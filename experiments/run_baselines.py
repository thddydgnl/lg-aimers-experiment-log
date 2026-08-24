#!/usr/bin/env python3
"""Run leakage-safe temporal baselines for the LG Aimers pitch-control task.

The primary validation protocol is train on seasons before 2024 and validate on
the complete 2024 season.  Every model receives one row at a time conceptually;
no statistic is computed from validation/test rows during prediction.
"""

from __future__ import annotations

import argparse
import gc
import json
import platform
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import joblib
import numpy as np
import pandas as pd
import scipy
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler


TARGET = "control_success"
SEASON = "season"
RANDOM_SEED = 42

FEATURES = [
    "season",
    "game_month",
    "game_dayofweek",
    "inning",
    "top_bottom",
    "game_type",
    "balls_before",
    "strikes_before",
    "outs_before",
    "run_top_before",
    "run_bot_before",
    "run_total_before",
    "score_diff_home",
    "score_diff_pitcher_team",
    "runner_on_1b",
    "runner_on_2b",
    "runner_on_3b",
    "num_runners_on",
    "base_state",
    "home_win_expectancy",
    "away_win_expectancy",
    "li",
    "pitcher_id",
    "batter_id",
    "pitcher_hand",
    "batter_hand",
    "pitcher_team_id",
    "batter_team_id",
    "asof_pitcher_n",
    "asof_pitcher_success_rate",
    "asof_pitcher_reverse_rate",
    "asof_pitcher_middle_rate",
    "asof_pitcher_ball_rate",
    "asof_pitcher_strike_rate",
    "asof_pitcher_prev1_game_success_rate",
    "asof_pitcher_prev3_game_success_rate",
    "asof_pitcher_prev5_game_success_rate",
    "asof_pitcher_prev1_game_middle_rate",
    "asof_pitcher_prev3_game_middle_rate",
    "asof_pitcher_prev5_game_middle_rate",
    "asof_batter_n",
    "asof_batter_success_rate",
    "asof_batter_middle_rate",
    "asof_pitcher_pitchmix_n",
    "asof_pitcher_fastball_rate",
    "asof_pitcher_breaking_rate",
    "asof_pitcher_offspeed_rate",
]

STRING_CATEGORICAL = ["top_bottom", "game_type", "base_state"]
LINEAR_CATEGORICAL = [
    "top_bottom",
    "game_type",
    "base_state",
    "pitcher_id",
    "batter_id",
    "pitcher_hand",
    "batter_hand",
    "pitcher_team_id",
    "batter_team_id",
]
HGB_CATEGORICAL = [
    "top_bottom",
    "game_type",
    "base_state",
    "pitcher_hand",
    "batter_hand",
    "pitcher_team_id",
    "batter_team_id",
]
HGB_DROPPED = ["pitcher_id", "batter_id"]

INT8_COLUMNS = [
    "game_month",
    "game_dayofweek",
    "inning",
    "balls_before",
    "strikes_before",
    "outs_before",
    "runner_on_1b",
    "runner_on_2b",
    "runner_on_3b",
    "num_runners_on",
    "pitcher_hand",
    "batter_hand",
    TARGET,
]
INT16_COLUMNS = [
    "season",
    "run_top_before",
    "run_bot_before",
    "run_total_before",
    "score_diff_home",
    "score_diff_pitcher_team",
    "pitcher_team_id",
    "batter_team_id",
]
INT32_COLUMNS = [
    "pitcher_id",
    "batter_id",
    "asof_pitcher_n",
    "asof_batter_n",
    "asof_pitcher_pitchmix_n",
]
FLOAT_COLUMNS = [
    column
    for column in FEATURES
    if column
    not in set(STRING_CATEGORICAL + INT8_COLUMNS + INT16_COLUMNS + INT32_COLUMNS)
]

MODEL_CHOICES = (
    "constant",
    "asof_pitcher",
    "smoothed_asof_blend",
    "linear_sgd",
    "official_rf",
    "hist_gradient_boosting",
)


@dataclass
class MetricSummary:
    n: int
    target_rate: float
    prediction_mean: float
    prediction_std: float
    brier: float
    reference_brier: float
    raw_skill: float | None
    competition_score: float
    log_loss: float
    roc_auc: float | None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("open/data/train.csv"))
    parser.add_argument("--validation-season", type=int, default=2024)
    parser.add_argument(
        "--models",
        nargs="+",
        choices=MODEL_CHOICES,
        default=list(MODEL_CHOICES),
    )
    parser.add_argument("--output-dir", type=Path, default=Path("experiments/results"))
    parser.add_argument("--artifact-dir", type=Path, default=Path("experiments/artifacts"))
    parser.add_argument("--save-models", action="store_true")
    parser.add_argument(
        "--max-train-rows",
        type=int,
        default=None,
        help="Deterministic debug subsample; omit for the official experiment.",
    )
    parser.add_argument(
        "--max-valid-rows",
        type=int,
        default=None,
        help="Deterministic debug subsample; omit for the official experiment.",
    )
    return parser.parse_args()


def dtype_map() -> dict[str, str]:
    result: dict[str, str] = {column: "string" for column in STRING_CATEGORICAL}
    result.update({column: "int8" for column in INT8_COLUMNS})
    result.update({column: "int16" for column in INT16_COLUMNS})
    result.update({column: "int32" for column in INT32_COLUMNS})
    result.update({column: "float32" for column in FLOAT_COLUMNS})
    return result


def load_train(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    started = time.perf_counter()
    frame = pd.read_csv(
        path,
        usecols=FEATURES + [TARGET],
        dtype=dtype_map(),
        encoding="utf-8-sig",
    )
    elapsed = time.perf_counter() - started
    print(
        f"Loaded {len(frame):,} rows x {frame.shape[1]} columns "
        f"in {elapsed:.1f}s ({frame.memory_usage(deep=True).sum() / 2**20:.1f} MiB).",
        flush=True,
    )
    return frame


def deterministic_sample(frame: pd.DataFrame, maximum: int | None) -> pd.DataFrame:
    if maximum is None or len(frame) <= maximum:
        return frame
    return frame.sample(n=maximum, random_state=RANDOM_SEED).sort_index()


def validate_split(
    frame: pd.DataFrame,
    validation_season: int,
    max_train_rows: int | None,
    max_valid_rows: int | None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    train = frame.loc[frame[SEASON] < validation_season].copy()
    valid = frame.loc[frame[SEASON] == validation_season].copy()
    if train.empty or valid.empty:
        seasons = sorted(frame[SEASON].unique().tolist())
        raise ValueError(
            f"Temporal split is empty for validation season {validation_season}; "
            f"available seasons={seasons}."
        )
    train = deterministic_sample(train, max_train_rows)
    valid = deterministic_sample(valid, max_valid_rows)
    if int(train[SEASON].max()) >= int(valid[SEASON].min()):
        raise AssertionError("Temporal split leakage detected.")
    print(
        f"Split: train={len(train):,} rows ({int(train[SEASON].min())}-"
        f"{int(train[SEASON].max())}), valid={len(valid):,} rows "
        f"({validation_season}).",
        flush=True,
    )
    return train, valid


def metric_summary(y_true: np.ndarray, prediction: np.ndarray) -> MetricSummary:
    y = np.asarray(y_true, dtype=np.float64)
    p = np.asarray(prediction, dtype=np.float64)
    if y.shape != p.shape:
        raise ValueError(f"Shape mismatch: y={y.shape}, p={p.shape}")
    if not np.isfinite(p).all() or np.any((p < 0.0) | (p > 1.0)):
        raise ValueError("Predictions must be finite probabilities in [0, 1].")
    rate = float(y.mean())
    brier = float(np.mean(np.square(p - y)))
    reference = float(rate * (1.0 - rate))
    raw_skill = float(1.0 - brier / reference) if reference > 0 else None
    score = float(max(0.0, 100_000.0 * raw_skill)) if raw_skill is not None else 0.0
    clipped = np.clip(p, 1e-7, 1.0 - 1e-7)
    auc: float | None
    if np.unique(y).size < 2:
        auc = None
    else:
        try:
            candidate_auc = float(roc_auc_score(y, p))
            auc = candidate_auc if np.isfinite(candidate_auc) else None
        except ValueError:
            auc = None
    return MetricSummary(
        n=len(y),
        target_rate=rate,
        prediction_mean=float(p.mean()),
        prediction_std=float(p.std()),
        brier=brier,
        reference_brier=reference,
        raw_skill=raw_skill,
        competition_score=score,
        log_loss=float(log_loss(y, clipped, labels=[0, 1])),
        roc_auc=auc,
    )


def segment_metrics(valid: pd.DataFrame, prediction: np.ndarray) -> dict[str, Any]:
    masks: dict[str, np.ndarray] = {
        "all": np.ones(len(valid), dtype=bool),
        "game_type_R": valid["game_type"].eq("R").to_numpy(dtype=bool, na_value=False),
        "game_type_F": valid["game_type"].eq("F").to_numpy(dtype=bool, na_value=False),
        "pitcher_cold_start_n0": valid["asof_pitcher_n"].eq(0).to_numpy(),
        "batter_cold_start_n0": valid["asof_batter_n"].eq(0).to_numpy(),
        "recent_form_missing": valid[
            "asof_pitcher_prev5_game_success_rate"
        ].isna().to_numpy(),
    }
    y = valid[TARGET].to_numpy(dtype=np.int8, copy=False)
    output: dict[str, Any] = {}
    for name, mask in masks.items():
        if int(mask.sum()) == 0:
            continue
        output[name] = asdict(metric_summary(y[mask], prediction[mask]))
    return output


def constant_prediction(valid: pd.DataFrame, prior: float) -> np.ndarray:
    return np.full(len(valid), prior, dtype=np.float64)


def raw_pitcher_prediction(valid: pd.DataFrame, prior: float) -> np.ndarray:
    return (
        valid["asof_pitcher_success_rate"]
        .fillna(prior)
        .to_numpy(dtype=np.float64, copy=False)
    )


def smoothed_asof_prediction(
    frame: pd.DataFrame,
    prior: float,
    alpha: float,
    pitcher_weight: float,
) -> np.ndarray:
    pitcher_n = frame["asof_pitcher_n"].to_numpy(dtype=np.float64, copy=False)
    batter_n = frame["asof_batter_n"].to_numpy(dtype=np.float64, copy=False)
    pitcher_rate = (
        frame["asof_pitcher_success_rate"]
        .fillna(prior)
        .to_numpy(dtype=np.float64, copy=False)
    )
    batter_rate = (
        frame["asof_batter_success_rate"]
        .fillna(prior)
        .to_numpy(dtype=np.float64, copy=False)
    )
    pitcher_smoothed = (pitcher_n * pitcher_rate + alpha * prior) / (
        pitcher_n + alpha
    )
    batter_smoothed = (batter_n * batter_rate + alpha * prior) / (batter_n + alpha)
    return pitcher_weight * pitcher_smoothed + (1.0 - pitcher_weight) * batter_smoothed


def tune_smoothed_asof(
    train: pd.DataFrame, validation_season: int
) -> tuple[dict[str, float], list[dict[str, float]]]:
    calibration_season = validation_season - 1
    history = train.loc[train[SEASON] < calibration_season]
    calibration = train.loc[train[SEASON] == calibration_season]
    if history.empty or calibration.empty:
        return {"alpha": 100.0, "pitcher_weight": 0.75}, []
    prior = float(history[TARGET].mean())
    y = calibration[TARGET].to_numpy(dtype=np.int8, copy=False)
    trials: list[dict[str, float]] = []
    for alpha in (10.0, 30.0, 100.0, 300.0, 1000.0):
        for weight in (0.50, 0.65, 0.80, 0.95):
            prediction = smoothed_asof_prediction(calibration, prior, alpha, weight)
            metrics = metric_summary(y, prediction)
            trials.append(
                {
                    "alpha": alpha,
                    "pitcher_weight": weight,
                    "brier": metrics.brier,
                    "competition_score": metrics.competition_score,
                }
            )
    best = min(trials, key=lambda row: row["brier"])
    return {
        "alpha": float(best["alpha"]),
        "pitcher_weight": float(best["pitcher_weight"]),
    }, trials


def make_linear_sgd() -> Pipeline:
    numeric = [column for column in FEATURES if column not in LINEAR_CATEGORICAL]
    numeric_pipeline = Pipeline(
        [
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        [
            ("impute", SimpleImputer(strategy="most_frequent")),
            (
                "onehot",
                OneHotEncoder(
                    handle_unknown="ignore",
                    min_frequency=10,
                    dtype=np.float32,
                ),
            ),
        ]
    )
    preprocessor = ColumnTransformer(
        [
            ("numeric", numeric_pipeline, numeric),
            ("categorical", categorical_pipeline, LINEAR_CATEGORICAL),
        ],
        sparse_threshold=1.0,
    )
    classifier = SGDClassifier(
        loss="log_loss",
        penalty="l2",
        # Strong regularisation is intentional: the Brier objective punishes the
        # overconfident probabilities produced by a lightly regularised SGD fit.
        alpha=0.3,
        learning_rate="constant",
        eta0=0.001,
        max_iter=100,
        tol=1e-4,
        average=True,
        random_state=RANDOM_SEED,
        n_jobs=-1,
    )
    return Pipeline([("pre", preprocessor), ("clf", classifier)])


def make_official_rf() -> Pipeline:
    numeric = [column for column in FEATURES if column not in STRING_CATEGORICAL]
    preprocessor = ColumnTransformer(
        [
            (
                "cat",
                OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1),
                STRING_CATEGORICAL,
            ),
            ("num", SimpleImputer(strategy="median"), numeric),
        ]
    )
    classifier = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        min_samples_leaf=200,
        max_features="sqrt",
        n_jobs=-1,
        random_state=RANDOM_SEED,
    )
    return Pipeline([("pre", preprocessor), ("clf", classifier)])


def make_hist_gradient_boosting() -> Pipeline:
    model_features = [column for column in FEATURES if column not in HGB_DROPPED]
    numeric = [column for column in model_features if column not in HGB_CATEGORICAL]
    preprocessor = ColumnTransformer(
        [
            (
                "cat",
                OrdinalEncoder(
                    handle_unknown="use_encoded_value",
                    unknown_value=-1,
                    encoded_missing_value=-1,
                ),
                HGB_CATEGORICAL,
            ),
            ("num", "passthrough", numeric),
        ]
    )
    categorical_mask = [True] * len(HGB_CATEGORICAL) + [False] * len(numeric)
    classifier = HistGradientBoostingClassifier(
        loss="log_loss",
        learning_rate=0.05,
        max_iter=250,
        max_leaf_nodes=31,
        min_samples_leaf=100,
        l2_regularization=5.0,
        categorical_features=categorical_mask,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=20,
        random_state=RANDOM_SEED,
    )
    return Pipeline([("pre", preprocessor), ("clf", classifier)])


def run_trainable_model(
    name: str,
    factory: Callable[[], Pipeline],
    train: pd.DataFrame,
    valid: pd.DataFrame,
    artifact_path: Path | None,
) -> tuple[np.ndarray, float, float, dict[str, Any]]:
    print(f"[{name}] fitting...", flush=True)
    model = factory()
    fit_started = time.perf_counter()
    model.fit(train[FEATURES], train[TARGET])
    fit_seconds = time.perf_counter() - fit_started
    print(f"[{name}] predicting...", flush=True)
    predict_started = time.perf_counter()
    prediction = model.predict_proba(valid[FEATURES])[:, 1]
    predict_seconds = time.perf_counter() - predict_started
    details: dict[str, Any] = {}
    if name == "hist_gradient_boosting":
        details["n_iter"] = int(model.named_steps["clf"].n_iter_)
    if name == "linear_sgd":
        details["n_iter"] = int(np.max(model.named_steps["clf"].n_iter_))
    if artifact_path is not None:
        artifact_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(model, artifact_path, compress=3)
        details["artifact"] = str(artifact_path)
    del model
    gc.collect()
    return prediction, fit_seconds, predict_seconds, details


def main() -> None:
    args = parse_args()
    run_started = time.perf_counter()
    frame = load_train(args.data)
    train, valid = validate_split(
        frame,
        args.validation_season,
        args.max_train_rows,
        args.max_valid_rows,
    )
    del frame
    gc.collect()

    y_valid = valid[TARGET].to_numpy(dtype=np.int8, copy=False)
    train_prior = float(train[TARGET].mean())
    model_results: dict[str, Any] = {}

    def record(
        name: str,
        prediction: np.ndarray,
        fit_seconds: float = 0.0,
        predict_seconds: float = 0.0,
        details: dict[str, Any] | None = None,
    ) -> None:
        metrics = metric_summary(y_valid, prediction)
        model_results[name] = {
            "metrics": asdict(metrics),
            "segments": segment_metrics(valid, prediction),
            "fit_seconds": fit_seconds,
            "predict_seconds": predict_seconds,
            "details": details or {},
        }
        auc_text = "NA" if metrics.roc_auc is None else f"{metrics.roc_auc:.6f}"
        print(
            f"[{name}] Brier={metrics.brier:.8f}, "
            f"score={metrics.competition_score:,.1f}, "
            f"AUC={auc_text}, fit={fit_seconds:.1f}s, "
            f"predict={predict_seconds:.1f}s",
            flush=True,
        )

    if "constant" in args.models:
        record("constant", constant_prediction(valid, train_prior), details={"prior": train_prior})

    if "asof_pitcher" in args.models:
        record(
            "asof_pitcher",
            raw_pitcher_prediction(valid, train_prior),
            details={"missing_fallback": train_prior},
        )

    if "smoothed_asof_blend" in args.models:
        selected, trials = tune_smoothed_asof(train, args.validation_season)
        prediction = smoothed_asof_prediction(
            valid,
            train_prior,
            selected["alpha"],
            selected["pitcher_weight"],
        )
        record(
            "smoothed_asof_blend",
            prediction,
            details={
                "selected_on_season": args.validation_season - 1,
                "selected": selected,
                "tuning_trials": trials,
                "prediction_prior": train_prior,
            },
        )

    factories: dict[str, Callable[[], Pipeline]] = {
        "linear_sgd": make_linear_sgd,
        "official_rf": make_official_rf,
        "hist_gradient_boosting": make_hist_gradient_boosting,
    }
    for name in ("linear_sgd", "official_rf", "hist_gradient_boosting"):
        if name not in args.models:
            continue
        artifact_path = None
        if args.save_models:
            artifact_path = args.artifact_dir / f"{name}_through_{args.validation_season - 1}.joblib"
        prediction, fit_seconds, predict_seconds, details = run_trainable_model(
            name,
            factories[name],
            train,
            valid,
            artifact_path,
        )
        record(name, prediction, fit_seconds, predict_seconds, details)
        del prediction
        gc.collect()

    ordered = sorted(
        model_results,
        key=lambda name: model_results[name]["metrics"]["brier"],
    )
    metadata = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "data_path": str(args.data),
        "validation_protocol": f"season < {args.validation_season} -> season == {args.validation_season}",
        "validation_season": args.validation_season,
        "train_rows": len(train),
        "valid_rows": len(valid),
        "train_seasons": [int(train[SEASON].min()), int(train[SEASON].max())],
        "train_target_rate": train_prior,
        "valid_target_rate": float(valid[TARGET].mean()),
        "debug_subsample": args.max_train_rows is not None or args.max_valid_rows is not None,
        "max_train_rows": args.max_train_rows,
        "max_valid_rows": args.max_valid_rows,
        "features": FEATURES,
        "row_independent_inference": True,
        "versions": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scipy": scipy.__version__,
            "scikit_learn": sklearn.__version__,
            "joblib": joblib.__version__,
        },
        "command": " ".join(sys.argv),
        "elapsed_seconds": time.perf_counter() - run_started,
    }
    payload = {"metadata": metadata, "ranking": ordered, "models": model_results}

    args.output_dir.mkdir(parents=True, exist_ok=True)
    stem = f"baseline_valid_{args.validation_season}"
    json_path = args.output_dir / f"{stem}.json"
    csv_path = args.output_dir / f"{stem}.csv"
    json_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    rows = []
    for rank, name in enumerate(ordered, start=1):
        rows.append(
            {
                "rank": rank,
                "model": name,
                **model_results[name]["metrics"],
                "fit_seconds": model_results[name]["fit_seconds"],
                "predict_seconds": model_results[name]["predict_seconds"],
            }
        )
    pd.DataFrame(rows).to_csv(csv_path, index=False)

    print("\nFinal ranking (lower Brier is better):", flush=True)
    for rank, name in enumerate(ordered, start=1):
        metrics = model_results[name]["metrics"]
        print(
            f"  {rank}. {name:<26} Brier={metrics['brier']:.8f} "
            f"score={metrics['competition_score']:,.1f}",
            flush=True,
        )
    print(f"Saved {json_path} and {csv_path}.", flush=True)


if __name__ == "__main__":
    main()
