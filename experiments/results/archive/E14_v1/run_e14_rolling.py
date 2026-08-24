#!/usr/bin/env python3
"""Leakage-safe rolling evaluation of the E14 season-to-date pitcher feature."""

from __future__ import annotations

import argparse
import gc
import json
import math
import platform
import sys
import time
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.run_baselines import (  # noqa: E402
    FEATURES as BASE_FEATURES,
    HGB_CATEGORICAL,
    HGB_DROPPED,
    INT16_COLUMNS,
    INT32_COLUMNS,
    INT8_COLUMNS,
    LINEAR_CATEGORICAL,
    RANDOM_SEED,
    STRING_CATEGORICAL,
    TARGET,
    dtype_map,
    load_train,
)


SEASON = "season"
PITCHER = "pitcher_id"
ASOF_N = "asof_pitcher_n"
ASOF_RATE = "asof_pitcher_success_rate"
E14_FEATURES = [
    "e14_n_season",
    "e14_s_season",
    "e14_log_n_season",
    "e14_rate_season",
    "e14_rate_delta",
    "e14_n_season_zero",
    "e14_pitcher_unseen",
    "e14_counter_invalid",
]
E14_K = 120.0
E14_FALLBACK_PRIOR = 0.5


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=ROOT / "open/data/train.csv")
    parser.add_argument(
        "--validation-seasons", nargs="+", type=int, default=[2022, 2023, 2024]
    )
    parser.add_argument(
        "--output-dir", type=Path, default=ROOT / "experiments/results"
    )
    parser.add_argument("--max-history-rows", type=int, default=None)
    parser.add_argument("--max-valid-rows", type=int, default=None)
    return parser.parse_args()


def metric(y_true: np.ndarray, prediction: np.ndarray) -> dict[str, float | None]:
    y = np.asarray(y_true, dtype=np.float64)
    p = np.asarray(prediction, dtype=np.float64)
    if y.shape != p.shape or not np.isfinite(p).all() or np.any((p < 0) | (p > 1)):
        raise ValueError("Invalid target/prediction shape or probability.")
    rate = float(y.mean())
    brier = float(np.mean((p - y) ** 2))
    reference = rate * (1.0 - rate)
    raw_skill = float(1.0 - brier / reference) if reference else None
    clipped = np.clip(p, 1e-7, 1.0 - 1e-7)
    try:
        auc: float | None = float(roc_auc_score(y, p))
    except ValueError:
        auc = None
    return {
        "n": int(len(y)),
        "target_rate": rate,
        "prediction_mean": float(p.mean()),
        "prediction_std": float(p.std()),
        "brier": brier,
        "reference_brier": reference,
        "raw_skill": raw_skill,
        "competition_score": float(max(0.0, 100_000.0 * raw_skill))
        if raw_skill is not None
        else 0.0,
        "log_loss": float(log_loss(y, clipped, labels=[0, 1])),
        "roc_auc": auc,
    }


def season_end_state(frame: pd.DataFrame) -> tuple[dict[int, dict[int, tuple[int, int]]], dict[int, tuple[int, int]]]:
    """Return state before each included season and the final state after it."""
    before: dict[int, dict[int, tuple[int, int]]] = {}
    state: dict[int, tuple[int, int]] = {}
    for season in sorted(int(value) for value in frame[SEASON].unique()):
        before[season] = dict(state)
        season_rows = frame.loc[frame[SEASON] == season]
        last_rows = season_rows.groupby(PITCHER, sort=False, observed=True).tail(1)
        for row in last_rows.itertuples(index=False):
            n = int(getattr(row, ASOF_N) or 0)
            rate = getattr(row, ASOF_RATE)
            rate_value = 0.0 if pd.isna(rate) else float(rate)
            successes_before = int(np.rint(rate_value * n))
            state[int(getattr(row, PITCHER))] = (
                n + 1,
                successes_before + int(getattr(row, TARGET)),
            )
    return before, state


def prior_before_each_season(frame: pd.DataFrame) -> dict[int, float]:
    result: dict[int, float] = {}
    running_target: list[float] = []
    for season in sorted(int(value) for value in frame[SEASON].unique()):
        result[season] = (
            float(np.mean(running_target)) if running_target else E14_FALLBACK_PRIOR
        )
        values = frame.loc[frame[SEASON] == season, TARGET].to_numpy(dtype=float)
        running_target.extend(values.tolist())
    return result


def _row_states(
    frame: pd.DataFrame,
    states_before: dict[int, dict[int, tuple[int, int]]],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n_end = np.zeros(len(frame), dtype=np.int64)
    s_end = np.zeros(len(frame), dtype=np.int64)
    unseen = np.zeros(len(frame), dtype=np.int8)
    seasons = frame[SEASON].to_numpy(dtype=np.int16, copy=False)
    pitchers = frame[PITCHER].to_numpy(dtype=np.int32, copy=False)
    for index, (season, pitcher) in enumerate(zip(seasons, pitchers)):
        state = states_before.get(int(season), {}).get(int(pitcher))
        if state is None:
            unseen[index] = 1
        else:
            n_end[index], s_end[index] = state
    return n_end, s_end, unseen


def build_e14_features(
    frame: pd.DataFrame,
    states_before: dict[int, dict[int, tuple[int, int]]],
    prior_by_season: dict[int, float],
    validation_prior: float,
    k: float = E14_K,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Build E14 features using only a row's as-of values and frozen prior state."""
    n_end, s_end, unseen = _row_states(frame, states_before)
    n_asof = frame[ASOF_N].fillna(0).to_numpy(dtype=np.int64, copy=False)
    career_rate = frame[ASOF_RATE].fillna(validation_prior).to_numpy(
        dtype=np.float64, copy=False
    )
    successes_asof = np.rint(career_rate * n_asof).astype(np.int64)
    n_delta = n_asof - n_end
    s_delta = successes_asof - s_end
    invalid = (n_delta < 0) | (s_delta < 0) | (s_delta > n_delta)
    safe_n = np.where(invalid, 0, n_delta)
    safe_s = np.where(invalid, 0, s_delta)
    row_priors = np.array(
        [prior_by_season.get(int(season), validation_prior) for season in frame[SEASON]],
        dtype=np.float64,
    )
    rate_season = (safe_s + k * row_priors) / (safe_n + k)
    result = pd.DataFrame(
        {
            "e14_n_season": safe_n.astype(np.int32),
            "e14_s_season": safe_s.astype(np.int32),
            "e14_log_n_season": np.log1p(safe_n).astype(np.float32),
            "e14_rate_season": rate_season.astype(np.float32),
            "e14_rate_delta": (rate_season - career_rate).astype(np.float32),
            "e14_n_season_zero": (safe_n == 0).astype(np.int8),
            "e14_pitcher_unseen": unseen,
            "e14_counter_invalid": invalid.astype(np.int8),
        },
        index=frame.index,
    )
    metadata = {
        "k": float(k),
        "validation_prior": float(validation_prior),
        "invalid_rows": int(invalid.sum()),
        "unseen_pitcher_rows": int(unseen.sum()),
        "n_zero_rows": int((safe_n == 0).sum()),
        "n_median": float(np.median(safe_n)) if len(safe_n) else 0.0,
        "n_positive_rate": float(np.mean(safe_n > 0)) if len(safe_n) else 0.0,
    }
    return result, metadata


def make_linear(features: list[str]) -> Pipeline:
    numeric = [column for column in features if column not in LINEAR_CATEGORICAL]
    numeric_pipeline = Pipeline(
        [("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())]
    )
    categorical_pipeline = Pipeline(
        [
            ("impute", SimpleImputer(strategy="most_frequent")),
            (
                "onehot",
                OneHotEncoder(
                    handle_unknown="ignore", min_frequency=10, dtype=np.float32
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


def make_hgb(features: list[str]) -> Pipeline:
    model_features = [column for column in features if column not in HGB_DROPPED]
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


def deterministic_sample(frame: pd.DataFrame, maximum: int | None) -> pd.DataFrame:
    if maximum is None or len(frame) <= maximum:
        return frame
    return frame.sample(n=maximum, random_state=RANDOM_SEED).sort_index()


def fit_predict(
    name: str,
    factory: Callable[[list[str]], Pipeline],
    train_x: pd.DataFrame,
    train_y: np.ndarray,
    valid_x: pd.DataFrame,
) -> tuple[np.ndarray, dict[str, float | int | None]]:
    print(f"[{name}] fit={len(train_x):,} rows, features={train_x.shape[1]}", flush=True)
    model = factory(list(train_x.columns))
    started = time.perf_counter()
    model.fit(train_x, train_y)
    fit_seconds = time.perf_counter() - started
    prediction_started = time.perf_counter()
    prediction = model.predict_proba(valid_x)[:, 1].astype(np.float64)
    predict_seconds = time.perf_counter() - prediction_started
    classifier = model.named_steps["clf"]
    iteration = getattr(classifier, "n_iter_", None)
    if isinstance(iteration, np.ndarray):
        iteration = int(np.max(iteration))
    elif iteration is not None:
        iteration = int(iteration)
    del model
    gc.collect()
    return prediction, {
        "fit_seconds": fit_seconds,
        "predict_seconds": predict_seconds,
        "n_iter": iteration,
    }


def segment_brier(
    valid: pd.DataFrame, prediction: np.ndarray, y: np.ndarray
) -> dict[str, dict[str, Any]]:
    n = valid["e14_n_season"].to_numpy(dtype=np.int64, copy=False)
    bins = [(-1, 0, "n_0"), (0, 49, "n_1_49"), (49, 199, "n_50_199"),
            (199, 499, "n_200_499"), (499, 10**9, "n_500_plus")]
    output: dict[str, dict[str, Any]] = {}
    for lower, upper, label in bins:
        mask = (n > lower) & (n <= upper)
        if int(mask.sum()):
            output[label] = metric(y[mask], prediction[mask])
    for label, mask in {
        "game_type_R": valid["game_type"].eq("R").to_numpy(
            dtype=bool, na_value=False
        ),
        "game_type_F": valid["game_type"].eq("F").to_numpy(
            dtype=bool, na_value=False
        ),
        "pitcher_unseen": valid["e14_pitcher_unseen"].eq(1).to_numpy(
            dtype=bool, na_value=False
        ),
    }.items():
        if int(mask.sum()):
            output[label] = metric(y[mask], prediction[mask])
    return output


def feature_invariance(
    frame: pd.DataFrame,
    states_before: dict[int, dict[int, tuple[int, int]]],
    priors: dict[int, float],
    validation_prior: float,
) -> float:
    sample = frame.iloc[: min(5, len(frame))].copy()
    if sample.empty:
        return 0.0
    reference, _ = build_e14_features(sample, states_before, priors, validation_prior)
    order = list(reversed(range(len(sample))))
    shuffled, _ = build_e14_features(sample.iloc[order], states_before, priors, validation_prior)
    shuffled = shuffled.iloc[order]
    delta = np.max(np.abs(reference.to_numpy(dtype=float) - shuffled.to_numpy(dtype=float)))
    duplicated = pd.concat([sample, sample.iloc[[0]]], ignore_index=False)
    duplicate_features, _ = build_e14_features(
        duplicated, states_before, priors, validation_prior
    )
    delta = max(
        float(delta),
        float(
            np.max(
                np.abs(
                    reference.to_numpy(dtype=float)
                    - duplicate_features.iloc[: len(sample)].to_numpy(dtype=float)
                )
            )
        ),
    )
    return float(delta)


def run_fold(frame: pd.DataFrame, validation_season: int, args: argparse.Namespace) -> dict[str, Any]:
    started = time.perf_counter()
    history = frame.loc[frame[SEASON] < validation_season].copy()
    valid = frame.loc[frame[SEASON] == validation_season].copy()
    history = deterministic_sample(history, args.max_history_rows)
    valid = deterministic_sample(valid, args.max_valid_rows)
    if history.empty or valid.empty:
        raise ValueError(f"Empty fold for season {validation_season}")
    # State assets are built only from the outer history; validation rows never
    # contribute to n_end, s_end, priors, model fitting, or K selection.
    states_before, final_state = season_end_state(history)
    train_priors = prior_before_each_season(history)
    validation_prior = float(history[TARGET].mean())
    validation_states = {validation_season: final_state}
    validation_priors = {validation_season: validation_prior}
    train_e14, train_e14_meta = build_e14_features(
        history, states_before, train_priors, validation_prior
    )
    valid_e14, valid_e14_meta = build_e14_features(
        valid, validation_states, validation_priors, validation_prior
    )
    invariant_delta = feature_invariance(
        valid, validation_states, validation_priors, validation_prior
    )
    if invariant_delta >= 1e-12:
        raise AssertionError(f"E14 feature invariance failed: {invariant_delta:.3e}")

    train_y = history[TARGET].to_numpy(dtype=np.int8, copy=False)
    valid_y = valid[TARGET].to_numpy(dtype=np.int8, copy=False)
    baseline_predictions: dict[str, np.ndarray] = {}
    e14_predictions: dict[str, np.ndarray] = {}
    fit_details: dict[str, Any] = {"baseline": {}, "e14": {}}
    factories = {"linear": make_linear, "hgb": make_hgb}
    for name, factory in factories.items():
        baseline_predictions[name], fit_details["baseline"][name] = fit_predict(
            f"{validation_season}/baseline/{name}",
            factory,
            history[BASE_FEATURES],
            train_y,
            valid[BASE_FEATURES],
        )
        e14_train = pd.concat([history[BASE_FEATURES], train_e14], axis=1)
        e14_valid = pd.concat([valid[BASE_FEATURES], valid_e14], axis=1)
        e14_predictions[name], fit_details["e14"][name] = fit_predict(
            f"{validation_season}/e14/{name}", factory, e14_train, train_y, e14_valid
        )
        del e14_train, e14_valid
        gc.collect()

    baseline_blend = 0.9 * baseline_predictions["linear"] + 0.1 * baseline_predictions["hgb"]
    e14_blend = 0.9 * e14_predictions["linear"] + 0.1 * e14_predictions["hgb"]
    standalone = valid_e14["e14_rate_season"].to_numpy(dtype=float)
    baseline_summary = metric(valid_y, baseline_blend)
    e14_summary = metric(valid_y, e14_blend)
    result = {
        "validation_season": validation_season,
        "history_rows": len(history),
        "valid_rows": len(valid),
        "history_seasons": sorted(int(value) for value in history[SEASON].unique()),
        "validation_target_rate": float(valid_y.mean()),
        "history_prior": validation_prior,
        "k": E14_K,
        "state_pitchers": len(final_state),
        "train_e14": train_e14_meta,
        "valid_e14": valid_e14_meta,
        "feature_invariance_max_abs_delta": invariant_delta,
        "baseline_s2": baseline_summary,
        "e14_s2": e14_summary,
        "e14_brier_delta": float(e14_summary["brier"] - baseline_summary["brier"]),
        "e14_score_delta": float(
            e14_summary["competition_score"] - baseline_summary["competition_score"]
        ),
        "standalone_e14_rate": metric(valid_y, standalone),
        "segments_e14": segment_brier(
            pd.concat([valid, valid_e14], axis=1), e14_blend, valid_y
        ),
        "residual_correlation": float(
            np.corrcoef(valid_y - baseline_blend, valid_y - e14_blend)[0, 1]
        ),
        "fit_details": fit_details,
        "elapsed_seconds": time.perf_counter() - started,
    }
    print(
        f"[{validation_season}] baseline Brier={baseline_summary['brier']:.8f}, "
        f"E14 Brier={e14_summary['brier']:.8f}, "
        f"delta={result['e14_brier_delta']:+.8f}, "
        f"score delta={result['e14_score_delta']:+.1f}",
        flush=True,
    )
    del history, valid, train_e14, valid_e14, baseline_predictions, e14_predictions
    gc.collect()
    return result


def main() -> None:
    args = parse_args()
    started = time.perf_counter()
    frame = load_train(args.data)
    results = [run_fold(frame, season, args) for season in sorted(args.validation_seasons)]
    del frame
    gc.collect()
    deltas = [float(result["e14_brier_delta"]) for result in results]
    wins = sum(delta < 0.0 for delta in deltas)
    aggregate = {
        "folds": len(results),
        "e14_wins": wins,
        "mean_brier_delta": float(np.mean(deltas)),
        "worst_brier_delta": float(np.max(deltas)),
        "mean_score_delta": float(
            np.mean([float(result["e14_score_delta"]) for result in results])
        ),
        "gate_pass": bool(wins >= 2 and np.max(deltas) <= 0.0005),
        "gate_definition": "wins >= 2/3 and worst Brier delta <= 0.0005",
    }
    payload = {
        "metadata": {
            "created_at_utc": pd.Timestamp.utcnow().isoformat(),
            "data": str(args.data),
            "validation_seasons": sorted(args.validation_seasons),
            "protocol": "outer history season < Y; validation season == Y",
            "row_independent_inference": True,
            "prior_source": "history seasons strictly before each row season; validation prior from season < Y",
            "k_source": "fixed EDA/theoretical value 120; not tuned on validation targets",
            "feature_cutoff": "asof_pitcher_* row values minus frozen prior-season state",
            "versions": {
                "python": platform.python_version(),
                "numpy": np.__version__,
                "pandas": pd.__version__,
                "scikit_learn": __import__("sklearn").__version__,
            },
            "command": " ".join(sys.argv),
            "elapsed_seconds": time.perf_counter() - started,
        },
        "aggregate": aggregate,
        "folds": results,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.output_dir / "e14_rolling.json"
    csv_path = args.output_dir / "e14_rolling.csv"
    json_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8"
    )
    rows = []
    for result in results:
        rows.append(
            {
                "validation_season": result["validation_season"],
                "history_rows": result["history_rows"],
                "valid_rows": result["valid_rows"],
                "baseline_brier": result["baseline_s2"]["brier"],
                "e14_brier": result["e14_s2"]["brier"],
                "e14_brier_delta": result["e14_brier_delta"],
                "baseline_score": result["baseline_s2"]["competition_score"],
                "e14_score": result["e14_s2"]["competition_score"],
                "e14_score_delta": result["e14_score_delta"],
                "standalone_e14_brier": result["standalone_e14_rate"]["brier"],
                "invalid_rows": result["valid_e14"]["invalid_rows"],
                "unseen_pitcher_rows": result["valid_e14"]["unseen_pitcher_rows"],
                "feature_invariance_max_abs_delta": result[
                    "feature_invariance_max_abs_delta"
                ],
                "residual_correlation": result["residual_correlation"],
            }
        )
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    print(json.dumps(aggregate, ensure_ascii=False, indent=2), flush=True)
    print(f"Saved {json_path} and {csv_path}.", flush=True)


if __name__ == "__main__":
    main()
