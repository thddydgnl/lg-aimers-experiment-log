#!/usr/bin/env python3
"""Evaluate the E22R soft pitch-group probabilities as S4 features.

The Trackman table is used only to recover a historical ``pitch_type_group``
label for matched training rows.  No current pitch measurement is joined to a
prediction row.  The stage-1 classifier sees only pre-pitch columns from the
main table and is fit on the outer history of each rolling fold.
"""

from __future__ import annotations

import argparse
import gc
import json
import platform
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from eda.run_structural_eda import (  # noqa: E402
    linkage_section,
    load_trackman,
    load_train as load_structural_train,
)
from experiments.run_baselines import FEATURES as BASE_FEATURES  # noqa: E402
from experiments.run_baselines import TARGET, RANDOM_SEED, load_train  # noqa: E402
from experiments.run_e14_rolling import (  # noqa: E402
    E14_FEATURES,
    E14_K,
    build_e14_features,
    fit_predict,
    make_hgb,
    make_linear,
    metric,
    prior_before_each_season,
    season_end_state,
)
from experiments.run_e15_pseudo_forward import candidate_priors  # noqa: E402


GROUPS = ["fastball", "breaking", "offspeed", "other"]
E22_FEATURES = [
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
    "asof_pitcher_pitchmix_n",
    "asof_pitcher_fastball_rate",
    "asof_pitcher_breaking_rate",
    "asof_pitcher_offspeed_rate",
]
E22_CATEGORICAL = [
    "top_bottom",
    "game_type",
    "base_state",
    "pitcher_hand",
    "batter_hand",
    "pitcher_team_id",
    "batter_team_id",
]
E22_PROB_FEATURES = [f"e22_p_group_{group}" for group in GROUPS]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=ROOT / "open/data/train.csv")
    parser.add_argument(
        "--validation-seasons", nargs="+", type=int, default=[2022, 2023, 2024]
    )
    parser.add_argument("--output-dir", type=Path, default=ROOT / "experiments/results")
    parser.add_argument("--prior-mode", default="r_recent3")
    return parser.parse_args()


def load_group_labels() -> pd.Series:
    """Recover exact historical row_id -> pitch group labels once."""
    structural = load_structural_train()
    trackman, game_ids, _ = load_trackman()
    _, joined = linkage_section(structural, trackman, len(game_ids))
    labels = joined[["row_id", "pitch_type_group"]].drop_duplicates("row_id")
    labels = labels.set_index("row_id")["pitch_type_group"]
    labels.name = "e22_pitch_type_group"
    print(
        f"Recovered E22 labels: {len(labels):,} rows, "
        f"groups={labels.value_counts().to_dict()}",
        flush=True,
    )
    return labels


def make_stage1() -> Pipeline:
    numeric = [column for column in E22_FEATURES if column not in E22_CATEGORICAL]
    categorical = Pipeline(
        [
            ("impute", __import__("sklearn").impute.SimpleImputer(strategy="most_frequent")),
            (
                "onehot",
                OneHotEncoder(handle_unknown="ignore", min_frequency=10, dtype=np.float32),
            ),
        ]
    )
    numeric_pipeline = Pipeline(
        [
            ("impute", __import__("sklearn").impute.SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
        ]
    )
    preprocessor = ColumnTransformer(
        [
            ("numeric", numeric_pipeline, numeric),
            ("categorical", categorical, E22_CATEGORICAL),
        ],
        sparse_threshold=1.0,
    )
    classifier = SGDClassifier(
        loss="log_loss",
        penalty="l2",
        alpha=0.0005,
        learning_rate="optimal",
        max_iter=60,
        tol=1e-4,
        average=True,
        random_state=RANDOM_SEED,
    )
    return Pipeline([("pre", preprocessor), ("clf", classifier)])


def align_group_probabilities(model: Pipeline, frame: pd.DataFrame) -> np.ndarray:
    raw = np.asarray(model.predict_proba(frame[E22_FEATURES]), dtype=np.float64)
    classes = list(model.named_steps["clf"].classes_)
    result = np.zeros((len(frame), len(GROUPS)), dtype=np.float64)
    for source, label in enumerate(classes):
        if label in GROUPS:
            result[:, GROUPS.index(label)] = raw[:, source]
    row_sum = result.sum(axis=1)
    missing = row_sum <= 0.0
    if missing.any():
        result[missing] = 1.0 / len(GROUPS)
        row_sum[missing] = 1.0
    result /= row_sum[:, None]
    return result.astype(np.float32)


def prior_group_probabilities(labels: pd.Series) -> np.ndarray:
    counts = labels.value_counts()
    vector = np.array([float(counts.get(group, 0.0)) for group in GROUPS], dtype=np.float64)
    if vector.sum() <= 0:
        vector[:] = 1.0
    vector /= vector.sum()
    return vector.astype(np.float32)


def fit_stage1(history: pd.DataFrame) -> tuple[Pipeline | None, np.ndarray, dict[str, Any]]:
    labeled = history.dropna(subset=["e22_pitch_type_group"])
    if labeled.empty:
        return None, np.full(len(GROUPS), 1.0 / len(GROUPS), dtype=np.float32), {
            "labeled_rows": 0,
            "classes": [],
            "fallback": True,
        }
    model = make_stage1()
    started = time.perf_counter()
    model.fit(labeled[E22_FEATURES], labeled["e22_pitch_type_group"].astype(str))
    prior = prior_group_probabilities(labeled["e22_pitch_type_group"])
    details = {
        "labeled_rows": int(len(labeled)),
        "classes": [str(value) for value in model.named_steps["clf"].classes_],
        "prior": prior.tolist(),
        "fit_seconds": time.perf_counter() - started,
        "features": E22_FEATURES,
        "uses_current_trackman": False,
    }
    return model, prior, details


def group_features_for_fold(
    history: pd.DataFrame, valid: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Fit stage 1 only on outer history and predict both partitions."""
    model, prior, details = fit_stage1(history)
    if model is None:
        train_probs = np.tile(prior, (len(history), 1))
        valid_probs = np.tile(prior, (len(valid), 1))
    else:
        # This is an outer-history fit; validation probabilities are genuinely
        # out of sample.  History probabilities are used only to fit stage 2,
        # and the classifier never sees control_success.
        train_probs = align_group_probabilities(model, history)
        valid_probs = align_group_probabilities(model, valid)
        del model
        gc.collect()
    train_frame = pd.DataFrame(train_probs, columns=E22_PROB_FEATURES, index=history.index)
    valid_frame = pd.DataFrame(valid_probs, columns=E22_PROB_FEATURES, index=valid.index)
    details.update(
        {
            "train_probability_mean": train_probs.mean(axis=0).tolist(),
            "valid_probability_mean": valid_probs.mean(axis=0).tolist(),
            "probability_invariance": 0.0,
        }
    )
    return train_frame, valid_frame, details


def run_fold(
    frame: pd.DataFrame, season: int, labels: pd.Series, args: argparse.Namespace
) -> dict[str, Any]:
    started = time.perf_counter()
    history = frame.loc[frame["season"] < season].copy()
    valid = frame.loc[frame["season"] == season].copy()
    history["e22_pitch_type_group"] = history["row_id"].map(labels)
    valid["e22_pitch_type_group"] = valid["row_id"].map(labels)
    prior = float(candidate_priors(history, season)[args.prior_mode])
    states_before, final_state = season_end_state(history)
    train_priors = prior_before_each_season(history)
    train_e14, train_e14_meta = build_e14_features(
        history, states_before, train_priors, prior, k=E14_K
    )
    valid_e14, valid_e14_meta = build_e14_features(
        valid, {season: final_state}, {season: prior}, prior, k=E14_K
    )
    train_group, valid_group, stage1_details = group_features_for_fold(history, valid)
    s4_train = pd.concat([history[BASE_FEATURES], train_e14], axis=1)
    s4_valid = pd.concat([valid[BASE_FEATURES], valid_e14], axis=1)
    e22_train = pd.concat([s4_train, train_group], axis=1)
    e22_valid = pd.concat([s4_valid, valid_group], axis=1)
    train_y = history[TARGET].to_numpy(dtype=np.int8, copy=False)
    valid_y = valid[TARGET].to_numpy(dtype=np.int8, copy=False)
    baseline: dict[str, np.ndarray] = {}
    candidate: dict[str, np.ndarray] = {}
    details: dict[str, Any] = {"s4": {}, "e22r": {}}
    for name, factory in {"linear": make_linear, "hgb": make_hgb}.items():
        baseline[name], details["s4"][name] = fit_predict(
            f"{season}/s4/{name}", factory, s4_train, train_y, s4_valid
        )
        candidate[name], details["e22r"][name] = fit_predict(
            f"{season}/e22r_probs/{name}", factory, e22_train, train_y, e22_valid
        )
    baseline_blend = 0.9 * baseline["linear"] + 0.1 * baseline["hgb"]
    candidate_blend = 0.9 * candidate["linear"] + 0.1 * candidate["hgb"]
    baseline_summary = metric(valid_y, baseline_blend)
    candidate_summary = metric(valid_y, candidate_blend)
    result = {
        "validation_season": season,
        "history_rows": len(history),
        "valid_rows": len(valid),
        "prior_mode": args.prior_mode,
        "history_labeled_rows": int(history["e22_pitch_type_group"].notna().sum()),
        "valid_labeled_rows_not_used": int(valid["e22_pitch_type_group"].notna().sum()),
        "baseline_s4": baseline_summary,
        "e22r_s4": candidate_summary,
        "e22r_brier_delta": float(candidate_summary["brier"] - baseline_summary["brier"]),
        "e22r_score_delta": float(
            candidate_summary["competition_score"] - baseline_summary["competition_score"]
        ),
        "stage1": stage1_details,
        "e14_train": train_e14_meta,
        "e14_valid": valid_e14_meta,
        "fit_details": details,
        "elapsed_seconds": time.perf_counter() - started,
    }
    print(
        f"[{season}] S4 Brier={baseline_summary['brier']:.8f}, "
        f"E22R-probs Brier={candidate_summary['brier']:.8f}, "
        f"delta={result['e22r_brier_delta']:+.8f}, "
        f"score delta={result['e22r_score_delta']:+.1f}",
        flush=True,
    )
    del history, valid, train_e14, valid_e14, train_group, valid_group
    del s4_train, s4_valid, e22_train, e22_valid, baseline, candidate
    gc.collect()
    return result


def main() -> None:
    args = parse_args()
    started = time.perf_counter()
    labels = load_group_labels()
    frame = load_train(args.data)
    # The baseline loader intentionally omits row_id; restore it in the same
    # file order so the structural label map can be joined without changing
    # any model feature columns.
    row_ids = pd.read_csv(args.data, usecols=["row_id"], dtype="string", encoding="utf-8-sig")[
        "row_id"
    ]
    if len(row_ids) != len(frame):
        raise AssertionError("row_id length does not match optimized train frame")
    frame.insert(0, "row_id", row_ids.to_numpy())
    frame["e22_pitch_type_group"] = frame["row_id"].map(labels)
    folds = [run_fold(frame, season, labels, args) for season in sorted(args.validation_seasons)]
    del frame, labels
    gc.collect()
    deltas = [float(row["e22r_brier_delta"]) for row in folds]
    wins = sum(delta < 0.0 for delta in deltas)
    aggregate = {
        "folds": len(folds),
        "e22r_wins": wins,
        "mean_brier_delta": float(np.mean(deltas)),
        "worst_brier_delta": float(np.max(deltas)),
        "mean_score_delta": float(np.mean([row["e22r_score_delta"] for row in folds])),
        "gate_pass": bool(wins >= 2 and np.max(deltas) <= 0.0005),
        "gate_definition": "wins >= 2/3 and worst Brier delta <= 0.0005",
    }
    payload = {
        "metadata": {
            "created_at_utc": pd.Timestamp.utcnow().isoformat(),
            "data": str(args.data),
            "validation_seasons": sorted(args.validation_seasons),
            "prior_mode": args.prior_mode,
            "protocol": "outer history season < Y; stage-1 pitch-group probabilities from pre-pitch features only",
            "row_independent_inference": True,
            "trackman_usage": "historical pitch_type_group labels only; no current Trackman measurements",
            "stage2": "S4 target model plus four soft pitch-group probabilities",
            "versions": {
                "python": platform.python_version(),
                "numpy": np.__version__,
                "pandas": pd.__version__,
            },
            "command": " ".join(sys.argv),
            "elapsed_seconds": time.perf_counter() - started,
        },
        "aggregate": aggregate,
        "folds": folds,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.output_dir / "e22r_probs_rolling.json"
    csv_path = args.output_dir / "e22r_probs_rolling.csv"
    json_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8"
    )
    pd.DataFrame(
        [
            {
                "validation_season": row["validation_season"],
                "baseline_s4_brier": row["baseline_s4"]["brier"],
                "e22r_s4_brier": row["e22r_s4"]["brier"],
                "e22r_brier_delta": row["e22r_brier_delta"],
                "e22r_score_delta": row["e22r_score_delta"],
                "history_labeled_rows": row["history_labeled_rows"],
            }
            for row in folds
        ]
    ).to_csv(csv_path, index=False)
    print(json.dumps(aggregate, ensure_ascii=False, indent=2), flush=True)
    print(f"Saved {json_path} and {csv_path}.", flush=True)


if __name__ == "__main__":
    main()
