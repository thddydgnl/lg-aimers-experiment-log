#!/usr/bin/env python3
"""Evaluate the explicit E22R soft mixture ``sum_g P(g|x)P(y|g,x)``."""

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

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.run_baselines import FEATURES as BASE_FEATURES  # noqa: E402
from experiments.run_baselines import TARGET, load_train  # noqa: E402
from experiments.run_e14_rolling import (  # noqa: E402
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
from experiments.run_e22r_probs_rolling import (  # noqa: E402
    E22_FEATURES,
    GROUPS,
    align_group_probabilities,
    load_group_labels,
    make_stage1,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=ROOT / "open/data/train.csv")
    parser.add_argument(
        "--validation-seasons", nargs="+", type=int, default=[2022, 2023, 2024]
    )
    parser.add_argument("--output-dir", type=Path, default=ROOT / "experiments/results")
    parser.add_argument("--prior-mode", default="r_recent3")
    return parser.parse_args()


def add_row_ids(frame: pd.DataFrame, path: Path) -> pd.DataFrame:
    row_ids = pd.read_csv(path, usecols=["row_id"], dtype="string", encoding="utf-8-sig")[
        "row_id"
    ]
    if len(row_ids) != len(frame):
        raise AssertionError("row_id length does not match optimized train frame")
    frame = frame.copy()
    frame.insert(0, "row_id", row_ids.to_numpy())
    return frame


def stage1_probabilities(history: pd.DataFrame, valid: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    labeled = history.dropna(subset=["e22_pitch_type_group"])
    stage1 = make_stage1()
    started = time.perf_counter()
    stage1.fit(labeled[E22_FEATURES], labeled["e22_pitch_type_group"].astype(str))
    train_probs = align_group_probabilities(stage1, history)
    valid_probs = align_group_probabilities(stage1, valid)
    details = {
        "labeled_rows": int(len(labeled)),
        "classes": [str(v) for v in stage1.named_steps["clf"].classes_],
        "fit_seconds": time.perf_counter() - started,
        "uses_current_trackman": False,
        "valid_labels_not_used": int(valid["e22_pitch_type_group"].notna().sum()),
    }
    del stage1
    gc.collect()
    return train_probs, valid_probs, details


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
    train_e14, train_meta = build_e14_features(
        history, states_before, train_priors, prior, k=E14_K
    )
    valid_e14, valid_meta = build_e14_features(
        valid, {season: final_state}, {season: prior}, prior, k=E14_K
    )
    s4_train = pd.concat([history[BASE_FEATURES], train_e14], axis=1)
    s4_valid = pd.concat([valid[BASE_FEATURES], valid_e14], axis=1)
    train_probs, valid_probs, stage1_details = stage1_probabilities(history, valid)
    train_y = history[TARGET].to_numpy(dtype=np.int8, copy=False)
    valid_y = valid[TARGET].to_numpy(dtype=np.int8, copy=False)

    # Baseline S4 blend.
    baseline_predictions: dict[str, np.ndarray] = {}
    baseline_details: dict[str, Any] = {}
    for name, factory in {"linear": make_linear, "hgb": make_hgb}.items():
        baseline_predictions[name], baseline_details[name] = fit_predict(
            f"{season}/s4/{name}", factory, s4_train, train_y, s4_valid
        )

    mixture_predictions: dict[str, np.ndarray] = {}
    mixture_details: dict[str, Any] = {}
    labeled_mask = history["e22_pitch_type_group"].notna().to_numpy(dtype=bool)
    for name, factory in {"linear": make_linear, "hgb": make_hgb}.items():
        per_group: list[np.ndarray] = []
        group_meta: dict[str, Any] = {}
        for group in GROUPS:
            group_mask = labeled_mask & history["e22_pitch_type_group"].eq(group).to_numpy(
                dtype=bool, na_value=False
            )
            if int(group_mask.sum()) < 100:
                raise ValueError(f"Too few labeled rows for group {group}: {int(group_mask.sum())}")
            prediction, fit_meta = fit_predict(
                f"{season}/mixture/{name}/{group}",
                factory,
                s4_train.loc[group_mask],
                train_y[group_mask],
                s4_valid,
            )
            per_group.append(prediction)
            group_meta[group] = {"rows": int(group_mask.sum()), **fit_meta}
        stacked = np.column_stack(per_group)
        mixture_predictions[name] = np.sum(valid_probs * stacked, axis=1)
        mixture_details[name] = group_meta
        del per_group, stacked
        gc.collect()

    baseline_blend = 0.9 * baseline_predictions["linear"] + 0.1 * baseline_predictions["hgb"]
    mixture_blend = 0.9 * mixture_predictions["linear"] + 0.1 * mixture_predictions["hgb"]
    baseline_summary = metric(valid_y, baseline_blend)
    mixture_summary = metric(valid_y, mixture_blend)
    result = {
        "validation_season": season,
        "history_rows": len(history),
        "valid_rows": len(valid),
        "prior_mode": args.prior_mode,
        "history_labeled_rows": int(labeled_mask.sum()),
        "baseline_s4": baseline_summary,
        "e22r_mixture_s4": mixture_summary,
        "e22r_mixture_brier_delta": float(mixture_summary["brier"] - baseline_summary["brier"]),
        "e22r_mixture_score_delta": float(
            mixture_summary["competition_score"] - baseline_summary["competition_score"]
        ),
        "stage1": stage1_details,
        "group_rows": {
            group: int((labeled_mask & history["e22_pitch_type_group"].eq(group).to_numpy(dtype=bool, na_value=False)).sum())
            for group in GROUPS
        },
        "fit_details": {"baseline": baseline_details, "mixture": mixture_details},
        "e14_train": train_meta,
        "e14_valid": valid_meta,
        "elapsed_seconds": time.perf_counter() - started,
    }
    print(
        f"[{season}] S4 Brier={baseline_summary['brier']:.8f}, "
        f"mixture Brier={mixture_summary['brier']:.8f}, "
        f"delta={result['e22r_mixture_brier_delta']:+.8f}, "
        f"score delta={result['e22r_mixture_score_delta']:+.1f}",
        flush=True,
    )
    del history, valid, train_e14, valid_e14, s4_train, s4_valid
    del baseline_predictions, mixture_predictions
    gc.collect()
    return result


def main() -> None:
    args = parse_args()
    started = time.perf_counter()
    labels = load_group_labels()
    frame = add_row_ids(load_train(args.data), args.data)
    folds = [run_fold(frame, season, labels, args) for season in sorted(args.validation_seasons)]
    del frame, labels
    gc.collect()
    deltas = [float(row["e22r_mixture_brier_delta"]) for row in folds]
    wins = sum(delta < 0.0 for delta in deltas)
    aggregate = {
        "folds": len(folds),
        "e22r_mixture_wins": wins,
        "mean_brier_delta": float(np.mean(deltas)),
        "worst_brier_delta": float(np.max(deltas)),
        "mean_score_delta": float(
            np.mean([row["e22r_mixture_score_delta"] for row in folds])
        ),
        "gate_pass": bool(wins >= 2 and np.max(deltas) <= 0.0005),
        "gate_definition": "wins >= 2/3 and worst Brier delta <= 0.0005",
    }
    payload = {
        "metadata": {
            "created_at_utc": pd.Timestamp.utcnow().isoformat(),
            "data": str(args.data),
            "validation_seasons": sorted(args.validation_seasons),
            "prior_mode": args.prior_mode,
            "protocol": "outer history season < Y; group-specific S4 models mixed by stage-1 soft probabilities",
            "row_independent_inference": True,
            "trackman_usage": "historical pitch_type_group labels only; no current Trackman measurements",
            "stage1": "pre-pitch features only; validation labels excluded",
            "stage2": "four group-specific S4 target models; explicit probability marginalization",
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
    json_path = args.output_dir / "e22r_mixture_rolling.json"
    csv_path = args.output_dir / "e22r_mixture_rolling.csv"
    json_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8"
    )
    pd.DataFrame(
        [
            {
                "validation_season": row["validation_season"],
                "baseline_s4_brier": row["baseline_s4"]["brier"],
                "e22r_mixture_s4_brier": row["e22r_mixture_s4"]["brier"],
                "e22r_mixture_brier_delta": row["e22r_mixture_brier_delta"],
                "e22r_mixture_score_delta": row["e22r_mixture_score_delta"],
                "history_labeled_rows": row["history_labeled_rows"],
            }
            for row in folds
        ]
    ).to_csv(csv_path, index=False)
    print(json.dumps(aggregate, ensure_ascii=False, indent=2), flush=True)
    print(f"Saved {json_path} and {csv_path}.", flush=True)


if __name__ == "__main__":
    main()
