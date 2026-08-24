#!/usr/bin/env python3
"""Finalize the target-blind V5 game-month ablation after its first-fold stop."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.run_v5_h1_residual import cluster_bootstrap_score_gain


PRED = ROOT / "experiments/results/predictions"
PREREG = ROOT / "experiments/params/v5_drop_month_c_preregister.json"
OUTPUT = ROOT / "experiments/results/v5_drop_month_c_selection.json"
YEAR = 2022


def load(name: str) -> dict[str, np.ndarray]:
    with np.load(PRED / name, allow_pickle=False) as archive:
        return {key: np.asarray(archive[key]) for key in archive.files}


def metrics(y: np.ndarray, prediction: np.ndarray, mask: np.ndarray) -> dict[str, float | int]:
    target = np.asarray(y[mask], dtype=np.float64)
    pred = np.asarray(prediction[mask], dtype=np.float64)
    rate = float(target.mean())
    brier = float(np.mean(np.square(pred - target)))
    score = 100_000.0 * (1.0 - brier / (rate * (1.0 - rate)))
    return {
        "rows": int(mask.sum()),
        "target_rate": rate,
        "prediction_mean": float(pred.mean()),
        "prediction_std": float(pred.std()),
        "brier": brier,
        "raw_competition_score": score,
    }


def main() -> None:
    candidate = load("v5_drop_month_c_dev2223_2022.npz")
    anchors = {
        "exact_parent_C": (
            load("v3_sparse_c_backtest_2022.npz"),
            "catboost_outcome",
        ),
        "honest_r_identity": (
            load("v5_honest_m3_r_identity_2022.npz"),
            "final_prediction",
        ),
        "honest_r_grid": (
            load("v5_honest_m3_r_grid_2022.npz"),
            "final_prediction",
        ),
    }
    for artifact, _ in anchors.values():
        for key in ("y", "row_index", "cluster"):
            if not np.array_equal(candidate[key], artifact[key]):
                raise ValueError(f"alignment mismatch: {key}")

    game_type = pd.read_csv(
        ROOT / "open/data/train.csv", usecols=["game_type"], low_memory=False
    )["game_type"].to_numpy()[candidate["row_index"]]
    masks = {
        "all": np.ones(len(game_type), dtype=bool),
        "R": game_type == "R",
    }
    pred = np.asarray(candidate["catboost_outcome"], dtype=np.float64)
    candidate_metrics = {
        scope: metrics(candidate["y"], pred, mask) for scope, mask in masks.items()
    }
    comparisons: dict[str, object] = {}
    for index, (name, (artifact, key)) in enumerate(anchors.items()):
        anchor_pred = np.asarray(artifact[key], dtype=np.float64)
        anchor_metrics = {
            scope: metrics(candidate["y"], anchor_pred, mask)
            for scope, mask in masks.items()
        }
        gains = {
            scope: float(
                candidate_metrics[scope]["raw_competition_score"]
                - anchor_metrics[scope]["raw_competition_score"]
            )
            for scope in masks
        }
        comparisons[name] = {
            "anchor_metrics": anchor_metrics,
            "candidate_metrics": candidate_metrics,
            "score_gains": gains,
            "bootstrap_R": cluster_bootstrap_score_gain(
                candidate["y"],
                anchor_pred,
                pred,
                candidate["cluster"].astype(str),
                masks["R"],
                1000,
                520220 + index,
            ),
        }

    exact = comparisons["exact_parent_C"]
    eligible = bool(
        exact["score_gains"]["R"] > 0.0
        and exact["bootstrap_R"]["ci_low"] > 0.0
    )
    report = {
        "experiment_id": "V5_DROP_MONTH_C",
        "mode": "development_early_stop_after_2022",
        "preregister_sha256": hashlib.sha256(PREREG.read_bytes()).hexdigest(),
        "years_with_completed_predictions_read": [2022],
        "years_not_read_for_selection": [2023, 2024],
        "candidate_change": "drop game_month only from exact component C",
        "candidate_metrics": candidate_metrics,
        "comparisons": comparisons,
        "eligible": eligible,
        "status": "eligible" if eligible else "failed_2022_gate_direction_closed",
        "execution_note": (
            "The combined 2022/2023 training command reached its 1200-second host "
            "timeout during the 2023 fit. The completed 2022 artifact already failed "
            "the preregistered point-gain gate, so 2023 was not rerun and 2024 remains locked."
        ),
    }
    OUTPUT.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({
        "status": report["status"],
        "candidate_R_score": candidate_metrics["R"]["raw_competition_score"],
        "exact_parent_R_gain": exact["score_gains"]["R"],
        "exact_parent_R_ci_low": exact["bootstrap_R"]["ci_low"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
