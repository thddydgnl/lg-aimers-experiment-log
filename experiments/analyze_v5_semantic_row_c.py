#!/usr/bin/env python3
"""Apply the preregistered 2022 early-stop gate to semantic-row component C."""

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


PREDICTIONS = ROOT / "experiments/results/predictions"
PREREGISTRATION = ROOT / "experiments/params/v5_semantic_row_c_preregister.json"
OUTPUT = ROOT / "experiments/results/v5_semantic_row_c_selection.json"
YEAR = 2022


def load(name: str) -> dict[str, np.ndarray]:
    with np.load(PREDICTIONS / name, allow_pickle=False) as archive:
        return {key: np.asarray(archive[key]) for key in archive.files}


def score(y: np.ndarray, prediction: np.ndarray, mask: np.ndarray) -> dict[str, float | int]:
    target = np.asarray(y[mask], dtype=np.float64)
    pred = np.asarray(prediction[mask], dtype=np.float64)
    target_rate = float(target.mean())
    brier = float(np.mean(np.square(pred - target)))
    return {
        "rows": int(mask.sum()),
        "target_rate": target_rate,
        "prediction_mean": float(pred.mean()),
        "prediction_std": float(pred.std()),
        "brier": brier,
        "raw_competition_score": float(
            100_000.0 * (1.0 - brier / (target_rate * (1.0 - target_rate)))
        ),
    }


def main() -> None:
    candidate = load("v5_semantic_row_c_2022_2022.npz")
    anchors = {
        "exact_parent_C": (load("v3_sparse_c_backtest_2022.npz"), "catboost_outcome"),
        "honest_r_identity": (
            load("v5_honest_m3_r_identity_2022.npz"),
            "final_prediction",
        ),
        "honest_r_grid": (
            load("v5_honest_m3_r_grid_2022.npz"),
            "final_prediction",
        ),
    }
    for name, (artifact, _) in anchors.items():
        for key in ("y", "row_index", "cluster"):
            if not np.array_equal(candidate[key], artifact[key]):
                raise ValueError(f"alignment mismatch: {name}/{key}")

    game_type = pd.read_csv(
        ROOT / "open/data/train.csv", usecols=["game_type"], low_memory=False
    )["game_type"].to_numpy()[candidate["row_index"]]
    masks = {
        "all": np.ones(len(game_type), dtype=bool),
        "R": game_type == "R",
    }
    prediction = np.asarray(candidate["catboost_outcome"], dtype=np.float64)
    candidate_metrics = {
        scope: score(candidate["y"], prediction, mask)
        for scope, mask in masks.items()
    }

    comparisons: dict[str, object] = {}
    for index, (name, (artifact, key)) in enumerate(anchors.items()):
        anchor_prediction = np.asarray(artifact[key], dtype=np.float64)
        anchor_metrics = {
            scope: score(candidate["y"], anchor_prediction, mask)
            for scope, mask in masks.items()
        }
        comparisons[name] = {
            "anchor_metrics": anchor_metrics,
            "score_gains": {
                scope: float(
                    candidate_metrics[scope]["raw_competition_score"]
                    - anchor_metrics[scope]["raw_competition_score"]
                )
                for scope in masks
            },
            "bootstrap_R": cluster_bootstrap_score_gain(
                candidate["y"],
                anchor_prediction,
                prediction,
                candidate["cluster"].astype(str),
                masks["R"],
                1000,
                580220 + index,
            ),
        }

    passed_comparisons = {
        name: bool(
            details["score_gains"]["R"] > 0.0
            and details["bootstrap_R"]["ci_low"] > 0.0
        )
        for name, details in comparisons.items()
    }
    eligible = all(passed_comparisons.values())
    report = {
        "experiment_id": "V5_SEMANTIC_ROW_C",
        "mode": "development_early_stop_after_2022",
        "preregister_sha256": hashlib.sha256(PREREGISTRATION.read_bytes()).hexdigest(),
        "years_with_completed_predictions_read": [2022],
        "years_not_read_for_selection": [2023, 2024],
        "candidate_metrics": candidate_metrics,
        "comparisons": comparisons,
        "passed_comparisons": passed_comparisons,
        "eligible": eligible,
        "status": "eligible_for_2023" if eligible else "failed_2022_gate_direction_closed",
        "conclusion": (
            "Proceed to the fixed 2023 recipe."
            if eligible
            else "Do not train or inspect 2023/2024 for this feature bundle."
        ),
    }
    if OUTPUT.exists():
        raise FileExistsError(f"Selection report already exists: {OUTPUT}")
    OUTPUT.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({
        "status": report["status"],
        "candidate_R_score": candidate_metrics["R"]["raw_competition_score"],
        "comparisons": {
            name: {
                "gain": details["score_gains"]["R"],
                "ci_low": details["bootstrap_R"]["ci_low"],
                "ci_high": details["bootstrap_R"]["ci_high"],
                "pass": passed_comparisons[name],
            }
            for name, details in comparisons.items()
        },
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
