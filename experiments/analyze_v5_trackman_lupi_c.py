#!/usr/bin/env python3
"""Apply the preregistered 2022 early-stop gate to TrackMan LUPI component C."""

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

from experiments.analyze_v5_semantic_row_c import load, score
from experiments.run_v5_h1_residual import cluster_bootstrap_score_gain


PREDICTIONS = ROOT / "experiments/results/predictions"
PREREGISTRATION = ROOT / "experiments/params/v5_trackman_lupi_c_preregister.json"
OUTPUT = ROOT / "experiments/results/v5_trackman_lupi_c_selection.json"


def main() -> None:
    candidate = load("v5_trackman_lupi_c_2022_2022.npz")
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
    regular = game_type == "R"
    all_rows = np.ones(len(regular), dtype=bool)
    prediction = np.asarray(candidate["catboost_outcome"], dtype=np.float64)
    exact_prediction = np.asarray(
        anchors["exact_parent_C"][0]["catboost_outcome"], dtype=np.float64
    )
    routed_prediction = exact_prediction.copy()
    routed_prediction[regular] = prediction[regular]
    candidate_metrics = {
        "R": score(candidate["y"], prediction, regular),
        "all_raw": score(candidate["y"], prediction, all_rows),
        "all_routed": score(candidate["y"], routed_prediction, all_rows),
    }

    comparisons: dict[str, object] = {}
    for index, (name, (artifact, key)) in enumerate(anchors.items()):
        anchor_prediction = np.asarray(artifact[key], dtype=np.float64)
        anchor_r = score(candidate["y"], anchor_prediction, regular)
        comparisons[name] = {
            "anchor_R_metrics": anchor_r,
            "score_gain_R": float(
                candidate_metrics["R"]["raw_competition_score"]
                - anchor_r["raw_competition_score"]
            ),
            "bootstrap_R": cluster_bootstrap_score_gain(
                candidate["y"],
                anchor_prediction,
                prediction,
                candidate["cluster"].astype(str),
                regular,
                1000,
                581220 + index,
            ),
        }
    comparisons["exact_parent_C"]["routed_full_score_gain"] = float(
        candidate_metrics["all_routed"]["raw_competition_score"]
        - score(candidate["y"], exact_prediction, all_rows)["raw_competition_score"]
    )

    passed = {
        name: bool(
            details["score_gain_R"] > 0.0
            and details["bootstrap_R"]["ci_low"] > 0.0
        )
        for name, details in comparisons.items()
    }
    eligible = all(passed.values())
    report = {
        "experiment_id": "V5_TRACKMAN_LUPI_C_V1",
        "mode": "development_early_stop_after_2022",
        "preregister_sha256": hashlib.sha256(PREREGISTRATION.read_bytes()).hexdigest(),
        "years_with_completed_predictions_read": [2022],
        "years_not_read_for_selection": [2023, 2024],
        "f_rows_routed_to_exact_parent": True,
        "candidate_metrics": candidate_metrics,
        "comparisons": comparisons,
        "passed_comparisons": passed,
        "eligible": eligible,
        "status": "eligible_for_2023" if eligible else "failed_2022_gate_direction_closed",
        "conclusion": (
            "Proceed to the unchanged 2023 recipe."
            if eligible
            else "Do not train or inspect 2023/2024 for this LUPI recipe."
        ),
    }
    if OUTPUT.exists():
        raise FileExistsError(f"Selection report already exists: {OUTPUT}")
    OUTPUT.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({
        "status": report["status"],
        "candidate_R_score": candidate_metrics["R"]["raw_competition_score"],
        "routed_full_score": candidate_metrics["all_routed"]["raw_competition_score"],
        "comparisons": {
            name: {
                "gain": details["score_gain_R"],
                "ci_low": details["bootstrap_R"]["ci_low"],
                "ci_high": details["bootstrap_R"]["ci_high"],
                "pass": passed[name],
            }
            for name, details in comparisons.items()
        },
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
