#!/usr/bin/env python3
"""Reusable immutable 2022 early gate against exact C and honest R anchors."""

from __future__ import annotations

import argparse
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-id", required=True)
    parser.add_argument("--candidate-stage", required=True)
    parser.add_argument("--candidate-key", default="catboost_outcome")
    parser.add_argument("--preregister", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=582220)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    candidate = load(f"{args.candidate_stage}_2022.npz")
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
    prediction = np.asarray(candidate[args.candidate_key], dtype=np.float64)
    exact_prediction = np.asarray(
        anchors["exact_parent_C"][0]["catboost_outcome"], dtype=np.float64
    )
    routed = exact_prediction.copy()
    routed[regular] = prediction[regular]
    candidate_metrics = {
        "R": score(candidate["y"], prediction, regular),
        "all_raw": score(candidate["y"], prediction, all_rows),
        "all_routed": score(candidate["y"], routed, all_rows),
    }

    comparisons = {}
    for index, (name, (artifact, key)) in enumerate(anchors.items()):
        anchor_prediction = np.asarray(artifact[key], dtype=np.float64)
        anchor_metrics = score(candidate["y"], anchor_prediction, regular)
        comparisons[name] = {
            "anchor_R_metrics": anchor_metrics,
            "score_gain_R": float(
                candidate_metrics["R"]["raw_competition_score"]
                - anchor_metrics["raw_competition_score"]
            ),
            "bootstrap_R": cluster_bootstrap_score_gain(
                candidate["y"],
                anchor_prediction,
                prediction,
                candidate["cluster"].astype(str),
                regular,
                1000,
                args.seed + index,
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
    preregister = args.preregister.resolve()
    output = args.output.resolve()
    report = {
        "experiment_id": args.experiment_id,
        "mode": "development_early_stop_after_2022",
        "candidate_stage": args.candidate_stage,
        "candidate_key": args.candidate_key,
        "preregister": str(preregister.relative_to(ROOT)),
        "preregister_sha256": hashlib.sha256(preregister.read_bytes()).hexdigest(),
        "years_with_completed_predictions_read": [2022],
        "years_not_read_for_selection": [2023, 2024],
        "f_rows_routed_to_exact_parent": True,
        "candidate_metrics": candidate_metrics,
        "comparisons": comparisons,
        "passed_comparisons": passed,
        "eligible": eligible,
        "status": "eligible_for_2023" if eligible else "failed_2022_gate_direction_closed",
    }
    if output.exists():
        raise FileExistsError(f"Selection report already exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
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
