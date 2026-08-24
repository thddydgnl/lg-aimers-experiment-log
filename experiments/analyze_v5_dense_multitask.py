#!/usr/bin/env python3
"""Immutable early source gate for dense auxiliary MultiRMSE."""

from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.analyze_v5_dense_pitchtype_moe import (  # noqa: E402
    digest,
    evaluate,
    load,
    safe,
)


RESULTS = ROOT / "experiments/results"
PRED = RESULTS / "predictions"
TRAIN = ROOT / "open/data/train.csv"
PREREG = ROOT / "experiments/params/v5_dense_multitask_preregister.json"
REPORT = RESULTS / "v5_dense_multitask_source_gate.json"
YEAR = 2020
STAGE = "v5_dense_multitask_source2020"
KEY = "catboost_dense_multitask"


def main() -> None:
    if REPORT.exists():
        raise FileExistsError(f"immutable result already exists: {REPORT}")
    prereg = json.loads(PREREG.read_text(encoding="utf-8"))
    candidate_path = PRED / f"{STAGE}_{YEAR}.npz"
    parent_path = PRED / "v4_m3_c_backtest_2020_2020.npz"
    candidate = load(candidate_path)
    parent_artifact = load(parent_path)
    for key in ("y", "row_index", "cluster"):
        if not np.array_equal(candidate[key], parent_artifact[key]):
            raise ValueError(f"alignment mismatch: {key}")
    types = pd.read_csv(TRAIN, usecols=["game_type"])["game_type"].astype(str)
    regular = types.iloc[
        candidate["row_index"].astype(np.int64)
    ].to_numpy(dtype=str) == "R"
    full = np.ones(len(regular), dtype=bool)
    stage_report = json.loads(
        (RESULTS / f"{STAGE}.json").read_text(encoding="utf-8")
    )
    details = stage_report["folds"][0]["fit_details"][KEY]
    semantic = {
        "history_usable_coverage": float(details["history_usable_coverage"]),
        "target_heads": details["target_heads"],
        "success_scale": float(details["success_scale"]),
        "current_pitch_group_used_at_inference": bool(
            details["current_pitch_group_used_at_inference"]
        ),
        "validation_auxiliary_labels_used": bool(
            details["validation_auxiliary_labels_used"]
        ),
        "row_independent_inference": bool(details["row_independent_inference"]),
    }
    semantic_pass = bool(
        semantic["history_usable_coverage"]
        >= float(
            prereg["semantic_gate"][
                "minimum_history_usable_coverage_each_year"
            ]
        )
        and len(semantic["target_heads"])
        == int(prereg["semantic_gate"]["expected_head_count"])
        and semantic["success_scale"]
        == float(prereg["semantic_gate"]["success_scale_exact"])
        and not semantic["current_pitch_group_used_at_inference"]
        and not semantic["validation_auxiliary_labels_used"]
        and semantic["row_independent_inference"]
    )
    semantic["pass"] = semantic_pass
    parent = parent_artifact["catboost_outcome"].astype(np.float64)
    direction = candidate[KEY].astype(np.float64)
    trials = []
    if semantic_pass:
        for gamma in prereg["candidate"]["top_level_blend_grid"]:
            trial = evaluate(
                candidate,
                parent,
                direction,
                full,
                {"full": full, "R": regular},
                float(gamma),
                int(prereg["bootstrap_iterations"]),
                1510000 + int(float(gamma) * 100),
            )
            trial["gamma"] = float(gamma)
            trials.append(trial)
    minimum_full = float(prereg["source_gate"]["minimum_full_gain_each_year"])
    minimum_r = float(prereg["source_gate"]["minimum_r_gain_each_year"])
    for trial in trials:
        routes = trial["routes"]
        trial["passes_2020"] = bool(
            routes["full"]["gain"] >= minimum_full
            and routes["R"]["gain"] >= minimum_r
            and routes["full"]["pitcher_cluster_95_ci"]["ci_low"] > 0.0
            and routes["R"]["pitcher_cluster_95_ci"]["ci_low"] > 0.0
        )
    any_pass = bool(any(trial["passes_2020"] for trial in trials))
    if any_pass:
        raise AssertionError("2021 must be run because a gamma passes 2020")
    selected = max(
        trials,
        key=lambda item: (
            item["routes"]["R"]["gain"],
            item["routes"]["full"]["gain"],
            -item["gamma"],
        ),
    )
    report = {
        "experiment_id": prereg["experiment_id"],
        "status": "source_failed_early_2020",
        "preregister_sha256": digest(PREREG),
        "script_sha256": digest(Path(__file__)),
        "years_read": [2020],
        "years_not_read": [2021, 2022, 2023, 2024],
        "early_stop_reason": (
            "No preregistered gamma satisfies the mandatory 2020 full/R "
            "gain and cluster-CI conjunction, so 2021 cannot rescue the gate."
        ),
        "semantic": semantic,
        "trials": trials,
        "selected_diagnostic_not_locked": selected,
        "source_gate": {
            "minimum_full_gain_each_year": minimum_full,
            "minimum_R_gain_each_year": minimum_r,
            "pass": False,
        },
        "artifacts": {
            "parent": str(parent_path.relative_to(ROOT)),
            "candidate": str(candidate_path.relative_to(ROOT)),
            "candidate_sha256": digest(candidate_path),
        },
        "goal_status": "active",
        "goal_completion_claimed": False,
    }
    REPORT.write_text(
        json.dumps(safe(report), ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(safe(report), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
