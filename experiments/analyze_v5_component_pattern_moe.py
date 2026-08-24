#!/usr/bin/env python3
"""Immutable source gate for the coherent component-pattern MoE."""

from __future__ import annotations

import json
from pathlib import Path
import sys
from typing import Any

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
PREREG = ROOT / "experiments/params/v5_component_pattern_moe_preregister.json"
REPORT = RESULTS / "v5_component_pattern_moe_source_gate.json"
YEARS = (2020, 2021)
KEY = "catboost_component_pattern_moe"


def main() -> None:
    if REPORT.exists():
        raise FileExistsError(f"immutable result already exists: {REPORT}")
    prereg = json.loads(PREREG.read_text(encoding="utf-8"))
    all_types = pd.read_csv(TRAIN, usecols=["game_type"])["game_type"].astype(str)
    folds: dict[int, dict[str, Any]] = {}
    semantic_pass = True
    expected_classes = sorted(prereg["candidate"]["pattern_classes"])
    expected_experts = sorted(prereg["candidate"]["success_eligible_patterns"])
    for year in YEARS:
        stage = f"v5_component_pattern_moe_source{year}"
        candidate_path = PRED / f"{stage}_{year}.npz"
        parent_path = PRED / f"v4_m3_c_backtest_{year}_{year}.npz"
        candidate = load(candidate_path)
        parent_artifact = load(parent_path)
        for key in ("y", "row_index", "cluster"):
            if not np.array_equal(candidate[key], parent_artifact[key]):
                raise ValueError(f"alignment mismatch: {year}/{key}")
        regular = all_types.iloc[
            candidate["row_index"].astype(np.int64)
        ].to_numpy(dtype=str) == "R"
        stage_report = json.loads(
            (RESULTS / f"{stage}.json").read_text(encoding="utf-8")
        )
        details = stage_report["folds"][0]["fit_details"][KEY]
        semantic = {
            "history_pattern_coverage": float(
                details["history_pattern_coverage"]
            ),
            "pattern_classes": sorted(details["pattern_classes"]),
            "success_eligible_patterns": sorted(
                details["success_eligible_patterns"]
            ),
            "expert_rows": {
                name: int(value["fit_rows"])
                for name, value in details["experts"].items()
            },
            "current_validation_pattern_used": bool(
                details["current_validation_pattern_used"]
            ),
            "row_independent_inference": bool(
                details["row_independent_inference"]
            ),
        }
        fold_semantic = bool(
            semantic["history_pattern_coverage"]
            >= float(
                prereg["semantic_gate"][
                    "minimum_history_pattern_coverage_each_year"
                ]
            )
            and semantic["pattern_classes"] == expected_classes
            and semantic["success_eligible_patterns"] == expected_experts
            and min(semantic["expert_rows"].values())
            >= int(prereg["semantic_gate"]["minimum_rows_per_success_expert"])
            and not semantic["current_validation_pattern_used"]
            and semantic["row_independent_inference"]
        )
        semantic["pass"] = fold_semantic
        semantic_pass &= fold_semantic
        folds[year] = {
            "candidate": candidate,
            "parent": parent_artifact["catboost_outcome"].astype(np.float64),
            "regular": regular,
            "full": np.ones(len(regular), dtype=bool),
            "semantic": semantic,
            "candidate_path": candidate_path,
            "parent_path": parent_path,
        }

    trials = []
    cache: dict[tuple[int, float], np.ndarray] = {}
    if semantic_pass:
        for gamma in prereg["candidate"]["top_level_blend_grid"]:
            metrics: dict[str, Any] = {}
            for year in YEARS:
                fold = folds[year]
                metrics[str(year)] = evaluate(
                    fold["candidate"],
                    fold["parent"],
                    fold["candidate"][KEY].astype(np.float64),
                    fold["full"],
                    {"full": fold["full"], "R": fold["regular"]},
                    float(gamma),
                    int(prereg["bootstrap_iterations"]),
                    1710000 + 10000 * year + int(float(gamma) * 100),
                )
                cache[(year, float(gamma))] = np.clip(
                    fold["parent"]
                    + float(gamma)
                    * (
                        fold["candidate"][KEY].astype(np.float64)
                        - fold["parent"]
                    ),
                    1e-6,
                    1.0 - 1e-6,
                )
            full_gains = [
                metrics[str(year)]["routes"]["full"]["gain"] for year in YEARS
            ]
            r_gains = [
                metrics[str(year)]["routes"]["R"]["gain"] for year in YEARS
            ]
            trials.append(
                {
                    "gamma": float(gamma),
                    "minimum_full_gain": float(min(full_gains)),
                    "minimum_R_gain": float(min(r_gains)),
                    "mean_full_gain": float(np.mean(full_gains)),
                    "years": metrics,
                }
            )
    selected = max(
        trials,
        key=lambda item: (
            item["minimum_full_gain"], item["minimum_R_gain"], -item["gamma"]
        ),
    )
    minimum_full = float(prereg["source_gate"]["minimum_full_gain_each_year"])
    minimum_r = float(prereg["source_gate"]["minimum_r_gain_each_year"])
    checks = [semantic_pass]
    for year in YEARS:
        route = selected["years"][str(year)]["routes"]
        checks.extend(
            [
                route["full"]["gain"] >= minimum_full,
                route["R"]["gain"] >= minimum_r,
                route["full"]["pitcher_cluster_95_ci"]["ci_low"] > 0.0,
                route["R"]["pitcher_cluster_95_ci"]["ci_low"] > 0.0,
            ]
        )
    passed = bool(all(checks))
    artifacts: dict[str, Any] = {}
    for year in YEARS:
        output = PRED / f"v5_component_pattern_moe_source_{year}.npz"
        if output.exists():
            raise FileExistsError(f"immutable artifact exists: {output}")
        fold = folds[year]
        np.savez_compressed(
            output,
            y=fold["candidate"]["y"].astype(np.int8),
            row_index=fold["candidate"]["row_index"].astype(np.int64),
            cluster=fold["candidate"]["cluster"],
            parent_exact_c=fold["parent"],
            component_pattern_moe_raw=fold["candidate"][KEY].astype(np.float64),
            final_prediction=cache[(year, selected["gamma"])],
        )
        artifacts[str(year)] = {
            "path": str(output.relative_to(ROOT)),
            "sha256": digest(output),
        }
    report = {
        "experiment_id": prereg["experiment_id"],
        "status": "source_pass" if passed else "source_failed",
        "preregister_sha256": digest(PREREG),
        "script_sha256": digest(Path(__file__)),
        "years_read": list(YEARS),
        "years_not_read": [2022, 2023, 2024],
        "semantic": {str(year): folds[year]["semantic"] for year in YEARS},
        "trials": trials,
        "selected": selected,
        "source_gate": {
            "minimum_full_gain_each_year": minimum_full,
            "minimum_R_gain_each_year": minimum_r,
            "pass": passed,
        },
        "artifacts": artifacts,
        "goal_status": "active",
        "goal_completion_claimed": False,
    }
    REPORT.write_text(
        json.dumps(safe(report), ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(
        json.dumps(
            safe({"status": report["status"], "selected": selected}),
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
