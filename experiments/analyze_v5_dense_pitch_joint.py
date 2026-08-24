#!/usr/bin/env python3
"""Immutable two-year source gate for the dense-pitch joint outcome model."""

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
PREREG = ROOT / "experiments/params/v5_dense_pitch_joint_preregister.json"
REPORT = RESULTS / "v5_dense_pitch_joint_source_gate.json"
YEARS = (2020, 2021)
STAGES = {year: f"v5_dense_pitch_joint_source{year}" for year in YEARS}
PARENTS = {year: f"v4_m3_c_backtest_{year}_{year}.npz" for year in YEARS}
KEY = "catboost_dense_pitch_joint"


def main() -> None:
    if REPORT.exists():
        raise FileExistsError(f"immutable result already exists: {REPORT}")
    prereg = json.loads(PREREG.read_text(encoding="utf-8"))
    all_types = pd.read_csv(TRAIN, usecols=["game_type"])["game_type"].astype(str)
    expected_classes = sorted(
        f"{outcome}|pitch={group}"
        for outcome in prereg["candidate"]["outcome_classes"]
        for group in prereg["candidate"]["pitch_groups"]
    )
    folds: dict[int, dict[str, Any]] = {}
    semantic_pass = True
    for year in YEARS:
        candidate_path = PRED / f"{STAGES[year]}_{year}.npz"
        parent_path = PRED / PARENTS[year]
        candidate = load(candidate_path)
        parent_artifact = load(parent_path)
        for key in ("y", "row_index", "cluster"):
            if not np.array_equal(candidate[key], parent_artifact[key]):
                raise ValueError(f"alignment mismatch: {year}/{key}")
        regular = all_types.iloc[
            candidate["row_index"].astype(np.int64)
        ].to_numpy(dtype=str) == "R"
        metadata = json.loads(
            (RESULTS / f"{STAGES[year]}.json").read_text(encoding="utf-8")
        )
        details = metadata["folds"][0]["fit_details"][KEY]
        semantic = {
            "history_dense_group_coverage": float(
                details["dense_group_coverage_all"]
            ),
            "history_dense_group_coverage_R": float(
                details["dense_group_coverage_R"]
            ),
            "joint_classes": details["joint_classes"],
            "expected_joint_classes": expected_classes,
            "joint_usable_rows": int(details["joint_usable_rows"]),
            "joint_dropped_rows": int(details["joint_dropped_rows"]),
            "current_pitch_group_used_at_inference": bool(
                details["current_pitch_group_used_at_inference"]
            ),
            "separate_selector_used": bool(details["separate_selector_used"]),
            "row_independent_inference": bool(
                details["row_independent_inference"]
            ),
        }
        fold_semantic = bool(
            semantic["history_dense_group_coverage"]
            >= float(
                prereg["semantic_gate"][
                    "minimum_history_dense_group_coverage_each_year"
                ]
            )
            and semantic["joint_classes"] == expected_classes
            and not semantic["current_pitch_group_used_at_inference"]
            and not semantic["separate_selector_used"]
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

    trials: list[dict[str, Any]] = []
    prediction_cache: dict[tuple[int, float], np.ndarray] = {}
    if semantic_pass:
        for gamma in prereg["candidate"]["top_level_blend_grid"]:
            year_metrics: dict[str, Any] = {}
            for year in YEARS:
                fold = folds[year]
                evaluated = evaluate(
                    fold["candidate"],
                    fold["parent"],
                    fold["candidate"][KEY].astype(np.float64),
                    fold["full"],
                    {"full": fold["full"], "R": fold["regular"]},
                    float(gamma),
                    int(prereg["bootstrap_iterations"]),
                    1410000 + 10000 * year + int(float(gamma) * 100),
                )
                prediction_cache[(year, float(gamma))] = np.clip(
                    fold["parent"]
                    + float(gamma)
                    * (
                        fold["candidate"][KEY].astype(np.float64)
                        - fold["parent"]
                    ),
                    1e-6,
                    1.0 - 1e-6,
                )
                year_metrics[str(year)] = evaluated
            full_gains = [
                year_metrics[str(year)]["routes"]["full"]["gain"]
                for year in YEARS
            ]
            r_gains = [
                year_metrics[str(year)]["routes"]["R"]["gain"]
                for year in YEARS
            ]
            trials.append(
                {
                    "gamma": float(gamma),
                    "minimum_full_gain": float(min(full_gains)),
                    "minimum_R_gain": float(min(r_gains)),
                    "mean_full_gain": float(np.mean(full_gains)),
                    "years": year_metrics,
                }
            )

    selected = (
        max(
            trials,
            key=lambda item: (
                item["minimum_full_gain"],
                item["minimum_R_gain"],
                -item["gamma"],
            ),
        )
        if trials
        else None
    )
    minimum_full = float(prereg["source_gate"]["minimum_full_gain_each_year"])
    minimum_r = float(prereg["source_gate"]["minimum_r_gain_each_year"])
    checks = [semantic_pass, selected is not None]
    if selected is not None:
        for year in YEARS:
            routes = selected["years"][str(year)]["routes"]
            checks.extend(
                [
                    routes["full"]["gain"] >= minimum_full,
                    routes["R"]["gain"] >= minimum_r,
                    routes["full"]["pitcher_cluster_95_ci"]["ci_low"] > 0.0,
                    routes["R"]["pitcher_cluster_95_ci"]["ci_low"] > 0.0,
                ]
            )
    passed = bool(all(checks))

    artifacts: dict[str, Any] = {}
    if selected is not None:
        for year in YEARS:
            output = PRED / f"v5_dense_pitch_joint_source_{year}.npz"
            if output.exists():
                raise FileExistsError(f"immutable artifact exists: {output}")
            fold = folds[year]
            np.savez_compressed(
                output,
                y=fold["candidate"]["y"].astype(np.int8),
                row_index=fold["candidate"]["row_index"].astype(np.int64),
                cluster=fold["candidate"]["cluster"],
                parent_exact_c=fold["parent"],
                dense_pitch_joint_raw=fold["candidate"][KEY].astype(np.float64),
                final_prediction=prediction_cache[(year, selected["gamma"])],
            )
            artifacts[str(year)] = {
                "path": str(output.relative_to(ROOT)),
                "sha256": digest(output),
                "parent": str(fold["parent_path"].relative_to(ROOT)),
                "raw_candidate": str(
                    fold["candidate_path"].relative_to(ROOT)
                ),
            }

    report = {
        "experiment_id": prereg["experiment_id"],
        "status": "source_pass" if passed else "source_failed",
        "preregister_sha256": digest(PREREG),
        "script_sha256": digest(Path(__file__)),
        "years_read": list(YEARS),
        "years_not_read": [2022, 2023, 2024],
        "semantic": {
            str(year): folds[year]["semantic"] for year in YEARS
        },
        "trials": trials,
        "selected": selected,
        "source_gate": {
            "minimum_full_gain_each_year": minimum_full,
            "minimum_R_gain_each_year": minimum_r,
            "cluster_ci_lower_positive_each_year": True,
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
            safe(
                {
                    "status": report["status"],
                    "selected": selected,
                    "semantic": report["semantic"],
                }
            ),
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
