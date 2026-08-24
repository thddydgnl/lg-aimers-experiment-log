#!/usr/bin/env python3
"""Immutable 2020/2021 source gate for the fine-pitch joint model."""

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
PREREG = ROOT / "experiments/params/v5_fine_pitch_joint_preregister.json"
REPORT = RESULTS / "v5_fine_pitch_joint_source_gate.json"
YEARS = (2020, 2021)
STAGES = {year: f"v5_fine_pitch_joint_source{year}" for year in YEARS}
KEY = "catboost_fine_pitch_joint"


def main() -> None:
    if REPORT.exists():
        raise FileExistsError(f"immutable result already exists: {REPORT}")
    prereg = json.loads(PREREG.read_text(encoding="utf-8"))
    all_types = pd.read_csv(TRAIN, usecols=["game_type"])["game_type"].astype(str)
    expected_types = sorted(prereg["candidate"]["fine_pitch_types"])
    folds: dict[int, dict[str, Any]] = {}
    semantic_pass = True
    for year in YEARS:
        candidate_path = PRED / f"{STAGES[year]}_{year}.npz"
        parent_path = PRED / f"v4_m3_c_backtest_{year}_{year}.npz"
        candidate = load(candidate_path)
        parent = load(parent_path)
        for alignment_key in ("y", "row_index", "cluster"):
            if not np.array_equal(candidate[alignment_key], parent[alignment_key]):
                raise ValueError(f"alignment mismatch: {year}/{alignment_key}")
        metadata_path = RESULTS / f"{STAGES[year]}.json"
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        details = metadata["folds"][0]["fit_details"][KEY]
        classes = sorted(str(value) for value in details["joint_classes"])
        observed_types = sorted(
            {value.split("|pitch=", 1)[1] for value in classes}
        )
        success_classes = [
            value for value in classes if value.startswith("success|")
        ]
        semantic = {
            "fine_label_coverage_all": float(details["fine_label_coverage_all"]),
            "fine_label_coverage_R": float(details["fine_label_coverage_R"]),
            "joint_usable_rows": int(details["joint_usable_rows"]),
            "joint_dropped_rows": int(details["joint_dropped_rows"]),
            "joint_class_count": len(classes),
            "success_joint_class_count": len(success_classes),
            "observed_fine_types": observed_types,
            "expected_fine_types": expected_types,
            "fine_probability_features": details["fine_probability_features"],
            "current_pitch_type_used_at_inference": bool(
                details["current_pitch_type_used_at_inference"]
            ),
            "current_pitch_trackman_used_at_inference": bool(
                details["current_pitch_trackman_used_at_inference"]
            ),
            "row_independent_inference": bool(
                details["row_independent_inference"]
            ),
        }
        semantic["pass"] = bool(
            semantic["fine_label_coverage_all"]
            >= float(
                prereg["semantic_gate"][
                    "minimum_history_fine_label_coverage_each_year"
                ]
            )
            and semantic["success_joint_class_count"]
            >= int(
                prereg["semantic_gate"][
                    "minimum_success_joint_classes_each_year"
                ]
            )
            and observed_types == expected_types
            and len(semantic["fine_probability_features"]) == 10
            and not semantic["current_pitch_type_used_at_inference"]
            and not semantic["current_pitch_trackman_used_at_inference"]
            and semantic["row_independent_inference"]
        )
        semantic_pass &= semantic["pass"]
        game_type = all_types.iloc[
            candidate["row_index"].astype(np.int64)
        ].to_numpy(dtype=str)
        folds[year] = {
            "candidate": candidate,
            "parent": parent["catboost_outcome"].astype(np.float64),
            "masks": {
                "full": np.ones(len(game_type), dtype=bool),
                "R": game_type == "R",
            },
            "semantic": semantic,
            "paths": {
                "candidate": candidate_path,
                "parent": parent_path,
                "metadata": metadata_path,
            },
        }

    trials: list[dict[str, Any]] = []
    cache: dict[tuple[int, float], np.ndarray] = {}
    if semantic_pass:
        for gamma in prereg["source_protocol"]["top_level_blend_grid"]:
            metrics: dict[str, Any] = {}
            for year in YEARS:
                fold = folds[year]
                metrics[str(year)] = evaluate(
                    fold["candidate"],
                    fold["parent"],
                    fold["candidate"][KEY].astype(np.float64),
                    fold["masks"]["full"],
                    fold["masks"],
                    float(gamma),
                    int(prereg["source_protocol"]["bootstrap_iterations"]),
                    2710000 + 10000 * year + int(float(gamma) * 100),
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
            r_gains = [
                metrics[str(year)]["routes"]["R"]["gain"] for year in YEARS
            ]
            full_gains = [
                metrics[str(year)]["routes"]["full"]["gain"] for year in YEARS
            ]
            trials.append(
                {
                    "gamma": float(gamma),
                    "minimum_R_gain": float(min(r_gains)),
                    "minimum_full_gain": float(min(full_gains)),
                    "mean_R_gain": float(np.mean(r_gains)),
                    "years": metrics,
                }
            )
    selected = (
        max(
            trials,
            key=lambda item: (
                item["minimum_R_gain"],
                item["minimum_full_gain"],
                item["mean_R_gain"],
                -item["gamma"],
            ),
        )
        if trials
        else None
    )
    gate = prereg["source_protocol"]["gate"]
    checks = [semantic_pass, selected is not None]
    if selected is not None:
        for year in YEARS:
            routes = selected["years"][str(year)]["routes"]
            checks.extend(
                [
                    routes["R"]["gain"]
                    >= float(gate["minimum_R_gain_each_year"]),
                    routes["full"]["gain"]
                    >= float(gate["minimum_full_gain_each_year"]),
                    routes["R"]["pitcher_cluster_95_ci"]["ci_low"] > 0.0,
                    routes["full"]["pitcher_cluster_95_ci"]["ci_low"] > 0.0,
                ]
            )
    passed = bool(all(checks))
    artifacts: dict[str, Any] = {}
    if selected is not None:
        for year in YEARS:
            fold = folds[year]
            output = PRED / f"v5_fine_pitch_joint_source_{year}.npz"
            if output.exists():
                raise FileExistsError(f"immutable artifact exists: {output}")
            np.savez_compressed(
                output,
                y=fold["candidate"]["y"].astype(np.int8),
                row_index=fold["candidate"]["row_index"].astype(np.int64),
                cluster=fold["candidate"]["cluster"],
                parent_exact_c=fold["parent"],
                fine_pitch_joint_raw=fold["candidate"][KEY].astype(np.float64),
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
        "input_sha256": {
            str(year): {
                name: digest(path)
                for name, path in folds[year]["paths"].items()
            }
            for year in YEARS
        },
        "trials": trials,
        "selected": selected,
        "source_gate": {"requirements": gate, "pass": passed},
        "artifacts": artifacts,
        "goal_status": "active",
        "goal_completion_claimed": False,
    }
    REPORT.write_text(
        json.dumps(safe(report), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(
        json.dumps(
            safe(
                {
                    "status": report["status"],
                    "semantic": report["semantic"],
                    "selected": selected,
                }
            ),
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
