#!/usr/bin/env python3
"""Immutable source gate for the expanded-TrackMan auto-pitch joint model."""

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
    digest, evaluate, load, safe,
)

RESULTS = ROOT / "experiments/results"
PRED = RESULTS / "predictions"
TRAIN = ROOT / "open/data/train.csv"
PREREG = ROOT / "experiments/params/v5_expanded_auto_pitch_joint_preregister.json"
REPORT = RESULTS / "v5_expanded_auto_pitch_joint_source_gate.json"
YEARS = (2020, 2021)
STAGES = {year: f"v5_expanded_auto_pitch_joint_source{year}" for year in YEARS}
KEY = "catboost_auto_pitch_joint"


def main() -> None:
    if REPORT.exists():
        raise FileExistsError(f"immutable result already exists: {REPORT}")
    prereg = json.loads(PREREG.read_text(encoding="utf-8"))
    game_types = pd.read_csv(TRAIN, usecols=["game_type"])["game_type"].astype(str)
    expected_types = sorted(prereg["candidate"]["fine_pitch_types"])
    expected_recipe = prereg["candidate"]["selector_recipe"]
    folds: dict[int, dict[str, Any]] = {}
    semantic_pass = True

    for year in YEARS:
        candidate_path = PRED / f"{STAGES[year]}_{year}.npz"
        parent_path = PRED / f"v4_m3_c_backtest_{year}_{year}.npz"
        metadata_path = RESULTS / f"{STAGES[year]}.json"
        candidate = load(candidate_path)
        parent_artifact = load(parent_path)
        for alignment_key in ("y", "row_index", "cluster"):
            if not np.array_equal(candidate[alignment_key], parent_artifact[alignment_key]):
                raise ValueError(f"alignment mismatch: {year}/{alignment_key}")
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        fold_meta = metadata["folds"][0]
        details = fold_meta["fit_details"][KEY]
        latent = fold_meta["fine_pitch_latent"]
        classes = sorted(str(value) for value in details["joint_classes"])
        observed_types = sorted({value.split("|pitch=", 1)[1] for value in classes})
        success_classes = [value for value in classes if value.startswith("success|")]
        probability_features = details["auto_probability_features"]
        semantic = {
            "auto_label_coverage_all": float(details["auto_label_coverage_all"]),
            "auto_label_coverage_R": float(details["auto_label_coverage_R"]),
            "success_joint_class_count": len(success_classes),
            "observed_fine_types": observed_types,
            "auto_probability_features": probability_features,
            "selector_architecture": latent["architecture"],
            "identity_minimum_purity": float(latent["identity_minimum_purity"]),
            "mapped_pitchers": int(latent["mapped_pitchers"]),
            "expanded_trackman_rows": int(latent["expanded_trackman_rows"]),
            "expansion_factor": float(latent["expansion_factor"]),
            "pitcher_k": float(latent["pitcher_k"]),
            "pitcher_count_k": float(latent["pitcher_count_k"]),
            "count_weight": float(latent["count_weight"]),
            "catboost_geometric_weight": float(latent["catboost_geometric_weight"]),
            "selector_log_loss_improvement": float(
                latent["selector_log_loss_improvement"]
            ),
            "selector_top1_improvement": float(latent["selector_top1_improvement"]),
            "training_self_pitch_subtracted": bool(
                latent["training_self_pitch_subtracted"]
            ),
            "current_pitch_type_used_at_inference": bool(
                details["current_pitch_type_used_at_inference"]
            ),
            "current_pitch_trackman_used_at_inference": bool(
                details["current_pitch_trackman_used_at_inference"]
            ),
            "row_independent_inference": bool(details["row_independent_inference"]),
        }
        semantic_gate = prereg["semantic_gate"]
        source_gate = prereg["source_protocol"]["gate"]
        semantic["pass"] = bool(
            semantic["auto_label_coverage_all"]
            >= float(semantic_gate["minimum_history_auto_label_coverage_each_year"])
            and semantic["success_joint_class_count"]
            >= int(semantic_gate["minimum_success_joint_classes_each_year"])
            and observed_types == expected_types
            and len(probability_features) == int(semantic_gate["e92_feature_count"])
            and all(column.startswith("e92_") for column in probability_features)
            and semantic["identity_minimum_purity"]
            >= float(semantic_gate["identity_minimum_purity"])
            and semantic["expansion_factor"]
            >= float(source_gate["minimum_expansion_factor_each_year"])
            and semantic["pitcher_k"] == float(expected_recipe["pitcher_k"])
            and semantic["pitcher_count_k"] == float(expected_recipe["pitcher_count_k"])
            and semantic["count_weight"] == float(expected_recipe["count_weight"])
            and semantic["catboost_geometric_weight"]
            == float(expected_recipe["catboost_geometric_weight"])
            and semantic["selector_log_loss_improvement"] > 0.0
            and semantic["selector_top1_improvement"] >= 0.0
            and semantic["training_self_pitch_subtracted"]
            and not semantic["current_pitch_type_used_at_inference"]
            and not semantic["current_pitch_trackman_used_at_inference"]
            and semantic["row_independent_inference"]
        )
        semantic_pass &= semantic["pass"]
        game_type = game_types.iloc[
            candidate["row_index"].astype(np.int64)
        ].to_numpy(dtype=str)
        parent = parent_artifact["catboost_outcome"].astype(np.float64)
        raw = candidate[KEY].astype(np.float64)
        routed_raw = np.where(game_type == "R", raw, parent)
        folds[year] = {
            "candidate": candidate,
            "parent": parent,
            "raw": raw,
            "routed_raw": routed_raw,
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
    prediction_cache: dict[tuple[int, float], np.ndarray] = {}
    if semantic_pass:
        for gamma in prereg["source_protocol"]["top_level_blend_grid"]:
            metrics: dict[str, Any] = {}
            for year in YEARS:
                fold = folds[year]
                metrics[str(year)] = evaluate(
                    fold["candidate"], fold["parent"], fold["routed_raw"],
                    fold["masks"]["full"], fold["masks"], float(gamma),
                    int(prereg["source_protocol"]["bootstrap_iterations"]),
                    3710000 + 10000 * year + int(float(gamma) * 100),
                )
                prediction_cache[(year, float(gamma))] = np.clip(
                    fold["parent"] + float(gamma)
                    * (fold["routed_raw"] - fold["parent"]),
                    1e-6, 1.0 - 1e-6,
                )
            r_gains = [metrics[str(y)]["routes"]["R"]["gain"] for y in YEARS]
            full_gains = [metrics[str(y)]["routes"]["full"]["gain"] for y in YEARS]
            trials.append({
                "gamma": float(gamma),
                "minimum_R_gain": float(min(r_gains)),
                "minimum_full_gain": float(min(full_gains)),
                "mean_R_gain": float(np.mean(r_gains)),
                "years": metrics,
            })
    selected = max(
        trials,
        key=lambda item: (
            item["minimum_R_gain"], item["minimum_full_gain"],
            item["mean_R_gain"], -item["gamma"],
        ),
    ) if trials else None
    gate = prereg["source_protocol"]["gate"]
    checks = [semantic_pass, selected is not None]
    if selected is not None:
        for year in YEARS:
            routes = selected["years"][str(year)]["routes"]
            checks.extend([
                routes["R"]["gain"] >= float(gate["minimum_R_gain_each_year"]),
                routes["full"]["gain"] >= float(gate["minimum_full_gain_each_year"]),
                routes["R"]["pitcher_cluster_95_ci"]["ci_low"] > 0.0,
                routes["full"]["pitcher_cluster_95_ci"]["ci_low"] > 0.0,
            ])
    passed = bool(all(checks))
    artifacts: dict[str, Any] = {}
    if selected is not None:
        for year in YEARS:
            fold = folds[year]
            output = PRED / f"v5_expanded_auto_pitch_joint_source_{year}.npz"
            if output.exists():
                raise FileExistsError(f"immutable artifact exists: {output}")
            np.savez_compressed(
                output,
                y=fold["candidate"]["y"].astype(np.int8),
                row_index=fold["candidate"]["row_index"].astype(np.int64),
                cluster=fold["candidate"]["cluster"],
                parent_exact_c=fold["parent"],
                expanded_auto_pitch_joint_raw=fold["raw"],
                routed_raw=fold["routed_raw"],
                final_prediction=prediction_cache[(year, selected["gamma"])],
            )
            artifacts[str(year)] = {
                "path": str(output.relative_to(ROOT)), "sha256": digest(output)
            }
    report = {
        "experiment_id": prereg["experiment_id"],
        "status": "source_pass" if passed else "source_failed",
        "selector_evidence_sha256": digest(
            ROOT / prereg["selector_evidence"]
        ),
        "preregister_sha256": digest(PREREG),
        "script_sha256": digest(Path(__file__)),
        "years_read": list(YEARS),
        "years_not_read": [2022, 2023, 2024],
        "semantic": {str(y): folds[y]["semantic"] for y in YEARS},
        "input_sha256": {
            str(y): {name: digest(path) for name, path in folds[y]["paths"].items()}
            for y in YEARS
        },
        "trials": trials,
        "selected": selected,
        "source_gate": {"requirements": gate, "pass": passed},
        "artifacts": artifacts,
        "goal_status": "active",
        "goal_completion_claimed": False,
    }
    REPORT.write_text(
        json.dumps(safe(report), ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(safe({
        "status": report["status"], "semantic": report["semantic"],
        "selected": selected,
    }), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
