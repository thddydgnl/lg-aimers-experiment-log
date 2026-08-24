#!/usr/bin/env python3
"""Immutable source gate for the profile-conditioned auto-pitch joint model."""

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
PREREG = ROOT / "experiments/params/v5_auto_pitch_profile_joint_preregister.json"
REPORT = RESULTS / "v5_auto_pitch_profile_joint_source_gate.json"
YEARS = (2020, 2021)
STAGES = {year: f"v5_auto_pitch_profile_joint_source{year}" for year in YEARS}
BASELINE_STAGES = {year: f"v5_auto_pitch_joint_source{year}" for year in YEARS}
KEY = "catboost_auto_pitch_joint"


def main() -> None:
    if REPORT.exists():
        raise FileExistsError(f"immutable result already exists: {REPORT}")
    prereg = json.loads(PREREG.read_text(encoding="utf-8"))
    all_types = pd.read_csv(TRAIN, usecols=["game_type"])["game_type"].astype(str)
    expected_types = sorted(prereg["candidate"]["fine_pitch_types"])
    expected_specs = prereg["profile_contract"]["specs"]
    folds: dict[int, dict[str, Any]] = {}
    semantic_pass = True

    for year in YEARS:
        candidate_path = PRED / f"{STAGES[year]}_{year}.npz"
        parent_path = PRED / f"v4_m3_c_backtest_{year}_{year}.npz"
        metadata_path = RESULTS / f"{STAGES[year]}.json"
        baseline_metadata_path = RESULTS / f"{BASELINE_STAGES[year]}.json"
        candidate = load(candidate_path)
        parent_artifact = load(parent_path)
        for alignment_key in ("y", "row_index", "cluster"):
            if not np.array_equal(candidate[alignment_key], parent_artifact[alignment_key]):
                raise ValueError(f"alignment mismatch: {year}/{alignment_key}")

        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        baseline_metadata = json.loads(
            baseline_metadata_path.read_text(encoding="utf-8")
        )
        fold_meta = metadata["folds"][0]
        baseline_fold_meta = baseline_metadata["folds"][0]
        details = fold_meta["fit_details"][KEY]
        latent = fold_meta["fine_pitch_latent"]
        baseline_latent = baseline_fold_meta["fine_pitch_latent"]
        classes = sorted(str(value) for value in details["joint_classes"])
        observed_types = sorted({value.split("|pitch=", 1)[1] for value in classes})
        success_classes = [value for value in classes if value.startswith("success|")]
        top1_improvement = float(
            latent["valid_top1_accuracy"] - baseline_latent["valid_top1_accuracy"]
        )
        logloss_improvement = float(
            baseline_latent["valid_log_loss"] - latent["valid_log_loss"]
        )
        semantic = {
            "auto_label_coverage_all": float(details["auto_label_coverage_all"]),
            "auto_label_coverage_R": float(details["auto_label_coverage_R"]),
            "joint_class_count": len(classes),
            "success_joint_class_count": len(success_classes),
            "observed_fine_types": observed_types,
            "expected_fine_types": expected_types,
            "auto_probability_feature_count": len(details["auto_probability_features"]),
            "auto_profile_features": details["auto_profile_features"],
            "profile_features_enabled": bool(latent["profile_features_enabled"]),
            "profile_specs": latent["profile_specs"],
            "profile_training_self_excluded": bool(
                latent["profile_training_self_excluded"]
            ),
            "selector_top1_accuracy": float(latent["valid_top1_accuracy"]),
            "baseline_selector_top1_accuracy": float(
                baseline_latent["valid_top1_accuracy"]
            ),
            "selector_top1_improvement": top1_improvement,
            "selector_log_loss": float(latent["valid_log_loss"]),
            "baseline_selector_log_loss": float(
                baseline_latent["valid_log_loss"]
            ),
            "selector_log_loss_improvement": logloss_improvement,
            "current_pitch_type_used_at_inference": bool(
                details["current_pitch_type_used_at_inference"]
            ),
            "current_pitch_trackman_used_at_inference": bool(
                details["current_pitch_trackman_used_at_inference"]
            ),
            "row_independent_inference": bool(details["row_independent_inference"]),
        }
        gate = prereg["source_protocol"]["gate"]
        semantic["pass"] = bool(
            semantic["auto_label_coverage_all"]
            >= float(prereg["semantic_gate"]["minimum_history_auto_label_coverage_each_year"])
            and semantic["success_joint_class_count"]
            >= int(prereg["semantic_gate"]["minimum_success_joint_classes_each_year"])
            and observed_types == expected_types
            and semantic["auto_probability_feature_count"] == 10
            and len(semantic["auto_profile_features"])
            == int(prereg["profile_contract"]["probability_feature_count"])
            and semantic["profile_features_enabled"]
            and semantic["profile_specs"] == expected_specs
            and semantic["profile_training_self_excluded"]
            and top1_improvement
            >= float(gate["minimum_selector_top1_improvement_each_year"])
            and logloss_improvement > 0.0
            and not semantic["current_pitch_type_used_at_inference"]
            and not semantic["current_pitch_trackman_used_at_inference"]
            and semantic["row_independent_inference"]
        )
        semantic_pass &= semantic["pass"]
        game_type = all_types.iloc[candidate["row_index"].astype(np.int64)].to_numpy(dtype=str)
        parent = parent_artifact["catboost_outcome"].astype(np.float64)
        raw = candidate[KEY].astype(np.float64)
        routed_raw = np.where(game_type == "R", raw, parent)
        folds[year] = {
            "candidate": candidate,
            "parent": parent,
            "routed_raw": routed_raw,
            "masks": {"full": np.ones(len(game_type), dtype=bool), "R": game_type == "R"},
            "semantic": semantic,
            "paths": {
                "candidate": candidate_path,
                "parent": parent_path,
                "metadata": metadata_path,
                "baseline_metadata": baseline_metadata_path,
            },
        }
        if year == YEARS[0] and not semantic["pass"]:
            report = {
                "experiment_id": prereg["experiment_id"],
                "status": "source_failed_early_2020",
                "early_stop_reason": (
                    "The fixed 2020 selector semantic gate failed; 2021 cannot "
                    "restore the required every-year selector improvement."
                ),
                "preregister_sha256": digest(PREREG),
                "script_sha256": digest(Path(__file__)),
                "years_read": [year],
                "years_not_read": [2021, 2022, 2023, 2024],
                "semantic": {str(year): semantic},
                "input_sha256": {
                    str(year): {
                        name: digest(path)
                        for name, path in folds[year]["paths"].items()
                    }
                },
                "trials": [],
                "selected": None,
                "source_gate": {
                    "requirements": prereg["source_protocol"]["gate"],
                    "pass": False,
                },
                "artifacts": {},
                "goal_status": "active",
                "goal_completion_claimed": False,
            }
            REPORT.write_text(
                json.dumps(safe(report), ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            print(json.dumps(safe({
                "status": report["status"],
                "early_stop_reason": report["early_stop_reason"],
                "semantic": report["semantic"],
            }), ensure_ascii=False, indent=2))
            return

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
                    3510000 + 10000 * year + int(float(gamma) * 100),
                )
                prediction_cache[(year, float(gamma))] = np.clip(
                    fold["parent"] + float(gamma) * (
                        fold["routed_raw"] - fold["parent"]
                    ), 1e-6, 1.0 - 1e-6,
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
            output = PRED / f"v5_auto_pitch_profile_joint_source_{year}.npz"
            if output.exists():
                raise FileExistsError(f"immutable artifact exists: {output}")
            np.savez_compressed(
                output,
                y=fold["candidate"]["y"].astype(np.int8),
                row_index=fold["candidate"]["row_index"].astype(np.int64),
                cluster=fold["candidate"]["cluster"],
                parent_exact_c=fold["parent"],
                auto_pitch_profile_joint_raw=fold["candidate"][KEY].astype(np.float64),
                routed_raw=fold["routed_raw"],
                final_prediction=prediction_cache[(year, selected["gamma"])],
            )
            artifacts[str(year)] = {
                "path": str(output.relative_to(ROOT)), "sha256": digest(output)
            }
    report = {
        "experiment_id": prereg["experiment_id"],
        "status": "source_pass" if passed else "source_failed",
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
