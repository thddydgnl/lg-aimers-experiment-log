#!/usr/bin/env python3
"""Immutable 2022/2023 development gate for the locked dual-state triad."""

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
    load,
    safe,
    score,
)
from experiments.run_v5_h1_residual import (  # noqa: E402
    cluster_bootstrap_score_gain,
)


RESULTS = ROOT / "experiments/results"
PRED = RESULTS / "predictions"
TRAIN = ROOT / "open/data/train.csv"
PREREG = ROOT / "experiments/params/v5_dual_state_pitch_triad_preregister.json"
SOURCE = RESULTS / "v5_dual_state_pitch_triad_source.json"
EXPANDED_PREREG = (
    ROOT / "experiments/params/v5_expanded_auto_pitch_joint_preregister.json"
)
RECOVERY = (
    ROOT / "experiments/params/v5_dual_state_pitch_triad_dev_recovery.json"
)
REPORT = RESULTS / "v5_dual_state_pitch_triad_dev.json"
STAGE = "v5_dual_state_pitch_triad_expanded_auto_dev2223"
STAGE_REPORT = RESULTS / f"{STAGE}.json"
YEARS = (2022, 2023)
MODEL = "catboost_auto_pitch_joint"
BOOTSTRAP_ITERATIONS = 2000


def evaluate(
    y: np.ndarray,
    parent: np.ndarray,
    candidate: np.ndarray,
    cluster: np.ndarray,
    masks: dict[str, np.ndarray],
    seed: int,
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for route_index, (route, mask) in enumerate(masks.items()):
        parent_score = score(y, parent, mask)
        candidate_score = score(y, candidate, mask)
        interval = cluster_bootstrap_score_gain(
            y,
            parent,
            candidate,
            cluster,
            mask,
            iterations=BOOTSTRAP_ITERATIONS,
            seed=seed + 1000 * route_index,
        )
        point = float(candidate_score["score"] - parent_score["score"])
        if abs(point - float(interval["point"])) > 1e-8:
            raise AssertionError(f"score/CI point mismatch: {route}")
        result[route] = {
            "parent": parent_score,
            "candidate": candidate_score,
            "gain": point,
            "pitcher_cluster_95_ci": interval,
        }
    return result


def main() -> None:
    if REPORT.exists():
        raise FileExistsError(f"immutable result already exists: {REPORT}")
    prereg = json.loads(PREREG.read_text(encoding="utf-8"))
    source = json.loads(SOURCE.read_text(encoding="utf-8"))
    expanded_prereg = json.loads(EXPANDED_PREREG.read_text(encoding="utf-8"))
    recovery = json.loads(RECOVERY.read_text(encoding="utf-8"))
    stage_report = json.loads(STAGE_REPORT.read_text(encoding="utf-8"))
    run_lock = prereg["expanded_auto_development_run"]
    metadata = stage_report["metadata"]
    preserved_2022_path = ROOT / recovery["preserved_2022_artifact"]["path"]
    preserved_2022 = load(preserved_2022_path)
    stage_checks: dict[str, bool] = {
        "source_lock_passed": source["status"] == "source_lock_passed",
        "stage": metadata["stage"] == STAGE == run_lock["stage"],
        "models": metadata["models"] == run_lock["models"] == [MODEL],
        "features": metadata["features"] == run_lock["features"],
        "recovery_scope_execution_only": recovery["recovery_scope"]
        == "execution_only_no_recipe_change",
        "recovery_original_years": recovery["original_validation_seasons"]
        == list(YEARS),
        "recovery_only_missing_year": recovery["recovery_run"][
            "validation_seasons"
        ]
        == [2023],
        "recovery_stage_unchanged": bool(
            recovery["recovery_run"]["stage_name_unchanged"]
            and recovery["recovery_run"]["features_unchanged"]
            and recovery["recovery_run"]["model_unchanged"]
            and recovery["recovery_run"]["parameters_unchanged"]
        ),
        "recovery_2024_sealed": recovery[
            "sealed_years_not_read_or_generated_by_this_recovery"
        ]
        == [2024],
        "preserved_2022_sha256": digest(preserved_2022_path)
        == recovery["preserved_2022_artifact"]["sha256"],
        "preserved_2022_bytes": preserved_2022_path.stat().st_size
        == int(recovery["preserved_2022_artifact"]["bytes"]),
        "preserved_2022_rows": len(preserved_2022["y"])
        == int(recovery["preserved_2022_artifact"]["rows"]),
        "preserved_2022_prediction_key": MODEL in preserved_2022,
        "preserved_2022_finite": bool(
            np.isfinite(preserved_2022[MODEL]).all()
        ),
        "recovery_report_year": metadata["validation_seasons"] == [2023],
        "inner_validation_none": metadata["inner_validation"] == "none",
        "outcome_scheme": metadata["outcome_scheme"] == "reverse_any",
        "row_independent_inference": bool(metadata["row_independent_inference"]),
        "fold_order": [fold["validation_season"] for fold in stage_report["folds"]]
        == [2023],
    }
    expected_types = sorted(expanded_prereg["candidate"]["fine_pitch_types"])
    expected_selector = expanded_prereg["candidate"]["selector_recipe"]
    fold_semantic: dict[str, Any] = {}
    for fold in stage_report["folds"]:
        year = int(fold["validation_season"])
        details = fold["fit_details"][MODEL]
        latent = fold["fine_pitch_latent"]
        joint_classes = [str(value) for value in details["joint_classes"]]
        observed_types = sorted(
            {value.split("|pitch=", 1)[1] for value in joint_classes}
        )
        success_classes = [
            value for value in joint_classes if value.startswith("success|")
        ]
        probability_features = details["auto_probability_features"]
        checks = {
            "pitcher_id_dropped": "pitcher_id" in fold["dropped_features"],
            "eight_types": observed_types == expected_types,
            "eight_success_classes": len(success_classes) >= 8,
            "e92_probability_features": len(probability_features) == 10
            and all(value.startswith("e92_") for value in probability_features),
            "history_auto_coverage": float(details["auto_label_coverage_all"])
            >= 0.5,
            "identity_purity": float(latent["identity_minimum_purity"]) >= 0.99,
            "expansion_factor": float(latent["expansion_factor"]) >= 1.25,
            "pitcher_k": float(latent["pitcher_k"])
            == float(expected_selector["pitcher_k"]),
            "pitcher_count_k": float(latent["pitcher_count_k"])
            == float(expected_selector["pitcher_count_k"]),
            "count_weight": float(latent["count_weight"])
            == float(expected_selector["count_weight"]),
            "catboost_geometric_weight": float(latent["catboost_geometric_weight"])
            == float(expected_selector["catboost_geometric_weight"]),
            "training_self_pitch_subtracted": bool(
                latent["training_self_pitch_subtracted"]
            ),
            "no_current_pitch_type": not bool(
                details["current_pitch_type_used_at_inference"]
            ),
            "no_current_pitch_trackman": not bool(
                details["current_pitch_trackman_used_at_inference"]
            ),
            "row_independent": bool(details["row_independent_inference"]),
        }
        checks["pass"] = bool(all(checks.values()))
        fold_semantic[str(year)] = {
            "checks": checks,
            "observed_types": observed_types,
            "auto_label_coverage_all": float(details["auto_label_coverage_all"]),
            "identity_minimum_purity": float(latent["identity_minimum_purity"]),
            "expansion_factor": float(latent["expansion_factor"]),
            "selector_log_loss_improvement": float(
                latent["selector_log_loss_improvement"]
            ),
            "selector_top1_improvement": float(
                latent["selector_top1_improvement"]
            ),
        }
        stage_checks[f"fold_{year}_semantic"] = checks["pass"]
    fold_semantic["2022"] = {
        "checks": {
            "artifact_sha256": stage_checks["preserved_2022_sha256"],
            "artifact_bytes": stage_checks["preserved_2022_bytes"],
            "artifact_rows": stage_checks["preserved_2022_rows"],
            "prediction_key": stage_checks["preserved_2022_prediction_key"],
            "prediction_finite": stage_checks["preserved_2022_finite"],
            "identical_locked_stage_recipe": stage_checks[
                "recovery_stage_unchanged"
            ],
            "pass": bool(
                stage_checks["preserved_2022_sha256"]
                and stage_checks["preserved_2022_bytes"]
                and stage_checks["preserved_2022_rows"]
                and stage_checks["preserved_2022_prediction_key"]
                and stage_checks["preserved_2022_finite"]
                and stage_checks["recovery_stage_unchanged"]
            ),
        },
        "evidence_role": (
            "immutable execution checkpoint; detailed fold metadata was not "
            "returned after the original stdout pipe closed"
        ),
    }
    stage_checks["fold_2022_recovery_checkpoint"] = bool(
        fold_semantic["2022"]["checks"]["pass"]
    )
    if not all(stage_checks.values()):
        raise AssertionError(f"stage/semantic audit failed: {stage_checks}")

    weights = {
        item["name"]: float(item["weight"])
        for item in prereg["locked_recipe"]["components"]
    }
    expected_weights = {
        "direct_update": 0.55,
        "expanded_auto": 0.25,
        "current_numeric": 0.20,
    }
    if weights != expected_weights or abs(sum(weights.values()) - 1.0) > 1e-12:
        raise AssertionError("locked weights changed")
    game_types = pd.read_csv(TRAIN, usecols=["game_type"])["game_type"].astype(str)
    years: dict[str, Any] = {}
    input_hashes: dict[str, Any] = {}
    output_artifacts: dict[str, Any] = {}
    for year in YEARS:
        paths = {
            "parent": PRED / f"v5_honest_m3_r_identity_{year}.npz",
            "direct_update": PRED / f"v5_direct_season_update_dev_{year}.npz",
            "expanded_auto": PRED / f"{STAGE}_{year}.npz",
            "current_numeric": PRED / f"v5_three_axis_current_dev2223_{year}.npz",
        }
        artifacts = {name: load(path) for name, path in paths.items()}
        if year == 2022 and digest(paths["expanded_auto"]) != recovery[
            "preserved_2022_artifact"
        ]["sha256"]:
            raise ValueError("preserved 2022 expanded-auto artifact changed")
        parent_artifact = artifacts["parent"]
        for name, artifact in artifacts.items():
            for key in ("y", "row_index", "cluster"):
                if not np.array_equal(parent_artifact[key], artifact[key]):
                    raise ValueError(f"alignment mismatch: {year}/{name}/{key}")
        y = parent_artifact["y"].astype(np.int8)
        row_index = parent_artifact["row_index"].astype(np.int64)
        cluster = parent_artifact["cluster"]
        parent = parent_artifact["final_prediction"].astype(np.float64)
        components = {
            "direct_update": artifacts["direct_update"]["final_prediction"].astype(
                np.float64
            ),
            "expanded_auto": artifacts["expanded_auto"][MODEL].astype(np.float64),
            "current_numeric": artifacts["current_numeric"]["catboost_numeric"].astype(
                np.float64
            ),
        }
        if not all(
            np.isfinite(value).all()
            and np.all((value > 0.0) & (value < 1.0))
            for value in components.values()
        ):
            raise ValueError(f"invalid component probabilities: {year}")
        raw = np.clip(
            sum(weights[name] * components[name] for name in expected_weights),
            1e-6,
            1.0 - 1e-6,
        )
        game_type = game_types.iloc[row_index].to_numpy(dtype=str)
        r_mask = game_type == "R"
        candidate = np.where(r_mask, raw, parent)
        masks = {
            "full": np.ones(len(y), dtype=bool),
            "R": r_mask,
            "F": ~r_mask,
        }
        metrics = evaluate(
            y,
            parent,
            candidate,
            cluster,
            masks,
            seed=4930000 + 10000 * year,
        )
        years[str(year)] = {
            "rows": int(len(y)),
            "route_rows": {name: int(mask.sum()) for name, mask in masks.items()},
            "component_summary": {
                name: {
                    "weight": weights[name],
                    "mean": float(value.mean()),
                    "std": float(value.std()),
                }
                for name, value in components.items()
            },
            "metrics": metrics,
        }
        input_hashes[str(year)] = {
            name: digest(path) for name, path in paths.items()
        }
        output = PRED / f"v5_dual_state_pitch_triad_dev_{year}.npz"
        if output.exists():
            raise FileExistsError(f"immutable artifact already exists: {output}")
        np.savez_compressed(
            output,
            y=y,
            row_index=row_index,
            cluster=cluster,
            parent_m3=parent,
            direct_update=components["direct_update"],
            expanded_auto=components["expanded_auto"],
            current_numeric=components["current_numeric"],
            raw_mixture=raw,
            final_prediction=candidate,
        )
        output_artifacts[str(year)] = {
            "path": str(output.relative_to(ROOT)),
            "sha256": digest(output),
        }

    gate = prereg["development_gate"]
    minimum_full_point = float(
        min(years[str(year)]["metrics"]["full"]["gain"] for year in YEARS)
    )
    checks = {
        "stage_and_semantic": all(stage_checks.values()),
        "R_point_positive_each_year": all(
            years[str(year)]["metrics"]["R"]["gain"]
            > float(gate["minimum_R_point_gain_each_year"])
            for year in YEARS
        ),
        "R_ci_low_positive_each_year": all(
            years[str(year)]["metrics"]["R"]["pitcher_cluster_95_ci"]["ci_low"]
            > 0.0
            for year in YEARS
        ),
        "full_ci_low_positive_each_year": all(
            years[str(year)]["metrics"]["full"]["pitcher_cluster_95_ci"][
                "ci_low"
            ]
            > 0.0
            for year in YEARS
        ),
        "minimum_full_point_exceeds_required": minimum_full_point
        > float(gate["minimum_routed_full_point_gain_across_2022_2023"]),
    }
    passed = bool(all(checks.values()))
    expected_lower = float(
        1090.9100565103 + 0.75 * max(0.0, minimum_full_point)
    )
    report = {
        "experiment_id": prereg["experiment_id"],
        "status": "development_pass" if passed else "development_failed",
        "evidence_role": "stress test with partial prior development knowledge; not independent confirmation",
        "preregister_sha256": digest(PREREG),
        "source_report_sha256": digest(SOURCE),
        "expanded_preregister_sha256": digest(EXPANDED_PREREG),
        "recovery_record_sha256": digest(RECOVERY),
        "stage_report_sha256": digest(STAGE_REPORT),
        "script_sha256": digest(Path(__file__)),
        "years_read": list(YEARS),
        "years_not_read": [2024],
        "stage_checks": stage_checks,
        "fold_semantic": fold_semantic,
        "execution_recovery": {
            "record": str(RECOVERY.relative_to(ROOT)),
            "mode": recovery["recovery_scope"],
            "preserved_year": 2022,
            "recomputed_year": 2023,
            "recipe_changed": False,
        },
        "recipe": prereg["locked_recipe"],
        "input_sha256": input_hashes,
        "years": years,
        "development_gate": {
            "requirements": gate,
            "checks": checks,
            "minimum_routed_full_point_gain": minimum_full_point,
            "expected_lower_from_point_only": expected_lower,
            "pass": passed,
        },
        "2024_composite_read_or_generated": False,
        "advance_to_2024": passed,
        "output_artifacts": output_artifacts,
        "goal_status": "active",
        "goal_completion_claimed": False,
    }
    REPORT.write_text(
        json.dumps(safe(report), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(safe({
        "status": report["status"],
        "stage_checks": stage_checks,
        "years": {
            str(year): years[str(year)]["metrics"] for year in YEARS
        },
        "development_gate": report["development_gate"],
        "advance_to_2024": passed,
    }), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
