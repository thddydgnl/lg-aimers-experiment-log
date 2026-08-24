#!/usr/bin/env python3
"""Immutable 2022/2023 development gate for the locked three-axis recipe."""

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
LOCK = ROOT / "experiments/params/v5_three_axis_source_lock.json"
CONTRACT = ROOT / "experiments/params/v5_validation_contract_v2.json"
REPORT = RESULTS / "v5_three_axis_dev.json"
YEARS = (2022, 2023)
BOOTSTRAP_ITERATIONS = 2000

COMPONENTS = {
    "dense_pitch_joint": {
        "stage": "v5_three_axis_joint_dev2223",
        "model": "catboost_dense_pitch_joint",
        "key": "catboost_dense_pitch_joint",
        "features": [
            "base", "e14", "platoon", "hand_matchup", "e14_hand_cells",
            "e14_count_cells", "e14_type_count_cells", "trackman_rich",
            "batter_e14", "batter_middle_e14", "pitchmix_e14",
        ],
        "weight": 0.10,
    },
    "component_pattern_moe": {
        "stage": "v5_three_axis_component_dev2223",
        "model": "catboost_component_pattern_moe",
        "key": "catboost_component_pattern_moe",
        "features": [
            "base", "e14", "platoon", "hand_matchup", "e14_hand_cells",
            "e14_count_cells", "e14_type_count_cells", "trackman_rich",
            "batter_e14", "batter_middle_e14", "pitchmix_e14",
        ],
        "weight": 0.25,
    },
    "current_state_numeric": {
        "stage": "v5_three_axis_current_dev2223",
        "model": "catboost_numeric",
        "key": "catboost_numeric",
        "features": [
            "base", "current_state_context", "current_state_level",
            "trackman_platoon", "trackman_count",
        ],
        "weight": 0.65,
    },
}
ANCHORS = {
    "exact_c": ("v3_sparse_c_backtest_{year}.npz", "catboost_outcome"),
    "honest_identity": (
        "v5_honest_m3_r_identity_{year}.npz", "final_prediction"
    ),
    "honest_grid": ("v5_honest_m3_r_grid_{year}.npz", "final_prediction"),
}


def evaluate_pair(
    y: np.ndarray,
    anchor: np.ndarray,
    candidate: np.ndarray,
    cluster: np.ndarray,
    masks: dict[str, np.ndarray],
    seed: int,
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for route_index, (route, mask) in enumerate(masks.items()):
        anchor_metrics = score(y, anchor, mask)
        candidate_metrics = score(y, candidate, mask)
        interval = cluster_bootstrap_score_gain(
            y,
            anchor,
            candidate,
            cluster,
            mask,
            iterations=BOOTSTRAP_ITERATIONS,
            seed=seed + 1000 * route_index,
        )
        point = float(candidate_metrics["score"] - anchor_metrics["score"])
        if abs(point - float(interval["point"])) > 1e-8:
            raise AssertionError(f"score/CI point mismatch: {route}")
        result[route] = {
            "anchor": anchor_metrics,
            "candidate": candidate_metrics,
            "gain": point,
            "pitcher_cluster_95_ci": interval,
        }
    return result


def validate_stage_contract(
    name: str, spec: dict[str, Any]
) -> tuple[dict[str, Any], dict[str, Any]]:
    path = RESULTS / f"{spec['stage']}.json"
    report = json.loads(path.read_text(encoding="utf-8"))
    metadata = report["metadata"]
    checks = {
        "stage": metadata["stage"] == spec["stage"],
        "models": metadata["models"] == [spec["model"]],
        "features": metadata["features"] == spec["features"],
        "validation_seasons": metadata["validation_seasons"] == list(YEARS),
        "inner_validation_none": metadata["inner_validation"] == "none",
        "row_independent_inference": bool(metadata["row_independent_inference"]),
        "fold_years": [fold["validation_season"] for fold in report["folds"]]
        == list(YEARS),
        "gpu": metadata["booster_device"] == "gpu",
    }
    for fold in report["folds"]:
        if name in ("dense_pitch_joint", "component_pattern_moe"):
            checks[f"{fold['validation_season']}_pitcher_id_dropped"] = (
                "pitcher_id" in fold["dropped_features"]
            )
        details = fold["fit_details"][spec["model"]]
        if name == "dense_pitch_joint":
            checks[f"{fold['validation_season']}_no_current_pitch_group"] = not bool(
                details["current_pitch_group_used_at_inference"]
            )
            checks[f"{fold['validation_season']}_no_selector"] = not bool(
                details["separate_selector_used"]
            )
            checks[f"{fold['validation_season']}_row_independent"] = bool(
                details["row_independent_inference"]
            )
        elif name == "component_pattern_moe":
            checks[f"{fold['validation_season']}_no_current_pattern"] = not bool(
                details["current_validation_pattern_used"]
            )
            checks[f"{fold['validation_season']}_row_independent"] = bool(
                details["row_independent_inference"]
            )
    if not all(checks.values()):
        failed = [key for key, value in checks.items() if not value]
        raise AssertionError(f"stage contract failed for {name}: {failed}")
    return report, {
        "report_path": str(path.relative_to(ROOT)),
        "report_sha256": digest(path),
        "checks": checks,
        "command": metadata["command"],
    }


def main() -> None:
    if REPORT.exists():
        raise FileExistsError(f"immutable result already exists: {REPORT}")
    lock = json.loads(LOCK.read_text(encoding="utf-8"))
    contract = json.loads(CONTRACT.read_text(encoding="utf-8"))
    locked_components = lock["locked_recipe"]["components"]
    locked_names = [item["name"] for item in locked_components]
    locked_weights = np.asarray(
        [item["weight"] for item in locked_components], dtype=np.float64
    )
    expected_names = list(COMPONENTS)
    expected_weights = np.asarray(
        [COMPONENTS[name]["weight"] for name in expected_names],
        dtype=np.float64,
    )
    if locked_names != expected_names or not np.allclose(
        locked_weights, expected_weights, atol=1e-12, rtol=0.0
    ):
        raise AssertionError("analyzer recipe disagrees with source lock")
    if abs(float(locked_weights.sum()) - 1.0) > 1e-12:
        raise AssertionError("locked weights do not sum to one")

    stage_audit: dict[str, Any] = {}
    for name, spec in COMPONENTS.items():
        _, stage_audit[name] = validate_stage_contract(name, spec)

    all_types = pd.read_csv(TRAIN, usecols=["game_type"])["game_type"].astype(str)
    years: dict[str, Any] = {}
    output_artifacts: dict[str, Any] = {}
    full_gains: list[float] = []
    same_parent_r_checks: dict[str, Any] = {}
    input_hashes: dict[str, Any] = {}
    for year in YEARS:
        component_artifacts: dict[str, dict[str, np.ndarray]] = {}
        component_paths: dict[str, Path] = {}
        for name, spec in COMPONENTS.items():
            path = PRED / f"{spec['stage']}_{year}.npz"
            component_artifacts[name] = load(path)
            component_paths[name] = path
        anchor_artifacts: dict[str, dict[str, np.ndarray]] = {}
        anchor_paths: dict[str, Path] = {}
        for name, (template, _) in ANCHORS.items():
            path = PRED / template.format(year=year)
            anchor_artifacts[name] = load(path)
            anchor_paths[name] = path
        reference = component_artifacts[expected_names[0]]
        for group_name, artifacts in (
            ("component", component_artifacts), ("anchor", anchor_artifacts)
        ):
            for name, artifact in artifacts.items():
                for key in ("y", "row_index", "cluster"):
                    if not np.array_equal(reference[key], artifact[key]):
                        raise ValueError(
                            f"alignment mismatch: {year}/{group_name}/{name}/{key}"
                        )
        y = reference["y"].astype(np.int8)
        row_index = reference["row_index"].astype(np.int64)
        cluster = reference["cluster"]
        game_type = all_types.iloc[row_index].to_numpy(dtype=str)
        masks = {
            "full": np.ones(len(y), dtype=bool),
            "R": game_type == "R",
            "F": game_type == "F",
        }
        component_predictions = {
            name: component_artifacts[name][COMPONENTS[name]["key"]].astype(
                np.float64
            )
            for name in expected_names
        }
        candidate = np.clip(
            sum(
                COMPONENTS[name]["weight"] * component_predictions[name]
                for name in expected_names
            ),
            1e-6,
            1.0 - 1e-6,
        )
        if not np.isfinite(candidate).all():
            raise ValueError(f"non-finite candidate prediction: {year}")
        comparisons: dict[str, Any] = {}
        anchor_predictions: dict[str, np.ndarray] = {}
        for anchor_index, (anchor_name, (_, key)) in enumerate(ANCHORS.items()):
            anchor = anchor_artifacts[anchor_name][key].astype(np.float64)
            anchor_predictions[anchor_name] = anchor
            comparison = evaluate_pair(
                y,
                anchor,
                candidate,
                cluster,
                masks,
                seed=2300000 + 10000 * year + 100 * anchor_index,
            )
            comparisons[anchor_name] = comparison
            full_gains.append(float(comparison["full"]["gain"]))
        exact_r = comparisons["exact_c"]["R"]
        same_parent_r_checks[str(year)] = {
            "point_positive": bool(exact_r["gain"] > 0.0),
            "cluster_ci_lower_positive": bool(
                exact_r["pitcher_cluster_95_ci"]["ci_low"] > 0.0
            ),
        }
        years[str(year)] = {
            "rows": int(len(y)),
            "route_rows": {name: int(mask.sum()) for name, mask in masks.items()},
            "candidate_component_summary": {
                name: {
                    "weight": float(COMPONENTS[name]["weight"]),
                    "mean": float(prediction.mean()),
                    "std": float(prediction.std()),
                }
                for name, prediction in component_predictions.items()
            },
            "comparisons": comparisons,
        }
        output = PRED / f"v5_three_axis_dev_{year}.npz"
        if output.exists():
            raise FileExistsError(f"immutable artifact already exists: {output}")
        np.savez_compressed(
            output,
            y=y,
            row_index=row_index,
            cluster=cluster,
            dense_pitch_joint=component_predictions["dense_pitch_joint"],
            component_pattern_moe=component_predictions["component_pattern_moe"],
            current_state_numeric=component_predictions["current_state_numeric"],
            parent_exact_c=anchor_predictions["exact_c"],
            honest_identity=anchor_predictions["honest_identity"],
            honest_grid=anchor_predictions["honest_grid"],
            final_prediction=candidate,
        )
        output_artifacts[str(year)] = {
            "path": str(output.relative_to(ROOT)),
            "sha256": digest(output),
        }
        input_hashes[str(year)] = {
            "components": {
                name: digest(path) for name, path in component_paths.items()
            },
            "anchors": {name: digest(path) for name, path in anchor_paths.items()},
        }

    required_gain = float(
        contract["conservative_score"]["required_raw_gain_for_1190_at_haircut"]
    )
    g_dev = float(min(full_gains))
    same_parent_r_pass = bool(
        all(all(checks.values()) for checks in same_parent_r_checks.values())
    )
    f_2023 = years["2023"]["comparisons"]["exact_c"]["F"]
    postbreak_f_checks = {
        "point_positive": bool(f_2023["gain"] > 0.0),
        "cluster_ci_lower_positive": bool(
            f_2023["pitcher_cluster_95_ci"]["ci_low"] > 0.0
        ),
    }
    g_dev_checks = {
        "minimum_full_gain": g_dev,
        "required_strictly_greater_than": required_gain,
        "pass": bool(g_dev > required_gain),
    }
    development_pass = bool(
        same_parent_r_pass
        and all(postbreak_f_checks.values())
        and g_dev_checks["pass"]
    )
    report = {
        "experiment_id": "V5_THREE_AXIS_DEVELOPMENT_V1",
        "status": "development_pass" if development_pass else "development_failed",
        "source_lock": {
            "path": str(LOCK.relative_to(ROOT)),
            "sha256": digest(LOCK),
        },
        "validation_contract": {
            "path": str(CONTRACT.relative_to(ROOT)),
            "sha256": digest(CONTRACT),
        },
        "script_sha256": digest(Path(__file__)),
        "years_read": list(YEARS),
        "years_not_read": [2024],
        "stage_audit": stage_audit,
        "input_sha256": input_hashes,
        "recipe": {
            "components": expected_names,
            "weights": expected_weights,
            "formula": lock["locked_recipe"]["formula"],
            "selection_changed_after_source": False,
        },
        "years": years,
        "gates": {
            "same_parent_R": {
                "years": same_parent_r_checks,
                "pass": same_parent_r_pass,
            },
            "postbreak_2023_F": {
                "checks": postbreak_f_checks,
                "pass": bool(all(postbreak_f_checks.values())),
            },
            "G_dev": g_dev_checks,
            "development_pass": development_pass,
        },
        "artifacts": output_artifacts,
        "confirmation_2024_authorized": development_pass,
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
                    "gates": report["gates"],
                    "year_summary": {
                        year: {
                            anchor: {
                                route: {
                                    "gain": details["gain"],
                                    "ci_low": details["pitcher_cluster_95_ci"][
                                        "ci_low"
                                    ],
                                }
                                for route, details in comparison.items()
                            }
                            for anchor, comparison in value["comparisons"].items()
                        }
                        for year, value in years.items()
                    },
                }
            ),
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
