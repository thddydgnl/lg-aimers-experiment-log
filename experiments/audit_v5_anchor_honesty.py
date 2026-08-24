#!/usr/bin/env python3
"""Audit and rebuild the V5 historical M3 comparison anchor.

The persisted ``v3_sparse_m3_frozen`` artifacts use weights and an affine
calibration selected on the 2024 fold for every historical year.  They are a
valid reproduction of the published V3 candidate, but they are not a valid
one-year-ahead development baseline for 2022 or 2023.

This audit creates target-blind historical analogues.  For target season Y,
three non-negative M3 component weights (sum to one) and, when applicable, one
of the original predeclared affine grid points are selected using season Y-1
only.  The selected recipe is then transferred unchanged to Y.  The two
primary anchors are R-fitted because 2023 F has a documented label-regime
break:

* ``v5_honest_m3_r_identity``: source-fitted weights, no affine calibration.
* ``v5_honest_m3_r_grid``: source-fitted weights and source-selected original
  V3 affine-grid point.

All outputs use official train rows only.  The generated prediction artifacts
remain row independent and are for validation/audit, not submission.
"""

from __future__ import annotations

import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
from scipy.optimize import minimize


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.analyze_v4_temporal_residual_ridge import load_frames
from experiments.run_v5_h1_residual import cluster_bootstrap_score_gain


PREDICTIONS = ROOT / "experiments/results/predictions"
OUTPUT = ROOT / "experiments/results/v5_anchor_honesty_audit.json"
COMPONENT_ORDER = ("A", "C", "B")
TARGET_YEARS = (2022, 2023, 2024)
AFFINE_GRID = tuple(
    (float(slope), float(offset))
    for slope in (1.0, 1.05, 1.10)
    for offset in (-0.004, -0.006, -0.008)
)
PUBLISHED_WEIGHTS = np.asarray(
    [0.501443851662535, 0.27016033407769313, 0.22839581425977187],
    dtype=np.float64,
)
PUBLISHED_SLOPE = 1.05
PUBLISHED_OFFSET = -0.006


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        number = float(value)
        return number if np.isfinite(number) else None
    return value


def load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        return {key: np.asarray(archive[key]) for key in archive.files}


def component_matrix(artifact: dict[str, np.ndarray]) -> np.ndarray:
    return np.column_stack(
        [np.asarray(artifact[f"component_{key}"], dtype=np.float64) for key in COMPONENT_ORDER]
    )


def calibrated(raw: np.ndarray, slope: float, offset: float) -> np.ndarray:
    return np.clip(
        0.5 + slope * (np.asarray(raw, dtype=np.float64) - 0.5) + offset,
        1e-6,
        1.0 - 1e-6,
    )


def metrics(y: np.ndarray, prediction: np.ndarray, mask: np.ndarray) -> dict[str, float | int]:
    target = np.asarray(y[mask], dtype=np.float64)
    pred = np.asarray(prediction[mask], dtype=np.float64)
    brier = float(np.mean(np.square(pred - target)))
    rate = float(target.mean())
    reference = rate * (1.0 - rate)
    raw_score = 100_000.0 * (1.0 - brier / reference)
    return {
        "rows": int(len(target)),
        "target_rate": rate,
        "prediction_mean": float(pred.mean()),
        "mean_bias": float(pred.mean() - rate),
        "prediction_std": float(pred.std()),
        "brier": brier,
        "raw_competition_score": raw_score,
        "competition_score": max(0.0, raw_score),
    }


def masks_for(frame: Any) -> dict[str, np.ndarray]:
    game_type = frame["game_type"].to_numpy()
    return {
        "all": np.ones(len(frame), dtype=bool),
        "R": game_type == "R",
        "F": game_type == "F",
    }


def fit_weights(
    matrix: np.ndarray,
    y: np.ndarray,
    mask: np.ndarray,
    slope: float,
    offset: float,
) -> tuple[np.ndarray, float]:
    x_fit = np.asarray(matrix[mask], dtype=np.float64)
    y_fit = np.asarray(y[mask], dtype=np.float64)

    def objective(weights: np.ndarray) -> float:
        prediction = calibrated(x_fit @ weights, slope, offset)
        return float(np.mean(np.square(prediction - y_fit)))

    result = minimize(
        objective,
        np.full(len(COMPONENT_ORDER), 1.0 / len(COMPONENT_ORDER)),
        method="SLSQP",
        bounds=[(0.0, 1.0)] * len(COMPONENT_ORDER),
        constraints={"type": "eq", "fun": lambda weights: float(weights.sum() - 1.0)},
        options={"ftol": 1e-15, "maxiter": 2000},
    )
    if not result.success:
        raise RuntimeError(f"weight optimization failed: {result.message}")
    weights = np.clip(np.asarray(result.x, dtype=np.float64), 0.0, 1.0)
    weights /= weights.sum()
    return weights, objective(weights)


def select_recipe(
    artifact: dict[str, np.ndarray],
    mask: np.ndarray,
    calibration_mode: str,
) -> dict[str, Any]:
    matrix = component_matrix(artifact)
    y = np.asarray(artifact["y"], dtype=np.int8)
    grid = ((1.0, 0.0),) if calibration_mode == "identity" else AFFINE_GRID
    candidates: list[dict[str, Any]] = []
    for slope, offset in grid:
        weights, loss = fit_weights(matrix, y, mask, slope, offset)
        candidates.append(
            {
                "weights": dict(zip(COMPONENT_ORDER, weights.tolist())),
                "slope": slope,
                "offset": offset,
                "source_brier": loss,
            }
        )
    return min(
        candidates,
        key=lambda item: (
            float(item["source_brier"]),
            float(item["slope"]),
            abs(float(item["offset"])),
        ),
    )


def apply_recipe(artifact: dict[str, np.ndarray], recipe: dict[str, Any]) -> np.ndarray:
    weights = np.asarray(
        [recipe["weights"][key] for key in COMPONENT_ORDER], dtype=np.float64
    )
    raw = component_matrix(artifact) @ weights
    return calibrated(raw, float(recipe["slope"]), float(recipe["offset"]))


def per_scope_metrics(
    artifact: dict[str, np.ndarray], prediction: np.ndarray, masks: dict[str, np.ndarray]
) -> dict[str, dict[str, float | int]]:
    return {
        scope: metrics(artifact["y"], prediction, mask)
        for scope, mask in masks.items()
    }


def save_anchor(
    name: str,
    target_year: int,
    artifact: dict[str, np.ndarray],
    prediction: np.ndarray,
) -> Path:
    path = PREDICTIONS / f"{name}_{target_year}.npz"
    np.savez_compressed(
        path,
        y=np.asarray(artifact["y"]),
        row_index=np.asarray(artifact["row_index"]),
        cluster=np.asarray(artifact["cluster"]),
        final_prediction=np.asarray(prediction, dtype=np.float64),
    )
    return path


def recheck_parent_direction(
    frames: dict[int, Any],
    artifacts: dict[int, dict[str, np.ndarray]],
    name: str,
    stems: dict[int, str],
    gamma: float,
) -> dict[str, Any]:
    report: dict[str, Any] = {
        "parent": "V3 component C for the same target season",
        "route": "R_only_F_parent_unchanged",
        "locked_gamma": gamma,
        "per_year": {},
    }
    for year, stem in stems.items():
        path = PREDICTIONS / f"{stem}_{year}.npz"
        if not path.exists():
            report["per_year"][str(year)] = {"missing": str(path.relative_to(ROOT))}
            continue
        candidate_artifact = load_npz(path)
        reference = artifacts[year]
        for key in ("y", "row_index", "cluster"):
            if not np.array_equal(candidate_artifact[key], reference[key]):
                raise ValueError(f"{name}/{year} alignment mismatch for {key}")
        parent = np.asarray(reference["component_C"], dtype=np.float64)
        candidate = np.asarray(candidate_artifact["catboost_outcome"], dtype=np.float64)
        masks = masks_for(frames[year])
        routed = parent.copy()
        routed[masks["R"]] += gamma * (candidate[masks["R"]] - parent[masks["R"]])
        routed = np.clip(routed, 1e-6, 1.0 - 1e-6)
        parent_metrics = per_scope_metrics(reference, parent, masks)
        candidate_metrics = per_scope_metrics(reference, candidate, masks)
        routed_metrics = per_scope_metrics(reference, routed, masks)
        scope_report: dict[str, Any] = {}
        for scope in ("all", "R", "F"):
            scope_report[scope] = {
                "parent": parent_metrics[scope],
                "candidate_full_replacement": candidate_metrics[scope],
                "locked_gamma_route": routed_metrics[scope],
                "candidate_full_gain": float(
                    candidate_metrics[scope]["raw_competition_score"]
                    - parent_metrics[scope]["raw_competition_score"]
                ),
                "locked_gamma_gain": float(
                    routed_metrics[scope]["raw_competition_score"]
                    - parent_metrics[scope]["raw_competition_score"]
                ),
            }
        scope_report["bootstrap_R_locked_gamma"] = cluster_bootstrap_score_gain(
            reference["y"],
            parent,
            routed,
            reference["cluster"].astype(str),
            masks["R"],
            1000,
            510000 + year + int(round(gamma * 1000)),
        )
        report["per_year"][str(year)] = scope_report
    r_gains = [
        year_report["R"]["locked_gamma_gain"]
        for year_report in report["per_year"].values()
        if "R" in year_report
    ]
    r_ci_lows = [
        year_report["bootstrap_R_locked_gamma"]["ci_low"]
        for year_report in report["per_year"].values()
        if "bootstrap_R_locked_gamma" in year_report
    ]
    report["honest_parent_gate"] = {
        "positive_point_every_available_year": bool(r_gains and min(r_gains) > 0.0),
        "positive_ci_lower_every_available_year": bool(r_ci_lows and min(r_ci_lows) > 0.0),
        "pass": bool(
            r_gains
            and r_ci_lows
            and min(r_gains) > 0.0
            and min(r_ci_lows) > 0.0
        ),
    }
    return report


def main() -> None:
    frames, artifacts = load_frames()
    masks = {year: masks_for(frame) for year, frame in frames.items()}

    published: dict[str, Any] = {}
    for year in sorted(artifacts):
        prediction = calibrated(
            component_matrix(artifacts[year]) @ PUBLISHED_WEIGHTS,
            PUBLISHED_SLOPE,
            PUBLISHED_OFFSET,
        )
        published[str(year)] = per_scope_metrics(artifacts[year], prediction, masks[year])

    honest: dict[str, Any] = {}
    generated: list[str] = []
    for fit_scope in ("R", "all"):
        for calibration_mode in ("identity", "grid"):
            anchor_name = f"v5_honest_m3_{fit_scope.lower()}_{calibration_mode}"
            anchor_report: dict[str, Any] = {
                "fit_scope": fit_scope,
                "calibration_mode": calibration_mode,
                "per_target": {},
            }
            for target_year in TARGET_YEARS:
                source_year = target_year - 1
                recipe = select_recipe(
                    artifacts[source_year], masks[source_year][fit_scope], calibration_mode
                )
                source_prediction = apply_recipe(artifacts[source_year], recipe)
                target_prediction = apply_recipe(artifacts[target_year], recipe)
                path = save_anchor(anchor_name, target_year, artifacts[target_year], target_prediction)
                generated.append(str(path.relative_to(ROOT)))
                anchor_report["per_target"][str(target_year)] = {
                    "source_year": source_year,
                    "recipe_selected_without_target_labels": recipe,
                    "source_metrics": per_scope_metrics(
                        artifacts[source_year], source_prediction, masks[source_year]
                    ),
                    "target_metrics": per_scope_metrics(
                        artifacts[target_year], target_prediction, masks[target_year]
                    ),
                    "artifact": str(path.relative_to(ROOT)),
                }
            honest[anchor_name] = anchor_report

    parent_rechecks = {
        "recent_denominator": recheck_parent_direction(
            frames,
            artifacts,
            "recent_denominator",
            {
                2022: "v5_recent_denominator_c_dev2223",
                2023: "v5_recent_denominator_c_dev2223",
                2024: "v5_recent_denominator_c_confirm24",
            },
            0.6,
        ),
        "conditional_history_joint": recheck_parent_direction(
            frames,
            artifacts,
            "conditional_history_joint",
            {
                2022: "v5_h2_hist_joint_dev2223",
                2023: "v5_h2_hist_joint_dev2223",
                2024: "v5_h2_hist_joint_confirm24",
            },
            0.5,
        ),
    }

    report = {
        "audit_id": "V5_ANCHOR_HONESTY_AUDIT_V1",
        "protocol": {
            "official_train_only": True,
            "test_rows_read": False,
            "row_independent_predictions": True,
            "target_years": list(TARGET_YEARS),
            "rule": "for target Y, select weights/calibration using Y-1 only and transfer unchanged",
            "component_order": list(COMPONENT_ORDER),
            "weight_constraints": "nonnegative, sum to one",
            "primary_fit_scope": "R because 2023 F is a documented regime break",
            "identity_anchor": "source-fitted weights with slope=1 and offset=0",
            "grid_anchor": {
                "source_fitted_weights": True,
                "grid": {
                    "slope": [1.0, 1.05, 1.10],
                    "offset": [-0.004, -0.006, -0.008],
                },
                "selection": "minimum source-season Brier only",
            },
        },
        "finding": {
            "published_anchor_selection_fold": 2024,
            "published_weights": dict(zip(COMPONENT_ORDER, PUBLISHED_WEIGHTS.tolist())),
            "published_affine": {
                "slope": PUBLISHED_SLOPE,
                "offset": PUBLISHED_OFFSET,
            },
            "historical_use_is_not_one_year_ahead": True,
            "invalidated_uses": [
                "V5 recent-denominator selection versus v3_sparse_m3_frozen on 2022/2023",
                "V5 H2 conditional-history selection versus v3_sparse_m3_frozen on 2022/2023",
                "V5 transfer catalog ranking versus v3_sparse_m3_frozen on 2022/2023",
            ],
            "interpretation": (
                "The frozen artifacts remain exact V3 reproductions, but their 2022/2023 "
                "calibration and ensemble weights contain choices made on 2024. They cannot "
                "serve as development anchors for a forward-transfer completion claim."
            ),
        },
        "published_frozen_anchor": published,
        "honest_one_year_ahead_anchors": honest,
        "same_parent_rechecks": parent_rechecks,
        "generated_artifacts": generated,
        "decision": {
            "previous_v5_positive_development_gates_retracted": True,
            "completion_evidence_from_previous_v5_candidates": False,
            "required_next_rule": (
                "A structural candidate must first beat its exact parent recipe in 2022, 2023, "
                "and locked 2024. Any ensemble claim must also survive both primary honest R "
                "anchors; published V3 is retained only as the actual-LB deployment reference."
            ),
        },
    }
    OUTPUT.write_text(
        json.dumps(json_safe(report), ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    summary = {
        "output": str(OUTPUT.relative_to(ROOT)),
        "published_R_bias": {
            year: published[str(year)]["R"]["mean_bias"] for year in TARGET_YEARS
        },
        "parent_recheck": {
            name: {
                "R_locked_gamma_gains": {
                    year: values["R"]["locked_gamma_gain"]
                    for year, values in result["per_year"].items()
                    if "R" in values
                },
                "pass": result["honest_parent_gate"]["pass"],
            }
            for name, result in parent_rechecks.items()
        },
        "generated_artifacts": generated,
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
