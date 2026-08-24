#!/usr/bin/env python3
"""Jointly reweight the nine grouped components of the current V4 champion."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.analyze_v4_nested_context_expansion import (  # noqa: E402
    AXES as EXPANSION_AXES,
    add_expansion_columns,
    library as nested_library,
)
from experiments.analyze_v4_nested_deviations import (  # noqa: E402
    AXES as NESTED_AXES,
    add_columns,
)
from experiments.analyze_v4_temporal_residual_ridge import (  # noqa: E402
    MEDIAN_OFFSET,
    M3_WEIGHTS,
    REQUIRED_LOCAL,
    json_safe,
    load_frames,
    score,
)
from experiments.v4_current_ensemble import (  # noqa: E402
    CALIBRATION_SLOPE,
    CONTEXT_WEIGHT,
    LEVEL_WEIGHT,
    PREDICTIONS,
    STABILITY_B_WEIGHT,
    STABILITY_C_WEIGHT,
    ensemble_weights,
    load_npz,
)


OUTPUT_JSON = ROOT / "experiments/results/v4_joint_reweight.json"
OUTPUT_NPZ = PREDICTIONS / "v4_joint_reweight_2024.npz"
NESTED_REPORT = ROOT / "experiments/results/v4_nested_deviations.json"
EXPANSION_REPORT = ROOT / "experiments/results/v4_nested_context_constrained.json"
YEARS = (2022, 2023, 2024)
SELECTION_YEARS = (2022, 2023)
COMPONENT_NAMES = (
    "ridge",
    "contrast",
    "current_context",
    "current_level",
    "trackman_stability_c",
    "trackman_stability_b",
    "nested_base",
    "nested_context_core",
    "linear_direction",
)
ADJUST_GRID = (-0.50, -0.35, -0.25, -0.15, -0.10, -0.05, 0.0, 0.05,
               0.10, 0.15, 0.25, 0.35, 0.50)
LINEAR_GRID = (0.0, 0.01, 0.02, 0.03, 0.04, 0.05, 0.075, 0.10)


def aligned(reference: dict[str, np.ndarray], candidate: dict[str, np.ndarray],
            label: str) -> None:
    for key in ("y", "row_index"):
        if not np.array_equal(reference[key], candidate[key]):
            raise ValueError(f"Alignment mismatch for {label}/{key}")


def build_components(
    frames: dict[int, Any], artifacts: dict[int, dict[str, np.ndarray]], year: int,
    source: tuple[int, int], nested_values: dict[str, Any],
    expansion_values: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    reference = artifacts[year]
    residual = load_npz(PREDICTIONS / f"v4_residual_ensemble_{year}.npz")
    aligned(reference, residual, f"residual/{year}")
    m3 = np.asarray(residual["m3"], dtype=np.float64)
    ridge = 0.75 * (np.asarray(residual["ridge"], dtype=np.float64) - m3)
    contrast = 0.85 * (np.asarray(residual["contrast"], dtype=np.float64) - m3)
    residual_reconstruction = m3 + ridge + contrast
    residual_error = float(np.max(np.abs(
        residual_reconstruction
        - np.asarray(residual["residual_ensemble"], dtype=np.float64)
    )))

    numeric = {}
    for key, stem in (
        ("base", "v4_numeric_cat_current_tmctx_seed42"),
        ("context", "v4_numeric_cat_current_context_tmctx_seed42"),
        ("level", "v4_numeric_cat_current_context_level_tmctx_seed42"),
    ):
        item = load_npz(PREDICTIONS / f"{stem}_{year}.npz")
        aligned(reference, item, f"{stem}/{year}")
        numeric[key] = np.asarray(item["catboost_numeric"], dtype=np.float64)
    context = CONTEXT_WEIGHT * (numeric["context"] - numeric["base"])
    level = LEVEL_WEIGHT * (numeric["level"] - numeric["context"])

    c_stem = ("v4_outcome_c_trackman_stability_backtest" if year < 2024
              else "v4_outcome_c_trackman_stability")
    b_stem = ("v4_outcome_b_trackman_stability_backtest" if year < 2024
              else "v4_outcome_b_trackman_stability")
    c_item = load_npz(PREDICTIONS / f"{c_stem}_{year}.npz")
    b_item = load_npz(PREDICTIONS / f"{b_stem}_{year}.npz")
    aligned(reference, c_item, f"{c_stem}/{year}")
    aligned(reference, b_item, f"{b_stem}/{year}")
    stability_c = (
        CALIBRATION_SLOPE * STABILITY_C_WEIGHT * M3_WEIGHTS["C"]
        * (np.asarray(c_item["catboost_outcome"], dtype=np.float64)
           - np.asarray(reference["component_C"], dtype=np.float64))
    )
    stability_b = (
        CALIBRATION_SLOPE * STABILITY_B_WEIGHT * M3_WEIGHTS["B"]
        * (np.asarray(b_item["catboost_outcome"], dtype=np.float64)
           - np.asarray(reference["component_B"], dtype=np.float64))
    )

    nested_k = {key: float(value)
                for key, value in nested_values["selected_k"].items()}
    nested_weights = {
        key: float(value)
        for key, value in nested_values["selected_weights"]["weights"].items()
    }
    mask, values = nested_library(
        frames, artifacts, source, year, "ALL", "ALL", NESTED_AXES,
        tuple(sorted(set(nested_k.values())))
    )
    nested = np.zeros(len(m3), dtype=np.float64)
    nested[mask] = sum(
        nested_weights[axis.name] * values[axis.name][nested_k[axis.name]]
        for axis in NESTED_AXES
    )

    expansion_k = {key: float(value)
                   for key, value in expansion_values["selected_k"].items()}
    expansion_weights = {
        key: float(value)
        for key, value in expansion_values["selected_weights"].items()
    }
    mask, values = nested_library(
        frames, artifacts, source, year, "R", "R_CORE", EXPANSION_AXES,
        tuple(sorted(set(expansion_k.values())))
    )
    expansion = np.zeros(len(m3), dtype=np.float64)
    expansion[mask] = sum(
        expansion_weights[axis.name]
        * values[axis.name][expansion_k[axis.name]]
        for axis in EXPANSION_AXES
    )

    champion = np.clip(
        m3 + ridge + contrast + context + level + stability_c + stability_b
        + nested + expansion,
        0.0,
        1.0,
    )
    linear_item = load_npz(PREDICTIONS / f"v2_linear_cfg00_{year}.npz")
    aligned(reference, linear_item, f"v2_linear_cfg00/{year}")
    linear_direction = (
        np.asarray(linear_item["linear"], dtype=np.float64) - champion
    )
    matrix = np.column_stack([
        ridge,
        contrast,
        context,
        level,
        stability_c,
        stability_b,
        nested,
        expansion,
        linear_direction,
    ])
    diagnostics = {
        "residual_reconstruction_max_abs": residual_error,
        "champion_min": float(champion.min()),
        "champion_max": float(champion.max()),
    }
    return champion, matrix, diagnostics


def quadratic_terms(y: np.ndarray, baseline: np.ndarray,
                    matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    residual = y - baseline
    denominator = float(len(y)) * float(np.mean(y)) * float(1.0 - np.mean(y))
    return (
        100_000.0 * 2.0 * (matrix.T @ residual) / denominator,
        100_000.0 * (matrix.T @ matrix) / denominator,
    )


def gain(weights: np.ndarray, linear: np.ndarray,
         gram: np.ndarray) -> float:
    return float(weights @ linear - weights @ gram @ weights)


def coordinate_search(
    terms: dict[int, tuple[np.ndarray, np.ndarray]],
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    weights = np.zeros(len(COMPONENT_NAMES), dtype=np.float64)
    grids = (*([ADJUST_GRID] * 8), LINEAR_GRID)
    trace: list[dict[str, Any]] = []
    for sweep in range(10):
        changed = False
        for index, (name, grid) in enumerate(zip(COMPONENT_NAMES, grids)):
            best: tuple[tuple[float, float], float, dict[str, float]] | None = None
            for value in grid:
                candidate = weights.copy()
                candidate[index] = value
                gains = {
                    str(year): gain(candidate, *terms[year])
                    for year in SELECTION_YEARS
                }
                rank = (float(min(gains.values())),
                        float(np.mean(list(gains.values()))))
                if best is None or rank > best[0]:
                    best = (rank, value, gains)
            assert best is not None
            if best[1] != weights[index]:
                changed = True
                weights[index] = best[1]
            trace.append({
                "sweep": sweep + 1,
                "component": name,
                "adjustment": float(weights[index]),
                "robust_min_gain": best[0][0],
                "mean_gain": best[0][1],
                "gains": best[2],
            })
        if not changed:
            break
    return weights, trace


def main() -> None:
    frames, artifacts = load_frames()
    add_columns(frames, artifacts)
    add_expansion_columns(frames, artifacts)
    nested_report = json.loads(NESTED_REPORT.read_text(encoding="utf-8"))
    nested_values = nested_report["routes"][nested_report["selected_route"]]
    expansion_report = json.loads(EXPANSION_REPORT.read_text(encoding="utf-8"))
    expansion_values = expansion_report["routes"]["core_from_r"]
    sources = {2022: (2020, 2021), 2023: (2021, 2022), 2024: (2022, 2023)}
    champions: dict[int, np.ndarray] = {}
    matrices: dict[int, np.ndarray] = {}
    diagnostics: dict[str, Any] = {}
    baselines: dict[str, Any] = {}
    terms: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    for year in YEARS:
        champion, matrix, diag = build_components(
            frames, artifacts, year, sources[year], nested_values, expansion_values
        )
        champions[year] = champion
        matrices[year] = matrix
        diagnostics[str(year)] = diag
        baselines[str(year)] = score(artifacts[year]["y"], champion)
        if year in SELECTION_YEARS:
            terms[year] = quadratic_terms(
                np.asarray(artifacts[year]["y"], dtype=np.float64),
                champion,
                matrix,
            )

    adjustments, trace = coordinate_search(terms)
    scale_trials: list[dict[str, Any]] = []
    for scale in (0.25, 0.50, 0.75, 1.00):
        candidate = adjustments * scale
        gains = {str(year): gain(candidate, *terms[year])
                 for year in SELECTION_YEARS}
        scale_trials.append({
            "scale": scale,
            "gains": gains,
            "robust_min_gain": float(min(gains.values())),
            "mean_gain": float(np.mean(list(gains.values()))),
        })
    selected_scale = max(
        scale_trials,
        key=lambda row: (row["robust_min_gain"], row["mean_gain"]),
    )
    selected_adjustments = adjustments * float(selected_scale["scale"])
    predictions: dict[str, np.ndarray] = {}
    confirmations: dict[str, Any] = {}
    for label, values in (
        ("selected", selected_adjustments),
        ("half", adjustments * 0.5),
        ("full", adjustments),
    ):
        prediction = np.clip(champions[2024] + matrices[2024] @ values, 0.0, 1.0)
        metric = score(artifacts[2024]["y"], prediction)
        confirmations[label] = {
            "metrics": metric,
            "gain": float(
                metric["raw_competition_score"]
                - baselines["2024"]["raw_competition_score"]
            ),
            "expected_lb_median": float(
                metric["raw_competition_score"] + MEDIAN_OFFSET
            ),
        }
        predictions[label] = prediction
        print(f"[{label}] gain={confirmations[label]['gain']:+.4f} "
              f"local={metric['raw_competition_score']:.4f}", flush=True)

    primary = predictions["selected"]
    primary_metrics = confirmations["selected"]["metrics"]
    np.savez_compressed(
        OUTPUT_NPZ,
        y=artifacts[2024]["y"],
        row_index=artifacts[2024]["row_index"],
        cluster=artifacts[2024]["cluster"],
        champion=champions[2024],
        joint_reweight=primary,
        candidate_half=predictions["half"],
        candidate_full=predictions["full"],
    )
    report = {
        "protocol": {
            "official_train_only": True,
            "test_rows_read": False,
            "leaderboard_values_used": False,
            "row_independent": True,
            "selection": "worst-fold coordinate search on 2022 and 2023",
            "confirmation": "apply once to 2024",
            "adjustment_semantics": (
                "first eight values multiply existing contributions by 1+value; "
                "linear_direction is a direct blend coefficient"
            ),
        },
        "fixed_estimator": {
            "median_offset": MEDIAN_OFFSET,
            "required_local_score": REQUIRED_LOCAL,
            "target_lb": 1190.0,
        },
        "current_ensemble_weights": ensemble_weights(),
        "component_names": list(COMPONENT_NAMES),
        "base_metrics": baselines,
        "diagnostics": diagnostics,
        "coordinate_adjustments": {
            name: float(value)
            for name, value in zip(COMPONENT_NAMES, adjustments)
        },
        "coordinate_trace": trace,
        "scale_trials": scale_trials,
        "selected_scale": selected_scale,
        "selected_adjustments": {
            name: float(value)
            for name, value in zip(COMPONENT_NAMES, selected_adjustments)
        },
        "confirmations_2024": confirmations,
        "primary_2024": {
            "metrics": primary_metrics,
            "expected_lb_median": float(
                primary_metrics["raw_competition_score"] + MEDIAN_OFFSET
            ),
            "crosses_required_local_score": bool(
                primary_metrics["raw_competition_score"] > REQUIRED_LOCAL
            ),
        },
        "prediction_artifact": str(OUTPUT_NPZ.relative_to(ROOT)),
    }
    OUTPUT_JSON.write_text(
        json.dumps(json_safe(report), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps({
        "selected_scale": selected_scale,
        "adjustments": report["selected_adjustments"],
        "score_2024": primary_metrics["raw_competition_score"],
        "expected_lb_median": (
            primary_metrics["raw_competition_score"] + MEDIAN_OFFSET
        ),
    }, ensure_ascii=False, indent=2))
    print(f"Saved {OUTPUT_JSON}")
    print(f"Saved {OUTPUT_NPZ}")


if __name__ == "__main__":
    main()
