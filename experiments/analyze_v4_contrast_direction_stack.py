#!/usr/bin/env python3
"""Add preselected pitcher-context contrasts to the current V4 stack."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.analyze_v4_contrast_effects import (  # noqa: E402
    AXES,
    ROUTES,
    add_columns,
    transition_library,
)
from experiments.analyze_v4_temporal_residual_ridge import (  # noqa: E402
    MEDIAN_OFFSET,
    REQUIRED_LOCAL,
    json_safe,
    load_frames,
    score,
)


PRED = ROOT / "experiments/results/predictions"
SOURCE_REPORT = ROOT / "experiments/results/v4_contrast_effects.json"
BASE_REPORT = ROOT / "experiments/results/v4_nested_direction_stack.json"
REPORT = ROOT / "experiments/results/v4_contrast_direction_stack.json"


def load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path) as archive:
        return {key: np.asarray(archive[key]) for key in archive.files}


def raw_score(y: np.ndarray, prediction: np.ndarray) -> float:
    return float(score(y, np.clip(prediction, 0.0, 1.0))["raw_competition_score"])


def main() -> None:
    source = json.loads(SOURCE_REPORT.read_text(encoding="utf-8"))
    base_report = json.loads(BASE_REPORT.read_text(encoding="utf-8"))
    route_name = source["selected_route"]
    route = next(item for item in ROUTES if item.name == route_name)
    spec = source["routes"][route_name]
    selected_k = {key: float(value) for key, value in spec["selected_k"].items()}
    selected_weights = {
        key: float(value) for key, value in spec["selected_weights"]["weights"].items()
    }

    frames, artifacts = load_frames()
    add_columns(frames, artifacts)
    stacks = {
        year: load_npz(PRED / f"v4_nested_direction_stack_{year}.npz")
        for year in (2023, 2024)
    }
    nested_coefficients = base_report["candidates"]["base_plus_nested_axes_joint"][
        "coefficients"
    ]
    base = {
        2023: stacks[2023]["base"].astype(np.float64)
        + sum(
            float(nested_coefficients[name]) * stacks[2023][f"direction_{name}"]
            for name in nested_coefficients
        ),
        2024: stacks[2024]["base_plus_nested_axes_joint"].astype(np.float64),
    }

    directions: dict[int, np.ndarray] = {}
    axis_directions: dict[int, dict[str, np.ndarray]] = {}
    for year, source_years in ((2023, (2021, 2022)), (2024, (2022, 2023))):
        mask, library = transition_library(frames, artifacts, source_years, year, route)
        correction = np.zeros(int(mask.sum()), dtype=np.float64)
        axis_directions[year] = {}
        for axis in AXES:
            weighted = selected_weights[axis] * library[axis][selected_k[axis]]
            full = np.zeros(len(mask), dtype=np.float64)
            full[mask] = weighted
            axis_directions[year][axis] = full
            correction += weighted
        full = np.zeros(len(mask), dtype=np.float64)
        full[mask] = correction
        directions[year] = full

    y = {year: stacks[year]["y"].astype(np.float64) for year in (2023, 2024)}
    base_scores = {year: raw_score(y[year], base[year]) for year in (2023, 2024)}
    residual23 = y[2023] - base[2023]
    denominator = float(np.dot(directions[2023], directions[2023]))
    scalar = float(np.dot(directions[2023], residual23) / denominator)
    fixed23 = base[2023] + scalar * directions[2023]
    fixed24 = base[2024] + scalar * directions[2024]
    fixed_s23 = raw_score(y[2023], fixed23)
    fixed_s24 = raw_score(y[2024], fixed24)

    matrix23 = np.column_stack([axis_directions[2023][axis] for axis in AXES])
    matrix24 = np.column_stack([axis_directions[2024][axis] for axis in AXES])
    coefficients = np.linalg.lstsq(matrix23, residual23, rcond=None)[0]
    joint23 = base[2023] + matrix23 @ coefficients
    joint24 = base[2024] + matrix24 @ coefficients
    joint_s23 = raw_score(y[2023], joint23)
    joint_s24 = raw_score(y[2024], joint24)

    candidates = {
        "base_plus_contrast_fixed_direction": {
            "selected_scalar": scalar,
            "selection_gain": fixed_s23 - base_scores[2023],
            "confirmation_gain": fixed_s24 - base_scores[2024],
            "confirmation_score": fixed_s24,
            "expected_lb_median": fixed_s24 + MEDIAN_OFFSET,
        },
        "base_plus_contrast_axes_joint": {
            "coefficients": {
                axis: float(value) for axis, value in zip(AXES, coefficients)
            },
            "selection_gain": joint_s23 - base_scores[2023],
            "confirmation_gain": joint_s24 - base_scores[2024],
            "confirmation_score": joint_s24,
            "expected_lb_median": joint_s24 + MEDIAN_OFFSET,
        },
    }
    best = max(candidates.items(), key=lambda item: float(item[1]["confirmation_score"]))
    output23 = PRED / "v4_contrast_direction_stack_2023.npz"
    output24 = PRED / "v4_contrast_direction_stack_2024.npz"
    np.savez_compressed(
        output23,
        y=y[2023], row_index=stacks[2023]["row_index"], cluster=stacks[2023]["cluster"],
        base=base[2023], direction_contrast=directions[2023],
        **{f"direction_{name}": values for name, values in axis_directions[2023].items()},
    )
    np.savez_compressed(
        output24,
        y=y[2024], row_index=stacks[2024]["row_index"], cluster=stacks[2024]["cluster"],
        base=base[2024], direction_contrast=directions[2024],
        base_plus_contrast_fixed_direction=np.clip(fixed24, 0.0, 1.0),
        base_plus_contrast_axes_joint=np.clip(joint24, 0.0, 1.0),
        **{f"direction_{name}": values for name, values in axis_directions[2024].items()},
    )
    report = {
        "protocol": {
            "official_train_only": True,
            "test_rows_read": False,
            "route_k_weights_preselected_on_two_historical_transfers": True,
            "meta_selection_year": 2023,
            "confirmation_year": 2024,
            "row_independent_lookup": True,
        },
        "source_route": route_name,
        "selected_k": selected_k,
        "selected_weights": selected_weights,
        "base_scores": base_scores,
        "candidates": candidates,
        "best_observed_confirmation_diagnostic": {
            "name": best[0], **best[1], "required_local_score": REQUIRED_LOCAL,
            "crosses_required_local_score": float(best[1]["confirmation_score"]) > REQUIRED_LOCAL,
            "warning": "2024 ranking is diagnostic, not a selection rule",
        },
        "prediction_artifacts": {
            "2023": str(output23.relative_to(ROOT)),
            "2024": str(output24.relative_to(ROOT)),
        },
    }
    REPORT.write_text(json.dumps(json_safe(report), ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(json_safe(candidates), ensure_ascii=False, indent=2))
    print(f"Saved {REPORT}")


if __name__ == "__main__":
    main()
