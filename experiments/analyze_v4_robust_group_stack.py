#!/usr/bin/env python3
"""Add preselected robust group-effect directions to the V4 stack."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.analyze_v4_robust_group_effects import (  # noqa: E402
    FEATURES,
    ROUTES,
    add_columns,
    build_corrections,
)
from experiments.analyze_v4_temporal_residual_ridge import (  # noqa: E402
    MEDIAN_OFFSET,
    REQUIRED_LOCAL,
    json_safe,
    load_frames,
    score,
)


PRED = ROOT / "experiments/results/predictions"
SOURCE_REPORT = ROOT / "experiments/results/v4_robust_group_effects.json"
STACK_REPORT = ROOT / "experiments/results/v4_conditional_ridge_stack.json"
REPORT = ROOT / "experiments/results/v4_robust_group_stack.json"


def load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path) as archive:
        return {key: np.asarray(archive[key]) for key in archive.files}


def raw_score(y: np.ndarray, prediction: np.ndarray) -> float:
    return float(score(y, np.clip(prediction, 0.0, 1.0))["raw_competition_score"])


def scalar_fit(direction: np.ndarray, residual: np.ndarray) -> float:
    denominator = float(np.dot(direction, direction))
    return float(np.dot(direction, residual) / denominator) if denominator else 0.0


def main() -> None:
    source_report = json.loads(SOURCE_REPORT.read_text(encoding="utf-8"))
    stack_report = json.loads(STACK_REPORT.read_text(encoding="utf-8"))
    selected_routes = source_report["combined_2024"]["routes"]
    selected = source_report["best_by_route"]
    route_lookup = {route.name: route for route in ROUTES}
    frames, artifacts = load_frames()
    add_columns(frames, artifacts)

    stack = {
        year: load_npz(PRED / f"v4_conditional_ridge_stack_{year}.npz")
        for year in (2023, 2024)
    }
    for year in (2023, 2024):
        if not np.array_equal(stack[year]["row_index"], artifacts[year]["row_index"]):
            raise ValueError(f"Artifact alignment mismatch for {year}")

    # Reconstruct the accepted conditional-consensus + HGB two-direction base.
    best_stack = stack_report["candidates"]["hybrid_plus_consensus_hgb_joint"]
    coefficients = best_stack["coefficients"]
    base = {
        2023: (
            stack[2023]["hybrid"].astype(np.float64)
            + float(coefficients["consensus"]) * stack[2023]["direction_consensus"]
            + float(coefficients["hgb"]) * stack[2023]["direction_hgb"]
        ),
        2024: stack[2024]["hybrid_plus_consensus_hgb_joint"].astype(np.float64),
    }

    directions: dict[int, dict[str, np.ndarray]] = {2023: {}, 2024: {}}
    for route_name in selected_routes:
        route = route_lookup[route_name]
        spec = selected[route_name]
        for year, source_year in ((2023, 2022), (2024, 2023)):
            target_mask, library = build_corrections(
                frames,
                artifacts,
                source_year,
                year,
                route,
                FEATURES[spec["feature"]],
            )
            direction = np.zeros(len(artifacts[year]["y"]), dtype=np.float64)
            direction[target_mask] = float(spec["gamma"]) * library[float(spec["k"])]
            directions[year][route_name] = direction
    for year in (2023, 2024):
        directions[year]["combined"] = sum(
            (directions[year][name] for name in selected_routes),
            np.zeros(len(base[year]), dtype=np.float64),
        )

    y = {year: stack[year]["y"].astype(np.float64) for year in (2023, 2024)}
    base_score = {year: raw_score(y[year], base[year]) for year in (2023, 2024)}
    candidates: dict[str, dict[str, object]] = {}
    payload: dict[str, np.ndarray] = {
        "y": y[2024],
        "row_index": stack[2024]["row_index"],
        "cluster": stack[2024]["cluster"],
        "base": base[2024],
    }
    for name in (*selected_routes, "combined"):
        gamma = scalar_fit(directions[2023][name], y[2023] - base[2023])
        p23 = base[2023] + gamma * directions[2023][name]
        p24 = base[2024] + gamma * directions[2024][name]
        s23 = raw_score(y[2023], p23)
        s24 = raw_score(y[2024], p24)
        candidate_name = f"base_plus_{name}"
        candidates[candidate_name] = {
            "selected_scalar": gamma,
            "selection_gain": s23 - base_score[2023],
            "confirmation_gain": s24 - base_score[2024],
            "confirmation_score": s24,
            "expected_lb_median": s24 + MEDIAN_OFFSET,
        }
        payload[candidate_name] = np.clip(p24, 0.0, 1.0)

    matrix23 = np.column_stack([directions[2023][name] for name in selected_routes])
    matrix24 = np.column_stack([directions[2024][name] for name in selected_routes])
    joint = np.linalg.lstsq(matrix23, y[2023] - base[2023], rcond=None)[0]
    p23 = base[2023] + matrix23 @ joint
    p24 = base[2024] + matrix24 @ joint
    s23 = raw_score(y[2023], p23)
    s24 = raw_score(y[2024], p24)
    candidates["base_plus_routes_joint"] = {
        "coefficients": {
            name: float(value) for name, value in zip(selected_routes, joint)
        },
        "selection_gain": s23 - base_score[2023],
        "confirmation_gain": s24 - base_score[2024],
        "confirmation_score": s24,
        "expected_lb_median": s24 + MEDIAN_OFFSET,
    }
    payload["base_plus_routes_joint"] = np.clip(p24, 0.0, 1.0)

    best = max(candidates.items(), key=lambda item: float(item[1]["confirmation_score"]))
    output24 = PRED / "v4_robust_group_stack_2024.npz"
    output23 = PRED / "v4_robust_group_stack_2023.npz"
    np.savez_compressed(
        output24,
        **payload,
        **{f"direction_{name}": directions[2024][name] for name in directions[2024]},
    )
    np.savez_compressed(
        output23,
        y=y[2023],
        row_index=stack[2023]["row_index"],
        cluster=stack[2023]["cluster"],
        base=base[2023],
        **{f"direction_{name}": directions[2023][name] for name in directions[2023]},
    )
    report = {
        "protocol": {
            "official_train_only": True,
            "test_rows_read": False,
            "route_recipes_preselected_by_two_historical_transfers": True,
            "meta_selection_year": 2023,
            "confirmation_year": 2024,
            "row_independent_lookup": True,
        },
        "selected_routes": selected_routes,
        "selected_specs": {name: selected[name] for name in selected_routes},
        "base_scores": base_score,
        "candidates": candidates,
        "best_observed_confirmation_diagnostic": {
            "name": best[0],
            **best[1],
            "required_local_score": REQUIRED_LOCAL,
            "crosses_required_local_score": (
                float(best[1]["confirmation_score"]) > REQUIRED_LOCAL
            ),
            "warning": "2024 ranking is diagnostic, not a selection rule",
        },
        "prediction_artifacts": {
            "2023": str(output23.relative_to(ROOT)),
            "2024": str(output24.relative_to(ROOT)),
        },
    }
    REPORT.write_text(
        json.dumps(json_safe(report), ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(json_safe(report["candidates"]), ensure_ascii=False, indent=2))
    print(f"Saved {REPORT}")


if __name__ == "__main__":
    main()
