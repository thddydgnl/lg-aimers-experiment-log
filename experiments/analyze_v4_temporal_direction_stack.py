#!/usr/bin/env python3
"""Combine robust temporal residual-model directions with the V4 hybrid.

The model recipes were chosen before the 2024 confirmation by robust transfer
tests in ``analyze_v4_temporal_residual_models.py``.  This script refits those
fixed family representatives for 2022->2023 and 2023->2024, estimates only
small meta coefficients on 2023, and then confirms once on 2024.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.analyze_v4_temporal_residual_models import (  # noqa: E402
    add_raw_columns,
    build_data,
    model_recipes,
)
from experiments.analyze_v4_temporal_residual_ridge import (  # noqa: E402
    MEDIAN_OFFSET,
    REQUIRED_LOCAL,
    json_safe,
    load_frames,
    score,
)


PRED = ROOT / "experiments/results/predictions"
REPORT = ROOT / "experiments/results/v4_temporal_direction_stack.json"
RECIPE_NAMES = (
    "ridge_aug_a10000_full",
    "extra_d8_leaf1000_full",
    "hgb_l15_leaf1000_full",
    "lgb_l15_leaf1000_full",
    "cat_d4_l2_100_full",
)


def load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path) as archive:
        return {key: np.asarray(archive[key]) for key in archive.files}


def raw_metric(y: np.ndarray, prediction: np.ndarray) -> float:
    return float(score(y, np.clip(prediction, 0.0, 1.0))["raw_competition_score"])


def fit_gamma(direction: np.ndarray, residual: np.ndarray) -> float:
    denominator = float(np.dot(direction, direction))
    return float(np.dot(direction, residual) / denominator) if denominator else 0.0


def standardized_ridge(
    design: np.ndarray, residual: np.ndarray, penalty: float
) -> np.ndarray:
    scale = np.sqrt(np.mean(np.square(design), axis=0))
    scale = np.where(scale > 1e-12, scale, 1.0)
    x = design / scale
    beta = np.linalg.solve(
        x.T @ x + len(x) * penalty * np.eye(x.shape[1]),
        x.T @ residual,
    )
    return beta / scale


def main() -> None:
    frames, m3_artifacts = load_frames()
    add_raw_columns(frames, m3_artifacts)
    public = {
        year: load_npz(PRED / f"v4_public_residual_postprocess_{year}.npz")
        for year in (2023, 2024)
    }
    for year in (2023, 2024):
        if not np.array_equal(public[year]["row_index"], m3_artifacts[year]["row_index"]):
            raise ValueError(f"Artifact alignment mismatch for {year}")

    recipe_lookup = {recipe.name: recipe for recipe in model_recipes()}
    missing = set(RECIPE_NAMES) - set(recipe_lookup)
    if missing:
        raise KeyError(f"Missing fixed recipes: {sorted(missing)}")

    data = {
        2023: build_data(frames, m3_artifacts, 2022, 2023, "full"),
        2024: build_data(frames, m3_artifacts, 2023, 2024, "full"),
    }
    directions: dict[int, dict[str, np.ndarray]] = {2023: {}, 2024: {}}
    timings: dict[str, dict[str, float]] = {}
    for name in RECIPE_NAMES:
        recipe = recipe_lookup[name]
        timings[name] = {}
        for year in (2023, 2024):
            transfer = data[year]
            model = recipe.factory()
            started = time.perf_counter()
            model.fit(transfer["x_source"], transfer["residual"])
            correction = np.asarray(model.predict(transfer["x_target"]), dtype=np.float64)
            direction = np.zeros(len(public[year]["y"]), dtype=np.float64)
            direction[transfer["target_core"]] = correction
            directions[year][name] = direction
            timings[name][str(year)] = time.perf_counter() - started
            print(
                f"[{name}] {year}: rows={len(correction)} "
                f"seconds={timings[name][str(year)]:.1f}",
                flush=True,
            )

    y23 = public[2023]["y"].astype(np.float64)
    y24 = public[2024]["y"].astype(np.float64)
    accepted23 = public[2023]["accepted"].astype(np.float64)
    accepted24 = public[2024]["accepted"].astype(np.float64)
    hybrid23 = public[2023]["split_r_selected_f_fixed"].astype(np.float64)
    hybrid24 = public[2024]["split_r_selected_f_fixed"].astype(np.float64)

    report_candidates: dict[str, dict[str, object]] = {}
    payload: dict[str, np.ndarray] = {
        "y": y24,
        "row_index": public[2024]["row_index"],
        "cluster": public[2024]["cluster"],
        "accepted": accepted24,
        "hybrid": hybrid24,
    }
    baseline_23 = raw_metric(y23, accepted23)
    baseline_24 = raw_metric(y24, accepted24)
    hybrid_23_score = raw_metric(y23, hybrid23)
    hybrid_24_score = raw_metric(y24, hybrid24)

    for name in RECIPE_NAMES:
        d23 = directions[2023][name]
        d24 = directions[2024][name]
        gamma_accepted = fit_gamma(d23, y23 - accepted23)
        gamma_hybrid = fit_gamma(d23, y23 - hybrid23)
        for base_name, b23, b24, gamma in (
            ("accepted", accepted23, accepted24, gamma_accepted),
            ("hybrid", hybrid23, hybrid24, gamma_hybrid),
        ):
            candidate_name = f"{base_name}_plus_{name}"
            p23 = b23 + gamma * d23
            p24 = b24 + gamma * d24
            score23 = raw_metric(y23, p23)
            score24 = raw_metric(y24, p24)
            report_candidates[candidate_name] = {
                "gamma_selected_2023": gamma,
                "selection_score_2023": score23,
                "selection_gain_over_base": score23 - raw_metric(y23, b23),
                "confirmation_score_2024": score24,
                "confirmation_gain_over_base": score24 - raw_metric(y24, b24),
                "confirmation_gain_over_accepted": score24 - baseline_24,
                "expected_lb_median": score24 + MEDIAN_OFFSET,
            }
            payload[candidate_name] = np.clip(p24, 0.0, 1.0)

    matrix23 = np.column_stack([directions[2023][name] for name in RECIPE_NAMES])
    matrix24 = np.column_stack([directions[2024][name] for name in RECIPE_NAMES])
    joint_specs: dict[str, np.ndarray] = {
        "joint_lstsq": np.linalg.lstsq(matrix23, y23 - hybrid23, rcond=None)[0]
    }
    for penalty in (1e-3, 1e-2, 1e-1, 1.0):
        joint_specs[f"joint_ridge_{penalty:g}"] = standardized_ridge(
            matrix23, y23 - hybrid23, penalty
        )
    for name, coefficients in joint_specs.items():
        p23 = hybrid23 + matrix23 @ coefficients
        p24 = hybrid24 + matrix24 @ coefficients
        score23 = raw_metric(y23, p23)
        score24 = raw_metric(y24, p24)
        report_candidates[name] = {
            "coefficients": {
                recipe: float(value) for recipe, value in zip(RECIPE_NAMES, coefficients)
            },
            "selection_score_2023": score23,
            "selection_gain_over_base": score23 - hybrid_23_score,
            "confirmation_score_2024": score24,
            "confirmation_gain_over_base": score24 - hybrid_24_score,
            "confirmation_gain_over_accepted": score24 - baseline_24,
            "expected_lb_median": score24 + MEDIAN_OFFSET,
        }
        payload[name] = np.clip(p24, 0.0, 1.0)

    best = max(
        report_candidates.items(),
        key=lambda item: float(item[1]["confirmation_score_2024"]),
    )
    output_npz = PRED / "v4_temporal_direction_stack_2024.npz"
    np.savez_compressed(output_npz, **payload)
    report = {
        "protocol": {
            "official_train_only": True,
            "test_rows_read": False,
            "fixed_recipe_selection": (
                "family representatives were fixed by 2021->2022 and 2022->2023 "
                "robust transfer before this script"
            ),
            "meta_selection_year": 2023,
            "confirmation_year": 2024,
            "row_independent_inference": True,
        },
        "baselines": {
            "accepted_2023": baseline_23,
            "accepted_2024": baseline_24,
            "hybrid_2023": hybrid_23_score,
            "hybrid_2024": hybrid_24_score,
        },
        "recipes": list(RECIPE_NAMES),
        "timings": timings,
        "candidates": report_candidates,
        "best_observed_confirmation_diagnostic": {
            "name": best[0],
            **best[1],
            "required_local_score": REQUIRED_LOCAL,
            "crosses_required_local_score": (
                float(best[1]["confirmation_score_2024"]) > REQUIRED_LOCAL
            ),
            "warning": "2024 ranking is diagnostic, not a selection rule",
        },
        "prediction_artifact": str(output_npz.relative_to(ROOT)),
    }
    REPORT.write_text(
        json.dumps(json_safe(report), ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(
        json.dumps(
            json_safe(
                {
                    "baselines": report["baselines"],
                    "candidates": report_candidates,
                    "best": report["best_observed_confirmation_diagnostic"],
                }
            ),
            ensure_ascii=False,
            indent=2,
        ),
        flush=True,
    )
    print(f"Saved {REPORT}", flush=True)
    print(f"Saved {output_npz}", flush=True)


if __name__ == "__main__":
    main()
