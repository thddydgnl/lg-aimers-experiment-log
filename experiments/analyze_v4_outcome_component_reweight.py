#!/usr/bin/env python3
"""Reweight A-model failure-type probabilities using two historical folds."""

from __future__ import annotations

import itertools
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.analyze_v4_pitchtype_failure_prior import (  # noqa: E402
    brier_gain,
    solve_weights,
)
from experiments.analyze_v4_temporal_residual_ridge import (  # noqa: E402
    MEDIAN_OFFSET,
    REQUIRED_LOCAL,
    json_safe,
    score,
)
from experiments.v4_current_ensemble import PREDICTIONS  # noqa: E402


OUTPUT_JSON = ROOT / "experiments/results/v4_outcome_component_reweight.json"
OUTPUT_NPZ = PREDICTIONS / "v4_outcome_component_reweight_2024.npz"
YEARS = (2022, 2023, 2024)
COMPONENT_STAGE = "v4_outcome_a_components"


def load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path) as archive:
        return {key: np.asarray(archive[key]) for key in archive.files}


def component_matrix(artifact: dict[str, np.ndarray], recipe: str) -> tuple[np.ndarray, list[str]]:
    middle = np.asarray(artifact["catboost_outcome__p_0_middle"], dtype=np.float64)
    reverse = np.asarray(artifact["catboost_outcome__p_1_reverse"], dtype=np.float64)
    success = np.asarray(artifact["catboost_outcome__p_2_success"], dtype=np.float64)
    wide = np.asarray(artifact["catboost_outcome__p_3_wide"], dtype=np.float64)
    failure = 1.0 - success
    if recipe == "contrasts":
        return np.column_stack((middle - wide, reverse - wide)), [
            "middle_minus_wide",
            "reverse_minus_wide",
        ]
    if recipe == "contrasts_failure":
        return np.column_stack((middle - wide, reverse - wide, failure)), [
            "middle_minus_wide",
            "reverse_minus_wide",
            "failure_total",
        ]
    if recipe == "raw_failures":
        return np.column_stack((middle, reverse, wide)), [
            "middle",
            "reverse",
            "wide",
        ]
    if recipe == "shares":
        denominator = np.maximum(failure, 1e-6)
        return np.column_stack(
            (
                failure * (middle / denominator - 1.0 / 3.0),
                failure * (reverse / denominator - 1.0 / 3.0),
            )
        ), ["middle_share_centered", "reverse_share_centered"]
    raise ValueError(f"Unknown recipe: {recipe}")


def main() -> None:
    bases: dict[int, dict[str, np.ndarray]] = {}
    components: dict[int, dict[str, np.ndarray]] = {}
    for year in YEARS:
        bases[year] = load_npz(
            PREDICTIONS / f"v4_pitchtype_failure_tagged_locked_{year}.npz"
        )
        components[year] = load_npz(
            PREDICTIONS / f"{COMPONENT_STAGE}_{year}.npz"
        )
        for key in ("y", "row_index", "cluster"):
            if not np.array_equal(bases[year][key], components[year][key]):
                raise ValueError(f"Alignment mismatch for {year}/{key}")

    recipes = ("contrasts", "contrasts_failure", "raw_failures", "shares")
    trials: list[dict[str, Any]] = []
    payloads: dict[str, tuple[dict[int, np.ndarray], list[str]]] = {}
    selected_payload: tuple[str, np.ndarray, dict[str, Any]] | None = None
    for recipe in recipes:
        matrices: dict[int, np.ndarray] = {}
        feature_names: list[str] = []
        for year in YEARS:
            matrices[year], feature_names = component_matrix(components[year], recipe)
        payloads[recipe] = (matrices, feature_names)
        for ridge, aggregate_name, shrink in itertools.product(
            (0.01, 0.1, 1.0, 10.0),
            ("median", "mean"),
            (0.10, 0.20, 0.35, 0.50, 0.75, 1.00),
        ):
            solutions = []
            for year in (2022, 2023):
                solutions.append(
                    solve_weights(
                        matrices[year],
                        bases[year]["y"],
                        bases[year]["tagged_locked"],
                        ridge,
                    )
                )
            aggregate = (
                np.median(np.stack(solutions), axis=0)
                if aggregate_name == "median"
                else np.mean(np.stack(solutions), axis=0)
            )
            weights = shrink * aggregate
            gains = {
                str(year): brier_gain(
                    bases[year]["y"],
                    bases[year]["tagged_locked"],
                    matrices[year] @ weights,
                )
                for year in (2022, 2023)
            }
            row = {
                "recipe": recipe,
                "feature_names": feature_names,
                "ridge": ridge,
                "aggregate": aggregate_name,
                "shrink": shrink,
                "weights": weights.tolist(),
                "gains": gains,
                "robust_min_gain": float(min(gains.values())),
                "mean_gain": float(np.mean(list(gains.values()))),
                "max_abs_correction": float(
                    max(
                        np.max(np.abs(matrices[year] @ weights))
                        for year in (2022, 2023)
                    )
                ),
            }
            trials.append(row)
            key = (row["robust_min_gain"], row["mean_gain"])
            if selected_payload is None or key > (
                selected_payload[2]["robust_min_gain"],
                selected_payload[2]["mean_gain"],
            ):
                selected_payload = (recipe, weights.copy(), row)

    if selected_payload is None:
        raise RuntimeError("No component reweight candidate selected")
    selected_recipe, selected_weights, selected = selected_payload
    selected_matrices, selected_feature_names = payloads[selected_recipe]
    metrics: dict[str, dict[str, float | int]] = {}
    fold_artifacts: dict[str, str] = {}
    for year in YEARS:
        base = np.asarray(bases[year]["tagged_locked"], dtype=np.float64)
        correction = selected_matrices[year] @ selected_weights
        prediction = np.clip(base + correction, 0.0, 1.0)
        metrics[str(year)] = score(bases[year]["y"], prediction)
        output_path = PREDICTIONS / f"v4_outcome_component_reweight_{year}.npz"
        np.savez_compressed(
            output_path,
            y=bases[year]["y"],
            row_index=bases[year]["row_index"],
            cluster=bases[year]["cluster"],
            base=base,
            component_matrix=selected_matrices[year],
            weights=selected_weights,
            correction=correction,
            component_reweight=prediction,
        )
        fold_artifacts[str(year)] = str(output_path.relative_to(ROOT))

    base_2024 = score(bases[2024]["y"], bases[2024]["tagged_locked"])
    local_2024 = float(metrics["2024"]["raw_competition_score"])
    confirmation_gain = local_2024 - float(base_2024["raw_competition_score"])
    report = {
        "protocol": {
            "official_train_only": True,
            "test_rows_read": False,
            "selection_folds": [2022, 2023],
            "confirmation_fold": 2024,
            "A_model_recipe_and_seed_unchanged": True,
            "failure_components_saved_without_refit_selection": True,
        },
        "selected": selected,
        "selected_feature_names": selected_feature_names,
        "top_trials": sorted(
            trials,
            key=lambda row: (row["robust_min_gain"], row["mean_gain"]),
            reverse=True,
        )[:100],
        "metrics": metrics,
        "confirmation_2024": {
            "gain": confirmation_gain,
            "local_score": local_2024,
            "expected_lb_median": local_2024 + MEDIAN_OFFSET,
            "required_local_score": REQUIRED_LOCAL,
            "crosses_required_local_score": local_2024 > REQUIRED_LOCAL,
        },
        "fold_artifacts": fold_artifacts,
    }
    OUTPUT_JSON.write_text(
        json.dumps(json_safe(report), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "selected": selected,
                "confirmation_2024": report["confirmation_2024"],
            },
            ensure_ascii=False,
            indent=2,
        ),
        flush=True,
    )
    print(f"Saved {OUTPUT_JSON}", flush=True)
    print(f"Saved {OUTPUT_NPZ}", flush=True)


if __name__ == "__main__":
    main()
