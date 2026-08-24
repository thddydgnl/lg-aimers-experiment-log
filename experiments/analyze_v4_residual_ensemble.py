#!/usr/bin/env python3
"""Selection-only blend of robust Ridge and pitcher-context contrasts."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

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
    Config,
    json_safe,
    load_frames,
    score,
    transfer,
)


RIDGE_JSON = ROOT / "experiments/results/v4_temporal_residual_ridge.json"
RIDGE_NPZ = ROOT / "experiments/results/predictions/v4_temporal_residual_ridge_2024.npz"
CONTRAST_JSON = ROOT / "experiments/results/v4_contrast_effects.json"
CONTRAST_NPZ = ROOT / "experiments/results/predictions/v4_contrast_effects_2024.npz"
OUTPUT_JSON = ROOT / "experiments/results/v4_residual_ensemble.json"
OUTPUT_NPZ = ROOT / "experiments/results/predictions/v4_residual_ensemble_2024.npz"
WEIGHT_GRID = np.round(np.arange(0.0, 2.01, 0.05), 8)


def config_from_dict(values: dict[str, Any]) -> Config:
    return Config(
        scope=str(values["scope"]),
        k_pitcher=float(values["k_pitcher"]),
        k_hand=float(values["k_hand"]),
        k_pressure=float(values["k_pressure"]),
        alpha=float(values["alpha"]),
        gamma=float(values["gamma"]),
        training_mode=str(values["training_mode"]),
    )


def main() -> None:
    ridge_report = json.loads(RIDGE_JSON.read_text(encoding="utf-8"))
    contrast_report = json.loads(CONTRAST_JSON.read_text(encoding="utf-8"))
    frames, artifacts = load_frames()
    add_columns(frames, artifacts)
    baselines = {
        season: score(artifacts[season]["y"], artifacts[season]["m3"])
        for season in (2022, 2023, 2024)
    }

    ridge_configs = {
        mode: config_from_dict(item["config"])
        for mode, item in ridge_report["selection"]["best_by_training_mode"].items()
    }
    ridge_selection: dict[int, np.ndarray] = {}
    for source, target in ((2021, 2022), (2022, 2023)):
        predictions = []
        for config in ridge_configs.values():
            prediction, _ = transfer(
                frames[source],
                frames[target],
                artifacts[source]["m3"],
                artifacts[target]["m3"],
                config,
            )
            predictions.append(prediction)
        ridge_selection[target] = np.mean(predictions, axis=0)

    selected_route_name = str(contrast_report["selected_route"])
    route = next(item for item in ROUTES if item.name == selected_route_name)
    selected_route = contrast_report["routes"][selected_route_name]
    selected_k = {
        axis: float(value) for axis, value in selected_route["selected_k"].items()
    }
    selected_weights = {
        axis: float(value)
        for axis, value in selected_route["selected_weights"]["weights"].items()
    }
    contrast_selection: dict[int, np.ndarray] = {}
    for source_years, target in (((2020, 2021), 2022), ((2021, 2022), 2023)):
        mask, library = transition_library(
            frames, artifacts, source_years, target, route
        )
        correction = sum(
            selected_weights[axis] * library[axis][selected_k[axis]]
            for axis in AXES
        )
        prediction = np.asarray(artifacts[target]["m3"], dtype=np.float64).copy()
        prediction[mask] = np.clip(prediction[mask] + correction, 0.0, 1.0)
        contrast_selection[target] = prediction

    trials: list[dict[str, Any]] = []
    best: tuple[tuple[float, float], dict[str, Any]] | None = None
    for ridge_weight in WEIGHT_GRID:
        for contrast_weight in WEIGHT_GRID:
            if ridge_weight == 0.0 and contrast_weight == 0.0:
                continue
            gains: dict[str, float] = {}
            metrics_by_year: dict[str, Any] = {}
            for target in (2022, 2023):
                m3 = np.asarray(artifacts[target]["m3"], dtype=np.float64)
                prediction = np.clip(
                    m3
                    + ridge_weight * (ridge_selection[target] - m3)
                    + contrast_weight * (contrast_selection[target] - m3),
                    0.0,
                    1.0,
                )
                metrics = score(artifacts[target]["y"], prediction)
                gains[str(target)] = float(
                    metrics["raw_competition_score"]
                    - baselines[target]["raw_competition_score"]
                )
                metrics_by_year[str(target)] = metrics
            row = {
                "ridge_weight": float(ridge_weight),
                "contrast_weight": float(contrast_weight),
                "gains": gains,
                "robust_min_gain": float(min(gains.values())),
                "mean_gain": float(np.mean(list(gains.values()))),
                "metrics": metrics_by_year,
            }
            trials.append(row)
            rank = (row["robust_min_gain"], row["mean_gain"])
            if best is None or rank > best[0]:
                best = (rank, row)
    assert best is not None
    selected = best[1]

    # Preserve the selection-fold ensemble predictions as first-class artifacts.
    # Downstream candidate selection can then reuse exactly the same 2022/2023
    # baseline without recomputing it or accidentally consulting 2024.
    for target in (2022, 2023):
        m3 = np.asarray(artifacts[target]["m3"], dtype=np.float64)
        prediction = np.clip(
            m3
            + selected["ridge_weight"] * (ridge_selection[target] - m3)
            + selected["contrast_weight"] * (contrast_selection[target] - m3),
            0.0,
            1.0,
        )
        selection_path = OUTPUT_NPZ.with_name(
            f"{OUTPUT_NPZ.stem.rsplit('_', 1)[0]}_{target}.npz"
        )
        np.savez_compressed(
            selection_path,
            y=artifacts[target]["y"],
            row_index=artifacts[target]["row_index"],
            cluster=artifacts[target]["cluster"],
            m3=m3,
            ridge=ridge_selection[target],
            contrast=contrast_selection[target],
            residual_ensemble=prediction,
        )

    with np.load(RIDGE_NPZ, allow_pickle=False) as archive:
        ridge_2024 = np.asarray(archive["temporal_residual_ridge"], dtype=np.float64)
    with np.load(CONTRAST_NPZ, allow_pickle=False) as archive:
        contrast_2024 = np.asarray(archive["contrast_effects"], dtype=np.float64)
    m3_2024 = np.asarray(artifacts[2024]["m3"], dtype=np.float64)
    prediction_2024 = np.clip(
        m3_2024
        + selected["ridge_weight"] * (ridge_2024 - m3_2024)
        + selected["contrast_weight"] * (contrast_2024 - m3_2024),
        0.0,
        1.0,
    )
    metrics_2024 = score(artifacts[2024]["y"], prediction_2024)
    gain_2024 = float(
        metrics_2024["raw_competition_score"]
        - baselines[2024]["raw_competition_score"]
    )
    np.savez_compressed(
        OUTPUT_NPZ,
        y=artifacts[2024]["y"],
        row_index=artifacts[2024]["row_index"],
        cluster=artifacts[2024]["cluster"],
        m3=m3_2024,
        ridge=ridge_2024,
        contrast=contrast_2024,
        residual_ensemble=prediction_2024,
    )
    report = {
        "protocol": {
            "official_train_only": True,
            "test_rows_read": False,
            "leaderboard_values_used": False,
            "selection": "maximize worst total gain on 2022 and 2023",
            "confirmation": "selected blend applied once to 2024",
        },
        "fixed_estimator": {
            "median_offset": MEDIAN_OFFSET,
            "target_lb": 1190.0,
            "required_local_score": REQUIRED_LOCAL,
        },
        "baselines": baselines,
        "selected": selected,
        "top_trials": sorted(
            trials,
            key=lambda row: (row["robust_min_gain"], row["mean_gain"]),
            reverse=True,
        )[:50],
        "confirmation_2024": {
            "metrics": metrics_2024,
            "gain": gain_2024,
            "expected_lb_median": float(
                metrics_2024["raw_competition_score"] + MEDIAN_OFFSET
            ),
            "crosses_required_local_score": bool(
                metrics_2024["raw_competition_score"] > REQUIRED_LOCAL
            ),
        },
        "prediction_artifact": str(OUTPUT_NPZ.relative_to(ROOT)),
    }
    OUTPUT_JSON.write_text(
        json.dumps(json_safe(report), ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "selected_weights": {
                    "ridge": selected["ridge_weight"],
                    "contrast": selected["contrast_weight"],
                },
                "selection_gains": selected["gains"],
                "gain_2024": gain_2024,
                "score_2024": metrics_2024["raw_competition_score"],
                "expected_lb_median": metrics_2024["raw_competition_score"]
                + MEDIAN_OFFSET,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    print(f"Saved {OUTPUT_JSON}")
    print(f"Saved {OUTPUT_NPZ}")


if __name__ == "__main__":
    main()
