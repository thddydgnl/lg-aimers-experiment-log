#!/usr/bin/env python3
"""Robust next-season pitcher/batter residual level effects for M3."""

from __future__ import annotations

import itertools
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.analyze_v4_temporal_residual_ridge import (  # noqa: E402
    MEDIAN_OFFSET,
    REQUIRED_LOCAL,
    json_safe,
    load_frames,
    score,
)


OUTPUT_JSON = ROOT / "experiments/results/v4_level_effects.json"
OUTPUT_NPZ = ROOT / "experiments/results/predictions/v4_level_effects_2024.npz"
SELECTION_TRANSITIONS = (((2020, 2021), 2022), ((2021, 2022), 2023))
CONFIRMATION = ((2022, 2023), 2024)
K_GRID = (500.0, 1000.0, 2000.0, 3000.0, 5000.0, 10000.0, 20000.0, 50000.0)
DECAY_GRID = (0.40, 0.60, 0.75, 0.85, 1.00)
WEIGHT_GRID = (0.0, 0.20, 0.35, 0.50, 0.65, 0.80, 1.00, 1.25, 1.50, 2.00)
AXES = {"pitcher": "pitcher_id", "batter": "batter_id"}


@dataclass(frozen=True)
class Route:
    name: str
    source_scope: str
    target_scope: str


ROUTES = (
    Route("r_from_r", "R", "R"),
    Route("all_from_all", "ALL", "ALL"),
)


def mask_for(frame: pd.DataFrame, scope: str) -> np.ndarray:
    if scope == "ALL":
        return np.ones(len(frame), dtype=bool)
    return frame["game_type"].eq(scope).to_numpy()


def level_lookup(
    frames: dict[int, pd.DataFrame],
    artifacts: dict[int, dict[str, np.ndarray]],
    source_years: tuple[int, ...],
    target_year: int,
    route: Route,
    entity: str,
    decay: float,
    k: float,
) -> tuple[np.ndarray, np.ndarray]:
    source_entities: list[np.ndarray] = []
    source_residuals: list[np.ndarray] = []
    source_weights: list[np.ndarray] = []
    latest = max(source_years)
    for season in source_years:
        mask = mask_for(frames[season], route.source_scope)
        source_entities.append(frames[season].loc[mask, entity].to_numpy())
        source_residuals.append(
            (
                np.asarray(artifacts[season]["y"], dtype=np.float64)
                - np.asarray(artifacts[season]["m3"], dtype=np.float64)
            )[mask]
        )
        source_weights.append(
            np.full(int(mask.sum()), decay ** (latest - season), dtype=np.float64)
        )
    entities = np.concatenate(source_entities)
    residual = np.concatenate(source_residuals)
    weight = np.concatenate(source_weights)
    table = pd.DataFrame(
        {"entity": entities, "weighted_residual": weight * residual, "weight": weight}
    ).groupby("entity", sort=False, observed=True).sum()
    table["effect"] = table["weighted_residual"] / (table["weight"] + k)
    target_mask = mask_for(frames[target_year], route.target_scope)
    target_entities = frames[target_year].loc[target_mask, entity]
    effect = table["effect"].reindex(target_entities.to_numpy()).fillna(0.0)
    return target_mask, effect.to_numpy(dtype=np.float64)


def gain(
    y: np.ndarray,
    baseline: np.ndarray,
    mask: np.ndarray,
    correction: np.ndarray,
) -> float:
    residual = y[mask] - baseline[mask]
    improvement = (
        2.0 * float(np.dot(residual, correction))
        - float(np.dot(correction, correction))
    ) / float(len(y))
    rate = float(np.mean(y))
    return 100_000.0 * improvement / (rate * (1.0 - rate))


def main() -> None:
    frames, artifacts = load_frames()
    baselines = {
        season: score(artifacts[season]["y"], artifacts[season]["m3"])
        for season in (2020, 2021, 2022, 2023, 2024)
    }
    route_reports: dict[str, Any] = {}
    route_predictions: dict[str, np.ndarray] = {}

    for route in ROUTES:
        selected_axis: dict[str, dict[str, Any]] = {}
        libraries: dict[
            tuple[tuple[int, ...], int, str, float, float], tuple[np.ndarray, np.ndarray]
        ] = {}
        for axis, entity in AXES.items():
            trials: list[dict[str, Any]] = []
            for decay in DECAY_GRID:
                for k in K_GRID:
                    gains: dict[str, float] = {}
                    for source_years, target_year in SELECTION_TRANSITIONS:
                        key = (source_years, target_year, axis, decay, k)
                        libraries[key] = level_lookup(
                            frames,
                            artifacts,
                            source_years,
                            target_year,
                            route,
                            entity,
                            decay,
                            k,
                        )
                        mask, effect = libraries[key]
                        gains[f"{'+'.join(map(str, source_years))}_to_{target_year}"] = gain(
                            np.asarray(artifacts[target_year]["y"], dtype=np.float64),
                            np.asarray(artifacts[target_year]["m3"], dtype=np.float64),
                            mask,
                            effect,
                        )
                    trials.append(
                        {
                            "decay": decay,
                            "k": k,
                            "gains": gains,
                            "robust_min_gain": float(min(gains.values())),
                            "mean_gain": float(np.mean(list(gains.values()))),
                        }
                    )
            best_axis = max(
                trials,
                key=lambda row: (row["robust_min_gain"], row["mean_gain"]),
            )
            selected_axis[axis] = {
                **best_axis,
                "top_trials": [
                    dict(row)
                    for row in sorted(
                        trials,
                        key=lambda row: (
                            row["robust_min_gain"], row["mean_gain"]
                        ),
                        reverse=True,
                    )[:20]
                ],
            }

        weight_trials: list[dict[str, Any]] = []
        axes = list(AXES)
        for values in itertools.product(WEIGHT_GRID, repeat=len(axes)):
            if not any(values):
                continue
            weights = dict(zip(axes, values))
            gains: dict[str, float] = {}
            for source_years, target_year in SELECTION_TRANSITIONS:
                correction = None
                target_mask = None
                for axis in axes:
                    spec = selected_axis[axis]
                    key = (
                        source_years,
                        target_year,
                        axis,
                        float(spec["decay"]),
                        float(spec["k"]),
                    )
                    mask, effect = libraries[key]
                    target_mask = mask
                    part = weights[axis] * effect
                    correction = part if correction is None else correction + part
                assert target_mask is not None and correction is not None
                gains[f"{'+'.join(map(str, source_years))}_to_{target_year}"] = gain(
                    np.asarray(artifacts[target_year]["y"], dtype=np.float64),
                    np.asarray(artifacts[target_year]["m3"], dtype=np.float64),
                    target_mask,
                    correction,
                )
            weight_trials.append(
                {
                    "weights": weights,
                    "gains": gains,
                    "robust_min_gain": float(min(gains.values())),
                    "mean_gain": float(np.mean(list(gains.values()))),
                }
            )
        selected_weights = max(
            weight_trials,
            key=lambda row: (row["robust_min_gain"], row["mean_gain"]),
        )

        correction_2024 = None
        target_mask = None
        for axis, entity in AXES.items():
            spec = selected_axis[axis]
            mask, effect = level_lookup(
                frames,
                artifacts,
                CONFIRMATION[0],
                CONFIRMATION[1],
                route,
                entity,
                float(spec["decay"]),
                float(spec["k"]),
            )
            target_mask = mask
            part = float(selected_weights["weights"][axis]) * effect
            correction_2024 = part if correction_2024 is None else correction_2024 + part
        assert target_mask is not None and correction_2024 is not None
        prediction = np.asarray(artifacts[2024]["m3"], dtype=np.float64).copy()
        prediction[target_mask] = np.clip(
            prediction[target_mask] + correction_2024, 0.0, 1.0
        )
        metrics = score(artifacts[2024]["y"], prediction)
        confirmation_gain = float(
            metrics["raw_competition_score"]
            - baselines[2024]["raw_competition_score"]
        )
        route_predictions[route.name] = prediction
        route_reports[route.name] = {
            "route": route.__dict__,
            "selected_axis": selected_axis,
            "selected_weights": selected_weights,
            "top_weight_trials": sorted(
                weight_trials,
                key=lambda row: (row["robust_min_gain"], row["mean_gain"]),
                reverse=True,
            )[:30],
            "confirmation_2024": {
                "metrics": metrics,
                "gain": confirmation_gain,
                "correction_mean": float(correction_2024.mean()),
                "correction_std": float(correction_2024.std()),
                "correction_max_abs": float(np.max(np.abs(correction_2024))),
            },
        }
        print(
            f"[{route.name}] min={selected_weights['robust_min_gain']:+.4f} "
            f"mean={selected_weights['mean_gain']:+.4f} "
            f"confirm={confirmation_gain:+.4f} local={metrics['raw_competition_score']:.4f}",
            flush=True,
        )

    selected_route = max(
        ROUTES,
        key=lambda route: (
            route_reports[route.name]["selected_weights"]["robust_min_gain"],
            route_reports[route.name]["selected_weights"]["mean_gain"],
        ),
    )
    primary = route_predictions[selected_route.name]
    metrics = score(artifacts[2024]["y"], primary)
    payload: dict[str, np.ndarray] = {
        "y": artifacts[2024]["y"],
        "row_index": artifacts[2024]["row_index"],
        "cluster": artifacts[2024]["cluster"],
        "m3": artifacts[2024]["m3"],
        "level_effects": primary,
    }
    for name, prediction in route_predictions.items():
        payload[f"route_{name}"] = prediction
    np.savez_compressed(OUTPUT_NPZ, **payload)
    report = {
        "protocol": {
            "official_train_only": True,
            "test_rows_read": False,
            "leaderboard_values_used": False,
            "row_independent_target_lookup": True,
            "selection": "worst gain across two two-season-source transfers",
            "confirmation": "2022+2023 residual levels -> 2024",
        },
        "fixed_estimator": {
            "median_offset": MEDIAN_OFFSET,
            "target_lb": 1190.0,
            "required_local_score": REQUIRED_LOCAL,
        },
        "baselines": baselines,
        "routes": route_reports,
        "selected_route": selected_route.name,
        "primary_2024": {
            "metrics": metrics,
            "expected_lb_median": float(metrics["raw_competition_score"] + MEDIAN_OFFSET),
            "crosses_required_local_score": bool(
                metrics["raw_competition_score"] > REQUIRED_LOCAL
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
                "selected_route": selected_route.name,
                "score_2024": metrics["raw_competition_score"],
                "expected_lb_median": metrics["raw_competition_score"] + MEDIAN_OFFSET,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    print(f"Saved {OUTPUT_JSON}")
    print(f"Saved {OUTPUT_NPZ}")


if __name__ == "__main__":
    main()
