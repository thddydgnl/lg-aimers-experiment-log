#!/usr/bin/env python3
"""Strict temporal pitcher-context residual contrasts for M3.

The table value is a child-cell residual mean minus its pitcher's residual
mean, shrunk by child sample size.  This removes the unstable pitcher level
and keeps only context-dependent behaviour.  Configuration selection uses two
two-season-source transfers, 2020+2021->2022 and 2021+2022->2023; 2024 is read
once after selection.
"""

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


OUTPUT_JSON = ROOT / "experiments/results/v4_contrast_effects.json"
OUTPUT_NPZ = ROOT / "experiments/results/predictions/v4_contrast_effects_2024.npz"
SELECTION_TRANSITIONS = (
    ((2020, 2021), 2022),
    ((2021, 2022), 2023),
)
CONFIRMATION = ((2022, 2023), 2024)
K_GRID = (300.0, 500.0, 800.0, 1000.0, 1500.0, 2000.0, 3000.0, 5000.0, 10000.0)
WEIGHT_GRID = (
    0.0,
    0.20,
    0.35,
    0.50,
    0.65,
    0.80,
    1.00,
    1.25,
    1.50,
    1.75,
    2.00,
    2.50,
)

AXES: dict[str, tuple[str, ...]] = {
    "batter_hand": ("pitcher_id", "batter_hand"),
    "two_strike": ("pitcher_id", "two_strike"),
    "runner_present": ("pitcher_id", "runner_present"),
}


@dataclass(frozen=True)
class Route:
    name: str
    source_scope: str
    target_scope: str


ROUTES = (
    Route("core_from_r", "R", "R_CORE"),
    Route("r_from_r", "R", "R"),
    Route("all_from_all", "ALL", "ALL"),
)


def add_columns(
    frames: dict[int, pd.DataFrame], artifacts: dict[int, dict[str, np.ndarray]]
) -> None:
    full = pd.read_csv(
        ROOT / "open/data/train.csv",
        usecols=["num_runners_on"],
        encoding="utf-8-sig",
        low_memory=False,
    )
    for season, frame in frames.items():
        rows = np.asarray(artifacts[season]["row_index"], dtype=np.int64)
        frame["num_runners_on"] = full.iloc[rows]["num_runners_on"].to_numpy()
        frame["two_strike"] = frame["strikes_before"].eq(2).astype(np.int8)
        frame["runner_present"] = frame["num_runners_on"].gt(0).astype(np.int8)


def mask_for(frame: pd.DataFrame, scope: str) -> np.ndarray:
    if scope == "ALL":
        return np.ones(len(frame), dtype=bool)
    if scope == "R":
        return frame["game_type"].eq("R").to_numpy()
    return frame["domain"].eq(scope).to_numpy()


def source_bundle(
    frames: dict[int, pd.DataFrame],
    artifacts: dict[int, dict[str, np.ndarray]],
    seasons: tuple[int, ...],
    scope: str,
) -> tuple[pd.DataFrame, np.ndarray]:
    frame_parts: list[pd.DataFrame] = []
    residual_parts: list[np.ndarray] = []
    for season in seasons:
        mask = mask_for(frames[season], scope)
        frame_parts.append(frames[season].loc[mask].reset_index(drop=True))
        residual_parts.append(
            (
                np.asarray(artifacts[season]["y"], dtype=np.float64)
                - np.asarray(artifacts[season]["m3"], dtype=np.float64)
            )[mask]
        )
    return pd.concat(frame_parts, ignore_index=True), np.concatenate(residual_parts)


def contrast_lookup(
    source: pd.DataFrame,
    residual: np.ndarray,
    target: pd.DataFrame,
    keys: tuple[str, ...],
    k: float,
) -> np.ndarray:
    work = source.loc[:, list(keys)].copy()
    work["_residual"] = residual
    parent = work.groupby("pitcher_id", sort=False, observed=True)["_residual"].mean()
    child = work.groupby(list(keys), sort=False, observed=True)["_residual"].agg(
        ["mean", "size"]
    )
    pitcher_index = child.index.get_level_values("pitcher_id")
    parent_mean = parent.reindex(pitcher_index).to_numpy(dtype=np.float64)
    count = child["size"].to_numpy(dtype=np.float64)
    child["effect"] = (
        count
        * (child["mean"].to_numpy(dtype=np.float64) - parent_mean)
        / (count + k)
    )
    target_index = pd.MultiIndex.from_frame(target.loc[:, list(keys)])
    return child["effect"].reindex(target_index).fillna(0.0).to_numpy(dtype=np.float64)


def transition_library(
    frames: dict[int, pd.DataFrame],
    artifacts: dict[int, dict[str, np.ndarray]],
    source_years: tuple[int, ...],
    target_year: int,
    route: Route,
) -> tuple[np.ndarray, dict[str, dict[float, np.ndarray]]]:
    source, residual = source_bundle(
        frames, artifacts, source_years, route.source_scope
    )
    target_mask = mask_for(frames[target_year], route.target_scope)
    target = frames[target_year].loc[target_mask].reset_index(drop=True)
    library = {
        axis: {
            k: contrast_lookup(source, residual, target, keys, k) for k in K_GRID
        }
        for axis, keys in AXES.items()
    }
    return target_mask, library


def gain_from_correction(
    y: np.ndarray,
    baseline: np.ndarray,
    mask: np.ndarray,
    correction: np.ndarray,
) -> float:
    residual = y[mask] - baseline[mask]
    brier_improvement = (
        2.0 * float(np.dot(residual, correction))
        - float(np.dot(correction, correction))
    ) / float(len(y))
    rate = float(np.mean(y))
    return 100_000.0 * brier_improvement / (rate * (1.0 - rate))


def main() -> None:
    frames, artifacts = load_frames()
    add_columns(frames, artifacts)
    baselines = {
        season: score(artifacts[season]["y"], artifacts[season]["m3"])
        for season in (2020, 2021, 2022, 2023, 2024)
    }
    route_reports: dict[str, Any] = {}
    route_predictions: dict[str, np.ndarray] = {}

    for route in ROUTES:
        selection_libraries = {
            (source_years, target_year): transition_library(
                frames, artifacts, source_years, target_year, route
            )
            for source_years, target_year in SELECTION_TRANSITIONS
        }
        selected_k: dict[str, float] = {}
        axis_trials: dict[str, list[dict[str, Any]]] = {}
        for axis in AXES:
            trials: list[dict[str, Any]] = []
            for k in K_GRID:
                gains: dict[str, float] = {}
                for source_years, target_year in SELECTION_TRANSITIONS:
                    mask, library = selection_libraries[(source_years, target_year)]
                    gains[f"{'+'.join(map(str, source_years))}_to_{target_year}"] = (
                        gain_from_correction(
                            np.asarray(artifacts[target_year]["y"], dtype=np.float64),
                            np.asarray(artifacts[target_year]["m3"], dtype=np.float64),
                            mask,
                            library[axis][k],
                        )
                    )
                trials.append(
                    {
                        "k": k,
                        "gains": gains,
                        "robust_min_gain": float(min(gains.values())),
                        "mean_gain": float(np.mean(list(gains.values()))),
                    }
                )
            best = max(
                trials,
                key=lambda row: (row["robust_min_gain"], row["mean_gain"]),
            )
            selected_k[axis] = float(best["k"])
            axis_trials[axis] = trials

        weight_trials: list[dict[str, Any]] = []
        axes = list(AXES)
        for values in itertools.product(WEIGHT_GRID, repeat=len(axes)):
            if not any(values):
                continue
            weights = dict(zip(axes, values))
            gains: dict[str, float] = {}
            for source_years, target_year in SELECTION_TRANSITIONS:
                mask, library = selection_libraries[(source_years, target_year)]
                correction = sum(
                    weights[axis] * library[axis][selected_k[axis]] for axis in axes
                )
                gains[f"{'+'.join(map(str, source_years))}_to_{target_year}"] = (
                    gain_from_correction(
                        np.asarray(artifacts[target_year]["y"], dtype=np.float64),
                        np.asarray(artifacts[target_year]["m3"], dtype=np.float64),
                        mask,
                        correction,
                    )
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

        target_mask, confirmation_library = transition_library(
            frames, artifacts, CONFIRMATION[0], CONFIRMATION[1], route
        )
        correction_2024 = sum(
            selected_weights["weights"][axis]
            * confirmation_library[axis][selected_k[axis]]
            for axis in axes
        )
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
            "selected_k": selected_k,
            "selected_weights": selected_weights,
            "axis_trials": axis_trials,
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
    primary_metrics = score(artifacts[2024]["y"], primary)
    payload: dict[str, np.ndarray] = {
        "y": artifacts[2024]["y"],
        "row_index": artifacts[2024]["row_index"],
        "cluster": artifacts[2024]["cluster"],
        "m3": artifacts[2024]["m3"],
        "contrast_effects": primary,
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
            "selection_transitions": [
                f"{'+'.join(map(str, source))}->{target}"
                for source, target in SELECTION_TRANSITIONS
            ],
            "confirmation": "2022+2023 residual tables -> 2024",
            "effect": "shrunk child residual mean minus pitcher residual mean",
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
        json.dumps(json_safe(report), ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "selected_route": selected_route.name,
                "score_2024": primary_metrics["raw_competition_score"],
                "expected_lb_median": primary_metrics["raw_competition_score"]
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
