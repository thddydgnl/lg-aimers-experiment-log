#!/usr/bin/env python3
"""Hierarchical next-season residual deviations on pitcher context cells.

Four effects are estimated from completed prior seasons: pitcher->platoon,
platoon->count advantage, advantage->exact count, and platoon->runner state.
Each child mean is centered on its parent and shrunk by child sample size.
Shrinkage and blend weights are selected by their worst gain on two untouched
temporal transfers.  The chosen construction is then applied once to 2024.
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
from experiments.v4_current_ensemble import (  # noqa: E402
    current_ensemble,
    ensemble_weights,
)


OUTPUT_JSON = ROOT / "experiments/results/v4_nested_deviations.json"
OUTPUT_NPZ = ROOT / "experiments/results/predictions/v4_nested_deviations_2024.npz"
SELECTION_TRANSITIONS = (
    ((2020, 2021), 2022),
    ((2021, 2022), 2023),
)
CONFIRMATION = ((2022, 2023), 2024)
K_GRID = (100.0, 200.0, 400.0, 800.0, 1200.0, 2000.0, 3500.0,
          6000.0, 10000.0, 16000.0)
WEIGHT_GRID = (-0.50, -0.25, 0.0, 0.10, 0.20, 0.35, 0.50, 0.65,
               0.80, 1.00, 1.25, 1.50, 2.00)


@dataclass(frozen=True)
class Axis:
    name: str
    parent: tuple[str, ...]
    child: tuple[str, ...]


AXES = (
    Axis("platoon", ("pitcher_id",), ("pitcher_id", "batter_hand")),
    Axis(
        "count_advantage",
        ("pitcher_id", "batter_hand"),
        ("pitcher_id", "batter_hand", "count_advantage"),
    ),
    Axis(
        "exact_count",
        ("pitcher_id", "batter_hand", "count_advantage"),
        ("pitcher_id", "batter_hand", "count_advantage", "balls_before",
         "strikes_before"),
    ),
    Axis(
        "runner",
        ("pitcher_id", "batter_hand"),
        ("pitcher_id", "batter_hand", "runner_present"),
    ),
)


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


def add_columns(frames: dict[int, pd.DataFrame],
                artifacts: dict[int, dict[str, np.ndarray]]) -> None:
    full = pd.read_csv(
        ROOT / "open/data/train.csv",
        usecols=["num_runners_on"],
        encoding="utf-8-sig",
        low_memory=False,
    )
    for season, frame in frames.items():
        row_index = np.asarray(artifacts[season]["row_index"], dtype=np.int64)
        frame["num_runners_on"] = full.iloc[row_index]["num_runners_on"].to_numpy()
        frame["runner_present"] = frame["num_runners_on"].gt(0).astype(np.int8)
        frame["count_advantage"] = (
            frame["strikes_before"].gt(frame["balls_before"])
        ).astype(np.int8)


def mask_for(frame: pd.DataFrame, scope: str) -> np.ndarray:
    if scope == "ALL":
        return np.ones(len(frame), dtype=bool)
    if scope == "R":
        return frame["game_type"].eq("R").to_numpy()
    return frame["domain"].eq(scope).to_numpy()


def source_bundle(frames: dict[int, pd.DataFrame],
                  artifacts: dict[int, dict[str, np.ndarray]],
                  seasons: tuple[int, ...], scope: str
                  ) -> tuple[pd.DataFrame, np.ndarray]:
    frame_parts: list[pd.DataFrame] = []
    residual_parts: list[np.ndarray] = []
    for season in seasons:
        mask = mask_for(frames[season], scope)
        frame_parts.append(frames[season].loc[mask].reset_index(drop=True))
        residual_parts.append(
            (np.asarray(artifacts[season]["y"], dtype=np.float64)
             - np.asarray(artifacts[season]["m3"], dtype=np.float64))[mask]
        )
    return pd.concat(frame_parts, ignore_index=True), np.concatenate(residual_parts)


def nested_lookup(source: pd.DataFrame, residual: np.ndarray,
                  target: pd.DataFrame, axis: Axis, k: float) -> np.ndarray:
    columns = list(dict.fromkeys((*axis.parent, *axis.child)))
    work = source.loc[:, columns].copy()
    work["_residual"] = residual
    parent = work.groupby(list(axis.parent), sort=False, observed=True)[
        "_residual"
    ].mean()
    child = work.groupby(list(axis.child), sort=False, observed=True)[
        "_residual"
    ].agg(["mean", "size"])
    if len(axis.parent) == 1:
        parent_index = pd.Index(
            child.index.get_level_values(axis.parent[0]), name=axis.parent[0]
        )
    else:
        parent_index = pd.MultiIndex.from_arrays(
            [child.index.get_level_values(column) for column in axis.parent],
            names=list(axis.parent),
        )
    parent_mean = parent.reindex(parent_index).to_numpy(dtype=np.float64)
    count = child["size"].to_numpy(dtype=np.float64)
    child["effect"] = (
        count * (child["mean"].to_numpy(dtype=np.float64) - parent_mean)
        / (count + k)
    )
    if len(axis.child) == 1:
        target_index = pd.Index(target[axis.child[0]], name=axis.child[0])
    else:
        target_index = pd.MultiIndex.from_frame(target.loc[:, list(axis.child)])
    return child["effect"].reindex(target_index).fillna(0.0).to_numpy(
        dtype=np.float64
    )


def transition_library(frames: dict[int, pd.DataFrame],
                       artifacts: dict[int, dict[str, np.ndarray]],
                       source_years: tuple[int, ...], target_year: int,
                       route: Route
                       ) -> tuple[np.ndarray, dict[str, dict[float, np.ndarray]]]:
    source, residual = source_bundle(frames, artifacts, source_years,
                                     route.source_scope)
    target_mask = mask_for(frames[target_year], route.target_scope)
    target = frames[target_year].loc[target_mask].reset_index(drop=True)
    library = {
        axis.name: {
            k: nested_lookup(source, residual, target, axis, k) for k in K_GRID
        }
        for axis in AXES
    }
    return target_mask, library


def gain(y: np.ndarray, baseline: np.ndarray, mask: np.ndarray,
         correction: np.ndarray) -> float:
    residual = y[mask] - baseline[mask]
    improvement = (
        2.0 * float(np.dot(residual, correction))
        - float(np.dot(correction, correction))
    ) / float(len(y))
    rate = float(np.mean(y))
    return 100_000.0 * improvement / (rate * (1.0 - rate))


def quadratic_terms(y: np.ndarray, baseline: np.ndarray, mask: np.ndarray,
                    matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    residual = y[mask] - baseline[mask]
    denominator = float(len(y)) * float(np.mean(y)) * float(1.0 - np.mean(y))
    linear = 100_000.0 * 2.0 * (matrix.T @ residual) / denominator
    gram = 100_000.0 * (matrix.T @ matrix) / denominator
    return linear, gram, denominator


def quadratic_gain(weights: np.ndarray, linear: np.ndarray,
                   gram: np.ndarray) -> float:
    return float(weights @ linear - weights @ gram @ weights)


def main() -> None:
    frames, artifacts = load_frames()
    add_columns(frames, artifacts)
    current = {season: current_ensemble(season, artifacts[season])
               for season in (2022, 2023, 2024)}
    baselines = {season: score(artifacts[season]["y"], current[season])
                 for season in current}
    route_reports: dict[str, Any] = {}
    route_predictions: dict[str, np.ndarray] = {}

    for route in ROUTES:
        libraries = {
            (source, target): transition_library(
                frames, artifacts, source, target, route
            )
            for source, target in SELECTION_TRANSITIONS
        }
        selected_k: dict[str, float] = {}
        axis_trials: dict[str, list[dict[str, Any]]] = {}
        for axis in AXES:
            trials: list[dict[str, Any]] = []
            for k in K_GRID:
                gains: dict[str, float] = {}
                for source, target in SELECTION_TRANSITIONS:
                    mask, library = libraries[(source, target)]
                    gains[f"{'+'.join(map(str, source))}_to_{target}"] = gain(
                        np.asarray(artifacts[target]["y"], dtype=np.float64),
                        current[target], mask, library[axis.name][k]
                    )
                trials.append({
                    "k": k,
                    "gains": gains,
                    "robust_min_gain": float(min(gains.values())),
                    "mean_gain": float(np.mean(list(gains.values()))),
                })
            best = max(trials,
                       key=lambda row: (row["robust_min_gain"], row["mean_gain"]))
            selected_k[axis.name] = float(best["k"])
            axis_trials[axis.name] = trials

        terms: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        for source, target in SELECTION_TRANSITIONS:
            mask, library = libraries[(source, target)]
            matrix = np.column_stack([
                library[axis.name][selected_k[axis.name]] for axis in AXES
            ])
            linear, gram, _ = quadratic_terms(
                np.asarray(artifacts[target]["y"], dtype=np.float64),
                current[target], mask, matrix
            )
            terms[str(target)] = (linear, gram)

        weight_trials: list[dict[str, Any]] = []
        for values in itertools.product(WEIGHT_GRID, repeat=len(AXES)):
            weights = np.asarray(values, dtype=np.float64)
            gains = {
                season: quadratic_gain(weights, *terms[season])
                for season in terms
            }
            weight_trials.append({
                "weights": {axis.name: float(value)
                            for axis, value in zip(AXES, values)},
                "gains": gains,
                "robust_min_gain": float(min(gains.values())),
                "mean_gain": float(np.mean(list(gains.values()))),
            })
        selected = max(
            weight_trials,
            key=lambda row: (row["robust_min_gain"], row["mean_gain"]),
        )

        target_mask, confirmation_library = transition_library(
            frames, artifacts, CONFIRMATION[0], CONFIRMATION[1], route
        )
        correction = sum(
            selected["weights"][axis.name]
            * confirmation_library[axis.name][selected_k[axis.name]]
            for axis in AXES
        )
        prediction = np.asarray(current[2024], dtype=np.float64).copy()
        prediction[target_mask] = np.clip(
            prediction[target_mask] + correction, 0.0, 1.0
        )
        metrics = score(artifacts[2024]["y"], prediction)
        confirm_gain = float(
            metrics["raw_competition_score"]
            - baselines[2024]["raw_competition_score"]
        )
        route_predictions[route.name] = prediction
        route_reports[route.name] = {
            "route": route.__dict__,
            "selected_k": selected_k,
            "axis_trials": axis_trials,
            "selected_weights": selected,
            "top_weight_trials": sorted(
                weight_trials,
                key=lambda row: (row["robust_min_gain"], row["mean_gain"]),
                reverse=True,
            )[:40],
            "confirmation_2024": {
                "metrics": metrics,
                "gain": confirm_gain,
                "correction_mean": float(correction.mean()),
                "correction_std": float(correction.std()),
                "correction_max_abs": float(np.max(np.abs(correction))),
            },
        }
        print(f"[{route.name}] min={selected['robust_min_gain']:+.4f} "
              f"mean={selected['mean_gain']:+.4f} "
              f"confirm={confirm_gain:+.4f} "
              f"local={metrics['raw_competition_score']:.4f}", flush=True)

    selected_route = max(
        ROUTES,
        key=lambda item: (
            route_reports[item.name]["selected_weights"]["robust_min_gain"],
            route_reports[item.name]["selected_weights"]["mean_gain"],
        ),
    )
    primary = route_predictions[selected_route.name]
    primary_metrics = score(artifacts[2024]["y"], primary)
    payload: dict[str, np.ndarray] = {
        "y": artifacts[2024]["y"],
        "row_index": artifacts[2024]["row_index"],
        "cluster": artifacts[2024]["cluster"],
        "current_ensemble": current[2024],
        "nested_deviations": primary,
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
            "selection": "maximize worst gain on two prior-season transfers",
            "selection_transitions": [
                f"{'+'.join(map(str, source))}->{target}"
                for source, target in SELECTION_TRANSITIONS
            ],
            "confirmation": "2022+2023 tables applied once to 2024",
            "effect": "shrunk child residual mean minus parent residual mean",
        },
        "fixed_estimator": {
            "median_offset": MEDIAN_OFFSET,
            "required_local_score": REQUIRED_LOCAL,
            "target_lb": 1190.0,
        },
        "current_ensemble_weights": ensemble_weights(),
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
        json.dumps(json_safe(report), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps({
        "selected_route": selected_route.name,
        "score_2024": primary_metrics["raw_competition_score"],
        "expected_lb_median": (
            primary_metrics["raw_competition_score"] + MEDIAN_OFFSET
        ),
    }, ensure_ascii=False, indent=2))
    print(f"Saved {OUTPUT_JSON}")
    print(f"Saved {OUTPUT_NPZ}")


if __name__ == "__main__":
    main()
