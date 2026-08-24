#!/usr/bin/env python3
"""Replace or augment hard-window context contrasts with decayed differentials."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


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
    mask_for,
)
from experiments.analyze_v4_temporal_residual_ridge import (  # noqa: E402
    MEDIAN_OFFSET,
    REQUIRED_LOCAL,
    json_safe,
    load_frames,
    score,
)
from experiments.v4_current_ensemble import (  # noqa: E402
    PREDICTIONS,
    current_ensemble,
    ensemble_weights,
    load_npz,
)


OUTPUT_JSON = ROOT / "experiments/results/v4_decayed_differentials.json"
OUTPUT_NPZ = PREDICTIONS / "v4_decayed_differentials_2024.npz"
NESTED_REPORT = ROOT / "experiments/results/v4_nested_deviations.json"
EXPANSION_REPORT = ROOT / "experiments/results/v4_nested_context_constrained.json"
SELECTION_YEARS = (2022, 2023)
ALL_YEARS = (2022, 2023, 2024)
K_GRID = (200.0, 500.0, 800.0, 1000.0, 1500.0, 2000.0, 3500.0,
          5000.0, 8000.0, 12000.0, 20000.0)
DECAY_GRID: tuple[float | None, ...] = (
    None, 0.30, 0.40, 0.50, 0.55, 0.65, 0.70, 0.80, 0.85, 0.90, 1.00
)
SINGLE_WEIGHT_GRID = (-1.00, -0.75, -0.50, -0.25, -0.10, 0.0, 0.10,
                      0.20, 0.35, 0.50, 0.65, 0.80, 1.00, 1.25, 1.50,
                      2.00)
OLD_ADJUST_GRID = (-1.00, -0.75, -0.50, -0.25, -0.10, 0.0, 0.10, 0.25,
                   0.50)
NEW_WEIGHT_GRID = SINGLE_WEIGHT_GRID
CONTRAST_WEIGHT = 0.85


AXES = (
    ("same_hand", "same_hand"),
    ("two_strike", "two_strike"),
    ("runner_present", "runner_present"),
)
ROUTES = (
    ("r_from_r", "R", "R"),
    ("core_from_r", "R", "R_CORE"),
    ("all_from_all", "ALL", "ALL"),
)


def add_differential_columns(frames: dict[int, pd.DataFrame]) -> None:
    for frame in frames.values():
        frame["same_hand"] = frame["pitcher_hand"].eq(
            frame["batter_hand"]
        ).astype(np.int8)
        frame["two_strike"] = frame["strikes_before"].eq(2).astype(np.int8)


def champion_predictions(
    frames: dict[int, pd.DataFrame],
    artifacts: dict[int, dict[str, np.ndarray]],
) -> dict[int, np.ndarray]:
    nested_report = json.loads(NESTED_REPORT.read_text(encoding="utf-8"))
    nested_route = nested_report["selected_route"]
    nested_values = nested_report["routes"][nested_route]
    nested_k = {key: float(value)
                for key, value in nested_values["selected_k"].items()}
    nested_weights = {
        key: float(value)
        for key, value in nested_values["selected_weights"]["weights"].items()
    }
    expansion_report = json.loads(EXPANSION_REPORT.read_text(encoding="utf-8"))
    expansion_values = expansion_report["routes"]["core_from_r"]
    expansion_k = {
        key: float(value) for key, value in expansion_values["selected_k"].items()
    }
    expansion_weights = {
        key: float(value)
        for key, value in expansion_values["selected_weights"].items()
    }
    sources = {2022: (2020, 2021), 2023: (2021, 2022), 2024: (2022, 2023)}
    result: dict[int, np.ndarray] = {}
    for target, source in sources.items():
        prediction = current_ensemble(target, artifacts[target])
        mask, values = nested_library(
            frames, artifacts, source, target, "ALL", "ALL", NESTED_AXES,
            tuple(sorted(set(nested_k.values())))
        )
        correction = sum(
            nested_weights[axis.name] * values[axis.name][nested_k[axis.name]]
            for axis in NESTED_AXES
        )
        prediction[mask] = np.clip(prediction[mask] + correction, 0.0, 1.0)

        mask, values = nested_library(
            frames, artifacts, source, target, "R", "R_CORE", EXPANSION_AXES,
            tuple(sorted(set(expansion_k.values())))
        )
        correction = sum(
            expansion_weights[axis.name]
            * values[axis.name][expansion_k[axis.name]]
            for axis in EXPANSION_AXES
        )
        prediction[mask] = np.clip(prediction[mask] + correction, 0.0, 1.0)
        result[target] = prediction
    return result


def source_seasons(target_year: int, decay: float | None) -> tuple[int, ...]:
    if decay is None:
        return (target_year - 2, target_year - 1)
    return tuple(range(2020, target_year))


def differential_raw(
    frames: dict[int, pd.DataFrame],
    artifacts: dict[int, dict[str, np.ndarray]],
    target_year: int,
    context: str,
    source_scope: str,
    decay: float | None,
) -> pd.DataFrame:
    parts: list[pd.DataFrame] = []
    for season in source_seasons(target_year, decay):
        mask = mask_for(frames[season], source_scope)
        weight = (1.0 if decay is None
                  else float(decay ** (target_year - 1 - season)))
        residual = (
            np.asarray(artifacts[season]["y"], dtype=np.float64)
            - np.asarray(artifacts[season]["m3"], dtype=np.float64)
        )[mask]
        parts.append(pd.DataFrame({
            "pitcher_id": frames[season].loc[mask, "pitcher_id"].to_numpy(),
            "context": frames[season].loc[mask, context].to_numpy(dtype=np.int8),
            "weighted_residual": residual * weight,
            "weight": np.full(int(mask.sum()), weight, dtype=np.float64),
        }))
    work = pd.concat(parts, ignore_index=True)
    grouped = work.groupby(["pitcher_id", "context"], observed=True, sort=False)[
        ["weighted_residual", "weight"]
    ].sum().unstack()
    for value in (0, 1):
        for column in ("weighted_residual", "weight"):
            if (column, value) not in grouped:
                grouped[(column, value)] = 0.0
    n0 = grouped[("weight", 0)].fillna(0.0)
    n1 = grouped[("weight", 1)].fillna(0.0)
    mean0 = grouped[("weighted_residual", 0)] / n0.replace(0.0, np.nan)
    mean1 = grouped[("weighted_residual", 1)] / n1.replace(0.0, np.nan)
    effective_n = (n0 * n1) / (n0 + n1).replace(0.0, np.nan)
    return pd.DataFrame({
        "difference": mean1 - mean0,
        "effective_n": effective_n,
    }).dropna()


def lookup_vector(table: pd.DataFrame, target: pd.DataFrame,
                  context: str, k: float) -> np.ndarray:
    effect = (table["difference"] * table["effective_n"]
              / (table["effective_n"] + k))
    mapped = target["pitcher_id"].map(effect).fillna(0.0).to_numpy(np.float64)
    sign = np.where(target[context].to_numpy(dtype=np.int8) == 1, 0.5, -0.5)
    return mapped * sign


def full_vector(mask: np.ndarray, values: np.ndarray) -> np.ndarray:
    result = np.zeros(len(mask), dtype=np.float64)
    result[mask] = values
    return result


def score_gain(y: np.ndarray, baseline: np.ndarray,
               correction: np.ndarray) -> float:
    residual = y - baseline
    improvement = (
        2.0 * float(np.dot(residual, correction))
        - float(np.dot(correction, correction))
    ) / float(len(y))
    rate = float(np.mean(y))
    return 100_000.0 * improvement / (rate * (1.0 - rate))


def quadratic_terms(y: np.ndarray, baseline: np.ndarray,
                    matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    residual = y - baseline
    denominator = float(len(y)) * float(np.mean(y)) * float(1.0 - np.mean(y))
    return (
        100_000.0 * 2.0 * (matrix.T @ residual) / denominator,
        100_000.0 * (matrix.T @ matrix) / denominator,
    )


def qgain(weights: np.ndarray, linear: np.ndarray,
          gram: np.ndarray) -> float:
    return float(weights @ linear - weights @ gram @ weights)


def coordinate_search(
    fold_terms: dict[int, tuple[np.ndarray, np.ndarray]],
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    weights = np.zeros(4, dtype=np.float64)
    trace: list[dict[str, Any]] = []
    grids = (OLD_ADJUST_GRID, NEW_WEIGHT_GRID, NEW_WEIGHT_GRID, NEW_WEIGHT_GRID)
    names = ("old_contrast_adjustment", *[name for name, _ in AXES])
    for sweep in range(10):
        changed = False
        for index, (name, grid) in enumerate(zip(names, grids)):
            best: tuple[tuple[float, float], float, dict[str, float]] | None = None
            for value in grid:
                candidate = weights.copy()
                candidate[index] = value
                gains = {
                    str(year): qgain(candidate, *fold_terms[year])
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
                "axis": name,
                "value": float(weights[index]),
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
    add_differential_columns(frames)
    champion = champion_predictions(frames, artifacts)
    champion_metrics = {
        year: score(artifacts[year]["y"], champion[year]) for year in ALL_YEARS
    }
    old_delta: dict[int, np.ndarray] = {}
    for year in ALL_YEARS:
        residual = load_npz(PREDICTIONS / f"v4_residual_ensemble_{year}.npz")
        old_delta[year] = CONTRAST_WEIGHT * (
            np.asarray(residual["contrast"], dtype=np.float64)
            - np.asarray(residual["m3"], dtype=np.float64)
        )

    route_reports: dict[str, Any] = {}
    route_predictions: dict[str, np.ndarray] = {}
    for route_name, source_scope, target_scope in ROUTES:
        masks = {year: mask_for(frames[year], target_scope) for year in ALL_YEARS}
        cache: dict[tuple[str, float | None, int], pd.DataFrame] = {}
        vectors: dict[tuple[str, float | None, float, int], np.ndarray] = {}
        selected_config: dict[str, dict[str, Any]] = {}
        axis_trials: dict[str, list[dict[str, Any]]] = {}
        for name, context in AXES:
            trials: list[dict[str, Any]] = []
            for decay in DECAY_GRID:
                for year in SELECTION_YEARS:
                    cache[(name, decay, year)] = differential_raw(
                        frames, artifacts, year, context, source_scope, decay
                    )
                for k in K_GRID:
                    fold_vectors: dict[int, np.ndarray] = {}
                    for year in SELECTION_YEARS:
                        target = frames[year].loc[masks[year]].reset_index(drop=True)
                        value = lookup_vector(cache[(name, decay, year)], target,
                                              context, k)
                        fold_vectors[year] = full_vector(masks[year], value)
                    best_weight: tuple[tuple[float, float], float,
                                       dict[str, float]] | None = None
                    for weight in SINGLE_WEIGHT_GRID:
                        gains = {
                            str(year): score_gain(
                                np.asarray(artifacts[year]["y"], dtype=np.float64),
                                champion[year], weight * fold_vectors[year]
                            )
                            for year in SELECTION_YEARS
                        }
                        rank = (float(min(gains.values())),
                                float(np.mean(list(gains.values()))))
                        if best_weight is None or rank > best_weight[0]:
                            best_weight = (rank, weight, gains)
                    assert best_weight is not None
                    trials.append({
                        "decay": decay,
                        "k": k,
                        "weight": best_weight[1],
                        "gains": best_weight[2],
                        "robust_min_gain": best_weight[0][0],
                        "mean_gain": best_weight[0][1],
                    })
            selected = max(
                trials,
                key=lambda row: (row["robust_min_gain"], row["mean_gain"]),
            )
            selected_config[name] = selected
            axis_trials[name] = sorted(
                trials,
                key=lambda row: (row["robust_min_gain"], row["mean_gain"]),
                reverse=True,
            )[:40]
            for year in ALL_YEARS:
                decay = selected["decay"]
                table = differential_raw(
                    frames, artifacts, year, context, source_scope, decay
                )
                target = frames[year].loc[masks[year]].reset_index(drop=True)
                value = lookup_vector(table, target, context, float(selected["k"]))
                vectors[(name, decay, float(selected["k"]), year)] = full_vector(
                    masks[year], value
                )

        fold_terms: dict[int, tuple[np.ndarray, np.ndarray]] = {}
        matrices: dict[int, np.ndarray] = {}
        for year in ALL_YEARS:
            matrix = np.column_stack([
                old_delta[year],
                *[
                    vectors[(name, selected_config[name]["decay"],
                             float(selected_config[name]["k"]), year)]
                    for name, _ in AXES
                ],
            ])
            matrices[year] = matrix
            if year in SELECTION_YEARS:
                fold_terms[year] = quadratic_terms(
                    np.asarray(artifacts[year]["y"], dtype=np.float64),
                    champion[year], matrix
                )
        weights, trace = coordinate_search(fold_terms)
        selection_gains = {
            str(year): qgain(weights, *fold_terms[year])
            for year in SELECTION_YEARS
        }
        prediction = np.clip(champion[2024] + matrices[2024] @ weights, 0.0, 1.0)
        metrics = score(artifacts[2024]["y"], prediction)
        confirm_gain = float(
            metrics["raw_competition_score"]
            - champion_metrics[2024]["raw_competition_score"]
        )
        route_predictions[route_name] = prediction
        route_reports[route_name] = {
            "source_scope": source_scope,
            "target_scope": target_scope,
            "selected_axis_config": selected_config,
            "joint_weights": {
                "old_contrast_adjustment": float(weights[0]),
                **{name: float(value)
                   for (name, _), value in zip(AXES, weights[1:])},
            },
            "selection_gains": selection_gains,
            "robust_min_gain": float(min(selection_gains.values())),
            "mean_gain": float(np.mean(list(selection_gains.values()))),
            "coordinate_trace": trace,
            "axis_trials": axis_trials,
            "confirmation_2024": {
                "metrics": metrics,
                "gain": confirm_gain,
                "correction_mean": float((matrices[2024] @ weights).mean()),
                "correction_std": float((matrices[2024] @ weights).std()),
                "correction_max_abs": float(
                    np.max(np.abs(matrices[2024] @ weights))
                ),
            },
        }
        print(f"[{route_name}] min={min(selection_gains.values()):+.4f} "
              f"mean={np.mean(list(selection_gains.values())):+.4f} "
              f"confirm={confirm_gain:+.4f} "
              f"local={metrics['raw_competition_score']:.4f}", flush=True)

    selected_route = max(
        route_reports,
        key=lambda name: (route_reports[name]["robust_min_gain"],
                          route_reports[name]["mean_gain"]),
    )
    primary = route_predictions[selected_route]
    primary_metrics = score(artifacts[2024]["y"], primary)
    payload: dict[str, np.ndarray] = {
        "y": artifacts[2024]["y"],
        "row_index": artifacts[2024]["row_index"],
        "cluster": artifacts[2024]["cluster"],
        "champion": champion[2024],
        "decayed_differentials": primary,
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
            "selection": "worst-fold decay/K selection then joint coordinate search",
            "selection_years": list(SELECTION_YEARS),
            "confirmation_year": 2024,
            "old_contrast_adjustment": (
                "0 keeps the current contrast; -1 removes it before replacement"
            ),
        },
        "fixed_estimator": {
            "median_offset": MEDIAN_OFFSET,
            "required_local_score": REQUIRED_LOCAL,
            "target_lb": 1190.0,
        },
        "current_ensemble_weights": ensemble_weights(),
        "champion_metrics": champion_metrics,
        "routes": route_reports,
        "selected_route": selected_route,
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
        "selected_route": selected_route,
        "score_2024": primary_metrics["raw_competition_score"],
        "expected_lb_median": (
            primary_metrics["raw_competition_score"] + MEDIAN_OFFSET
        ),
    }, ensure_ascii=False, indent=2))
    print(f"Saved {OUTPUT_JSON}")
    print(f"Saved {OUTPUT_NPZ}")


if __name__ == "__main__":
    main()
