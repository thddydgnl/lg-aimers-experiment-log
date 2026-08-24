#!/usr/bin/env python3
"""Expand the robust exact-count residual effect with current-row contexts."""

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

from experiments.analyze_v4_nested_deviations import (  # noqa: E402
    AXES as BASE_AXES,
    Axis,
    CONFIRMATION,
    SELECTION_TRANSITIONS,
    add_columns,
    mask_for,
    nested_lookup,
    source_bundle,
)
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


OUTPUT_JSON = ROOT / "experiments/results/v4_nested_context_expansion.json"
OUTPUT_NPZ = (
    ROOT / "experiments/results/predictions/v4_nested_context_expansion_2024.npz"
)
BASE_REPORT = ROOT / "experiments/results/v4_nested_deviations.json"
K_GRID = (200.0, 400.0, 800.0, 1200.0, 2000.0, 3500.0, 6000.0,
          10000.0, 16000.0, 24000.0, 32000.0)
WEIGHT_GRID = (-0.50, -0.25, 0.0, 0.10, 0.20, 0.35, 0.50, 0.65,
               0.80, 1.00, 1.25, 1.50, 2.00)


PH = ("pitcher_id", "batter_hand")
COUNT = (*PH, "balls_before", "strikes_before")
PITCHER_COUNT = ("pitcher_id", "balls_before", "strikes_before")
AXES = (
    Axis("exact_count_direct", PH, COUNT),
    Axis("pitcher_exact_count", ("pitcher_id",), PITCHER_COUNT),
    Axis("count_runner", COUNT, (*COUNT, "runner_present")),
    Axis("count_outs", COUNT, (*COUNT, "outs_before")),
    Axis("count_inning", COUNT, (*COUNT, "inning_phase")),
    Axis("count_side", COUNT, (*COUNT, "top_bottom")),
    Axis("count_game_type", COUNT, (*COUNT, "game_type")),
    Axis("count_base", COUNT, (*COUNT, "base_state")),
    Axis("count_score", COUNT, (*COUNT, "score_state")),
    Axis("count_leverage", COUNT, (*COUNT, "leverage_bin")),
    Axis("count_month", COUNT, (*COUNT, "month_phase")),
    Axis(
        "pitcher_count_runner",
        PITCHER_COUNT,
        (*PITCHER_COUNT, "runner_present"),
    ),
)


def add_expansion_columns(
    frames: dict[int, pd.DataFrame],
    artifacts: dict[int, dict[str, np.ndarray]],
) -> None:
    columns = [
        "inning",
        "outs_before",
        "top_bottom",
        "base_state",
        "score_diff_pitcher_team",
        "li",
    ]
    full = pd.read_csv(
        ROOT / "open/data/train.csv",
        usecols=columns,
        encoding="utf-8-sig",
        low_memory=False,
    )
    for season, frame in frames.items():
        row_index = np.asarray(artifacts[season]["row_index"], dtype=np.int64)
        selected = full.iloc[row_index].reset_index(drop=True)
        for column in columns:
            frame[column] = selected[column].to_numpy()
        inning = frame["inning"].to_numpy(dtype=np.float64)
        frame["inning_phase"] = np.where(
            inning <= 3, 0, np.where(inning <= 6, 1, np.where(inning <= 9, 2, 3))
        ).astype(np.int8)
        score_diff = frame["score_diff_pitcher_team"].to_numpy(dtype=np.float64)
        frame["score_state"] = np.where(
            score_diff < 0, 0, np.where(score_diff > 0, 2, 1)
        ).astype(np.int8)
        leverage = frame["li"].to_numpy(dtype=np.float64)
        frame["leverage_bin"] = np.where(
            leverage < 0.7, 0, np.where(leverage < 1.5, 1, 2)
        ).astype(np.int8)
        month = frame["game_month"].to_numpy(dtype=np.float64)
        frame["month_phase"] = np.where(
            month <= 4, 0, np.where(month <= 6, 1, np.where(month <= 8, 2, 3))
        ).astype(np.int8)


def library(frames: dict[int, pd.DataFrame],
            artifacts: dict[int, dict[str, np.ndarray]],
            source_years: tuple[int, ...], target_year: int,
            source_scope: str, target_scope: str,
            axes: tuple[Axis, ...], k_grid: tuple[float, ...]
            ) -> tuple[np.ndarray, dict[str, dict[float, np.ndarray]]]:
    source, residual = source_bundle(frames, artifacts, source_years, source_scope)
    target_mask = mask_for(frames[target_year], target_scope)
    target = frames[target_year].loc[target_mask].reset_index(drop=True)
    result = {
        axis.name: {
            k: nested_lookup(source, residual, target, axis, k) for k in k_grid
        }
        for axis in axes
    }
    return target_mask, result


def gain(y: np.ndarray, baseline: np.ndarray, mask: np.ndarray,
         correction: np.ndarray) -> float:
    residual = y[mask] - baseline[mask]
    improvement = (
        2.0 * float(np.dot(residual, correction))
        - float(np.dot(correction, correction))
    ) / float(len(y))
    rate = float(np.mean(y))
    return 100_000.0 * improvement / (rate * (1.0 - rate))


def terms(y: np.ndarray, baseline: np.ndarray, mask: np.ndarray,
          matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    residual = y[mask] - baseline[mask]
    denominator = float(len(y)) * float(np.mean(y)) * float(1.0 - np.mean(y))
    linear = 100_000.0 * 2.0 * (matrix.T @ residual) / denominator
    gram = 100_000.0 * (matrix.T @ matrix) / denominator
    return linear, gram


def qgain(weights: np.ndarray, linear: np.ndarray,
          gram: np.ndarray) -> float:
    return float(weights @ linear - weights @ gram @ weights)


def coordinate_search(fold_terms: dict[str, tuple[np.ndarray, np.ndarray]]) -> tuple[
        np.ndarray, list[dict[str, Any]]]:
    weights = np.zeros(len(AXES), dtype=np.float64)
    trace: list[dict[str, Any]] = []
    for sweep in range(8):
        changed = False
        for index, axis in enumerate(AXES):
            best: tuple[tuple[float, float], float, dict[str, float]] | None = None
            for value in WEIGHT_GRID:
                candidate = weights.copy()
                candidate[index] = value
                gains = {
                    season: qgain(candidate, *fold_terms[season])
                    for season in fold_terms
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
                "axis": axis.name,
                "selected_value": float(weights[index]),
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
    current = {season: current_ensemble(season, artifacts[season])
               for season in (2022, 2023, 2024)}
    base_report = json.loads(BASE_REPORT.read_text(encoding="utf-8"))
    base_route_name = base_report["selected_route"]
    base_route = base_report["routes"][base_route_name]
    base_k = {key: float(value) for key, value in base_route["selected_k"].items()}
    base_weights = {
        key: float(value)
        for key, value in base_route["selected_weights"]["weights"].items()
    }

    base_predictions: dict[int, np.ndarray] = {}
    transition_map = {
        2022: SELECTION_TRANSITIONS[0][0],
        2023: SELECTION_TRANSITIONS[1][0],
        2024: CONFIRMATION[0],
    }
    for target_year, source_years in transition_map.items():
        mask, base_library = library(
            frames, artifacts, source_years, target_year, "ALL", "ALL",
            BASE_AXES, tuple(sorted(set(base_k.values())))
        )
        correction = sum(
            base_weights[axis.name] * base_library[axis.name][base_k[axis.name]]
            for axis in BASE_AXES
        )
        prediction = current[target_year].copy()
        prediction[mask] = np.clip(prediction[mask] + correction, 0.0, 1.0)
        base_predictions[target_year] = prediction

    route_reports: dict[str, Any] = {}
    route_predictions: dict[str, np.ndarray] = {}
    for route_name, source_scope, target_scope in (
        ("all_from_all", "ALL", "ALL"),
        ("core_from_r", "R", "R_CORE"),
    ):
        selection_libraries = {
            (source, target): library(
                frames, artifacts, source, target, source_scope, target_scope,
                AXES, K_GRID
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
                    mask, values = selection_libraries[(source, target)]
                    gains[str(target)] = gain(
                        np.asarray(artifacts[target]["y"], dtype=np.float64),
                        base_predictions[target], mask, values[axis.name][k]
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

        fold_terms: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        for source, target in SELECTION_TRANSITIONS:
            mask, values = selection_libraries[(source, target)]
            matrix = np.column_stack([
                values[axis.name][selected_k[axis.name]] for axis in AXES
            ])
            fold_terms[str(target)] = terms(
                np.asarray(artifacts[target]["y"], dtype=np.float64),
                base_predictions[target], mask, matrix
            )
        weights, trace = coordinate_search(fold_terms)
        selected_gains = {
            season: qgain(weights, *fold_terms[season]) for season in fold_terms
        }

        mask_2024, values_2024 = library(
            frames, artifacts, CONFIRMATION[0], CONFIRMATION[1], source_scope,
            target_scope, AXES, K_GRID
        )
        correction_2024 = sum(
            weight * values_2024[axis.name][selected_k[axis.name]]
            for axis, weight in zip(AXES, weights)
        )
        prediction = base_predictions[2024].copy()
        prediction[mask_2024] = np.clip(
            prediction[mask_2024] + correction_2024, 0.0, 1.0
        )
        metrics = score(artifacts[2024]["y"], prediction)
        base_metrics = score(artifacts[2024]["y"], base_predictions[2024])
        confirm_gain = float(
            metrics["raw_competition_score"]
            - base_metrics["raw_competition_score"]
        )
        route_predictions[route_name] = prediction
        route_reports[route_name] = {
            "source_scope": source_scope,
            "target_scope": target_scope,
            "selected_k": selected_k,
            "selected_weights": {
                axis.name: float(value) for axis, value in zip(AXES, weights)
            },
            "selection_gains": selected_gains,
            "robust_min_gain": float(min(selected_gains.values())),
            "mean_gain": float(np.mean(list(selected_gains.values()))),
            "coordinate_trace": trace,
            "axis_trials": axis_trials,
            "confirmation_2024": {
                "metrics": metrics,
                "gain_over_nested_base": confirm_gain,
                "gain_over_current": float(
                    metrics["raw_competition_score"]
                    - score(artifacts[2024]["y"], current[2024])[
                        "raw_competition_score"
                    ]
                ),
                "correction_mean": float(correction_2024.mean()),
                "correction_std": float(correction_2024.std()),
                "correction_max_abs": float(np.max(np.abs(correction_2024))),
            },
        }
        print(f"[{route_name}] min={min(selected_gains.values()):+.4f} "
              f"mean={np.mean(list(selected_gains.values())):+.4f} "
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
        "current_ensemble": current[2024],
        "nested_base": base_predictions[2024],
        "nested_context_expansion": primary,
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
            "selection": "per-axis K then worst-fold coordinate weight search",
            "selection_transitions": [
                f"{'+'.join(map(str, source))}->{target}"
                for source, target in SELECTION_TRANSITIONS
            ],
            "confirmation": "apply selected expansion once to 2024",
            "weight_grid": list(WEIGHT_GRID),
        },
        "fixed_estimator": {
            "median_offset": MEDIAN_OFFSET,
            "required_local_score": REQUIRED_LOCAL,
            "target_lb": 1190.0,
        },
        "current_ensemble_weights": ensemble_weights(),
        "nested_base": {
            "route": base_route_name,
            "k": base_k,
            "weights": base_weights,
        },
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
