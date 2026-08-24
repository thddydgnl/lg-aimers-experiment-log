#!/usr/bin/env python3
"""Leakage-safe affine calibration checks for the conservative V4 stack."""

from __future__ import annotations

import itertools
import json
import sys
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
    score,
)


PREDICTIONS = ROOT / "experiments/results/predictions"
OUTPUT_JSON = ROOT / "experiments/results/v4_final_calibration.json"
OUTPUT_NPZ = PREDICTIONS / "v4_final_calibration_2024.npz"
YEARS = (2022, 2023, 2024)
ANCHOR_TEAM_ID = 13
SLOPE_GRID = np.round(np.arange(-0.30, 0.3001, 0.005), 6)
SHIFT_GRID = np.round(np.arange(-0.015, 0.01501, 0.0005), 6)


def load() -> tuple[dict[int, dict[str, np.ndarray]], dict[int, pd.DataFrame]]:
    artifacts = {}
    full = pd.read_csv(
        ROOT / "open/data/train.csv",
        usecols=["game_type", "pitcher_team_id", "batter_team_id"],
        encoding="utf-8-sig",
        low_memory=False,
    )
    rows = {}
    for year in YEARS:
        with np.load(
            PREDICTIONS / f"v4_joint_neural_conservative_{year}.npz"
        ) as archive:
            artifacts[year] = {key: np.asarray(archive[key])
                               for key in archive.files}
        selected = full.iloc[artifacts[year]["row_index"]].reset_index(drop=True)
        anchor = (
            selected["pitcher_team_id"].eq(ANCHOR_TEAM_ID)
            | selected["batter_team_id"].eq(ANCHOR_TEAM_ID)
        )
        selected["domain"] = np.where(
            selected["game_type"].eq("F"),
            "F",
            np.where(anchor, "R_ANCHOR", "R_CORE"),
        )
        rows[year] = selected
    return artifacts, rows


def terms(y: np.ndarray, prediction: np.ndarray,
          matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    residual = y - prediction
    denominator = float(len(y)) * float(np.mean(y)) * float(1.0 - np.mean(y))
    return (
        100_000.0 * 2.0 * (matrix.T @ residual) / denominator,
        100_000.0 * (matrix.T @ matrix) / denominator,
    )


def gain(weights: np.ndarray, linear: np.ndarray,
         gram: np.ndarray) -> float:
    return float(weights @ linear - weights @ gram @ weights)


def matrices_for(
    artifacts: dict[int, dict[str, np.ndarray]],
    rows: dict[int, pd.DataFrame],
    variant: str,
) -> tuple[dict[int, np.ndarray], list[str], list[np.ndarray]]:
    matrices = {}
    if variant == "global_affine":
        names = ["global_slope_adjustment", "global_shift"]
        grids = [SLOPE_GRID, SHIFT_GRID]
        for year in YEARS:
            prediction = artifacts[year]["conservative"].astype(np.float64)
            matrices[year] = np.column_stack([
                prediction - 0.5,
                np.ones(len(prediction), dtype=np.float64),
            ])
        return matrices, names, grids

    if variant == "game_type_affine":
        groups = ("R", "F")
        names = [item for group in groups for item in
                 (f"{group}_slope_adjustment", f"{group}_shift")]
        grids = [grid for _ in groups for grid in (SLOPE_GRID, SHIFT_GRID)]
        for year in YEARS:
            prediction = artifacts[year]["conservative"].astype(np.float64)
            columns = []
            for group in groups:
                mask = rows[year]["game_type"].eq(group).to_numpy(dtype=np.float64)
                columns.extend(((prediction - 0.5) * mask, mask))
            matrices[year] = np.column_stack(columns)
        return matrices, names, grids

    if variant in ("domain_slopes", "domain_affine"):
        groups = ("R_CORE", "R_ANCHOR", "F")
        affine = variant == "domain_affine"
        names = []
        grids = []
        for group in groups:
            names.append(f"{group}_slope_adjustment")
            grids.append(SLOPE_GRID)
            if affine:
                names.append(f"{group}_shift")
                grids.append(SHIFT_GRID)
        for year in YEARS:
            prediction = artifacts[year]["conservative"].astype(np.float64)
            columns = []
            for group in groups:
                mask = rows[year]["domain"].eq(group).to_numpy(dtype=np.float64)
                columns.append((prediction - 0.5) * mask)
                if affine:
                    columns.append(mask)
            matrices[year] = np.column_stack(columns)
        return matrices, names, grids
    raise ValueError(variant)


def coordinate_search(
    fold_terms: dict[int, tuple[np.ndarray, np.ndarray]],
    names: list[str], grids: list[np.ndarray],
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    weights = np.zeros(len(names), dtype=np.float64)
    trace = []
    for sweep in range(10):
        changed = False
        for index, (name, grid) in enumerate(zip(names, grids)):
            best = None
            for value in grid:
                candidate = weights.copy()
                candidate[index] = value
                gains = {
                    str(year): gain(candidate, *fold_terms[year])
                    for year in (2022, 2023)
                }
                rank = (float(min(gains.values())),
                        float(np.mean(list(gains.values()))))
                if best is None or rank > best[0]:
                    best = (rank, float(value), gains)
            assert best is not None
            if best[1] != weights[index]:
                changed = True
                weights[index] = best[1]
            trace.append({
                "sweep": sweep + 1,
                "parameter": name,
                "value": float(weights[index]),
                "robust_min_gain": best[0][0],
                "mean_gain": best[0][1],
                "gains": best[2],
            })
        if not changed:
            break
    return weights, trace


def main() -> None:
    artifacts, rows = load()
    baselines = {
        str(year): score(
            artifacts[year]["y"], artifacts[year]["conservative"]
        )
        for year in YEARS
    }
    variants = {}
    predictions = {}
    for variant in (
        "global_affine",
        "game_type_affine",
        "domain_slopes",
        "domain_affine",
    ):
        matrices, names, grids = matrices_for(artifacts, rows, variant)
        fold_terms = {
            year: terms(
                artifacts[year]["y"].astype(np.float64),
                artifacts[year]["conservative"].astype(np.float64),
                matrices[year],
            )
            for year in (2022, 2023)
        }
        weights, trace = coordinate_search(fold_terms, names, grids)
        gains = {
            str(year): gain(weights, *fold_terms[year])
            for year in (2022, 2023)
        }
        base = artifacts[2024]["conservative"].astype(np.float64)
        prediction = np.clip(base + matrices[2024] @ weights, 0.0, 1.0)
        metric = score(artifacts[2024]["y"], prediction)
        confirm_gain = float(
            metric["raw_competition_score"]
            - baselines["2024"]["raw_competition_score"]
        )
        predictions[variant] = prediction
        variants[variant] = {
            "parameters": {name: float(value)
                           for name, value in zip(names, weights)},
            "selection_gains": gains,
            "robust_min_gain": float(min(gains.values())),
            "mean_gain": float(np.mean(list(gains.values()))),
            "trace": trace,
            "confirmation_2024": {
                "metrics": metric,
                "gain": confirm_gain,
                "expected_lb_median": float(
                    metric["raw_competition_score"] + MEDIAN_OFFSET
                ),
            },
        }
        print(f"[{variant}] min={min(gains.values()):+.4f} "
              f"mean={np.mean(list(gains.values())):+.4f} "
              f"confirm={confirm_gain:+.4f} "
              f"local={metric['raw_competition_score']:.4f}", flush=True)

    selected_name = max(
        variants,
        key=lambda name: (variants[name]["robust_min_gain"],
                          variants[name]["mean_gain"]),
    )
    primary = predictions[selected_name]
    primary_metrics = variants[selected_name]["confirmation_2024"]["metrics"]
    payload = {
        "y": artifacts[2024]["y"],
        "row_index": artifacts[2024]["row_index"],
        "cluster": artifacts[2024]["cluster"],
        "base": artifacts[2024]["conservative"],
        "calibrated": primary,
    }
    for name, prediction in predictions.items():
        payload[f"candidate_{name}"] = prediction
    np.savez_compressed(OUTPUT_NPZ, **payload)
    report = {
        "protocol": {
            "official_train_only": True,
            "test_rows_read": False,
            "leaderboard_values_used": False,
            "test_distribution_used": False,
            "row_independent": True,
            "selection": "maximize worst affine gain on 2022 and 2023",
            "confirmation": "apply fixed parameters once to 2024",
        },
        "fixed_estimator": {
            "median_offset": MEDIAN_OFFSET,
            "required_local_score": REQUIRED_LOCAL,
            "target_lb": 1190.0,
        },
        "base_metrics": baselines,
        "variants": variants,
        "selected_variant": selected_name,
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
        "selected_variant": selected_name,
        "score_2024": primary_metrics["raw_competition_score"],
        "expected_lb_median": (
            primary_metrics["raw_competition_score"] + MEDIAN_OFFSET
        ),
    }, ensure_ascii=False, indent=2))
    print(f"Saved {OUTPUT_JSON}")
    print(f"Saved {OUTPUT_NPZ}")


if __name__ == "__main__":
    main()
