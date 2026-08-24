#!/usr/bin/env python3
"""Blend only the incremental effect of current-state CatBoost interactions.

Weights are selected on the 2022 and 2023 temporal folds.  The chosen pair is
then applied once to 2024.  The construction never reads test rows or
leaderboard values; all inputs are official-train OOF prediction artifacts.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np


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
BASE_STEM = "v4_numeric_cat_current_tmctx_seed42"
CONTEXT_STEM = "v4_numeric_cat_current_context_tmctx_seed42"
LEVEL_STEM = "v4_numeric_cat_current_context_level_tmctx_seed42"
RESIDUAL_STEM = "v4_residual_ensemble"
OUTPUT_JSON = ROOT / "experiments/results/v4_context_cat_delta.json"
OUTPUT_NPZ = PREDICTIONS / "v4_context_cat_delta_2024.npz"
WEIGHT_GRID = np.round(np.arange(-2.0, 3.0001, 0.05), 8)


def load(season: int) -> dict[str, np.ndarray]:
    result: dict[str, np.ndarray] = {}
    with np.load(PREDICTIONS / f"{RESIDUAL_STEM}_{season}.npz") as archive:
        for key in ("y", "row_index", "cluster", "m3", "residual_ensemble"):
            result[key] = np.asarray(archive[key])
    for key, stem in (
        ("numeric_base", BASE_STEM),
        ("numeric_context", CONTEXT_STEM),
        ("numeric_level", LEVEL_STEM),
    ):
        with np.load(PREDICTIONS / f"{stem}_{season}.npz") as archive:
            if not np.array_equal(result["row_index"], archive["row_index"]):
                raise ValueError(f"Prediction alignment mismatch: {stem}/{season}")
            result[key] = np.asarray(archive["catboost_numeric"], dtype=np.float64)
    return result


def predict(
    artifact: dict[str, np.ndarray], context_weight: float, level_weight: float
) -> np.ndarray:
    context_delta = artifact["numeric_context"] - artifact["numeric_base"]
    level_delta = artifact["numeric_level"] - artifact["numeric_context"]
    return np.clip(
        artifact["residual_ensemble"]
        + context_weight * context_delta
        + level_weight * level_delta,
        0.0,
        1.0,
    )


def main() -> None:
    artifacts = {season: load(season) for season in (2022, 2023, 2024)}
    baselines = {
        season: score(item["y"], item["residual_ensemble"])
        for season, item in artifacts.items()
    }
    trials: list[dict[str, Any]] = []
    best: tuple[tuple[float, float], dict[str, Any]] | None = None
    for context_weight in WEIGHT_GRID:
        for level_weight in WEIGHT_GRID:
            gains: dict[str, float] = {}
            metrics: dict[str, Any] = {}
            for season in (2022, 2023):
                item = artifacts[season]
                current = score(
                    item["y"], predict(item, context_weight, level_weight)
                )
                gains[str(season)] = float(
                    current["raw_competition_score"]
                    - baselines[season]["raw_competition_score"]
                )
                metrics[str(season)] = current
            row = {
                "context_weight": float(context_weight),
                "level_weight": float(level_weight),
                "gains": gains,
                "robust_min_gain": float(min(gains.values())),
                "mean_gain": float(np.mean(list(gains.values()))),
                "metrics": metrics,
            }
            trials.append(row)
            rank = (row["robust_min_gain"], row["mean_gain"])
            if best is None or rank > best[0]:
                best = (rank, row)
    assert best is not None
    selected = best[1]
    item_2024 = artifacts[2024]
    prediction_2024 = predict(
        item_2024, selected["context_weight"], selected["level_weight"]
    )
    metrics_2024 = score(item_2024["y"], prediction_2024)
    gain_2024 = float(
        metrics_2024["raw_competition_score"]
        - baselines[2024]["raw_competition_score"]
    )
    np.savez_compressed(
        OUTPUT_NPZ,
        y=item_2024["y"],
        row_index=item_2024["row_index"],
        cluster=item_2024["cluster"],
        m3=item_2024["m3"],
        residual_ensemble=item_2024["residual_ensemble"],
        numeric_base=item_2024["numeric_base"],
        numeric_context=item_2024["numeric_context"],
        numeric_level=item_2024["numeric_level"],
        context_cat_delta=prediction_2024,
    )
    report = {
        "protocol": {
            "official_train_only": True,
            "test_rows_read": False,
            "leaderboard_values_used": False,
            "selection": "maximize worst gain on 2022 and 2023",
            "confirmation": "apply selected weights once to 2024",
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
        json.dumps(json_safe(report), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "selected_weights": {
                    "context": selected["context_weight"],
                    "level": selected["level_weight"],
                },
                "selection_gains": selected["gains"],
                "gain_2024": gain_2024,
                "score_2024": metrics_2024["raw_competition_score"],
                "expected_lb_median": (
                    metrics_2024["raw_competition_score"] + MEDIAN_OFFSET
                ),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    print(f"Saved {OUTPUT_JSON}")
    print(f"Saved {OUTPUT_NPZ}")


if __name__ == "__main__":
    main()
