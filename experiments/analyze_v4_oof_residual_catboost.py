#!/usr/bin/env python3
"""Residual CatBoost stackers over frozen OOT model predictions.

The residual model is selected on 2022->2023, refit on pooled 2022+2023, and
confirmed once on 2024.  All feature encoders and model predictions are based
only on earlier seasons for each validation row.
"""

from __future__ import annotations

import gc
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.analyze_v4_deep_oof_stacker import (  # noqa: E402
    accepted_prediction,
    make_frames,
)
from experiments.analyze_v4_temporal_residual_ridge import (  # noqa: E402
    MEDIAN_OFFSET,
    REQUIRED_LOCAL,
    json_safe,
    score,
)
from experiments.run_v2_rolling import BOOSTER_CATEGORICAL  # noqa: E402
import pandas as pd  # noqa: E402


PREDICTIONS = ROOT / "experiments/results/predictions"
OUTPUT_REPORT = ROOT / "experiments/results/v4_oof_residual_catboost.json"
OUTPUT_SELECTION = PREDICTIONS / "v4_oof_residual_catboost_2023.npz"
OUTPUT_CONFIRMATION = PREDICTIONS / "v4_oof_residual_catboost_2024.npz"


@dataclass(frozen=True)
class Recipe:
    name: str
    depth: int
    iterations: int
    learning_rate: float
    l2_leaf_reg: float


RECIPES = (
    Recipe("cat_d3_l100", 3, 350, 0.035, 100.0),
    Recipe("cat_d4_l150", 4, 450, 0.030, 150.0),
    Recipe("cat_d6_l250", 6, 550, 0.025, 250.0),
)


def prepare(frame: pd.DataFrame, features: list[str],
            categorical: list[str]) -> pd.DataFrame:
    result = frame.loc[:, features].copy()
    for column in categorical:
        result[column] = (
            result[column].astype("string").fillna("__missing__").astype(str)
        )
    return result


def fit_predict(recipe: Recipe, train_x: pd.DataFrame, train_y: np.ndarray,
                valid_x: pd.DataFrame, categorical: list[str]) -> tuple[np.ndarray, float]:
    from catboost import CatBoostRegressor

    model = CatBoostRegressor(
        loss_function="RMSE",
        iterations=recipe.iterations,
        depth=recipe.depth,
        learning_rate=recipe.learning_rate,
        l2_leaf_reg=recipe.l2_leaf_reg,
        random_seed=2026,
        task_type="GPU",
        devices="0",
        verbose=False,
        allow_writing_files=False,
        bootstrap_type="Bayesian",
        bagging_temperature=0.5,
        random_strength=0.5,
    )
    started = time.perf_counter()
    model.fit(train_x, train_y, cat_features=categorical)
    prediction = np.asarray(model.predict(valid_x), dtype=np.float64)
    elapsed = time.perf_counter() - started
    del model
    gc.collect()
    return prediction, elapsed


def apply(base: np.ndarray, route: np.ndarray, correction: np.ndarray,
          gamma: float) -> np.ndarray:
    result = np.asarray(base, dtype=np.float64).copy()
    result[route] = np.clip(
        result[route] + gamma * correction, 0.0, 1.0
    )
    return result


def main() -> None:
    frames, artifacts, features = make_frames()
    route = {
        year: frames[year]["game_type"].eq("R").to_numpy()
        for year in (2022, 2023, 2024)
    }
    y = {
        year: np.asarray(artifacts[year]["y"], dtype=np.float64)
        for year in (2022, 2023, 2024)
    }
    accepted = {
        year: accepted_prediction(artifacts[year], frames[year], frames[year])
        for year in (2022, 2023, 2024)
    }
    categorical = [
        column for column in features
        if column in BOOSTER_CATEGORICAL
        or not pd.api.types.is_numeric_dtype(frames[2022][column].dtype)
    ]
    prepared = {
        year: prepare(frames[year], features, categorical)
        for year in (2022, 2023, 2024)
    }

    selection_payload: dict[str, np.ndarray] = {
        "y": y[2023],
        "row_index": np.asarray(artifacts[2023]["row_index"]),
        "cluster": np.asarray(artifacts[2023]["cluster"]),
        "accepted": accepted[2023],
        "game_type_r": route[2023],
    }
    selection_rows: list[dict[str, Any]] = []
    for recipe in RECIPES:
        print(f"[selection] {recipe.name}", flush=True)
        residual = y[2022][route[2022]] - accepted[2022][route[2022]]
        correction, elapsed = fit_predict(
            recipe,
            prepared[2022].loc[route[2022]],
            residual,
            prepared[2023].loc[route[2023]],
            categorical,
        )
        target_residual = y[2023][route[2023]] - accepted[2023][route[2023]]
        denominator = float(np.dot(correction, correction))
        unconstrained = (
            float(np.dot(correction, target_residual) / denominator)
            if denominator > 0.0 else 0.0
        )
        gamma = float(np.clip(unconstrained, -1.0, 1.0))
        candidate = apply(accepted[2023], route[2023], correction, gamma)
        metric = score(y[2023], candidate)
        baseline_metric = score(y[2023], accepted[2023])
        selection_rows.append({
            "recipe": recipe,
            "unconstrained_gamma": unconstrained,
            "selected_gamma": gamma,
            "gain": float(
                metric["raw_competition_score"]
                - baseline_metric["raw_competition_score"]
            ),
            "metrics": metric,
            "fit_predict_seconds": elapsed,
            "correction_mean": float(correction.mean()),
            "correction_std": float(correction.std()),
        })
        selection_payload[f"correction_{recipe.name}"] = correction
        print(f"  gamma={gamma:+.6f} gain={selection_rows[-1]['gain']:+.4f}", flush=True)

    ranked = sorted(selection_rows, key=lambda item: item["gain"], reverse=True)
    train_x = pd.concat(
        [prepared[2022].loc[route[2022]], prepared[2023].loc[route[2023]]],
        axis=0,
        ignore_index=True,
    )
    train_residual = np.concatenate([
        y[2022][route[2022]] - accepted[2022][route[2022]],
        y[2023][route[2023]] - accepted[2023][route[2023]],
    ])
    confirmation_payload: dict[str, np.ndarray] = {
        "y": y[2024],
        "row_index": np.asarray(artifacts[2024]["row_index"]),
        "cluster": np.asarray(artifacts[2024]["cluster"]),
        "accepted": accepted[2024],
        "game_type_r": route[2024],
    }
    confirmations: dict[str, Any] = {}
    recipe_lookup = {recipe.name: recipe for recipe in RECIPES}
    for selected in ranked:
        recipe = recipe_lookup[selected["recipe"].name]
        print(f"[confirmation] {recipe.name}", flush=True)
        correction, elapsed = fit_predict(
            recipe,
            train_x,
            train_residual,
            prepared[2024].loc[route[2024]],
            categorical,
        )
        gamma = float(selected["selected_gamma"])
        candidate = apply(accepted[2024], route[2024], correction, gamma)
        metric = score(y[2024], candidate)
        baseline_metric = score(y[2024], accepted[2024])
        confirmations[recipe.name] = {
            "selected_gamma": gamma,
            "gain_over_accepted": float(
                metric["raw_competition_score"]
                - baseline_metric["raw_competition_score"]
            ),
            "metrics": metric,
            "expected_lb_median": float(
                metric["raw_competition_score"] + MEDIAN_OFFSET
            ),
            "crosses_required_local_score": bool(
                metric["raw_competition_score"] > REQUIRED_LOCAL
            ),
            "fit_predict_seconds": elapsed,
            "correction_mean": float(correction.mean()),
            "correction_std": float(correction.std()),
        }
        confirmation_payload[f"correction_{recipe.name}"] = correction
        confirmation_payload[f"candidate_{recipe.name}"] = candidate
        print(
            f"  gain={confirmations[recipe.name]['gain_over_accepted']:+.4f} "
            f"local={metric['raw_competition_score']:.4f}",
            flush=True,
        )

    np.savez_compressed(OUTPUT_SELECTION, **selection_payload)
    np.savez_compressed(OUTPUT_CONFIRMATION, **confirmation_payload)
    report = {
        "protocol": {
            "official_train_only": True,
            "test_rows_read": False,
            "meta_predictions_out_of_time": True,
            "selection_transfer": [2022, 2023],
            "confirmation_training_years": [2022, 2023],
            "confirmation_year": 2024,
            "route": "R rows only",
            "row_independent_inference": True,
        },
        "features": features,
        "categorical_features": categorical,
        "selection_ranked": [
            {
                **{key: value for key, value in row.items() if key != "recipe"},
                "recipe": row["recipe"].__dict__,
            }
            for row in ranked
        ],
        "accepted_baseline_2024": score(y[2024], accepted[2024]),
        "confirmations_2024": confirmations,
        "required_local_score": REQUIRED_LOCAL,
        "selection_artifact": str(OUTPUT_SELECTION.relative_to(ROOT)),
        "confirmation_artifact": str(OUTPUT_CONFIRMATION.relative_to(ROOT)),
    }
    OUTPUT_REPORT.write_text(
        json.dumps(json_safe(report), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(json_safe({
        "selection": {
            row["recipe"].name: {"gamma": row["selected_gamma"], "gain": row["gain"]}
            for row in ranked
        },
        "confirmations_2024": confirmations,
    }), ensure_ascii=False, indent=2), flush=True)
    print(f"Saved {OUTPUT_REPORT}", flush=True)


if __name__ == "__main__":
    main()
