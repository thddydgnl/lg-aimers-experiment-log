#!/usr/bin/env python3
"""Robust next-season residual model zoo for the fixed V3 M3 blend.

Every recipe is selected by its *worst* gain over two untouched temporal
transfers (2021->2022 and 2022->2023).  Only after selection is it refit on
2023 and evaluated on 2024.  The script reads official train data and stored
OOF predictions only; it never reads test rows or leaderboard values.
"""

from __future__ import annotations

import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

# LightGBM must be imported before CatBoost on Windows.  Importing CatBoost's
# bundled OpenMP runtime or pandas' native stack first makes LightGBM 4.7 fail
# in DatasetSetField with a native access violation even on a tiny finite array.
from lightgbm import LGBMRegressor

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.analyze_v4_temporal_residual_ridge import (  # noqa: E402
    MEDIAN_OFFSET,
    REQUIRED_LOCAL,
    Config,
    correction_diagnostics,
    json_safe,
    load_frames,
    score,
    transfer_data,
)


OUTPUT_JSON = ROOT / "experiments/results/v4_temporal_residual_models.json"
OUTPUT_NPZ = (
    ROOT / "experiments/results/predictions/v4_temporal_residual_models_2024.npz"
)
TRANSITIONS = ((2021, 2022), (2022, 2023))
GAMMAS = (0.03, 0.05, 0.08, 0.10, 0.15, 0.20, 0.30, 0.40, 0.55, 0.75, 1.00)

RAW_NUMERIC_COLUMNS = [
    "game_month",
    "game_dayofweek",
    "inning",
    "balls_before",
    "strikes_before",
    "outs_before",
    "run_top_before",
    "run_bot_before",
    "run_total_before",
    "score_diff_home",
    "score_diff_pitcher_team",
    "runner_on_1b",
    "runner_on_2b",
    "runner_on_3b",
    "num_runners_on",
    "home_win_expectancy",
    "away_win_expectancy",
    "li",
    "pitcher_hand",
    "batter_hand",
    "asof_pitcher_n",
    "asof_pitcher_success_rate",
    "asof_pitcher_reverse_rate",
    "asof_pitcher_middle_rate",
    "asof_pitcher_ball_rate",
    "asof_pitcher_strike_rate",
    "asof_pitcher_prev1_game_success_rate",
    "asof_pitcher_prev3_game_success_rate",
    "asof_pitcher_prev5_game_success_rate",
    "asof_pitcher_prev1_game_middle_rate",
    "asof_pitcher_prev3_game_middle_rate",
    "asof_pitcher_prev5_game_middle_rate",
    "asof_batter_n",
    "asof_batter_success_rate",
    "asof_batter_middle_rate",
    "asof_pitcher_pitchmix_n",
    "asof_pitcher_fastball_rate",
    "asof_pitcher_breaking_rate",
    "asof_pitcher_offspeed_rate",
]


@dataclass(frozen=True)
class Recipe:
    name: str
    family: str
    feature_set: str
    training_mode: str
    factory: Callable[[], Any]


def model_recipes() -> list[Recipe]:
    recipes: list[Recipe] = []
    for mode in ("loo", "full"):
        recipes.extend(
            [
                Recipe(
                    f"ridge_aug_a1000_{mode}",
                    "ridge",
                    "augmented",
                    mode,
                    lambda: make_pipeline(
                        StandardScaler(), Ridge(alpha=1000.0, fit_intercept=True)
                    ),
                ),
                Recipe(
                    f"ridge_aug_a10000_{mode}",
                    "ridge",
                    "augmented",
                    mode,
                    lambda: make_pipeline(
                        StandardScaler(), Ridge(alpha=10000.0, fit_intercept=True)
                    ),
                ),
                Recipe(
                    f"hgb_l7_leaf2000_{mode}",
                    "hgb",
                    "augmented",
                    mode,
                    lambda: HistGradientBoostingRegressor(
                        loss="squared_error",
                        learning_rate=0.04,
                        max_iter=250,
                        max_leaf_nodes=7,
                        min_samples_leaf=2000,
                        l2_regularization=30.0,
                        random_state=2026,
                    ),
                ),
                Recipe(
                    f"hgb_l15_leaf1000_{mode}",
                    "hgb",
                    "augmented",
                    mode,
                    lambda: HistGradientBoostingRegressor(
                        loss="squared_error",
                        learning_rate=0.03,
                        max_iter=300,
                        max_leaf_nodes=15,
                        min_samples_leaf=1000,
                        l2_regularization=100.0,
                        random_state=2026,
                    ),
                ),
                Recipe(
                    f"lgb_l7_leaf2000_{mode}",
                    "lightgbm",
                    "augmented",
                    mode,
                    lambda: LGBMRegressor(
                        objective="regression_l2",
                        n_estimators=400,
                        learning_rate=0.025,
                        num_leaves=7,
                        max_depth=4,
                        min_child_samples=2000,
                        reg_alpha=10.0,
                        reg_lambda=100.0,
                        colsample_bytree=0.85,
                        random_state=2026,
                        n_jobs=-1,
                        verbosity=-1,
                    ),
                ),
                Recipe(
                    f"lgb_l15_leaf1000_{mode}",
                    "lightgbm",
                    "augmented",
                    mode,
                    lambda: LGBMRegressor(
                        objective="regression_l2",
                        n_estimators=500,
                        learning_rate=0.02,
                        num_leaves=15,
                        max_depth=6,
                        min_child_samples=1000,
                        reg_alpha=20.0,
                        reg_lambda=150.0,
                        colsample_bytree=0.85,
                        random_state=2026,
                        n_jobs=-1,
                        verbosity=-1,
                    ),
                ),
                Recipe(
                    f"extra_d8_leaf1000_{mode}",
                    "extra_trees",
                    "augmented",
                    mode,
                    lambda: ExtraTreesRegressor(
                        n_estimators=160,
                        max_depth=8,
                        min_samples_leaf=1000,
                        max_features=0.8,
                        n_jobs=-1,
                        random_state=2026,
                    ),
                ),
                Recipe(
                    f"cat_d4_l2_100_{mode}",
                    "catboost",
                    "augmented",
                    mode,
                    lambda: CatBoostRegressor(
                        loss_function="RMSE",
                        iterations=400,
                        depth=4,
                        learning_rate=0.035,
                        l2_leaf_reg=100.0,
                        random_seed=2026,
                        task_type="GPU",
                        devices="0",
                        verbose=False,
                        allow_writing_files=False,
                    ),
                ),
                Recipe(
                    f"cat_d6_l2_150_{mode}",
                    "catboost",
                    "augmented",
                    mode,
                    lambda: CatBoostRegressor(
                        loss_function="RMSE",
                        iterations=500,
                        depth=6,
                        learning_rate=0.025,
                        l2_leaf_reg=150.0,
                        random_seed=2026,
                        task_type="GPU",
                        devices="0",
                        verbose=False,
                        allow_writing_files=False,
                    ),
                ),
            ]
        )
    return recipes


def prediction_matrix(artifact: dict[str, np.ndarray], mask: np.ndarray) -> np.ndarray:
    a = np.asarray(artifact["component_A"], dtype=np.float64)[mask]
    b = np.asarray(artifact["component_B"], dtype=np.float64)[mask]
    c = np.asarray(artifact["component_C"], dtype=np.float64)[mask]
    raw = np.asarray(artifact["m3_raw"], dtype=np.float64)[mask]
    calibrated = np.asarray(artifact["m3"], dtype=np.float64)[mask]
    clipped = np.clip(calibrated, 1e-5, 1.0 - 1e-5)
    components = np.column_stack([a, b, c])
    return np.column_stack(
        [
            calibrated,
            raw,
            np.log(clipped / (1.0 - clipped)),
            calibrated * (1.0 - calibrated),
            components.std(axis=1),
            a - b,
            a - c,
            b - c,
            a,
            b,
            c,
        ]
    )


def raw_matrix(frame: pd.DataFrame, mask: np.ndarray) -> np.ndarray:
    values = frame.loc[mask, RAW_NUMERIC_COLUMNS].to_numpy(dtype=np.float64)
    month = frame.loc[mask, "game_month"].to_numpy(dtype=np.float64)
    inning = frame.loc[mask, "inning"].to_numpy(dtype=np.float64)
    extra = np.column_stack(
        [
            np.sin(2.0 * np.pi * month / 12.0),
            np.cos(2.0 * np.pi * month / 12.0),
            np.minimum(inning, 10.0),
            np.log1p(frame.loc[mask, "asof_pitcher_n"].to_numpy(dtype=np.float64)),
            np.log1p(frame.loc[mask, "asof_batter_n"].to_numpy(dtype=np.float64)),
            np.log1p(
                frame.loc[mask, "asof_pitcher_pitchmix_n"].to_numpy(dtype=np.float64)
            ),
        ]
    )
    return np.column_stack([values, extra])


def fill_from_source(source: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    source = np.asarray(source, dtype=np.float64).copy()
    target = np.asarray(target, dtype=np.float64).copy()
    finite_source = np.where(np.isfinite(source), source, np.nan)
    medians = np.nanmedian(finite_source, axis=0)
    medians[~np.isfinite(medians)] = 0.0
    source_bad = ~np.isfinite(source)
    target_bad = ~np.isfinite(target)
    if source_bad.any():
        source[source_bad] = np.take(medians, np.where(source_bad)[1])
    if target_bad.any():
        target[target_bad] = np.take(medians, np.where(target_bad)[1])
    return source.astype(np.float32), target.astype(np.float32)


def add_raw_columns(
    frames: dict[int, pd.DataFrame], artifacts: dict[int, dict[str, np.ndarray]]
) -> None:
    full = pd.read_csv(
        ROOT / "open/data/train.csv",
        usecols=RAW_NUMERIC_COLUMNS,
        encoding="utf-8-sig",
        low_memory=False,
    )
    for season, frame in frames.items():
        row_index = np.asarray(artifacts[season]["row_index"], dtype=np.int64)
        selected = full.iloc[row_index].reset_index(drop=True)
        for column in RAW_NUMERIC_COLUMNS:
            frame[column] = selected[column].to_numpy()


def build_data(
    frames: dict[int, pd.DataFrame],
    artifacts: dict[int, dict[str, np.ndarray]],
    source: int,
    target: int,
    mode: str,
) -> dict[str, Any]:
    config = Config("r_all", 800.0, 800.0, 1600.0, 1.0, 1.0, mode)
    data = transfer_data(
        frames[source], frames[target], artifacts[source]["m3"], config
    )
    source_pred = prediction_matrix(artifacts[source], data["source_core"])
    target_pred = prediction_matrix(artifacts[target], data["target_core"])
    source_raw = raw_matrix(frames[source], data["source_core"])
    target_raw = raw_matrix(frames[target], data["target_core"])
    x_source = np.column_stack([data["x_source"], source_pred, source_raw])
    x_target = np.column_stack([data["x_target"], target_pred, target_raw])
    x_source, x_target = fill_from_source(x_source, x_target)
    return {**data, "x_source": x_source, "x_target": x_target}


def prediction_with_correction(
    baseline: np.ndarray,
    target_core: np.ndarray,
    correction: np.ndarray,
    gamma: float,
) -> np.ndarray:
    prediction = np.asarray(baseline, dtype=np.float64).copy()
    prediction[target_core] = np.clip(
        prediction[target_core] + gamma * correction, 0.0, 1.0
    )
    return prediction


def evaluate_recipe(
    recipe: Recipe,
    data_by_mode: dict[str, dict[tuple[int, int], dict[str, Any]]],
    artifacts: dict[int, dict[str, np.ndarray]],
    baselines: dict[int, dict[str, Any]],
) -> tuple[dict[str, Any], dict[tuple[int, int], np.ndarray]]:
    corrections: dict[tuple[int, int], np.ndarray] = {}
    fit_seconds = 0.0
    for transition in TRANSITIONS:
        data = data_by_mode[recipe.training_mode][transition]
        model = recipe.factory()
        started = time.perf_counter()
        model.fit(data["x_source"], data["residual"])
        corrections[transition] = np.asarray(
            model.predict(data["x_target"]), dtype=np.float64
        )
        fit_seconds += time.perf_counter() - started

    gamma_trials: list[dict[str, Any]] = []
    best: tuple[tuple[float, float], float, dict[str, Any]] | None = None
    for gamma in GAMMAS:
        transition_results: dict[str, Any] = {}
        gains: list[float] = []
        for transition in TRANSITIONS:
            source, target = transition
            data = data_by_mode[recipe.training_mode][transition]
            prediction = prediction_with_correction(
                artifacts[target]["m3"],
                data["target_core"],
                corrections[transition],
                gamma,
            )
            metrics = score(artifacts[target]["y"], prediction)
            gain = float(
                metrics["raw_competition_score"]
                - baselines[target]["raw_competition_score"]
            )
            gains.append(gain)
            transition_results[f"{source}_to_{target}"] = {
                "gain": gain,
                "metrics": metrics,
            }
        robust_min = float(min(gains))
        mean_gain = float(np.mean(gains))
        row = {
            "gamma": gamma,
            "robust_min_gain": robust_min,
            "mean_gain": mean_gain,
            "transitions": transition_results,
        }
        gamma_trials.append(row)
        rank = (robust_min, mean_gain)
        if best is None or rank > best[0]:
            best = (rank, gamma, row)
    assert best is not None
    return (
        {
            "recipe": {
                "name": recipe.name,
                "family": recipe.family,
                "feature_set": recipe.feature_set,
                "training_mode": recipe.training_mode,
            },
            "selected_gamma": best[1],
            "robust_min_gain": best[0][0],
            "mean_gain": best[0][1],
            "selection": best[2],
            "fit_seconds": fit_seconds,
            "gamma_trials": gamma_trials,
        },
        corrections,
    )


def confirm(
    recipe: Recipe,
    gamma: float,
    data: dict[str, Any],
    artifacts: dict[int, dict[str, np.ndarray]],
    baseline_2024: dict[str, Any],
) -> tuple[np.ndarray, dict[str, Any]]:
    model = recipe.factory()
    started = time.perf_counter()
    model.fit(data["x_source"], data["residual"])
    correction = np.asarray(model.predict(data["x_target"]), dtype=np.float64)
    prediction = prediction_with_correction(
        artifacts[2024]["m3"], data["target_core"], correction, gamma
    )
    metrics = score(artifacts[2024]["y"], prediction)
    return prediction, {
        "metrics": metrics,
        "gain": float(
            metrics["raw_competition_score"]
            - baseline_2024["raw_competition_score"]
        ),
        "expected_lb_median": float(
            metrics["raw_competition_score"] + MEDIAN_OFFSET
        ),
        "crosses_required_local_score": bool(
            metrics["raw_competition_score"] > REQUIRED_LOCAL
        ),
        "fit_seconds": time.perf_counter() - started,
        "diagnostics": correction_diagnostics(data, correction),
    }


def main() -> None:
    frames, artifacts = load_frames()
    add_raw_columns(frames, artifacts)
    baselines = {
        season: score(artifacts[season]["y"], artifacts[season]["m3"])
        for season in (2021, 2022, 2023, 2024)
    }
    data_by_mode: dict[str, dict[tuple[int, int], dict[str, Any]]] = {}
    for mode in ("loo", "full"):
        data_by_mode[mode] = {
            transition: build_data(frames, artifacts, *transition, mode)
            for transition in (*TRANSITIONS, (2023, 2024))
        }

    recipes = model_recipes()
    results: list[dict[str, Any]] = []
    failures: list[dict[str, str]] = []
    recipe_lookup = {recipe.name: recipe for recipe in recipes}
    for index, recipe in enumerate(recipes, start=1):
        try:
            result, _ = evaluate_recipe(recipe, data_by_mode, artifacts, baselines)
        except Exception as exc:  # keep independent model families running
            failures.append(
                {
                    "recipe": recipe.name,
                    "exception": type(exc).__name__,
                    "message": str(exc),
                }
            )
            print(
                f"[{index:02d}/{len(recipes):02d}] {recipe.name}: "
                f"FAILED {type(exc).__name__}: {exc}",
                flush=True,
            )
            continue
        results.append(result)
        print(
            f"[{index:02d}/{len(recipes):02d}] {recipe.name}: "
            f"min={result['robust_min_gain']:+.4f} "
            f"mean={result['mean_gain']:+.4f} gamma={result['selected_gamma']:.2f}",
            flush=True,
        )

    if not results:
        raise RuntimeError("All temporal residual model recipes failed")
    ranked = sorted(
        results,
        key=lambda row: (row["robust_min_gain"], row["mean_gain"]),
        reverse=True,
    )
    best_by_family: dict[str, dict[str, Any]] = {}
    for row in ranked:
        family = row["recipe"]["family"]
        best_by_family.setdefault(family, row)

    confirmation_rows: dict[str, Any] = {}
    confirmation_predictions: dict[str, np.ndarray] = {}
    names_to_confirm = [ranked[0]["recipe"]["name"]]
    names_to_confirm.extend(
        row["recipe"]["name"] for row in best_by_family.values()
    )
    for name in dict.fromkeys(names_to_confirm):
        row = next(item for item in results if item["recipe"]["name"] == name)
        recipe = recipe_lookup[name]
        prediction, confirmation = confirm(
            recipe,
            float(row["selected_gamma"]),
            data_by_mode[recipe.training_mode][(2023, 2024)],
            artifacts,
            baselines[2024],
        )
        confirmation_rows[name] = confirmation
        confirmation_predictions[name] = prediction
        print(
            f"[confirm] {name}: gain={confirmation['gain']:+.4f} "
            f"local={confirmation['metrics']['raw_competition_score']:.4f}",
            flush=True,
        )

    primary_name = ranked[0]["recipe"]["name"]
    primary_prediction = confirmation_predictions[primary_name]
    npz_payload: dict[str, np.ndarray] = {
        "y": artifacts[2024]["y"],
        "row_index": artifacts[2024]["row_index"],
        "cluster": artifacts[2024]["cluster"],
        "m3": artifacts[2024]["m3"],
        "temporal_residual_model": primary_prediction,
    }
    for name, prediction in confirmation_predictions.items():
        npz_payload[f"candidate_{name}"] = prediction
    np.savez_compressed(OUTPUT_NPZ, **npz_payload)

    report = {
        "protocol": {
            "official_train_only": True,
            "test_rows_read": False,
            "leaderboard_values_used": False,
            "row_independent_target_features": True,
            "selection": "maximize worst gain over 2021->2022 and 2022->2023",
            "confirmation": "refit selected recipes on 2023 and transfer to 2024",
            "route": "R_CORE only; R_ANCHOR and F unchanged",
            "feature_count": int(
                data_by_mode["loo"][(2021, 2022)]["x_source"].shape[1]
            ),
        },
        "fixed_estimator": {
            "median_offset": MEDIAN_OFFSET,
            "required_local_score": REQUIRED_LOCAL,
            "target_lb": 1190.0,
        },
        "baselines": baselines,
        "ranked_selection": ranked,
        "failures": failures,
        "best_by_family": best_by_family,
        "primary_name": primary_name,
        "confirmations_2024": confirmation_rows,
        "prediction_artifact": str(OUTPUT_NPZ.relative_to(ROOT)),
    }
    OUTPUT_JSON.write_text(
        json.dumps(json_safe(report), ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "primary": primary_name,
                "selection_min_gain": ranked[0]["robust_min_gain"],
                "selection_mean_gain": ranked[0]["mean_gain"],
                "confirmation": confirmation_rows[primary_name],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    print(f"Saved {OUTPUT_JSON}")
    print(f"Saved {OUTPUT_NPZ}")


if __name__ == "__main__":
    main()
