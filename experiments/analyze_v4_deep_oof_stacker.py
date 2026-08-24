#!/usr/bin/env python3
"""Leakage-safe deep stacking on frozen rolling predictions.

Each input prediction is out-of-time for its row.  Architecture and blend
strength are selected on the 2022->2023 transfer, then the models are refit on
pooled 2022+2023 rows and confirmed once on 2024.  The stack is routed only to
regular-season rows; F rows retain the locked baseline.
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

# Importing the rolling module first preserves its Windows native-DLL order.
from experiments.run_v2_rolling import TorchTabularModel  # noqa: E402
from experiments.run_baselines import FEATURES  # noqa: E402
from experiments.analyze_v4_temporal_residual_ridge import (  # noqa: E402
    MEDIAN_OFFSET,
    REQUIRED_LOCAL,
    json_safe,
    score,
)
import pandas as pd  # noqa: E402


PREDICTIONS = ROOT / "experiments/results/predictions"
OUTPUT_REPORT = ROOT / "experiments/results/v4_deep_oof_stacker.json"
OUTPUT_PREDICTIONS = PREDICTIONS / "v4_deep_oof_stacker_2024.npz"
OUTPUT_SELECTION_PREDICTIONS = PREDICTIONS / "v4_deep_oof_stacker_2023.npz"
YEARS = (2022, 2023, 2024)
ACCEPTED_COEFFICIENTS = np.asarray(
    [0.35044580887215393, -0.2200834470317333], dtype=np.float64
)


@dataclass(frozen=True)
class Recipe:
    name: str
    architecture: str
    params: dict[str, Any]


def recipes() -> list[Recipe]:
    common = {
        "epochs": 12,
        "predict_batch_size": 8192,
        "learning_rate": 8e-4,
        "weight_decay": 5e-4,
        "dropout": 0.15,
        "loss": "brier",
        "amp": True,
        "device": "cuda",
        "random_seed": 2026,
    }
    return [
        Recipe(
            "embedding_mlp_brier",
            "deep_mlp",
            {
                **common,
                "batch_size": 4096,
                "hidden_dims": [256, 128, 64],
                "embedding_dim": 16,
            },
        ),
        Recipe(
            "tabm_brier",
            "tabm",
            {
                **common,
                "batch_size": 1024,
                "tabm_k": 16,
                "tabm_blocks": 3,
                "tabm_width": 256,
            },
        ),
        Recipe(
            "tabm_bce",
            "tabm",
            {
                **common,
                "batch_size": 1024,
                "tabm_k": 16,
                "tabm_blocks": 3,
                "tabm_width": 256,
                "loss": "bce",
            },
        ),
        Recipe(
            "tabm_bce_brier",
            "tabm",
            {
                **common,
                "batch_size": 1024,
                "tabm_k": 16,
                "tabm_blocks": 3,
                "tabm_width": 256,
                "loss": "bce_brier",
                "brier_weight": 1.0,
            },
        ),
        Recipe(
            "tabm_small_brier",
            "tabm",
            {
                **common,
                "epochs": 8,
                "batch_size": 2048,
                "tabm_k": 8,
                "tabm_blocks": 2,
                "tabm_width": 128,
            },
        ),
        Recipe(
            "tabm_k32_brier",
            "tabm",
            {
                **common,
                "epochs": 10,
                "batch_size": 768,
                "predict_batch_size": 6144,
                "tabm_k": 32,
                "tabm_blocks": 3,
                "tabm_width": 384,
            },
        ),
    ]


def load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path) as archive:
        return {key: np.asarray(archive[key]) for key in archive.files}


def aligned(reference: dict[str, np.ndarray], candidate: dict[str, np.ndarray],
            label: str) -> None:
    for key in ("y", "row_index", "cluster"):
        if not np.array_equal(reference[key], candidate[key]):
            raise ValueError(f"Artifact alignment mismatch for {label}/{key}")


def dynamic_stem(year: int, early: str, confirmation: str) -> str:
    return early if year < 2024 else confirmation


def prediction_sources(year: int) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    locked = load_npz(
        PREDICTIONS / f"v4_pitchtype_failure_tagged_locked_{year}.npz"
    )
    specifications = {
        "meta_tabm": ("v4_tabm_enhanced_all", "tabm_outcome"),
        "meta_tabm_seed42": ("v4_tabm_enhanced_seed42_all", "tabm_outcome"),
        "meta_tabm_rfit": ("v4_tabm_enhanced_rfit_all", "tabm_outcome"),
        "meta_tabm_alltype": ("v4_tabm_enhanced_alltype_all", "tabm_outcome"),
        "meta_tabm_successcall": (
            "v4_tabm_enhanced_successcall_all", "tabm_outcome"
        ),
        "meta_tabm_binary": ("v4_tabm_binary_brier_enhanced_all", "tabm"),
        "meta_numeric": ("v4_numeric_cat_current_tmctx_seed42", "catboost_numeric"),
        "meta_numeric_context": (
            "v4_numeric_cat_current_context_tmctx_seed42", "catboost_numeric"
        ),
        "meta_numeric_level": (
            "v4_numeric_cat_current_context_level_tmctx_seed42", "catboost_numeric"
        ),
        "meta_residual_ensemble": ("v4_residual_ensemble", "residual_ensemble"),
        "meta_neural_conservative": (
            "v4_joint_neural_conservative", "conservative"
        ),
        "meta_stability_a": (
            dynamic_stem(
                year,
                "v4_outcome_a_trackman_stability_backtest",
                "v4_outcome_a_trackman_stability",
            ),
            "catboost_outcome",
        ),
        "meta_stability_b": (
            dynamic_stem(
                year,
                "v4_outcome_b_trackman_stability_backtest",
                "v4_outcome_b_trackman_stability",
            ),
            "catboost_outcome",
        ),
        "meta_stability_c": (
            dynamic_stem(
                year,
                "v4_outcome_c_trackman_stability_backtest",
                "v4_outcome_c_trackman_stability",
            ),
            "catboost_outcome",
        ),
    }
    values = {
        "meta_locked": np.asarray(locked["tagged_locked"], dtype=np.float32),
        "meta_pre_pitchtype": np.asarray(locked["champion"], dtype=np.float32),
    }
    for name, (stem, key) in specifications.items():
        item = load_npz(PREDICTIONS / f"{stem}_{year}.npz")
        aligned(locked, item, f"{name}/{year}")
        values[name] = np.asarray(item[key], dtype=np.float32)
    return locked, values


def make_frames() -> tuple[
    dict[int, pd.DataFrame],
    dict[int, dict[str, np.ndarray]],
    list[str],
]:
    raw_features = [feature for feature in FEATURES if feature != "season"]
    raw = pd.read_csv(
        ROOT / "open/data/train.csv",
        usecols=raw_features,
        encoding="utf-8-sig",
        low_memory=False,
    )
    frames: dict[int, pd.DataFrame] = {}
    artifacts: dict[int, dict[str, np.ndarray]] = {}
    meta_names: list[str] | None = None
    for year in YEARS:
        locked, meta = prediction_sources(year)
        row_index = np.asarray(locked["row_index"], dtype=np.int64)
        frame = raw.iloc[row_index].reset_index(drop=True).copy()
        current_names = list(meta)
        if meta_names is None:
            meta_names = current_names
        elif current_names != meta_names:
            raise ValueError("Meta feature order changed between seasons")
        for name, values in meta.items():
            frame[name] = values
        frame["count_state"] = (
            frame["balls_before"].astype(str)
            + "-"
            + frame["strikes_before"].astype(str)
        )
        frame["hand_matchup"] = (
            frame["pitcher_hand"].astype(str)
            + "-"
            + frame["batter_hand"].astype(str)
        )
        frames[year] = frame
        artifacts[year] = locked
    assert meta_names is not None
    return frames, artifacts, [*raw_features, *meta_names, "count_state", "hand_matchup"]


def accepted_prediction(artifact: dict[str, np.ndarray], frame: pd.DataFrame,
                        meta: pd.DataFrame) -> np.ndarray:
    base = np.asarray(artifact["tagged_locked"], dtype=np.float64)
    route = frame["game_type"].eq("R").to_numpy()
    directions = np.column_stack(
        [
            meta["meta_tabm"].to_numpy(dtype=np.float64) - base,
            meta["meta_stability_b"].to_numpy(dtype=np.float64) - base,
        ]
    )
    result = base.copy()
    result[route] += directions[route] @ ACCEPTED_COEFFICIENTS
    return np.clip(result, 0.0, 1.0)


def fit_predict(recipe: Recipe, train_x: pd.DataFrame, train_y: np.ndarray,
                valid_x: pd.DataFrame) -> tuple[np.ndarray, float]:
    model = TorchTabularModel(list(train_x.columns), recipe.architecture, recipe.params)
    started = time.perf_counter()
    model.fit(train_x, train_y)
    prediction = np.asarray(model.predict_proba(valid_x)[:, 1], dtype=np.float64)
    elapsed = time.perf_counter() - started
    del model
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass
    return prediction, elapsed


def routed_candidate(base: np.ndarray, raw: np.ndarray, route: np.ndarray,
                     gamma: float) -> np.ndarray:
    result = np.asarray(base, dtype=np.float64).copy()
    result[route] = np.clip(
        result[route] + gamma * (raw - result[route]), 0.0, 1.0
    )
    return result


def main() -> None:
    frames, artifacts, features = make_frames()
    route = {
        year: frames[year]["game_type"].eq("R").to_numpy() for year in YEARS
    }
    y = {
        year: np.asarray(artifacts[year]["y"], dtype=np.float64) for year in YEARS
    }
    locked = {
        year: np.asarray(artifacts[year]["tagged_locked"], dtype=np.float64)
        for year in YEARS
    }
    accepted = {
        year: accepted_prediction(artifacts[year], frames[year], frames[year])
        for year in YEARS
    }

    selection_rows: list[dict[str, Any]] = []
    selection_payload: dict[str, np.ndarray] = {
        "y": y[2023],
        "row_index": np.asarray(artifacts[2023]["row_index"]),
        "cluster": np.asarray(artifacts[2023]["cluster"]),
        "accepted": accepted[2023],
        "game_type_r": route[2023],
    }
    for recipe in recipes():
        print(f"[selection] {recipe.name}", flush=True)
        selection_prediction, elapsed = fit_predict(
            recipe,
            frames[2022].loc[route[2022], features],
            y[2022][route[2022]],
            frames[2023].loc[route[2023], features],
        )
        selection_payload[f"raw_{recipe.name}"] = selection_prediction
        direction = selection_prediction - accepted[2023][route[2023]]
        residual = y[2023][route[2023]] - accepted[2023][route[2023]]
        denominator = float(np.dot(direction, direction))
        unconstrained = (
            float(np.dot(direction, residual) / denominator)
            if denominator > 0.0 else 0.0
        )
        # An anti-correlated model can be a useful residual direction (the
        # already locked TrackMan-B arm is one such case).  Bound both signs
        # conservatively instead of silently discarding a stable negative arm.
        gamma = float(np.clip(unconstrained, -0.25, 0.50))
        candidate = routed_candidate(
            accepted[2023], selection_prediction, route[2023], gamma
        )
        baseline_metric = score(y[2023], accepted[2023])
        metric = score(y[2023], candidate)
        selection_rows.append(
            {
                "recipe": recipe,
                "unconstrained_gamma": unconstrained,
                "selected_gamma": gamma,
                "selection_gain": float(
                    metric["raw_competition_score"]
                    - baseline_metric["raw_competition_score"]
                ),
                "selection_metrics": metric,
                "fit_predict_seconds": elapsed,
            }
        )
        print(
            f"  gamma={gamma:.6f} gain={selection_rows[-1]['selection_gain']:+.4f}",
            flush=True,
        )

    ranked = sorted(
        selection_rows,
        key=lambda item: item["selection_gain"],
        reverse=True,
    )
    confirmations: dict[str, dict[str, Any]] = {}
    prediction_payload: dict[str, np.ndarray] = {
        "y": y[2024],
        "row_index": np.asarray(artifacts[2024]["row_index"]),
        "cluster": np.asarray(artifacts[2024]["cluster"]),
        "locked": locked[2024],
        "accepted": accepted[2024],
        "game_type_r": route[2024],
    }
    train_frame = pd.concat(
        [
            frames[2022].loc[route[2022], features],
            frames[2023].loc[route[2023], features],
        ],
        axis=0,
        ignore_index=True,
    )
    train_y = np.concatenate([y[2022][route[2022]], y[2023][route[2023]]])
    recipe_lookup = {recipe.name: recipe for recipe in recipes()}
    for selected in ranked:
        recipe = recipe_lookup[selected["recipe"].name]
        print(f"[confirmation] {recipe.name}", flush=True)
        raw_prediction, elapsed = fit_predict(
            recipe,
            train_frame,
            train_y,
            frames[2024].loc[route[2024], features],
        )
        gamma = float(selected["selected_gamma"])
        candidate = routed_candidate(
            accepted[2024], raw_prediction, route[2024], gamma
        )
        baseline_metric = score(y[2024], accepted[2024])
        metric = score(y[2024], candidate)
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
            "raw_prediction_mean": float(raw_prediction.mean()),
            "raw_prediction_std": float(raw_prediction.std()),
        }
        prediction_payload[f"raw_{recipe.name}"] = raw_prediction
        prediction_payload[f"candidate_{recipe.name}"] = candidate
        print(
            f"  gain={confirmations[recipe.name]['gain_over_accepted']:+.4f} "
            f"local={metric['raw_competition_score']:.4f}",
            flush=True,
        )

    np.savez_compressed(OUTPUT_SELECTION_PREDICTIONS, **selection_payload)
    np.savez_compressed(OUTPUT_PREDICTIONS, **prediction_payload)
    report = {
        "protocol": {
            "official_train_only": True,
            "test_rows_read": False,
            "all_meta_predictions_are_out_of_time": True,
            "selection_transfer": [2022, 2023],
            "confirmation_training_years": [2022, 2023],
            "confirmation_year": 2024,
            "route": "R rows only; F rows retain accepted locked stack",
            "row_independent_inference": True,
        },
        "feature_count": len(features),
        "features": features,
        "accepted_baseline_2024": score(y[2024], accepted[2024]),
        "selection_ranked": [
            {
                **{key: value for key, value in row.items() if key != "recipe"},
                "recipe": {
                    "name": row["recipe"].name,
                    "architecture": row["recipe"].architecture,
                    "params": row["recipe"].params,
                },
            }
            for row in ranked
        ],
        "confirmations_2024": confirmations,
        "required_local_score": REQUIRED_LOCAL,
        "prediction_artifact": str(OUTPUT_PREDICTIONS.relative_to(ROOT)),
        "selection_prediction_artifact": str(
            OUTPUT_SELECTION_PREDICTIONS.relative_to(ROOT)
        ),
    }
    OUTPUT_REPORT.write_text(
        json.dumps(json_safe(report), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(json_safe({
        "selection": {
            row["recipe"].name: {
                "gamma": row["selected_gamma"],
                "gain": row["selection_gain"],
            }
            for row in ranked
        },
        "confirmations_2024": confirmations,
    }), ensure_ascii=False, indent=2), flush=True)
    print(f"Saved {OUTPUT_REPORT}", flush=True)
    print(f"Saved {OUTPUT_SELECTION_PREDICTIONS}", flush=True)
    print(f"Saved {OUTPUT_PREDICTIONS}", flush=True)


if __name__ == "__main__":
    main()
