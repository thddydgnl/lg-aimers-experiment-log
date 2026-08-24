#!/usr/bin/env python3
"""External-fold recent-workload decoder features for V5 rolling models."""

from __future__ import annotations

import gc
import json
import os
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier

HORIZONS = (1, 3, 5)
PROFILE_STATS = ("count", "mean", "std", "q25", "q50", "q75")


def derive_game_ids(frame: pd.DataFrame) -> np.ndarray:
    season = frame["season"].fillna(-1).to_numpy(dtype=np.int64, copy=False)
    month = frame["game_month"].fillna(-1).to_numpy(dtype=np.int64, copy=False)
    dow = frame["game_dayofweek"].fillna(-1).to_numpy(dtype=np.int64, copy=False)
    pitcher_team = frame["pitcher_team_id"].fillna(-1).to_numpy(
        dtype=np.int64, copy=False
    )
    batter_team = frame["batter_team_id"].fillna(-1).to_numpy(
        dtype=np.int64, copy=False
    )
    key = np.stack([
        season, month, dow,
        np.minimum(pitcher_team, batter_team),
        np.maximum(pitcher_team, batter_team),
    ], axis=1)
    half = frame["top_bottom"].eq("B").to_numpy(dtype=np.int64, na_value=False)
    progress = frame["inning"].fillna(-1).to_numpy(
        dtype=np.int64, copy=False
    ) * 2 + half
    runs = frame["run_total_before"].fillna(-1).to_numpy(
        dtype=np.int64, copy=False
    )
    if not len(frame):
        return np.empty(0, dtype=np.int64)
    boundary = np.concatenate([
        np.array([True]),
        np.any(key[1:] != key[:-1], axis=1)
        | (progress[1:] < progress[:-1])
        | (runs[1:] < runs[:-1]),
    ])
    return boundary.cumsum(dtype=np.int64) - 1


def _aggregate_table(rows: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    grouped = rows.groupby(keys, observed=True)["appearance_pitches"]
    result = grouped.agg(["count", "sum", "mean", "std"])
    quantiles = grouped.quantile([0.25, 0.5, 0.75]).unstack(-1)
    quantiles.columns = ["q25", "q50", "q75"]
    return result.join(quantiles).fillna(0.0)


def profile_features(
    samples: pd.DataFrame,
    history: pd.DataFrame,
    leave_one_out: bool,
) -> pd.DataFrame:
    by_type = _aggregate_table(history, ["pitcher_id", "game_type"])
    overall = _aggregate_table(history, ["pitcher_id"])
    global_mean = float(history["appearance_pitches"].mean())
    global_std = float(history["appearance_pitches"].std(ddof=0))
    type_key = pd.MultiIndex.from_arrays(
        [samples["pitcher_id"], samples["game_type"]],
        names=["pitcher_id", "game_type"],
    )
    typed = by_type.reindex(type_key).reset_index(drop=True)
    all_pitcher = overall.reindex(samples["pitcher_id"].to_numpy()).reset_index(
        drop=True
    )
    if leave_one_out:
        own = samples["appearance_pitches"].to_numpy(dtype=np.float64)
        for table in (typed, all_pitcher):
            count = table["count"].fillna(0.0).to_numpy(dtype=np.float64)
            total = table["sum"].fillna(0.0).to_numpy(dtype=np.float64)
            remaining = np.maximum(count - 1.0, 0.0)
            table["count"] = remaining
            table["mean"] = np.divide(
                total - own,
                remaining,
                out=np.full(len(samples), np.nan),
                where=remaining > 0.0,
            )
    output = pd.DataFrame(index=samples.index)
    for prefix, table in (("type", typed), ("all", all_pitcher)):
        for statistic in PROFILE_STATS:
            fallback = global_mean if statistic not in {"count", "std"} else 0.0
            if statistic == "std":
                fallback = global_std
            output[f"profile_{prefix}_{statistic}"] = table[statistic].fillna(
                fallback
            ).to_numpy(dtype=np.float32)
    output["profile_global_mean"] = np.float32(global_mean)
    output["profile_global_std"] = np.float32(global_std)
    return output


def model_features(
    rows: pd.DataFrame,
    profiles: pd.DataFrame,
) -> tuple[pd.DataFrame, list[str]]:
    values: dict[str, Any] = {
        "pitcher_id": rows["pitcher_id"].astype("string").fillna("__unknown__"),
        "game_type": rows["game_type"].astype("string").fillna("__unknown__"),
    }
    for horizon in HORIZONS:
        for suffix in ("success_rate", "middle_rate"):
            source = f"asof_pitcher_prev{horizon}_game_{suffix}"
            values[source] = pd.to_numeric(rows[source], errors="coerce").fillna(-1.0)
        for suffix in (
            "n_lower", "success_count_lower", "middle_count_lower",
            "boundary_ambiguous",
        ):
            source = f"e74_prev{horizon}_{suffix}"
            values[source] = pd.to_numeric(rows[source], errors="coerce").fillna(0.0)
    output = pd.DataFrame(values, index=rows.index)
    for column in profiles.columns:
        output[column] = profiles[column].to_numpy()
    return output, ["pitcher_id", "game_type"]


DECODER_FEATURE_COLUMNS = [
    column
    for horizon in HORIZONS
    for column in (
        f"e101_prev{horizon}_multiplier_mode",
        f"e101_prev{horizon}_n_mode",
        f"e101_prev{horizon}_log_n_mode",
        f"e101_prev{horizon}_p_multiplier1",
        f"e101_prev{horizon}_multiplier_entropy",
        f"e101_prev{horizon}_success_count_mode",
        f"e101_prev{horizon}_middle_count_mode",
        f"e101_prev{horizon}_success_posterior_k20",
        f"e101_prev{horizon}_reliability_k20",
    )
] + [
    "e101_nested_mode",
    "e101_prev23_n_mode",
    "e101_prev45_n_mode",
    "e101_prev23_success_rate",
    "e101_prev45_success_rate",
    "e101_prev23_middle_rate",
    "e101_prev45_middle_rate",
    "e101_prev23_valid",
    "e101_prev45_valid",
]


def reconstruct_appearances(
    history: pd.DataFrame,
    lower_builder: Callable[[pd.DataFrame], pd.DataFrame],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Create target-free appearance labels from completed official history."""
    work = history.copy()
    work["_decoder_game_id"] = derive_game_ids(work)
    work["_decoder_position"] = np.arange(len(work), dtype=np.int64)
    appearances = (
        work.groupby(
            ["_decoder_game_id", "pitcher_id"], sort=False, observed=True
        )
        .agg(
            first_position=("_decoder_position", "min"),
            appearance_pitches=("_decoder_position", "size"),
            season=("season", "first"),
            game_type=("game_type", "first"),
        )
        .reset_index()
        .sort_values("first_position", kind="mergesort")
        .reset_index(drop=True)
    )
    for horizon in HORIZONS:
        appearances[f"true_prev{horizon}_n"] = appearances.groupby(
            "pitcher_id", sort=False, observed=True
        )["appearance_pitches"].transform(
            lambda values, h=horizon: values.shift(1).rolling(
                h, min_periods=h
            ).sum()
        )
    representatives = work.iloc[
        appearances["first_position"].to_numpy(dtype=np.int64)
    ].reset_index(drop=True)
    lower = lower_builder(representatives).reset_index(drop=True)
    representatives = pd.concat([representatives, lower], axis=1)
    return appearances, representatives


def _fit_models(
    appearances: pd.DataFrame,
    representatives: pd.DataFrame,
    params: dict[str, Any],
    seed_base: int,
) -> tuple[dict[int, CatBoostClassifier], dict[str, Any]]:
    profiles = profile_features(appearances, appearances, True)
    train_x, categorical = model_features(representatives, profiles)
    models: dict[int, CatBoostClassifier] = {}
    metadata: dict[str, Any] = {}
    for horizon in HORIZONS:
        truth = appearances[f"true_prev{horizon}_n"].to_numpy(dtype=np.float64)
        lower = representatives[f"e74_prev{horizon}_n_lower"].to_numpy(
            dtype=np.float64
        )
        usable = np.isfinite(truth) & (lower > 0.0)
        multiplier = np.rint(truth[usable] / lower[usable]).astype(str)
        if int(usable.sum()) < 100 or len(np.unique(multiplier)) < 2:
            metadata[str(horizon)] = {
                "fit_rows": int(usable.sum()),
                "fallback_only": True,
            }
            continue
        model = CatBoostClassifier(
            loss_function=str(params["loss_function"]),
            iterations=int(params["iterations"]),
            depth=int(params["depth"]),
            learning_rate=float(params["learning_rate"]),
            l2_leaf_reg=float(params["l2_leaf_reg"]),
            random_seed=int(seed_base + horizon),
            random_strength=float(params["random_strength"]),
            bootstrap_type=str(params["bootstrap_type"]),
            bagging_temperature=float(params["bagging_temperature"]),
            allow_writing_files=False,
            verbose=False,
            thread_count=6,
            task_type=(
                "GPU"
                if os.environ.get("V2_BOOSTER_DEVICE", "cpu").lower() == "gpu"
                else "CPU"
            ),
        )
        model.fit(train_x.loc[usable], multiplier, cat_features=categorical)
        models[horizon] = model
        metadata[str(horizon)] = {
            "fit_rows": int(usable.sum()),
            "fallback_only": False,
            "classes": [
                int(round(float(value))) for value in np.asarray(model.classes_)
            ],
        }
    return models, metadata


def _predict_rows(
    rows: pd.DataFrame,
    profile_history: pd.DataFrame,
    models: dict[int, CatBoostClassifier],
    params: dict[str, Any],
    lower_builder: Callable[[pd.DataFrame], pd.DataFrame],
) -> pd.DataFrame:
    original_index = rows.index
    samples = rows.reset_index(drop=True).copy()
    lower_features = lower_builder(samples).reset_index(drop=True)
    augmented = pd.concat([samples, lower_features], axis=1)
    profiles = profile_features(samples, profile_history, False)
    features, _ = model_features(augmented, profiles)
    values: dict[str, np.ndarray] = {}
    mode_n: dict[int, np.ndarray] = {}
    success_count: dict[int, np.ndarray] = {}
    middle_count: dict[int, np.ndarray] = {}
    for horizon in HORIZONS:
        lower = augmented[f"e74_prev{horizon}_n_lower"].to_numpy(
            dtype=np.float64
        )
        model = models.get(horizon)
        if model is None:
            selected_multiplier = np.ones(len(rows), dtype=np.int32)
            probability_one = np.ones(len(rows), dtype=np.float64)
            entropy = np.zeros(len(rows), dtype=np.float64)
        else:
            probability = np.asarray(model.predict_proba(features), dtype=np.float64)
            classes = np.rint(
                np.asarray(model.classes_).astype(np.float64)
            ).astype(np.int32)
            selected_multiplier = classes[np.argmax(probability, axis=1)]
            one_index = np.flatnonzero(classes == 1)
            probability_one = (
                probability[:, one_index[0]]
                if len(one_index) else np.zeros(len(rows), dtype=np.float64)
            )
            entropy = -np.sum(
                probability * np.log(np.clip(probability, 1e-12, 1.0)), axis=1
            )
        maximum = int(params["maximum_denominator"][str(horizon)])
        maximum_multiplier = np.floor(
            maximum / np.maximum(lower, 1.0)
        ).astype(np.int32)
        selected_multiplier = np.clip(
            selected_multiplier, 1, np.maximum(maximum_multiplier, 1)
        )
        decoded = lower * selected_multiplier
        decoded[lower <= 0.0] = 0.0
        success_rate = pd.to_numeric(
            augmented[f"asof_pitcher_prev{horizon}_game_success_rate"],
            errors="coerce",
        ).fillna(0.5).to_numpy(dtype=np.float64)
        middle_rate = pd.to_numeric(
            augmented[f"asof_pitcher_prev{horizon}_game_middle_rate"],
            errors="coerce",
        ).fillna(0.0).to_numpy(dtype=np.float64)
        successes = np.rint(success_rate * decoded)
        middles = np.rint(middle_rate * decoded)
        mode_n[horizon] = decoded
        success_count[horizon] = successes
        middle_count[horizon] = middles
        values[f"e101_prev{horizon}_multiplier_mode"] = selected_multiplier
        values[f"e101_prev{horizon}_n_mode"] = decoded
        values[f"e101_prev{horizon}_log_n_mode"] = np.log1p(decoded)
        values[f"e101_prev{horizon}_p_multiplier1"] = probability_one
        values[f"e101_prev{horizon}_multiplier_entropy"] = entropy
        values[f"e101_prev{horizon}_success_count_mode"] = successes
        values[f"e101_prev{horizon}_middle_count_mode"] = middles
        values[f"e101_prev{horizon}_success_posterior_k20"] = (
            successes + 10.0
        ) / (decoded + 20.0)
        values[f"e101_prev{horizon}_reliability_k20"] = decoded / (
            decoded + 20.0
        )

    n23 = mode_n[3] - mode_n[1]
    n45 = mode_n[5] - mode_n[3]
    s23 = success_count[3] - success_count[1]
    s45 = success_count[5] - success_count[3]
    m23 = middle_count[3] - middle_count[1]
    m45 = middle_count[5] - middle_count[3]
    valid23 = (n23 >= 0.0) & (s23 >= 0.0) & (m23 >= 0.0) & (s23 + m23 <= n23)
    valid45 = (n45 >= 0.0) & (s45 >= 0.0) & (m45 >= 0.0) & (s45 + m45 <= n45)

    def rate(count: np.ndarray, denominator: np.ndarray, valid: np.ndarray) -> np.ndarray:
        return np.divide(
            count,
            denominator,
            out=np.full(len(rows), np.nan, dtype=np.float64),
            where=valid & (denominator > 0.0),
        )

    values.update({
        "e101_nested_mode": (valid23 & valid45).astype(np.int8),
        "e101_prev23_n_mode": np.where(valid23, n23, 0.0),
        "e101_prev45_n_mode": np.where(valid45, n45, 0.0),
        "e101_prev23_success_rate": rate(s23, n23, valid23),
        "e101_prev45_success_rate": rate(s45, n45, valid45),
        "e101_prev23_middle_rate": rate(m23, n23, valid23),
        "e101_prev45_middle_rate": rate(m45, n45, valid45),
        "e101_prev23_valid": valid23.astype(np.int8),
        "e101_prev45_valid": valid45.astype(np.int8),
    })
    output = pd.DataFrame(values, index=original_index)
    return output[DECODER_FEATURE_COLUMNS].astype(np.float32)


def build_recent_workload_decoder_fold_features(
    history_all: pd.DataFrame,
    train_rows: pd.DataFrame,
    valid_rows: pd.DataFrame,
    outer_year: int,
    preregistration: Path,
    lower_builder: Callable[[pd.DataFrame], pd.DataFrame],
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Build external-fold train features and frozen-history validation features."""
    prereg = json.loads(preregistration.read_text(encoding="utf-8"))
    params = prereg["model"]
    appearances, representatives = reconstruct_appearances(
        history_all, lower_builder
    )
    train_parts: list[pd.DataFrame] = []
    fit_metadata: dict[str, Any] = {}
    for heldout_year in sorted(int(value) for value in train_rows["season"].unique()):
        row_mask = train_rows["season"].eq(heldout_year)
        fit_mask = appearances["season"].ne(heldout_year)
        fit_appearances = appearances.loc[fit_mask].reset_index(drop=True)
        fit_representatives = representatives.loc[fit_mask].reset_index(drop=True)
        models, metadata = _fit_models(
            fit_appearances,
            fit_representatives,
            params,
            seed_base=int(params["random_seed"]) + outer_year * 100 + heldout_year * 10,
        )
        train_parts.append(
            _predict_rows(
                train_rows.loc[row_mask], fit_appearances, models, params,
                lower_builder,
            )
        )
        fit_metadata[f"train_oof_{heldout_year}"] = metadata
        del models
        gc.collect()
    train_features = pd.concat(train_parts, axis=0).reindex(train_rows.index)

    final_models, final_metadata = _fit_models(
        appearances,
        representatives,
        params,
        seed_base=int(params["random_seed"]) + outer_year * 10,
    )
    valid_features = _predict_rows(
        valid_rows, appearances, final_models, params, lower_builder
    )

    sample = valid_rows.iloc[: min(32, len(valid_rows))].copy()
    reference = _predict_rows(
        sample, appearances, final_models, params, lower_builder
    )
    order = np.arange(len(sample) - 1, -1, -1)
    shuffled = _predict_rows(
        sample.iloc[order], appearances, final_models, params, lower_builder
    )
    shuffled = shuffled.reindex(sample.index)
    difference = np.nanmax(
        np.abs(reference.to_numpy(dtype=np.float64) - shuffled.to_numpy(dtype=np.float64))
    ) if len(sample) else 0.0
    duplicated = pd.concat([sample, sample.iloc[[0]]], axis=0) if len(sample) else sample
    duplicate_features = _predict_rows(
        duplicated, appearances, final_models, params, lower_builder
    )
    if len(sample):
        difference = max(
            float(difference),
            float(np.nanmax(np.abs(
                duplicate_features.iloc[: len(sample)].to_numpy(dtype=np.float64)
                - reference.to_numpy(dtype=np.float64)
            ))),
        )
    metadata = {
        "enabled": True,
        "outer_year": int(outer_year),
        "history_appearances": int(len(appearances)),
        "train_external_fold_models": fit_metadata,
        "validation_models": final_metadata,
        "feature_columns": list(DECODER_FEATURE_COLUMNS),
        "row_order_duplicate_max_abs_difference": float(difference),
        "row_order_duplicate_invariance": bool(difference == 0.0),
        "control_success_used_by_decoder": False,
        "other_validation_rows_used": False,
    }
    del final_models
    gc.collect()
    return train_features, valid_features, metadata
