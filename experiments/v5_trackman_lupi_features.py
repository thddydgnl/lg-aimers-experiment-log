#!/usr/bin/env python3
"""Cross-season imputation of privileged TrackMan attributes.

The auxiliary models may observe historical TrackMan labels while fitting, but
their inputs and every deployable feature contain only official pre-pitch row
fields.  A row from season S is always scored by auxiliary models fitted on
matched regular-season rows with season strictly less than S.
"""

from __future__ import annotations

import gc
import os
import time
from typing import Any

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, CatBoostRegressor

from experiments.run_baselines import FEATURES as BASE_FEATURES
from experiments.run_baselines import LINEAR_CATEGORICAL


SEASON = "season"
PITCH_GROUPS = ("fastball", "breaking", "offspeed", "other")
PHYSICS = (
    "rel_speed",
    "spin_rate",
    "induced_vert_break",
    "horz_break",
    "extension",
    "rel_height",
    "rel_side",
    "zone_speed",
)
CATEGORICAL = tuple(name for name in LINEAR_CATEGORICAL if name in BASE_FEATURES)
RAW_FEATURES = (
    *(f"e81_lupi_p_{group}" for group in PITCH_GROUPS),
    "e81_lupi_pitchmix_entropy",
    *(f"e81_lupi_pred_{name}" for name in PHYSICS),
    "e81_lupi_available",
)
DELTA_FEATURES = (
    *(f"e81_lupi_{name}_minus_profile" for name in PHYSICS),
    *(f"e81_lupi_p_{group}_minus_profile" for group in PITCH_GROUPS),
)
AUXILIARY_PARAMS = {
    "iterations": 250,
    "depth": 6,
    "learning_rate": 0.05,
    "l2_leaf_reg": 20.0,
    "random_strength": 0.5,
    "random_seed": 20260821,
    "border_count": 64,
}


def _prepare_input(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.loc[:, BASE_FEATURES].copy()
    for name in CATEGORICAL:
        result[name] = (
            result[name]
            .astype("string")
            .fillna("__MISSING__")
            .astype(object)
        )
    for name in result.columns:
        if name not in CATEGORICAL:
            result[name] = pd.to_numeric(result[name], errors="coerce").astype(
                np.float32
            )
    return result


def _model_common(iterations: int) -> dict[str, Any]:
    device = os.environ.get("V2_BOOSTER_DEVICE", "cpu").lower()
    return {
        **AUXILIARY_PARAMS,
        "iterations": int(iterations),
        "allow_writing_files": False,
        "verbose": False,
        "thread_count": min(6, os.cpu_count() or 1),
        "task_type": "GPU" if device == "gpu" else "CPU",
    }


def _empty(index: pd.Index) -> pd.DataFrame:
    result = pd.DataFrame(np.nan, index=index, columns=RAW_FEATURES, dtype=np.float32)
    result["e81_lupi_available"] = np.zeros(len(index), dtype=np.int8)
    return result


def _assign_predictions(
    output: pd.DataFrame,
    query: pd.DataFrame,
    classifier: CatBoostClassifier,
    regressor: CatBoostRegressor,
    physics_mean: np.ndarray,
    physics_scale: np.ndarray,
) -> int:
    route = query["game_type"].eq("R")
    selected = query.loc[route]
    if selected.empty:
        return 0
    query_x = _prepare_input(selected)
    probabilities = np.asarray(classifier.predict_proba(query_x), dtype=np.float64)
    class_index = {str(value): index for index, value in enumerate(classifier.classes_)}
    aligned = np.zeros((len(selected), len(PITCH_GROUPS)), dtype=np.float64)
    for index, group in enumerate(PITCH_GROUPS):
        if group in class_index:
            aligned[:, index] = probabilities[:, class_index[group]]
    total = aligned.sum(axis=1, keepdims=True)
    aligned = np.divide(
        aligned,
        total,
        out=np.full_like(aligned, 1.0 / len(PITCH_GROUPS)),
        where=total > 0.0,
    )
    standardized_physics = np.asarray(regressor.predict(query_x), dtype=np.float64)
    predicted_physics = standardized_physics * physics_scale + physics_mean
    entropy = -np.sum(
        np.where(aligned > 0.0, aligned * np.log(np.maximum(aligned, 1e-12)), 0.0),
        axis=1,
    )
    values: dict[str, np.ndarray] = {
        **{
            f"e81_lupi_p_{group}": aligned[:, index]
            for index, group in enumerate(PITCH_GROUPS)
        },
        "e81_lupi_pitchmix_entropy": entropy,
        **{
            f"e81_lupi_pred_{name}": predicted_physics[:, index]
            for index, name in enumerate(PHYSICS)
        },
        "e81_lupi_available": np.ones(len(selected), dtype=np.int8),
    }
    output.loc[selected.index, list(values)] = pd.DataFrame(
        {name: value.astype(np.float32) for name, value in values.items()},
        index=selected.index,
    )
    del query_x, probabilities, aligned, standardized_physics, predicted_physics
    return int(len(selected))


def build_cross_season_lupi_features(
    history: pd.DataFrame,
    valid: pd.DataFrame,
    joined_trackman: pd.DataFrame,
    *,
    smoke_source_rows: int | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Return strict out-of-time auxiliary predictions for one outer fold."""
    started = time.perf_counter()
    train_output = _empty(history.index)
    valid_output = _empty(valid.index)
    matched = joined_trackman.loc[
        joined_trackman["game_type"].eq("R")
        & joined_trackman["pitch_type_group"].isin(PITCH_GROUPS)
    ]
    fold_details: dict[str, Any] = {}
    query_seasons = sorted(
        set(int(value) for value in history[SEASON].unique())
        | set(int(value) for value in valid[SEASON].unique())
    )
    iterations = 20 if smoke_source_rows is not None else AUXILIARY_PARAMS["iterations"]
    for query_season in query_seasons:
        source = matched.loc[matched[SEASON].lt(query_season)]
        if smoke_source_rows is not None and len(source) > smoke_source_rows:
            source = source.sample(
                n=smoke_source_rows, random_state=20260821 + query_season
            ).sort_index()
        query_parts = [
            ("train", history.loc[history[SEASON].eq(query_season)], train_output),
            ("valid", valid.loc[valid[SEASON].eq(query_season)], valid_output),
        ]
        requested_r_rows = int(
            sum(part["game_type"].eq("R").sum() for _, part, _ in query_parts)
        )
        if source.empty or requested_r_rows == 0:
            fold_details[str(query_season)] = {
                "source_seasons": [],
                "source_rows": int(len(source)),
                "physics_complete_rows": 0,
                "query_r_rows": requested_r_rows,
                "scored_r_rows": 0,
            }
            continue

        source_x = _prepare_input(source)
        classifier = CatBoostClassifier(
            loss_function="MultiClass",
            **_model_common(iterations),
        )
        classifier.fit(
            source_x,
            source["pitch_type_group"].astype(str),
            cat_features=list(CATEGORICAL),
        )

        physics_frame = source.loc[:, PHYSICS].apply(pd.to_numeric, errors="coerce")
        complete = physics_frame.notna().all(axis=1).to_numpy()
        physics_frame = physics_frame.loc[complete]
        physics_mean = physics_frame.mean(axis=0).to_numpy(dtype=np.float64)
        physics_scale = physics_frame.std(axis=0, ddof=0).to_numpy(dtype=np.float64)
        physics_scale = np.maximum(physics_scale, 1e-6)
        standardized = (
            physics_frame.to_numpy(dtype=np.float64) - physics_mean
        ) / physics_scale
        regressor = CatBoostRegressor(
            loss_function="MultiRMSE",
            **_model_common(iterations),
        )
        regressor.fit(
            source_x.loc[physics_frame.index],
            standardized,
            cat_features=list(CATEGORICAL),
        )

        scored = 0
        for _, query, output in query_parts:
            scored += _assign_predictions(
                output,
                query,
                classifier,
                regressor,
                physics_mean,
                physics_scale,
            )
        fold_details[str(query_season)] = {
            "source_seasons": sorted(int(value) for value in source[SEASON].unique()),
            "source_rows": int(len(source)),
            "physics_complete_rows": int(len(physics_frame)),
            "query_r_rows": requested_r_rows,
            "scored_r_rows": int(scored),
        }
        del (
            source_x,
            classifier,
            regressor,
            physics_frame,
            standardized,
            source,
        )
        gc.collect()

    metadata = {
        "enabled": True,
        "method": "two auxiliary CatBoost models: pitch-group MultiClass and standardized-physics MultiRMSE",
        "input_features": list(BASE_FEATURES),
        "categorical_features": list(CATEGORICAL),
        "physics_targets": list(PHYSICS),
        "pitch_groups": list(PITCH_GROUPS),
        "model_params": {**AUXILIARY_PARAMS, "effective_iterations": iterations},
        "source_rule": "matched official R rows with season strictly before query row season",
        "main_target_used_by_auxiliary_models": False,
        "current_row_trackman_used_at_inference": False,
        "row_independent_inference": True,
        "smoke_source_rows": smoke_source_rows,
        "folds": fold_details,
        "elapsed_seconds": float(time.perf_counter() - started),
    }
    return train_output, valid_output, metadata


def add_profile_deltas(
    lupi: pd.DataFrame,
    rich_profile: pd.DataFrame,
) -> pd.DataFrame:
    """Contrast predicted current-pitch attributes with frozen pitcher profiles."""
    if "e58_profile_unseen" not in rich_profile:
        raise ValueError("TrackMan LUPI requires the rich TrackMan profile bundle")
    result = lupi.copy()
    available = (
        result["e81_lupi_available"].eq(1)
        & pd.to_numeric(rich_profile["e58_profile_unseen"], errors="coerce").eq(0)
    )
    for name in PHYSICS:
        profile_name = f"e58_{name}_mean"
        result[f"e81_lupi_{name}_minus_profile"] = (
            result[f"e81_lupi_pred_{name}"]
            - pd.to_numeric(rich_profile[profile_name], errors="coerce")
        ).where(available).astype(np.float32)
    for group in PITCH_GROUPS:
        profile_name = f"e58_{group}_rate"
        result[f"e81_lupi_p_{group}_minus_profile"] = (
            result[f"e81_lupi_p_{group}"]
            - pd.to_numeric(rich_profile[profile_name], errors="coerce")
        ).where(available).astype(np.float32)
    return result
