#!/usr/bin/env python3
"""Deploy the source-locked pitcher/count/hand fine-pitch selector."""

from __future__ import annotations

import gc
import time
from typing import Any

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier

from experiments.v5_expanded_trackman_profiles import (
    build_expanded_trackman_profile_source,
)


FINE_TYPES = (
    "Fastball",
    "Slider",
    "Curveball",
    "ChangeUp",
    "Splitter",
    "Sinker",
    "Cutter",
    "Other",
)
PROBABILITY_COLUMNS = [f"e92_p_{value.lower()}" for value in FINE_TYPES]


def _normalize(values: pd.Series) -> pd.Series:
    normalized = values.astype("string").replace(
        {"Changeup": "ChangeUp", "Four-Seam": "Fastball", "SInker": "Sinker"}
    )
    return normalized.where(normalized.isin(FINE_TYPES[:-1]), "Other")


def _count_table(rows: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    table = rows.groupby(
        [*keys, "_fine_auto"], sort=False, observed=True, dropna=False
    ).size().unstack("_fine_auto", fill_value=0)
    return table.reindex(columns=FINE_TYPES, fill_value=0).astype(np.float64)


def _mapped_counts(
    table: pd.DataFrame, query: pd.DataFrame, keys: list[str]
) -> np.ndarray:
    if len(keys) == 1:
        index = pd.Index(query[keys[0]].to_numpy(), name=table.index.name)
    else:
        index = pd.MultiIndex.from_frame(query[keys])
        index.names = table.index.names
    values = table.reindex(index).to_numpy(dtype=np.float64)
    missing = np.isnan(values).all(axis=1)
    values[missing] = 0.0
    return values


def _smooth(counts: np.ndarray, prior: np.ndarray, k: float) -> np.ndarray:
    total = counts.sum(axis=1)
    return (counts + k * prior) / (total[:, None] + k)


def _geometric(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    result = np.exp(
        0.5 * np.log(np.clip(left, 1e-12, 1.0))
        + 0.5 * np.log(np.clip(right, 1e-12, 1.0))
    )
    return result / result.sum(axis=1, keepdims=True)


def _metric(probability: np.ndarray, truth: pd.Series) -> dict[str, float]:
    truth_index = np.asarray(
        [FINE_TYPES.index(value) for value in truth.astype(str)], dtype=np.int16
    )
    chosen = probability[np.arange(len(truth_index)), truth_index]
    return {
        "log_loss": float(-np.mean(np.log(np.clip(chosen, 1e-12, 1.0)))),
        "top1_accuracy": float(
            np.mean(probability.argmax(axis=1) == truth_index)
        ),
    }


def _output_frame(probability: np.ndarray, index: pd.Index) -> pd.DataFrame:
    result = pd.DataFrame(
        probability.astype(np.float32), columns=PROBABILITY_COLUMNS, index=index
    )
    result["e92_entropy"] = -np.sum(
        probability * np.log(np.clip(probability, 1e-12, 1.0)), axis=1
    ).astype(np.float32)
    result["e92_max_probability"] = probability.max(axis=1).astype(np.float32)
    return result


def build_locked_matchup_hand_probabilities(
    history: pd.DataFrame,
    valid: pd.DataFrame,
    joined_trackman: pd.DataFrame,
    raw_trackman: pd.DataFrame,
    base_features: list[str],
    categorical_features: list[str],
    random_seed: int,
    use_gpu: bool,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Build the frozen selector; training e92 columns are inert by contract.

    The eight-expert MoE removes every e92 column from each expert's training
    matrix.  We therefore expose a constant, target-free prior on training rows
    and fit the source-locked R-history selector only for outer validation.
    This prevents in-sample auxiliary labels from entering the control experts.
    """
    if set(FINE_TYPES) != set(history["auto_fine_pitch_type"].dropna().unique()):
        missing = sorted(
            set(FINE_TYPES)
            - set(history["auto_fine_pitch_type"].dropna().astype(str).unique())
        )
        if missing:
            raise ValueError(f"matchup selector history lacks pitch classes: {missing}")
    years = pd.unique(valid["season"])
    if len(years) != 1:
        raise ValueError("matchup selector requires one outer validation season")
    target_year = int(years[0])
    history_r = history.loc[history["game_type"].eq("R")].copy()
    labeled = history_r.loc[history_r["auto_fine_pitch_type"].notna()].copy()
    if len(labeled) < 1000:
        raise ValueError("too few R-history fine-pitch labels")

    categorical = [
        column for column in categorical_features if column in base_features
    ]

    def prepare(frame: pd.DataFrame) -> pd.DataFrame:
        result = frame[base_features].copy()
        for column in categorical:
            result[column] = (
                result[column].astype("string").fillna("__missing__").astype(str)
            )
        return result

    model = CatBoostClassifier(
        loss_function="MultiClass",
        iterations=400,
        depth=6,
        learning_rate=0.06,
        l2_leaf_reg=20.0,
        random_seed=random_seed + target_year + 100,
        allow_writing_files=False,
        thread_count=6,
        task_type="GPU" if use_gpu else "CPU",
    )
    started = time.perf_counter()
    model.fit(
        prepare(labeled),
        labeled["auto_fine_pitch_type"].astype(str),
        cat_features=categorical,
        verbose=False,
    )

    def aligned(frame: pd.DataFrame) -> np.ndarray:
        raw = np.asarray(model.predict_proba(prepare(frame)), dtype=np.float64)
        result = np.zeros((len(frame), len(FINE_TYPES)), dtype=np.float64)
        classes = [str(value) for value in model.classes_]
        for source_index, label in enumerate(classes):
            if label in FINE_TYPES:
                result[:, FINE_TYPES.index(label)] = raw[:, source_index]
        denominator = result.sum(axis=1)
        invalid = denominator <= 0.0
        result[invalid] = 1.0 / len(FINE_TYPES)
        denominator[invalid] = 1.0
        return result / denominator[:, None]

    # Predict R and F separately so the locked R diagnostic is reproduced on
    # exactly the same query slice used by the source selector experiment.
    valid_r_mask = valid["game_type"].eq("R").to_numpy(dtype=bool)
    baseline = np.empty((len(valid), len(FINE_TYPES)), dtype=np.float64)
    baseline[valid_r_mask] = aligned(valid.loc[valid_r_mask])
    if (~valid_r_mask).any():
        baseline[~valid_r_mask] = aligned(valid.loc[~valid_r_mask])

    allowed_seasons = sorted(int(value) for value in history_r["season"].unique())
    major, expansion_meta = build_expanded_trackman_profile_source(
        joined_trackman, raw_trackman, allowed_seasons, 0.99
    )
    major["_fine_auto"] = _normalize(major["auto_pitch_type"])
    global_counts = (
        major["_fine_auto"].value_counts().reindex(FINE_TYPES).fillna(0.0)
        .to_numpy(dtype=np.float64)
    )
    if global_counts.sum() <= 0.0:
        raise ValueError("empty expanded matchup pitch profile")
    global_prior = global_counts / global_counts.sum()
    global_matrix = np.broadcast_to(
        global_prior, (len(valid), len(FINE_TYPES))
    )
    pitcher = _smooth(
        _mapped_counts(_count_table(major, ["pitcher_id"]), valid, ["pitcher_id"]),
        global_matrix,
        100.0,
    )
    pitcher_count = _smooth(
        _mapped_counts(
            _count_table(major, ["pitcher_id", "balls_before", "strikes_before"]),
            valid,
            ["pitcher_id", "balls_before", "strikes_before"],
        ),
        pitcher,
        20.0,
    )
    pitcher_hand = _smooth(
        _mapped_counts(
            _count_table(major, ["pitcher_id", "batter_hand"]),
            valid,
            ["pitcher_id", "batter_hand"],
        ),
        pitcher,
        40.0,
    )
    pitcher_hand_count = _smooth(
        _mapped_counts(
            _count_table(
                major,
                [
                    "pitcher_id", "batter_hand", "balls_before", "strikes_before"
                ],
            ),
            valid,
            ["pitcher_id", "batter_hand", "balls_before", "strikes_before"],
        ),
        pitcher_hand,
        20.0,
    )
    profile = _geometric(pitcher_count, pitcher_hand_count)
    candidate = _geometric(baseline, profile)

    matched_r = valid_r_mask & valid["auto_fine_pitch_type"].isin(
        FINE_TYPES
    ).to_numpy(dtype=bool)
    baseline_metric = _metric(
        baseline[matched_r], valid.loc[matched_r, "auto_fine_pitch_type"]
    )
    candidate_metric = _metric(
        candidate[matched_r], valid.loc[matched_r, "auto_fine_pitch_type"]
    )
    train_probability = np.broadcast_to(
        global_prior, (len(history), len(FINE_TYPES))
    ).copy()
    metadata = {
        "enabled": True,
        "architecture": "locked_geometric_pitcher_count_and_pitcher_hand_count_selector",
        "candidate_id": "geometric_count_hand__cb0.5",
        "allowed_history_seasons": allowed_seasons,
        "history_game_type": "R",
        "history_labeled_rows": int(len(labeled)),
        "valid_R_matched_rows_not_used": int(matched_r.sum()),
        "selector_baseline_R": baseline_metric,
        "selector_candidate_R": candidate_metric,
        "selector_log_loss_improvement_R": float(
            baseline_metric["log_loss"] - candidate_metric["log_loss"]
        ),
        "selector_top1_improvement_R": float(
            candidate_metric["top1_accuracy"] - baseline_metric["top1_accuracy"]
        ),
        "selector_fit_seconds": float(time.perf_counter() - started),
        "profile": {
            "pitcher_k": 100.0,
            "pitcher_count_k": 20.0,
            "pitcher_hand_k": 40.0,
            "pitcher_hand_count_k": 20.0,
            "catboost_geometric_weight": 0.5,
            "selected_batter_identity_used": False,
            "batter_hand_used": True,
        },
        "expanded_trackman": expansion_meta,
        "training_e92_values": "constant official-history global pitch prior",
        "training_e92_consumed_by_control_experts": False,
        "current_pitch_type_at_inference": False,
        "current_pitch_trackman_at_inference": False,
        "row_independent": True,
        "feature_columns": [
            *PROBABILITY_COLUMNS, "e92_entropy", "e92_max_probability"
        ],
    }
    del model, major, labeled, history_r, baseline, profile
    gc.collect()
    return (
        _output_frame(train_probability, history.index),
        _output_frame(candidate, valid.index),
        metadata,
    )
