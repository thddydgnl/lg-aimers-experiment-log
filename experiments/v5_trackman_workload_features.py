#!/usr/bin/env python3
"""Target-free, season-frozen TrackMan appearance workload features."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


CELL_K = 100.0
APPEARANCE_K = 10.0
UNKNOWN_PITCHER = -1
FIXED_INNINGS = tuple(range(21))
WORKLOAD_PROFILE_COLUMNS = [
    "e100_profile_n_log",
    "e100_appearance_count_log",
    "e100_mean_appearance_pitches",
    "e100_starter_rate",
    "e100_cell_n_log",
    "e100_expected_appearance_pitch_index",
    "e100_prob_index_ge25",
    "e100_prob_index_ge50",
    "e100_prob_index_ge75",
    "e100_expected_progress",
    "e100_cell_unseen",
    "e100_profile_unseen",
]


def _empty_profile() -> pd.DataFrame:
    index = pd.MultiIndex.from_arrays(
        [np.array([], dtype=np.int64), np.array([], dtype=np.int16)],
        names=["pitcher_id", "inning"],
    )
    return pd.DataFrame(index=index, columns=WORKLOAD_PROFILE_COLUMNS, dtype=np.float32)


def _appearance_rows(rows: pd.DataFrame) -> pd.DataFrame:
    """Return regular-season pitches with a deterministic appearance index."""
    required = {
        "game_type", "trackman_game_id", "pitch_no", "pitcher_id", "inning"
    }
    missing = sorted(required.difference(rows.columns))
    if missing:
        raise ValueError(f"TrackMan workload source is missing columns: {missing}")
    regular = rows.loc[
        rows["game_type"].eq("R"),
        ["trackman_game_id", "pitch_no", "pitcher_id", "inning"],
    ].dropna().copy()
    if regular.empty:
        return regular.assign(appearance_pitch_index=np.array([], dtype=np.int16))
    regular["pitch_no"] = pd.to_numeric(
        regular["pitch_no"], errors="raise"
    ).astype(np.int32)
    regular["pitcher_id"] = pd.to_numeric(
        regular["pitcher_id"], errors="raise"
    ).astype(np.int64)
    if (regular["pitcher_id"] == UNKNOWN_PITCHER).any():
        raise ValueError(f"reserved pitcher id {UNKNOWN_PITCHER} is present")
    regular["inning"] = pd.to_numeric(
        regular["inning"], errors="raise"
    ).round().astype(np.int16)
    if regular.duplicated(["trackman_game_id", "pitch_no"]).any():
        raise ValueError("TrackMan workload source has duplicate game/pitch keys")
    regular = regular.sort_values(
        ["trackman_game_id", "pitch_no"], kind="mergesort"
    )
    regular["appearance_pitch_index"] = (
        regular.groupby(
            ["trackman_game_id", "pitcher_id"], sort=False, observed=True
        ).cumcount() + 1
    ).astype(np.int16)
    return regular


def workload_profile_table(rows: pd.DataFrame) -> pd.DataFrame:
    """Build pitcher-by-inning workload states from completed TrackMan rows."""
    regular = _appearance_rows(rows)
    if regular.empty:
        return _empty_profile()

    appearance_key = ["trackman_game_id", "pitcher_id"]
    appearances = regular.groupby(
        appearance_key, sort=False, observed=True
    ).agg(
        appearance_pitches=("pitch_no", "size"),
        first_inning=("inning", "min"),
    ).reset_index()
    appearances["starter"] = appearances["first_inning"].eq(1).astype(np.int8)

    pitcher_appearances = appearances.groupby(
        "pitcher_id", sort=False, observed=True
    ).agg(
        appearance_count=("appearance_pitches", "size"),
        appearance_pitch_sum=("appearance_pitches", "sum"),
        starter_sum=("starter", "sum"),
    )
    pitcher_pitch_n = regular.groupby(
        "pitcher_id", sort=False, observed=True
    ).size().astype(np.float64)

    global_mean_appearance = float(appearances["appearance_pitches"].mean())
    global_starter_rate = float(appearances["starter"].mean())
    global_index_mean = float(regular["appearance_pitch_index"].mean())
    global_tail = {
        threshold: float(regular["appearance_pitch_index"].ge(threshold).mean())
        for threshold in (25, 50, 75)
    }

    work = regular.assign(
        ge25=regular["appearance_pitch_index"].ge(25).astype(np.int8),
        ge50=regular["appearance_pitch_index"].ge(50).astype(np.int8),
        ge75=regular["appearance_pitch_index"].ge(75).astype(np.int8),
    )
    cell = work.groupby(
        ["pitcher_id", "inning"], sort=False, observed=True
    ).agg(
        cell_n=("appearance_pitch_index", "size"),
        index_sum=("appearance_pitch_index", "sum"),
        ge25_sum=("ge25", "sum"),
        ge50_sum=("ge50", "sum"),
        ge75_sum=("ge75", "sum"),
    )
    league_inning = work.groupby("inning", sort=False, observed=True).agg(
        cell_n=("appearance_pitch_index", "size"),
        index_mean=("appearance_pitch_index", "mean"),
        ge25_rate=("ge25", "mean"),
        ge50_rate=("ge50", "mean"),
        ge75_rate=("ge75", "mean"),
    )

    pitcher_ids = np.sort(regular["pitcher_id"].unique().astype(np.int64))
    innings = np.array(
        sorted(set(FIXED_INNINGS).union(int(x) for x in regular["inning"].unique())),
        dtype=np.int16,
    )
    index = pd.MultiIndex.from_product(
        [pitcher_ids, innings], names=["pitcher_id", "inning"]
    )
    output = pd.DataFrame(index=index)
    index_pitchers = index.get_level_values("pitcher_id")
    index_innings = index.get_level_values("inning")

    profile_n = pitcher_pitch_n.reindex(index_pitchers).to_numpy(dtype=np.float64)
    appearance_count = pitcher_appearances["appearance_count"].reindex(
        index_pitchers
    ).to_numpy(dtype=np.float64)
    appearance_pitch_sum = pitcher_appearances[
        "appearance_pitch_sum"
    ].reindex(index_pitchers).to_numpy(dtype=np.float64)
    starter_sum = pitcher_appearances["starter_sum"].reindex(
        index_pitchers
    ).to_numpy(dtype=np.float64)
    mean_appearance = (
        appearance_pitch_sum + APPEARANCE_K * global_mean_appearance
    ) / (appearance_count + APPEARANCE_K)
    starter_rate = (
        starter_sum + APPEARANCE_K * global_starter_rate
    ) / (appearance_count + APPEARANCE_K)

    cell_lookup = cell.reindex(index)
    cell_n = cell_lookup["cell_n"].fillna(0.0).to_numpy(dtype=np.float64)
    inning_prior = league_inning.reindex(index_innings)
    prior_index = inning_prior["index_mean"].fillna(
        global_index_mean
    ).to_numpy(dtype=np.float64)
    prior_tail = {
        threshold: inning_prior[f"ge{threshold}_rate"].fillna(
            global_tail[threshold]
        ).to_numpy(dtype=np.float64)
        for threshold in (25, 50, 75)
    }
    expected_index = (
        cell_lookup["index_sum"].fillna(0.0).to_numpy(dtype=np.float64)
        + CELL_K * prior_index
    ) / (cell_n + CELL_K)
    tail_rates = {
        threshold: (
            cell_lookup[f"ge{threshold}_sum"].fillna(0.0).to_numpy(
                dtype=np.float64
            ) + CELL_K * prior_tail[threshold]
        ) / (cell_n + CELL_K)
        for threshold in (25, 50, 75)
    }

    output["e100_profile_n_log"] = np.log1p(profile_n).astype(np.float32)
    output["e100_appearance_count_log"] = np.log1p(
        appearance_count
    ).astype(np.float32)
    output["e100_mean_appearance_pitches"] = mean_appearance.astype(np.float32)
    output["e100_starter_rate"] = starter_rate.astype(np.float32)
    output["e100_cell_n_log"] = np.log1p(cell_n).astype(np.float32)
    output["e100_expected_appearance_pitch_index"] = expected_index.astype(
        np.float32
    )
    for threshold in (25, 50, 75):
        output[f"e100_prob_index_ge{threshold}"] = tail_rates[threshold].astype(
            np.float32
        )
    output["e100_expected_progress"] = (
        expected_index / np.maximum(mean_appearance, 1.0)
    ).astype(np.float32)
    output["e100_cell_unseen"] = (cell_n <= 0).astype(np.int8)
    output["e100_profile_unseen"] = 0

    sentinel_index = pd.MultiIndex.from_product(
        [[UNKNOWN_PITCHER], innings], names=["pitcher_id", "inning"]
    )
    sentinel = pd.DataFrame(index=sentinel_index)
    sentinel_innings = sentinel_index.get_level_values("inning")
    sentinel_prior = league_inning.reindex(sentinel_innings)
    sentinel_expected_index = sentinel_prior["index_mean"].fillna(
        global_index_mean
    ).to_numpy(dtype=np.float64)
    sentinel["e100_profile_n_log"] = 0.0
    sentinel["e100_appearance_count_log"] = 0.0
    sentinel["e100_mean_appearance_pitches"] = np.float32(
        global_mean_appearance
    )
    sentinel["e100_starter_rate"] = np.float32(global_starter_rate)
    sentinel["e100_cell_n_log"] = 0.0
    sentinel["e100_expected_appearance_pitch_index"] = (
        sentinel_expected_index.astype(np.float32)
    )
    for threshold in (25, 50, 75):
        sentinel[f"e100_prob_index_ge{threshold}"] = sentinel_prior[
            f"ge{threshold}_rate"
        ].fillna(global_tail[threshold]).to_numpy(dtype=np.float32)
    sentinel["e100_expected_progress"] = (
        sentinel_expected_index / max(global_mean_appearance, 1.0)
    ).astype(np.float32)
    sentinel["e100_cell_unseen"] = 1
    sentinel["e100_profile_unseen"] = 1
    return pd.concat([output, sentinel], axis=0)[WORKLOAD_PROFILE_COLUMNS]


def workload_profile_states_before_each_season(
    joined: pd.DataFrame,
    seasons: list[int],
    window: int | None = None,
) -> tuple[dict[int, pd.DataFrame], pd.DataFrame]:
    """Create strictly pre-season states plus the final validation state."""
    if window is not None and window < 1:
        raise ValueError("TrackMan workload window must be >= 1")
    before: dict[int, pd.DataFrame] = {}
    for season in sorted(seasons):
        rows = joined.loc[joined["season"] < season]
        if window is not None:
            rows = rows.loc[rows["season"] >= season - window]
        before[season] = workload_profile_table(rows)
    if seasons:
        cutoff = max(seasons) + 1
        rows = joined.loc[joined["season"] < cutoff]
        if window is not None:
            rows = rows.loc[rows["season"] >= cutoff - window]
        final = workload_profile_table(rows)
    else:
        final = _empty_profile()
    return before, final


def build_workload_profile_features(
    frame: pd.DataFrame,
    profiles_before: dict[int, pd.DataFrame],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Look up frozen workload features without aggregating prediction rows."""
    values = np.zeros(
        (len(frame), len(WORKLOAD_PROFILE_COLUMNS)), dtype=np.float32
    )
    cell_unseen_index = WORKLOAD_PROFILE_COLUMNS.index("e100_cell_unseen")
    profile_unseen_index = WORKLOAD_PROFILE_COLUMNS.index("e100_profile_unseen")
    values[:, cell_unseen_index] = 1.0
    values[:, profile_unseen_index] = 1.0
    seasons = pd.to_numeric(frame["season"], errors="raise").to_numpy(
        dtype=np.int16, copy=False
    )
    pitchers = pd.to_numeric(frame["pitcher_id"], errors="raise").to_numpy(
        dtype=np.int64, copy=False
    )
    innings = pd.to_numeric(frame["inning"], errors="coerce").fillna(0).round().to_numpy(
        dtype=np.int16, copy=False
    )
    for season in sorted(set(int(value) for value in seasons)):
        mask = seasons == season
        profile = profiles_before.get(season)
        if profile is None or profile.empty:
            continue
        key = pd.MultiIndex.from_arrays(
            [pitchers[mask], innings[mask]], names=["pitcher_id", "inning"]
        )
        exact = profile.reindex(key)
        fallback_key = pd.MultiIndex.from_arrays(
            [
                np.full(int(mask.sum()), UNKNOWN_PITCHER, dtype=np.int64),
                innings[mask],
            ],
            names=["pitcher_id", "inning"],
        )
        fallback = profile.reindex(fallback_key)
        fallback.index = exact.index
        missing = exact["e100_profile_unseen"].isna()
        selected = exact.copy()
        selected.loc[missing, WORKLOAD_PROFILE_COLUMNS] = fallback.loc[
            missing, WORKLOAD_PROFILE_COLUMNS
        ].to_numpy()
        matrix = selected[WORKLOAD_PROFILE_COLUMNS].to_numpy(dtype=np.float32)
        unresolved = np.isnan(matrix[:, profile_unseen_index])
        if unresolved.any():
            matrix[unresolved] = 0.0
            matrix[unresolved, cell_unseen_index] = 1.0
            matrix[unresolved, profile_unseen_index] = 1.0
        values[np.flatnonzero(mask)] = matrix
    result = pd.DataFrame(
        values, columns=WORKLOAD_PROFILE_COLUMNS, index=frame.index
    )
    return result, {
        "unseen_rows": int((result["e100_profile_unseen"] > 0).sum()),
        "known_rows": int((result["e100_profile_unseen"] == 0).sum()),
        "unseen_cells": int((result["e100_cell_unseen"] > 0).sum()),
        "feature_count": len(WORKLOAD_PROFILE_COLUMNS),
        "cell_k": CELL_K,
        "appearance_k": APPEARANCE_K,
        "cutoff": "linked TrackMan R rows with season strictly before row season",
        "target_free": True,
        "row_independent": True,
    }
