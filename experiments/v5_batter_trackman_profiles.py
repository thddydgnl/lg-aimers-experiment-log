#!/usr/bin/env python3
"""Cutoff-safe batter approach profiles from the official TrackMan history.

Exact game linkage is used only to recover anonymous batter identities from
completed seasons.  The actual feature values then aggregate the official raw
TrackMan history strictly before each query season.  No current-pitch or
validation-season TrackMan value is exposed to a row.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from experiments.v5_expanded_trackman_profiles import _best_map


METRICS = (
    "rel_speed",
    "spin_rate",
    "induced_vert_break",
    "horz_break",
    "extension",
    "rel_height",
    "rel_side",
    "zone_speed",
)
GROUPS = ("fastball", "breaking", "offspeed", "other")
HAND_K = 80.0
COUNT_K = 120.0

OVERALL_COLUMNS = (
    "e101_batter_tm_n_log",
    *(f"e101_{group}_rate" for group in GROUPS),
    *(f"e101_{metric}_mean" for metric in METRICS),
    *(f"e101_{metric}_sd" for metric in METRICS),
    "e101_batter_tm_unseen",
)
HAND_COLUMNS = (
    "e102_hand_n_log",
    *(f"e102_{group}_rate_delta" for group in GROUPS),
    *(f"e102_{metric}_mean_delta" for metric in METRICS),
    "e102_hand_unseen",
)
COUNT_COLUMNS = (
    "e103_count_n_log",
    *(f"e103_{group}_rate_delta" for group in GROUPS),
    "e103_count_unseen",
)
FEATURE_COLUMNS = (*OVERALL_COLUMNS, *HAND_COLUMNS, *COUNT_COLUMNS)


def _empty_state(cutoff: int) -> dict[str, Any]:
    return {
        "cutoff": int(cutoff),
        "overall": pd.DataFrame(),
        "hand": pd.DataFrame(),
        "count": pd.DataFrame(),
        "metadata": {
            "cutoff": int(cutoff),
            "mapped_batters": 0,
            "profile_rows": 0,
            "identity_minimum_purity": None,
            "history_seasons": [],
        },
    }


def _mode_by_entity(frame: pd.DataFrame, entity: str, value: str) -> pd.Series:
    return (
        frame.dropna(subset=[entity, value])
        .groupby([entity, value], observed=True, sort=False)
        .size()
        .rename("n")
        .reset_index()
        .sort_values([entity, "n"], ascending=[True, False], kind="stable")
        .drop_duplicates(entity)
        .set_index(entity)[value]
    )


def _pitcher_hand_map(
    exact: pd.DataFrame,
    raw: pd.DataFrame,
) -> tuple[dict[str, int], dict[str, Any]]:
    pitcher_map, _ = _best_map(
        exact, "pitcher_id", "pitcher_trackman_id", 0.99, True
    )
    raw_hand = _mode_by_entity(raw, "pitcher_trackman_id", "pitcher_hand")
    pairs = pd.DataFrame(
        {
            "main_hand": [
                int(
                    exact.loc[
                        exact["pitcher_id"].eq(pitcher_id), "pitcher_hand"
                    ].mode().iloc[0]
                )
                for pitcher_id in pitcher_map
            ],
            "raw_hand": [raw_hand.get(trackman_id) for trackman_id in pitcher_map.values()],
        }
    ).dropna()
    mapping, metadata = _best_map(
        pairs, "raw_hand", "main_hand", 0.99, True
    )
    return {str(key): int(value) for key, value in mapping.items()}, metadata


def _aggregate_rates(rows: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    sizes = rows.groupby(keys, observed=True, sort=False).size().rename("n")
    group_counts = (
        rows.groupby([*keys, "pitch_type_group"], observed=True, sort=False)
        .size()
        .unstack("pitch_type_group", fill_value=0)
        .reindex(columns=GROUPS, fill_value=0)
    )
    rates = group_counts.div(sizes, axis=0)
    rates.columns = [f"rate_{group}" for group in GROUPS]
    return pd.concat([sizes, rates], axis=1)


def build_batter_trackman_state(
    exact_joined: pd.DataFrame,
    raw_trackman: pd.DataFrame,
    cutoff: int,
) -> dict[str, Any]:
    """Build one completed-history state for rows belonging to ``cutoff``."""

    exact = exact_joined.loc[
        exact_joined["season"].lt(cutoff)
        & exact_joined["game_type"].eq("R")
    ].copy()
    raw = raw_trackman.loc[raw_trackman["season"].lt(cutoff)].copy()
    if exact.empty or raw.empty:
        return _empty_state(cutoff)

    batter_map, batter_meta = _best_map(
        exact, "batter_id", "batter_trackman_id", 0.99, True
    )
    inverse_batter = {
        trackman_id: int(batter_id)
        for batter_id, trackman_id in batter_map.items()
    }
    hand_map, hand_meta = _pitcher_hand_map(exact, raw)
    team_codes = sorted(
        set(exact["pitcher_team"].dropna().astype(str))
        | set(exact["batter_team"].dropna().astype(str))
    )
    major = raw.loc[
        raw["batter_trackman_id"].isin(inverse_batter)
        & raw["pitcher_team"].astype(str).isin(team_codes)
        & raw["batter_team"].astype(str).isin(team_codes)
    ].copy()
    major["batter_id"] = major["batter_trackman_id"].map(inverse_batter)
    major["main_pitcher_hand"] = (
        major["pitcher_hand"].astype(str).map(hand_map)
    )
    major["pitch_type_group"] = major["pitch_type_group"].where(
        major["pitch_type_group"].isin(GROUPS), "other"
    )
    major = major.dropna(subset=["batter_id", "main_pitcher_hand"])
    major["batter_id"] = major["batter_id"].astype(np.int64)
    major["main_pitcher_hand"] = major["main_pitcher_hand"].astype(np.int8)

    overall = _aggregate_rates(major, ["batter_id"])
    means = major.groupby("batter_id", observed=True, sort=False)[list(METRICS)].mean()
    means.columns = [f"mean_{metric}" for metric in METRICS]
    standard = major.groupby("batter_id", observed=True, sort=False)[list(METRICS)].std()
    standard.columns = [f"sd_{metric}" for metric in METRICS]
    overall = pd.concat([overall, means, standard], axis=1)

    hand = _aggregate_rates(major, ["batter_id", "main_pitcher_hand"])
    hand_means = major.groupby(
        ["batter_id", "main_pitcher_hand"], observed=True, sort=False
    )[list(METRICS)].mean()
    hand_means.columns = [f"mean_{metric}" for metric in METRICS]
    hand = pd.concat([hand, hand_means], axis=1)

    count = _aggregate_rates(
        major, ["batter_id", "balls_before", "strikes_before"]
    )
    metadata = {
        "cutoff": int(cutoff),
        "history_seasons": sorted(int(value) for value in raw["season"].unique()),
        "mapped_batters": int(len(inverse_batter)),
        "profile_rows": int(len(major)),
        "identity_minimum_purity": float(batter_meta["minimum_purity"]),
        "hand_minimum_purity": float(hand_meta["minimum_purity"]),
        "major_team_code_count": int(len(team_codes)),
        "current_or_future_trackman_rows": 0,
        "target_columns_read": False,
    }
    return {
        "cutoff": int(cutoff),
        "overall": overall,
        "hand": hand,
        "count": count,
        "metadata": metadata,
    }


def _lookup(frame: pd.DataFrame, table: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    if table.empty:
        return pd.DataFrame(index=frame.index, columns=table.columns, dtype=np.float64)
    if len(keys) == 1:
        index = pd.Index(frame[keys[0]].to_numpy(), name=table.index.name)
    else:
        index = pd.MultiIndex.from_frame(frame[keys])
        index.names = table.index.names
    result = table.reindex(index)
    result.index = frame.index
    return result


def features_from_state(frame: pd.DataFrame, state: dict[str, Any]) -> pd.DataFrame:
    def column(
        table: pd.DataFrame, name: str, default: float = np.nan
    ) -> pd.Series:
        if name in table.columns:
            return table[name]
        return pd.Series(default, index=frame.index, dtype=np.float64)

    values = pd.DataFrame(index=frame.index)
    overall = _lookup(frame, state["overall"], ["batter_id"])
    overall_n = column(overall, "n")
    known = overall_n.notna()
    values["e101_batter_tm_n_log"] = np.log1p(overall_n)
    for group in GROUPS:
        values[f"e101_{group}_rate"] = column(overall, f"rate_{group}")
    for metric in METRICS:
        values[f"e101_{metric}_mean"] = column(overall, f"mean_{metric}")
        values[f"e101_{metric}_sd"] = column(overall, f"sd_{metric}")
    values["e101_batter_tm_unseen"] = (~known).astype(np.int8)

    hand_query = frame[["batter_id", "pitcher_hand"]].rename(
        columns={"pitcher_hand": "main_pitcher_hand"}
    )
    hand_query.index = frame.index
    hand = _lookup(
        hand_query, state["hand"], ["batter_id", "main_pitcher_hand"]
    )
    hand_n = column(hand, "n").fillna(0.0)
    values["e102_hand_n_log"] = np.log1p(hand_n)
    for group in GROUPS:
        parent = column(overall, f"rate_{group}")
        raw = column(hand, f"rate_{group}")
        smoothed = (hand_n * raw.fillna(0.0) + HAND_K * parent) / (hand_n + HAND_K)
        values[f"e102_{group}_rate_delta"] = (smoothed - parent).where(known)
    for metric in METRICS:
        parent = column(overall, f"mean_{metric}")
        raw = column(hand, f"mean_{metric}")
        smoothed = (hand_n * raw.fillna(0.0) + HAND_K * parent) / (hand_n + HAND_K)
        values[f"e102_{metric}_mean_delta"] = (smoothed - parent).where(known)
    values["e102_hand_unseen"] = column(hand, "n").isna().astype(np.int8)

    count = _lookup(
        frame,
        state["count"],
        ["batter_id", "balls_before", "strikes_before"],
    )
    count_n = column(count, "n").fillna(0.0)
    values["e103_count_n_log"] = np.log1p(count_n)
    for group in GROUPS:
        parent = column(overall, f"rate_{group}")
        raw = column(count, f"rate_{group}")
        smoothed = (count_n * raw.fillna(0.0) + COUNT_K * parent) / (count_n + COUNT_K)
        values[f"e103_{group}_rate_delta"] = (smoothed - parent).where(known)
    values["e103_count_unseen"] = column(count, "n").isna().astype(np.int8)
    return values[list(FEATURE_COLUMNS)].astype(np.float32)


def build_batter_trackman_fold_features(
    history: pd.DataFrame,
    valid: pd.DataFrame,
    exact_joined: pd.DataFrame,
    raw_trackman: pd.DataFrame,
    validation_season: int,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    seasons = sorted(
        set(int(value) for value in history["season"].unique())
        | {int(validation_season)}
    )
    states = {
        season: build_batter_trackman_state(exact_joined, raw_trackman, season)
        for season in seasons
    }
    train = pd.DataFrame(index=history.index, columns=FEATURE_COLUMNS, dtype=np.float32)
    for season in sorted(int(value) for value in history["season"].unique()):
        mask = history["season"].eq(season)
        train.loc[mask, :] = features_from_state(history.loc[mask], states[season])
    validation = features_from_state(valid, states[int(validation_season)])

    sample = valid.iloc[: min(64, len(valid))]
    normal = features_from_state(sample, states[int(validation_season)])
    reversed_frame = sample.iloc[::-1]
    reversed_values = features_from_state(
        reversed_frame, states[int(validation_season)]
    ).reindex(sample.index)
    difference = np.nanmax(
        np.abs(normal.to_numpy(dtype=np.float64) - reversed_values.to_numpy(dtype=np.float64))
    )
    if not np.isfinite(difference):
        difference = 0.0
    metadata = {
        "enabled": True,
        "feature_columns": list(FEATURE_COLUMNS),
        "feature_count": len(FEATURE_COLUMNS),
        "hand_k": HAND_K,
        "count_k": COUNT_K,
        "states": {str(year): states[year]["metadata"] for year in seasons},
        "validation_unseen_rate": float(validation["e101_batter_tm_unseen"].mean()),
        "row_order_invariance_max_abs": float(difference),
        "row_independent": True,
        "target_free": True,
        "cutoff": "raw and identity-linkage TrackMan seasons strictly before each query season",
    }
    return train.astype(np.float32), validation.astype(np.float32), metadata
