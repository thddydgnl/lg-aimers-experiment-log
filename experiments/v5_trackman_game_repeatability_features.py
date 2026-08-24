#!/usr/bin/env python3
"""Cutoff-safe pitcher repeatability inside games and pitch groups.

The older e74 profile removes pitch-group means but still mixes slow changes
between games with pitch-to-pitch execution noise.  This module removes a
separate mean for every pitcher x TrackMan game x coarse pitch group, pools the
remaining variance, and exposes the complementary game-to-game drift.  Only
completed official TrackMan seasons are used; inference is a frozen pitcher
lookup and is independent across evaluation rows.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


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

PROFILE_COLUMNS = [
    "e118_observation_n_log",
    "e118_game_n_log",
    "e118_cell_n_log",
    *[
        f"e118_{metric}_{suffix}"
        for metric in METRICS
        for suffix in ("within_game_group_sd", "between_game_group_sd", "within_share")
    ],
    "e118_release_within_corr",
    "e118_release_ellipse_log_area",
    "e118_release_ellipse_log_axis_ratio",
    "e118_profile_unseen",
]


def _empty() -> pd.DataFrame:
    return pd.DataFrame(columns=PROFILE_COLUMNS).set_index(
        pd.Index([], name="pitcher_id")
    )


def game_repeatability_profile_table(rows: pd.DataFrame) -> pd.DataFrame:
    """Build one frozen target-free profile per anonymous pitcher."""

    regular = rows.loc[rows["game_type"].eq("R")].copy()
    required = {"pitcher_id", "trackman_game_id", "pitch_type_group", *METRICS}
    missing = sorted(required.difference(regular.columns))
    if missing:
        raise ValueError(f"joined TrackMan rows miss columns: {missing}")
    regular = regular.dropna(subset=["pitcher_id", "trackman_game_id", "pitch_type_group"])
    if regular.empty:
        return _empty()

    cell_keys = ["pitcher_id", "trackman_game_id", "pitch_type_group"]
    pitch_keys = ["pitcher_id", "pitch_type_group"]
    cell = regular.groupby(cell_keys, sort=False, observed=True)
    cell_size = cell.size().rename("cell_n")
    usable_cells = cell_size.loc[cell_size.ge(3)].index
    if len(usable_cells) == 0:
        return _empty()

    pitcher_index = pd.Index(
        sorted(regular["pitcher_id"].astype(np.int64).unique()), name="pitcher_id"
    )
    output = pd.DataFrame(index=pitcher_index)
    pitcher = regular.groupby("pitcher_id", sort=False, observed=True)
    output["e118_observation_n_log"] = np.log1p(
        pitcher.size().reindex(pitcher_index).fillna(0.0)
    ).astype(np.float32)
    output["e118_game_n_log"] = np.log1p(
        pitcher["trackman_game_id"].nunique().reindex(pitcher_index).fillna(0.0)
    ).astype(np.float32)
    usable_frame = cell_size.loc[usable_cells].reset_index()
    output["e118_cell_n_log"] = np.log1p(
        usable_frame.groupby("pitcher_id", observed=True).size().reindex(
            pitcher_index
        ).fillna(0.0)
    ).astype(np.float32)

    # One multi-column aggregation is materially faster than rebuilding the
    # same 100k+ group index once per physical metric.
    moments_all = cell[list(METRICS)].agg(["count", "mean", "var"]).reindex(
        usable_cells
    )
    pitch_means_all = regular.groupby(
        pitch_keys, sort=False, observed=True
    )[list(METRICS)].mean()
    pitcher_values = moments_all.index.get_level_values("pitcher_id")
    pitch_values = moments_all.index.get_level_values("pitch_type_group")
    parent_index = pd.MultiIndex.from_arrays(
        [pitcher_values, pitch_values], names=pitch_keys
    )
    parent_means_all = pitch_means_all.reindex(parent_index)

    within_variances: dict[str, pd.Series] = {}
    for metric in METRICS:
        moments = moments_all[metric]
        degrees = (moments["count"] - 1.0).clip(lower=0.0)
        within_num = (degrees * moments["var"].fillna(0.0)).groupby(
            level="pitcher_id", observed=True
        ).sum()
        within_den = degrees.groupby(level="pitcher_id", observed=True).sum()
        within_var = (within_num / within_den.replace(0.0, np.nan)).reindex(
            pitcher_index
        )
        within_variances[metric] = within_var

        parent_mean = parent_means_all[metric].to_numpy(dtype=np.float64)
        cell_mean = moments["mean"].to_numpy(dtype=np.float64)
        cell_weight = moments["count"].fillna(0.0).to_numpy(dtype=np.float64)
        between_contribution = cell_weight * np.square(cell_mean - parent_mean)
        between_num = pd.Series(
            between_contribution, index=moments.index
        ).groupby(level="pitcher_id", observed=True).sum()
        between_den = moments["count"].groupby(
            level="pitcher_id", observed=True
        ).sum()
        between_var = (between_num / between_den.replace(0.0, np.nan)).reindex(
            pitcher_index
        )

        total = within_var + between_var
        output[f"e118_{metric}_within_game_group_sd"] = np.sqrt(
            within_var.clip(lower=0.0)
        )
        output[f"e118_{metric}_between_game_group_sd"] = np.sqrt(
            between_var.clip(lower=0.0)
        )
        output[f"e118_{metric}_within_share"] = (
            within_var / total.replace(0.0, np.nan)
        ).clip(0.0, 1.0)

    # Release covariance after removing each game x pitch-group mean.
    usable_key = pd.MultiIndex.from_frame(regular[cell_keys])
    usable_mask = usable_key.isin(usable_cells)
    residual_frame = regular.loc[usable_mask, ["pitcher_id", "rel_height", "rel_side"]].copy()
    height_cell_mean = moments_all[("rel_height", "mean")].reindex(
        usable_key[usable_mask]
    ).to_numpy(dtype=np.float64)
    side_cell_mean = moments_all[("rel_side", "mean")].reindex(
        usable_key[usable_mask]
    ).to_numpy(dtype=np.float64)
    residual_frame["height_resid"] = (
        regular.loc[usable_mask, "rel_height"].to_numpy(dtype=np.float64)
        - height_cell_mean
    )
    residual_frame["side_resid"] = (
        regular.loc[usable_mask, "rel_side"].to_numpy(dtype=np.float64)
        - side_cell_mean
    )
    h = residual_frame["height_resid"]
    s = residual_frame["side_resid"]
    key = residual_frame["pitcher_id"]
    cross = (h * s).groupby(key, observed=True).sum().reindex(pitcher_index)
    hss = (h * h).groupby(key, observed=True).sum().reindex(pitcher_index)
    sss = (s * s).groupby(key, observed=True).sum().reindex(pitcher_index)
    corr = (cross / np.sqrt(hss * sss).replace(0.0, np.nan)).clip(-1.0, 1.0)
    output["e118_release_within_corr"] = corr
    height_var = within_variances["rel_height"].reindex(pitcher_index)
    side_var = within_variances["rel_side"].reindex(pitcher_index)
    covariance = corr * np.sqrt(height_var * side_var)
    trace = height_var + side_var
    discriminant = np.sqrt(
        np.maximum(np.square(height_var - side_var) + 4.0 * np.square(covariance), 0.0)
    )
    largest = np.maximum((trace + discriminant) / 2.0, 0.0)
    smallest = np.maximum((trace - discriminant) / 2.0, 0.0)
    output["e118_release_ellipse_log_area"] = 0.5 * np.log(
        largest * smallest + 1e-12
    )
    output["e118_release_ellipse_log_axis_ratio"] = 0.5 * np.log(
        (largest + 1e-12) / (smallest + 1e-12)
    )
    output["e118_profile_unseen"] = 0.0
    return output[PROFILE_COLUMNS]


def game_repeatability_states_before_each_season(
    joined: pd.DataFrame, seasons: list[int], window: int | None = None
) -> tuple[dict[int, pd.DataFrame], pd.DataFrame]:
    if window is not None and window < 1:
        raise ValueError("TrackMan profile window must be >= 1")
    before: dict[int, pd.DataFrame] = {}
    for season in sorted(seasons):
        rows = joined.loc[joined["season"].lt(season)]
        if window is not None:
            rows = rows.loc[rows["season"].ge(season - window)]
        before[season] = game_repeatability_profile_table(rows)
    if seasons:
        cutoff = max(seasons) + 1
        rows = joined.loc[joined["season"].lt(cutoff)]
        if window is not None:
            rows = rows.loc[rows["season"].ge(cutoff - window)]
        final = game_repeatability_profile_table(rows)
    else:
        final = _empty()
    return before, final


def build_game_repeatability_features(
    frame: pd.DataFrame, profiles_before: dict[int, pd.DataFrame]
) -> tuple[pd.DataFrame, dict[str, Any]]:
    values = np.full((len(frame), len(PROFILE_COLUMNS)), np.nan, dtype=np.float32)
    unseen_index = PROFILE_COLUMNS.index("e118_profile_unseen")
    values[:, unseen_index] = 1.0
    seasons = frame["season"].to_numpy(dtype=np.int16, copy=False)
    pitchers = frame["pitcher_id"].to_numpy(dtype=np.int64, copy=False)
    for season in sorted(set(int(value) for value in seasons)):
        mask = seasons == season
        profile = profiles_before.get(season)
        if profile is None or profile.empty:
            continue
        lookup = profile.reindex(pitchers[mask])
        known = lookup["e118_profile_unseen"].notna().to_numpy(dtype=bool)
        matrix = lookup[PROFILE_COLUMNS].to_numpy(dtype=np.float32)
        indices = np.flatnonzero(mask)
        values[indices[known]] = matrix[known]
        values[indices[known], unseen_index] = 0.0
    result = pd.DataFrame(values, columns=PROFILE_COLUMNS, index=frame.index)
    return result, {
        "unseen_rows": int(result["e118_profile_unseen"].gt(0).sum()),
        "known_rows": int(result["e118_profile_unseen"].eq(0).sum()),
        "feature_count": len(PROFILE_COLUMNS),
        "cutoff": "official linked regular TrackMan seasons strictly before row season",
        "target_free": True,
        "row_independent": True,
        "cell": "pitcher x TrackMan game x coarse pitch group, minimum 3 pitches",
    }
