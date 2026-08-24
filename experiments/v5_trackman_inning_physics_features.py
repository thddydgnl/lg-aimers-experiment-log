#!/usr/bin/env python3
"""Cutoff-safe pitcher physical profiles conditional on the public inning.

The current pitch's TrackMan measurements are never used.  For every modeled
season, completed earlier official TrackMan seasons are reduced to frozen
pitcher x inning-band profiles.  Coarse pitch-group means and variances are
used as the parent so that the features separate within-repertoire physical
change from changes in pitch mix.
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
PITCH_GROUPS = ("fastball", "breaking", "offspeed", "other")
SHRINKAGE_K = 100.0

PROFILE_COLUMNS = [
    "e119_inning_n_log",
    "e119_inning_share",
    "e119_observed_group_count",
    *[f"e119_{group}_mix_delta" for group in PITCH_GROUPS],
    *[
        f"e119_{metric}_{suffix}"
        for metric in METRICS
        for suffix in (
            "within_group_mean_delta",
            "total_mean_delta",
            "within_group_log_sd_ratio",
        )
    ],
    "e119_inning_profile_unseen",
]


def inning_band(values: pd.Series | np.ndarray) -> np.ndarray:
    """Map innings to the preregistered early/middle/late bands."""

    innings = np.asarray(values, dtype=np.int16)
    return np.where(innings <= 3, 0, np.where(innings <= 6, 1, 2)).astype(
        np.int8
    )


def _empty() -> pd.DataFrame:
    index = pd.MultiIndex.from_arrays(
        [np.array([], dtype=np.int64), np.array([], dtype=np.int8)],
        names=["pitcher_id", "inning_band"],
    )
    return pd.DataFrame(columns=PROFILE_COLUMNS, index=index)


def inning_physics_profile_table(
    rows: pd.DataFrame, k: float = SHRINKAGE_K
) -> pd.DataFrame:
    """Build frozen pitcher x inning-band target-free physical profiles."""

    if k <= 0:
        raise ValueError("inning physics shrinkage k must be positive")
    regular = rows.loc[rows["game_type"].eq("R")].copy()
    required = {"pitcher_id", "inning", "pitch_type_group", *METRICS}
    missing = sorted(required.difference(regular.columns))
    if missing:
        raise ValueError(f"joined TrackMan rows miss columns: {missing}")
    regular = regular.dropna(
        subset=["pitcher_id", "inning", "pitch_type_group"]
    )
    regular = regular.loc[regular["pitch_type_group"].isin(PITCH_GROUPS)].copy()
    if regular.empty:
        return _empty()
    regular["pitcher_id"] = regular["pitcher_id"].astype(np.int64)
    regular["inning_band"] = inning_band(regular["inning"])

    parent_keys = ["pitcher_id", "pitch_type_group"]
    cell_keys = ["pitcher_id", "inning_band", "pitch_type_group"]
    band_keys = ["pitcher_id", "inning_band"]

    parent_group = regular.groupby(parent_keys, sort=False, observed=True)
    cell_group = regular.groupby(cell_keys, sort=False, observed=True)
    parent_n = parent_group.size().astype(np.float64)
    cell_n = cell_group.size().astype(np.float64)
    pitcher_n = regular.groupby("pitcher_id", sort=False, observed=True).size().astype(
        np.float64
    )
    band_n = regular.groupby(band_keys, sort=False, observed=True).size().astype(
        np.float64
    )

    band_index = band_n.index
    full_index = pd.MultiIndex.from_product(
        [
            band_index.get_level_values("pitcher_id").unique(),
            (0, 1, 2),
            PITCH_GROUPS,
        ],
        names=cell_keys,
    )
    # Keep only pitcher-band pairs actually observed; the group level remains
    # complete so absent pitch groups shrink exactly to their pitcher parent.
    observed_bands = pd.MultiIndex.from_arrays(
        [
            full_index.get_level_values("pitcher_id"),
            full_index.get_level_values("inning_band"),
        ],
        names=band_keys,
    ).isin(band_index)
    full_index = full_index[observed_bands]

    pitchers = full_index.get_level_values("pitcher_id")
    bands = full_index.get_level_values("inning_band")
    groups = full_index.get_level_values("pitch_type_group")
    parent_index = pd.MultiIndex.from_arrays(
        [pitchers, groups], names=parent_keys
    )
    band_lookup_index = pd.MultiIndex.from_arrays(
        [pitchers, bands], names=band_keys
    )

    parent_count = parent_n.reindex(parent_index).fillna(0.0).to_numpy()
    cell_count = cell_n.reindex(full_index).fillna(0.0).to_numpy()
    total_count = pitcher_n.reindex(pitchers).to_numpy(dtype=np.float64)
    current_band_count = band_n.reindex(band_lookup_index).to_numpy(dtype=np.float64)
    parent_rate = np.divide(
        parent_count,
        total_count,
        out=np.zeros_like(parent_count),
        where=total_count > 0,
    )
    band_rate = (cell_count + k * parent_rate) / (current_band_count + k)

    output = pd.DataFrame(index=band_index)
    output["e119_inning_n_log"] = np.log1p(band_n).astype(np.float32)
    output["e119_inning_share"] = (
        band_n
        / pitcher_n.reindex(band_index.get_level_values("pitcher_id")).to_numpy()
    ).astype(np.float32)
    output["e119_observed_group_count"] = (
        cell_n.gt(0).groupby(level=band_keys, observed=True).sum().reindex(band_index)
    ).astype(np.float32)

    rate_frame = pd.DataFrame(
        {
            "band_rate": band_rate,
            "parent_rate": parent_rate,
        },
        index=full_index,
    )
    for group in PITCH_GROUPS:
        group_rows = rate_frame.xs(group, level="pitch_type_group")
        output[f"e119_{group}_mix_delta"] = (
            group_rows["band_rate"] - group_rows["parent_rate"]
        ).reindex(band_index)

    parent_moments = parent_group[list(METRICS)].agg(["count", "mean", "var"])
    cell_moments = cell_group[list(METRICS)].agg(["count", "mean", "var"])
    for metric in METRICS:
        metric_cell_count = cell_moments[(metric, "count")].reindex(
            full_index
        ).fillna(0.0).to_numpy(dtype=np.float64)
        parent_mean = parent_moments[(metric, "mean")].reindex(
            parent_index
        ).to_numpy(dtype=np.float64)
        parent_var = parent_moments[(metric, "var")].reindex(
            parent_index
        ).to_numpy(dtype=np.float64)
        cell_mean = cell_moments[(metric, "mean")].reindex(full_index).to_numpy(
            dtype=np.float64
        )
        cell_var = cell_moments[(metric, "var")].reindex(full_index).to_numpy(
            dtype=np.float64
        )

        valid_parent_mean = np.isfinite(parent_mean)
        mean_weight = np.where(valid_parent_mean, metric_cell_count, 0.0)
        smoothed_mean = np.where(
            valid_parent_mean,
            (
                np.nan_to_num(cell_mean, nan=0.0) * mean_weight
                + k * np.nan_to_num(parent_mean, nan=0.0)
            )
            / (mean_weight + k),
            0.0,
        )
        parent_mean_safe = np.nan_to_num(parent_mean, nan=0.0)
        within_delta = smoothed_mean - parent_mean_safe

        # Variances use effective degrees of freedom and the same fixed prior.
        effective_df = np.maximum(metric_cell_count - 1.0, 0.0)
        valid_parent_var = np.isfinite(parent_var) & (parent_var >= 0.0)
        smoothed_var = np.where(
            valid_parent_var,
            (
                effective_df * np.nan_to_num(cell_var, nan=0.0)
                + k * np.nan_to_num(parent_var, nan=0.0)
            )
            / (effective_df + k),
            np.nan,
        )
        log_sd_ratio = np.where(
            valid_parent_var,
            0.5
            * np.log(
                (np.maximum(smoothed_var, 0.0) + 1e-8)
                / (np.maximum(parent_var, 0.0) + 1e-8)
            ),
            0.0,
        )

        metric_frame = pd.DataFrame(
            {
                "within": band_rate * within_delta,
                "band_total": band_rate * smoothed_mean,
                "parent_total": parent_rate * parent_mean_safe,
                "log_sd": band_rate * log_sd_ratio,
            },
            index=full_index,
        )
        aggregated = metric_frame.groupby(level=band_keys, observed=True).sum()
        output[f"e119_{metric}_within_group_mean_delta"] = aggregated[
            "within"
        ].reindex(band_index)
        output[f"e119_{metric}_total_mean_delta"] = (
            aggregated["band_total"] - aggregated["parent_total"]
        ).reindex(band_index)
        output[f"e119_{metric}_within_group_log_sd_ratio"] = aggregated[
            "log_sd"
        ].reindex(band_index)

    output["e119_inning_profile_unseen"] = 0.0
    output = output.replace([np.inf, -np.inf], np.nan)
    return output[PROFILE_COLUMNS].astype(np.float32)


def inning_physics_states_before_each_season(
    joined: pd.DataFrame,
    seasons: list[int],
    window: int | None = None,
    k: float = SHRINKAGE_K,
) -> tuple[dict[int, pd.DataFrame], pd.DataFrame]:
    if window is not None and window < 1:
        raise ValueError("TrackMan profile window must be >= 1")
    before: dict[int, pd.DataFrame] = {}
    for season in sorted(seasons):
        rows = joined.loc[joined["season"].lt(season)]
        if window is not None:
            rows = rows.loc[rows["season"].ge(season - window)]
        before[season] = inning_physics_profile_table(rows, k=k)
    if seasons:
        cutoff = max(seasons) + 1
        rows = joined.loc[joined["season"].lt(cutoff)]
        if window is not None:
            rows = rows.loc[rows["season"].ge(cutoff - window)]
        final = inning_physics_profile_table(rows, k=k)
    else:
        final = _empty()
    return before, final


def build_inning_physics_features(
    frame: pd.DataFrame, profiles_before: dict[int, pd.DataFrame]
) -> tuple[pd.DataFrame, dict[str, Any]]:
    values = np.zeros((len(frame), len(PROFILE_COLUMNS)), dtype=np.float32)
    unseen_index = PROFILE_COLUMNS.index("e119_inning_profile_unseen")
    values[:, unseen_index] = 1.0
    seasons = frame["season"].to_numpy(dtype=np.int16, copy=False)
    pitchers = frame["pitcher_id"].to_numpy(dtype=np.int64, copy=False)
    bands = inning_band(frame["inning"])
    for season in sorted(set(int(value) for value in seasons)):
        mask = seasons == season
        profile = profiles_before.get(season)
        if profile is None or profile.empty:
            continue
        key = pd.MultiIndex.from_arrays(
            [pitchers[mask], bands[mask]], names=["pitcher_id", "inning_band"]
        )
        lookup = profile.reindex(key)
        known = lookup["e119_inning_profile_unseen"].notna().to_numpy(dtype=bool)
        matrix = lookup[PROFILE_COLUMNS].to_numpy(dtype=np.float32)
        indices = np.flatnonzero(mask)
        values[indices[known]] = matrix[known]
        values[indices[known], unseen_index] = 0.0
    result = pd.DataFrame(values, columns=PROFILE_COLUMNS, index=frame.index)
    return result, {
        "unseen_rows": int(result["e119_inning_profile_unseen"].gt(0).sum()),
        "known_rows": int(result["e119_inning_profile_unseen"].eq(0).sum()),
        "feature_count": len(PROFILE_COLUMNS),
        "inning_bands": {"0": "1-3", "1": "4-6", "2": "7+"},
        "shrinkage_k": SHRINKAGE_K,
        "cutoff": "official linked regular TrackMan seasons strictly before row season",
        "target_free": True,
        "row_independent": True,
        "current_pitch_trackman_used": False,
    }
