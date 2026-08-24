#!/usr/bin/env python3
"""Leakage-safe rolling evaluation of six historical Trackman profile features."""

from __future__ import annotations

import argparse
import gc
import json
import platform
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from eda.run_structural_eda import linkage_section, load_trackman, load_train as load_structural_train  # noqa: E402
from experiments.run_baselines import FEATURES as BASE_FEATURES  # noqa: E402
from experiments.run_baselines import TARGET, load_train  # noqa: E402
from experiments.run_e14_rolling import (  # noqa: E402
    E14_K,
    build_e14_features,
    fit_predict,
    make_hgb,
    make_linear,
    metric,
    prior_before_each_season,
    season_end_state,
)
from experiments.run_e15_pseudo_forward import candidate_priors  # noqa: E402


PROFILE_COLUMNS = [
    "e20_rel_speed_mean",
    "e20_rel_speed_sd",
    "e20_horz_break_mean",
    "e20_rel_side_mean",
    "e20_rel_side_sd",
    "e20_fastball_rate",
    "e20_profile_n_log",
    "e20_profile_unseen",
]
TRACKMAN_COLUMNS = ["rel_speed", "horz_break", "rel_side"]

RICH_TRACKMAN_COLUMNS = [
    "rel_speed",
    "spin_rate",
    "induced_vert_break",
    "horz_break",
    "extension",
    "rel_height",
    "rel_side",
    "zone_speed",
]
RICH_PITCH_GROUPS = ["fastball", "breaking", "offspeed", "other"]
RICH_PROFILE_COLUMNS = [
    "e58_profile_n_log",
    *[
        f"e58_{metric}_{stat}"
        for metric in RICH_TRACKMAN_COLUMNS
        for stat in ("mean", "sd")
    ],
    *[f"e58_{group}_rate" for group in RICH_PITCH_GROUPS],
    *[
        f"e58_{group}_{metric}_mean"
        for group in RICH_PITCH_GROUPS
        for metric in RICH_TRACKMAN_COLUMNS
    ],
    "e58_profile_unseen",
]
TRACKMAN_PLATOON_COLUMNS = [
    "e59_platoon_n_log",
    *[f"e59_{metric}_delta" for metric in RICH_TRACKMAN_COLUMNS],
    *[f"e59_{group}_rate_delta" for group in RICH_PITCH_GROUPS],
    "e59_platoon_unseen",
]
TRACKMAN_COUNT_COLUMNS = [
    "e71_count_n_log",
    *[f"e71_{group}_rate_delta" for group in RICH_PITCH_GROUPS],
    "e71_count_unseen",
]

STABILITY_PROFILE_COLUMNS = [
    *[
        f"e74_{metric}_{kind}_sd"
        for metric in RICH_TRACKMAN_COLUMNS
        for kind in ("within_group", "between_group")
    ],
    "e74_rel_height_side_within_corr",
    "e74_speed_ivb_within_corr",
    "e74_speed_horz_within_corr",
    "e74_release_ellipse_log_area",
    "e74_release_ellipse_log_axis_ratio",
    "e74_pitchmix_entropy",
    "e74_profile_unseen",
]

TREND_SOURCE_COLUMNS = [
    *[
        f"e58_{metric}_{stat}"
        for metric in RICH_TRACKMAN_COLUMNS
        for stat in ("mean", "sd")
    ],
    *[f"e58_{group}_rate" for group in RICH_PITCH_GROUPS],
]
TREND_PROFILE_COLUMNS = [
    "e75_recent_profile_n_log",
    *[f"e75_{column.removeprefix('e58_')}_delta" for column in TREND_SOURCE_COLUMNS],
    "e75_recent_profile_unseen",
]

GROUP_STABILITY_GROUPS = ("fastball", "breaking", "offspeed")
GROUP_STABILITY_PROFILE_COLUMNS = [
    *[f"e76_{group}_n_log" for group in GROUP_STABILITY_GROUPS],
    *[
        f"e76_{group}_{metric}_sd"
        for group in GROUP_STABILITY_GROUPS
        for metric in RICH_TRACKMAN_COLUMNS
    ],
    "e76_profile_unseen",
]


def json_safe(value: Any) -> Any:
    """Convert numpy scalars and non-finite diagnostics to JSON-safe values."""
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [json_safe(item) for item in value]
    if isinstance(value, tuple):
        return [json_safe(item) for item in value]
    if isinstance(value, (np.floating, float)):
        value = float(value)
        return value if np.isfinite(value) else None
    if isinstance(value, (np.integer, int)):
        return int(value)
    return value


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=ROOT / "open/data/train.csv")
    parser.add_argument(
        "--validation-seasons", nargs="+", type=int, default=[2022, 2023, 2024]
    )
    parser.add_argument("--output-dir", type=Path, default=ROOT / "experiments/results")
    parser.add_argument("--prior-mode", default="r_recent3")
    return parser.parse_args()


def load_joined_trackman() -> pd.DataFrame:
    train = load_structural_train()
    trackman, game_ids, _ = load_trackman()
    _, joined = linkage_section(train, trackman, len(game_ids))
    print(f"E20R joined historical rows: {len(joined):,}", flush=True)
    return joined


def load_joined_and_raw_trackman() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load the conservative game linkage and the full cleaned official log.

    The raw table is returned only for history-profile experiments.  Identity
    recovery still comes exclusively from the exact fingerprint linkage; an
    unmatched TrackMan game is never asserted to be a main-table game.
    """
    train = load_structural_train()
    trackman, game_ids, _ = load_trackman()
    _, joined = linkage_section(train, trackman, len(game_ids))
    print(
        f"E20R joined historical rows: {len(joined):,}; "
        f"raw cleaned TrackMan rows: {len(trackman):,}",
        flush=True,
    )
    return joined, trackman


def profile_table(rows: pd.DataFrame) -> pd.DataFrame:
    """Aggregate profile values by pitcher, combining within-group moments."""
    regular = rows.loc[rows["game_type"].eq("R")].copy()
    columns = [*TRACKMAN_COLUMNS, "e20_profile_n", *PROFILE_COLUMNS]
    if regular.empty:
        return pd.DataFrame(columns=columns).set_index(pd.Index([], name="pitcher_id"))
    group = regular.groupby(["pitcher_id", "pitch_type_group"], observed=True)
    output = pd.DataFrame(index=pd.Index(sorted(regular["pitcher_id"].dropna().unique()), name="pitcher_id"))
    output["e20_profile_n"] = regular.groupby("pitcher_id", observed=True).size().astype(np.int32)
    output["e20_profile_n_log"] = np.log1p(output["e20_profile_n"]).astype(np.float32)
    for source, mean_name, sd_name in (
        ("rel_speed", "e20_rel_speed_mean", "e20_rel_speed_sd"),
        ("rel_side", "e20_rel_side_mean", "e20_rel_side_sd"),
    ):
        moments = group[source].agg(["count", "mean", "var"]).reset_index()
        moments = moments.dropna(subset=["mean"])
        for pitcher, subset in moments.groupby("pitcher_id", sort=False):
            counts = subset["count"].to_numpy(dtype=np.float64)
            means = subset["mean"].to_numpy(dtype=np.float64)
            variances = subset["var"].fillna(0.0).to_numpy(dtype=np.float64)
            total = counts.sum()
            if total <= 0:
                continue
            weighted_mean = float(np.sum(counts * means) / total)
            second = float(np.sum(counts * (variances + means**2)) / total)
            output.loc[int(pitcher), mean_name] = weighted_mean
            output.loc[int(pitcher), sd_name] = np.sqrt(max(0.0, second - weighted_mean**2))
    horz = group["horz_break"].mean().reset_index(name="mean").dropna(subset=["mean"])
    if not horz.empty:
        output["e20_horz_break_mean"] = horz.groupby("pitcher_id")["mean"].mean()
    fastball = regular["pitch_type_group"].eq("fastball").groupby(regular["pitcher_id"]).mean()
    output["e20_fastball_rate"] = fastball
    output["e20_profile_unseen"] = 0
    return output[PROFILE_COLUMNS]


def rich_profile_table(rows: pd.DataFrame) -> pd.DataFrame:
    """Aggregate a fuller target-free Trackman profile by anonymous pitcher."""
    regular = rows.loc[rows["game_type"].eq("R")].copy()
    if regular.empty:
        return pd.DataFrame(columns=RICH_PROFILE_COLUMNS).set_index(
            pd.Index([], name="pitcher_id")
        )
    pitcher = regular.groupby("pitcher_id", sort=False, observed=True)
    output = pd.DataFrame(index=pitcher.size().index)
    output.index.name = "pitcher_id"
    output["e58_profile_n_log"] = np.log1p(pitcher.size()).astype(np.float32)
    for metric in RICH_TRACKMAN_COLUMNS:
        moments = pitcher[metric].agg(["mean", "std"])
        output[f"e58_{metric}_mean"] = moments["mean"]
        output[f"e58_{metric}_sd"] = moments["std"].fillna(0.0)
    denominator = pitcher.size().astype(np.float64)
    for group in RICH_PITCH_GROUPS:
        subset = regular.loc[regular["pitch_type_group"].eq(group)]
        group_size = subset.groupby("pitcher_id", observed=True).size()
        output[f"e58_{group}_rate"] = (group_size / denominator).fillna(0.0)
        group_means = subset.groupby("pitcher_id", observed=True)[
            RICH_TRACKMAN_COLUMNS
        ].mean()
        for metric in RICH_TRACKMAN_COLUMNS:
            output[f"e58_{group}_{metric}_mean"] = group_means[metric]
    output["e58_profile_unseen"] = 0
    return output[RICH_PROFILE_COLUMNS]


def stability_profile_table(rows: pd.DataFrame) -> pd.DataFrame:
    """Summarize target-free repeatability after removing pitch-group means.

    Overall TrackMan variance conflates a pitcher's repertoire with execution
    spread.  The pooled within-group standard deviation below isolates the
    latter, while the complementary between-group term preserves repertoire
    separation.  All statistics are computed from completed regular-season
    TrackMan rows only.
    """

    regular = rows.loc[rows["game_type"].eq("R")].copy()
    if regular.empty:
        return pd.DataFrame(columns=STABILITY_PROFILE_COLUMNS).set_index(
            pd.Index([], name="pitcher_id")
        )
    pitcher = regular.groupby("pitcher_id", sort=False, observed=True)
    cell = regular.groupby(
        ["pitcher_id", "pitch_type_group"], sort=False, observed=True
    )
    output = pd.DataFrame(index=pitcher.size().index)
    output.index.name = "pitcher_id"
    within_variances: dict[str, pd.Series] = {}
    for metric in RICH_TRACKMAN_COLUMNS:
        moments = cell[metric].agg(["count", "var"])
        degrees = (moments["count"] - 1.0).clip(lower=0.0)
        numerator = (degrees * moments["var"].fillna(0.0)).groupby(
            level="pitcher_id", observed=True
        ).sum()
        denominator = degrees.groupby(
            level="pitcher_id", observed=True
        ).sum()
        within_var = (numerator / denominator.replace(0.0, np.nan)).reindex(
            output.index
        )
        total_var = pitcher[metric].var().reindex(output.index)
        between_var = (total_var - within_var).clip(lower=0.0)
        within_variances[metric] = within_var
        output[f"e74_{metric}_within_group_sd"] = np.sqrt(within_var)
        output[f"e74_{metric}_between_group_sd"] = np.sqrt(between_var)

    residuals: dict[str, pd.Series] = {}
    for metric in (
        "rel_height", "rel_side", "rel_speed", "induced_vert_break", "horz_break"
    ):
        residuals[metric] = regular[metric] - cell[metric].transform("mean")

    def pooled_corr(left: str, right: str) -> pd.Series:
        x = residuals[left]
        y = residuals[right]
        key = regular["pitcher_id"]
        numerator = (x * y).groupby(key, observed=True).sum()
        x_ss = (x * x).groupby(key, observed=True).sum()
        y_ss = (y * y).groupby(key, observed=True).sum()
        denominator = np.sqrt(x_ss * y_ss).replace(0.0, np.nan)
        return (numerator / denominator).clip(-1.0, 1.0).reindex(output.index)

    release_corr = pooled_corr("rel_height", "rel_side")
    output["e74_rel_height_side_within_corr"] = release_corr
    output["e74_speed_ivb_within_corr"] = pooled_corr(
        "rel_speed", "induced_vert_break"
    )
    output["e74_speed_horz_within_corr"] = pooled_corr(
        "rel_speed", "horz_break"
    )
    height_var = within_variances["rel_height"].reindex(output.index)
    side_var = within_variances["rel_side"].reindex(output.index)
    covariance = release_corr * np.sqrt(height_var * side_var)
    trace = height_var + side_var
    discriminant = np.sqrt(
        np.maximum((height_var - side_var) ** 2 + 4.0 * covariance**2, 0.0)
    )
    largest = np.maximum((trace + discriminant) / 2.0, 0.0)
    smallest = np.maximum((trace - discriminant) / 2.0, 0.0)
    output["e74_release_ellipse_log_area"] = 0.5 * np.log(
        np.maximum(largest * smallest, 0.0) + 1e-12
    )
    output["e74_release_ellipse_log_axis_ratio"] = 0.5 * np.log(
        (largest + 1e-12) / (smallest + 1e-12)
    )
    mix_counts = regular.groupby(
        ["pitcher_id", "pitch_type_group"], observed=True
    ).size()
    mix_rates = mix_counts / mix_counts.groupby(level="pitcher_id").sum()
    output["e74_pitchmix_entropy"] = (
        -(mix_rates * np.log(np.maximum(mix_rates, 1e-12)))
        .groupby(level="pitcher_id", observed=True)
        .sum()
        .reindex(output.index)
    )
    output["e74_profile_unseen"] = 0
    return output[STABILITY_PROFILE_COLUMNS]


def group_stability_profile_table(rows: pd.DataFrame) -> pd.DataFrame:
    """Expose pitch-group-specific physical repeatability by pitcher."""

    regular = rows.loc[rows["game_type"].eq("R")].copy()
    if regular.empty:
        return pd.DataFrame(columns=GROUP_STABILITY_PROFILE_COLUMNS).set_index(
            pd.Index([], name="pitcher_id")
        )
    pitcher_index = pd.Index(
        sorted(regular["pitcher_id"].dropna().unique()), name="pitcher_id"
    )
    output = pd.DataFrame(index=pitcher_index)
    for group in GROUP_STABILITY_GROUPS:
        subset = regular.loc[regular["pitch_type_group"].eq(group)]
        grouped = subset.groupby("pitcher_id", sort=False, observed=True)
        output[f"e76_{group}_n_log"] = np.log1p(
            grouped.size().reindex(pitcher_index)
        )
        standard_deviations = grouped[RICH_TRACKMAN_COLUMNS].std().reindex(
            pitcher_index
        )
        for metric in RICH_TRACKMAN_COLUMNS:
            output[f"e76_{group}_{metric}_sd"] = standard_deviations[metric]
    output["e76_profile_unseen"] = 0
    return output[GROUP_STABILITY_PROFILE_COLUMNS]


def trackman_platoon_profile_table(rows: pd.DataFrame, k: float) -> pd.DataFrame:
    """Shrink pitcher-by-batter-hand Trackman profiles to each pitcher mean."""
    if k <= 0:
        raise ValueError("Trackman platoon smoothing k must be positive")
    regular = rows.loc[rows["game_type"].eq("R")].copy()
    if regular.empty:
        return pd.DataFrame(columns=TRACKMAN_PLATOON_COLUMNS).set_index(
            pd.MultiIndex.from_arrays([[], []], names=["pitcher_id", "batter_hand"])
        )
    overall = regular.groupby("pitcher_id", sort=False, observed=True)
    cell = regular.groupby(
        ["pitcher_id", "batter_hand"], sort=False, observed=True
    )
    cell_size = cell.size().astype(np.float64)
    output = pd.DataFrame(index=cell_size.index)
    output["e59_platoon_n_log"] = np.log1p(cell_size).astype(np.float32)
    pitcher_index = output.index.get_level_values("pitcher_id")
    global_means = regular[RICH_TRACKMAN_COLUMNS].mean()
    for metric in RICH_TRACKMAN_COLUMNS:
        overall_mean = overall[metric].mean().reindex(pitcher_index).to_numpy(
            dtype=np.float64
        )
        overall_mean = np.where(
            np.isfinite(overall_mean), overall_mean, float(global_means[metric])
        )
        stats = cell[metric].agg(["sum", "count"]).reindex(output.index)
        count = stats["count"].fillna(0.0).to_numpy(dtype=np.float64)
        total = stats["sum"].fillna(0.0).to_numpy(dtype=np.float64)
        smoothed = (total + k * overall_mean) / (count + k)
        output[f"e59_{metric}_delta"] = (smoothed - overall_mean).astype(
            np.float32
        )
    overall_size = overall.size().astype(np.float64)
    for group in RICH_PITCH_GROUPS:
        group_mask = regular["pitch_type_group"].eq(group)
        overall_rate = (
            group_mask.groupby(regular["pitcher_id"]).sum() / overall_size
        ).reindex(pitcher_index).fillna(0.0).to_numpy(dtype=np.float64)
        group_count = group_mask.groupby(
            [regular["pitcher_id"], regular["batter_hand"]]
        ).sum().reindex(output.index).fillna(0.0).to_numpy(dtype=np.float64)
        smoothed = (group_count + k * overall_rate) / (cell_size.to_numpy() + k)
        output[f"e59_{group}_rate_delta"] = (smoothed - overall_rate).astype(
            np.float32
        )
    output["e59_platoon_unseen"] = 0
    return output[TRACKMAN_PLATOON_COLUMNS]


def trackman_count_profile_table(rows: pd.DataFrame, k: float) -> pd.DataFrame:
    """Shrink pitcher-by-count pitch-mix rates to each pitcher's overall mix."""
    if k <= 0:
        raise ValueError("Trackman count smoothing k must be positive")
    regular = rows.loc[rows["game_type"].eq("R")].copy()
    index_names = ["pitcher_id", "balls_before", "strikes_before"]
    if regular.empty:
        return pd.DataFrame(columns=TRACKMAN_COUNT_COLUMNS).set_index(
            pd.MultiIndex.from_arrays([[], [], []], names=index_names)
        )

    overall = regular.groupby("pitcher_id", sort=False, observed=True)
    cell = regular.groupby(index_names, sort=False, observed=True)
    cell_size = cell.size().astype(np.float64)
    output = pd.DataFrame(index=cell_size.index)
    output["e71_count_n_log"] = np.log1p(cell_size).astype(np.float32)
    pitcher_index = output.index.get_level_values("pitcher_id")
    overall_size = overall.size().astype(np.float64)

    for group in RICH_PITCH_GROUPS:
        indicator = regular["pitch_type_group"].eq(group).astype(np.float64)
        overall_count = indicator.groupby(
            regular["pitcher_id"], sort=False, observed=True
        ).sum()
        overall_rate = (
            overall_count / overall_size
        ).reindex(pitcher_index).fillna(0.0).to_numpy(dtype=np.float64)
        cell_count = indicator.groupby(
            [regular[name] for name in index_names],
            sort=False,
            observed=True,
        ).sum().reindex(output.index).fillna(0.0).to_numpy(dtype=np.float64)
        smoothed = (cell_count + k * overall_rate) / (
            cell_size.to_numpy(dtype=np.float64) + k
        )
        output[f"e71_{group}_rate_delta"] = (
            smoothed - overall_rate
        ).astype(np.float32)

    output["e71_count_unseen"] = 0
    return output[TRACKMAN_COUNT_COLUMNS]


def profile_states_before_each_season(
    joined: pd.DataFrame, seasons: list[int], window: int | None = None
) -> tuple[dict[int, pd.DataFrame], pd.DataFrame]:
    if window is not None and window < 1:
        raise ValueError("Trackman profile window must be >= 1")
    before: dict[int, pd.DataFrame] = {}
    for season in sorted(seasons):
        rows = joined.loc[joined["season"] < season]
        if window is not None:
            rows = rows.loc[rows["season"] >= season - window]
        before[season] = profile_table(rows)
    if seasons:
        cutoff = max(seasons) + 1
        rows = joined.loc[joined["season"] < cutoff]
        if window is not None:
            rows = rows.loc[rows["season"] >= cutoff - window]
        final = profile_table(rows)
    else:
        final = profile_table(joined.iloc[:0])
    return before, final


def rich_profile_states_before_each_season(
    joined: pd.DataFrame, seasons: list[int], window: int | None = None
) -> tuple[dict[int, pd.DataFrame], pd.DataFrame]:
    if window is not None and window < 1:
        raise ValueError("Trackman profile window must be >= 1")
    before: dict[int, pd.DataFrame] = {}
    for season in sorted(seasons):
        rows = joined.loc[joined["season"] < season]
        if window is not None:
            rows = rows.loc[rows["season"] >= season - window]
        before[season] = rich_profile_table(rows)
    if seasons:
        cutoff = max(seasons) + 1
        rows = joined.loc[joined["season"] < cutoff]
        if window is not None:
            rows = rows.loc[rows["season"] >= cutoff - window]
        final = rich_profile_table(rows)
    else:
        final = rich_profile_table(joined.iloc[:0])
    return before, final


def stability_profile_states_before_each_season(
    joined: pd.DataFrame, seasons: list[int], window: int | None = None
) -> tuple[dict[int, pd.DataFrame], pd.DataFrame]:
    if window is not None and window < 1:
        raise ValueError("TrackMan profile window must be >= 1")
    before: dict[int, pd.DataFrame] = {}
    for season in sorted(seasons):
        rows = joined.loc[joined["season"] < season]
        if window is not None:
            rows = rows.loc[rows["season"] >= season - window]
        before[season] = stability_profile_table(rows)
    if seasons:
        cutoff = max(seasons) + 1
        rows = joined.loc[joined["season"] < cutoff]
        if window is not None:
            rows = rows.loc[rows["season"] >= cutoff - window]
        final = stability_profile_table(rows)
    else:
        final = stability_profile_table(joined.iloc[:0])
    return before, final


def group_stability_profile_states_before_each_season(
    joined: pd.DataFrame, seasons: list[int], window: int | None = None
) -> tuple[dict[int, pd.DataFrame], pd.DataFrame]:
    if window is not None and window < 1:
        raise ValueError("TrackMan profile window must be >= 1")
    before: dict[int, pd.DataFrame] = {}
    for season in sorted(seasons):
        rows = joined.loc[joined["season"] < season]
        if window is not None:
            rows = rows.loc[rows["season"] >= season - window]
        before[season] = group_stability_profile_table(rows)
    if seasons:
        cutoff = max(seasons) + 1
        rows = joined.loc[joined["season"] < cutoff]
        if window is not None:
            rows = rows.loc[rows["season"] >= cutoff - window]
        final = group_stability_profile_table(rows)
    else:
        final = group_stability_profile_table(joined.iloc[:0])
    return before, final


def trend_profile_table(rows: pd.DataFrame, cutoff: int, window: int) -> pd.DataFrame:
    """Return recent-minus-long TrackMan profile deltas at one season cutoff."""

    all_rows = rows.loc[rows["season"] < cutoff]
    recent_rows = all_rows.loc[all_rows["season"] >= cutoff - window]
    long = rich_profile_table(all_rows)
    recent = rich_profile_table(recent_rows)
    index = long.index.union(recent.index)
    output = pd.DataFrame(index=index)
    output.index.name = "pitcher_id"
    output["e75_recent_profile_n_log"] = recent[
        "e58_profile_n_log"
    ].reindex(index)
    for column in TREND_SOURCE_COLUMNS:
        output[f"e75_{column.removeprefix('e58_')}_delta"] = (
            recent[column].reindex(index) - long[column].reindex(index)
        )
    output["e75_recent_profile_unseen"] = np.where(
        recent["e58_profile_unseen"].reindex(index).notna(), 0, 1
    )
    return output[TREND_PROFILE_COLUMNS]


def trend_profile_states_before_each_season(
    joined: pd.DataFrame, seasons: list[int], window: int = 2
) -> tuple[dict[int, pd.DataFrame], pd.DataFrame]:
    if window < 1:
        raise ValueError("TrackMan trend window must be >= 1")
    before = {
        season: trend_profile_table(joined, season, window)
        for season in sorted(seasons)
    }
    cutoff = max(seasons) + 1 if seasons else 0
    final = (
        trend_profile_table(joined, cutoff, window)
        if seasons
        else pd.DataFrame(columns=TREND_PROFILE_COLUMNS).set_index(
            pd.Index([], name="pitcher_id")
        )
    )
    return before, final


def trackman_platoon_states_before_each_season(
    joined: pd.DataFrame,
    seasons: list[int],
    k: float,
    window: int | None = None,
) -> tuple[dict[int, pd.DataFrame], pd.DataFrame]:
    if window is not None and window < 1:
        raise ValueError("Trackman profile window must be >= 1")
    before: dict[int, pd.DataFrame] = {}
    for season in sorted(seasons):
        rows = joined.loc[joined["season"] < season]
        if window is not None:
            rows = rows.loc[rows["season"] >= season - window]
        before[season] = trackman_platoon_profile_table(rows, k)
    if seasons:
        cutoff = max(seasons) + 1
        rows = joined.loc[joined["season"] < cutoff]
        if window is not None:
            rows = rows.loc[rows["season"] >= cutoff - window]
        final = trackman_platoon_profile_table(rows, k)
    else:
        final = trackman_platoon_profile_table(joined.iloc[:0], k)
    return before, final


def trackman_count_states_before_each_season(
    joined: pd.DataFrame,
    seasons: list[int],
    k: float,
    window: int | None = None,
) -> tuple[dict[int, pd.DataFrame], pd.DataFrame]:
    """Create season-cutoff-safe pitcher-by-count pitch-mix states."""
    if window is not None and window < 1:
        raise ValueError("Trackman profile window must be >= 1")
    before: dict[int, pd.DataFrame] = {}
    for season in sorted(seasons):
        rows = joined.loc[joined["season"] < season]
        if window is not None:
            rows = rows.loc[rows["season"] >= season - window]
        before[season] = trackman_count_profile_table(rows, k)
    if seasons:
        cutoff = max(seasons) + 1
        rows = joined.loc[joined["season"] < cutoff]
        if window is not None:
            rows = rows.loc[rows["season"] >= cutoff - window]
        final = trackman_count_profile_table(rows, k)
    else:
        final = trackman_count_profile_table(joined.iloc[:0], k)
    return before, final


def build_profile_features(
    frame: pd.DataFrame, profiles_before: dict[int, pd.DataFrame]
) -> tuple[pd.DataFrame, dict[str, Any]]:
    values = np.full((len(frame), len(PROFILE_COLUMNS)), np.nan, dtype=np.float32)
    values[:, PROFILE_COLUMNS.index("e20_profile_unseen")] = 1.0
    seasons = frame["season"].to_numpy(dtype=np.int16, copy=False)
    pitchers = frame["pitcher_id"].to_numpy(dtype=np.int64, copy=False)
    for season in sorted(set(int(value) for value in seasons)):
        mask = seasons == season
        profile = profiles_before.get(season)
        if profile is None or profile.empty:
            continue
        lookup = profile.reindex(pitchers[mask])
        known = lookup["e20_profile_unseen"].notna().to_numpy(dtype=bool)
        matrix = lookup[PROFILE_COLUMNS].to_numpy(dtype=np.float32)
        indices = np.flatnonzero(mask)
        values[indices[known]] = matrix[known]
        values[indices[known], PROFILE_COLUMNS.index("e20_profile_unseen")] = 0.0
    result = pd.DataFrame(values, columns=PROFILE_COLUMNS, index=frame.index)
    metadata = {
        "unseen_rows": int((result["e20_profile_unseen"] > 0).sum()),
        "known_rows": int((result["e20_profile_unseen"] == 0).sum()),
        "profile_n_log_median_known": float(
            np.nanmedian(result.loc[result["e20_profile_unseen"] == 0, "e20_profile_n_log"])
        )
        if int((result["e20_profile_unseen"] == 0).sum())
        else 0.0,
        "cutoff": "Trackman matched regular rows with season strictly before row season",
    }
    return result, metadata


def build_rich_profile_features(
    frame: pd.DataFrame, profiles_before: dict[int, pd.DataFrame]
) -> tuple[pd.DataFrame, dict[str, Any]]:
    values = np.full(
        (len(frame), len(RICH_PROFILE_COLUMNS)), np.nan, dtype=np.float32
    )
    unseen_index = RICH_PROFILE_COLUMNS.index("e58_profile_unseen")
    values[:, unseen_index] = 1.0
    seasons = frame["season"].to_numpy(dtype=np.int16, copy=False)
    pitchers = frame["pitcher_id"].to_numpy(dtype=np.int64, copy=False)
    for season in sorted(set(int(value) for value in seasons)):
        mask = seasons == season
        profile = profiles_before.get(season)
        if profile is None or profile.empty:
            continue
        lookup = profile.reindex(pitchers[mask])
        known = lookup["e58_profile_unseen"].notna().to_numpy(dtype=bool)
        matrix = lookup[RICH_PROFILE_COLUMNS].to_numpy(dtype=np.float32)
        indices = np.flatnonzero(mask)
        values[indices[known]] = matrix[known]
        values[indices[known], unseen_index] = 0.0
    result = pd.DataFrame(values, columns=RICH_PROFILE_COLUMNS, index=frame.index)
    return result, {
        "unseen_rows": int((result["e58_profile_unseen"] > 0).sum()),
        "known_rows": int((result["e58_profile_unseen"] == 0).sum()),
        "feature_count": len(RICH_PROFILE_COLUMNS),
        "cutoff": "Trackman matched regular rows with season strictly before row season",
        "target_free": True,
    }


def build_stability_profile_features(
    frame: pd.DataFrame, profiles_before: dict[int, pd.DataFrame]
) -> tuple[pd.DataFrame, dict[str, Any]]:
    values = np.full(
        (len(frame), len(STABILITY_PROFILE_COLUMNS)), np.nan, dtype=np.float32
    )
    unseen_index = STABILITY_PROFILE_COLUMNS.index("e74_profile_unseen")
    values[:, unseen_index] = 1.0
    seasons = frame["season"].to_numpy(dtype=np.int16, copy=False)
    pitchers = frame["pitcher_id"].to_numpy(dtype=np.int64, copy=False)
    for season in sorted(set(int(value) for value in seasons)):
        mask = seasons == season
        profile = profiles_before.get(season)
        if profile is None or profile.empty:
            continue
        lookup = profile.reindex(pitchers[mask])
        known = lookup["e74_profile_unseen"].notna().to_numpy(dtype=bool)
        matrix = lookup[STABILITY_PROFILE_COLUMNS].to_numpy(dtype=np.float32)
        indices = np.flatnonzero(mask)
        values[indices[known]] = matrix[known]
        values[indices[known], unseen_index] = 0.0
    result = pd.DataFrame(
        values, columns=STABILITY_PROFILE_COLUMNS, index=frame.index
    )
    return result, {
        "unseen_rows": int((result["e74_profile_unseen"] > 0).sum()),
        "known_rows": int((result["e74_profile_unseen"] == 0).sum()),
        "feature_count": len(STABILITY_PROFILE_COLUMNS),
        "cutoff": "TrackMan regular rows with season strictly before row season",
        "target_free": True,
        "variance_decomposition": "pooled within pitch group plus between group",
    }


def build_group_stability_profile_features(
    frame: pd.DataFrame, profiles_before: dict[int, pd.DataFrame]
) -> tuple[pd.DataFrame, dict[str, Any]]:
    values = np.full(
        (len(frame), len(GROUP_STABILITY_PROFILE_COLUMNS)),
        np.nan,
        dtype=np.float32,
    )
    unseen_index = GROUP_STABILITY_PROFILE_COLUMNS.index("e76_profile_unseen")
    values[:, unseen_index] = 1.0
    seasons = frame["season"].to_numpy(dtype=np.int16, copy=False)
    pitchers = frame["pitcher_id"].to_numpy(dtype=np.int64, copy=False)
    for season in sorted(set(int(value) for value in seasons)):
        mask = seasons == season
        profile = profiles_before.get(season)
        if profile is None or profile.empty:
            continue
        lookup = profile.reindex(pitchers[mask])
        known = lookup["e76_profile_unseen"].notna().to_numpy(dtype=bool)
        matrix = lookup[GROUP_STABILITY_PROFILE_COLUMNS].to_numpy(dtype=np.float32)
        indices = np.flatnonzero(mask)
        values[indices[known]] = matrix[known]
        values[indices[known], unseen_index] = 0.0
    result = pd.DataFrame(
        values, columns=GROUP_STABILITY_PROFILE_COLUMNS, index=frame.index
    )
    return result, {
        "unseen_rows": int((result["e76_profile_unseen"] > 0).sum()),
        "known_rows": int((result["e76_profile_unseen"] == 0).sum()),
        "feature_count": len(GROUP_STABILITY_PROFILE_COLUMNS),
        "cutoff": "TrackMan regular rows with season strictly before row season",
        "target_free": True,
        "groups": list(GROUP_STABILITY_GROUPS),
    }


def build_trend_profile_features(
    frame: pd.DataFrame, profiles_before: dict[int, pd.DataFrame]
) -> tuple[pd.DataFrame, dict[str, Any]]:
    values = np.full(
        (len(frame), len(TREND_PROFILE_COLUMNS)), np.nan, dtype=np.float32
    )
    unseen_index = TREND_PROFILE_COLUMNS.index("e75_recent_profile_unseen")
    values[:, unseen_index] = 1.0
    seasons = frame["season"].to_numpy(dtype=np.int16, copy=False)
    pitchers = frame["pitcher_id"].to_numpy(dtype=np.int64, copy=False)
    for season in sorted(set(int(value) for value in seasons)):
        mask = seasons == season
        profile = profiles_before.get(season)
        if profile is None or profile.empty:
            continue
        lookup = profile.reindex(pitchers[mask])
        known = lookup["e75_recent_profile_unseen"].notna().to_numpy(dtype=bool)
        matrix = lookup[TREND_PROFILE_COLUMNS].to_numpy(dtype=np.float32)
        indices = np.flatnonzero(mask)
        values[indices[known]] = matrix[known]
        values[indices[known], unseen_index] = matrix[known, unseen_index]
    result = pd.DataFrame(values, columns=TREND_PROFILE_COLUMNS, index=frame.index)
    return result, {
        "unseen_rows": int((result["e75_recent_profile_unseen"] > 0).sum()),
        "known_rows": int((result["e75_recent_profile_unseen"] == 0).sum()),
        "feature_count": len(TREND_PROFILE_COLUMNS),
        "cutoff": "completed TrackMan seasons strictly before row season",
        "target_free": True,
    }


def build_trackman_platoon_features(
    frame: pd.DataFrame, profiles_before: dict[int, pd.DataFrame]
) -> tuple[pd.DataFrame, dict[str, Any]]:
    values = np.zeros(
        (len(frame), len(TRACKMAN_PLATOON_COLUMNS)), dtype=np.float32
    )
    unseen_index = TRACKMAN_PLATOON_COLUMNS.index("e59_platoon_unseen")
    values[:, unseen_index] = 1.0
    seasons = frame["season"].to_numpy(dtype=np.int16, copy=False)
    pitchers = frame["pitcher_id"].to_numpy(dtype=np.int64, copy=False)
    batter_hands = frame["batter_hand"].to_numpy(dtype=np.int64, copy=False)
    for season in sorted(set(int(value) for value in seasons)):
        mask = seasons == season
        profile = profiles_before.get(season)
        if profile is None or profile.empty:
            continue
        key = pd.MultiIndex.from_arrays(
            [pitchers[mask], batter_hands[mask]],
            names=["pitcher_id", "batter_hand"],
        )
        lookup = profile.reindex(key)
        known = lookup["e59_platoon_unseen"].notna().to_numpy(dtype=bool)
        matrix = lookup[TRACKMAN_PLATOON_COLUMNS].to_numpy(dtype=np.float32)
        indices = np.flatnonzero(mask)
        values[indices[known]] = matrix[known]
        values[indices[known], unseen_index] = 0.0
    result = pd.DataFrame(values, columns=TRACKMAN_PLATOON_COLUMNS, index=frame.index)
    return result, {
        "unseen_rows": int((result["e59_platoon_unseen"] > 0).sum()),
        "known_rows": int((result["e59_platoon_unseen"] == 0).sum()),
        "feature_count": len(TRACKMAN_PLATOON_COLUMNS),
        "cutoff": "Trackman matched regular rows with season strictly before row season",
        "target_free": True,
    }


def build_trackman_count_features(
    frame: pd.DataFrame, profiles_before: dict[int, pd.DataFrame]
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Look up row-independent, cutoff-safe pitcher-by-count Trackman features."""
    values = np.zeros((len(frame), len(TRACKMAN_COUNT_COLUMNS)), dtype=np.float32)
    unseen_index = TRACKMAN_COUNT_COLUMNS.index("e71_count_unseen")
    values[:, unseen_index] = 1.0
    seasons = frame["season"].to_numpy(dtype=np.int16, copy=False)
    pitchers = frame["pitcher_id"].to_numpy(dtype=np.int64, copy=False)
    balls = frame["balls_before"].to_numpy(dtype=np.int64, copy=False)
    strikes = frame["strikes_before"].to_numpy(dtype=np.int64, copy=False)
    for season in sorted(set(int(value) for value in seasons)):
        mask = seasons == season
        profile = profiles_before.get(season)
        if profile is None or profile.empty:
            continue
        key = pd.MultiIndex.from_arrays(
            [pitchers[mask], balls[mask], strikes[mask]],
            names=["pitcher_id", "balls_before", "strikes_before"],
        )
        lookup = profile.reindex(key)
        known = lookup["e71_count_unseen"].notna().to_numpy(dtype=bool)
        matrix = lookup[TRACKMAN_COUNT_COLUMNS].to_numpy(dtype=np.float32)
        indices = np.flatnonzero(mask)
        values[indices[known]] = matrix[known]
        values[indices[known], unseen_index] = 0.0
    result = pd.DataFrame(values, columns=TRACKMAN_COUNT_COLUMNS, index=frame.index)
    return result, {
        "unseen_rows": int((result["e71_count_unseen"] > 0).sum()),
        "known_rows": int((result["e71_count_unseen"] == 0).sum()),
        "feature_count": len(TRACKMAN_COUNT_COLUMNS),
        "cutoff": "Trackman regular rows with season strictly before row season",
        "target_free": True,
        "row_independent": True,
    }


def invariance(frame: pd.DataFrame, profiles: dict[int, pd.DataFrame]) -> float:
    sample = frame.iloc[: min(8, len(frame))]
    if sample.empty:
        return 0.0
    first, _ = build_profile_features(sample, profiles)
    order = list(reversed(range(len(sample))))
    second, _ = build_profile_features(sample.iloc[order], profiles)
    second = second.iloc[order]
    return float(np.max(np.abs(first.to_numpy(dtype=float) - second.to_numpy(dtype=float))))


def run_fold(
    frame: pd.DataFrame, joined: pd.DataFrame, season: int, args: argparse.Namespace
) -> dict[str, Any]:
    started = time.perf_counter()
    history = frame.loc[frame["season"] < season].copy()
    valid = frame.loc[frame["season"] == season].copy()
    prior = float(candidate_priors(history, season)[args.prior_mode])
    states_before, final_state = season_end_state(history)
    train_priors = prior_before_each_season(history)
    train_e14, e14_train_meta = build_e14_features(history, states_before, train_priors, prior, k=E14_K)
    valid_e14, e14_valid_meta = build_e14_features(valid, {season: final_state}, {season: prior}, prior, k=E14_K)
    tm_history = joined.loc[joined["season"] < season]
    all_seasons = sorted(int(value) for value in tm_history["season"].unique())
    profiles, final_profile = profile_states_before_each_season(tm_history, all_seasons)
    profiles[season] = final_profile
    train_e20, train_meta = build_profile_features(history, profiles)
    valid_e20, valid_meta = build_profile_features(valid, profiles)
    invariant_delta = invariance(valid, {season: final_profile})
    if invariant_delta >= 1e-12:
        raise AssertionError(f"E20R feature invariance failed: {invariant_delta:.3e}")
    s4_train = pd.concat([history[BASE_FEATURES], train_e14], axis=1)
    s4_valid = pd.concat([valid[BASE_FEATURES], valid_e14], axis=1)
    e20_train = pd.concat([s4_train, train_e20], axis=1)
    e20_valid = pd.concat([s4_valid, valid_e20], axis=1)
    train_y = history[TARGET].to_numpy(dtype=np.int8, copy=False)
    valid_y = valid[TARGET].to_numpy(dtype=np.int8, copy=False)
    baseline: dict[str, np.ndarray] = {}
    candidate: dict[str, np.ndarray] = {}
    details: dict[str, Any] = {"s4": {}, "e20r": {}}
    for name, factory in {"linear": make_linear, "hgb": make_hgb}.items():
        baseline[name], details["s4"][name] = fit_predict(
            f"{season}/s4/{name}", factory, s4_train, train_y, s4_valid
        )
        candidate[name], details["e20r"][name] = fit_predict(
            f"{season}/e20r/{name}", factory, e20_train, train_y, e20_valid
        )
    baseline_blend = 0.9 * baseline["linear"] + 0.1 * baseline["hgb"]
    candidate_blend = 0.9 * candidate["linear"] + 0.1 * candidate["hgb"]
    baseline_summary = metric(valid_y, baseline_blend)
    candidate_summary = metric(valid_y, candidate_blend)
    result = {
        "validation_season": season,
        "history_rows": len(history),
        "valid_rows": len(valid),
        "prior_mode": args.prior_mode,
        "baseline_s4": baseline_summary,
        "e20r_s4": candidate_summary,
        "e20r_brier_delta": float(candidate_summary["brier"] - baseline_summary["brier"]),
        "e20r_score_delta": float(candidate_summary["competition_score"] - baseline_summary["competition_score"]),
        "feature_invariance_max_abs_delta": invariant_delta,
        "e14_train": e14_train_meta,
        "e14_valid": e14_valid_meta,
        "e20_train": train_meta,
        "e20_valid": valid_meta,
        "trackman_history_rows": len(tm_history),
        "fit_details": details,
        "elapsed_seconds": time.perf_counter() - started,
    }
    print(
        f"[{season}] S4 Brier={baseline_summary['brier']:.8f}, E20R Brier={candidate_summary['brier']:.8f}, "
        f"delta={result['e20r_brier_delta']:+.8f}, score delta={result['e20r_score_delta']:+.1f}",
        flush=True,
    )
    del history, valid, train_e14, valid_e14, train_e20, valid_e20, s4_train, s4_valid, e20_train, e20_valid
    del baseline, candidate, profiles, final_profile
    gc.collect()
    return result


def main() -> None:
    args = parse_args()
    started = time.perf_counter()
    joined = load_joined_trackman()
    frame = load_train(args.data)
    folds = [run_fold(frame, joined, season, args) for season in sorted(args.validation_seasons)]
    del frame, joined
    gc.collect()
    deltas = [float(row["e20r_brier_delta"]) for row in folds]
    wins = sum(delta < 0.0 for delta in deltas)
    aggregate = {
        "folds": len(folds),
        "e20r_wins": wins,
        "mean_brier_delta": float(np.mean(deltas)),
        "worst_brier_delta": float(np.max(deltas)),
        "mean_score_delta": float(np.mean([row["e20r_score_delta"] for row in folds])),
        "gate_pass": bool(wins >= 2 and np.max(deltas) <= 0.0005),
        "gate_definition": "wins >= 2/3 and worst Brier delta <= 0.0005",
    }
    payload = {
        "metadata": {
            "created_at_utc": pd.Timestamp.utcnow().isoformat(),
            "data": str(args.data),
            "validation_seasons": sorted(args.validation_seasons),
            "prior_mode": args.prior_mode,
            "protocol": "outer history season < Y; S4 plus six historical Trackman profile features",
            "row_independent_inference": True,
            "trackman_usage": "matched historical regular rows only; no current Trackman measurements",
            "profile_features": PROFILE_COLUMNS,
            "versions": {
                "python": platform.python_version(),
                "numpy": np.__version__,
                "pandas": pd.__version__,
            },
            "command": " ".join(sys.argv),
            "elapsed_seconds": time.perf_counter() - started,
        },
        "aggregate": aggregate,
        "folds": folds,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.output_dir / "e20r_rolling.json"
    csv_path = args.output_dir / "e20r_rolling.csv"
    json_path.write_text(
        json.dumps(json_safe(payload), ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    pd.DataFrame(
        [
            {
                "validation_season": row["validation_season"],
                "baseline_s4_brier": row["baseline_s4"]["brier"],
                "e20r_s4_brier": row["e20r_s4"]["brier"],
                "e20r_brier_delta": row["e20r_brier_delta"],
                "e20r_score_delta": row["e20r_score_delta"],
                "profile_unseen_rows": row["e20_valid"]["unseen_rows"],
            }
            for row in folds
        ]
    ).to_csv(csv_path, index=False)
    print(json.dumps(aggregate, ensure_ascii=False, indent=2), flush=True)
    print(f"Saved {json_path} and {csv_path}.", flush=True)


if __name__ == "__main__":
    main()
