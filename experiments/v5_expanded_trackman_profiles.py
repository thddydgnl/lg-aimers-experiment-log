#!/usr/bin/env python3
"""Build deployable pitcher profiles from the full official TrackMan log."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def _best_map(
    rows: pd.DataFrame,
    left: str,
    right: str,
    minimum_purity: float = 0.99,
    injective: bool = True,
) -> tuple[dict[Any, Any], dict[str, Any]]:
    counts = (
        rows[[left, right]].dropna()
        .groupby([left, right], sort=False, observed=True)
        .size().rename("n").reset_index()
    )
    if counts.empty:
        raise ValueError(f"cannot recover empty mapping: {left}->{right}")
    totals = counts.groupby(left, observed=True)["n"].transform("sum")
    counts["purity"] = counts["n"] / totals
    best = counts.sort_values(
        [left, "n", "purity"], ascending=[True, False, False], kind="stable"
    ).drop_duplicates(left)
    best = best.loc[best["purity"].ge(minimum_purity)].copy()
    if injective:
        best = best.sort_values(
            [right, "n", "purity"],
            ascending=[True, False, False], kind="stable",
        ).drop_duplicates(right)
    mapping = dict(zip(best[left], best[right]))
    return mapping, {
        "entities": int(len(best)),
        "minimum_purity": float(best["purity"].min()),
        "mean_purity": float(best["purity"].mean()),
        "injective": bool(len(set(mapping.values())) == len(mapping)),
    }


def _recover_batter_hand_map(
    exact: pd.DataFrame,
    raw: pd.DataFrame,
) -> tuple[dict[str, Any], dict[str, Any]]:
    raw_mode = (
        raw.dropna(subset=["batter_trackman_id", "batter_hand"])
        .groupby(["batter_trackman_id", "batter_hand"], observed=True)
        .size().rename("n").reset_index()
        .sort_values(
            ["batter_trackman_id", "n"],
            ascending=[True, False], kind="stable",
        )
        .drop_duplicates("batter_trackman_id")
        .set_index("batter_trackman_id")["batter_hand"]
    )
    pairs = exact[["batter_hand", "batter_trackman_id"]].dropna().copy()
    pairs["raw_batter_hand"] = pairs["batter_trackman_id"].map(raw_mode)
    pairs = pairs.dropna(subset=["raw_batter_hand"])
    mapping, metadata = _best_map(
        pairs, "raw_batter_hand", "batter_hand", 0.99, True
    )
    return {str(key): value for key, value in mapping.items()}, metadata


def build_expanded_trackman_profile_source(
    exact_joined: pd.DataFrame,
    raw_trackman: pd.DataFrame,
    allowed_seasons: list[int],
    minimum_purity: float = 0.99,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Map full historical major-league TrackMan rows to anonymous pitchers.

    Exact linkage supplies identity only.  An unmatched game is never treated
    as a main-table game and contributes solely to completed-history aggregate
    profiles.
    """
    allowed = sorted(int(value) for value in allowed_seasons)
    exact = exact_joined.loc[
        exact_joined["season"].isin(allowed)
        & exact_joined["game_type"].eq("R")
    ].copy()
    raw = raw_trackman.loc[raw_trackman["season"].isin(allowed)].copy()
    pitcher_map, pitcher_meta = _best_map(
        exact, "pitcher_id", "pitcher_trackman_id", minimum_purity, True
    )
    inverse_pitcher = {
        trackman_id: int(pitcher_id)
        for pitcher_id, trackman_id in pitcher_map.items()
    }
    team_codes = sorted(
        set(exact["pitcher_team"].dropna().astype(str))
        | set(exact["batter_team"].dropna().astype(str))
    )
    hand_map, hand_meta = _recover_batter_hand_map(exact, raw)
    major = raw.loc[
        raw["pitcher_trackman_id"].isin(inverse_pitcher)
        & raw["pitcher_team"].astype(str).isin(team_codes)
        & raw["batter_team"].astype(str).isin(team_codes)
    ].copy()
    major["pitcher_id"] = major["pitcher_trackman_id"].map(
        inverse_pitcher
    ).astype(np.int64)
    major["batter_hand"] = major["batter_hand"].astype(str).map(hand_map)
    missing_hand = int(major["batter_hand"].isna().sum())
    if missing_hand:
        raise ValueError(
            f"expanded TrackMan has unmapped batter-hand rows: {missing_hand}"
        )
    major["game_type"] = "R"
    exact_rows = int(len(exact))
    metadata = {
        "allowed_history_seasons": allowed,
        "identity": {"pitcher": pitcher_meta, "batter_hand": hand_meta},
        "major_team_codes": team_codes,
        "major_team_code_count": int(len(team_codes)),
        "raw_history_rows": int(len(raw)),
        "exact_regular_rows": exact_rows,
        "expanded_major_rows": int(len(major)),
        "row_expansion_factor": float(len(major) / max(1, exact_rows)),
        "mapped_pitchers": int(len(inverse_pitcher)),
        "unmapped_batter_hand_rows": missing_hand,
        "unmatched_game_claimed_as_main_row": False,
        "profile_aggregation_only": True,
        "target_columns_read": False,
        "current_validation_trackman_used": False,
        "external_data_used": False,
    }
    return major, metadata

