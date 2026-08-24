#!/usr/bin/env python3
"""Split-half grouping ceilings and residual base-rate bias for EXPERIMENT_PLAN_V2 section 1.6.

Every number in section 1.6 of EXPERIMENT_PLAN_V2.md comes from this script.
It is a diagnostic probe, not an experiment: nothing here produces a submission
candidate, and no artifact is written.  The split-half protocol estimates group
means on one random half of a season and scores them on the other half, so the
reported values are honest within-season ceilings with no in-sample overfit.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

RANDOM_SEED = 42
PROBE_COLUMNS = [
    "season",
    "game_type",
    "control_success",
    "pitcher_id",
    "batter_hand",
    "pitcher_hand",
    "balls_before",
    "strikes_before",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=ROOT / "open/data/train.csv")
    parser.add_argument("--season", type=int, default=2024)
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def competition_score(y: np.ndarray, prediction: np.ndarray) -> float:
    rate = float(y.mean())
    reference = rate * (1.0 - rate)
    brier = float(np.mean(np.square(prediction - y)))
    return float(max(0.0, 100_000.0 * (1.0 - brier / reference)))


def empirical_bayes(train: pd.DataFrame, keys: list[str], prior: float, k: float) -> pd.Series:
    grouped = train.groupby(keys, observed=True)["control_success"].agg(["sum", "size"])
    return (grouped["sum"] + k * prior) / (grouped["size"] + k)


def lookup(frame: pd.DataFrame, keys: list[str], table: pd.Series, fallback: float) -> np.ndarray:
    index = frame[keys[0]] if len(keys) == 1 else frame.set_index(keys).index
    values = np.asarray(index.map(table), dtype=np.float64)
    return np.where(np.isnan(values), fallback, values)


def main() -> None:
    args = parse_args()
    frame = pd.read_csv(args.data, usecols=PROBE_COLUMNS, encoding="utf-8-sig")
    season = frame.loc[frame["season"] == args.season].copy()
    if season.empty:
        raise ValueError(f"No rows for season {args.season}")

    season["count_state"] = (
        season["balls_before"].astype(str) + "-" + season["strikes_before"].astype(str)
    )
    season["hand_pair"] = (
        season["pitcher_hand"].astype(str) + "-" + season["batter_hand"].astype(str)
    )
    season["count_hand"] = season["count_state"] + "|" + season["hand_pair"]
    season["pitcher_hand_cell"] = (
        season["pitcher_id"].astype(str) + "|" + season["batter_hand"].astype(str)
    )

    generator = np.random.default_rng(args.seed)
    estimate_mask = generator.random(len(season)) < 0.5
    estimate = season.loc[estimate_mask]
    score_half = season.loc[~estimate_mask]
    y = score_half["control_success"].to_numpy(dtype=np.float64)
    prior = float(estimate["control_success"].mean())

    report: dict = {
        "season": args.season,
        "seed": args.seed,
        "estimate_rows": int(len(estimate)),
        "score_rows": int(len(score_half)),
        "score_half_target_rate": float(y.mean()),
        "estimate_half_prior": prior,
    }

    # --- Check 1: situational vs handedness groupings -----------------------
    situational: dict[str, float] = {}
    for name, keys in (
        ("count_state", ["count_state"]),
        ("game_type", ["game_type"]),
        ("pitcher_hand x batter_hand", ["hand_pair"]),
        ("count x hand", ["count_hand"]),
    ):
        table = empirical_bayes(estimate, keys, prior, 50.0)
        situational[name] = competition_score(y, lookup(score_half, keys, table, prior))
    report["check1_situational_ceilings_k50"] = situational

    # --- Check 2: pitcher identity and the platoon split --------------------
    identity: dict[str, dict[str, float]] = {}
    for name, keys in (
        ("pitcher_id", ["pitcher_id"]),
        ("pitcher_id x batter_hand", ["pitcher_hand_cell"]),
    ):
        identity[name] = {
            f"k={k:g}": competition_score(
                y, lookup(score_half, keys, empirical_bayes(estimate, keys, prior, k), prior)
            )
            for k in (0.0, 50.0, 200.0, 500.0)
        }
    report["check2_identity_ceilings"] = identity

    # Marginal contribution: fit the pitcher main effect first, then shrink the
    # platoon residual separately toward zero.  This is the form section 4.1
    # implements, so it is the number that matters for B1'.
    marginal: dict[str, dict[str, float]] = {}
    for k_pitcher in (200.0, 500.0):
        pitcher_table = empirical_bayes(estimate, ["pitcher_id"], prior, k_pitcher)
        base = lookup(score_half, ["pitcher_id"], pitcher_table, prior)
        base_score = competition_score(y, base)

        residual_frame = estimate.copy()
        residual_frame["base"] = lookup(estimate, ["pitcher_id"], pitcher_table, prior)
        residual_frame["residual"] = residual_frame["control_success"] - residual_frame["base"]
        counts = residual_frame.groupby("pitcher_hand_cell", observed=True)["residual"].agg(
            ["sum", "size"]
        )
        entry = {"pitcher_only": base_score}
        for k_platoon in (50.0, 200.0, 500.0):
            delta_table = counts["sum"] / (counts["size"] + k_platoon)
            delta = lookup(score_half, ["pitcher_hand_cell"], delta_table, 0.0)
            blended = np.clip(base + delta, 1e-6, 1.0 - 1e-6)
            entry[f"plus_platoon_k={k_platoon:g}"] = competition_score(y, blended)
        marginal[f"k_pitcher={k_pitcher:g}"] = entry
    report["check2_platoon_marginal"] = marginal

    # --- Check 3: season-level base rate trend ------------------------------
    regular = frame.loc[frame["game_type"] == "R"]
    by_season = regular.groupby("season")["control_success"].mean()
    seasons = by_season.index.to_numpy(dtype=np.float64)
    rates = by_season.to_numpy(dtype=np.float64)
    trend = {}
    for window in (3, 4, 5, 6):
        slope, intercept = np.polyfit(seasons[-window:], rates[-window:], 1)
        trend[f"linfit_last_{window}"] = {
            "predicted_2025": float(slope * 2025 + intercept),
            "slope_per_year": float(slope),
        }
    report["check3_regular_season_rate"] = {
        "by_season": {int(s): float(v) for s, v in by_season.items()},
        "year_over_year_delta": [float(v) for v in np.diff(rates)],
        "r_recent3": float(by_season.loc[[2022, 2023, 2024]].mean()),
        "r_recent2": float(by_season.loc[[2023, 2024]].mean()),
        "r_last": float(by_season.loc[2024]),
        "extrapolation": trend,
    }

    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False),
            encoding="utf-8",
        )
        print(f"Saved {args.output}.", flush=True)


if __name__ == "__main__":
    main()
