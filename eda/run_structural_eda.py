#!/usr/bin/env python3
"""Structural EDA for the LG Aimers 9th competition data.

`run_eda.py` answers "what do the columns look like".  This script answers
"what generated the rows".  It reconstructs the latent structure that the
anonymised main table still leaks:

  1. row order  -> game boundaries, chronological ordering, per-entity counters
  2. games      -> an exact 1:1 match against `trackman_history.csv`
  3. that match -> deterministic pitcher / batter / team / hand code recovery
  4. asof_*     -> integer event counts, hidden failure category, and the
                   season-to-date statistics that survive into the 2025 test set

Unlike `run_eda.py` this script uses numpy/pandas (already pinned in
`requirements-baseline.txt`) because the game matching and the pitch level
join are not practical in pure stdlib.  Run it with the project venv:

    .venv/bin/python eda/run_structural_eda.py

Nothing here writes to `open/`; the raw CSVs are only read.
"""

from __future__ import annotations

import hashlib
import html
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "open" / "data"
RESULT_DIR = ROOT / "eda" / "results"
FIGURE_DIR = ROOT / "eda" / "figures"

TRAIN_PATH = DATA_DIR / "train.csv"
TEST_PATH = DATA_DIR / "test.csv"
TRACKMAN_PATH = DATA_DIR / "trackman_history.csv"

# balls 0-3, strikes 0-2, outs 0-2, half 0/1, inning clipped to 1-15
STATE_CODES = 1152


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------
def state_code(inning, half, balls, strikes, outs) -> np.ndarray:
    inning = np.clip(inning, 1, 15)
    return ((((inning * 2 + half) * 4 + balls) * 3 + strikes) * 3 + outs).astype(np.int64)


def game_fingerprint(game_index: np.ndarray, n_games: int, codes: np.ndarray) -> np.ndarray:
    """Order independent fingerprint of the multiset of pitch states in a game."""
    counts = np.bincount(game_index * STATE_CODES + codes, minlength=n_games * STATE_CODES)
    counts = counts.reshape(n_games, STATE_CODES).astype(np.int32)
    return np.array([hashlib.md5(row.tobytes()).hexdigest()[:20] for row in counts])


def brier(pred: np.ndarray, y: np.ndarray) -> float:
    return float(np.mean((np.clip(pred, 0.0, 1.0) - y) ** 2))


def competition_score(pred: np.ndarray, y: np.ndarray) -> float:
    rate = float(np.mean(y))
    return max(0.0, 100000.0 * (1.0 - brier(pred, y) / (rate * (1.0 - rate))))


def empirical_bayes(successes: np.ndarray, n: np.ndarray, k: float, prior: float) -> np.ndarray:
    return (successes + k * prior) / (n + k)


# --------------------------------------------------------------------------
# 1. row order -> games
# --------------------------------------------------------------------------
def load_train() -> pd.DataFrame:
    train = pd.read_csv(TRAIN_PATH)
    lo = np.minimum(train["pitcher_team_id"], train["batter_team_id"]).to_numpy()
    hi = np.maximum(train["pitcher_team_id"], train["batter_team_id"]).to_numpy()
    key = np.stack([train["season"], train["game_month"], train["game_dayofweek"], lo, hi], axis=1)
    half = (train["top_bottom"] == "B").to_numpy().astype(np.int64)
    progress = train["inning"].to_numpy() * 2 + half
    runs = train["run_total_before"].to_numpy()

    # A new game starts when the (date-ish, team pair) key changes, or when the
    # within-game monotone counters go backwards.
    boundary = np.concatenate(
        [
            [True],
            np.any(key[1:] != key[:-1], axis=1)
            | (progress[1:] < progress[:-1])
            | (runs[1:] < runs[:-1]),
        ]
    )
    train["gid"] = boundary.cumsum() - 1
    train["half"] = half
    train["state"] = state_code(
        train["inning"].to_numpy(),
        half,
        train["balls_before"].to_numpy(),
        train["strikes_before"].to_numpy(),
        train["outs_before"].to_numpy(),
    )
    # integer event counts recovered from the rounded rate columns
    train["p_succ"] = np.round(
        train["asof_pitcher_success_rate"].fillna(0.0) * train["asof_pitcher_n"]
    )
    train["b_succ"] = np.round(
        train["asof_batter_success_rate"].fillna(0.0) * train["asof_batter_n"]
    )
    return train


def load_trackman() -> pd.DataFrame:
    trackman = pd.read_csv(TRACKMAN_PATH)
    clean = trackman[
        (trackman["balls_before"] <= 3)
        & (trackman["strikes_before"] <= 2)
        & (trackman["outs_before"] <= 2)
    ].copy()
    clean = clean.sort_values(["trackman_game_id", "pitch_no"], kind="stable").reset_index(drop=True)
    codes, uniques = pd.factorize(clean["trackman_game_id"])
    clean["g"] = codes
    clean["state"] = state_code(
        clean["inning"].to_numpy(),
        (clean["top_bottom"] == "Bottom").to_numpy().astype(np.int64),
        clean["balls_before"].to_numpy(),
        clean["strikes_before"].to_numpy(),
        clean["outs_before"].to_numpy(),
    )
    clean["venue"] = clean["trackman_game_id"].str.split("-").str[1]
    return clean, uniques, len(trackman)


def structure_section(train: pd.DataFrame) -> dict[str, Any]:
    per_pitcher = train.groupby("pitcher_id").cumcount()
    per_batter = train.groupby("batter_id").cumcount()
    games = train.groupby("gid").agg(
        season=("season", "first"),
        game_type=("game_type", "first"),
        month=("game_month", "first"),
        pitches=("season", "size"),
        max_inning=("inning", "max"),
        pitchers=("pitcher_id", "nunique"),
    )
    by_season = (
        games.groupby(["season", "game_type"]).size().unstack(fill_value=0).astype(int)
    )
    return {
        "row_order_is_chronological": {
            "asof_pitcher_n_equals_cumcount": bool((per_pitcher == train["asof_pitcher_n"]).all()),
            "asof_batter_n_equals_cumcount": bool((per_batter == train["asof_batter_n"]).all()),
        },
        "games_total": int(train["gid"].nunique()),
        "games_by_season_and_type": {
            str(season): {col: int(by_season.loc[season, col]) for col in by_season.columns}
            for season in by_season.index
        },
        "pitches_per_game_percentiles": {
            str(q): float(np.percentile(games["pitches"], q)) for q in (1, 25, 50, 75, 99)
        },
        "max_inning_counts": {str(k): int(v) for k, v in games["max_inning"].value_counts().items()},
        "pitchers_per_game_mean": float(games["pitchers"].mean()),
        "games_per_season_month": {
            str(season): {
                str(month): int(count)
                for month, count in games[games["season"] == season]["month"].value_counts().items()
            }
            for season in sorted(games["season"].unique())
        },
    }


# --------------------------------------------------------------------------
# 2/3. trackman linkage
# --------------------------------------------------------------------------
def linkage_section(train: pd.DataFrame, trackman: pd.DataFrame, n_games_tm: int) -> tuple[dict[str, Any], pd.DataFrame]:
    n_games_tr = int(train["gid"].max()) + 1
    train_sig = pd.DataFrame(
        {
            "gid": np.arange(n_games_tr),
            "sig": game_fingerprint(train["gid"].to_numpy(), n_games_tr, train["state"].to_numpy()),
        }
    )
    tm_sig = pd.DataFrame(
        {
            "g": np.arange(n_games_tm),
            "sig": game_fingerprint(trackman["g"].to_numpy(), n_games_tm, trackman["state"].to_numpy()),
        }
    )
    merged = train_sig.merge(tm_sig, on="sig")
    unique = merged[~merged["gid"].duplicated(keep=False) & ~merged["g"].duplicated(keep=False)]
    game_map = dict(zip(unique["gid"], unique["g"]))

    left = train[train["gid"].isin(game_map)].copy()
    left["g"] = left["gid"].map(game_map)
    left = left.sort_values("g", kind="stable").reset_index(drop=True)
    right = (
        trackman[trackman["g"].isin(set(unique["g"]))]
        .sort_values(["g", "pitch_no"], kind="stable")
        .reset_index(drop=True)
    )
    elementwise = float(np.mean(left["state"].to_numpy() == right["state"].to_numpy()))

    def recover(left_col: str, right_col: str) -> dict[str, Any]:
        table = pd.crosstab(left[left_col], right[right_col].to_numpy())
        top = table.max(axis=1)
        total = table.sum(axis=1)
        best = table.idxmax(axis=1)
        return {
            "entities": int(len(table)),
            "purity_mean": float((top / total).mean()),
            "purity_median": float((top / total).median()),
            "purity_is_1": int(((top / total) == 1.0).sum()),
            "purity_ge_099": int(((top / total) >= 0.99).sum()),
            "distinct_targets": int(best.nunique()),
            "injective": bool(best.nunique() == len(table)),
            "map": {str(k): (int(v) if isinstance(v, (int, np.integer)) else str(v)) for k, v in best.items()},
        }

    pitcher = recover("pitcher_id", "pitcher_trackman_id")
    batter = recover("batter_id", "batter_trackman_id")
    team = recover("pitcher_team_id", "pitcher_team")
    hand_p = pd.crosstab(left["pitcher_hand"], right["pitcher_hand"].to_numpy())
    hand_b = pd.crosstab(left["batter_hand"], right["batter_hand"].to_numpy())

    mapped_pitchers = set(pitcher["map"].keys())
    mapped_batters = set(batter["map"].keys())
    coverage = {
        "rows_with_mapped_pitcher": float(train["pitcher_id"].astype(str).isin(mapped_pitchers).mean()),
        "rows_with_mapped_batter": float(train["batter_id"].astype(str).isin(mapped_batters).mean()),
        "by_season": {
            str(season): float(
                train[train["season"] == season]["pitcher_id"].astype(str).isin(mapped_pitchers).mean()
            )
            for season in sorted(train["season"].unique())
        },
        "by_game_type": {
            str(gt): float(train[train["game_type"] == gt]["pitcher_id"].astype(str).isin(mapped_pitchers).mean())
            for gt in sorted(train["game_type"].unique())
        },
    }

    joined = pd.concat(
        [
            left.reset_index(drop=True),
            right.drop(columns=["g", "state", "season", "game_month", "game_dayofweek", "inning",
                                "top_bottom", "balls_before", "strikes_before", "outs_before",
                                "pitcher_hand", "batter_hand"]).reset_index(drop=True),
        ],
        axis=1,
    )

    venue_by_type = pd.crosstab(joined["venue"], joined["game_type"])
    section = {
        "train_games": n_games_tr,
        "trackman_games": n_games_tm,
        "signature_matched_pairs": int(len(merged)),
        "unambiguous_one_to_one": int(len(unique)),
        "match_rate_of_train_games": float(len(unique) / n_games_tr),
        "aligned_rows": int(len(left)),
        "elementwise_state_agreement": elementwise,
        "matched_games_by_type": {
            str(k): int(v) for k, v in joined.groupby("game_type")["gid"].nunique().items()
        },
        "pitcher_id_recovery": pitcher,
        "batter_id_recovery": batter,
        "team_id_recovery": team,
        "hand_code_pitcher": {str(i): {str(c): int(hand_p.loc[i, c]) for c in hand_p.columns} for i in hand_p.index},
        "hand_code_batter": {str(i): {str(c): int(hand_b.loc[i, c]) for c in hand_b.columns} for i in hand_b.index},
        "coverage": coverage,
        "venue_by_game_type": {
            str(v): {str(c): int(venue_by_type.loc[v, c]) for c in venue_by_type.columns}
            for v in venue_by_type.index
        },
        "game_dates_recovered": int(joined["gid"].nunique()),
    }
    return section, joined


# --------------------------------------------------------------------------
# 4. label taxonomy and asof_* arithmetic
# --------------------------------------------------------------------------
def label_section(train: pd.DataFrame, joined: pd.DataFrame) -> dict[str, Any]:
    warm = train["asof_pitcher_n"] > 0
    success = train.loc[warm, "asof_pitcher_success_rate"].to_numpy()
    reverse = train.loc[warm, "asof_pitcher_reverse_rate"].to_numpy()
    middle = train.loc[warm, "asof_pitcher_middle_rate"].to_numpy()
    ball = train.loc[warm, "asof_pitcher_ball_rate"].to_numpy()
    strike = train.loc[warm, "asof_pitcher_strike_rate"].to_numpy()
    failure = 1.0 - success

    # does the running success count differ by exactly the previous label?
    nxt = train.groupby("pitcher_id")["p_succ"].shift(-1)
    known = nxt.notna()
    delta_ok = float((nxt[known] - train.loc[known, "p_succ"] == train.loc[known, "control_success"]).mean())

    by_group = joined.groupby("pitch_type_group")["control_success"].agg(["size", "mean"])
    physical = {}
    regular = joined[joined["game_type"] == "R"]
    sizes = regular.groupby("pitcher_trackman_id").size()
    heavy = regular[regular["pitcher_trackman_id"].isin(sizes[sizes >= 300].index)]
    for column in ["rel_speed", "spin_rate", "induced_vert_break", "horz_break",
                   "extension", "rel_height", "rel_side", "zone_speed"]:
        present = heavy[heavy[column].notna()]
        pitcher_mean = present.groupby("pitcher_trackman_id")[column].mean()
        pitcher_rate = present.groupby("pitcher_trackman_id")["control_success"].mean()
        within_x = present[column] - present["pitcher_trackman_id"].map(pitcher_mean)
        within_y = present["control_success"] - present["pitcher_trackman_id"].map(pitcher_rate)
        physical[column] = {
            "r_pooled": float(np.corrcoef(present[column], present["control_success"])[0, 1]),
            "r_between_pitcher": float(np.corrcoef(pitcher_mean, pitcher_rate)[0, 1]),
            "r_within_pitcher": float(np.corrcoef(within_x, within_y)[0, 1]),
        }

    return {
        "failure_taxonomy": {
            "reverse_le_failure": float(np.mean(reverse <= failure + 1e-6)),
            "middle_le_failure": float(np.mean(middle <= failure + 1e-6)),
            "reverse_plus_middle_eq_failure": float(np.mean(np.abs(reverse + middle - failure) < 1e-6)),
            "reverse_plus_middle_gt_failure": float(np.mean(reverse + middle > failure + 1e-6)),
            "hidden_residual_mean": float(np.mean(failure - reverse - middle)),
            "hidden_residual_share_of_failures": float(
                np.mean(failure - reverse - middle) / np.mean(failure)
            ),
            "ball_plus_strike_le_1": float(np.mean(ball + strike <= 1.0 + 1e-6)),
            "ball_strike_residual_mean": float(np.mean(1.0 - ball - strike)),
        },
        "integer_count_recovery": {
            "successor_delta_equals_label": delta_ok,
        },
        "pitch_type_group_effect": {
            str(k): {"rows": int(by_group.loc[k, "size"]), "success_rate": float(by_group.loc[k, "mean"])}
            for k in by_group.index
        },
        "physical_between_vs_within": physical,
        "pitchers_used_for_between_within": int((sizes >= 300).sum()),
    }


# --------------------------------------------------------------------------
# 5. drift, ceilings, season-to-date recovery
# --------------------------------------------------------------------------
def drift_section(train: pd.DataFrame) -> dict[str, Any]:
    regular = train[train["game_type"] == "R"]
    decomposition = {}
    previous = None
    for season in sorted(regular["season"].unique()):
        current = regular[regular["season"] == season]
        if previous is not None:
            a = previous.groupby("pitcher_id")["control_success"].agg(["size", "mean"])
            b = current.groupby("pitcher_id")["control_success"].agg(["size", "mean"])
            joined = a.join(b, lsuffix="_a", rsuffix="_b", how="inner")
            joined = joined[(joined["size_a"] >= 200) & (joined["size_b"] >= 200)]
            weight = joined["size_b"] / joined["size_b"].sum()
            within = float((weight * (joined["mean_b"] - joined["mean_a"])).sum())
            total = float(current["control_success"].mean() - previous["control_success"].mean())
            decomposition[f"{season - 1}->{season}"] = {
                "total": total,
                "within_pitcher": within,
                "composition": total - within,
            }
        previous = current

    seen_p: set = set()
    seen_b: set = set()
    carryover = {}
    for season in sorted(train["season"].unique()):
        current = train[train["season"] == season]
        if seen_p:
            carryover[str(season)] = {
                "pitcher_seen_before": float(current["pitcher_id"].isin(seen_p).mean()),
                "batter_seen_before": float(current["batter_id"].isin(seen_b).mean()),
            }
        seen_p |= set(current["pitcher_id"])
        seen_b |= set(current["batter_id"])

    rates = train.groupby("season")["control_success"].mean()
    years = np.array(sorted(train["season"].unique()), dtype=float)
    extrapolation = {
        "all_seasons_linear_2025": float(np.polyval(np.polyfit(years, rates.values, 1), 2025)),
        "last3_linear_2025": float(np.polyval(np.polyfit(years[-3:], rates.values[-3:], 1), 2025)),
    }
    for gt in sorted(train["game_type"].unique()):
        sub = train[train["game_type"] == gt].groupby("season")["control_success"].mean()
        extrapolation[f"{gt}_by_season"] = {str(k): float(v) for k, v in sub.items()}
        recent = sub.values[-3:]
        extrapolation[f"{gt}_last3_linear_2025"] = float(
            np.polyval(np.polyfit(years[-3:], recent, 1), 2025)
        )

    reference = float(rates.iloc[-1] * (1 - rates.iloc[-1]))
    sensitivity = {
        f"{int(e * 10000) / 100:.2f}pp": float(100000 * e * e / reference)
        for e in (0.002, 0.005, 0.010, 0.015, 0.020, 0.030)
    }
    return {
        "within_vs_composition_regular_only": decomposition,
        "entity_carryover_by_season": carryover,
        "season_rates": {str(k): float(v) for k, v in rates.items()},
        "base_rate_extrapolation": extrapolation,
        "base_rate_error_cost_points": sensitivity,
    }


def ceiling_section(train: pd.DataFrame) -> dict[str, Any]:
    d24 = train[train["season"] == 2024].reset_index(drop=True)
    y = d24["control_success"].to_numpy()
    rate = float(y.mean())
    reference = rate * (1 - rate)
    rng = np.random.default_rng(0)
    half = rng.random(len(d24)) < 0.5

    def split_half(keys: list[str], k: float) -> float:
        a, b = d24[half], d24[~half]
        grouped = a.groupby(keys)["control_success"].agg(["sum", "size"])
        mu = float(a["control_success"].mean())
        shrunk = (grouped["sum"] + k * mu) / (grouped["size"] + k)
        pred = b.set_index(keys).index.map(shrunk).to_numpy(dtype=float)
        pred = np.where(np.isnan(pred), mu, pred)
        bs = float(np.mean((pred - b["control_success"].to_numpy()) ** 2))
        return max(0.0, 100000.0 * (1 - bs / reference))

    ceilings = {}
    for keys, label in [
        (["pitcher_id"], "pitcher"),
        (["batter_id"], "batter"),
        (["pitcher_id", "batter_hand"], "pitcher x batter_hand"),
        (["pitcher_id", "balls_before", "strikes_before"], "pitcher x count"),
        (["pitcher_id", "batter_id"], "pitcher x batter raw pair"),
    ]:
        ceilings[label] = {f"eb_k={k}": split_half(keys, k) for k in (0, 50, 200)}

    prior_season = train[train["season"] == 2023].groupby("pitcher_id")["control_success"].agg(["sum", "size"])
    mu23 = float(train[train["season"] == 2023]["control_success"].mean())
    out_of_time = {}
    for k in (0, 50, 200, 500):
        shrunk = (prior_season["sum"] + k * mu23) / (prior_season["size"] + k)
        pred = d24["pitcher_id"].map(shrunk).fillna(mu23).to_numpy()
        out_of_time[f"eb_k={k}"] = competition_score(pred, y)
        recentred = pred - pred.mean() + rate
        out_of_time[f"eb_k={k}_recentred"] = competition_score(recentred, y)
    return {
        "split_half_within_2024": ceilings,
        "out_of_time_2023_pitcher_rate_on_2024": out_of_time,
        "climatology_brier_2024": reference,
    }


def season_to_date_section(train: pd.DataFrame) -> dict[str, Any]:
    """The single most consequential derived feature: per-row season-to-date
    history, reconstructed from the row's own asof_* values plus a per-entity
    constant frozen from the training cutoff."""
    frozen = train[train["season"] <= 2023].groupby("pitcher_id").tail(1)
    end_state = pd.DataFrame(
        {
            "pid": frozen["pitcher_id"].to_numpy(),
            "n_end": frozen["asof_pitcher_n"].to_numpy() + 1,
            "s_end": frozen["p_succ"].to_numpy() + frozen["control_success"].to_numpy(),
        }
    ).set_index("pid")

    d = train[train["season"] == 2024].copy()
    d["n_end"] = d["pitcher_id"].map(end_state["n_end"]).fillna(0.0)
    d["s_end"] = d["pitcher_id"].map(end_state["s_end"]).fillna(0.0)
    d["n_season"] = d["asof_pitcher_n"] - d["n_end"]
    d["s_season"] = d["p_succ"] - d["s_end"]

    y = d["control_success"].to_numpy()
    rate = float(y.mean())
    mu = float(train[train["season"] <= 2023]["control_success"].mean())

    results = {"constant_train_mean": competition_score(np.full(len(d), mu), y)}
    for k in (100, 300, 1000):
        pred = empirical_bayes(d["p_succ"].to_numpy(), d["asof_pitcher_n"].to_numpy(), k, mu)
        results[f"career_asof_rate_eb_k={k}"] = competition_score(pred, y)
    for k in (50, 100, 300):
        pred = empirical_bayes(d["s_season"].to_numpy(), d["n_season"].to_numpy(), k, mu)
        results[f"season_to_date_eb_k={k}"] = competition_score(pred, y)
    pred = d["asof_pitcher_prev5_game_success_rate"].fillna(mu).to_numpy()
    results["given_prev5_game_rate"] = competition_score(pred, y)

    best = empirical_bayes(d["s_season"].to_numpy(), d["n_season"].to_numpy(), 50, mu)
    career = empirical_bayes(d["p_succ"].to_numpy(), d["asof_pitcher_n"].to_numpy(), 300, mu)
    return {
        "coverage_2024": {
            "pitcher_known_from_cutoff": float((d["n_end"] > 0).mean()),
            "season_to_date_n_gt_0": float((d["n_season"] > 0).mean()),
            "season_to_date_n_ge_200": float((d["n_season"] >= 200).mean()),
            "season_to_date_n_median": float(d.loc[d["n_season"] > 0, "n_season"].median()),
        },
        "single_feature_scores_2024": results,
        "prediction_means": {
            "season_to_date_eb_k=50": float(best.mean()),
            "career_eb_k=300": float(career.mean()),
            "actual_2024": rate,
        },
        "recentred_to_true_mean": {
            "season_to_date_eb_k=50": competition_score(best - best.mean() + rate, y),
            "career_eb_k=300": competition_score(career - career.mean() + rate, y),
        },
    }


def test_sample_section(train: pd.DataFrame) -> dict[str, Any]:
    test = pd.read_csv(TEST_PATH)
    frozen = train.groupby("pitcher_id").tail(1)
    end_state = pd.DataFrame(
        {
            "pid": frozen["pitcher_id"].to_numpy(),
            "n_end": frozen["asof_pitcher_n"].to_numpy() + 1,
            "s_end": frozen["p_succ"].to_numpy() + frozen["control_success"].to_numpy(),
        }
    ).set_index("pid")
    rows = []
    for _, row in test.iterrows():
        pid = row["pitcher_id"]
        n = row["asof_pitcher_n"]
        known = pid in end_state.index
        entry: dict[str, Any] = {
            "row_id": row["row_id"],
            "game_month": int(row["game_month"]),
            "pitcher_in_train": bool(known),
            "asof_pitcher_n": int(n),
        }
        if known:
            n_end = int(end_state.loc[pid, "n_end"])
            s_end = int(end_state.loc[pid, "s_end"])
            entry["career_n_through_2024"] = n_end
            entry["pitches_thrown_in_2025"] = int(n) - n_end
            if int(n) - n_end > 0:
                total = round(row["asof_pitcher_success_rate"] * n)
                entry["success_rate_2025_only"] = float((total - s_end) / (int(n) - n_end))
                entry["career_success_rate"] = float(row["asof_pitcher_success_rate"])
        rows.append(entry)
    return {
        "note": "row_id is not chronological in test: TEST_000001 is month 7 while TEST_000017 is month 3",
        "months_in_row_id_order": [int(m) for m in test["game_month"]],
        "rows": rows,
    }


# --------------------------------------------------------------------------
# figures
# --------------------------------------------------------------------------
def svg_document(width: int, height: int, content: str, title: str) -> str:
    return f'''<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
<rect width="100%" height="100%" fill="#FFFFFF"/>
<style>
text {{ font-family: "Apple SD Gothic Neo", "Noto Sans KR", Arial, sans-serif; fill: #172033; }}
.title {{ font-size: 22px; font-weight: 700; }}
.axis {{ font-size: 11px; fill: #667085; }}
.label {{ font-size: 12px; }}
.value {{ font-size: 11px; font-weight: 600; }}
.legend {{ font-size: 12px; }}
</style>
<text x="32" y="36" class="title">{html.escape(title)}</text>
{content}
</svg>'''


def grouped_bar_svg(rows: list[tuple[str, float]], title: str, unit: str, color: str = "#2563EB") -> str:
    width, row_height, left = 940, 30, 300
    height = 80 + row_height * len(rows)
    chart_width = width - left - 110
    maximum = max((abs(v) for _, v in rows), default=1.0) or 1.0
    parts = []
    for index, (label, value) in enumerate(rows):
        y = 60 + index * row_height
        bar = abs(value) / maximum * chart_width
        parts.append(f'<text x="{left - 10}" y="{y + 16}" text-anchor="end" class="label">{html.escape(label)}</text>')
        parts.append(f'<rect x="{left}" y="{y}" width="{bar:.2f}" height="20" rx="3" fill="{color}" opacity="0.88"/>')
        parts.append(f'<text x="{left + bar + 8:.2f}" y="{y + 15}" class="value">{value:,.0f}{unit}</text>')
    return svg_document(width, height, "\n".join(parts), title)


def between_within_svg(physical: dict[str, dict[str, float]], title: str) -> str:
    width, row_height, left = 940, 34, 210
    height = 110 + row_height * len(physical)
    chart_width = width - left - 140
    centre = left + chart_width / 2
    scale = chart_width / 2 / 0.20
    parts = [
        f'<line x1="{centre}" y1="72" x2="{centre}" y2="{height - 40}" stroke="#98A2B3" stroke-dasharray="4 4"/>',
        f'<text x="{centre}" y="{height - 22}" text-anchor="middle" class="axis">0</text>',
        f'<text x="{left}" y="{height - 22}" text-anchor="middle" class="axis">-0.20</text>',
        f'<text x="{left + chart_width}" y="{height - 22}" text-anchor="middle" class="axis">+0.20</text>',
        f'<rect x="{width - 130}" y="58" width="12" height="12" fill="#DC2626"/>',
        f'<text x="{width - 114}" y="69" class="legend">투수 간</text>',
        f'<rect x="{width - 130}" y="76" width="12" height="12" fill="#2563EB"/>',
        f'<text x="{width - 114}" y="87" class="legend">투구 내</text>',
    ]
    for index, (name, values) in enumerate(physical.items()):
        y = 78 + index * row_height
        parts.append(f'<text x="{left - 10}" y="{y + 14}" text-anchor="end" class="label">{html.escape(name)}</text>')
        for offset, key, color in ((0, "r_between_pitcher", "#DC2626"), (13, "r_within_pitcher", "#2563EB")):
            r = float(values[key])
            x0 = centre if r >= 0 else centre + r * scale
            parts.append(
                f'<rect x="{x0:.1f}" y="{y + offset}" width="{abs(r) * scale:.1f}" height="11" fill="{color}" opacity="0.9"/>'
            )
    return svg_document(width, height, "\n".join(parts), title)


def make_figures(summary: dict[str, Any]) -> list[str]:
    figures = []
    scores = summary["season_to_date"]["single_feature_scores_2024"]
    order = [
        ("주어진 asof 통산 성공률 (EB k=300)", scores["career_asof_rate_eb_k=300"]),
        ("주어진 최근 5경기 성공률", scores["given_prev5_game_rate"]),
        ("학습 평균 상수", scores["constant_train_mean"]),
        ("복원한 시즌 내 성공률 (EB k=300)", scores["season_to_date_eb_k=300"]),
        ("복원한 시즌 내 성공률 (EB k=100)", scores["season_to_date_eb_k=100"]),
        ("복원한 시즌 내 성공률 (EB k=50)", scores["season_to_date_eb_k=50"]),
    ]
    (FIGURE_DIR / "season_to_date_vs_given.svg").write_text(
        grouped_bar_svg(order, "2024 단일 피처 대회 환산 점수", "점"), encoding="utf-8"
    )
    figures.append("season_to_date_vs_given.svg")

    (FIGURE_DIR / "trackman_between_within.svg").write_text(
        between_within_svg(summary["labels"]["physical_between_vs_within"],
                           "Trackman 물리량과 Target: 투수 간 vs 투구 내 상관"),
        encoding="utf-8",
    )
    figures.append("trackman_between_within.svg")

    decomposition = summary["drift"]["within_vs_composition_regular_only"]
    rows = []
    for key, value in decomposition.items():
        rows.append((f"{key} 투수 내 변화", value["within_pitcher"] * 10000))
        rows.append((f"{key} 선수 구성 변화", value["composition"] * 10000))
    (FIGURE_DIR / "drift_within_vs_composition.svg").write_text(
        grouped_bar_svg(rows, "R 경기 시즌 간 성공률 변화 분해 (0.01%p 단위)", ""), encoding="utf-8"
    )
    figures.append("drift_within_vs_composition.svg")
    return figures


# --------------------------------------------------------------------------
def main() -> None:
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    train = load_train()
    trackman, game_ids, trackman_raw_rows = load_trackman()

    summary: dict[str, Any] = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "method": {
            "game_reconstruction": "train row order + (season, month, dayofweek, team pair) key + monotone inning/score counters",
            "trackman_match": "exact multiset fingerprint of (inning, half, balls, strikes, outs) per game, kept only when 1:1",
            "dependencies": "numpy, pandas (see requirements-baseline.txt)",
            "trackman_rows_dropped_as_invalid": int(trackman_raw_rows - len(trackman)),
        },
    }
    summary["structure"] = structure_section(train)
    summary["linkage"], joined = linkage_section(train, trackman, len(game_ids))
    summary["labels"] = label_section(train, joined)
    summary["drift"] = drift_section(train)
    summary["ceilings"] = ceiling_section(train)
    summary["season_to_date"] = season_to_date_section(train)
    summary["test_sample"] = test_sample_section(train)
    summary["figures"] = make_figures(summary)

    out = RESULT_DIR / "structural_summary.json"
    out.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"wrote {out}")
    print(f"games={summary['structure']['games_total']} "
          f"matched={summary['linkage']['unambiguous_one_to_one']} "
          f"pitchers_mapped={summary['linkage']['pitcher_id_recovery']['entities']} "
          f"coverage={summary['linkage']['coverage']['rows_with_mapped_pitcher']:.4f}")


if __name__ == "__main__":
    main()
