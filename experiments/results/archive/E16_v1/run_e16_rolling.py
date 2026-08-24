#!/usr/bin/env python3
"""Leakage-safe rolling evaluation of the E16 role/home-team context features.

E16 is deliberately small: role statistics are frozen by season from the
outer history only, while ``e16_home_team_id`` is reconstructed from the
current row's already-available team and half-inning columns.  The candidate
is compared with S4 (S2 + E14 + ``r_recent3`` prior) using the same 90:10
Linear/HGB blend and the same rolling gate as the preceding experiments.
"""

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

from experiments.run_baselines import FEATURES as BASE_FEATURES  # noqa: E402
from experiments.run_baselines import TARGET, load_train  # noqa: E402
from experiments.run_e14_rolling import (  # noqa: E402
    E14_FEATURES,
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


E16_FEATURES = [
    "e16_home_team_id",
    "e16_role_start_ratio",
    "e16_role_games_log",
    "e16_role_unseen",
]

# The existing factories use module-level category lists.  Keep the original
# lists for the S4 baseline and use an explicit extended list only while
# constructing an E16 pipeline.
from experiments import run_e14_rolling as _e14  # noqa: E402

ORIGINAL_LINEAR_CATEGORICAL = list(_e14.LINEAR_CATEGORICAL)
ORIGINAL_HGB_CATEGORICAL = list(_e14.HGB_CATEGORICAL)
E16_LINEAR_CATEGORICAL = [*ORIGINAL_LINEAR_CATEGORICAL, "e16_home_team_id"]
E16_HGB_CATEGORICAL = [*ORIGINAL_HGB_CATEGORICAL, "e16_home_team_id"]


def make_s4_linear(features: list[str]):
    _e14.LINEAR_CATEGORICAL = ORIGINAL_LINEAR_CATEGORICAL
    _e14.HGB_CATEGORICAL = ORIGINAL_HGB_CATEGORICAL
    return make_linear(features)


def make_s4_hgb(features: list[str]):
    _e14.LINEAR_CATEGORICAL = ORIGINAL_LINEAR_CATEGORICAL
    _e14.HGB_CATEGORICAL = ORIGINAL_HGB_CATEGORICAL
    return make_hgb(features)


def make_e16_linear(features: list[str]):
    _e14.LINEAR_CATEGORICAL = E16_LINEAR_CATEGORICAL
    _e14.HGB_CATEGORICAL = E16_HGB_CATEGORICAL
    return make_linear(features)


def make_e16_hgb(features: list[str]):
    _e14.LINEAR_CATEGORICAL = E16_LINEAR_CATEGORICAL
    _e14.HGB_CATEGORICAL = E16_HGB_CATEGORICAL
    return make_hgb(features)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=ROOT / "open/data/train.csv")
    parser.add_argument(
        "--validation-seasons", nargs="+", type=int, default=[2022, 2023, 2024]
    )
    parser.add_argument("--output-dir", type=Path, default=ROOT / "experiments/results")
    parser.add_argument("--prior-mode", default="r_recent3")
    return parser.parse_args()


def derive_game_ids(frame: pd.DataFrame) -> np.ndarray:
    """Reconstruct games from the same row-order keys used by structural EDA."""
    season = frame["season"].fillna(-1).to_numpy(dtype=np.int64, copy=False)
    month = frame["game_month"].fillna(-1).to_numpy(dtype=np.int64, copy=False)
    dow = frame["game_dayofweek"].fillna(-1).to_numpy(dtype=np.int64, copy=False)
    pitcher_team = frame["pitcher_team_id"].fillna(-1).to_numpy(dtype=np.int64, copy=False)
    batter_team = frame["batter_team_id"].fillna(-1).to_numpy(dtype=np.int64, copy=False)
    lo = np.minimum(pitcher_team, batter_team)
    hi = np.maximum(pitcher_team, batter_team)
    key = np.stack([season, month, dow, lo, hi], axis=1)
    half = frame["top_bottom"].eq("B").to_numpy(dtype=np.int64, na_value=False)
    progress = frame["inning"].fillna(-1).to_numpy(dtype=np.int64, copy=False) * 2 + half
    runs = frame["run_total_before"].fillna(-1).to_numpy(dtype=np.int64, copy=False)
    if not len(frame):
        return np.empty(0, dtype=np.int64)
    boundary = np.concatenate(
        [
            np.array([True]),
            np.any(key[1:] != key[:-1], axis=1)
            | (progress[1:] < progress[:-1])
            | (runs[1:] < runs[:-1]),
        ]
    )
    return boundary.cumsum(dtype=np.int64) - 1


def role_states_before_each_season(
    frame: pd.DataFrame,
) -> tuple[dict[int, dict[int, tuple[int, int]]], dict[int, tuple[int, int]]]:
    """Return pitcher (starts, games) maps before each season and after history."""
    if frame.empty:
        return {}, {}
    work = frame[["season", "pitcher_id"]].copy()
    work["_gid"] = derive_game_ids(frame)
    states_before: dict[int, dict[int, tuple[int, int]]] = {}
    state: dict[int, tuple[int, int]] = {}
    seasons = sorted(int(value) for value in work["season"].dropna().unique())
    for season in seasons:
        states_before[season] = dict(state)
        current = work.loc[work["season"] == season]
        # One row per pitcher-game gives appearances; one row per game gives
        # the first pitcher, which is the stable starter proxy.
        appearances = current.drop_duplicates(["_gid", "pitcher_id"], keep="first")
        starts = current.drop_duplicates("_gid", keep="first")["pitcher_id"].value_counts()
        games = appearances["pitcher_id"].value_counts()
        for pitcher in games.index:
            pid = int(pitcher)
            old_starts, old_games = state.get(pid, (0, 0))
            state[pid] = (
                old_starts + int(starts.get(pitcher, 0)),
                old_games + int(games.get(pitcher, 0)),
            )
    return states_before, state


def build_e16_features(
    frame: pd.DataFrame,
    role_states: dict[int, dict[int, tuple[int, int]]],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Build row-independent role and home-team features from frozen assets."""
    seasons = frame["season"].to_numpy(dtype=np.int16, copy=False)
    pitchers = frame["pitcher_id"].to_numpy(dtype=np.int64, copy=False)
    ratios = np.full(len(frame), 0.5, dtype=np.float32)
    games_log = np.zeros(len(frame), dtype=np.float32)
    unseen = np.ones(len(frame), dtype=np.int8)
    for index, (season, pitcher) in enumerate(zip(seasons, pitchers)):
        state = role_states.get(int(season), {}).get(int(pitcher))
        if state is None:
            continue
        starts, games = state
        if games > 0:
            ratios[index] = np.float32(starts / games)
            games_log[index] = np.float32(np.log1p(games))
            unseen[index] = 0

    top = frame["top_bottom"].astype("string").to_numpy()
    pitcher_team = frame["pitcher_team_id"].astype("string").to_numpy()
    batter_team = frame["batter_team_id"].astype("string").to_numpy()
    home_team = np.where(top == "T", pitcher_team, batter_team)
    features = pd.DataFrame(
        {
            "e16_home_team_id": pd.Series(home_team, index=frame.index, dtype="string"),
            "e16_role_start_ratio": ratios,
            "e16_role_games_log": games_log,
            "e16_role_unseen": unseen,
        },
        index=frame.index,
    )
    metadata = {
        "role_unseen_rows": int(unseen.sum()),
        "role_known_rows": int((unseen == 0).sum()),
        "role_start_ratio_mean": float(ratios.mean()) if len(ratios) else 0.0,
        "home_team_formula": "top_bottom == 'T' ? pitcher_team_id : batter_team_id",
        "role_formula": "first pitcher in reconstructed game is starter; map frozen before row season",
    }
    return features, metadata


def feature_invariance(
    frame: pd.DataFrame, role_states: dict[int, dict[int, tuple[int, int]]]
) -> float:
    sample = frame.iloc[: min(8, len(frame))].copy()
    if sample.empty:
        return 0.0
    reference, _ = build_e16_features(sample, role_states)
    order = list(reversed(range(len(sample))))
    shuffled, _ = build_e16_features(sample.iloc[order], role_states)
    shuffled = shuffled.iloc[order]
    numeric_ref = reference.drop(columns=["e16_home_team_id"]).to_numpy(dtype=float)
    numeric_shuf = shuffled.drop(columns=["e16_home_team_id"]).to_numpy(dtype=float)
    delta = float(np.max(np.abs(numeric_ref - numeric_shuf)))
    delta = max(
        delta,
        float(
            reference["e16_home_team_id"].astype("string").reset_index(drop=True).ne(
                shuffled["e16_home_team_id"].astype("string").reset_index(drop=True)
            ).any()
        ),
    )
    duplicated = pd.concat([sample, sample.iloc[[0]]], ignore_index=False)
    duplicate_features, _ = build_e16_features(duplicated, role_states)
    delta = max(
        delta,
        float(
            np.max(
                np.abs(
                    numeric_ref
                    - duplicate_features.iloc[: len(sample)]
                    .drop(columns=["e16_home_team_id"])
                    .to_numpy(dtype=float)
                )
            )
        ),
    )
    return delta


def run_fold(frame: pd.DataFrame, season: int, args: argparse.Namespace) -> dict[str, Any]:
    started = time.perf_counter()
    history = frame.loc[frame["season"] < season].copy()
    valid = frame.loc[frame["season"] == season].copy()
    if history.empty or valid.empty:
        raise ValueError(f"Empty fold for season {season}")

    prior = float(candidate_priors(history, season)[args.prior_mode])
    states_before, final_state = season_end_state(history)
    role_before, role_final = role_states_before_each_season(history)
    train_priors = prior_before_each_season(history)
    train_e14, train_e14_meta = build_e14_features(
        history, states_before, train_priors, prior, k=E14_K
    )
    valid_e14, valid_e14_meta = build_e14_features(
        valid, {season: final_state}, {season: prior}, prior, k=E14_K
    )
    role_before[season] = role_final
    train_e16, train_e16_meta = build_e16_features(history, role_before)
    valid_e16, valid_e16_meta = build_e16_features(valid, role_before)
    invariant_delta = feature_invariance(valid, {season: role_final})
    if invariant_delta >= 1e-12:
        raise AssertionError(f"E16 feature invariance failed: {invariant_delta:.3e}")

    s4_train = pd.concat([history[BASE_FEATURES], train_e14], axis=1)
    s4_valid = pd.concat([valid[BASE_FEATURES], valid_e14], axis=1)
    e16_train = pd.concat([s4_train, train_e16], axis=1)
    e16_valid = pd.concat([s4_valid, valid_e16], axis=1)
    train_y = history[TARGET].to_numpy(dtype=np.int8, copy=False)
    valid_y = valid[TARGET].to_numpy(dtype=np.int8, copy=False)
    baseline: dict[str, np.ndarray] = {}
    candidate: dict[str, np.ndarray] = {}
    details: dict[str, Any] = {"s4": {}, "e16": {}}
    for name, s4_factory, e16_factory in [
        ("linear", make_s4_linear, make_e16_linear),
        ("hgb", make_s4_hgb, make_e16_hgb),
    ]:
        baseline[name], details["s4"][name] = fit_predict(
            f"{season}/s4/{name}", s4_factory, s4_train, train_y, s4_valid
        )
        candidate[name], details["e16"][name] = fit_predict(
            f"{season}/e16/{name}", e16_factory, e16_train, train_y, e16_valid
        )
    baseline_blend = 0.9 * baseline["linear"] + 0.1 * baseline["hgb"]
    candidate_blend = 0.9 * candidate["linear"] + 0.1 * candidate["hgb"]
    baseline_summary = metric(valid_y, baseline_blend)
    candidate_summary = metric(valid_y, candidate_blend)
    unseen_mask = valid_e16["e16_role_unseen"].to_numpy(dtype=bool)
    known_mask = ~unseen_mask
    segments: dict[str, Any] = {}
    if unseen_mask.any():
        segments["role_unseen"] = {
            "s4": metric(valid_y[unseen_mask], baseline_blend[unseen_mask]),
            "e16": metric(valid_y[unseen_mask], candidate_blend[unseen_mask]),
        }
    if known_mask.any():
        segments["role_known"] = {
            "s4": metric(valid_y[known_mask], baseline_blend[known_mask]),
            "e16": metric(valid_y[known_mask], candidate_blend[known_mask]),
        }
    result = {
        "validation_season": season,
        "history_rows": len(history),
        "valid_rows": len(valid),
        "prior_mode": args.prior_mode,
        "history_prior": prior,
        "baseline_s4": baseline_summary,
        "e16_s4": candidate_summary,
        "e16_brier_delta": float(candidate_summary["brier"] - baseline_summary["brier"]),
        "e16_score_delta": float(
            candidate_summary["competition_score"] - baseline_summary["competition_score"]
        ),
        "feature_invariance_max_abs_delta": invariant_delta,
        "segments": segments,
        "e14_train": train_e14_meta,
        "e14_valid": valid_e14_meta,
        "e16_train": train_e16_meta,
        "e16_valid": valid_e16_meta,
        "fit_details": details,
        "elapsed_seconds": time.perf_counter() - started,
    }
    print(
        f"[{season}] S4 Brier={baseline_summary['brier']:.8f}, "
        f"E16 Brier={candidate_summary['brier']:.8f}, "
        f"delta={result['e16_brier_delta']:+.8f}, "
        f"score delta={result['e16_score_delta']:+.1f}",
        flush=True,
    )
    del history, valid, train_e14, valid_e14, train_e16, valid_e16
    del s4_train, s4_valid, e16_train, e16_valid, baseline, candidate
    gc.collect()
    return result


def main() -> None:
    args = parse_args()
    started = time.perf_counter()
    frame = load_train(args.data)
    folds = [run_fold(frame, season, args) for season in sorted(args.validation_seasons)]
    del frame
    gc.collect()
    deltas = [float(row["e16_brier_delta"]) for row in folds]
    wins = sum(delta < 0.0 for delta in deltas)
    aggregate = {
        "folds": len(folds),
        "e16_wins": wins,
        "mean_brier_delta": float(np.mean(deltas)),
        "worst_brier_delta": float(np.max(deltas)),
        "mean_score_delta": float(np.mean([row["e16_score_delta"] for row in folds])),
        "gate_pass": bool(wins >= 2 and np.max(deltas) <= 0.0005),
        "gate_definition": "wins >= 2/3 and worst Brier delta <= 0.0005",
    }
    payload = {
        "metadata": {
            "created_at_utc": pd.Timestamp.utcnow().isoformat(),
            "data": str(args.data),
            "validation_seasons": sorted(args.validation_seasons),
            "prior_mode": args.prior_mode,
            "protocol": "outer history season < Y; S4 vs S4 + frozen E16 role/home features",
            "row_independent_inference": True,
            "role_cutoff": "role counts from seasons strictly before each row season",
            "home_team_formula": "top_bottom == 'T' ? pitcher_team_id : batter_team_id",
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
    json_path = args.output_dir / "e16_rolling.json"
    csv_path = args.output_dir / "e16_rolling.csv"
    json_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8"
    )
    pd.DataFrame(
        [
            {
                "validation_season": row["validation_season"],
                "baseline_s4_brier": row["baseline_s4"]["brier"],
                "e16_s4_brier": row["e16_s4"]["brier"],
                "e16_brier_delta": row["e16_brier_delta"],
                "e16_score_delta": row["e16_score_delta"],
                "role_unseen_rows": row["e16_valid"]["role_unseen_rows"],
                "feature_invariance_max_abs_delta": row[
                    "feature_invariance_max_abs_delta"
                ],
            }
            for row in folds
        ]
    ).to_csv(csv_path, index=False)
    print(json.dumps(aggregate, ensure_ascii=False, indent=2), flush=True)
    print(f"Saved {json_path} and {csv_path}.", flush=True)


if __name__ == "__main__":
    main()
