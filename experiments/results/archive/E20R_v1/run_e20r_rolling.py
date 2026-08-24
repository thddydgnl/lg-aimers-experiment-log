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


def profile_states_before_each_season(
    joined: pd.DataFrame, seasons: list[int]
) -> tuple[dict[int, pd.DataFrame], pd.DataFrame]:
    before: dict[int, pd.DataFrame] = {}
    for season in sorted(seasons):
        before[season] = profile_table(joined.loc[joined["season"] < season])
    final = profile_table(joined.loc[joined["season"] < (max(seasons) + 1)]) if seasons else profile_table(joined.iloc[:0])
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
