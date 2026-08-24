#!/usr/bin/env python3
"""Evaluate hierarchical empirical-Bayes pitcher shrinkage on top of S4."""

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


EB_K = 120.0
EB_FEATURES = [
    "eb_pitcher_rate",
    "eb_rate_delta",
    "eb_group_prior",
    "eb_group_seen",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=ROOT / "open/data/train.csv")
    parser.add_argument(
        "--validation-seasons", nargs="+", type=int, default=[2022, 2023, 2024]
    )
    parser.add_argument("--output-dir", type=Path, default=ROOT / "experiments/results")
    parser.add_argument("--prior-mode", default="r_recent3")
    parser.add_argument("--k", type=float, default=EB_K)
    return parser.parse_args()


def group_key(game_type: object, pitcher_hand: object) -> tuple[str, int]:
    game = "__missing__" if pd.isna(game_type) else str(game_type)
    hand = -1 if pd.isna(pitcher_hand) else int(pitcher_hand)
    return game, hand


def group_priors_before_each_season(
    frame: pd.DataFrame,
) -> tuple[dict[int, tuple[float, dict[tuple[str, int], float]]], tuple[float, dict[tuple[str, int], float]]]:
    total = 0.0
    count = 0
    sums: dict[tuple[str, int], float] = {}
    counts: dict[tuple[str, int], int] = {}
    before: dict[int, tuple[float, dict[tuple[str, int], float]]] = {}
    for season in sorted(int(value) for value in frame["season"].unique()):
        global_prior = total / count if count else 0.5
        group_map = {
            key: sums[key] / counts[key]
            for key in sums
            if counts[key]
        }
        before[season] = (global_prior, group_map)
        rows = frame.loc[frame["season"] == season, ["game_type", "pitcher_hand", TARGET]]
        for row in rows.itertuples(index=False):
            key = group_key(row.game_type, row.pitcher_hand)
            value = float(getattr(row, TARGET))
            sums[key] = sums.get(key, 0.0) + value
            counts[key] = counts.get(key, 0) + 1
            total += value
            count += 1
    final_map = {key: sums[key] / counts[key] for key in sums if counts[key]}
    final_prior = total / count if count else 0.5
    return before, (final_prior, final_map)


def build_eb_features(
    frame: pd.DataFrame,
    priors_before: dict[int, tuple[float, dict[tuple[str, int], float]]],
    validation_prior: float,
    k: float,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    n = frame["asof_pitcher_n"].fillna(0).to_numpy(dtype=np.int64, copy=False)
    rate = frame["asof_pitcher_success_rate"].fillna(validation_prior).to_numpy(
        dtype=np.float64, copy=False
    )
    successes = np.rint(rate * n).astype(np.int64)
    group_priors = np.zeros(len(frame), dtype=np.float64)
    seen = np.zeros(len(frame), dtype=np.int8)
    for index, row in enumerate(frame[["season", "game_type", "pitcher_hand"]].itertuples(index=False)):
        global_prior, group_map = priors_before.get(int(row.season), (validation_prior, {}))
        key = group_key(row.game_type, row.pitcher_hand)
        group_priors[index] = group_map.get(key, global_prior)
        seen[index] = int(key in group_map)
    eb_rate = (successes + k * group_priors) / (n + k)
    result = pd.DataFrame(
        {
            "eb_pitcher_rate": eb_rate.astype(np.float32),
            "eb_rate_delta": (eb_rate - rate).astype(np.float32),
            "eb_group_prior": group_priors.astype(np.float32),
            "eb_group_seen": seen,
        },
        index=frame.index,
    )
    return result, {
        "k": float(k),
        "group_seen_rows": int(seen.sum()),
        "group_unseen_rows": int((seen == 0).sum()),
    }


def feature_invariance(frame: pd.DataFrame, priors, validation_prior: float, k: float) -> float:
    sample = frame.iloc[: min(5, len(frame))].copy()
    if sample.empty:
        return 0.0
    reference, _ = build_eb_features(sample, priors, validation_prior, k)
    shuffled, _ = build_eb_features(sample.iloc[::-1], priors, validation_prior, k)
    shuffled = shuffled.iloc[::-1]
    return float(np.max(np.abs(reference.to_numpy(float) - shuffled.to_numpy(float))))


def run_fold(frame: pd.DataFrame, season: int, args: argparse.Namespace) -> dict[str, Any]:
    started = time.perf_counter()
    history = frame.loc[frame["season"] < season].copy()
    valid = frame.loc[frame["season"] == season].copy()
    prior = float(candidate_priors(history, season)[args.prior_mode])
    state_before, final_state = season_end_state(history)
    e14_train_priors = prior_before_each_season(history)
    train_e14, train_e14_meta = build_e14_features(
        history, state_before, e14_train_priors, prior, k=E14_K
    )
    valid_e14, valid_e14_meta = build_e14_features(
        valid, {season: final_state}, {season: prior}, prior, k=E14_K
    )
    eb_before, eb_final = group_priors_before_each_season(history)
    eb_valid_priors = {season: eb_final}
    eb_train, eb_train_meta = build_eb_features(history, eb_before, prior, args.k)
    eb_valid, eb_valid_meta = build_eb_features(valid, eb_valid_priors, prior, args.k)
    invariant_delta = feature_invariance(valid, eb_valid_priors, prior, args.k)
    if invariant_delta >= 1e-12:
        raise AssertionError(f"E11 invariance failed: {invariant_delta:.3e}")
    base_train = pd.concat([history[BASE_FEATURES], train_e14], axis=1)
    base_valid = pd.concat([valid[BASE_FEATURES], valid_e14], axis=1)
    eb_train_frame = pd.concat([base_train, eb_train], axis=1)
    eb_valid_frame = pd.concat([base_valid, eb_valid], axis=1)
    train_y = history[TARGET].to_numpy(dtype=np.int8, copy=False)
    valid_y = valid[TARGET].to_numpy(dtype=np.int8, copy=False)
    baseline: dict[str, np.ndarray] = {}
    candidate: dict[str, np.ndarray] = {}
    fit_details: dict[str, Any] = {"baseline": {}, "e11": {}}
    for name, factory in {"linear": make_linear, "hgb": make_hgb}.items():
        baseline[name], fit_details["baseline"][name] = fit_predict(
            f"{season}/s4/{name}", factory, base_train, train_y, base_valid
        )
        candidate[name], fit_details["e11"][name] = fit_predict(
            f"{season}/e11/{name}", factory, eb_train_frame, train_y, eb_valid_frame
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
        "validation_prior": prior,
        "k": args.k,
        "baseline_s4": baseline_summary,
        "e11_s4": candidate_summary,
        "e11_brier_delta": float(candidate_summary["brier"] - baseline_summary["brier"]),
        "e11_score_delta": float(
            candidate_summary["competition_score"] - baseline_summary["competition_score"]
        ),
        "feature_invariance_max_abs_delta": invariant_delta,
        "train_e14": train_e14_meta,
        "valid_e14": valid_e14_meta,
        "train_eb": eb_train_meta,
        "valid_eb": eb_valid_meta,
        "residual_correlation": float(
            np.corrcoef(valid_y - baseline_blend, valid_y - candidate_blend)[0, 1]
        ),
        "fit_details": fit_details,
        "elapsed_seconds": time.perf_counter() - started,
    }
    print(
        f"[{season}] S4 Brier={baseline_summary['brier']:.8f}, "
        f"E11 Brier={candidate_summary['brier']:.8f}, "
        f"delta={result['e11_brier_delta']:+.8f}, "
        f"score delta={result['e11_score_delta']:+.1f}",
        flush=True,
    )
    del history, valid, train_e14, valid_e14, eb_train, eb_valid
    gc.collect()
    return result


def main() -> None:
    args = parse_args()
    started = time.perf_counter()
    frame = load_train(args.data)
    folds = [run_fold(frame, season, args) for season in sorted(args.validation_seasons)]
    del frame
    gc.collect()
    deltas = [float(row["e11_brier_delta"]) for row in folds]
    wins = sum(delta < 0 for delta in deltas)
    aggregate = {
        "folds": len(folds),
        "e11_wins": wins,
        "mean_brier_delta": float(np.mean(deltas)),
        "worst_brier_delta": float(np.max(deltas)),
        "mean_score_delta": float(np.mean([row["e11_score_delta"] for row in folds])),
        "gate_pass": bool(wins >= 2 and np.max(deltas) <= 0.0005),
        "gate_definition": "wins >= 2/3 and worst Brier delta <= 0.0005",
    }
    payload = {
        "metadata": {
            "created_at_utc": pd.Timestamp.utcnow().isoformat(),
            "data": str(args.data),
            "validation_seasons": sorted(args.validation_seasons),
            "prior_mode": args.prior_mode,
            "k": args.k,
            "protocol": "S4 vs S4 + group(game_type, pitcher_hand) EB feature; all priors before fold",
            "row_independent_inference": True,
            "versions": {
                "python": platform.python_version(),
                "numpy": np.__version__,
                "pandas": pd.__version__,
            },
            "elapsed_seconds": time.perf_counter() - started,
        },
        "aggregate": aggregate,
        "folds": folds,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.output_dir / "e11_rolling.json"
    csv_path = args.output_dir / "e11_rolling.csv"
    json_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8"
    )
    pd.DataFrame(
        [
            {
                "validation_season": row["validation_season"],
                "baseline_s4_brier": row["baseline_s4"]["brier"],
                "e11_s4_brier": row["e11_s4"]["brier"],
                "e11_brier_delta": row["e11_brier_delta"],
                "e11_score_delta": row["e11_score_delta"],
                "eb_group_unseen_rows": row["valid_eb"]["group_unseen_rows"],
                "feature_invariance_max_abs_delta": row[
                    "feature_invariance_max_abs_delta"
                ],
                "residual_correlation": row["residual_correlation"],
            }
            for row in folds
        ]
    ).to_csv(csv_path, index=False)
    print(json.dumps(aggregate, ensure_ascii=False, indent=2), flush=True)
    print(f"Saved {json_path} and {csv_path}.", flush=True)


if __name__ == "__main__":
    main()
