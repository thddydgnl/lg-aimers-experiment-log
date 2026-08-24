#!/usr/bin/env python3
"""Evaluate the E10 regime filter: exclude pre-2022 F rows from model fitting."""

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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=ROOT / "open/data/train.csv")
    parser.add_argument(
        "--validation-seasons", nargs="+", type=int, default=[2022, 2023, 2024]
    )
    parser.add_argument("--output-dir", type=Path, default=ROOT / "experiments/results")
    parser.add_argument("--prior-mode", default="r_recent3")
    parser.add_argument("--f-cutoff-season", type=int, default=2022)
    return parser.parse_args()


def run_fold(frame: pd.DataFrame, season: int, args: argparse.Namespace) -> dict[str, Any]:
    started = time.perf_counter()
    history = frame.loc[frame["season"] < season].copy()
    valid = frame.loc[frame["season"] == season].copy()
    prior = float(candidate_priors(history, season)[args.prior_mode])
    states_before, final_state = season_end_state(history)
    train_priors = prior_before_each_season(history)
    train_e14, train_meta = build_e14_features(
        history, states_before, train_priors, prior, k=E14_K
    )
    valid_e14, valid_meta = build_e14_features(
        valid, {season: final_state}, {season: prior}, prior, k=E14_K
    )
    full_train_x = pd.concat([history[BASE_FEATURES], train_e14], axis=1)
    valid_x = pd.concat([valid[BASE_FEATURES], valid_e14], axis=1)
    exclude_mask = (history["season"] < args.f_cutoff_season) & history["game_type"].eq("F")
    filtered_train_x = full_train_x.loc[~exclude_mask]
    train_y_full = history[TARGET].to_numpy(dtype=np.int8, copy=False)
    train_y_filtered = history.loc[~exclude_mask, TARGET].to_numpy(dtype=np.int8, copy=False)
    valid_y = valid[TARGET].to_numpy(dtype=np.int8, copy=False)
    baseline: dict[str, np.ndarray] = {}
    candidate: dict[str, np.ndarray] = {}
    details: dict[str, Any] = {"baseline": {}, "e10": {}}
    for name, factory in {"linear": make_linear, "hgb": make_hgb}.items():
        baseline[name], details["baseline"][name] = fit_predict(
            f"{season}/s4/{name}", factory, full_train_x, train_y_full, valid_x
        )
        candidate[name], details["e10"][name] = fit_predict(
            f"{season}/e10/{name}",
            factory,
            filtered_train_x,
            train_y_filtered,
            valid_x,
        )
    baseline_blend = 0.9 * baseline["linear"] + 0.1 * baseline["hgb"]
    candidate_blend = 0.9 * candidate["linear"] + 0.1 * candidate["hgb"]
    baseline_summary = metric(valid_y, baseline_blend)
    candidate_summary = metric(valid_y, candidate_blend)
    f_mask = valid["game_type"].eq("F").to_numpy(dtype=bool, na_value=False)
    r_mask = valid["game_type"].eq("R").to_numpy(dtype=bool, na_value=False)
    segments = {
        "all": {"baseline": baseline_summary, "e10": candidate_summary},
        "F": {
            "baseline": metric(valid_y[f_mask], baseline_blend[f_mask]),
            "e10": metric(valid_y[f_mask], candidate_blend[f_mask]),
        },
        "R": {
            "baseline": metric(valid_y[r_mask], baseline_blend[r_mask]),
            "e10": metric(valid_y[r_mask], candidate_blend[r_mask]),
        },
    }
    result = {
        "validation_season": season,
        "history_rows": len(history),
        "valid_rows": len(valid),
        "prior_mode": args.prior_mode,
        "f_cutoff_season": args.f_cutoff_season,
        "excluded_history_rows": int(exclude_mask.sum()),
        "excluded_history_target_rate": float(
            history.loc[exclude_mask, TARGET].mean()
        )
        if int(exclude_mask.sum())
        else None,
        "baseline_s4": baseline_summary,
        "e10_s4": candidate_summary,
        "e10_brier_delta": float(candidate_summary["brier"] - baseline_summary["brier"]),
        "e10_score_delta": float(
            candidate_summary["competition_score"] - baseline_summary["competition_score"]
        ),
        "segments": segments,
        "e14_train": train_meta,
        "e14_valid": valid_meta,
        "fit_details": details,
        "elapsed_seconds": time.perf_counter() - started,
    }
    print(
        f"[{season}] S4 Brier={baseline_summary['brier']:.8f}, "
        f"E10 Brier={candidate_summary['brier']:.8f}, "
        f"delta={result['e10_brier_delta']:+.8f}, "
        f"score delta={result['e10_score_delta']:+.1f}, "
        f"excluded={int(exclude_mask.sum()):,}",
        flush=True,
    )
    del history, valid, train_e14, valid_e14, full_train_x, filtered_train_x
    gc.collect()
    return result


def main() -> None:
    args = parse_args()
    started = time.perf_counter()
    frame = load_train(args.data)
    folds = [run_fold(frame, season, args) for season in sorted(args.validation_seasons)]
    del frame
    gc.collect()
    deltas = [float(row["e10_brier_delta"]) for row in folds]
    wins = sum(delta < 0 for delta in deltas)
    aggregate = {
        "folds": len(folds),
        "e10_wins": wins,
        "mean_brier_delta": float(np.mean(deltas)),
        "worst_brier_delta": float(np.max(deltas)),
        "mean_score_delta": float(np.mean([row["e10_score_delta"] for row in folds])),
        "gate_pass": bool(wins >= 2 and np.max(deltas) <= 0.0005),
        "gate_definition": "wins >= 2/3 and worst Brier delta <= 0.0005",
    }
    payload = {
        "metadata": {
            "created_at_utc": pd.Timestamp.utcnow().isoformat(),
            "data": str(args.data),
            "validation_seasons": sorted(args.validation_seasons),
            "prior_mode": args.prior_mode,
            "f_cutoff_season": args.f_cutoff_season,
            "protocol": "S4 vs S4 trained without F rows before cutoff; validation unchanged",
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
    json_path = args.output_dir / "e10_rolling.json"
    csv_path = args.output_dir / "e10_rolling.csv"
    json_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8"
    )
    pd.DataFrame(
        [
            {
                "validation_season": row["validation_season"],
                "baseline_s4_brier": row["baseline_s4"]["brier"],
                "e10_s4_brier": row["e10_s4"]["brier"],
                "e10_brier_delta": row["e10_brier_delta"],
                "e10_score_delta": row["e10_score_delta"],
                "excluded_history_rows": row["excluded_history_rows"],
            }
            for row in folds
        ]
    ).to_csv(csv_path, index=False)
    print(json.dumps(aggregate, ensure_ascii=False, indent=2), flush=True)
    print(f"Saved {json_path} and {csv_path}.", flush=True)


if __name__ == "__main__":
    main()
