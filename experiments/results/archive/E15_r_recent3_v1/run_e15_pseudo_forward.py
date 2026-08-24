#!/usr/bin/env python3
"""Evaluate leakage-safe future base-rate extrapolation candidates."""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.run_baselines import TARGET, load_train  # noqa: E402
from experiments.run_e14_rolling import (  # noqa: E402
    E14_K,
    build_e14_features,
    metric,
    season_end_state,
    prior_before_each_season,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=ROOT / "open/data/train.csv")
    parser.add_argument(
        "--validation-seasons", nargs="+", type=int, default=[2022, 2023, 2024]
    )
    parser.add_argument("--output-dir", type=Path, default=ROOT / "experiments/results")
    return parser.parse_args()


def safe_mean(frame: pd.DataFrame, mask: pd.Series, fallback: float) -> float:
    values = frame.loc[mask, TARGET].to_numpy(dtype=float, copy=False)
    return float(values.mean()) if len(values) else float(fallback)


def candidate_priors(history: pd.DataFrame, validation_season: int) -> dict[str, float]:
    fallback = float(history[TARGET].mean())
    recent3_start = validation_season - 3
    recent2_start = validation_season - 2
    recent3 = history[history["season"] >= recent3_start]
    recent2 = history[history["season"] >= recent2_start]
    all_rate = fallback
    recent3_rate = safe_mean(history, history["season"] >= recent3_start, fallback)
    recent2_rate = safe_mean(history, history["season"] >= recent2_start, fallback)
    r_recent3 = safe_mean(
        history,
        (history["season"] >= recent3_start) & history["game_type"].eq("R"),
        recent3_rate,
    )
    r_recent2 = safe_mean(
        history,
        (history["season"] >= recent2_start) & history["game_type"].eq("R"),
        r_recent3,
    )
    f_start = max(2022, recent2_start)
    f_recent = safe_mean(
        history,
        (history["season"] >= f_start) & history["game_type"].eq("F"),
        fallback,
    )
    f_share = float(history["game_type"].eq("F").mean())
    r_f_recombined = (1.0 - f_share) * r_recent2 + f_share * f_recent
    return {
        "all_history": all_rate,
        "recent3": recent3_rate,
        "recent2": recent2_rate,
        "r_recent3": r_recent3,
        "r_recent2_f_recent": r_f_recombined,
    }


def main() -> None:
    args = parse_args()
    started = time.perf_counter()
    frame = load_train(args.data)
    candidate_rows: list[dict] = []
    downstream_rows: list[dict] = []
    for season in sorted(args.validation_seasons):
        history = frame.loc[frame["season"] < season].copy()
        valid = frame.loc[frame["season"] == season].copy()
        if history.empty or valid.empty:
            raise ValueError(f"Empty pseudo-forward fold: {season}")
        priors = candidate_priors(history, season)
        actual = float(valid[TARGET].mean())
        states_before, final_state = season_end_state(history)
        train_priors = prior_before_each_season(history)
        history_prior = float(history[TARGET].mean())
        # The E14 training representation is fixed using only its row-season
        # past.  Only the held-out prior changes across E15 candidates.
        for name, prior in priors.items():
            absolute_error = abs(prior - actual)
            squared_error = (prior - actual) ** 2
            candidate_rows.append(
                {
                    "validation_season": season,
                    "candidate": name,
                    "predicted_prior": prior,
                    "actual_target_rate": actual,
                    "absolute_error": absolute_error,
                    "squared_error": squared_error,
                    "history_prior": history_prior,
                }
            )
            valid_e14, e14_meta = build_e14_features(
                valid,
                {season: final_state},
                {season: prior},
                prior,
                k=E14_K,
            )
            standalone = valid_e14["e14_rate_season"].to_numpy(dtype=float)
            summary = metric(valid[TARGET].to_numpy(dtype=np.int8), standalone)
            downstream_rows.append(
                {
                    "validation_season": season,
                    "candidate": name,
                    "predicted_prior": prior,
                    "standalone_e14_brier": summary["brier"],
                    "standalone_e14_score": summary["competition_score"],
                    "e14_mean_prediction": summary["prediction_mean"],
                    "invalid_rows": e14_meta["invalid_rows"],
                }
            )
        print(
            f"[{season}] actual={actual:.6f}; "
            + ", ".join(f"{name}={value:.6f}" for name, value in priors.items()),
            flush=True,
        )
    candidates = pd.DataFrame(candidate_rows)
    downstream = pd.DataFrame(downstream_rows)
    aggregate = (
        candidates.groupby("candidate", as_index=False)
        .agg(
            mean_absolute_error=("absolute_error", "mean"),
            worst_absolute_error=("absolute_error", "max"),
            mean_squared_error=("squared_error", "mean"),
            folds=("validation_season", "nunique"),
        )
        .sort_values(["mean_absolute_error", "worst_absolute_error"])
    )
    downstream_aggregate = (
        downstream.groupby("candidate", as_index=False)
        .agg(
            mean_brier=("standalone_e14_brier", "mean"),
            worst_brier=("standalone_e14_brier", "max"),
            mean_score=("standalone_e14_score", "mean"),
            folds=("validation_season", "nunique"),
        )
        .sort_values("mean_brier")
    )
    payload = {
        "metadata": {
            "created_at_utc": pd.Timestamp.utcnow().isoformat(),
            "data": str(args.data),
            "validation_seasons": sorted(args.validation_seasons),
            "protocol": "prior candidate uses only seasons < validation season",
            "k": E14_K,
            "row_independent_inference": True,
            "versions": {
                "python": platform.python_version(),
                "numpy": np.__version__,
                "pandas": pd.__version__,
            },
            "elapsed_seconds": time.perf_counter() - started,
        },
        "candidate_rank_by_prior_error": aggregate.to_dict(orient="records"),
        "candidate_rank_by_standalone_e14": downstream_aggregate.to_dict(
            orient="records"
        ),
        "folds": candidate_rows,
        "downstream_folds": downstream_rows,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.output_dir / "e15_pseudo_forward.json"
    csv_path = args.output_dir / "e15_pseudo_forward.csv"
    downstream_csv = args.output_dir / "e15_pseudo_forward_downstream.csv"
    json_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8"
    )
    candidates.to_csv(csv_path, index=False)
    downstream.to_csv(downstream_csv, index=False)
    print("\nPrior error ranking:", flush=True)
    print(aggregate.to_string(index=False), flush=True)
    print("\nStandalone E14 downstream ranking:", flush=True)
    print(downstream_aggregate.to_string(index=False), flush=True)
    print(f"Saved {json_path}, {csv_path}, and {downstream_csv}.", flush=True)


if __name__ == "__main__":
    main()
