#!/usr/bin/env python3
"""Evaluate the preregistered row-independent R base-rate trend correction.

The target-year offset is a constant computed only from official labels in
seasons strictly before the target year.  In particular, this script never
solves an offset from the mean or any other aggregate of target predictions.
Development folds are opened sequentially; 2024 cannot be loaded unless the
immutable 2022 and 2023 reports both passed their preregistered gates.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
PREDICTIONS = ROOT / "experiments/results/predictions"
RESULTS = ROOT / "experiments/results"
PREREGISTRATION = ROOT / "experiments/params/v5_fixed_r_baserate_trend_preregister.json"
TRAIN = ROOT / "open/data/train.csv"
TARGET = "control_success"
PHI = 0.75
EPSILON = 1e-6
BOOTSTRAP_ITERATIONS = 2000
ACTUAL_ANCHOR = 1090.9100565103
HAIRCUT = 0.75
REQUIRED_RAW_GAIN = 132.11992465293324

PARENTS: dict[str, tuple[dict[int, str], str]] = {
    "exact_component_C": (
        {
            2022: "v3_sparse_c_backtest_2022.npz",
            2023: "v3_sparse_c_backtest_2023.npz",
            2024: "v3_outcome_trackmanrich_overall_e14k50_batter80_middle100_2024.npz",
        },
        "catboost_outcome",
    ),
    "honest_r_identity": (
        {
            year: f"v5_honest_m3_r_identity_{year}.npz"
            for year in (2022, 2023, 2024)
        },
        "final_prediction",
    ),
    "honest_r_grid": (
        {
            year: f"v5_honest_m3_r_grid_{year}.npz"
            for year in (2022, 2023, 2024)
        },
        "final_prediction",
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--year", type=int, choices=(2022, 2023, 2024), required=True)
    return parser.parse_args()


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        number = float(value)
        return number if np.isfinite(number) else None
    return value


def load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        return {key: np.asarray(archive[key]) for key in archive.files}


def logit(value: np.ndarray | float) -> np.ndarray | float:
    clipped = np.clip(value, EPSILON, 1.0 - EPSILON)
    return np.log(clipped / (1.0 - clipped))


def sigmoid(value: np.ndarray | float) -> np.ndarray | float:
    return 1.0 / (1.0 + np.exp(-np.asarray(value)))


def historical_offset(frame: pd.DataFrame, year: int) -> dict[str, Any]:
    history = frame.loc[(frame["season"] < year) & frame["game_type"].eq("R")].copy()
    if history.empty:
        raise ValueError(f"no R history before {year}")
    rates = history.groupby("season", sort=True)[TARGET].mean()
    if len(rates) < 3 or int(rates.index[-1]) != year - 1:
        raise ValueError(f"insufficient consecutive R history before {year}: {rates.index.tolist()}")
    differences = rates.diff().dropna()
    forecast = float(rates.iloc[-1] + PHI * differences.mean())
    recent_start = year - 3
    recent = history.loc[history["season"] >= recent_start, TARGET]
    reference_prior = float(recent.mean())
    forecast = float(np.clip(forecast, 0.02, 0.98))
    reference_prior = float(np.clip(reference_prior, 0.02, 0.98))
    offset = float(logit(forecast) - logit(reference_prior))
    return {
        "history_seasons": [int(item) for item in rates.index],
        "season_rates_R": {str(int(key)): float(value) for key, value in rates.items()},
        "annual_differences": [float(value) for value in differences.to_numpy()],
        "mean_annual_difference": float(differences.mean()),
        "phi": PHI,
        "forecast_rate": forecast,
        "reference_recent3_R_rate": reference_prior,
        "fixed_logit_offset": offset,
        "target_prediction_aggregate_used": False,
        "target_labels_used": False,
    }


def apply_offset(parent: np.ndarray, route_r: np.ndarray, offset: float) -> np.ndarray:
    prediction = np.asarray(parent, dtype=np.float64).copy()
    prediction[route_r] = sigmoid(logit(prediction[route_r]) + offset)
    return np.clip(prediction, EPSILON, 1.0 - EPSILON)


def metrics(y: np.ndarray, prediction: np.ndarray, mask: np.ndarray) -> dict[str, float | int]:
    target = np.asarray(y[mask], dtype=np.float64)
    pred = np.asarray(prediction[mask], dtype=np.float64)
    rate = float(target.mean())
    brier = float(np.mean(np.square(pred - target)))
    score = 100_000.0 * (1.0 - brier / max(rate * (1.0 - rate), 1e-12))
    return {
        "rows": int(mask.sum()),
        "target_rate": rate,
        "prediction_mean": float(pred.mean()),
        "prediction_std": float(pred.std()),
        "brier": brier,
        "raw_competition_score": score,
    }


def cluster_bootstrap_score_gain(
    y: np.ndarray,
    parent: np.ndarray,
    candidate: np.ndarray,
    cluster: np.ndarray,
    mask: np.ndarray,
    iterations: int,
    seed: int,
) -> dict[str, float | int]:
    work = pd.DataFrame(
        {
            "cluster": cluster[mask].astype(str),
            "y": y[mask].astype(np.float64),
            "parent_error": np.square(parent[mask] - y[mask]),
            "candidate_error": np.square(candidate[mask] - y[mask]),
        }
    )
    grouped = work.groupby("cluster", sort=False, observed=True).agg(
        n=("y", "size"),
        y_sum=("y", "sum"),
        parent_error=("parent_error", "sum"),
        candidate_error=("candidate_error", "sum"),
    )
    values = grouped.to_numpy(dtype=np.float64)
    rng = np.random.default_rng(seed)
    cluster_count = len(values)
    gains = np.empty(iterations, dtype=np.float64)
    for iteration in range(iterations):
        sampled = values[rng.integers(0, cluster_count, size=cluster_count)].sum(axis=0)
        n, y_sum, parent_error, candidate_error = sampled
        rate = y_sum / n
        reference = max(rate * (1.0 - rate), 1e-12)
        gains[iteration] = 100_000.0 * (
            parent_error / n - candidate_error / n
        ) / reference
    point = 100_000.0 * (
        work["parent_error"].mean() - work["candidate_error"].mean()
    ) / max(float(work["y"].mean() * (1.0 - work["y"].mean())), 1e-12)
    return {
        "point": float(point),
        "ci_low": float(np.quantile(gains, 0.025)),
        "ci_high": float(np.quantile(gains, 0.975)),
        "bootstrap_std": float(gains.std(ddof=1)),
        "iterations": int(iterations),
        "cluster_count": int(cluster_count),
    }


def report_path(year: int) -> Path:
    return RESULTS / f"v5_fixed_r_baserate_trend_{year}.json"


def preregister_sha256() -> str:
    return hashlib.sha256(PREREGISTRATION.read_bytes()).hexdigest()


def require_prior_passes(year: int, prereg_hash: str) -> list[dict[str, Any]]:
    required = [] if year == 2022 else [2022] if year == 2023 else [2022, 2023]
    reports: list[dict[str, Any]] = []
    for prior_year in required:
        path = report_path(prior_year)
        if not path.exists():
            raise FileNotFoundError(f"required prior gate report is missing: {path}")
        report = json.loads(path.read_text(encoding="utf-8"))
        if report.get("preregister_sha256") != prereg_hash:
            raise ValueError(f"preregistration hash mismatch in {path}")
        if not bool(report.get("development_gate_pass")):
            raise PermissionError(f"prior development gate did not pass: {path}")
        reports.append(report)
    return reports


def main() -> None:
    args = parse_args()
    year = int(args.year)
    output = report_path(year)
    artifact_path = PREDICTIONS / f"v5_fixed_r_baserate_trend_{year}.npz"
    if output.exists() or artifact_path.exists():
        raise FileExistsError(f"immutable result already exists for {year}")

    prereg = json.loads(PREREGISTRATION.read_text(encoding="utf-8"))
    if prereg.get("status") != "preregistered_before_development_score_computation":
        raise ValueError("unexpected preregistration status")
    prereg_hash = preregister_sha256()
    prior_reports = require_prior_passes(year, prereg_hash)

    # The rate formula receives only rows strictly before the target year.
    rate_frame = pd.read_csv(
        TRAIN,
        usecols=["season", "game_type", TARGET],
        low_memory=False,
    )
    offset_details = historical_offset(rate_frame, year)

    parents: dict[str, tuple[dict[str, np.ndarray], np.ndarray]] = {}
    reference: dict[str, np.ndarray] | None = None
    for name, (paths, key) in PARENTS.items():
        artifact = load_npz(PREDICTIONS / paths[year])
        if reference is None:
            reference = artifact
        else:
            for alignment_key in ("y", "row_index", "cluster"):
                if not np.array_equal(reference[alignment_key], artifact[alignment_key]):
                    raise ValueError(f"alignment mismatch: {name}/{alignment_key}/{year}")
        parents[name] = (artifact, np.asarray(artifact[key], dtype=np.float64))
    if reference is None:
        raise RuntimeError("no parents loaded")

    target_rows = rate_frame.loc[
        reference["row_index"].astype(np.int64), ["game_type"]
    ]
    route_r = target_rows["game_type"].to_numpy() == "R"
    masks = {
        "all": np.ones(len(route_r), dtype=bool),
        "R": route_r,
    }
    y = np.asarray(reference["y"], dtype=np.float64)
    comparisons: dict[str, Any] = {}
    candidate_predictions: dict[str, np.ndarray] = {}
    passed: dict[str, bool] = {}
    for index, (name, (artifact, parent_prediction)) in enumerate(parents.items()):
        candidate = apply_offset(
            parent_prediction,
            route_r,
            float(offset_details["fixed_logit_offset"]),
        )
        candidate_predictions[name] = candidate
        parent_metrics = {
            scope: metrics(y, parent_prediction, mask) for scope, mask in masks.items()
        }
        candidate_metrics = {
            scope: metrics(y, candidate, mask) for scope, mask in masks.items()
        }
        gains = {
            scope: float(
                candidate_metrics[scope]["raw_competition_score"]
                - parent_metrics[scope]["raw_competition_score"]
            )
            for scope in masks
        }
        bootstrap = {
            scope: cluster_bootstrap_score_gain(
                y,
                parent_prediction,
                candidate,
                artifact["cluster"].astype(str),
                mask,
                BOOTSTRAP_ITERATIONS,
                620_000 + 100 * year + 10 * index + scope_index,
            )
            for scope_index, (scope, mask) in enumerate(masks.items())
        }
        passed[name] = bool(
            gains["R"] > 0.0
            and float(bootstrap["R"]["ci_low"]) > 0.0
            and gains["all"] > 0.0
        )
        comparisons[name] = {
            "parent_metrics": parent_metrics,
            "candidate_metrics": candidate_metrics,
            "score_gains": gains,
            "pitcher_cluster_bootstrap": bootstrap,
            "development_parent_gate_pass": passed[name],
        }

    development_gate_pass = bool(all(passed.values()))
    report: dict[str, Any] = {
        "experiment_id": "V5_FIXED_R_BASERATE_TREND_V1",
        "year": year,
        "mode": "locked_confirmation" if year == 2024 else "sequential_development",
        "preregister_sha256": prereg_hash,
        "prediction_files_loaded_for_target_year_only": year,
        "prior_gate_reports": [item["year"] for item in prior_reports],
        "target_prediction_aggregate_used_to_build_offset": False,
        "target_labels_used_to_build_offset": False,
        "offset_details": offset_details,
        "route": "R_only_F_unchanged",
        "comparisons": comparisons,
        "passed_parents": passed,
        "development_gate_pass": development_gate_pass,
        "status": (
            "locked_confirmation_evaluated"
            if year == 2024
            else "eligible_for_next_fold"
            if development_gate_pass
            else "failed_development_gate_direction_closed"
        ),
    }

    if year == 2024:
        development_full_gains = [
            float(details["score_gains"]["all"])
            for prior in prior_reports
            for details in prior["comparisons"].values()
        ]
        confirmation_full_gains = [
            float(details["score_gains"]["all"])
            for details in comparisons.values()
        ]
        confirmation_full_ci_lows = [
            float(details["pitcher_cluster_bootstrap"]["all"]["ci_low"])
            for details in comparisons.values()
        ]
        g_dev = float(min(development_full_gains))
        g_confirm = float(min(confirmation_full_gains))
        g_ci = float(min(confirmation_full_ci_lows))
        g_robust = float(min(g_dev, g_confirm, g_ci))
        expected_lb_lower = float(ACTUAL_ANCHOR + HAIRCUT * max(0.0, g_robust))
        report["goal_evaluation"] = {
            "G_dev": g_dev,
            "G_confirm": g_confirm,
            "G_ci": g_ci,
            "G_robust": g_robust,
            "required_raw_robust_gain": REQUIRED_RAW_GAIN,
            "actual_anchor": ACTUAL_ANCHOR,
            "haircut": HAIRCUT,
            "expected_lb_lower": expected_lb_lower,
            "goal_pass": bool(expected_lb_lower > 1190.0),
        }

    np.savez_compressed(
        artifact_path,
        y=np.asarray(reference["y"]),
        row_index=np.asarray(reference["row_index"]),
        cluster=np.asarray(reference["cluster"]),
        route_r=route_r,
        fixed_logit_offset=np.full(len(y), float(offset_details["fixed_logit_offset"])),
        **candidate_predictions,
    )
    output.write_text(
        json.dumps(json_safe(report), ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "year": year,
                "status": report["status"],
                "offset": offset_details,
                "parents": {
                    name: {
                        "R_gain": details["score_gains"]["R"],
                        "R_ci_low": details["pitcher_cluster_bootstrap"]["R"]["ci_low"],
                        "full_gain": details["score_gains"]["all"],
                        "pass": passed[name],
                    }
                    for name, details in comparisons.items()
                },
                "goal_evaluation": report.get("goal_evaluation"),
            },
            ensure_ascii=False,
            indent=2,
            allow_nan=False,
        )
    )


if __name__ == "__main__":
    main()
