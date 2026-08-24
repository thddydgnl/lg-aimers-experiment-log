#!/usr/bin/env python3
"""Source-only auxiliary audit for exact recent-workload decoding."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.run_e16_rolling import derive_game_ids  # noqa: E402
from experiments.run_v2_rolling import build_recent_denominator_features  # noqa: E402

TRAIN = ROOT / "open/data/train.csv"
PREREG = ROOT / "experiments/params/v5_recent_workload_decoder_preregister.json"
REPORT = ROOT / "experiments/results/v5_recent_workload_decoder_source.json"
YEARS = (2020, 2021)
HORIZONS = (1, 3, 5)


def sha256(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [safe(item) for item in value]
    if isinstance(value, (np.floating, float)):
        number = float(value)
        return number if np.isfinite(number) else None
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    return value


def load_rows() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    columns = [
        "season", "game_month", "game_dayofweek", "pitcher_team_id",
        "batter_team_id", "top_bottom", "inning", "run_total_before",
        "pitcher_id", "game_type", "asof_pitcher_success_rate",
    ]
    for horizon in HORIZONS:
        columns.extend([
            f"asof_pitcher_prev{horizon}_game_success_rate",
            f"asof_pitcher_prev{horizon}_game_middle_rate",
        ])
    frame = pd.read_csv(TRAIN, usecols=columns, encoding="utf-8-sig")
    frame["_game_id"] = derive_game_ids(frame)
    frame["_row_index"] = np.arange(len(frame), dtype=np.int64)
    appearances = (
        frame.groupby(["_game_id", "pitcher_id"], sort=False, observed=True)
        .agg(
            first_row=("_row_index", "min"),
            appearance_pitches=("_row_index", "size"),
            season=("season", "first"),
            game_type=("game_type", "first"),
        )
        .reset_index()
        .sort_values("first_row", kind="mergesort")
        .reset_index(drop=True)
    )
    for horizon in HORIZONS:
        appearances[f"true_prev{horizon}_n"] = appearances.groupby(
            "pitcher_id", sort=False, observed=True
        )["appearance_pitches"].transform(
            lambda values, h=horizon: values.shift(1).rolling(
                h, min_periods=h
            ).sum()
        )
    representatives = frame.iloc[appearances["first_row"].to_numpy()].reset_index(
        drop=True
    )
    lower = build_recent_denominator_features(representatives).reset_index(drop=True)
    return frame, appearances, pd.concat([representatives, lower], axis=1)


def aggregate_table(rows: pd.DataFrame) -> pd.DataFrame:
    grouped = rows.groupby(["pitcher_id", "game_type"], observed=True)[
        "appearance_pitches"
    ]
    result = grouped.agg(["count", "sum", "mean", "std"])
    quantiles = grouped.quantile([0.25, 0.5, 0.75]).unstack(-1)
    quantiles.columns = ["q25", "q50", "q75"]
    return result.join(quantiles).fillna(0.0)


def overall_table(rows: pd.DataFrame) -> pd.DataFrame:
    grouped = rows.groupby("pitcher_id", observed=True)["appearance_pitches"]
    result = grouped.agg(["count", "sum", "mean", "std"])
    quantiles = grouped.quantile([0.25, 0.5, 0.75]).unstack(-1)
    quantiles.columns = ["q25", "q50", "q75"]
    return result.join(quantiles).fillna(0.0)


PROFILE_STATS = ("count", "mean", "std", "q25", "q50", "q75")


def profile_features(
    samples: pd.DataFrame,
    history: pd.DataFrame,
    leave_one_out: bool,
) -> pd.DataFrame:
    by_type = aggregate_table(history)
    overall = overall_table(history)
    global_mean = float(history["appearance_pitches"].mean())
    global_std = float(history["appearance_pitches"].std(ddof=0))
    output = pd.DataFrame(index=samples.index)
    type_key = pd.MultiIndex.from_arrays(
        [samples["pitcher_id"], samples["game_type"]],
        names=["pitcher_id", "game_type"],
    )
    typed = by_type.reindex(type_key).reset_index(drop=True)
    all_pitcher = overall.reindex(samples["pitcher_id"].to_numpy()).reset_index(
        drop=True
    )
    if leave_one_out:
        own = samples["appearance_pitches"].to_numpy(dtype=np.float64)
        count = typed["count"].fillna(0.0).to_numpy(dtype=np.float64)
        total = typed["sum"].fillna(0.0).to_numpy(dtype=np.float64)
        loo_count = np.maximum(count - 1.0, 0.0)
        typed["count"] = loo_count
        typed["mean"] = np.divide(
            total - own,
            loo_count,
            out=np.full(len(samples), np.nan),
            where=loo_count > 0,
        )
        count_all = all_pitcher["count"].fillna(0.0).to_numpy(dtype=np.float64)
        total_all = all_pitcher["sum"].fillna(0.0).to_numpy(dtype=np.float64)
        loo_all = np.maximum(count_all - 1.0, 0.0)
        all_pitcher["count"] = loo_all
        all_pitcher["mean"] = np.divide(
            total_all - own,
            loo_all,
            out=np.full(len(samples), np.nan),
            where=loo_all > 0,
        )
    for prefix, table in (("type", typed), ("all", all_pitcher)):
        for statistic in PROFILE_STATS:
            fallback = global_mean if statistic not in {"count", "std"} else 0.0
            if statistic == "std":
                fallback = global_std
            output[f"profile_{prefix}_{statistic}"] = table[statistic].fillna(
                fallback
            ).to_numpy(dtype=np.float32)
    output["profile_global_mean"] = np.float32(global_mean)
    output["profile_global_std"] = np.float32(global_std)
    return output


def model_features(
    rows: pd.DataFrame,
    profiles: pd.DataFrame,
) -> tuple[pd.DataFrame, list[str]]:
    columns: dict[str, Any] = {
        "pitcher_id": rows["pitcher_id"].astype("string").fillna("__unknown__"),
        "game_type": rows["game_type"].astype("string").fillna("__unknown__"),
    }
    for horizon in HORIZONS:
        for suffix in ("success_rate", "middle_rate"):
            source = f"asof_pitcher_prev{horizon}_game_{suffix}"
            columns[source] = pd.to_numeric(rows[source], errors="coerce").fillna(-1.0)
        for suffix in (
            "n_lower", "success_count_lower", "middle_count_lower",
            "boundary_ambiguous",
        ):
            source = f"e74_prev{horizon}_{suffix}"
            columns[source] = pd.to_numeric(rows[source], errors="coerce").fillna(0.0)
    output = pd.DataFrame(columns, index=rows.index)
    for column in profiles.columns:
        output[column] = profiles[column].to_numpy()
    return output, ["pitcher_id", "game_type"]


def summarize_decoder(
    truth: np.ndarray,
    baseline: np.ndarray,
    candidate: np.ndarray,
    weights: np.ndarray,
) -> dict[str, Any]:
    valid = np.isfinite(truth) & (baseline > 0)
    truth = truth[valid]
    baseline = baseline[valid]
    candidate = candidate[valid]
    weights = weights[valid]
    base_exact = float(np.average(baseline == truth, weights=weights))
    candidate_exact = float(np.average(candidate == truth, weights=weights))
    base_mae = float(np.average(np.abs(baseline - truth), weights=weights))
    candidate_mae = float(np.average(np.abs(candidate - truth), weights=weights))
    return {
        "appearances": int(valid.sum()),
        "represented_pitch_rows": int(weights.sum()),
        "lower_divides_true_rate": float(np.average(
            np.mod(truth, baseline) == 0.0, weights=weights
        )),
        "baseline_exact_accuracy": base_exact,
        "candidate_exact_accuracy": candidate_exact,
        "exact_accuracy_improvement": candidate_exact - base_exact,
        "baseline_mae": base_mae,
        "candidate_mae": candidate_mae,
        "mae_reduction_fraction": (
            (base_mae - candidate_mae) / base_mae if base_mae > 0 else 0.0
        ),
    }


def main() -> None:
    if REPORT.exists():
        raise FileExistsError(f"immutable report exists: {REPORT}")
    prereg = json.loads(PREREG.read_text(encoding="utf-8"))
    if prereg["status"] != "locked_before_source_decoder_metrics":
        raise ValueError("unexpected preregistration state")
    started = time.perf_counter()
    _, appearances, rows = load_rows()
    folds: dict[int, Any] = {}
    all_checks: list[bool] = []
    fitted_classes: dict[int, dict[int, list[int]]] = {}
    for year in YEARS:
        history_mask = appearances["season"].lt(year)
        valid_mask = appearances["season"].eq(year)
        history = appearances.loc[history_mask].copy().reset_index(drop=True)
        valid = appearances.loc[valid_mask].copy().reset_index(drop=True)
        history_rows = rows.loc[history_mask].copy().reset_index(drop=True)
        valid_rows = rows.loc[valid_mask].copy().reset_index(drop=True)
        history_profiles = profile_features(history, history, True)
        valid_profiles = profile_features(valid, history, False)
        train_x, categorical = model_features(history_rows, history_profiles)
        valid_x, valid_categorical = model_features(valid_rows, valid_profiles)
        if categorical != valid_categorical:
            raise AssertionError("decoder schema mismatch")
        fold: dict[str, Any] = {}
        fitted_classes[year] = {}
        for horizon in HORIZONS:
            lower_column = f"e74_prev{horizon}_n_lower"
            truth_column = f"true_prev{horizon}_n"
            train_truth = history[truth_column].to_numpy(dtype=np.float64)
            train_lower = history_rows[lower_column].to_numpy(dtype=np.float64)
            usable = np.isfinite(train_truth) & (train_lower > 0.0)
            multiplier = np.rint(train_truth[usable] / train_lower[usable]).astype(str)
            params = prereg["model"]
            model = CatBoostClassifier(
                loss_function=str(params["loss_function"]),
                iterations=int(params["iterations"]),
                depth=int(params["depth"]),
                learning_rate=float(params["learning_rate"]),
                l2_leaf_reg=float(params["l2_leaf_reg"]),
                random_seed=int(params["random_seed"]) + year * 10 + horizon,
                random_strength=float(params["random_strength"]),
                bootstrap_type=str(params["bootstrap_type"]),
                bagging_temperature=float(params["bagging_temperature"]),
                allow_writing_files=False,
                verbose=False,
                thread_count=6,
                task_type=(
                    "GPU"
                    if os.environ.get("V2_BOOSTER_DEVICE", "cpu").lower() == "gpu"
                    else "CPU"
                ),
            )
            model.fit(
                train_x.loc[usable],
                multiplier,
                cat_features=categorical,
            )
            classes = np.rint(
                np.asarray(model.classes_).astype(np.float64)
            ).astype(int)
            probability = np.asarray(model.predict_proba(valid_x), dtype=np.float64)
            selected_multiplier = classes[np.argmax(probability, axis=1)]
            valid_lower = valid_rows[lower_column].to_numpy(dtype=np.float64)
            maximum = int(params["maximum_denominator"][str(horizon)])
            prediction = np.minimum(
                valid_lower * selected_multiplier,
                np.floor(maximum / np.maximum(valid_lower, 1.0)) * valid_lower,
            )
            prediction[valid_lower <= 0.0] = valid_lower[valid_lower <= 0.0]
            metrics = summarize_decoder(
                valid[truth_column].to_numpy(dtype=np.float64),
                valid_lower,
                prediction,
                valid["appearance_pitches"].to_numpy(dtype=np.float64),
            )
            requirements = prereg["source_protocol"]["gate"]
            checks = {
                "accuracy": metrics["exact_accuracy_improvement"] >= float(
                    requirements[
                        "minimum_row_weighted_exact_accuracy_improvement_each_horizon_each_year"
                    ]
                ),
                "mae": metrics["mae_reduction_fraction"] >= float(
                    requirements[
                        "minimum_row_weighted_mae_reduction_fraction_each_horizon_each_year"
                    ]
                ),
                "divisibility": metrics["lower_divides_true_rate"] == float(
                    requirements[
                        "decoded_lower_divides_true_rate_each_horizon_each_year"
                    ]
                ),
            }
            all_checks.extend(checks.values())
            fold[str(horizon)] = {
                "metrics": metrics,
                "checks": checks,
                "fit_appearances": int(usable.sum()),
                "classes": classes.tolist(),
            }
            fitted_classes[year][horizon] = classes.tolist()
        folds[year] = fold
    passed = bool(all(all_checks))
    report = {
        "experiment_id": prereg["experiment_id"],
        "status": "source_pass" if passed else "source_failed",
        "preregister_sha256": sha256(PREREG),
        "script_sha256": sha256(Path(__file__)),
        "train_sha256": sha256(TRAIN),
        "years_read": list(YEARS),
        "years_not_read": [2022, 2023, 2024],
        "control_success_columns_read": 0,
        "test_rows_read": 0,
        "rows": int(len(appearances)),
        "folds": folds,
        "source_gate_pass": passed,
        "decision": (
            "freeze decoder and preregister downstream exact-C ablation"
            if passed else
            "close without reading control-success metrics"
        ),
        "elapsed_seconds": time.perf_counter() - started,
        "goal_status": "active",
        "goal_completion_claimed": False,
    }
    REPORT.write_text(
        json.dumps(safe(report), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(safe(report), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
