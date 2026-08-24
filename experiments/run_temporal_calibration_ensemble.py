#!/usr/bin/env python3
"""Evaluate strictly temporal calibration and Linear-HGB ensembles.

For an outer validation season Y:

1. fit base models on seasons < Y-1;
2. predict season Y-1 and fit calibrators/ensemble weights there;
3. evaluate on season Y without using any season-Y target or aggregate.

Two deployment protocols are compared. ``frozen`` keeps the base model from
step 1, while ``refit`` retrains the base model on all seasons < Y and transfers
the calibrator learned in step 2.  The latter uses more recent data but can
change the probability scale underneath the calibrator.
"""

from __future__ import annotations

import argparse
import gc
import json
import platform
import sys
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd
import scipy
import sklearn
from scipy.optimize import brentq
from scipy.special import expit
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from run_baselines import (
    FEATURES,
    RANDOM_SEED,
    SEASON,
    TARGET,
    deterministic_sample,
    load_train,
    make_hist_gradient_boosting,
    make_linear_sgd,
    metric_summary,
    segment_metrics,
)


CALIBRATION_METHODS = (
    "logit_intercept",
    "affine_brier",
    "platt",
    "isotonic",
)
BASE_NAMES = ("linear", "hgb")
PROTOCOLS = ("frozen", "refit")
FIXED_LINEAR_WEIGHTS = (0.8, 0.9)
EPSILON = 1e-6


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("open/data/train.csv"))
    parser.add_argument(
        "--validation-seasons",
        nargs="+",
        type=int,
        default=[2022, 2023, 2024],
    )
    parser.add_argument("--output-dir", type=Path, default=Path("experiments/results"))
    parser.add_argument("--max-history-rows", type=int, default=None)
    parser.add_argument("--max-calibration-rows", type=int, default=None)
    parser.add_argument("--max-outer-train-rows", type=int, default=None)
    parser.add_argument("--max-valid-rows", type=int, default=None)
    return parser.parse_args()


def probabilities(values: np.ndarray | pd.Series) -> np.ndarray:
    result = np.asarray(values, dtype=np.float64)
    if result.ndim != 1:
        raise ValueError(f"Expected 1-D probabilities, got {result.shape}.")
    if not np.isfinite(result).all() or np.any((result < 0.0) | (result > 1.0)):
        raise ValueError("Probabilities must be finite and in [0, 1].")
    return result


def logits(prediction: np.ndarray) -> np.ndarray:
    clipped = np.clip(probabilities(prediction), EPSILON, 1.0 - EPSILON)
    return np.log(clipped) - np.log1p(-clipped)


def fit_calibrator(
    method: str,
    prediction: np.ndarray,
    target: np.ndarray,
) -> dict[str, Any]:
    p = probabilities(prediction)
    y = np.asarray(target, dtype=np.float64)
    if p.shape != y.shape:
        raise ValueError(f"Calibration shape mismatch: p={p.shape}, y={y.shape}")
    if method == "logit_intercept":
        z = logits(p)
        target_rate = float(y.mean())

        def mean_error(offset: float) -> float:
            return float(expit(z + offset).mean() - target_rate)

        offset = float(brentq(mean_error, -30.0, 30.0))
        return {"method": method, "offset": offset}

    if method == "affine_brier":
        p_centered = p - p.mean()
        denominator = float(np.dot(p_centered, p_centered))
        slope = 0.0 if denominator == 0.0 else float(
            np.dot(p_centered, y - y.mean()) / denominator
        )
        # Negative/reversed calibration is not sensible to transfer to a future
        # season; cap only extreme slopes while retaining direct Brier fitting.
        slope = float(np.clip(slope, 0.0, 3.0))
        intercept = float(y.mean() - slope * p.mean())
        return {"method": method, "intercept": intercept, "slope": slope}

    if method == "platt":
        model = LogisticRegression(C=1e6, solver="lbfgs", max_iter=300)
        model.fit(logits(p).reshape(-1, 1), y.astype(np.int8))
        return {
            "method": method,
            "intercept": float(model.intercept_[0]),
            "slope": float(model.coef_[0, 0]),
        }

    if method == "isotonic":
        model = IsotonicRegression(
            y_min=EPSILON,
            y_max=1.0 - EPSILON,
            out_of_bounds="clip",
        )
        model.fit(p, y)
        return {
            "method": method,
            "x_thresholds": model.X_thresholds_.astype(float).tolist(),
            "y_thresholds": model.y_thresholds_.astype(float).tolist(),
        }

    raise ValueError(f"Unknown calibration method: {method}")


def apply_calibrator(specification: dict[str, Any], prediction: np.ndarray) -> np.ndarray:
    p = probabilities(prediction)
    method = specification["method"]
    if method == "logit_intercept":
        calibrated = expit(logits(p) + float(specification["offset"]))
    elif method == "affine_brier":
        calibrated = (
            float(specification["intercept"])
            + float(specification["slope"]) * p
        )
    elif method == "platt":
        calibrated = expit(
            float(specification["intercept"])
            + float(specification["slope"]) * logits(p)
        )
    elif method == "isotonic":
        calibrated = np.interp(
            p,
            np.asarray(specification["x_thresholds"], dtype=np.float64),
            np.asarray(specification["y_thresholds"], dtype=np.float64),
        )
    else:
        raise ValueError(f"Unknown calibration method: {method}")
    return np.clip(np.asarray(calibrated, dtype=np.float64), 0.0, 1.0)


def calibrator_summary(
    specification: dict[str, Any],
    prediction: np.ndarray,
    target: np.ndarray,
) -> dict[str, Any]:
    output = dict(specification)
    if "x_thresholds" in output:
        output["n_thresholds"] = len(output["x_thresholds"])
    output["calibration_metrics"] = asdict(
        metric_summary(target, apply_calibrator(specification, prediction))
    )
    return output


def brier_optimal_weight(
    linear_prediction: np.ndarray,
    hgb_prediction: np.ndarray,
    target: np.ndarray,
) -> float:
    linear = probabilities(linear_prediction)
    hgb = probabilities(hgb_prediction)
    y = np.asarray(target, dtype=np.float64)
    difference = linear - hgb
    denominator = float(np.dot(difference, difference))
    if denominator == 0.0:
        return 0.5
    weight = float(np.dot(y - hgb, difference) / denominator)
    return float(np.clip(weight, 0.0, 1.0))


def blend(linear_prediction: np.ndarray, hgb_prediction: np.ndarray, weight: float) -> np.ndarray:
    return weight * probabilities(linear_prediction) + (1.0 - weight) * probabilities(
        hgb_prediction
    )


def fit_predict_pair(
    label: str,
    factory: Callable[[], Pipeline],
    train: pd.DataFrame,
    first: pd.DataFrame,
    second: pd.DataFrame | None,
) -> tuple[np.ndarray, np.ndarray | None, dict[str, Any]]:
    model = factory()
    print(f"[{label}] fit on {len(train):,} rows...", flush=True)
    started = time.perf_counter()
    model.fit(train[FEATURES], train[TARGET])
    fit_seconds = time.perf_counter() - started

    prediction_started = time.perf_counter()
    first_prediction = model.predict_proba(first[FEATURES])[:, 1]
    second_prediction = None
    if second is not None:
        second_prediction = model.predict_proba(second[FEATURES])[:, 1]
    predict_seconds = time.perf_counter() - prediction_started
    classifier = model.named_steps["clf"]
    iteration = getattr(classifier, "n_iter_", None)
    if isinstance(iteration, np.ndarray):
        iteration = int(np.max(iteration))
    elif iteration is not None:
        iteration = int(iteration)
    details = {
        "fit_seconds": fit_seconds,
        "predict_seconds": predict_seconds,
        "n_iter": iteration,
    }
    print(
        f"[{label}] fit={fit_seconds:.1f}s, predict={predict_seconds:.1f}s",
        flush=True,
    )
    del model
    gc.collect()
    return first_prediction, second_prediction, details


def make_strategy_record(
    name: str,
    protocol: str,
    family: str,
    calibration: str,
    prediction: np.ndarray,
    valid: pd.DataFrame,
    details: dict[str, Any] | None = None,
) -> dict[str, Any]:
    y = valid[TARGET].to_numpy(dtype=np.int8, copy=False)
    return {
        "name": name,
        "protocol": protocol,
        "family": family,
        "calibration": calibration,
        "metrics": asdict(metric_summary(y, prediction)),
        "segments": segment_metrics(valid, prediction),
        "details": details or {},
    }


def build_calibration_bundle(
    calibration_prediction: dict[str, np.ndarray],
    calibration_target: np.ndarray,
) -> dict[str, Any]:
    base_specs: dict[str, dict[str, Any]] = {name: {} for name in BASE_NAMES}
    for base_name in BASE_NAMES:
        for method in CALIBRATION_METHODS:
            specification = fit_calibrator(
                method,
                calibration_prediction[base_name],
                calibration_target,
            )
            base_specs[base_name][method] = calibrator_summary(
                specification,
                calibration_prediction[base_name],
                calibration_target,
            )

    raw_weight = brier_optimal_weight(
        calibration_prediction["linear"],
        calibration_prediction["hgb"],
        calibration_target,
    )
    raw_blend = blend(
        calibration_prediction["linear"],
        calibration_prediction["hgb"],
        raw_weight,
    )
    blend_specs: dict[str, dict[str, Any]] = {}
    for method in CALIBRATION_METHODS:
        specification = fit_calibrator(method, raw_blend, calibration_target)
        blend_specs[method] = calibrator_summary(
            specification,
            raw_blend,
            calibration_target,
        )

    calibrated_member_weights: dict[str, float] = {}
    for method in CALIBRATION_METHODS:
        linear_calibrated = apply_calibrator(
            base_specs["linear"][method], calibration_prediction["linear"]
        )
        hgb_calibrated = apply_calibrator(
            base_specs["hgb"][method], calibration_prediction["hgb"]
        )
        calibrated_member_weights[method] = brier_optimal_weight(
            linear_calibrated,
            hgb_calibrated,
            calibration_target,
        )

    return {
        "base": base_specs,
        "raw_ensemble_weight_linear": raw_weight,
        "raw_ensemble_calibration": blend_specs,
        "calibrated_member_weights_linear": calibrated_member_weights,
        "calibration_raw_metrics": {
            name: asdict(metric_summary(calibration_target, prediction))
            for name, prediction in calibration_prediction.items()
        },
    }


def evaluate_protocol(
    protocol: str,
    outer_prediction: dict[str, np.ndarray],
    calibration_bundle: dict[str, Any],
    valid: pd.DataFrame,
) -> dict[str, dict[str, Any]]:
    records: dict[str, dict[str, Any]] = {}

    def record(
        suffix: str,
        family: str,
        calibration: str,
        prediction: np.ndarray,
        details: dict[str, Any] | None = None,
    ) -> None:
        name = f"{protocol}__{suffix}"
        records[name] = make_strategy_record(
            name,
            protocol,
            family,
            calibration,
            prediction,
            valid,
            details,
        )

    for base_name in BASE_NAMES:
        raw = outer_prediction[base_name]
        record(f"{base_name}__raw", base_name, "none", raw)
        for method in CALIBRATION_METHODS:
            specification = calibration_bundle["base"][base_name][method]
            record(
                f"{base_name}__{method}",
                base_name,
                method,
                apply_calibrator(specification, raw),
                {"calibrator_source": "previous_season"},
            )

    linear = outer_prediction["linear"]
    hgb = outer_prediction["hgb"]
    record(
        "ensemble__raw_equal",
        "ensemble",
        "none",
        blend(linear, hgb, 0.5),
        {"linear_weight": 0.5},
    )
    for fixed_weight in FIXED_LINEAR_WEIGHTS:
        weight_label = int(round(100 * fixed_weight))
        record(
            f"ensemble__raw_fixed_l{weight_label}",
            "ensemble",
            "none",
            blend(linear, hgb, fixed_weight),
            {
                "linear_weight": fixed_weight,
                "weight_source": "fixed_rolling_development",
            },
        )
    raw_weight = float(calibration_bundle["raw_ensemble_weight_linear"])
    raw_weighted = blend(linear, hgb, raw_weight)
    record(
        "ensemble__raw_weighted",
        "ensemble",
        "none",
        raw_weighted,
        {"linear_weight": raw_weight, "weight_source": "previous_season"},
    )

    for method in CALIBRATION_METHODS:
        post_specification = calibration_bundle["raw_ensemble_calibration"][method]
        record(
            f"ensemble__post_{method}",
            "ensemble",
            f"post_{method}",
            apply_calibrator(post_specification, raw_weighted),
            {"linear_weight": raw_weight, "parameter_source": "previous_season"},
        )

        linear_calibrated = apply_calibrator(
            calibration_bundle["base"]["linear"][method], linear
        )
        hgb_calibrated = apply_calibrator(
            calibration_bundle["base"]["hgb"][method], hgb
        )
        member_weight = float(
            calibration_bundle["calibrated_member_weights_linear"][method]
        )
        record(
            f"ensemble__members_{method}",
            "ensemble",
            f"members_{method}",
            blend(linear_calibrated, hgb_calibrated, member_weight),
            {"linear_weight": member_weight, "parameter_source": "previous_season"},
        )
    return records


def fold_frames(
    frame: pd.DataFrame,
    validation_season: int,
    args: argparse.Namespace,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    calibration_season = validation_season - 1
    history = frame.loc[frame[SEASON] < calibration_season]
    calibration = frame.loc[frame[SEASON] == calibration_season]
    outer_train = frame.loc[frame[SEASON] < validation_season]
    valid = frame.loc[frame[SEASON] == validation_season]
    if any(part.empty for part in (history, calibration, outer_train, valid)):
        raise ValueError(
            f"Empty split for validation={validation_season}, "
            f"available={sorted(frame[SEASON].unique().tolist())}"
        )
    history = deterministic_sample(history, args.max_history_rows)
    calibration = deterministic_sample(calibration, args.max_calibration_rows)
    outer_train = deterministic_sample(outer_train, args.max_outer_train_rows)
    valid = deterministic_sample(valid, args.max_valid_rows)
    if int(history[SEASON].max()) >= calibration_season:
        raise AssertionError("History/calibration temporal leakage.")
    if int(outer_train[SEASON].max()) >= validation_season:
        raise AssertionError("Outer train/validation temporal leakage.")
    return history, calibration, outer_train, valid


def verify_baseline_reproduction(
    output_dir: Path,
    validation_season: int,
    strategies: dict[str, dict[str, Any]],
    debug_subsample: bool,
) -> dict[str, float] | None:
    if debug_subsample:
        return None
    baseline_path = output_dir / f"baseline_valid_{validation_season}.json"
    if not baseline_path.exists():
        return None
    baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
    comparisons = {
        "linear": (
            strategies["refit__linear__raw"]["metrics"]["brier"],
            baseline["models"]["linear_sgd"]["metrics"]["brier"],
        ),
        "hgb": (
            strategies["refit__hgb__raw"]["metrics"]["brier"],
            baseline["models"]["hist_gradient_boosting"]["metrics"]["brier"],
        ),
    }
    deltas = {name: float(got - expected) for name, (got, expected) in comparisons.items()}
    if any(not np.isclose(delta, 0.0, rtol=0.0, atol=1e-14) for delta in deltas.values()):
        raise AssertionError(f"Baseline reproduction failed: {deltas}")
    return deltas


def flatten_fold_rows(
    validation_season: int,
    strategies: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for rank, name in enumerate(
        sorted(strategies, key=lambda key: strategies[key]["metrics"]["brier"]),
        start=1,
    ):
        record = strategies[name]
        rows.append(
            {
                "validation_season": validation_season,
                "rank": rank,
                "strategy": name,
                "protocol": record["protocol"],
                "family": record["family"],
                "calibration": record["calibration"],
                **record["metrics"],
                "linear_weight": record["details"].get("linear_weight"),
            }
        )
    return rows


def aggregate_rows(all_rows: pd.DataFrame) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    for strategy, group in all_rows.groupby("strategy", sort=False):
        ordered = group.sort_values("validation_season")
        total_n = float(ordered["n"].sum())
        row: dict[str, Any] = {
            "strategy": strategy,
            "protocol": ordered["protocol"].iloc[0],
            "family": ordered["family"].iloc[0],
            "calibration": ordered["calibration"].iloc[0],
            "folds": len(ordered),
            "mean_brier": float(ordered["brier"].mean()),
            "weighted_brier": float((ordered["brier"] * ordered["n"]).sum() / total_n),
            "mean_raw_skill": float(ordered["raw_skill"].mean()),
            "worst_raw_skill": float(ordered["raw_skill"].min()),
            "positive_skill_folds": int(ordered["raw_skill"].gt(0).sum()),
            "mean_score": float(ordered["competition_score"].mean()),
            "mean_auc": float(ordered["roc_auc"].mean()),
        }
        for fold in ordered.itertuples(index=False):
            row[f"brier_{fold.validation_season}"] = float(fold.brier)
            row[f"score_{fold.validation_season}"] = float(fold.competition_score)
        records.append(row)
    return pd.DataFrame(records).sort_values(
        ["mean_brier", "worst_raw_skill"], ascending=[True, False]
    )


def run_fold(
    frame: pd.DataFrame,
    validation_season: int,
    args: argparse.Namespace,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    fold_started = time.perf_counter()
    calibration_season = validation_season - 1
    history, calibration, outer_train, valid = fold_frames(
        frame, validation_season, args
    )
    print(
        f"\n=== Outer {validation_season}: history={len(history):,} "
        f"(<{calibration_season}), calibration={len(calibration):,} "
        f"({calibration_season}), refit={len(outer_train):,}, valid={len(valid):,} ===",
        flush=True,
    )

    calibration_prediction: dict[str, np.ndarray] = {}
    frozen_prediction: dict[str, np.ndarray] = {}
    refit_prediction: dict[str, np.ndarray] = {}
    fit_details: dict[str, Any] = {"frozen": {}, "refit": {}}

    factories: dict[str, Callable[[], Pipeline]] = {
        "linear": make_linear_sgd,
        "hgb": make_hist_gradient_boosting,
    }
    for base_name in BASE_NAMES:
        cal_prediction, outer_prediction, details = fit_predict_pair(
            f"{validation_season}/{base_name}/history",
            factories[base_name],
            history,
            calibration,
            valid,
        )
        calibration_prediction[base_name] = cal_prediction
        if outer_prediction is None:
            raise AssertionError("Missing frozen outer prediction.")
        frozen_prediction[base_name] = outer_prediction
        fit_details["frozen"][base_name] = details

    calibration_target = calibration[TARGET].to_numpy(dtype=np.int8, copy=False)
    calibration_bundle = build_calibration_bundle(
        calibration_prediction, calibration_target
    )
    print(
        f"[{validation_season}] previous-season optimal raw linear weight="
        f"{calibration_bundle['raw_ensemble_weight_linear']:.4f}",
        flush=True,
    )

    for base_name in BASE_NAMES:
        outer_prediction, _, details = fit_predict_pair(
            f"{validation_season}/{base_name}/refit",
            factories[base_name],
            outer_train,
            valid,
            None,
        )
        refit_prediction[base_name] = outer_prediction
        fit_details["refit"][base_name] = details

    strategies: dict[str, dict[str, Any]] = {}
    strategies.update(
        evaluate_protocol("frozen", frozen_prediction, calibration_bundle, valid)
    )
    strategies.update(
        evaluate_protocol("refit", refit_prediction, calibration_bundle, valid)
    )
    ranking = sorted(strategies, key=lambda key: strategies[key]["metrics"]["brier"])
    debug_subsample = any(
        value is not None
        for value in (
            args.max_history_rows,
            args.max_calibration_rows,
            args.max_outer_train_rows,
            args.max_valid_rows,
        )
    )
    reproduction_delta = verify_baseline_reproduction(
        args.output_dir,
        validation_season,
        strategies,
        debug_subsample,
    )
    payload = {
        "metadata": {
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "validation_season": validation_season,
            "calibration_season": calibration_season,
            "protocol": (
                f"fit season < {calibration_season}; calibrate {calibration_season}; "
                f"evaluate {validation_season}"
            ),
            "history_rows": len(history),
            "calibration_rows": len(calibration),
            "outer_train_rows": len(outer_train),
            "valid_rows": len(valid),
            "calibration_target_rate": float(calibration[TARGET].mean()),
            "valid_target_rate": float(valid[TARGET].mean()),
            "debug_subsample": debug_subsample,
            "row_independent_inference": True,
            "baseline_reproduction_delta": reproduction_delta,
            "elapsed_seconds": time.perf_counter() - fold_started,
            "versions": {
                "python": platform.python_version(),
                "numpy": np.__version__,
                "pandas": pd.__version__,
                "scipy": scipy.__version__,
                "scikit_learn": sklearn.__version__,
            },
            "command": " ".join(sys.argv),
        },
        "fit_details": fit_details,
        "calibration_bundle": calibration_bundle,
        "ranking": ranking,
        "strategies": strategies,
    }
    rows = flatten_fold_rows(validation_season, strategies)
    top = ranking[:5]
    print(f"[{validation_season}] top strategies:", flush=True)
    for rank, name in enumerate(top, start=1):
        metrics = strategies[name]["metrics"]
        print(
            f"  {rank}. {name:<48} Brier={metrics['brier']:.8f} "
            f"score={metrics['competition_score']:,.1f}",
            flush=True,
        )
    del history, calibration, outer_train, valid
    gc.collect()
    return payload, rows


def main() -> None:
    args = parse_args()
    started = time.perf_counter()
    frame = load_train(args.data)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    all_rows: list[dict[str, Any]] = []
    for validation_season in sorted(set(args.validation_seasons)):
        payload, rows = run_fold(frame, validation_season, args)
        all_rows.extend(rows)
        stem = f"calibration_ensemble_valid_{validation_season}"
        json_path = args.output_dir / f"{stem}.json"
        csv_path = args.output_dir / f"{stem}.csv"
        json_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False),
            encoding="utf-8",
        )
        pd.DataFrame(rows).to_csv(csv_path, index=False)
        print(f"Saved {json_path} and {csv_path}.", flush=True)

    rolling = pd.DataFrame(all_rows)
    aggregate = aggregate_rows(rolling)
    rolling_path = args.output_dir / "calibration_ensemble_rolling.csv"
    aggregate_path = args.output_dir / "calibration_ensemble_aggregate.csv"
    rolling.to_csv(rolling_path, index=False)
    aggregate.to_csv(aggregate_path, index=False)
    print("\nAggregate top 10 (lower mean Brier is better):", flush=True)
    for rank, row in enumerate(aggregate.head(10).itertuples(index=False), start=1):
        print(
            f"  {rank}. {row.strategy:<48} mean={row.mean_brier:.8f} "
            f"worst_skill={row.worst_raw_skill:.6f} positive={row.positive_skill_folds}/"
            f"{row.folds}",
            flush=True,
        )
    print(
        f"Saved {rolling_path} and {aggregate_path}; total={time.perf_counter() - started:.1f}s.",
        flush=True,
    )


if __name__ == "__main__":
    main()
