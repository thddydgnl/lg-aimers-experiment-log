#!/usr/bin/env python3
"""Out-of-time feasibility check for a privileged TrackMan physics teacher.

This is deliberately diagnostic-only.  It tests whether post-pitch physical
measurements carry stable target information beyond pitch identity and the
pre-pitch context.  A later deployable model, if this gate passes, may only use
profiles distilled from completed historical seasons; it may never use the
current prediction row's TrackMan measurements.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from typing import Any

# LightGBM must load before pandas/sklearn on this Windows runtime.  Loading a
# second OpenMP runtime first can crash DatasetSetField with a native access
# violation before the first tree is trained.
from lightgbm import LGBMClassifier
import numpy as np
import pandas as pd
from sklearn.metrics import log_loss, roc_auc_score


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.run_e20r_rolling import load_joined_trackman  # noqa: E402


PREREG = (
    ROOT
    / "experiments/params/v5_privileged_trackman_teacher_feasibility_preregister.json"
)
OUTPUT = (
    ROOT / "experiments/results/v5_privileged_trackman_teacher_feasibility.json"
)
TARGET_YEARS = (2021, 2022, 2023)
TARGET = "control_success"
PHYSICS = (
    "rel_speed",
    "spin_rate",
    "induced_vert_break",
    "horz_break",
    "extension",
    "rel_height",
    "rel_side",
    "zone_speed",
)
NUMERIC_CONTEXT = ("balls_before", "strikes_before", "outs_before")
CATEGORICAL = (
    "pitcher_hand",
    "batter_hand",
    "pitch_type_group",
    "tagged_pitch_type",
    "auto_pitch_type",
)
CONTROL_FEATURES = (*NUMERIC_CONTEXT, *CATEGORICAL)
FULL_FEATURES = (*CONTROL_FEATURES, *PHYSICS)
MODEL_PARAMS = {
    "objective": "binary",
    "n_estimators": 300,
    "learning_rate": 0.04,
    "num_leaves": 31,
    "max_depth": -1,
    "min_child_samples": 400,
    "subsample": 1.0,
    "colsample_bytree": 1.0,
    "reg_alpha": 1.0,
    "reg_lambda": 20.0,
    "random_state": 20260821,
    "n_jobs": 6,
    "verbosity": -1,
    "deterministic": True,
    "force_col_wise": True,
}


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def encode_features(
    train: pd.DataFrame,
    valid: pd.DataFrame,
    feature_names: tuple[str, ...],
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    train_x = train.loc[:, feature_names].copy()
    valid_x = valid.loc[:, feature_names].copy()
    categorical = [name for name in feature_names if name in CATEGORICAL]
    for name in categorical:
        # The category vocabulary contains feature values only, never labels.
        values = pd.concat(
            [train_x[name].astype("string"), valid_x[name].astype("string")],
            ignore_index=True,
        ).fillna("__MISSING__")
        categories = sorted(values.unique().tolist())
        mapping = {value: index for index, value in enumerate(categories)}
        train_x[name] = (
            train_x[name].astype("string").fillna("__MISSING__").map(mapping)
        ).astype("category")
        valid_x[name] = (
            valid_x[name].astype("string").fillna("__MISSING__").map(mapping)
        ).astype("category")
    for name in feature_names:
        if name not in categorical:
            train_x[name] = pd.to_numeric(train_x[name], errors="coerce")
            valid_x[name] = pd.to_numeric(valid_x[name], errors="coerce")
    return train_x, valid_x, categorical


def fit_predict(
    train: pd.DataFrame,
    valid: pd.DataFrame,
    feature_names: tuple[str, ...],
) -> np.ndarray:
    train_x, valid_x, categorical = encode_features(train, valid, feature_names)
    model = LGBMClassifier(**MODEL_PARAMS)
    model.fit(
        train_x,
        np.ascontiguousarray(train[TARGET].to_numpy(dtype=np.float32)),
        categorical_feature=categorical,
    )
    return model.predict_proba(valid_x)[:, 1].astype(np.float64)


def normalized_gain(y: np.ndarray, control: np.ndarray, full: np.ndarray) -> float:
    rate = float(np.mean(y))
    denominator = rate * (1.0 - rate)
    paired = np.square(y - control) - np.square(y - full)
    return 100_000.0 * float(np.mean(paired)) / denominator


def clustered_interval(
    y: np.ndarray,
    control: np.ndarray,
    full: np.ndarray,
    clusters: np.ndarray,
    seed: int,
    replicates: int = 1000,
) -> dict[str, float]:
    rate = float(np.mean(y))
    denominator = rate * (1.0 - rate)
    paired = np.square(y - control) - np.square(y - full)
    work = pd.DataFrame({"cluster": clusters, "paired": paired})
    grouped = work.groupby("cluster", sort=False, observed=True)["paired"].agg(
        ["sum", "count"]
    )
    sums = grouped["sum"].to_numpy(dtype=np.float64)
    counts = grouped["count"].to_numpy(dtype=np.int64)
    rng = np.random.default_rng(seed)
    values = np.empty(replicates, dtype=np.float64)
    n_clusters = len(grouped)
    for index in range(replicates):
        sampled = rng.integers(0, n_clusters, size=n_clusters)
        mean = float(sums[sampled].sum() / counts[sampled].sum())
        values[index] = 100_000.0 * mean / denominator
    return {
        "point": normalized_gain(y, control, full),
        "lower_95": float(np.quantile(values, 0.025)),
        "upper_95": float(np.quantile(values, 0.975)),
        "replicates": int(replicates),
        "clusters": int(n_clusters),
    }


def metrics(y: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    return {
        "brier": float(np.mean(np.square(y - prediction))),
        "log_loss": float(log_loss(y, prediction, labels=[0, 1])),
        "roc_auc": float(roc_auc_score(y, prediction)),
        "prediction_mean": float(np.mean(prediction)),
        "prediction_std": float(np.std(prediction)),
    }


def main() -> None:
    prereg = json.loads(PREREG.read_text(encoding="utf-8"))
    if prereg["status"] != "locked_before_execution":
        raise ValueError("Preregister status must remain locked_before_execution")
    joined = load_joined_trackman()
    rows = joined.loc[
        joined["game_type"].eq("R")
        & joined["season"].le(max(TARGET_YEARS))
        & joined[TARGET].notna()
    ].copy()
    del joined
    rows[TARGET] = rows[TARGET].astype(np.int8)

    folds: dict[str, Any] = {}
    for year in TARGET_YEARS:
        history = rows.loc[rows["season"].lt(year)].copy()
        valid = rows.loc[rows["season"].eq(year)].copy()
        if history.empty or valid.empty:
            raise ValueError(f"Empty fold for {year}")
        print(
            f"teacher feasibility {year}: train={len(history):,} valid={len(valid):,}",
            flush=True,
        )
        control = fit_predict(history, valid, CONTROL_FEATURES)
        full = fit_predict(history, valid, FULL_FEATURES)
        y = valid[TARGET].to_numpy(dtype=np.float64)
        control_metrics = metrics(y, control)
        full_metrics = metrics(y, full)
        interval = clustered_interval(
            y,
            control,
            full,
            valid["pitcher_id"].to_numpy(),
            seed=20260821 + year,
        )
        folds[str(year)] = {
            "history_seasons": sorted(
                int(value) for value in history["season"].unique()
            ),
            "train_rows": int(len(history)),
            "valid_rows": int(len(valid)),
            "valid_pitchers": int(valid["pitcher_id"].nunique()),
            "target_rate": float(np.mean(y)),
            "control": control_metrics,
            "physics_teacher": full_metrics,
            "paired_normalized_brier_gain": interval,
            "auc_delta": float(
                full_metrics["roc_auc"] - control_metrics["roc_auc"]
            ),
            "log_loss_delta_full_minus_control": float(
                full_metrics["log_loss"] - control_metrics["log_loss"]
            ),
        }
        print(
            f"  gain={interval['point']:.3f} "
            f"CI=[{interval['lower_95']:.3f}, {interval['upper_95']:.3f}] "
            f"auc_delta={folds[str(year)]['auc_delta']:.6f}",
            flush=True,
        )

    point_positive = all(
        folds[str(year)]["paired_normalized_brier_gain"]["point"] > 0.0
        for year in TARGET_YEARS
    )
    ci_positive_count = sum(
        folds[str(year)]["paired_normalized_brier_gain"]["lower_95"] > 0.0
        for year in TARGET_YEARS
    )
    auc_positive = all(
        folds[str(year)]["auc_delta"] > 0.0 for year in TARGET_YEARS
    )
    passed = point_positive and ci_positive_count >= 2 and auc_positive
    result = {
        "experiment_id": prereg["experiment_id"],
        "status": "passed_proceed_to_history_only_distillation"
        if passed
        else "failed_reject_direction_without_2024",
        "protocol": {
            "diagnostic_only": True,
            "official_data_only": True,
            "test_rows_read": False,
            "2024_rows_read": False,
            "identifiers_as_features": False,
            "current_trackman_permitted_in_deployable_candidate": False,
        },
        "preregister": str(PREREG.relative_to(ROOT)),
        "preregister_sha256": file_sha256(PREREG),
        "script_sha256": file_sha256(Path(__file__)),
        "model_params": MODEL_PARAMS,
        "folds": folds,
        "gate": {
            "point_positive_all_years": bool(point_positive),
            "positive_ci_years": int(ci_positive_count),
            "auc_positive_all_years": bool(auc_positive),
            "passed": bool(passed),
        },
        "next_action": (
            "Preregister a history-only profile distillation on 2022/2023."
            if passed
            else "Do not build or confirm this direction on 2024."
        ),
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"wrote {OUTPUT}", flush=True)


if __name__ == "__main__":
    main()
