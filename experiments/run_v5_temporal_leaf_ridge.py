#!/usr/bin/env python3
"""Strict A->B->C temporal tree-leaf residual experiment."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.run_v5_h1_residual import cluster_bootstrap_score_gain


TARGET = "control_success"
PRED = ROOT / "experiments/results/predictions"
PREREG = ROOT / "experiments/params/v5_temporal_leaf_ridge_preregister.json"
OUTPUT = ROOT / "experiments/results/v5_temporal_leaf_ridge_source21.json"
DROP = {"row_id", TARGET, "pitcher_id", "batter_id", "game_type"}
C_SOURCES = {
    2020: ("v4_m3_c_backtest_2020", "catboost_outcome"),
    2021: ("v4_m3_c_backtest_2021", "catboost_outcome"),
    2022: ("v3_sparse_c_backtest", "catboost_outcome"),
    2023: ("v3_sparse_c_backtest", "catboost_outcome"),
    2024: (
        "v3_outcome_trackmanrich_overall_e14k50_batter80_middle100",
        "catboost_outcome",
    ),
}


def load_c(year: int) -> dict[str, np.ndarray]:
    stage, key = C_SOURCES[year]
    with np.load(PRED / f"{stage}_{year}.npz", allow_pickle=False) as archive:
        result = {name: np.asarray(archive[name]) for name in archive.files}
    result["prediction"] = np.asarray(result[key], dtype=np.float64)
    return result


def score(y: np.ndarray, prediction: np.ndarray) -> dict[str, float | int]:
    target = np.asarray(y, dtype=np.float64)
    pred = np.asarray(prediction, dtype=np.float64)
    rate = float(target.mean())
    brier = float(np.mean(np.square(pred - target)))
    raw = 100_000.0 * (1.0 - brier / (rate * (1.0 - rate)))
    return {
        "rows": int(len(target)),
        "target_rate": rate,
        "prediction_mean": float(pred.mean()),
        "prediction_std": float(pred.std()),
        "brier": brier,
        "raw_competition_score": raw,
    }


def prepare_features(frame: pd.DataFrame) -> tuple[list[str], list[str], list[str]]:
    features = [column for column in frame.columns if column not in DROP]
    categorical = [
        column
        for column in features
        if frame[column].dtype == "object" or str(frame[column].dtype).startswith("string")
    ]
    numeric = [column for column in features if column not in categorical]
    return features, categorical, numeric


def run_transition(frame: pd.DataFrame, target_year: int) -> dict[str, Any]:
    block_b_year = target_year - 1
    a_mask = (frame["season"] < block_b_year) & frame["game_type"].eq("R")
    b_artifact = load_c(block_b_year)
    c_artifact = load_c(target_year)
    b_rows = b_artifact["row_index"].astype(np.int64)
    c_rows = c_artifact["row_index"].astype(np.int64)
    b_mask = frame.iloc[b_rows]["game_type"].eq("R").to_numpy()
    c_mask = frame.iloc[c_rows]["game_type"].eq("R").to_numpy()
    b_rows = b_rows[b_mask]
    c_rows = c_rows[c_mask]
    y_b = b_artifact["y"][b_mask].astype(np.float64)
    p_b = b_artifact["prediction"][b_mask]
    y_c = c_artifact["y"][c_mask].astype(np.float64)
    p_c = c_artifact["prediction"][c_mask]
    cluster_c = c_artifact["cluster"][c_mask].astype(str)

    features, categorical, numeric = prepare_features(frame)
    transformer = ColumnTransformer(
        [
            (
                "cat",
                OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1),
                categorical,
            ),
            ("num", SimpleImputer(strategy="median"), numeric),
        ],
        remainder="drop",
    )
    x_a = transformer.fit_transform(frame.loc[a_mask, features]).astype(np.float32)
    x_b = transformer.transform(frame.iloc[b_rows][features]).astype(np.float32)
    x_c = transformer.transform(frame.iloc[c_rows][features]).astype(np.float32)
    y_a = frame.loc[a_mask, TARGET].to_numpy(dtype=np.int8)

    tree = lgb.LGBMClassifier(
        objective="binary",
        n_estimators=120,
        learning_rate=0.03,
        num_leaves=12,
        min_child_samples=2000,
        reg_lambda=100.0,
        max_bin=32,
        random_state=2026,
        n_jobs=6,
        verbosity=-1,
    )
    tree.fit(x_a, y_a)
    leaf_b = tree.predict(x_b, pred_leaf=True)
    leaf_c = tree.predict(x_c, pred_leaf=True)
    encoder = OneHotEncoder(handle_unknown="ignore", dtype=np.float32)
    z_b = encoder.fit_transform(leaf_b)
    z_c = encoder.transform(leaf_c)
    residual = y_b - p_b
    ridge = Ridge(
        alpha=10_000.0,
        fit_intercept=True,
        solver="lsqr",
        tol=1e-5,
        max_iter=1000,
    )
    ridge.fit(z_b, residual)
    raw_correction = np.asarray(ridge.predict(z_c), dtype=np.float64)
    correction = 0.2 * np.clip(raw_correction, -0.02, 0.02)
    candidate = np.clip(p_c + correction, 1e-6, 1.0 - 1e-6)
    parent_metrics = score(y_c, p_c)
    candidate_metrics = score(y_c, candidate)
    bootstrap = cluster_bootstrap_score_gain(
        y_c,
        p_c,
        candidate,
        cluster_c,
        np.ones(len(y_c), dtype=bool),
        1000,
        530000 + target_year,
    )
    artifact_path = PRED / f"v5_temporal_leaf_ridge_{target_year}.npz"
    np.savez_compressed(
        artifact_path,
        y=y_c.astype(np.int8),
        row_index=c_rows,
        cluster=cluster_c,
        parent_C=p_c,
        final_prediction=candidate,
    )
    return {
        "target_year": target_year,
        "blocks": {
            "A_seasons": sorted(frame.loc[a_mask, "season"].astype(int).unique().tolist()),
            "B_season": block_b_year,
            "C_season": target_year,
            "A_rows_R": int(a_mask.sum()),
            "B_rows_R": int(len(b_rows)),
            "C_rows_R": int(len(c_rows)),
        },
        "feature_count": len(features),
        "categorical_count": len(categorical),
        "tree_leaf_columns": int(z_b.shape[1]),
        "parent_metrics": parent_metrics,
        "candidate_metrics": candidate_metrics,
        "score_gain": float(
            candidate_metrics["raw_competition_score"]
            - parent_metrics["raw_competition_score"]
        ),
        "bootstrap": bootstrap,
        "raw_correction_mean": float(raw_correction.mean()),
        "raw_correction_std": float(raw_correction.std()),
        "applied_correction_mean": float(correction.mean()),
        "applied_correction_std": float(correction.std()),
        "artifact": str(artifact_path.relative_to(ROOT)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target-year", type=int, default=2021)
    args = parser.parse_args()
    if args.target_year not in (2021, 2022, 2023, 2024):
        raise ValueError("target year must be 2021-2024")
    frame = pd.read_csv(ROOT / "open/data/train.csv", low_memory=False)
    result = run_transition(frame, args.target_year)
    eligible = bool(
        result["score_gain"] > 0.0 and result["bootstrap"]["ci_low"] > 0.0
    )
    report = {
        "experiment_id": "V5_TEMPORAL_LEAF_RIDGE",
        "mode": "source_feasibility" if args.target_year == 2021 else "development",
        "preregister_sha256": hashlib.sha256(PREREG.read_bytes()).hexdigest(),
        "target_years_read": [args.target_year],
        "result": result,
        "eligible": eligible,
        "status": "eligible" if eligible else "failed_gate",
    }
    output = OUTPUT if args.target_year == 2021 else ROOT / (
        f"experiments/results/v5_temporal_leaf_ridge_{args.target_year}.json"
    )
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({
        "target_year": args.target_year,
        "parent_score": result["parent_metrics"]["raw_competition_score"],
        "candidate_score": result["candidate_metrics"]["raw_competition_score"],
        "gain": result["score_gain"],
        "ci_low": result["bootstrap"]["ci_low"],
        "status": report["status"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
