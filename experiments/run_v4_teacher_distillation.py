#!/usr/bin/env python3
"""Distill a rolling OOF ensemble into a compact row-independent CatBoost.

The student uses only official row features.  Teacher probabilities are OOF
for every training row and therefore contain no target-fold label leakage.
An optional supervised fraction mixes the official binary label into the soft
target; it is selected only on a later outer fold.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.analyze_v4_temporal_residual_ridge import (  # noqa: E402
    MEDIAN_OFFSET,
    REQUIRED_LOCAL,
    json_safe,
    score,
)
from experiments.run_baselines import FEATURES, TARGET  # noqa: E402
from experiments.run_v2_rolling import BOOSTER_CATEGORICAL  # noqa: E402


PRED = ROOT / "experiments/results/predictions"
MODEL_DIR = ROOT / "experiments/results/models"
RESULT_DIR = ROOT / "experiments/results"

TEACHERS = {
    "routed": ("v4_routed_tabm_stack_locked", "routed_tabm_stack"),
    "supported": ("v4_supported_meta_stack", "final_prediction"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", required=True)
    parser.add_argument("--teacher", choices=sorted(TEACHERS), required=True)
    parser.add_argument("--train-years", nargs="+", type=int, default=[2022, 2023])
    parser.add_argument("--valid-year", type=int, default=2024)
    parser.add_argument("--alphas", nargs="+", type=float, default=[0.0, 0.1, 0.25])
    parser.add_argument("--iterations", type=int, default=1500)
    parser.add_argument("--depth", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=0.03)
    parser.add_argument("--l2-leaf-reg", type=float, default=50.0)
    parser.add_argument("--seed", type=int, default=2026)
    return parser.parse_args()


def load(path: Path) -> dict[str, np.ndarray]:
    with np.load(path) as archive:
        return {key: np.asarray(archive[key]) for key in archive.files}


def prepare(frame: pd.DataFrame, categorical: list[str]) -> pd.DataFrame:
    result = frame.copy()
    for column in categorical:
        result[column] = (
            result[column].astype("string").fillna("__missing__").astype(str)
        )
    return result


def main() -> None:
    args = parse_args()
    if any(not 0.0 <= value <= 1.0 for value in args.alphas):
        raise ValueError("--alphas must be in [0, 1]")
    if max(args.train_years) >= args.valid_year:
        raise ValueError("Every train year must precede --valid-year")
    started_all = time.perf_counter()
    raw = pd.read_csv(
        ROOT / "open/data/train.csv",
        usecols=[*FEATURES, TARGET],
        encoding="utf-8-sig",
        low_memory=False,
    )
    teacher_stem, teacher_key = TEACHERS[args.teacher]
    artifacts = {
        year: load(PRED / f"{teacher_stem}_{year}.npz")
        for year in [*args.train_years, args.valid_year]
    }
    frames: dict[int, pd.DataFrame] = {}
    for year, artifact in artifacts.items():
        row_index = artifact["row_index"].astype(np.int64)
        frames[year] = raw.iloc[row_index].copy()
        if not np.array_equal(
            frames[year][TARGET].to_numpy(dtype=np.int8),
            artifact["y"].astype(np.int8),
        ):
            raise ValueError(f"Target alignment mismatch for {year}")

    categorical = [column for column in BOOSTER_CATEGORICAL if column in FEATURES]
    train_frame = pd.concat([frames[year] for year in args.train_years], axis=0)
    valid_frame = frames[args.valid_year]
    train_x = prepare(train_frame[FEATURES], categorical)
    valid_x = prepare(valid_frame[FEATURES], categorical)
    teacher_train = np.concatenate([
        artifacts[year][teacher_key].astype(np.float64) for year in args.train_years
    ])
    teacher_valid = artifacts[args.valid_year][teacher_key].astype(np.float64)
    actual_train = train_frame[TARGET].to_numpy(dtype=np.float64)
    actual_valid = valid_frame[TARGET].to_numpy(dtype=np.float64)

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    predictions: dict[str, np.ndarray] = {}
    rows: list[dict[str, object]] = []
    for alpha in args.alphas:
        soft_target = (1.0 - alpha) * teacher_train + alpha * actual_train
        model = CatBoostRegressor(
            loss_function="RMSE",
            eval_metric="RMSE",
            iterations=args.iterations,
            depth=args.depth,
            learning_rate=args.learning_rate,
            l2_leaf_reg=args.l2_leaf_reg,
            random_seed=args.seed,
            random_strength=1.0,
            border_count=64,
            max_ctr_complexity=2,
            task_type=(
                "GPU"
                if os.environ.get("V2_BOOSTER_DEVICE", "cpu").lower() == "gpu"
                else "CPU"
            ),
            thread_count=6,
            allow_writing_files=False,
            verbose=False,
        )
        started_fit = time.perf_counter()
        model.fit(train_x, soft_target, cat_features=categorical)
        prediction = np.clip(model.predict(valid_x), 1e-6, 1.0 - 1e-6)
        label = f"alpha_{alpha:g}".replace(".", "p")
        model_path = MODEL_DIR / f"{args.stage}_{label}.cbm"
        model.save_model(model_path)
        predictions[label] = prediction
        actual_metrics = score(actual_valid, prediction)
        row = {
            "alpha": float(alpha),
            "label": label,
            "fit_seconds": time.perf_counter() - started_fit,
            "model_path": str(model_path.relative_to(ROOT)),
            "model_bytes": int(model_path.stat().st_size),
            "teacher_rmse": float(np.sqrt(np.mean(np.square(prediction - teacher_valid)))),
            "teacher_correlation": float(np.corrcoef(prediction, teacher_valid)[0, 1]),
            "prediction_mean": float(prediction.mean()),
            "prediction_std": float(prediction.std()),
            "actual_metrics": actual_metrics,
            "expected_lb_median": float(
                actual_metrics["raw_competition_score"] + MEDIAN_OFFSET
            ),
        }
        rows.append(row)
        print(json.dumps(json_safe(row)), flush=True)
        del model
        gc.collect()

    best = max(rows, key=lambda row: float(row["actual_metrics"]["raw_competition_score"]))
    output = PRED / f"{args.stage}_{args.valid_year}.npz"
    np.savez_compressed(
        output,
        y=actual_valid.astype(np.int8),
        row_index=artifacts[args.valid_year]["row_index"],
        cluster=artifacts[args.valid_year]["cluster"],
        teacher=teacher_valid,
        **predictions,
    )
    report = {
        "protocol": {
            "official_train_only": True,
            "teacher_is_outer_oof": True,
            "student_features": "current row official fields only",
            "train_years": args.train_years,
            "validation_year": args.valid_year,
            "test_rows_read": False,
            "row_independent": True,
        },
        "teacher": {"stem": teacher_stem, "key": teacher_key},
        "params": {
            "iterations": args.iterations,
            "depth": args.depth,
            "learning_rate": args.learning_rate,
            "l2_leaf_reg": args.l2_leaf_reg,
            "seed": args.seed,
        },
        "rows": rows,
        "best": best,
        "required_local_score": REQUIRED_LOCAL,
        "crosses_required_local_score": (
            float(best["actual_metrics"]["raw_competition_score"]) > REQUIRED_LOCAL
        ),
        "prediction_artifact": str(output.relative_to(ROOT)),
        "elapsed_seconds": time.perf_counter() - started_all,
    }
    report_path = RESULT_DIR / f"{args.stage}.json"
    report_path.write_text(
        json.dumps(json_safe(report), ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"Saved {report_path}", flush=True)


if __name__ == "__main__":
    main()
