#!/usr/bin/env python3
"""Rebuild the public detailed-context CatBoost family from official data only.

No published prediction, encoder, model, or metadata artifact is loaded.  Only
the public method's source-level feature definitions and hyperparameters are
reused; every table and model is rebuilt under an outer season cutoff.
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


ROOT = Path(__file__).resolve().parents[1]
PUBLIC_SRC = (
    ROOT
    / "research/external_repos/kyungjunoh-baseball/baseball-main/src"
)
if str(PUBLIC_SRC) not in sys.path:
    sys.path.insert(0, str(PUBLIC_SRC))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from catboost import CatBoostClassifier  # noqa: E402
from common import (  # noqa: E402
    SeasonAnchor,
    SplitEncoder,
    build_features,
    damped_base_rate,
    load_csv,
    shift_to_base_rate,
)
from screen_batter_season_anchor import BatterSeasonAnchor  # noqa: E402
from screen_cat_detailed_outcome_context import (  # noqa: E402
    BATTER,
    HO87,
    PARAMS,
    detail_context,
    reconstruct_details,
)
from screen_cat_player_trajectory import PriorSeasonFeatures  # noqa: E402
from experiments.analyze_v4_temporal_residual_ridge import (  # noqa: E402
    json_safe,
    score,
)


TARGET = "control_success"
PRED_DIR = ROOT / "experiments/results/predictions"
RESULT_DIR = ROOT / "experiments/results"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", required=True)
    parser.add_argument("--validation-seasons", nargs="+", type=int, required=True)
    parser.add_argument("--seeds", nargs="+", type=int, default=[42])
    parser.add_argument("--device", default=os.environ.get("V4_PUBLIC_DEVICE", "0"))
    parser.add_argument("--data", type=Path, default=ROOT / "open/data/train.csv")
    return parser.parse_args()


def metric(y: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    return score(y, np.clip(prediction, 0.0, 1.0))


def main() -> None:
    args = parse_args()
    started_all = time.perf_counter()
    data = load_csv(str(args.data))
    labels = reconstruct_details(data)
    PRED_DIR.mkdir(parents=True, exist_ok=True)
    fold_reports: list[dict[str, object]] = []

    for validation_season in args.validation_seasons:
        started_fold = time.perf_counter()
        train = data.loc[data["season"] < validation_season].copy()
        valid = data.loc[data["season"] == validation_season].copy()
        seasons = sorted(int(value) for value in train["season"].unique())
        seasons.append(int(validation_season))

        split_encoder = SplitEncoder(keys=HO87).fit(train, target_seasons=seasons)
        pitcher_anchor = SeasonAnchor().fit(train, target_seasons=seasons)
        batter_anchor = BatterSeasonAnchor(50).fit(train, seasons)
        base_train = np.asarray(
            np.column_stack([
                build_features(train).to_numpy(dtype=np.float32),
                split_encoder.transform(train).to_numpy(dtype=np.float32),
                pitcher_anchor.transform(train).to_numpy(dtype=np.float32),
                batter_anchor.transform(train)[BATTER].to_numpy(dtype=np.float32),
            ]),
            dtype=np.float32,
        )
        base_valid = np.asarray(
            np.column_stack([
                build_features(valid).to_numpy(dtype=np.float32),
                split_encoder.transform(valid).to_numpy(dtype=np.float32),
                pitcher_anchor.transform(valid).to_numpy(dtype=np.float32),
                batter_anchor.transform(valid)[BATTER].to_numpy(dtype=np.float32),
            ]),
            dtype=np.float32,
        )
        prior_features = PriorSeasonFeatures(train, seasons)
        binary_train = prior_features.recent_context(train).filter(regex="k200$")
        binary_valid = prior_features.recent_context(valid).filter(regex="k200$")
        detailed_train = detail_context(train, train, labels, k=200.0)
        detailed_valid = detail_context(train, valid, labels, k=200.0)
        pitcher_columns = [
            column
            for column in detailed_train
            if any(f"dctx_{name}_" in column for name in ("pc", "ph", "pg"))
        ]
        arm_features = {
            "detail_pitcher12": (
                detailed_train[pitcher_columns].to_numpy(dtype=np.float32),
                detailed_valid[pitcher_columns].to_numpy(dtype=np.float32),
            ),
            "binary_detail_all25": (
                np.column_stack([
                    binary_train.to_numpy(dtype=np.float32),
                    detailed_train.to_numpy(dtype=np.float32),
                ]).astype(np.float32, copy=False),
                np.column_stack([
                    binary_valid.to_numpy(dtype=np.float32),
                    detailed_valid.to_numpy(dtype=np.float32),
                ]).astype(np.float32, copy=False),
            ),
        }
        y_train = train[TARGET].to_numpy(dtype=np.int8)
        y_valid = valid[TARGET].to_numpy(dtype=np.int8)
        target_rate = float(
            damped_base_rate(
                train.groupby("season")[TARGET].mean(), validation_season, 0.75
            )
        )
        payload: dict[str, np.ndarray] = {
            "y": y_valid,
            "row_index": valid.index.to_numpy(dtype=np.int64),
            "cluster": np.asarray(valid["pitcher_id"].astype(str), dtype=np.str_),
        }
        fit_rows: list[dict[str, object]] = []
        shifted_means: dict[str, list[np.ndarray]] = {}
        raw_means: dict[str, list[np.ndarray]] = {}
        for arm_name, (extra_train, extra_valid) in arm_features.items():
            x_train = np.column_stack([base_train, extra_train]).astype(
                np.float32, copy=False
            )
            x_valid = np.column_stack([base_valid, extra_valid]).astype(
                np.float32, copy=False
            )
            raw_means[arm_name] = []
            shifted_means[arm_name] = []
            for seed in args.seeds:
                started_fit = time.perf_counter()
                settings = dict(PARAMS)
                settings.update({
                    "loss_function": "Logloss",
                    "bootstrap_type": "Bayesian",
                    "task_type": "GPU",
                    "devices": args.device,
                    "random_seed": int(seed),
                    "verbose": False,
                    "allow_writing_files": False,
                })
                model = CatBoostClassifier(**settings)
                model.fit(x_train, y_train)
                raw_prediction = model.predict_proba(x_valid)[:, 1].astype(np.float64)
                shifted_prediction = shift_to_base_rate(raw_prediction, target_rate).astype(
                    np.float64
                )
                payload[f"{arm_name}_seed{seed}_raw"] = raw_prediction
                payload[f"{arm_name}_seed{seed}_shifted"] = shifted_prediction
                raw_means[arm_name].append(raw_prediction)
                shifted_means[arm_name].append(shifted_prediction)
                fit_row = {
                    "arm": arm_name,
                    "seed": int(seed),
                    "features": int(x_train.shape[1]),
                    "seconds": time.perf_counter() - started_fit,
                    "raw_metrics": metric(y_valid, raw_prediction),
                    "shifted_metrics": metric(y_valid, shifted_prediction),
                }
                fit_rows.append(fit_row)
                print(json.dumps(json_safe(fit_row)), flush=True)
                del model
                gc.collect()
            payload[f"{arm_name}_mean_raw"] = np.mean(raw_means[arm_name], axis=0)
            payload[f"{arm_name}_mean_shifted"] = np.mean(
                shifted_means[arm_name], axis=0
            )
            del x_train, x_valid
            gc.collect()

        payload["public_v17_detail_raw"] = (
            0.4 * payload["detail_pitcher12_mean_raw"]
            + 0.6 * payload["binary_detail_all25_mean_raw"]
        )
        payload["public_v17_detail_shifted"] = shift_to_base_rate(
            payload["public_v17_detail_raw"], target_rate
        ).astype(np.float64)
        artifact = PRED_DIR / f"{args.stage}_{validation_season}.npz"
        np.savez_compressed(artifact, **payload)
        fold_reports.append({
            "validation_season": int(validation_season),
            "history_rows": int(len(train)),
            "valid_rows": int(len(valid)),
            "target_rate": target_rate,
            "detail_label_coverage": float(labels.loc[train.index].notna().all(axis=1).mean()),
            "base_features": int(base_train.shape[1]),
            "detail_features": int(detailed_train.shape[1]),
            "binary_features": int(binary_train.shape[1]),
            "fit_rows": fit_rows,
            "ensemble_metrics": {
                "raw": metric(y_valid, payload["public_v17_detail_raw"]),
                "shifted": metric(y_valid, payload["public_v17_detail_shifted"]),
            },
            "artifact": str(artifact.relative_to(ROOT)),
            "seconds": time.perf_counter() - started_fold,
        })
        del (
            train, valid, split_encoder, pitcher_anchor, batter_anchor,
            base_train, base_valid, prior_features, binary_train, binary_valid,
            detailed_train, detailed_valid, arm_features,
        )
        gc.collect()

    report = {
        "metadata": {
            "stage": args.stage,
            "official_train_only": True,
            "external_model_artifacts_used": False,
            "external_prediction_artifacts_used": False,
            "method_source": "public feature definitions and hyperparameters only",
            "validation_protocol": "train season < target; validate season == target",
            "seeds": args.seeds,
            "device": args.device,
            "params": PARAMS,
            "elapsed_seconds": time.perf_counter() - started_all,
        },
        "folds": fold_reports,
    }
    path = RESULT_DIR / f"{args.stage}.json"
    path.write_text(
        json.dumps(json_safe(report), ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"Saved {path}", flush=True)


if __name__ == "__main__":
    main()
