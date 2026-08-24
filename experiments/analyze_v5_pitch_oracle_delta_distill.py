#!/usr/bin/env python3
"""Strict-forward source test for dense pitch-oracle delta distillation."""

from __future__ import annotations

import json
import os
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.analyze_v5_dense_pitchtype_moe import (  # noqa: E402
    digest,
    evaluate,
    load,
    safe,
)
from experiments.run_v2_rolling import BOOSTER_CATEGORICAL  # noqa: E402


RESULTS = ROOT / "experiments/results"
PRED = RESULTS / "predictions"
TRAIN = ROOT / "open/data/train.csv"
PREREG = ROOT / "experiments/params/v5_pitch_oracle_delta_distill_preregister.json"
REPORT = RESULTS / "v5_pitch_oracle_delta_distill_source.json"
OUTPUT = PRED / "v5_pitch_oracle_delta_distill_source_2021.npz"
TEACHER = PRED / "v5_dense_physics_pitchtype_moe_source2020_2020.npz"
PARENT20 = PRED / "v4_m3_c_backtest_2020_2020.npz"
PARENT21 = PRED / "v4_m3_c_backtest_2021_2021.npz"
PARENT_KEY = "catboost_outcome"
TEACHER_KEY = (
    "catboost_dense_pitchtype_moe__diagnostic_true_group_oracle"
)
AVAILABLE_KEY = (
    "catboost_dense_pitchtype_moe__diagnostic_true_group_available"
)


def logit(probability: np.ndarray) -> np.ndarray:
    value = np.clip(probability, 1e-6, 1.0 - 1e-6)
    return np.log(value / (1.0 - value))


def prepare_features(
    raw: pd.DataFrame,
    row_index: np.ndarray,
    parent: np.ndarray,
    feature_columns: list[str] | None = None,
) -> pd.DataFrame:
    frame = raw.iloc[row_index].drop(
        columns=["row_id", "season", "control_success"], errors="ignore"
    ).copy()
    frame["parent_exact_c"] = parent.astype(np.float32)
    frame["parent_exact_c_logit"] = logit(parent).astype(np.float32)
    if feature_columns is not None:
        missing = sorted(set(feature_columns) - set(frame.columns))
        if missing:
            raise ValueError(f"student feature columns missing: {missing}")
        frame = frame.loc[:, feature_columns]
    for column in frame.columns:
        if column in BOOSTER_CATEGORICAL:
            frame[column] = (
                frame[column].astype("string").fillna("__MISSING__").astype(str)
            )
        else:
            frame[column] = pd.to_numeric(frame[column], errors="coerce").astype(
                np.float32
            )
    return frame.reset_index(drop=True)


def main() -> None:
    if REPORT.exists() or OUTPUT.exists():
        raise FileExistsError("immutable source result already exists")
    from catboost import CatBoostRegressor

    prereg = json.loads(PREREG.read_text(encoding="utf-8"))
    teacher = load(TEACHER)
    parent20_artifact = load(PARENT20)
    parent21_artifact = load(PARENT21)
    for key in ("y", "row_index", "cluster"):
        if not np.array_equal(teacher[key], parent20_artifact[key]):
            raise ValueError(f"teacher/parent 2020 alignment mismatch: {key}")
    raw = pd.read_csv(TRAIN)
    if not np.array_equal(
        raw.iloc[parent21_artifact["row_index"].astype(np.int64)][
            "control_success"
        ].to_numpy(dtype=np.int8),
        parent21_artifact["y"].astype(np.int8),
    ):
        raise ValueError("2021 evaluation target alignment mismatch")

    train_types = raw.iloc[teacher["row_index"].astype(np.int64)][
        "game_type"
    ].astype(str).to_numpy()
    valid_types = raw.iloc[parent21_artifact["row_index"].astype(np.int64)][
        "game_type"
    ].astype(str).to_numpy()
    available = teacher[AVAILABLE_KEY].astype(bool)
    teacher_coverage = float(available.mean())
    fit_mask = available & (train_types == "R")
    parent20 = parent20_artifact[PARENT_KEY].astype(np.float64)
    parent21 = parent21_artifact[PARENT_KEY].astype(np.float64)
    teacher_delta = np.clip(
        teacher[TEACHER_KEY].astype(np.float64) - parent20,
        float(prereg["student"]["target_clip"][0]),
        float(prereg["student"]["target_clip"][1]),
    )
    train_x = prepare_features(
        raw, teacher["row_index"].astype(np.int64), parent20
    )
    feature_columns = list(train_x.columns)
    valid_x = prepare_features(
        raw,
        parent21_artifact["row_index"].astype(np.int64),
        parent21,
        feature_columns,
    )
    forbidden_present = sorted(
        set(feature_columns)
        & {"control_success", "current_pitch_group", "teacher_oracle", "row_id"}
    )
    if forbidden_present:
        raise ValueError(f"forbidden student features: {forbidden_present}")

    configured = dict(prereg["student"]["catboost_regressor"])
    settings: dict[str, Any] = {
        **configured,
        "eval_metric": "RMSE",
        "allow_writing_files": False,
        "thread_count": 6,
        "task_type": (
            "GPU"
            if os.environ.get("V2_BOOSTER_DEVICE", "gpu").lower() == "gpu"
            else "CPU"
        ),
    }
    categorical = [
        column for column in feature_columns if column in BOOSTER_CATEGORICAL
    ]
    model = CatBoostRegressor(**settings)
    started = time.perf_counter()
    model.fit(
        train_x.loc[fit_mask],
        teacher_delta[fit_mask],
        cat_features=categorical,
        verbose=False,
    )
    fit_seconds = time.perf_counter() - started
    prediction_started = time.perf_counter()
    correction = np.asarray(model.predict(valid_x), dtype=np.float64)
    predict_seconds = time.perf_counter() - prediction_started
    sentinel = float(model.predict(valid_x.iloc[[0]])[0])
    invariance_delta = float(abs(sentinel - correction[0]))

    regular = valid_types == "R"
    full = np.ones(len(regular), dtype=bool)
    direction = np.clip(parent21 + correction, 1e-6, 1.0 - 1e-6)
    semantic_pass = bool(
        teacher_coverage
        >= float(prereg["semantic_gate"]["minimum_teacher_availability"])
        and not forbidden_present
        and invariance_delta
        <= float(
            prereg["semantic_gate"][
                "single_row_prediction_invariance_max_abs"
            ]
        )
    )
    trials = []
    if semantic_pass:
        for scale in prereg["student"]["correction_scale_grid"]:
            trial = evaluate(
                parent21_artifact,
                parent21,
                direction,
                regular,
                {"full": full, "R": regular},
                float(scale),
                int(prereg["bootstrap_iterations"]),
                1610000 + int(float(scale) * 100),
            )
            trial["correction_scale"] = float(scale)
            trials.append(trial)
    selected = (
        max(
            trials,
            key=lambda item: (
                item["routes"]["R"]["gain"],
                item["routes"]["full"]["gain"],
                -item["correction_scale"],
            ),
        )
        if trials
        else None
    )
    checks = [semantic_pass, selected is not None]
    if selected is not None:
        routes = selected["routes"]
        checks.extend(
            [
                routes["R"]["gain"]
                >= float(prereg["source_gate"]["minimum_2021_R_gain"]),
                routes["full"]["gain"]
                >= float(prereg["source_gate"]["minimum_2021_full_gain"]),
                routes["R"]["pitcher_cluster_95_ci"]["ci_low"] > 0.0,
                routes["full"]["pitcher_cluster_95_ci"]["ci_low"] > 0.0,
            ]
        )
    passed = bool(all(checks))
    selected_scale = float(selected["correction_scale"]) if selected else 0.0
    final_prediction = parent21.copy()
    final_prediction[regular] = np.clip(
        parent21[regular] + selected_scale * correction[regular],
        1e-6,
        1.0 - 1e-6,
    )
    np.savez_compressed(
        OUTPUT,
        y=parent21_artifact["y"].astype(np.int8),
        row_index=parent21_artifact["row_index"].astype(np.int64),
        cluster=parent21_artifact["cluster"],
        parent_exact_c=parent21,
        student_correction=correction,
        student_direction=direction,
        final_prediction=final_prediction,
        game_type_r=regular.astype(np.int8),
    )
    importance = np.asarray(model.get_feature_importance(), dtype=np.float64)
    top_importance = [
        {"feature": column, "importance": float(value)}
        for column, value in sorted(
            zip(feature_columns, importance),
            key=lambda item: item[1],
            reverse=True,
        )[:30]
    ]
    report = {
        "experiment_id": prereg["experiment_id"],
        "status": "source_pass" if passed else "source_failed",
        "preregister_sha256": digest(PREREG),
        "script_sha256": digest(Path(__file__)),
        "years_read": [2020, 2021],
        "years_not_read": [2022, 2023, 2024],
        "teacher": {
            "path": str(TEACHER.relative_to(ROOT)),
            "sha256": digest(TEACHER),
            "availability": teacher_coverage,
            "fit_rows_R": int(fit_mask.sum()),
            "delta_mean_R": float(teacher_delta[fit_mask].mean()),
            "delta_std_R": float(teacher_delta[fit_mask].std()),
            "control_target_2020_used_by_student": False,
        },
        "student": {
            "model_params": settings,
            "feature_columns": feature_columns,
            "forbidden_features_present": forbidden_present,
            "fit_seconds": fit_seconds,
            "predict_seconds": predict_seconds,
            "correction_mean": float(correction.mean()),
            "correction_std": float(correction.std()),
            "correction_min": float(correction.min()),
            "correction_max": float(correction.max()),
            "single_row_invariance_max_abs": invariance_delta,
            "top_feature_importance": top_importance,
        },
        "semantic_gate_pass": semantic_pass,
        "trials": trials,
        "selected": selected,
        "source_gate_pass": passed,
        "artifact": {
            "path": str(OUTPUT.relative_to(ROOT)),
            "sha256": digest(OUTPUT),
        },
        "goal_status": "active",
        "goal_completion_claimed": False,
    }
    REPORT.write_text(
        json.dumps(safe(report), ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(
        json.dumps(
            safe(
                {
                    "status": report["status"],
                    "teacher": report["teacher"],
                    "student": {
                        key: report["student"][key]
                        for key in (
                            "fit_seconds", "predict_seconds", "correction_mean",
                            "correction_std", "single_row_invariance_max_abs",
                        )
                    },
                    "selected": selected,
                }
            ),
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
