#!/usr/bin/env python3
"""Predict control-relevant fine-pitch selection without current pitch data."""

from __future__ import annotations

import gc
import json
import os
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.analyze_v5_fine_pitchtype_latent import (  # noqa: E402
    FINE_TYPES,
    PREDICTIONS,
    SOURCE_YEARS,
    evaluate,
    json_safe,
    load_anchor,
    load_fine_labels,
    load_main_frame,
    prepare_catboost,
    sha256,
)
from experiments.run_baselines import RANDOM_SEED, TARGET  # noqa: E402
from experiments.run_v5_h1_residual import (  # noqa: E402
    cluster_bootstrap_score_gain,
)


PREREGISTRATION = (
    ROOT
    / "experiments/params/v5_control_relevant_pitch_selection_preregister.json"
)
OUTPUT = (
    ROOT / "experiments/results/v5_control_relevant_pitch_selection_source.json"
)
LABEL_COLUMN = "fine_tagged"
OUTCOME_K = 100.0
REPERTOIRE_K = 200.0
GAMMAS = (0.1, 0.25, 0.5, 0.75, 1.0, 1.5)
VARIANTS = {
    "d4_l50": {
        "iterations": 500,
        "depth": 4,
        "learning_rate": 0.04,
        "l2_leaf_reg": 50.0,
    },
    "d6_l100": {
        "iterations": 600,
        "depth": 6,
        "learning_rate": 0.03,
        "l2_leaf_reg": 100.0,
    },
}


def build_teacher_and_state(
    history: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, Any], dict[str, Any]]:
    """Create a leave-one-control-label-out teacher and a frozen full state."""
    work = history.loc[history[LABEL_COLUMN].notna()].copy()
    work["centered_control"] = work[TARGET].astype(np.float64) - 0.5
    grouped = work.groupby(
        ["pitcher_id", LABEL_COLUMN], sort=False, observed=True
    )["centered_control"]
    cell_sum = grouped.transform("sum").to_numpy(dtype=np.float64)
    cell_count = grouped.transform("count").to_numpy(dtype=np.float64)
    centered = work["centered_control"].to_numpy(dtype=np.float64)
    q_loo = (cell_sum - centered) / (cell_count - 1.0 + OUTCOME_K)

    stats = grouped.agg(["sum", "count"])
    q_table = (stats["sum"] / (stats["count"] + OUTCOME_K)).unstack()
    q_table = q_table.reindex(columns=FINE_TYPES).fillna(0.0)
    count_table = stats["count"].unstack(fill_value=0)
    count_table = count_table.reindex(columns=FINE_TYPES, fill_value=0)
    global_mix = (
        work[LABEL_COLUMN]
        .value_counts(normalize=True)
        .reindex(FINE_TYPES)
        .fillna(0.0)
        .to_numpy(dtype=np.float64)
    )
    q_values = q_table.to_numpy(dtype=np.float64)
    count_values = count_table.reindex(q_table.index).to_numpy(dtype=np.float64)
    weighted_counts = count_values + REPERTOIRE_K * global_mix[None, :]
    full_numerator = np.sum(weighted_counts * q_values, axis=1)
    total_count = count_values.sum(axis=1)
    state_summary = pd.DataFrame(
        {
            "baseline_numerator": full_numerator,
            "total_count": total_count,
        },
        index=q_table.index,
    )

    pitchers = work["pitcher_id"]
    type_index = np.array(
        [FINE_TYPES.index(value) for value in work[LABEL_COLUMN].astype(str)],
        dtype=np.int16,
    )
    full_q_rows = q_table.reindex(pitchers.to_numpy()).to_numpy(dtype=np.float64)
    full_count_rows = count_table.reindex(pitchers.to_numpy()).to_numpy(
        dtype=np.float64
    )
    baseline_numerator = pitchers.map(state_summary["baseline_numerator"]).to_numpy(
        dtype=np.float64
    )
    pitcher_total = pitchers.map(state_summary["total_count"]).to_numpy(
        dtype=np.float64
    )
    row_number = np.arange(len(work))
    own_q_full = full_q_rows[row_number, type_index]
    own_count = full_count_rows[row_number, type_index]
    own_global_mix = global_mix[type_index]
    old_term = (own_count + REPERTOIRE_K * own_global_mix) * own_q_full
    new_term = (own_count - 1.0 + REPERTOIRE_K * own_global_mix) * q_loo
    baseline_loo = (baseline_numerator - old_term + new_term) / (
        pitcher_total - 1.0 + REPERTOIRE_K
    )
    work["control_relevant_pitch_teacher"] = q_loo - baseline_loo
    state = {
        "q_table": q_table,
        "count_table": count_table,
        "global_mix": global_mix,
    }
    metadata = {
        "training_rows": int(len(work)),
        "state_pitchers": int(len(q_table)),
        "teacher_mean": float(work["control_relevant_pitch_teacher"].mean()),
        "teacher_std": float(work["control_relevant_pitch_teacher"].std()),
        "teacher_max_abs": float(
            work["control_relevant_pitch_teacher"].abs().max()
        ),
        "own_control_label_removed": True,
        "center": 0.5,
    }
    return work, state, metadata


def apply_state(
    valid_r: pd.DataFrame,
    state: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    q_table: pd.DataFrame = state["q_table"]
    count_table: pd.DataFrame = state["count_table"]
    global_mix: np.ndarray = state["global_mix"]
    pitchers = valid_r["pitcher_id"].to_numpy()
    q_matrix = q_table.reindex(pitchers).to_numpy(dtype=np.float64)
    count_matrix = count_table.reindex(pitchers).to_numpy(dtype=np.float64)
    unseen = np.isnan(count_matrix).all(axis=1)
    q_matrix[np.isnan(q_matrix)] = 0.0
    count_matrix[np.isnan(count_matrix)] = 0.0
    mix_matrix = (count_matrix + REPERTOIRE_K * global_mix[None, :]) / (
        count_matrix.sum(axis=1, keepdims=True) + REPERTOIRE_K
    )
    baseline = np.sum(mix_matrix * q_matrix, axis=1)
    matched = valid_r[LABEL_COLUMN].notna().to_numpy(dtype=bool)
    oracle = np.zeros(len(valid_r), dtype=np.float64)
    truth = valid_r.loc[matched, LABEL_COLUMN].astype(str).to_numpy()
    truth_index = np.array(
        [FINE_TYPES.index(value) for value in truth], dtype=np.int16
    )
    oracle[matched] = q_matrix[matched, truth_index] - baseline[matched]
    return oracle, matched, {
        "valid_rows": int(len(valid_r)),
        "matched_rows": int(matched.sum()),
        "unseen_pitchers": int(unseen.sum()),
        "oracle_mean": float(oracle[matched].mean()),
        "oracle_std": float(oracle[matched].std()),
    }


def fit_student(
    history_teacher: pd.DataFrame,
    valid_r: pd.DataFrame,
    variant_name: str,
    year: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    params = VARIANTS[variant_name]
    train_x, categorical = prepare_catboost(history_teacher)
    valid_x, valid_categorical = prepare_catboost(valid_r)
    if categorical != valid_categorical:
        raise AssertionError("categorical feature schema changed")
    model = CatBoostRegressor(
        loss_function="RMSE",
        random_seed=RANDOM_SEED + year + (0 if variant_name == "d4_l50" else 1000),
        allow_writing_files=False,
        thread_count=6,
        task_type=(
            "GPU"
            if os.environ.get("V2_BOOSTER_DEVICE", "cpu").lower() == "gpu"
            else "CPU"
        ),
        **params,
    )
    started = time.perf_counter()
    model.fit(
        train_x,
        history_teacher["control_relevant_pitch_teacher"].to_numpy(
            dtype=np.float64
        ),
        cat_features=categorical,
        verbose=False,
    )
    prediction = np.asarray(model.predict(valid_x), dtype=np.float64)
    metadata = {
        "variant": variant_name,
        "params": params,
        "prediction_mean": float(prediction.mean()),
        "prediction_std": float(prediction.std()),
        "prediction_max_abs": float(np.max(np.abs(prediction))),
        "fit_seconds": float(time.perf_counter() - started),
    }
    del model, train_x, valid_x
    gc.collect()
    return prediction, metadata


def correlation(left: np.ndarray, right: np.ndarray) -> float:
    if len(left) < 2 or np.std(left) <= 0.0 or np.std(right) <= 0.0:
        return 0.0
    return float(np.corrcoef(left, right)[0, 1])


def main() -> None:
    if OUTPUT.exists():
        raise FileExistsError(f"result already exists: {OUTPUT}")
    started = time.perf_counter()
    labels, linkage_meta = load_fine_labels()
    frame = load_main_frame(labels)
    del labels
    gc.collect()

    folds: dict[int, dict[str, Any]] = {}
    diagnostics: dict[str, Any] = {}
    for year in SOURCE_YEARS:
        anchor = load_anchor(year)
        valid = frame.iloc[anchor["row_index"]].copy()
        if not valid["season"].eq(year).all():
            raise ValueError(f"unexpected validation season in {year} anchor")
        history = frame.loc[
            (frame["season"] < year) & frame["game_type"].eq("R")
        ].copy()
        history_teacher, state, teacher_meta = build_teacher_and_state(history)
        r_mask = valid["game_type"].eq("R").to_numpy(dtype=bool)
        valid_r = valid.loc[r_mask].copy()
        oracle, matched, state_meta = apply_state(valid_r, state)
        predictions: dict[str, np.ndarray] = {}
        variant_meta: dict[str, Any] = {}
        for variant_name in VARIANTS:
            prediction, fit_meta = fit_student(
                history_teacher, valid_r, variant_name, year
            )
            fit_meta["matched_oracle_correlation"] = correlation(
                prediction[matched], oracle[matched]
            )
            fit_meta["matched_oracle_mse"] = float(
                np.mean(np.square(prediction[matched] - oracle[matched]))
            )
            predictions[variant_name] = prediction
            variant_meta[variant_name] = fit_meta
        folds[year] = {
            "anchor": anchor,
            "valid": valid,
            "game_type": valid["game_type"].astype(str).to_numpy(),
            "r_mask": r_mask,
            "oracle": oracle,
            "matched": matched,
            "predictions": predictions,
        }
        diagnostics[str(year)] = {
            "teacher": teacher_meta,
            "validation_state": state_meta,
            "students": variant_meta,
        }
        del history, history_teacher, state, valid_r
        gc.collect()

    trials: list[dict[str, Any]] = []
    for variant_name in VARIANTS:
        for gamma in GAMMAS:
            years: dict[str, Any] = {}
            for year in SOURCE_YEARS:
                fold = folds[year]
                anchor = fold["anchor"]
                base = anchor["catboost_outcome"].astype(np.float64)
                candidate = base.copy()
                candidate[fold["r_mask"]] = np.clip(
                    candidate[fold["r_mask"]]
                    + gamma * fold["predictions"][variant_name],
                    0.0,
                    1.0,
                )
                years[str(year)] = evaluate(
                    anchor["y"], base, candidate, fold["game_type"]
                )
            full_gains = [years[str(year)]["gains"]["all"] for year in SOURCE_YEARS]
            r_gains = [years[str(year)]["gains"]["R"] for year in SOURCE_YEARS]
            trials.append(
                {
                    "variant": variant_name,
                    "gamma": gamma,
                    "min_full_gain": float(min(full_gains)),
                    "min_R_gain": float(min(r_gains)),
                    "mean_full_gain": float(np.mean(full_gains)),
                    "years": years,
                }
            )
    trials.sort(
        key=lambda row: (
            row["min_full_gain"],
            row["min_R_gain"],
            row["mean_full_gain"],
            -row["gamma"],
        ),
        reverse=True,
    )
    selected = trials[0]

    intervals: dict[str, Any] = {}
    artifacts: dict[str, str] = {}
    for offset, year in enumerate(SOURCE_YEARS):
        fold = folds[year]
        anchor = fold["anchor"]
        base = anchor["catboost_outcome"].astype(np.float64)
        candidate = base.copy()
        candidate[fold["r_mask"]] = np.clip(
            candidate[fold["r_mask"]]
            + selected["gamma"] * fold["predictions"][selected["variant"]],
            0.0,
            1.0,
        )
        intervals[str(year)] = cluster_bootstrap_score_gain(
            anchor["y"],
            base,
            candidate,
            anchor["cluster"].astype(str),
            fold["r_mask"],
            2000,
            591000 + offset,
        )
        artifact = (
            PREDICTIONS
            / f"v5_control_relevant_pitch_selection_source_{year}.npz"
        )
        if artifact.exists():
            raise FileExistsError(f"prediction artifact exists: {artifact}")
        np.savez_compressed(
            artifact,
            y=anchor["y"],
            row_index=anchor["row_index"],
            cluster=anchor["cluster"],
            base=base,
            selector_prediction=fold["predictions"][selected["variant"]],
            final_prediction=candidate,
        )
        artifacts[str(year)] = str(artifact.relative_to(ROOT))

    correlations = {
        str(year): diagnostics[str(year)]["students"][selected["variant"]][
            "matched_oracle_correlation"
        ]
        for year in SOURCE_YEARS
    }
    conditions = {
        "minimum_full_gain_each_year": bool(selected["min_full_gain"] >= 5.0),
        "minimum_R_gain_each_year": bool(selected["min_R_gain"] >= 5.0),
        "ci_lower_positive_each_year": bool(
            all(intervals[str(year)]["ci_low"] > 0.0 for year in SOURCE_YEARS)
        ),
        "oracle_teacher_correlation_positive_each_year": bool(
            all(correlations[str(year)] > 0.0 for year in SOURCE_YEARS)
        ),
    }
    gate_pass = all(conditions.values())
    payload = {
        "experiment_id": "V5_CONTROL_RELEVANT_PITCH_SELECTION_V1",
        "status": "source_gate_pass" if gate_pass else "failed_source_gate",
        "preregister_sha256": sha256(PREREGISTRATION),
        "policy": {
            "test_rows_read": False,
            "latest_control_label_season_read": max(SOURCE_YEARS),
            "row_independent": True,
            "current_pitch_type_at_inference": False,
            "training_teacher_own_control_label_removed": True,
        },
        "linkage": linkage_meta,
        "diagnostics": diagnostics,
        "candidate_count": len(trials),
        "selected": selected,
        "selected_correlations": correlations,
        "selected_r_cluster_intervals": intervals,
        "conditions": conditions,
        "gate_pass": gate_pass,
        "decision": "open preregistered 2022" if gate_pass else "close without 2022+",
        "all_candidates": trials,
        "selected_prediction_artifacts": artifacts,
        "artifact_hashes": {
            "open/data/train.csv": sha256(ROOT / "open/data/train.csv"),
            "preregister": sha256(PREREGISTRATION),
            **{
                f"anchor_{year}": sha256(
                    PREDICTIONS / f"v4_m3_c_backtest_{year}_{year}.npz"
                )
                for year in SOURCE_YEARS
            },
        },
        "elapsed_seconds": float(time.perf_counter() - started),
    }
    OUTPUT.write_text(
        json.dumps(json_safe(payload), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(
        json.dumps(
            json_safe(
                {
                    "status": payload["status"],
                    "selected": {
                        key: selected[key]
                        for key in (
                            "variant",
                            "gamma",
                            "min_full_gain",
                            "min_R_gain",
                            "mean_full_gain",
                        )
                    },
                    "correlations": correlations,
                    "intervals": intervals,
                    "conditions": conditions,
                    "elapsed_seconds": payload["elapsed_seconds"],
                }
            ),
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
