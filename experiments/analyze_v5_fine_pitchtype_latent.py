#!/usr/bin/env python3
"""Source-only screen for a deployable fine-pitch latent control prior.

True current pitch type is used only for an explicitly separated historical
oracle.  Every deployable correction uses a probability vector predicted from
the current row's official pre-pitch fields and states frozen from outer
history seasons.
"""

from __future__ import annotations

import gc
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

from experiments.analyze_v4_pitchtype_failure_prior import (  # noqa: E402
    normalize_fine_pitch_type,
)
from experiments.run_baselines import (  # noqa: E402
    FEATURES as BASE_FEATURES,
    RANDOM_SEED,
    TARGET,
    load_train,
)
from experiments.run_e20r_rolling import load_joined_trackman  # noqa: E402
from experiments.run_v2_rolling import BOOSTER_CATEGORICAL  # noqa: E402
from experiments.run_v5_h1_residual import (  # noqa: E402
    cluster_bootstrap_score_gain,
)


PREDICTIONS = ROOT / "experiments/results/predictions"
PREREGISTRATION = (
    ROOT / "experiments/params/v5_fine_pitchtype_latent_preregister.json"
)
OUTPUT = ROOT / "experiments/results/v5_fine_pitchtype_latent_source.json"
SOURCE_YEARS = (2020, 2021)
FINE_TYPES = (
    "Fastball",
    "Slider",
    "Curveball",
    "ChangeUp",
    "Splitter",
    "Sinker",
    "Cutter",
    "Other",
)
LABEL_SOURCES = ("tagged", "auto")
OUTCOME_KS = (10.0, 50.0, 100.0)
REPERTOIRE_KS = (50.0, 200.0)
SELECTOR_WEIGHTS = (0.5, 0.75, 1.0)
GAMMAS = (0.1, 0.25, 0.5, 0.75, 1.0, 1.5)
ORACLE_REQUIRED_GAIN = 132.11992465


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, (np.floating, float)):
        numeric = float(value)
        return numeric if np.isfinite(numeric) else None
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    return value


def load_fine_labels() -> tuple[pd.DataFrame, dict[str, Any]]:
    joined = load_joined_trackman()
    labels = joined[["row_id", "tagged_pitch_type", "auto_pitch_type"]].copy()
    labels["row_id"] = labels["row_id"].astype(str)
    labels = labels.drop_duplicates("row_id", keep="first")
    labels["fine_tagged"] = normalize_fine_pitch_type(labels["tagged_pitch_type"])
    labels["fine_auto"] = normalize_fine_pitch_type(labels["auto_pitch_type"])
    metadata = {
        "joined_rows": int(len(joined)),
        "unique_labeled_rows": int(len(labels)),
        "counts": {
            source: {
                str(key): int(value)
                for key, value in labels[f"fine_{source}"].value_counts().items()
            }
            for source in LABEL_SOURCES
        },
    }
    result = labels[["row_id", "fine_tagged", "fine_auto"]].copy()
    del joined, labels
    gc.collect()
    return result, metadata


def load_main_frame(labels: pd.DataFrame) -> pd.DataFrame:
    frame = load_train(ROOT / "open/data/train.csv")
    row_ids = pd.read_csv(
        ROOT / "open/data/train.csv",
        usecols=["row_id"],
        dtype="string",
        encoding="utf-8-sig",
    )["row_id"]
    if len(row_ids) != len(frame):
        raise ValueError("row_id and optimized frame lengths differ")
    frame.insert(0, "row_id", row_ids.astype(str).to_numpy())
    label_map = labels.set_index("row_id")
    for source in LABEL_SOURCES:
        frame[f"fine_{source}"] = frame["row_id"].map(label_map[f"fine_{source}"])
    return frame


def load_anchor(year: int) -> dict[str, np.ndarray]:
    path = PREDICTIONS / f"v4_m3_c_backtest_{year}_{year}.npz"
    with np.load(path, allow_pickle=False) as archive:
        result = {key: np.asarray(archive[key]) for key in archive.files}
    required = {"y", "row_index", "cluster", "catboost_outcome"}
    if not required.issubset(result):
        raise ValueError(f"anchor missing keys for {year}: {sorted(required - result.keys())}")
    return result


def prepare_catboost(frame: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    categorical = [column for column in BOOSTER_CATEGORICAL if column in BASE_FEATURES]
    result = frame[BASE_FEATURES].copy()
    for column in categorical:
        result[column] = (
            result[column].astype("string").fillna("__missing__").astype(str)
        )
    return result, categorical


def fit_selector(
    history: pd.DataFrame,
    valid_r: pd.DataFrame,
    source: str,
    year: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    label_column = f"fine_{source}"
    labeled = history.loc[history[label_column].notna()].copy()
    train_x, categorical = prepare_catboost(labeled)
    valid_x, valid_categorical = prepare_catboost(valid_r)
    if categorical != valid_categorical:
        raise AssertionError("categorical schema changed")
    model = CatBoostClassifier(
        loss_function="MultiClass",
        iterations=400,
        depth=6,
        learning_rate=0.06,
        l2_leaf_reg=20.0,
        random_seed=RANDOM_SEED + year + (0 if source == "tagged" else 100),
        allow_writing_files=False,
        thread_count=6,
        task_type=(
            "GPU"
            if os.environ.get("V2_BOOSTER_DEVICE", "cpu").lower() == "gpu"
            else "CPU"
        ),
    )
    started = time.perf_counter()
    model.fit(
        train_x,
        labeled[label_column].astype(str),
        cat_features=categorical,
        verbose=False,
    )
    raw = np.asarray(model.predict_proba(valid_x), dtype=np.float64)
    probabilities = np.zeros((len(valid_r), len(FINE_TYPES)), dtype=np.float64)
    classes = [str(value) for value in model.classes_]
    for source_index, label in enumerate(classes):
        if label in FINE_TYPES:
            probabilities[:, FINE_TYPES.index(label)] = raw[:, source_index]
    denominator = probabilities.sum(axis=1)
    invalid = denominator <= 0.0
    probabilities[invalid] = 1.0 / len(FINE_TYPES)
    denominator[invalid] = 1.0
    probabilities /= denominator[:, None]

    matched = valid_r[label_column].notna().to_numpy(dtype=bool)
    truth = valid_r.loc[matched, label_column].astype(str).to_numpy()
    truth_index = np.array([FINE_TYPES.index(value) for value in truth], dtype=np.int16)
    matched_probability = probabilities[matched]
    top1 = matched_probability.argmax(axis=1)
    chosen = matched_probability[np.arange(len(truth_index)), truth_index]
    diagnostics = {
        "history_labeled_rows": int(len(labeled)),
        "valid_rows": int(len(valid_r)),
        "valid_matched_rows": int(matched.sum()),
        "classes": classes,
        "top1_accuracy": float(np.mean(top1 == truth_index)),
        "multiclass_log_loss": float(-np.mean(np.log(np.clip(chosen, 1e-12, 1.0)))),
        "probability_mean": probabilities.mean(axis=0).tolist(),
        "fit_seconds": float(time.perf_counter() - started),
        "current_pitch_type_used_for_prediction": False,
    }
    del model, train_x, valid_x, labeled, raw
    gc.collect()
    return probabilities, diagnostics


def build_control_matrices(
    history: pd.DataFrame,
    valid_r: pd.DataFrame,
    source: str,
    outcome_k: float,
    repertoire_k: float,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    label_column = f"fine_{source}"
    work = history.loc[history[label_column].notna()].copy()
    season_mean = work.groupby("season", observed=True)[TARGET].transform("mean")
    work["centered_control"] = work[TARGET].astype(float) - season_mean

    type_prior = (
        work.groupby(label_column, observed=True)["centered_control"]
        .mean()
        .reindex(FINE_TYPES)
        .fillna(0.0)
    )
    stats = work.groupby(
        ["pitcher_id", label_column], sort=False, observed=True
    )["centered_control"].agg(["sum", "count"])
    stats = stats.reset_index()
    stats["value"] = (
        stats["sum"].to_numpy(dtype=np.float64)
        + outcome_k
        * stats[label_column].map(type_prior).to_numpy(dtype=np.float64)
    ) / (stats["count"].to_numpy(dtype=np.float64) + outcome_k)
    values = stats.pivot(index="pitcher_id", columns=label_column, values="value")
    values = values.reindex(columns=FINE_TYPES)
    for pitch_type in FINE_TYPES:
        values[pitch_type] = values[pitch_type].fillna(float(type_prior[pitch_type]))

    counts = (
        work.groupby(["pitcher_id", label_column], sort=False, observed=True)
        .size()
        .unstack(fill_value=0)
        .reindex(columns=FINE_TYPES, fill_value=0)
    )
    global_mix = (
        work[label_column]
        .value_counts(normalize=True)
        .reindex(FINE_TYPES)
        .fillna(0.0)
        .to_numpy(dtype=np.float64)
    )
    count_matrix = counts.to_numpy(dtype=np.float64)
    total = count_matrix.sum(axis=1)
    mix = (count_matrix + repertoire_k * global_mix[None, :]) / (
        total[:, None] + repertoire_k
    )
    mix_table = pd.DataFrame(mix, index=counts.index, columns=FINE_TYPES)

    pitchers = valid_r["pitcher_id"]
    q_matrix = values.reindex(pitchers.to_numpy()).to_numpy(dtype=np.float64)
    mix_matrix = mix_table.reindex(pitchers.to_numpy()).to_numpy(dtype=np.float64)
    unseen = np.isnan(mix_matrix).all(axis=1)
    if unseen.any():
        mix_matrix[unseen] = global_mix
    for column, pitch_type in enumerate(FINE_TYPES):
        missing_q = np.isnan(q_matrix[:, column])
        q_matrix[missing_q, column] = float(type_prior[pitch_type])
        missing_mix = np.isnan(mix_matrix[:, column])
        mix_matrix[missing_mix, column] = global_mix[column]
    mix_matrix /= np.maximum(mix_matrix.sum(axis=1, keepdims=True), 1e-12)
    metadata = {
        "matched_history_rows": int(len(work)),
        "state_pitchers": int(len(values)),
        "valid_unseen_pitchers": int(unseen.sum()),
        "type_control_priors": {
            pitch_type: float(type_prior[pitch_type]) for pitch_type in FINE_TYPES
        },
        "global_repertoire": {
            pitch_type: float(global_mix[index])
            for index, pitch_type in enumerate(FINE_TYPES)
        },
    }
    return q_matrix, mix_matrix, metadata


def metric(y: np.ndarray, prediction: np.ndarray, mask: np.ndarray) -> dict[str, Any]:
    target = np.asarray(y[mask], dtype=np.float64)
    pred = np.asarray(prediction[mask], dtype=np.float64)
    rate = float(target.mean())
    brier = float(np.mean(np.square(pred - target)))
    return {
        "rows": int(mask.sum()),
        "target_rate": rate,
        "prediction_mean": float(pred.mean()),
        "prediction_std": float(pred.std()),
        "brier": brier,
        "score": float(100_000.0 * (1.0 - brier / (rate * (1.0 - rate)))),
    }


def evaluate(
    y: np.ndarray,
    base: np.ndarray,
    candidate: np.ndarray,
    game_type: np.ndarray,
) -> dict[str, Any]:
    masks = {
        "all": np.ones(len(y), dtype=bool),
        "R": game_type == "R",
    }
    base_metrics = {scope: metric(y, base, mask) for scope, mask in masks.items()}
    candidate_metrics = {
        scope: metric(y, candidate, mask) for scope, mask in masks.items()
    }
    return {
        "base": base_metrics,
        "candidate": candidate_metrics,
        "gains": {
            scope: float(candidate_metrics[scope]["score"] - base_metrics[scope]["score"])
            for scope in masks
        },
    }


def main() -> None:
    if OUTPUT.exists():
        raise FileExistsError(f"result already exists: {OUTPUT}")
    started = time.perf_counter()
    labels, linkage_meta = load_fine_labels()
    frame = load_main_frame(labels)
    del labels
    gc.collect()

    fold_data: dict[int, dict[str, Any]] = {}
    selector_meta: dict[str, Any] = {}
    correction_cache: dict[tuple[int, str, float, float], dict[str, Any]] = {}

    for year in SOURCE_YEARS:
        anchor = load_anchor(year)
        valid = frame.iloc[anchor["row_index"]].copy()
        if not valid["season"].eq(year).all():
            raise ValueError(f"anchor row_index is not season {year}")
        if not np.array_equal(valid[TARGET].to_numpy(dtype=np.int8), anchor["y"]):
            raise ValueError(f"anchor target mismatch for {year}")
        history = frame.loc[(frame["season"] < year) & frame["game_type"].eq("R")].copy()
        r_mask = valid["game_type"].eq("R").to_numpy(dtype=bool)
        valid_r = valid.loc[r_mask].copy()
        fold_data[year] = {
            "anchor": anchor,
            "valid": valid,
            "game_type": valid["game_type"].astype(str).to_numpy(),
            "r_mask": r_mask,
        }

        for source in LABEL_SOURCES:
            probabilities, select_meta = fit_selector(history, valid_r, source, year)
            selector_meta[f"{year}_{source}"] = select_meta
            for outcome_k in OUTCOME_KS:
                for repertoire_k in REPERTOIRE_KS:
                    q_matrix, mix_matrix, state_meta = build_control_matrices(
                        history,
                        valid_r,
                        source,
                        outcome_k,
                        repertoire_k,
                    )
                    classifier_delta = np.sum(
                        (probabilities - mix_matrix) * q_matrix,
                        axis=1,
                    )
                    label_column = f"fine_{source}"
                    matched = valid_r[label_column].notna().to_numpy(dtype=bool)
                    truth = valid_r.loc[matched, label_column].astype(str).to_numpy()
                    truth_index = np.array(
                        [FINE_TYPES.index(value) for value in truth], dtype=np.int16
                    )
                    oracle_delta = np.zeros(len(valid_r), dtype=np.float64)
                    baseline_expected = np.sum(mix_matrix * q_matrix, axis=1)
                    oracle_delta[matched] = (
                        q_matrix[matched, truth_index] - baseline_expected[matched]
                    )
                    correction_cache[(year, source, outcome_k, repertoire_k)] = {
                        "classifier_delta": classifier_delta,
                        "oracle_delta": oracle_delta,
                        "matched": matched,
                        "state": state_meta,
                    }
        del history, valid_r
        gc.collect()

    trials: list[dict[str, Any]] = []
    for source in LABEL_SOURCES:
        for outcome_k in OUTCOME_KS:
            for repertoire_k in REPERTOIRE_KS:
                for selector_weight in SELECTOR_WEIGHTS:
                    for gamma in GAMMAS:
                        years: dict[str, Any] = {}
                        for year in SOURCE_YEARS:
                            fold = fold_data[year]
                            anchor = fold["anchor"]
                            base = anchor["catboost_outcome"].astype(np.float64)
                            candidate = base.copy()
                            delta = correction_cache[
                                (year, source, outcome_k, repertoire_k)
                            ]["classifier_delta"]
                            candidate[fold["r_mask"]] = np.clip(
                                candidate[fold["r_mask"]]
                                + gamma * selector_weight * delta,
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
                                "source": source,
                                "outcome_k": outcome_k,
                                "repertoire_k": repertoire_k,
                                "selector_weight": selector_weight,
                                "gamma": gamma,
                                "effective_scale": selector_weight * gamma,
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
            -row["effective_scale"],
        ),
        reverse=True,
    )
    selected = trials[0]

    selected_intervals: dict[str, Any] = {}
    selected_artifacts: dict[str, str] = {}
    for offset, year in enumerate(SOURCE_YEARS):
        fold = fold_data[year]
        anchor = fold["anchor"]
        base = anchor["catboost_outcome"].astype(np.float64)
        candidate = base.copy()
        delta = correction_cache[
            (
                year,
                selected["source"],
                selected["outcome_k"],
                selected["repertoire_k"],
            )
        ]["classifier_delta"]
        candidate[fold["r_mask"]] = np.clip(
            candidate[fold["r_mask"]]
            + selected["effective_scale"] * delta,
            0.0,
            1.0,
        )
        selected_intervals[str(year)] = cluster_bootstrap_score_gain(
            anchor["y"],
            base,
            candidate,
            anchor["cluster"].astype(str),
            fold["r_mask"],
            2000,
            590000 + offset,
        )
        artifact_path = (
            PREDICTIONS / f"v5_fine_pitchtype_latent_source_{year}.npz"
        )
        if artifact_path.exists():
            raise FileExistsError(f"prediction artifact exists: {artifact_path}")
        np.savez_compressed(
            artifact_path,
            y=anchor["y"],
            row_index=anchor["row_index"],
            cluster=anchor["cluster"],
            base=base,
            final_prediction=candidate,
        )
        selected_artifacts[str(year)] = str(artifact_path.relative_to(ROOT))

    oracle_trials: list[dict[str, Any]] = []
    for source in LABEL_SOURCES:
        for outcome_k in OUTCOME_KS:
            for repertoire_k in REPERTOIRE_KS:
                for gamma in GAMMAS:
                    years: dict[str, Any] = {}
                    for year in SOURCE_YEARS:
                        fold = fold_data[year]
                        anchor = fold["anchor"]
                        base = anchor["catboost_outcome"].astype(np.float64)
                        candidate = base.copy()
                        oracle = correction_cache[
                            (year, source, outcome_k, repertoire_k)
                        ]["oracle_delta"]
                        candidate[fold["r_mask"]] = np.clip(
                            candidate[fold["r_mask"]] + gamma * oracle,
                            0.0,
                            1.0,
                        )
                        years[str(year)] = evaluate(
                            anchor["y"], base, candidate, fold["game_type"]
                        )
                    gains = [years[str(year)]["gains"]["all"] for year in SOURCE_YEARS]
                    oracle_trials.append(
                        {
                            "source": source,
                            "outcome_k": outcome_k,
                            "repertoire_k": repertoire_k,
                            "gamma": gamma,
                            "min_full_gain": float(min(gains)),
                            "mean_full_gain": float(np.mean(gains)),
                            "years": years,
                        }
                    )
    oracle_trials.sort(
        key=lambda row: (row["min_full_gain"], row["mean_full_gain"]), reverse=True
    )
    oracle_selected = oracle_trials[0]

    conditions = {
        "minimum_full_gain_each_year": bool(selected["min_full_gain"] >= 5.0),
        "minimum_R_gain_each_year": bool(selected["min_R_gain"] >= 5.0),
        "ci_lower_positive_each_year": bool(
            all(selected_intervals[str(year)]["ci_low"] > 0.0 for year in SOURCE_YEARS)
        ),
        "diagnostic_oracle_full_equivalent": bool(
            oracle_selected["min_full_gain"] >= ORACLE_REQUIRED_GAIN
        ),
    }
    gate_pass = all(conditions.values())
    payload = {
        "experiment_id": "V5_FINE_PITCHTYPE_LATENT_V1",
        "status": "source_gate_pass" if gate_pass else "failed_source_gate",
        "preregister_sha256": sha256(PREREGISTRATION),
        "policy": {
            "test_rows_read": False,
            "latest_control_label_season_read": max(SOURCE_YEARS),
            "row_independent": True,
            "true_current_pitch_type_in_deployable_prediction": False,
        },
        "linkage": linkage_meta,
        "selector_diagnostics": selector_meta,
        "candidate_count": len(trials),
        "selected": selected,
        "selected_r_cluster_intervals": selected_intervals,
        "diagnostic_oracle": {
            "selected": oracle_selected,
            "candidate_count": len(oracle_trials),
            "for_deployment": False,
        },
        "conditions": conditions,
        "gate_pass": gate_pass,
        "decision": (
            "preregister shared fine-type conditional model before 2022"
            if gate_pass
            else "close without 2022+"
        ),
        "top_candidates": trials[:30],
        "top_oracle_candidates": oracle_trials[:15],
        "selected_prediction_artifacts": selected_artifacts,
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
                            "source",
                            "outcome_k",
                            "repertoire_k",
                            "selector_weight",
                            "gamma",
                            "effective_scale",
                            "min_full_gain",
                            "min_R_gain",
                        )
                    },
                    "selected_ci": selected_intervals,
                    "oracle": {
                        key: oracle_selected[key]
                        for key in (
                            "source",
                            "outcome_k",
                            "repertoire_k",
                            "gamma",
                            "min_full_gain",
                        )
                    },
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
