#!/usr/bin/env python3
"""Source-only row-local current-season pitchmix selector experiment."""

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

from experiments.analyze_v5_counter_reconstructed_pitch_hierarchy import (  # noqa: E402
    build_control_matrices,
    derive_coarse_labels,
    direction_record,
    gains_from_record,
)
from experiments.analyze_v5_fine_pitchtype_latent import (  # noqa: E402
    PREDICTIONS,
    SOURCE_YEARS,
    TARGET,
    evaluate,
    json_safe,
    load_anchor,
)
from experiments.run_baselines import (  # noqa: E402
    FEATURES as BASE_FEATURES,
    RANDOM_SEED,
)
from experiments.run_v2_rolling import BOOSTER_CATEGORICAL  # noqa: E402
from experiments.run_v5_h1_residual import (  # noqa: E402
    cluster_bootstrap_score_gain,
)


TRAIN = ROOT / "open/data/train.csv"
PREREG = ROOT / "experiments/params/v5_row_local_pitchmix_selector_preregister.json"
REPORT = ROOT / "experiments/results/v5_row_local_pitchmix_selector_source.json"
GROUPS = ("fastball", "breaking", "offspeed")
RATE_COLUMNS = (
    "asof_pitcher_fastball_rate",
    "asof_pitcher_breaking_rate",
    "asof_pitcher_offspeed_rate",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_source() -> pd.DataFrame:
    with np.load(
        PREDICTIONS / "v4_m3_c_backtest_2021_2021.npz", allow_pickle=False
    ) as archive:
        last_index = int(np.max(archive["row_index"]))
    columns = list(dict.fromkeys(["row_id", *BASE_FEATURES, TARGET]))
    frame = pd.read_csv(TRAIN, usecols=columns, nrows=last_index + 1)
    if set(frame["season"].unique()) != {2019, 2020, 2021}:
        raise ValueError("Row-local selector read a control label after 2021")
    return frame


def row_pitchmix_counts(frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    n = (
        pd.to_numeric(frame["asof_pitcher_pitchmix_n"], errors="coerce")
        .fillna(0)
        .to_numpy(dtype=np.int64)
    )
    counts = np.column_stack(
        [
            np.rint(
                pd.to_numeric(frame[column], errors="coerce")
                .fillna(0.0)
                .to_numpy(dtype=np.float64)
                * n
            ).astype(np.int64)
            for column in RATE_COLUMNS
        ]
    )
    return n, counts


def pitchmix_states_before_each_season(
    frame: pd.DataFrame,
) -> tuple[
    dict[int, dict[int, tuple[int, int, int, int]]],
    dict[int, tuple[int, int, int, int]],
]:
    """Freeze the observable as-of state at the final row of each season."""
    before: dict[int, dict[int, tuple[int, int, int, int]]] = {}
    state: dict[int, tuple[int, int, int, int]] = {}
    for season in sorted(int(value) for value in frame["season"].unique()):
        before[season] = dict(state)
        rows = frame.loc[frame["season"].eq(season)]
        tail = rows.groupby("pitcher_id", sort=False, observed=True).tail(1)
        n, counts = row_pitchmix_counts(tail)
        for pitcher, total, values in zip(
            tail["pitcher_id"].to_numpy(), n, counts, strict=True
        ):
            state[int(pitcher)] = (
                int(total), int(values[0]), int(values[1]), int(values[2])
            )
    return before, state


def state_features(
    frame: pd.DataFrame,
    states_before: dict[int, dict[int, tuple[int, int, int, int]]],
    ks: list[int],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    n, counts = row_pitchmix_counts(frame)
    pre_n = np.zeros(len(frame), dtype=np.int64)
    pre_counts = np.zeros((len(frame), len(GROUPS)), dtype=np.int64)
    unseen = np.ones(len(frame), dtype=bool)
    for index, (season, pitcher) in enumerate(
        zip(frame["season"].to_numpy(), frame["pitcher_id"].to_numpy(), strict=True)
    ):
        state = states_before.get(int(season), {}).get(int(pitcher))
        if state is not None:
            unseen[index] = False
            pre_n[index] = state[0]
            pre_counts[index] = state[1:]

    # Completed-history league prior for unseen pitchers, computed separately
    # for each row season from only its frozen preseason state dictionary.
    global_prior_by_season: dict[int, np.ndarray] = {}
    for season in np.unique(frame["season"]):
        state = states_before.get(int(season), {})
        denominator = float(sum(value[0] for value in state.values()))
        if denominator <= 0.0:
            global_prior_by_season[int(season)] = np.full(len(GROUPS), 1.0 / len(GROUPS))
        else:
            numerator = np.asarray(
                [sum(value[index + 1] for value in state.values()) for index in range(3)],
                dtype=np.float64,
            )
            prior = numerator / denominator
            prior /= np.maximum(prior.sum(), 1e-12)
            global_prior_by_season[int(season)] = prior
    global_prior = np.vstack(
        [global_prior_by_season[int(season)] for season in frame["season"]]
    )
    completed = np.divide(
        pre_counts,
        pre_n[:, None],
        out=global_prior.copy(),
        where=pre_n[:, None] > 0,
    )
    completed /= np.maximum(completed.sum(axis=1, keepdims=True), 1e-12)
    delta_n = n - pre_n
    delta_counts = counts - pre_counts
    invalid = (
        (delta_n < 0)
        | np.any(delta_counts < 0, axis=1)
        | np.any(delta_counts > delta_n[:, None], axis=1)
    )
    safe_n = np.where(invalid, 0, delta_n)
    safe_counts = np.where(invalid[:, None], 0, delta_counts)
    values: dict[str, np.ndarray] = {
        "pmx_state_n": safe_n.astype(np.float32),
        "pmx_state_log_n": np.log1p(safe_n).astype(np.float32),
        "pmx_state_unseen": unseen.astype(np.int8),
        "pmx_state_invalid": invalid.astype(np.int8),
    }
    for column, group in enumerate(GROUPS):
        values[f"pmx_completed_p_{group}"] = completed[:, column].astype(np.float32)
    for k in ks:
        rate = (safe_counts + float(k) * completed) / (safe_n[:, None] + float(k))
        rate /= np.maximum(rate.sum(axis=1, keepdims=True), 1e-12)
        for column, group in enumerate(GROUPS):
            values[f"pmx_state_p_{group}_k{k}"] = rate[:, column].astype(np.float32)
            values[f"pmx_state_delta_{group}_k{k}"] = (
                rate[:, column] - completed[:, column]
            ).astype(np.float32)
    return pd.DataFrame(values, index=frame.index), {
        "rows": int(len(frame)),
        "unseen_rows": int(unseen.sum()),
        "invalid_rows": int(invalid.sum()),
        "positive_state_n_rate": float(np.mean(safe_n > 0)),
        "median_state_n": float(np.median(safe_n)),
    }


def prepare_model_frame(
    frame: pd.DataFrame, state: pd.DataFrame | None
) -> tuple[pd.DataFrame, list[str]]:
    result = frame[BASE_FEATURES].copy()
    if state is not None:
        result = pd.concat([result, state], axis=1)
    categorical = [column for column in BOOSTER_CATEGORICAL if column in result.columns]
    for column in categorical:
        result[column] = result[column].astype("string").fillna("__missing__").astype(str)
    return result, categorical


def fit_selector(
    history: pd.DataFrame,
    valid_r: pd.DataFrame,
    history_state: pd.DataFrame | None,
    valid_state: pd.DataFrame | None,
    year: int,
    name: str,
    prereg: dict[str, Any],
) -> tuple[np.ndarray, dict[str, Any]]:
    labeled = history.loc[history["coarse_reconstructed"].notna()].copy()
    train_state = None if history_state is None else history_state.loc[labeled.index]
    train_x, categorical = prepare_model_frame(labeled, train_state)
    valid_x, valid_categorical = prepare_model_frame(valid_r, valid_state)
    if categorical != valid_categorical:
        raise AssertionError("Selector categorical schema changed")
    model = CatBoostClassifier(
        loss_function="MultiClass",
        iterations=400,
        depth=6,
        learning_rate=0.06,
        l2_leaf_reg=20.0,
        random_seed=RANDOM_SEED + year + (1000 if name == "state" else 700),
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
        labeled["coarse_reconstructed"].astype(str),
        cat_features=categorical,
        verbose=False,
    )
    raw = np.asarray(model.predict_proba(valid_x), dtype=np.float64)
    probabilities = np.zeros((len(valid_r), len(GROUPS)), dtype=np.float64)
    classes = [str(value) for value in model.classes_]
    for source_index, label in enumerate(classes):
        if label in GROUPS:
            probabilities[:, GROUPS.index(label)] = raw[:, source_index]
    probabilities /= np.maximum(probabilities.sum(axis=1, keepdims=True), 1e-12)
    meta = {
        "name": name,
        "history_labeled_rows": int(len(labeled)),
        "valid_rows": int(len(valid_r)),
        "feature_count": int(train_x.shape[1]),
        "classes": classes,
        "fit_seconds": float(time.perf_counter() - started),
    }
    del model, train_x, valid_x, raw, labeled
    gc.collect()
    return probabilities, meta


def probability_metrics(probabilities: np.ndarray, labels: pd.Series) -> dict[str, Any]:
    matched = labels.notna().to_numpy(dtype=bool)
    truth = np.asarray([GROUPS.index(value) for value in labels.loc[matched]], dtype=np.int16)
    chosen = probabilities[matched, truth]
    return {
        "rows": int(matched.sum()),
        "log_loss": float(-np.mean(np.log(np.clip(chosen, 1e-12, 1.0)))),
        "top1_accuracy": float(np.mean(probabilities[matched].argmax(axis=1) == truth)),
    }


def history_context_probabilities(
    history: pd.DataFrame,
    valid_r: pd.DataFrame,
    variant: str,
    k: float,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    work = history.loc[history["coarse_reconstructed"].notna()].copy()
    global_mix = (
        work["coarse_reconstructed"]
        .value_counts(normalize=True)
        .reindex(GROUPS)
        .fillna(0.0)
        .to_numpy(dtype=np.float64)
    )
    pitcher_counts = (
        work.groupby(["pitcher_id", "coarse_reconstructed"], observed=True)
        .size()
        .unstack(fill_value=0)
        .reindex(columns=GROUPS, fill_value=0)
    )
    matrix = pitcher_counts.to_numpy(dtype=np.float64)
    total = matrix.sum(axis=1)
    pitcher_table = pd.DataFrame(
        (matrix + k * global_mix[None, :]) / (total[:, None] + k),
        index=pitcher_counts.index,
        columns=GROUPS,
    )
    pitcher = pitcher_table.reindex(valid_r["pitcher_id"].to_numpy()).to_numpy(
        dtype=np.float64
    )
    unseen = np.isnan(pitcher).all(axis=1)
    pitcher[unseen] = global_mix
    pitcher /= np.maximum(pitcher.sum(axis=1, keepdims=True), 1e-12)
    if variant == "none":
        return pitcher.copy(), pitcher, {
            "variant": variant,
            "k": float(k),
            "history_rows": int(len(work)),
            "unseen_pitcher_rows": int(unseen.sum()),
        }
    keys = ["pitcher_id", "balls_before", "strikes_before"]
    if variant == "pitcher_hand_count":
        keys.append("batter_hand")
    elif variant != "pitcher_count":
        raise ValueError(f"Unknown context variant: {variant}")
    cell_counts = (
        work.groupby([*keys, "coarse_reconstructed"], observed=True)
        .size()
        .unstack(fill_value=0)
        .reindex(columns=GROUPS, fill_value=0)
    )
    cell_matrix = cell_counts.to_numpy(dtype=np.float64)
    cell_total = cell_matrix.sum(axis=1)
    cell_index = cell_counts.index
    if len(keys) == 1:
        parent_keys = np.asarray(cell_index)
    else:
        parent_keys = cell_index.get_level_values("pitcher_id").to_numpy()
    parent = pitcher_table.reindex(parent_keys).to_numpy(dtype=np.float64)
    missing_parent = np.isnan(parent).all(axis=1)
    parent[missing_parent] = global_mix
    cell_prob = (cell_matrix + k * parent) / (cell_total[:, None] + k)
    cell_table = pd.DataFrame(cell_prob, index=cell_index, columns=GROUPS)
    lookup = pd.MultiIndex.from_frame(valid_r[keys])
    context = cell_table.reindex(lookup).to_numpy(dtype=np.float64)
    missing = np.isnan(context).all(axis=1)
    context[missing] = pitcher[missing]
    context /= np.maximum(context.sum(axis=1, keepdims=True), 1e-12)
    return context, pitcher, {
        "variant": variant,
        "k": float(k),
        "history_rows": int(len(work)),
        "cells": int(len(cell_table)),
        "unseen_pitcher_rows": int(unseen.sum()),
        "unseen_context_rows": int(missing.sum()),
    }


def normalize(probabilities: np.ndarray) -> np.ndarray:
    result = np.clip(probabilities, 1e-12, None)
    return result / result.sum(axis=1, keepdims=True)


def main() -> None:
    if REPORT.exists():
        raise FileExistsError(f"Preserve immutable row-local selector report: {REPORT}")
    prereg = json.loads(PREREG.read_text(encoding="utf-8"))
    if prereg["status"] != "preregistered_before_source_metrics":
        raise ValueError("Unexpected preregistration status")
    started = time.perf_counter()
    frame = load_source()
    state_ks = [int(value) for value in prereg["row_local_state"]["state_k"]]
    folds: dict[int, dict[str, Any]] = {}
    model_meta: dict[str, Any] = {}
    state_meta: dict[str, Any] = {}

    for year in SOURCE_YEARS:
        anchor = load_anchor(year)
        valid = frame.iloc[anchor["row_index"]].copy()
        if not valid["season"].eq(year).all():
            raise ValueError(f"{year}: anchor season mismatch")
        if not np.array_equal(
            valid[TARGET].to_numpy(dtype=np.int8), anchor["y"].astype(np.int8)
        ):
            raise ValueError(f"{year}: anchor target mismatch")
        history_all = frame.loc[frame["season"].lt(year)].copy()
        history_all["coarse_reconstructed"] = derive_coarse_labels(history_all)
        valid["coarse_reconstructed"] = derive_coarse_labels(valid)
        states_before, final_state = pitchmix_states_before_each_season(history_all)
        history_state, history_state_meta = state_features(
            history_all, states_before, state_ks
        )
        valid_state, valid_state_meta = state_features(
            valid, {year: final_state}, state_ks
        )
        history_r = history_all.loc[history_all["game_type"].eq("R")].copy()
        history_state_r = history_state.loc[history_r.index]
        regular = valid["game_type"].eq("R").to_numpy(dtype=bool)
        valid_r = valid.loc[regular].copy()
        valid_state_r = valid_state.loc[valid_r.index]
        base_prob, base_meta = fit_selector(
            history_r, valid_r, None, None, year, "base", prereg
        )
        state_prob, enhanced_meta = fit_selector(
            history_r,
            valid_r,
            history_state_r,
            valid_state_r,
            year,
            "state",
            prereg,
        )
        model_meta[str(year)] = {"base": base_meta, "state": enhanced_meta}
        state_meta[str(year)] = {
            "history": history_state_meta,
            "valid": valid_state_meta,
        }
        folds[year] = {
            "anchor": anchor,
            "valid": valid,
            "valid_r": valid_r,
            "history_r": history_r,
            "valid_state_r": valid_state_r,
            "regular": regular,
            "game_type": valid["game_type"].astype(str).to_numpy(),
            "base_probability": base_prob,
            "state_probability": state_prob,
            "base_metrics": probability_metrics(
                base_prob, valid_r["coarse_reconstructed"]
            ),
            "state_metrics": probability_metrics(
                state_prob, valid_r["coarse_reconstructed"]
            ),
        }
        del history_all, history_state, valid_state, history_state_r
        gc.collect()

    variants = [str(value) for value in prereg["completed_history_context"]["variants"]]
    context_ks = [int(value) for value in prereg["completed_history_context"]["context_k"]]
    lambdas = [float(value) for value in prereg["completed_history_context"]["tilt_lambda"]]
    weights = [
        float(value)
        for value in prereg["selector_selection"]["blend_state_catboost_weight"]
    ]
    selector_trials: list[dict[str, Any]] = []
    context_cache: dict[tuple[int, str, int], tuple[np.ndarray, np.ndarray]] = {}
    context_meta: dict[str, Any] = {}
    for year in SOURCE_YEARS:
        fold = folds[year]
        for variant in variants:
            for context_k in context_ks:
                context, pitcher, meta = history_context_probabilities(
                    fold["history_r"], fold["valid_r"], variant, float(context_k)
                )
                context_cache[(year, variant, context_k)] = (context, pitcher)
                context_meta[f"{year}:{variant}:k{context_k}"] = meta

    for state_k in state_ks:
        for variant in variants:
            relevant_context_ks = context_ks if variant != "none" else [max(context_ks)]
            relevant_lambdas = lambdas if variant != "none" else [0.0]
            for context_k in relevant_context_ks:
                for tilt_lambda in relevant_lambdas:
                    structured_by_year: dict[int, np.ndarray] = {}
                    for year in SOURCE_YEARS:
                        fold = folds[year]
                        state_probability = np.column_stack(
                            [
                                fold["valid_state_r"][f"pmx_state_p_{group}_k{state_k}"]
                                .to_numpy(dtype=np.float64)
                                for group in GROUPS
                            ]
                        )
                        if variant == "none":
                            structured = state_probability
                        else:
                            context, pitcher = context_cache[
                                (year, variant, context_k)
                            ]
                            ratio = np.divide(
                                context,
                                pitcher,
                                out=np.ones_like(context),
                                where=pitcher > 1e-12,
                            )
                            structured = normalize(
                                state_probability * np.power(ratio, tilt_lambda)
                            )
                        structured_by_year[year] = structured
                    for weight in weights:
                        years: dict[str, Any] = {}
                        for year in SOURCE_YEARS:
                            fold = folds[year]
                            probability = normalize(
                                weight * fold["state_probability"]
                                + (1.0 - weight) * structured_by_year[year]
                            )
                            metric = probability_metrics(
                                probability, fold["valid_r"]["coarse_reconstructed"]
                            )
                            base_metric = fold["base_metrics"]
                            years[str(year)] = {
                                **metric,
                                "log_loss_improvement": float(
                                    base_metric["log_loss"] - metric["log_loss"]
                                ),
                                "accuracy_improvement": float(
                                    metric["top1_accuracy"]
                                    - base_metric["top1_accuracy"]
                                ),
                            }
                        selector_trials.append(
                            {
                                "state_k": state_k,
                                "context_variant": variant,
                                "context_k": context_k,
                                "tilt_lambda": tilt_lambda,
                                "state_catboost_weight": weight,
                                "min_log_loss_improvement": float(
                                    min(
                                        value["log_loss_improvement"]
                                        for value in years.values()
                                    )
                                ),
                                "mean_log_loss_improvement": float(
                                    np.mean(
                                        [
                                            value["log_loss_improvement"]
                                            for value in years.values()
                                        ]
                                    )
                                ),
                                "min_accuracy_improvement": float(
                                    min(
                                        value["accuracy_improvement"]
                                        for value in years.values()
                                    )
                                ),
                                "years": years,
                            }
                        )
    selector_trials.sort(
        key=lambda row: (
            row["min_log_loss_improvement"],
            row["mean_log_loss_improvement"],
            row["min_accuracy_improvement"],
            row["context_variant"] == "none",
            row["state_k"],
            row["context_k"],
            -row["state_catboost_weight"],
        ),
        reverse=True,
    )
    selected_selector = selector_trials[0]

    def selected_probability(year: int) -> np.ndarray:
        fold = folds[year]
        state_k = selected_selector["state_k"]
        state_probability = np.column_stack(
            [
                fold["valid_state_r"][f"pmx_state_p_{group}_k{state_k}"].to_numpy(
                    dtype=np.float64
                )
                for group in GROUPS
            ]
        )
        variant = selected_selector["context_variant"]
        if variant == "none":
            structured = state_probability
        else:
            context, pitcher = context_cache[
                (year, variant, selected_selector["context_k"])
            ]
            ratio = np.divide(
                context,
                pitcher,
                out=np.ones_like(context),
                where=pitcher > 1e-12,
            )
            structured = normalize(
                state_probability
                * np.power(ratio, selected_selector["tilt_lambda"])
            )
        weight = selected_selector["state_catboost_weight"]
        return normalize(
            weight * fold["state_probability"] + (1.0 - weight) * structured
        )

    locked_probabilities = {year: selected_probability(year) for year in SOURCE_YEARS}
    control = prereg["control_correction_after_selector_lock"]
    outcome_ks = [int(value) for value in control["outcome_k"]]
    repertoire_ks = [int(value) for value in control["repertoire_k"]]
    gammas = [float(value) for value in control["gammas"]]
    records: dict[tuple[int, int, int], tuple[dict[str, Any], np.ndarray]] = {}
    control_meta: dict[str, Any] = {}
    for year in SOURCE_YEARS:
        fold = folds[year]
        base = fold["anchor"]["catboost_outcome"].astype(np.float64)
        for outcome_k in outcome_ks:
            for repertoire_k in repertoire_ks:
                q, mix, meta = build_control_matrices(
                    fold["history_r"],
                    fold["valid_r"],
                    "coarse_reconstructed",
                    GROUPS,
                    float(outcome_k),
                    float(repertoire_k),
                )
                delta = np.sum((locked_probabilities[year] - mix) * q, axis=1)
                records[(year, outcome_k, repertoire_k)] = (
                    direction_record(
                        fold["anchor"], base, fold["regular"], delta
                    ),
                    delta,
                )
                control_meta[f"{year}:k{outcome_k}:r{repertoire_k}"] = meta
    control_trials: list[dict[str, Any]] = []
    for outcome_k in outcome_ks:
        for repertoire_k in repertoire_ks:
            for gamma in gammas:
                years: dict[str, Any] = {}
                for year in SOURCE_YEARS:
                    record, _ = records[(year, outcome_k, repertoire_k)]
                    full_gain, r_gain = gains_from_record(record, gamma)
                    years[str(year)] = {
                        "full_gain": full_gain,
                        "R_gain": r_gain,
                    }
                control_trials.append(
                    {
                        "outcome_k": outcome_k,
                        "repertoire_k": repertoire_k,
                        "gamma": gamma,
                        "min_full_gain": float(
                            min(value["full_gain"] for value in years.values())
                        ),
                        "min_R_gain": float(
                            min(value["R_gain"] for value in years.values())
                        ),
                        "mean_full_gain": float(
                            np.mean([value["full_gain"] for value in years.values()])
                        ),
                        "years": years,
                    }
                )
    control_trials.sort(
        key=lambda row: (
            row["min_full_gain"],
            row["min_R_gain"],
            row["mean_full_gain"],
            row["outcome_k"],
            row["repertoire_k"],
            -row["gamma"],
        ),
        reverse=True,
    )
    selected_control = control_trials[0]
    intervals: dict[str, Any] = {}
    detailed_metrics: dict[str, Any] = {}
    for offset, year in enumerate(SOURCE_YEARS):
        fold = folds[year]
        anchor = fold["anchor"]
        base = anchor["catboost_outcome"].astype(np.float64)
        _, delta = records[
            (year, selected_control["outcome_k"], selected_control["repertoire_k"])
        ]
        candidate = base.copy()
        candidate[fold["regular"]] = np.clip(
            candidate[fold["regular"]] + selected_control["gamma"] * delta,
            1e-6,
            1.0 - 1e-6,
        )
        intervals[str(year)] = cluster_bootstrap_score_gain(
            anchor["y"],
            base,
            candidate,
            anchor["cluster"].astype(str),
            fold["regular"],
            2000,
            53500 + offset,
        )
        detailed_metrics[str(year)] = evaluate(
            anchor["y"], base, candidate, fold["game_type"]
        )

    selector_required = float(
        prereg["source_gate"]["selector_log_loss_improvement_each_year"]
    )
    gate = prereg["source_gate"]
    conditions = {
        "selector_log_loss_improvement_each_year": bool(
            selected_selector["min_log_loss_improvement"] >= selector_required
        ),
        "minimum_full_gain_each_year": bool(
            selected_control["min_full_gain"]
            >= float(gate["minimum_full_gain_each_year"])
        ),
        "minimum_R_gain_each_year": bool(
            selected_control["min_R_gain"]
            >= float(gate["minimum_r_gain_each_year"])
        ),
        "R_cluster_ci_lower_positive_each_year": bool(
            all(value["ci_low"] > 0.0 for value in intervals.values())
        ),
    }
    passed = bool(all(conditions.values()))
    payload = {
        "experiment_id": prereg["experiment_id"],
        "status": "passed_source_gate" if passed else "failed_source_gate",
        "preregister": str(PREREG.relative_to(ROOT)),
        "preregister_sha256": sha256(PREREG),
        "policy": {
            "official_data_only": True,
            "test_rows_read": False,
            "latest_control_label_season_used_for_metrics": 2021,
            "selector_selected_without_control_target": True,
            "current_pitch_group_at_inference": False,
            "other_validation_rows_at_inference": False,
            "row_independent_inference": True,
            "automatic_submission": False,
        },
        "model_metadata": model_meta,
        "state_metadata": state_meta,
        "base_selector_metrics": {
            str(year): folds[year]["base_metrics"] for year in SOURCE_YEARS
        },
        "state_selector_metrics": {
            str(year): folds[year]["state_metrics"] for year in SOURCE_YEARS
        },
        "selector_candidate_count": int(len(selector_trials)),
        "selected_selector": selected_selector,
        "control_candidate_count": int(len(control_trials)),
        "selected_control": selected_control,
        "selected_control_detailed_metrics": detailed_metrics,
        "selected_R_pitcher_cluster_intervals": intervals,
        "conditions": conditions,
        "gate_pass": passed,
        "decision": (
            "freeze row-local selector recipe before 2022"
            if passed
            else "close without 2022+ labels"
        ),
        "top_selector_candidates": selector_trials[:30],
        "top_control_candidates": control_trials[:15],
        "context_metadata": context_meta,
        "control_state_metadata": control_meta,
        "elapsed_seconds": float(time.perf_counter() - started),
    }
    REPORT.write_text(
        json.dumps(json_safe(payload), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(
        json.dumps(
            json_safe(
                {
                    "status": payload["status"],
                    "base_selector_metrics": payload["base_selector_metrics"],
                    "state_selector_metrics": payload["state_selector_metrics"],
                    "selected_selector": selected_selector,
                    "selected_control": selected_control,
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
