#!/usr/bin/env python3
"""Source-only audit of frozen-boundary current-appearance deconvolution."""

from __future__ import annotations

import gc
import hashlib
import json
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.analyze_v5_fine_pitchtype_latent import load_anchor  # noqa: E402
from experiments.run_e14_rolling import (  # noqa: E402
    build_e14_features,
    prior_before_each_season,
    season_end_state,
)
from experiments.run_v2_rolling import build_recent_denominator_features  # noqa: E402
from experiments.run_v5_h1_residual import cluster_bootstrap_score_gain  # noqa: E402
from experiments.v5_recent_workload_decoder_features import (  # noqa: E402
    _fit_models,
    _predict_rows,
    derive_game_ids,
    reconstruct_appearances,
)


TRAIN = ROOT / "open/data/train.csv"
PREREG = (
    ROOT
    / "experiments/params/v5_boundary_appearance_deconvolution_source_preregister.json"
)
DECODER_PREREG = ROOT / "experiments/params/v5_recent_workload_decoder_preregister.json"
REPORT = ROOT / "experiments/results/v5_boundary_appearance_deconvolution_source.json"
TARGET = "control_success"
YEARS = (2020, 2021)
WINDOW = 5


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return safe(value.tolist())
    if isinstance(value, (np.floating, float)):
        number = float(value)
        return number if np.isfinite(number) else None
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    return value


def score(y: np.ndarray, prediction: np.ndarray, mask: np.ndarray) -> dict[str, Any]:
    target = y[mask].astype(np.float64)
    estimate = prediction[mask].astype(np.float64)
    rate = float(target.mean())
    reference = max(rate * (1.0 - rate), 1e-12)
    brier = float(np.mean(np.square(estimate - target)))
    return {
        "rows": int(mask.sum()),
        "target_rate": rate,
        "prediction_mean": float(estimate.mean()),
        "prediction_std": float(estimate.std()),
        "brier": brier,
        "score": float(100000.0 * (1.0 - brier / reference)),
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
        "F": game_type == "F",
    }
    base_metrics = {name: score(y, base, mask) for name, mask in masks.items()}
    candidate_metrics = {
        name: score(y, candidate, mask) for name, mask in masks.items()
    }
    return {
        "base": base_metrics,
        "candidate": candidate_metrics,
        "gains": {
            name: float(candidate_metrics[name]["score"] - base_metrics[name]["score"])
            for name in masks
        },
    }


def load_source() -> pd.DataFrame:
    anchor = load_anchor(max(YEARS))
    final_index = int(np.max(anchor["row_index"]))
    columns = [
        "season",
        "game_month",
        "game_dayofweek",
        "pitcher_team_id",
        "batter_team_id",
        "top_bottom",
        "inning",
        "run_total_before",
        "pitcher_id",
        "game_type",
        "asof_pitcher_n",
        "asof_pitcher_success_rate",
        TARGET,
    ]
    for horizon in (1, 3, 5):
        columns.extend(
            [
                f"asof_pitcher_prev{horizon}_game_success_rate",
                f"asof_pitcher_prev{horizon}_game_middle_rate",
            ]
        )
    frame = pd.read_csv(
        TRAIN,
        usecols=columns,
        nrows=final_index + 1,
        encoding="utf-8-sig",
    )
    frame.index = np.arange(len(frame), dtype=np.int64)
    if set(int(value) for value in frame["season"].unique()) != {2019, 2020, 2021}:
        raise ValueError("source audit read a season outside 2019-2021")
    return frame


def build_appearance_panel(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    work = frame.copy()
    work["_game_id"] = derive_game_ids(work)
    work["_row_index"] = np.arange(len(work), dtype=np.int64)
    group = work.groupby(["_game_id", "pitcher_id"], sort=False, observed=True)
    work["_appearance_id"] = group.ngroup().astype(np.int64)
    work["_within_appearance_n"] = group.cumcount().astype(np.int32)
    cumulative_success = group[TARGET].cumsum() - work[TARGET]
    work["_within_appearance_s"] = cumulative_success.astype(np.int32)
    appearances = (
        work.groupby("_appearance_id", sort=False, observed=True)
        .agg(
            first_row=("_row_index", "min"),
            appearance_pitches=("_row_index", "size"),
            appearance_successes=(TARGET, "sum"),
            season=("season", "first"),
            pitcher_id=("pitcher_id", "first"),
            game_type=("game_type", "first"),
        )
        .sort_values("first_row", kind="mergesort")
    )
    appearances["season_appearance_k"] = appearances.groupby(
        ["season", "pitcher_id"], sort=False, observed=True
    ).cumcount()
    for column in ("appearance_pitches", "appearance_successes"):
        destination = "true_prev5_n" if column == "appearance_pitches" else "true_prev5_s"
        appearances[destination] = appearances.groupby(
            "pitcher_id", sort=False, observed=True
        )[column].transform(
            lambda values: values.shift(1).rolling(WINDOW, min_periods=WINDOW).sum()
        )
    mapping = appearances[
        ["season_appearance_k", "true_prev5_n", "true_prev5_s"]
    ]
    for column in mapping.columns:
        work[column] = work["_appearance_id"].map(mapping[column])
    return work, appearances


def frozen_history_arrays(
    history: pd.DataFrame,
    rows: pd.DataFrame,
) -> dict[str, Any]:
    history = history.sort_values("first_row", kind="mergesort")
    pitcher_stats = history.groupby("pitcher_id", sort=False, observed=True).agg(
        workload_mean=("appearance_pitches", "mean"),
        workload_std=("appearance_pitches", "std"),
        workload_count=("appearance_pitches", "size"),
        pitches=("appearance_pitches", "sum"),
        successes=("appearance_successes", "sum"),
    )
    global_mean = float(history["appearance_pitches"].mean())
    global_std = float(history["appearance_pitches"].std(ddof=0))
    global_rate = float(history["appearance_successes"].sum() / history["appearance_pitches"].sum())
    pitcher = rows["pitcher_id"]
    mean = pitcher.map(pitcher_stats["workload_mean"]).fillna(global_mean).to_numpy(float)
    std = pitcher.map(pitcher_stats["workload_std"]).fillna(global_std).to_numpy(float)
    rate_table = pitcher_stats["successes"] / pitcher_stats["pitches"]
    success_rate = pitcher.map(rate_table).fillna(global_rate).to_numpy(float)
    history_count = pitcher.map(pitcher_stats["workload_count"]).fillna(0).to_numpy(int)

    suffix = history.groupby("pitcher_id", sort=False, observed=True).tail(WINDOW)
    suffix_n: dict[int, np.ndarray] = {}
    suffix_s: dict[int, np.ndarray] = {}
    for length in range(1, WINDOW + 1):
        tail = suffix.groupby("pitcher_id", sort=False, observed=True).tail(length)
        sums = tail.groupby("pitcher_id", sort=False, observed=True).agg(
            n=("appearance_pitches", "sum"),
            s=("appearance_successes", "sum"),
            count=("appearance_pitches", "size"),
        )
        valid = sums["count"].eq(length)
        suffix_n[length] = pitcher.map(sums["n"].where(valid)).to_numpy(float)
        suffix_s[length] = pitcher.map(sums["s"].where(valid)).to_numpy(float)
    return {
        "mean": mean,
        "std": std,
        "success_rate": success_rate,
        "history_count": history_count,
        "suffix_n": suffix_n,
        "suffix_s": suffix_s,
        "global_mean": global_mean,
        "global_std": global_std,
        "global_rate": global_rate,
    }


def boundary_posterior(
    e14_n: np.ndarray,
    e14_s: np.ndarray,
    recent_n: np.ndarray,
    recent_s: np.ndarray,
    frozen: dict[str, Any],
) -> dict[str, np.ndarray]:
    count = len(e14_n)
    log_weight = np.full((count, WINDOW), -np.inf, dtype=np.float64)
    candidate_n = np.zeros((count, WINDOW), dtype=np.float64)
    candidate_s = np.zeros((count, WINDOW), dtype=np.float64)
    mean = np.asarray(frozen["mean"], dtype=np.float64)
    std = np.maximum(np.asarray(frozen["std"], dtype=np.float64), 8.0)
    historical_rate = np.clip(
        np.asarray(frozen["success_rate"], dtype=np.float64), 1e-4, 1.0 - 1e-4
    )
    for k in range(WINDOW):
        suffix_length = WINDOW - k
        boundary_n = np.asarray(frozen["suffix_n"][suffix_length], dtype=np.float64)
        boundary_s = np.asarray(frozen["suffix_s"][suffix_length], dtype=np.float64)
        completed_n = recent_n - boundary_n
        completed_s = recent_s - boundary_s
        current_n = e14_n - completed_n
        current_s = e14_s - completed_s
        valid = (
            np.isfinite(boundary_n)
            & np.isfinite(boundary_s)
            & np.isfinite(recent_n)
            & np.isfinite(recent_s)
            & (recent_n > 0.0)
            & (completed_n >= 0.0)
            & (completed_s >= 0.0)
            & (completed_s <= completed_n)
            & (current_n >= 0.0)
            & (current_n <= 140.0)
            & (current_s >= 0.0)
            & (current_s <= current_n)
        )
        if k == 0:
            valid &= (completed_n == 0.0) & (completed_s == 0.0)
            likelihood = np.zeros(count, dtype=np.float64)
        else:
            sum_std = np.maximum(np.sqrt(float(k)) * std, 8.0)
            workload_z = (completed_n - float(k) * mean) / sum_std
            success_var = np.maximum(
                completed_n * historical_rate * (1.0 - historical_rate), 4.0
            )
            success_z = (completed_s - completed_n * historical_rate) / np.sqrt(
                success_var
            )
            likelihood = (
                -0.5 * np.square(workload_z)
                - np.log(sum_std)
                - 0.5 * np.square(success_z)
                - 0.5 * np.log(success_var)
            )
        over_mean = np.maximum((current_n - mean) / std, 0.0)
        likelihood -= 0.5 * np.square(over_mean)
        log_weight[valid, k] = likelihood[valid]
        candidate_n[:, k] = current_n
        candidate_s[:, k] = current_s

    maximum = np.max(log_weight, axis=1)
    active = np.isfinite(maximum)
    weight = np.zeros_like(log_weight)
    weight[active] = np.exp(log_weight[active] - maximum[active, None])
    total = weight.sum(axis=1)
    weight[active] /= total[active, None]
    expected_n = np.sum(weight * candidate_n, axis=1)
    expected_s = np.sum(weight * candidate_s, axis=1)
    mode_k = np.argmax(weight, axis=1).astype(np.int8)
    confidence = np.max(weight, axis=1)
    entropy = -np.sum(
        np.where(weight > 0.0, weight * np.log(np.maximum(weight, 1e-12)), 0.0),
        axis=1,
    )
    expected_n[~active] = 0.0
    expected_s[~active] = 0.0
    mode_k[~active] = -1
    confidence[~active] = 0.0
    entropy[~active] = 0.0
    return {
        "active": active,
        "expected_n": expected_n,
        "expected_s": expected_s,
        "mode_k": mode_k,
        "confidence": confidence,
        "entropy": entropy,
        "weights": weight,
    }


def semantic_metrics(
    valid: pd.DataFrame,
    exact: dict[str, np.ndarray],
) -> dict[str, Any]:
    true_n = valid["true_prev5_n"].to_numpy(dtype=np.float64)
    true_s = valid["true_prev5_s"].to_numpy(dtype=np.float64)
    official = pd.to_numeric(
        valid["asof_pitcher_prev5_game_success_rate"], errors="coerce"
    ).to_numpy(dtype=np.float64)
    comparable = np.isfinite(true_n) & np.isfinite(true_s) & np.isfinite(official)
    exact_rate = np.divide(
        true_s,
        true_n,
        out=np.zeros(len(valid), dtype=np.float64),
        where=true_n > 0.0,
    )
    rate_match = np.abs(official - exact_rate) <= 5.1e-7
    count_match = np.rint(official * true_n) == true_s
    truth_k = valid["season_appearance_k"].to_numpy(dtype=np.int64)
    boundary = comparable & (truth_k < WINDOW)
    recovered = boundary & exact["active"]
    correct = recovered & (exact["mode_k"] == truth_k)
    false_active = exact["active"] & (truth_k >= WINDOW)
    true_current_n = valid["_within_appearance_n"].to_numpy(dtype=np.float64)
    true_current_s = valid["_within_appearance_s"].to_numpy(dtype=np.float64)
    return {
        "comparable_pitch_rows": int(comparable.sum()),
        "official_prev5_rate_match_rate": float(rate_match[comparable].mean()),
        "official_prev5_success_count_match_rate": float(count_match[comparable].mean()),
        "true_boundary_pitch_rows": int(boundary.sum()),
        "true_boundary_pitch_row_rate": float(boundary.mean()),
        "recovered_true_boundary_pitch_rows": int(recovered.sum()),
        "recovered_true_boundary_pitch_row_rate": float(recovered.mean()),
        "mode_k_accuracy_on_recovered_true_boundary": float(
            correct.sum() / max(recovered.sum(), 1)
        ),
        "false_active_nonboundary_pitch_rows": int(false_active.sum()),
        "false_active_rate_among_nonboundary": float(
            false_active.sum() / max((truth_k >= WINDOW).sum(), 1)
        ),
        "current_n_mae_on_recovered_true_boundary": float(
            np.mean(np.abs(exact["expected_n"][recovered] - true_current_n[recovered]))
        ) if recovered.any() else None,
        "current_s_mae_on_recovered_true_boundary": float(
            np.mean(np.abs(exact["expected_s"][recovered] - true_current_s[recovered]))
        ) if recovered.any() else None,
    }


def direction_from_state(
    current_n: np.ndarray,
    current_s: np.ndarray,
    active: np.ndarray,
    e14_n: np.ndarray,
    e14_s: np.ndarray,
    prior: float,
    k: float,
) -> np.ndarray:
    current_rate = (current_s + k * prior) / (current_n + k)
    season_rate = (e14_s + k * prior) / (e14_n + k)
    return np.where(active, current_rate - season_rate, 0.0)


def grid_candidates(
    folds: dict[int, dict[str, Any]],
    state_name: str,
    prereg: dict[str, Any],
) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    for k_value in prereg["source_recipe_grid"]["posterior_k"]:
        k = float(k_value)
        for gamma_value in prereg["source_recipe_grid"]["gamma"]:
            gamma = float(gamma_value)
            years: dict[str, Any] = {}
            for year in YEARS:
                fold = folds[year]
                state = fold[state_name]
                direction = direction_from_state(
                    state["n"],
                    state["s"],
                    state["active"],
                    fold["e14_n"],
                    fold["e14_s"],
                    fold["prior"],
                    k,
                )
                candidate = np.clip(
                    fold["base"] + gamma * direction, 1e-6, 1.0 - 1e-6
                )
                years[str(year)] = evaluate(
                    fold["y"], fold["base"], candidate, fold["game_type"]
                )
            candidates.append(
                {
                    "posterior_k": int(k_value),
                    "gamma": gamma,
                    "min_full_gain": float(
                        min(years[str(year)]["gains"]["all"] for year in YEARS)
                    ),
                    "min_R_gain": float(
                        min(years[str(year)]["gains"]["R"] for year in YEARS)
                    ),
                    "min_F_gain": float(
                        min(years[str(year)]["gains"]["F"] for year in YEARS)
                    ),
                    "mean_full_gain": float(
                        np.mean([years[str(year)]["gains"]["all"] for year in YEARS])
                    ),
                    "years": years,
                }
            )
    candidates.sort(
        key=lambda row: (
            row["min_full_gain"],
            row["min_R_gain"],
            row["mean_full_gain"],
            -row["gamma"],
            row["posterior_k"],
        ),
        reverse=True,
    )
    return candidates


def intervals_for_selected(
    folds: dict[int, dict[str, Any]],
    state_name: str,
    selected: dict[str, Any],
    seed_base: int,
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for offset, year in enumerate(YEARS):
        fold = folds[year]
        state = fold[state_name]
        direction = direction_from_state(
            state["n"],
            state["s"],
            state["active"],
            fold["e14_n"],
            fold["e14_s"],
            fold["prior"],
            float(selected["posterior_k"]),
        )
        candidate = np.clip(
            fold["base"] + float(selected["gamma"]) * direction,
            1e-6,
            1.0 - 1e-6,
        )
        masks = {
            "all": np.ones(len(candidate), dtype=bool),
            "R": fold["game_type"] == "R",
            "F": fold["game_type"] == "F",
        }
        result[str(year)] = {
            scope: cluster_bootstrap_score_gain(
                fold["y"],
                fold["base"],
                candidate,
                fold["cluster"],
                mask,
                iterations=2000,
                seed=seed_base + offset * 10 + index,
            )
            for index, (scope, mask) in enumerate(masks.items())
        }
    return result


def decoder_for_year(
    frame: pd.DataFrame,
    appearances: pd.DataFrame,
    year: int,
    params: dict[str, Any],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    history_frame = frame.loc[frame["season"].lt(year)].reset_index(drop=True)
    decoder_appearances, representatives = reconstruct_appearances(
        history_frame, build_recent_denominator_features
    )
    models, metadata = _fit_models(
        decoder_appearances,
        representatives,
        params,
        seed_base=int(params["random_seed"]) + year * 10,
    )
    valid_appearances = appearances.loc[appearances["season"].eq(year)]
    representative_rows = frame.iloc[
        valid_appearances["first_row"].to_numpy(dtype=np.int64)
    ].copy()
    decoded_representatives = _predict_rows(
        representative_rows,
        decoder_appearances,
        models,
        params,
        build_recent_denominator_features,
    )
    decoded_representatives.index = valid_appearances.index
    del models, decoder_appearances, representatives, representative_rows
    gc.collect()
    return decoded_representatives, metadata


def invariance_check(
    e14_n: np.ndarray,
    e14_s: np.ndarray,
    recent_n: np.ndarray,
    recent_s: np.ndarray,
    frozen: dict[str, Any],
) -> dict[str, Any]:
    size = min(64, len(e14_n))
    index = np.arange(size)

    def subset(source: dict[str, Any], selected: np.ndarray) -> dict[str, Any]:
        return {
            "mean": source["mean"][selected],
            "std": source["std"][selected],
            "success_rate": source["success_rate"][selected],
            "history_count": source["history_count"][selected],
            "suffix_n": {key: value[selected] for key, value in source["suffix_n"].items()},
            "suffix_s": {key: value[selected] for key, value in source["suffix_s"].items()},
        }

    reference = boundary_posterior(
        e14_n[index], e14_s[index], recent_n[index], recent_s[index], subset(frozen, index)
    )
    reverse = index[::-1]
    reversed_result = boundary_posterior(
        e14_n[reverse],
        e14_s[reverse],
        recent_n[reverse],
        recent_s[reverse],
        subset(frozen, reverse),
    )
    max_difference = 0.0
    for column in ("expected_n", "expected_s", "confidence", "entropy"):
        max_difference = max(
            max_difference,
            float(np.max(np.abs(reference[column] - reversed_result[column][::-1]))),
        )
    duplicated_index = np.concatenate([index, index[:1]])
    duplicate = boundary_posterior(
        e14_n[duplicated_index],
        e14_s[duplicated_index],
        recent_n[duplicated_index],
        recent_s[duplicated_index],
        subset(frozen, duplicated_index),
    )
    for column in ("expected_n", "expected_s", "confidence", "entropy"):
        max_difference = max(
            max_difference,
            float(np.max(np.abs(reference[column] - duplicate[column][:size]))),
            float(abs(reference[column][0] - duplicate[column][-1])),
        )
    return {
        "sample_rows": int(size),
        "reverse_duplicate_max_abs_difference": max_difference,
        "passed": bool(max_difference == 0.0),
    }


def main() -> None:
    if REPORT.exists():
        raise FileExistsError(f"immutable report exists: {REPORT}")
    prereg = json.loads(PREREG.read_text(encoding="utf-8"))
    if prereg["status"] != "locked_before_source_semantic_or_control_metrics":
        raise ValueError("unexpected preregistration state")
    decoder_prereg = json.loads(DECODER_PREREG.read_text(encoding="utf-8"))
    started = time.perf_counter()
    frame = load_source()
    frame, appearances = build_appearance_panel(frame)
    states_before, _ = season_end_state(frame)
    priors = prior_before_each_season(frame)

    folds: dict[int, dict[str, Any]] = {}
    semantic: dict[str, Any] = {}
    semantic_checks: list[bool] = []
    for year in YEARS:
        anchor = load_anchor(year)
        valid = frame.iloc[anchor["row_index"]].copy()
        if not valid["season"].eq(year).all():
            raise ValueError(f"{year}: anchor season mismatch")
        if not np.array_equal(valid[TARGET].to_numpy(dtype=np.int8), anchor["y"].astype(np.int8)):
            raise ValueError(f"{year}: target mismatch")
        prior = float(priors[year])
        e14, e14_meta = build_e14_features(
            valid, states_before, priors, prior, k=120.0
        )
        history_appearances = appearances.loc[appearances["season"].lt(year)].copy()
        frozen = frozen_history_arrays(history_appearances, valid)
        exact = boundary_posterior(
            e14["e14_n_season"].to_numpy(dtype=np.float64),
            e14["e14_s_season"].to_numpy(dtype=np.float64),
            valid["true_prev5_n"].to_numpy(dtype=np.float64),
            valid["true_prev5_s"].to_numpy(dtype=np.float64),
            frozen,
        )
        fold_semantic = semantic_metrics(valid, exact)
        requirements = prereg["semantic_audit"]
        checks = {
            "official_rate": fold_semantic["official_prev5_rate_match_rate"]
            >= float(requirements["minimum_official_prev5_denominator_match_rate"]),
            "official_success_count": fold_semantic[
                "official_prev5_success_count_match_rate"
            ] >= float(requirements["minimum_official_prev5_success_count_match_rate"]),
            "exact_k_accuracy": fold_semantic[
                "mode_k_accuracy_on_recovered_true_boundary"
            ] >= float(requirements["minimum_exact_denominator_true_boundary_k_accuracy"]),
            "boundary_coverage": fold_semantic[
                "recovered_true_boundary_pitch_row_rate"
            ] >= float(requirements["minimum_exact_denominator_covered_pitch_row_rate"]),
        }
        semantic_checks.extend(checks.values())
        semantic[str(year)] = {
            **fold_semantic,
            "checks": checks,
            "e14": e14_meta,
            "frozen_history_global_workload_mean": frozen["global_mean"],
            "frozen_history_global_workload_std": frozen["global_std"],
        }
        truth_k = valid["season_appearance_k"].to_numpy(dtype=np.int64)
        oracle_active = (truth_k < WINDOW) & (frozen["history_count"] >= WINDOW)
        folds[year] = {
            "y": anchor["y"].astype(np.int8),
            "base": anchor["catboost_outcome"].astype(np.float64),
            "cluster": anchor["cluster"],
            "game_type": valid["game_type"].astype(str).to_numpy(),
            "valid": valid,
            "e14_n": e14["e14_n_season"].to_numpy(dtype=np.float64),
            "e14_s": e14["e14_s_season"].to_numpy(dtype=np.float64),
            "prior": prior,
            "frozen": frozen,
            "exact": exact,
            "oracle": {
                "n": valid["_within_appearance_n"].to_numpy(dtype=np.float64),
                "s": valid["_within_appearance_s"].to_numpy(dtype=np.float64),
                "active": oracle_active,
            },
        }

    oracle_candidates = grid_candidates(folds, "oracle", prereg)
    oracle_selected = oracle_candidates[0]
    oracle_intervals = intervals_for_selected(
        folds, "oracle", oracle_selected, seed_base=61200
    )
    headroom = prereg["headroom_gate"]
    oracle_conditions = {
        "semantic_gate": bool(all(semantic_checks)),
        "minimum_full_gain_each_year": bool(
            oracle_selected["min_full_gain"]
            >= float(headroom["oracle_minimum_full_gain_each_source_year"])
        ),
        "full_cluster_ci_lower_positive_each_year": bool(
            all(oracle_intervals[str(year)]["all"]["ci_low"] > 0.0 for year in YEARS)
        ),
    }
    oracle_pass = bool(all(oracle_conditions.values()))

    if not oracle_pass:
        report = {
            "experiment_id": prereg["experiment_id"],
            "status": "source_oracle_failed",
            "preregister": str(PREREG.relative_to(ROOT)),
            "preregister_sha256": sha256(PREREG),
            "script_sha256": sha256(Path(__file__)),
            "train_sha256": sha256(TRAIN),
            "years_read": list(YEARS),
            "years_not_read": [2022, 2023, 2024],
            "semantic": semantic,
            "oracle": {
                "role": "nondeployable_headroom_only",
                "same_appearance_prior_labels_used": True,
                "selected": oracle_selected,
                "intervals": oracle_intervals,
                "conditions": oracle_conditions,
                "gate_pass": False,
                "prediction_artifact_saved": False,
            },
            "legal_decoder_or_control_metrics_computed": False,
            "decision": "close boundary-appearance axis without reading 2022+ labels",
            "policy": prereg["data_policy"],
            "elapsed_seconds": float(time.perf_counter() - started),
            "goal_completion_claimed": False,
        }
        REPORT.write_text(
            json.dumps(safe(report), indent=2, ensure_ascii=False), encoding="utf-8"
        )
        print(json.dumps(safe(report), indent=2, ensure_ascii=False))
        return

    decoder_metadata: dict[str, Any] = {}
    invariance: dict[str, Any] = {}
    legal_semantic: dict[str, Any] = {}
    for year in YEARS:
        fold = folds[year]
        decoded_by_appearance, metadata = decoder_for_year(
            frame, appearances, year, decoder_prereg["model"]
        )
        valid = fold["valid"]
        appearance_ids = valid["_appearance_id"].to_numpy(dtype=np.int64)
        recent_n = decoded_by_appearance["e101_prev5_n_mode"].reindex(
            appearance_ids
        ).to_numpy(dtype=np.float64)
        recent_s = decoded_by_appearance["e101_prev5_success_count_mode"].reindex(
            appearance_ids
        ).to_numpy(dtype=np.float64)
        legal = boundary_posterior(
            fold["e14_n"], fold["e14_s"], recent_n, recent_s, fold["frozen"]
        )
        fold["legal"] = {
            "n": legal["expected_n"],
            "s": legal["expected_s"],
            "active": legal["active"],
        }
        decoder_metadata[str(year)] = metadata
        invariance[str(year)] = invariance_check(
            fold["e14_n"], fold["e14_s"], recent_n, recent_s, fold["frozen"]
        )
        legal_semantic[str(year)] = semantic_metrics(valid, legal)

    legal_candidates = grid_candidates(folds, "legal", prereg)
    legal_selected = legal_candidates[0]
    legal_intervals = intervals_for_selected(
        folds, "legal", legal_selected, seed_base=61300
    )
    gate = prereg["legal_source_gate"]
    legal_conditions = {
        "minimum_full_gain_each_year": bool(
            legal_selected["min_full_gain"]
            >= float(gate["minimum_full_gain_each_source_year"])
        ),
        "minimum_R_gain_each_year": bool(
            legal_selected["min_R_gain"] >= float(gate["minimum_R_gain_each_source_year"])
        ),
        "minimum_F_gain_each_year": bool(
            legal_selected["min_F_gain"] >= float(gate["minimum_F_gain_each_source_year"])
        ),
        "full_cluster_ci_lower_positive_each_year": bool(
            all(legal_intervals[str(year)]["all"]["ci_low"] > 0.0 for year in YEARS)
        ),
        "row_invariance": bool(all(value["passed"] for value in invariance.values())),
    }
    legal_pass = bool(all(legal_conditions.values()))
    report = {
        "experiment_id": prereg["experiment_id"],
        "status": "source_pass" if legal_pass else "legal_source_failed",
        "preregister": str(PREREG.relative_to(ROOT)),
        "preregister_sha256": sha256(PREREG),
        "script_sha256": sha256(Path(__file__)),
        "train_sha256": sha256(TRAIN),
        "years_read": list(YEARS),
        "years_not_read": [2022, 2023, 2024],
        "semantic": semantic,
        "oracle": {
            "role": "nondeployable_headroom_only",
            "same_appearance_prior_labels_used": True,
            "selected": oracle_selected,
            "intervals": oracle_intervals,
            "conditions": oracle_conditions,
            "gate_pass": True,
            "prediction_artifact_saved": False,
        },
        "legal": {
            "selected": legal_selected,
            "intervals": legal_intervals,
            "conditions": legal_conditions,
            "gate_pass": legal_pass,
            "semantic": legal_semantic,
            "decoder_metadata": decoder_metadata,
            "row_invariance": invariance,
            "prediction_artifact_saved": False,
        },
        "decision": (
            "freeze legal direct recipe before 2022/2023 development"
            if legal_pass
            else "close direct recipe; learned feature follow-up allowed by passed oracle"
        ),
        "policy": prereg["data_policy"],
        "elapsed_seconds": float(time.perf_counter() - started),
        "goal_completion_claimed": False,
    }
    REPORT.write_text(
        json.dumps(safe(report), indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(safe(report), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
