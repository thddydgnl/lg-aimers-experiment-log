#!/usr/bin/env python3
"""Pitcher-adaptive state-space approximation from official as-of counters."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

TARGET = "control_success"
SEASON = "season"
PITCHER = "pitcher_id"


def _season_table(history: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    regular = history.loc[history["game_type"].eq("R")].copy()
    league = regular.groupby(SEASON, observed=True)[TARGET].mean().sort_index()
    table = (
        regular.groupby([PITCHER, SEASON], observed=True)[TARGET]
        .agg(["sum", "count"])
        .reset_index()
        .sort_values([PITCHER, SEASON], kind="stable")
    )
    table["rate"] = table["sum"] / table["count"]
    table["league"] = table[SEASON].map(league)
    table["effect"] = table["rate"] - table["league"]
    table["sampling_var"] = (
        table["rate"] * (1.0 - table["rate"]) / table["count"].clip(lower=1)
    )
    grouped = table.groupby(PITCHER, sort=False, observed=True)
    table["previous_effect"] = grouped["effect"].shift(1)
    table["previous_sampling_var"] = grouped["sampling_var"].shift(1)
    table["previous_count"] = grouped["count"].shift(1)
    table["process_observation"] = np.maximum(
        np.square(table["effect"] - table["previous_effect"])
        - table["sampling_var"]
        - table["previous_sampling_var"],
        0.0,
    )
    return table, league


def _process_variances(
    table: pd.DataFrame,
    global_minimum_rows: int = 100,
    individual_minimum_rows: int = 30,
    prior_dof: float = 3.0,
    winsor_quantile: float = 0.90,
    upper_multiple: float = 10.0,
) -> tuple[pd.Series, dict[str, Any]]:
    valid_pair = table["previous_effect"].notna()
    global_mask = (
        valid_pair
        & table["count"].ge(global_minimum_rows)
        & table["previous_count"].ge(global_minimum_rows)
    )
    global_values = table.loc[global_mask, "process_observation"].to_numpy(
        dtype=np.float64
    )
    if len(global_values):
        cap = float(np.quantile(global_values, winsor_quantile))
        global_process = float(np.mean(np.minimum(global_values, cap)))
    else:
        cap = 0.0
        global_process = 0.0
    global_process = max(global_process, 1e-6)

    individual_mask = (
        valid_pair
        & table["count"].ge(individual_minimum_rows)
        & table["previous_count"].ge(individual_minimum_rows)
    )
    individual = table.loc[individual_mask].groupby(
        PITCHER, observed=True
    )["process_observation"].agg(["sum", "count"])
    estimate = (
        individual["sum"] + prior_dof * global_process
    ) / (individual["count"] + prior_dof)
    estimate = estimate.clip(
        lower=global_process / upper_multiple,
        upper=global_process * upper_multiple,
    )
    return estimate, {
        "global_pair_count": int(global_mask.sum()),
        "individual_pair_count": int(individual_mask.sum()),
        "pitchers_with_individual_process": int(len(estimate)),
        "global_process_variance": global_process,
        "winsor_cap": cap,
        "process_q10": float(estimate.quantile(0.10)) if len(estimate) else None,
        "process_median": float(estimate.median()) if len(estimate) else None,
        "process_q90": float(estimate.quantile(0.90)) if len(estimate) else None,
    }


def _lifetime_end_state(history: pd.DataFrame) -> pd.DataFrame:
    work = history[
        [PITCHER, "asof_pitcher_n", "asof_pitcher_success_rate", TARGET]
    ].copy()
    work["asof_pitcher_n"] = pd.to_numeric(
        work["asof_pitcher_n"], errors="coerce"
    ).fillna(0).astype(np.int64)
    work["success_before"] = np.rint(
        pd.to_numeric(
            work["asof_pitcher_success_rate"], errors="coerce"
        ).fillna(0.0).to_numpy(dtype=np.float64)
        * work["asof_pitcher_n"].to_numpy(dtype=np.float64)
    ).astype(np.int64)
    # Official rows are chronological; n is also monotone within pitcher.  The
    # stable last maximum resolves the final completed historical pitch.
    work["_order"] = np.arange(len(work), dtype=np.int64)
    last = work.sort_values(
        [PITCHER, "asof_pitcher_n", "_order"], kind="stable"
    ).groupby(PITCHER, observed=True).tail(1).set_index(PITCHER)
    return pd.DataFrame({
        "end_n": last["asof_pitcher_n"] + 1,
        "end_success": last["success_before"] + last[TARGET].astype(np.int64),
    })


def build_adaptive_state_probability(
    frame: pd.DataFrame,
    target_year: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Return a row-independent state posterior for rows in ``target_year``."""
    history = frame.loc[frame[SEASON].lt(target_year)].copy()
    valid = frame.loc[frame[SEASON].eq(target_year)].copy()
    if history.empty or valid.empty:
        raise ValueError(f"empty history or validation for {target_year}")
    table, league = _season_table(history)
    process, process_meta = _process_variances(table)
    latest_year = int(league.index.max())
    latest_global = float(league.loc[latest_year])
    latest = (
        table.sort_values([PITCHER, SEASON], kind="stable")
        .groupby(PITCHER, observed=True)
        .tail(1)
        .set_index(PITCHER)
    )
    boundary = _lifetime_end_state(history)

    pitcher = valid[PITCHER]
    end_n = pitcher.map(boundary["end_n"]).fillna(0).to_numpy(dtype=np.int64)
    end_success = pitcher.map(boundary["end_success"]).fillna(0).to_numpy(
        dtype=np.int64
    )
    n_asof = pd.to_numeric(
        valid["asof_pitcher_n"], errors="coerce"
    ).fillna(0).to_numpy(dtype=np.int64)
    successes_asof = np.rint(
        pd.to_numeric(
            valid["asof_pitcher_success_rate"], errors="coerce"
        ).fillna(0.0).to_numpy(dtype=np.float64)
        * n_asof
    ).astype(np.int64)
    current_n = n_asof - end_n
    current_success = successes_asof - end_success
    invalid = (
        (current_n < 0)
        | (current_success < 0)
        | (current_success > current_n)
    )
    current_n = np.where(invalid, 0, current_n)
    current_success = np.where(invalid, 0, current_success)

    latest_effect = pitcher.map(latest["effect"]).fillna(0.0).to_numpy(
        dtype=np.float64
    )
    latest_count = pitcher.map(latest["count"]).fillna(0.0).to_numpy(
        dtype=np.float64
    )
    latest_sampling = pitcher.map(latest["sampling_var"]).fillna(
        latest_global * (1.0 - latest_global) / 120.0
    ).to_numpy(dtype=np.float64)
    effect_weight = latest_count / (latest_count + 120.0)
    prior = np.clip(
        latest_global + effect_weight * latest_effect,
        0.05,
        0.95,
    )
    process_value = pitcher.map(process).fillna(
        process_meta["global_process_variance"]
    ).to_numpy(dtype=np.float64)
    adaptive_k = latest_global * (1.0 - latest_global) / np.maximum(
        latest_sampling + process_value, 1e-9
    )
    adaptive_k = np.clip(adaptive_k, 20.0, 500.0)
    cold = latest_count <= 0
    adaptive_k[cold] = 120.0
    prior[cold] = latest_global
    posterior = (
        current_success + adaptive_k * prior
    ) / (current_n + adaptive_k)
    posterior = np.clip(posterior, 1e-6, 1.0 - 1e-6)
    metadata = {
        "target_year": int(target_year),
        "history_seasons": sorted(int(value) for value in history[SEASON].unique()),
        "latest_completed_R_year": latest_year,
        "latest_completed_R_mean": latest_global,
        "history_rows": int(len(history)),
        "valid_rows": int(len(valid)),
        "invalid_counter_rows": int(invalid.sum()),
        "cold_start_rows": int(cold.sum()),
        "current_n_positive_fraction": float(np.mean(current_n > 0)),
        "current_n_median": float(np.median(current_n)),
        "adaptive_k_q10": float(np.quantile(adaptive_k, 0.10)),
        "adaptive_k_median": float(np.median(adaptive_k)),
        "adaptive_k_q90": float(np.quantile(adaptive_k, 0.90)),
        "prior_mean": float(np.mean(prior)),
        "posterior_mean": float(np.mean(posterior)),
        "posterior_std": float(np.std(posterior)),
        "process": process_meta,
        "validation_target_used": False,
        "other_validation_rows_used": False,
        "row_independent": True,
    }
    return posterior, metadata
