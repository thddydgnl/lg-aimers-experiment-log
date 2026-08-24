#!/usr/bin/env python3
"""Strict next-season conditional residual Ridge experiments on the V3 M3 OOF.

The correction is learned from one completed season and transferred unchanged to
the next season.  Conditional tables use official train labels only, while each
target row is looked up independently.  No test rows or leaderboard values are
read by this experiment.
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.finalize_v3_sparse import (  # noqa: E402
    COMPONENTS,
    PREDICTIONS,
    calibrate,
    load_npz,
)


TARGET = "control_success"
ANCHOR_TEAM_ID = 13
OUTPUT_JSON = ROOT / "experiments/results/v4_temporal_residual_ridge.json"
OUTPUT_NPZ = (
    ROOT / "experiments/results/predictions/v4_temporal_residual_ridge_2024.npz"
)
MEDIAN_OFFSET = 140.1475834416
REQUIRED_LOCAL = 1190.0 - MEDIAN_OFFSET
M3_WEIGHTS = {
    "A": 0.501443851662535,
    "C": 0.27016033407769313,
    "B": 0.22839581425977187,
}
M3_EARLY_SOURCES = {
    2020: {
        "A": ("v4_m3_a_backtest_2020", "catboost_outcome"),
        "B": ("v4_m3_b_backtest_2020", "catboost_outcome"),
        "C": ("v4_m3_c_backtest_2020", "catboost_outcome"),
    },
    2021: {
        "A": ("v4_m3_a_backtest_2021", "catboost_outcome"),
        "B": ("v4_m3_b_backtest_2021", "catboost_outcome"),
        "C": ("v4_m3_c_backtest_2021", "catboost_outcome"),
    },
}


@dataclass(frozen=True)
class Config:
    scope: str
    k_pitcher: float
    k_hand: float
    k_pressure: float
    alpha: float
    gamma: float
    training_mode: str = "loo"


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        number = float(value)
        return number if np.isfinite(number) else None
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value


def score(y: np.ndarray, prediction: np.ndarray) -> dict[str, float | int]:
    y64 = np.asarray(y, dtype=np.float64)
    p64 = np.asarray(prediction, dtype=np.float64)
    brier = float(np.mean(np.square(p64 - y64)))
    rate = float(y64.mean())
    reference = rate * (1.0 - rate)
    raw = 100_000.0 * (1.0 - brier / reference)
    return {
        "rows": int(len(y64)),
        "target_rate": rate,
        "prediction_mean": float(p64.mean()),
        "prediction_std": float(p64.std()),
        "brier": brier,
        "competition_score": max(0.0, raw),
        "raw_competition_score": raw,
    }


def m3_for_season(season: int) -> dict[str, np.ndarray]:
    reference: dict[str, np.ndarray] | None = None
    predictions: dict[str, np.ndarray] = {}
    for key, sources in COMPONENTS.items():
        stage, column = (
            M3_EARLY_SOURCES[season][key]
            if season in M3_EARLY_SOURCES
            else sources[season]
        )
        artifact = load_npz(stage, season)
        if reference is None:
            reference = artifact
        else:
            for align_key in ("y", "row_index", "cluster"):
                if not np.array_equal(reference[align_key], artifact[align_key]):
                    raise ValueError(f"M3 alignment mismatch for {season}/{align_key}")
        predictions[key] = np.asarray(artifact[column], dtype=np.float64)
    if reference is None:
        raise RuntimeError(f"No M3 predictions found for {season}")
    raw = sum(M3_WEIGHTS[key] * predictions[key] for key in M3_WEIGHTS)
    return {
        **reference,
        **{f"component_{key}": value for key, value in predictions.items()},
        "m3_raw": raw,
        "m3": calibrate(raw),
    }


def add_context(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    balls = result["balls_before"].to_numpy(dtype=np.int8, copy=False)
    strikes = result["strikes_before"].to_numpy(dtype=np.int8, copy=False)
    pressure = np.zeros(len(result), dtype=np.int8)
    pressure[(balls == 3) & (strikes < 2)] = 1
    pressure[(balls < 3) & (strikes == 2)] = 2
    pressure[(balls == 3) & (strikes == 2)] = 3
    result["pressure_state"] = pressure
    anchor = result["pitcher_team_id"].eq(ANCHOR_TEAM_ID) | result[
        "batter_team_id"
    ].eq(ANCHOR_TEAM_ID)
    result["domain"] = np.where(
        result["game_type"].eq("F"), "F", np.where(anchor, "R_ANCHOR", "R_CORE")
    )
    return result


def aggregate(frame: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    table = frame.groupby(keys, sort=False, observed=True)[TARGET].agg(["sum", "size"])
    table["sum"] = table["sum"].astype(np.float64)
    table["size"] = table["size"].astype(np.float64)
    return table


def lookup(
    table: pd.DataFrame,
    rows: pd.DataFrame,
    keys: list[str],
) -> tuple[np.ndarray, np.ndarray]:
    if len(keys) == 1:
        index = pd.Index(rows[keys[0]].to_numpy(), name=keys[0])
    else:
        index = pd.MultiIndex.from_frame(rows[keys])
    found = table.reindex(index)
    return (
        found["sum"].fillna(0.0).to_numpy(dtype=np.float64),
        found["size"].fillna(0.0).to_numpy(dtype=np.float64),
    )


def posterior_block(
    rate: np.ndarray,
    parent: np.ndarray,
    count: np.ndarray,
    prefix: str,
    strength: float,
) -> dict[str, np.ndarray]:
    reliability = np.clip(
        count / np.maximum(count + strength, strength), 0.0, 1.0
    )
    delta = rate - parent
    return {
        f"{prefix}_rate": rate,
        f"{prefix}_delta": delta,
        f"{prefix}_reliability": reliability,
        f"{prefix}_log_n": np.log1p(np.maximum(count, 0.0)),
        f"{prefix}_trusted_delta": reliability * delta,
        f"{prefix}_post_sd": np.sqrt(
            np.clip(
                rate * (1.0 - rate)
                / np.maximum(count + strength + 1.0, strength + 1.0),
                0.0,
                None,
            )
        ),
    }


def stable_matrix(
    table_rows: pd.DataFrame,
    rows: pd.DataFrame,
    config: Config,
    *,
    leave_one_out: bool,
) -> tuple[np.ndarray, list[str]]:
    """Build 25 stable numeric features from frozen source-season tables."""
    if config.scope == "r_core":
        table_rows = table_rows.loc[table_rows["domain"].eq("R_CORE")]
    elif config.scope == "r_all":
        table_rows = table_rows.loc[table_rows["game_type"].eq("R")]
    else:
        raise ValueError(f"Unknown table scope: {config.scope}")
    if table_rows.empty:
        raise ValueError("Conditional source table is empty")

    league = float(table_rows[TARGET].mean())
    y = rows[TARGET].to_numpy(dtype=np.float64, copy=False) if leave_one_out else None

    pitcher_table = aggregate(table_rows, ["pitcher_id"])
    pitcher_sum, pitcher_n = lookup(pitcher_table, rows, ["pitcher_id"])
    if leave_one_out:
        pitcher_sum = pitcher_sum - y
        pitcher_n = np.maximum(0.0, pitcher_n - 1.0)
    pitcher_rate = (pitcher_sum + config.k_pitcher * league) / (
        pitcher_n + config.k_pitcher
    )

    hand_table = aggregate(table_rows, ["pitcher_id", "batter_hand"])
    hand_sum, hand_n = lookup(hand_table, rows, ["pitcher_id", "batter_hand"])
    if leave_one_out:
        hand_sum = hand_sum - y
        hand_n = np.maximum(0.0, hand_n - 1.0)
    hand_rate = (hand_sum + config.k_hand * pitcher_rate) / (
        hand_n + config.k_hand
    )

    pressure_keys = ["pitcher_id", "pressure_state", "batter_hand"]
    pressure_table = aggregate(table_rows, pressure_keys)
    pressure_sum, pressure_n = lookup(pressure_table, rows, pressure_keys)
    if leave_one_out:
        pressure_sum = pressure_sum - y
        pressure_n = np.maximum(0.0, pressure_n - 1.0)
    pressure_rate = (pressure_sum + config.k_pressure * hand_rate) / (
        pressure_n + config.k_pressure
    )

    balls = rows["balls_before"].to_numpy(dtype=np.float64, copy=False)
    strikes = rows["strikes_before"].to_numpy(dtype=np.float64, copy=False)
    pitcher_hand = rows["pitcher_hand"].to_numpy(dtype=np.float64, copy=False)
    batter_hand = rows["batter_hand"].to_numpy(dtype=np.float64, copy=False)
    context: dict[str, np.ndarray] = {
        "ctx_balls": balls / 3.0,
        "ctx_strikes": strikes / 2.0,
        "ctx_ball_advantage": (balls - strikes) / 3.0,
        "ctx_full_count": ((balls == 3) & (strikes == 2)).astype(np.float64),
        "ctx_three_ball": (balls == 3).astype(np.float64),
        "ctx_two_strike": (strikes == 2).astype(np.float64),
        "ctx_same_hand": (pitcher_hand == batter_hand).astype(np.float64),
    }
    pitcher_parent = np.full(len(rows), league, dtype=np.float64)
    features = {
        **context,
        **posterior_block(
            pitcher_rate, pitcher_parent, pitcher_n, "pitcher", config.k_pitcher
        ),
        **posterior_block(hand_rate, pitcher_rate, hand_n, "hand", config.k_hand),
        **posterior_block(
            pressure_rate,
            hand_rate,
            pressure_n,
            "pressure_hand",
            config.k_pressure,
        ),
    }
    names = list(features)
    matrix = np.column_stack([features[name] for name in names]).astype(np.float64)
    if matrix.shape[1] != 25:
        raise AssertionError(f"Stable matrix must contain 25 columns, got {matrix.shape}")
    if not np.isfinite(matrix).all():
        raise ValueError("Non-finite value in stable conditional matrix")
    return matrix, names


def transfer(
    source: pd.DataFrame,
    target: pd.DataFrame,
    source_prediction: np.ndarray,
    target_prediction: np.ndarray,
    config: Config,
) -> tuple[np.ndarray, dict[str, Any]]:
    data = transfer_data(source, target, source_prediction, config)
    model = make_pipeline(
        StandardScaler(), Ridge(alpha=config.alpha, fit_intercept=True)
    )
    model.fit(data["x_source"], data["residual"])
    correction = model.predict(data["x_target"])
    result = np.asarray(target_prediction, dtype=np.float64).copy()
    result[data["target_core"]] = np.clip(
        result[data["target_core"]] + config.gamma * correction, 0.0, 1.0
    )
    return result, correction_diagnostics(data, correction)


def transfer_data(
    source: pd.DataFrame,
    target: pd.DataFrame,
    source_prediction: np.ndarray,
    config: Config,
) -> dict[str, Any]:
    """Build matrices once so alpha/gamma grids do not repeat group aggregation."""
    source_core = source["domain"].eq("R_CORE").to_numpy()
    target_core = target["domain"].eq("R_CORE").to_numpy()
    x_source, names = stable_matrix(
        source,
        source.loc[source_core],
        config,
        leave_one_out=config.training_mode == "loo",
    )
    x_target, target_names = stable_matrix(
        source, target.loc[target_core], config, leave_one_out=False
    )
    if names != target_names:
        raise AssertionError("Source/target stable feature order differs")
    residual = (
        source.loc[source_core, TARGET].to_numpy(dtype=np.float64)
        - source_prediction[source_core]
    )
    return {
        "source_core": source_core,
        "target_core": target_core,
        "x_source": x_source,
        "x_target": x_target,
        "residual": residual,
        "feature_names": names,
    }


def correction_diagnostics(
    data: dict[str, Any], correction: np.ndarray
) -> dict[str, Any]:
    return {
        "source_core_rows": int(data["source_core"].sum()),
        "target_core_rows": int(data["target_core"].sum()),
        "feature_count": int(data["x_source"].shape[1]),
        "feature_names": data["feature_names"],
        "correction_mean": float(correction.mean()),
        "correction_std": float(correction.std()),
        "correction_max_abs": float(np.max(np.abs(correction))),
    }


def load_frames() -> tuple[dict[int, pd.DataFrame], dict[int, dict[str, np.ndarray]]]:
    artifacts = {
        season: m3_for_season(season)
        for season in (2020, 2021, 2022, 2023, 2024)
    }
    columns = [
        "season",
        "game_month",
        "game_type",
        "balls_before",
        "strikes_before",
        "pitcher_id",
        "batter_id",
        "pitcher_hand",
        "batter_hand",
        "pitcher_team_id",
        "batter_team_id",
        TARGET,
    ]
    full = pd.read_csv(
        ROOT / "open/data/train.csv",
        usecols=columns,
        encoding="utf-8-sig",
        low_memory=False,
    )
    frames: dict[int, pd.DataFrame] = {}
    for season, artifact in artifacts.items():
        row_index = np.asarray(artifact["row_index"], dtype=np.int64)
        frame = add_context(full.iloc[row_index].reset_index(drop=True))
        if not np.all(frame["season"].to_numpy() == season):
            raise ValueError(f"Row alignment season mismatch for {season}")
        if not np.array_equal(
            frame[TARGET].to_numpy(dtype=np.int8), artifact["y"].astype(np.int8)
        ):
            raise ValueError(f"Target alignment mismatch for {season}")
        frames[season] = frame
    return frames, artifacts


def main() -> None:
    frames, artifacts = load_frames()
    k_configs = [
        (80.0, 110.0, 220.0),
        (100.0, 100.0, 200.0),
        (100.0, 150.0, 300.0),
        (200.0, 200.0, 400.0),
        (400.0, 400.0, 800.0),
        (800.0, 800.0, 1600.0),
    ]
    alphas = [1.0, 10.0, 100.0, 300.0, 1000.0, 3000.0, 10000.0]
    gammas = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.55, 0.75, 0.95]
    selection_transitions = ((2021, 2022), (2022, 2023))
    baselines = {
        season: score(artifacts[season]["y"], artifacts[season]["m3"])
        for season in (2021, 2022, 2023, 2024)
    }
    baseline_2024 = score(artifacts[2024]["y"], artifacts[2024]["m3"])
    trials: list[dict[str, Any]] = []
    best: tuple[tuple[float, float], Config, dict[str, Any]] | None = None
    best_by_mode: dict[
        str, tuple[tuple[float, float], Config, dict[str, Any]]
    ] = {}
    # Estimate conditional effects from all regular-season rows for coverage and
    # stability, then route the correction only to R_CORE.  This scope rule is
    # fixed before the 2024 confirmation and avoids tuning a sparse table to one
    # source-to-target transition.
    for scope in ("r_all",):
      for training_mode in ("loo", "full"):
        for k_pitcher, k_hand, k_pressure in k_configs:
            matrix_config = Config(
                scope,
                k_pitcher,
                k_hand,
                k_pressure,
                alphas[0],
                gammas[0],
                training_mode,
            )
            transition_data = {
                (source, target): transfer_data(
                    frames[source],
                    frames[target],
                    artifacts[source]["m3"],
                    matrix_config,
                )
                for source, target in selection_transitions
            }
            for alpha in alphas:
                corrections: dict[tuple[int, int], np.ndarray] = {}
                for transition, data in transition_data.items():
                    model = make_pipeline(
                        StandardScaler(), Ridge(alpha=alpha, fit_intercept=True)
                    )
                    model.fit(data["x_source"], data["residual"])
                    corrections[transition] = model.predict(data["x_target"])
                for gamma in gammas:
                    config = Config(
                        scope,
                        k_pitcher,
                        k_hand,
                        k_pressure,
                        alpha,
                        gamma,
                        training_mode,
                    )
                    transition_results: dict[str, Any] = {}
                    gains: list[float] = []
                    for transition, data in transition_data.items():
                        source, target = transition
                        correction = corrections[transition]
                        prediction = np.asarray(
                            artifacts[target]["m3"], dtype=np.float64
                        ).copy()
                        prediction[data["target_core"]] = np.clip(
                            prediction[data["target_core"]] + gamma * correction,
                            0.0,
                            1.0,
                        )
                        metrics = score(artifacts[target]["y"], prediction)
                        gain = float(
                            metrics["raw_competition_score"]
                            - baselines[target]["raw_competition_score"]
                        )
                        gains.append(gain)
                        transition_results[f"{source}_to_{target}"] = {
                            "metrics": metrics,
                            "gain": gain,
                            "diagnostics": correction_diagnostics(data, correction),
                        }
                    robust_min_gain = float(min(gains))
                    mean_gain = float(np.mean(gains))
                    rank = (robust_min_gain, mean_gain)
                    row = {
                        "config": config.__dict__,
                        "selection_transitions": transition_results,
                        "robust_min_gain": robust_min_gain,
                        "mean_gain": mean_gain,
                    }
                    trials.append(row)
                    if best is None or rank > best[0]:
                        best = (rank, config, row)
                    mode_best = best_by_mode.get(training_mode)
                    if mode_best is None or rank > mode_best[0]:
                        best_by_mode[training_mode] = (rank, config, row)
    if best is None:
        raise RuntimeError("No temporal residual configuration evaluated")
    selected_rank, selected, selected_row = best
    mode_predictions: dict[str, np.ndarray] = {}
    mode_confirmations: dict[str, Any] = {}
    for training_mode in ("loo", "full"):
        mode_rank, mode_config, mode_selection_row = best_by_mode[training_mode]
        mode_prediction, mode_diagnostics = transfer(
            frames[2023],
            frames[2024],
            artifacts[2023]["m3"],
            artifacts[2024]["m3"],
            mode_config,
        )
        mode_metrics = score(artifacts[2024]["y"], mode_prediction)
        mode_predictions[training_mode] = mode_prediction
        mode_confirmations[training_mode] = {
            "selected_config": mode_config.__dict__,
            "selection_robust_min_gain": mode_rank[0],
            "selection_mean_gain": mode_rank[1],
            "selection_transitions": mode_selection_row["selection_transitions"],
            "metrics_2024": mode_metrics,
            "gain_2024": float(
                mode_metrics["raw_competition_score"]
                - baseline_2024["raw_competition_score"]
            ),
            "diagnostics": mode_diagnostics,
        }
    prediction_2024 = 0.5 * (
        mode_predictions["loo"] + mode_predictions["full"]
    )
    diagnostics_2024 = {
        "consensus": "0.5 * independently selected LOO + 0.5 * full-source-table",
        "mode_confirmations": mode_confirmations,
    }
    metrics_2024 = score(artifacts[2024]["y"], prediction_2024)
    gain_2024 = float(
        metrics_2024["raw_competition_score"]
        - baseline_2024["raw_competition_score"]
    )

    domain_metrics: dict[str, Any] = {}
    month_metrics: dict[str, Any] = {}
    for domain in ("R_CORE", "R_ANCHOR", "F"):
        mask = frames[2024]["domain"].eq(domain).to_numpy()
        domain_metrics[domain] = {
            "baseline": score(artifacts[2024]["y"][mask], artifacts[2024]["m3"][mask]),
            "candidate": score(artifacts[2024]["y"][mask], prediction_2024[mask]),
        }
    for month in sorted(frames[2024]["game_month"].unique()):
        mask = frames[2024]["game_month"].eq(month).to_numpy()
        before = score(artifacts[2024]["y"][mask], artifacts[2024]["m3"][mask])
        after = score(artifacts[2024]["y"][mask], prediction_2024[mask])
        month_metrics[str(int(month))] = {
            "rows": int(mask.sum()),
            "baseline_raw": before["raw_competition_score"],
            "candidate_raw": after["raw_competition_score"],
            "gain": after["raw_competition_score"] - before["raw_competition_score"],
        }

    np.savez_compressed(
        OUTPUT_NPZ,
        y=artifacts[2024]["y"],
        row_index=artifacts[2024]["row_index"],
        cluster=artifacts[2024]["cluster"],
        m3=artifacts[2024]["m3"],
        temporal_residual_ridge=prediction_2024,
        temporal_residual_ridge_loo=mode_predictions["loo"],
        temporal_residual_ridge_full=mode_predictions["full"],
    )
    report = {
        "protocol": {
            "official_train_only": True,
            "test_rows_read": False,
            "leaderboard_values_used": False,
            "row_independent_target_lookup": True,
            "selection": "maximize worst gain over 2021->2022 and 2022->2023 transfers",
            "confirmation": "selected configuration refit on 2023 and transferred to 2024",
            "route": "R_CORE only; R_ANCHOR and F predictions unchanged",
            "source_training_features": selected.training_mode,
            "scope_selection_rule": "all R table fixed for coverage; correction routed to R_CORE",
            "primary_method": "equal-weight consensus of independently selected LOO and full source-table Ridge",
        },
        "fixed_estimator": {
            "median_offset": MEDIAN_OFFSET,
            "target_lb": 1190.0,
            "required_local_score": REQUIRED_LOCAL,
        },
        "baseline": {str(season): metrics for season, metrics in baselines.items()},
        "selection": {
            "selected_config": selected.__dict__,
            "robust_min_gain": selected_rank[0],
            "mean_gain": selected_rank[1],
            "transitions": selected_row["selection_transitions"],
            "trial_count": len(trials),
            "top_trials": sorted(
                trials,
                key=lambda row: (row["robust_min_gain"], row["mean_gain"]),
                reverse=True,
            )[:20],
            "best_by_training_mode": {
                mode: {
                    "robust_min_gain": item[0][0],
                    "mean_gain": item[0][1],
                    "config": item[1].__dict__,
                    "transitions": item[2]["selection_transitions"],
                }
                for mode, item in best_by_mode.items()
            },
        },
        "confirmation_2024": {
            "metrics": metrics_2024,
            "gain": gain_2024,
            "expected_lb_median": metrics_2024["raw_competition_score"] + MEDIAN_OFFSET,
            "crosses_required_local_score": bool(
                metrics_2024["raw_competition_score"] > REQUIRED_LOCAL
            ),
            "diagnostics": diagnostics_2024,
            "domains": domain_metrics,
            "months": month_metrics,
            "prediction_artifact": str(OUTPUT_NPZ.relative_to(ROOT)),
        },
    }
    OUTPUT_JSON.write_text(
        json.dumps(json_safe(report), ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "selected": selected.__dict__,
                "selection_robust_min_gain": selected_rank[0],
                "selection_mean_gain": selected_rank[1],
                "confirmation_gain_2024": gain_2024,
                "confirmation_score_2024": metrics_2024["raw_competition_score"],
                "expected_lb_median": metrics_2024["raw_competition_score"] + MEDIAN_OFFSET,
                "crosses_required": metrics_2024["raw_competition_score"] > REQUIRED_LOCAL,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    print(f"Saved {OUTPUT_JSON}")
    print(f"Saved {OUTPUT_NPZ}")


if __name__ == "__main__":
    main()
