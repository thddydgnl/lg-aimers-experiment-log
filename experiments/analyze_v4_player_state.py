#!/usr/bin/env python3
"""Strict next-season player-state residual experiments on the V3 M3 OOF.

Each row uses only completed prior seasons plus its own official as-of counters.
No target-season labels, other target rows, test rows, or leaderboard values are
used to create target-row features.
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

from experiments.analyze_v4_temporal_residual_ridge import (
    MEDIAN_OFFSET,
    REQUIRED_LOCAL,
    add_context,
    json_safe,
    m3_for_season,
    score,
)


TARGET = "control_success"
OUTPUT_JSON = ROOT / "experiments/results/v4_player_state.json"
OUTPUT_NPZ = ROOT / "experiments/results/predictions/v4_player_state_2024.npz"


@dataclass(frozen=True)
class StateConfig:
    prior_k: float
    current_k: float
    decay: float
    trend: float
    alpha: float
    gamma: float


def map_series(rows: pd.DataFrame, entity: str, series: pd.Series) -> np.ndarray:
    return rows[entity].map(series).fillna(0.0).to_numpy(dtype=np.float64)


def entity_state_features(
    full: pd.DataFrame,
    rows: pd.DataFrame,
    target_year: int,
    entity: str,
    asof_n_column: str,
    asof_rate_column: str,
    prefix: str,
    config: StateConfig,
) -> pd.DataFrame:
    history = full.loc[full["season"] < target_year]
    if history.empty:
        raise ValueError(f"No history before {target_year}")
    last_year = target_year - 1
    previous_year = target_year - 2
    last_rows = history.loc[history["season"].eq(last_year)]
    league = float(last_rows[TARGET].mean()) if len(last_rows) else float(history[TARGET].mean())

    career = history.groupby(entity, sort=False, observed=True)[TARGET].agg(["sum", "size"])
    career_sum = map_series(rows, entity, career["sum"])
    career_n = map_series(rows, entity, career["size"])
    career_rate = (career_sum + config.prior_k * league) / (
        career_n + config.prior_k
    )

    def season_values(year: int) -> tuple[np.ndarray, np.ndarray]:
        subset = history.loc[history["season"].eq(year)]
        table = subset.groupby(entity, sort=False, observed=True)[TARGET].agg(
            ["sum", "size"]
        )
        return map_series(rows, entity, table["sum"]), map_series(
            rows, entity, table["size"]
        )

    last_sum, last_n = season_values(last_year)
    previous_sum, previous_n = season_values(previous_year)
    last_rate = (last_sum + config.prior_k * career_rate) / (
        last_n + config.prior_k
    )
    previous_rate = (previous_sum + config.prior_k * career_rate) / (
        previous_n + config.prior_k
    )

    recent = history.loc[history["season"] >= target_year - 3, [
        "season", entity, TARGET
    ]].copy()
    recent["_weight"] = np.power(
        config.decay,
        np.maximum(0, last_year - recent["season"].to_numpy(dtype=np.int16)),
    )
    recent["_weighted_target"] = recent["_weight"] * recent[TARGET]
    recent_table = recent.groupby(entity, sort=False, observed=True).agg(
        weighted_sum=("_weighted_target", "sum"),
        weighted_n=("_weight", "sum"),
    )
    recent_sum = map_series(rows, entity, recent_table["weighted_sum"])
    recent_n = map_series(rows, entity, recent_table["weighted_n"])
    recent_rate = (recent_sum + config.prior_k * career_rate) / (
        recent_n + config.prior_k
    )

    seasonal = history.groupby([entity, "season"], sort=False, observed=True)[
        TARGET
    ].agg(["sum", "size"])
    seasonal["rate"] = (seasonal["sum"] + config.prior_k * league) / (
        seasonal["size"] + config.prior_k
    )
    volatility_table = seasonal.groupby(level=0, observed=True)["rate"].std().fillna(0.0)
    volatility = map_series(rows, entity, volatility_table)

    last_reliability = last_n / (last_n + config.prior_k)
    previous_reliability = previous_n / (previous_n + config.prior_k)
    recent_reliability = recent_n / (recent_n + config.prior_k)
    slope = (last_rate - previous_rate) * np.minimum(
        last_reliability, previous_reliability
    )
    forecast_base = 0.50 * last_rate + 0.30 * recent_rate + 0.20 * career_rate
    forecast = np.clip(forecast_base + config.trend * slope, 0.02, 0.98)

    asof_n = rows[asof_n_column].fillna(0).to_numpy(dtype=np.float64)
    asof_rate = rows[asof_rate_column].fillna(league).to_numpy(dtype=np.float64)
    cumulative_success = np.rint(asof_n * asof_rate)
    current_n = asof_n - career_n
    current_sum = cumulative_success - career_sum
    invalid = (
        (current_n < 0.0)
        | (current_sum < 0.0)
        | (current_sum > current_n)
    )
    if invalid.any():
        raise ValueError(
            f"{prefix}/{target_year} exact as-of reconstruction failed for {int(invalid.sum())} rows"
        )
    current_raw = np.divide(
        current_sum,
        current_n,
        out=forecast.copy(),
        where=current_n > 0,
    )
    current_reliability = current_n / (current_n + config.current_k)
    posterior = (current_sum + config.current_k * forecast) / (
        current_n + config.current_k
    )
    posterior_sd = np.sqrt(
        np.clip(
            posterior * (1.0 - posterior)
            / (current_n + config.current_k + 1.0),
            0.0,
            None,
        )
    )

    features = pd.DataFrame(
        {
            f"{prefix}_career": career_rate,
            f"{prefix}_last": last_rate,
            f"{prefix}_previous": previous_rate,
            f"{prefix}_recent": recent_rate,
            f"{prefix}_forecast_base": forecast_base,
            f"{prefix}_forecast": forecast,
            f"{prefix}_slope": slope,
            f"{prefix}_volatility": volatility,
            f"{prefix}_career_log_n": np.log1p(career_n),
            f"{prefix}_last_log_n": np.log1p(last_n),
            f"{prefix}_recent_log_n": np.log1p(recent_n),
            f"{prefix}_last_reliability": last_reliability,
            f"{prefix}_recent_reliability": recent_reliability,
            f"{prefix}_current_log_n": np.log1p(current_n),
            f"{prefix}_current_raw": current_raw,
            f"{prefix}_current_reliability": current_reliability,
            f"{prefix}_posterior": posterior,
            f"{prefix}_posterior_minus_forecast": posterior - forecast,
            f"{prefix}_posterior_minus_career": posterior - career_rate,
            f"{prefix}_posterior_sd": posterior_sd,
            f"{prefix}_asof_rate": asof_rate,
        },
        index=rows.index,
        dtype=np.float64,
    )
    return features


def state_matrix(
    full: pd.DataFrame,
    rows: pd.DataFrame,
    target_year: int,
    config: StateConfig,
) -> tuple[np.ndarray, list[str]]:
    pitcher = entity_state_features(
        full,
        rows,
        target_year,
        "pitcher_id",
        "asof_pitcher_n",
        "asof_pitcher_success_rate",
        "pitcher_state",
        config,
    )
    batter = entity_state_features(
        full,
        rows,
        target_year,
        "batter_id",
        "asof_batter_n",
        "asof_batter_success_rate",
        "batter_state",
        config,
    )
    cross = pd.DataFrame(
        {
            "state_forecast_gap": pitcher["pitcher_state_forecast"].to_numpy()
            - batter["batter_state_forecast"].to_numpy(),
            "state_posterior_gap": pitcher["pitcher_state_posterior"].to_numpy()
            - batter["batter_state_posterior"].to_numpy(),
            "state_current_raw_gap": pitcher["pitcher_state_current_raw"].to_numpy()
            - batter["batter_state_current_raw"].to_numpy(),
            "state_joint_reliability": np.minimum(
                pitcher["pitcher_state_current_reliability"].to_numpy(),
                batter["batter_state_current_reliability"].to_numpy(),
            ),
            "state_joint_uncertainty": np.hypot(
                pitcher["pitcher_state_posterior_sd"].to_numpy(),
                batter["batter_state_posterior_sd"].to_numpy(),
            ),
        },
        index=rows.index,
    )
    balls = rows["balls_before"].to_numpy(dtype=np.float64)
    strikes = rows["strikes_before"].to_numpy(dtype=np.float64)
    context = pd.DataFrame(
        {
            "ctx_balls": balls / 3.0,
            "ctx_strikes": strikes / 2.0,
            "ctx_ball_advantage": (balls - strikes) / 3.0,
            "ctx_full_count": ((balls == 3) & (strikes == 2)).astype(float),
            "ctx_three_ball": (balls == 3).astype(float),
            "ctx_two_strike": (strikes == 2).astype(float),
            "ctx_same_hand": rows["pitcher_hand"].to_numpy()
            == rows["batter_hand"].to_numpy(),
        },
        index=rows.index,
        dtype=np.float64,
    )
    frame = pd.concat([pitcher, batter, cross, context], axis=1)
    matrix = frame.to_numpy(dtype=np.float64)
    if not np.isfinite(matrix).all():
        raise ValueError("Non-finite player-state feature")
    return matrix, list(frame.columns)


def load_data() -> tuple[
    pd.DataFrame,
    dict[int, pd.DataFrame],
    dict[int, dict[str, np.ndarray]],
]:
    artifacts = {season: m3_for_season(season) for season in (2022, 2023, 2024)}
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
        "asof_pitcher_n",
        "asof_pitcher_success_rate",
        "asof_batter_n",
        "asof_batter_success_rate",
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
        rows = np.asarray(artifact["row_index"], dtype=np.int64)
        frame = add_context(full.iloc[rows].reset_index(drop=True))
        if not np.array_equal(
            frame[TARGET].to_numpy(dtype=np.int8), artifact["y"].astype(np.int8)
        ):
            raise ValueError(f"OOF target alignment mismatch for {season}")
        frames[season] = frame
    return full, frames, artifacts


def fit_correction(
    x_source: np.ndarray,
    y_source: np.ndarray,
    p_source: np.ndarray,
    x_target: np.ndarray,
    alpha: float,
) -> np.ndarray:
    model = make_pipeline(StandardScaler(), Ridge(alpha=alpha, fit_intercept=True))
    model.fit(x_source, y_source - p_source)
    return model.predict(x_target)


def apply_correction(
    base: np.ndarray,
    core_mask: np.ndarray,
    correction: np.ndarray,
    gamma: float,
) -> np.ndarray:
    result = np.asarray(base, dtype=np.float64).copy()
    result[core_mask] = np.clip(
        result[core_mask] + gamma * correction, 0.0, 1.0
    )
    return result


def main() -> None:
    full, frames, artifacts = load_data()
    structural_configs = [
        (100.0, 50.0, 0.70, 0.25),
        (200.0, 50.0, 0.70, 0.25),
        (100.0, 100.0, 0.70, 0.25),
        (200.0, 100.0, 0.70, 0.25),
        (200.0, 80.0, 0.85, 0.25),
        (200.0, 80.0, 0.70, 0.00),
        (200.0, 80.0, 0.70, 0.50),
        (500.0, 100.0, 0.70, 0.25),
    ]
    alphas = [10.0, 100.0, 300.0, 1000.0, 3000.0, 10000.0]
    gammas = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.55, 0.75]
    core_2022 = frames[2022]["domain"].eq("R_CORE").to_numpy()
    core_2023 = frames[2023]["domain"].eq("R_CORE").to_numpy()
    core_2024 = frames[2024]["domain"].eq("R_CORE").to_numpy()
    baseline_2023 = score(artifacts[2023]["y"], artifacts[2023]["m3"])
    baseline_2024 = score(artifacts[2024]["y"], artifacts[2024]["m3"])

    trials: list[dict[str, Any]] = []
    best: tuple[float, StateConfig, list[str]] | None = None
    for prior_k, current_k, decay, trend in structural_configs:
        skeleton = StateConfig(prior_k, current_k, decay, trend, alphas[0], gammas[0])
        x_2022, names_2022 = state_matrix(
            full, frames[2022].loc[core_2022], 2022, skeleton
        )
        x_2023, names_2023 = state_matrix(
            full, frames[2023].loc[core_2023], 2023, skeleton
        )
        if names_2022 != names_2023:
            raise AssertionError("Player-state feature order mismatch")
        for alpha in alphas:
            correction = fit_correction(
                x_2022,
                artifacts[2022]["y"][core_2022],
                artifacts[2022]["m3"][core_2022],
                x_2023,
                alpha,
            )
            for gamma in gammas:
                config = StateConfig(prior_k, current_k, decay, trend, alpha, gamma)
                prediction = apply_correction(
                    artifacts[2023]["m3"], core_2023, correction, gamma
                )
                metrics = score(artifacts[2023]["y"], prediction)
                gain = float(
                    metrics["raw_competition_score"]
                    - baseline_2023["raw_competition_score"]
                )
                trials.append(
                    {
                        "config": config.__dict__,
                        "gain_2023": gain,
                        "metrics_2023": metrics,
                        "correction_mean": float(correction.mean()),
                        "correction_std": float(correction.std()),
                    }
                )
                if best is None or gain > best[0]:
                    best = (gain, config, names_2022)
        del x_2022, x_2023

    if best is None:
        raise RuntimeError("No player-state candidate evaluated")
    selected_gain, selected, feature_names = best
    x_2023, names_2023 = state_matrix(
        full, frames[2023].loc[core_2023], 2023, selected
    )
    x_2024, names_2024 = state_matrix(
        full, frames[2024].loc[core_2024], 2024, selected
    )
    if feature_names != names_2023 or feature_names != names_2024:
        raise AssertionError("Confirmation player-state feature order mismatch")
    correction_2024 = fit_correction(
        x_2023,
        artifacts[2023]["y"][core_2023],
        artifacts[2023]["m3"][core_2023],
        x_2024,
        selected.alpha,
    )
    prediction_2024 = apply_correction(
        artifacts[2024]["m3"], core_2024, correction_2024, selected.gamma
    )
    metrics_2024 = score(artifacts[2024]["y"], prediction_2024)
    gain_2024 = float(
        metrics_2024["raw_competition_score"]
        - baseline_2024["raw_competition_score"]
    )

    domains: dict[str, Any] = {}
    months: dict[str, Any] = {}
    for domain in ("R_CORE", "R_ANCHOR", "F"):
        mask = frames[2024]["domain"].eq(domain).to_numpy()
        before = score(artifacts[2024]["y"][mask], artifacts[2024]["m3"][mask])
        after = score(artifacts[2024]["y"][mask], prediction_2024[mask])
        domains[domain] = {
            "baseline": before,
            "candidate": after,
            "gain": after["raw_competition_score"] - before["raw_competition_score"],
        }
    for month in sorted(frames[2024]["game_month"].unique()):
        mask = frames[2024]["game_month"].eq(month).to_numpy()
        before = score(artifacts[2024]["y"][mask], artifacts[2024]["m3"][mask])
        after = score(artifacts[2024]["y"][mask], prediction_2024[mask])
        months[str(int(month))] = {
            "rows": int(mask.sum()),
            "gain": after["raw_competition_score"] - before["raw_competition_score"],
        }

    np.savez_compressed(
        OUTPUT_NPZ,
        y=artifacts[2024]["y"],
        row_index=artifacts[2024]["row_index"],
        cluster=artifacts[2024]["cluster"],
        m3=artifacts[2024]["m3"],
        player_state=prediction_2024,
    )
    report = {
        "protocol": {
            "official_train_only": True,
            "test_rows_read": False,
            "leaderboard_values_used": False,
            "row_independent": True,
            "exact_asof_reconstruction": True,
            "selection": "fit 2022 M3 residual and transfer to 2023",
            "confirmation": "refit selected recipe on 2023 residual and transfer to 2024",
            "route": "R_CORE only",
        },
        "fixed_estimator": {
            "target_lb": 1190.0,
            "median_offset": MEDIAN_OFFSET,
            "required_local_score": REQUIRED_LOCAL,
        },
        "baseline": {"2023": baseline_2023, "2024": baseline_2024},
        "selection": {
            "selected_config": selected.__dict__,
            "gain_2023": selected_gain,
            "feature_count": len(feature_names),
            "feature_names": feature_names,
            "trial_count": len(trials),
            "top_trials": sorted(trials, key=lambda item: item["gain_2023"], reverse=True)[:25],
        },
        "confirmation_2024": {
            "metrics": metrics_2024,
            "gain": gain_2024,
            "expected_lb_median": metrics_2024["raw_competition_score"] + MEDIAN_OFFSET,
            "crosses_required_local_score": bool(
                metrics_2024["raw_competition_score"] > REQUIRED_LOCAL
            ),
            "correction_mean": float(correction_2024.mean()),
            "correction_std": float(correction_2024.std()),
            "domains": domains,
            "months": months,
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
                "selection_gain_2023": selected_gain,
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
