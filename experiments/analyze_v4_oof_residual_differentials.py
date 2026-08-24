#!/usr/bin/env python3
"""Rebuild persistent pitcher residual-differential tables from local OOF data.

The three contexts (same-hand matchup, two strikes, runner present) and their
shrinkage constants were fixed by prior public official-data research.  Every
target season uses only the immediately preceding two seasons' strictly OOF
residuals.  No external model artifact or player mapping is consumed here.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.analyze_v4_temporal_residual_ridge import (  # noqa: E402
    MEDIAN_OFFSET,
    REQUIRED_LOCAL,
    json_safe,
    score,
)


PRED = ROOT / "experiments/results/predictions"
REPORT = ROOT / "experiments/results/v4_oof_residual_differentials.json"
MODEL_KEY = "catboost_numeric"
CONTEXTS = {
    "same_hand": 1000.0,
    "two_strikes": 1000.0,
    "runner_present": 2000.0,
}
TARGETS = (2022, 2023, 2024)


def load(path: Path) -> dict[str, np.ndarray]:
    with np.load(path) as archive:
        return {key: np.asarray(archive[key]) for key in archive.files}


def source_artifact(year: int) -> dict[str, np.ndarray]:
    stem = (
        "v4_numeric_cat_current_context_level_tmctx_seed42_early"
        if year < 2022
        else "v4_numeric_cat_current_context_level_tmctx_seed42"
    )
    return load(PRED / f"{stem}_{year}.npz")


def raw_score(y: np.ndarray, p: np.ndarray) -> float:
    return float(score(y, np.clip(p, 0.0, 1.0))["raw_competition_score"])


def fit_scalar(direction: np.ndarray, residual: np.ndarray) -> float:
    denominator = float(np.dot(direction, direction))
    return float(np.dot(direction, residual) / denominator) if denominator else 0.0


def context_values(frame: pd.DataFrame) -> dict[str, np.ndarray]:
    return {
        "same_hand": (
            pd.to_numeric(frame["pitcher_hand"], errors="coerce").to_numpy()
            == pd.to_numeric(frame["batter_hand"], errors="coerce").to_numpy()
        ).astype(np.int8),
        "two_strikes": (
            pd.to_numeric(frame["strikes_before"], errors="coerce").to_numpy() == 2
        ).astype(np.int8),
        "runner_present": (
            pd.to_numeric(frame["num_runners_on"], errors="coerce").to_numpy() > 0
        ).astype(np.int8),
    }


def differential_table(
    pitcher: np.ndarray, context: np.ndarray, residual: np.ndarray, k: float
) -> pd.Series:
    grouped = pd.DataFrame(
        {"pitcher": pitcher, "context": context, "residual": residual}
    ).groupby(["pitcher", "context"], sort=True)["residual"].agg(["mean", "size"])
    wide_mean = grouped["mean"].unstack()
    wide_size = grouped["size"].unstack().fillna(0.0)
    for value in (0, 1):
        if value not in wide_mean:
            wide_mean[value] = np.nan
            wide_size[value] = 0.0
    n0, n1 = wide_size[0], wide_size[1]
    effective = (n0 * n1) / (n0 + n1).replace(0.0, np.nan)
    difference = wide_mean[1] - wide_mean[0]
    return (difference * effective / (effective + k)).dropna()


def apply_table(
    table: pd.Series, pitcher: np.ndarray, context: np.ndarray
) -> np.ndarray:
    values = pd.Series(pitcher).map(table).fillna(0.0).to_numpy(dtype=np.float64)
    return values * np.where(context == 1, 0.5, -0.5)


def main() -> None:
    raw = pd.read_csv(
        ROOT / "open/data/train.csv",
        usecols=["season", "pitcher_id", "pitcher_hand", "batter_hand",
                 "strikes_before", "num_runners_on"],
        encoding="utf-8-sig",
        low_memory=False,
    )
    artifacts = {year: source_artifact(year) for year in range(2020, 2025)}
    frames: dict[int, pd.DataFrame] = {}
    y: dict[int, np.ndarray] = {}
    model_prediction: dict[int, np.ndarray] = {}
    for year, artifact in artifacts.items():
        frame = raw.iloc[artifact["row_index"].astype(np.int64)].reset_index(drop=True)
        if not frame["season"].eq(year).all():
            raise ValueError(f"row_index contains another season for {year}")
        frames[year] = frame
        y[year] = artifact["y"].astype(np.float64)
        model_prediction[year] = artifact[MODEL_KEY].astype(np.float64)

    axis_directions: dict[int, dict[str, np.ndarray]] = {}
    directions: dict[int, np.ndarray] = {}
    table_meta: dict[str, object] = {}
    for target in TARGETS:
        source_years = (target - 2, target - 1)
        source_pitcher = np.concatenate(
            [frames[year]["pitcher_id"].to_numpy(dtype=np.int64) for year in source_years]
        )
        source_residual = np.concatenate(
            [y[year] - model_prediction[year] for year in source_years]
        )
        source_context = {name: [] for name in CONTEXTS}
        for year in source_years:
            current = context_values(frames[year])
            for name in CONTEXTS:
                source_context[name].append(current[name])
        target_context = context_values(frames[target])
        target_pitcher = frames[target]["pitcher_id"].to_numpy(dtype=np.int64)
        axis_directions[target] = {}
        details = {}
        for name, k in CONTEXTS.items():
            table = differential_table(
                source_pitcher,
                np.concatenate(source_context[name]),
                source_residual,
                k,
            )
            values = apply_table(table, target_pitcher, target_context[name])
            axis_directions[target][name] = values
            details[name] = {
                "k": k,
                "source_seasons": source_years,
                "pitcher_cells": int(len(table)),
                "target_nonzero_rate": float(np.mean(values != 0.0)),
                "direction_std": float(values.std()),
            }
        directions[target] = sum(axis_directions[target].values())
        table_meta[str(target)] = details

    model_scores = {
        year: raw_score(y[year], model_prediction[year]) for year in TARGETS
    }
    route_r = {
        year: frames[year]["season"].to_numpy() == year for year in TARGETS
    }
    # Only the R mask is needed; take it from the locked artifact for reliable
    # handling of the 2023 F label-regime break.
    for year in TARGETS:
        locked = load(PRED / f"v4_routed_tabm_stack_locked_{year}.npz")
        if not np.array_equal(locked["row_index"], artifacts[year]["row_index"]):
            raise ValueError(f"locked alignment mismatch for {year}")
        route_r[year] = locked["game_type_r"].astype(bool)

    source_outputs = {}
    for year in TARGETS:
        path = PRED / f"v4_oof_residual_source_{year}.npz"
        direction_r = np.where(route_r[year], directions[year], 0.0)
        np.savez_compressed(
            path,
            y=y[year], row_index=artifacts[year]["row_index"],
            cluster=artifacts[year]["cluster"], game_type_r=route_r[year],
            source_model=model_prediction[year], direction_c3_all=directions[year],
            direction_c3_r=direction_r,
            source_plus_c3_all=np.clip(model_prediction[year] + directions[year], 0.0, 1.0),
            source_plus_c3_r=np.clip(model_prediction[year] + direction_r, 0.0, 1.0),
        )
        source_outputs[year] = str(path.relative_to(ROOT))

    screens = {}
    routed_directions: dict[str, dict[int, np.ndarray]] = {}
    for route in ("all", "R"):
        values = {
            year: (
                directions[year]
                if route == "all"
                else np.where(route_r[year], directions[year], 0.0)
            )
            for year in TARGETS
        }
        gamma22 = fit_scalar(values[2022], y[2022] - model_prediction[2022])
        gamma23 = fit_scalar(values[2023], y[2023] - model_prediction[2023])
        gain22 = raw_score(
            y[2022], model_prediction[2022] + gamma22 * values[2022]
        ) - model_scores[2022]
        transfer23 = raw_score(
            y[2023], model_prediction[2023] + gamma22 * values[2023]
        ) - model_scores[2023]
        fixed_gains = {
            year: raw_score(y[year], model_prediction[year] + values[year])
            - model_scores[year]
            for year in TARGETS
        }
        ratio = abs(gamma23 / gamma22) if abs(gamma22) > 1e-9 else float("inf")
        screens[route] = {
            "gamma_fit_2022": gamma22,
            "gamma_fit_2023": gamma23,
            "gamma_abs_ratio": ratio,
            "gain_fit_2022": gain22,
            "transfer_gain_2023": transfer23,
            "fixed_weight_one_gains": fixed_gains,
            "coefficient_stable": bool(gamma22 * gamma23 > 0 and 0.5 <= ratio <= 2.0),
        }
        routed_directions[route] = values

    # Route selection uses 2022 fit and untouched 2023 transfer only.
    eligible_routes = [
        route for route, row in screens.items()
        if row["gain_fit_2022"] > 0.05
        and row["transfer_gain_2023"] > 0.05
        and row["coefficient_stable"]
    ]
    selected_route = max(
        eligible_routes,
        key=lambda route: float(screens[route]["transfer_gain_2023"]),
        default="none",
    )

    current_artifacts = {
        year: load(PRED / f"v4_tabtransformer_seed_ensemble_{year}.npz")
        for year in (2023, 2024)
    }
    current = {
        year: current_artifacts[year]["base_plus_tabtransformer_seed_average"].astype(np.float64)
        for year in current_artifacts
    }
    current_scores = {year: raw_score(y[year], current[year]) for year in current}
    current_diagnostics = {}
    for route, values in routed_directions.items():
        refit_gamma = float(np.clip(
            fit_scalar(values[2023], y[2023] - current[2023]), -4.0, 4.0
        ))
        historical_gamma = float(screens[route]["gamma_fit_2022"])
        variants = {}
        for label, value in (
            ("fixed_weight_one", 1.0),
            ("historical_gamma_2022", historical_gamma),
            ("refit_gamma_2023", refit_gamma),
        ):
            candidate_scores = {
                year: raw_score(y[year], current[year] + value * values[year])
                for year in (2023, 2024)
            }
            variants[label] = {
                "gamma": value,
                "scores": candidate_scores,
                "gains": {
                    year: candidate_scores[year] - current_scores[year]
                    for year in candidate_scores
                },
                "expected_lb_median": candidate_scores[2024] + MEDIAN_OFFSET,
            }
        current_diagnostics[route] = variants
    confirmation = None
    outputs = {}
    if selected_route != "none":
        values = routed_directions[selected_route]
        gamma = float(np.clip(
            fit_scalar(values[2023], y[2023] - current[2023]), -4.0, 4.0
        ))
        prediction = {
            year: np.clip(current[year] + gamma * values[year], 0.0, 1.0)
            for year in (2023, 2024)
        }
        candidate_scores = {
            year: raw_score(y[year], prediction[year]) for year in prediction
        }
        confirmation = {
            "selected_route": selected_route,
            "gamma_fit_current_2023": gamma,
            "current_scores": current_scores,
            "candidate_scores": candidate_scores,
            "gains": {
                year: candidate_scores[year] - current_scores[year]
                for year in candidate_scores
            },
            "expected_lb_median": candidate_scores[2024] + MEDIAN_OFFSET,
            "required_local_score": REQUIRED_LOCAL,
            "crosses_required_local_score": candidate_scores[2024] > REQUIRED_LOCAL,
        }
        for year in (2023, 2024):
            path = PRED / f"v4_oof_residual_differentials_{year}.npz"
            np.savez_compressed(
                path,
                y=y[year], row_index=artifacts[year]["row_index"],
                cluster=artifacts[year]["cluster"], base=current[year],
                direction_c3=values[year], base_plus_c3=prediction[year],
                **{f"direction_{name}": axis_directions[year][name]
                   for name in CONTEXTS},
            )
            outputs[year] = str(path.relative_to(ROOT))

    report = {
        "protocol": {
            "official_train_only": True,
            "external_model_artifacts_used": False,
            "test_rows_read": False,
            "residual_predictions_are_strictly_oof": True,
            "source_window": "immediately preceding two seasons",
            "route_selection_years": [2022, 2023],
            "coefficient_refit_year": 2023,
            "confirmation_year": 2024,
            "selection_does_not_read_2024_labels": True,
        },
        "contexts": CONTEXTS,
        "table_meta": table_meta,
        "source_model_scores": model_scores,
        "route_screens": screens,
        "selected_route": selected_route,
        "current_base_diagnostics": current_diagnostics,
        "confirmation": confirmation,
        "prediction_artifacts": outputs,
        "source_prediction_artifacts": source_outputs,
    }
    REPORT.write_text(json.dumps(json_safe(report), ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(json_safe(report), ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
