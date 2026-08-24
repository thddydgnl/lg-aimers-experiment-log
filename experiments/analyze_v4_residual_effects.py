#!/usr/bin/env python3
"""Leakage-safe residual main-effect experiments for the fixed V3 M3 blend.

This script never reads test.csv or any external repository artifact.  It learns
small, strongly-shrunk group corrections from earlier-season OOF residuals.  The
hyperparameters are selected on 2022 -> 2023 R games, then transferred unchanged
to 2022 + 2023 R -> 2024.  A separate 2024 exploratory sweep is reported but is
never labelled as confirmatory evidence.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
PREDICTIONS = ROOT / "experiments" / "results" / "predictions"
OUTPUT_JSON = ROOT / "experiments" / "results" / "v4_residual_effects.json"
OUTPUT_NPZ = PREDICTIONS / "v4_residual_effects_strict_2024.npz"

COMPONENTS = {
    "A": {
        2022: "v3_sparse_a_backtest_2022.npz",
        2023: "v3_sparse_a_backtest_2023.npz",
        2024: "v3_outcome_trackmanrich_overall_components120_e14k50_batter80_middle100_2024.npz",
    },
    "B": {
        2022: "v3_sparse_b_backtest_2022.npz",
        2023: "v3_sparse_b_backtest_2023.npz",
        2024: "v3_outcome_batter80_middle100_hgroups500_2024.npz",
    },
    "C": {
        2022: "v3_sparse_c_backtest_2022.npz",
        2023: "v3_sparse_c_backtest_2023.npz",
        2024: "v3_outcome_trackmanrich_overall_e14k50_batter80_middle100_2024.npz",
    },
}
M3_WEIGHTS = {
    "A": 0.501443851662535,
    "C": 0.27016033407769313,
    "B": 0.22839581425977187,
}
CALIBRATION_SLOPE = 1.05
CALIBRATION_OFFSET = -0.006
FIXED_LB_OFFSET_MEDIAN = 140.1475834416
TARGET_LB = 1190.0
TARGET_LOCAL = TARGET_LB - FIXED_LB_OFFSET_MEDIAN

USECOLS = [
    "season",
    "game_type",
    "inning",
    "balls_before",
    "strikes_before",
    "num_runners_on",
    "base_state",
    "pitcher_id",
    "batter_id",
    "pitcher_hand",
    "batter_hand",
    "pitcher_team_id",
    "batter_team_id",
    "control_success",
]

FEATURES: dict[str, tuple[str, ...]] = {
    "pitcher": ("pitcher_id",),
    "batter": ("batter_id",),
    "pitcher_game_type": ("pitcher_id", "game_type"),
    "batter_game_type": ("batter_id", "game_type"),
    "pitcher_batter_hand": ("pitcher_id", "batter_hand"),
    "batter_pitcher_hand": ("batter_id", "pitcher_hand"),
    "pitcher_count": ("pitcher_id", "balls_before", "strikes_before"),
    "pitcher_ball_advantage": ("pitcher_id", "ball_advantage"),
    "pitcher_two_strike": ("pitcher_id", "two_strike"),
    "pitcher_runner_present": ("pitcher_id", "runner_present"),
    "pitcher_base_state": ("pitcher_id", "base_state"),
    "pitcher_inning_bucket": ("pitcher_id", "inning_bucket"),
    "pitcher_batter": ("pitcher_id", "batter_id"),
    "pitcher_team": ("pitcher_team_id",),
    "batter_team": ("batter_team_id",),
}

K_GRID = (100.0, 300.0, 1_000.0, 3_000.0, 10_000.0, 20_000.0, 50_000.0, 100_000.0)
WEIGHT_GRID = (0.25, 0.5, 1.0, 1.5, 2.0, 3.0, 5.0)


@dataclass
class SeasonData:
    season: int
    frame: pd.DataFrame
    y: np.ndarray
    row_index: np.ndarray
    cluster: np.ndarray
    raw: np.ndarray


def calibrate(prediction: np.ndarray) -> np.ndarray:
    return np.clip(
        0.5 + CALIBRATION_SLOPE * (prediction - 0.5) + CALIBRATION_OFFSET,
        1e-6,
        1.0 - 1e-6,
    )


def metrics(y: np.ndarray, prediction: np.ndarray) -> dict[str, float | int]:
    y64 = np.asarray(y, dtype=np.float64)
    p64 = np.asarray(prediction, dtype=np.float64)
    brier = float(np.mean(np.square(p64 - y64)))
    rate = float(y64.mean())
    reference = rate * (1.0 - rate)
    raw_score = float(100_000.0 * (1.0 - brier / reference))
    return {
        "rows": int(len(y64)),
        "target_rate": rate,
        "prediction_mean": float(p64.mean()),
        "prediction_std": float(p64.std()),
        "brier": brier,
        "reference_brier": reference,
        "raw_competition_score": raw_score,
        "competition_score": max(0.0, raw_score),
    }


def load_archive(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        return {key: archive[key] for key in archive.files}


def load_seasons() -> dict[int, SeasonData]:
    context = pd.read_csv(ROOT / "open" / "data" / "train.csv", usecols=USECOLS)
    context["ball_advantage"] = (context["balls_before"] > context["strikes_before"]).astype("int8")
    context["two_strike"] = (context["strikes_before"] == 2).astype("int8")
    context["runner_present"] = (context["num_runners_on"] > 0).astype("int8")
    context["inning_bucket"] = np.minimum(context["inning"].to_numpy(), 10).astype("int16")

    result: dict[int, SeasonData] = {}
    for season in (2022, 2023, 2024):
        reference: dict[str, np.ndarray] | None = None
        predictions: dict[str, np.ndarray] = {}
        for key in ("A", "B", "C"):
            artifact = load_archive(PREDICTIONS / COMPONENTS[key][season])
            if reference is None:
                reference = artifact
            else:
                for aligned in ("y", "row_index", "cluster"):
                    if not np.array_equal(reference[aligned], artifact[aligned]):
                        raise ValueError(f"{season} component alignment mismatch: {aligned}")
            predictions[key] = np.asarray(artifact["catboost_outcome"], dtype=np.float64)
        assert reference is not None
        raw = sum(M3_WEIGHTS[key] * predictions[key] for key in M3_WEIGHTS)
        row_index = np.asarray(reference["row_index"], dtype=np.int64)
        frame = context.iloc[row_index].reset_index(drop=True)
        y = np.asarray(reference["y"], dtype=np.float64)
        if not np.array_equal(frame["control_success"].to_numpy(dtype=np.float64), y):
            raise ValueError(f"{season} target/context alignment mismatch")
        if not bool((frame["season"].to_numpy() == season).all()):
            raise ValueError(f"{season} season/context alignment mismatch")
        result[season] = SeasonData(
            season=season,
            frame=frame,
            y=y,
            row_index=row_index,
            cluster=np.asarray(reference["cluster"]),
            raw=np.asarray(raw, dtype=np.float64),
        )
    return result


def combined_codes(
    source: pd.DataFrame,
    target: pd.DataFrame,
    keys: tuple[str, ...],
) -> tuple[np.ndarray, np.ndarray, int]:
    combined = pd.concat(
        [source.loc[:, list(keys)], target.loc[:, list(keys)]],
        axis=0,
        ignore_index=True,
    )
    if len(keys) == 1:
        values = combined[keys[0]].to_numpy()
        codes, uniques = pd.factorize(values, sort=False, use_na_sentinel=True)
    else:
        index = pd.MultiIndex.from_frame(combined, names=list(keys))
        codes, uniques = pd.factorize(index, sort=False, use_na_sentinel=True)
    codes = np.asarray(codes, dtype=np.int64)
    codes[codes < 0] = len(uniques)
    split = len(source)
    return codes[:split], codes[split:], int(len(uniques) + 1)


def effect_grid(
    source_frame: pd.DataFrame,
    source_residual: np.ndarray,
    source_weight: np.ndarray,
    target_frame: pd.DataFrame,
    keys: tuple[str, ...],
) -> dict[float, np.ndarray]:
    source_codes, target_codes, size = combined_codes(source_frame, target_frame, keys)
    weighted_count = np.bincount(source_codes, weights=source_weight, minlength=size)
    weighted_sum = np.bincount(
        source_codes,
        weights=source_weight * source_residual,
        minlength=size,
    )
    seen = weighted_count > 0.0
    output: dict[float, np.ndarray] = {}
    for k in K_GRID:
        table = np.zeros(size, dtype=np.float64)
        table[seen] = weighted_sum[seen] / (weighted_count[seen] + k)
        output[k] = table[target_codes]
    return output


def source_bundle(
    seasons: dict[int, SeasonData],
    years: Iterable[int],
    r_only_years: set[int] | None = None,
    year_weights: dict[int, float] | None = None,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    frames: list[pd.DataFrame] = []
    residuals: list[np.ndarray] = []
    weights: list[np.ndarray] = []
    r_only_years = r_only_years or set()
    year_weights = year_weights or {}
    for year in years:
        item = seasons[year]
        mask = np.ones(len(item.y), dtype=bool)
        if year in r_only_years:
            mask = item.frame["game_type"].to_numpy() == "R"
        frames.append(item.frame.loc[mask].reset_index(drop=True))
        residuals.append((item.y - item.raw)[mask])
        weights.append(np.full(int(mask.sum()), year_weights.get(year, 1.0), dtype=np.float64))
    return (
        pd.concat(frames, ignore_index=True),
        np.concatenate(residuals),
        np.concatenate(weights),
    )


def effect_library(
    source: tuple[pd.DataFrame, np.ndarray, np.ndarray],
    target: SeasonData,
) -> dict[tuple[str, float], np.ndarray]:
    frame, residual, weight = source
    library: dict[tuple[str, float], np.ndarray] = {}
    for feature, keys in FEATURES.items():
        for k, effect in effect_grid(frame, residual, weight, target.frame, keys).items():
            library[(feature, k)] = effect
    return library


def greedy_select(
    target: SeasonData,
    library: dict[tuple[str, float], np.ndarray],
    score_mask: np.ndarray,
    rounds: int,
) -> tuple[list[dict[str, float | str]], np.ndarray, list[dict[str, float | int]]]:
    correction = np.zeros(len(target.y), dtype=np.float64)
    selected: list[dict[str, float | str]] = []
    curve: list[dict[str, float | int]] = []
    used_features: set[str] = set()
    y = target.y[score_mask]
    base_metrics = metrics(y, calibrate(target.raw[score_mask]))
    curve.append({"round": 0, **base_metrics})
    current_brier = float(base_metrics["brier"])

    for round_index in range(1, rounds + 1):
        best: tuple[float, str, float, float, np.ndarray] | None = None
        for (feature, k), effect in library.items():
            if feature in used_features:
                continue
            for weight in WEIGHT_GRID:
                candidate_correction = correction + weight * effect
                prediction = calibrate(target.raw + candidate_correction)[score_mask]
                brier = float(np.mean(np.square(prediction - y)))
                if best is None or brier < best[0]:
                    best = (brier, feature, k, weight, candidate_correction)
        if best is None or best[0] >= current_brier - 1e-12:
            break
        current_brier, feature, k, weight, correction = best
        used_features.add(feature)
        selected.append({"feature": feature, "k": k, "weight": weight})
        curve.append(
            {
                "round": round_index,
                **metrics(y, calibrate(target.raw + correction)[score_mask]),
            }
        )
    return selected, correction, curve


def transfer_selection(
    selected: list[dict[str, float | str]],
    library: dict[tuple[str, float], np.ndarray],
) -> np.ndarray:
    if not library:
        raise ValueError("Empty effect library")
    first = next(iter(library.values()))
    correction = np.zeros(len(first), dtype=np.float64)
    for item in selected:
        key = (str(item["feature"]), float(item["k"]))
        correction += float(item["weight"]) * library[key]
    return correction


def exploratory_individual_sweep(
    target: SeasonData,
    libraries: dict[str, dict[tuple[str, float], np.ndarray]],
    limit: int = 100,
) -> list[dict[str, float | str]]:
    rows: list[dict[str, float | str]] = []
    for source_name, library in libraries.items():
        for (feature, k), effect in library.items():
            for weight in WEIGHT_GRID:
                result = metrics(target.y, calibrate(target.raw + weight * effect))
                rows.append(
                    {
                        "source": source_name,
                        "feature": feature,
                        "k": k,
                        "weight": weight,
                        "brier": float(result["brier"]),
                        "local_score": float(result["competition_score"]),
                    }
                )
    rows.sort(key=lambda item: float(item["brier"]))
    return rows[:limit]


def main() -> None:
    seasons = load_seasons()
    baseline = {
        str(year): {
            "raw": metrics(item.y, item.raw),
            "calibrated": metrics(item.y, calibrate(item.raw)),
        }
        for year, item in seasons.items()
    }

    # Hyperparameters are fixed on the earlier 2022 -> 2023 regular-season transfer.
    source_2022 = source_bundle(seasons, [2022])
    library_2023 = effect_library(source_2022, seasons[2023])
    mask_2023_r = seasons[2023].frame["game_type"].to_numpy() == "R"
    selected, _, selection_curve = greedy_select(
        seasons[2023], library_2023, mask_2023_r, rounds=5
    )

    # Apply the frozen feature/k/weight recipe using all stable OOF residual history.
    source_2022_2023r = source_bundle(seasons, [2022, 2023], r_only_years={2023})
    strict_library_2024 = effect_library(source_2022_2023r, seasons[2024])
    strict_correction = transfer_selection(selected, strict_library_2024)
    strict_raw = np.clip(seasons[2024].raw + strict_correction, 1e-6, 1.0 - 1e-6)
    strict_calibrated = calibrate(strict_raw)
    strict_metrics = metrics(seasons[2024].y, strict_calibrated)

    # Alternative source policies are diagnostic only; 2024 selects none of them.
    sources_2024 = {
        "2022_only": source_bundle(seasons, [2022]),
        "2023_r_only": source_bundle(seasons, [2023], r_only_years={2023}),
        "2022_plus_2023_r": source_2022_2023r,
        "2022_half_plus_2023_r": source_bundle(
            seasons,
            [2022, 2023],
            r_only_years={2023},
            year_weights={2022: 0.5, 2023: 1.0},
        ),
        "2022_plus_2023_all": source_bundle(seasons, [2022, 2023]),
    }
    libraries_2024 = {
        name: effect_library(source, seasons[2024])
        for name, source in sources_2024.items()
    }
    top_individual = exploratory_individual_sweep(seasons[2024], libraries_2024)
    exploratory_selected, exploratory_correction, exploratory_curve = greedy_select(
        seasons[2024], libraries_2024["2022_plus_2023_r"], np.ones(len(seasons[2024].y), dtype=bool), rounds=5
    )
    exploratory_prediction = calibrate(seasons[2024].raw + exploratory_correction)

    np.savez_compressed(
        OUTPUT_NPZ,
        y=seasons[2024].y.astype(np.int8),
        row_index=seasons[2024].row_index,
        cluster=seasons[2024].cluster,
        v3_m3_raw=seasons[2024].raw,
        correction=strict_correction,
        v4_residual_raw=strict_raw,
        v4_residual_effects=strict_calibrated,
    )

    strict_local = float(strict_metrics["competition_score"])
    report = {
        "protocol": {
            "official_data_only": True,
            "external_tables_or_values_used": False,
            "test_rows_read": False,
            "test_distribution_used": False,
            "row_independent_inference": True,
            "base_residuals": "earlier-season V3 M3 OOF predictions",
            "selection_transfer": "2022 OOF residuals -> 2023 R validation",
            "final_transfer": "2022 + 2023 R OOF residuals -> 2024 validation",
            "exploratory_2024_results_are_not_confirmatory": True,
        },
        "fixed_estimator": {
            "median_offset": FIXED_LB_OFFSET_MEDIAN,
            "target_lb": TARGET_LB,
            "required_local_score": TARGET_LOCAL,
        },
        "features": {name: list(keys) for name, keys in FEATURES.items()},
        "k_grid": list(K_GRID),
        "weight_grid": list(WEIGHT_GRID),
        "baseline": baseline,
        "strict_transfer": {
            "selected_on_2023_r": selected,
            "selection_curve_2023_r": selection_curve,
            "metrics_2024": strict_metrics,
            "expected_lb_median": strict_local + FIXED_LB_OFFSET_MEDIAN,
            "crosses_required_local_score": strict_local > TARGET_LOCAL,
            "prediction_artifact": str(OUTPUT_NPZ.relative_to(ROOT)),
        },
        "exploratory_2024": {
            "warning": "Selected on the 2024 development labels; rank/ceiling diagnostic only.",
            "top_individual": top_individual,
            "greedy_selected": exploratory_selected,
            "greedy_curve": exploratory_curve,
            "greedy_metrics": metrics(seasons[2024].y, exploratory_prediction),
        },
    }
    OUTPUT_JSON.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    print(json.dumps(report["strict_transfer"], ensure_ascii=False, indent=2))
    print(json.dumps(report["exploratory_2024"]["greedy_metrics"], ensure_ascii=False, indent=2))
    print(OUTPUT_JSON)
    print(OUTPUT_NPZ)


if __name__ == "__main__":
    main()
