#!/usr/bin/env python3
"""Robust one-season-ahead residual group effects for M3.

Hyperparameters are ranked by the lower gain across 2021->2022 and
2022->2023.  Selected non-overlapping domain corrections are then refit on
2023 and confirmed once on 2024.  Official train data and OOF predictions are
the only inputs.
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.analyze_v4_temporal_residual_ridge import (  # noqa: E402
    MEDIAN_OFFSET,
    REQUIRED_LOCAL,
    json_safe,
    load_frames,
    score,
)


OUTPUT_JSON = ROOT / "experiments/results/v4_robust_group_effects.json"
OUTPUT_NPZ = ROOT / "experiments/results/predictions/v4_robust_group_effects_2024.npz"
TRANSITIONS = ((2021, 2022), (2022, 2023))
K_GRID = (20.0, 50.0, 100.0, 200.0, 500.0, 1000.0, 3000.0, 10000.0)
GAMMA_GRID = (0.15, 0.25, 0.40, 0.60, 0.85, 1.10, 1.50, 2.00)

EXTRA_COLUMNS = [
    "game_dayofweek",
    "inning",
    "outs_before",
    "score_diff_pitcher_team",
    "num_runners_on",
    "base_state",
    "li",
    "asof_pitcher_n",
    "asof_pitcher_prev3_game_success_rate",
    "asof_batter_n",
    "asof_batter_success_rate",
]

FEATURES: dict[str, tuple[str, ...]] = {
    "global": ("unit",),
    "count": ("count_code",),
    "count_hands": ("count_code", "pitcher_hand", "batter_hand"),
    "inning": ("inning_bucket",),
    "inning_count": ("inning_bucket", "count_code"),
    "outs_count": ("outs_before", "count_code"),
    "base_count": ("base_state", "count_code"),
    "runners_count": ("num_runners_on", "count_code"),
    "score_count": ("score_bucket", "count_code"),
    "li_count": ("li_bucket", "count_code"),
    "month": ("game_month",),
    "month_count": ("game_month", "count_code"),
    "weekday_count": ("game_dayofweek", "count_code"),
    "pitcher": ("pitcher_id",),
    "pitcher_batter_hand": ("pitcher_id", "batter_hand"),
    "pitcher_count": ("pitcher_id", "count_code"),
    "pitcher_pressure_hand": ("pitcher_id", "pressure_state", "batter_hand"),
    "pitcher_inning": ("pitcher_id", "inning_bucket"),
    "pitcher_hand_count": ("pitcher_id", "batter_hand", "count_code"),
    "batter": ("batter_id",),
    "batter_pitcher_hand": ("batter_id", "pitcher_hand"),
    "batter_count": ("batter_id", "count_code"),
    "pitcher_batter": ("pitcher_id", "batter_id"),
    "pitcher_team": ("pitcher_team_id",),
    "batter_team": ("batter_team_id",),
    "team_matchup": ("pitcher_team_id", "batter_team_id"),
    "pitcher_team_count": ("pitcher_team_id", "count_code"),
    "batter_team_count": ("batter_team_id", "count_code"),
    "pitcher_form_count": ("pitcher_form_bin", "count_code"),
    "pitcher_experience_count": ("pitcher_n_bin", "count_code"),
    "batter_form_count": ("batter_form_bin", "count_code"),
}


@dataclass(frozen=True)
class Route:
    name: str
    source_domain: str
    target_domain: str


ROUTES = (
    Route("r_core_from_r", "R", "R_CORE"),
    Route("r_anchor_from_r", "R", "R_ANCHOR"),
    Route("r_anchor_from_anchor", "R_ANCHOR", "R_ANCHOR"),
    Route("f_from_f", "F", "F"),
)


def add_columns(
    frames: dict[int, pd.DataFrame], artifacts: dict[int, dict[str, np.ndarray]]
) -> None:
    full = pd.read_csv(
        ROOT / "open/data/train.csv",
        usecols=EXTRA_COLUMNS,
        encoding="utf-8-sig",
        low_memory=False,
    )
    for season, frame in frames.items():
        selected = full.iloc[np.asarray(artifacts[season]["row_index"], dtype=np.int64)]
        selected = selected.reset_index(drop=True)
        for column in EXTRA_COLUMNS:
            frame[column] = selected[column].to_numpy()
        frame["unit"] = np.int8(1)
        frame["count_code"] = (
            frame["balls_before"].to_numpy(dtype=np.int8) * 3
            + frame["strikes_before"].to_numpy(dtype=np.int8)
        )
        frame["inning_bucket"] = np.minimum(
            frame["inning"].to_numpy(dtype=np.int16), 10
        )
        frame["score_bucket"] = np.clip(
            frame["score_diff_pitcher_team"].to_numpy(dtype=np.float64), -4, 4
        ).astype(np.int8)
        frame["li_bucket"] = np.digitize(
            frame["li"].fillna(1.0).to_numpy(dtype=np.float64),
            np.asarray([0.5, 1.0, 2.0, 4.0]),
        ).astype(np.int8)
        frame["pitcher_form_bin"] = np.clip(
            np.floor(
                frame["asof_pitcher_prev3_game_success_rate"]
                .fillna(0.5)
                .to_numpy(dtype=np.float64)
                * 20.0
            ),
            0,
            20,
        ).astype(np.int8)
        frame["pitcher_n_bin"] = np.floor(
            np.log2(
                1.0
                + frame["asof_pitcher_n"].fillna(0.0).to_numpy(dtype=np.float64)
            )
        ).astype(np.int8)
        frame["batter_form_bin"] = np.clip(
            np.floor(
                frame["asof_batter_success_rate"]
                .fillna(0.5)
                .to_numpy(dtype=np.float64)
                * 20.0
            ),
            0,
            20,
        ).astype(np.int8)


def domain_mask(frame: pd.DataFrame, domain: str) -> np.ndarray:
    if domain == "R":
        return frame["game_type"].eq("R").to_numpy()
    return frame["domain"].eq(domain).to_numpy()


def group_codes(
    source: pd.DataFrame,
    target: pd.DataFrame,
    keys: tuple[str, ...],
) -> tuple[np.ndarray, np.ndarray, int]:
    joined = pd.concat(
        [source.loc[:, list(keys)], target.loc[:, list(keys)]],
        axis=0,
        ignore_index=True,
    )
    if len(keys) == 1:
        codes, uniques = pd.factorize(joined[keys[0]], sort=False, use_na_sentinel=True)
    else:
        index = pd.MultiIndex.from_frame(joined, names=list(keys))
        codes, uniques = pd.factorize(index, sort=False, use_na_sentinel=True)
    codes = np.asarray(codes, dtype=np.int64)
    codes[codes < 0] = len(uniques)
    split = len(source)
    return codes[:split], codes[split:], int(len(uniques) + 1)


def correction_library(
    source_frame: pd.DataFrame,
    source_residual: np.ndarray,
    target_frame: pd.DataFrame,
    keys: tuple[str, ...],
) -> dict[float, np.ndarray]:
    source_codes, target_codes, size = group_codes(source_frame, target_frame, keys)
    counts = np.bincount(source_codes, minlength=size).astype(np.float64)
    sums = np.bincount(
        source_codes, weights=source_residual, minlength=size
    ).astype(np.float64)
    return {k: (sums / (counts + k))[target_codes] for k in K_GRID}


def analytic_gain(
    y: np.ndarray,
    baseline: np.ndarray,
    mask: np.ndarray,
    correction: np.ndarray,
    gamma: float,
) -> float:
    residual = y[mask] - baseline[mask]
    n = float(len(y))
    brier_improvement = (
        2.0 * gamma * float(np.dot(residual, correction))
        - gamma * gamma * float(np.dot(correction, correction))
    ) / n
    rate = float(np.mean(y))
    reference = rate * (1.0 - rate)
    return 100_000.0 * brier_improvement / reference


def build_corrections(
    frames: dict[int, pd.DataFrame],
    artifacts: dict[int, dict[str, np.ndarray]],
    source_year: int,
    target_year: int,
    route: Route,
    keys: tuple[str, ...],
) -> tuple[np.ndarray, dict[float, np.ndarray]]:
    source_mask = domain_mask(frames[source_year], route.source_domain)
    target_mask = domain_mask(frames[target_year], route.target_domain)
    source = frames[source_year].loc[source_mask].reset_index(drop=True)
    target = frames[target_year].loc[target_mask].reset_index(drop=True)
    residual = (
        np.asarray(artifacts[source_year]["y"], dtype=np.float64)
        - np.asarray(artifacts[source_year]["m3"], dtype=np.float64)
    )[source_mask]
    return target_mask, correction_library(source, residual, target, keys)


def apply_candidate(
    baseline: np.ndarray,
    mask: np.ndarray,
    correction: np.ndarray,
    gamma: float,
) -> np.ndarray:
    result = np.asarray(baseline, dtype=np.float64).copy()
    result[mask] = np.clip(result[mask] + gamma * correction, 0.0, 1.0)
    return result


def main() -> None:
    frames, artifacts = load_frames()
    add_columns(frames, artifacts)
    baselines = {
        season: score(artifacts[season]["y"], artifacts[season]["m3"])
        for season in (2021, 2022, 2023, 2024)
    }
    trials: list[dict[str, Any]] = []
    best_by_route: dict[str, dict[str, Any]] = {}

    for route in ROUTES:
        for feature, keys in FEATURES.items():
            libraries = {
                transition: build_corrections(
                    frames, artifacts, transition[0], transition[1], route, keys
                )
                for transition in TRANSITIONS
            }
            for k in K_GRID:
                for gamma in GAMMA_GRID:
                    gains: dict[str, float] = {}
                    for source_year, target_year in TRANSITIONS:
                        target_mask, library = libraries[(source_year, target_year)]
                        gains[f"{source_year}_to_{target_year}"] = analytic_gain(
                            np.asarray(artifacts[target_year]["y"], dtype=np.float64),
                            np.asarray(artifacts[target_year]["m3"], dtype=np.float64),
                            target_mask,
                            library[k],
                            gamma,
                        )
                    row = {
                        "route": route.name,
                        "source_domain": route.source_domain,
                        "target_domain": route.target_domain,
                        "feature": feature,
                        "keys": keys,
                        "k": k,
                        "gamma": gamma,
                        "gains": gains,
                        "robust_min_gain": float(min(gains.values())),
                        "mean_gain": float(np.mean(list(gains.values()))),
                    }
                    trials.append(row)
                    previous = best_by_route.get(route.name)
                    rank = (row["robust_min_gain"], row["mean_gain"])
                    if previous is None or rank > (
                        previous["robust_min_gain"], previous["mean_gain"]
                    ):
                        best_by_route[route.name] = row
        best = best_by_route[route.name]
        print(
            f"[{route.name}] {best['feature']} k={best['k']:.0f} "
            f"gamma={best['gamma']:.2f} min={best['robust_min_gain']:+.4f} "
            f"mean={best['mean_gain']:+.4f}",
            flush=True,
        )

    route_predictions: dict[str, np.ndarray] = {}
    route_masks: dict[str, np.ndarray] = {}
    route_confirmations: dict[str, Any] = {}
    for route in ROUTES:
        selected = best_by_route[route.name]
        target_mask, library = build_corrections(
            frames,
            artifacts,
            2023,
            2024,
            route,
            FEATURES[selected["feature"]],
        )
        prediction = apply_candidate(
            artifacts[2024]["m3"],
            target_mask,
            library[float(selected["k"])],
            float(selected["gamma"]),
        )
        metrics = score(artifacts[2024]["y"], prediction)
        gain = float(
            metrics["raw_competition_score"]
            - baselines[2024]["raw_competition_score"]
        )
        route_predictions[route.name] = prediction
        route_masks[route.name] = target_mask
        route_confirmations[route.name] = {
            "selected": selected,
            "metrics": metrics,
            "gain": gain,
        }
        print(
            f"[confirm {route.name}] gain={gain:+.4f} "
            f"local={metrics['raw_competition_score']:.4f}",
            flush=True,
        )

    # Choose at most one source-table recipe per target domain using selection
    # evidence only.  This prevents the two R_ANCHOR routes from overwriting
    # each other based on their later 2024 confirmation.
    selected_route_names: list[str] = []
    for target_domain in ("R_CORE", "R_ANCHOR", "F"):
        eligible = [
            route
            for route in ROUTES
            if route.target_domain == target_domain
            and best_by_route[route.name]["robust_min_gain"] > 0.0
        ]
        if eligible:
            chosen = max(
                eligible,
                key=lambda route: (
                    best_by_route[route.name]["robust_min_gain"],
                    best_by_route[route.name]["mean_gain"],
                ),
            )
            selected_route_names.append(chosen.name)
    combined = np.asarray(artifacts[2024]["m3"], dtype=np.float64).copy()
    for route_name in selected_route_names:
        mask = route_masks[route_name]
        combined[mask] = route_predictions[route_name][mask]
    combined_routes = selected_route_names
    combined_metrics = score(artifacts[2024]["y"], combined)
    combined_gain = float(
        combined_metrics["raw_competition_score"]
        - baselines[2024]["raw_competition_score"]
    )
    payload: dict[str, np.ndarray] = {
        "y": artifacts[2024]["y"],
        "row_index": artifacts[2024]["row_index"],
        "cluster": artifacts[2024]["cluster"],
        "m3": artifacts[2024]["m3"],
        "robust_group_effects": combined,
    }
    for route_name, prediction in route_predictions.items():
        payload[f"route_{route_name}"] = prediction
    np.savez_compressed(OUTPUT_NPZ, **payload)

    report = {
        "protocol": {
            "official_train_only": True,
            "test_rows_read": False,
            "leaderboard_values_used": False,
            "row_independent_target_lookup": True,
            "selection": "maximize worst gain over 2021->2022 and 2022->2023",
            "confirmation": "refit selected group table on 2023 and transfer to 2024",
            "combination_rule": "combine non-overlapping routes only when both selection gains are positive",
        },
        "fixed_estimator": {
            "median_offset": MEDIAN_OFFSET,
            "target_lb": 1190.0,
            "required_local_score": REQUIRED_LOCAL,
        },
        "baselines": baselines,
        "trial_count": len(trials),
        "best_by_route": best_by_route,
        "route_confirmations_2024": route_confirmations,
        "combined_2024": {
            "routes": combined_routes,
            "metrics": combined_metrics,
            "gain": combined_gain,
            "expected_lb_median": float(
                combined_metrics["raw_competition_score"] + MEDIAN_OFFSET
            ),
            "crosses_required_local_score": bool(
                combined_metrics["raw_competition_score"] > REQUIRED_LOCAL
            ),
        },
        "top_trials": sorted(
            trials,
            key=lambda row: (row["robust_min_gain"], row["mean_gain"]),
            reverse=True,
        )[:50],
        "prediction_artifact": str(OUTPUT_NPZ.relative_to(ROOT)),
    }
    OUTPUT_JSON.write_text(
        json.dumps(json_safe(report), ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "combined_routes": combined_routes,
                "combined_gain_2024": combined_gain,
                "combined_score_2024": combined_metrics["raw_competition_score"],
                "expected_lb_median": combined_metrics["raw_competition_score"]
                + MEDIAN_OFFSET,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    print(f"Saved {OUTPUT_JSON}")
    print(f"Saved {OUTPUT_NPZ}")


if __name__ == "__main__":
    main()
