#!/usr/bin/env python3
"""Source screen for coarse numeric cohorts in official player IDs."""

from __future__ import annotations

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
from experiments.run_v5_h1_residual import cluster_bootstrap_score_gain  # noqa: E402

TRAIN = ROOT / "open/data/train.csv"
PREREG = ROOT / "experiments/params/v5_main_id_cohort_source_preregister.json"
REPORT = ROOT / "experiments/results/v5_main_id_cohort_source.json"
YEARS = (2020, 2021)
TARGET = "control_success"


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [safe(item) for item in value]
    if isinstance(value, (np.floating, float)):
        number = float(value)
        return number if np.isfinite(number) else None
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    return value


def metric(y: np.ndarray, prediction: np.ndarray, mask: np.ndarray) -> float:
    target = y[mask].astype(np.float64)
    estimate = prediction[mask].astype(np.float64)
    rate = float(target.mean())
    return float(
        100000.0
        * (1.0 - np.mean(np.square(estimate - target)) / max(rate * (1.0 - rate), 1e-12))
    )


def load_frame() -> pd.DataFrame:
    anchor = load_anchor(max(YEARS))
    frame = pd.read_csv(
        TRAIN,
        usecols=[
            "season", "game_type", "pitcher_id", "batter_id",
            "asof_pitcher_n", "asof_pitcher_success_rate",
            "asof_batter_n", "asof_batter_success_rate", TARGET,
        ],
        nrows=int(np.max(anchor["row_index"])) + 1,
        encoding="utf-8-sig",
    )
    if set(int(value) for value in frame["season"].unique()) != {2019, 2020, 2021}:
        raise ValueError("read a season after 2021")
    return frame


def cohort_direction(
    history: pd.DataFrame,
    valid: pd.DataFrame,
    entity: str,
    width: int,
    eb_k: float,
    cold_k: float,
) -> np.ndarray:
    history_r = history.loc[history["game_type"].eq("R")].copy()
    prior = float(history_r[TARGET].mean())

    def one(kind: str) -> np.ndarray:
        id_column = f"{kind}_id"
        n_column = f"asof_{kind}_n"
        rate_column = f"asof_{kind}_success_rate"
        bins = np.floor_divide(history_r[id_column].to_numpy(dtype=np.int64), width)
        table = history_r.assign(_cohort=bins).groupby(
            "_cohort", sort=False, observed=True
        )[TARGET].agg(["sum", "count"])
        posterior = (table["sum"] + eb_k * prior) / (table["count"] + eb_k)
        valid_bins = np.floor_divide(valid[id_column].to_numpy(dtype=np.int64), width)
        cohort = pd.Series(valid_bins).map(posterior).fillna(prior).to_numpy(float)
        career = pd.to_numeric(valid[rate_column], errors="coerce").fillna(prior).to_numpy(float)
        n = pd.to_numeric(valid[n_column], errors="coerce").fillna(0.0).to_numpy(float)
        return (cohort - career) * (cold_k / (np.maximum(n, 0.0) + cold_k))

    pitcher = one("pitcher")
    batter = one("batter")
    if entity == "pitcher":
        return pitcher
    if entity == "batter":
        return batter
    if entity == "equal_mean":
        return 0.5 * (pitcher + batter)
    raise ValueError(entity)


def main() -> None:
    if REPORT.exists():
        raise FileExistsError(f"immutable report exists: {REPORT}")
    prereg = json.loads(PREREG.read_text(encoding="utf-8"))
    if prereg["status"] != "locked_before_source_metrics":
        raise ValueError("unexpected preregistration state")
    started = time.perf_counter()
    frame = load_frame()
    folds: dict[int, dict[str, Any]] = {}
    directions: dict[tuple[int, str, int, int, int], np.ndarray] = {}
    spec = prereg["candidate"]
    for year in YEARS:
        anchor = load_anchor(year)
        valid = frame.iloc[anchor["row_index"]].copy()
        if not valid["season"].eq(year).all():
            raise ValueError(f"{year}: anchor mismatch")
        y = anchor["y"].astype(np.int8)
        if not np.array_equal(valid[TARGET].to_numpy(dtype=np.int8), y):
            raise ValueError(f"{year}: target mismatch")
        folds[year] = {
            "y": y,
            "base": anchor["catboost_outcome"].astype(np.float64),
            "cluster": anchor["cluster"],
            "game_type": valid["game_type"].astype(str).to_numpy(),
        }
        history = frame.loc[frame["season"].lt(year)]
        for entity in spec["entities"]:
            for width in spec["bin_width"]:
                for eb_k in spec["cohort_eb_k"]:
                    for cold_k in spec["cold_reliability_k"]:
                        directions[(year, entity, int(width), int(eb_k), int(cold_k))] = (
                            cohort_direction(
                                history, valid, entity, int(width), float(eb_k), float(cold_k)
                            )
                        )

    candidates: list[dict[str, Any]] = []
    for entity in spec["entities"]:
        for width in spec["bin_width"]:
            for eb_k in spec["cohort_eb_k"]:
                for cold_k in spec["cold_reliability_k"]:
                    for gamma_value in spec["gamma"]:
                        gamma = float(gamma_value)
                        years: dict[str, Any] = {}
                        for year in YEARS:
                            fold = folds[year]
                            route = fold["game_type"] == "R"
                            direction = directions[
                                (year, entity, int(width), int(eb_k), int(cold_k))
                            ]
                            prediction = fold["base"].copy()
                            prediction[route] = np.clip(
                                prediction[route] + gamma * direction[route],
                                1e-6,
                                1.0 - 1e-6,
                            )
                            all_mask = np.ones(len(route), dtype=bool)
                            years[str(year)] = {
                                "full_gain": metric(fold["y"], prediction, all_mask)
                                - metric(fold["y"], fold["base"], all_mask),
                                "R_gain": metric(fold["y"], prediction, route)
                                - metric(fold["y"], fold["base"], route),
                                "direction_mean_R": float(direction[route].mean()),
                                "direction_std_R": float(direction[route].std()),
                            }
                        candidates.append(
                            {
                                "entity": entity,
                                "bin_width": int(width),
                                "cohort_eb_k": int(eb_k),
                                "cold_reliability_k": int(cold_k),
                                "gamma": gamma,
                                "minimum_full_gain": float(
                                    min(years[str(year)]["full_gain"] for year in YEARS)
                                ),
                                "minimum_R_gain": float(
                                    min(years[str(year)]["R_gain"] for year in YEARS)
                                ),
                                "mean_full_gain": float(
                                    np.mean([years[str(year)]["full_gain"] for year in YEARS])
                                ),
                                "years": years,
                            }
                        )
    candidates.sort(
        key=lambda row: (
            row["minimum_full_gain"],
            row["minimum_R_gain"],
            row["mean_full_gain"],
            -row["gamma"],
            row["cohort_eb_k"],
            row["bin_width"],
        ),
        reverse=True,
    )
    selected = candidates[0]
    intervals: dict[str, Any] = {}
    for offset, year in enumerate(YEARS):
        fold = folds[year]
        route = fold["game_type"] == "R"
        direction = directions[
            (
                year,
                selected["entity"],
                selected["bin_width"],
                selected["cohort_eb_k"],
                selected["cold_reliability_k"],
            )
        ]
        prediction = fold["base"].copy()
        prediction[route] = np.clip(
            prediction[route] + selected["gamma"] * direction[route],
            1e-6,
            1.0 - 1e-6,
        )
        intervals[str(year)] = {
            "full": cluster_bootstrap_score_gain(
                fold["y"], fold["base"], prediction, fold["cluster"],
                np.ones(len(route), dtype=bool), 2000, 62100 + offset,
            ),
            "R": cluster_bootstrap_score_gain(
                fold["y"], fold["base"], prediction, fold["cluster"],
                route, 2000, 62200 + offset,
            ),
        }
    gate = prereg["source_gate"]
    conditions = {
        "minimum_full_gain": selected["minimum_full_gain"]
        >= float(gate["minimum_full_gain_each_year"]),
        "minimum_R_gain": selected["minimum_R_gain"]
        >= float(gate["minimum_R_gain_each_year"]),
        "full_ci_lower_positive": all(
            intervals[str(year)]["full"]["ci_low"] > 0.0 for year in YEARS
        ),
        "R_ci_lower_positive": all(
            intervals[str(year)]["R"]["ci_low"] > 0.0 for year in YEARS
        ),
    }
    passed = bool(all(conditions.values()))
    report = {
        "experiment_id": prereg["experiment_id"],
        "status": "source_pass" if passed else "source_failed_closed",
        "preregister_sha256": digest(PREREG),
        "script_sha256": digest(Path(__file__)),
        "train_sha256": digest(TRAIN),
        "years_read": list(YEARS),
        "years_not_read": [2022, 2023, 2024],
        "candidate_count": len(candidates),
        "selected": selected,
        "intervals": intervals,
        "conditions": conditions,
        "source_gate_pass": passed,
        "top_candidates": candidates[:20],
        "decision": (
            "freeze before development" if passed
            else "close numeric main-player-ID cohort axis without 2022+ labels"
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
