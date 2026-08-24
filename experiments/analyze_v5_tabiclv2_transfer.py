#!/usr/bin/env python3
"""Source-fold-only selection for the preregistered TabICLv2 blend."""

from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.stats import paired_bootstrap_brier_ci  # noqa: E402


RESULTS = ROOT / "experiments" / "results"
PREDICTIONS = RESULTS / "predictions"
PREREG = ROOT / "experiments" / "params" / "v5_tabiclv2_transfer_preregister.json"
TABICL_STAGE = "v5_tabiclv2_source2020_2021"
C_STAGES = {
    2020: "v4_m3_c_backtest_2020_2020",
    2021: "v4_m3_c_backtest_2021_2021",
}
REPORT = RESULTS / "v5_tabiclv2_source_selection.json"


def competition_score(y: np.ndarray, prediction: np.ndarray) -> float:
    y = np.asarray(y, dtype=np.float64)
    prediction = np.asarray(prediction, dtype=np.float64)
    reference = float(y.mean() * (1.0 - y.mean()))
    brier = float(np.mean(np.square(prediction - y)))
    return max(0.0, 100_000.0 * (1.0 - brier / reference))


def load_fold(year: int, game_types: pd.Series, centers: dict[int, float]) -> dict:
    tabicl_path = PREDICTIONS / f"{TABICL_STAGE}_{year}.npz"
    c_path = PREDICTIONS / f"{C_STAGES[year]}.npz"
    with np.load(tabicl_path) as tab, np.load(c_path) as parent:
        for key in ("y", "row_index", "cluster"):
            if not np.array_equal(tab[key], parent[key]):
                raise ValueError(f"{year}: TabICL/exact-C mismatch for {key}")
        row_index = tab["row_index"].astype(np.int64)
        return {
            "y": tab["y"].astype(np.float64),
            "row_index": row_index,
            "cluster": tab["cluster"].astype(str),
            "tabicl": tab["tabicl"].astype(np.float64),
            "parent": parent["catboost_outcome"].astype(np.float64),
            "regular": game_types.iloc[row_index].astype(str).eq("R").to_numpy(),
            "mu": centers[year],
        }


def main() -> None:
    prereg = json.loads(PREREG.read_text(encoding="utf-8"))
    stage_report = json.loads(
        (RESULTS / f"{TABICL_STAGE}.json").read_text(encoding="utf-8")
    )
    centers = {
        int(fold["validation_season"]): float(
            fold["fit_details"]["tabicl"]["context_target_mean"]
        )
        for fold in stage_report["folds"]
    }
    game_types = pd.read_csv(
        ROOT / "open" / "data" / "train.csv", usecols=["game_type"]
    )["game_type"]
    years = list(prereg["fold_protocol"]["source_selection_years"])
    folds = {year: load_fold(year, game_types, centers) for year in years}

    shrink_grid = prereg["source_only_selection"]["shrink_grid"]
    alpha_grid = prereg["source_only_selection"]["alpha_grid"]
    candidates: list[dict] = []
    for shrink in shrink_grid:
        for alpha in alpha_grid:
            year_metrics: dict[str, dict] = {}
            for year, fold in folds.items():
                calibrated = np.clip(
                    fold["mu"] + shrink * (fold["tabicl"] - fold["mu"]),
                    1e-6,
                    1.0 - 1e-6,
                )
                candidate = (1.0 - alpha) * fold["parent"] + alpha * calibrated
                regular = fold["regular"]
                full_gain = competition_score(fold["y"], candidate) - competition_score(
                    fold["y"], fold["parent"]
                )
                r_gain = competition_score(
                    fold["y"][regular], candidate[regular]
                ) - competition_score(
                    fold["y"][regular], fold["parent"][regular]
                )
                year_metrics[str(year)] = {
                    "full_gain": float(full_gain),
                    "r_gain": float(r_gain),
                    "candidate_score": competition_score(fold["y"], candidate),
                    "parent_score": competition_score(fold["y"], fold["parent"]),
                    "candidate_r_score": competition_score(
                        fold["y"][regular], candidate[regular]
                    ),
                    "parent_r_score": competition_score(
                        fold["y"][regular], fold["parent"][regular]
                    ),
                    "candidate_mean": float(candidate.mean()),
                    "candidate_std": float(candidate.std()),
                }
            min_full = min(value["full_gain"] for value in year_metrics.values())
            min_r = min(value["r_gain"] for value in year_metrics.values())
            mean_full = float(
                np.mean([value["full_gain"] for value in year_metrics.values()])
            )
            candidates.append(
                {
                    "shrink": float(shrink),
                    "alpha": float(alpha),
                    "min_full_gain": float(min_full),
                    "min_r_gain": float(min_r),
                    "mean_full_gain": mean_full,
                    "years": year_metrics,
                }
            )

    selected = max(
        candidates,
        key=lambda item: (
            item["min_full_gain"],
            item["min_r_gain"],
            item["mean_full_gain"],
            -item["alpha"],
            -item["shrink"],
        ),
    )
    intervals: dict[str, dict] = {}
    for year, fold in folds.items():
        calibrated = np.clip(
            fold["mu"]
            + selected["shrink"] * (fold["tabicl"] - fold["mu"]),
            1e-6,
            1.0 - 1e-6,
        )
        candidate = (
            (1.0 - selected["alpha"]) * fold["parent"]
            + selected["alpha"] * calibrated
        )
        regular = fold["regular"]
        intervals[str(year)] = {
            "full": paired_bootstrap_brier_ci(
                fold["y"],
                fold["parent"],
                candidate,
                iterations=2_000,
                seed=2026 + year,
                clusters=fold["cluster"],
            ),
            "R": paired_bootstrap_brier_ci(
                fold["y"][regular],
                fold["parent"][regular],
                candidate[regular],
                iterations=2_000,
                seed=4026 + year,
                clusters=fold["cluster"][regular],
            ),
        }

    threshold = float(
        prereg["source_only_selection"]["source_open_gate"]
        ["minimum_of_full_and_r_point_gains"]
    )
    all_point_positive = all(
        selected["years"][str(year)][scope + "_gain"] > 0.0
        for year in years
        for scope in ("full", "r")
    )
    all_ci_positive = all(
        intervals[str(year)][scope]["score_ci_low"] > 0.0
        for year in years
        for scope in ("full", "R")
    )
    minimum_point = min(selected["min_full_gain"], selected["min_r_gain"])
    passed = bool(
        all_point_positive and all_ci_positive and minimum_point >= threshold
    )
    report = {
        "experiment_id": prereg["experiment_id"],
        "selection_data": years,
        "confirmation_data_opened": [],
        "status": "source_gate_passed_locked_before_2022" if passed else "failed_source_gate",
        "selected": selected,
        "selected_intervals": intervals,
        "gate": {
            "all_full_and_r_point_gains_positive": all_point_positive,
            "all_full_and_r_ci_lowers_positive": all_ci_positive,
            "minimum_point_gain": minimum_point,
            "required_minimum_point_gain": threshold,
            "passed": passed,
        },
        "candidate_count": len(candidates),
        "candidates": candidates,
        "recipe_locked_before_2022": passed,
    }
    REPORT.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({key: value for key, value in report.items() if key != "candidates"}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
