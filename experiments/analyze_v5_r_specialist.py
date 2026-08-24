#!/usr/bin/env python3
"""Select and confirm the preregistered regular-season outcome specialist."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.run_v5_h1_residual import (  # noqa: E402
    cluster_bootstrap_score_gain,
    metrics,
)


PRED = ROOT / "experiments/results/predictions"
TRAIN = ROOT / "open/data/train.csv"
PREREG = ROOT / "experiments/params/v5_r_specialist_preregister.json"
CONTRACT = ROOT / "experiments/params/v5_validation_contract_v2.json"
SELECTION = ROOT / "experiments/results/v5_r_specialist_selection.json"
CONFIRMATION = ROOT / "experiments/results/v5_r_specialist_confirmation.json"
CANDIDATE_KEY = "catboost_outcome"
V3_ACTUAL_LB = 1090.9100565103
BASELINES = {
    "exact_parent_C": {
        2022: ("v3_sparse_c_backtest", "catboost_outcome"),
        2023: ("v3_sparse_c_backtest", "catboost_outcome"),
        2024: (
            "v3_outcome_trackmanrich_overall_e14k50_batter80_middle100",
            "catboost_outcome",
        ),
    },
    "honest_r_identity": {
        year: ("v5_honest_m3_r_identity", "final_prediction")
        for year in (2022, 2023, 2024)
    },
    "honest_r_grid": {
        year: ("v5_honest_m3_r_grid", "final_prediction")
        for year in (2022, 2023, 2024)
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("select", "confirm"), required=True)
    parser.add_argument("--bootstrap", type=int, default=1000)
    return parser.parse_args()


def load(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        return {key: np.asarray(archive[key]) for key in archive.files}


def load_candidate(year: int, stem: str) -> dict[str, np.ndarray]:
    artifact = load(PRED / f"{stem}_{year}.npz")
    if CANDIDATE_KEY not in artifact:
        raise KeyError(f"{stem}_{year}.npz has no {CANDIDATE_KEY}")
    return artifact


def baseline_prediction(
    baseline_name: str, year: int, candidate: dict[str, np.ndarray]
) -> np.ndarray:
    stem, key = BASELINES[baseline_name][year]
    artifact = load(PRED / f"{stem}_{year}.npz")
    for align_key in ("y", "row_index", "cluster"):
        if not np.array_equal(candidate[align_key], artifact[align_key]):
            raise ValueError(
                f"alignment mismatch {baseline_name}/{year}/{align_key}"
            )
    return artifact[key].astype(np.float64)


def evaluate(
    candidate: dict[str, np.ndarray],
    baseline: np.ndarray,
    all_types: np.ndarray,
    gamma: float,
    bootstrap: int,
    seed: int,
) -> dict:
    types = all_types[candidate["row_index"].astype(np.int64)]
    route = types == "R"
    prediction = baseline.copy()
    prediction[route] += gamma * (
        candidate[CANDIDATE_KEY][route].astype(np.float64) - baseline[route]
    )
    prediction = np.clip(prediction, 1e-6, 1.0 - 1e-6)
    y = candidate["y"].astype(np.int8)
    cluster = candidate["cluster"].astype(str)
    return {
        "metrics": metrics(y, baseline, prediction, types),
        "bootstrap_R": cluster_bootstrap_score_gain(
            y, baseline, prediction, cluster, route, bootstrap, seed
        ),
        "bootstrap_all": cluster_bootstrap_score_gain(
            y,
            baseline,
            prediction,
            cluster,
            np.ones(len(y), dtype=bool),
            bootstrap,
            seed + 10000,
        ),
        "prediction": prediction,
    }


def select(args: argparse.Namespace, prereg: dict, all_types: np.ndarray) -> None:
    if SELECTION.exists():
        raise FileExistsError(f"immutable selection already exists: {SELECTION}")
    trials: list[dict] = []
    variants = prereg["model"]["variants"]
    for variant_index, (variant_name, variant) in enumerate(sorted(variants.items())):
        candidates = {
            year: load_candidate(year, variant["development_stem"])
            for year in (2022, 2023)
        }
        for gamma in prereg["selection"]["gamma_grid"]:
            cells: dict[str, dict[str, dict]] = {}
            gains: list[float] = []
            ci_lows: list[float] = []
            for baseline_index, baseline_name in enumerate(BASELINES):
                cells[baseline_name] = {}
                for year in (2022, 2023):
                    candidate = candidates[year]
                    result = evaluate(
                        candidate,
                        baseline_prediction(baseline_name, year, candidate),
                        all_types,
                        float(gamma),
                        args.bootstrap,
                        810000
                        + 10000 * variant_index
                        + 1000 * baseline_index
                        + year
                        + int(round(float(gamma) * 100)),
                    )
                    result.pop("prediction")
                    cells[baseline_name][str(year)] = result
                    gains.append(float(result["metrics"]["R"]["gain"]))
                    ci_lows.append(float(result["bootstrap_R"]["ci_low"]))
            trials.append(
                {
                    "variant": variant_name,
                    "development_stem": variant["development_stem"],
                    "gamma": float(gamma),
                    "minimum_R_gain": float(min(gains)),
                    "median_R_gain": float(np.median(gains)),
                    "minimum_R_ci_low": float(min(ci_lows)),
                    "cells": cells,
                    "eligible": bool(min(gains) > 0.0 and min(ci_lows) > 0.0),
                }
            )
    eligible = [trial for trial in trials if trial["eligible"]]
    selected = (
        sorted(
            eligible,
            key=lambda trial: (
                -float(trial["minimum_R_gain"]),
                float(trial["gamma"]),
                str(trial["variant"]),
            ),
        )[0]
        if eligible
        else None
    )
    report = {
        "experiment_id": prereg["experiment_id"],
        "mode": "development_selection",
        "preregister_sha256": hashlib.sha256(PREREG.read_bytes()).hexdigest(),
        "contract_sha256": hashlib.sha256(CONTRACT.read_bytes()).hexdigest(),
        "years_read": [2022, 2023],
        "confirmation_year_read": False,
        "trials": trials,
        "selected": selected,
        "status": "locked" if selected is not None else "failed_no_eligible_candidate",
    }
    SELECTION.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    print(json.dumps({"status": report["status"], "selected": selected}, indent=2))
    print(f"Saved {SELECTION}")


def confirm(args: argparse.Namespace, prereg: dict, all_types: np.ndarray) -> None:
    if CONFIRMATION.exists():
        raise FileExistsError(f"immutable confirmation already exists: {CONFIRMATION}")
    selection = json.loads(SELECTION.read_text(encoding="utf-8"))
    prereg_hash = hashlib.sha256(PREREG.read_bytes()).hexdigest()
    contract_hash = hashlib.sha256(CONTRACT.read_bytes()).hexdigest()
    if selection.get("status") != "locked" or selection.get("selected") is None:
        raise ValueError("no eligible locked selection")
    if selection.get("preregister_sha256") != prereg_hash:
        raise ValueError("preregister changed after selection")
    if selection.get("contract_sha256") != contract_hash:
        raise ValueError("validation contract changed after selection")
    selected = selection["selected"]
    variant = prereg["model"]["variants"][selected["variant"]]
    candidate = load_candidate(2024, variant["confirmation_stem"])
    gamma = float(selected["gamma"])
    cells: dict[str, dict] = {}
    saved_predictions: dict[str, np.ndarray] = {}
    for baseline_index, baseline_name in enumerate(BASELINES):
        result = evaluate(
            candidate,
            baseline_prediction(baseline_name, 2024, candidate),
            all_types,
            gamma,
            args.bootstrap,
            910000 + 1000 * baseline_index + 2024,
        )
        saved_predictions[baseline_name] = result.pop("prediction")
        cells[baseline_name] = result
    development_full = [
        float(result["metrics"]["all"]["gain"])
        for baseline_cells in selected["cells"].values()
        for result in baseline_cells.values()
    ]
    confirm_full = [float(result["metrics"]["all"]["gain"]) for result in cells.values()]
    confirm_ci = [float(result["bootstrap_all"]["ci_low"]) for result in cells.values()]
    g_dev = min(development_full)
    g_confirm = min(confirm_full)
    g_ci = min(confirm_ci)
    g_robust = min(g_dev, g_confirm, g_ci)
    expected = V3_ACTUAL_LB + 0.75 * max(0.0, g_robust)
    report = {
        "experiment_id": prereg["experiment_id"],
        "mode": "locked_confirmation",
        "preregister_sha256": prereg_hash,
        "contract_sha256": contract_hash,
        "selected_variant": selected["variant"],
        "confirmation_stem": variant["confirmation_stem"],
        "gamma": gamma,
        "development": selected,
        "confirmation_2024": cells,
        "conservative_expected_score": {
            "actual_v3_anchor": V3_ACTUAL_LB,
            "G_dev_full_min": g_dev,
            "G_confirm_full_min": g_confirm,
            "G_ci_full_min": g_ci,
            "G_robust": g_robust,
            "haircut": 0.75,
            "expected_lb_lower": expected,
            "passes_1190": bool(expected > 1190.0),
        },
    }
    CONFIRMATION.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    np.savez_compressed(
        PRED / "v5_r_specialist_locked_2024.npz",
        y=candidate["y"],
        row_index=candidate["row_index"],
        cluster=candidate["cluster"],
        r_specialist=candidate[CANDIDATE_KEY],
        **{f"final_{name}": value for name, value in saved_predictions.items()},
    )
    print(json.dumps(report["conservative_expected_score"], indent=2))
    print(f"Saved {CONFIRMATION}")


def main() -> None:
    args = parse_args()
    prereg = json.loads(PREREG.read_text(encoding="utf-8"))
    all_types = pd.read_csv(TRAIN, usecols=["game_type"])["game_type"].to_numpy()
    if args.mode == "select":
        select(args, prereg, all_types)
    else:
        confirm(args, prereg, all_types)


if __name__ == "__main__":
    main()
