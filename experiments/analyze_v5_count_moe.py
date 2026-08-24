#!/usr/bin/env python3
"""Immutable early-gate, selection, and confirmation for count-routed MoE."""

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
PREREG = ROOT / "experiments/params/v5_count_moe_preregister.json"
CONTRACT = ROOT / "experiments/params/v5_validation_contract_v2.json"
EARLY = ROOT / "experiments/results/v5_count_moe_2022_gate.json"
SELECTION = ROOT / "experiments/results/v5_count_moe_selection.json"
CONFIRMATION = ROOT / "experiments/results/v5_count_moe_confirmation.json"
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
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("gate2022", "select", "confirm"), required=True)
    parser.add_argument("--bootstrap", type=int, default=1000)
    return parser.parse_args()


def load(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        return {key: np.asarray(archive[key]) for key in archive.files}


def file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_candidate(year: int, prereg: dict) -> dict[str, np.ndarray]:
    stem = prereg["stages"][str(year)]
    candidate = load(PRED / f"{stem}_{year}.npz")
    key = prereg["candidate_key"]
    if key not in candidate:
        raise KeyError(f"{stem}_{year}.npz has no {key}")
    return candidate


def load_baseline(
    name: str, year: int, candidate: dict[str, np.ndarray]
) -> np.ndarray:
    stem, key = BASELINES[name][year]
    artifact = load(PRED / f"{stem}_{year}.npz")
    for align_key in ("y", "row_index", "cluster"):
        if not np.array_equal(candidate[align_key], artifact[align_key]):
            raise ValueError(f"alignment mismatch {name}/{year}/{align_key}")
    return np.asarray(artifact[key], dtype=np.float64)


def evaluate(
    candidate: dict[str, np.ndarray],
    baseline: np.ndarray,
    types: np.ndarray,
    key: str,
    gamma: float,
    bootstrap: int,
    seed: int,
) -> dict:
    regular = types == "R"
    prediction = np.asarray(baseline, dtype=np.float64).copy()
    direction = np.asarray(candidate[key], dtype=np.float64) - baseline
    prediction[regular] += gamma * direction[regular]
    prediction = np.clip(prediction, 1e-6, 1.0 - 1e-6)
    y = candidate["y"].astype(np.int8)
    cluster = candidate["cluster"].astype(str)
    return {
        "gamma": gamma,
        "metrics": metrics(y, baseline, prediction, types),
        "bootstrap_R": cluster_bootstrap_score_gain(
            y, baseline, prediction, cluster, regular, bootstrap, seed
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


def strip_prediction(result: dict) -> dict:
    return {key: value for key, value in result.items() if key != "prediction"}


def all_types() -> np.ndarray:
    return pd.read_csv(TRAIN, usecols=["game_type"], low_memory=False)[
        "game_type"
    ].astype(str).to_numpy()


def cells_for_year(
    year: int,
    prereg: dict,
    gamma: float,
    types_all: np.ndarray,
    bootstrap: int,
) -> dict[str, dict]:
    candidate = load_candidate(year, prereg)
    types = types_all[candidate["row_index"].astype(np.int64)]
    cells = {}
    for baseline_index, name in enumerate(BASELINES):
        baseline = load_baseline(name, year, candidate)
        result = evaluate(
            candidate,
            baseline,
            types,
            prereg["candidate_key"],
            gamma,
            bootstrap,
            820000 + 10000 * year + 1000 * baseline_index + int(gamma * 100),
        )
        cells[name] = strip_prediction(result)
    return cells


def write_immutable(path: Path, report: dict) -> None:
    if path.exists():
        raise FileExistsError(f"immutable report already exists: {path}")
    path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )


def gate2022(args: argparse.Namespace, prereg: dict, types_all: np.ndarray) -> None:
    trials = []
    required = float(
        prereg["selection"]["early_2022_required_full_point_gain"]
    )
    for gamma in prereg["selection"]["gamma_grid"]:
        cells = cells_for_year(2022, prereg, float(gamma), types_all, args.bootstrap)
        full_gains = [float(cell["metrics"]["all"]["gain"]) for cell in cells.values()]
        r_gains = [float(cell["metrics"]["R"]["gain"]) for cell in cells.values()]
        ci_lows = [float(cell["bootstrap_R"]["ci_low"]) for cell in cells.values()]
        trials.append({
            "gamma": float(gamma),
            "cells": cells,
            "minimum_full_gain": min(full_gains),
            "minimum_R_gain": min(r_gains),
            "minimum_R_ci_low": min(ci_lows),
            "goal_scale_eligible": bool(
                min(full_gains) > required
                and min(r_gains) > 0.0
                and min(ci_lows) > 0.0
            ),
        })
    eligible = [trial for trial in trials if trial["goal_scale_eligible"]]
    report = {
        "experiment_id": prereg["experiment_id"],
        "mode": "development_2022_goal_scale_early_gate",
        "preregister_sha256": file_hash(PREREG),
        "contract_sha256": file_hash(CONTRACT),
        "years_read": [2022],
        "years_not_read": [2023, 2024],
        "required_full_gain": required,
        "trials": trials,
        "eligible_gammas": [trial["gamma"] for trial in eligible],
        "status": "eligible_for_2023" if eligible else "failed_2022_goal_scale_gate",
    }
    write_immutable(EARLY, report)
    print(json.dumps({
        "status": report["status"],
        "trials": [{
            "gamma": trial["gamma"],
            "min_full_gain": trial["minimum_full_gain"],
            "min_R_gain": trial["minimum_R_gain"],
            "min_R_ci_low": trial["minimum_R_ci_low"],
        } for trial in trials],
    }, ensure_ascii=False, indent=2))


def select(args: argparse.Namespace, prereg: dict, types_all: np.ndarray) -> None:
    early = json.loads(EARLY.read_text(encoding="utf-8"))
    if early["preregister_sha256"] != file_hash(PREREG):
        raise ValueError("preregister changed after 2022 early gate")
    if early["contract_sha256"] != file_hash(CONTRACT):
        raise ValueError("validation contract changed after 2022 early gate")
    if early["status"] != "eligible_for_2023":
        raise ValueError("2022 goal-scale gate did not pass")
    trials = []
    required = float(prereg["selection"]["early_2022_required_full_point_gain"])
    for gamma in early["eligible_gammas"]:
        cells = {
            str(year): cells_for_year(
                year, prereg, float(gamma), types_all, args.bootstrap
            )
            for year in (2022, 2023)
        }
        flat = [cell for year_cells in cells.values() for cell in year_cells.values()]
        full_gains = [float(cell["metrics"]["all"]["gain"]) for cell in flat]
        r_gains = [float(cell["metrics"]["R"]["gain"]) for cell in flat]
        ci_lows = [float(cell["bootstrap_R"]["ci_low"]) for cell in flat]
        trials.append({
            "gamma": float(gamma),
            "cells": cells,
            "minimum_full_gain": min(full_gains),
            "minimum_R_gain": min(r_gains),
            "minimum_R_ci_low": min(ci_lows),
            "eligible": bool(
                min(full_gains) > required
                and min(r_gains) > 0.0
                and min(ci_lows) > 0.0
            ),
        })
    eligible = [trial for trial in trials if trial["eligible"]]
    selected = max(eligible, key=lambda item: item["minimum_full_gain"]) if eligible else None
    report = {
        "experiment_id": prereg["experiment_id"],
        "mode": "development_selection",
        "preregister_sha256": file_hash(PREREG),
        "contract_sha256": file_hash(CONTRACT),
        "years_read": [2022, 2023],
        "confirmation_year_read": False,
        "trials": trials,
        "selected": selected,
        "status": "locked" if selected else "failed_no_goal_scale_candidate",
    }
    write_immutable(SELECTION, report)
    print(json.dumps({"status": report["status"], "selected": selected}, ensure_ascii=False, indent=2))


def confirm(args: argparse.Namespace, prereg: dict, types_all: np.ndarray) -> None:
    selection = json.loads(SELECTION.read_text(encoding="utf-8"))
    if selection["status"] != "locked" or selection["selected"] is None:
        raise ValueError("no locked count MoE selection")
    if selection["preregister_sha256"] != file_hash(PREREG):
        raise ValueError("preregister changed after selection")
    if selection["contract_sha256"] != file_hash(CONTRACT):
        raise ValueError("validation contract changed after selection")
    gamma = float(selection["selected"]["gamma"])
    candidate = load_candidate(2024, prereg)
    types = types_all[candidate["row_index"].astype(np.int64)]
    cells = {}
    predictions = {}
    for baseline_index, name in enumerate(BASELINES):
        baseline = load_baseline(name, 2024, candidate)
        result = evaluate(
            candidate,
            baseline,
            types,
            prereg["candidate_key"],
            gamma,
            args.bootstrap,
            920000 + 1000 * baseline_index,
        )
        predictions[f"final_{name}"] = result.pop("prediction")
        cells[name] = result
    development = selection["selected"]
    g_dev = float(development["minimum_full_gain"])
    g_confirm = min(float(cell["metrics"]["all"]["gain"]) for cell in cells.values())
    g_ci = min(float(cell["bootstrap_all"]["ci_low"]) for cell in cells.values())
    g_robust = min(g_dev, g_confirm, g_ci)
    expected = V3_ACTUAL_LB + 0.75 * max(0.0, g_robust)
    report = {
        "experiment_id": prereg["experiment_id"],
        "mode": "locked_2024_confirmation",
        "preregister_sha256": file_hash(PREREG),
        "contract_sha256": file_hash(CONTRACT),
        "selected_gamma": gamma,
        "development": development,
        "confirmation_2024": cells,
        "conservative_expected_score": {
            "actual_v3_anchor": V3_ACTUAL_LB,
            "G_dev_full_min": g_dev,
            "G_confirm_full_min": g_confirm,
            "G_confirm_full_ci_low_min": g_ci,
            "G_robust": g_robust,
            "haircut": 0.75,
            "expected_lb_lower": expected,
            "passes_1190": bool(expected > 1190.0),
        },
    }
    write_immutable(CONFIRMATION, report)
    np.savez_compressed(
        PRED / "v5_count_moe_locked_2024.npz",
        y=candidate["y"],
        row_index=candidate["row_index"],
        cluster=candidate["cluster"],
        count_moe=candidate[prereg["candidate_key"]],
        **predictions,
    )
    print(json.dumps(report["conservative_expected_score"], indent=2))


def main() -> None:
    args = parse_args()
    prereg = json.loads(PREREG.read_text(encoding="utf-8"))
    types_all = all_types()
    if args.mode == "gate2022":
        gate2022(args, prereg, types_all)
    elif args.mode == "select":
        select(args, prereg, types_all)
    else:
        confirm(args, prereg, types_all)


if __name__ == "__main__":
    main()
