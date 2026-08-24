#!/usr/bin/env python3
"""Immutable gates for the deployable pitch-type MoE and its oracle diagnostic."""

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
PREREG = ROOT / "experiments/params/v5_pitchtype_moe_preregister.json"
CONTRACT = ROOT / "experiments/params/v5_validation_contract_v2.json"
EARLY = ROOT / "experiments/results/v5_pitchtype_moe_2022_gate.json"
SELECTION = ROOT / "experiments/results/v5_pitchtype_moe_selection.json"
CONFIRMATION = ROOT / "experiments/results/v5_pitchtype_moe_confirmation.json"
V3_ACTUAL_LB = 1090.9100565103
GROUPS = ("fastball", "breaking", "offspeed", "other")
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
    if prereg["candidate_key"] not in candidate:
        raise KeyError(f"{stem}_{year}.npz has no {prereg['candidate_key']}")
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


def evaluate_direction(
    candidate: dict[str, np.ndarray],
    baseline: np.ndarray,
    types: np.ndarray,
    direction_prediction: np.ndarray,
    route: np.ndarray,
    gamma: float,
    bootstrap: int,
    seed: int,
) -> dict:
    prediction = np.asarray(baseline, dtype=np.float64).copy()
    direction = np.asarray(direction_prediction, dtype=np.float64) - baseline
    prediction[route] += gamma * direction[route]
    prediction = np.clip(prediction, 1e-6, 1.0 - 1e-6)
    y = candidate["y"].astype(np.int8)
    cluster = candidate["cluster"].astype(str)
    regular = types == "R"
    return {
        "gamma": float(gamma),
        "route_rows": int(route.sum()),
        "metrics": metrics(y, baseline, prediction, types),
        "bootstrap_R": cluster_bootstrap_score_gain(
            y, baseline, prediction, cluster, regular, bootstrap, seed
        ),
        "bootstrap_route": cluster_bootstrap_score_gain(
            y, baseline, prediction, cluster, route, bootstrap, seed + 5000
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


def deployable_cells_for_year(
    year: int,
    prereg: dict,
    gamma: float,
    types_all: np.ndarray,
    bootstrap: int,
) -> dict[str, dict]:
    candidate = load_candidate(year, prereg)
    types = types_all[candidate["row_index"].astype(np.int64)]
    regular = types == "R"
    direction = candidate[prereg["candidate_key"]]
    cells = {}
    for baseline_index, name in enumerate(BASELINES):
        baseline = load_baseline(name, year, candidate)
        result = evaluate_direction(
            candidate,
            baseline,
            types,
            direction,
            regular,
            gamma,
            bootstrap,
            830000 + 10000 * year + 1000 * baseline_index + int(gamma * 100),
        )
        cells[name] = strip_prediction(result)
    return cells


def stage1_diagnostic(candidate: dict[str, np.ndarray], types: np.ndarray) -> dict:
    prefix = "catboost_pitchtype_moe__"
    probabilities = np.column_stack(
        [candidate[f"{prefix}p_{group}"] for group in GROUPS]
    ).astype(np.float64)
    code = candidate[f"{prefix}diagnostic_true_group_code"].astype(np.int16)
    route = (types == "R") & (code >= 0)
    selected = probabilities[route]
    truth = code[route]
    true_probability = selected[np.arange(len(selected)), truth]
    entropy = -np.sum(
        selected * np.log(np.maximum(selected, 1e-12)), axis=1
    )
    return {
        "matched_regular_rows": int(route.sum()),
        "accuracy": float(np.mean(np.argmax(selected, axis=1) == truth)),
        "multiclass_log_loss": float(-np.mean(np.log(np.maximum(true_probability, 1e-12)))),
        "mean_true_group_probability": float(np.mean(true_probability)),
        "mean_entropy": float(np.mean(entropy)),
        "predicted_group_mean": {
            group: float(selected[:, index].mean())
            for index, group in enumerate(GROUPS)
        },
        "true_group_frequency": {
            group: float(np.mean(truth == index))
            for index, group in enumerate(GROUPS)
        },
    }


def oracle_diagnostic(
    candidate: dict[str, np.ndarray],
    types: np.ndarray,
    gamma_grid: list[float],
    bootstrap: int,
) -> dict:
    prefix = "catboost_pitchtype_moe__"
    oracle = candidate[f"{prefix}diagnostic_true_group_oracle"].astype(np.float64)
    available = candidate[
        f"{prefix}diagnostic_true_group_available"
    ].astype(bool)
    route = (types == "R") & available
    trials = []
    for gamma in gamma_grid:
        cells = {}
        for baseline_index, name in enumerate(BASELINES):
            baseline = load_baseline(name, 2022, candidate)
            result = evaluate_direction(
                candidate,
                baseline,
                types,
                oracle,
                route,
                float(gamma),
                bootstrap,
                840000 + 1000 * baseline_index + int(float(gamma) * 100),
            )
            cells[name] = strip_prediction(result)
        full_gains = [float(cell["metrics"]["all"]["gain"]) for cell in cells.values()]
        route_gains = [float(cell["bootstrap_route"]["point"]) for cell in cells.values()]
        route_lows = [float(cell["bootstrap_route"]["ci_low"]) for cell in cells.values()]
        trials.append({
            "gamma": float(gamma),
            "cells": cells,
            "minimum_full_gain": min(full_gains),
            "minimum_matched_gain": min(route_gains),
            "minimum_matched_ci_low": min(route_lows),
        })
    best = max(trials, key=lambda item: item["minimum_full_gain"])
    return {
        "goal_gate_eligible": false_value(),
        "goal_gate_exclusion_reason": (
            "uses privileged true current pitch group on historical validation"
        ),
        "matched_regular_rows": int(route.sum()),
        "trials": trials,
        "best_by_minimum_full_gain": best,
    }


def false_value() -> bool:
    """Make the diagnostic's exclusion explicit and JSON-serializable."""
    return False


def write_immutable(path: Path, report: dict) -> None:
    if path.exists():
        raise FileExistsError(f"immutable report already exists: {path}")
    path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )


def gate2022(args: argparse.Namespace, prereg: dict, types_all: np.ndarray) -> None:
    candidate = load_candidate(2022, prereg)
    types = types_all[candidate["row_index"].astype(np.int64)]
    trials = []
    required = float(prereg["selection"]["early_2022_required_full_point_gain"])
    for gamma in prereg["selection"]["gamma_grid"]:
        cells = deployable_cells_for_year(
            2022, prereg, float(gamma), types_all, args.bootstrap
        )
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
    diagnostics = {
        "stage1": stage1_diagnostic(candidate, types),
        "true_group_oracle": oracle_diagnostic(
            candidate,
            types,
            [float(value) for value in prereg["selection"]["gamma_grid"]],
            args.bootstrap,
        ),
    }
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
        "diagnostics_excluded_from_goal_gate": diagnostics,
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
        "stage1": diagnostics["stage1"],
        "oracle_best": diagnostics["true_group_oracle"]["best_by_minimum_full_gain"],
    }, ensure_ascii=False, indent=2))


def select(args: argparse.Namespace, prereg: dict, types_all: np.ndarray) -> None:
    early = json.loads(EARLY.read_text(encoding="utf-8"))
    if early["preregister_sha256"] != file_hash(PREREG):
        raise ValueError("preregister changed after 2022 early gate")
    if early["contract_sha256"] != file_hash(CONTRACT):
        raise ValueError("validation contract changed after 2022 early gate")
    if early["status"] != "eligible_for_2023":
        raise ValueError("2022 goal-scale gate did not pass")
    required = float(prereg["selection"]["early_2022_required_full_point_gain"])
    trials = []
    for gamma in early["eligible_gammas"]:
        cells = {
            str(year): deployable_cells_for_year(
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
        raise ValueError("no locked pitch-type MoE selection")
    if selection["preregister_sha256"] != file_hash(PREREG):
        raise ValueError("preregister changed after selection")
    if selection["contract_sha256"] != file_hash(CONTRACT):
        raise ValueError("validation contract changed after selection")
    gamma = float(selection["selected"]["gamma"])
    candidate = load_candidate(2024, prereg)
    types = types_all[candidate["row_index"].astype(np.int64)]
    regular = types == "R"
    direction = candidate[prereg["candidate_key"]]
    cells = {}
    predictions = {}
    for baseline_index, name in enumerate(BASELINES):
        baseline = load_baseline(name, 2024, candidate)
        result = evaluate_direction(
            candidate,
            baseline,
            types,
            direction,
            regular,
            gamma,
            args.bootstrap,
            930000 + 1000 * baseline_index,
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
        PRED / "v5_pitchtype_moe_locked_2024.npz",
        y=candidate["y"],
        row_index=candidate["row_index"],
        cluster=candidate["cluster"],
        pitchtype_moe=candidate[prereg["candidate_key"]],
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
