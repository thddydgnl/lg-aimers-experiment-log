#!/usr/bin/env python3
"""Apply the preregistered source gate to inning-physics predictions."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.analyze_v5_trackman_game_repeatability_source import (  # noqa: E402
    digest,
    load,
    safe,
    score,
)
from experiments.run_v5_h1_residual import cluster_bootstrap_score_gain  # noqa: E402


PRED = ROOT / "experiments/results/predictions"
PREREG = ROOT / "experiments/params/v5_trackman_inning_physics_preregister.json"
STAGE = ROOT / "experiments/results/v5_trackman_inning_physics_source.json"
REPORT = ROOT / "experiments/results/v5_trackman_inning_physics_source_gate.json"
TRAIN = ROOT / "open/data/train.csv"
YEARS = (2020, 2021)
PARENTS = {
    year: PRED / f"v4_m3_c_backtest_{year}_{year}.npz" for year in YEARS
}


def main() -> None:
    if REPORT.exists():
        raise FileExistsError(f"immutable report exists: {REPORT}")
    prereg = json.loads(PREREG.read_text(encoding="utf-8"))
    stage = json.loads(STAGE.read_text(encoding="utf-8"))
    if prereg["status"] != "locked_before_source_model_or_scores":
        raise ValueError("unexpected preregistration status")
    if stage["metadata"]["booster_device"] != "gpu":
        raise ValueError("candidate is not comparable to the GPU exact-C parent")
    if "trackman_inning_physics" not in stage["metadata"]["features"]:
        raise ValueError("stage does not contain the locked e119 feature")
    game_type = pd.read_csv(TRAIN, usecols=["game_type"])["game_type"].astype(str)

    folds: dict[int, dict[str, Any]] = {}
    for year in YEARS:
        parent = load(PARENTS[year])
        candidate_path = PRED / f"v5_trackman_inning_physics_source_{year}.npz"
        candidate = load(candidate_path)
        for key in ("y", "row_index", "cluster"):
            if not np.array_equal(parent[key], candidate[key]):
                raise ValueError(f"{year} alignment mismatch: {key}")
        rows = parent["row_index"].astype(np.int64)
        folds[year] = {
            "y": parent["y"].astype(np.int8),
            "cluster": parent["cluster"],
            "parent": parent["catboost_outcome"].astype(np.float64),
            "raw": candidate["catboost_outcome"].astype(np.float64),
            "regular": game_type.iloc[rows].eq("R").to_numpy(),
            "artifacts": {
                "parent": str(PARENTS[year].relative_to(ROOT)),
                "candidate": str(candidate_path.relative_to(ROOT)),
                "candidate_sha256": digest(candidate_path),
            },
        }

    gamma_trials = []
    source = folds[2020]
    full_source = np.ones(len(source["y"]), dtype=bool)
    for gamma_value in prereg["source_protocol"]["gamma_grid"]:
        gamma = float(gamma_value)
        mixed = source["parent"].copy()
        mixed[source["regular"]] = (
            (1.0 - gamma) * source["parent"][source["regular"]]
            + gamma * source["raw"][source["regular"]]
        )
        gamma_trials.append(
            {
                "gamma": gamma,
                "R_gain": score(source["y"], mixed, source["regular"])["score"]
                - score(source["y"], source["parent"], source["regular"])["score"],
                "full_gain": score(source["y"], mixed, full_source)["score"]
                - score(source["y"], source["parent"], full_source)["score"],
            }
        )
    selected = max(
        gamma_trials,
        key=lambda item: (item["R_gain"], item["full_gain"], -item["gamma"]),
    )
    gamma = float(selected["gamma"])

    results: dict[str, Any] = {}
    checks: list[bool] = []
    gate = prereg["source_protocol"]["advance_gate"]
    for year, fold in folds.items():
        regular = fold["regular"]
        mixed = fold["parent"].copy()
        mixed[regular] = (
            (1.0 - gamma) * fold["parent"][regular]
            + gamma * fold["raw"][regular]
        )
        routes: dict[str, Any] = {}
        for route, mask in (
            ("full", np.ones(len(fold["y"]), dtype=bool)),
            ("R", regular),
        ):
            parent_metrics = score(fold["y"], fold["parent"], mask)
            candidate_metrics = score(fold["y"], mixed, mask)
            interval = cluster_bootstrap_score_gain(
                fold["y"], fold["parent"], mixed, fold["cluster"], mask,
                iterations=2000, seed=11900 + year + (0 if route == "full" else 100),
            )
            gain = candidate_metrics["score"] - parent_metrics["score"]
            minimum = float(
                gate["minimum_full_gain_each_year"]
                if route == "full"
                else gate["minimum_R_gain_each_year"]
            )
            routes[route] = {
                "parent": parent_metrics,
                "candidate": candidate_metrics,
                "gain": gain,
                "pitcher_cluster_95_ci": interval,
                "passes_point": bool(gain >= minimum),
                "passes_ci": bool(interval["ci_low"] > 0.0),
            }
            checks.extend([gain >= minimum, interval["ci_low"] > 0.0])
        results[str(year)] = {"routes": routes, "artifacts": fold["artifacts"]}

    passed = bool(all(checks))
    report = {
        "experiment_id": prereg["experiment_id"],
        "status": "source_passed" if passed else "source_failed",
        "preregister_sha256": digest(PREREG),
        "stage_report_sha256": digest(STAGE),
        "feature_code_sha256": digest(
            ROOT / "experiments/v5_trackman_inning_physics_features.py"
        ),
        "analysis_code_sha256": digest(Path(__file__)),
        "booster_device": stage["metadata"]["booster_device"],
        "gamma_selection_year": 2020,
        "gamma_trials": gamma_trials,
        "selected_gamma": gamma,
        "results": results,
        "gate": {
            "requirements": gate,
            "passed": passed,
            "decision": (
                "write a development lock before 2022/2023"
                if passed
                else "close without reading 2022-2024 candidate labels"
            ),
        },
        "later_years_read": [],
        "goal_status": "active",
        "goal_completion_claimed": False,
    }
    REPORT.write_text(
        json.dumps(safe(report), ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(safe({
        "status": report["status"],
        "selected_gamma": gamma,
        "results": results,
        "gate": report["gate"],
    }), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
