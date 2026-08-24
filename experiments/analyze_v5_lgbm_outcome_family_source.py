#!/usr/bin/env python3
"""Locked 2020 selection / 2021 confirmation for LightGBM outcome diversity."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.run_v5_h1_residual import cluster_bootstrap_score_gain  # noqa: E402

PRED = ROOT / "experiments/results/predictions"
PREREG = ROOT / "experiments/params/v5_lgbm_outcome_family_preregister.json"
REPORT = ROOT / "experiments/results/v5_lgbm_outcome_family_source_gate.json"
TRAIN = ROOT / "open/data/train.csv"
YEARS = (2020, 2021)


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def score(y: np.ndarray, p: np.ndarray, mask: np.ndarray) -> float:
    yy = y[mask].astype(np.float64)
    pp = p[mask].astype(np.float64)
    rate = float(yy.mean())
    return float(100000.0 * (1.0 - np.mean((pp - yy) ** 2) / (rate * (1.0 - rate))))


def main() -> None:
    if REPORT.exists():
        raise FileExistsError(f"immutable report exists: {REPORT}")
    prereg = json.loads(PREREG.read_text(encoding="utf-8"))
    if prereg["status"] != "locked_before_2020_source_metrics":
        raise ValueError("unexpected preregistration status")
    game_type = pd.read_csv(TRAIN, usecols=["game_type"])["game_type"].astype(str)
    folds = {}
    for year in YEARS:
        candidate_path = PRED / f"v5_lgbm_outcome_family_source_{year}.npz"
        parent_path = PRED / f"v4_m3_c_backtest_{year}_{year}.npz"
        with np.load(candidate_path, allow_pickle=False) as z:
            candidate = {key: np.asarray(z[key]) for key in z.files}
        with np.load(parent_path, allow_pickle=False) as z:
            parent = {key: np.asarray(z[key]) for key in z.files}
        if not np.array_equal(candidate["row_index"], parent["row_index"]):
            raise ValueError(f"{year}: row mismatch")
        if not np.array_equal(candidate["y"], parent["y"]):
            raise ValueError(f"{year}: target mismatch")
        rows = candidate["row_index"].astype(np.int64)
        folds[year] = {
            "y": candidate["y"].astype(np.int8),
            "parent": parent["catboost_outcome"].astype(np.float64),
            "lgbm": candidate["lgbm_outcome"].astype(np.float64),
            "cluster": candidate["cluster"].astype(str),
            "regular": game_type.iloc[rows].eq("R").to_numpy(),
        }
    trials = []
    source = folds[2020]
    for gamma_value in prereg["source_protocol"]["gamma_grid"]:
        gamma = float(gamma_value)
        blend = (1.0 - gamma) * source["parent"] + gamma * source["lgbm"]
        trials.append({
            "gamma": gamma,
            "2020_R_gain": score(source["y"], blend, source["regular"])
            - score(source["y"], source["parent"], source["regular"]),
            "2020_full_gain": score(source["y"], blend, np.ones(len(blend), dtype=bool))
            - score(source["y"], source["parent"], np.ones(len(blend), dtype=bool)),
        })
    selected = max(
        trials,
        key=lambda item: (item["2020_R_gain"], item["2020_full_gain"], -item["gamma"]),
    )
    gamma = float(selected["gamma"])
    results = {}
    for offset, year in enumerate(YEARS):
        fold = folds[year]
        blend = (1.0 - gamma) * fold["parent"] + gamma * fold["lgbm"]
        routes = {"R": fold["regular"], "full": np.ones(len(blend), dtype=bool)}
        results[str(year)] = {}
        for route_offset, (route, mask) in enumerate(routes.items()):
            parent_score = score(fold["y"], fold["parent"], mask)
            candidate_score = score(fold["y"], blend, mask)
            interval = cluster_bootstrap_score_gain(
                fold["y"], fold["parent"], blend, fold["cluster"], mask,
                2000, 882300 + 10 * offset + route_offset,
            )
            results[str(year)][route] = {
                "parent_score": parent_score,
                "candidate_score": candidate_score,
                "gain": candidate_score - parent_score,
                "interval": interval,
            }
    requirements = prereg["source_protocol"]["gate"]
    checks = {}
    for year in YEARS:
        checks[f"{year}_R_gain"] = (
            results[str(year)]["R"]["gain"] >= requirements["minimum_R_gain_each_year"]
        )
        checks[f"{year}_full_gain"] = (
            results[str(year)]["full"]["gain"] >= requirements["minimum_full_gain_each_year"]
        )
        checks[f"{year}_R_ci"] = results[str(year)]["R"]["interval"]["ci_low"] > 0.0
        checks[f"{year}_full_ci"] = results[str(year)]["full"]["interval"]["ci_low"] > 0.0
    passed = bool(all(checks.values()))
    payload = {
        "experiment_id": prereg["experiment_id"],
        "status": "source_pass" if passed else "source_failed",
        "preregister_sha256": digest(PREREG),
        "model_params_sha256": digest(ROOT / "experiments/params/v5_lgbm_outcome_family.json"),
        "script_sha256": digest(Path(__file__)),
        "selection_trials": trials,
        "selected_gamma_from_2020": gamma,
        "results": results,
        "gate": {"requirements": requirements, "checks": checks, "pass": passed},
        "later_years_read": [],
        "goal_status": "active",
        "goal_completion_claimed": False,
    }
    REPORT.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({
        "status": payload["status"], "selected_gamma": gamma,
        "results": results, "gate": payload["gate"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
