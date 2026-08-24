#!/usr/bin/env python3
"""Apply the preregistered source gate to game-repeatability predictions."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.run_v5_h1_residual import cluster_bootstrap_score_gain


PRED = ROOT / "experiments/results/predictions"
PREREG = ROOT / "experiments/params/v5_trackman_game_repeatability_preregister.json"
STAGE = ROOT / "experiments/results/v5_trackman_game_repeatability_source.json"
REPORT = ROOT / "experiments/results/v5_trackman_game_repeatability_source_gate.json"
TRAIN = ROOT / "open/data/train.csv"
YEARS = (2020, 2021)
PARENTS = {
    2020: PRED / "v4_m3_c_backtest_2020_2020.npz",
    2021: PRED / "v4_m3_c_backtest_2021_2021.npz",
}


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        return {key: np.asarray(archive[key]) for key in archive.files}


def score(y: np.ndarray, p: np.ndarray, mask: np.ndarray) -> dict[str, Any]:
    yy = y[mask].astype(np.float64)
    pp = p[mask].astype(np.float64)
    rate = float(yy.mean())
    brier = float(np.mean(np.square(pp - yy)))
    return {
        "rows": int(mask.sum()),
        "target_rate": rate,
        "prediction_mean": float(pp.mean()),
        "prediction_std": float(pp.std()),
        "brier": brier,
        "score": float(100000.0 * (1.0 - brier / (rate * (1.0 - rate)))),
    }


def safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [safe(v) for v in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def main() -> None:
    if REPORT.exists():
        raise FileExistsError(f"immutable report exists: {REPORT}")
    prereg = json.loads(PREREG.read_text(encoding="utf-8"))
    stage = json.loads(STAGE.read_text(encoding="utf-8"))
    if prereg["status"] != "locked_before_source_model_or_scores":
        raise ValueError("unexpected preregistration status")
    if stage["metadata"]["booster_device"] != "gpu":
        raise ValueError("candidate is not comparable to the GPU exact-C parent")
    game_type = pd.read_csv(TRAIN, usecols=["game_type"])["game_type"].astype(str)

    folds: dict[int, dict[str, Any]] = {}
    for year in YEARS:
        parent = load(PARENTS[year])
        candidate_path = PRED / f"v5_trackman_game_repeatability_source_{year}.npz"
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
    selection_fold = folds[2020]
    full_mask = np.ones(len(selection_fold["y"]), dtype=bool)
    for gamma_value in prereg["source_protocol"]["gamma_grid"]:
        gamma = float(gamma_value)
        mixed = selection_fold["parent"].copy()
        regular = selection_fold["regular"]
        mixed[regular] = (
            (1.0 - gamma) * selection_fold["parent"][regular]
            + gamma * selection_fold["raw"][regular]
        )
        gamma_trials.append(
            {
                "gamma": gamma,
                "R_gain": score(selection_fold["y"], mixed, regular)["score"]
                - score(selection_fold["y"], selection_fold["parent"], regular)["score"],
                "full_gain": score(selection_fold["y"], mixed, full_mask)["score"]
                - score(selection_fold["y"], selection_fold["parent"], full_mask)["score"],
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
        mixed = fold["parent"].copy()
        regular = fold["regular"]
        mixed[regular] = (
            (1.0 - gamma) * fold["parent"][regular] + gamma * fold["raw"][regular]
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
                iterations=2000, seed=11800 + year + (0 if route == "full" else 100),
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
            ROOT / "experiments/v5_trackman_game_repeatability_features.py"
        ),
        "analysis_code_sha256": digest(Path(__file__)),
        "booster_device": stage["metadata"]["booster_device"],
        "invalid_comparator_run": {
            "reason": "CPU CatBoost was not comparable to the GPU exact-C parent",
            "archive": "experiments/results/archive/invalid_cpu_v5_trackman_game_repeatability_20260822_2328",
            "used_for_selection_or_gate": False,
        },
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
