#!/usr/bin/env python3
"""Gate the preregistered conditional failure chain on source seasons."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.run_v5_h1_residual import (  # noqa: E402
    cluster_bootstrap_score_gain,
)


PREDICTIONS = ROOT / "experiments/results/predictions"
PREREG = ROOT / "experiments/params/v5_hierarchical_failure_chain_preregister.json"
REPORT = ROOT / "experiments/results/v5_hierarchical_failure_chain_source_gate.json"
TRAIN = ROOT / "open/data/train.csv"
SOURCE_YEARS = (2020, 2021)
PARENT = {
    2020: "v4_m3_c_backtest_2020_2020.npz",
    2021: "v4_m3_c_backtest_2021_2021.npz",
}
CANDIDATE_STEM = "v5_hierarchical_failure_chain_source"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        return {key: np.asarray(archive[key]) for key in archive.files}


def score(y: np.ndarray, p: np.ndarray, mask: np.ndarray) -> dict[str, float]:
    target = y[mask].astype(np.float64)
    prediction = p[mask].astype(np.float64)
    rate = float(target.mean())
    reference = max(rate * (1.0 - rate), 1e-12)
    brier = float(np.mean(np.square(prediction - target)))
    return {
        "rows": int(mask.sum()),
        "target_rate": rate,
        "prediction_mean": float(prediction.mean()),
        "prediction_std": float(prediction.std()),
        "brier": brier,
        "score": float(100000.0 * (1.0 - brier / reference)),
    }


def safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def main() -> None:
    if REPORT.exists():
        raise FileExistsError(f"immutable result already exists: {REPORT}")
    prereg = json.loads(PREREG.read_text(encoding="utf-8"))
    threshold_full = float(prereg["source_gate"]["minimum_full_gain_each_year"])
    threshold_r = float(prereg["source_gate"]["minimum_r_gain_each_year"])
    iterations = int(prereg["bootstrap_iterations"])
    types = pd.read_csv(TRAIN, usecols=["game_type"])["game_type"].astype(str)
    checks: list[bool] = []
    report: dict[str, Any] = {
        "experiment_id": prereg["experiment_id"],
        "status": "source_screen",
        "preregister_sha256": sha256(PREREG),
        "script_sha256": sha256(Path(__file__)),
        "candidate": prereg["candidate"],
        "source_years": {},
    }

    for year in SOURCE_YEARS:
        parent_path = PREDICTIONS / PARENT[year]
        candidate_path = PREDICTIONS / f"{CANDIDATE_STEM}_{year}.npz"
        parent = load(parent_path)
        candidate = load(candidate_path)
        for key in ("y", "row_index", "cluster"):
            if not np.array_equal(parent[key], candidate[key]):
                raise ValueError(f"alignment mismatch: {year}/{key}")
        parent_p = parent["catboost_outcome"].astype(np.float64)
        candidate_p = candidate["catboost_failure_chain"].astype(np.float64)
        row_types = types.iloc[parent["row_index"].astype(np.int64)].to_numpy(dtype=str)
        masks = {
            "full": np.ones(len(parent_p), dtype=bool),
            "R": row_types == "R",
        }
        year_report: dict[str, Any] = {
            "parent_artifact": str(parent_path.relative_to(ROOT)),
            "candidate_artifact": str(candidate_path.relative_to(ROOT)),
            "candidate_sha256": sha256(candidate_path),
            "routes": {},
        }
        for route, mask in masks.items():
            parent_metrics = score(parent["y"], parent_p, mask)
            candidate_metrics = score(parent["y"], candidate_p, mask)
            interval = cluster_bootstrap_score_gain(
                parent["y"], parent_p, candidate_p, parent["cluster"], mask,
                iterations=iterations,
                seed=2700 + year + (0 if route == "full" else 100),
            )
            gain = candidate_metrics["score"] - parent_metrics["score"]
            if abs(gain - interval["point"]) > 1e-8:
                raise AssertionError(f"score/CI mismatch: {year}/{route}")
            threshold = threshold_full if route == "full" else threshold_r
            point_pass = bool(gain >= threshold)
            ci_pass = bool(interval["ci_low"] > 0.0)
            checks.extend((point_pass, ci_pass))
            year_report["routes"][route] = {
                "parent": parent_metrics,
                "candidate": candidate_metrics,
                "gain": gain,
                "pitcher_cluster_95_ci": interval,
                "passes_point_gate": point_pass,
                "passes_ci_gate": ci_pass,
            }
        report["source_years"][str(year)] = year_report

    passed = bool(all(checks))
    report["source_gate"] = {
        "minimum_full_gain_each_year": threshold_full,
        "minimum_r_gain_each_year": threshold_r,
        "ci_lower_positive_each_year": True,
        "passed": passed,
        "decision": (
            "freeze and advance to 2022/2023"
            if passed
            else "close H4 without reading 2022+ candidate labels"
        ),
    }
    REPORT.write_text(
        json.dumps(safe(report), ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report["source_gate"], ensure_ascii=False))
    for year in SOURCE_YEARS:
        routes = report["source_years"][str(year)]["routes"]
        print(
            year,
            "full",
            routes["full"]["gain"],
            routes["full"]["pitcher_cluster_95_ci"]["ci_low"],
            "R",
            routes["R"]["gain"],
            routes["R"]["pitcher_cluster_95_ci"]["ci_low"],
        )


if __name__ == "__main__":
    main()
