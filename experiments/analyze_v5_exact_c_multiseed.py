#!/usr/bin/env python3
"""Evaluate a preregistered three-seed uniform exact-C bag on source folds."""

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
PREREG = ROOT / "experiments/params/v5_exact_c_multiseed_preregister.json"
REPORT = ROOT / "experiments/results/v5_exact_c_multiseed_source.json"
TRAIN = ROOT / "open/data/train.csv"
SOURCE_YEARS = (2020, 2021)
PARENT_FILES = {
    2020: "v4_m3_c_backtest_2020_2020.npz",
    2021: "v4_m3_c_backtest_2021_2021.npz",
}
SEED_FILES = {
    1: "v5_exact_c_multiseed_s1_source_{year}.npz",
    7: "v5_exact_c_multiseed_s7_source_{year}.npz",
    2026: None,
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        return {key: np.asarray(archive[key]) for key in archive.files}


def assert_aligned(
    reference: dict[str, np.ndarray], other: dict[str, np.ndarray], label: str
) -> None:
    for key in ("y", "row_index", "cluster"):
        if not np.array_equal(reference[key], other[key]):
            raise ValueError(f"alignment mismatch for {label}/{key}")


def score(y: np.ndarray, prediction: np.ndarray, mask: np.ndarray) -> dict[str, float]:
    target = y[mask].astype(np.float64)
    probability = prediction[mask].astype(np.float64)
    rate = float(target.mean())
    reference = max(rate * (1.0 - rate), 1e-12)
    brier = float(np.mean(np.square(probability - target)))
    return {
        "rows": int(mask.sum()),
        "target_rate": rate,
        "prediction_mean": float(probability.mean()),
        "prediction_std": float(probability.std()),
        "brier": brier,
        "score": float(100000.0 * (1.0 - brier / reference)),
    }


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def main() -> None:
    if REPORT.exists():
        raise FileExistsError(f"immutable result already exists: {REPORT}")
    prereg = json.loads(PREREG.read_text(encoding="utf-8"))
    seeds = [int(seed) for seed in prereg["candidate"]["seeds"]]
    weights = np.asarray(prereg["candidate"]["weights"], dtype=np.float64)
    if seeds != [1, 7, 2026] or not np.allclose(weights, np.full(3, 1.0 / 3.0)):
        raise ValueError("preregistered seed recipe changed")

    game_type = pd.read_csv(TRAIN, usecols=["game_type"])["game_type"].astype(str)
    report: dict[str, Any] = {
        "experiment_id": prereg["experiment_id"],
        "status": "source_screen",
        "preregister_sha256": sha256(PREREG),
        "script_sha256": sha256(Path(__file__)),
        "candidate": prereg["candidate"],
        "source_years": {},
        "candidate_artifacts": {},
    }
    gate_checks: list[bool] = []
    threshold = float(prereg["source_gate"]["minimum_full_gain_each_year"])
    iterations = int(prereg["bootstrap_iterations"])

    for year in SOURCE_YEARS:
        parent_path = PREDICTIONS / PARENT_FILES[year]
        parent = load_npz(parent_path)
        seed_predictions: list[np.ndarray] = []
        seed_paths: dict[str, str] = {}
        for seed in seeds:
            if seed == 2026:
                artifact = parent
                path = parent_path
            else:
                path = PREDICTIONS / SEED_FILES[seed].format(year=year)
                artifact = load_npz(path)
                assert_aligned(parent, artifact, f"{year}/seed{seed}")
            seed_predictions.append(
                artifact["catboost_outcome"].astype(np.float64)
            )
            seed_paths[str(seed)] = str(path.relative_to(ROOT))

        candidate = np.average(
            np.column_stack(seed_predictions), axis=1, weights=weights
        )
        parent_prediction = parent["catboost_outcome"].astype(np.float64)
        rows = parent["row_index"].astype(np.int64)
        types = game_type.iloc[rows].to_numpy(dtype=str)
        masks = {
            "full": np.ones(len(rows), dtype=bool),
            "R": types == "R",
        }
        year_report: dict[str, Any] = {
            "parent_artifact": str(parent_path.relative_to(ROOT)),
            "seed_artifacts": seed_paths,
            "seed_pairwise_prediction_correlation": np.corrcoef(
                np.column_stack(seed_predictions), rowvar=False
            ),
            "seed_mean_row_std": float(
                np.std(np.column_stack(seed_predictions), axis=1).mean()
            ),
            "routes": {},
        }
        for route, mask in masks.items():
            parent_metrics = score(parent["y"], parent_prediction, mask)
            candidate_metrics = score(parent["y"], candidate, mask)
            interval = cluster_bootstrap_score_gain(
                parent["y"],
                parent_prediction,
                candidate,
                parent["cluster"],
                mask,
                iterations=iterations,
                seed=1700 + year + (0 if route == "full" else 100),
            )
            gain = candidate_metrics["score"] - parent_metrics["score"]
            if abs(gain - interval["point"]) > 1e-8:
                raise AssertionError(f"score/CI point mismatch: {year}/{route}")
            year_report["routes"][route] = {
                "parent": parent_metrics,
                "candidate": candidate_metrics,
                "gain": gain,
                "pitcher_cluster_95_ci": interval,
                "passes_point_gate": bool(gain >= threshold),
                "passes_ci_gate": bool(interval["ci_low"] > 0.0),
            }
            gate_checks.extend([gain >= threshold, interval["ci_low"] > 0.0])

        output_path = PREDICTIONS / f"v5_exact_c_multiseed_source_{year}.npz"
        if output_path.exists():
            raise FileExistsError(f"immutable artifact already exists: {output_path}")
        np.savez_compressed(
            output_path,
            y=parent["y"].astype(np.int8),
            row_index=rows,
            cluster=parent["cluster"],
            parent_exact_c=parent_prediction,
            candidate_uniform_three_seed=candidate,
        )
        report["candidate_artifacts"][str(year)] = {
            "path": str(output_path.relative_to(ROOT)),
            "sha256": sha256(output_path),
        }
        report["source_years"][str(year)] = year_report

    passed = bool(all(gate_checks))
    report["source_gate"] = {
        "threshold": threshold,
        "pitcher_cluster_ci_lower_positive": True,
        "passed": passed,
        "decision": (
            "freeze and advance to 2022/2023"
            if passed
            else "close without reading 2022+ candidate labels"
        ),
    }
    REPORT.write_text(
        json.dumps(json_safe(report), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(json_safe(report["source_gate"]), ensure_ascii=False))
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
