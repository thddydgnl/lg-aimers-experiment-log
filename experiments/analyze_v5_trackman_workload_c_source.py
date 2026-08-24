#!/usr/bin/env python3
"""Apply the locked V5 TrackMan workload source gate on 2022/2023."""

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

from experiments.run_v5_h1_residual import cluster_bootstrap_score_gain  # noqa: E402

TRAIN = ROOT / "open/data/train.csv"
PREREG = ROOT / "experiments/params/v5_trackman_workload_c_preregister.json"
FEATURE_AUDIT = ROOT / "experiments/results/v5_trackman_workload_feature_audit.json"
MODEL_REPORT = ROOT / "experiments/results/v5_trackman_workload_c_source.json"
REPORT = ROOT / "experiments/results/v5_trackman_workload_c_source_gate.json"
PREDICTIONS = ROOT / "experiments/results/predictions"
YEARS = (2022, 2023)
MODEL_PATHS = {
    year: PREDICTIONS / f"v5_trackman_workload_c_source_{year}.npz"
    for year in YEARS
}
BASELINE_PATHS = {
    "exact_parent_C": {
        year: (PREDICTIONS / f"v3_sparse_c_backtest_{year}.npz", "catboost_outcome")
        for year in YEARS
    },
    "honest_r_identity": {
        year: (
            PREDICTIONS / f"v5_honest_m3_r_identity_{year}.npz",
            "final_prediction",
        )
        for year in YEARS
    },
    "honest_r_grid": {
        year: (
            PREDICTIONS / f"v5_honest_m3_r_grid_{year}.npz",
            "final_prediction",
        )
        for year in YEARS
    },
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


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


def load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        return {key: np.asarray(archive[key]) for key in archive.files}


def summarize(
    y: np.ndarray, prediction: np.ndarray, mask: np.ndarray
) -> dict[str, float]:
    yy = y[mask].astype(np.float64)
    pp = prediction[mask].astype(np.float64)
    rate = float(yy.mean())
    brier = float(np.mean(np.square(pp - yy)))
    return {
        "rows": int(mask.sum()),
        "target_rate": rate,
        "prediction_mean": float(pp.mean()),
        "prediction_std": float(pp.std()),
        "brier": brier,
        "score": 100000.0 * (1.0 - brier / (rate * (1.0 - rate))),
    }


def main() -> None:
    if REPORT.exists():
        raise FileExistsError(f"immutable report exists: {REPORT}")
    prereg = json.loads(PREREG.read_text(encoding="utf-8"))
    feature_audit = json.loads(FEATURE_AUDIT.read_text(encoding="utf-8"))
    model_report = json.loads(MODEL_REPORT.read_text(encoding="utf-8"))
    if prereg["status"] != "locked_before_2022_2023_candidate_metrics":
        raise ValueError("unexpected preregistration state")
    if feature_audit["status"] != "passed":
        raise ValueError("feature invariance audit did not pass")

    models = {year: load_npz(path) for year, path in MODEL_PATHS.items()}
    maximum_row = max(int(models[year]["row_index"].max()) for year in YEARS)
    frame = pd.read_csv(
        TRAIN,
        usecols=["season", "game_type"],
        nrows=maximum_row + 1,
        encoding="utf-8-sig",
    )
    if int(frame["season"].max()) != max(YEARS):
        raise ValueError("source reader crossed the locked 2023 target boundary")

    folds: dict[int, dict[str, Any]] = {}
    alignment_checks: dict[str, bool] = {}
    baselines: dict[str, dict[int, np.ndarray]] = {
        name: {} for name in BASELINE_PATHS
    }
    for year in YEARS:
        model = models[year]
        row_index = model["row_index"].astype(np.int64)
        expected = frame.index[frame["season"].eq(year)].to_numpy(dtype=np.int64)
        aligned = np.array_equal(row_index, expected)
        alignment_checks[f"model_{year}_row_index"] = bool(aligned)
        if not aligned:
            raise ValueError(f"{year}: workload model row order mismatch")
        y = model["y"].astype(np.int8)
        cluster = model["cluster"].astype(str)
        candidate = model["catboost_outcome"].astype(np.float64)
        regular = frame.loc[row_index, "game_type"].astype(str).eq("R").to_numpy()
        folds[year] = {
            "y": y,
            "cluster": cluster,
            "candidate": candidate,
            "regular": regular,
            "row_index": row_index,
        }
        for name, yearly in BASELINE_PATHS.items():
            path, key = yearly[year]
            archive = load_npz(path)
            checks = {
                "row_index": np.array_equal(
                    archive["row_index"].astype(np.int64), row_index
                ),
                "target": np.array_equal(archive["y"].astype(np.int8), y),
                "cluster": np.array_equal(archive["cluster"].astype(str), cluster),
            }
            for check_name, passed in checks.items():
                alignment_checks[f"{name}_{year}_{check_name}"] = bool(passed)
            if not all(checks.values()):
                raise ValueError(f"{name} {year}: artifact alignment mismatch")
            baselines[name][year] = archive[key].astype(np.float64)

    trials: list[dict[str, Any]] = []
    for gamma_value in prereg["development"]["blend_grid"]:
        gamma = float(gamma_value)
        cells: dict[str, Any] = {}
        full_gains: list[float] = []
        r_gains: list[float] = []
        for baseline_name, yearly in baselines.items():
            cells[baseline_name] = {}
            for year in YEARS:
                fold = folds[year]
                regular = fold["regular"]
                baseline = yearly[year]
                prediction = baseline.copy()
                prediction[regular] = (
                    (1.0 - gamma) * baseline[regular]
                    + gamma * fold["candidate"][regular]
                )
                prediction = np.clip(prediction, 1e-6, 1.0 - 1e-6)
                full = np.ones(len(prediction), dtype=bool)
                baseline_full = summarize(fold["y"], baseline, full)
                candidate_full = summarize(fold["y"], prediction, full)
                baseline_r = summarize(fold["y"], baseline, regular)
                candidate_r = summarize(fold["y"], prediction, regular)
                full_gain = candidate_full["score"] - baseline_full["score"]
                r_gain = candidate_r["score"] - baseline_r["score"]
                full_gains.append(full_gain)
                r_gains.append(r_gain)
                cells[baseline_name][str(year)] = {
                    "full": {
                        "baseline": baseline_full,
                        "candidate": candidate_full,
                        "gain": full_gain,
                    },
                    "R": {
                        "baseline": baseline_r,
                        "candidate": candidate_r,
                        "gain": r_gain,
                    },
                    "F_prediction_preserved_exactly": bool(
                        np.array_equal(prediction[~regular], baseline[~regular])
                    ),
                }
        trials.append(
            {
                "gamma": gamma,
                "minimum_full_gain": float(min(full_gains)),
                "minimum_R_gain": float(min(r_gains)),
                "mean_full_gain": float(np.mean(full_gains)),
                "cells": cells,
            }
        )
    selected = max(
        trials,
        key=lambda item: (
            item["minimum_full_gain"],
            item["minimum_R_gain"],
            -item["gamma"],
        ),
    )

    intervals: dict[str, Any] = {}
    gamma = float(selected["gamma"])
    for baseline_offset, (baseline_name, yearly) in enumerate(baselines.items()):
        intervals[baseline_name] = {}
        for year_offset, year in enumerate(YEARS):
            fold = folds[year]
            regular = fold["regular"]
            baseline = yearly[year]
            prediction = baseline.copy()
            prediction[regular] = (
                (1.0 - gamma) * baseline[regular]
                + gamma * fold["candidate"][regular]
            )
            full = np.ones(len(prediction), dtype=bool)
            seed = 883100 + 100 * baseline_offset + 10 * year_offset
            intervals[baseline_name][str(year)] = {
                "R": cluster_bootstrap_score_gain(
                    fold["y"], baseline, prediction, fold["cluster"], regular,
                    1000, seed,
                ),
                "full": cluster_bootstrap_score_gain(
                    fold["y"], baseline, prediction, fold["cluster"], full,
                    1000, seed + 1,
                ),
            }

    threshold = float(
        prereg["development"]["advance_gate"]["minimum_development_full_gain"]
    )
    checks = {
        "minimum_full_gain_goal_sized": selected["minimum_full_gain"] >= threshold,
        "all_R_ci_low_positive": all(
            intervals[name][str(year)]["R"]["ci_low"] > 0.0
            for name in intervals for year in YEARS
        ),
        "all_full_ci_low_positive": all(
            intervals[name][str(year)]["full"]["ci_low"] > 0.0
            for name in intervals for year in YEARS
        ),
        "feature_row_order_invariance": bool(
            feature_audit["checks"]["source_order_invariant"]
            and feature_audit["checks"]["prediction_row_order_invariant"]
        ),
        "model_prediction_artifact_alignment": all(alignment_checks.values()),
        "F_preserved_every_cell": all(
            cell["F_prediction_preserved_exactly"]
            for trial in [selected]
            for baseline_cells in trial["cells"].values()
            for cell in baseline_cells.values()
        ),
    }
    passed = bool(all(checks.values()))
    report = {
        "experiment_id": prereg["experiment_id"],
        "status": "development_pass" if passed else "development_failed",
        "preregister_sha256": sha256(PREREG),
        "feature_audit_sha256": sha256(FEATURE_AUDIT),
        "model_report_sha256": sha256(MODEL_REPORT),
        "script_sha256": sha256(Path(__file__)),
        "years_read": list(YEARS),
        "confirmation_2024_read": False,
        "model_stage_status": model_report.get("metadata", {}).get("status", "complete"),
        "alignment_checks": alignment_checks,
        "trials": trials,
        "selected": selected,
        "intervals": intervals,
        "gate": {
            "requirements": prereg["development"]["advance_gate"],
            "checks": checks,
            "pass": passed,
            "decision": (
                "write confirmation lock before 2024"
                if passed
                else "close without fitting or reading a 2024 workload candidate"
            ),
        },
        "prediction_artifacts": {
            str(year): {
                "path": str(MODEL_PATHS[year].relative_to(ROOT)),
                "sha256": sha256(MODEL_PATHS[year]),
            }
            for year in YEARS
        },
        "goal_status": "active",
        "goal_completion_claimed": False,
    }
    REPORT.write_text(
        json.dumps(safe(report), ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(
        json.dumps(
            safe(
                {
                    "status": report["status"],
                    "selected": selected,
                    "intervals": intervals,
                    "gate": report["gate"],
                }
            ),
            ensure_ascii=False,
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
