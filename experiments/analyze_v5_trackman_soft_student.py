#!/usr/bin/env python3
"""Select the preregistered C/physics-soft-student convex blend on 2022/2023."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
PREDICTIONS = ROOT / "experiments/results/predictions"
PREREG = ROOT / "experiments/params/v5_privileged_trackman_soft_student_preregister.json"
OUTPUT = ROOT / "experiments/results/v5_trackman_soft_student_selection.json"
YEARS = (2022, 2023)
WEIGHTS = (0.1, 0.25, 0.5, 1.0)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        return {key: np.asarray(archive[key]) for key in archive.files}


def aligned(reference: dict[str, np.ndarray], candidate: dict[str, np.ndarray], label: str) -> None:
    for key in ("y", "row_index", "cluster"):
        if not np.array_equal(reference[key], candidate[key]):
            raise ValueError(f"Alignment mismatch for {label}/{key}")


def interval(
    y: np.ndarray,
    baseline: np.ndarray,
    candidate: np.ndarray,
    cluster: np.ndarray,
    mask: np.ndarray,
    seed: int,
    replicates: int = 1000,
) -> dict[str, float | int]:
    y = np.asarray(y, dtype=np.float64)[mask]
    baseline = np.asarray(baseline, dtype=np.float64)[mask]
    candidate = np.asarray(candidate, dtype=np.float64)[mask]
    cluster = np.asarray(cluster)[mask]
    paired = np.square(y - baseline) - np.square(y - candidate)
    rate = float(np.mean(y))
    scale = 100_000.0 / (rate * (1.0 - rate))
    grouped = pd.DataFrame({"cluster": cluster, "paired": paired}).groupby(
        "cluster", sort=False, observed=True
    )["paired"].agg(["sum", "count"])
    sums = grouped["sum"].to_numpy(dtype=np.float64)
    counts = grouped["count"].to_numpy(dtype=np.int64)
    rng = np.random.default_rng(seed)
    draws = np.empty(replicates, dtype=np.float64)
    for index in range(replicates):
        sampled = rng.integers(0, len(grouped), size=len(grouped))
        draws[index] = scale * float(sums[sampled].sum() / counts[sampled].sum())
    return {
        "point_gain": scale * float(np.mean(paired)),
        "lower_95": float(np.quantile(draws, 0.025)),
        "upper_95": float(np.quantile(draws, 0.975)),
        "rows": int(len(y)),
        "clusters": int(len(grouped)),
        "replicates": int(replicates),
    }


def main() -> None:
    prereg = json.loads(PREREG.read_text(encoding="utf-8"))
    if prereg["status"] != "locked_before_student_artifact_materialization":
        raise ValueError("Preregister status changed")
    if tuple(float(value) for value in prereg["blend"]["student_weight_grid"]) != WEIGHTS:
        raise ValueError("Weight grid differs from preregistration")

    parent: dict[int, dict[str, np.ndarray]] = {}
    student: dict[int, dict[str, np.ndarray]] = {}
    anchors: dict[int, dict[str, dict[str, np.ndarray]]] = {}
    for year in YEARS:
        parent[year] = load_npz(PREDICTIONS / f"v3_sparse_c_backtest_{year}.npz")
        student[year] = load_npz(
            PREDICTIONS / f"v5_trackman_soft_student_{year}_{year}.npz"
        )
        aligned(parent[year], student[year], f"student/{year}")
        anchors[year] = {
            "honest_identity": load_npz(PREDICTIONS / f"v5_honest_m3_r_identity_{year}.npz"),
            "honest_grid": load_npz(PREDICTIONS / f"v5_honest_m3_r_grid_{year}.npz"),
        }
        for name, artifact in anchors[year].items():
            aligned(parent[year], artifact, f"{name}/{year}")

    full = pd.read_csv(
        ROOT / "open/data/train.csv",
        usecols=["season", "game_type"],
        encoding="utf-8-sig",
        low_memory=False,
    )
    route: dict[int, np.ndarray] = {}
    for year in YEARS:
        rows = full.iloc[parent[year]["row_index"].astype(np.int64)]
        if not bool(rows["season"].eq(year).all()):
            raise ValueError(f"Season mismatch for {year}")
        route[year] = rows["game_type"].eq("R").to_numpy(dtype=bool)

    trials: list[dict[str, Any]] = []
    for weight_index, weight in enumerate(WEIGHTS):
        fold_metrics: dict[str, Any] = {}
        for year in YEARS:
            y = parent[year]["y"].astype(np.float64)
            base = parent[year]["catboost_outcome"].astype(np.float64)
            soft = student[year]["catboost_teacher"].astype(np.float64)
            r_prediction = (1.0 - weight) * base + weight * soft
            candidate = np.where(route[year], r_prediction, base)
            comparisons: dict[str, Any] = {
                "vs_exact_parent_r": interval(
                    y, base, candidate, parent[year]["cluster"], route[year],
                    seed=20260821 + 100 * weight_index + year,
                ),
                "vs_exact_parent_routed_full": interval(
                    y, base, candidate, parent[year]["cluster"],
                    np.ones(len(y), dtype=bool),
                    seed=20261821 + 100 * weight_index + year,
                ),
            }
            for anchor_index, (name, artifact) in enumerate(anchors[year].items(), start=1):
                comparisons[f"vs_{name}_r"] = interval(
                    y,
                    artifact["final_prediction"].astype(np.float64),
                    candidate,
                    parent[year]["cluster"],
                    route[year],
                    seed=20262821 + 100 * weight_index + 10 * anchor_index + year,
                )
            fold_metrics[str(year)] = comparisons
        min_point = min(
            fold_metrics[str(year)]["vs_exact_parent_r"]["point_gain"] for year in YEARS
        )
        min_lower = min(
            fold_metrics[str(year)]["vs_exact_parent_r"]["lower_95"] for year in YEARS
        )
        trial = {
            "student_weight": weight,
            "parent_weight": 1.0 - weight,
            "folds": fold_metrics,
            "min_parent_r_point_gain": float(min_point),
            "min_parent_r_lower_95": float(min_lower),
        }
        trials.append(trial)

    ranked = sorted(
        trials,
        key=lambda item: (
            item["min_parent_r_point_gain"], item["min_parent_r_lower_95"]
        ),
        reverse=True,
    )
    selected = ranked[0]
    parent_point = selected["min_parent_r_point_gain"] > 0.0
    parent_ci = selected["min_parent_r_lower_95"] > 0.0
    full_point = all(
        selected["folds"][str(year)]["vs_exact_parent_routed_full"]["point_gain"] > 0.0
        for year in YEARS
    )
    anchor_point = all(
        selected["folds"][str(year)][f"vs_{name}_r"]["point_gain"] > 0.0
        for year in YEARS
        for name in ("honest_identity", "honest_grid")
    )
    passed = parent_point and parent_ci and full_point and anchor_point
    payload = {
        "experiment_id": prereg["experiment_id"],
        "status": "passed_lock_before_2024" if passed else "failed_no_2024_run",
        "protocol": {
            "development_years": list(YEARS),
            "source_2023_teacher_artifact_materialized": False,
            "2024_student_run": False,
            "test_rows_read": False,
            "f_predictions_unchanged": True,
        },
        "preregister": str(PREREG.relative_to(ROOT)),
        "preregister_sha256": sha256(PREREG),
        "trials": trials,
        "selected": selected,
        "gate": {
            "positive_parent_r_point_both": bool(parent_point),
            "positive_parent_r_lower_both": bool(parent_ci),
            "positive_routed_full_point_both": bool(full_point),
            "positive_both_honest_anchor_r_points_both": bool(anchor_point),
            "passed": bool(passed),
        },
        "next_action": (
            "Lock the selected weight and run one 2024 confirmation."
            if passed
            else "Reject without source-2023 artifact or 2024 run."
        ),
    }
    OUTPUT.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(payload["gate"], ensure_ascii=False, indent=2), flush=True)
    for trial in trials:
        print(
            f"w={trial['student_weight']:.2f} "
            f"min_point={trial['min_parent_r_point_gain']:.3f} "
            f"min_lower={trial['min_parent_r_lower_95']:.3f}",
            flush=True,
        )
    print(f"wrote {OUTPUT}", flush=True)


if __name__ == "__main__":
    main()
