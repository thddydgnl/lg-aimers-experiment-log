#!/usr/bin/env python3
"""Apply the locked V5 teacher-profile development gate on 2022/2023 only."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
PREDICTIONS = ROOT / "experiments/results/predictions"
PREREG = ROOT / "experiments/params/v5_privileged_trackman_distill_c_preregister.json"
OUTPUT = ROOT / "experiments/results/v5_privileged_trackman_distill_c_selection.json"
YEARS = (2022, 2023)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        return {key: np.asarray(archive[key]) for key in archive.files}


def ensure_aligned(reference: dict[str, np.ndarray], candidate: dict[str, np.ndarray], label: str) -> None:
    for key in ("y", "row_index", "cluster"):
        if not np.array_equal(reference[key], candidate[key]):
            raise ValueError(f"Alignment mismatch for {label}/{key}")


def score_gain_interval(
    y: np.ndarray,
    baseline: np.ndarray,
    candidate: np.ndarray,
    clusters: np.ndarray,
    mask: np.ndarray,
    seed: int,
    replicates: int = 1000,
) -> dict[str, float | int]:
    y = np.asarray(y, dtype=np.float64)[mask]
    baseline = np.asarray(baseline, dtype=np.float64)[mask]
    candidate = np.asarray(candidate, dtype=np.float64)[mask]
    clusters = np.asarray(clusters)[mask]
    rate = float(np.mean(y))
    denominator = rate * (1.0 - rate)
    paired = np.square(y - baseline) - np.square(y - candidate)
    point = 100_000.0 * float(np.mean(paired)) / denominator
    grouped = pd.DataFrame({"cluster": clusters, "paired": paired}).groupby(
        "cluster", sort=False, observed=True
    )["paired"].agg(["sum", "count"])
    sums = grouped["sum"].to_numpy(dtype=np.float64)
    counts = grouped["count"].to_numpy(dtype=np.int64)
    rng = np.random.default_rng(seed)
    draws = np.empty(replicates, dtype=np.float64)
    for index in range(replicates):
        sampled = rng.integers(0, len(grouped), size=len(grouped))
        mean = float(sums[sampled].sum() / counts[sampled].sum())
        draws[index] = 100_000.0 * mean / denominator
    return {
        "rows": int(len(y)),
        "clusters": int(len(grouped)),
        "target_rate": rate,
        "point_gain": point,
        "lower_95": float(np.quantile(draws, 0.025)),
        "upper_95": float(np.quantile(draws, 0.975)),
        "replicates": int(replicates),
    }


def main() -> None:
    prereg = json.loads(PREREG.read_text(encoding="utf-8"))
    if prereg["status"] != "locked_before_teacher_score_generation":
        raise ValueError("Preregister status changed")
    references: dict[int, dict[str, np.ndarray]] = {}
    candidates: dict[int, dict[str, np.ndarray]] = {}
    anchors: dict[int, dict[str, dict[str, np.ndarray]]] = {}
    indices: list[np.ndarray] = []
    for year in YEARS:
        parent = load_npz(PREDICTIONS / f"v3_sparse_c_backtest_{year}.npz")
        candidate = load_npz(
            PREDICTIONS / f"v5_privileged_trackman_distill_c_dev2223_{year}.npz"
        )
        ensure_aligned(parent, candidate, f"candidate/{year}")
        year_anchors = {
            "honest_identity": load_npz(
                PREDICTIONS / f"v5_honest_m3_r_identity_{year}.npz"
            ),
            "honest_grid": load_npz(
                PREDICTIONS / f"v5_honest_m3_r_grid_{year}.npz"
            ),
        }
        for name, artifact in year_anchors.items():
            ensure_aligned(parent, artifact, f"{name}/{year}")
        references[year] = parent
        candidates[year] = candidate
        anchors[year] = year_anchors
        indices.append(parent["row_index"].astype(np.int64))

    wanted = np.unique(np.concatenate(indices))
    usecols = ["season", "game_type"]
    full = pd.read_csv(
        ROOT / "open/data/train.csv",
        usecols=usecols,
        encoding="utf-8-sig",
        low_memory=False,
    )
    folds: dict[str, Any] = {}
    for year in YEARS:
        parent = references[year]
        candidate = candidates[year]
        row_index = parent["row_index"].astype(np.int64)
        frame = full.iloc[row_index]
        if not bool(frame["season"].eq(year).all()):
            raise ValueError(f"Season alignment mismatch for {year}")
        route_r = frame["game_type"].eq("R").to_numpy(dtype=bool)
        all_rows = np.ones(len(route_r), dtype=bool)
        y = parent["y"].astype(np.float64)
        parent_prediction = parent["catboost_outcome"].astype(np.float64)
        augmented = candidate["catboost_outcome"].astype(np.float64)
        routed = np.where(route_r, augmented, parent_prediction)
        comparisons: dict[str, Any] = {
            "vs_exact_parent_r": score_gain_interval(
                y,
                parent_prediction,
                routed,
                parent["cluster"],
                route_r,
                seed=20260821 + year,
            ),
            "vs_exact_parent_routed_full": score_gain_interval(
                y,
                parent_prediction,
                routed,
                parent["cluster"],
                all_rows,
                seed=20261821 + year,
            ),
        }
        for offset, (name, artifact) in enumerate(anchors[year].items(), start=1):
            comparisons[f"vs_{name}_r"] = score_gain_interval(
                y,
                artifact["final_prediction"].astype(np.float64),
                routed,
                parent["cluster"],
                route_r,
                seed=20262821 + 10 * offset + year,
            )
        folds[str(year)] = {
            "r_rows": int(route_r.sum()),
            "f_rows": int((~route_r).sum()),
            "f_prediction_unchanged": bool(
                np.array_equal(routed[~route_r], parent_prediction[~route_r])
            ),
            "comparisons": comparisons,
        }

    parent_point = all(
        folds[str(year)]["comparisons"]["vs_exact_parent_r"]["point_gain"] > 0.0
        for year in YEARS
    )
    parent_ci = all(
        folds[str(year)]["comparisons"]["vs_exact_parent_r"]["lower_95"] > 0.0
        for year in YEARS
    )
    full_point = all(
        folds[str(year)]["comparisons"]["vs_exact_parent_routed_full"]["point_gain"] > 0.0
        for year in YEARS
    )
    anchor_point = all(
        folds[str(year)]["comparisons"][f"vs_{name}_r"]["point_gain"] > 0.0
        for year in YEARS
        for name in ("honest_identity", "honest_grid")
    )
    passed = parent_point and parent_ci and full_point and anchor_point
    payload = {
        "experiment_id": prereg["experiment_id"],
        "status": "passed_lock_before_2024" if passed else "failed_no_2024_run",
        "protocol": {
            "development_years": list(YEARS),
            "2024_candidate_run": False,
            "source_2023_teacher_scores_generated": False,
            "test_rows_read": False,
            "routed_f_to_exact_parent": True,
        },
        "preregister": str(PREREG.relative_to(ROOT)),
        "preregister_sha256": sha256(PREREG),
        "folds": folds,
        "gate": {
            "positive_parent_r_point_both": bool(parent_point),
            "positive_parent_r_lower_both": bool(parent_ci),
            "positive_routed_full_point_both": bool(full_point),
            "positive_both_honest_anchor_r_points_both": bool(anchor_point),
            "passed": bool(passed),
        },
        "next_action": (
            "Lock recipe, generate source-2023 scores, then run 2024 once."
            if passed
            else "Reject this feature bundle without source-2023 generation or 2024 run."
        ),
    }
    OUTPUT.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(payload["gate"], ensure_ascii=False, indent=2), flush=True)
    for year in YEARS:
        comparison = folds[str(year)]["comparisons"]["vs_exact_parent_r"]
        print(
            f"{year} R parent gain={comparison['point_gain']:.3f} "
            f"CI=[{comparison['lower_95']:.3f}, {comparison['upper_95']:.3f}]",
            flush=True,
        )
    print(f"wrote {OUTPUT}", flush=True)


if __name__ == "__main__":
    main()
