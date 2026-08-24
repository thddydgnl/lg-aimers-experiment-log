#!/usr/bin/env python3
"""Apply the locked V5 HGB state/context development gate on 2022/2023."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
PREDICTIONS = ROOT / "experiments/results/predictions"
PREREG = ROOT / "experiments/params/v5_hgb_state_context_preregister.json"
OUTPUT = ROOT / "experiments/results/v5_hgb_state_context_selection.json"
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


def ensure_aligned(reference: dict[str, np.ndarray], other: dict[str, np.ndarray], label: str) -> None:
    for key in ("y", "row_index", "cluster"):
        if not np.array_equal(reference[key], other[key]):
            raise ValueError(f"Alignment mismatch for {label}/{key}")


def normalized_score(y: np.ndarray, prediction: np.ndarray) -> float:
    rate = float(np.mean(y))
    denominator = rate * (1.0 - rate)
    return 100_000.0 * (1.0 - float(np.mean(np.square(y - prediction))) / denominator)


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
        draws[index] = (
            100_000.0 * float(sums[sampled].sum() / counts[sampled].sum()) / denominator
        )
    return {
        "rows": int(len(y)),
        "clusters": int(len(grouped)),
        "target_rate": rate,
        "baseline_score": normalized_score(y, baseline),
        "candidate_score": normalized_score(y, candidate),
        "point_gain": point,
        "lower_95": float(np.quantile(draws, 0.025)),
        "upper_95": float(np.quantile(draws, 0.975)),
        "replicates": int(replicates),
    }


def main() -> None:
    prereg = json.loads(PREREG.read_text(encoding="utf-8"))
    if prereg["status"] != "locked_before_execution":
        raise ValueError("Preregister status changed")
    train_route = pd.read_csv(
        ROOT / "open/data/train.csv",
        usecols=["season", "game_type"],
        encoding="utf-8-sig",
        low_memory=False,
    )
    folds: dict[str, Any] = {}
    for year in YEARS:
        parent = load_npz(PREDICTIONS / f"v5_hgb_context_parent_dev2223_{year}.npz")
        candidate = load_npz(PREDICTIONS / f"v5_hgb_state_context_dev2223_{year}.npz")
        exact_c = load_npz(PREDICTIONS / f"v3_sparse_c_backtest_{year}.npz")
        anchors = {
            "honest_identity": load_npz(PREDICTIONS / f"v5_honest_m3_r_identity_{year}.npz"),
            "honest_grid": load_npz(PREDICTIONS / f"v5_honest_m3_r_grid_{year}.npz"),
        }
        ensure_aligned(parent, candidate, f"candidate/{year}")
        ensure_aligned(parent, exact_c, f"exact_c/{year}")
        for name, artifact in anchors.items():
            ensure_aligned(parent, artifact, f"{name}/{year}")

        row_index = parent["row_index"].astype(np.int64)
        routed = train_route.iloc[row_index]
        if not bool(routed["season"].eq(year).all()):
            raise ValueError(f"Season alignment mismatch for {year}")
        r_mask = routed["game_type"].eq("R").to_numpy(dtype=bool)
        y = parent["y"].astype(np.float64)
        p_candidate = candidate["hgb"].astype(np.float64)
        comparisons = {
            "vs_exact_hgb_parent_r": score_gain_interval(
                y,
                parent["hgb"].astype(np.float64),
                p_candidate,
                parent["cluster"],
                r_mask,
                seed=20260821 + year,
            ),
            "vs_exact_c_r": score_gain_interval(
                y,
                exact_c["catboost_outcome"].astype(np.float64),
                p_candidate,
                parent["cluster"],
                r_mask,
                seed=20261821 + year,
            ),
        }
        for offset, (name, artifact) in enumerate(anchors.items(), start=1):
            comparisons[f"vs_{name}_r"] = score_gain_interval(
                y,
                artifact["final_prediction"].astype(np.float64),
                p_candidate,
                parent["cluster"],
                r_mask,
                seed=20262821 + 10 * offset + year,
            )
        folds[str(year)] = {
            "r_rows": int(r_mask.sum()),
            "f_rows_excluded": int((~r_mask).sum()),
            "comparisons": comparisons,
        }

    parent_point = all(
        folds[str(year)]["comparisons"]["vs_exact_hgb_parent_r"]["point_gain"] > 0.0
        for year in YEARS
    )
    parent_lower = all(
        folds[str(year)]["comparisons"]["vs_exact_hgb_parent_r"]["lower_95"] > 0.0
        for year in YEARS
    )
    anchors_point = all(
        folds[str(year)]["comparisons"][f"vs_{name}_r"]["point_gain"] > 0.0
        for year in YEARS
        for name in ("honest_identity", "honest_grid")
    )
    passed = parent_point and parent_lower and anchors_point
    payload = {
        "experiment_id": prereg["experiment_id"],
        "status": "passed_lock_before_2024" if passed else "failed_no_2024_run",
        "protocol": {
            "development_years": list(YEARS),
            "2024_candidate_run": False,
            "test_rows_read": False,
            "regular_season_primary_only": True,
            "prediction_calibration": "none",
        },
        "preregister": str(PREREG.relative_to(ROOT)),
        "preregister_sha256": sha256(PREREG),
        "folds": folds,
        "gate": {
            "positive_exact_parent_point_both": bool(parent_point),
            "positive_exact_parent_lower_both": bool(parent_lower),
            "positive_both_honest_anchor_points_both": bool(anchors_point),
            "passed": bool(passed),
        },
        "next_action": (
            "Freeze the exact HGB recipe and preregister one development-only convex blend before a single 2024 run."
            if passed
            else "Reject this HGB state/context recipe without running 2024."
        ),
    }
    OUTPUT.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload["gate"], ensure_ascii=False, indent=2), flush=True)
    for year in YEARS:
        print(f"\n{year}", flush=True)
        for name, result in folds[str(year)]["comparisons"].items():
            print(
                f"  {name}: candidate={result['candidate_score']:.3f} "
                f"gain={result['point_gain']:+.3f} "
                f"CI=[{result['lower_95']:+.3f}, {result['upper_95']:+.3f}]",
                flush=True,
            )
    print(f"wrote {OUTPUT}", flush=True)


if __name__ == "__main__":
    main()
