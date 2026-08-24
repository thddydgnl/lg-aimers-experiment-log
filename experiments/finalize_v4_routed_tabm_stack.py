#!/usr/bin/env python3
"""Lock the leakage-safe R-only TabM/TrackMan-B residual stack.

The two coefficients are estimated once from the pooled 2022 and 2023
rolling predictions.  The untouched 2024 fold is used only for confirmation.
Every validation and future inference row is routed independently from its
own official ``game_type`` value.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.analyze_v4_temporal_residual_ridge import (  # noqa: E402
    MEDIAN_OFFSET,
    REQUIRED_LOCAL,
    json_safe,
    score,
)


PREDICTIONS = ROOT / "experiments/results/predictions"
OUTPUT_REPORT = ROOT / "experiments/results/v4_routed_tabm_stack_locked.json"
SELECTION_YEARS = (2022, 2023)
CONFIRMATION_YEAR = 2024
YEARS = (*SELECTION_YEARS, CONFIRMATION_YEAR)


def load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path) as archive:
        return {key: np.asarray(archive[key]) for key in archive.files}


def ensure_aligned(reference: dict[str, np.ndarray], candidate: dict[str, np.ndarray],
                   label: str) -> None:
    for key in ("y", "row_index", "cluster"):
        if not np.array_equal(reference[key], candidate[key]):
            raise ValueError(f"Artifact alignment mismatch for {label}/{key}")


def source_artifacts(year: int) -> dict[str, dict[str, np.ndarray]]:
    locked = load_npz(
        PREDICTIONS / f"v4_pitchtype_failure_tagged_locked_{year}.npz"
    )
    tabm = load_npz(PREDICTIONS / f"v4_tabm_enhanced_all_{year}.npz")
    b_stem = (
        "v4_outcome_b_trackman_stability_backtest"
        if year < CONFIRMATION_YEAR
        else "v4_outcome_b_trackman_stability"
    )
    stability_b = load_npz(PREDICTIONS / f"{b_stem}_{year}.npz")
    ensure_aligned(locked, tabm, f"tabm/{year}")
    ensure_aligned(locked, stability_b, f"trackman_b/{year}")
    return {"locked": locked, "tabm": tabm, "stability_b": stability_b}


def direction_matrix(items: dict[str, dict[str, np.ndarray]]) -> np.ndarray:
    base = np.asarray(items["locked"]["tagged_locked"], dtype=np.float64)
    tabm = np.asarray(items["tabm"]["tabm_outcome"], dtype=np.float64)
    stability_b = np.asarray(
        items["stability_b"]["catboost_outcome"], dtype=np.float64
    )
    return np.column_stack((tabm - base, stability_b - base))


def main() -> None:
    game_type = pd.read_csv(
        ROOT / "open/data/train.csv",
        usecols=["game_type"],
        encoding="utf-8-sig",
        low_memory=False,
    )["game_type"].astype(str).to_numpy()
    sources = {year: source_artifacts(year) for year in YEARS}

    design_parts: list[np.ndarray] = []
    target_parts: list[np.ndarray] = []
    for year in SELECTION_YEARS:
        locked = sources[year]["locked"]
        base = np.asarray(locked["tagged_locked"], dtype=np.float64)
        y = np.asarray(locked["y"], dtype=np.float64)
        route = game_type[np.asarray(locked["row_index"], dtype=np.int64)] == "R"
        design_parts.append(direction_matrix(sources[year])[route])
        target_parts.append((y - base)[route])
    coefficients = np.linalg.lstsq(
        np.vstack(design_parts), np.concatenate(target_parts), rcond=None
    )[0]

    metrics: dict[str, dict[str, object]] = {}
    fold_artifacts: dict[str, str] = {}
    for year in YEARS:
        locked = sources[year]["locked"]
        y = np.asarray(locked["y"], dtype=np.float64)
        base = np.asarray(locked["tagged_locked"], dtype=np.float64)
        row_index = np.asarray(locked["row_index"], dtype=np.int64)
        route = game_type[row_index] == "R"
        correction = np.zeros(len(base), dtype=np.float64)
        correction[route] = direction_matrix(sources[year])[route] @ coefficients
        prediction = np.clip(base + correction, 0.0, 1.0)
        base_metric = score(y, base)
        candidate_metric = score(y, prediction)
        metrics[str(year)] = {
            "baseline": base_metric,
            "candidate": candidate_metric,
            "gain": float(
                candidate_metric["raw_competition_score"]
                - base_metric["raw_competition_score"]
            ),
            "routed_rows": int(route.sum()),
            "correction_mean": float(correction.mean()),
            "correction_std": float(correction.std()),
            "correction_max_abs": float(np.max(np.abs(correction))),
        }
        output_path = PREDICTIONS / f"v4_routed_tabm_stack_locked_{year}.npz"
        np.savez_compressed(
            output_path,
            y=y,
            row_index=row_index,
            cluster=np.asarray(locked["cluster"]),
            game_type_r=route,
            locked=base,
            tabm_direction=direction_matrix(sources[year])[:, 0],
            stability_b_direction=direction_matrix(sources[year])[:, 1],
            correction=correction,
            routed_tabm_stack=prediction,
        )
        fold_artifacts[str(year)] = str(output_path.relative_to(ROOT))

    local_2024 = float(
        metrics[str(CONFIRMATION_YEAR)]["candidate"]["raw_competition_score"]
    )
    report = {
        "protocol": {
            "status": "locked after positive selection and confirmation",
            "official_train_only": True,
            "test_rows_read": False,
            "leaderboard_values_used_for_coefficients": False,
            "selection_folds": list(SELECTION_YEARS),
            "confirmation_fold": CONFIRMATION_YEAR,
            "route": "apply correction only when the row's game_type is R",
            "row_independent_inference": True,
            "coefficient_fit": "pooled R-row least squares without intercept",
        },
        "sources": {
            "baseline": "v4_pitchtype_failure_tagged_locked/tagged_locked",
            "direction_1": "v4_tabm_enhanced_all/tabm_outcome minus baseline",
            "direction_2": (
                "v4_outcome_b_trackman_stability[_backtest]/catboost_outcome "
                "minus baseline"
            ),
        },
        "coefficients": {
            "tabm": float(coefficients[0]),
            "trackman_stability_b": float(coefficients[1]),
        },
        "folds": metrics,
        "confirmation_2024": {
            "local_score": local_2024,
            "expected_lb_median": local_2024 + MEDIAN_OFFSET,
            "required_local_score": REQUIRED_LOCAL,
            "crosses_required_local_score": local_2024 > REQUIRED_LOCAL,
        },
        "fold_artifacts": fold_artifacts,
    }
    OUTPUT_REPORT.write_text(
        json.dumps(json_safe(report), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(json_safe({
        "coefficients": report["coefficients"],
        "fold_gains": {year: value["gain"] for year, value in metrics.items()},
        "confirmation_2024": report["confirmation_2024"],
    }), ensure_ascii=False, indent=2), flush=True)
    print(f"Saved {OUTPUT_REPORT}", flush=True)


if __name__ == "__main__":
    main()
