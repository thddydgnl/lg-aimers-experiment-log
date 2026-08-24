#!/usr/bin/env python3
"""Select the V5 HGB blend on 2021 and transfer it unchanged to 2022/2023."""

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

from experiments.analyze_v4_temporal_residual_ridge import m3_for_season
from experiments.audit_v5_anchor_honesty import apply_recipe, select_recipe
from experiments.analyze_v5_hgb_state_context import (
    ensure_aligned,
    load_npz,
    normalized_score,
    score_gain_interval,
)


PREDICTIONS = ROOT / "experiments/results/predictions"
PREREG = ROOT / "experiments/params/v5_hgb_state_context_blend_preregister.json"
OUTPUT = ROOT / "experiments/results/v5_hgb_state_context_blend_selection.json"
TARGET_YEARS = (2022, 2023)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def route_mask(route: pd.DataFrame, artifact: dict[str, np.ndarray], year: int) -> np.ndarray:
    rows = route.iloc[artifact["row_index"].astype(np.int64)]
    if not bool(rows["season"].eq(year).all()):
        raise ValueError(f"Season alignment mismatch for {year}")
    return rows["game_type"].eq("R").to_numpy(dtype=bool)


def main() -> None:
    prereg = json.loads(PREREG.read_text(encoding="utf-8"))
    if prereg["status"] != "locked_before_source_2021_execution":
        raise ValueError("Preregister status changed")
    route = pd.read_csv(
        ROOT / "open/data/train.csv",
        usecols=["season", "game_type"],
        encoding="utf-8-sig",
        low_memory=False,
    )

    source_2020 = m3_for_season(2020)
    source_2021 = m3_for_season(2021)
    hgb_2021 = load_npz(PREDICTIONS / "v5_hgb_state_context_blend_source21_2021.npz")
    ensure_aligned(source_2021, hgb_2021, "source_hgb/2021")
    mask_2020 = route_mask(route, source_2020, 2020)
    mask_2021 = route_mask(route, source_2021, 2021)
    anchor_recipe_2021 = select_recipe(source_2020, mask_2020, "identity")
    anchor_2021 = apply_recipe(source_2021, anchor_recipe_2021)
    hgb_prediction_2021 = hgb_2021["hgb"].astype(np.float64)
    y_2021 = source_2021["y"].astype(np.float64)

    trials: list[dict[str, float]] = []
    for weight in prereg["source_selection"]["weight_grid_hgb"]:
        weight = float(weight)
        prediction = (1.0 - weight) * anchor_2021 + weight * hgb_prediction_2021
        brier = float(np.mean(np.square(y_2021[mask_2021] - prediction[mask_2021])))
        trials.append(
            {
                "hgb_weight": weight,
                "anchor_weight": 1.0 - weight,
                "source_brier": brier,
                "source_score": normalized_score(y_2021[mask_2021], prediction[mask_2021]),
            }
        )
    selected = min(trials, key=lambda item: (item["source_brier"], item["hgb_weight"]))
    hgb_weight = float(selected["hgb_weight"])

    folds: dict[str, Any] = {}
    for year in TARGET_YEARS:
        hgb = load_npz(PREDICTIONS / f"v5_hgb_state_context_dev2223_{year}.npz")
        identity = load_npz(PREDICTIONS / f"v5_honest_m3_r_identity_{year}.npz")
        grid = load_npz(PREDICTIONS / f"v5_honest_m3_r_grid_{year}.npz")
        exact_c = load_npz(PREDICTIONS / f"v3_sparse_c_backtest_{year}.npz")
        ensure_aligned(hgb, identity, f"identity/{year}")
        ensure_aligned(hgb, grid, f"grid/{year}")
        ensure_aligned(hgb, exact_c, f"exact_c/{year}")
        mask = route_mask(route, hgb, year)
        y = hgb["y"].astype(np.float64)
        prediction = (
            (1.0 - hgb_weight) * identity["final_prediction"].astype(np.float64)
            + hgb_weight * hgb["hgb"].astype(np.float64)
        )
        comparisons = {
            "vs_exact_identity_parent_r": score_gain_interval(
                y,
                identity["final_prediction"].astype(np.float64),
                prediction,
                hgb["cluster"],
                mask,
                seed=20260821 + year,
            ),
            "vs_honest_grid_r": score_gain_interval(
                y,
                grid["final_prediction"].astype(np.float64),
                prediction,
                hgb["cluster"],
                mask,
                seed=20261821 + year,
            ),
            "vs_exact_c_r": score_gain_interval(
                y,
                exact_c["catboost_outcome"].astype(np.float64),
                prediction,
                hgb["cluster"],
                mask,
                seed=20262821 + year,
            ),
        }
        folds[str(year)] = {
            "r_rows": int(mask.sum()),
            "f_rows_excluded": int((~mask).sum()),
            "hgb_weight_locked_from_2021": hgb_weight,
            "comparisons": comparisons,
        }

    nonzero = hgb_weight > 0.0
    parent_point = all(
        folds[str(year)]["comparisons"]["vs_exact_identity_parent_r"]["point_gain"] > 0.0
        for year in TARGET_YEARS
    )
    parent_lower = all(
        folds[str(year)]["comparisons"]["vs_exact_identity_parent_r"]["lower_95"] > 0.0
        for year in TARGET_YEARS
    )
    alternatives_point = all(
        folds[str(year)]["comparisons"][name]["point_gain"] > 0.0
        for year in TARGET_YEARS
        for name in ("vs_honest_grid_r", "vs_exact_c_r")
    )
    passed = nonzero and parent_point and parent_lower and alternatives_point
    payload = {
        "experiment_id": prereg["experiment_id"],
        "status": "passed_lock_before_2024" if passed else "failed_no_2024_run",
        "protocol": {
            "weight_source_year": 2021,
            "weight_source_anchor_fit_year": 2020,
            "development_target_years": list(TARGET_YEARS),
            "2024_candidate_run": False,
            "test_rows_read": False,
            "same_target_fold_weight_fitting": False,
            "calibration": "none",
        },
        "preregister": str(PREREG.relative_to(ROOT)),
        "preregister_sha256": sha256(PREREG),
        "source_anchor_recipe": anchor_recipe_2021,
        "source_trials": trials,
        "selected": selected,
        "folds": folds,
        "gate": {
            "nonzero_source_selected_hgb_weight": bool(nonzero),
            "positive_identity_parent_point_both": bool(parent_point),
            "positive_identity_parent_lower_both": bool(parent_lower),
            "positive_grid_and_exact_c_points_both": bool(alternatives_point),
            "passed": bool(passed),
        },
        "next_action": (
            "Freeze the 2021-selected coefficient and run 2024 exactly once."
            if passed
            else "Reject the complementary HGB blend without running 2024."
        ),
    }
    OUTPUT.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload["selected"], ensure_ascii=False, indent=2), flush=True)
    print(json.dumps(payload["gate"], ensure_ascii=False, indent=2), flush=True)
    for year in TARGET_YEARS:
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
