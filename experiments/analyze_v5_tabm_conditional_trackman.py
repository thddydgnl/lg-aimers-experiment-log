#!/usr/bin/env python3
"""Apply the locked V5 conditional-TrackMan TabM development gate."""

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

from experiments.analyze_v5_hgb_state_context import (  # noqa: E402
    ensure_aligned,
    load_npz,
    score_gain_interval,
)


PREDICTIONS = ROOT / "experiments/results/predictions"
PREREG = ROOT / "experiments/params/v5_tabm_conditional_trackman_preregister.json"
OUTPUT = ROOT / "experiments/results/v5_tabm_conditional_trackman_selection.json"
YEARS = (2022, 2023)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    prereg = json.loads(PREREG.read_text(encoding="utf-8"))
    if prereg["status"] != "locked_before_execution":
        raise ValueError("Preregister status changed")
    route = pd.read_csv(
        ROOT / "open/data/train.csv",
        usecols=["season", "game_type"],
        encoding="utf-8-sig",
        low_memory=False,
    )
    folds: dict[str, Any] = {}
    for year in YEARS:
        parent = load_npz(PREDICTIONS / f"v4_tabm_enhanced_successcall_all_{year}.npz")
        candidate = load_npz(
            PREDICTIONS / f"v5_tabm_successcall_conditional_tm_dev2223_{year}.npz"
        )
        exact_c = load_npz(PREDICTIONS / f"v3_sparse_c_backtest_{year}.npz")
        identity = load_npz(PREDICTIONS / f"v5_honest_m3_r_identity_{year}.npz")
        grid = load_npz(PREDICTIONS / f"v5_honest_m3_r_grid_{year}.npz")
        for label, artifact in (
            ("candidate", candidate),
            ("exact_c", exact_c),
            ("identity", identity),
            ("grid", grid),
        ):
            ensure_aligned(parent, artifact, f"{label}/{year}")
        rows = route.iloc[parent["row_index"].astype(np.int64)]
        if not bool(rows["season"].eq(year).all()):
            raise ValueError(f"Season alignment mismatch for {year}")
        mask = rows["game_type"].eq("R").to_numpy(dtype=bool)
        y = parent["y"].astype(np.float64)
        prediction = candidate["tabm_outcome"].astype(np.float64)
        comparisons = {
            "vs_exact_tabm_parent_r": score_gain_interval(
                y, parent["tabm_outcome"], prediction, parent["cluster"], mask,
                seed=20260821 + year,
            ),
            "vs_exact_c_r": score_gain_interval(
                y, exact_c["catboost_outcome"], prediction, parent["cluster"], mask,
                seed=20261821 + year,
            ),
            "vs_honest_identity_r": score_gain_interval(
                y, identity["final_prediction"], prediction, parent["cluster"], mask,
                seed=20262821 + year,
            ),
            "vs_honest_grid_r": score_gain_interval(
                y, grid["final_prediction"], prediction, parent["cluster"], mask,
                seed=20263821 + year,
            ),
        }
        folds[str(year)] = {
            "r_rows": int(mask.sum()),
            "f_rows_excluded": int((~mask).sum()),
            "comparisons": comparisons,
        }

    parent_point = all(
        folds[str(year)]["comparisons"]["vs_exact_tabm_parent_r"]["point_gain"] > 0.0
        for year in YEARS
    )
    parent_lower = all(
        folds[str(year)]["comparisons"]["vs_exact_tabm_parent_r"]["lower_95"] > 0.0
        for year in YEARS
    )
    anchors_point = all(
        folds[str(year)]["comparisons"][name]["point_gain"] > 0.0
        for year in YEARS
        for name in ("vs_exact_c_r", "vs_honest_identity_r", "vs_honest_grid_r")
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
        },
        "preregister": str(PREREG.relative_to(ROOT)),
        "preregister_sha256": sha256(PREREG),
        "folds": folds,
        "gate": {
            "positive_exact_parent_point_both": bool(parent_point),
            "positive_exact_parent_lower_both": bool(parent_lower),
            "positive_exact_c_and_honest_anchor_points_both": bool(anchors_point),
            "passed": bool(passed),
        },
        "next_action": (
            "Freeze this exact recipe and run 2024 once."
            if passed else "Reject without running 2024."
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
