#!/usr/bin/env python3
"""Freeze the first 2022-supported meta stack that crosses expected LB 1190.

The seven directions were selected in a forward diagnostic over candidates
that already had aligned 2022 and 2024 outer-OOF predictions.  This script does
not rescan the catalog: it freezes those sources, refits their bounded linear
coefficients on the documented 2024 development fold, and applies the exact
same coefficients to 2022 for the safety gate.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import lsq_linear


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.analyze_v4_temporal_residual_ridge import (  # noqa: E402
    MEDIAN_OFFSET,
    REQUIRED_LOCAL,
    json_safe,
    score,
)
from experiments.run_v2_rolling import paired_bootstrap_brier_ci  # noqa: E402


PRED = ROOT / "experiments/results/predictions"
REPORT = ROOT / "experiments/results/v4_supported_meta_stack.json"
MAX_2022_BRIER_WORSENING = 0.0005
BOUNDS = (-0.30, 0.30)

ARMS = (
    {
        "name": "current_state_binary",
        "family": "catboost_current_state",
        "2022": ("v4_current_state_binary_support22_2022.npz", "catboost"),
        "2024": ("v4_current_state_binary_2024.npz", "catboost"),
    },
    {
        "name": "f_only_outcome_expert",
        "family": "catboost_regime_expert",
        "2022": ("v4_regime_expert_f_all_support22_2022.npz", "catboost_outcome"),
        "2024": ("v4_regime_expert_f_all_2024.npz", "catboost_outcome"),
    },
    {
        "name": "count_0_2_outcome_expert",
        "family": "catboost_count_expert",
        "2022": ("v4_count_expert_0_2_support22_2022.npz", "catboost_outcome"),
        "2024": ("v4_count_expert_0_2_2024.npz", "catboost_outcome"),
    },
    {
        "name": "tabm_rfit_outcome",
        "family": "tabm",
        "2022": ("v4_tabm_enhanced_rfit_all_2022.npz", "tabm_outcome"),
        "2024": ("v4_tabm_enhanced_rfit_all_2024.npz", "tabm_outcome"),
    },
    {
        "name": "numeric_cat_no_current",
        "family": "catboost_numeric",
        "2022": ("v4_numeric_cat_nocurrent_tmctx_seed42_2022.npz", "catboost_numeric"),
        "2024": ("v4_numeric_cat_nocurrent_tmctx_seed42_2024.npz", "catboost_numeric"),
    },
    {
        "name": "component15_success_strike",
        "family": "catboost_component15",
        "2022": (
            "v4_outcome_component15_current_primary_2022.npz",
            "catboost_outcome__p_13_success_r0m0b0s1",
        ),
        "2024": (
            "v4_outcome_component15_current_primary_2024.npz",
            "catboost_outcome__p_13_success_r0m0b0s1",
        ),
    },
    {
        "name": "coarse_wide_component",
        "family": "catboost_coarse_outcome",
        "2022": ("v4_outcome_a_components_2022.npz", "catboost_outcome__p_3_wide"),
        "2024": ("v4_outcome_a_components_2024.npz", "catboost_outcome__p_3_wide"),
    },
)


def load(path: Path) -> dict[str, np.ndarray]:
    with np.load(path) as archive:
        return {key: np.asarray(archive[key]) for key in archive.files}


def metrics(y: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    return score(y, np.clip(prediction, 0.0, 1.0))


def main() -> None:
    base_artifact = {
        year: load(PRED / f"v4_failure_decomp_blend_{year}.npz")
        for year in (2022, 2024)
    }
    y = {year: base_artifact[year]["y"].astype(np.float64) for year in base_artifact}
    base = {
        year: base_artifact[year]["final_prediction"].astype(np.float64)
        for year in base_artifact
    }
    directions: dict[int, list[np.ndarray]] = {2022: [], 2024: []}
    source_rows: list[dict[str, object]] = []
    for arm in ARMS:
        source_row: dict[str, object] = {
            "name": arm["name"],
            "family": arm["family"],
            "sources": {},
        }
        for year in (2022, 2024):
            filename, key = arm[str(year)]
            artifact = load(PRED / filename)
            if not np.array_equal(
                artifact["row_index"], base_artifact[year]["row_index"]
            ):
                raise ValueError(f"row_index mismatch: {filename}")
            prediction = artifact[key].astype(np.float64)
            if not (
                prediction.ndim == 1
                and len(prediction) == len(y[year])
                and np.isfinite(prediction).all()
                and float(prediction.min()) >= 0.0
                and float(prediction.max()) <= 1.0
            ):
                raise ValueError(f"invalid probability prediction: {filename}::{key}")
            directions[year].append(prediction - base[year])
            source_row["sources"][str(year)] = {"file": filename, "key": key}
        source_rows.append(source_row)

    design24 = np.column_stack(directions[2024])
    fit = lsq_linear(
        design24,
        y[2024] - base[2024],
        bounds=BOUNDS,
        method="bvls",
        tol=1e-12,
        max_iter=1000,
    )
    if not fit.success:
        raise RuntimeError(fit.message)
    coefficients = fit.x.astype(np.float64)
    final = {
        2024: np.clip(base[2024] + design24 @ coefficients, 0.0, 1.0),
        2022: np.clip(
            base[2022] + np.column_stack(directions[2022]) @ coefficients,
            0.0,
            1.0,
        ),
    }
    base_metrics = {year: metrics(y[year], base[year]) for year in base}
    final_metrics = {year: metrics(y[year], final[year]) for year in final}
    brier_delta22 = float(final_metrics[2022]["brier"] - base_metrics[2022]["brier"])
    intervals = {
        year: paired_bootstrap_brier_ci(
            y[year].astype(np.int8),
            base[year],
            final[year],
            iterations=2000,
            seed=20260821,
            clusters=base_artifact[year]["cluster"],
        )
        for year in (2022, 2024)
    }

    artifacts: dict[int, str] = {}
    for year in (2022, 2024):
        path = PRED / f"v4_supported_meta_stack_{year}.npz"
        payload: dict[str, np.ndarray] = {
            "y": y[year],
            "row_index": base_artifact[year]["row_index"],
            "cluster": base_artifact[year]["cluster"],
            "base": base[year],
            "final_prediction": final[year],
        }
        for index, direction in enumerate(directions[year]):
            payload[f"direction_{index:02d}"] = direction
        np.savez_compressed(path, **payload)
        artifacts[year] = str(path.relative_to(ROOT))

    final_score = float(final_metrics[2024]["raw_competition_score"])
    report = {
        "protocol": {
            "official_train_only": True,
            "test_rows_read": False,
            "row_independent_inference": True,
            "outer_oof_training": "season strictly before target",
            "primary_development_fold": 2024,
            "support_fold": 2022,
            "2023_role": "record-only due documented F-label regime discontinuity",
            "coefficient_bounds": list(BOUNDS),
            "max_2022_brier_worsening": MAX_2022_BRIER_WORSENING,
            "selection_note": "first forward-supported stack crossing required local score",
        },
        "arms": source_rows,
        "coefficients": {
            arm["name"]: float(value) for arm, value in zip(ARMS, coefficients)
        },
        "negative_coefficient_warning": [
            arm["name"] for arm, value in zip(ARMS, coefficients) if value < 0.0
        ],
        "base_metrics": base_metrics,
        "final_metrics": final_metrics,
        "gain_2024": final_score - float(base_metrics[2024]["raw_competition_score"]),
        "brier_delta_2022": brier_delta22,
        "passes_2022_safety": brier_delta22 <= MAX_2022_BRIER_WORSENING,
        "paired_cluster_bootstrap": intervals,
        "expected_lb_median": final_score + MEDIAN_OFFSET,
        "required_local_score": REQUIRED_LOCAL,
        "crosses_required_local_score": final_score > REQUIRED_LOCAL,
        "prediction_artifacts": artifacts,
        "risk": (
            "Seven-arm 2024-selected meta fit with signed coefficients; expected LB is a "
            "fixed planning estimate, not a leaderboard guarantee."
        ),
    }
    REPORT.write_text(
        json.dumps(json_safe(report), ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(json_safe(report), ensure_ascii=False, indent=2), flush=True)
    print(f"Saved {REPORT}", flush=True)


if __name__ == "__main__":
    main()
