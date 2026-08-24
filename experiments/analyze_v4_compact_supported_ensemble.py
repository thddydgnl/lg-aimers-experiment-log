#!/usr/bin/env python3
"""Freeze and verify the compact, submission-oriented V4 ensemble.

The coefficients below were selected once on the 2024 development fold with
bounded least squares.  Earlier folds are diagnostics only: the deployment
student did not yet have enough prior supported-teacher seasons there, so its
base is conservatively replaced by the frozen V3 anchor.  Every component
prediction is outer-time and produced from official data only.
"""

from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.run_e14_rolling import metric  # noqa: E402


PRED = ROOT / "experiments/results/predictions"
OUTPUT_JSON = ROOT / "experiments/results/v4_compact_supported_ensemble.json"

LB_OFFSET = 140.1475834416
TARGET_EXPECTED = 1190.0
V3_STAGE = "v3_sparse_m3_frozen"
V3_KEY = "final_prediction"
STUDENT_STAGE = "v4_teacher_residual_centered_r_primary24"
STUDENT_KEY = "catboost_teacher"

ARMS = [
    ("v4_outcome_context_publicparam_primary24", "catboost_numeric", 0.5),
    ("v3_outcome_e14k50_batter80_middle100_dropseason", "catboost_outcome", -0.22535351311965496),
    ("v4_regime_expert_f_all", "catboost_outcome", 0.056220095242924296),
    ("v4_count_expert_0_2", "catboost_outcome", -0.07950414401143537),
    ("v3_outcome_rev_count", "catboost_outcome", 0.350754857696783),
    ("v4_outcome_ova", "catboost_outcome", -0.4829564100633987),
    ("v4_outcome_all_call_components", "catboost_outcome", 0.3425253994993398),
    ("v4_current_state_c", "catboost_outcome", -0.32887462171312243),
    ("v3_outcome_rev_e14multi", "catboost_outcome", -0.2695523527411383),
    ("v4_recent_form", "catboost_outcome", 0.4196925584812542),
    ("v4_outcome_balance_latest_type", "catboost_outcome", -0.2328937579626719),
    ("v3_outcome_e14k50_batter80", "catboost_outcome", 0.4546705925200295),
    ("v3_outcome_batter80_middle500", "catboost_outcome", -0.3291972582518642),
    ("v4_outcome_trackman_count_k200", "catboost_outcome", -0.2490291886699021),
    ("v4_numeric_cat_nocurrent_tmctx_seed42", "catboost_numeric", 0.12629870900520043),
    ("v3_catboost_platoon_cfg01", "catboost", -0.12653466013330458),
    ("v3_outcome_trackman_w2_e14k50_batter80_middle100", "catboost_outcome", -0.3069577878455414),
    ("v3_outcome_trackman_e14k80_batter80_middle100", "catboost_outcome", 0.34530933248020573),
]

# Backfills were intentionally written to a separate stage, leaving the
# original 2024 artifacts immutable.  Two stages already contained all folds.
ORIGINAL_SUPPORT = {
    "v4_outcome_all_call_components",
    "v4_numeric_cat_nocurrent_tmctx_seed42",
}


def load(stage: str, year: int, key: str) -> dict[str, np.ndarray]:
    path = PRED / f"{stage}_{year}.npz"
    if not path.is_file():
        raise FileNotFoundError(path)
    with np.load(path) as archive:
        if key not in archive.files:
            raise KeyError(f"{path.name}: {key}; available={archive.files}")
        return {
            "y": np.asarray(archive["y"], dtype=np.float64),
            "row_index": np.asarray(archive["row_index"]),
            "prediction": np.asarray(archive[key], dtype=np.float64),
        }


def arm_stage(stage: str, year: int) -> str:
    if year == 2024 or stage in ORIGINAL_SUPPORT:
        return stage
    return f"{stage}_support2223"


def main() -> None:
    fold_results: dict[str, dict] = {}
    output_payload: dict[int, dict[str, np.ndarray]] = {}
    for year in (2022, 2023, 2024):
        anchor = load(V3_STAGE, year, V3_KEY)
        if year == 2024:
            student = load(STUDENT_STAGE, year, STUDENT_KEY)
            if not np.array_equal(anchor["row_index"], student["row_index"]):
                raise ValueError("Student/anchor row mismatch")
            base = student["prediction"]
            base_name = f"{STUDENT_STAGE}:{STUDENT_KEY}"
        else:
            base = anchor["prediction"].copy()
            base_name = f"{V3_STAGE}:{V3_KEY} (conservative diagnostic base)"

        correction = np.zeros_like(base)
        for stage, key, coefficient in ARMS:
            item = load(arm_stage(stage, year), year, key)
            if not np.array_equal(anchor["row_index"], item["row_index"]):
                raise ValueError(f"Row mismatch: {stage}/{year}")
            if not np.array_equal(anchor["y"], item["y"]):
                raise ValueError(f"Target mismatch: {stage}/{year}")
            correction += coefficient * (item["prediction"] - anchor["prediction"])

        candidate = np.clip(base + correction, 1e-6, 1.0 - 1e-6)
        candidate_metrics = metric(anchor["y"], candidate)
        anchor_metrics = metric(anchor["y"], anchor["prediction"])
        brier_delta = float(candidate_metrics["brier"] - anchor_metrics["brier"])
        fold_results[str(year)] = {
            "base": base_name,
            "rows": int(len(candidate)),
            "candidate": candidate_metrics,
            "v3_anchor": anchor_metrics,
            "brier_delta_vs_v3": brier_delta,
            "earlier_fold_safety_pass": bool(year == 2024 or brier_delta <= 0.0005),
        }
        output_payload[year] = {
            "y": anchor["y"],
            "row_index": anchor["row_index"],
            "v3_anchor": anchor["prediction"],
            "base": base,
            "correction": correction,
            "final_prediction": candidate,
        }

    local_score = float(fold_results["2024"]["candidate"]["competition_score"])
    expected = local_score + LB_OFFSET
    report = {
        "candidate": "V4_compact_supported_1193",
        "selection_protocol": {
            "coefficient_selection_fold": 2024,
            "coefficient_bound": [-0.5, 0.5],
            "component_predictions": "strict outer-time predictions",
            "official_data_only": True,
            "test_aggregate_usage": False,
            "earlier_fold_role": "safety diagnostics only",
        },
        "formula": "clip(student + sum(coef_i * (arm_i - v3_anchor)))",
        "student": f"{STUDENT_STAGE}:{STUDENT_KEY}",
        "anchor": f"{V3_STAGE}:{V3_KEY}",
        "arms": [
            {"stage": stage, "key": key, "coefficient": coefficient}
            for stage, key, coefficient in ARMS
        ],
        "folds": fold_results,
        "expected_score": {
            "fixed_formula": "2024_local_score + 140.1475834416",
            "local_2024": local_score,
            "fixed_offset": LB_OFFSET,
            "expected_public_private": expected,
            "target": TARGET_EXPECTED,
            "target_pass": bool(expected > TARGET_EXPECTED),
        },
        "deployment": {
            "status": "awaiting full-history refit and package verification",
            "models": 22,
            "model_breakdown": "3 V3 anchor + 1 teacher-residual student + 18 arms",
        },
    }

    for year, payload in output_payload.items():
        path = PRED / f"v4_compact_supported_ensemble_{year}.npz"
        np.savez_compressed(path, **payload)
    OUTPUT_JSON.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    print(json.dumps(report["expected_score"], ensure_ascii=False, indent=2))
    for year, item in fold_results.items():
        print(
            year,
            f"score={item['candidate']['competition_score']:.9f}",
            f"brier_delta_vs_v3={item['brier_delta_vs_v3']:+.12f}",
            f"safety={item['earlier_fold_safety_pass']}",
        )
    print(f"Saved {OUTPUT_JSON}")


if __name__ == "__main__":
    main()
