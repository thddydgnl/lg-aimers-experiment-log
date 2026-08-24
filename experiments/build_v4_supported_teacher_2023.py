#!/usr/bin/env python3
"""Apply frozen V4 meta coefficients to the missing 2023 OOF fold.

No coefficient is fitted here.  The failure blend and seven-arm coefficients
are copied verbatim from the previously frozen 2024 development reports.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.analyze_v4_temporal_residual_ridge import json_safe, score  # noqa: E402


PRED = ROOT / "experiments/results/predictions"
REPORT = ROOT / "experiments/results/v4_supported_teacher_2023.json"
FAILURE_WEIGHT = 0.05193190904632398
COEFFICIENTS = np.asarray(
    [
        -0.1637719665469707,
        0.07834527193698473,
        -0.10015256781489247,
        0.15520969234361248,
        0.1322910876495326,
        -0.06116162445919242,
        0.036385254422802354,
    ],
    dtype=np.float64,
)
SOURCES = (
    ("v4_current_state_binary_support23_2023.npz", "catboost"),
    ("v4_regime_expert_f_all_support23_2023.npz", "catboost_outcome"),
    ("v4_count_expert_0_2_support23_exact_2023.npz", "catboost_outcome"),
    ("v4_tabm_enhanced_rfit_all_2023.npz", "tabm_outcome"),
    ("v4_numeric_cat_nocurrent_tmctx_seed42_2023.npz", "catboost_numeric"),
    (
        "v4_outcome_component15_current_support23_2023.npz",
        "catboost_outcome__p_13_success_r0m0b0s1",
    ),
    ("v4_outcome_a_components_2023.npz", "catboost_outcome__p_3_wide"),
)


def load(name: str) -> dict[str, np.ndarray]:
    with np.load(PRED / name) as archive:
        return {key: np.asarray(archive[key]) for key in archive.files}


def logit(probability: np.ndarray) -> np.ndarray:
    value = np.clip(probability, 1e-6, 1.0 - 1e-6)
    return np.log(value / (1.0 - value))


def sigmoid(value: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(value, -30.0, 30.0)))


def main() -> None:
    anchor = load("v4_post4_c3_axis_screen_2023.npz")
    failure = load("v4_failure_decomp_current_support23_2023.npz")
    row_index = anchor["row_index"]
    if not np.array_equal(row_index, failure["row_index"]):
        raise ValueError("failure row alignment mismatch")
    y = anchor["y"].astype(np.float64)
    pre_failure = anchor["selected_prediction_plus_tabtransformer"].astype(np.float64)
    all_failure = failure["catboost_failure_decomp"].astype(np.float64)
    failure_base = sigmoid(
        (1.0 - FAILURE_WEIGHT) * logit(pre_failure)
        + FAILURE_WEIGHT * logit(all_failure)
    )

    directions = []
    source_rows = []
    for filename, key in SOURCES:
        artifact = load(filename)
        if not np.array_equal(row_index, artifact["row_index"]):
            raise ValueError(f"row alignment mismatch: {filename}")
        prediction = artifact[key].astype(np.float64)
        directions.append(prediction - failure_base)
        source_rows.append({"file": filename, "key": key})
    design = np.column_stack(directions)
    final = np.clip(failure_base + design @ COEFFICIENTS, 0.0, 1.0)

    failure_path = PRED / "v4_failure_decomp_blend_2023.npz"
    np.savez_compressed(
        failure_path,
        y=y,
        row_index=row_index,
        cluster=anchor["cluster"],
        base=pre_failure,
        final_prediction=failure_base,
        all_failure=all_failure,
    )
    final_path = PRED / "v4_supported_meta_stack_2023.npz"
    payload = {
        "y": y,
        "row_index": row_index,
        "cluster": anchor["cluster"],
        "base": failure_base,
        "final_prediction": final,
    }
    payload.update({f"direction_{i:02d}": value for i, value in enumerate(directions)})
    np.savez_compressed(final_path, **payload)

    report = {
        "protocol": {
            "coefficients_fitted_here": False,
            "outer_oof_training": "season strictly before 2023",
            "official_train_only": True,
            "test_rows_read": False,
            "row_independent_inference": True,
        },
        "failure_weight": FAILURE_WEIGHT,
        "coefficients": COEFFICIENTS.tolist(),
        "sources": source_rows,
        "pre_failure_metrics": score(y, pre_failure),
        "failure_base_metrics": score(y, failure_base),
        "final_metrics": score(y, final),
        "artifacts": {
            "failure": str(failure_path.relative_to(ROOT)),
            "supported": str(final_path.relative_to(ROOT)),
        },
    }
    REPORT.write_text(
        json.dumps(json_safe(report), ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(json_safe(report), ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
