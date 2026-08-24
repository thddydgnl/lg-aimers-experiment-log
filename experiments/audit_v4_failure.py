#!/usr/bin/env python3
"""Audit why the V4 development estimate did not transfer to the leaderboard.

This script never reads test rows.  It only compares already frozen outer-time
OOF predictions and the scalar leaderboard scores supplied by the user.  The
main purpose is to make the V4 failure a permanent validation constraint for
V5 rather than another offset to tune around.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
from scipy.optimize import lsq_linear


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.analyze_v4_compact_supported_ensemble import (  # noqa: E402
    ARMS,
    STUDENT_KEY,
    STUDENT_STAGE,
    V3_KEY,
    V3_STAGE,
    arm_stage,
    load,
)


OUTPUT = ROOT / "experiments/results/v4_failure_audit.json"
V3_ACTUAL_LB = 1090.9100565103


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--v4-actual",
        type=float,
        default=1005.0,
        help=(
            "Scalar V4 leaderboard score.  The default is the user's rounded "
            "report; rerun with the exact score when it is available."
        ),
    )
    parser.add_argument("--output", type=Path, default=OUTPUT)
    return parser.parse_args()


def metrics(y: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    prediction = np.clip(np.asarray(prediction, dtype=np.float64), 1e-6, 1 - 1e-6)
    y = np.asarray(y, dtype=np.float64)
    rate = float(y.mean())
    reference = rate * (1.0 - rate)
    brier = float(np.mean(np.square(prediction - y)))
    raw_skill = 1.0 - brier / reference
    return {
        "rows": int(len(y)),
        "target_rate": rate,
        "prediction_mean": float(prediction.mean()),
        "prediction_std": float(prediction.std()),
        "brier": brier,
        "raw_skill": raw_skill,
        "competition_score": max(0.0, 100000.0 * raw_skill),
        "raw_competition_score": 100000.0 * raw_skill,
    }


def load_fold(year: int) -> dict[str, object]:
    anchor = load(V3_STAGE, year, V3_KEY)
    directions: list[np.ndarray] = []
    for stage, key, _ in ARMS:
        item = load(arm_stage(stage, year), year, key)
        if not np.array_equal(anchor["row_index"], item["row_index"]):
            raise ValueError(f"row mismatch: {stage}/{year}")
        if not np.array_equal(anchor["y"], item["y"]):
            raise ValueError(f"target mismatch: {stage}/{year}")
        directions.append(item["prediction"] - anchor["prediction"])
    return {
        "y": anchor["y"],
        "row_index": anchor["row_index"],
        "anchor": anchor["prediction"],
        "design": np.column_stack(directions),
    }


def fit_bounded(design: np.ndarray, residual: np.ndarray) -> np.ndarray:
    fit = lsq_linear(
        design,
        residual,
        bounds=(-0.5, 0.5),
        method="bvls",
        tol=1e-10,
        max_iter=1000,
    )
    if not fit.success:
        raise RuntimeError(fit.message)
    return np.asarray(fit.x, dtype=np.float64)


def score_with(
    fold: dict[str, object], coefficients: np.ndarray, base: np.ndarray | None = None
) -> dict[str, object]:
    y = np.asarray(fold["y"], dtype=np.float64)
    anchor = np.asarray(fold["anchor"], dtype=np.float64)
    design = np.asarray(fold["design"], dtype=np.float64)
    base_prediction = anchor if base is None else np.asarray(base, dtype=np.float64)
    prediction = np.clip(base_prediction + design @ coefficients, 1e-6, 1 - 1e-6)
    result = metrics(y, prediction)
    anchor_result = metrics(y, anchor)
    result["score_delta_vs_v3"] = (
        result["raw_competition_score"] - anchor_result["raw_competition_score"]
    )
    result["brier_delta_vs_v3"] = result["brier"] - anchor_result["brier"]
    return result


def design_diagnostics(design: np.ndarray) -> dict[str, float | int]:
    centered = design - design.mean(axis=0, keepdims=True)
    scale = centered.std(axis=0)
    nonconstant = scale > 0
    standardized = centered[:, nonconstant] / scale[nonconstant]
    correlation = np.corrcoef(standardized, rowvar=False)
    upper = np.abs(correlation[np.triu_indices_from(correlation, k=1)])
    # Work on the small Gram matrix instead of taking an SVD of all rows.
    gram = standardized.T @ standardized / max(1, len(standardized))
    eigenvalues = np.linalg.eigvalsh(gram)
    positive = eigenvalues[eigenvalues > 1e-10]
    condition = float(positive.max() / positive.min()) if len(positive) else float("inf")
    effective_rank = float(
        np.exp(
            -np.sum(
                (positive / positive.sum())
                * np.log(np.clip(positive / positive.sum(), 1e-15, None))
            )
        )
    )
    return {
        "rows": int(design.shape[0]),
        "columns": int(design.shape[1]),
        "matrix_rank": int(np.linalg.matrix_rank(gram, tol=1e-10)),
        "effective_rank": effective_rank,
        "max_abs_pairwise_correlation": float(upper.max()) if len(upper) else 0.0,
        "median_abs_pairwise_correlation": float(np.median(upper)) if len(upper) else 0.0,
        "condition_number": condition,
    }


def main() -> None:
    args = parse_args()
    folds = {year: load_fold(year) for year in (2022, 2023, 2024)}
    frozen = np.asarray([coefficient for _, _, coefficient in ARMS], dtype=np.float64)

    anchor_metrics = {
        str(year): metrics(fold["y"], fold["anchor"])
        for year, fold in folds.items()
    }
    frozen_on_anchor = {
        str(year): score_with(fold, frozen) for year, fold in folds.items()
    }

    student = load(STUDENT_STAGE, 2024, STUDENT_KEY)
    fold24 = folds[2024]
    if not np.array_equal(student["row_index"], fold24["row_index"]):
        raise ValueError("student/2024 row mismatch")
    student_metrics = metrics(fold24["y"], student["prediction"])
    final_on_student = score_with(fold24, frozen, base=student["prediction"])

    coefficients_by_fit: dict[str, list[float]] = {}
    transfer: dict[str, dict[str, object]] = {}
    for fit_year, fit_fold in folds.items():
        coefficients = fit_bounded(
            np.asarray(fit_fold["design"]),
            np.asarray(fit_fold["y"]) - np.asarray(fit_fold["anchor"]),
        )
        coefficients_by_fit[str(fit_year)] = coefficients.tolist()
        transfer[str(fit_year)] = {
            str(eval_year): score_with(eval_fold, coefficients)
            for eval_year, eval_fold in folds.items()
        }

    pooled_design = np.vstack([folds[2022]["design"], folds[2023]["design"]])
    pooled_residual = np.concatenate(
        [
            folds[year]["y"] - folds[year]["anchor"]
            for year in (2022, 2023)
        ]
    )
    pooled_coefficients = fit_bounded(pooled_design, pooled_residual)
    coefficients_by_fit["2022_2023_pooled"] = pooled_coefficients.tolist()
    transfer["2022_2023_pooled"] = {
        str(eval_year): score_with(eval_fold, pooled_coefficients)
        for eval_year, eval_fold in folds.items()
    }

    scalar_coefficients: dict[str, dict[str, float | bool]] = {}
    for index, (stage, key, frozen_coefficient) in enumerate(ARMS):
        row: dict[str, float | bool] = {"frozen": float(frozen_coefficient)}
        signs: list[int] = []
        for year, fold in folds.items():
            direction = np.asarray(fold["design"])[:, index]
            residual = np.asarray(fold["y"]) - np.asarray(fold["anchor"])
            denominator = float(direction @ direction)
            gamma = float(direction @ residual / denominator) if denominator else 0.0
            row[f"gamma_{year}"] = gamma
            signs.append(int(np.sign(gamma)))
        row["same_sign_2022_2024"] = bool(signs[0] == signs[2] and signs[0] != 0)
        scalar_coefficients[f"{stage}:{key}"] = row

    local_delta = (
        final_on_student["raw_competition_score"]
        - anchor_metrics["2024"]["raw_competition_score"]
    )
    actual_delta = float(args.v4_actual - V3_ACTUAL_LB)
    report = {
        "protocol": {
            "official_prediction_artifacts_only": True,
            "test_rows_read": False,
            "leaderboard_inputs_are_scalar_only": True,
            "v4_actual_source": "user-reported rounded value unless --v4-actual is supplied",
            "v4_actual_is_exact": bool(args.v4_actual != 1005.0),
        },
        "leaderboard": {
            "v3_actual": V3_ACTUAL_LB,
            "v4_actual": float(args.v4_actual),
            "actual_delta_v4_minus_v3": actual_delta,
        },
        "development_rank_reversal": {
            "v3_local_2024": anchor_metrics["2024"]["raw_competition_score"],
            "v4_local_2024": final_on_student["raw_competition_score"],
            "local_delta_v4_minus_v3": local_delta,
            "actual_delta_v4_minus_v3": actual_delta,
            "delta_error": actual_delta - local_delta,
            "obsolete_fixed_offset_v4_prediction": (
                final_on_student["raw_competition_score"] + 140.1475834416
            ),
            "obsolete_prediction_error": (
                args.v4_actual
                - (final_on_student["raw_competition_score"] + 140.1475834416)
            ),
        },
        "decomposition_2024": {
            "v3_anchor": anchor_metrics["2024"],
            "student_only": student_metrics,
            "frozen_18_arm_correction_on_v3": frozen_on_anchor["2024"],
            "student_plus_frozen_18_arms": final_on_student,
        },
        "frozen_coefficients": {
            f"{stage}:{key}": float(coefficient)
            for stage, key, coefficient in ARMS
        },
        "frozen_at_bound_count": int(np.sum(np.abs(frozen) >= 0.499999)),
        "design_diagnostics": {
            str(year): design_diagnostics(np.asarray(fold["design"]))
            for year, fold in folds.items()
        },
        "anchor_metrics": anchor_metrics,
        "frozen_18_arm_on_anchor": frozen_on_anchor,
        "coefficients_refit_by_fold": coefficients_by_fit,
        "cross_fold_transfer": transfer,
        "univariate_direction_stability": scalar_coefficients,
        "same_sign_2022_2024_count": int(
            sum(row["same_sign_2022_2024"] for row in scalar_coefficients.values())
        ),
        "conclusion": {
            "fixed_offset_valid": False,
            "v4_2024_meta_score_is_external_confirmation": False,
            "high_dimensional_meta_stack_requires_actual_lb_to_complete_goal": True,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    summary = {
        "leaderboard": report["leaderboard"],
        "development_rank_reversal": report["development_rank_reversal"],
        "frozen_at_bound_count": report["frozen_at_bound_count"],
        "same_sign_2022_2024_count": report["same_sign_2022_2024_count"],
        "transfer_to_2024": {
            fit_year: values["2024"]["score_delta_vs_v3"]
            for fit_year, values in transfer.items()
        },
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False))
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
