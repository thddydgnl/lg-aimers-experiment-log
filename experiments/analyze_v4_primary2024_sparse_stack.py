#!/usr/bin/env python3
"""Sparse 2024-primary stacking with the documented 2022 safety gate.

The 2024 labels are a development/meta-fit fold for the 2025 hidden test, as
specified by EXPERIMENT_PLAN_V3.  Candidate predictions remain outer-OOF
(trained on season < target).  The same fitted coefficients must keep the 2022
Brier deterioration at or below 0.0005.  At most one arm per family and three
arms total are admitted to limit winner's curse.
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


PRED = ROOT / "experiments/results/predictions"
CATALOG = ROOT / "experiments/results/v4_oof_direction_catalog.json"
REPORT = ROOT / "experiments/results/v4_primary2024_sparse_stack.json"
MAX_ARMS = 3
MAX_2022_BRIER_WORSENING = 0.0005


def load(path: Path) -> dict[str, np.ndarray]:
    with np.load(path) as archive:
        return {key: np.asarray(archive[key]) for key in archive.files}


def metrics(y: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    return score(y, np.clip(prediction, 0.0, 1.0))


def fit_scalar(direction: np.ndarray, residual: np.ndarray) -> tuple[float, float]:
    denominator = float(np.dot(direction, direction))
    raw = float(np.dot(direction, residual) / denominator) if denominator else 0.0
    return raw, float(np.clip(raw, 0.0, 1.0))


def family(stem: str) -> str:
    lowered = stem.lower()
    if "lgbm" in lowered:
        return "lightgbm"
    if "numeric_cat" in lowered:
        return "catboost_current_state"
    if "outcome" in lowered:
        return "catboost_outcome"
    if "tabm" in lowered:
        return "tabm"
    if "pitchtype" in lowered:
        return "pitchtype_prior"
    return stem.split("_")[1] if "_" in stem else stem


def prediction_like(values: np.ndarray) -> bool:
    return bool(
        values.ndim == 1
        and np.isfinite(values).all()
        and float(values.min()) >= 0.0
        and float(values.max()) <= 1.0
        and 0.15 <= float(values.mean()) <= 0.85
        and 0.005 <= float(values.std()) <= 0.25
    )


def bounded_fit(design: np.ndarray, residual: np.ndarray) -> np.ndarray:
    result = lsq_linear(
        design,
        residual,
        bounds=(np.zeros(design.shape[1]), np.ones(design.shape[1])),
        method="bvls",
        tol=1e-10,
        max_iter=500,
    )
    if not result.success:
        raise RuntimeError(result.message)
    return result.x.astype(np.float64)


def main() -> None:
    catalog = json.loads(CATALOG.read_text(encoding="utf-8"))
    accepted = {
        year: load(PRED / f"v4_routed_tabm_stack_locked_{year}.npz")
        for year in (2022, 2024)
    }
    accepted_prediction = {
        year: accepted[year]["routed_tabm_stack"].astype(np.float64)
        for year in accepted
    }
    y = {year: accepted[year]["y"].astype(np.float64) for year in accepted}
    route_r = {year: accepted[year]["game_type_r"].astype(bool) for year in accepted}
    latest = load(PRED / "v4_post4_c3_axis_screen_2024.npz")
    base = latest["selected_prediction_plus_tabtransformer"].astype(np.float64)
    base_metrics_2024 = metrics(y[2024], base)
    support_metrics_2022 = metrics(y[2022], accepted_prediction[2022])

    rows_by_name: dict[str, dict[str, object]] = {}
    for row in catalog["top_screened"]:
        if (
            row["route"] == "R"
            and bool(row["coefficient_stable"])
            and float(row["gain_fit_2022"]) > 0.05
        ):
            rows_by_name.setdefault(f"{row['stem']}::{row['key']}", row)

    raw_candidates: dict[str, dict[str, object]] = {}
    failures: list[dict[str, str]] = []
    for short_name, row in rows_by_name.items():
        try:
            artifacts = {
                year: load(PRED / f"{row['stem']}_{year}.npz")
                for year in (2022, 2024)
            }
            for year in artifacts:
                if not np.array_equal(
                    artifacts[year]["row_index"], accepted[year]["row_index"]
                ):
                    raise ValueError(f"row_index mismatch for {year}")
                if not prediction_like(artifacts[year][row["key"]]):
                    raise ValueError(f"key is not a full probability prediction for {year}")
            raw_candidates[short_name] = {
                "stem": row["stem"],
                "key": row["key"],
                "family": family(str(row["stem"])),
                "historical_gain_fit_2022": float(row["gain_fit_2022"]),
                "historical_transfer_gain_2023": float(row["transfer_gain_2023"]),
                "prediction": {
                    year: artifacts[year][row["key"]].astype(np.float64)
                    for year in artifacts
                },
            }
        except Exception as exc:
            failures.append({
                "candidate": short_name,
                "exception": type(exc).__name__,
                "message": str(exc),
            })

    # Add the recency arm that was trained after the original catalog.
    recent = {
        2022: load(PRED / "v4_numeric_cat_ctxlvl_tm_rfit_recent3_oof_2022.npz"),
        2024: load(PRED / "v4_numeric_cat_ctxlvl_tm_rfit_recent3_confirm_2024.npz"),
    }
    raw_candidates["recent_r_cat::catboost_numeric"] = {
        "stem": "v4_numeric_cat_ctxlvl_tm_rfit_recent3",
        "key": "catboost_numeric",
        "family": "catboost_recent_r",
        "historical_gain_fit_2022": 6.900334307591038,
        "historical_transfer_gain_2023": 8.292885783856036,
        "prediction": {
            year: recent[year]["catboost_numeric"].astype(np.float64)
            for year in recent
        },
    }

    for stem, key, candidate_family in (
        ("v4_xgb_ctxlvl_tm_primary", "xgboost", "xgboost"),
        ("v4_catbrier_ctxlvl_tm_primary", "catboost_brier", "catboost_brier"),
        (
            "v4_outcome_component15_current_primary",
            "catboost_outcome",
            "catboost_failure_components",
        ),
    ):
        artifacts = {
            year: load(PRED / f"{stem}_{year}.npz") for year in (2022, 2024)
        }
        raw_candidates[f"{stem}::{key}"] = {
            "stem": stem,
            "key": key,
            "family": candidate_family,
            "historical_gain_fit_2022": None,
            "historical_transfer_gain_2023": None,
            "prediction": {
                year: artifacts[year][key].astype(np.float64) for year in artifacts
            },
        }

    individual: dict[str, dict[str, object]] = {}
    directions: dict[str, dict[int, np.ndarray]] = {}
    for name, candidate in raw_candidates.items():
        predictions = candidate.pop("prediction")
        direction = {
            year: np.where(
                route_r[year], predictions[year] - accepted_prediction[year], 0.0
            )
            for year in (2022, 2024)
        }
        gamma_raw, gamma = fit_scalar(direction[2024], y[2024] - base)
        prediction24 = np.clip(base + gamma * direction[2024], 0.0, 1.0)
        support22 = np.clip(
            accepted_prediction[2022] + gamma * direction[2022], 0.0, 1.0
        )
        metric24 = metrics(y[2024], prediction24)
        metric22 = metrics(y[2022], support22)
        gain24 = (
            float(metric24["raw_competition_score"])
            - float(base_metrics_2024["raw_competition_score"])
        )
        brier_delta22 = float(metric22["brier"] - support_metrics_2022["brier"])
        individual[name] = {
            **candidate,
            "gamma_fit_2024_raw": gamma_raw,
            "gamma_fit_2024": gamma,
            "gain_2024": gain24,
            "score_2024": float(metric24["raw_competition_score"]),
            "brier_delta_2022": brier_delta22,
            "score_gain_2022": (
                float(metric22["raw_competition_score"])
                - float(support_metrics_2022["raw_competition_score"])
            ),
            "passes_2022_safety": brier_delta22 <= MAX_2022_BRIER_WORSENING,
            "passes_primary_gate": bool(
                gamma > 0.0
                and gain24 > 0.05
                and brier_delta22 <= MAX_2022_BRIER_WORSENING
            ),
        }
        directions[name] = direction

    ranked = sorted(
        (name for name, row in individual.items() if row["passes_primary_gate"]),
        key=lambda name: float(individual[name]["gain_2024"]),
        reverse=True,
    )

    selected: list[str] = []
    selected_families: set[str] = set()
    coefficients = np.empty(0, dtype=np.float64)
    current_score = float(base_metrics_2024["raw_competition_score"])
    greedy_trace: list[dict[str, object]] = []
    for name in ranked:
        candidate_family = str(individual[name]["family"])
        if candidate_family in selected_families:
            continue
        candidate_direction = directions[name][2024]
        if any(
            abs(float(np.corrcoef(candidate_direction, directions[prior][2024])[0, 1]))
            >= 0.985
            for prior in selected
        ):
            continue
        trial = [*selected, name]
        design24 = np.column_stack([directions[item][2024] for item in trial])
        trial_coefficients = bounded_fit(design24, y[2024] - base)
        prediction24 = np.clip(base + design24 @ trial_coefficients, 0.0, 1.0)
        score24 = float(metrics(y[2024], prediction24)["raw_competition_score"])
        design22 = np.column_stack([directions[item][2022] for item in trial])
        prediction22 = np.clip(
            accepted_prediction[2022] + design22 @ trial_coefficients, 0.0, 1.0
        )
        metric22 = metrics(y[2022], prediction22)
        brier_delta22 = float(metric22["brier"] - support_metrics_2022["brier"])
        accepted_trial = bool(
            score24 - current_score > 0.10
            and brier_delta22 <= MAX_2022_BRIER_WORSENING
        )
        greedy_trace.append({
            "candidate": name,
            "family": candidate_family,
            "trial_coefficients": dict(zip(trial, trial_coefficients.tolist())),
            "incremental_gain_2024": score24 - current_score,
            "total_gain_2024": (
                score24 - float(base_metrics_2024["raw_competition_score"])
            ),
            "brier_delta_2022": brier_delta22,
            "accepted": accepted_trial,
        })
        if accepted_trial:
            selected = trial
            selected_families.add(candidate_family)
            coefficients = trial_coefficients
            current_score = score24
            if len(selected) >= MAX_ARMS:
                break

    if selected:
        final_design24 = np.column_stack([directions[name][2024] for name in selected])
        final_prediction24 = np.clip(base + final_design24 @ coefficients, 0.0, 1.0)
        final_design22 = np.column_stack([directions[name][2022] for name in selected])
        final_prediction22 = np.clip(
            accepted_prediction[2022] + final_design22 @ coefficients, 0.0, 1.0
        )
    else:
        final_prediction24 = base.copy()
        final_prediction22 = accepted_prediction[2022].copy()
    final_metrics = {
        2022: metrics(y[2022], final_prediction22),
        2024: metrics(y[2024], final_prediction24),
    }

    artifacts: dict[int, str] = {}
    for year, prediction, base_prediction in (
        (2022, final_prediction22, accepted_prediction[2022]),
        (2024, final_prediction24, base),
    ):
        path = PRED / f"v4_primary2024_sparse_stack_{year}.npz"
        payload: dict[str, np.ndarray] = {
            "y": y[year],
            "row_index": accepted[year]["row_index"],
            "cluster": accepted[year]["cluster"],
            "base": base_prediction,
            "final_prediction": prediction,
        }
        for index, name in enumerate(selected):
            payload[f"direction_{index:02d}"] = directions[name][year]
        np.savez_compressed(path, **payload)
        artifacts[year] = str(path.relative_to(ROOT))

    final_score = float(final_metrics[2024]["raw_competition_score"])
    report = {
        "protocol": {
            "official_train_only": True,
            "test_rows_read": False,
            "outer_oof_training": "season strictly before target",
            "primary_development_fold": 2024,
            "support_fold": 2022,
            "2023_role": "record only due documented F label regime break",
            "max_2022_brier_worsening": MAX_2022_BRIER_WORSENING,
            "max_arms": MAX_ARMS,
            "one_arm_per_family": True,
            "correlation_cap": 0.985,
            "coefficient_bounds": [0.0, 1.0],
        },
        "candidate_count": len(individual),
        "load_failures": failures,
        "base_metrics": {"2024": base_metrics_2024, "2022_proxy": support_metrics_2022},
        "individual_ranked": [
            {"name": name, **individual[name]}
            for name in sorted(
                individual,
                key=lambda item: float(individual[item]["gain_2024"]),
                reverse=True,
            )
        ],
        "greedy_trace": greedy_trace,
        "selected": selected,
        "coefficients": dict(zip(selected, coefficients.tolist())),
        "final_metrics": final_metrics,
        "gain_2024": final_score - float(base_metrics_2024["raw_competition_score"]),
        "brier_delta_2022": (
            float(final_metrics[2022]["brier"])
            - float(support_metrics_2022["brier"])
        ),
        "expected_lb_median": final_score + MEDIAN_OFFSET,
        "required_local_score": REQUIRED_LOCAL,
        "crosses_required_local_score": final_score > REQUIRED_LOCAL,
        "prediction_artifacts": artifacts,
        "warning": "2024 meta-fit score is exploratory and subject to winner's curse.",
    }
    REPORT.write_text(
        json.dumps(json_safe(report), ensure_ascii=False, indent=2), encoding="utf-8"
    )
    summary = {
        "candidate_count": report["candidate_count"],
        "selected": selected,
        "coefficients": report["coefficients"],
        "base_2024": base_metrics_2024["raw_competition_score"],
        "final_2024": final_score,
        "gain_2024": report["gain_2024"],
        "brier_delta_2022": report["brier_delta_2022"],
        "expected_lb_median": report["expected_lb_median"],
        "crosses_required_local_score": report["crosses_required_local_score"],
        "top_individual": report["individual_ranked"][:10],
    }
    print(json.dumps(json_safe(summary), ensure_ascii=False, indent=2), flush=True)
    print(f"Saved {REPORT}", flush=True)


if __name__ == "__main__":
    main()
