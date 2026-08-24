#!/usr/bin/env python3
"""Evaluate stored prediction and delta artifacts with their correct semantics.

Older generic catalog code treated every one-dimensional array as a complete
prediction.  This script explicitly distinguishes probabilities from additive
directions, screens each R-only direction on 2022 -> 2023, then refits its
scalar against the locked current stack on 2023 and confirms on 2024.
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np


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
REPORT = ROOT / "experiments/results/v4_semantic_direction_catalog.json"
OUTPUT = PRED / "v4_semantic_direction_catalog_2024.npz"
GAMMA_LIMIT = 4.0


@dataclass(frozen=True)
class Spec:
    name: str
    stems: dict[int, str]
    key: str
    kind: str  # prediction | delta
    column: int | None = None


def common(name: str, key: str, kind: str, column: int | None = None) -> Spec:
    return Spec(name, {year: name for year in (2022, 2023, 2024)}, key, kind, column)


SPECS = [
    common("v4_joint_neural_conservative", "joint_delta", "delta"),
    common("v4_joint_neural_conservative", "neural_delta", "delta"),
    common("v4_neural_resnet_delta", "neural_delta", "delta"),
    common("v4_neural_resnet_delta", "candidate", "prediction"),
    common("v4_outcome_component_reweight", "correction", "delta"),
    common("v4_pitchtype_failure_prior", "correction", "delta"),
    common("v4_pitchtype_failure_prior", "source_directions", "delta", 0),
    common("v4_pitchtype_failure_prior", "source_directions", "delta", 1),
    common("v4_pitchtype_failure_prior", "source_directions", "delta", 2),
    common("v4_pitchtype_failure_tagged_locked", "tagged_direction", "delta"),
    common("v4_routed_tabm_stack_locked", "tabm_direction", "delta"),
    common("v4_routed_tabm_stack_locked", "stability_b_direction", "delta"),
    common("v4_routed_tabm_stack_locked", "correction", "delta"),
    common("v4_residual_ensemble", "residual_ensemble", "prediction"),
    common("v4_tabm_binary_brier_enhanced_all", "tabm", "prediction"),
    common("v4_tabm_enhanced_all", "tabm_outcome", "prediction"),
    common("v4_tabm_enhanced_alltype_all", "tabm_outcome", "prediction"),
    common("v4_tabm_enhanced_rfit_all", "tabm_outcome", "prediction"),
    common("v4_tabm_enhanced_seed42_all", "tabm_outcome", "prediction"),
    common("v4_tabm_enhanced_successcall_all", "tabm_outcome", "prediction"),
    Spec(
        "v4_tabm_enhanced_decay65_alias",
        {
            2022: "v4_tabm_enhanced_decay65_backtest",
            2023: "v4_tabm_enhanced_decay65_backtest",
            2024: "v4_tabm_enhanced_decay65_2024",
        },
        "tabm_outcome",
        "prediction",
    ),
]


def path_for(spec: Spec, year: int) -> Path:
    stem = spec.stems[year]
    # The 2024 stage itself ends in _2024, so the rolling harness appended a
    # second season suffix.  All other stages use the conventional name.
    return PRED / f"{stem}_{year}.npz"


def load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path) as archive:
        return {key: np.asarray(archive[key]) for key in archive.files}


def raw_score(y: np.ndarray, prediction: np.ndarray) -> float:
    return float(score(y, np.clip(prediction, 0.0, 1.0))["raw_competition_score"])


def fit_scalar(direction: np.ndarray, residual: np.ndarray) -> tuple[float, float]:
    denominator = float(np.dot(direction, direction))
    raw = float(np.dot(direction, residual) / denominator) if denominator else 0.0
    return raw, float(np.clip(raw, -GAMMA_LIMIT, GAMMA_LIMIT))


def current_predictions() -> tuple[dict[int, np.ndarray], dict[int, dict[str, np.ndarray]]]:
    artifacts = {
        year: load_npz(PRED / f"v4_oof_direction_locked_{year}.npz")
        for year in (2023, 2024)
    }
    return {
        year: artifacts[year]["oof_direction_locked"].astype(np.float64)
        for year in artifacts
    }, artifacts


def spec_label(spec: Spec) -> str:
    suffix = "" if spec.column is None else f"[{spec.column}]"
    return f"{spec.name}::{spec.key}{suffix}::{spec.kind}"


def main() -> None:
    accepted_artifacts = {
        year: load_npz(PRED / f"v4_routed_tabm_stack_locked_{year}.npz")
        for year in (2022, 2023, 2024)
    }
    accepted = {
        year: accepted_artifacts[year]["routed_tabm_stack"].astype(np.float64)
        for year in accepted_artifacts
    }
    y = {
        year: accepted_artifacts[year]["y"].astype(np.float64)
        for year in accepted_artifacts
    }
    route_r = {
        year: accepted_artifacts[year]["game_type_r"].astype(bool)
        for year in accepted_artifacts
    }
    accepted_scores = {year: raw_score(y[year], accepted[year]) for year in accepted}
    current, current_artifacts = current_predictions()
    current_scores = {year: raw_score(y[year], current[year]) for year in current}

    directions: dict[str, dict[int, np.ndarray]] = {}
    screen: list[dict[str, object]] = []
    failures: list[dict[str, str]] = []
    for spec in SPECS:
        label = spec_label(spec)
        try:
            values: dict[int, np.ndarray] = {}
            for year in (2022, 2023, 2024):
                artifact = load_npz(path_for(spec, year))
                if not np.array_equal(
                    artifact["row_index"], accepted_artifacts[year]["row_index"]
                ):
                    raise ValueError(f"row_index mismatch in {year}")
                array = artifact[spec.key].astype(np.float64)
                if spec.column is not None:
                    array = array[:, spec.column]
                if array.ndim != 1 or len(array) != len(y[year]):
                    raise ValueError(f"invalid shape {array.shape} in {year}")
                raw = array if spec.kind == "delta" else array - accepted[year]
                values[year] = np.where(route_r[year], raw, 0.0)
            gamma22_raw, gamma22 = fit_scalar(
                values[2022], y[2022] - accepted[2022]
            )
            gamma23_raw, gamma23 = fit_scalar(
                values[2023], y[2023] - accepted[2023]
            )
            gain22 = raw_score(
                y[2022], accepted[2022] + gamma22 * values[2022]
            ) - accepted_scores[2022]
            gain23 = raw_score(
                y[2023], accepted[2023] + gamma22 * values[2023]
            ) - accepted_scores[2023]
            ratio = abs(gamma23 / gamma22) if abs(gamma22) > 1e-9 else float("inf")
            stable = bool(gamma22 * gamma23 > 0 and 0.5 <= ratio <= 2.0)
            screen.append(
                {
                    "name": label,
                    "kind": spec.kind,
                    "gamma_fit_2022_raw": gamma22_raw,
                    "gamma_fit_2022": gamma22,
                    "gain_fit_2022": gain22,
                    "transfer_gain_2023": gain23,
                    "gamma_fit_2023_accepted_raw": gamma23_raw,
                    "gamma_fit_2023_accepted": gamma23,
                    "gamma_abs_ratio_2023_to_2022": ratio,
                    "coefficient_stable": stable,
                }
            )
            directions[label] = values
        except Exception as exc:
            failures.append(
                {"name": label, "exception": type(exc).__name__, "message": str(exc)}
            )

    ranked = sorted(
        screen, key=lambda row: float(row["transfer_gain_2023"]), reverse=True
    )
    eligible = [
        row for row in ranked
        if float(row["gain_fit_2022"]) > 0.05
        and float(row["transfer_gain_2023"]) > 0.05
        and bool(row["coefficient_stable"])
    ]
    selected: list[dict[str, object]] = []
    for row in eligible:
        candidate = directions[str(row["name"])][2023]
        if any(
            abs(float(np.corrcoef(candidate, directions[str(old["name"])][2023])[0, 1]))
            >= 0.98
            for old in selected
        ):
            continue
        selected.append(row)
        if len(selected) >= 8:
            break

    confirmations: dict[str, dict[str, float]] = {}
    payload: dict[str, np.ndarray] = {
        "y": y[2024],
        "row_index": accepted_artifacts[2024]["row_index"],
        "cluster": accepted_artifacts[2024]["cluster"],
        "base": current[2024],
    }
    for index, row in enumerate(selected):
        name = str(row["name"])
        values = directions[name]
        raw_gamma, gamma = fit_scalar(values[2023], y[2023] - current[2023])
        p23 = current[2023] + gamma * values[2023]
        p24 = current[2024] + gamma * values[2024]
        s23 = raw_score(y[2023], p23)
        s24 = raw_score(y[2024], p24)
        confirmations[name] = {
            "gamma_fit_2023_raw": raw_gamma,
            "gamma_fit_2023": gamma,
            "selection_gain_over_current": s23 - current_scores[2023],
            "confirmation_gain_over_current": s24 - current_scores[2024],
            "confirmation_score": s24,
            "expected_lb_median": s24 + MEDIAN_OFFSET,
        }
        payload[f"candidate_{index:02d}"] = np.clip(p24, 0.0, 1.0)
        payload[f"direction_{index:02d}"] = values[2024]

    # A joint model is reported only as a diagnostic.  Candidate admission and
    # diversity use 2022/2023 alone, while coefficients use current-base 2023.
    joint: dict[str, object] | None = None
    if selected:
        matrix23 = np.column_stack(
            [directions[str(row["name"])][2023] for row in selected]
        )
        matrix24 = np.column_stack(
            [directions[str(row["name"])][2024] for row in selected]
        )
        scale = np.sqrt(np.mean(np.square(matrix23), axis=0))
        scale = np.where(scale > 1e-12, scale, 1.0)
        standardized = matrix23 / scale
        penalty = 1.0
        beta_std = np.linalg.solve(
            standardized.T @ standardized
            + len(standardized) * penalty * np.eye(standardized.shape[1]),
            standardized.T @ (y[2023] - current[2023]),
        )
        beta = beta_std / scale
        p23 = current[2023] + matrix23 @ beta
        p24 = current[2024] + matrix24 @ beta
        s23 = raw_score(y[2023], p23)
        s24 = raw_score(y[2024], p24)
        joint = {
            "penalty": penalty,
            "coefficients": {
                str(row["name"]): float(value)
                for row, value in zip(selected, beta)
            },
            "selection_gain_over_current": s23 - current_scores[2023],
            "confirmation_gain_over_current": s24 - current_scores[2024],
            "confirmation_score": s24,
            "expected_lb_median": s24 + MEDIAN_OFFSET,
        }
        payload["joint_ridge_1"] = np.clip(p24, 0.0, 1.0)

    diagnostic_items: list[tuple[str, dict[str, object]]] = list(confirmations.items())
    if joint is not None:
        diagnostic_items.append(("joint_ridge_1", joint))
    best = max(
        diagnostic_items,
        key=lambda item: float(item[1]["confirmation_score"]),
        default=("current", {"confirmation_score": current_scores[2024],
                             "confirmation_gain_over_current": 0.0,
                             "expected_lb_median": current_scores[2024] + MEDIAN_OFFSET}),
    )
    np.savez_compressed(OUTPUT, **payload)
    report = {
        "protocol": {
            "official_train_only": True,
            "test_rows_read": False,
            "semantic_array_types_explicit": True,
            "screen_fit_year": 2022,
            "screen_transfer_year": 2023,
            "coefficient_refit_year": 2023,
            "confirmation_year": 2024,
            "selection_does_not_read_2024_labels": True,
            "route": "R only",
        },
        "current_scores": current_scores,
        "screened": ranked,
        "eligible": eligible,
        "diverse_selected": selected,
        "confirmations": confirmations,
        "joint_diagnostic": joint,
        "failures": failures,
        "best_observed_confirmation_diagnostic": {
            "name": best[0],
            **best[1],
            "required_local_score": REQUIRED_LOCAL,
            "crosses_required_local_score": float(best[1]["confirmation_score"]) > REQUIRED_LOCAL,
            "warning": "2024 ranking is diagnostic, not a selection rule",
        },
        "prediction_artifact": str(OUTPUT.relative_to(ROOT)),
    }
    REPORT.write_text(
        json.dumps(json_safe(report), ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(json_safe({
        "eligible": eligible,
        "confirmations": confirmations,
        "joint": joint,
        "best": report["best_observed_confirmation_diagnostic"],
    }), ensure_ascii=False, indent=2), flush=True)
    print(f"Saved {REPORT}", flush=True)


if __name__ == "__main__":
    main()
