#!/usr/bin/env python3
"""Replace the current-state CatBoost arm with its OOF-C3 improved version."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.finalize_v4_oof_direction_locked import nested_base  # noqa: E402
from experiments.analyze_v4_temporal_residual_ridge import (  # noqa: E402
    MEDIAN_OFFSET,
    REQUIRED_LOCAL,
    json_safe,
    score,
)


PRED = ROOT / "experiments/results/predictions"
REPORT = ROOT / "experiments/results/v4_c3_source_replacement.json"
GAMMA_LIMIT = 1.0


def load(path: Path) -> dict[str, np.ndarray]:
    with np.load(path) as archive:
        return {key: np.asarray(archive[key]) for key in archive.files}


def raw_score(y: np.ndarray, p: np.ndarray) -> float:
    return float(score(y, np.clip(p, 0.0, 1.0))["raw_competition_score"])


def fit(direction: np.ndarray, residual: np.ndarray) -> tuple[float, float]:
    denominator = float(np.dot(direction, direction))
    raw = float(np.dot(direction, residual) / denominator) if denominator else 0.0
    return raw, float(np.clip(raw, -GAMMA_LIMIT, GAMMA_LIMIT))


def main() -> None:
    accepted = {
        year: load(PRED / f"v4_routed_tabm_stack_locked_{year}.npz")
        for year in (2022, 2023, 2024)
    }
    source = {
        year: load(PRED / f"v4_oof_residual_source_{year}.npz")
        for year in (2022, 2023, 2024)
    }
    y = {year: accepted[year]["y"].astype(np.float64) for year in accepted}
    accepted_prediction = {
        year: accepted[year]["routed_tabm_stack"].astype(np.float64)
        for year in accepted
    }
    direction = {}
    for year in accepted:
        if not np.array_equal(source[year]["row_index"], accepted[year]["row_index"]):
            raise ValueError(f"source row_index mismatch for {year}")
        raw = source[year]["source_plus_c3_r"].astype(np.float64) - accepted_prediction[year]
        direction[year] = np.where(accepted[year]["game_type_r"].astype(bool), raw, 0.0)

    gamma22_raw, gamma22 = fit(direction[2022], y[2022] - accepted_prediction[2022])
    gamma23_accepted_raw, gamma23_accepted = fit(
        direction[2023], y[2023] - accepted_prediction[2023]
    )
    accepted_scores = {
        year: raw_score(y[year], accepted_prediction[year]) for year in accepted
    }
    gain22 = raw_score(
        y[2022], accepted_prediction[2022] + gamma22 * direction[2022]
    ) - accepted_scores[2022]
    transfer23 = raw_score(
        y[2023], accepted_prediction[2023] + gamma22 * direction[2023]
    ) - accepted_scores[2023]
    ratio = abs(gamma23_accepted / gamma22) if abs(gamma22) > 1e-9 else float("inf")
    passed = bool(
        gain22 > 0.05 and transfer23 > 0.05
        and gamma22 * gamma23_accepted > 0.0 and 0.5 <= ratio <= 2.0
    )

    nested_artifacts = {
        year: load(PRED / f"v4_nested_direction_stack_{year}.npz")
        for year in (2023, 2024)
    }
    base = {
        year: nested_base(year, nested_artifacts[year]) for year in (2023, 2024)
    }
    base_scores = {year: raw_score(y[year], base[year]) for year in base}
    gamma23_raw, gamma23 = fit(direction[2023], y[2023] - base[2023])
    replacement = {
        year: np.clip(base[year] + gamma23 * direction[year], 0.0, 1.0)
        for year in base
    }
    replacement_scores = {
        year: raw_score(y[year], replacement[year]) for year in replacement
    }

    replacement_gamma_diagnostics = {}
    for label, value in (
        ("current_base_refit_2023", gamma23),
        ("accepted_fit_2022", gamma22),
        ("accepted_fit_2023", gamma23_accepted),
        ("accepted_two_year_mean", 0.5 * (gamma22 + gamma23_accepted)),
    ):
        candidate = {
            year: np.clip(base[year] + value * direction[year], 0.0, 1.0)
            for year in base
        }
        candidate_scores = {
            year: raw_score(y[year], candidate[year]) for year in candidate
        }
        replacement_gamma_diagnostics[label] = {
            "gamma": value,
            "scores": candidate_scores,
            "gains_over_base": {
                year: candidate_scores[year] - base_scores[year] for year in base
            },
            "expected_lb_median": candidate_scores[2024] + MEDIAN_OFFSET,
        }

    # The two-seed TabTransformer direction was independently admitted on the
    # 2022->2023 screen. Refit only its scalar after replacing the CatBoost arm.
    tab = {
        year: load(PRED / f"v4_tabtransformer_seed_ensemble_{year}.npz")
        for year in (2023, 2024)
    }
    tab_direction = {
        year: tab[year]["direction_tabtransformer_seed_average"].astype(np.float64)
        for year in tab
    }
    tab_gamma_raw, tab_gamma = fit(
        tab_direction[2023], y[2023] - replacement[2023]
    )
    stacked = {
        year: np.clip(replacement[year] + tab_gamma * tab_direction[year], 0.0, 1.0)
        for year in replacement
    }
    stacked_scores = {year: raw_score(y[year], stacked[year]) for year in stacked}

    outputs = {}
    for year in (2023, 2024):
        path = PRED / f"v4_c3_source_replacement_{year}.npz"
        np.savez_compressed(
            path,
            y=y[year], row_index=accepted[year]["row_index"],
            cluster=accepted[year]["cluster"], base=base[year],
            direction_c3_source_r=direction[year], replacement=replacement[year],
            direction_tabtransformer_seed_average=tab_direction[year],
            replacement_plus_tabtransformer=stacked[year],
        )
        outputs[year] = str(path.relative_to(ROOT))

    report = {
        "protocol": {
            "official_train_only": True,
            "external_model_artifacts_used": False,
            "test_rows_read": False,
            "source_arm_screen_fit_year": 2022,
            "source_arm_screen_transfer_year": 2023,
            "coefficient_refit_year": 2023,
            "confirmation_year": 2024,
            "selection_does_not_read_2024_labels": True,
            "route": "R only",
        },
        "historical_screen": {
            "gamma_fit_2022_raw": gamma22_raw,
            "gamma_fit_2022": gamma22,
            "gain_fit_2022": gain22,
            "transfer_gain_2023": transfer23,
            "gamma_fit_2023_accepted_raw": gamma23_accepted_raw,
            "gamma_fit_2023_accepted": gamma23_accepted,
            "gamma_abs_ratio": ratio,
            "passes_gate": passed,
        },
        "base_scores": base_scores,
        "replacement": {
            "gamma_fit_2023_raw": gamma23_raw,
            "gamma_fit_2023": gamma23,
            "scores": replacement_scores,
            "gains_over_base": {
                year: replacement_scores[year] - base_scores[year] for year in base
            },
            "expected_lb_median": replacement_scores[2024] + MEDIAN_OFFSET,
        },
        "replacement_gamma_diagnostics": replacement_gamma_diagnostics,
        "replacement_plus_tabtransformer": {
            "gamma_fit_2023_raw": tab_gamma_raw,
            "gamma_fit_2023": tab_gamma,
            "scores": stacked_scores,
            "gains_over_replacement": {
                year: stacked_scores[year] - replacement_scores[year] for year in stacked
            },
            "expected_lb_median": stacked_scores[2024] + MEDIAN_OFFSET,
            "required_local_score": REQUIRED_LOCAL,
            "crosses_required_local_score": stacked_scores[2024] > REQUIRED_LOCAL,
        },
        "prediction_artifacts": outputs,
        "warning": "2024 scores are diagnostic confirmations, not selection inputs.",
    }
    REPORT.write_text(json.dumps(json_safe(report), ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(json_safe(report), ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
