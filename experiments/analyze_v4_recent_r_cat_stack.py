#!/usr/bin/env python3
"""Add the historically selected recent-three-season R CatBoost arm."""

from __future__ import annotations

import json
import sys
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
REPORT = ROOT / "experiments/results/v4_recent_r_cat_stack.json"
KEY = "catboost_numeric"


def load(path: Path) -> dict[str, np.ndarray]:
    with np.load(path) as archive:
        return {key: np.asarray(archive[key]) for key in archive.files}


def raw_score(y: np.ndarray, prediction: np.ndarray) -> float:
    return float(score(y, np.clip(prediction, 0.0, 1.0))["raw_competition_score"])


def main() -> None:
    years = (2023, 2024)
    accepted = {
        year: load(PRED / f"v4_routed_tabm_stack_locked_{year}.npz") for year in years
    }
    candidate = {
        2023: load(PRED / "v4_numeric_cat_ctxlvl_tm_rfit_recent3_oof_2023.npz"),
        2024: load(PRED / "v4_numeric_cat_ctxlvl_tm_rfit_recent3_confirm_2024.npz"),
    }
    latest = {
        year: load(PRED / f"v4_post4_c3_axis_screen_{year}.npz") for year in years
    }
    for year in years:
        for label, artifact in (("candidate", candidate[year]), ("latest", latest[year])):
            if not np.array_equal(artifact["row_index"], accepted[year]["row_index"]):
                raise ValueError(f"{label} row_index mismatch for {year}")
    y = {year: accepted[year]["y"].astype(np.float64) for year in years}
    base = {
        year: latest[year]["selected_prediction_plus_tabtransformer"].astype(np.float64)
        for year in years
    }
    direction = {
        year: np.where(
            accepted[year]["game_type_r"].astype(bool),
            candidate[year][KEY].astype(np.float64)
            - accepted[year]["routed_tabm_stack"].astype(np.float64),
            0.0,
        )
        for year in years
    }
    denominator = float(np.dot(direction[2023], direction[2023]))
    gamma_raw = float(np.dot(direction[2023], y[2023] - base[2023]) / denominator)
    gamma = float(np.clip(gamma_raw, -1.0, 1.0))
    prediction = {
        year: np.clip(base[year] + gamma * direction[year], 0.0, 1.0)
        for year in years
    }
    base_scores = {year: raw_score(y[year], base[year]) for year in years}
    scores = {year: raw_score(y[year], prediction[year]) for year in years}

    paths: dict[int, str] = {}
    for year in years:
        path = PRED / f"v4_recent_r_cat_stack_{year}.npz"
        np.savez_compressed(
            path,
            y=y[year],
            row_index=accepted[year]["row_index"],
            cluster=accepted[year]["cluster"],
            game_type_r=accepted[year]["game_type_r"],
            base=base[year],
            direction_recent_r_cat=direction[year],
            final_prediction=prediction[year],
        )
        paths[year] = str(path.relative_to(ROOT))

    report = {
        "protocol": {
            "official_train_only": True,
            "test_rows_read": False,
            "candidate_selection": "max min gain among four fixed recency/R CatBoost variants",
            "screen_fit_year": 2022,
            "screen_transfer_year": 2023,
            "selected_before_2024_training": True,
            "coefficient_refit_year": 2023,
            "confirmation_year": 2024,
        },
        "selected_candidate": "R-only, most recent three seasons, 82 features",
        "historical_screen": {
            "gain_fit_2022": 6.900334307591038,
            "transfer_gain_2023": 8.292885783856036,
            "gamma_fit_2022": 0.25021178596695465,
            "gamma_fit_2023_accepted": 0.30432601102638424,
            "gamma_ratio": 1.2162736853114362,
        },
        "gamma_fit_2023_current_raw": gamma_raw,
        "gamma_fit_2023_current": gamma,
        "base_scores": base_scores,
        "scores": scores,
        "gains": {year: scores[year] - base_scores[year] for year in years},
        "expected_lb_median": scores[2024] + MEDIAN_OFFSET,
        "required_local_score": REQUIRED_LOCAL,
        "crosses_required_local_score": scores[2024] > REQUIRED_LOCAL,
        "prediction_artifacts": paths,
        "warning": "2024 is diagnostic confirmation and was not used for selection.",
    }
    REPORT.write_text(
        json.dumps(json_safe(report), ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(json_safe(report), ensure_ascii=False, indent=2), flush=True)
    print(f"Saved {REPORT}", flush=True)


if __name__ == "__main__":
    main()
