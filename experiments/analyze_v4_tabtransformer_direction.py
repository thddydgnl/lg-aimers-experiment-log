#!/usr/bin/env python3
"""Add the pre-screened R-only TabTransformer direction to the current stack."""

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
SCREEN = ROOT / "experiments/results/v4_deep_arch_r_oof_screen.json"
REPORT = ROOT / "experiments/results/v4_tabtransformer_direction.json"
MODEL_KEY = "tabtransformer_outcome"
GAMMA_LIMIT = 1.0


def load(path: Path) -> dict[str, np.ndarray]:
    with np.load(path) as archive:
        return {key: np.asarray(archive[key]) for key in archive.files}


def raw_score(y: np.ndarray, p: np.ndarray) -> float:
    return float(score(y, np.clip(p, 0.0, 1.0))["raw_competition_score"])


def main() -> None:
    screen = json.loads(SCREEN.read_text(encoding="utf-8"))
    selected = next(row for row in screen["ranked"] if row["model"] == MODEL_KEY)
    if not selected["passes_2024_training_gate"]:
        raise ValueError("TabTransformer did not pass the historical transfer gate")

    years = (2023, 2024)
    current = {year: load(PRED / f"v4_oof_direction_locked_{year}.npz") for year in years}
    accepted = {year: load(PRED / f"v4_routed_tabm_stack_locked_{year}.npz") for year in years}
    model = {
        2023: load(PRED / "v4_deep_arch_r_oof_2023.npz"),
        2024: load(PRED / "v4_tabtransformer_r_confirm_2024.npz"),
    }
    y = {year: accepted[year]["y"].astype(np.float64) for year in years}
    base = {year: current[year]["oof_direction_locked"].astype(np.float64) for year in years}
    direction = {}
    for year in years:
        for label, artifact in (("current", current[year]), ("model", model[year])):
            if not np.array_equal(artifact["row_index"], accepted[year]["row_index"]):
                raise ValueError(f"{label} row_index mismatch for {year}")
        route = accepted[year]["game_type_r"].astype(bool)
        model_delta = (
            model[year][MODEL_KEY].astype(np.float64)
            - accepted[year]["routed_tabm_stack"].astype(np.float64)
        )
        direction[year] = np.where(route, model_delta, 0.0)

    residual = y[2023] - base[2023]
    denominator = float(np.dot(direction[2023], direction[2023]))
    gamma_raw = float(np.dot(direction[2023], residual) / denominator)
    gamma = float(np.clip(gamma_raw, -GAMMA_LIMIT, GAMMA_LIMIT))
    prediction = {
        year: np.clip(base[year] + gamma * direction[year], 0.0, 1.0)
        for year in years
    }
    base_scores = {year: raw_score(y[year], base[year]) for year in years}
    scores = {year: raw_score(y[year], prediction[year]) for year in years}

    outputs = {}
    for year in years:
        path = PRED / f"v4_tabtransformer_direction_{year}.npz"
        np.savez_compressed(
            path,
            y=y[year],
            row_index=accepted[year]["row_index"],
            cluster=accepted[year]["cluster"],
            game_type_r=accepted[year]["game_type_r"],
            base=base[year],
            direction_tabtransformer_r=direction[year],
            base_plus_tabtransformer=prediction[year],
        )
        outputs[year] = str(path.relative_to(ROOT))

    report = {
        "protocol": {
            "official_train_only": True,
            "test_rows_read": False,
            "architecture_gate": "fit 2022 and transfer unchanged to 2023",
            "scalar_refit_year": 2023,
            "confirmation_year": 2024,
            "selection_does_not_read_2024_labels": True,
            "route": "R only",
        },
        "historical_screen": selected,
        "gamma_fit_2023_raw": gamma_raw,
        "gamma_fit_2023": gamma,
        "base_scores": base_scores,
        "candidate_scores": scores,
        "gains": {year: scores[year] - base_scores[year] for year in years},
        "expected_lb_median": scores[2024] + MEDIAN_OFFSET,
        "required_local_score": REQUIRED_LOCAL,
        "crosses_required_local_score": scores[2024] > REQUIRED_LOCAL,
        "prediction_artifacts": outputs,
        "warning": "2024 is a diagnostic confirmation fold and was not used for architecture or scalar selection.",
    }
    REPORT.write_text(
        json.dumps(json_safe(report), ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(json_safe(report), ensure_ascii=False, indent=2), flush=True)
    print(f"Saved {REPORT}", flush=True)


if __name__ == "__main__":
    main()
