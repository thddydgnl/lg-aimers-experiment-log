#!/usr/bin/env python3
"""Confirm the historically admitted two-seed TabTransformer direction."""

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
SCREEN = ROOT / "experiments/results/v4_tabtransformer_seed_ensemble_screen.json"
REPORT = ROOT / "experiments/results/v4_tabtransformer_seed_ensemble.json"
KEY = "tabtransformer_outcome"


def load(path: Path) -> dict[str, np.ndarray]:
    with np.load(path) as archive:
        return {key: np.asarray(archive[key]) for key in archive.files}


def raw_score(y: np.ndarray, p: np.ndarray) -> float:
    return float(score(y, np.clip(p, 0.0, 1.0))["raw_competition_score"])


def main() -> None:
    screen = json.loads(SCREEN.read_text(encoding="utf-8"))
    if not screen["passes_2024_training_gate"]:
        raise ValueError("The two-seed average did not pass its historical gate")
    accepted = {
        year: load(PRED / f"v4_routed_tabm_stack_locked_{year}.npz")
        for year in (2023, 2024)
    }
    base_artifacts = {
        year: load(PRED / f"v4_oof_direction_locked_{year}.npz")
        for year in (2023, 2024)
    }
    seed2026 = {
        2023: load(PRED / "v4_deep_arch_r_oof_2023.npz"),
        2024: load(PRED / "v4_tabtransformer_r_confirm_2024.npz"),
    }
    seed42 = {
        2023: load(PRED / "v4_tabtransformer_r_seed42_oof_2023.npz"),
        2024: load(PRED / "v4_tabtransformer_r_seed42_confirm_2024.npz"),
    }
    y = {year: accepted[year]["y"].astype(np.float64) for year in accepted}
    base = {
        year: base_artifacts[year]["oof_direction_locked"].astype(np.float64)
        for year in base_artifacts
    }
    direction = {}
    for year in (2023, 2024):
        for label, artifact in (("base", base_artifacts[year]),
                                ("seed2026", seed2026[year]),
                                ("seed42", seed42[year])):
            if not np.array_equal(artifact["row_index"], accepted[year]["row_index"]):
                raise ValueError(f"{label} row_index mismatch for {year}")
        mean_prediction = 0.5 * (
            seed2026[year][KEY].astype(np.float64)
            + seed42[year][KEY].astype(np.float64)
        )
        raw = mean_prediction - accepted[year]["routed_tabm_stack"].astype(np.float64)
        direction[year] = np.where(
            accepted[year]["game_type_r"].astype(bool), raw, 0.0
        )
    residual = y[2023] - base[2023]
    denominator = float(np.dot(direction[2023], direction[2023]))
    gamma = float(np.dot(direction[2023], residual) / denominator)
    predictions = {
        year: np.clip(base[year] + gamma * direction[year], 0.0, 1.0)
        for year in (2023, 2024)
    }
    base_scores = {year: raw_score(y[year], base[year]) for year in (2023, 2024)}
    scores = {year: raw_score(y[year], predictions[year]) for year in (2023, 2024)}
    outputs = {}
    for year in (2023, 2024):
        path = PRED / f"v4_tabtransformer_seed_ensemble_{year}.npz"
        np.savez_compressed(
            path,
            y=y[year], row_index=accepted[year]["row_index"],
            cluster=accepted[year]["cluster"], game_type_r=accepted[year]["game_type_r"],
            base=base[year], direction_tabtransformer_seed_average=direction[year],
            base_plus_tabtransformer_seed_average=predictions[year],
        )
        outputs[year] = str(path.relative_to(ROOT))
    report = {
        "protocol": {
            "official_train_only": True,
            "test_rows_read": False,
            "two_seed_average_admitted_on_2022_to_2023": True,
            "scalar_refit_year": 2023,
            "confirmation_year": 2024,
            "selection_does_not_read_2024_labels": True,
            "route": "R only",
        },
        "historical_screen": screen,
        "gamma_fit_2023": gamma,
        "base_scores": base_scores,
        "candidate_scores": scores,
        "gains": {year: scores[year] - base_scores[year] for year in (2023, 2024)},
        "expected_lb_median": scores[2024] + MEDIAN_OFFSET,
        "required_local_score": REQUIRED_LOCAL,
        "crosses_required_local_score": scores[2024] > REQUIRED_LOCAL,
        "prediction_artifacts": outputs,
    }
    REPORT.write_text(json.dumps(json_safe(report), ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(json_safe(report), ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
