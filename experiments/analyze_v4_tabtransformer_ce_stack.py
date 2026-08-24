#!/usr/bin/env python3
"""Stack the historically selected CE TabTransformer on the seed-average base."""

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
SCREEN = ROOT / "experiments/results/v4_tabtransformer_r_ce_oof_screen.json"
REPORT = ROOT / "experiments/results/v4_tabtransformer_ce_stack.json"
KEY = "tabtransformer_outcome"


def load(path: Path) -> dict[str, np.ndarray]:
    with np.load(path) as archive:
        return {key: np.asarray(archive[key]) for key in archive.files}


def raw_score(y: np.ndarray, p: np.ndarray) -> float:
    return float(score(y, np.clip(p, 0.0, 1.0))["raw_competition_score"])


def main() -> None:
    screen = json.loads(SCREEN.read_text(encoding="utf-8"))
    selected = next(row for row in screen["ranked"] if row["model"] == KEY)
    if not selected["passes_2024_training_gate"]:
        raise ValueError("CE TabTransformer did not pass its historical gate")
    years = (2023, 2024)
    accepted = {year: load(PRED / f"v4_routed_tabm_stack_locked_{year}.npz") for year in years}
    stack = {year: load(PRED / f"v4_tabtransformer_seed_ensemble_{year}.npz") for year in years}
    model = {
        2023: load(PRED / "v4_tabtransformer_r_ce_oof_2023.npz"),
        2024: load(PRED / "v4_tabtransformer_r_ce_confirm_2024.npz"),
    }
    y = {year: accepted[year]["y"].astype(np.float64) for year in years}
    base = {
        year: stack[year]["base_plus_tabtransformer_seed_average"].astype(np.float64)
        for year in years
    }
    direction = {}
    for year in years:
        for label, artifact in (("stack", stack[year]), ("model", model[year])):
            if not np.array_equal(artifact["row_index"], accepted[year]["row_index"]):
                raise ValueError(f"{label} row_index mismatch for {year}")
        raw = model[year][KEY].astype(np.float64) - accepted[year]["routed_tabm_stack"].astype(np.float64)
        direction[year] = np.where(accepted[year]["game_type_r"].astype(bool), raw, 0.0)
    residual = y[2023] - base[2023]
    denominator = float(np.dot(direction[2023], direction[2023]))
    gamma = float(np.clip(np.dot(direction[2023], residual) / denominator, -1.0, 1.0))
    prediction = {
        year: np.clip(base[year] + gamma * direction[year], 0.0, 1.0)
        for year in years
    }
    base_scores = {year: raw_score(y[year], base[year]) for year in years}
    scores = {year: raw_score(y[year], prediction[year]) for year in years}
    outputs = {}
    for year in years:
        path = PRED / f"v4_tabtransformer_ce_stack_{year}.npz"
        np.savez_compressed(
            path, y=y[year], row_index=accepted[year]["row_index"],
            cluster=accepted[year]["cluster"], game_type_r=accepted[year]["game_type_r"],
            base=base[year], direction_tabtransformer_ce_r=direction[year],
            base_plus_tabtransformer_ce=prediction[year],
        )
        outputs[year] = str(path.relative_to(ROOT))
    report = {
        "protocol": {
            "official_train_only": True,
            "test_rows_read": False,
            "ce_architecture_admitted_on_2022_to_2023": True,
            "base_direction_admitted_on_2022_to_2023": True,
            "coefficient_refit_year": 2023,
            "confirmation_year": 2024,
            "selection_does_not_read_2024_labels": True,
            "route": "R only",
        },
        "historical_screen": selected,
        "gamma_fit_2023": gamma,
        "base_scores": base_scores,
        "candidate_scores": scores,
        "gains": {year: scores[year] - base_scores[year] for year in years},
        "expected_lb_median": scores[2024] + MEDIAN_OFFSET,
        "required_local_score": REQUIRED_LOCAL,
        "crosses_required_local_score": scores[2024] > REQUIRED_LOCAL,
        "prediction_artifacts": outputs,
    }
    REPORT.write_text(json.dumps(json_safe(report), ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(json_safe(report), ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
