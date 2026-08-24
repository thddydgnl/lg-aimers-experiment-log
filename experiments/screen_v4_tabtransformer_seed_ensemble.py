#!/usr/bin/env python3
"""Screen a predeclared two-seed TabTransformer average on 2022 -> 2023."""

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
REPORT = ROOT / "experiments/results/v4_tabtransformer_seed_ensemble_screen.json"
STEMS = ("v4_deep_arch_r_oof", "v4_tabtransformer_r_seed42_oof")
KEY = "tabtransformer_outcome"


def load(path: Path) -> dict[str, np.ndarray]:
    with np.load(path) as archive:
        return {key: np.asarray(archive[key]) for key in archive.files}


def raw_score(y: np.ndarray, p: np.ndarray) -> float:
    return float(score(y, np.clip(p, 0.0, 1.0))["raw_competition_score"])


def scalar(direction: np.ndarray, residual: np.ndarray) -> float:
    denominator = float(np.dot(direction, direction))
    return float(np.dot(direction, residual) / denominator) if denominator else 0.0


def main() -> None:
    accepted = {
        year: load(PRED / f"v4_routed_tabm_stack_locked_{year}.npz")
        for year in (2022, 2023)
    }
    models = {
        stem: {year: load(PRED / f"{stem}_{year}.npz") for year in (2022, 2023)}
        for stem in STEMS
    }
    y = {year: accepted[year]["y"].astype(np.float64) for year in accepted}
    base = {
        year: accepted[year]["routed_tabm_stack"].astype(np.float64)
        for year in accepted
    }
    direction = {}
    for year in (2022, 2023):
        values = []
        for stem in STEMS:
            if not np.array_equal(models[stem][year]["row_index"], accepted[year]["row_index"]):
                raise ValueError(f"row_index mismatch for {stem}/{year}")
            values.append(models[stem][year][KEY].astype(np.float64))
        route = accepted[year]["game_type_r"].astype(bool)
        direction[year] = np.where(route, np.mean(values, axis=0) - base[year], 0.0)
    gamma22 = scalar(direction[2022], y[2022] - base[2022])
    gamma23 = scalar(direction[2023], y[2023] - base[2023])
    base_scores = {year: raw_score(y[year], base[year]) for year in (2022, 2023)}
    gain22 = raw_score(y[2022], base[2022] + gamma22 * direction[2022]) - base_scores[2022]
    gain23 = raw_score(y[2023], base[2023] + gamma22 * direction[2023]) - base_scores[2023]
    ratio = abs(gamma23 / gamma22) if abs(gamma22) > 1e-9 else float("inf")
    stable = bool(gamma22 * gamma23 > 0 and 0.5 <= ratio <= 2.0)
    report = {
        "protocol": {"fit_year": 2022, "transfer_year": 2023, "2024_read": False},
        "stems": STEMS,
        "gamma_fit_2022": gamma22,
        "gamma_fit_2023": gamma23,
        "gain_fit_2022": gain22,
        "transfer_gain_2023": gain23,
        "gamma_abs_ratio": ratio,
        "coefficient_stable": stable,
        "passes_2024_training_gate": bool(gain22 > 0.05 and gain23 > 0.05 and stable),
    }
    REPORT.write_text(json.dumps(json_safe(report), ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(json_safe(report), ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
