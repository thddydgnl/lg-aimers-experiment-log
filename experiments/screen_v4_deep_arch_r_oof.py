#!/usr/bin/env python3
"""Screen the R-only deep architecture batch before any 2024 training."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.analyze_v4_temporal_residual_ridge import json_safe, score  # noqa: E402


PRED = ROOT / "experiments/results/predictions"
DEFAULT_STAGE = "v4_deep_arch_r_oof"
DEFAULT_MODELS = ("deep_mlp_outcome", "deepfm_outcome", "tabtransformer_outcome")
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", default=DEFAULT_STAGE)
    parser.add_argument("--models", nargs="+", default=list(DEFAULT_MODELS))
    parser.add_argument("--report", default="v4_deep_arch_r_oof_screen.json")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report_path = ROOT / "experiments/results" / args.report
    accepted = {
        year: load(PRED / f"v4_routed_tabm_stack_locked_{year}.npz")
        for year in (2022, 2023)
    }
    deep = {
        year: load(PRED / f"{args.stage}_{year}.npz")
        for year in (2022, 2023)
    }
    y = {year: accepted[year]["y"].astype(np.float64) for year in accepted}
    base = {
        year: accepted[year]["routed_tabm_stack"].astype(np.float64)
        for year in accepted
    }
    route = {year: accepted[year]["game_type_r"].astype(bool) for year in accepted}
    base_scores = {year: raw_score(y[year], base[year]) for year in accepted}
    rows = []
    for model in args.models:
        values = {}
        for year in (2022, 2023):
            if not np.array_equal(deep[year]["row_index"], accepted[year]["row_index"]):
                raise ValueError(f"row_index mismatch for {year}")
            raw = deep[year][model].astype(np.float64) - base[year]
            values[year] = np.where(route[year], raw, 0.0)
        gamma22_raw, gamma22 = fit(values[2022], y[2022] - base[2022])
        gamma23_raw, gamma23 = fit(values[2023], y[2023] - base[2023])
        gain22 = raw_score(y[2022], base[2022] + gamma22 * values[2022]) - base_scores[2022]
        transfer23 = raw_score(y[2023], base[2023] + gamma22 * values[2023]) - base_scores[2023]
        ratio = abs(gamma23 / gamma22) if abs(gamma22) > 1e-9 else float("inf")
        stable = bool(gamma22 * gamma23 > 0.0 and 0.5 <= ratio <= 2.0)
        passed = bool(gain22 > 0.05 and transfer23 > 0.05 and stable)
        rows.append({
            "model": model,
            "gamma_fit_2022_raw": gamma22_raw,
            "gamma_fit_2022": gamma22,
            "gain_fit_2022": gain22,
            "transfer_gain_2023": transfer23,
            "gamma_fit_2023_raw": gamma23_raw,
            "gamma_fit_2023": gamma23,
            "gamma_abs_ratio_2023_to_2022": ratio,
            "coefficient_stable": stable,
            "passes_2024_training_gate": passed,
        })
    ranked = sorted(rows, key=lambda row: float(row["transfer_gain_2023"]), reverse=True)
    report = {
        "protocol": {
            "official_train_only": True,
            "test_rows_read": False,
            "fit_year": 2022,
            "untouched_transfer_year": 2023,
            "2024_model_trained": False,
            "route": "R only",
            "stage": args.stage,
        },
        "baseline_scores": base_scores,
        "ranked": ranked,
        "passed_models": [row["model"] for row in ranked if row["passes_2024_training_gate"]],
    }
    report_path.write_text(
        json.dumps(json_safe(report), ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(json_safe(report), ensure_ascii=False, indent=2), flush=True)
    print(f"Saved {report_path}", flush=True)


if __name__ == "__main__":
    main()
