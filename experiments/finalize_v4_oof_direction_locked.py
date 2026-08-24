#!/usr/bin/env python3
"""Materialize the leakage-safe winner from the V4 OOF direction catalog.

The model family/direction was admitted using the 2022 -> 2023 transfer screen.
Its scalar is refit on 2023 only, and 2024 remains a confirmation fold.  This
script deliberately reconstructs the prediction from named source artifacts
instead of depending on an anonymous ``candidate_XX`` array.
"""

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
CATALOG_REPORT = ROOT / "experiments/results/v4_oof_direction_catalog.json"
NESTED_REPORT = ROOT / "experiments/results/v4_nested_direction_stack.json"
REPORT = ROOT / "experiments/results/v4_oof_direction_locked.json"

SOURCE_STEM = "v4_numeric_cat_current_context_level_tmctx_seed42"
SOURCE_KEY = "catboost_numeric"
SOURCE_NAME = f"{SOURCE_STEM}::{SOURCE_KEY}::R"
GAMMA_LIMIT = 1.0


def load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path) as archive:
        return {key: np.asarray(archive[key]) for key in archive.files}


def raw_score(y: np.ndarray, prediction: np.ndarray) -> float:
    return float(score(y, np.clip(prediction, 0.0, 1.0))["raw_competition_score"])


def nested_base(year: int, artifact: dict[str, np.ndarray]) -> np.ndarray:
    if year == 2024:
        return artifact["base_plus_nested_axes_joint"].astype(np.float64)
    report = json.loads(NESTED_REPORT.read_text(encoding="utf-8"))
    coefficients = report["candidates"]["base_plus_nested_axes_joint"]["coefficients"]
    return artifact["base"].astype(np.float64) + sum(
        float(coefficients[name]) * artifact[f"direction_{name}"].astype(np.float64)
        for name in coefficients
    )


def main() -> None:
    catalog = json.loads(CATALOG_REPORT.read_text(encoding="utf-8"))
    screened = next(
        row for row in catalog["diverse_selected"] if row["name"] == SOURCE_NAME
    )
    recorded = catalog["individual_confirmations"][SOURCE_NAME]

    stacks: dict[int, dict[str, np.ndarray]] = {}
    accepted: dict[int, dict[str, np.ndarray]] = {}
    source: dict[int, dict[str, np.ndarray]] = {}
    base: dict[int, np.ndarray] = {}
    direction: dict[int, np.ndarray] = {}
    y: dict[int, np.ndarray] = {}
    for year in (2023, 2024):
        stacks[year] = load_npz(PRED / f"v4_nested_direction_stack_{year}.npz")
        accepted[year] = load_npz(PRED / f"v4_routed_tabm_stack_locked_{year}.npz")
        source[year] = load_npz(PRED / f"{SOURCE_STEM}_{year}.npz")
        expected_index = accepted[year]["row_index"]
        for label, artifact in (("nested", stacks[year]), ("source", source[year])):
            if not np.array_equal(artifact["row_index"], expected_index):
                raise ValueError(f"{label} row_index mismatch for {year}")
        y[year] = accepted[year]["y"].astype(np.float64)
        base[year] = nested_base(year, stacks[year])
        route_r = accepted[year]["game_type_r"].astype(bool)
        accepted_prediction = accepted[year]["routed_tabm_stack"].astype(np.float64)
        raw_direction = source[year][SOURCE_KEY].astype(np.float64) - accepted_prediction
        direction[year] = np.where(route_r, raw_direction, 0.0)

    residual = y[2023] - base[2023]
    denominator = float(np.dot(direction[2023], direction[2023]))
    gamma_raw = float(np.dot(direction[2023], residual) / denominator)
    gamma = float(np.clip(gamma_raw, -GAMMA_LIMIT, GAMMA_LIMIT))
    predictions = {
        year: np.clip(base[year] + gamma * direction[year], 0.0, 1.0)
        for year in (2023, 2024)
    }
    base_scores = {year: raw_score(y[year], base[year]) for year in (2023, 2024)}
    scores = {year: raw_score(y[year], predictions[year]) for year in (2023, 2024)}

    if not np.isclose(gamma, recorded["gamma_fit_2023"], rtol=0.0, atol=1e-12):
        raise AssertionError("Reconstructed scalar differs from catalog")
    if not np.isclose(scores[2024], recorded["confirmation_score"], rtol=0.0, atol=1e-9):
        raise AssertionError("Reconstructed confirmation score differs from catalog")

    outputs: dict[int, str] = {}
    for year in (2023, 2024):
        path = PRED / f"v4_oof_direction_locked_{year}.npz"
        np.savez_compressed(
            path,
            y=y[year],
            row_index=accepted[year]["row_index"],
            cluster=accepted[year]["cluster"],
            game_type_r=accepted[year]["game_type_r"],
            base=base[year],
            direction_current_state_catboost_r=direction[year],
            oof_direction_locked=predictions[year],
        )
        outputs[year] = str(path.relative_to(ROOT))

    report = {
        "protocol": {
            "official_train_only": True,
            "test_rows_read": False,
            "model_direction_admission": "fit 2022, positive transfer to 2023",
            "coefficient_stability_required": True,
            "scalar_refit_year": 2023,
            "confirmation_year": 2024,
            "selection_does_not_read_2024_labels": True,
        },
        "source": {
            "name": SOURCE_NAME,
            "screen": screened,
        },
        "scalar": {"raw": gamma_raw, "clipped": gamma, "bounds": [-GAMMA_LIMIT, GAMMA_LIMIT]},
        "base_scores": base_scores,
        "locked_scores": scores,
        "gains": {year: scores[year] - base_scores[year] for year in (2023, 2024)},
        "expected_lb_median": scores[2024] + MEDIAN_OFFSET,
        "required_local_score": REQUIRED_LOCAL,
        "crosses_required_local_score": scores[2024] > REQUIRED_LOCAL,
        "prediction_artifacts": outputs,
        "warning": "2024 is a diagnostic confirmation fold; it was not used to admit or fit this direction.",
    }
    REPORT.write_text(
        json.dumps(json_safe(report), ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(json_safe(report), ensure_ascii=False, indent=2), flush=True)
    print(f"Saved {REPORT}", flush=True)


if __name__ == "__main__":
    main()
