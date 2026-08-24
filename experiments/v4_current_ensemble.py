"""Shared reconstruction of the strongest pre-V4-neural OOF ensemble."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from experiments.analyze_v4_temporal_residual_ridge import M3_WEIGHTS


ROOT = Path(__file__).resolve().parents[1]
PREDICTIONS = ROOT / "experiments/results/predictions"
CONTEXT_WEIGHT = 0.15
LEVEL_WEIGHT = 0.50
STABILITY_C_WEIGHT = 1.05
STABILITY_B_WEIGHT = 0.925
CALIBRATION_SLOPE = 1.05


def load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path) as archive:
        return {key: np.asarray(archive[key]) for key in archive.files}


def ensure_aligned(reference: dict[str, np.ndarray],
                   candidate: dict[str, np.ndarray], label: str) -> None:
    for key in ("y", "row_index"):
        if not np.array_equal(reference[key], candidate[key]):
            raise ValueError(f"Artifact alignment mismatch for {label}/{key}")


def current_ensemble(season: int,
                     artifact: dict[str, np.ndarray]) -> np.ndarray:
    residual = load_npz(PREDICTIONS / f"v4_residual_ensemble_{season}.npz")
    ensure_aligned(artifact, residual, f"residual/{season}")
    numeric: dict[str, np.ndarray] = {}
    for key, stem in (
        ("base", "v4_numeric_cat_current_tmctx_seed42"),
        ("context", "v4_numeric_cat_current_context_tmctx_seed42"),
        ("level", "v4_numeric_cat_current_context_level_tmctx_seed42"),
    ):
        item = load_npz(PREDICTIONS / f"{stem}_{season}.npz")
        ensure_aligned(artifact, item, f"{stem}/{season}")
        numeric[key] = np.asarray(item["catboost_numeric"], dtype=np.float64)

    c_stem = ("v4_outcome_c_trackman_stability_backtest" if season < 2024
              else "v4_outcome_c_trackman_stability")
    b_stem = ("v4_outcome_b_trackman_stability_backtest" if season < 2024
              else "v4_outcome_b_trackman_stability")
    c_item = load_npz(PREDICTIONS / f"{c_stem}_{season}.npz")
    b_item = load_npz(PREDICTIONS / f"{b_stem}_{season}.npz")
    ensure_aligned(artifact, c_item, f"{c_stem}/{season}")
    ensure_aligned(artifact, b_item, f"{b_stem}/{season}")

    context_delta = numeric["context"] - numeric["base"]
    level_delta = numeric["level"] - numeric["context"]
    c_delta = (np.asarray(c_item["catboost_outcome"], dtype=np.float64)
               - np.asarray(artifact["component_C"], dtype=np.float64))
    b_delta = (np.asarray(b_item["catboost_outcome"], dtype=np.float64)
               - np.asarray(artifact["component_B"], dtype=np.float64))
    prediction = (
        np.asarray(residual["residual_ensemble"], dtype=np.float64)
        + CONTEXT_WEIGHT * context_delta
        + LEVEL_WEIGHT * level_delta
        + CALIBRATION_SLOPE * STABILITY_C_WEIGHT * M3_WEIGHTS["C"] * c_delta
        + CALIBRATION_SLOPE * STABILITY_B_WEIGHT * M3_WEIGHTS["B"] * b_delta
    )
    return np.clip(prediction, 0.0, 1.0)


def ensemble_weights() -> dict[str, float]:
    return {
        "context": CONTEXT_WEIGHT,
        "level": LEVEL_WEIGHT,
        "trackman_stability_c": STABILITY_C_WEIGHT,
        "trackman_stability_b": STABILITY_B_WEIGHT,
        "calibration_slope": CALIBRATION_SLOPE,
    }
