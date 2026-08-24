#!/usr/bin/env python3
"""Reproduce the selected V3 sparse blends and their leakage-safe gates."""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.stats import aggregate_gate, paired_bootstrap_brier_ci  # noqa: E402


PREDICTIONS = ROOT / "experiments" / "results" / "predictions"
OUTPUT_JSON = ROOT / "experiments" / "results" / "v3_sparse_ensemble.json"
OUTPUT_CSV = ROOT / "experiments" / "results" / "v3_sparse_ensemble.csv"

COMPONENTS = {
    "A": {
        2022: ("v3_sparse_a_backtest", "catboost_outcome"),
        2023: ("v3_sparse_a_backtest", "catboost_outcome"),
        2024: (
            "v3_outcome_trackmanrich_overall_components120_e14k50_batter80_middle100",
            "catboost_outcome",
        ),
    },
    "B": {
        2022: ("v3_sparse_b_backtest", "catboost_outcome"),
        2023: ("v3_sparse_b_backtest", "catboost_outcome"),
        2024: ("v3_outcome_batter80_middle100_hgroups500", "catboost_outcome"),
    },
    "C": {
        2022: ("v3_sparse_c_backtest", "catboost_outcome"),
        2023: ("v3_sparse_c_backtest", "catboost_outcome"),
        2024: (
            "v3_outcome_trackmanrich_overall_e14k50_batter80_middle100",
            "catboost_outcome",
        ),
    },
}
WEIGHTS = {
    "V3_sparse_m2_1100": {
        "A": 0.6293619759116473,
        "B": 0.37063802408835267,
    },
    "V3_sparse_m3_1103": {
        "A": 0.501443851662535,
        "C": 0.27016033407769313,
        "B": 0.22839581425977187,
    },
}
CALIBRATION = {"slope": 1.05, "offset": -0.006}
ANCHOR_OFFSETS = {
    "S4": 142.0223445187,
    "S5": 145.4097607666,
    "S6": 138.2728223645,
    "S8": 137.9826529035,
}


def load_npz(stage: str, season: int) -> dict[str, np.ndarray]:
    path = PREDICTIONS / f"{stage}_{season}.npz"
    with np.load(path, allow_pickle=False) as archive:
        return {key: archive[key] for key in archive.files}


def assert_aligned(reference: dict[str, np.ndarray], other: dict[str, np.ndarray]) -> None:
    for key in ("y", "row_index", "cluster"):
        if not np.array_equal(reference[key], other[key]):
            raise ValueError(f"Prediction artifact alignment mismatch for {key}")


def score(y: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    y64 = np.asarray(y, dtype=np.float64)
    p64 = np.asarray(prediction, dtype=np.float64)
    brier = float(np.mean(np.square(p64 - y64)))
    rate = float(y64.mean())
    reference = rate * (1.0 - rate)
    raw_skill = 100_000.0 * (1.0 - brier / reference)
    return {
        "rows": int(len(y64)),
        "target_rate": rate,
        "prediction_mean": float(p64.mean()),
        "prediction_std": float(p64.std()),
        "brier": brier,
        "reference_brier": reference,
        "raw_competition_score": raw_skill,
        "competition_score": max(0.0, raw_skill),
    }


def v2_ensemble(season: int) -> tuple[dict[str, np.ndarray], np.ndarray]:
    hgb = load_npz("v2_base", season)
    linear = load_npz("v2_linear_tuned", season)
    catboost = load_npz("v2_catboost", season)
    assert_aligned(hgb, linear)
    assert_aligned(hgb, catboost)
    prediction = (
        0.75 * hgb["hgb"]
        + 0.15 * linear["linear"]
        + 0.10 * catboost["catboost"]
    )
    return hgb, prediction


def component_predictions(season: int) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    reference: dict[str, np.ndarray] | None = None
    predictions: dict[str, np.ndarray] = {}
    for key, season_sources in COMPONENTS.items():
        stage, column = season_sources[season]
        artifact = load_npz(stage, season)
        if reference is None:
            reference = artifact
        else:
            assert_aligned(reference, artifact)
        predictions[key] = np.asarray(artifact[column], dtype=np.float64)
    if reference is None:
        raise RuntimeError("No component predictions loaded")
    return reference, predictions


def calibrate(prediction: np.ndarray) -> np.ndarray:
    return np.clip(
        0.5 + CALIBRATION["slope"] * (prediction - 0.5) + CALIBRATION["offset"],
        1e-6,
        1.0 - 1e-6,
    )


def expected_lb(local_score: float) -> dict[str, object]:
    anchors = {name: local_score + offset for name, offset in ANCHOR_OFFSETS.items()}
    values = np.asarray(list(anchors.values()), dtype=np.float64)
    return {
        "anchors": anchors,
        "median": float(np.median(values)),
        "minimum": float(values.min()),
        "maximum": float(values.max()),
        "crosses_1100_median": bool(np.median(values) > 1100.0),
        "crosses_1100_conservative_s8": bool(anchors["S8"] > 1100.0),
    }


def main() -> None:
    report: dict[str, object] = {
        "protocol": {
            "selection_fold": 2024,
            "safety_fold": 2022,
            "record_only_fold": 2023,
            "row_independent": True,
            "test_distribution_used": False,
            "bootstrap_unit": "pitcher_id cluster",
            "bootstrap_iterations": 2000,
            "selection_risk": (
                "Weights and the fixed affine calibration were selected on the 2024 "
                "development fold; intervals are exploratory and leaderboard confirmation is required."
            ),
        },
        "components": COMPONENTS,
        "weights": WEIGHTS,
        "calibration": {
            **CALIBRATION,
            "formula": "clip(0.5 + 1.05 * (p - 0.5) - 0.006)",
            "predeclared_grid": {
                "slope": [1.0, 1.05, 1.1],
                "offset": [-0.004, -0.006, -0.008],
            },
        },
        "anchor_offsets": ANCHOR_OFFSETS,
        "candidates": {},
    }
    csv_rows: list[dict[str, object]] = []
    intervals: dict[str, dict[int, dict]] = {name: {} for name in WEIGHTS}

    for season in (2022, 2023, 2024):
        reference, components = component_predictions(season)
        v2_reference, baseline_prediction = v2_ensemble(season)
        assert_aligned(reference, v2_reference)
        for candidate, weights in WEIGHTS.items():
            raw_prediction = sum(weights[key] * components[key] for key in weights)
            calibrated_prediction = calibrate(raw_prediction)
            raw_metrics = score(reference["y"], raw_prediction)
            calibrated_metrics = score(reference["y"], calibrated_prediction)
            baseline_metrics = score(reference["y"], baseline_prediction)
            interval = paired_bootstrap_brier_ci(
                reference["y"],
                baseline_prediction,
                calibrated_prediction,
                iterations=2000,
                seed=42,
                reference_brier=calibrated_metrics["reference_brier"],
                clusters=reference["cluster"],
            )
            intervals[candidate][season] = interval
            candidate_report = report["candidates"].setdefault(
                candidate, {"per_season": {}}
            )
            season_report: dict[str, object] = {
                "raw": raw_metrics,
                "calibrated": calibrated_metrics,
                "v2_ensemble_baseline": baseline_metrics,
                "paired_vs_v2_ensemble": interval,
            }
            if season == 2024:
                season_report["expected_lb"] = expected_lb(
                    calibrated_metrics["competition_score"]
                )
            candidate_report["per_season"][str(season)] = season_report
            expected = season_report.get("expected_lb", {})
            csv_rows.append(
                {
                    "candidate": candidate,
                    "season": season,
                    "raw_score": raw_metrics["competition_score"],
                    "calibrated_score": calibrated_metrics["competition_score"],
                    "v2_ensemble_score": baseline_metrics["competition_score"],
                    "brier_delta_vs_v2": interval["point"],
                    "ci_low": interval["ci_low"],
                    "ci_high": interval["ci_high"],
                    "expected_lb_median": expected.get("median", ""),
                    "expected_lb_min": expected.get("minimum", ""),
                    "expected_lb_max": expected.get("maximum", ""),
                }
            )

    for candidate in WEIGHTS:
        candidate_report = report["candidates"][candidate]
        candidate_report["gate"] = aggregate_gate(
            intervals[candidate], primary_season=2024, secondary_season=2022
        )

    OUTPUT_JSON.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    with OUTPUT_CSV.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(csv_rows[0]))
        writer.writeheader()
        writer.writerows(csv_rows)
    print(OUTPUT_JSON)
    print(OUTPUT_CSV)


if __name__ == "__main__":
    main()
