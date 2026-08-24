#!/usr/bin/env python3
"""Combine the joint V4 reweight direction with the stable neural delta."""

from __future__ import annotations

import itertools
import json
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.analyze_v4_joint_reweight import (  # noqa: E402
    COMPONENT_NAMES,
    EXPANSION_REPORT,
    NESTED_REPORT,
    add_columns,
    add_expansion_columns,
    build_components,
)
from experiments.analyze_v4_temporal_residual_ridge import (  # noqa: E402
    MEDIAN_OFFSET,
    REQUIRED_LOCAL,
    json_safe,
    load_frames,
    score,
)
from experiments.v4_current_ensemble import PREDICTIONS  # noqa: E402


OUTPUT_JSON = ROOT / "experiments/results/v4_joint_neural.json"
OUTPUT_NPZ = PREDICTIONS / "v4_joint_neural_2024.npz"
JOINT_REPORT = ROOT / "experiments/results/v4_joint_reweight.json"
YEARS = (2022, 2023, 2024)
SCALE_GRID = (0.0, 0.25, 0.50, 0.75, 1.00)
NEURAL_GRID = (0.0, 0.25, 0.50, 0.75, 1.00, 1.25, 1.50, 2.00)


def gain(y: np.ndarray, baseline: np.ndarray,
         correction: np.ndarray) -> float:
    residual = y - baseline
    improvement = (
        2.0 * float(np.dot(residual, correction))
        - float(np.dot(correction, correction))
    ) / float(len(y))
    rate = float(np.mean(y))
    return 100_000.0 * improvement / (rate * (1.0 - rate))


def main() -> None:
    frames, artifacts = load_frames()
    add_columns(frames, artifacts)
    add_expansion_columns(frames, artifacts)
    nested_report = json.loads(NESTED_REPORT.read_text(encoding="utf-8"))
    nested_values = nested_report["routes"][nested_report["selected_route"]]
    expansion_report = json.loads(EXPANSION_REPORT.read_text(encoding="utf-8"))
    expansion_values = expansion_report["routes"]["core_from_r"]
    joint_report = json.loads(JOINT_REPORT.read_text(encoding="utf-8"))
    adjustment = np.asarray([
        joint_report["coordinate_adjustments"][name] for name in COMPONENT_NAMES
    ], dtype=np.float64)
    sources = {2022: (2020, 2021), 2023: (2021, 2022), 2024: (2022, 2023)}
    champions = {}
    joint_delta = {}
    neural_delta = {}
    base_metrics = {}
    for year in YEARS:
        champion, matrix, _ = build_components(
            frames, artifacts, year, sources[year], nested_values, expansion_values
        )
        champions[year] = champion
        joint_delta[year] = matrix @ adjustment
        with np.load(PREDICTIONS / f"v4_neural_resnet_delta_{year}.npz") as archive:
            if not np.array_equal(archive["row_index"], artifacts[year]["row_index"]):
                raise ValueError(f"Neural alignment mismatch for {year}")
            neural_delta[year] = np.asarray(archive["neural_delta"], dtype=np.float64)
        base_metrics[str(year)] = score(artifacts[year]["y"], champion)

    trials = []
    for scale, neural_weight in itertools.product(SCALE_GRID, NEURAL_GRID):
        gains = {}
        for year in (2022, 2023):
            correction = scale * joint_delta[year] + neural_weight * neural_delta[year]
            gains[str(year)] = gain(
                np.asarray(artifacts[year]["y"], dtype=np.float64),
                champions[year], correction
            )
        trials.append({
            "joint_scale": scale,
            "neural_weight": neural_weight,
            "gains": gains,
            "robust_min_gain": float(min(gains.values())),
            "mean_gain": float(np.mean(list(gains.values()))),
        })
    selected = max(
        trials,
        key=lambda row: (row["robust_min_gain"], row["mean_gain"]),
    )
    conservative_trials = [row for row in trials if row["joint_scale"] <= 0.50]
    conservative = max(
        conservative_trials,
        key=lambda row: (row["robust_min_gain"], row["mean_gain"]),
    )

    fold_artifacts = {}
    for year in YEARS:
        conservative_prediction = np.clip(
            champions[year]
            + float(conservative["joint_scale"]) * joint_delta[year]
            + float(conservative["neural_weight"]) * neural_delta[year],
            0.0,
            1.0,
        )
        path = PREDICTIONS / f"v4_joint_neural_conservative_{year}.npz"
        np.savez_compressed(
            path,
            y=artifacts[year]["y"],
            row_index=artifacts[year]["row_index"],
            cluster=artifacts[year]["cluster"],
            champion=champions[year],
            joint_delta=joint_delta[year],
            neural_delta=neural_delta[year],
            conservative=conservative_prediction,
        )
        fold_artifacts[str(year)] = str(path.relative_to(ROOT))

    confirmations = {}
    predictions = {}
    for name, row in (("selected", selected), ("conservative", conservative)):
        correction = (
            float(row["joint_scale"]) * joint_delta[2024]
            + float(row["neural_weight"]) * neural_delta[2024]
        )
        prediction = np.clip(champions[2024] + correction, 0.0, 1.0)
        metric = score(artifacts[2024]["y"], prediction)
        confirmations[name] = {
            "configuration": row,
            "metrics": metric,
            "gain": float(
                metric["raw_competition_score"]
                - base_metrics["2024"]["raw_competition_score"]
            ),
            "expected_lb_median": float(
                metric["raw_competition_score"] + MEDIAN_OFFSET
            ),
        }
        predictions[name] = prediction
        print(f"[{name}] scale={row['joint_scale']:.2f} "
              f"neural={row['neural_weight']:.2f} "
              f"gain={confirmations[name]['gain']:+.4f} "
              f"local={metric['raw_competition_score']:.4f}", flush=True)

    primary = predictions["selected"]
    primary_metrics = confirmations["selected"]["metrics"]
    np.savez_compressed(
        OUTPUT_NPZ,
        y=artifacts[2024]["y"],
        row_index=artifacts[2024]["row_index"],
        cluster=artifacts[2024]["cluster"],
        champion=champions[2024],
        joint_neural=primary,
        conservative=predictions["conservative"],
        joint_delta=joint_delta[2024],
        neural_delta=neural_delta[2024],
    )
    report = {
        "protocol": {
            "official_train_only": True,
            "test_rows_read": False,
            "leaderboard_values_used": False,
            "row_independent": True,
            "joint_direction_preselected": True,
            "neural_recipe_preselected": True,
            "combination_selection": "worst gain on 2022 and 2023",
            "confirmation": "apply once to 2024",
        },
        "fixed_estimator": {
            "median_offset": MEDIAN_OFFSET,
            "required_local_score": REQUIRED_LOCAL,
            "target_lb": 1190.0,
        },
        "base_metrics": base_metrics,
        "selected": selected,
        "conservative_selected": conservative,
        "top_trials": sorted(
            trials,
            key=lambda row: (row["robust_min_gain"], row["mean_gain"]),
            reverse=True,
        )[:30],
        "confirmations_2024": confirmations,
        "fold_artifacts": fold_artifacts,
        "primary_2024": {
            "metrics": primary_metrics,
            "expected_lb_median": float(
                primary_metrics["raw_competition_score"] + MEDIAN_OFFSET
            ),
            "crosses_required_local_score": bool(
                primary_metrics["raw_competition_score"] > REQUIRED_LOCAL
            ),
        },
        "prediction_artifact": str(OUTPUT_NPZ.relative_to(ROOT)),
    }
    OUTPUT_JSON.write_text(
        json.dumps(json_safe(report), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps({
        "selected": selected,
        "score_2024": primary_metrics["raw_competition_score"],
        "expected_lb_median": (
            primary_metrics["raw_competition_score"] + MEDIAN_OFFSET
        ),
    }, ensure_ascii=False, indent=2))
    print(f"Saved {OUTPUT_JSON}")
    print(f"Saved {OUTPUT_NPZ}")


if __name__ == "__main__":
    main()
