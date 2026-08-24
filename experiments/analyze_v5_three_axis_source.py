#!/usr/bin/env python3
"""Reproduce and freeze the 2020/2021 three-axis source discovery.

This is intentionally a discovery audit, not confirmatory evidence.  It only
reads the two source years that were used to select the components and weights.
"""

from __future__ import annotations

from itertools import combinations
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.analyze_v5_dense_pitchtype_moe import (  # noqa: E402
    digest,
    load,
    safe,
    score,
)
from experiments.run_v5_h1_residual import (  # noqa: E402
    cluster_bootstrap_score_gain,
)


RESULTS = ROOT / "experiments/results"
PRED = RESULTS / "predictions"
TRAIN = ROOT / "open/data/train.csv"
LOCK = ROOT / "experiments/params/v5_three_axis_source_lock.json"
REPORT = RESULTS / "v5_three_axis_source.json"
YEARS = (2020, 2021)
BOOTSTRAP_ITERATIONS = 2000
WEIGHT_DENOMINATOR = 20
TOP_K = 20

PRIMITIVES = (
    "exact_c",
    "dense_physics_moe",
    "dense_pitch_joint",
    "component_pattern_moe",
    "current_state_numeric",
    "exact_c_seed1",
    "exact_c_seed7",
)
INPUTS = {
    "exact_c": ("v4_m3_c_backtest_{year}_{year}.npz", "catboost_outcome"),
    "dense_physics_moe": (
        "v5_dense_physics_pitchtype_moe_source_{year}.npz",
        "dense_physics_moe_raw",
    ),
    "dense_pitch_joint": (
        "v5_dense_pitch_joint_source_{year}.npz",
        "dense_pitch_joint_raw",
    ),
    "component_pattern_moe": (
        "v5_component_pattern_moe_source_{year}.npz",
        "component_pattern_moe_raw",
    ),
    "current_state_numeric": (
        "v4_numeric_cat_current_context_level_tmctx_seed42_early_{year}.npz",
        "catboost_numeric",
    ),
    "exact_c_seed1": (
        "v5_exact_c_multiseed_s1_source_{year}.npz",
        "catboost_outcome",
    ),
    "exact_c_seed7": (
        "v5_exact_c_multiseed_s7_source_{year}.npz",
        "catboost_outcome",
    ),
}
EXPECTED_NAMES = (
    "dense_pitch_joint",
    "component_pattern_moe",
    "current_state_numeric",
)
EXPECTED_WEIGHTS = (0.10, 0.25, 0.65)


def gain_from_gram(
    weights: np.ndarray,
    gram: np.ndarray,
    base_mse: float,
    reference: float,
) -> float:
    candidate_mse = float(weights @ gram @ weights)
    return float(100000.0 * (base_mse - candidate_mse) / reference)


def main() -> None:
    if REPORT.exists():
        raise FileExistsError(f"immutable result already exists: {REPORT}")
    lock = json.loads(LOCK.read_text(encoding="utf-8"))
    all_types = pd.read_csv(TRAIN, usecols=["game_type"])["game_type"].astype(str)

    folds: dict[int, dict[str, Any]] = {}
    input_hashes: dict[str, dict[str, str]] = {}
    for year in YEARS:
        artifacts: dict[str, dict[str, np.ndarray]] = {}
        paths: dict[str, Path] = {}
        for name in PRIMITIVES:
            template, _ = INPUTS[name]
            path = PRED / template.format(year=year)
            artifacts[name] = load(path)
            paths[name] = path
        reference_artifact = artifacts["exact_c"]
        for name, artifact in artifacts.items():
            for key in ("y", "row_index", "cluster"):
                if not np.array_equal(reference_artifact[key], artifact[key]):
                    raise ValueError(f"alignment mismatch: {year}/{name}/{key}")
        predictions = np.column_stack(
            [artifacts[name][INPUTS[name][1]].astype(np.float64) for name in PRIMITIVES]
        )
        if not np.isfinite(predictions).all():
            raise ValueError(f"non-finite primitive prediction: {year}")
        if np.any((predictions <= 0.0) | (predictions >= 1.0)):
            raise ValueError(f"primitive prediction outside open unit interval: {year}")
        y = reference_artifact["y"].astype(np.float64)
        row_index = reference_artifact["row_index"].astype(np.int64)
        game_type = all_types.iloc[row_index].to_numpy(dtype=str)
        masks = {
            "full": np.ones(len(y), dtype=bool),
            "R": game_type == "R",
            "F": game_type == "F",
        }
        grams: dict[str, np.ndarray] = {}
        base_mse: dict[str, float] = {}
        references: dict[str, float] = {}
        errors = predictions - y[:, None]
        for route, mask in masks.items():
            route_errors = errors[mask]
            grams[route] = route_errors.T @ route_errors / float(mask.sum())
            base_mse[route] = float(grams[route][0, 0])
            rate = float(y[mask].mean())
            references[route] = max(rate * (1.0 - rate), 1e-12)
        folds[year] = {
            "y": reference_artifact["y"].astype(np.int8),
            "row_index": row_index,
            "cluster": reference_artifact["cluster"],
            "predictions": predictions,
            "masks": masks,
            "grams": grams,
            "base_mse": base_mse,
            "references": references,
            "paths": paths,
        }
        input_hashes[str(year)] = {
            name: digest(path) for name, path in paths.items()
        }

    trials: list[dict[str, Any]] = []
    for indices in combinations(range(len(PRIMITIVES)), 3):
        names = tuple(PRIMITIVES[index] for index in indices)
        for first in range(1, WEIGHT_DENOMINATOR - 1):
            for second in range(1, WEIGHT_DENOMINATOR - first):
                third = WEIGHT_DENOMINATOR - first - second
                if third <= 0:
                    continue
                local_weights = np.asarray(
                    [first, second, third], dtype=np.float64
                ) / WEIGHT_DENOMINATOR
                global_weights = np.zeros(len(PRIMITIVES), dtype=np.float64)
                global_weights[list(indices)] = local_weights
                metrics: dict[str, dict[str, float]] = {}
                for year in YEARS:
                    fold = folds[year]
                    metrics[str(year)] = {
                        route: gain_from_gram(
                            global_weights,
                            fold["grams"][route],
                            fold["base_mse"][route],
                            fold["references"][route],
                        )
                        for route in ("full", "R", "F")
                    }
                full_gains = [metrics[str(year)]["full"] for year in YEARS]
                r_gains = [metrics[str(year)]["R"] for year in YEARS]
                trials.append(
                    {
                        "names": names,
                        "weights": local_weights,
                        "minimum_full_gain": float(min(full_gains)),
                        "minimum_R_gain": float(min(r_gains)),
                        "mean_full_gain": float(np.mean(full_gains)),
                        "years": metrics,
                    }
                )

    ranked = sorted(
        trials,
        key=lambda item: (
            item["minimum_full_gain"],
            item["minimum_R_gain"],
            item["mean_full_gain"],
        ),
        reverse=True,
    )
    selected = ranked[0]
    if tuple(selected["names"]) != EXPECTED_NAMES or not np.allclose(
        selected["weights"], EXPECTED_WEIGHTS, atol=1e-12, rtol=0.0
    ):
        raise AssertionError(
            f"source selection does not reproduce lock: {selected['names']} / "
            f"{selected['weights']}"
        )
    locked_components = lock["locked_recipe"]["components"]
    locked_names = tuple(item["name"] for item in locked_components)
    locked_weights = np.asarray(
        [item["weight"] for item in locked_components], dtype=np.float64
    )
    if locked_names != EXPECTED_NAMES or not np.allclose(
        locked_weights, EXPECTED_WEIGHTS, atol=1e-12, rtol=0.0
    ):
        raise AssertionError("JSON source lock disagrees with reproduced selection")

    selected_global_weights = np.zeros(len(PRIMITIVES), dtype=np.float64)
    for name, weight in zip(selected["names"], selected["weights"]):
        selected_global_weights[PRIMITIVES.index(name)] = float(weight)
    confirmatory_metrics: dict[str, Any] = {}
    artifacts_out: dict[str, Any] = {}
    for year in YEARS:
        fold = folds[year]
        parent = fold["predictions"][:, PRIMITIVES.index("exact_c")]
        prediction = np.clip(
            fold["predictions"] @ selected_global_weights,
            1e-6,
            1.0 - 1e-6,
        )
        route_metrics: dict[str, Any] = {}
        for route_index, (route, mask) in enumerate(fold["masks"].items()):
            parent_score = score(fold["y"], parent, mask)
            candidate_score = score(fold["y"], prediction, mask)
            interval = cluster_bootstrap_score_gain(
                fold["y"],
                parent,
                prediction,
                fold["cluster"],
                mask,
                iterations=BOOTSTRAP_ITERATIONS,
                seed=1900000 + 10000 * year + 1000 * route_index,
            )
            point = float(candidate_score["score"] - parent_score["score"])
            if abs(point - float(interval["point"])) > 1e-8:
                raise AssertionError(f"score/CI point mismatch: {year}/{route}")
            route_metrics[route] = {
                "parent": parent_score,
                "candidate": candidate_score,
                "gain": point,
                "pitcher_cluster_95_ci": interval,
            }
        confirmatory_metrics[str(year)] = route_metrics
        output = PRED / f"v5_three_axis_source_{year}.npz"
        if output.exists():
            raise FileExistsError(f"immutable artifact already exists: {output}")
        np.savez_compressed(
            output,
            y=fold["y"],
            row_index=fold["row_index"],
            cluster=fold["cluster"],
            parent_exact_c=parent,
            dense_pitch_joint=fold["predictions"][:, PRIMITIVES.index("dense_pitch_joint")],
            component_pattern_moe=fold["predictions"][:, PRIMITIVES.index("component_pattern_moe")],
            current_state_numeric=fold["predictions"][:, PRIMITIVES.index("current_state_numeric")],
            final_prediction=prediction,
        )
        artifacts_out[str(year)] = {
            "path": str(output.relative_to(ROOT)),
            "sha256": digest(output),
        }

    report = {
        "experiment_id": lock["experiment_id"],
        "status": "source_discovery_reproduced_and_locked",
        "evidence_role": "discovery_only_not_confirmatory",
        "lock_sha256": digest(LOCK),
        "script_sha256": digest(Path(__file__)),
        "years_read": list(YEARS),
        "years_not_read": [2022, 2023, 2024],
        "search": {
            "primitives": PRIMITIVES,
            "weight_step": 1.0 / WEIGHT_DENOMINATOR,
            "exactly_positive_components": 3,
            "trial_count": len(trials),
            "ranking": lock["source_search"]["ranking"],
            "top_20": ranked[:TOP_K],
        },
        "selected": selected,
        "selected_bootstrap_diagnostics": confirmatory_metrics,
        "input_sha256": input_hashes,
        "artifacts": artifacts_out,
        "development_metrics_read": False,
        "goal_status": "active",
        "goal_completion_claimed": False,
    }
    REPORT.write_text(
        json.dumps(safe(report), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(
        json.dumps(
            safe(
                {
                    "status": report["status"],
                    "trial_count": len(trials),
                    "selected": selected,
                    "bootstrap": confirmatory_metrics,
                }
            ),
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
