#!/usr/bin/env python3
"""Development gate for the source-locked dense-physics/seed ensemble."""

from __future__ import annotations

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
    PRED,
    digest,
    load,
    safe,
    score,
)
from experiments.run_v5_h1_residual import (  # noqa: E402
    cluster_bootstrap_score_gain,
)


LOCK = ROOT / "experiments/params/v5_dense_physics_seed_ensemble_dev_lock.json"
SOURCE_REPORT = ROOT / "experiments/results/v5_dense_physics_seed_ensemble_source.json"
REPORT = ROOT / "experiments/results/v5_dense_physics_seed_ensemble_dev.json"
TRAIN = ROOT / "open/data/train.csv"
YEARS = (2022, 2023)
PARENT_STEM = "v3_sparse_c_backtest"
SEED_STEMS = {
    1: "v5_exact_c_multiseed_s1_dev2223",
    7: "v5_exact_c_multiseed_s7_dev2223",
}
PHYSICS_STEMS = {
    2022: "v5_dense_physics_pitchtype_moe_dev2022",
    2023: "v5_dense_physics_pitchtype_moe_dev2023",
}
PHYSICS_KEY = "catboost_dense_pitchtype_moe"
BASELINES = {
    "exact_parent_C": (PARENT_STEM, "catboost_outcome"),
    "honest_r_identity": ("v5_honest_m3_r_identity", "final_prediction"),
    "honest_r_grid": ("v5_honest_m3_r_grid", "final_prediction"),
}


def evaluate_baseline(
    reference: dict[str, np.ndarray],
    baseline: np.ndarray,
    candidate: np.ndarray,
    masks: dict[str, np.ndarray],
    iterations: int,
    seed: int,
) -> dict[str, Any]:
    routes: dict[str, Any] = {}
    for route_index, (route, mask) in enumerate(masks.items()):
        base_metrics = score(reference["y"], baseline, mask)
        candidate_metrics = score(reference["y"], candidate, mask)
        interval = cluster_bootstrap_score_gain(
            reference["y"], baseline, candidate, reference["cluster"], mask,
            iterations=iterations, seed=seed + 1000 * route_index,
        )
        gain = candidate_metrics["score"] - base_metrics["score"]
        if abs(gain - interval["point"]) > 1e-8:
            raise AssertionError(f"score/CI mismatch: {route}")
        routes[route] = {
            "baseline": base_metrics,
            "candidate": candidate_metrics,
            "gain": gain,
            "pitcher_cluster_95_ci": interval,
        }
    return {"routes": routes}


def main() -> None:
    if REPORT.exists():
        raise FileExistsError(f"immutable result already exists: {REPORT}")
    lock = json.loads(LOCK.read_text(encoding="utf-8"))
    source = json.loads(SOURCE_REPORT.read_text(encoding="utf-8"))
    if source["status"] != "source_pass" or not source["source_gate"]["passed"]:
        raise ValueError("source recipe did not pass")
    if lock["status"] != "locked_before_2022_2023_candidate_training_or_metrics":
        raise ValueError("development lock status changed")
    if lock["recipe"]["final"] != (
        "0.5 * seed_bag + 0.5 * dense_physics_r_blend"
    ):
        raise ValueError("locked final formula changed")

    official_types = pd.read_csv(TRAIN, usecols=["game_type"])["game_type"].astype(str)
    iterations = int(lock["bootstrap_iterations"])
    years: dict[str, Any] = {}
    full_gains: list[float] = []
    same_parent_r_checks: list[bool] = []
    postbreak_f_checks: list[bool] = []

    for year in YEARS:
        parent_path = PRED / f"{PARENT_STEM}_{year}.npz"
        parent_artifact = load(parent_path)
        parent = parent_artifact["catboost_outcome"].astype(np.float64)
        seed_predictions = [parent]
        seed_paths: dict[str, str] = {"2026": str(parent_path.relative_to(ROOT))}
        for seed_value, stem in SEED_STEMS.items():
            path = PRED / f"{stem}_{year}.npz"
            artifact = load(path)
            for key in ("y", "row_index", "cluster"):
                if not np.array_equal(parent_artifact[key], artifact[key]):
                    raise ValueError(f"seed alignment mismatch: {year}/{seed_value}/{key}")
            seed_predictions.append(artifact["catboost_outcome"].astype(np.float64))
            seed_paths[str(seed_value)] = str(path.relative_to(ROOT))
        seed_bag = np.mean(np.column_stack(seed_predictions), axis=1)

        physics_path = PRED / f"{PHYSICS_STEMS[year]}_{year}.npz"
        physics_artifact = load(physics_path)
        for key in ("y", "row_index", "cluster"):
            if not np.array_equal(parent_artifact[key], physics_artifact[key]):
                raise ValueError(f"physics alignment mismatch: {year}/{key}")
        types = official_types.iloc[
            parent_artifact["row_index"].astype(np.int64)
        ].to_numpy(dtype=str)
        regular = types == "R"
        finals = types == "F"
        physics_blend = parent.copy()
        physics_raw = physics_artifact[PHYSICS_KEY].astype(np.float64)
        physics_blend[regular] += 0.5 * (physics_raw[regular] - parent[regular])
        physics_blend = np.clip(physics_blend, 1e-6, 1.0 - 1e-6)
        final = np.clip(0.5 * seed_bag + 0.5 * physics_blend, 1e-6, 1.0 - 1e-6)
        direction = final - parent
        masks = {
            "full": np.ones(len(parent), dtype=bool),
            "R": regular,
            "F": finals,
        }
        baseline_results: dict[str, Any] = {}
        prediction_payload: dict[str, np.ndarray] = {
            "parent_exact_c": parent,
            "seed_bag": seed_bag,
            "dense_physics_blend": physics_blend,
            "direction": direction,
        }
        for baseline_index, (name, (stem, key)) in enumerate(BASELINES.items()):
            path = PRED / f"{stem}_{year}.npz"
            artifact = load(path)
            for align_key in ("y", "row_index", "cluster"):
                if not np.array_equal(parent_artifact[align_key], artifact[align_key]):
                    raise ValueError(f"baseline alignment mismatch: {year}/{name}/{align_key}")
            baseline = artifact[key].astype(np.float64)
            candidate = np.clip(baseline + direction, 1e-6, 1.0 - 1e-6)
            result = evaluate_baseline(
                parent_artifact,
                baseline,
                candidate,
                masks,
                iterations,
                1410000 + 10000 * year + 100000 * baseline_index,
            )
            baseline_results[name] = {
                "artifact": str(path.relative_to(ROOT)),
                **result,
            }
            prediction_payload[f"final_{name}"] = candidate
            full_gains.append(float(result["routes"]["full"]["gain"]))

        exact_r = baseline_results["exact_parent_C"]["routes"]["R"]
        same_parent_r_checks.extend(
            (
                exact_r["gain"] > 0.0,
                exact_r["pitcher_cluster_95_ci"]["ci_low"] > 0.0,
            )
        )
        if year == 2023:
            exact_f = baseline_results["exact_parent_C"]["routes"]["F"]
            postbreak_f_checks.extend(
                (
                    exact_f["gain"] > 0.0,
                    exact_f["pitcher_cluster_95_ci"]["ci_low"] > 0.0,
                )
            )
        output = PRED / f"v5_dense_physics_seed_ensemble_dev_{year}.npz"
        if output.exists():
            raise FileExistsError(f"immutable output exists: {output}")
        np.savez_compressed(
            output,
            y=parent_artifact["y"].astype(np.int8),
            row_index=parent_artifact["row_index"].astype(np.int64),
            cluster=parent_artifact["cluster"],
            **prediction_payload,
        )
        years[str(year)] = {
            "rows": int(len(parent)),
            "R_rows": int(regular.sum()),
            "F_rows": int(finals.sum()),
            "components": {
                "seed_artifacts": seed_paths,
                "dense_physics_artifact": str(physics_path.relative_to(ROOT)),
            },
            "baselines": baseline_results,
            "output": {
                "path": str(output.relative_to(ROOT)),
                "sha256": digest(output),
            },
        }

    g_dev = float(min(full_gains))
    required = float(
        lock["development_gate"]["required_G_dev_strictly_greater_than"]
    )
    same_parent_r_pass = bool(all(same_parent_r_checks))
    postbreak_f_pass = bool(all(postbreak_f_checks))
    goal_scale_pass = bool(g_dev > required)
    passed = bool(same_parent_r_pass and postbreak_f_pass and goal_scale_pass)
    report = {
        "experiment_id": lock["experiment_id"],
        "status": "locked_for_2024" if passed else "development_failed",
        "dev_lock_sha256": digest(LOCK),
        "source_report_sha256": digest(SOURCE_REPORT),
        "script_sha256": digest(Path(__file__)),
        "years_read": list(YEARS),
        "confirmation_2024_read": False,
        "recipe": lock["recipe"],
        "years": years,
        "development_gate": {
            "same_parent_R_point_and_ci_each_year_pass": same_parent_r_pass,
            "postbreak_2023_F_point_and_ci_pass": postbreak_f_pass,
            "G_dev_full_min_all_years_all_baselines": g_dev,
            "required_G_dev_strictly_greater_than": required,
            "goal_scale_pass": goal_scale_pass,
            "passed": passed,
            "decision": (
                "freeze and open 2024 exactly once"
                if passed
                else "close without training or reading 2024 candidate"
            ),
        },
    }
    REPORT.write_text(
        json.dumps(safe(report), ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "status": report["status"],
                "development_gate": report["development_gate"],
                "gains": {
                    str(year): {
                        baseline: {
                            route: {
                                "gain": years[str(year)]["baselines"][baseline][
                                    "routes"
                                ][route]["gain"],
                                "ci_low": years[str(year)]["baselines"][baseline][
                                    "routes"
                                ][route]["pitcher_cluster_95_ci"]["ci_low"],
                            }
                            for route in ("full", "R", "F")
                        }
                        for baseline in BASELINES
                    }
                    for year in YEARS
                },
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
