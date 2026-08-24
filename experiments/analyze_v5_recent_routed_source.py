#!/usr/bin/env python3
"""Reproduce the source selection and gate for the recent routed recipe."""

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

from experiments.analyze_v5_dense_pitchtype_moe import digest, load, safe, score
from experiments.run_v5_h1_residual import cluster_bootstrap_score_gain

RESULTS = ROOT / "experiments/results"
PRED = RESULTS / "predictions"
TRAIN = ROOT / "open/data/train.csv"
LOCK = ROOT / "experiments/params/v5_recent_routed_source_lock.json"
REPORT = RESULTS / "v5_recent_routed_source.json"
YEARS = (2020, 2021)
INPUTS = {
    "expanded_fine_pitch_moe": ("v5_expanded_fine_pitch_moe_source_{year}.npz", "final_prediction"),
    "expanded_auto_pitch_joint": ("v5_expanded_auto_pitch_joint_source_{year}.npz", "final_prediction"),
    "partial_expanded_fine_pitch_moe": ("v5_partial_expanded_fine_pitch_moe_source_{year}.npz", "final_prediction"),
    "dense_physics_pitchtype_moe": ("v5_dense_physics_pitchtype_moe_source_{year}.npz", "final_prediction"),
    "dense_moe_reliability": ("v5_dense_moe_reliability_gate_source_{year}.npz", "final_prediction"),
}


def evaluate(
    y: np.ndarray, parent: np.ndarray, candidate: np.ndarray,
    cluster: np.ndarray, masks: dict[str, np.ndarray], seed: int,
) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for route_index, (name, mask) in enumerate(masks.items()):
        base = score(y, parent, mask)
        cand = score(y, candidate, mask)
        ci = cluster_bootstrap_score_gain(
            y, parent, candidate, cluster, mask, iterations=2000,
            seed=seed + route_index * 1000,
        )
        out[name] = {"parent": base, "candidate": cand, "gain": float(cand["score"] - base["score"]), "pitcher_cluster_95_ci": ci}
    return out


def main() -> None:
    if REPORT.exists():
        raise FileExistsError(f"immutable report exists: {REPORT}")
    lock = json.loads(LOCK.read_text(encoding="utf-8"))
    all_types = pd.read_csv(TRAIN, usecols=["game_type"])["game_type"].astype(str)
    folds: dict[int, dict[str, Any]] = {}
    hashes: dict[str, Any] = {}
    names = list(INPUTS)
    for year in YEARS:
        parent_path = PRED / f"v4_m3_c_backtest_{year}_{year}.npz"
        parent_artifact = load(parent_path)
        component_paths: dict[str, Path] = {}
        component_artifacts: dict[str, dict[str, np.ndarray]] = {}
        for name, (template, _) in INPUTS.items():
            path = PRED / template.format(year=year)
            component_paths[name] = path
            component_artifacts[name] = load(path)
            for key in ("y", "row_index", "cluster"):
                if not np.array_equal(parent_artifact[key], component_artifacts[name][key]):
                    raise ValueError(f"alignment mismatch: {year}/{name}/{key}")
        f_path = PRED / f"v5_recent_game_f_update_source_{year}.npz"
        f_artifact = load(f_path)
        for key in ("y", "row_index", "cluster"):
            if not np.array_equal(parent_artifact[key], f_artifact[key]):
                raise ValueError(f"F alignment mismatch: {year}/{key}")
        row_index = parent_artifact["row_index"].astype(np.int64)
        regular = all_types.iloc[row_index].to_numpy(dtype=str) == "R"
        matrix = np.column_stack([
            component_artifacts[name][INPUTS[name][1]].astype(np.float64) for name in names
        ])
        folds[year] = {
            "artifact": parent_artifact,
            "parent": parent_artifact["catboost_outcome"].astype(np.float64),
            "matrix": matrix, "f_prediction": f_artifact["final_prediction"].astype(np.float64),
            "regular": regular,
            "masks": {"full": np.ones(len(regular), dtype=bool), "R": regular, "F": ~regular},
        }
        hashes[str(year)] = {
            "parent": digest(parent_path), "F": digest(f_path),
            **{name: digest(path) for name, path in component_paths.items()},
        }

    trials: list[dict[str, Any]] = []
    cache: dict[tuple[tuple[str, str], float, int], np.ndarray] = {}
    for left, right in combinations(range(len(names)), 2):
        pair = (names[left], names[right])
        for units in range(1, 20):
            weight = units / 20.0
            years: dict[str, Any] = {}
            for year in YEARS:
                fold = folds[year]
                r_prediction = weight * fold["matrix"][:, left] + (1.0 - weight) * fold["matrix"][:, right]
                candidate = np.clip(np.where(fold["regular"], r_prediction, fold["f_prediction"]), 1e-6, 1.0 - 1e-6)
                cache[(pair, weight, year)] = candidate
                years[str(year)] = {
                    route: float(score(fold["artifact"]["y"], candidate, mask)["score"] - score(fold["artifact"]["y"], fold["parent"], mask)["score"])
                    for route, mask in fold["masks"].items()
                }
            trials.append({
                "components": pair, "weights": [weight, 1.0 - weight],
                "minimum_full_gain": float(min(years[str(y)]["full"] for y in YEARS)),
                "minimum_R_gain": float(min(years[str(y)]["R"] for y in YEARS)),
                "mean_full_gain": float(np.mean([years[str(y)]["full"] for y in YEARS])),
                "years": years,
            })
    selected = max(trials, key=lambda x: (x["minimum_full_gain"], x["minimum_R_gain"], x["mean_full_gain"]))
    expected = lock["locked_recipe"]["R"]["components"]
    if tuple(item["name"] for item in expected) != tuple(selected["components"]) or not np.allclose([item["weight"] for item in expected], selected["weights"], atol=1e-12, rtol=0.0):
        raise AssertionError(f"source selection disagrees with lock: {selected}")

    confirmatory: dict[str, Any] = {}
    checks: dict[str, Any] = {}
    gate = lock["source_gate"]
    passed = True
    pair = tuple(selected["components"])
    weight = float(selected["weights"][0])
    artifacts: dict[str, Any] = {}
    for year in YEARS:
        fold = folds[year]
        candidate = cache[(pair, weight, year)]
        confirmatory[str(year)] = evaluate(
            fold["artifact"]["y"], fold["parent"], candidate,
            fold["artifact"]["cluster"], fold["masks"], 8470000 + year * 10000,
        )
        local = {
            "R_point": confirmatory[str(year)]["R"]["gain"] >= float(gate["minimum_R_gain_each_year"]),
            "F_point": confirmatory[str(year)]["F"]["gain"] >= float(gate["minimum_F_gain_each_year"]),
            "full_point": confirmatory[str(year)]["full"]["gain"] >= float(gate["minimum_full_gain_each_year"]),
            "R_ci": confirmatory[str(year)]["R"]["pitcher_cluster_95_ci"]["ci_low"] > 0.0,
            "F_ci": confirmatory[str(year)]["F"]["pitcher_cluster_95_ci"]["ci_low"] > 0.0,
            "full_ci": confirmatory[str(year)]["full"]["pitcher_cluster_95_ci"]["ci_low"] > 0.0,
        }
        checks[str(year)] = local
        passed = passed and all(local.values())
        output = PRED / f"v5_recent_routed_source_{year}.npz"
        if output.exists():
            raise FileExistsError(f"immutable artifact exists: {output}")
        np.savez_compressed(
            output, y=fold["artifact"]["y"].astype(np.int8), row_index=fold["artifact"]["row_index"].astype(np.int64),
            cluster=fold["artifact"]["cluster"], parent_exact_c=fold["parent"], final_prediction=candidate,
        )
        artifacts[str(year)] = {"path": str(output.relative_to(ROOT)), "sha256": digest(output)}
    report = {
        "experiment_id": lock["experiment_id"], "status": "source_pass" if passed else "source_failed",
        "lock_sha256": digest(LOCK), "years_read": list(YEARS), "years_not_read": [2022, 2023, 2024],
        "input_sha256": hashes, "trials": trials, "selected": selected,
        "confirmatory_metrics": confirmatory, "source_gate": {"requirements": gate, "checks": checks, "pass": bool(passed)},
        "artifacts": artifacts, "goal_status": "active", "goal_completion_claimed": False,
    }
    REPORT.write_text(json.dumps(safe(report), ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(safe({"status": report["status"], "selected": selected, "metrics": confirmatory, "checks": checks}), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
