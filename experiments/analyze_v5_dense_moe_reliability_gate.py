#!/usr/bin/env python3
"""Source-screen a row-local confidence gate for immutable dense MoE."""

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
    PREFIX,
    digest,
    load,
    safe,
    score,
)
from experiments.run_v5_h1_residual import (  # noqa: E402
    cluster_bootstrap_score_gain,
)


PREREG = (
    ROOT / "experiments/params/v5_dense_moe_reliability_gate_preregister.json"
)
DENSE_REPORT = ROOT / "experiments/results/v5_dense_pitchtype_moe_source_gate_v2.json"
REPORT = ROOT / "experiments/results/v5_dense_moe_reliability_gate_source.json"
TRAIN = ROOT / "open/data/train.csv"
YEARS = (2020, 2021)
PARENT = {
    2020: "v4_m3_c_backtest_2020_2020.npz",
    2021: "v4_m3_c_backtest_2021_2021.npz",
}
STAGES = {
    2020: "v5_dense_pitchtype_moe_source2020",
    2021: "v5_dense_pitchtype_moe_source2021",
}
KEY = "catboost_dense_pitchtype_moe"
GROUPS = ("fastball", "breaking", "offspeed")


def evaluate_candidate(
    artifact: dict[str, np.ndarray],
    parent: np.ndarray,
    candidate: np.ndarray,
    masks: dict[str, np.ndarray],
    iterations: int,
    seed: int,
) -> dict[str, Any]:
    result: dict[str, Any] = {"routes": {}}
    for index, (name, mask) in enumerate(masks.items()):
        parent_metrics = score(artifact["y"], parent, mask)
        candidate_metrics = score(artifact["y"], candidate, mask)
        interval = cluster_bootstrap_score_gain(
            artifact["y"], parent, candidate, artifact["cluster"], mask,
            iterations=iterations, seed=seed + 1000 * index,
        )
        gain = candidate_metrics["score"] - parent_metrics["score"]
        if abs(gain - interval["point"]) > 1e-8:
            raise AssertionError(f"score/CI mismatch: {name}")
        result["routes"][name] = {
            "parent": parent_metrics,
            "candidate": candidate_metrics,
            "gain": gain,
            "pitcher_cluster_95_ci": interval,
        }
    return result


def main() -> None:
    if REPORT.exists():
        raise FileExistsError(f"immutable result already exists: {REPORT}")
    prereg = json.loads(PREREG.read_text(encoding="utf-8"))
    dense_report = json.loads(DENSE_REPORT.read_text(encoding="utf-8"))
    if dense_report["status"] != "source_failed":
        raise ValueError("unexpected immutable dense-MoE source status")
    all_types = pd.read_csv(TRAIN, usecols=["game_type"])["game_type"].astype(str)
    folds: dict[int, dict[str, Any]] = {}
    for year in YEARS:
        parent_path = PRED / PARENT[year]
        dense_path = PRED / f"{STAGES[year]}_{year}.npz"
        parent_artifact = load(parent_path)
        dense = load(dense_path)
        for key in ("y", "row_index", "cluster"):
            if not np.array_equal(parent_artifact[key], dense[key]):
                raise ValueError(f"alignment mismatch: {year}/{key}")
        parent = parent_artifact["catboost_outcome"].astype(np.float64)
        probabilities = np.column_stack(
            [dense[f"{PREFIX}p_{group}"] for group in GROUPS]
        ).astype(np.float64)
        reliability = np.clip(
            (probabilities.max(axis=1) - 1.0 / 3.0) / (2.0 / 3.0),
            0.0,
            1.0,
        )
        types = all_types.iloc[
            parent_artifact["row_index"].astype(np.int64)
        ].to_numpy(dtype=str)
        regular = types == "R"
        folds[year] = {
            "artifact": dense,
            "parent": parent,
            "dense": dense[KEY].astype(np.float64),
            "reliability": reliability,
            "regular": regular,
            "masks": {
                "full": np.ones(len(parent), dtype=bool),
                "R": regular,
            },
            "paths": {"parent": parent_path, "dense": dense_path},
        }

    iterations = int(prereg["bootstrap_iterations"])
    trials: list[dict[str, Any]] = []
    prediction_cache: dict[tuple[int, float], np.ndarray] = {}
    for outer_scale in prereg["reliability"]["outer_scale_grid"]:
        years: dict[str, Any] = {}
        for year in YEARS:
            fold = folds[year]
            effective = np.clip(
                float(outer_scale) * fold["reliability"], 0.0, 1.0
            )
            prediction = fold["parent"].copy()
            route = fold["regular"]
            prediction[route] += effective[route] * (
                fold["dense"][route] - fold["parent"][route]
            )
            prediction = np.clip(prediction, 1e-6, 1.0 - 1e-6)
            prediction_cache[(year, float(outer_scale))] = prediction
            evaluated = evaluate_candidate(
                fold["artifact"],
                fold["parent"],
                prediction,
                fold["masks"],
                iterations,
                1010000 + 10000 * year + int(float(outer_scale) * 100),
            )
            evaluated["effective_scale"] = {
                "mean_R": float(effective[route].mean()),
                "median_R": float(np.median(effective[route])),
                "p90_R": float(np.quantile(effective[route], 0.9)),
                "saturated_R_rate": float(np.mean(effective[route] >= 1.0)),
            }
            years[str(year)] = evaluated
        full_gains = [
            years[str(year)]["routes"]["full"]["gain"] for year in YEARS
        ]
        r_gains = [
            years[str(year)]["routes"]["R"]["gain"] for year in YEARS
        ]
        trials.append(
            {
                "outer_scale": float(outer_scale),
                "minimum_full_gain": float(min(full_gains)),
                "minimum_R_gain": float(min(r_gains)),
                "mean_full_gain": float(np.mean(full_gains)),
                "years": years,
            }
        )
    selected = max(
        trials,
        key=lambda item: (
            item["minimum_full_gain"],
            item["minimum_R_gain"],
            -item["outer_scale"],
        ),
    )
    minimum_full = float(prereg["source_gate"]["minimum_full_gain_each_year"])
    minimum_r = float(prereg["source_gate"]["minimum_r_gain_each_year"])
    checks: list[bool] = []
    for year in YEARS:
        routes = selected["years"][str(year)]["routes"]
        checks.extend(
            (
                routes["full"]["gain"] >= minimum_full,
                routes["R"]["gain"] >= minimum_r,
                routes["full"]["pitcher_cluster_95_ci"]["ci_low"] > 0.0,
                routes["R"]["pitcher_cluster_95_ci"]["ci_low"] > 0.0,
            )
        )
    passed = bool(all(checks))
    artifacts: dict[str, Any] = {}
    for year in YEARS:
        output = PRED / f"v5_dense_moe_reliability_gate_source_{year}.npz"
        if output.exists():
            raise FileExistsError(f"immutable artifact exists: {output}")
        fold = folds[year]
        np.savez_compressed(
            output,
            y=fold["artifact"]["y"].astype(np.int8),
            row_index=fold["artifact"]["row_index"].astype(np.int64),
            cluster=fold["artifact"]["cluster"],
            parent_exact_c=fold["parent"],
            reliability=fold["reliability"],
            final_prediction=prediction_cache[(year, selected["outer_scale"])],
        )
        artifacts[str(year)] = {
            "path": str(output.relative_to(ROOT)),
            "sha256": digest(output),
            "parent": str(fold["paths"]["parent"].relative_to(ROOT)),
            "dense": str(fold["paths"]["dense"].relative_to(ROOT)),
        }
    report = {
        "experiment_id": prereg["experiment_id"],
        "status": "source_pass" if passed else "source_failed",
        "preregister_sha256": digest(PREREG),
        "script_sha256": digest(Path(__file__)),
        "dense_report": {
            "path": str(DENSE_REPORT.relative_to(ROOT)),
            "sha256": digest(DENSE_REPORT),
        },
        "years_read": list(YEARS),
        "years_not_read": [2022, 2023, 2024],
        "trials": trials,
        "selected": selected,
        "artifacts": artifacts,
        "source_gate": {
            "minimum_full_gain_each_year": minimum_full,
            "minimum_R_gain_each_year": minimum_r,
            "ci_lower_positive_each_year": True,
            "passed": passed,
            "decision": (
                "freeze and advance to 2022/2023"
                if passed
                else "close without reading 2022+ candidate labels"
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
                "selected_outer_scale": selected["outer_scale"],
                "minimum_full_gain": selected["minimum_full_gain"],
                "minimum_R_gain": selected["minimum_R_gain"],
                "per_year": {
                    str(year): {
                        route: {
                            "gain": selected["years"][str(year)]["routes"][route]["gain"],
                            "ci_low": selected["years"][str(year)]["routes"][route][
                                "pitcher_cluster_95_ci"
                            ]["ci_low"],
                        }
                        for route in ("full", "R")
                    }
                    for year in YEARS
                },
                "effective_scale": {
                    str(year): selected["years"][str(year)]["effective_scale"]
                    for year in YEARS
                },
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
