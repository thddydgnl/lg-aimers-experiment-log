#!/usr/bin/env python3
"""Immutable source selection and cluster gate for hashed-cross logistic."""

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

from experiments.analyze_v5_dense_pitchtype_moe import load, safe, score  # noqa: E402
from experiments.analyze_v5_game_centered_brier_source import digest, route_metrics  # noqa: E402
from experiments.run_v5_hashed_cross_source import (  # noqa: E402
    ALPHAS, DIMENSIONS, config_key,
)


YEARS = (2020, 2021)
TRAIN = ROOT / "open/data/train.csv"
PRED = ROOT / "experiments/results/predictions"
PREREG = ROOT / "experiments/params/v5_hashed_cross_preregister.json"
TRAINING_REPORT = ROOT / "experiments/results/v5_hashed_cross_source_training.json"
REPORT = ROOT / "experiments/results/v5_hashed_cross_source_gate.json"


def route(parent: np.ndarray, model: np.ndarray, regular: np.ndarray, gamma: float) -> np.ndarray:
    result = parent.astype(np.float64, copy=True)
    result[regular] = np.clip(
        (1.0 - gamma) * parent[regular] + gamma * model[regular], 1e-6, 1.0 - 1e-6
    )
    return result


def main() -> None:
    if REPORT.exists():
        raise FileExistsError(f"immutable report exists: {REPORT}")
    prereg = json.loads(PREREG.read_text(encoding="utf-8"))
    if prereg["status"] != "locked_before_source_metrics":
        raise ValueError("unexpected preregistration status")
    if prereg["source_protocol"]["years"] != list(YEARS):
        raise ValueError("source-year contract changed")
    if int(prereg["source_protocol"]["bootstrap_iterations"]) != 2000:
        raise ValueError("bootstrap contract changed")

    maximum_row = 0
    folds: dict[int, dict[str, Any]] = {}
    input_hashes: dict[str, Any] = {}
    for year in YEARS:
        parent_path = PRED / f"v4_m3_c_backtest_{year}_{year}.npz"
        model_path = PRED / f"v5_hashed_cross_source_{year}.npz"
        parent = load(parent_path)
        model = load(model_path)
        for field in ("y", "row_index", "cluster"):
            if not np.array_equal(parent[field], model[field]):
                raise ValueError(f"{year}: {field} alignment mismatch")
        row_index = parent["row_index"].astype(np.int64)
        maximum_row = max(maximum_row, int(row_index.max()))
        folds[year] = {
            "y": parent["y"].astype(np.int8),
            "row_index": row_index,
            "cluster": parent["cluster"],
            "parent": parent["catboost_outcome"].astype(np.float64),
            "models": {
                config_key(dimension, alpha): model[config_key(dimension, alpha)].astype(np.float64)
                for dimension in DIMENSIONS for alpha in ALPHAS
            },
        }
        input_hashes[str(year)] = {
            "parent": digest(parent_path), "hashed_models": digest(model_path)
        }

    frame = pd.read_csv(
        TRAIN,
        usecols=["season", "game_type", "control_success"],
        nrows=maximum_row + 1,
    )
    if int(frame["season"].max()) != 2021:
        raise ValueError("source reader crossed 2021 boundary")
    for year in YEARS:
        fold = folds[year]
        rows = frame.loc[fold["row_index"]]
        if not rows["season"].eq(year).all():
            raise ValueError(f"{year}: season mismatch")
        if not np.array_equal(rows["control_success"].to_numpy(np.int8), fold["y"]):
            raise ValueError(f"{year}: target mismatch")
        fold["regular"] = rows["game_type"].astype(str).eq("R").to_numpy()

    trials: list[dict[str, Any]] = []
    for dimension in DIMENSIONS:
        for alpha in ALPHAS:
            key = config_key(dimension, alpha)
            for gamma_value in prereg["candidate_family"]["blend_grid"]:
                gamma = float(gamma_value)
                year_metrics: dict[str, Any] = {}
                for year in YEARS:
                    fold = folds[year]
                    candidate = route(
                        fold["parent"], fold["models"][key], fold["regular"], gamma
                    )
                    year_metrics[str(year)] = {}
                    for name, mask in {
                        "full": np.ones(len(candidate), dtype=bool),
                        "R": fold["regular"].astype(bool),
                    }.items():
                        parent_score = score(fold["y"], fold["parent"], mask)["score"]
                        candidate_score = score(fold["y"], candidate, mask)["score"]
                        year_metrics[str(year)][name] = {
                            "gain": float(candidate_score - parent_score)
                        }
                trials.append({
                    "key": key, "dimension": dimension, "alpha": alpha, "gamma": gamma,
                    "minimum_full_gain": float(min(year_metrics[str(y)]["full"]["gain"] for y in YEARS)),
                    "minimum_R_gain": float(min(year_metrics[str(y)]["R"]["gain"] for y in YEARS)),
                    "mean_full_gain": float(np.mean([year_metrics[str(y)]["full"]["gain"] for y in YEARS])),
                    "years": year_metrics,
                })
    selected = max(
        trials,
        key=lambda item: (
            item["minimum_full_gain"], item["minimum_R_gain"], item["mean_full_gain"],
            item["alpha"], -item["dimension"], -item["gamma"],
        ),
    )

    selected_metrics: dict[str, Any] = {}
    artifacts: dict[str, Any] = {}
    for offset, year in enumerate(YEARS):
        fold = folds[year]
        candidate = route(
            fold["parent"], fold["models"][selected["key"]], fold["regular"],
            float(selected["gamma"]),
        )
        selected_metrics[str(year)] = route_metrics(
            fold["y"], fold["parent"], candidate, fold["cluster"],
            fold["regular"].astype(bool), seed=8233000 + 10000 * offset,
        )
        output = PRED / f"v5_hashed_cross_selected_source_{year}.npz"
        if output.exists():
            raise FileExistsError(f"immutable artifact exists: {output}")
        np.savez_compressed(
            output, y=fold["y"], row_index=fold["row_index"], cluster=fold["cluster"],
            parent_exact_c=fold["parent"].astype(np.float32),
            hashed_cross=fold["models"][selected["key"]].astype(np.float32),
            final_prediction=candidate.astype(np.float32),
            dimension=np.asarray(selected["dimension"], dtype=np.int64),
            alpha=np.asarray(selected["alpha"], dtype=np.float64),
            gamma=np.asarray(selected["gamma"], dtype=np.float64),
        )
        artifacts[str(year)] = {
            "path": str(output.relative_to(ROOT)), "sha256": digest(output)
        }

    other_dimension = next(value for value in DIMENSIONS if value != selected["dimension"])
    other_key = config_key(other_dimension, float(selected["alpha"]))
    other_minimum_r = min(
        next(
            item["minimum_R_gain"] for item in trials
            if item["key"] == other_key and item["gamma"] == selected["gamma"]
        ),
        float("inf"),
    )
    requirements = prereg["source_protocol"]["advance_gate"]
    checks: dict[str, bool] = {}
    for year in YEARS:
        result = selected_metrics[str(year)]
        checks[f"{year}_full_gain"] = result["full"]["gain"] >= float(requirements["minimum_full_gain_each_year"])
        checks[f"{year}_R_gain"] = result["R"]["gain"] >= float(requirements["minimum_R_gain_each_year"])
        checks[f"{year}_full_ci"] = result["full"]["pitcher_cluster_95_ci"]["ci_low"] > float(requirements["full_pitcher_cluster_95_ci_low_each_year"])
        checks[f"{year}_R_ci"] = result["R"]["pitcher_cluster_95_ci"]["ci_low"] > float(requirements["R_pitcher_cluster_95_ci_low_each_year"])
    checks["other_dimension_positive"] = other_minimum_r > float(requirements["other_dimension_minimum_R_gain"])
    passed = bool(all(checks.values()))
    report = {
        "experiment_id": prereg["experiment_id"],
        "status": "source_pass" if passed else "source_failed",
        "preregister_sha256": digest(PREREG),
        "training_report_sha256": digest(TRAINING_REPORT),
        "script_sha256": digest(Path(__file__)),
        "years_read": list(YEARS), "years_not_read": [2022, 2023, 2024],
        "selection": selected, "selected_metrics": selected_metrics,
        "other_dimension_same_alpha_gamma": {
            "dimension": other_dimension, "key": other_key,
            "minimum_R_gain": other_minimum_r,
        },
        "trials": trials, "input_sha256": input_hashes,
        "gate": {"requirements": requirements, "checks": checks, "pass": passed},
        "artifacts": artifacts, "goal_status": "active",
        "goal_completion_claimed": False,
    }
    REPORT.write_text(json.dumps(safe(report), ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(safe({
        "status": report["status"], "selection": selected,
        "selected_metrics": selected_metrics,
        "other_dimension": report["other_dimension_same_alpha_gamma"],
        "gate": report["gate"],
    }), ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
