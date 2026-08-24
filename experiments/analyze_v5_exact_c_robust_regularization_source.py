#!/usr/bin/env python3
"""Immutable source gate for the regularized exact-C recipe."""

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
from experiments.analyze_v5_game_centered_brier_source import (  # noqa: E402
    digest,
    route_metrics,
)


YEARS = (2020, 2021)
TRAIN = ROOT / "open/data/train.csv"
PRED = ROOT / "experiments/results/predictions"
PREREG = ROOT / "experiments/params/v5_exact_c_robust_regularization_preregister.json"
MODEL_PARAMS = ROOT / "experiments/params/v5_exact_c_robust_regularization_model.json"
ENGINE = ROOT / "experiments/run_v2_rolling.py"
REPORT = ROOT / "experiments/results/v5_exact_c_robust_regularization_source_gate.json"
KEY = "catboost_outcome"


def main() -> None:
    if REPORT.exists():
        raise FileExistsError(f"immutable report exists: {REPORT}")
    prereg = json.loads(PREREG.read_text(encoding="utf-8"))
    if prereg["status"] != "locked_before_source_metrics":
        raise ValueError("unexpected preregistration status")
    if prereg["source_protocol"]["years"] != list(YEARS):
        raise ValueError("source-year contract changed")
    if int(prereg["source_protocol"]["bootstrap_iterations"]) != 2000:
        raise ValueError("route_metrics is locked to 2000 bootstrap iterations")

    folds: dict[int, dict[str, np.ndarray]] = {}
    maximum_row = 0
    input_hashes: dict[str, Any] = {}
    for year in YEARS:
        parent_path = PRED / f"v4_m3_c_backtest_{year}_{year}.npz"
        candidate_path = PRED / f"v5_exact_c_robust_regularization_source_{year}.npz"
        parent_artifact = load(parent_path)
        candidate_artifact = load(candidate_path)
        for field in ("y", "row_index", "cluster"):
            if not np.array_equal(parent_artifact[field], candidate_artifact[field]):
                raise ValueError(f"{year}: {field} alignment mismatch")
        row_index = parent_artifact["row_index"].astype(np.int64)
        maximum_row = max(maximum_row, int(row_index.max()))
        folds[year] = {
            "y": parent_artifact["y"].astype(np.int8),
            "row_index": row_index,
            "cluster": parent_artifact["cluster"],
            "parent": parent_artifact[KEY].astype(np.float64),
            "model": candidate_artifact[KEY].astype(np.float64),
        }
        input_hashes[str(year)] = {
            "parent": digest(parent_path),
            "regularized_model": digest(candidate_path),
        }

    frame = pd.read_csv(
        TRAIN,
        usecols=["season", "game_type", "control_success"],
        nrows=maximum_row + 1,
    )
    if int(frame["season"].max()) != max(YEARS):
        raise ValueError("source reader crossed the locked 2021 boundary")
    for year in YEARS:
        fold = folds[year]
        rows = frame.loc[fold["row_index"]]
        if not rows["season"].eq(year).all():
            raise ValueError(f"{year}: season mismatch")
        if not np.array_equal(
            rows["control_success"].to_numpy(dtype=np.int8), fold["y"]
        ):
            raise ValueError(f"{year}: target mismatch")
        fold["regular"] = rows["game_type"].astype(str).eq("R").to_numpy()

    trials: list[dict[str, Any]] = []
    for gamma_value in prereg["candidate"]["blend_grid"]:
        gamma = float(gamma_value)
        year_metrics: dict[str, Any] = {}
        for year in YEARS:
            fold = folds[year]
            candidate = np.clip(
                (1.0 - gamma) * fold["parent"] + gamma * fold["model"],
                1e-6,
                1.0 - 1e-6,
            )
            masks = {
                "full": np.ones(len(candidate), dtype=bool),
                "R": fold["regular"].astype(bool),
            }
            year_metrics[str(year)] = {}
            for route, mask in masks.items():
                parent_score = score(fold["y"], fold["parent"], mask)["score"]
                candidate_score = score(fold["y"], candidate, mask)["score"]
                year_metrics[str(year)][route] = {
                    "gain": float(candidate_score - parent_score)
                }
        trials.append(
            {
                "gamma": gamma,
                "minimum_full_gain": float(
                    min(year_metrics[str(year)]["full"]["gain"] for year in YEARS)
                ),
                "minimum_R_gain": float(
                    min(year_metrics[str(year)]["R"]["gain"] for year in YEARS)
                ),
                "mean_full_gain": float(
                    np.mean(
                        [year_metrics[str(year)]["full"]["gain"] for year in YEARS]
                    )
                ),
                "years": year_metrics,
            }
        )
    selected = max(
        trials,
        key=lambda item: (
            item["minimum_full_gain"],
            item["minimum_R_gain"],
            item["mean_full_gain"],
            -item["gamma"],
        ),
    )

    selected_metrics: dict[str, Any] = {}
    artifacts: dict[str, Any] = {}
    gamma = float(selected["gamma"])
    for offset, year in enumerate(YEARS):
        fold = folds[year]
        candidate = np.clip(
            (1.0 - gamma) * fold["parent"] + gamma * fold["model"],
            1e-6,
            1.0 - 1e-6,
        )
        selected_metrics[str(year)] = route_metrics(
            fold["y"],
            fold["parent"],
            candidate,
            fold["cluster"],
            fold["regular"].astype(bool),
            seed=8229000 + 10000 * offset,
        )
        output = PRED / f"v5_exact_c_robust_regularization_selected_source_{year}.npz"
        if output.exists():
            raise FileExistsError(f"immutable artifact exists: {output}")
        np.savez_compressed(
            output,
            y=fold["y"],
            row_index=fold["row_index"],
            cluster=fold["cluster"],
            parent_exact_c=fold["parent"].astype(np.float32),
            regularized_exact_c=fold["model"].astype(np.float32),
            final_prediction=candidate.astype(np.float32),
            gamma=np.asarray(gamma, dtype=np.float64),
        )
        artifacts[str(year)] = {
            "path": str(output.relative_to(ROOT)),
            "sha256": digest(output),
        }

    requirements = prereg["source_protocol"]["advance_gate"]
    checks: dict[str, bool] = {}
    for year in YEARS:
        result = selected_metrics[str(year)]
        checks[f"{year}_full_gain"] = bool(
            result["full"]["gain"]
            >= float(requirements["minimum_full_gain_each_year"])
        )
        checks[f"{year}_R_gain"] = bool(
            result["R"]["gain"] >= float(requirements["minimum_R_gain_each_year"])
        )
        checks[f"{year}_full_ci"] = bool(
            result["full"]["pitcher_cluster_95_ci"]["ci_low"]
            > float(requirements["full_pitcher_cluster_95_ci_low_each_year"])
        )
        checks[f"{year}_R_ci"] = bool(
            result["R"]["pitcher_cluster_95_ci"]["ci_low"]
            > float(requirements["R_pitcher_cluster_95_ci_low_each_year"])
        )
    passed = bool(all(checks.values()))
    report = {
        "experiment_id": prereg["experiment_id"],
        "status": "source_pass" if passed else "source_failed",
        "preregister_sha256": digest(PREREG),
        "model_params_sha256": digest(MODEL_PARAMS),
        "engine_sha256": digest(ENGINE),
        "script_sha256": digest(Path(__file__)),
        "years_read": list(YEARS),
        "years_not_read": [2022, 2023, 2024],
        "selection": selected,
        "selected_metrics": selected_metrics,
        "trials": trials,
        "input_sha256": input_hashes,
        "gate": {"requirements": requirements, "checks": checks, "pass": passed},
        "artifacts": artifacts,
        "goal_status": "active",
        "goal_completion_claimed": False,
    }
    REPORT.write_text(
        json.dumps(safe(report), ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(
        json.dumps(
            safe(
                {
                    "status": report["status"],
                    "selection": selected,
                    "selected_metrics": selected_metrics,
                    "gate": report["gate"],
                }
            ),
            ensure_ascii=False,
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
