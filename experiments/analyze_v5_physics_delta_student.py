#!/usr/bin/env python3
"""Immutable 2022 Goal-scale gate for the physics-delta LUPI student."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.analyze_v5_pitchtype_moe import (  # noqa: E402
    BASELINES,
    evaluate_direction,
    load,
    load_baseline,
    strip_prediction,
)


PRED = ROOT / "experiments/results/predictions"
TRAIN = ROOT / "open/data/train.csv"
PREREG = ROOT / "experiments/params/v5_physics_delta_student_preregister.json"
CONTRACT = ROOT / "experiments/params/v5_validation_contract_v2.json"
REPORT = ROOT / "experiments/results/v5_physics_delta_student_2022_gate.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bootstrap", type=int, default=1000)
    return parser.parse_args()


def file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_immutable(path: Path, payload: dict) -> None:
    if path.exists():
        raise FileExistsError(f"immutable report already exists: {path}")
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )


def main() -> None:
    args = parse_args()
    prereg = json.loads(PREREG.read_text(encoding="utf-8"))
    stem = prereg["stages"]["2022"]
    candidate = load(PRED / f"{stem}_2022.npz")
    types_all = pd.read_csv(TRAIN, usecols=["game_type"], low_memory=False)[
        "game_type"
    ].astype(str).to_numpy()
    types = types_all[candidate["row_index"].astype(np.int64)]
    regular = types == "R"
    residual = candidate[prereg["candidate_key"]].astype(np.float64) - 0.5
    required = float(prereg["selection"]["early_2022_required_full_point_gain"])
    trials = []
    for gamma in prereg["selection"]["gamma_grid"]:
        cells = {}
        for baseline_index, name in enumerate(BASELINES):
            baseline = load_baseline(name, 2022, candidate)
            direction_prediction = np.clip(
                baseline + residual, 1e-6, 1.0 - 1e-6
            )
            result = evaluate_direction(
                candidate,
                baseline,
                types,
                direction_prediction,
                regular,
                float(gamma),
                args.bootstrap,
                871000 + 1000 * baseline_index + int(float(gamma) * 100),
            )
            cells[name] = strip_prediction(result)
        full_gains = [float(cell["metrics"]["all"]["gain"]) for cell in cells.values()]
        r_gains = [float(cell["metrics"]["R"]["gain"]) for cell in cells.values()]
        ci_lows = [float(cell["bootstrap_R"]["ci_low"]) for cell in cells.values()]
        trials.append({
            "gamma": float(gamma),
            "cells": cells,
            "minimum_full_gain": min(full_gains),
            "minimum_R_gain": min(r_gains),
            "minimum_R_ci_low": min(ci_lows),
            "goal_scale_eligible": bool(
                min(full_gains) > required
                and min(r_gains) > 0.0
                and min(ci_lows) > 0.0
            ),
        })
    eligible = [trial for trial in trials if trial["goal_scale_eligible"]]
    report = {
        "experiment_id": prereg["experiment_id"],
        "mode": "development_2022_goal_scale_early_gate",
        "preregister_sha256": file_hash(PREREG),
        "contract_sha256": file_hash(CONTRACT),
        "student_artifact_sha256": file_hash(PRED / f"{stem}_2022.npz"),
        "years_read_for_control_target": [2022],
        "years_not_read_for_control_target": [2023, 2024],
        "student_residual_mean_R": float(residual[regular].mean()),
        "student_residual_std_R": float(residual[regular].std()),
        "required_full_gain": required,
        "trials": trials,
        "eligible_gammas": [trial["gamma"] for trial in eligible],
        "status": "eligible_for_2023" if eligible else "failed_2022_goal_scale_gate",
    }
    write_immutable(REPORT, report)
    print(json.dumps({
        "status": report["status"],
        "student_residual_mean_R": report["student_residual_mean_R"],
        "student_residual_std_R": report["student_residual_std_R"],
        "trials": [
            {
                "gamma": trial["gamma"],
                "minimum_full_gain": trial["minimum_full_gain"],
                "minimum_R_gain": trial["minimum_R_gain"],
                "minimum_R_ci_low": trial["minimum_R_ci_low"],
            }
            for trial in trials
        ],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
