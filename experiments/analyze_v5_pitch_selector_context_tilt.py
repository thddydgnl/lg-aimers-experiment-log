#!/usr/bin/env python3
"""Apply the immutable 2022 Goal gate to the locked context-tilt selector."""

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
from experiments.run_e22r_probs_rolling import GROUPS  # noqa: E402


PRED = ROOT / "experiments/results/predictions"
TRAIN = ROOT / "open/data/train.csv"
PREREG = ROOT / "experiments/params/v5_pitch_selector_context_tilt_preregister.json"
CONTRACT = ROOT / "experiments/params/v5_validation_contract_v2.json"
SELECTION = ROOT / "experiments/results/v5_pitch_selector_context_tilt_selection.json"
MATERIALIZATION = ROOT / "experiments/results/v5_pitch_selector_context_tilt_2022.json"
ARTIFACT = PRED / "v5_pitch_selector_context_tilt_2022.npz"
REPORT = ROOT / "experiments/results/v5_pitch_selector_context_tilt_2022_gate.json"


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


def stage1_diagnostic(candidate: dict[str, np.ndarray], types: np.ndarray) -> dict:
    probability = np.column_stack(
        [candidate[f"selector_p_{group}"] for group in GROUPS]
    ).astype(np.float64)
    truth = candidate["diagnostic_true_group_code"].astype(np.int16)
    route = (types == "R") & (truth >= 0)
    selected = probability[route]
    labels = truth[route]
    true_probability = selected[np.arange(len(selected)), labels]
    return {
        "matched_regular_rows": int(route.sum()),
        "accuracy": float(np.mean(np.argmax(selected, axis=1) == labels)),
        "multiclass_log_loss": float(
            -np.mean(np.log(np.maximum(true_probability, 1e-12)))
        ),
        "mean_true_group_probability": float(np.mean(true_probability)),
        "probability_mean": {
            group: float(selected[:, index].mean())
            for index, group in enumerate(GROUPS)
        },
    }


def main() -> None:
    args = parse_args()
    prereg = json.loads(PREREG.read_text(encoding="utf-8"))
    selection = json.loads(SELECTION.read_text(encoding="utf-8"))
    materialization = json.loads(MATERIALIZATION.read_text(encoding="utf-8"))
    if selection["preregister_sha256"] != file_hash(PREREG):
        raise ValueError("selection/preregister hash mismatch")
    if materialization["selection_sha256"] != file_hash(SELECTION):
        raise ValueError("materialization/selection hash mismatch")
    candidate = load(ARTIFACT)
    types_all = pd.read_csv(TRAIN, usecols=["game_type"], low_memory=False)[
        "game_type"
    ].astype(str).to_numpy()
    types = types_all[candidate["row_index"].astype(np.int64)]
    regular = types == "R"
    direction = candidate[prereg["candidate_2022_key"]].astype(np.float64)
    required = float(prereg["goal_gate"]["required_minimum_full_gain"])
    trials = []
    for gamma in prereg["goal_gate"]["gamma_grid"]:
        cells = {}
        for baseline_index, name in enumerate(BASELINES):
            baseline = load_baseline(name, 2022, candidate)
            result = evaluate_direction(
                candidate,
                baseline,
                types,
                direction,
                regular,
                float(gamma),
                args.bootstrap,
                861000 + 1000 * baseline_index + int(float(gamma) * 100),
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
        "selection_sha256": file_hash(SELECTION),
        "materialization_sha256": file_hash(MATERIALIZATION),
        "years_read_for_control_target": [2022],
        "years_not_read_for_control_target": [2023, 2024],
        "required_full_gain": required,
        "selector_diagnostic": stage1_diagnostic(candidate, types),
        "trials": trials,
        "eligible_gammas": [trial["gamma"] for trial in eligible],
        "status": "eligible_for_2023" if eligible else "failed_2022_goal_scale_gate",
    }
    write_immutable(REPORT, report)
    print(json.dumps({
        "status": report["status"],
        "selected_selector": selection["selected"],
        "selector_diagnostic": report["selector_diagnostic"],
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
