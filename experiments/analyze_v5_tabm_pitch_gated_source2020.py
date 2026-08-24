#!/usr/bin/env python3
"""Immutable 2020 first gate for end-to-end pitch-gated control TabM."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.analyze_v5_dense_pitchtype_moe import evaluate, load, safe  # noqa: E402

RESULTS = ROOT / "experiments/results"
PRED = RESULTS / "predictions"
PREREG = ROOT / "experiments/params/v5_tabm_pitch_gated_preregister.json"
PARAMS = ROOT / "experiments/params/v5_tabm_pitch_gated.json"
STAGE = "v5_tabm_pitch_gated_source2020"
REPORT = RESULTS / "v5_tabm_pitch_gated_source2020_gate.json"
YEAR = 2020
KEY = "tabm_pitch_gated"


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def main() -> None:
    if REPORT.exists():
        raise FileExistsError(f"immutable report exists: {REPORT}")
    prereg = json.loads(PREREG.read_text(encoding="utf-8"))
    params = json.loads(PARAMS.read_text(encoding="utf-8"))
    candidate_path = PRED / f"{STAGE}_{YEAR}.npz"
    parent_path = PRED / f"v4_m3_c_backtest_{YEAR}_{YEAR}.npz"
    metadata_path = RESULTS / f"{STAGE}.json"
    candidate = load(candidate_path)
    parent_artifact = load(parent_path)
    for key in ("y", "row_index", "cluster"):
        if not np.array_equal(candidate[key], parent_artifact[key]):
            raise ValueError(f"candidate/parent alignment mismatch: {key}")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    details = metadata["folds"][0]["fit_details"][KEY]
    semantic = {
        "architecture": details["architecture"],
        "probability_formula": details["probability_formula"],
        "history_usable_coverage": float(details["history_usable_coverage"]),
        "current_pitch_group_used_at_inference": bool(
            details["current_pitch_group_used_at_inference"]
        ),
        "validation_pitch_group_used": bool(details["validation_pitch_group_used"]),
        "validation_target_used_in_training": bool(
            details["validation_target_used_in_training"]
        ),
        "row_independent_inference": bool(details["row_independent_inference"]),
        "epochs": int(details["n_iter"]),
    }
    semantic["pass"] = bool(
        semantic["architecture"]
        == "supervised_soft_pitch_gate_times_control_experts"
        and semantic["probability_formula"]
        == "sum_g softmax(gate)_g * sigmoid(expert_g)"
        and semantic["history_usable_coverage"] >= 0.995
        and not semantic["current_pitch_group_used_at_inference"]
        and not semantic["validation_pitch_group_used"]
        and not semantic["validation_target_used_in_training"]
        and semantic["row_independent_inference"]
        and semantic["epochs"] == int(params["epochs"])
    )
    game_types = pd.read_csv(
        ROOT / "open/data/train.csv", usecols=["game_type"]
    )["game_type"].astype(str)
    game_type = game_types.iloc[
        candidate["row_index"].astype(np.int64)
    ].to_numpy(dtype=str)
    parent = parent_artifact["catboost_outcome"].astype(np.float64)
    raw = candidate[KEY].astype(np.float64)
    routed_raw = np.where(game_type == "R", raw, parent)
    masks = {"full": np.ones(len(raw), dtype=bool), "R": game_type == "R"}
    trials = []
    if semantic["pass"]:
        for gamma_value in prereg["source_protocol"]["gamma_grid"]:
            gamma = float(gamma_value)
            result = evaluate(
                candidate, parent, routed_raw, masks["full"], masks, gamma,
                2000, 882400 + int(gamma * 100),
            )
            trials.append({"gamma": gamma, **result})
    selected = (
        max(
            trials,
            key=lambda item: (
                item["routes"]["R"]["gain"],
                item["routes"]["full"]["gain"],
                -item["gamma"],
            ),
        )
        if trials else None
    )
    gate = prereg["source_protocol"]["gate"]
    checks = {"semantic": bool(semantic["pass"]), "selected": selected is not None}
    if selected is not None:
        checks.update({
            "R_gain": selected["routes"]["R"]["gain"] >= gate["minimum_R_gain_each_year"],
            "full_gain": selected["routes"]["full"]["gain"] >= gate["minimum_full_gain_each_year"],
            "R_ci": selected["routes"]["R"]["pitcher_cluster_95_ci"]["ci_low"] > 0.0,
            "full_ci": selected["routes"]["full"]["pitcher_cluster_95_ci"]["ci_low"] > 0.0,
        })
    passed = bool(all(checks.values()))
    report = {
        "experiment_id": prereg["experiment_id"],
        "status": "source_2020_pass" if passed else "source_failed_2020",
        "preregister_sha256": digest(PREREG),
        "params_sha256": digest(PARAMS),
        "script_sha256": digest(Path(__file__)),
        "years_read": [YEAR],
        "years_not_read": [2021, 2022, 2023, 2024],
        "semantic": semantic,
        "trials": trials,
        "selected": selected,
        "source_gate": {"requirements": gate, "checks": checks, "pass": passed},
        "next_action": "lock_gamma_then_train_2021" if passed else "close_before_2021",
        "goal_status": "active",
        "goal_completion_claimed": False,
    }
    REPORT.write_text(json.dumps(safe(report), ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(safe({
        "status": report["status"], "semantic": semantic,
        "selected": selected, "checks": checks,
    }), ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
