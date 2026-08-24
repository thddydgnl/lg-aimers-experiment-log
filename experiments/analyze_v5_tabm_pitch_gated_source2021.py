#!/usr/bin/env python3
"""Locked 2021 confirmation of the 2020-selected pitch-gated recipe."""

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
LOCK = ROOT / "experiments/params/v5_tabm_pitch_gated_source2020_lock.json"
REPORT = RESULTS / "v5_tabm_pitch_gated_source2021_gate.json"
STAGE = "v5_tabm_pitch_gated_source2021"
YEAR = 2021
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
    lock = json.loads(LOCK.read_text(encoding="utf-8"))
    if lock["status"] != "locked_after_2020_before_2021_training":
        raise ValueError("unexpected source lock status")
    gamma = float(lock["locked_recipe"]["gamma"])
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
        "history_usable_coverage": float(details["history_usable_coverage"]),
        "current_pitch_group_used_at_inference": bool(details["current_pitch_group_used_at_inference"]),
        "validation_pitch_group_used": bool(details["validation_pitch_group_used"]),
        "validation_target_used_in_training": bool(details["validation_target_used_in_training"]),
        "row_independent_inference": bool(details["row_independent_inference"]),
        "epochs": int(details["n_iter"]),
    }
    semantic["pass"] = bool(
        semantic["architecture"] == "supervised_soft_pitch_gate_times_control_experts"
        and semantic["history_usable_coverage"] >= 0.995
        and not semantic["current_pitch_group_used_at_inference"]
        and not semantic["validation_pitch_group_used"]
        and not semantic["validation_target_used_in_training"]
        and semantic["row_independent_inference"]
        and semantic["epochs"] == int(lock["locked_recipe"]["epochs"])
    )
    game_types = pd.read_csv(
        ROOT / "open/data/train.csv", usecols=["game_type"]
    )["game_type"].astype(str)
    game_type = game_types.iloc[candidate["row_index"].astype(np.int64)].to_numpy(dtype=str)
    parent = parent_artifact["catboost_outcome"].astype(np.float64)
    raw = candidate[KEY].astype(np.float64)
    routed_raw = np.where(game_type == "R", raw, parent)
    masks = {"full": np.ones(len(raw), dtype=bool), "R": game_type == "R"}
    result = evaluate(
        candidate, parent, routed_raw, masks["full"], masks, gamma,
        2000, 882500,
    )
    gate = lock["next_gate"]
    checks = {
        "semantic": semantic["pass"],
        "gamma_unchanged": gamma == 0.75,
        "R_gain": result["routes"]["R"]["gain"] >= gate["minimum_R_gain"],
        "full_gain": result["routes"]["full"]["gain"] >= gate["minimum_full_gain"],
        "R_ci": result["routes"]["R"]["pitcher_cluster_95_ci"]["ci_low"] > 0.0,
        "full_ci": result["routes"]["full"]["pitcher_cluster_95_ci"]["ci_low"] > 0.0,
    }
    passed = bool(all(checks.values()))
    payload = {
        "experiment_id": lock["experiment_id"],
        "status": "source_pass" if passed else "source_failed_2021_confirmation",
        "source_lock_sha256": digest(LOCK),
        "locked_gamma": gamma,
        "gamma_reselected": False,
        "years_read": [2021],
        "years_not_read": [2022, 2023, 2024],
        "semantic": semantic,
        "result": result,
        "gate": {"requirements": gate, "checks": checks, "pass": passed},
        "next_action": "freeze_for_2022_2023_development" if passed else "close_before_later_years",
        "goal_status": "active",
        "goal_completion_claimed": False,
    }
    REPORT.write_text(json.dumps(safe(payload), ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(safe({
        "status": payload["status"], "locked_gamma": gamma,
        "result": result, "checks": checks,
    }), ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
