#!/usr/bin/env python3
"""Apply the frozen gamma to the untouched 2021 neural multitask fold."""

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
LOCK = ROOT / "experiments/params/v5_neural_dense_multitask_source_lock.json"
REPORT = RESULTS / "v5_neural_dense_multitask_source2021_gate.json"
STAGE = "v5_neural_dense_multitask_source2021"
YEAR = 2021
KEY = "tabm_dense_multitask"


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def main() -> None:
    if REPORT.exists():
        raise FileExistsError(f"immutable report exists: {REPORT}")
    lock = json.loads(LOCK.read_text(encoding="utf-8"))
    first_gate = ROOT / lock["first_gate_report"]
    if digest(first_gate) != lock["first_gate_report_sha256"]:
        raise ValueError("2020 first-gate report changed after gamma lock")
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
        "target_heads": details["target_heads"],
        "head_weights": details["head_weights"],
        "history_usable_coverage": float(details["history_usable_coverage"]),
        "current_pitch_group_used_at_inference": bool(
            details["current_pitch_group_used_at_inference"]
        ),
        "validation_auxiliary_labels_used": bool(
            details["validation_auxiliary_labels_used"]
        ),
        "row_independent_inference": bool(details["row_independent_inference"]),
        "gamma_from_2020_lock": gamma,
    }
    semantic["pass"] = bool(
        semantic["architecture"]
        == "tabm_shared_representation_eight_bernoulli_heads"
        and semantic["target_heads"] == lock["locked_recipe"]["heads"]
        and np.allclose(
            semantic["head_weights"], lock["locked_recipe"]["head_weights"],
            atol=0.0, rtol=0.0,
        )
        and semantic["history_usable_coverage"] >= 0.995
        and not semantic["current_pitch_group_used_at_inference"]
        and not semantic["validation_auxiliary_labels_used"]
        and semantic["row_independent_inference"]
    )
    game_types = pd.read_csv(
        ROOT / "open/data/train.csv", usecols=["game_type"]
    )["game_type"].astype(str)
    game_type = game_types.iloc[
        candidate["row_index"].astype(np.int64)
    ].to_numpy(dtype=str)
    parent = parent_artifact["catboost_outcome"].astype(np.float64)
    raw = candidate[KEY].astype(np.float64)
    routed = np.where(game_type == "R", raw, parent)
    masks = {"full": np.ones(len(raw), dtype=bool), "R": game_type == "R"}
    result = evaluate(
        candidate,
        parent,
        routed,
        masks["full"],
        masks,
        gamma,
        2000,
        8132021,
    )
    gate = lock["next_gate"]
    checks = {
        "semantic": bool(semantic["pass"]),
        "gamma_unchanged": gamma == 0.25,
        "R_gain": bool(result["routes"]["R"]["gain"] >= float(gate["minimum_R_gain"])),
        "full_gain": bool(
            result["routes"]["full"]["gain"] >= float(gate["minimum_full_gain"])
        ),
        "R_ci": bool(result["routes"]["R"]["pitcher_cluster_95_ci"]["ci_low"] > 0.0),
        "full_ci": bool(
            result["routes"]["full"]["pitcher_cluster_95_ci"]["ci_low"] > 0.0
        ),
    }
    passed = bool(all(checks.values()))
    report = {
        "experiment_id": lock["experiment_id"],
        "status": "source_pass" if passed else "source_failed_2021",
        "source_lock_sha256": digest(LOCK),
        "script_sha256": digest(Path(__file__)),
        "years_read": [2020, 2021],
        "years_not_read": [2022, 2023, 2024],
        "semantic": semantic,
        "gamma": gamma,
        "result": result,
        "source_gate": {"requirements": gate, "checks": checks, "pass": passed},
        "next_action": (
            "lock_recipe_before_2022_2023_development"
            if passed else "close_without_2022_or_later_metrics"
        ),
        "goal_status": "active",
        "goal_completion_claimed": False,
    }
    REPORT.write_text(
        json.dumps(safe(report), ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(
        json.dumps(
            safe({"status": report["status"], "result": result, "checks": checks}),
            ensure_ascii=False,
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
