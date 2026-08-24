#!/usr/bin/env python3
"""Gate the disclosed adaptive one-season neural multitask variant."""

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
PREREG = ROOT / "experiments/params/v5_recent_neural_multitask_preregister.json"
REPORT = RESULTS / "v5_recent_neural_multitask_source_gate.json"
STAGE = "v5_recent_neural_multitask_source2021"
KEY = "tabm_dense_multitask"
YEAR = 2021


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def main() -> None:
    if REPORT.exists():
        raise FileExistsError(f"immutable report exists: {REPORT}")
    prereg = json.loads(PREREG.read_text(encoding="utf-8"))
    gamma = float(prereg["locked_recipe"]["gamma"])
    candidate_path = PRED / f"{STAGE}_{YEAR}.npz"
    parent_path = PRED / f"v4_m3_c_backtest_{YEAR}_{YEAR}.npz"
    metadata_path = RESULTS / f"{STAGE}.json"
    candidate = load(candidate_path)
    parent_artifact = load(parent_path)
    for key in ("y", "row_index", "cluster"):
        if not np.array_equal(candidate[key], parent_artifact[key]):
            raise ValueError(f"alignment mismatch: {key}")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    fold = metadata["folds"][0]
    details = fold["fit_details"][KEY]
    semantic = {
        "history_window": metadata["metadata"]["history_window"],
        "fit_history_seasons": fold["fit_history_seasons"],
        "architecture": details["architecture"],
        "target_heads": details["target_heads"],
        "head_weights": details["head_weights"],
        "history_usable_coverage": float(details["history_usable_coverage"]),
        "validation_auxiliary_labels_used": bool(details["validation_auxiliary_labels_used"]),
        "current_pitch_group_used_at_inference": bool(details["current_pitch_group_used_at_inference"]),
        "row_independent_inference": bool(details["row_independent_inference"]),
    }
    semantic["pass"] = bool(
        semantic["history_window"] == 1
        and semantic["fit_history_seasons"] == [2020]
        and semantic["architecture"]
        == "tabm_shared_representation_eight_bernoulli_heads"
        and semantic["history_usable_coverage"] >= 0.995
        and not semantic["validation_auxiliary_labels_used"]
        and not semantic["current_pitch_group_used_at_inference"]
        and semantic["row_independent_inference"]
    )
    game_types = pd.read_csv(
        ROOT / "open/data/train.csv", usecols=["game_type"]
    )["game_type"].astype(str)
    game_type = game_types.iloc[candidate["row_index"].astype(np.int64)].to_numpy(dtype=str)
    parent = parent_artifact["catboost_outcome"].astype(np.float64)
    raw = candidate[KEY].astype(np.float64)
    routed = np.where(game_type == "R", raw, parent)
    masks = {"full": np.ones(len(raw), dtype=bool), "R": game_type == "R"}
    result = evaluate(
        candidate, parent, routed, masks["full"], masks, gamma, 2000, 8242021
    )
    gate = prereg["protocol"]["source_gate_2021"]
    checks = {
        "semantic": bool(semantic["pass"]),
        "gamma_unchanged": gamma == 0.25,
        "R_gain": bool(result["routes"]["R"]["gain"] >= float(gate["minimum_R_gain"])),
        "full_gain": bool(result["routes"]["full"]["gain"] >= float(gate["minimum_full_gain"])),
        "R_ci": bool(result["routes"]["R"]["pitcher_cluster_95_ci"]["ci_low"] > 0.0),
        "full_ci": bool(result["routes"]["full"]["pitcher_cluster_95_ci"]["ci_low"] > 0.0),
    }
    passed = bool(all(checks.values()))
    report = {
        "experiment_id": prereg["experiment_id"],
        "status": "adaptive_source_pass" if passed else "adaptive_source_failed",
        "preregister_sha256": digest(PREREG),
        "script_sha256": digest(Path(__file__)),
        "years_read": [2020, 2021],
        "years_not_read": [2022, 2023, 2024],
        "adaptive_origin_disclosed": True,
        "semantic": semantic,
        "gamma": gamma,
        "result": result,
        "source_gate": {"requirements": gate, "checks": checks, "pass": passed},
        "next_action": "lock_before_fresh_development" if passed else "close",
        "goal_status": "active",
        "goal_completion_claimed": False,
    }
    REPORT.write_text(json.dumps(safe(report), ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(safe({"status": report["status"], "result": result, "checks": checks}), ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
