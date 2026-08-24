#!/usr/bin/env python3
"""Apply the immutable 2020 first gate to neural dense multitask TabM."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.analyze_v5_dense_pitchtype_moe import evaluate, load, safe  # noqa: E402

RESULTS = ROOT / "experiments/results"
PRED = RESULTS / "predictions"
PREREG = ROOT / "experiments/params/v5_neural_dense_multitask_preregister.json"
PARAMS = ROOT / "experiments/params/v5_neural_dense_multitask_tabm.json"
STAGE = "v5_neural_dense_multitask_source2020"
REPORT = RESULTS / "v5_neural_dense_multitask_source_gate.json"
YEAR = 2020
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
    expected_heads = prereg["candidate"]["heads"]
    expected_weights = [float(value) for value in prereg["candidate"]["head_weights"]]
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
        "epochs": int(details["n_iter"]),
    }
    semantic["pass"] = bool(
        semantic["architecture"]
        == "tabm_shared_representation_eight_bernoulli_heads"
        and semantic["target_heads"] == expected_heads
        and np.allclose(semantic["head_weights"], expected_weights, atol=0.0, rtol=0.0)
        and semantic["history_usable_coverage"] >= 0.995
        and not semantic["current_pitch_group_used_at_inference"]
        and not semantic["validation_auxiliary_labels_used"]
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
    trials: list[dict[str, Any]] = []
    if semantic["pass"]:
        for gamma in prereg["source_protocol"]["blend_grid"]:
            result = evaluate(
                candidate,
                parent,
                routed_raw,
                masks["full"],
                masks,
                float(gamma),
                int(prereg["source_protocol"]["bootstrap_iterations"]),
                8120000 + int(float(gamma) * 100),
            )
            trials.append({"gamma": float(gamma), **result})
    selected = (
        max(
            trials,
            key=lambda item: (
                item["routes"]["R"]["gain"],
                item["routes"]["full"]["gain"],
                -item["gamma"],
            ),
        )
        if trials
        else None
    )
    gate = prereg["source_protocol"]["gate_each_year"]
    checks = {"semantic": bool(semantic["pass"]), "selected": selected is not None}
    if selected is not None:
        checks.update(
            {
                "R_gain": bool(
                    selected["routes"]["R"]["gain"] >= float(gate["minimum_R_gain"])
                ),
                "full_gain": bool(
                    selected["routes"]["full"]["gain"]
                    >= float(gate["minimum_full_gain"])
                ),
                "R_ci": bool(
                    selected["routes"]["R"]["pitcher_cluster_95_ci"]["ci_low"] > 0.0
                ),
                "full_ci": bool(
                    selected["routes"]["full"]["pitcher_cluster_95_ci"]["ci_low"] > 0.0
                ),
            }
        )
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
        "input_sha256": {
            "candidate": digest(candidate_path),
            "parent": digest(parent_path),
            "metadata": digest(metadata_path),
        },
        "trials": trials,
        "selected": selected,
        "source_gate": {"requirements": gate, "checks": checks, "pass": passed},
        "next_action": (
            "train_locked_2021_source_fold"
            if passed
            else "close_without_2021_or_later_metrics"
        ),
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
                    "semantic": semantic,
                    "selected": selected,
                    "checks": checks,
                }
            ),
            ensure_ascii=False,
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
