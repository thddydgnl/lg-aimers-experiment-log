#!/usr/bin/env python3
"""Fit an immutable 2020 meta-head and evaluate it one year ahead."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.analyze_v5_dense_pitchtype_moe import evaluate, load, safe  # noqa: E402

RESULTS = ROOT / "experiments/results"
PRED = RESULTS / "predictions"
PREREG = ROOT / "experiments/params/v5_temporal_multitask_head_preregister.json"
REPORT = RESULTS / "v5_temporal_multitask_head_2021_gate.json"
HEADS = ("success", "reverse", "middle", "ball", "strike", "fastball", "breaking", "offspeed")
STAGES = {
    2020: "v5_neural_dense_multitask_source2020",
    2021: "v5_neural_dense_multitask_source2021",
}


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def head_matrix(archive: dict[str, np.ndarray]) -> np.ndarray:
    probability = np.column_stack(
        [archive[f"tabm_dense_multitask__head_{head}"] for head in HEADS]
    ).astype(np.float64)
    probability = np.clip(probability, 1e-5, 1.0 - 1e-5)
    return np.log(probability / (1.0 - probability))


def main() -> None:
    if REPORT.exists():
        raise FileExistsError(f"immutable report exists: {REPORT}")
    prereg = json.loads(PREREG.read_text(encoding="utf-8"))
    archives = {
        year: load(PRED / f"{stage}_{year}.npz")
        for year, stage in STAGES.items()
    }
    parents = {
        year: load(PRED / f"v4_m3_c_backtest_{year}_{year}.npz")
        for year in STAGES
    }
    for year in STAGES:
        for key in ("y", "row_index", "cluster"):
            if not np.array_equal(archives[year][key], parents[year][key]):
                raise ValueError(f"alignment mismatch: {year}/{key}")
    game_types = pd.read_csv(
        ROOT / "open/data/train.csv", usecols=["game_type"]
    )["game_type"].astype(str)
    fold_types = {
        year: game_types.iloc[
            archives[year]["row_index"].astype(np.int64)
        ].to_numpy(dtype=str)
        for year in STAGES
    }
    fit_mask = fold_types[2020] == "R"
    scaler = StandardScaler()
    fit_x = scaler.fit_transform(head_matrix(archives[2020])[fit_mask])
    settings = prereg["recipe"]["logistic_regression"]
    model = LogisticRegression(
        C=float(settings["C"]),
        penalty=str(settings["penalty"]),
        solver=str(settings["solver"]),
        fit_intercept=bool(settings["fit_intercept"]),
        max_iter=int(settings["max_iter"]),
        random_state=int(settings["random_state"]),
    )
    model.fit(fit_x, archives[2020]["y"][fit_mask].astype(np.int8))
    apply_x = scaler.transform(head_matrix(archives[2021]))
    temporal = model.predict_proba(apply_x)[:, 1].astype(np.float64)
    parent = parents[2021]["catboost_outcome"].astype(np.float64)
    routed = np.where(fold_types[2021] == "R", temporal, parent)
    masks = {
        "full": np.ones(len(parent), dtype=bool),
        "R": fold_types[2021] == "R",
    }
    gamma = float(prereg["recipe"]["component_weights"][1])
    result = evaluate(
        archives[2021], parent, routed, masks["full"], masks,
        gamma, 2000, 8352021,
    )
    gate = prereg["gate_2021"]
    checks = {
        "head_order": list(HEADS) == prereg["recipe"]["base_heads"],
        "fit_year_only_2020": True,
        "fit_scope_R": bool(fit_mask.sum() == np.sum(fold_types[2020] == "R")),
        "component_count": int(prereg["recipe"]["component_count"]) == 2,
        "nonnegative_sum_one": bool(
            min(preregistered := prereg["recipe"]["component_weights"]) >= 0.0
            and abs(sum(preregistered) - 1.0) <= 1e-12
        ),
        "R_gain": bool(result["routes"]["R"]["gain"] >= float(gate["minimum_R_gain"])),
        "full_gain": bool(result["routes"]["full"]["gain"] >= float(gate["minimum_full_gain"])),
        "R_ci": bool(result["routes"]["R"]["pitcher_cluster_95_ci"]["ci_low"] > 0.0),
        "full_ci": bool(result["routes"]["full"]["pitcher_cluster_95_ci"]["ci_low"] > 0.0),
    }
    passed = bool(all(checks.values()))
    report = {
        "experiment_id": prereg["experiment_id"],
        "status": "temporal_source_pass" if passed else "temporal_source_failed",
        "preregister_sha256": digest(PREREG),
        "script_sha256": digest(Path(__file__)),
        "fit_year": 2020,
        "evaluation_year": 2021,
        "years_not_read_for_temporal_head": [2022, 2023, 2024],
        "meta_model": {
            "heads": list(HEADS),
            "fit_rows": int(fit_mask.sum()),
            "scaler_mean": scaler.mean_.tolist(),
            "scaler_scale": scaler.scale_.tolist(),
            "intercept": model.intercept_.tolist(),
            "coefficients": model.coef_.tolist(),
            "iterations": model.n_iter_.tolist(),
            "2021_target_used_for_fit": False,
        },
        "gamma": gamma,
        "result": result,
        "gate": {"requirements": gate, "checks": checks, "pass": passed},
        "next_action": "lock_rolling_recipe" if passed else "close",
        "goal_status": "active",
        "goal_completion_claimed": False,
    }
    REPORT.write_text(json.dumps(safe(report), ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(safe({"status": report["status"], "meta_model": report["meta_model"], "result": result, "checks": checks}), ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
