#!/usr/bin/env python3
"""Single locked 2024 confirmation for the regime-aware recent routed recipe."""

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

from experiments.analyze_v5_dense_pitchtype_moe import digest, load, safe, score
from experiments.analyze_v5_recent_game_f_update import decode
from experiments.run_v5_h1_residual import cluster_bootstrap_score_gain

RESULTS = ROOT / "experiments/results"
PRED = RESULTS / "predictions"
TRAIN = ROOT / "open/data/train.csv"
LOCK = ROOT / "experiments/params/v5_recent_routed_confirmation_lock.json"
REPORT = RESULTS / "v5_recent_routed_confirmation.json"
YEAR = 2024
STAGES = {
    "fine": ("v5_recent_routed_fine_confirm2024", "catboost_fine_pitch_moe", "catboost_fine_pitch_moe"),
    "auto": ("v5_recent_routed_auto_confirm2024", "catboost_auto_pitch_joint", "catboost_auto_pitch_joint"),
}
FEATURES = ["base", "e14", "platoon", "hand_matchup", "e14_hand_cells", "e14_count_cells", "e14_type_count_cells", "trackman_rich", "batter_e14", "batter_middle_e14", "pitchmix_e14", "expanded_auto_pitch_latent"]
ANCHORS = {
    "exact_c": ("v3_outcome_trackmanrich_overall_e14k50_batter80_middle100_2024.npz", "catboost_outcome"),
    "honest_identity": ("v5_honest_m3_r_identity_2024.npz", "final_prediction"),
    "honest_grid": ("v5_honest_m3_r_grid_2024.npz", "final_prediction"),
}


def verify_lock(lock: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    loaded: dict[str, Any] = {}
    for name in ("contract", "regime_recipe", "development_gate"):
        spec = lock["immutable_inputs"][name]
        path = ROOT / spec["path"]
        if digest(path) != spec["sha256"]:
            raise ValueError(f"locked input changed: {name}")
        loaded[name] = json.loads(path.read_text(encoding="utf-8"))
    if loaded["development_gate"]["status"] != lock["immutable_inputs"]["development_gate"]["required_status"]:
        raise ValueError("development did not authorize confirmation")
    for year, expected in lock["immutable_inputs"]["development_artifacts"].items():
        path = PRED / f"v5_recent_routed_regime_dev_{year}.npz"
        if digest(path) != expected:
            raise ValueError(f"development artifact changed: {year}")
    return loaded["contract"], loaded["development_gate"]


def audit_stage(name: str, stage: str, model: str) -> dict[str, Any]:
    path = RESULTS / f"{stage}.json"
    report = json.loads(path.read_text(encoding="utf-8"))
    metadata = report["metadata"]
    fold = report["folds"][0]
    details = fold["fit_details"][model]
    checks = {
        "stage": metadata["stage"] == stage,
        "model": metadata["models"] == [model],
        "features": metadata["features"] == FEATURES,
        "year": metadata["validation_seasons"] == [YEAR],
        "inner_validation_none": metadata["inner_validation"] == "none",
        "row_independent": bool(metadata["row_independent_inference"]),
        "gpu": metadata["booster_device"] == "gpu",
        "pitcher_id_dropped": "pitcher_id" in fold["dropped_features"],
        "no_current_pitch": not bool(details.get("current_pitch_type_used_at_inference", False)),
        "no_current_trackman": not bool(details.get("current_pitch_trackman_used_at_inference", False)),
        "detail_row_independent": bool(details["row_independent_inference"]),
    }
    if name == "fine":
        checks["eight_experts"] = int(details["expert_count"]) == 8
    if not all(checks.values()):
        raise AssertionError(f"confirmation stage audit failed {name}: {[k for k,v in checks.items() if not v]}")
    return {"path": str(path.relative_to(ROOT)), "sha256": digest(path), "checks": checks, "command": metadata["command"]}


def evaluate(
    y: np.ndarray, anchor: np.ndarray, candidate: np.ndarray,
    cluster: np.ndarray, masks: dict[str, np.ndarray], seed: int,
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for route_index, (name, mask) in enumerate(masks.items()):
        base = score(y, anchor, mask)
        cand = score(y, candidate, mask)
        ci = cluster_bootstrap_score_gain(
            y, anchor, candidate, cluster, mask, iterations=2000,
            seed=seed + 1000 * route_index,
        )
        result[name] = {"anchor": base, "candidate": cand, "gain": float(cand["score"] - base["score"]), "pitcher_cluster_95_ci": ci}
    return result


def main() -> None:
    if REPORT.exists():
        raise FileExistsError(f"immutable report exists: {REPORT}")
    lock = json.loads(LOCK.read_text(encoding="utf-8"))
    contract, development = verify_lock(lock)
    stage_audit = {
        name: audit_stage(name, stage, model)
        for name, (stage, model, _) in STAGES.items()
    }
    components: dict[str, dict[str, np.ndarray]] = {}
    component_paths: dict[str, Path] = {}
    for name, (stage, _, _) in STAGES.items():
        path = PRED / f"{stage}_{YEAR}.npz"
        components[name] = load(path)
        component_paths[name] = path
    anchors: dict[str, dict[str, np.ndarray]] = {}
    anchor_paths: dict[str, Path] = {}
    for name, (filename, _) in ANCHORS.items():
        path = PRED / filename
        anchors[name] = load(path)
        anchor_paths[name] = path
    reference = anchors["exact_c"]
    for group, collection in (("component", components), ("anchor", anchors)):
        for name, artifact in collection.items():
            for key in ("y", "row_index", "cluster"):
                if not np.array_equal(reference[key], artifact[key]):
                    raise ValueError(f"alignment mismatch: {group}/{name}/{key}")
    usecols = ["season", "game_type", "control_success", "asof_pitcher_prev1_game_success_rate", "asof_pitcher_prev1_game_middle_rate"]
    frame = pd.read_csv(TRAIN, usecols=usecols)
    rows = frame.iloc[reference["row_index"].astype(np.int64)]
    if not rows["season"].eq(YEAR).all() or not np.array_equal(rows["control_success"].to_numpy(dtype=np.int8), reference["y"].astype(np.int8)):
        raise ValueError("2024 row/target mismatch")
    regular = rows["game_type"].astype(str).eq("R").to_numpy()
    masks = {"full": np.ones(len(rows), dtype=bool), "R": regular, "F": ~regular}
    parent = reference["catboost_outcome"].astype(np.float64)
    fine_raw = components["fine"][STAGES["fine"][2]].astype(np.float64)
    auto_raw = components["auto"][STAGES["auto"][2]].astype(np.float64)
    fine = parent + 0.5 * (fine_raw - parent)
    auto = parent + 0.25 * (auto_raw - parent)
    r_prediction = 0.6 * fine + 0.4 * auto
    decoded_n, decoded_s, decoded_valid = decode(rows, 1, 200, 5.1e-7)
    f_prediction = (decoded_s + 100.0 * parent) / (decoded_n + 100.0)
    f_prediction = np.where(decoded_valid, f_prediction, parent)
    candidate = np.clip(np.where(regular, r_prediction, f_prediction), 1e-6, 1.0 - 1e-6)
    comparisons: dict[str, Any] = {}
    for anchor_index, (name, (_, key)) in enumerate(ANCHORS.items()):
        comparisons[name] = evaluate(
            reference["y"], anchors[name][key].astype(np.float64), candidate,
            reference["cluster"], masks, 8830000 + 100 * anchor_index,
        )
    threshold = float(contract["non_relaxation_checks"]["required_raw_gain_unchanged"])
    exact = comparisons["exact_c"]
    segment_checks = {
        "R_point_positive": exact["R"]["gain"] > 0.0,
        "R_ci_lower_positive": exact["R"]["pitcher_cluster_95_ci"]["ci_low"] > 0.0,
        "F_point_positive": exact["F"]["gain"] > 0.0,
        "F_ci_lower_positive": exact["F"]["pitcher_cluster_95_ci"]["ci_low"] > 0.0,
    }
    full_points = [item["full"]["gain"] for item in comparisons.values()]
    full_ci_lows = [item["full"]["pitcher_cluster_95_ci"]["ci_low"] for item in comparisons.values()]
    anchor_checks = {
        "all_points_above_threshold": min(full_points) > threshold,
        "all_ci_lowers_above_threshold": min(full_ci_lows) > threshold,
    }
    g_dev = float(development["gates"]["postbreak_G_dev"]["G_dev"])
    g_confirm = float(min(full_points))
    g_ci = float(min(full_ci_lows))
    g_robust = float(min(g_dev, g_confirm, g_ci))
    actual_anchor = float(contract["non_relaxation_checks"]["v3_actual_anchor_unchanged"])
    haircut = float(contract["non_relaxation_checks"]["haircut_unchanged"])
    expected_lower = float(actual_anchor + haircut * max(0.0, g_robust))
    confirmation_pass = bool(all(segment_checks.values()) and all(anchor_checks.values()))
    output = PRED / "v5_recent_routed_regime_confirm_2024.npz"
    if output.exists():
        raise FileExistsError(f"immutable artifact exists: {output}")
    np.savez_compressed(
        output, y=reference["y"].astype(np.int8), row_index=reference["row_index"].astype(np.int64),
        cluster=reference["cluster"], parent_exact_c=parent, fine_raw=fine_raw, auto_raw=auto_raw,
        decoded_n=decoded_n, decoded_successes=decoded_s, decoded_valid=decoded_valid.astype(np.int8),
        final_prediction=candidate,
    )
    report = {
        "experiment_id": lock["experiment_id"], "status": "confirmation_pass" if confirmation_pass else "confirmation_failed",
        "lock_sha256": digest(LOCK), "contract_sha256": digest(ROOT / lock["immutable_inputs"]["contract"]["path"]),
        "year_read": YEAR, "test_rows_read": False, "stage_audit": stage_audit,
        "input_sha256": {
            **{name: digest(path) for name, path in component_paths.items()},
            **{name: digest(path) for name, path in anchor_paths.items()},
        },
        "rows": {"full": int(len(rows)), "R": int(regular.sum()), "F": int((~regular).sum()), "F_decoded_coverage": float(decoded_valid[~regular].mean())},
        "comparisons": comparisons,
        "gates": {
            "same_parent_segments": {"checks": segment_checks, "pass": all(segment_checks.values())},
            "all_anchor_full": {"threshold": threshold, "points": full_points, "ci_lowers": full_ci_lows, "checks": anchor_checks, "pass": all(anchor_checks.values())},
            "confirmation_pass": confirmation_pass,
        },
        "conservative_expected_score": {
            "actual_v3_anchor": actual_anchor, "G_dev": g_dev, "G_confirm": g_confirm,
            "G_ci": g_ci, "G_robust": g_robust, "haircut": haircut,
            "expected_lb_lower": expected_lower, "passes_1190": expected_lower > 1190.0,
        },
        "artifact": {"path": str(output.relative_to(ROOT)), "sha256": digest(output)},
        "goal_status": "active", "goal_completion_claimed": False,
    }
    REPORT.write_text(json.dumps(safe(report), ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(safe({"status": report["status"], "gates": report["gates"], "expected": report["conservative_expected_score"], "same_parent": exact}), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
