#!/usr/bin/env python3
"""Immutable 2022/2023 gate for the locked recent routed recipe."""

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
LOCK = ROOT / "experiments/params/v5_recent_routed_source_lock.json"
CONTRACT = ROOT / "experiments/params/v5_validation_contract_v2.json"
SOURCE = RESULTS / "v5_recent_routed_source.json"
REPORT = RESULTS / "v5_recent_routed_dev.json"
YEARS = (2022, 2023)
STAGES = {
    "expanded_fine_pitch_moe": {
        "stage": "v5_recent_routed_fine_dev2223",
        "model": "catboost_fine_pitch_moe",
        "key": "catboost_fine_pitch_moe",
        "features": ["base", "e14", "platoon", "hand_matchup", "e14_hand_cells", "e14_count_cells", "e14_type_count_cells", "trackman_rich", "batter_e14", "batter_middle_e14", "pitchmix_e14", "expanded_auto_pitch_latent"],
    },
    "expanded_auto_pitch_joint": {
        "stage": "v5_recent_routed_auto_dev2223",
        "model": "catboost_auto_pitch_joint",
        "key": "catboost_auto_pitch_joint",
        "features": ["base", "e14", "platoon", "hand_matchup", "e14_hand_cells", "e14_count_cells", "e14_type_count_cells", "trackman_rich", "batter_e14", "batter_middle_e14", "pitchmix_e14", "expanded_auto_pitch_latent"],
    },
}
ANCHORS = {
    "exact_c": ("v3_sparse_c_backtest_{year}.npz", "catboost_outcome"),
    "honest_identity": ("v5_honest_m3_r_identity_{year}.npz", "final_prediction"),
    "honest_grid": ("v5_honest_m3_r_grid_{year}.npz", "final_prediction"),
}


def load_rows() -> pd.DataFrame:
    columns = ["season", "game_type", "control_success", "asof_pitcher_prev1_game_success_rate", "asof_pitcher_prev1_game_middle_rate"]
    pieces: list[pd.DataFrame] = []
    offset = 0
    for chunk in pd.read_csv(TRAIN, usecols=columns, chunksize=250_000):
        chunk.index = np.arange(offset, offset + len(chunk), dtype=np.int64)
        offset += len(chunk)
        part = chunk.loc[chunk["season"].le(max(YEARS))]
        if len(part):
            pieces.append(part)
        if int(chunk["season"].min()) > max(YEARS):
            break
    frame = pd.concat(pieces)
    if int(frame["season"].max()) != max(YEARS):
        raise ValueError("development loader did not stop at 2023")
    return frame


def audit_stage(name: str, spec: dict[str, Any]) -> dict[str, Any]:
    path = RESULTS / f"{spec['stage']}.json"
    report = json.loads(path.read_text(encoding="utf-8"))
    metadata = report["metadata"]
    checks: dict[str, bool] = {
        "stage": metadata["stage"] == spec["stage"],
        "model": metadata["models"] == [spec["model"]],
        "features": metadata["features"] == spec["features"],
        "years": metadata["validation_seasons"] == list(YEARS),
        "inner_validation_none": metadata["inner_validation"] == "none",
        "row_independent": bool(metadata["row_independent_inference"]),
        "gpu": metadata["booster_device"] == "gpu",
        "pitcher_id_dropped": all("pitcher_id" in fold["dropped_features"] for fold in report["folds"]),
    }
    for fold in report["folds"]:
        year = int(fold["validation_season"])
        details = fold["fit_details"][spec["model"]]
        checks[f"{year}_no_current_pitch"] = not bool(details.get("current_pitch_type_used_at_inference", False))
        checks[f"{year}_no_current_trackman"] = not bool(details.get("current_pitch_trackman_used_at_inference", False))
        checks[f"{year}_row_independent"] = bool(details["row_independent_inference"])
        if name == "expanded_fine_pitch_moe":
            checks[f"{year}_experts"] = int(details["expert_count"]) == 8
    if not all(checks.values()):
        raise AssertionError(f"stage audit failed {name}: {[k for k,v in checks.items() if not v]}")
    return {"path": str(path.relative_to(ROOT)), "sha256": digest(path), "checks": checks, "command": metadata["command"]}


def evaluate(
    y: np.ndarray, anchor: np.ndarray, candidate: np.ndarray,
    cluster: np.ndarray, masks: dict[str, np.ndarray], seed: int,
) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for route_index, (name, mask) in enumerate(masks.items()):
        base = score(y, anchor, mask)
        cand = score(y, candidate, mask)
        ci = cluster_bootstrap_score_gain(
            y, anchor, candidate, cluster, mask, iterations=2000,
            seed=seed + 1000 * route_index,
        )
        out[name] = {"anchor": base, "candidate": cand, "gain": float(cand["score"] - base["score"]), "pitcher_cluster_95_ci": ci}
    return out


def main() -> None:
    if REPORT.exists():
        raise FileExistsError(f"immutable report exists: {REPORT}")
    lock = json.loads(LOCK.read_text(encoding="utf-8"))
    contract = json.loads(CONTRACT.read_text(encoding="utf-8"))
    source = json.loads(SOURCE.read_text(encoding="utf-8"))
    if source["status"] != "source_pass" or source["lock_sha256"] != digest(LOCK):
        raise ValueError("source lock/report mismatch")
    recipe = lock["locked_recipe"]
    if [item["name"] for item in recipe["R"]["components"]] != list(STAGES):
        raise ValueError("component order mismatch")
    if not np.allclose([item["weight"] for item in recipe["R"]["components"]], [0.6, 0.4], atol=1e-12, rtol=0.0):
        raise ValueError("component weights mismatch")
    stage_audit = {name: audit_stage(name, spec) for name, spec in STAGES.items()}
    frame = load_rows()
    years: dict[str, Any] = {}
    artifacts: dict[str, Any] = {}
    input_hashes: dict[str, Any] = {}
    full_gains: list[float] = []
    same_parent: dict[str, Any] = {}
    for year in YEARS:
        components: dict[str, dict[str, np.ndarray]] = {}
        component_paths: dict[str, Path] = {}
        for name, spec in STAGES.items():
            path = PRED / f"{spec['stage']}_{year}.npz"
            components[name] = load(path)
            component_paths[name] = path
        anchors: dict[str, dict[str, np.ndarray]] = {}
        anchor_paths: dict[str, Path] = {}
        for name, (template, _) in ANCHORS.items():
            path = PRED / template.format(year=year)
            anchors[name] = load(path)
            anchor_paths[name] = path
        reference = anchors["exact_c"]
        for group, collection in (("component", components), ("anchor", anchors)):
            for name, artifact in collection.items():
                for key in ("y", "row_index", "cluster"):
                    if not np.array_equal(reference[key], artifact[key]):
                        raise ValueError(f"alignment mismatch {year}/{group}/{name}/{key}")
        indices = reference["row_index"].astype(np.int64)
        rows = frame.loc[indices]
        if not rows["season"].eq(year).all() or not np.array_equal(rows["control_success"].to_numpy(dtype=np.int8), reference["y"].astype(np.int8)):
            raise ValueError(f"row/target mismatch: {year}")
        regular = rows["game_type"].astype(str).eq("R").to_numpy()
        masks = {"full": np.ones(len(rows), dtype=bool), "R": regular, "F": ~regular}
        parent = reference["catboost_outcome"].astype(np.float64)
        fine_raw = components["expanded_fine_pitch_moe"][STAGES["expanded_fine_pitch_moe"]["key"]].astype(np.float64)
        auto_raw = components["expanded_auto_pitch_joint"][STAGES["expanded_auto_pitch_joint"]["key"]].astype(np.float64)
        fine = parent + 0.5 * (fine_raw - parent)
        auto = parent + 0.25 * (auto_raw - parent)
        r_prediction = 0.6 * fine + 0.4 * auto
        decoded_n, decoded_s, decoded_valid = decode(rows, 1, 200, 5.1e-7)
        f_prediction = (decoded_s + 100.0 * parent) / (decoded_n + 100.0)
        f_prediction = np.where(decoded_valid, f_prediction, parent)
        candidate = np.clip(np.where(regular, r_prediction, f_prediction), 1e-6, 1.0 - 1e-6)
        comparisons: dict[str, Any] = {}
        for anchor_index, (name, (_, key)) in enumerate(ANCHORS.items()):
            anchor = anchors[name][key].astype(np.float64)
            comparisons[name] = evaluate(
                reference["y"], anchor, candidate, reference["cluster"], masks,
                8590000 + 10000 * year + 100 * anchor_index,
            )
            full_gains.append(float(comparisons[name]["full"]["gain"]))
        exact = comparisons["exact_c"]
        same_parent[str(year)] = {
            "R_point_positive": exact["R"]["gain"] > 0.0,
            "R_ci_lower_positive": exact["R"]["pitcher_cluster_95_ci"]["ci_low"] > 0.0,
            "F_point_positive": exact["F"]["gain"] > 0.0,
            "F_ci_lower_positive": exact["F"]["pitcher_cluster_95_ci"]["ci_low"] > 0.0,
        }
        years[str(year)] = {
            "rows": int(len(rows)), "R_rows": int(regular.sum()), "F_rows": int((~regular).sum()),
            "F_decoded_coverage": float(decoded_valid[~regular].mean()), "comparisons": comparisons,
        }
        output = PRED / f"v5_recent_routed_dev_{year}.npz"
        if output.exists():
            raise FileExistsError(f"immutable artifact exists: {output}")
        np.savez_compressed(
            output, y=reference["y"].astype(np.int8), row_index=indices, cluster=reference["cluster"],
            parent_exact_c=parent, fine_raw=fine_raw, auto_raw=auto_raw,
            decoded_n=decoded_n, decoded_successes=decoded_s, decoded_valid=decoded_valid.astype(np.int8),
            final_prediction=candidate,
        )
        artifacts[str(year)] = {"path": str(output.relative_to(ROOT)), "sha256": digest(output)}
        input_hashes[str(year)] = {
            **{name: digest(path) for name, path in component_paths.items()},
            **{name: digest(path) for name, path in anchor_paths.items()},
        }
    same_parent_pass = all(all(v.values()) for v in same_parent.values())
    g_dev = float(min(full_gains))
    threshold = float(lock["development_protocol"]["minimum_G_dev_strictly_greater_than"])
    passed = bool(same_parent_pass and g_dev > threshold)
    report = {
        "experiment_id": lock["experiment_id"], "status": "development_pass" if passed else "development_failed",
        "lock_sha256": digest(LOCK), "contract_sha256": digest(CONTRACT), "source_sha256": digest(SOURCE),
        "years_read": list(YEARS), "confirmation_2024_read": False, "stage_audit": stage_audit,
        "input_sha256": input_hashes, "years": years,
        "gates": {
            "same_parent": {"years": same_parent, "pass": same_parent_pass},
            "G_dev": {"minimum_full_gain": g_dev, "required_strictly_greater_than": threshold, "pass": g_dev > threshold},
            "development_pass": passed,
        },
        "artifacts": artifacts, "confirmation_2024_authorized": passed,
        "goal_status": "active", "goal_completion_claimed": False,
    }
    REPORT.write_text(json.dumps(safe(report), ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(safe({"status": report["status"], "gates": report["gates"], "same_parent_metrics": {
        str(y): {route: years[str(y)]["comparisons"]["exact_c"][route] for route in ("full", "R", "F")} for y in YEARS
    }}), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
