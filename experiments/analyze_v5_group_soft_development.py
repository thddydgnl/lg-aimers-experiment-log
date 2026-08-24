#!/usr/bin/env python3
"""Immutable 2022/2023 development gate for locked group-soft routing."""

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

from experiments.analyze_v5_dense_pitchtype_moe import digest, load, safe, score  # noqa: E402
from experiments.run_v5_h1_residual import cluster_bootstrap_score_gain  # noqa: E402


RESULTS = ROOT / "experiments/results"
PRED = RESULTS / "predictions"
TRAIN = ROOT / "open/data/train.csv"
LOCK = ROOT / "experiments/params/v5_group_soft_development_lock.json"
CONTRACT = ROOT / "experiments/params/v5_validation_contract_v2.json"
PARAMS = ROOT / "experiments/params/v5_group_soft_alpha0.json"
ENGINE = ROOT / "experiments/run_v2_rolling.py"
STAGE_REPORT = RESULTS / "v5_group_soft_locked_dev.json"
REPORT = RESULTS / "v5_group_soft_development_gate.json"
YEARS = (2022, 2023)
BOOTSTRAP_ITERATIONS = 2000
ANCHORS = {
    "exact_c": ("v3_sparse_c_backtest_{year}.npz", "catboost_outcome"),
    "honest_identity": ("v5_honest_m3_r_identity_{year}.npz", "final_prediction"),
    "honest_grid": ("v5_honest_m3_r_grid_{year}.npz", "final_prediction"),
}
EXPECTED_FEATURES = [
    "base", "e14", "platoon", "hand_matchup", "e14_hand_cells",
    "e14_count_cells", "e14_type_count_cells", "trackman_rich",
    "batter_e14", "batter_middle_e14",
]


def routed_prediction(
    parent: np.ndarray,
    model: np.ndarray,
    regular: np.ndarray,
    gamma: float,
) -> np.ndarray:
    result = parent.astype(np.float64, copy=True)
    result[regular] = np.clip(
        (1.0 - gamma) * parent[regular] + gamma * model[regular],
        1e-6,
        1.0 - 1e-6,
    )
    return result


def evaluate_pair(
    y: np.ndarray,
    anchor: np.ndarray,
    candidate: np.ndarray,
    cluster: np.ndarray,
    masks: dict[str, np.ndarray],
    seed: int,
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for route_index, (route, mask) in enumerate(masks.items()):
        anchor_metrics = score(y, anchor, mask)
        candidate_metrics = score(y, candidate, mask)
        interval = cluster_bootstrap_score_gain(
            y,
            anchor,
            candidate,
            cluster,
            mask,
            iterations=BOOTSTRAP_ITERATIONS,
            seed=seed + 1000 * route_index,
        )
        point = float(candidate_metrics["score"] - anchor_metrics["score"])
        if abs(point - float(interval["point"])) > 1e-8:
            raise AssertionError(f"score/CI point mismatch: {route}")
        result[route] = {
            "anchor": anchor_metrics,
            "candidate": candidate_metrics,
            "gain": point,
            "pitcher_cluster_95_ci": interval,
        }
    return result


def main() -> None:
    if REPORT.exists():
        raise FileExistsError(f"immutable report exists: {REPORT}")
    lock = json.loads(LOCK.read_text(encoding="utf-8"))
    contract = json.loads(CONTRACT.read_text(encoding="utf-8"))
    stage = json.loads(STAGE_REPORT.read_text(encoding="utf-8"))
    if lock["status"] != "locked_after_source_pass_before_2022_or_2023_candidate_metrics":
        raise ValueError("unexpected lock status")
    fixed = lock["fixed_recipe"]
    gamma = float(fixed["gamma"])
    if gamma != 0.5 or float(fixed["hard_label_fraction"]) != 0.0:
        raise ValueError("locked source selection changed")
    expected_hashes = lock["immutable_input_sha256"]
    immutable_checks = {
        relative: digest(ROOT / relative)
        == expected_hash
        for relative, expected_hash in expected_hashes.items()
    }
    if not all(immutable_checks.values()):
        raise ValueError(f"immutable source input changed: {immutable_checks}")

    metadata = stage["metadata"]
    stage_checks: dict[str, bool] = {
        "stage": metadata["stage"] == "v5_group_soft_locked_dev",
        "models": metadata["models"] == ["catboost_group_soft"],
        "features": metadata["features"] == EXPECTED_FEATURES,
        "validation_seasons": metadata["validation_seasons"] == list(YEARS),
        "fit_game_types_R": "--fit-game-types R" in metadata["command"],
        "inner_validation_none": metadata["inner_validation"] == "none",
        "gpu": metadata["booster_device"] == "gpu",
        "row_independent": bool(metadata["row_independent_inference"]),
        "params_hash": digest(PARAMS)
        == expected_hashes["experiments/params/v5_group_soft_alpha0.json"],
        "engine_hash": digest(ENGINE)
        == expected_hashes["experiments/run_v2_rolling.py"],
        "fold_years": [fold["validation_season"] for fold in stage["folds"]]
        == list(YEARS),
    }
    for fold in stage["folds"]:
        year = int(fold["validation_season"])
        details = fold["fit_details"]["catboost_group_soft"]
        stage_checks[f"{year}_fit_game_types_R"] = fold["fit_game_types"] == ["R"]
        stage_checks[f"{year}_pitcher_id_dropped"] = "pitcher_id" in fold["dropped_features"]
        stage_checks[f"{year}_hard_fraction"] = float(details["hard_label_fraction"]) == 0.0
        stage_checks[f"{year}_loo_target"] = (
            details["training_target"]
            == "hierarchical_leave_one_out_empirical_bayes"
        )
        stage_checks[f"{year}_no_validation_labels"] = not bool(
            details["validation_labels_used_for_target_or_fit"]
        )
        stage_checks[f"{year}_row_independent"] = bool(details["row_independent_inference"])
    if not all(stage_checks.values()):
        raise ValueError(f"stage contract failed: {stage_checks}")

    folds: dict[int, dict[str, Any]] = {}
    maximum_row = 0
    input_hashes: dict[str, Any] = {}
    for year in YEARS:
        model_path = PRED / f"v5_group_soft_locked_dev_{year}.npz"
        model_artifact = load(model_path)
        exact_path = PRED / ANCHORS["exact_c"][0].format(year=year)
        exact_artifact = load(exact_path)
        for field in ("y", "row_index", "cluster"):
            if not np.array_equal(model_artifact[field], exact_artifact[field]):
                raise ValueError(f"{year}: model/exact {field} alignment mismatch")
        row_index = exact_artifact["row_index"].astype(np.int64)
        maximum_row = max(maximum_row, int(row_index.max()))
        fold: dict[str, Any] = {
            "y": exact_artifact["y"].astype(np.int8),
            "row_index": row_index,
            "cluster": exact_artifact["cluster"],
            "exact": exact_artifact[ANCHORS["exact_c"][1]].astype(np.float64),
            "model": model_artifact["catboost_group_soft"].astype(np.float64),
            "anchors": {},
        }
        input_hashes[str(year)] = {
            "model": digest(model_path),
            "anchors": {"exact_c": digest(exact_path)},
        }
        for anchor_name, (template, key) in ANCHORS.items():
            if anchor_name == "exact_c":
                fold["anchors"][anchor_name] = fold["exact"]
                continue
            path = PRED / template.format(year=year)
            artifact = load(path)
            for field in ("y", "row_index", "cluster"):
                if not np.array_equal(exact_artifact[field], artifact[field]):
                    raise ValueError(f"{year}/{anchor_name}: {field} mismatch")
            fold["anchors"][anchor_name] = artifact[key].astype(np.float64)
            input_hashes[str(year)]["anchors"][anchor_name] = digest(path)
        folds[year] = fold

    frame = pd.read_csv(
        TRAIN,
        usecols=["season", "game_type", "control_success"],
        nrows=maximum_row + 1,
    )
    if int(frame["season"].max()) != max(YEARS):
        raise ValueError("development reader crossed the locked 2023 boundary")

    comparisons: dict[str, Any] = {}
    artifacts: dict[str, Any] = {}
    for year_index, year in enumerate(YEARS):
        fold = folds[year]
        rows = frame.loc[fold["row_index"]]
        if not rows["season"].eq(year).all():
            raise ValueError(f"{year}: season mismatch")
        if not np.array_equal(
            rows["control_success"].to_numpy(dtype=np.int8), fold["y"]
        ):
            raise ValueError(f"{year}: target mismatch")
        regular = rows["game_type"].astype(str).eq("R").to_numpy()
        candidate = routed_prediction(fold["exact"], fold["model"], regular, gamma)
        masks = {
            "full": np.ones(len(candidate), dtype=bool),
            "R": regular,
            "F": ~regular,
        }
        comparisons[str(year)] = {}
        for anchor_index, (anchor_name, anchor) in enumerate(fold["anchors"].items()):
            comparisons[str(year)][anchor_name] = evaluate_pair(
                fold["y"],
                anchor,
                candidate,
                fold["cluster"],
                masks,
                seed=8253000 + 100000 * year_index + 10000 * anchor_index,
            )
        output = PRED / f"v5_group_soft_selected_dev_{year}.npz"
        if output.exists():
            raise FileExistsError(f"immutable artifact exists: {output}")
        np.savez_compressed(
            output,
            y=fold["y"],
            row_index=fold["row_index"],
            cluster=fold["cluster"],
            parent_exact_c=fold["exact"].astype(np.float32),
            group_soft=fold["model"].astype(np.float32),
            final_prediction=candidate.astype(np.float32),
            gamma=np.asarray(gamma, dtype=np.float64),
        )
        artifacts[str(year)] = {
            "path": str(output.relative_to(ROOT)),
            "sha256": digest(output),
        }

    full_gains = [
        float(comparisons[str(year)][anchor]["full"]["gain"])
        for year in YEARS
        for anchor in ANCHORS
    ]
    g_dev = float(min(full_gains))
    threshold = float(
        contract["conservative_score"]["required_raw_gain_for_1190_at_haircut"]
    )
    same_parent_checks: dict[str, bool] = {}
    for year in YEARS:
        exact_r = comparisons[str(year)]["exact_c"]["R"]
        same_parent_checks[f"{year}_R_point_positive"] = float(exact_r["gain"]) > 0.0
        same_parent_checks[f"{year}_R_ci_low_positive"] = float(
            exact_r["pitcher_cluster_95_ci"]["ci_low"]
        ) > 0.0
    gate = {
        "G_dev": g_dev,
        "required_G_dev_strictly_above": threshold,
        "G_dev_pass": g_dev > threshold,
        "same_parent_checks": same_parent_checks,
    }
    gate["pass"] = bool(gate["G_dev_pass"] and all(same_parent_checks.values()))
    report = {
        "experiment_id": lock["experiment_id"],
        "status": "development_pass" if gate["pass"] else "development_failed",
        "lock_sha256": digest(LOCK),
        "contract_sha256": digest(CONTRACT),
        "stage_report_sha256": digest(STAGE_REPORT),
        "script_sha256": digest(Path(__file__)),
        "immutable_source_checks": immutable_checks,
        "stage_checks": stage_checks,
        "years_read": list(YEARS),
        "years_not_read": [2024],
        "fixed_recipe": fixed,
        "comparisons": comparisons,
        "gate": gate,
        "input_sha256": input_hashes,
        "artifacts": artifacts,
        "goal_status": "active",
        "goal_completion_claimed": False,
    }
    REPORT.write_text(
        json.dumps(safe(report), ensure_ascii=False, indent=2), encoding="utf-8"
    )
    summary = {
        "status": report["status"],
        "G_dev": g_dev,
        "threshold": threshold,
        "same_parent_checks": same_parent_checks,
        "full_gains": {
            str(year): {
                anchor: comparisons[str(year)][anchor]["full"]["gain"]
                for anchor in ANCHORS
            }
            for year in YEARS
        },
        "same_parent_R": {
            str(year): comparisons[str(year)]["exact_c"]["R"]
            for year in YEARS
        },
        "gate_pass": gate["pass"],
    }
    print(json.dumps(safe(summary), ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
