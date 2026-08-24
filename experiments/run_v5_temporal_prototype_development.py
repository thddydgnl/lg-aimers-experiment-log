#!/usr/bin/env python3
"""Locked 2022/2023 development gate for temporal prototype retrieval."""

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
from experiments.run_v5_temporal_prototype_source import (  # noqa: E402
    READ_COLUMNS,
    SEASON,
    TARGET,
    build_fold_retrieval,
)


RESULTS = ROOT / "experiments/results"
PRED = RESULTS / "predictions"
TRAIN = ROOT / "open/data/train.csv"
LOCK = ROOT / "experiments/params/v5_temporal_prototype_development_lock.json"
CONTRACT = ROOT / "experiments/params/v5_validation_contract_v2.json"
REPORT = RESULTS / "v5_temporal_prototype_development_gate.json"
YEARS = (2022, 2023)
BOOTSTRAP_ITERATIONS = 2000
ANCHORS = {
    "exact_c": ("v3_sparse_c_backtest_{year}.npz", "catboost_outcome"),
    "honest_identity": ("v5_honest_m3_r_identity_{year}.npz", "final_prediction"),
    "honest_grid": ("v5_honest_m3_r_grid_{year}.npz", "final_prediction"),
}


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
    if lock["status"] != "locked_after_source_pass_before_development_metrics":
        raise ValueError("unexpected development lock status")
    fixed = lock["fixed_recipe"]
    if fixed != {
        "neighbors": 16,
        "mode": "rate",
        "retrieval_amplitude": 0.5,
        "parent_gamma": 0.25,
        "F_routing": "unchanged exact-C parent",
    }:
        raise ValueError("locked source recipe changed")
    immutable_checks = {
        relative: digest(ROOT / relative) == expected
        for relative, expected in lock["immutable_input_sha256"].items()
    }
    if not all(immutable_checks.values()):
        raise ValueError(f"immutable input changed: {immutable_checks}")

    folds: dict[int, dict[str, Any]] = {}
    maximum_row = 0
    input_hashes: dict[str, Any] = {}
    for year in YEARS:
        exact_path = PRED / ANCHORS["exact_c"][0].format(year=year)
        exact = load(exact_path)
        row_index = exact["row_index"].astype(np.int64)
        maximum_row = max(maximum_row, int(row_index.max()))
        fold: dict[str, Any] = {
            "y": exact["y"].astype(np.int8),
            "row_index": row_index,
            "cluster": exact["cluster"],
            "exact": exact[ANCHORS["exact_c"][1]].astype(np.float64),
            "anchors": {},
        }
        input_hashes[str(year)] = {"anchors": {"exact_c": digest(exact_path)}}
        for anchor_name, (template, key) in ANCHORS.items():
            if anchor_name == "exact_c":
                fold["anchors"][anchor_name] = fold["exact"]
                continue
            path = PRED / template.format(year=year)
            artifact = load(path)
            for field in ("y", "row_index", "cluster"):
                if not np.array_equal(exact[field], artifact[field]):
                    raise ValueError(f"{year}/{anchor_name}: {field} mismatch")
            fold["anchors"][anchor_name] = artifact[key].astype(np.float64)
            input_hashes[str(year)]["anchors"][anchor_name] = digest(path)
        folds[year] = fold

    frame = pd.read_csv(TRAIN, usecols=READ_COLUMNS, nrows=maximum_row + 1)
    if int(frame[SEASON].max()) != max(YEARS):
        raise ValueError("development reader crossed the locked 2023 boundary")

    comparisons: dict[str, Any] = {}
    fold_details: dict[str, Any] = {}
    artifacts: dict[str, Any] = {}
    for year_index, year in enumerate(YEARS):
        fold = folds[year]
        rows = frame.loc[fold["row_index"]]
        if not rows[SEASON].eq(year).all():
            raise ValueError(f"{year}: season mismatch")
        if not np.array_equal(rows[TARGET].to_numpy(dtype=np.int8), fold["y"]):
            raise ValueError(f"{year}: target mismatch")
        retrieval_library, details = build_fold_retrieval(
            frame, fold["row_index"], year, [int(fixed["neighbors"])]
        )
        retrieval = retrieval_library[
            (
                int(fixed["neighbors"]),
                f"{fixed['mode']}:{float(fixed['retrieval_amplitude'])}",
            )
        ]
        regular = rows["game_type"].astype(str).eq("R").to_numpy()
        candidate = fold["exact"].copy()
        gamma = float(fixed["parent_gamma"])
        candidate[regular] = np.clip(
            (1.0 - gamma) * fold["exact"][regular]
            + gamma * retrieval[regular],
            1e-6,
            1.0 - 1e-6,
        )
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
                seed=8275000 + 100000 * year_index + 10000 * anchor_index,
            )
        details["selected_recipe"] = fixed
        fold_details[str(year)] = details
        output = PRED / f"v5_temporal_prototype_selected_dev_{year}.npz"
        if output.exists():
            raise FileExistsError(f"immutable artifact exists: {output}")
        np.savez_compressed(
            output,
            y=fold["y"],
            row_index=fold["row_index"],
            cluster=fold["cluster"],
            parent_exact_c=fold["exact"].astype(np.float32),
            retrieval=retrieval.astype(np.float32),
            final_prediction=candidate.astype(np.float32),
            neighbors=np.asarray(fixed["neighbors"], dtype=np.int64),
            retrieval_amplitude=np.asarray(
                fixed["retrieval_amplitude"], dtype=np.float64
            ),
            parent_gamma=np.asarray(gamma, dtype=np.float64),
        )
        artifacts[str(year)] = {
            "path": str(output.relative_to(ROOT)),
            "sha256": digest(output),
        }
        print(
            f"[{year}] prototypes={details['prototype_count']:,} "
            f"exact_full_gain={comparisons[str(year)]['exact_c']['full']['gain']:+.3f}",
            flush=True,
        )

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
        "script_sha256": digest(Path(__file__)),
        "immutable_checks": immutable_checks,
        "years_read": list(YEARS),
        "years_not_read": [2024],
        "fixed_recipe": fixed,
        "comparisons": comparisons,
        "fold_details": fold_details,
        "gate": gate,
        "input_sha256": input_hashes,
        "artifacts": artifacts,
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
                    "G_dev": g_dev,
                    "threshold": threshold,
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
                    "gate": gate,
                }
            ),
            ensure_ascii=False,
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
