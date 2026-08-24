#!/usr/bin/env python3
"""Apply the source-locked previous-year loss gate on the 2021->2022 path."""

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

from experiments.analyze_v5_dense_pitchtype_moe import digest, safe  # noqa: E402
from experiments.analyze_v5_prevyear_loss_gate_source import (  # noqa: E402
    ROW_COLUMNS,
    apply_gate,
    feature_frame,
    fit_gate,
    route_metrics,
)
from experiments.run_v5_h1_residual import load_anchor  # noqa: E402


TRAIN = ROOT / "open/data/train.csv"
PRED = ROOT / "experiments/results/predictions"
PREREG = ROOT / "experiments/params/v5_prevyear_loss_gate_preregister.json"
SOURCE_LOCK = ROOT / "experiments/params/v5_prevyear_loss_gate_source_lock.json"
SOURCE_REPORT = ROOT / "experiments/results/v5_prevyear_loss_gate_source.json"
DIRECT_PATHS = {
    2021: PRED / "v5_direct_season_update_source_2021.npz",
    2022: PRED / "v5_direct_season_update_dev_2022.npz",
}
REPORT = ROOT / "experiments/results/v5_prevyear_loss_gate_dev2022.json"
ARTIFACT = PRED / "v5_prevyear_loss_gate_dev_2022.npz"


def load_frame_through_2022() -> pd.DataFrame:
    pieces: list[pd.DataFrame] = []
    offset = 0
    for chunk in pd.read_csv(TRAIN, usecols=ROW_COLUMNS, chunksize=250_000):
        chunk.index = np.arange(offset, offset + len(chunk), dtype=np.int64)
        offset += len(chunk)
        selected = chunk.loc[chunk["season"].le(2022)]
        if len(selected):
            pieces.append(selected)
        if int(chunk["season"].min()) > 2022:
            break
    frame = pd.concat(pieces, axis=0)
    if int(frame["season"].max()) != 2022:
        raise AssertionError("development loader crossed or missed 2022")
    return frame


def load_fold(frame: pd.DataFrame, year: int) -> dict[str, Any]:
    anchor = load_anchor(year)
    direct_path = DIRECT_PATHS[year]
    direct = dict(np.load(direct_path, allow_pickle=False))
    for key in ("y", "row_index", "cluster"):
        if not np.array_equal(anchor[key], direct[key]):
            raise AssertionError(f"anchor/direct alignment failure {year}: {key}")
    row_index = anchor["row_index"].astype(np.int64)
    rows = frame.loc[row_index]
    if not rows["season"].eq(year).all():
        raise AssertionError(f"row season mismatch: {year}")
    if not np.array_equal(
        rows["control_success"].to_numpy(dtype=np.int8),
        anchor["y"].astype(np.int8),
    ):
        raise AssertionError(f"target mismatch: {year}")
    return {
        "year": year,
        "y": anchor["y"].astype(np.int8),
        "row_index": row_index,
        "cluster": anchor["cluster"],
        "parent": anchor["final_prediction"].astype(np.float64),
        "alternate": direct["final_prediction"].astype(np.float64),
        "n_current": direct["n_current"].astype(np.float64),
        "s_current": direct["s_current"].astype(np.float64),
        "rows": rows,
        "regular": rows["game_type"].astype(str).eq("R").to_numpy(),
        "direct_path": direct_path,
    }


def tree_spec(tree: Any, columns: list[str]) -> dict[str, Any]:
    nodes: list[dict[str, Any]] = []
    for node in range(int(tree.tree_.node_count)):
        feature_index = int(tree.tree_.feature[node])
        nodes.append({
            "node": node,
            "left": int(tree.tree_.children_left[node]),
            "right": int(tree.tree_.children_right[node]),
            "feature": columns[feature_index] if feature_index >= 0 else None,
            "threshold": (
                float(tree.tree_.threshold[node]) if feature_index >= 0 else None
            ),
            "samples": int(tree.tree_.n_node_samples[node]),
            "value": float(tree.tree_.value[node].reshape(-1)[0]),
        })
    return {"nodes": nodes}


def main() -> None:
    if REPORT.exists() or ARTIFACT.exists():
        raise FileExistsError("immutable 2022 development output already exists")
    prereg = json.loads(PREREG.read_text(encoding="utf-8"))
    lock = json.loads(SOURCE_LOCK.read_text(encoding="utf-8"))
    if not lock.get("advance_to_2022"):
        raise RuntimeError("source gate did not authorize the 2022 read")
    if lock["preregister_sha256"] != digest(PREREG):
        raise AssertionError("preregister hash changed after source lock")
    frame = load_frame_through_2022()
    train_fold = load_fold(frame, 2021)
    valid_fold = load_fold(frame, 2022)
    train_features = feature_frame(train_fold)
    valid_features = feature_frame(valid_fold)
    selected = lock["selected_hyperparameters"]
    columns = list(lock["feature_columns"])
    expected_columns = prereg["feature_sets"][selected["feature_set"]]
    if columns != expected_columns:
        raise AssertionError("locked feature columns disagree with preregistration")
    tree, leaf_weights, tree_meta = fit_gate(
        train_fold,
        train_features,
        columns,
        int(selected["max_depth"]),
        int(selected["min_samples_leaf"]),
        int(prereg["search_grid"]["random_state"]),
        np.asarray(prereg["leaf_weight_fit"]["grid"], dtype=np.float64),
    )
    train_prediction, train_weights = apply_gate(
        train_fold, train_features, columns, tree, leaf_weights,
        float(selected["global_shrink"]),
    )
    valid_prediction, valid_weights = apply_gate(
        valid_fold, valid_features, columns, tree, leaf_weights,
        float(selected["global_shrink"]),
    )
    metrics = {
        "2021_refit": route_metrics(train_fold, train_prediction, 8312021),
        "2022_forward": route_metrics(valid_fold, valid_prediction, 8312022),
    }
    threshold = float(
        prereg["development_gate"][
            "required_2022_R_and_full_point_gain_strictly_greater_than"
        ]
    )
    forward = metrics["2022_forward"]
    checks = {
        "R_point_strict": forward["R"]["gain"] > threshold,
        "full_point_strict": forward["full"]["gain"] > threshold,
        "R_ci_lower_positive": (
            forward["R"]["pitcher_cluster_95_ci"]["ci_low"] > 0.0
        ),
        "full_ci_lower_positive": (
            forward["full"]["pitcher_cluster_95_ci"]["ci_low"] > 0.0
        ),
        "F_unchanged": abs(forward["F"]["gain"]) <= 1e-12,
    }
    passed = bool(all(checks.values()))
    np.savez_compressed(
        ARTIFACT,
        y=valid_fold["y"], row_index=valid_fold["row_index"],
        cluster=valid_fold["cluster"], parent_m3=valid_fold["parent"],
        direct_update=valid_fold["alternate"], gate_weight=valid_weights,
        final_prediction=valid_prediction,
    )
    regular_weights = valid_weights[valid_fold["regular"]]
    report = {
        "experiment_id": prereg["experiment_id"],
        "status": "development_2022_pass" if passed else "development_2022_failed_closed",
        "years_read": [2021, 2022],
        "years_not_read_by_this_script": [2023, 2024],
        "preregister_sha256": digest(PREREG),
        "source_lock_sha256": digest(SOURCE_LOCK),
        "source_report_sha256": digest(SOURCE_REPORT),
        "script_sha256": digest(Path(__file__)),
        "locked_hyperparameters": selected,
        "feature_columns": columns,
        "refitted_tree": {**tree_meta, **tree_spec(tree, columns)},
        "refitted_leaf_weights": {
            str(key): float(value) for key, value in leaf_weights.items()
        },
        "2022_gate_weight_summary_R": {
            "mean": float(regular_weights.mean()),
            "std": float(regular_weights.std()),
            "min": float(regular_weights.min()),
            "max": float(regular_weights.max()),
            "changed_fraction": float(np.mean(regular_weights > 0.0)),
        },
        "metrics": metrics,
        "development_gate": {
            "required_strict_gain": threshold,
            "checks": checks,
            "pass": passed,
        },
        "input_sha256": {
            str(year): {
                "direct": digest(DIRECT_PATHS[year]),
                "anchor": "deterministic load_anchor",
            }
            for year in (2021, 2022)
        },
        "artifact": {
            "path": str(ARTIFACT.relative_to(ROOT)),
            "sha256": digest(ARTIFACT),
        },
        "2023_composite_generated_or_read": False,
        "2024_composite_generated_or_read": False,
        "goal_status": "active",
        "goal_completion_claimed": False,
    }
    REPORT.write_text(
        json.dumps(safe(report), ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(safe({
        "status": report["status"], "tree": report["refitted_tree"],
        "weight_summary": report["2022_gate_weight_summary_R"],
        "metrics": metrics, "checks": checks,
    }), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
