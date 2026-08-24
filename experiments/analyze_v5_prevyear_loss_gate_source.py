#!/usr/bin/env python3
"""Select a shallow previous-season loss gate on the sealed 2020->2021 source path."""

from __future__ import annotations

import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeRegressor


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.analyze_v5_dense_pitchtype_moe import digest, safe, score  # noqa: E402
from experiments.run_v5_h1_residual import (  # noqa: E402
    cluster_bootstrap_score_gain,
    load_anchor,
)


TRAIN = ROOT / "open/data/train.csv"
PRED = ROOT / "experiments/results/predictions"
PREREG = ROOT / "experiments/params/v5_prevyear_loss_gate_preregister.json"
REPORT = ROOT / "experiments/results/v5_prevyear_loss_gate_source.json"
LOCK = ROOT / "experiments/params/v5_prevyear_loss_gate_source_lock.json"
YEARS = (2020, 2021)
DIRECT_TEMPLATE = "v5_direct_season_update_source_{year}.npz"
BOOTSTRAP_ITERATIONS = 2000


ROW_COLUMNS = [
    "season", "game_type", "game_month", "inning", "balls_before",
    "strikes_before", "outs_before", "li", "pitcher_hand", "batter_hand",
    "asof_pitcher_n", "asof_pitcher_success_rate",
    "asof_pitcher_reverse_rate", "asof_pitcher_middle_rate",
    "asof_pitcher_ball_rate", "asof_pitcher_strike_rate",
    "asof_pitcher_prev1_game_success_rate",
    "asof_pitcher_prev3_game_success_rate",
    "asof_pitcher_prev5_game_success_rate",
    "asof_pitcher_prev1_game_middle_rate",
    "asof_pitcher_prev3_game_middle_rate",
    "asof_pitcher_prev5_game_middle_rate", "control_success",
]


def load_source_frame() -> pd.DataFrame:
    pieces: list[pd.DataFrame] = []
    offset = 0
    for chunk in pd.read_csv(TRAIN, usecols=ROW_COLUMNS, chunksize=250_000):
        chunk.index = np.arange(offset, offset + len(chunk), dtype=np.int64)
        offset += len(chunk)
        selected = chunk.loc[chunk["season"].le(2021)]
        if len(selected):
            pieces.append(selected)
        if int(chunk["season"].min()) > 2021:
            break
    frame = pd.concat(pieces, axis=0)
    if int(frame["season"].max()) != 2021:
        raise AssertionError("source loader crossed or missed the 2021 boundary")
    return frame


def load_fold(frame: pd.DataFrame, year: int) -> dict[str, Any]:
    anchor = load_anchor(year)
    direct_path = PRED / DIRECT_TEMPLATE.format(year=year)
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
    parent = anchor["final_prediction"].astype(np.float64)
    alternate = direct["final_prediction"].astype(np.float64)
    n_current = direct["n_current"].astype(np.float64)
    s_current = direct["s_current"].astype(np.float64)
    return {
        "year": year,
        "y": anchor["y"].astype(np.int8),
        "row_index": row_index,
        "cluster": anchor["cluster"],
        "parent": parent,
        "alternate": alternate,
        "n_current": n_current,
        "s_current": s_current,
        "rows": rows,
        "regular": rows["game_type"].astype(str).eq("R").to_numpy(),
        "direct_path": direct_path,
    }


def feature_frame(fold: dict[str, Any]) -> pd.DataFrame:
    rows = fold["rows"]
    parent = fold["parent"]
    alternate = fold["alternate"]
    n_current = fold["n_current"]
    s_current = fold["s_current"]
    career = pd.to_numeric(
        rows["asof_pitcher_success_rate"], errors="coerce"
    ).fillna(0.5).to_numpy(dtype=np.float64)
    current = np.divide(
        s_current,
        n_current,
        out=career.copy(),
        where=n_current > 0,
    )
    values: dict[str, np.ndarray] = {
        "parent_prediction": parent,
        "alternate_prediction": alternate,
        "prediction_delta": alternate - parent,
        "abs_prediction_delta": np.abs(alternate - parent),
        "log1p_n_current": np.log1p(n_current),
        "direct_reliability_k500": n_current / (n_current + 500.0),
        "current_success_rate": current,
        "current_minus_career_success": current - career,
        "log1p_asof_pitcher_n": np.log1p(
            pd.to_numeric(rows["asof_pitcher_n"], errors="coerce")
            .fillna(0.0).to_numpy(dtype=np.float64)
        ),
    }
    passthrough = [
        "asof_pitcher_success_rate", "asof_pitcher_reverse_rate",
        "asof_pitcher_middle_rate", "asof_pitcher_ball_rate",
        "asof_pitcher_strike_rate",
        "asof_pitcher_prev1_game_success_rate",
        "asof_pitcher_prev3_game_success_rate",
        "asof_pitcher_prev5_game_success_rate",
        "asof_pitcher_prev1_game_middle_rate",
        "asof_pitcher_prev3_game_middle_rate",
        "asof_pitcher_prev5_game_middle_rate", "game_month", "inning",
        "balls_before", "strikes_before", "outs_before", "li",
        "pitcher_hand", "batter_hand",
    ]
    for column in passthrough:
        values[column] = pd.to_numeric(
            rows[column], errors="coerce"
        ).fillna(-1.0).to_numpy(dtype=np.float64)
    result = pd.DataFrame(values, index=rows.index)
    if not np.isfinite(result.to_numpy(dtype=np.float64)).all():
        raise AssertionError("non-finite gate feature")
    return result


def fit_gate(
    fold: dict[str, Any], features: pd.DataFrame, columns: list[str],
    max_depth: int, min_samples_leaf: int, random_state: int,
    weight_grid: np.ndarray,
) -> tuple[DecisionTreeRegressor, dict[int, float], dict[str, Any]]:
    mask = fold["regular"]
    y = fold["y"][mask].astype(np.float64)
    parent = fold["parent"][mask]
    alternate = fold["alternate"][mask]
    target = np.square(parent - y) - np.square(alternate - y)
    tree = DecisionTreeRegressor(
        criterion="squared_error",
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        random_state=random_state,
    )
    tree.fit(features.loc[mask, columns], target)
    leaves = tree.apply(features.loc[mask, columns])
    weights: dict[int, float] = {}
    leaf_rows: list[dict[str, Any]] = []
    for leaf in sorted(int(value) for value in np.unique(leaves)):
        local = leaves == leaf
        losses = []
        for weight in weight_grid:
            prediction = parent[local] + weight * (alternate[local] - parent[local])
            losses.append(float(np.mean(np.square(prediction - y[local]))))
        best_index = min(range(len(losses)), key=lambda i: (losses[i], weight_grid[i]))
        best_weight = float(weight_grid[best_index])
        weights[leaf] = best_weight
        leaf_rows.append({
            "leaf": leaf,
            "rows": int(local.sum()),
            "weight": best_weight,
            "direct_advantage_mean": float(target[local].mean()),
        })
    return tree, weights, {
        "leaf_count": len(weights),
        "leaves": leaf_rows,
        "node_count": int(tree.tree_.node_count),
        "actual_depth": int(tree.tree_.max_depth),
    }


def apply_gate(
    fold: dict[str, Any], features: pd.DataFrame, columns: list[str],
    tree: DecisionTreeRegressor, leaf_weights: dict[int, float], shrink: float,
) -> tuple[np.ndarray, np.ndarray]:
    prediction = fold["parent"].copy()
    weights = np.zeros(len(prediction), dtype=np.float64)
    mask = fold["regular"]
    leaves = tree.apply(features.loc[mask, columns])
    local_weight = np.asarray(
        [shrink * leaf_weights[int(leaf)] for leaf in leaves], dtype=np.float64
    )
    local_weight = np.clip(local_weight, 0.0, 1.0)
    weights[mask] = local_weight
    prediction[mask] = (
        fold["parent"][mask]
        + local_weight * (fold["alternate"][mask] - fold["parent"][mask])
    )
    return np.clip(prediction, 1e-6, 1.0 - 1e-6), weights


def route_metrics(
    fold: dict[str, Any], candidate: np.ndarray, seed: int,
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    masks = {
        "full": np.ones(len(candidate), dtype=bool),
        "R": fold["regular"],
        "F": ~fold["regular"],
    }
    for route_index, (route, mask) in enumerate(masks.items()):
        parent_score = score(fold["y"], fold["parent"], mask)
        candidate_score = score(fold["y"], candidate, mask)
        interval = cluster_bootstrap_score_gain(
            fold["y"], fold["parent"], candidate, fold["cluster"], mask,
            iterations=BOOTSTRAP_ITERATIONS,
            seed=seed + 1000 * route_index,
        )
        result[route] = {
            "parent": parent_score,
            "candidate": candidate_score,
            "gain": float(candidate_score["score"] - parent_score["score"]),
            "pitcher_cluster_95_ci": interval,
        }
    return result


def main() -> None:
    if REPORT.exists() or LOCK.exists():
        raise FileExistsError("immutable source report/lock already exists")
    prereg = json.loads(PREREG.read_text(encoding="utf-8"))
    frame = load_source_frame()
    folds = {year: load_fold(frame, year) for year in YEARS}
    features = {year: feature_frame(folds[year]) for year in YEARS}
    weight_grid = np.asarray(prereg["leaf_weight_fit"]["grid"], dtype=np.float64)
    random_state = int(prereg["search_grid"]["random_state"])
    trials: list[dict[str, Any]] = []
    fitted: dict[tuple[str, int, int], tuple[Any, ...]] = {}
    for feature_name in prereg["search_grid"]["feature_set"]:
        columns = list(prereg["feature_sets"][feature_name])
        for max_depth in prereg["search_grid"]["max_depth"]:
            for min_leaf in prereg["search_grid"]["min_samples_leaf"]:
                key = (feature_name, int(max_depth), int(min_leaf))
                tree, leaf_weights, tree_meta = fit_gate(
                    folds[2020], features[2020], columns, int(max_depth),
                    int(min_leaf), random_state, weight_grid,
                )
                fitted[key] = (tree, leaf_weights, tree_meta, columns)
                for shrink in prereg["search_grid"]["global_shrink"]:
                    source_pred, source_weights = apply_gate(
                        folds[2020], features[2020], columns, tree,
                        leaf_weights, float(shrink),
                    )
                    valid_pred, valid_weights = apply_gate(
                        folds[2021], features[2021], columns, tree,
                        leaf_weights, float(shrink),
                    )
                    source_r = score(
                        folds[2020]["y"], source_pred, folds[2020]["regular"]
                    )["score"] - score(
                        folds[2020]["y"], folds[2020]["parent"],
                        folds[2020]["regular"],
                    )["score"]
                    valid_r = score(
                        folds[2021]["y"], valid_pred, folds[2021]["regular"]
                    )["score"] - score(
                        folds[2021]["y"], folds[2021]["parent"],
                        folds[2021]["regular"],
                    )["score"]
                    full_mask = np.ones(len(valid_pred), dtype=bool)
                    valid_full = score(
                        folds[2021]["y"], valid_pred, full_mask
                    )["score"] - score(
                        folds[2021]["y"], folds[2021]["parent"], full_mask
                    )["score"]
                    trials.append({
                        "feature_set": feature_name,
                        "max_depth": int(max_depth),
                        "min_samples_leaf": int(min_leaf),
                        "global_shrink": float(shrink),
                        "2020_R_gain": float(source_r),
                        "2021_R_gain": float(valid_r),
                        "2021_full_gain": float(valid_full),
                        "2021_changed_R_fraction": float(
                            np.mean(valid_weights[folds[2021]["regular"]] > 0.0)
                        ),
                        "tree": tree_meta,
                    })
    selected = max(
        trials,
        key=lambda item: (
            item["2021_R_gain"], item["2021_full_gain"], item["2020_R_gain"],
            -item["max_depth"], item["min_samples_leaf"],
            tuple(-ord(ch) for ch in item["feature_set"]),
            -item["global_shrink"],
        ),
    )
    selected_key = (
        selected["feature_set"], selected["max_depth"],
        selected["min_samples_leaf"],
    )
    tree, leaf_weights, tree_meta, columns = fitted[selected_key]
    source_prediction, source_weights = apply_gate(
        folds[2020], features[2020], columns, tree, leaf_weights,
        selected["global_shrink"],
    )
    validation_prediction, validation_weights = apply_gate(
        folds[2021], features[2021], columns, tree, leaf_weights,
        selected["global_shrink"],
    )
    metrics = {
        "2020_fit": route_metrics(folds[2020], source_prediction, 8302020),
        "2021_forward": route_metrics(
            folds[2021], validation_prediction, 8302021
        ),
    }
    gate = prereg["source_gate"]
    forward = metrics["2021_forward"]
    checks = {
        "R_point": forward["R"]["gain"] >= gate["minimum_2021_R_point_gain"],
        "full_point": forward["full"]["gain"] >= gate["minimum_2021_routed_full_point_gain"],
        "R_ci": forward["R"]["pitcher_cluster_95_ci"]["ci_low"] > 0.0,
        "full_ci": forward["full"]["pitcher_cluster_95_ci"]["ci_low"] > 0.0,
        "changed_fraction": selected["2021_changed_R_fraction"] >= gate["minimum_2021_changed_R_fraction"],
        "leaf_count": tree_meta["leaf_count"] <= gate["maximum_leaf_count"],
    }
    passed = bool(all(checks.values()))
    artifact = PRED / "v5_prevyear_loss_gate_source_2021.npz"
    if artifact.exists():
        raise FileExistsError(f"immutable artifact exists: {artifact}")
    np.savez_compressed(
        artifact,
        y=folds[2021]["y"], row_index=folds[2021]["row_index"],
        cluster=folds[2021]["cluster"], parent_m3=folds[2021]["parent"],
        direct_update=folds[2021]["alternate"],
        gate_weight=validation_weights, final_prediction=validation_prediction,
    )
    lock = {
        "experiment_id": prereg["experiment_id"],
        "status": "source_pass_locked_for_2022" if passed else "source_failed_closed",
        "preregister_sha256": digest(PREREG),
        "selected_hyperparameters": {
            key: selected[key] for key in (
                "feature_set", "max_depth", "min_samples_leaf", "global_shrink"
            )
        },
        "feature_columns": columns,
        "temporal_refit_rule": prereg["temporal_protocol"]["rule"],
        "advance_to_2022": passed,
        "2022_details_unread_at_lock": True,
        "2023_and_2024_composite_unread": True,
    }
    LOCK.write_text(json.dumps(safe(lock), ensure_ascii=False, indent=2), encoding="utf-8")
    report = {
        "experiment_id": prereg["experiment_id"],
        "status": "source_pass" if passed else "source_failed",
        "years_read": [2020, 2021],
        "years_not_read": [2022, 2023, 2024],
        "preregister_sha256": digest(PREREG),
        "script_sha256": digest(Path(__file__)),
        "search_trial_count": len(trials),
        "selected": selected,
        "selected_training_tree": tree_meta,
        "metrics": metrics,
        "source_gate": {"requirements": gate, "checks": checks, "pass": passed},
        "input_sha256": {
            str(year): {
                "direct": digest(folds[year]["direct_path"]),
                "anchor": "deterministic load_anchor",
            }
            for year in YEARS
        },
        "artifact": {
            "path": str(artifact.relative_to(ROOT)), "sha256": digest(artifact)
        },
        "lock": {"path": str(LOCK.relative_to(ROOT)), "sha256": digest(LOCK)},
        "top_20": sorted(
            trials,
            key=lambda item: (
                item["2021_R_gain"], item["2021_full_gain"],
                item["2020_R_gain"], -item["max_depth"],
                item["min_samples_leaf"], -item["global_shrink"],
            ),
            reverse=True,
        )[:20],
        "goal_status": "active",
        "goal_completion_claimed": False,
    }
    REPORT.write_text(json.dumps(safe(report), ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(safe({
        "status": report["status"], "selected": selected,
        "checks": checks, "metrics": metrics,
    }), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
