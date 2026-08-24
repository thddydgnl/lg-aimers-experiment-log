#!/usr/bin/env python3
"""Locked post-break F specialist source and confirmation gates."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier


ROOT = Path(__file__).resolve().parents[1]
TRAIN = ROOT / "open" / "data" / "train.csv"
PARAM = ROOT / "experiments" / "params" / "v5_postbreak_f_specialist_preregister.json"
LOCK = ROOT / "experiments" / "params" / "v5_postbreak_f_specialist_source_lock.json"
RESULTS = ROOT / "experiments" / "results"
PREDICTIONS = RESULTS / "predictions"
SOURCE_REPORT = RESULTS / "v5_postbreak_f_specialist_source.json"
CONFIRM_REPORT = RESULTS / "v5_postbreak_f_specialist_confirm2024.json"
THRESHOLD = 132.11992465293324

PARENT_PATHS = {
    2023: PREDICTIONS / "v5_recent_routed_regime_dev_2023.npz",
    2024: PREDICTIONS / "v3_outcome_trackmanrich_overall_e14k50_batter80_middle100_2024.npz",
}
ANCHOR_PATHS = {
    year: {
        "honest_identity": PREDICTIONS / f"v5_honest_m3_r_identity_{year}.npz",
        "honest_grid": PREDICTIONS / f"v5_honest_m3_r_grid_{year}.npz",
    }
    for year in (2023, 2024)
}
BASE_CATEGORICAL = {
    "game_dayofweek",
    "top_bottom",
    "base_state",
    "pitcher_hand",
    "batter_hand",
    "pitcher_team_id",
    "batter_team_id",
    "f_count_state",
    "f_hand_matchup",
}
VIEW_PRIORITY = {"no_id": 0, "fixed_mean": 1, "pitcher_id": 2}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", required=True, choices=("source", "confirm"))
    parser.add_argument("--bootstrap", type=int, default=2000)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        return {key: np.asarray(archive[key]) for key in archive.files}


def prediction_key(artifact: dict[str, np.ndarray]) -> str:
    for key in ("parent_exact_c", "catboost_outcome", "final_prediction"):
        if key in artifact:
            return key
    raise KeyError(f"No prediction key in {sorted(artifact)}")


def aligned_prediction(artifact: dict[str, np.ndarray], row_index: np.ndarray) -> np.ndarray:
    source_index = artifact["row_index"].astype(np.int64)
    if len(np.unique(source_index)) != len(source_index):
        raise ValueError("artifact row indices are not unique")
    positions = pd.Series(np.arange(len(source_index), dtype=np.int64), index=source_index)
    selected = positions.reindex(row_index.astype(np.int64))
    if selected.isna().any():
        raise ValueError("artifact does not cover requested rows")
    return artifact[prediction_key(artifact)][selected.to_numpy(dtype=np.int64)].astype(np.float64)


def aligned_anchor(path: Path, row_index: np.ndarray) -> np.ndarray:
    return aligned_prediction(load_npz(path), row_index)


def raw_score(y: np.ndarray, prediction: np.ndarray) -> float:
    rate = float(y.mean())
    reference = max(rate * (1.0 - rate), 1e-12)
    brier = float(np.mean(np.square(prediction - y)))
    return 100000.0 * (1.0 - brier / reference)


def summary(y: np.ndarray, prediction: np.ndarray) -> dict[str, float | int]:
    return {
        "rows": int(len(y)),
        "target_rate": float(y.mean()),
        "prediction_mean": float(prediction.mean()),
        "prediction_std": float(prediction.std()),
        "brier": float(np.mean(np.square(prediction - y))),
        "score": raw_score(y, prediction),
    }


def bootstrap_gain(
    y: np.ndarray,
    parent: np.ndarray,
    candidate: np.ndarray,
    cluster: np.ndarray,
    iterations: int,
    seed: int,
) -> dict[str, float | int]:
    work = pd.DataFrame(
        {
            "cluster": cluster.astype(str),
            "y": y.astype(np.float64),
            "parent_error": np.square(parent - y),
            "candidate_error": np.square(candidate - y),
        }
    )
    grouped = work.groupby("cluster", sort=False, observed=True).agg(
        n=("y", "size"),
        y_sum=("y", "sum"),
        parent_error=("parent_error", "sum"),
        candidate_error=("candidate_error", "sum"),
    )
    values = grouped.to_numpy(dtype=np.float64)
    rng = np.random.default_rng(seed)
    draws = np.empty(iterations, dtype=np.float64)
    cluster_count = len(values)
    for iteration in range(iterations):
        sampled = values[rng.integers(0, cluster_count, cluster_count)].sum(axis=0)
        n, y_sum, parent_error, candidate_error = sampled
        rate = y_sum / n
        reference = max(rate * (1.0 - rate), 1e-12)
        draws[iteration] = 100000.0 * (parent_error - candidate_error) / n / reference
    point = raw_score(y, candidate) - raw_score(y, parent)
    return {
        "point": float(point),
        "ci_low": float(np.quantile(draws, 0.025)),
        "ci_high": float(np.quantile(draws, 0.975)),
        "bootstrap_std": float(draws.std(ddof=1)),
        "iterations": int(iterations),
        "cluster_count": int(cluster_count),
    }


def comparison(
    y: np.ndarray,
    parent: np.ndarray,
    candidate: np.ndarray,
    cluster: np.ndarray,
    iterations: int,
    seed: int,
) -> dict[str, Any]:
    return {
        "parent": summary(y, parent),
        "candidate": summary(y, candidate),
        "gain": float(raw_score(y, candidate) - raw_score(y, parent)),
        "pitcher_cluster_95_ci": bootstrap_gain(
            y, parent, candidate, cluster, iterations, seed
        ),
    }


def season_start_state(frame: pd.DataFrame, year: int, entity: str) -> pd.DataFrame:
    if entity == "pitcher_id":
        n_column = "asof_pitcher_n"
        rate_column = "asof_pitcher_success_rate"
    elif entity == "batter_id":
        n_column = "asof_batter_n"
        rate_column = "asof_batter_success_rate"
    else:
        raise ValueError(entity)
    history = frame.loc[frame["season"] < year, [entity, n_column, rate_column, "control_success"]]
    last = history.groupby(entity, sort=False, observed=True).tail(1).copy()
    n_before = last[n_column].fillna(0).to_numpy(dtype=np.float64)
    successes_before = np.rint(
        last[rate_column].fillna(0).to_numpy(dtype=np.float64) * n_before
    )
    last["n_end"] = n_before + 1.0
    last["s_end"] = successes_before + last["control_success"].to_numpy(dtype=np.float64)
    return last.set_index(entity)[["n_end", "s_end"]]


def add_current_success_state(
    features: pd.DataFrame,
    rows: pd.DataFrame,
    state: pd.DataFrame,
    entity: str,
    prior: float,
    k: float,
) -> None:
    if entity == "pitcher_id":
        prefix = "f_pitcher"
        n_column = "asof_pitcher_n"
        rate_column = "asof_pitcher_success_rate"
    else:
        prefix = "f_batter"
        n_column = "asof_batter_n"
        rate_column = "asof_batter_success_rate"
    ids = rows[entity]
    n_end = ids.map(state["n_end"]).fillna(0).to_numpy(dtype=np.float64)
    s_end = ids.map(state["s_end"]).fillna(0).to_numpy(dtype=np.float64)
    total_n = rows[n_column].fillna(0).to_numpy(dtype=np.float64)
    total_s = np.rint(rows[rate_column].fillna(0).to_numpy(dtype=np.float64) * total_n)
    current_n = np.maximum(total_n - n_end, 0.0)
    current_s = np.clip(total_s - s_end, 0.0, current_n)
    posterior = (current_s + k * prior) / (current_n + k)
    career = rows[rate_column].fillna(prior).to_numpy(dtype=np.float64)
    features[f"{prefix}_current_n"] = current_n
    features[f"{prefix}_current_s"] = current_s
    features[f"{prefix}_current_log_n"] = np.log1p(current_n)
    features[f"{prefix}_current_rate"] = posterior
    features[f"{prefix}_current_minus_career"] = posterior - career
    features[f"{prefix}_unseen"] = (~ids.isin(state.index)).astype(np.int8).to_numpy()


def feature_matrix(
    rows: pd.DataFrame,
    pitcher_state: pd.DataFrame,
    batter_state: pd.DataFrame,
    prior: float,
    view: str,
) -> tuple[pd.DataFrame, list[str]]:
    excluded = {"row_id", "control_success", "season", "game_type", "pitcher_id", "batter_id"}
    columns = [column for column in rows.columns if column not in excluded]
    features = rows[columns].copy()
    features["f_count_state"] = (
        rows["balls_before"].astype(str) + "-" + rows["strikes_before"].astype(str)
    )
    features["f_hand_matchup"] = (
        rows["pitcher_hand"].astype(str) + "-" + rows["batter_hand"].astype(str)
    )
    features["f_log_pitcher_n"] = np.log1p(rows["asof_pitcher_n"].fillna(0).to_numpy(dtype=np.float64))
    features["f_log_batter_n"] = np.log1p(rows["asof_batter_n"].fillna(0).to_numpy(dtype=np.float64))
    add_current_success_state(features, rows, pitcher_state, "pitcher_id", prior, 100.0)
    add_current_success_state(features, rows, batter_state, "batter_id", prior, 100.0)
    if view == "pitcher_id":
        features["pitcher_id"] = rows["pitcher_id"].astype(str)
    elif view != "no_id":
        raise ValueError(view)
    categorical = []
    for column in features.columns:
        if column in BASE_CATEGORICAL or column == "pitcher_id" or features[column].dtype == object:
            features[column] = features[column].fillna("__NA__").astype(str)
            categorical.append(column)
        else:
            features[column] = pd.to_numeric(features[column], errors="coerce")
    return features, categorical


def model_params() -> dict[str, Any]:
    return {
        "loss_function": "Logloss",
        "iterations": 400,
        "depth": 5,
        "learning_rate": 0.04,
        "l2_leaf_reg": 20.0,
        "random_seed": 2026,
        "random_strength": 0.0,
        "bootstrap_type": "Bayesian",
        "bagging_temperature": 0.0,
        "task_type": "GPU",
        "allow_writing_files": False,
        "verbose": False,
    }


def fit_predict_view(
    fit_rows: pd.DataFrame,
    eval_rows: pd.DataFrame,
    pitcher_state: pd.DataFrame,
    batter_state: pd.DataFrame,
    view: str,
) -> np.ndarray:
    prior = float(fit_rows["control_success"].mean())
    x_fit, categorical = feature_matrix(fit_rows, pitcher_state, batter_state, prior, view)
    x_eval, eval_categorical = feature_matrix(eval_rows, pitcher_state, batter_state, prior, view)
    if list(x_fit.columns) != list(x_eval.columns) or categorical != eval_categorical:
        raise ValueError("fit/eval feature mismatch")
    model = CatBoostClassifier(**model_params())
    model.fit(x_fit, fit_rows["control_success"].to_numpy(dtype=np.int8), cat_features=categorical)
    prediction = model.predict_proba(x_eval)[:, 1].astype(np.float64)
    return np.clip(prediction, 1e-6, 1.0 - 1e-6)


def view_predictions(
    fit_rows: pd.DataFrame,
    eval_rows: pd.DataFrame,
    pitcher_state: pd.DataFrame,
    batter_state: pd.DataFrame,
    needed: set[str] | None = None,
) -> dict[str, np.ndarray]:
    requested = needed or {"no_id", "pitcher_id", "fixed_mean"}
    base_views = {"no_id", "pitcher_id"} if "fixed_mean" in requested else set(requested)
    predictions = {
        view: fit_predict_view(fit_rows, eval_rows, pitcher_state, batter_state, view)
        for view in sorted(base_views)
    }
    if "fixed_mean" in requested:
        predictions["fixed_mean"] = 0.5 * predictions["no_id"] + 0.5 * predictions["pitcher_id"]
    return {view: predictions[view] for view in requested}


def load_aligned_frame(year: int) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    frame = pd.read_csv(TRAIN)
    frame.index = np.arange(len(frame), dtype=np.int64)
    artifact = load_npz(PARENT_PATHS[year])
    row_index = artifact["row_index"].astype(np.int64)
    rows = frame.loc[row_index].copy()
    y = artifact["y"].astype(np.int8)
    if not np.array_equal(rows["control_success"].to_numpy(dtype=np.int8), y):
        raise ValueError(f"{year} target alignment mismatch")
    parent = aligned_prediction(artifact, row_index)
    return frame, rows, parent


def source_run(iterations: int) -> None:
    prereg = json.loads(PARAM.read_text(encoding="utf-8"))
    frame, rows, parent = load_aligned_frame(2023)
    if set(rows["season"].unique()) != {2023}:
        raise ValueError("2023 artifact contains another season")
    f_rows = rows["game_type"].astype(str).eq("F")
    pitcher_state = season_start_state(frame, 2023, "pitcher_id")
    batter_state = season_start_state(frame, 2023, "batter_id")

    selection_fit = rows[f_rows & rows["game_month"].isin([4, 5])]
    selection_eval_mask = f_rows & rows["game_month"].isin([6, 7])
    selection_eval = rows[selection_eval_mask]
    selection_predictions = view_predictions(
        selection_fit, selection_eval, pitcher_state, batter_state
    )
    selection_parent = parent[selection_eval_mask.to_numpy()]
    selection_y = selection_eval["control_success"].to_numpy(dtype=np.int8)
    selection_cluster = selection_eval["pitcher_id"].astype(str).to_numpy()
    selection_scores = {
        view: float(raw_score(selection_y, prediction) - raw_score(selection_y, selection_parent))
        for view, prediction in selection_predictions.items()
    }
    selected_view = sorted(
        selection_scores,
        key=lambda view: (-selection_scores[view], VIEW_PRIORITY[view]),
    )[0]
    selection_metrics = comparison(
        selection_y,
        selection_parent,
        selection_predictions[selected_view],
        selection_cluster,
        iterations,
        20230607,
    )

    confirmation_fit = rows[f_rows & rows["game_month"].isin([4, 5, 6, 7])]
    confirmation_eval_mask = f_rows & rows["game_month"].isin([8, 9])
    confirmation_eval = rows[confirmation_eval_mask]
    confirmation_prediction = view_predictions(
        confirmation_fit,
        confirmation_eval,
        pitcher_state,
        batter_state,
        {selected_view},
    )[selected_view]
    confirmation_parent = parent[confirmation_eval_mask.to_numpy()]
    confirmation_y = confirmation_eval["control_success"].to_numpy(dtype=np.int8)
    confirmation_cluster = confirmation_eval["pitcher_id"].astype(str).to_numpy()
    confirmation_metrics = comparison(
        confirmation_y,
        confirmation_parent,
        confirmation_prediction,
        confirmation_cluster,
        iterations,
        20230809,
    )

    staged = parent.copy()
    staged[selection_eval_mask.to_numpy()] = selection_predictions[selected_view]
    staged[confirmation_eval_mask.to_numpy()] = confirmation_prediction
    row_index = rows.index.to_numpy(dtype=np.int64)
    y_all = rows["control_success"].to_numpy(dtype=np.int8)
    cluster_all = rows["pitcher_id"].astype(str).to_numpy()
    anchors = {
        "exact_c": parent,
        **{
            name: aligned_anchor(path, row_index)
            for name, path in ANCHOR_PATHS[2023].items()
        },
    }
    full_comparisons = {
        name: comparison(y_all, anchor, staged, cluster_all, iterations, 20230000 + offset)
        for offset, (name, anchor) in enumerate(anchors.items(), start=1)
    }
    full_points = [value["gain"] for value in full_comparisons.values()]
    full_ci_lowers = [value["pitcher_cluster_95_ci"]["ci_low"] for value in full_comparisons.values()]
    g_dev = float(min(full_points + full_ci_lowers))
    checks = {
        "selection_F_point_at_least_800": selection_metrics["gain"] >= 800.0,
        "selection_F_ci_lower_positive": selection_metrics["pitcher_cluster_95_ci"]["ci_low"] > 0.0,
        "confirmation_F_point_at_least_800": confirmation_metrics["gain"] >= 800.0,
        "confirmation_F_ci_lower_positive": confirmation_metrics["pitcher_cluster_95_ci"]["ci_low"] > 0.0,
        "staged_full_points_above_threshold": all(value > THRESHOLD for value in full_points),
        "staged_full_ci_lowers_above_threshold": all(value > THRESHOLD for value in full_ci_lowers),
    }
    passed = all(checks.values())
    PREDICTIONS.mkdir(parents=True, exist_ok=True)
    source_artifact = PREDICTIONS / "v5_postbreak_f_specialist_source_2023.npz"
    np.savez_compressed(
        source_artifact,
        y=y_all,
        row_index=row_index,
        cluster=cluster_all,
        parent_exact_c=parent,
        staged_prediction=staged,
        specialist_mask=(selection_eval_mask | confirmation_eval_mask).to_numpy(dtype=bool),
    )
    report = {
        "experiment_id": prereg["experiment_id"],
        "status": "source_pass" if passed else "source_failed_closed",
        "preregister_sha256": sha256(PARAM),
        "script_sha256": sha256(Path(__file__)),
        "selected_view": selected_view,
        "selection_all_view_gains": selection_scores,
        "selection": selection_metrics,
        "confirmation": confirmation_metrics,
        "staged_2023_full_comparisons": full_comparisons,
        "G_dev": g_dev,
        "checks": checks,
        "rows": {
            "selection_fit": int(len(selection_fit)),
            "selection_eval": int(len(selection_eval)),
            "confirmation_fit": int(len(confirmation_fit)),
            "confirmation_eval": int(len(confirmation_eval)),
        },
        "artifact": {"path": str(source_artifact.relative_to(ROOT)), "sha256": sha256(source_artifact)},
        "confirmation_2024_authorized": passed,
        "goal_completion_claimed": False,
    }
    SOURCE_REPORT.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    lock = {
        "experiment_id": prereg["experiment_id"],
        "status": "locked_after_source_pass_before_2024" if passed else "source_failed_closed",
        "selected_view": selected_view,
        "preregister_sha256": sha256(PARAM),
        "script_sha256": sha256(Path(__file__)),
        "source_report_sha256": sha256(SOURCE_REPORT),
        "source_artifact_sha256": sha256(source_artifact),
        "G_dev": g_dev,
        "confirmation_2024_authorized": passed,
        "goal_completion_claimed": False,
    }
    LOCK.write_text(json.dumps(lock, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


def confirm_run(iterations: int) -> None:
    if not LOCK.exists():
        raise FileNotFoundError("source lock is missing")
    lock = json.loads(LOCK.read_text(encoding="utf-8"))
    if not lock.get("confirmation_2024_authorized"):
        raise RuntimeError("source gate did not authorize 2024")
    if lock["preregister_sha256"] != sha256(PARAM) or lock["script_sha256"] != sha256(Path(__file__)):
        raise RuntimeError("preregister or implementation changed after source lock")
    frame, rows, parent = load_aligned_frame(2024)
    if set(rows["season"].unique()) != {2024}:
        raise ValueError("2024 artifact contains another season")
    fit_rows = frame[(frame["season"] == 2023) & frame["game_type"].astype(str).eq("F")]
    f_mask = rows["game_type"].astype(str).eq("F")
    eval_rows = rows[f_mask]
    pitcher_state = season_start_state(frame, 2024, "pitcher_id")
    batter_state = season_start_state(frame, 2024, "batter_id")
    selected_view = str(lock["selected_view"])
    f_prediction = view_predictions(
        fit_rows, eval_rows, pitcher_state, batter_state, {selected_view}
    )[selected_view]
    candidate = parent.copy()
    candidate[f_mask.to_numpy()] = f_prediction
    row_index = rows.index.to_numpy(dtype=np.int64)
    y = rows["control_success"].to_numpy(dtype=np.int8)
    cluster = rows["pitcher_id"].astype(str).to_numpy()
    anchors = {
        "exact_c": parent,
        **{
            name: aligned_anchor(path, row_index)
            for name, path in ANCHOR_PATHS[2024].items()
        },
    }
    comparisons = {
        name: comparison(y, anchor, candidate, cluster, iterations, 20240000 + offset)
        for offset, (name, anchor) in enumerate(anchors.items(), start=1)
    }
    f_metrics = comparison(
        y[f_mask.to_numpy()],
        parent[f_mask.to_numpy()],
        candidate[f_mask.to_numpy()],
        cluster[f_mask.to_numpy()],
        iterations,
        20242424,
    )
    points = [value["gain"] for value in comparisons.values()]
    ci_lowers = [value["pitcher_cluster_95_ci"]["ci_low"] for value in comparisons.values()]
    checks = {
        "F_point_at_least_800": f_metrics["gain"] >= 800.0,
        "F_ci_lower_positive": f_metrics["pitcher_cluster_95_ci"]["ci_low"] > 0.0,
        "all_full_points_above_threshold": all(value > THRESHOLD for value in points),
        "all_full_ci_lowers_above_threshold": all(value > THRESHOLD for value in ci_lowers),
    }
    passed = all(checks.values())
    g_confirm = float(min(points))
    g_ci = float(min(ci_lowers))
    g_robust = float(min(float(lock["G_dev"]), g_confirm, g_ci))
    expected_lower = 1090.9100565103 + 0.75 * max(0.0, g_robust)
    PREDICTIONS.mkdir(parents=True, exist_ok=True)
    artifact = PREDICTIONS / "v5_postbreak_f_specialist_confirm_2024.npz"
    np.savez_compressed(
        artifact,
        y=y,
        row_index=row_index,
        cluster=cluster,
        parent_exact_c=parent,
        final_prediction=candidate,
        f_mask=f_mask.to_numpy(dtype=bool),
    )
    report = {
        "experiment_id": lock["experiment_id"],
        "status": "confirmation_pass" if passed else "confirmation_failed_closed",
        "selected_view": selected_view,
        "lock_sha256": sha256(LOCK),
        "year_read": 2024,
        "test_rows_read": False,
        "rows": {"fit_2023_F": int(len(fit_rows)), "validation_2024_F": int(f_mask.sum()), "full": int(len(rows))},
        "F_same_parent": f_metrics,
        "full_comparisons": comparisons,
        "checks": checks,
        "conservative_expected_score": {
            "actual_v3_anchor": 1090.9100565103,
            "G_dev": float(lock["G_dev"]),
            "G_confirm": g_confirm,
            "G_ci": g_ci,
            "G_robust": g_robust,
            "haircut": 0.75,
            "expected_lb_lower": expected_lower,
            "passes_1190": expected_lower > 1190.0,
        },
        "artifact": {"path": str(artifact.relative_to(ROOT)), "sha256": sha256(artifact)},
        "goal_completion_claimed": False,
    }
    CONFIRM_REPORT.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


def main() -> None:
    args = parse_args()
    if args.mode == "source":
        source_run(args.bootstrap)
    else:
        confirm_run(args.bootstrap)


if __name__ == "__main__":
    main()
