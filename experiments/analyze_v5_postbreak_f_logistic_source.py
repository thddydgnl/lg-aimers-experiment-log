#!/usr/bin/env python3
"""Immutable forward-2023 source gate for a low-dimensional F logistic model."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.analyze_v5_dense_pitchtype_moe import safe, score
from experiments.run_v5_h1_residual import cluster_bootstrap_score_gain

TRAIN = ROOT / "open/data/train.csv"
PRED = ROOT / "experiments/results/predictions"
PREREG = ROOT / "experiments/params/v5_postbreak_f_logistic_preregister.json"
REPORT = ROOT / "experiments/results/v5_postbreak_f_logistic_source.json"
ARTIFACT = PRED / "v5_postbreak_f_logistic_source_2023.npz"
TARGET_YEAR = 2023

RATE_COLUMNS = [
    "asof_pitcher_prev1_game_success_rate",
    "asof_pitcher_prev3_game_success_rate",
    "asof_pitcher_prev5_game_success_rate",
    "asof_pitcher_prev1_game_middle_rate",
    "asof_pitcher_prev3_game_middle_rate",
    "asof_pitcher_prev5_game_middle_rate",
    "asof_pitcher_reverse_rate",
    "asof_pitcher_middle_rate",
    "asof_pitcher_ball_rate",
    "asof_pitcher_strike_rate",
]


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as data:
        return {key: data[key] for key in data.files}


def prediction(artifact: dict[str, np.ndarray]) -> np.ndarray:
    for key in ("catboost_outcome", "final_prediction"):
        if key in artifact:
            return artifact[key].astype(np.float64)
    raise KeyError("prediction key not found")


def load_frame() -> pd.DataFrame:
    columns = [
        "season", "game_month", "game_type", "pitcher_id",
        "asof_pitcher_n", "asof_pitcher_success_rate", "control_success",
        *RATE_COLUMNS,
    ]
    return pd.read_csv(TRAIN, usecols=columns)


def previous_end_state(frame: pd.DataFrame) -> pd.DataFrame:
    history = frame.loc[frame["season"].lt(TARGET_YEAR)].copy()
    n = history["asof_pitcher_n"].fillna(0).to_numpy(dtype=np.int64)
    rate = history["asof_pitcher_success_rate"].fillna(0.0).to_numpy(dtype=np.float64)
    history["_end_n"] = n + 1
    history["_end_s"] = np.rint(rate * n).astype(np.int64) + history["control_success"].to_numpy(dtype=np.int64)
    return history.groupby("pitcher_id", sort=False, observed=True).tail(1).set_index("pitcher_id")[["_end_n", "_end_s"]]


def feature_views(rows: pd.DataFrame, parent: np.ndarray, end: pd.DataFrame) -> tuple[dict[str, pd.DataFrame], dict[str, Any]]:
    pitchers = rows["pitcher_id"]
    end_n = pitchers.map(end["_end_n"]).fillna(0).to_numpy(dtype=np.int64)
    end_s = pitchers.map(end["_end_s"]).fillna(0).to_numpy(dtype=np.int64)
    asof_n = rows["asof_pitcher_n"].fillna(0).to_numpy(dtype=np.int64)
    career = rows["asof_pitcher_success_rate"].fillna(0.5).to_numpy(dtype=np.float64)
    asof_s = np.rint(career * asof_n).astype(np.int64)
    current_n = asof_n - end_n
    current_s = asof_s - end_s
    invalid = (current_n < 0) | (current_s < 0) | (current_s > current_n)
    current_n = np.where(invalid, 0, current_n).astype(np.float64)
    current_s = np.where(invalid, 0, current_s).astype(np.float64)
    clipped_parent = np.clip(parent, 1e-6, 1.0 - 1e-6)
    posterior = (current_s + 50.0 * clipped_parent) / (current_n + 50.0)
    posterior = np.where(invalid, clipped_parent, posterior)
    success_columns = [f"asof_pitcher_prev{h}_game_success_rate" for h in (1, 3, 5)]
    middle_columns = [f"asof_pitcher_prev{h}_game_middle_rate" for h in (1, 3, 5)]
    success = rows[success_columns].apply(pd.to_numeric, errors="coerce")
    middle = rows[middle_columns].apply(pd.to_numeric, errors="coerce")
    core = pd.DataFrame({
        "logit_exact_c": np.log(clipped_parent / (1.0 - clipped_parent)),
        "current_posterior_k50": posterior,
        "current_posterior_minus_exact_c": posterior - clipped_parent,
        "current_reliability_k50": current_n / (current_n + 50.0),
        "log1p_current_n": np.log1p(current_n),
        "prev1_success": success.iloc[:, 0].to_numpy(dtype=np.float64),
        "prev3_success": success.iloc[:, 1].to_numpy(dtype=np.float64),
        "prev5_success": success.iloc[:, 2].to_numpy(dtype=np.float64),
        "recent_success_mean": success.mean(axis=1, skipna=True).to_numpy(dtype=np.float64),
        "recent_success_std": success.std(axis=1, skipna=True, ddof=0).to_numpy(dtype=np.float64),
    }, index=rows.index)
    extended = core.copy()
    extended["prev1_middle"] = middle.iloc[:, 0].to_numpy(dtype=np.float64)
    extended["prev3_middle"] = middle.iloc[:, 1].to_numpy(dtype=np.float64)
    extended["prev5_middle"] = middle.iloc[:, 2].to_numpy(dtype=np.float64)
    extended["recent_middle_mean"] = middle.mean(axis=1, skipna=True).to_numpy(dtype=np.float64)
    extended["recent_middle_std"] = middle.std(axis=1, skipna=True, ddof=0).to_numpy(dtype=np.float64)
    for column in (
        "asof_pitcher_reverse_rate", "asof_pitcher_middle_rate",
        "asof_pitcher_ball_rate", "asof_pitcher_strike_rate",
    ):
        extended[column] = pd.to_numeric(rows[column], errors="coerce").to_numpy(dtype=np.float64)
    meta = {
        "invalid_counter_rows": int(invalid.sum()),
        "current_n_median": float(np.median(current_n)),
        "current_n_positive_fraction": float(np.mean(current_n > 0)),
        "core_columns": list(core.columns),
        "extended_columns": list(extended.columns),
    }
    return {"core": core, "extended": extended}, meta


def make_model(c_value: float) -> Pipeline:
    return Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
        ("model", LogisticRegression(
            C=float(c_value), penalty="l2", solver="lbfgs", max_iter=1000,
            random_state=2026,
        )),
    ])


def raw_gain(y: np.ndarray, parent: np.ndarray, candidate: np.ndarray) -> float:
    mask = np.ones(len(y), dtype=bool)
    return float(score(y, candidate, mask)["score"] - score(y, parent, mask)["score"])


def aligned_anchor(path: Path, row_index: np.ndarray) -> np.ndarray:
    artifact = load_npz(path)
    if np.array_equal(artifact["row_index"].astype(np.int64), row_index):
        return prediction(artifact)
    lookup = pd.Series(prediction(artifact), index=artifact["row_index"].astype(np.int64))
    result = lookup.reindex(row_index)
    if result.isna().any():
        raise ValueError(f"anchor alignment failed: {path}")
    return result.to_numpy(dtype=np.float64)


def comparison(y: np.ndarray, anchor: np.ndarray, candidate: np.ndarray, cluster: np.ndarray, seed: int) -> dict[str, Any]:
    mask = np.ones(len(y), dtype=bool)
    anchor_metric = score(y, anchor, mask)
    candidate_metric = score(y, candidate, mask)
    return {
        "anchor": anchor_metric,
        "candidate": candidate_metric,
        "gain": float(candidate_metric["score"] - anchor_metric["score"]),
        "pitcher_cluster_95_ci": cluster_bootstrap_score_gain(
            y, anchor, candidate, cluster, mask, iterations=2000, seed=seed,
        ),
    }


def main() -> None:
    if REPORT.exists() or ARTIFACT.exists():
        raise FileExistsError("immutable source output already exists")
    prereg = json.loads(PREREG.read_text(encoding="utf-8"))
    frame = load_frame()
    parent_path = PRED / "v3_sparse_c_backtest_2023.npz"
    artifact = load_npz(parent_path)
    row_index = artifact["row_index"].astype(np.int64)
    rows = frame.loc[row_index].copy()
    if not rows["season"].eq(TARGET_YEAR).all():
        raise ValueError("parent rows are not 2023")
    y = artifact["y"].astype(np.int8)
    if not np.array_equal(y, rows["control_success"].to_numpy(dtype=np.int8)):
        raise ValueError("target alignment failed")
    parent = prediction(artifact)
    views, feature_meta = feature_views(rows, parent, previous_end_state(frame))
    f_mask = rows["game_type"].astype(str).eq("F").to_numpy()
    splits = prereg["source_protocol"]["forward_splits"]
    c_grid = [float(value) for value in prereg["model"]["C_grid"]]
    gamma_grid = [float(value) for value in prereg["model"]["outer_parent_blend_gamma_grid"]]
    view_order = list(prereg["row_local_features"]["feature_view_order"])
    component_cache: dict[tuple[str, float, int], tuple[np.ndarray, np.ndarray]] = {}
    trials: list[dict[str, Any]] = []
    for view_index, view_name in enumerate(view_order):
        features = views[view_name]
        for c_value in c_grid:
            for split_index, split in enumerate(splits):
                fit_mask = f_mask & rows["game_month"].isin(split["fit_months"]).to_numpy()
                eval_mask = f_mask & rows["game_month"].isin(split["eval_months"]).to_numpy()
                model = make_model(c_value)
                model.fit(features.loc[fit_mask], y[fit_mask])
                component_cache[(view_name, c_value, split_index)] = (
                    eval_mask,
                    model.predict_proba(features.loc[eval_mask])[:, 1].astype(np.float64),
                )
        for c_value in c_grid:
            for gamma in gamma_grid:
                split_metrics: list[dict[str, Any]] = []
                combined_mask = np.zeros(len(rows), dtype=bool)
                combined_candidate = parent.copy()
                for split_index, split in enumerate(splits):
                    eval_mask, component = component_cache[(view_name, c_value, split_index)]
                    candidate = np.clip(
                        (1.0 - gamma) * parent[eval_mask] + gamma * component,
                        1e-6, 1.0 - 1e-6,
                    )
                    combined_mask |= eval_mask
                    combined_candidate[eval_mask] = candidate
                    split_y = y[eval_mask]
                    split_parent = parent[eval_mask]
                    split_metrics.append({
                        "split_index": split_index,
                        "fit_months": split["fit_months"],
                        "eval_months": split["eval_months"],
                        "rows": int(eval_mask.sum()),
                        "gain": raw_gain(split_y, split_parent, candidate),
                        "parent_mean": float(split_parent.mean()),
                        "component_mean": float(component.mean()),
                        "candidate_mean": float(candidate.mean()),
                        "target_rate": float(split_y.mean()),
                    })
                combined_gain = raw_gain(
                    y[combined_mask], parent[combined_mask], combined_candidate[combined_mask]
                )
                trials.append({
                    "view": view_name,
                    "view_order": view_index,
                    "C": c_value,
                    "gamma": gamma,
                    "minimum_split_F_gain": float(min(item["gain"] for item in split_metrics)),
                    "combined_forward_OOF_F_gain": float(combined_gain),
                    "split_metrics": split_metrics,
                })
    selected = max(trials, key=lambda item: (
        item["minimum_split_F_gain"], item["combined_forward_OOF_F_gain"],
        -item["view_order"], -item["C"], -item["gamma"],
    ))
    selected_mask = np.zeros(len(rows), dtype=bool)
    selected_candidate = parent.copy()
    selected_component = np.full(len(rows), np.nan, dtype=np.float64)
    for split_index, _ in enumerate(splits):
        eval_mask, component = component_cache[(selected["view"], selected["C"], split_index)]
        selected_mask |= eval_mask
        selected_component[eval_mask] = component
        selected_candidate[eval_mask] = np.clip(
            (1.0 - selected["gamma"]) * parent[eval_mask] + selected["gamma"] * component,
            1e-6, 1.0 - 1e-6,
        )
    combined = comparison(
        y[selected_mask], parent[selected_mask], selected_candidate[selected_mask],
        rows.loc[selected_mask, "pitcher_id"].astype(str).to_numpy(), 8462023,
    )
    full_candidate = parent.copy()
    full_candidate[selected_mask] = selected_candidate[selected_mask]
    anchors = {
        "exact_c": parent,
        "honest_identity": aligned_anchor(PRED / "v5_honest_m3_r_identity_2023.npz", row_index),
        "honest_grid": aligned_anchor(PRED / "v5_honest_m3_r_grid_2023.npz", row_index),
    }
    full_comparisons = {
        name: comparison(y, anchor, full_candidate, artifact["cluster"], 8470000 + index * 1000)
        for index, (name, anchor) in enumerate(anchors.items())
    }
    threshold = float(
        prereg["source_protocol"]["source_gate"]
        ["staged_2023_full_point_and_pitcher_cluster_ci_lower_gain_over_exact_C_and_both_honest_anchors_strictly_above"]
    )
    checks = {
        "each_split_point_positive": all(item["gain"] > 0.0 for item in selected["split_metrics"]),
        "combined_point_at_least_400": combined["gain"] >= 400.0,
        "combined_ci_lower_positive": combined["pitcher_cluster_95_ci"]["ci_low"] > 0.0,
        "all_anchor_full_points_above_threshold": all(item["gain"] > threshold for item in full_comparisons.values()),
        "all_anchor_full_ci_lowers_above_threshold": all(
            item["pitcher_cluster_95_ci"]["ci_low"] > threshold
            for item in full_comparisons.values()
        ),
    }
    passed = all(checks.values())
    np.savez_compressed(
        ARTIFACT,
        y=y, row_index=row_index, cluster=artifact["cluster"],
        parent_exact_c=parent, final_prediction=full_candidate,
        forward_eval_mask=selected_mask.astype(np.int8),
        logistic_component=selected_component,
    )
    report = {
        "experiment_id": prereg["experiment_id"],
        "status": "source_pass" if passed else "source_failed",
        "preregister_sha256": digest(PREREG),
        "script_sha256": digest(Path(__file__)),
        "years_read": [2023], "years_not_read": [2024],
        "parent_artifact": {"path": str(parent_path.relative_to(ROOT)), "sha256": digest(parent_path)},
        "feature_metadata": feature_meta,
        "trials": trials,
        "selected": selected,
        "combined_forward_OOF_F": combined,
        "staged_2023_full_comparisons": full_comparisons,
        "source_gate": {"checks": checks, "pass": bool(passed), "threshold": threshold},
        "artifact": {"path": str(ARTIFACT.relative_to(ROOT)), "sha256": digest(ARTIFACT)},
        "confirmation_2024_authorized": bool(passed),
        "goal_status": "active", "goal_completion_claimed": False
    }
    REPORT.write_text(json.dumps(safe(report), ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(safe({
        "status": report["status"], "selected": selected,
        "combined_forward_OOF_F": combined, "full_comparisons": full_comparisons,
        "checks": checks,
    }), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
