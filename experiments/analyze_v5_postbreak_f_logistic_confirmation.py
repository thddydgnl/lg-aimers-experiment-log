#!/usr/bin/env python3
"""Single locked 2024 confirmation for the post-break F logistic recipe."""

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
LOCK = ROOT / "experiments/params/v5_postbreak_f_logistic_confirmation_lock.json"
REPORT = ROOT / "experiments/results/v5_postbreak_f_logistic_confirmation.json"
ARTIFACT = PRED / "v5_postbreak_f_logistic_confirmation_2024.npz"

RATE_COLUMNS = [
    "asof_pitcher_prev1_game_success_rate",
    "asof_pitcher_prev3_game_success_rate",
    "asof_pitcher_prev5_game_success_rate",
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


def end_state(frame: pd.DataFrame, target_year: int) -> pd.DataFrame:
    history = frame.loc[frame["season"].lt(target_year)].copy()
    n = history["asof_pitcher_n"].fillna(0).to_numpy(dtype=np.int64)
    rate = history["asof_pitcher_success_rate"].fillna(0.0).to_numpy(dtype=np.float64)
    history["_end_n"] = n + 1
    history["_end_s"] = np.rint(rate * n).astype(np.int64) + history["control_success"].to_numpy(dtype=np.int64)
    return history.groupby("pitcher_id", sort=False, observed=True).tail(1).set_index("pitcher_id")[["_end_n", "_end_s"]]


def core_features(rows: pd.DataFrame, parent: np.ndarray, state: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    pitchers = rows["pitcher_id"]
    frozen_n = pitchers.map(state["_end_n"]).fillna(0).to_numpy(dtype=np.int64)
    frozen_s = pitchers.map(state["_end_s"]).fillna(0).to_numpy(dtype=np.int64)
    asof_n = rows["asof_pitcher_n"].fillna(0).to_numpy(dtype=np.int64)
    career = rows["asof_pitcher_success_rate"].fillna(0.5).to_numpy(dtype=np.float64)
    asof_s = np.rint(career * asof_n).astype(np.int64)
    current_n = asof_n - frozen_n
    current_s = asof_s - frozen_s
    invalid = (current_n < 0) | (current_s < 0) | (current_s > current_n)
    current_n = np.where(invalid, 0, current_n).astype(np.float64)
    current_s = np.where(invalid, 0, current_s).astype(np.float64)
    p = np.clip(parent, 1e-6, 1.0 - 1e-6)
    posterior = (current_s + 50.0 * p) / (current_n + 50.0)
    posterior = np.where(invalid, p, posterior)
    recent = rows[RATE_COLUMNS].apply(pd.to_numeric, errors="coerce")
    features = pd.DataFrame({
        "logit_exact_c": np.log(p / (1.0 - p)),
        "current_posterior_k50": posterior,
        "current_posterior_minus_exact_c": posterior - p,
        "current_reliability_k50": current_n / (current_n + 50.0),
        "log1p_current_n": np.log1p(current_n),
        "prev1_success": recent.iloc[:, 0].to_numpy(dtype=np.float64),
        "prev3_success": recent.iloc[:, 1].to_numpy(dtype=np.float64),
        "prev5_success": recent.iloc[:, 2].to_numpy(dtype=np.float64),
        "recent_success_mean": recent.mean(axis=1, skipna=True).to_numpy(dtype=np.float64),
        "recent_success_std": recent.std(axis=1, skipna=True, ddof=0).to_numpy(dtype=np.float64),
    }, index=rows.index)
    return features, {
        "invalid_counter_rows": int(invalid.sum()),
        "current_n_median": float(np.median(current_n)),
        "current_n_positive_fraction": float(np.mean(current_n > 0)),
        "columns": list(features.columns),
    }


def aligned_rows(frame: pd.DataFrame, artifact: dict[str, np.ndarray], year: int) -> pd.DataFrame:
    rows = frame.loc[artifact["row_index"].astype(np.int64)].copy()
    if not rows["season"].eq(year).all():
        raise ValueError(f"season alignment failed: {year}")
    if not np.array_equal(rows["control_success"].to_numpy(dtype=np.int8), artifact["y"].astype(np.int8)):
        raise ValueError(f"target alignment failed: {year}")
    return rows


def aligned_anchor(path: Path, row_index: np.ndarray) -> np.ndarray:
    artifact = load_npz(path)
    if np.array_equal(artifact["row_index"].astype(np.int64), row_index):
        return prediction(artifact)
    lookup = pd.Series(prediction(artifact), index=artifact["row_index"].astype(np.int64))
    result = lookup.reindex(row_index)
    if result.isna().any():
        raise ValueError(f"anchor alignment failed: {path}")
    return result.to_numpy(dtype=np.float64)


def make_model(c_value: float) -> Pipeline:
    return Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
        ("model", LogisticRegression(
            C=float(c_value), penalty="l2", solver="lbfgs", max_iter=1000,
            random_state=2026,
        )),
    ])


def comparison(y: np.ndarray, anchor: np.ndarray, candidate: np.ndarray, cluster: np.ndarray, mask: np.ndarray, seed: int) -> dict[str, Any]:
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
        raise FileExistsError("immutable confirmation output already exists")
    lock = json.loads(LOCK.read_text(encoding="utf-8"))
    for item in lock["immutable_inputs"]:
        path = ROOT / item["path"]
        if digest(path).lower() != item["sha256"].lower():
            raise ValueError(f"immutable input hash mismatch: {path}")
    frame = load_frame()
    train_path = PRED / "v3_sparse_c_backtest_2023.npz"
    valid_path = PRED / "v3_outcome_trackmanrich_overall_e14k50_batter80_middle100_2024.npz"
    train_artifact = load_npz(train_path)
    valid_artifact = load_npz(valid_path)
    train_rows = aligned_rows(frame, train_artifact, 2023)
    valid_rows = aligned_rows(frame, valid_artifact, 2024)
    train_parent = prediction(train_artifact)
    valid_parent = prediction(valid_artifact)
    train_x, train_meta = core_features(train_rows, train_parent, end_state(frame, 2023))
    valid_x, valid_meta = core_features(valid_rows, valid_parent, end_state(frame, 2024))
    expected_columns = lock["selected_recipe"]["feature_columns"]
    if list(train_x.columns) != expected_columns or list(valid_x.columns) != expected_columns:
        raise ValueError("locked feature-column mismatch")
    train_f = train_rows["game_type"].astype(str).eq("F").to_numpy()
    valid_f = valid_rows["game_type"].astype(str).eq("F").to_numpy()
    model = make_model(float(lock["selected_recipe"]["C"]))
    model.fit(train_x.loc[train_f], train_artifact["y"].astype(np.int8)[train_f])
    component = model.predict_proba(valid_x.loc[valid_f])[:, 1].astype(np.float64)
    gamma = float(lock["selected_recipe"]["gamma"])
    candidate = valid_parent.copy()
    candidate[valid_f] = np.clip(
        (1.0 - gamma) * valid_parent[valid_f] + gamma * component,
        1e-6, 1.0 - 1e-6,
    )
    y = valid_artifact["y"].astype(np.int8)
    cluster = valid_artifact["cluster"]
    all_mask = np.ones(len(y), dtype=bool)
    r_mask = ~valid_f
    anchors = {
        "exact_c": valid_parent,
        "honest_identity": aligned_anchor(PRED / "v5_honest_m3_r_identity_2024.npz", valid_artifact["row_index"].astype(np.int64)),
        "honest_grid": aligned_anchor(PRED / "v5_honest_m3_r_grid_2024.npz", valid_artifact["row_index"].astype(np.int64)),
    }
    full_comparisons = {
        name: comparison(y, anchor, candidate, cluster, all_mask, 8510000 + index * 1000)
        for index, (name, anchor) in enumerate(anchors.items())
    }
    f_same_parent = comparison(y, valid_parent, candidate, cluster, valid_f, 8522024)
    r_same_parent = comparison(y, valid_parent, candidate, cluster, r_mask, 8532024)
    threshold = float(lock["required_raw_gain"])
    source = json.loads((ROOT / lock["source_report"]).read_text(encoding="utf-8"))
    source_values: list[float] = []
    for item in source["staged_2023_full_comparisons"].values():
        source_values.extend([float(item["gain"]), float(item["pitcher_cluster_95_ci"]["ci_low"])])
    g_dev = float(min(source_values))
    g_confirm = float(min(item["gain"] for item in full_comparisons.values()))
    g_ci = float(min(item["pitcher_cluster_95_ci"]["ci_low"] for item in full_comparisons.values()))
    g_robust = float(min(g_dev, g_confirm, g_ci))
    expected = float(lock["actual_v3_anchor"] + lock["haircut"] * max(0.0, g_robust))
    checks = {
        "R_bit_identity": bool(np.array_equal(candidate[r_mask], valid_parent[r_mask])),
        "R_point_zero": abs(float(r_same_parent["gain"])) <= 1e-12,
        "F_point_positive": f_same_parent["gain"] > 0.0,
        "F_ci_lower_positive": f_same_parent["pitcher_cluster_95_ci"]["ci_low"] > 0.0,
        "all_anchor_full_points_above_threshold": all(item["gain"] > threshold for item in full_comparisons.values()),
        "all_anchor_full_ci_lowers_above_threshold": all(
            item["pitcher_cluster_95_ci"]["ci_low"] > threshold
            for item in full_comparisons.values()
        ),
        "expected_score_strictly_above_1190": expected > 1190.0,
    }
    passed = all(checks.values())
    np.savez_compressed(
        ARTIFACT,
        y=y, row_index=valid_artifact["row_index"].astype(np.int64),
        cluster=cluster, parent_exact_c=valid_parent,
        logistic_component_F=component, F_mask=valid_f.astype(np.int8),
        final_prediction=candidate,
    )
    report = {
        "experiment_id": lock["experiment_id"],
        "status": "confirmation_pass" if passed else "confirmation_failed_closed",
        "lock_sha256": digest(LOCK), "year_read": 2024,
        "test_rows_read": False,
        "selected_recipe": lock["selected_recipe"],
        "fit_rows_2023_F": int(train_f.sum()), "valid_rows_2024_F": int(valid_f.sum()),
        "feature_metadata": {"fit_2023": train_meta, "confirm_2024": valid_meta},
        "component_2024_F": {
            "mean": float(component.mean()), "std": float(component.std()),
            "target_rate": float(y[valid_f].mean()),
        },
        "R_same_parent": r_same_parent, "F_same_parent": f_same_parent,
        "full_comparisons": full_comparisons,
        "G_dev": g_dev, "G_confirm": g_confirm, "G_ci": g_ci,
        "G_robust": g_robust, "conservative_expected_score": expected,
        "checks": checks,
        "artifact": {"path": str(ARTIFACT.relative_to(ROOT)), "sha256": digest(ARTIFACT)},
        "package_authorized": bool(passed),
        "goal_status": "active", "goal_completion_claimed": False,
    }
    REPORT.write_text(json.dumps(safe(report), ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(safe({
        "status": report["status"], "F_same_parent": f_same_parent,
        "full_comparisons": full_comparisons,
        "G_dev": g_dev, "G_confirm": g_confirm, "G_ci": g_ci,
        "G_robust": g_robust, "conservative_expected_score": expected,
        "checks": checks,
    }), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
