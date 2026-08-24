#!/usr/bin/env python3
"""Strict 2020/2021 source gate for a global pairwise rank component."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pandas as pd
from catboost import CatBoostRanker, Pool


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.analyze_v5_dense_pitchtype_moe import load, safe, score  # noqa: E402
from experiments.run_v5_h1_residual import cluster_bootstrap_score_gain  # noqa: E402


TRAIN = ROOT / "open/data/train.csv"
PRED = ROOT / "experiments/results/predictions"
PREREG = ROOT / "experiments/params/v5_global_pairwise_rank_preregister.json"
REPORT = ROOT / "experiments/results/v5_global_pairwise_rank_source.json"
YEARS = (2020, 2021)
PARENTS = {
    year: PRED / f"v4_m3_c_backtest_{year}_{year}.npz" for year in YEARS
}
CATEGORICAL = [
    "game_month", "game_dayofweek", "inning", "top_bottom", "game_type",
    "balls_before", "strikes_before", "outs_before", "base_state",
    "num_runners_on", "pitcher_id", "batter_id", "pitcher_hand",
    "batter_hand", "pitcher_team_id", "batter_team_id",
]
DROP = {"row_id", "control_success"}


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def prepare(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.drop(columns=[column for column in DROP if column in frame]).copy()
    for column in CATEGORICAL:
        result[column] = result[column].astype("string").fillna("__missing__")
    for column in result.columns:
        if column not in CATEGORICAL:
            result[column] = pd.to_numeric(result[column], errors="coerce").astype(np.float32)
    return result


def rank_groups(length: int, size: int = 64) -> np.ndarray:
    return np.arange(length, dtype=np.int64) // size


def affine_calibrate(
    score_fit: np.ndarray,
    y_fit: np.ndarray,
    score_target: np.ndarray,
    shrink: float,
) -> tuple[np.ndarray, dict[str, float]]:
    center = float(np.mean(score_fit))
    target_rate = float(np.mean(y_fit))
    centered = score_fit - center
    slope = float(np.dot(centered, y_fit - target_rate) / max(np.dot(centered, centered), 1e-12))
    slope *= shrink
    prediction = target_rate + slope * (score_target - center)
    return np.clip(prediction, 1e-6, 1.0 - 1e-6), {
        "calibration_score_mean": center,
        "calibration_target_rate": target_rate,
        "raw_brier_slope": slope / shrink,
        "applied_slope": slope,
    }


def metrics(
    y: np.ndarray,
    parent: np.ndarray,
    candidate: np.ndarray,
    regular: np.ndarray,
) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for name, mask in {"full": np.ones(len(y), dtype=bool), "R": regular, "F": ~regular}.items():
        before = score(y, parent, mask)
        after = score(y, candidate, mask)
        output[name] = {
            "parent_score": float(before["score"]),
            "candidate_score": float(after["score"]),
            "gain": float(after["score"] - before["score"]),
        }
    return output


def main() -> None:
    if REPORT.exists():
        raise FileExistsError(f"immutable report exists: {REPORT}")
    prereg = json.loads(PREREG.read_text(encoding="utf-8"))
    if prereg["status"] != "locked_before_source_metrics":
        raise ValueError("unexpected preregistration status")
    # Exactly 2019-2021.  No feature or target from a later season is read.
    frame = pd.read_csv(TRAIN, nrows=728_588)
    if set(frame["season"].unique()) != {2019, 2020, 2021}:
        raise ValueError("source loader crossed the 2021 boundary")

    folds: dict[int, dict[str, Any]] = {}
    model_meta: dict[str, Any] = {}
    for year in YEARS:
        artifact = load(PARENTS[year])
        row_index = artifact["row_index"].astype(np.int64)
        valid = frame.loc[row_index]
        if not valid["season"].eq(year).all():
            raise ValueError(f"{year}: validation season mismatch")
        y = artifact["y"].astype(np.int8)
        if not np.array_equal(valid["control_success"].to_numpy(dtype=np.int8), y):
            raise ValueError(f"{year}: target mismatch")

        history = frame.loc[frame["season"].lt(year) & frame["game_type"].eq("R")]
        latest = int(history["season"].max())
        latest_index = history.index[history["season"].eq(latest)].to_numpy(dtype=np.int64)
        split_at = int(np.floor(0.80 * len(latest_index)))
        calibration_index = latest_index[split_at:]
        fit_mask = ~history.index.isin(calibration_index)
        fit = history.loc[fit_mask]
        calibration = frame.loc[calibration_index]

        x_fit = prepare(fit)
        x_calibration = prepare(calibration)
        x_valid = prepare(valid)
        if list(x_fit.columns) != list(x_calibration.columns) or list(x_fit.columns) != list(x_valid.columns):
            raise ValueError("feature order mismatch")
        cat_indices = [x_fit.columns.get_loc(column) for column in CATEGORICAL]
        params = prereg["model"]
        model = CatBoostRanker(
            loss_function=params["loss_function"],
            iterations=int(params["iterations"]),
            depth=int(params["depth"]),
            learning_rate=float(params["learning_rate"]),
            l2_leaf_reg=float(params["l2_leaf_reg"]),
            random_seed=int(params["random_seed"]),
            task_type=str(params["task_type"]),
            devices="0",
            verbose=False,
            allow_writing_files=False,
        )
        model.fit(
            Pool(
                x_fit,
                label=fit["control_success"].to_numpy(dtype=np.float32),
                group_id=rank_groups(len(fit)),
                cat_features=cat_indices,
            )
        )
        calibration_score = model.predict(x_calibration).astype(np.float64)
        valid_score = model.predict(x_valid).astype(np.float64)
        folds[year] = {
            "artifact": artifact,
            "valid": valid,
            "y": y,
            "parent": artifact["catboost_outcome"].astype(np.float64),
            "regular": valid["game_type"].eq("R").to_numpy(),
            "calibration_y": calibration["control_success"].to_numpy(dtype=np.float64),
            "calibration_score": calibration_score,
            "valid_score": valid_score,
        }
        model_meta[str(year)] = {
            "history_seasons": sorted(int(v) for v in history["season"].unique()),
            "latest_history_season": latest,
            "fit_rows": int(len(fit)),
            "calibration_rows": int(len(calibration)),
            "valid_rows": int(len(valid)),
            "feature_count": int(x_fit.shape[1]),
            "calibration_positive_rate": float(calibration["control_success"].mean()),
        }

    trials: list[dict[str, Any]] = []
    predictions: dict[tuple[float, float, int], np.ndarray] = {}
    calibration_meta: dict[str, Any] = {}
    for shrink_value in prereg["calibration_slope_shrink_grid"]:
        shrink = float(shrink_value)
        rank_probability: dict[int, np.ndarray] = {}
        for year in YEARS:
            fold = folds[year]
            probability, meta = affine_calibrate(
                fold["calibration_score"], fold["calibration_y"], fold["valid_score"], shrink
            )
            rank_probability[year] = probability
            calibration_meta[f"shrink={shrink}:year={year}"] = meta
        for gamma_value in prereg["blend_gamma_grid"]:
            gamma = float(gamma_value)
            years: dict[str, Any] = {}
            for year in YEARS:
                fold = folds[year]
                candidate = fold["parent"].copy()
                r = fold["regular"]
                candidate[r] = (1.0 - gamma) * candidate[r] + gamma * rank_probability[year][r]
                predictions[(shrink, gamma, year)] = candidate
                years[str(year)] = metrics(fold["y"], fold["parent"], candidate, r)
            trials.append({
                "slope_shrink": shrink,
                "gamma": gamma,
                "minimum_full_gain": float(min(years[str(y)]["full"]["gain"] for y in YEARS)),
                "minimum_R_gain": float(min(years[str(y)]["R"]["gain"] for y in YEARS)),
                "mean_full_gain": float(np.mean([years[str(y)]["full"]["gain"] for y in YEARS])),
                "years": years,
            })
    selected = max(trials, key=lambda x: (x["minimum_full_gain"], x["minimum_R_gain"], x["mean_full_gain"], -x["gamma"]))

    intervals: dict[str, Any] = {}
    for year in YEARS:
        fold = folds[year]
        candidate = predictions[(selected["slope_shrink"], selected["gamma"], year)]
        intervals[str(year)] = {}
        for offset, (route, mask) in enumerate({"full": np.ones(len(fold["y"]), bool), "R": fold["regular"]}.items()):
            intervals[str(year)][route] = cluster_bootstrap_score_gain(
                fold["y"], fold["parent"], candidate, fold["artifact"]["cluster"], mask,
                iterations=2000, seed=8126000 + year + offset * 1000,
            )
    gate = prereg["source_gate"]
    checks: dict[str, bool] = {}
    for year in YEARS:
        checks[f"{year}_full_gain"] = selected["years"][str(year)]["full"]["gain"] >= float(gate["minimum_full_gain_each_year"])
        checks[f"{year}_R_gain"] = selected["years"][str(year)]["R"]["gain"] >= float(gate["minimum_R_gain_each_year"])
        checks[f"{year}_full_ci"] = intervals[str(year)]["full"]["ci_low"] > float(gate["pitcher_cluster_95_ci_low_each_year"])
        checks[f"{year}_R_ci"] = intervals[str(year)]["R"]["ci_low"] > float(gate["pitcher_cluster_95_ci_low_each_year"])
    passed = all(checks.values())

    artifacts: dict[str, Any] = {}
    for year in YEARS:
        fold = folds[year]
        candidate = predictions[(selected["slope_shrink"], selected["gamma"], year)]
        path = PRED / f"v5_global_pairwise_rank_source_{year}.npz"
        if path.exists():
            raise FileExistsError(f"immutable artifact exists: {path}")
        np.savez_compressed(
            path, y=fold["y"], row_index=fold["artifact"]["row_index"],
            cluster=fold["artifact"]["cluster"], parent=fold["parent"],
            rank_score=fold["valid_score"], final_prediction=candidate,
        )
        artifacts[str(year)] = {"path": str(path.relative_to(ROOT)), "sha256": digest(path)}

    report = {
        "experiment_id": prereg["experiment_id"],
        "status": "source_pass" if passed else "source_failed",
        "preregister_sha256": digest(PREREG),
        "script_sha256": digest(Path(__file__)),
        "years_read": list(YEARS), "years_not_read": [2022, 2023, 2024],
        "model_metadata": model_meta, "calibration_metadata": calibration_meta,
        "trials": trials, "selected": selected, "intervals": intervals,
        "gate": {"requirements": gate, "checks": checks, "pass": passed},
        "artifacts": artifacts, "goal_status": "active", "goal_completion_claimed": False,
    }
    REPORT.write_text(json.dumps(safe(report), ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(safe({"status": report["status"], "selected": selected, "intervals": intervals, "checks": checks}), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
