#!/usr/bin/env python3
"""Locked 2022/2023 temporal state-offset gate evaluation.

The meta learner sees only exact-C predictions produced on strictly earlier
outer seasons.  An adaptive, row-local current-season state supplies XGBoost's
fixed base margin; shallow trees learn a centered correction and never read a
different row from the target fold at inference time.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pandas as pd
from scipy.special import expit, logit
from xgboost import XGBClassifier

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.run_v5_h1_residual import cluster_bootstrap_score_gain  # noqa: E402
from experiments.v5_adaptive_state_space import (  # noqa: E402
    _lifetime_end_state,
    build_adaptive_state_probability,
)

TRAIN = ROOT / "open/data/train.csv"
PREREG = ROOT / "experiments/params/v5_temporal_state_offset_gate_preregister.json"
REPORT = ROOT / "experiments/results/v5_temporal_state_offset_gate_source.json"
PREDICTIONS = ROOT / "experiments/results/predictions"
YEARS = (2022, 2023)
OUTER_START = 2020
PARENT_ARTIFACTS = {
    2020: PREDICTIONS / "v4_m3_c_backtest_2020_2020.npz",
    2021: PREDICTIONS / "v4_m3_c_backtest_2021_2021.npz",
    2022: PREDICTIONS / "v3_sparse_c_backtest_2022.npz",
    2023: PREDICTIONS / "v3_sparse_c_backtest_2023.npz",
}
HONEST_ARTIFACTS = {
    "honest_r_identity": {
        year: PREDICTIONS / f"v5_honest_m3_r_identity_{year}.npz"
        for year in YEARS
    },
    "honest_r_grid": {
        year: PREDICTIONS / f"v5_honest_m3_r_grid_{year}.npz"
        for year in YEARS
    },
}

RAW_COLUMNS = [
    "season", "game_month", "game_dayofweek", "inning", "top_bottom",
    "game_type", "balls_before", "strikes_before", "outs_before",
    "run_total_before", "score_diff_home", "score_diff_pitcher_team",
    "runner_on_1b", "runner_on_2b", "runner_on_3b", "num_runners_on",
    "base_state", "home_win_expectancy", "away_win_expectancy", "li",
    "pitcher_id", "pitcher_hand", "batter_hand", "asof_pitcher_n",
    "asof_pitcher_success_rate", "asof_pitcher_reverse_rate",
    "asof_pitcher_middle_rate", "asof_pitcher_ball_rate",
    "asof_pitcher_strike_rate", "asof_pitcher_prev1_game_success_rate",
    "asof_pitcher_prev3_game_success_rate",
    "asof_pitcher_prev5_game_success_rate",
    "asof_pitcher_prev1_game_middle_rate",
    "asof_pitcher_prev3_game_middle_rate",
    "asof_pitcher_prev5_game_middle_rate", "asof_batter_n",
    "asof_batter_success_rate", "asof_batter_middle_rate",
    "asof_pitcher_fastball_rate", "asof_pitcher_breaking_rate",
    "asof_pitcher_offspeed_rate", "control_success",
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [safe(item) for item in value]
    if isinstance(value, (np.floating, float)):
        number = float(value)
        return number if np.isfinite(number) else None
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    return value


def load_archive(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        return {key: np.asarray(archive[key]) for key in archive.files}


def parent_prediction(archive: dict[str, np.ndarray]) -> np.ndarray:
    if "catboost_outcome" in archive:
        return archive["catboost_outcome"].astype(np.float64)
    if "parent" in archive:
        return archive["parent"].astype(np.float64)
    raise KeyError(f"no exact-C prediction key: {sorted(archive)}")


def current_season_n(frame: pd.DataFrame, target_year: int) -> np.ndarray:
    history = frame.loc[frame["season"].lt(target_year)]
    valid = frame.loc[frame["season"].eq(target_year)]
    boundary = _lifetime_end_state(history)
    end_n = valid["pitcher_id"].map(boundary["end_n"]).fillna(0).to_numpy(
        dtype=np.int64
    )
    n_asof = pd.to_numeric(valid["asof_pitcher_n"], errors="coerce").fillna(
        0
    ).to_numpy(dtype=np.int64)
    value = n_asof - end_n
    return np.maximum(value, 0)


def fixed_category_code(series: pd.Series, name: str) -> np.ndarray:
    text = series.astype("string").fillna("__missing__").astype(str)
    if name == "top_bottom":
        return text.map({"B": 0, "T": 1}).fillna(-1).to_numpy(dtype=np.float32)
    if name == "base_state":
        return text.map({
            "___": 0, "1__": 1, "_2_": 2, "12_": 3,
            "__3": 4, "1_3": 5, "_23": 6, "123": 7,
        }).fillna(-1).to_numpy(dtype=np.float32)
    return pd.to_numeric(series, errors="coerce").fillna(-1).to_numpy(
        dtype=np.float32
    )


def build_features(
    rows: pd.DataFrame,
    parent: np.ndarray,
    state: np.ndarray,
    season_n: np.ndarray,
) -> pd.DataFrame:
    parent = np.clip(np.asarray(parent, dtype=np.float64), 1e-6, 1.0 - 1e-6)
    state = np.clip(np.asarray(state, dtype=np.float64), 1e-6, 1.0 - 1e-6)
    parent_logit = logit(parent)
    state_logit = logit(state)
    values: dict[str, np.ndarray] = {
        "parent_probability": parent.astype(np.float32),
        "parent_logit": parent_logit.astype(np.float32),
        "adaptive_state_probability": state.astype(np.float32),
        "adaptive_state_logit": state_logit.astype(np.float32),
        "parent_minus_state": (parent - state).astype(np.float32),
        "parent_logit_minus_state_logit": (parent_logit - state_logit).astype(
            np.float32
        ),
        "current_season_pitcher_n": season_n.astype(np.float32),
        "log1p_current_season_pitcher_n": np.log1p(season_n).astype(np.float32),
    }
    direct = [
        "game_month", "game_dayofweek", "inning", "balls_before",
        "strikes_before", "outs_before", "run_total_before",
        "score_diff_home", "score_diff_pitcher_team", "runner_on_1b",
        "runner_on_2b", "runner_on_3b", "num_runners_on",
        "home_win_expectancy", "away_win_expectancy", "li",
        "asof_pitcher_success_rate", "asof_pitcher_reverse_rate",
        "asof_pitcher_middle_rate", "asof_pitcher_ball_rate",
        "asof_pitcher_strike_rate", "asof_pitcher_prev1_game_success_rate",
        "asof_pitcher_prev3_game_success_rate",
        "asof_pitcher_prev5_game_success_rate",
        "asof_pitcher_prev1_game_middle_rate",
        "asof_pitcher_prev3_game_middle_rate",
        "asof_pitcher_prev5_game_middle_rate",
        "asof_batter_success_rate", "asof_batter_middle_rate",
        "asof_pitcher_fastball_rate", "asof_pitcher_breaking_rate",
        "asof_pitcher_offspeed_rate",
    ]
    for column in direct:
        values[column] = pd.to_numeric(rows[column], errors="coerce").to_numpy(
            dtype=np.float32
        )
    for column in ["top_bottom", "base_state", "pitcher_hand", "batter_hand"]:
        values[column] = fixed_category_code(rows[column], column)
    pitcher_hand = values["pitcher_hand"]
    batter_hand = values["batter_hand"]
    values["same_hand"] = (pitcher_hand == batter_hand).astype(np.float32)
    values["count_code"] = (
        3.0 * values["balls_before"] + values["strikes_before"]
    ).astype(np.float32)
    values["asof_pitcher_n_log1p"] = np.log1p(
        pd.to_numeric(rows["asof_pitcher_n"], errors="coerce").fillna(0).to_numpy(
            dtype=np.float64
        )
    ).astype(np.float32)
    values["asof_batter_n_log1p"] = np.log1p(
        pd.to_numeric(rows["asof_batter_n"], errors="coerce").fillna(0).to_numpy(
            dtype=np.float64
        )
    ).astype(np.float32)
    return pd.DataFrame(values, index=rows.index)


def fill_from_training(
    train_x: pd.DataFrame, valid_x: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, float]]:
    medians: dict[str, float] = {}
    train = train_x.copy()
    valid = valid_x.copy()
    for column in train.columns:
        value = float(np.nanmedian(train[column].to_numpy(dtype=np.float64)))
        if not np.isfinite(value):
            value = 0.0
        medians[column] = value
        train[column] = train[column].fillna(value)
        valid[column] = valid[column].fillna(value)
    return train.astype(np.float32), valid.astype(np.float32), medians


def competition_score(y: np.ndarray, prediction: np.ndarray, mask: np.ndarray) -> float:
    yy = y[mask].astype(np.float64)
    pp = prediction[mask].astype(np.float64)
    rate = float(yy.mean())
    brier = float(np.mean(np.square(pp - yy)))
    return 100000.0 * (1.0 - brier / (rate * (1.0 - rate)))


def summarize(
    y: np.ndarray, prediction: np.ndarray, mask: np.ndarray
) -> dict[str, float]:
    yy = y[mask].astype(np.float64)
    pp = prediction[mask].astype(np.float64)
    rate = float(yy.mean())
    brier = float(np.mean(np.square(pp - yy)))
    return {
        "rows": int(mask.sum()),
        "target_rate": rate,
        "prediction_mean": float(pp.mean()),
        "prediction_std": float(pp.std()),
        "brier": brier,
        "score": 100000.0 * (1.0 - brier / (rate * (1.0 - rate))),
    }


def fit_target(
    frame: pd.DataFrame,
    target_year: int,
    prereg: dict[str, Any],
    folds: dict[int, dict[str, Any]],
) -> tuple[np.ndarray, dict[str, Any]]:
    train_parts: list[pd.DataFrame] = []
    target_parts: list[np.ndarray] = []
    state_parts: list[np.ndarray] = []
    weight_parts: list[np.ndarray] = []
    source_counts: dict[str, int] = {}
    for outer_year in range(OUTER_START, target_year):
        fold = folds[outer_year]
        regular = fold["game_type"] == "R"
        features = build_features(
            fold["rows"], fold["parent"], fold["state"], fold["season_n"]
        )
        train_parts.append(features.loc[regular])
        target_parts.append(fold["y"][regular].astype(np.int8))
        state_parts.append(fold["state"][regular].astype(np.float64))
        weight = 0.8 ** (target_year - 1 - outer_year)
        weight_parts.append(np.full(int(regular.sum()), weight, dtype=np.float64))
        source_counts[str(outer_year)] = int(regular.sum())

    valid_fold = folds[target_year]
    valid_x = build_features(
        valid_fold["rows"], valid_fold["parent"], valid_fold["state"],
        valid_fold["season_n"],
    )
    train_x = pd.concat(train_parts, axis=0)
    train_y = np.concatenate(target_parts)
    train_state = np.concatenate(state_parts)
    train_weight = np.concatenate(weight_parts)
    train_x, valid_x, medians = fill_from_training(train_x, valid_x)
    base_train = logit(np.clip(train_state, 1e-6, 1.0 - 1e-6))
    base_valid = logit(np.clip(valid_fold["state"], 1e-6, 1.0 - 1e-6))

    settings = dict(prereg["fixed_recipe"]["xgboost"])
    model = XGBClassifier(**settings)
    model.fit(
        train_x,
        train_y,
        sample_weight=train_weight,
        base_margin=base_train,
        verbose=False,
    )
    train_margin = np.asarray(
        model.predict(train_x, output_margin=True, base_margin=base_train),
        dtype=np.float64,
    )
    correction = train_margin - base_train
    correction_center = float(np.average(correction, weights=train_weight))
    valid_margin = np.asarray(
        model.predict(valid_x, output_margin=True, base_margin=base_valid),
        dtype=np.float64,
    )
    prediction = np.clip(expit(valid_margin - correction_center), 1e-6, 1.0 - 1e-6)
    importance = sorted(
        zip(train_x.columns, model.feature_importances_),
        key=lambda item: float(item[1]),
        reverse=True,
    )
    details = {
        "target_year": target_year,
        "meta_source_rows": source_counts,
        "meta_train_rows": int(len(train_x)),
        "meta_train_positive_rate": float(np.average(train_y, weights=train_weight)),
        "season_weight_min": float(train_weight.min()),
        "season_weight_max": float(train_weight.max()),
        "base_train_mean": float(train_state.mean()),
        "base_valid_mean": float(valid_fold["state"].mean()),
        "correction_center": correction_center,
        "uncentered_valid_correction_mean": float((valid_margin - base_valid).mean()),
        "prediction_mean": float(prediction.mean()),
        "prediction_std": float(prediction.std()),
        "feature_count": int(train_x.shape[1]),
        "feature_columns": list(train_x.columns),
        "training_medians": medians,
        "top_feature_importance": [
            {"feature": name, "importance": float(value)}
            for name, value in importance[:20]
        ],
        "xgboost": settings,
        "same_fold_meta_fit": False,
        "validation_target_used_for_fit": False,
        "other_validation_rows_used_at_inference": False,
        "row_independent": True,
    }
    return prediction, details


def main() -> None:
    if REPORT.exists():
        raise FileExistsError(f"immutable report exists: {REPORT}")
    prereg = json.loads(PREREG.read_text(encoding="utf-8"))
    if prereg["status"] != "locked_before_2022_2023_candidate_metrics":
        raise ValueError("unexpected preregistration state")

    archives = {year: load_archive(path) for year, path in PARENT_ARTIFACTS.items()}
    maximum_row = max(int(archives[year]["row_index"].max()) for year in YEARS)
    frame = pd.read_csv(TRAIN, usecols=RAW_COLUMNS, nrows=maximum_row + 1)
    if int(frame["season"].max()) != 2023:
        raise ValueError("source reader crossed the locked 2023 boundary")

    folds: dict[int, dict[str, Any]] = {}
    state_metadata: dict[str, Any] = {}
    for year in range(OUTER_START, max(YEARS) + 1):
        archive = archives[year]
        row_index = archive["row_index"].astype(np.int64)
        expected = frame.index[frame["season"].eq(year)].to_numpy(dtype=np.int64)
        if not np.array_equal(row_index, expected):
            raise ValueError(f"{year}: parent row order mismatch")
        rows = frame.loc[row_index]
        y = archive["y"].astype(np.int8)
        if not np.array_equal(rows["control_success"].to_numpy(dtype=np.int8), y):
            raise ValueError(f"{year}: parent target mismatch")
        state, metadata = build_adaptive_state_probability(frame, year)
        season_n = current_season_n(frame, year)
        folds[year] = {
            "row_index": row_index,
            "rows": rows,
            "y": y,
            "cluster": archive["cluster"].astype(str),
            "parent": parent_prediction(archive),
            "state": state,
            "season_n": season_n,
            "game_type": rows["game_type"].astype(str).to_numpy(),
        }
        state_metadata[str(year)] = metadata

    stacks: dict[int, np.ndarray] = {}
    fit_details: dict[str, Any] = {}
    artifacts: dict[str, str] = {}
    for year in YEARS:
        prediction, details = fit_target(frame, year, prereg, folds)
        stacks[year] = prediction
        fit_details[str(year)] = details
        path = PREDICTIONS / f"v5_temporal_state_offset_gate_source_{year}.npz"
        if path.exists():
            raise FileExistsError(f"immutable prediction artifact exists: {path}")
        np.savez_compressed(
            path,
            y=folds[year]["y"],
            row_index=folds[year]["row_index"],
            cluster=folds[year]["cluster"],
            exact_parent=folds[year]["parent"].astype(np.float32),
            adaptive_state=folds[year]["state"].astype(np.float32),
            temporal_state_offset_gate=prediction.astype(np.float32),
        )
        artifacts[str(year)] = str(path.relative_to(ROOT))

    baselines: dict[str, dict[int, np.ndarray]] = {
        "exact_parent_C": {year: folds[year]["parent"] for year in YEARS}
    }
    for name, paths in HONEST_ARTIFACTS.items():
        baselines[name] = {}
        for year, path in paths.items():
            archive = load_archive(path)
            if not np.array_equal(
                archive["row_index"].astype(np.int64), folds[year]["row_index"]
            ):
                raise ValueError(f"{name} {year}: row order mismatch")
            if not np.array_equal(archive["y"].astype(np.int8), folds[year]["y"]):
                raise ValueError(f"{name} {year}: target mismatch")
            baselines[name][year] = archive["final_prediction"].astype(np.float64)

    trials: list[dict[str, Any]] = []
    for gamma_value in prereg["development"]["blend_grid"]:
        gamma = float(gamma_value)
        cells: dict[str, Any] = {}
        full_gains: list[float] = []
        r_gains: list[float] = []
        for baseline_name, yearly in baselines.items():
            cells[baseline_name] = {}
            for year in YEARS:
                fold = folds[year]
                regular = fold["game_type"] == "R"
                base = yearly[year]
                candidate = base.copy()
                candidate[regular] = (
                    (1.0 - gamma) * base[regular] + gamma * stacks[year][regular]
                )
                candidate = np.clip(candidate, 1e-6, 1.0 - 1e-6)
                full = np.ones(len(candidate), dtype=bool)
                base_full = summarize(fold["y"], base, full)
                candidate_full = summarize(fold["y"], candidate, full)
                base_r = summarize(fold["y"], base, regular)
                candidate_r = summarize(fold["y"], candidate, regular)
                full_gain = candidate_full["score"] - base_full["score"]
                r_gain = candidate_r["score"] - base_r["score"]
                full_gains.append(full_gain)
                r_gains.append(r_gain)
                cells[baseline_name][str(year)] = {
                    "full": {
                        "baseline": base_full,
                        "candidate": candidate_full,
                        "gain": full_gain,
                    },
                    "R": {
                        "baseline": base_r,
                        "candidate": candidate_r,
                        "gain": r_gain,
                    },
                }
        trials.append({
            "gamma": gamma,
            "minimum_full_gain": float(min(full_gains)),
            "minimum_R_gain": float(min(r_gains)),
            "mean_full_gain": float(np.mean(full_gains)),
            "cells": cells,
        })
    selected = max(
        trials,
        key=lambda item: (
            item["minimum_full_gain"], item["minimum_R_gain"], -item["gamma"]
        ),
    )

    intervals: dict[str, Any] = {}
    gamma = float(selected["gamma"])
    for baseline_offset, (baseline_name, yearly) in enumerate(baselines.items()):
        intervals[baseline_name] = {}
        for year_offset, year in enumerate(YEARS):
            fold = folds[year]
            regular = fold["game_type"] == "R"
            base = yearly[year]
            candidate = base.copy()
            candidate[regular] = (
                (1.0 - gamma) * base[regular] + gamma * stacks[year][regular]
            )
            full = np.ones(len(candidate), dtype=bool)
            seed = 882600 + 100 * baseline_offset + 10 * year_offset
            intervals[baseline_name][str(year)] = {
                "R": cluster_bootstrap_score_gain(
                    fold["y"], base, candidate, fold["cluster"], regular,
                    1000, seed,
                ),
                "full": cluster_bootstrap_score_gain(
                    fold["y"], base, candidate, fold["cluster"], full,
                    1000, seed + 1,
                ),
            }

    threshold = float(
        prereg["development"]["advance_gate"]["minimum_development_full_gain"]
    )
    checks = {
        "minimum_full_gain_goal_sized": selected["minimum_full_gain"] >= threshold,
        "all_R_ci_low_positive": all(
            intervals[name][str(year)]["R"]["ci_low"] > 0.0
            for name in intervals for year in YEARS
        ),
        "all_full_ci_low_positive": all(
            intervals[name][str(year)]["full"]["ci_low"] > 0.0
            for name in intervals for year in YEARS
        ),
    }
    passed = bool(all(checks.values()))
    report = {
        "experiment_id": prereg["experiment_id"],
        "status": "development_pass" if passed else "development_failed",
        "preregister_sha256": sha256(PREREG),
        "script_sha256": sha256(Path(__file__)),
        "adaptive_state_implementation_sha256": sha256(
            ROOT / "experiments/v5_adaptive_state_space.py"
        ),
        "years_read": list(YEARS),
        "confirmation_2024_read": False,
        "fit_details": fit_details,
        "state_metadata": state_metadata,
        "trials": trials,
        "selected": selected,
        "intervals": intervals,
        "gate": {
            "requirements": prereg["development"]["advance_gate"],
            "checks": checks,
            "pass": passed,
            "decision": (
                "write confirmation lock before 2024"
                if passed
                else "close without fitting or reading a 2024 candidate"
            ),
        },
        "prediction_artifacts": artifacts,
        "goal_status": "active",
        "goal_completion_claimed": False,
    }
    REPORT.write_text(
        json.dumps(safe(report), ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(safe({
        "status": report["status"],
        "selected": selected,
        "intervals": intervals,
        "gate": report["gate"],
        "fit_details": fit_details,
    }), ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
