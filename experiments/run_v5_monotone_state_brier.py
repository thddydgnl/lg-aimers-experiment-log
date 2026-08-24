#!/usr/bin/env python3
"""Run the preregistered compact monotone direct-Brier state ablation."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys
import time
from typing import Any

# Load LightGBM before pandas/sklearn on Windows.  Loading its native DLL after
# the large pandas frame has caused intermittent access violations here.
from lightgbm import LGBMRegressor
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.analyze_v5_hgb_state_context import (  # noqa: E402
    ensure_aligned,
    load_npz,
    score_gain_interval,
)
from experiments.run_baselines import FEATURES as BASE_FEATURES, TARGET  # noqa: E402
from experiments.run_e14_rolling import season_end_state  # noqa: E402
from experiments.run_v2_rolling import (  # noqa: E402
    build_hierarchical_entity_features,
    entity_season_end_state,
)


PREREG = ROOT / "experiments/params/v5_monotone_state_brier_preregister.json"
OUTPUT = ROOT / "experiments/results/v5_monotone_state_brier_selection.json"
PREDICTIONS = ROOT / "experiments/results/predictions"
YEARS = (2022, 2023)
RANDOM_SEED = 20260821

BASE_DROP = {"pitcher_id", "batter_id"}
CATEGORICAL = (
    "game_month",
    "game_dayofweek",
    "inning",
    "top_bottom",
    "game_type",
    "balls_before",
    "strikes_before",
    "outs_before",
    "base_state",
    "num_runners_on",
    "pitcher_hand",
    "batter_hand",
    "pitcher_team_id",
    "batter_team_id",
)
MONOTONE_POSITIVE = {
    "asof_pitcher_success_rate",
    "state_pitcher_history",
    "state_pitcher_posterior",
    "state_pitcher_delta",
    "state_batter_history",
    "state_batter_posterior",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def recent_r_priors(frame: pd.DataFrame) -> tuple[dict[int, float], float]:
    seasons = sorted(int(value) for value in frame["season"].unique())
    priors: dict[int, float] = {}
    for season in seasons:
        past = frame.loc[
            frame["game_type"].eq("R")
            & frame["season"].lt(season)
            & frame["season"].ge(season - 3),
            TARGET,
        ]
        priors[season] = float(past.mean()) if len(past) else 0.5
    latest = seasons[-1] + 1
    completed = frame.loc[
        frame["game_type"].eq("R") & frame["season"].ge(latest - 3), TARGET
    ]
    return priors, float(completed.mean()) if len(completed) else 0.5


def state_block(
    frame: pd.DataFrame,
    pitcher_states: dict[int, dict[int, tuple[int, int]]],
    batter_states: dict[int, dict[int, tuple[int, int]]],
    priors: dict[int, float],
    fallback_prior: float,
) -> pd.DataFrame:
    pitcher, _ = build_hierarchical_entity_features(
        frame,
        pitcher_states,
        priors,
        fallback_prior,
        "pitcher_id",
        "asof_pitcher_n",
        "asof_pitcher_success_rate",
        "state_pitcher_raw",
        history_k=200.0,
        current_ks=(50.0,),
    )
    batter, _ = build_hierarchical_entity_features(
        frame,
        batter_states,
        priors,
        fallback_prior,
        "batter_id",
        "asof_batter_n",
        "asof_batter_success_rate",
        "state_batter_raw",
        history_k=200.0,
        current_ks=(100.0,),
    )
    p_history = pitcher["state_pitcher_raw_history_rate"].to_numpy(dtype=np.float64)
    p_posterior = pitcher["state_pitcher_raw_posterior_k50"].to_numpy(dtype=np.float64)
    p_reliability = pitcher["state_pitcher_raw_reliability_k50"].to_numpy(dtype=np.float64)
    p_raw = pitcher["state_pitcher_raw_raw_season_rate"].to_numpy(dtype=np.float64)
    b_history = batter["state_batter_raw_history_rate"].to_numpy(dtype=np.float64)
    b_posterior = batter["state_batter_raw_posterior_k100"].to_numpy(dtype=np.float64)
    b_reliability = batter["state_batter_raw_reliability_k100"].to_numpy(dtype=np.float64)
    balls = pd.to_numeric(frame["balls_before"], errors="coerce").fillna(0).to_numpy(
        dtype=np.float64
    )
    strikes = pd.to_numeric(frame["strikes_before"], errors="coerce").fillna(0).to_numpy(
        dtype=np.float64
    )
    same_hand = (
        pd.to_numeric(frame["pitcher_hand"], errors="coerce").to_numpy()
        == pd.to_numeric(frame["batter_hand"], errors="coerce").to_numpy()
    ).astype(np.float64)
    runner = (
        pd.to_numeric(frame["num_runners_on"], errors="coerce").fillna(0).to_numpy()
        > 0
    ).astype(np.float64)
    p_n = 50.0 * p_reliability / np.maximum(1.0 - p_reliability, 1e-9)
    b_n = 100.0 * b_reliability / np.maximum(1.0 - b_reliability, 1e-9)
    return pd.DataFrame(
        {
            "state_pitcher_history": p_history,
            "state_pitcher_posterior": p_posterior,
            "state_pitcher_reliability": p_reliability,
            "state_pitcher_delta": p_posterior - p_history,
            "state_pitcher_raw_rate": p_raw,
            "state_batter_history": b_history,
            "state_batter_posterior": b_posterior,
            "state_batter_reliability": b_reliability,
            "state_batter_delta": b_posterior - b_history,
            "state_pitcher_batter_gap": p_posterior - b_posterior,
            "state_pitcher_x_count_advantage": p_posterior * (strikes > balls),
            "state_pitcher_x_ball_strike_gap": p_posterior * (balls - strikes),
            "state_pitcher_x_same_hand": p_posterior * same_hand,
            "state_pitcher_x_runner": p_posterior * runner,
            "state_pitcher_uncertainty": np.sqrt(
                np.clip(p_posterior * (1.0 - p_posterior) / (p_n + 51.0), 0.0, None)
            ),
            "state_batter_uncertainty": np.sqrt(
                np.clip(b_posterior * (1.0 - b_posterior) / (b_n + 101.0), 0.0, None)
            ),
        },
        index=frame.index,
        dtype=np.float32,
    )


def encode_frames(
    train: pd.DataFrame, valid: pd.DataFrame, features: list[str]
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    train_x = train[features].copy()
    valid_x = valid[features].copy()
    categorical = [name for name in CATEGORICAL if name in features]
    for column in features:
        if column in categorical:
            source = train_x[column].astype("string").fillna("__missing__")
            categories = sorted(str(value) for value in source.unique())
            mapping = {value: index for index, value in enumerate(categories)}
            train_x[column] = source.map(mapping).fillna(-1).astype(np.int32)
            valid_x[column] = (
                valid_x[column]
                .astype("string")
                .fillna("__missing__")
                .map(mapping)
                .fillna(-1)
                .astype(np.int32)
            )
        else:
            train_x[column] = pd.to_numeric(train_x[column], errors="coerce").astype(
                np.float32
            )
            valid_x[column] = pd.to_numeric(valid_x[column], errors="coerce").astype(
                np.float32
            )
    return train_x, valid_x, categorical


def fit_model(
    train: pd.DataFrame,
    valid: pd.DataFrame,
    features: list[str],
    target: np.ndarray,
    sample_weight: np.ndarray,
) -> tuple[np.ndarray, dict[str, Any]]:
    train_x, valid_x, categorical = encode_frames(train, valid, features)
    constraints = [1 if name in MONOTONE_POSITIVE else 0 for name in features]
    model = LGBMRegressor(
        objective="regression_l2",
        learning_rate=0.02,
        n_estimators=1200,
        num_leaves=15,
        min_child_samples=1000,
        reg_lambda=20.0,
        colsample_bytree=1.0,
        subsample=1.0,
        random_state=RANDOM_SEED,
        n_jobs=6,
        deterministic=True,
        force_col_wise=True,
        verbosity=-1,
        monotone_constraints=constraints,
        monotone_constraints_method="advanced",
    )
    started = time.perf_counter()
    model.fit(
        train_x,
        np.asarray(target, dtype=np.float64),
        sample_weight=sample_weight,
        categorical_feature=categorical,
    )
    prediction = np.clip(model.predict(valid_x), 0.001, 0.999).astype(np.float64)
    importance = sorted(
        zip(features, model.feature_importances_.astype(float)),
        key=lambda item: item[1],
        reverse=True,
    )[:25]
    return prediction, {
        "fit_seconds": time.perf_counter() - started,
        "feature_count": len(features),
        "categorical": categorical,
        "monotone_positive": [
            name for name, value in zip(features, constraints) if value > 0
        ],
        "top_split_importance": [
            {"feature": name, "importance": value} for name, value in importance
        ],
    }


def main() -> None:
    prereg = json.loads(PREREG.read_text(encoding="utf-8"))
    if prereg["status"] != "locked_before_execution":
        raise ValueError("Preregister status changed")
    full = pd.read_csv(
        ROOT / "open/data/train.csv", encoding="utf-8-sig", low_memory=False
    )
    parent_features = [name for name in BASE_FEATURES if name not in BASE_DROP]
    folds: dict[str, Any] = {}
    for year in YEARS:
        history_all = full.loc[full["season"].lt(year)].copy()
        valid = full.loc[full["season"].eq(year)].copy()
        fit_mask = history_all["game_type"].eq("R")
        history = history_all.loc[fit_mask].copy()
        priors, target_prior = recent_r_priors(history_all)

        pitcher_before, pitcher_final = season_end_state(history_all)
        batter_before, batter_final = entity_season_end_state(
            history_all,
            "batter_id",
            "asof_batter_n",
            "asof_batter_success_rate",
        )
        state_train_all = state_block(
            history_all, pitcher_before, batter_before, priors, target_prior
        )
        state_valid = state_block(
            valid,
            {year: pitcher_final},
            {year: batter_final},
            {year: target_prior},
            target_prior,
        )
        state_train = state_train_all.loc[history.index]
        candidate_features = [*parent_features, *state_train.columns.tolist()]
        parent_train = history[parent_features].copy()
        parent_valid = valid[parent_features].copy()
        candidate_train = pd.concat([parent_train, state_train], axis=1)
        candidate_valid = pd.concat([parent_valid, state_valid], axis=1)
        max_history_year = int(history["season"].max())
        sample_weight = np.power(
            0.8,
            max_history_year - history["season"].to_numpy(dtype=np.int16),
        ).astype(np.float64)

        parent_prediction, parent_meta = fit_model(
            parent_train,
            parent_valid,
            parent_features,
            history[TARGET].to_numpy(dtype=np.float64),
            sample_weight,
        )
        candidate_prediction, candidate_meta = fit_model(
            candidate_train,
            candidate_valid,
            candidate_features,
            history[TARGET].to_numpy(dtype=np.float64),
            sample_weight,
        )
        row_index = valid.index.to_numpy(dtype=np.int64)
        y = valid[TARGET].to_numpy(dtype=np.int8)
        cluster = valid["pitcher_id"].astype(str).to_numpy()
        r_mask = valid["game_type"].eq("R").to_numpy(dtype=bool)
        artifact_path = PREDICTIONS / f"v5_monotone_state_brier_v1_dev2223_{year}.npz"
        np.savez_compressed(
            artifact_path,
            y=y,
            row_index=row_index,
            cluster=cluster,
            parent=parent_prediction,
            candidate=candidate_prediction,
        )
        artifact = {
            "y": y,
            "row_index": row_index,
            "cluster": cluster,
            "parent": parent_prediction,
            "candidate": candidate_prediction,
        }
        exact_c = load_npz(PREDICTIONS / f"v3_sparse_c_backtest_{year}.npz")
        identity = load_npz(PREDICTIONS / f"v5_honest_m3_r_identity_{year}.npz")
        grid = load_npz(PREDICTIONS / f"v5_honest_m3_r_grid_{year}.npz")
        for label, reference in (
            ("exact_c", exact_c),
            ("honest_identity", identity),
            ("honest_grid", grid),
        ):
            ensure_aligned(artifact, reference, f"{label}/{year}")
        comparisons = {
            "vs_exact_brier_parent_r": score_gain_interval(
                y, parent_prediction, candidate_prediction, cluster, r_mask,
                seed=20260821 + year,
            ),
            "vs_exact_c_r": score_gain_interval(
                y, exact_c["catboost_outcome"], candidate_prediction, cluster, r_mask,
                seed=20261821 + year,
            ),
            "vs_honest_identity_r": score_gain_interval(
                y, identity["final_prediction"], candidate_prediction, cluster, r_mask,
                seed=20262821 + year,
            ),
            "vs_honest_grid_r": score_gain_interval(
                y, grid["final_prediction"], candidate_prediction, cluster, r_mask,
                seed=20263821 + year,
            ),
        }
        folds[str(year)] = {
            "history_rows_all": int(len(history_all)),
            "fit_rows_r": int(len(history)),
            "valid_rows_r": int(r_mask.sum()),
            "target_prior_from_history": target_prior,
            "sample_weight_min": float(sample_weight.min()),
            "parent": parent_meta,
            "candidate": candidate_meta,
            "comparisons": comparisons,
            "prediction_artifact": str(artifact_path.relative_to(ROOT)),
        }
        print(
            f"{year}: candidate R={comparisons['vs_exact_brier_parent_r']['candidate_score']:.3f} "
            f"parent_gain={comparisons['vs_exact_brier_parent_r']['point_gain']:+.3f}",
            flush=True,
        )

    parent_point = all(
        folds[str(year)]["comparisons"]["vs_exact_brier_parent_r"]["point_gain"] > 0.0
        for year in YEARS
    )
    parent_lower = all(
        folds[str(year)]["comparisons"]["vs_exact_brier_parent_r"]["lower_95"] > 0.0
        for year in YEARS
    )
    anchors_point = all(
        folds[str(year)]["comparisons"][name]["point_gain"] > 0.0
        for year in YEARS
        for name in ("vs_exact_c_r", "vs_honest_identity_r", "vs_honest_grid_r")
    )
    passed = parent_point and parent_lower and anchors_point
    payload = {
        "experiment_id": prereg["experiment_id"],
        "status": "passed_lock_before_2024" if passed else "failed_no_2024_run",
        "protocol": {
            "development_years": list(YEARS),
            "2024_candidate_run": False,
            "test_rows_read": False,
            "regular_season_primary_only": True,
            "same_target_fold_calibration": False,
        },
        "preregister": str(PREREG.relative_to(ROOT)),
        "preregister_sha256": sha256(PREREG),
        "folds": folds,
        "gate": {
            "positive_exact_parent_point_both": bool(parent_point),
            "positive_exact_parent_lower_both": bool(parent_lower),
            "positive_exact_c_and_honest_anchor_points_both": bool(anchors_point),
            "passed": bool(passed),
        },
        "next_action": (
            "Freeze this exact recipe and run 2024 once."
            if passed else "Reject without running 2024."
        ),
    }
    OUTPUT.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload["gate"], ensure_ascii=False, indent=2), flush=True)
    for year in YEARS:
        print(f"\n{year}", flush=True)
        for name, result in folds[str(year)]["comparisons"].items():
            print(
                f"  {name}: candidate={result['candidate_score']:.3f} "
                f"gain={result['point_gain']:+.3f} "
                f"CI=[{result['lower_95']:+.3f}, {result['upper_95']:+.3f}]",
                flush=True,
            )
    print(f"wrote {OUTPUT}", flush=True)


if __name__ == "__main__":
    main()
