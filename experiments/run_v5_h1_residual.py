#!/usr/bin/env python3
"""Preregistered V5 H1 strictly-forward OOF residual experiment.

Selection mode reads only the 2020, 2021, and 2022 V3 OOF panels.  It learns
from 2020/2021 errors and selects one compact model plus one correction scale
on 2022 R rows.  Locked mode reads that immutable selection artifact, retrains
the same recipe for each forward fold, and audits 2023 before confirming 2024.

The correction uses only row-local official as-of values and constants frozen
before the row season.  It is fitted and applied only to regular-season rows;
F rows remain bit-for-bit equal to the V3 anchor.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time
from typing import Any

# HistGradientBoosting asks OpenMP for its effective thread count and joblib
# may otherwise try to create a Windows multiprocessing pipe in this managed
# workspace.  One thread changes runtime only, not the fitted recipe.
os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
PREDICTIONS = ROOT / "experiments/results/predictions"
PREREG = ROOT / "experiments/params/v5_h1_residual_preregister.json"
TRAIN = ROOT / "open/data/train.csv"
SELECTION_REPORT = ROOT / "experiments/results/v5_h1_residual_selection.json"
LOCKED_REPORT = ROOT / "experiments/results/v5_h1_residual_locked.json"
V3_ACTUAL_LB = 1090.9100565103
EARLY_WEIGHTS = {
    "A": 0.501443851662535,
    "C": 0.27016033407769313,
    "B": 0.22839581425977187,
}
RAW_COLUMNS = [
    "season",
    "game_month",
    "inning",
    "top_bottom",
    "game_type",
    "balls_before",
    "strikes_before",
    "outs_before",
    "score_diff_pitcher_team",
    "num_runners_on",
    "home_win_expectancy",
    "away_win_expectancy",
    "li",
    "pitcher_id",
    "batter_id",
    "pitcher_hand",
    "batter_hand",
    "asof_pitcher_n",
    "asof_pitcher_success_rate",
    "asof_pitcher_reverse_rate",
    "asof_pitcher_middle_rate",
    "asof_pitcher_ball_rate",
    "asof_pitcher_strike_rate",
    "asof_pitcher_prev1_game_success_rate",
    "asof_pitcher_prev3_game_success_rate",
    "asof_pitcher_prev5_game_success_rate",
    "asof_pitcher_prev1_game_middle_rate",
    "asof_pitcher_prev3_game_middle_rate",
    "asof_pitcher_prev5_game_middle_rate",
    "asof_batter_n",
    "asof_batter_success_rate",
    "asof_batter_middle_rate",
    "asof_pitcher_pitchmix_n",
    "asof_pitcher_fastball_rate",
    "asof_pitcher_breaking_rate",
    "asof_pitcher_offspeed_rate",
    "control_success",
]
CATEGORICAL_COLUMNS = [
    "game_month_cat",
    "inning_cat",
    "top_bottom_cat",
    "count_state_cat",
    "hand_matchup_cat",
]
FAMILY_ORDER = {
    "ridge_a1000": 0,
    "hgb_leaf15": 1,
    "catboost_d4": 2,
    "catboost_d5": 3,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("select", "locked"), required=True)
    parser.add_argument("--bootstrap", type=int, default=1000)
    parser.add_argument("--device", choices=("cpu", "gpu"), default="gpu")
    return parser.parse_args()


def load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        return {key: np.asarray(archive[key]) for key in archive.files}


def load_anchor(year: int) -> dict[str, np.ndarray]:
    if year >= 2022:
        return load_npz(PREDICTIONS / f"v3_sparse_m3_frozen_{year}.npz")
    artifacts = {
        name: load_npz(
            PREDICTIONS / f"v4_m3_{name.lower()}_backtest_{year}_{year}.npz"
        )
        for name in EARLY_WEIGHTS
    }
    reference = artifacts["A"]
    for name, artifact in artifacts.items():
        for key in ("y", "row_index", "cluster"):
            if not np.array_equal(reference[key], artifact[key]):
                raise ValueError(f"early V3 alignment mismatch: {year}/{name}/{key}")
    raw = sum(
        EARLY_WEIGHTS[name]
        * artifacts[name]["catboost_outcome"].astype(np.float64)
        for name in EARLY_WEIGHTS
    )
    prediction = np.clip(
        0.5 + 1.05 * (raw - 0.5) - 0.006, 1e-6, 1.0 - 1e-6
    )
    return {
        "y": reference["y"].astype(np.int8),
        "row_index": reference["row_index"].astype(np.int64),
        "cluster": reference["cluster"],
        "final_prediction": prediction,
    }


def panel_frame(artifacts: dict[int, dict[str, np.ndarray]]) -> pd.DataFrame:
    parts: list[pd.DataFrame] = []
    for year in sorted(artifacts):
        artifact = artifacts[year]
        parts.append(
            pd.DataFrame(
                {
                    "row_index": artifact["row_index"].astype(np.int64),
                    "year": year,
                    "y": artifact["y"].astype(np.int8),
                    "anchor": artifact["final_prediction"].astype(np.float64),
                    "cluster": artifact["cluster"].astype(str),
                }
            )
        )
    panel = pd.concat(parts, ignore_index=True)
    if panel["row_index"].duplicated().any():
        raise ValueError("OOF row indices are not unique")
    return panel


def build_hierarchical_component_features(
    frame: pd.DataFrame,
    states_before: dict[int, dict[int, tuple[int, ...]]],
    priors_before: dict[int, dict[str, float]],
    component_columns: dict[str, str],
    history_k: float = 200.0,
    current_k: float = 50.0,
) -> pd.DataFrame:
    """Independent beta-binomial posteriors for official failure counters."""
    seasons = frame["season"].to_numpy(dtype=np.int16, copy=False)
    pitchers = frame["pitcher_id"].to_numpy(dtype=np.int64, copy=False)
    n_end = np.zeros(len(frame), dtype=np.int64)
    component_end = np.zeros((len(frame), len(component_columns)), dtype=np.int64)
    unseen = np.ones(len(frame), dtype=np.int8)
    for index, (season, pitcher) in enumerate(zip(seasons, pitchers)):
        state = states_before.get(int(season), {}).get(int(pitcher))
        if state is not None:
            n_end[index] = state[0]
            component_end[index] = state[1:]
            unseen[index] = 0
    n_asof = frame["asof_pitcher_n"].fillna(0).to_numpy(dtype=np.int64)
    n_delta = n_asof - n_end
    values: dict[str, np.ndarray] = {
        "h1_component_log_n_season": np.log1p(np.maximum(n_delta, 0)).astype(
            np.float32
        ),
        "h1_component_unseen": unseen,
    }
    invalid_any = n_delta < 0
    for component_index, (name, column) in enumerate(component_columns.items()):
        league = np.asarray(
            [
                priors_before.get(int(season), {}).get(name, 0.5)
                for season in seasons
            ],
            dtype=np.float64,
        )
        history_rate = (
            component_end[:, component_index] + history_k * league
        ) / (n_end + history_k)
        career = pd.to_numeric(frame[column], errors="coerce").to_numpy(
            dtype=np.float64
        )
        career = np.where(np.isfinite(career), career, league)
        count_asof = np.rint(career * n_asof).astype(np.int64)
        count_delta = count_asof - component_end[:, component_index]
        invalid = (n_delta < 0) | (count_delta < 0) | (count_delta > n_delta)
        invalid_any |= invalid
        safe_n = np.where(invalid, 0, n_delta)
        safe_count = np.where(invalid, 0, count_delta)
        posterior = (safe_count + current_k * history_rate) / (
            safe_n + current_k
        )
        values[f"h1_{name}_history_rate"] = history_rate.astype(np.float32)
        values[f"h1_{name}_posterior_k50"] = posterior.astype(np.float32)
        values[f"h1_{name}_reliability_k50"] = (
            safe_n / (safe_n + current_k)
        ).astype(np.float32)
        values[f"h1_{name}_posterior_var_k50"] = (
            posterior * (1.0 - posterior) / (safe_n + current_k + 1.0)
        ).astype(np.float32)
    values["h1_component_invalid"] = invalid_any.astype(np.int8)
    return pd.DataFrame(values, index=frame.index)


def build_feature_frame(
    full_frame: pd.DataFrame,
    panel: pd.DataFrame,
    max_year: int,
) -> pd.DataFrame:
    # Importing these proven state builders keeps the exact legal cutoff used
    # by the prior rolling harness.  The source is truncated at max_year so
    # selection mode never even constructs state from 2023/2024 labels.
    from experiments.run_v2_rolling import (
        COMPONENT_RATE_COLUMNS,
        build_entity_season_features,
        build_hierarchical_entity_features,
        candidate_priors_before_each_season,
        component_states_before_each_season,
        entity_season_end_state,
    )

    source = full_frame.loc[full_frame["season"] <= max_year].copy()
    eval_rows = panel["row_index"].to_numpy(dtype=np.int64)
    frame = source.loc[eval_rows].copy()
    if not np.array_equal(frame.index.to_numpy(dtype=np.int64), eval_rows):
        raise ValueError("feature row order mismatch")
    if not np.array_equal(
        frame["control_success"].to_numpy(dtype=np.int8),
        panel["y"].to_numpy(dtype=np.int8),
    ):
        raise ValueError("feature target does not align with OOF artifacts")

    priors = candidate_priors_before_each_season(source, "r_recent3")
    fallback_prior = float(priors.get(max_year, 0.5))
    pitcher_before, _ = entity_season_end_state(
        source,
        "pitcher_id",
        "asof_pitcher_n",
        "asof_pitcher_success_rate",
    )
    batter_before, _ = entity_season_end_state(
        source,
        "batter_id",
        "asof_batter_n",
        "asof_batter_success_rate",
    )
    pitcher_hier, _ = build_hierarchical_entity_features(
        frame,
        pitcher_before,
        priors,
        fallback_prior,
        "pitcher_id",
        "asof_pitcher_n",
        "asof_pitcher_success_rate",
        "h1_pitcher",
        history_k=200.0,
        current_ks=(20.0, 50.0, 100.0, 200.0),
    )
    batter_hier, _ = build_hierarchical_entity_features(
        frame,
        batter_before,
        priors,
        fallback_prior,
        "batter_id",
        "asof_batter_n",
        "asof_batter_success_rate",
        "h1_batter",
        history_k=200.0,
        current_ks=(20.0, 50.0, 100.0, 200.0),
    )
    pitcher_season, _ = build_entity_season_features(
        frame,
        pitcher_before,
        priors,
        fallback_prior,
        "pitcher_id",
        "asof_pitcher_n",
        "asof_pitcher_success_rate",
        "h1_pitcher_season",
        50.0,
    )
    batter_season, _ = build_entity_season_features(
        frame,
        batter_before,
        priors,
        fallback_prior,
        "batter_id",
        "asof_batter_n",
        "asof_batter_success_rate",
        "h1_batter_season",
        50.0,
    )
    component_before, component_priors, _, _ = component_states_before_each_season(
        source
    )
    component_hier = build_hierarchical_component_features(
        frame,
        component_before,
        component_priors,
        COMPONENT_RATE_COLUMNS,
    )

    anchor = panel["anchor"].to_numpy(dtype=np.float64)
    clipped_anchor = np.clip(anchor, 1e-6, 1.0 - 1e-6)
    features = pd.DataFrame(index=frame.index)
    features["anchor"] = anchor.astype(np.float32)
    features["anchor_logit"] = np.log(
        clipped_anchor / (1.0 - clipped_anchor)
    ).astype(np.float32)
    features["game_month_cat"] = frame["game_month"].astype(str)
    features["inning_cat"] = frame["inning"].astype(str)
    features["top_bottom_cat"] = frame["top_bottom"].astype("string").fillna(
        "__missing__"
    )
    features["count_state_cat"] = (
        frame["balls_before"].astype(str)
        + "|"
        + frame["strikes_before"].astype(str)
    )
    features["hand_matchup_cat"] = (
        frame["pitcher_hand"].astype(str)
        + "|"
        + frame["batter_hand"].astype(str)
    )
    raw_numeric = [
        "balls_before",
        "strikes_before",
        "outs_before",
        "score_diff_pitcher_team",
        "num_runners_on",
        "home_win_expectancy",
        "away_win_expectancy",
        "li",
        "asof_pitcher_success_rate",
        "asof_pitcher_reverse_rate",
        "asof_pitcher_middle_rate",
        "asof_pitcher_ball_rate",
        "asof_pitcher_strike_rate",
        "asof_pitcher_prev1_game_success_rate",
        "asof_pitcher_prev3_game_success_rate",
        "asof_pitcher_prev5_game_success_rate",
        "asof_pitcher_prev1_game_middle_rate",
        "asof_pitcher_prev3_game_middle_rate",
        "asof_pitcher_prev5_game_middle_rate",
        "asof_batter_success_rate",
        "asof_batter_middle_rate",
        "asof_pitcher_fastball_rate",
        "asof_pitcher_breaking_rate",
        "asof_pitcher_offspeed_rate",
    ]
    for column in raw_numeric:
        features[column] = pd.to_numeric(frame[column], errors="coerce").astype(
            np.float32
        )
    for source_column, target_column in (
        ("asof_pitcher_n", "log_asof_pitcher_n"),
        ("asof_batter_n", "log_asof_batter_n"),
        ("asof_pitcher_pitchmix_n", "log_asof_pitchmix_n"),
    ):
        values = pd.to_numeric(frame[source_column], errors="coerce").fillna(0.0)
        features[target_column] = np.log1p(values).astype(np.float32)
    for derived in (
        pitcher_hier,
        batter_hier,
        pitcher_season,
        batter_season,
        component_hier,
    ):
        for column in derived.columns:
            features[column] = pd.to_numeric(
                derived[column], errors="coerce"
            ).astype(np.float32)
    for prefix, n_column in (
        ("h1_pitcher", "h1_pitcher_season_n_season"),
        ("h1_batter", "h1_batter_season_n_season"),
    ):
        posterior = features[f"{prefix}_posterior_k50"].to_numpy(dtype=np.float64)
        n = features[n_column].to_numpy(dtype=np.float64)
        features[f"{prefix}_posterior_var_k50"] = (
            posterior * (1.0 - posterior) / (n + 51.0)
        ).astype(np.float32)
    if list(features.columns[: len(CATEGORICAL_COLUMNS) + 2]) != [
        "anchor",
        "anchor_logit",
        *CATEGORICAL_COLUMNS,
    ]:
        raise AssertionError("unexpected feature order")
    return features


def make_training_target(
    panel: pd.DataFrame, training_mask: np.ndarray
) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    train = panel.loc[training_mask, ["year", "y", "anchor"]].copy()
    residual = train["y"].to_numpy(dtype=np.float64) - train["anchor"].to_numpy(
        dtype=np.float64
    )
    centers: dict[str, float] = {}
    for year in sorted(train["year"].unique()):
        mask = train["year"].to_numpy() == year
        center = float(residual[mask].mean())
        residual[mask] -= center
        centers[str(int(year))] = center
    counts = train["year"].value_counts().to_dict()
    weight = np.asarray(
        [1.0 / counts[int(year)] for year in train["year"]], dtype=np.float64
    )
    weight /= weight.mean()
    return residual, weight, centers


def fit_predict(
    config_name: str,
    config: dict[str, Any],
    train_x: pd.DataFrame,
    train_y: np.ndarray,
    train_weight: np.ndarray,
    valid_x: pd.DataFrame,
    device: str,
) -> tuple[np.ndarray, dict[str, Any]]:
    numeric_columns = [
        column for column in train_x.columns if column not in CATEGORICAL_COLUMNS
    ]
    family = config["family"]
    started = time.perf_counter()
    details: dict[str, Any] = {
        "family": family,
        "train_rows": int(len(train_x)),
        "feature_count": int(train_x.shape[1]),
    }
    if family == "ridge":
        preprocess = ColumnTransformer(
            [
                (
                    "numeric",
                    Pipeline(
                        [
                            ("impute", SimpleImputer(strategy="median")),
                            ("scale", StandardScaler()),
                        ]
                    ),
                    numeric_columns,
                ),
                (
                    "categorical",
                    OneHotEncoder(handle_unknown="ignore"),
                    CATEGORICAL_COLUMNS,
                ),
            ]
        )
        model = Pipeline(
            [
                ("preprocess", preprocess),
                (
                    "model",
                    Ridge(alpha=float(config["alpha"]), solver="lsqr"),
                ),
            ]
        )
        model.fit(train_x, train_y, model__sample_weight=train_weight)
        prediction = model.predict(valid_x)
    elif family == "hist_gradient_boosting":
        preprocess = ColumnTransformer(
            [
                (
                    "numeric",
                    SimpleImputer(strategy="median"),
                    numeric_columns,
                ),
                (
                    "categorical",
                    Pipeline(
                        [
                            (
                                "impute",
                                SimpleImputer(strategy="most_frequent"),
                            ),
                            (
                                "ordinal",
                                OrdinalEncoder(
                                    handle_unknown="use_encoded_value",
                                    unknown_value=-1,
                                ),
                            ),
                        ]
                    ),
                    CATEGORICAL_COLUMNS,
                ),
            ]
        )
        categorical_indices = list(
            range(len(numeric_columns), len(numeric_columns) + len(CATEGORICAL_COLUMNS))
        )
        estimator = HistGradientBoostingRegressor(
            loss=config["loss"],
            learning_rate=float(config["learning_rate"]),
            max_iter=int(config["max_iter"]),
            max_leaf_nodes=int(config["max_leaf_nodes"]),
            min_samples_leaf=int(config["min_samples_leaf"]),
            l2_regularization=float(config["l2_regularization"]),
            random_state=int(config["random_state"]),
            early_stopping=False,
            categorical_features=categorical_indices,
        )
        model = Pipeline([("preprocess", preprocess), ("model", estimator)])
        model.fit(train_x, train_y, model__sample_weight=train_weight)
        prediction = model.predict(valid_x)
    elif family == "catboost_regression":
        from catboost import CatBoostRegressor

        fit_x = train_x.copy()
        predict_x = valid_x.copy()
        for column in CATEGORICAL_COLUMNS:
            fit_x[column] = fit_x[column].astype("string").fillna("__missing__").astype(str)
            predict_x[column] = (
                predict_x[column].astype("string").fillna("__missing__").astype(str)
            )
        settings = {
            key: value
            for key, value in config.items()
            if key != "family"
        }
        settings.update(
            {
                "allow_writing_files": False,
                "thread_count": 6,
                "task_type": device.upper(),
                "verbose": False,
            }
        )
        model = CatBoostRegressor(**settings)
        model.fit(
            fit_x,
            train_y,
            cat_features=CATEGORICAL_COLUMNS,
            sample_weight=train_weight,
        )
        prediction = model.predict(predict_x)
        importance = model.get_feature_importance()
        details["feature_importance"] = [
            {"feature": feature, "importance": float(value)}
            for feature, value in sorted(
                zip(train_x.columns, importance),
                key=lambda item: item[1],
                reverse=True,
            )[:25]
        ]
    else:
        raise ValueError(f"unknown family: {family}")
    prediction = np.asarray(prediction, dtype=np.float64)
    details.update(
        {
            "fit_predict_seconds": time.perf_counter() - started,
            "correction_mean": float(prediction.mean()),
            "correction_std": float(prediction.std()),
            "correction_max_abs": float(np.max(np.abs(prediction))),
        }
    )
    return prediction, details


def raw_score(y: np.ndarray, prediction: np.ndarray, mask: np.ndarray) -> float:
    target = y[mask].astype(np.float64)
    pred = np.clip(prediction[mask].astype(np.float64), 1e-6, 1.0 - 1e-6)
    rate = float(target.mean())
    reference = rate * (1.0 - rate)
    brier = float(np.mean(np.square(pred - target)))
    return 100000.0 * (1.0 - brier / reference)


def metrics(
    y: np.ndarray,
    anchor: np.ndarray,
    prediction: np.ndarray,
    game_type: np.ndarray,
) -> dict[str, dict[str, float]]:
    result: dict[str, dict[str, float]] = {}
    for route, mask in (
        ("all", np.ones(len(y), dtype=bool)),
        ("R", game_type == "R"),
        ("F", game_type == "F"),
    ):
        anchor_score = raw_score(y, anchor, mask)
        candidate_score = raw_score(y, prediction, mask)
        result[route] = {
            "rows": int(mask.sum()),
            "target_rate": float(y[mask].mean()),
            "anchor_score": anchor_score,
            "candidate_score": candidate_score,
            "gain": candidate_score - anchor_score,
            "anchor_mean": float(anchor[mask].mean()),
            "candidate_mean": float(prediction[mask].mean()),
        }
    return result


def cluster_bootstrap_score_gain(
    y: np.ndarray,
    anchor: np.ndarray,
    prediction: np.ndarray,
    cluster: np.ndarray,
    mask: np.ndarray,
    iterations: int,
    seed: int,
) -> dict[str, float]:
    work = pd.DataFrame(
        {
            "cluster": cluster[mask].astype(str),
            "y": y[mask].astype(np.float64),
            "anchor_error": np.square(anchor[mask] - y[mask]),
            "candidate_error": np.square(prediction[mask] - y[mask]),
        }
    )
    grouped = work.groupby("cluster", sort=False, observed=True).agg(
        n=("y", "size"),
        y_sum=("y", "sum"),
        anchor_error=("anchor_error", "sum"),
        candidate_error=("candidate_error", "sum"),
    )
    values = grouped.to_numpy(dtype=np.float64)
    rng = np.random.default_rng(seed)
    gains = np.empty(iterations, dtype=np.float64)
    cluster_count = len(values)
    for iteration in range(iterations):
        sampled = values[rng.integers(0, cluster_count, size=cluster_count)].sum(axis=0)
        n, y_sum, anchor_error, candidate_error = sampled
        rate = y_sum / n
        reference = max(rate * (1.0 - rate), 1e-12)
        gains[iteration] = 100000.0 * (
            anchor_error / n - candidate_error / n
        ) / reference
    point = 100000.0 * (
        work["anchor_error"].mean() - work["candidate_error"].mean()
    ) / max(float(work["y"].mean() * (1.0 - work["y"].mean())), 1e-12)
    return {
        "point": float(point),
        "ci_low": float(np.quantile(gains, 0.025)),
        "ci_high": float(np.quantile(gains, 0.975)),
        "bootstrap_std": float(gains.std(ddof=1)),
        "iterations": int(iterations),
        "cluster_count": int(cluster_count),
    }


def apply_correction(
    anchor: np.ndarray,
    correction: np.ndarray,
    game_type: np.ndarray,
    scale: float,
) -> np.ndarray:
    prediction = anchor.astype(np.float64).copy()
    route = game_type == "R"
    prediction[route] += scale * correction[route]
    return np.clip(prediction, 1e-6, 1.0 - 1e-6)


def load_inputs(years: list[int]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    artifacts = {year: load_anchor(year) for year in years}
    panel = panel_frame(artifacts)
    max_year = max(years)
    full = pd.read_csv(TRAIN, usecols=RAW_COLUMNS)
    full.index = np.arange(len(full), dtype=np.int64)
    features = build_feature_frame(full, panel, max_year)
    aligned = full.loc[panel["row_index"].to_numpy(dtype=np.int64)]
    panel = panel.copy()
    panel["game_type"] = aligned["game_type"].astype(str).to_numpy()
    panel["pitcher_id"] = aligned["pitcher_id"].astype(str).to_numpy()
    if not np.array_equal(features.index.to_numpy(), panel["row_index"].to_numpy()):
        raise ValueError("final feature-panel alignment mismatch")
    return panel, features.reset_index(drop=True), full


def run_selection(args: argparse.Namespace, prereg: dict[str, Any]) -> None:
    if SELECTION_REPORT.exists():
        raise FileExistsError(
            f"Selection artifact already exists and is immutable: {SELECTION_REPORT}"
        )
    panel, features, _ = load_inputs([2020, 2021, 2022])
    train_mask = (panel["year"] < 2022).to_numpy() & panel["game_type"].eq("R").to_numpy()
    valid_mask = panel["year"].eq(2022).to_numpy()
    valid_r = valid_mask & panel["game_type"].eq("R").to_numpy()
    train_y, train_weight, centers = make_training_target(panel, train_mask)
    candidates: list[dict[str, Any]] = []
    correction_by_config: dict[str, np.ndarray] = {}
    detail_by_config: dict[str, dict[str, Any]] = {}
    for config_name, config in prereg["candidate_configs"].items():
        print(f"[select] fitting {config_name}", flush=True)
        correction, details = fit_predict(
            config_name,
            config,
            features.loc[train_mask],
            train_y,
            train_weight,
            features.loc[valid_mask],
            args.device,
        )
        correction_by_config[config_name] = correction
        detail_by_config[config_name] = details
        valid_panel = panel.loc[valid_mask].reset_index(drop=True)
        y = valid_panel["y"].to_numpy(dtype=np.int8)
        anchor = valid_panel["anchor"].to_numpy(dtype=np.float64)
        game_type = valid_panel["game_type"].to_numpy()
        cluster = valid_panel["cluster"].to_numpy()
        for scale in prereg["correction_scales"]:
            prediction = apply_correction(anchor, correction, game_type, float(scale))
            score_metrics = metrics(y, anchor, prediction, game_type)
            bootstrap = cluster_bootstrap_score_gain(
                y,
                anchor,
                prediction,
                cluster,
                game_type == "R",
                args.bootstrap,
                202600 + FAMILY_ORDER[config_name] * 10 + int(float(scale) * 4),
            )
            candidates.append(
                {
                    "config": config_name,
                    "scale": float(scale),
                    "metrics": score_metrics,
                    "bootstrap_R": bootstrap,
                    "eligible": bool(
                        score_metrics["R"]["gain"] > 0.0
                        and bootstrap["ci_low"] > 0.0
                    ),
                }
            )
    eligible = [candidate for candidate in candidates if candidate["eligible"]]
    selected = None
    if eligible:
        selected = sorted(
            eligible,
            key=lambda candidate: (
                -candidate["metrics"]["R"]["gain"],
                candidate["scale"],
                FAMILY_ORDER[candidate["config"]],
            ),
        )[0]
    prereg_hash = hashlib.sha256(PREREG.read_bytes()).hexdigest()
    report = {
        "experiment_id": prereg["experiment_id"],
        "mode": "selection",
        "preregister_sha256": prereg_hash,
        "years_read": [2020, 2021, 2022],
        "later_anchor_files_read": False,
        "training_rows_R": int(train_mask.sum()),
        "validation_rows_R": int(valid_r.sum()),
        "historical_residual_centers": centers,
        "feature_columns": list(features.columns),
        "candidate_model_details": detail_by_config,
        "candidates": candidates,
        "selected": selected,
        "status": "locked" if selected is not None else "failed_no_eligible_candidate",
    }
    SELECTION_REPORT.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    if selected is not None:
        valid_panel = panel.loc[valid_mask].reset_index(drop=True)
        correction = correction_by_config[selected["config"]]
        prediction = apply_correction(
            valid_panel["anchor"].to_numpy(dtype=np.float64),
            correction,
            valid_panel["game_type"].to_numpy(),
            float(selected["scale"]),
        )
        np.savez_compressed(
            PREDICTIONS / "v5_h1_residual_locked_2022.npz",
            y=valid_panel["y"].to_numpy(dtype=np.int8),
            row_index=valid_panel["row_index"].to_numpy(dtype=np.int64),
            cluster=valid_panel["cluster"].to_numpy(),
            anchor=valid_panel["anchor"].to_numpy(dtype=np.float64),
            correction=correction,
            final_prediction=prediction,
        )
    print(json.dumps({"status": report["status"], "selected": selected}, indent=2))
    print(f"Saved {SELECTION_REPORT}")


def run_locked(args: argparse.Namespace, prereg: dict[str, Any]) -> None:
    if LOCKED_REPORT.exists():
        raise FileExistsError(
            f"Locked report already exists and is immutable: {LOCKED_REPORT}"
        )
    selection = json.loads(SELECTION_REPORT.read_text(encoding="utf-8"))
    if selection.get("status") != "locked" or selection.get("selected") is None:
        raise ValueError("selection report has no eligible locked candidate")
    expected_hash = hashlib.sha256(PREREG.read_bytes()).hexdigest()
    if selection.get("preregister_sha256") != expected_hash:
        raise ValueError("preregister changed after selection")
    selected = selection["selected"]
    config_name = selected["config"]
    scale = float(selected["scale"])
    config = prereg["candidate_configs"][config_name]
    panel, features, _ = load_inputs([2020, 2021, 2022, 2023, 2024])
    folds: dict[str, Any] = {}
    for valid_year in (2022, 2023, 2024):
        train_mask = (panel["year"] < valid_year).to_numpy() & panel[
            "game_type"
        ].eq("R").to_numpy()
        valid_mask = panel["year"].eq(valid_year).to_numpy()
        train_y, train_weight, centers = make_training_target(panel, train_mask)
        print(
            f"[locked] {config_name} train<{valid_year} rows={train_mask.sum():,}",
            flush=True,
        )
        correction, model_details = fit_predict(
            config_name,
            config,
            features.loc[train_mask],
            train_y,
            train_weight,
            features.loc[valid_mask],
            args.device,
        )
        valid_panel = panel.loc[valid_mask].reset_index(drop=True)
        y = valid_panel["y"].to_numpy(dtype=np.int8)
        anchor = valid_panel["anchor"].to_numpy(dtype=np.float64)
        game_type = valid_panel["game_type"].to_numpy()
        cluster = valid_panel["cluster"].to_numpy()
        prediction = apply_correction(anchor, correction, game_type, scale)
        score_metrics = metrics(y, anchor, prediction, game_type)
        bootstrap_all = cluster_bootstrap_score_gain(
            y,
            anchor,
            prediction,
            cluster,
            np.ones(len(y), dtype=bool),
            args.bootstrap,
            202600 + valid_year,
        )
        bootstrap_r = cluster_bootstrap_score_gain(
            y,
            anchor,
            prediction,
            cluster,
            game_type == "R",
            args.bootstrap,
            303700 + valid_year,
        )
        folds[str(valid_year)] = {
            "training_years": sorted(
                int(value) for value in panel.loc[train_mask, "year"].unique()
            ),
            "training_rows_R": int(train_mask.sum()),
            "historical_residual_centers": centers,
            "model": model_details,
            "metrics": score_metrics,
            "bootstrap_all": bootstrap_all,
            "bootstrap_R": bootstrap_r,
        }
        np.savez_compressed(
            PREDICTIONS / f"v5_h1_residual_locked_{valid_year}.npz",
            y=y,
            row_index=valid_panel["row_index"].to_numpy(dtype=np.int64),
            cluster=cluster,
            anchor=anchor,
            correction=correction,
            final_prediction=prediction,
        )
    development_gain = float(
        np.median(
            [
                folds["2022"]["metrics"]["all"]["gain"],
                folds["2023"]["metrics"]["all"]["gain"],
            ]
        )
    )
    confirmation_gain = float(folds["2024"]["metrics"]["all"]["gain"])
    confirmation_ci_low = float(folds["2024"]["bootstrap_all"]["ci_low"])
    robust_gain = min(development_gain, confirmation_gain, confirmation_ci_low)
    expected_lb_lower = V3_ACTUAL_LB + 0.75 * max(0.0, robust_gain)
    report = {
        "experiment_id": prereg["experiment_id"],
        "mode": "locked_transfer_and_confirmation",
        "selection_report": str(SELECTION_REPORT.relative_to(ROOT)),
        "preregister_sha256": expected_hash,
        "locked_config": config_name,
        "locked_scale": scale,
        "route": "R_only_F_unchanged",
        "feature_columns": list(features.columns),
        "folds": folds,
        "conservative_expected_score": {
            "actual_v3_anchor": V3_ACTUAL_LB,
            "G_dev_full": development_gain,
            "G_confirm_full": confirmation_gain,
            "G_ci_full": confirmation_ci_low,
            "G_robust": robust_gain,
            "haircut": 0.75,
            "expected_lb_lower": expected_lb_lower,
            "passes_1190": bool(expected_lb_lower > 1190.0),
        },
    }
    LOCKED_REPORT.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "locked_config": config_name,
                "locked_scale": scale,
                "gains": {
                    year: folds[year]["metrics"]["all"]["gain"]
                    for year in ("2022", "2023", "2024")
                },
                "conservative_expected_score": report[
                    "conservative_expected_score"
                ],
            },
            indent=2,
        )
    )
    print(f"Saved {LOCKED_REPORT}")


def main() -> None:
    args = parse_args()
    prereg = json.loads(PREREG.read_text(encoding="utf-8"))
    if args.mode == "select":
        run_selection(args, prereg)
    else:
        run_locked(args, prereg)


if __name__ == "__main__":
    main()
