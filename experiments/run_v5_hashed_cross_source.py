#!/usr/bin/env python3
"""Train immutable 2020/2021 signed-hashed R-only logistic source artifacts."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys
import time
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.feature_extraction import FeatureHasher
from sklearn.linear_model import SGDClassifier


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.analyze_v5_dense_pitchtype_moe import load, safe  # noqa: E402


YEARS = (2020, 2021)
TRAIN = ROOT / "open/data/train.csv"
PRED = ROOT / "experiments/results/predictions"
PREREG = ROOT / "experiments/params/v5_hashed_cross_preregister.json"
REPORT = ROOT / "experiments/results/v5_hashed_cross_source_training.json"
DIMENSIONS = (2**18, 2**20)
ALPHAS = (3e-5, 1e-4, 3e-4)
SEED = 2026

INPUT_COLUMNS = [
    "season", "game_type", "game_month", "game_dayofweek", "inning",
    "top_bottom", "balls_before", "strikes_before", "outs_before",
    "run_total_before", "score_diff_pitcher_team", "base_state",
    "num_runners_on", "home_win_expectancy", "li", "pitcher_id",
    "batter_id", "pitcher_hand", "batter_hand", "pitcher_team_id",
    "batter_team_id", "asof_pitcher_n", "asof_pitcher_success_rate",
    "asof_pitcher_reverse_rate", "asof_pitcher_middle_rate",
    "asof_pitcher_ball_rate", "asof_pitcher_strike_rate",
    "asof_pitcher_prev1_game_success_rate",
    "asof_pitcher_prev3_game_success_rate",
    "asof_pitcher_prev5_game_success_rate",
    "asof_pitcher_prev1_game_middle_rate",
    "asof_pitcher_prev3_game_middle_rate",
    "asof_pitcher_prev5_game_middle_rate", "asof_batter_n",
    "asof_batter_success_rate", "asof_batter_middle_rate",
    "asof_pitcher_pitchmix_n", "asof_pitcher_fastball_rate",
    "asof_pitcher_breaking_rate", "asof_pitcher_offspeed_rate",
    "control_success",
]


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            value.update(chunk)
    return value.hexdigest()


def as_text(frame: pd.DataFrame, column: str) -> np.ndarray:
    return frame[column].fillna("__NA__").astype(str).to_numpy(dtype=str)


def fixed_bin(
    frame: pd.DataFrame,
    column: str,
    step: float,
    lower: float,
    upper: float,
) -> np.ndarray:
    values = pd.to_numeric(frame[column], errors="coerce").to_numpy(dtype=np.float64)
    missing = ~np.isfinite(values)
    values[missing] = lower
    values = np.clip(values, lower, upper)
    result = np.floor((values - lower) / step + 1e-9).astype(np.int32)
    result[missing] = -1
    return result.astype(str)


def log_count_bin(frame: pd.DataFrame, column: str) -> np.ndarray:
    values = pd.to_numeric(frame[column], errors="coerce").to_numpy(dtype=np.float64)
    missing = ~np.isfinite(values)
    values = np.maximum(np.nan_to_num(values, nan=0.0), 0.0)
    result = np.floor(np.log2(values + 1.0)).astype(np.int16)
    result[missing] = -1
    return result.astype(str)


def token_rows(frame: pd.DataFrame) -> Iterable[list[str]]:
    p = as_text(frame, "pitcher_id")
    b = as_text(frame, "batter_id")
    pt = as_text(frame, "pitcher_team_id")
    bt = as_text(frame, "batter_team_id")
    ph = as_text(frame, "pitcher_hand")
    bh = as_text(frame, "batter_hand")
    month = as_text(frame, "game_month")
    dow = as_text(frame, "game_dayofweek")
    top = as_text(frame, "top_bottom")
    balls = as_text(frame, "balls_before")
    strikes = as_text(frame, "strikes_before")
    outs = as_text(frame, "outs_before")
    base = as_text(frame, "base_state")
    runners = as_text(frame, "num_runners_on")
    innings = pd.to_numeric(frame["inning"], errors="coerce").fillna(-1).to_numpy()
    inning_band = np.where(innings < 0, -1, np.minimum((innings - 1) // 3, 4)).astype(str)
    count = np.char.add(np.char.add(balls, "-"), strikes)
    hand = np.char.add(np.char.add(ph, "-"), bh)

    score = fixed_bin(frame, "score_diff_pitcher_team", 1.0, -8.0, 8.0)
    runs = fixed_bin(frame, "run_total_before", 2.0, 0.0, 20.0)
    win = fixed_bin(frame, "home_win_expectancy", 5.0, 0.0, 100.0)
    leverage = fixed_bin(frame, "li", 0.5, 0.0, 10.0)

    pn = log_count_bin(frame, "asof_pitcher_n")
    bn = log_count_bin(frame, "asof_batter_n")
    mixn = log_count_bin(frame, "asof_pitcher_pitchmix_n")
    rate_columns = [
        "asof_pitcher_success_rate", "asof_pitcher_reverse_rate",
        "asof_pitcher_middle_rate", "asof_pitcher_ball_rate",
        "asof_pitcher_strike_rate", "asof_pitcher_prev1_game_success_rate",
        "asof_pitcher_prev3_game_success_rate",
        "asof_pitcher_prev5_game_success_rate",
        "asof_pitcher_prev1_game_middle_rate",
        "asof_pitcher_prev3_game_middle_rate",
        "asof_pitcher_prev5_game_middle_rate", "asof_batter_success_rate",
        "asof_batter_middle_rate", "asof_pitcher_fastball_rate",
        "asof_pitcher_breaking_rate", "asof_pitcher_offspeed_rate",
    ]
    rate_bins = {
        column: fixed_bin(
            frame,
            column,
            0.025 if column in {
                "asof_pitcher_success_rate", "asof_pitcher_reverse_rate",
                "asof_pitcher_middle_rate", "asof_pitcher_ball_rate",
                "asof_pitcher_strike_rate",
            } else 0.05,
            0.0,
            1.0,
        )
        for column in rate_columns
    }

    for index in range(len(frame)):
        tokens = [
            "bias", f"p={p[index]}", f"b={b[index]}",
            f"pt={pt[index]}", f"bt={bt[index]}", f"ph={ph[index]}",
            f"bh={bh[index]}", f"hand={hand[index]}",
            f"month={month[index]}", f"dow={dow[index]}",
            f"ib={inning_band[index]}", f"top={top[index]}",
            f"count={count[index]}", f"outs={outs[index]}",
            f"base={base[index]}", f"runners={runners[index]}",
            f"score={score[index]}", f"runs={runs[index]}",
            f"win={win[index]}", f"li={leverage[index]}",
            f"pn={pn[index]}", f"bn={bn[index]}", f"mixn={mixn[index]}",
        ]
        for column in rate_columns:
            tokens.append(f"{column}={rate_bins[column][index]}")
        tokens.extend([
            f"p|bh={p[index]}|{bh[index]}",
            f"p|count={p[index]}|{count[index]}",
            f"p|count|bh={p[index]}|{count[index]}|{bh[index]}",
            f"p|base={p[index]}|{base[index]}",
            f"p|ib={p[index]}|{inning_band[index]}",
            f"p|bt={p[index]}|{bt[index]}",
            f"p|pn={p[index]}|{pn[index]}",
            f"b|ph={b[index]}|{ph[index]}",
            f"b|count={b[index]}|{count[index]}",
            f"b|count|ph={b[index]}|{count[index]}|{ph[index]}",
            f"b|pt={b[index]}|{pt[index]}",
            f"b|bn={b[index]}|{bn[index]}",
            f"p|b={p[index]}|{b[index]}",
            f"pt|bt={pt[index]}|{bt[index]}",
            f"hand|count={hand[index]}|{count[index]}",
            f"count|base|outs={count[index]}|{base[index]}|{outs[index]}",
            f"count|ib={count[index]}|{inning_band[index]}",
            f"pt|count={pt[index]}|{count[index]}",
            f"bt|count={bt[index]}|{count[index]}",
            f"pn|ps={pn[index]}|{rate_bins['asof_pitcher_success_rate'][index]}",
            f"bn|bs={bn[index]}|{rate_bins['asof_batter_success_rate'][index]}",
        ])
        yield tokens


def config_key(dimension: int, alpha: float) -> str:
    exponent = int(round(-np.log10(alpha)))
    mantissa = int(round(alpha * (10**exponent)))
    return f"d{int(np.log2(dimension))}_a{mantissa}e{exponent}"


def main() -> None:
    if REPORT.exists():
        raise FileExistsError(f"immutable training report exists: {REPORT}")
    prereg = json.loads(PREREG.read_text(encoding="utf-8"))
    if prereg["status"] != "locked_before_source_metrics":
        raise ValueError("unexpected preregistration status")
    hashing = prereg["candidate_family"]["hashing"]
    model_contract = prereg["candidate_family"]["model"]
    if tuple(hashing["dimensions"]) != DIMENSIONS:
        raise ValueError("hash dimension contract changed")
    if tuple(float(value) for value in model_contract["alphas"]) != ALPHAS:
        raise ValueError("alpha contract changed")

    parents = {
        year: load(PRED / f"v4_m3_c_backtest_{year}_{year}.npz") for year in YEARS
    }
    maximum_row = max(int(parents[year]["row_index"].max()) for year in YEARS)
    frame = pd.read_csv(TRAIN, usecols=INPUT_COLUMNS, nrows=maximum_row + 1)
    if int(frame["season"].max()) != 2021:
        raise ValueError("source reader crossed the locked 2021 boundary")

    report: dict[str, object] = {
        "experiment_id": prereg["experiment_id"],
        "status": "source_predictions_materialized_no_metrics_read",
        "preregister_sha256": digest(PREREG),
        "script_sha256": digest(Path(__file__)),
        "years_read": list(YEARS),
        "years_not_read": [2022, 2023, 2024],
        "folds": {},
        "target_metrics_computed": False,
    }
    for year in YEARS:
        output = PRED / f"v5_hashed_cross_source_{year}.npz"
        if output.exists():
            raise FileExistsError(f"immutable source artifact exists: {output}")
        parent = parents[year]
        validation_index = parent["row_index"].astype(np.int64)
        history_mask = (frame["season"].to_numpy() < year) & frame["game_type"].eq("R").to_numpy()
        valid = frame.loc[validation_index]
        if not valid["season"].eq(year).all():
            raise ValueError(f"{year}: validation season mismatch")
        if not np.array_equal(
            valid["control_success"].to_numpy(dtype=np.int8),
            parent["y"].astype(np.int8),
        ):
            raise ValueError(f"{year}: target alignment mismatch")
        fit = frame.loc[history_mask]
        fit_y = fit["control_success"].to_numpy(dtype=np.int8)
        artifact: dict[str, np.ndarray] = {
            "y": parent["y"],
            "row_index": parent["row_index"],
            "cluster": parent["cluster"],
        }
        fold_details: dict[str, object] = {
            "fit_rows": int(len(fit)),
            "valid_rows": int(len(valid)),
            "fit_seasons": sorted(int(value) for value in fit["season"].unique()),
            "configs": {},
        }
        for dimension in DIMENSIONS:
            started = time.perf_counter()
            hasher = FeatureHasher(
                n_features=dimension,
                input_type="string",
                alternate_sign=True,
                dtype=np.float32,
            )
            fit_x = hasher.transform(token_rows(fit))
            valid_x = hasher.transform(token_rows(valid))
            matrix_seconds = time.perf_counter() - started
            for alpha in ALPHAS:
                key = config_key(dimension, alpha)
                model = SGDClassifier(
                    loss="log_loss",
                    penalty="l2",
                    alpha=alpha,
                    max_iter=int(model_contract["max_iter"]),
                    tol=float(model_contract["tol"]),
                    average=bool(model_contract["average"]),
                    random_state=SEED,
                    n_jobs=-1,
                )
                fit_started = time.perf_counter()
                model.fit(fit_x, fit_y)
                prediction = model.predict_proba(valid_x)[:, 1].astype(np.float32)
                fit_seconds = time.perf_counter() - fit_started
                reversed_prediction = model.predict_proba(valid_x[::-1])[:, 1][::-1]
                invariance_error = float(
                    np.max(np.abs(prediction.astype(np.float64) - reversed_prediction))
                )
                if invariance_error > 1e-7:
                    raise ValueError(f"{year}/{key}: row-order invariance failed")
                artifact[key] = prediction
                fold_details["configs"][key] = {
                    "dimension": dimension,
                    "alpha": alpha,
                    "matrix_seconds": matrix_seconds,
                    "fit_predict_seconds": fit_seconds,
                    "n_iter": int(model.n_iter_),
                    "coefficient_l2": float(np.linalg.norm(model.coef_)),
                    "prediction_min": float(prediction.min()),
                    "prediction_max": float(prediction.max()),
                    "prediction_mean": float(prediction.mean()),
                    "row_order_invariance_max_abs": invariance_error,
                }
            del fit_x, valid_x
        np.savez_compressed(output, **artifact)
        fold_details["artifact"] = str(output.relative_to(ROOT))
        fold_details["artifact_sha256"] = digest(output)
        report["folds"][str(year)] = fold_details
        print(
            json.dumps(
                safe({"year": year, "artifact": fold_details["artifact"], "details": fold_details}),
                ensure_ascii=False,
                indent=2,
            ),
            flush=True,
        )
    REPORT.write_text(
        json.dumps(safe(report), ensure_ascii=False, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
