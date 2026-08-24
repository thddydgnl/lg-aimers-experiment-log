#!/usr/bin/env python3
"""Strict 2020/2021 source gate for fixed temporal prototype retrieval."""

from __future__ import annotations

import json
from pathlib import Path
import sys
from typing import Any

import faiss
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.analyze_v5_dense_pitchtype_moe import digest, load, safe, score  # noqa: E402
from experiments.analyze_v5_game_centered_brier_source import route_metrics  # noqa: E402


YEARS = (2020, 2021)
TRAIN = ROOT / "open/data/train.csv"
PRED = ROOT / "experiments/results/predictions"
PREREG = ROOT / "experiments/params/v5_temporal_prototype_retrieval_preregister.json"
REPORT = ROOT / "experiments/results/v5_temporal_prototype_source_gate.json"
TARGET = "control_success"
SEASON = "season"
PITCHER = "pitcher_id"
BUCKET_KEYS = ["balls_before", "strikes_before", "pitcher_hand", "batter_hand"]
GROUP_KEYS = [SEASON, PITCHER, *BUCKET_KEYS]
RATE_COLUMNS = [
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
COUNT_COLUMNS = ["asof_pitcher_n", "asof_batter_n", "asof_pitcher_pitchmix_n"]
CONTEXT_COLUMNS = [
    "game_month",
    "inning",
    "outs_before",
    "num_runners_on",
    "score_diff_pitcher_team",
    "li",
]
NUMERIC_COLUMNS = [
    *RATE_COLUMNS,
    "log_asof_pitcher_n",
    "log_asof_batter_n",
    "log_asof_pitcher_pitchmix_n",
    "scaled_game_month",
    "scaled_inning",
    "scaled_outs_before",
    "scaled_num_runners_on",
    "scaled_score_diff_pitcher_team",
    "log_li",
]
READ_COLUMNS = sorted(
    set(
        [
            SEASON,
            "game_type",
            PITCHER,
            "batter_id",
            "pitcher_hand",
            "batter_hand",
            "balls_before",
            "strikes_before",
            TARGET,
        ]
        + RATE_COLUMNS
        + COUNT_COLUMNS
        + CONTEXT_COLUMNS
    )
)


def numeric_state(frame: pd.DataFrame) -> pd.DataFrame:
    values = pd.DataFrame(index=frame.index)
    for column in RATE_COLUMNS:
        values[column] = pd.to_numeric(frame[column], errors="coerce")
    for column in COUNT_COLUMNS:
        raw = pd.to_numeric(frame[column], errors="coerce").fillna(0.0).clip(lower=0.0)
        values[f"log_{column}"] = np.log1p(raw)
    values["scaled_game_month"] = (
        pd.to_numeric(frame["game_month"], errors="coerce") - 6.0
    ) / 4.0
    values["scaled_inning"] = (
        pd.to_numeric(frame["inning"], errors="coerce").clip(1.0, 12.0) - 5.0
    ) / 4.0
    values["scaled_outs_before"] = pd.to_numeric(
        frame["outs_before"], errors="coerce"
    ) / 2.0
    values["scaled_num_runners_on"] = pd.to_numeric(
        frame["num_runners_on"], errors="coerce"
    ) / 3.0
    values["scaled_score_diff_pitcher_team"] = pd.to_numeric(
        frame["score_diff_pitcher_team"], errors="coerce"
    ).clip(-8.0, 8.0) / 4.0
    leverage = pd.to_numeric(frame["li"], errors="coerce").clip(lower=0.0)
    values["log_li"] = np.log1p(leverage)
    return values[NUMERIC_COLUMNS]


def state_prior(frame: pd.DataFrame, historical_rate: float) -> np.ndarray:
    n = pd.to_numeric(frame["asof_pitcher_n"], errors="coerce").fillna(0.0)
    n_values = n.clip(lower=0.0).to_numpy(dtype=np.float64)
    rate = pd.to_numeric(
        frame["asof_pitcher_success_rate"], errors="coerce"
    ).fillna(historical_rate).to_numpy(dtype=np.float64)
    return np.clip(
        (n_values * rate + 50.0 * historical_rate) / (n_values + 50.0),
        1e-4,
        1.0 - 1e-4,
    )


def build_fold_retrieval(
    frame: pd.DataFrame,
    valid_index: np.ndarray,
    year: int,
    neighbors_grid: list[int],
) -> tuple[dict[tuple[int, str], np.ndarray], dict[str, Any]]:
    history = frame.loc[
        frame[SEASON].lt(year) & frame["game_type"].astype(str).eq("R")
    ].copy()
    valid = frame.loc[valid_index].copy()
    regular = valid["game_type"].astype(str).eq("R").to_numpy()
    historical_rate = float(history[TARGET].mean())

    history_numeric = numeric_state(history)
    valid_numeric = numeric_state(valid)
    medians = history_numeric.median(axis=0, skipna=True).fillna(0.0)
    history_numeric = history_numeric.fillna(medians)
    valid_numeric = valid_numeric.fillna(medians)
    history_prior = state_prior(history, historical_rate)
    valid_prior = state_prior(valid, historical_rate)

    work = pd.concat(
        [history[GROUP_KEYS].reset_index(drop=True), history_numeric.reset_index(drop=True)],
        axis=1,
    )
    work["_target"] = history[TARGET].to_numpy(dtype=np.float64)
    work["_state_prior"] = history_prior
    grouped = work.groupby(GROUP_KEYS, sort=True, observed=True, dropna=False)
    prototype = grouped[NUMERIC_COLUMNS + ["_state_prior", "_target"]].mean()
    prototype["_rows"] = grouped.size().astype(np.float64)
    prototype = prototype.reset_index()
    prototype["_shrunk_residual"] = (
        prototype["_rows"] / (prototype["_rows"] + 64.0)
    ) * (prototype["_target"] - prototype["_state_prior"])

    center = prototype[NUMERIC_COLUMNS].median(axis=0)
    q25 = prototype[NUMERIC_COLUMNS].quantile(0.25)
    q75 = prototype[NUMERIC_COLUMNS].quantile(0.75)
    scale = (q75 - q25) / 1.349
    fallback = prototype[NUMERIC_COLUMNS].std(axis=0, ddof=0)
    scale = scale.where(scale.gt(1e-3), fallback).where(lambda value: value.gt(1e-3), 1.0)
    prototype_z = np.clip(
        (prototype[NUMERIC_COLUMNS] - center) / scale,
        -6.0,
        6.0,
    ).to_numpy(dtype=np.float32)
    query_z = np.clip(
        (valid_numeric[NUMERIC_COLUMNS] - center) / scale,
        -6.0,
        6.0,
    ).to_numpy(dtype=np.float32)

    maximum_k = max(neighbors_grid)
    neighbor_rate = {
        k: np.full(len(valid), np.nan, dtype=np.float64) for k in neighbors_grid
    }
    neighbor_residual = {
        k: np.full(len(valid), np.nan, dtype=np.float64) for k in neighbors_grid
    }
    query_table = valid[BUCKET_KEYS].reset_index(drop=True)
    query_table["_position"] = np.arange(len(valid), dtype=np.int64)
    query_regular = query_table.loc[regular]
    prototype_bucket_groups = prototype.groupby(
        BUCKET_KEYS, sort=False, observed=True, dropna=False
    ).indices
    query_bucket_groups = query_regular.groupby(
        BUCKET_KEYS, sort=False, observed=True, dropna=False
    ).indices
    bucket_sizes: list[int] = []
    missing_buckets: list[str] = []
    for key, query_local_positions in query_bucket_groups.items():
        normalized_key = key if isinstance(key, tuple) else (key,)
        prototype_positions = prototype_bucket_groups.get(normalized_key)
        if prototype_positions is None:
            missing_buckets.append("|".join(str(value) for value in normalized_key))
            continue
        query_positions = query_regular.iloc[query_local_positions]["_position"].to_numpy(
            dtype=np.int64
        )
        proto_positions = np.asarray(prototype_positions, dtype=np.int64)
        bucket_sizes.append(int(len(proto_positions)))
        index = faiss.IndexFlatL2(len(NUMERIC_COLUMNS))
        index.add(np.ascontiguousarray(prototype_z[proto_positions]))
        available_k = min(maximum_k, len(proto_positions))
        squared_distance, local_neighbors = index.search(
            np.ascontiguousarray(query_z[query_positions]), available_k
        )
        selected_proto = proto_positions[local_neighbors]
        distance = np.sqrt(np.maximum(squared_distance, 0.0)).astype(np.float64)
        reliability = np.sqrt(
            np.minimum(
                prototype["_rows"].to_numpy(dtype=np.float64)[selected_proto],
                256.0,
            )
        )
        weights = reliability / (1.0 + distance)
        target_values = prototype["_target"].to_numpy(dtype=np.float64)[selected_proto]
        residual_values = prototype["_shrunk_residual"].to_numpy(
            dtype=np.float64
        )[selected_proto]
        for k in neighbors_grid:
            use_k = min(k, available_k)
            denominator = weights[:, :use_k].sum(axis=1)
            neighbor_rate[k][query_positions] = (
                weights[:, :use_k] * target_values[:, :use_k]
            ).sum(axis=1) / denominator
            neighbor_residual[k][query_positions] = (
                weights[:, :use_k] * residual_values[:, :use_k]
            ).sum(axis=1) / denominator
    if missing_buckets:
        raise ValueError(f"{year}: missing historical buckets: {missing_buckets}")
    for k in neighbors_grid:
        if not np.isfinite(neighbor_rate[k][regular]).all():
            raise ValueError(f"{year}: non-finite k={k} neighbor rate")
        if not np.isfinite(neighbor_residual[k][regular]).all():
            raise ValueError(f"{year}: non-finite k={k} neighbor residual")

    outputs: dict[tuple[int, str], np.ndarray] = {}
    amplitudes = [0.25, 0.5, 1.0]
    for k in neighbors_grid:
        for amplitude in amplitudes:
            rate_prediction = valid_prior.copy()
            rate_prediction[regular] = (
                (1.0 - amplitude) * valid_prior[regular]
                + amplitude * neighbor_rate[k][regular]
            )
            outputs[(k, f"rate:{amplitude}")] = np.clip(
                rate_prediction, 1e-6, 1.0 - 1e-6
            )
            residual_prediction = valid_prior.copy()
            residual_prediction[regular] = (
                valid_prior[regular]
                + amplitude * neighbor_residual[k][regular]
            )
            outputs[(k, f"residual:{amplitude}")] = np.clip(
                residual_prediction, 1e-6, 1.0 - 1e-6
            )
    details = {
        "history_rows": int(len(history)),
        "history_seasons": sorted(int(value) for value in history[SEASON].unique()),
        "validation_rows": int(len(valid)),
        "validation_R_rows": int(regular.sum()),
        "historical_R_rate": historical_rate,
        "prototype_count": int(len(prototype)),
        "bucket_count": int(len(prototype_bucket_groups)),
        "bucket_size_min": int(min(bucket_sizes)),
        "bucket_size_median": float(np.median(bucket_sizes)),
        "bucket_size_max": int(max(bucket_sizes)),
        "numeric_feature_count": int(len(NUMERIC_COLUMNS)),
        "numeric_features": NUMERIC_COLUMNS,
        "imputation_medians": {column: float(value) for column, value in medians.items()},
        "scaling_center": {column: float(value) for column, value in center.items()},
        "scaling_scale": {column: float(value) for column, value in scale.items()},
        "query_prior_mean_R": float(valid_prior[regular].mean()),
        "neighbor_rate_mean_R": {
            str(k): float(neighbor_rate[k][regular].mean()) for k in neighbors_grid
        },
        "neighbor_residual_mean_R": {
            str(k): float(neighbor_residual[k][regular].mean()) for k in neighbors_grid
        },
        "row_independent_query": True,
        "validation_labels_used_in_library_scaling_or_retrieval": False,
    }
    return outputs, details


def main() -> None:
    if REPORT.exists():
        raise FileExistsError(f"immutable report exists: {REPORT}")
    prereg = json.loads(PREREG.read_text(encoding="utf-8"))
    if prereg["status"] != "locked_before_source_metrics":
        raise ValueError("unexpected preregistration status")
    if prereg["source_protocol"]["years"] != list(YEARS):
        raise ValueError("source-year contract changed")
    grid = prereg["candidate_grid"]
    neighbors_grid = [int(value) for value in grid["neighbors"]]
    modes = [str(value) for value in grid["mode"]]
    amplitudes = [float(value) for value in grid["retrieval_amplitude"]]
    gammas = [float(value) for value in grid["parent_gamma"]]
    faiss.omp_set_num_threads(6)

    parents: dict[int, dict[str, np.ndarray]] = {}
    maximum_row = 0
    input_hashes: dict[str, Any] = {}
    for year in YEARS:
        path = PRED / f"v4_m3_c_backtest_{year}_{year}.npz"
        artifact = load(path)
        row_index = artifact["row_index"].astype(np.int64)
        maximum_row = max(maximum_row, int(row_index.max()))
        parents[year] = {
            "y": artifact["y"].astype(np.int8),
            "row_index": row_index,
            "cluster": artifact["cluster"],
            "parent": artifact["catboost_outcome"].astype(np.float64),
        }
        input_hashes[str(year)] = {"parent": digest(path)}
    frame = pd.read_csv(TRAIN, usecols=READ_COLUMNS, nrows=maximum_row + 1)
    if int(frame[SEASON].max()) != max(YEARS):
        raise ValueError("source reader crossed the locked 2021 boundary")

    fold_predictions: dict[int, dict[tuple[int, str], np.ndarray]] = {}
    fold_details: dict[str, Any] = {}
    for year in YEARS:
        parent = parents[year]
        rows = frame.loc[parent["row_index"]]
        if not rows[SEASON].eq(year).all():
            raise ValueError(f"{year}: season alignment mismatch")
        if not np.array_equal(rows[TARGET].to_numpy(dtype=np.int8), parent["y"]):
            raise ValueError(f"{year}: target alignment mismatch")
        predictions, details = build_fold_retrieval(
            frame, parent["row_index"], year, neighbors_grid
        )
        fold_predictions[year] = predictions
        fold_details[str(year)] = details
        print(
            f"[{year}] history={details['history_rows']:,} "
            f"prototypes={details['prototype_count']:,} "
            f"buckets={details['bucket_count']}",
            flush=True,
        )

    trials: list[dict[str, Any]] = []
    for k in neighbors_grid:
        for mode in modes:
            for amplitude in amplitudes:
                retrieval_key = (k, f"{mode}:{amplitude}")
                for gamma in gammas:
                    year_metrics: dict[str, Any] = {}
                    for year in YEARS:
                        parent = parents[year]
                        rows = frame.loc[parent["row_index"]]
                        regular = rows["game_type"].astype(str).eq("R").to_numpy()
                        candidate = parent["parent"].copy()
                        retrieval = fold_predictions[year][retrieval_key]
                        candidate[regular] = np.clip(
                            (1.0 - gamma) * parent["parent"][regular]
                            + gamma * retrieval[regular],
                            1e-6,
                            1.0 - 1e-6,
                        )
                        year_metrics[str(year)] = {}
                        for route, mask in {
                            "full": np.ones(len(candidate), dtype=bool),
                            "R": regular,
                        }.items():
                            parent_score = score(
                                parent["y"], parent["parent"], mask
                            )["score"]
                            candidate_score = score(parent["y"], candidate, mask)[
                                "score"
                            ]
                            year_metrics[str(year)][route] = {
                                "gain": float(candidate_score - parent_score)
                            }
                    trials.append(
                        {
                            "neighbors": k,
                            "mode": mode,
                            "retrieval_amplitude": amplitude,
                            "parent_gamma": gamma,
                            "minimum_full_gain": float(
                                min(
                                    year_metrics[str(year)]["full"]["gain"]
                                    for year in YEARS
                                )
                            ),
                            "minimum_R_gain": float(
                                min(
                                    year_metrics[str(year)]["R"]["gain"]
                                    for year in YEARS
                                )
                            ),
                            "mean_full_gain": float(
                                np.mean(
                                    [
                                        year_metrics[str(year)]["full"]["gain"]
                                        for year in YEARS
                                    ]
                                )
                            ),
                            "years": year_metrics,
                        }
                    )
    selected = max(
        trials,
        key=lambda item: (
            item["minimum_full_gain"],
            item["minimum_R_gain"],
            item["mean_full_gain"],
            -item["neighbors"],
            -item["retrieval_amplitude"],
            -item["parent_gamma"],
        ),
    )

    selected_metrics: dict[str, Any] = {}
    artifacts: dict[str, Any] = {}
    selected_key = (
        int(selected["neighbors"]),
        f"{selected['mode']}:{float(selected['retrieval_amplitude'])}",
    )
    gamma = float(selected["parent_gamma"])
    for offset, year in enumerate(YEARS):
        parent = parents[year]
        rows = frame.loc[parent["row_index"]]
        regular = rows["game_type"].astype(str).eq("R").to_numpy()
        retrieval = fold_predictions[year][selected_key]
        candidate = parent["parent"].copy()
        candidate[regular] = np.clip(
            (1.0 - gamma) * parent["parent"][regular]
            + gamma * retrieval[regular],
            1e-6,
            1.0 - 1e-6,
        )
        selected_metrics[str(year)] = route_metrics(
            parent["y"],
            parent["parent"],
            candidate,
            parent["cluster"],
            regular,
            seed=8264000 + 10000 * offset,
        )
        output = PRED / f"v5_temporal_prototype_selected_source_{year}.npz"
        if output.exists():
            raise FileExistsError(f"immutable artifact exists: {output}")
        np.savez_compressed(
            output,
            y=parent["y"],
            row_index=parent["row_index"],
            cluster=parent["cluster"],
            parent_exact_c=parent["parent"].astype(np.float32),
            retrieval=retrieval.astype(np.float32),
            final_prediction=candidate.astype(np.float32),
            neighbors=np.asarray(selected["neighbors"], dtype=np.int64),
            retrieval_amplitude=np.asarray(
                selected["retrieval_amplitude"], dtype=np.float64
            ),
            parent_gamma=np.asarray(gamma, dtype=np.float64),
        )
        artifacts[str(year)] = {
            "path": str(output.relative_to(ROOT)),
            "sha256": digest(output),
        }

    requirements = prereg["source_protocol"]["advance_gate"]
    checks: dict[str, bool] = {}
    for year in YEARS:
        result = selected_metrics[str(year)]
        checks[f"{year}_full_gain"] = bool(
            result["full"]["gain"]
            >= float(requirements["minimum_full_gain_each_year"])
        )
        checks[f"{year}_R_gain"] = bool(
            result["R"]["gain"] >= float(requirements["minimum_R_gain_each_year"])
        )
        checks[f"{year}_full_ci"] = bool(
            result["full"]["pitcher_cluster_95_ci"]["ci_low"]
            > float(requirements["full_pitcher_cluster_95_ci_low_each_year"])
        )
        checks[f"{year}_R_ci"] = bool(
            result["R"]["pitcher_cluster_95_ci"]["ci_low"]
            > float(requirements["R_pitcher_cluster_95_ci_low_each_year"])
        )
    passed = bool(all(checks.values()))
    report = {
        "experiment_id": prereg["experiment_id"],
        "status": "source_pass" if passed else "source_failed",
        "preregister_sha256": digest(PREREG),
        "script_sha256": digest(Path(__file__)),
        "train_sha256": digest(TRAIN),
        "faiss_version": getattr(faiss, "__version__", "unknown"),
        "years_read": list(YEARS),
        "years_not_read": [2022, 2023, 2024],
        "selection": selected,
        "selected_metrics": selected_metrics,
        "trials": trials,
        "fold_details": fold_details,
        "input_sha256": input_hashes,
        "gate": {"requirements": requirements, "checks": checks, "pass": passed},
        "artifacts": artifacts,
        "goal_status": "active",
        "goal_completion_claimed": False,
    }
    REPORT.write_text(
        json.dumps(safe(report), ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(
        json.dumps(
            safe(
                {
                    "status": report["status"],
                    "selection": selected,
                    "selected_metrics": selected_metrics,
                    "gate": report["gate"],
                }
            ),
            ensure_ascii=False,
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
