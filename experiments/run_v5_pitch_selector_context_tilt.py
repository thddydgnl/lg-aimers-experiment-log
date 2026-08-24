#!/usr/bin/env python3
"""Select a hierarchical repertoire tilt without reading control targets."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.run_baselines import TARGET, load_train  # noqa: E402
from experiments.run_e22r_probs_rolling import GROUPS, load_group_labels  # noqa: E402
from experiments.run_v2_rolling import build_e22_catboost_probabilities  # noqa: E402


DATA = ROOT / "open/data/train.csv"
PRED = ROOT / "experiments/results/predictions"
PREREG = ROOT / "experiments/params/v5_pitch_selector_context_tilt_preregister.json"
SELECTION = ROOT / "experiments/results/v5_pitch_selector_context_tilt_selection.json"
MATERIALIZATION = ROOT / "experiments/results/v5_pitch_selector_context_tilt_2022.json"
OUTPUT_2022 = PRED / "v5_pitch_selector_context_tilt_2022.npz"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("select", "materialize2022"), required=True)
    return parser.parse_args()


def file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_immutable(path: Path, payload: dict[str, Any]) -> None:
    if path.exists():
        raise FileExistsError(f"immutable output already exists: {path}")
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )


def load_frame() -> pd.DataFrame:
    frame = load_train(DATA)
    if TARGET in frame.columns:
        frame = frame.drop(columns=[TARGET])
    row_ids = pd.read_csv(
        DATA, usecols=["row_id"], dtype="string", encoding="utf-8-sig"
    )["row_id"]
    labels = load_group_labels()
    frame.insert(0, "row_id", row_ids.to_numpy())
    frame["e22_pitch_type_group"] = frame["row_id"].map(labels)
    inning = pd.to_numeric(frame["inning"], errors="coerce").fillna(0).to_numpy()
    frame["inning_bucket"] = np.select(
        [inning <= 3, inning <= 6, inning <= 9], ["early", "middle", "late"],
        default="extra",
    )
    return frame


def historical_frequency(history: pd.DataFrame) -> np.ndarray:
    labels = history.loc[
        history["game_type"].eq("R")
        & history["e22_pitch_type_group"].isin(GROUPS),
        "e22_pitch_type_group",
    ].astype(str)
    counts = labels.value_counts()
    values = np.asarray(
        [float(counts.get(group, 0.0)) for group in GROUPS], dtype=np.float64
    )
    if values.sum() <= 0.0:
        values[:] = 1.0
    return values / values.sum()


def official_prior(rows: pd.DataFrame, source: np.ndarray, k: float) -> np.ndarray:
    rates = np.column_stack(
        [
            pd.to_numeric(rows["asof_pitcher_fastball_rate"], errors="coerce"),
            pd.to_numeric(rows["asof_pitcher_breaking_rate"], errors="coerce"),
            pd.to_numeric(rows["asof_pitcher_offspeed_rate"], errors="coerce"),
        ]
    ).astype(np.float64)
    finite = np.isfinite(rates).all(axis=1)
    rates = np.clip(np.nan_to_num(rates, nan=0.0), 0.0, 1.0)
    raw = np.column_stack([rates, np.clip(1.0 - rates.sum(axis=1), 0.0, 1.0)])
    total = raw.sum(axis=1, keepdims=True)
    valid = finite & (total[:, 0] > 0.0)
    raw = np.divide(
        raw, total, out=np.tile(source, (len(rows), 1)), where=total > 0.0
    )
    raw[~valid] = source
    n = pd.to_numeric(rows["asof_pitcher_pitchmix_n"], errors="coerce").fillna(0.0)
    n = np.clip(n.to_numpy(dtype=np.float64), 0.0, None)
    n[~valid] = 0.0
    denominator = n + float(k)
    result = np.tile(source, (len(rows), 1))
    usable = denominator > 0.0
    result[usable] = (
        n[usable, None] * raw[usable] + float(k) * source[None, :]
    ) / denominator[usable, None]
    result /= result.sum(axis=1, keepdims=True)
    return result


def grouped_counts(
    history: pd.DataFrame, rows: pd.DataFrame, columns: list[str]
) -> tuple[np.ndarray, np.ndarray]:
    labeled = history.loc[
        history["game_type"].eq("R")
        & history["e22_pitch_type_group"].isin(GROUPS),
        [*columns, "e22_pitch_type_group"],
    ].copy()
    table = labeled.groupby(
        [*columns, "e22_pitch_type_group"], sort=False, observed=True
    ).size().unstack(fill_value=0)
    table = table.reindex(columns=GROUPS, fill_value=0)
    if len(columns) == 1:
        aligned = table.reindex(rows[columns[0]].to_numpy())
    else:
        aligned = table.reindex(pd.MultiIndex.from_frame(rows[columns]))
    counts = aligned.fillna(0.0).to_numpy(dtype=np.float64)
    return counts, counts.sum(axis=1)


def context_tilt(
    history: pd.DataFrame,
    rows: pd.DataFrame,
    columns: list[str],
    source: np.ndarray,
    parent_k: float,
    cell_k: float,
    tau: float,
    prior_k: float,
    ratio_clip: tuple[float, float],
) -> np.ndarray:
    parent_counts, parent_n = grouped_counts(history, rows, ["pitcher_id"])
    parent = (
        parent_counts + float(parent_k) * source[None, :]
    ) / (parent_n[:, None] + float(parent_k))
    cell_counts, cell_n = grouped_counts(history, rows, columns)
    cell = (
        cell_counts + float(cell_k) * parent
    ) / (cell_n[:, None] + float(cell_k))
    current = official_prior(rows, source, float(prior_k))
    ratio = np.divide(cell, np.maximum(parent, 1e-12))
    ratio = np.clip(ratio, float(ratio_clip[0]), float(ratio_clip[1]))
    result = current * np.power(ratio, float(tau))
    result /= result.sum(axis=1, keepdims=True)
    return result


def fold_count_cache(
    history: pd.DataFrame,
    rows: pd.DataFrame,
    prereg: dict[str, Any],
) -> dict[str, Any]:
    """Precompute every expensive groupby exactly once for a selector fold."""
    source = historical_frequency(history)
    parent_counts, parent_n = grouped_counts(history, rows, ["pitcher_id"])
    parent = (
        parent_counts + float(prereg["parent_k"]) * source[None, :]
    ) / (parent_n[:, None] + float(prereg["parent_k"]))
    cells = {
        spec_name: grouped_counts(history, rows, list(columns))
        for spec_name, columns in prereg["context_specs"].items()
    }
    priors = {
        float(k): official_prior(rows, source, float(k))
        for k in prereg["candidate_grid"]["official_prior_k"]
    }
    return {"source": source, "parent": parent, "cells": cells, "priors": priors}


def context_tilt_from_cache(
    cache: dict[str, Any],
    candidate: dict[str, Any],
    prereg: dict[str, Any],
) -> np.ndarray:
    parent = cache["parent"]
    cell_counts, cell_n = cache["cells"][candidate["spec_name"]]
    cell_k = float(candidate["cell_k"])
    cell = (cell_counts + cell_k * parent) / (cell_n[:, None] + cell_k)
    ratio = np.divide(cell, np.maximum(parent, 1e-12))
    ratio = np.clip(
        ratio, float(prereg["ratio_clip"][0]), float(prereg["ratio_clip"][1])
    )
    current = cache["priors"][float(candidate["official_prior_k"])]
    result = current * np.power(ratio, float(candidate["tau"]))
    result /= result.sum(axis=1, keepdims=True)
    return result


def geometric_pool(
    table_probability: np.ndarray, catboost_probability: np.ndarray, alpha: float
) -> np.ndarray:
    left = np.clip(table_probability, 1e-12, 1.0)
    right = np.clip(catboost_probability, 1e-12, 1.0)
    result = np.power(left, 1.0 - float(alpha)) * np.power(right, float(alpha))
    result /= result.sum(axis=1, keepdims=True)
    return result


def selector_metrics(probability: np.ndarray, truth: np.ndarray) -> dict[str, Any]:
    true_probability = probability[np.arange(len(truth)), truth]
    return {
        "rows": int(len(truth)),
        "multiclass_log_loss": float(
            -np.mean(np.log(np.maximum(true_probability, 1e-12)))
        ),
        "accuracy": float(np.mean(np.argmax(probability, axis=1) == truth)),
        "mean_true_group_probability": float(np.mean(true_probability)),
    }


def candidate_grid(prereg: dict[str, Any]) -> list[dict[str, Any]]:
    grid = prereg["candidate_grid"]
    result = []
    for spec_name in prereg["context_specs"]:
        for window in grid["history_window"]:
            for cell_k in grid["cell_k"]:
                for tau in grid["tau"]:
                    for prior_k in grid["official_prior_k"]:
                        for alpha in grid["catboost_geometric_weight"]:
                            window_name = "all" if window is None else str(int(window))
                            result.append({
                                "candidate_id": (
                                    f"{spec_name}_w{window_name}_ck{float(cell_k):g}"
                                    f"_t{int(round(float(tau) * 100)):03d}"
                                    f"_pk{float(prior_k):g}_a{int(round(float(alpha) * 100)):03d}"
                                ),
                                "spec_name": spec_name,
                                "history_window": window,
                                "cell_k": float(cell_k),
                                "tau": float(tau),
                                "official_prior_k": float(prior_k),
                                "catboost_weight": float(alpha),
                            })
    return result


def candidate_probability(
    history: pd.DataFrame,
    rows: pd.DataFrame,
    catboost_probability: np.ndarray,
    candidate: dict[str, Any],
    prereg: dict[str, Any],
) -> np.ndarray:
    window = candidate["history_window"]
    source_history = history
    if window is not None:
        cutoff = int(rows["season"].iloc[0]) - int(window)
        source_history = history.loc[history["season"].ge(cutoff)]
    source = historical_frequency(source_history)
    table_probability = context_tilt(
        source_history,
        rows,
        list(prereg["context_specs"][candidate["spec_name"]]),
        source,
        float(prereg["parent_k"]),
        float(candidate["cell_k"]),
        float(candidate["tau"]),
        float(candidate["official_prior_k"]),
        tuple(float(value) for value in prereg["ratio_clip"]),
    )
    return geometric_pool(
        table_probability, catboost_probability, float(candidate["catboost_weight"])
    )


def select(prereg: dict[str, Any]) -> None:
    frame = load_frame()
    candidates = candidate_grid(prereg)
    scores = {candidate["candidate_id"]: [] for candidate in candidates}
    fold_reports: dict[str, Any] = {}
    truth_map = {group: index for index, group in enumerate(GROUPS)}
    for year in prereg["selection_contract"]["years"]:
        history = frame.loc[frame["season"].lt(year)].copy()
        valid = frame.loc[frame["season"].eq(year)].copy()
        _, catboost_frame, stage1_meta = build_e22_catboost_probabilities(
            history, valid
        )
        labels = valid["e22_pitch_type_group"].astype("string")
        route = valid["game_type"].eq("R") & labels.isin(GROUPS)
        rows = valid.loc[route].copy()
        truth = labels.loc[route].map(truth_map).to_numpy(dtype=np.int16)
        catboost_probability = catboost_frame.to_numpy(dtype=np.float64)[
            route.to_numpy(dtype=bool)
        ]
        trials = []
        fold_cache = fold_count_cache(history, rows, prereg)
        previous_table_key: tuple[Any, ...] | None = None
        table_probability: np.ndarray | None = None
        for candidate in candidates:
            table_key = (
                candidate["spec_name"], candidate["history_window"],
                candidate["cell_k"], candidate["tau"],
                candidate["official_prior_k"],
            )
            if table_key != previous_table_key:
                table_probability = context_tilt_from_cache(
                    fold_cache, candidate, prereg
                )
                previous_table_key = table_key
            if table_probability is None:
                raise RuntimeError("selector table probability was not initialized")
            probability = geometric_pool(
                table_probability, catboost_probability,
                float(candidate["catboost_weight"]),
            )
            measured = selector_metrics(probability, truth)
            scores[candidate["candidate_id"]].append(
                measured["multiclass_log_loss"]
            )
            trials.append({**candidate, "metrics": measured})
        pure_catboost = selector_metrics(catboost_probability, truth)
        fold_reports[str(year)] = {
            "history_seasons": sorted(int(value) for value in history["season"].unique()),
            "matched_regular_rows": int(len(truth)),
            "pure_catboost": pure_catboost,
            "stage1": stage1_meta,
            "trials": trials,
        }
        print(
            f"selector fold {year}: matched={len(truth):,}, "
            f"catboost_logloss={pure_catboost['multiclass_log_loss']:.6f}",
            flush=True,
        )

    by_id = {candidate["candidate_id"]: candidate for candidate in candidates}
    aggregates = []
    for candidate_id, values in scores.items():
        aggregates.append({
            **by_id[candidate_id],
            "fold_log_losses": [float(value) for value in values],
            "worst_log_loss": float(max(values)),
            "mean_log_loss": float(np.mean(values)),
        })
    selected = min(
        aggregates,
        key=lambda item: (
            item["worst_log_loss"], item["mean_log_loss"], item["candidate_id"]
        ),
    )
    report = {
        "experiment_id": prereg["experiment_id"],
        "mode": "selector_only_lock",
        "preregister_sha256": file_hash(PREREG),
        "competition_target_read_for_selection": False,
        "years_read": list(prereg["selection_contract"]["years"]),
        "target_years_not_read": [2022, 2023, 2024],
        "folds": fold_reports,
        "aggregates": aggregates,
        "selected": selected,
        "status": "locked_before_2022_control_target_evaluation",
    }
    write_immutable(SELECTION, report)
    print(json.dumps({"status": report["status"], "selected": selected}, indent=2))


def materialize2022(prereg: dict[str, Any]) -> None:
    selection = json.loads(SELECTION.read_text(encoding="utf-8"))
    if selection["preregister_sha256"] != file_hash(PREREG):
        raise ValueError("preregister changed after selector lock")
    if selection["status"] != "locked_before_2022_control_target_evaluation":
        raise ValueError("selector is not locked")
    if OUTPUT_2022.exists() or MATERIALIZATION.exists():
        raise FileExistsError("2022 materialization is immutable")
    frame = load_frame()
    source_path = ROOT / prereg["source_2022_artifact"]
    with np.load(source_path, allow_pickle=False) as archive:
        source_artifact = {key: np.asarray(archive[key]) for key in archive.files}
    row_index = source_artifact["row_index"].astype(np.int64)
    rows = frame.loc[row_index].copy()
    if not np.all(rows["season"].to_numpy(dtype=np.int16) == 2022):
        raise ValueError("source artifact is not the 2022 fold")
    history = frame.loc[frame["season"].lt(2022)].copy()
    prefix = "catboost_pitchtype_moe__"
    catboost_probability = np.column_stack(
        [source_artifact[f"{prefix}p_{group}"] for group in GROUPS]
    ).astype(np.float64)
    selected = selection["selected"]
    selector_probability = candidate_probability(
        history, rows, catboost_probability, selected, prereg
    )
    experts = np.column_stack(
        [source_artifact[f"{prefix}expert_{group}"] for group in GROUPS]
    ).astype(np.float64)
    prediction = np.clip(
        np.sum(selector_probability * experts, axis=1), 1e-6, 1.0 - 1e-6
    )
    output_payload = {
        "y": source_artifact["y"],
        "row_index": source_artifact["row_index"],
        "cluster": source_artifact["cluster"],
        prereg["candidate_2022_key"]: prediction,
        **{
            f"selector_p_{group}": selector_probability[:, index]
            for index, group in enumerate(GROUPS)
        },
        "diagnostic_true_group_code": source_artifact[
            f"{prefix}diagnostic_true_group_code"
        ],
    }
    np.savez_compressed(OUTPUT_2022, **output_payload)
    report = {
        "experiment_id": prereg["experiment_id"],
        "mode": "locked_selector_2022_materialization",
        "preregister_sha256": file_hash(PREREG),
        "selection_sha256": file_hash(SELECTION),
        "source_artifact": str(source_path.relative_to(ROOT)),
        "source_artifact_sha256": file_hash(source_path),
        "output_artifact": str(OUTPUT_2022.relative_to(ROOT)),
        "output_artifact_sha256": file_hash(OUTPUT_2022),
        "selected": selected,
        "rows": int(len(prediction)),
        "prediction_mean": float(prediction.mean()),
        "prediction_std": float(prediction.std()),
        "competition_target_used_to_construct_prediction": False,
        "current_pitch_group_used_to_construct_prediction": False,
        "status": "materialized_for_2022_goal_gate",
    }
    write_immutable(MATERIALIZATION, report)
    print(json.dumps(report, indent=2))


def main() -> None:
    args = parse_args()
    prereg = json.loads(PREREG.read_text(encoding="utf-8"))
    if args.mode == "select":
        select(prereg)
    else:
        materialize2022(prereg)


if __name__ == "__main__":
    main()
