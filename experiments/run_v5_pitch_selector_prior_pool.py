#!/usr/bin/env python3
"""Lock a pitch selector on 2020/2021 labels, then materialize 2022 MoE."""

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
from experiments.run_e22r_probs_rolling import (  # noqa: E402
    GROUPS,
    load_group_labels,
)
from experiments.run_v2_rolling import (  # noqa: E402
    build_e22_catboost_probabilities,
)


DATA = ROOT / "open/data/train.csv"
PRED = ROOT / "experiments/results/predictions"
PREREG = ROOT / "experiments/params/v5_pitch_selector_prior_pool_preregister.json"
SELECTION = ROOT / "experiments/results/v5_pitch_selector_prior_pool_selection.json"
MATERIALIZATION = ROOT / "experiments/results/v5_pitch_selector_moe_locked_2022.json"
SOURCE_2022 = PRED / "v5_pitchtype_moe_c_dev22_2022.npz"
OUTPUT_2022 = PRED / "v5_pitch_selector_moe_locked_2022.npz"


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
        # Selector fitting and selection never receive the competition target.
        frame = frame.drop(columns=[TARGET])
    row_ids = pd.read_csv(
        DATA, usecols=["row_id"], dtype="string", encoding="utf-8-sig"
    )["row_id"]
    if len(row_ids) != len(frame):
        raise ValueError("row_id alignment mismatch")
    labels = load_group_labels()
    frame.insert(0, "row_id", row_ids.to_numpy())
    frame["e22_pitch_type_group"] = frame["row_id"].map(labels)
    return frame


def source_frequency(history: pd.DataFrame) -> np.ndarray:
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


def official_prior(
    rows: pd.DataFrame, source: np.ndarray, k: float
) -> np.ndarray:
    rates = np.column_stack(
        [
            pd.to_numeric(rows["asof_pitcher_fastball_rate"], errors="coerce"),
            pd.to_numeric(rows["asof_pitcher_breaking_rate"], errors="coerce"),
            pd.to_numeric(rows["asof_pitcher_offspeed_rate"], errors="coerce"),
        ]
    ).astype(np.float64)
    finite = np.isfinite(rates).all(axis=1)
    rates = np.nan_to_num(rates, nan=0.0, posinf=0.0, neginf=0.0)
    rates = np.clip(rates, 0.0, 1.0)
    other = np.clip(1.0 - rates.sum(axis=1), 0.0, 1.0)
    raw = np.column_stack([rates, other])
    total = raw.sum(axis=1, keepdims=True)
    valid = finite & (total[:, 0] > 0.0)
    raw = np.divide(
        raw,
        total,
        out=np.tile(source, (len(rows), 1)),
        where=total > 0.0,
    )
    raw[~valid] = source
    n = pd.to_numeric(
        rows["asof_pitcher_pitchmix_n"], errors="coerce"
    ).fillna(0.0).to_numpy(dtype=np.float64)
    n = np.clip(n, 0.0, None)
    n[~valid] = 0.0
    denominator = n + float(k)
    result = np.empty_like(raw)
    usable = denominator > 0.0
    result[usable] = (
        n[usable, None] * raw[usable] + float(k) * source[None, :]
    ) / denominator[usable, None]
    result[~usable] = source
    result /= result.sum(axis=1, keepdims=True)
    return result


def pool_probabilities(
    context: np.ndarray, prior: np.ndarray, kind: str, alpha: float
) -> np.ndarray:
    context = np.clip(np.asarray(context, dtype=np.float64), 1e-12, 1.0)
    prior = np.clip(np.asarray(prior, dtype=np.float64), 1e-12, 1.0)
    if kind == "linear":
        result = float(alpha) * context + (1.0 - float(alpha)) * prior
    elif kind == "geometric":
        result = np.power(context, float(alpha)) * np.power(
            prior, 1.0 - float(alpha)
        )
    else:
        raise ValueError(f"unknown pool kind: {kind}")
    result /= result.sum(axis=1, keepdims=True)
    return result


def selector_metrics(probabilities: np.ndarray, truth: np.ndarray) -> dict[str, Any]:
    true_probability = probabilities[np.arange(len(truth)), truth]
    return {
        "rows": int(len(truth)),
        "multiclass_log_loss": float(
            -np.mean(np.log(np.maximum(true_probability, 1e-12)))
        ),
        "accuracy": float(np.mean(np.argmax(probabilities, axis=1) == truth)),
        "mean_true_group_probability": float(np.mean(true_probability)),
        "probability_mean": {
            group: float(probabilities[:, index].mean())
            for index, group in enumerate(GROUPS)
        },
    }


def candidate_grid(prereg: dict[str, Any]) -> list[dict[str, Any]]:
    grid = prereg["candidate_grid"]
    candidates = []
    for k in grid["prior_k"]:
        for kind in grid["pool"]:
            for alpha in grid["catboost_weight_alpha"]:
                candidates.append({
                    "candidate_id": (
                        f"k{float(k):g}_{kind}_a{int(round(float(alpha) * 100)):03d}"
                    ),
                    "prior_k": float(k),
                    "pool": str(kind),
                    "alpha": float(alpha),
                })
    return candidates


def select(prereg: dict[str, Any]) -> None:
    frame = load_frame()
    candidates = candidate_grid(prereg)
    fold_payload: dict[str, Any] = {}
    scores: dict[str, list[float]] = {
        item["candidate_id"]: [] for item in candidates
    }
    for year in prereg["selection_years"]:
        history = frame.loc[frame["season"].lt(year)].copy()
        valid = frame.loc[frame["season"].eq(year)].copy()
        _, context_frame, stage1_meta = build_e22_catboost_probabilities(
            history, valid
        )
        context = context_frame.to_numpy(dtype=np.float64)
        source = source_frequency(history)
        labels = valid["e22_pitch_type_group"].astype("string")
        route = valid["game_type"].eq("R") & labels.isin(GROUPS)
        truth_map = {group: index for index, group in enumerate(GROUPS)}
        truth = labels.loc[route].map(truth_map).to_numpy(dtype=np.int16)
        context = context[route.to_numpy(dtype=bool)]
        rows = valid.loc[route]
        prior_cache = {
            float(k): official_prior(rows, source, float(k))
            for k in prereg["candidate_grid"]["prior_k"]
        }
        fold_trials = []
        for item in candidates:
            probability = pool_probabilities(
                context,
                prior_cache[item["prior_k"]],
                item["pool"],
                item["alpha"],
            )
            measured = selector_metrics(probability, truth)
            scores[item["candidate_id"]].append(measured["multiclass_log_loss"])
            fold_trials.append({**item, "metrics": measured})
        fold_payload[str(year)] = {
            "history_seasons": sorted(
                int(value) for value in history["season"].unique()
            ),
            "source_group_frequency": {
                group: float(source[index]) for index, group in enumerate(GROUPS)
            },
            "stage1": stage1_meta,
            "trials": fold_trials,
        }
        print(
            f"selector fold {year}: matched={len(truth):,}, "
            f"context_logloss={selector_metrics(context, truth)['multiclass_log_loss']:.6f}",
            flush=True,
        )

    aggregates = []
    by_id = {item["candidate_id"]: item for item in candidates}
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
            item["worst_log_loss"],
            item["mean_log_loss"],
            item["candidate_id"],
        ),
    )
    report = {
        "experiment_id": prereg["experiment_id"],
        "mode": "selector_only_lock",
        "preregister_sha256": file_hash(PREREG),
        "control_success_read_for_selection": False,
        "years_read": list(prereg["selection_years"]),
        "control_target_year_not_read": 2022,
        "folds": fold_payload,
        "aggregates": aggregates,
        "selected": selected,
        "status": "locked_before_2022_control_target_evaluation",
    }
    write_immutable(SELECTION, report)
    print(json.dumps({"status": report["status"], "selected": selected}, indent=2))


def materialize2022(prereg: dict[str, Any]) -> None:
    selection = json.loads(SELECTION.read_text(encoding="utf-8"))
    if selection["status"] != "locked_before_2022_control_target_evaluation":
        raise ValueError("selector is not locked")
    if selection["preregister_sha256"] != file_hash(PREREG):
        raise ValueError("preregister changed after selector lock")
    if OUTPUT_2022.exists() or MATERIALIZATION.exists():
        raise FileExistsError("2022 selector materialization is immutable")
    frame = load_frame()
    with np.load(SOURCE_2022, allow_pickle=False) as archive:
        source_artifact = {
            key: np.asarray(archive[key]) for key in archive.files
        }
    row_index = source_artifact["row_index"].astype(np.int64)
    rows = frame.loc[row_index]
    if not np.all(rows["season"].to_numpy(dtype=np.int16) == 2022):
        raise ValueError("source artifact is not the 2022 fold")
    history = frame.loc[frame["season"].lt(2022)]
    source = source_frequency(history)
    selected = selection["selected"]
    prior = official_prior(rows, source, float(selected["prior_k"]))
    prefix = "catboost_pitchtype_moe__"
    context = np.column_stack(
        [source_artifact[f"{prefix}p_{group}"] for group in GROUPS]
    )
    pooled = pool_probabilities(
        context, prior, str(selected["pool"]), float(selected["alpha"])
    )
    experts = np.column_stack(
        [source_artifact[f"{prefix}expert_{group}"] for group in GROUPS]
    )
    prediction = np.sum(pooled * experts, axis=1)
    prediction = np.clip(prediction, 1e-6, 1.0 - 1e-6)
    np.savez_compressed(
        OUTPUT_2022,
        y=source_artifact["y"],
        row_index=source_artifact["row_index"],
        cluster=source_artifact["cluster"],
        pitchtype_moe_selector=prediction,
        **{
            f"selector_p_{group}": pooled[:, index]
            for index, group in enumerate(GROUPS)
        },
    )
    report = {
        "experiment_id": prereg["experiment_id"],
        "mode": "locked_selector_2022_materialization",
        "preregister_sha256": file_hash(PREREG),
        "selection_sha256": file_hash(SELECTION),
        "source_artifact": str(SOURCE_2022.relative_to(ROOT)),
        "source_artifact_sha256": file_hash(SOURCE_2022),
        "output_artifact": str(OUTPUT_2022.relative_to(ROOT)),
        "output_artifact_sha256": file_hash(OUTPUT_2022),
        "selected": selected,
        "rows": int(len(prediction)),
        "prediction_mean": float(prediction.mean()),
        "prediction_std": float(prediction.std()),
        "control_success_used_to_construct_prediction": False,
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
