#!/usr/bin/env python3
"""Compare fixed, leakage-safe blends of the preserved S4--S7 candidates."""

from __future__ import annotations

import argparse
import gc
import json
import platform
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments import run_e14_rolling as e14_module  # noqa: E402
from experiments.run_baselines import FEATURES as BASE_FEATURES  # noqa: E402
from experiments.run_baselines import TARGET, load_train  # noqa: E402
from experiments.run_e14_rolling import (  # noqa: E402
    E14_K,
    build_e14_features,
    fit_predict,
    make_hgb,
    make_linear,
    metric,
    prior_before_each_season,
    season_end_state,
)
from experiments.run_e15_pseudo_forward import candidate_priors  # noqa: E402
from experiments.run_e16_rolling import (  # noqa: E402
    build_e16_features,
    make_e16_hgb,
    make_e16_linear,
    role_states_before_each_season,
)
from experiments.run_e22r_probs_rolling import (  # noqa: E402
    E22_FEATURES,
    E22_PROB_FEATURES,
    align_group_probabilities,
    load_group_labels,
    make_stage1,
)
from experiments.run_e22r_mixture_rolling import GROUPS  # noqa: E402


ORIGINAL_LINEAR_CATEGORICAL = list(e14_module.LINEAR_CATEGORICAL)
ORIGINAL_HGB_CATEGORICAL = list(e14_module.HGB_CATEGORICAL)


def make_base_linear(features: list[str]):
    e14_module.LINEAR_CATEGORICAL = ORIGINAL_LINEAR_CATEGORICAL
    e14_module.HGB_CATEGORICAL = ORIGINAL_HGB_CATEGORICAL
    return make_linear(features)


def make_base_hgb(features: list[str]):
    e14_module.LINEAR_CATEGORICAL = ORIGINAL_LINEAR_CATEGORICAL
    e14_module.HGB_CATEGORICAL = ORIGINAL_HGB_CATEGORICAL
    return make_hgb(features)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=ROOT / "open/data/train.csv")
    parser.add_argument(
        "--validation-seasons", nargs="+", type=int, default=[2022, 2023, 2024]
    )
    parser.add_argument("--output-dir", type=Path, default=ROOT / "experiments/results")
    parser.add_argument("--prior-mode", default="r_recent3")
    return parser.parse_args()


def attach_row_ids(frame: pd.DataFrame, path: Path) -> pd.DataFrame:
    row_ids = pd.read_csv(path, usecols=["row_id"], dtype="string", encoding="utf-8-sig")[
        "row_id"
    ]
    if len(row_ids) != len(frame):
        raise AssertionError("row_id length does not match train frame")
    frame = frame.copy()
    frame.insert(0, "row_id", row_ids.to_numpy())
    return frame


def run_fold(
    frame: pd.DataFrame,
    season: int,
    labels: pd.Series,
    args: argparse.Namespace,
) -> dict[str, Any]:
    started = time.perf_counter()
    history = frame.loc[frame["season"] < season].copy()
    valid = frame.loc[frame["season"] == season].copy()
    history["e22_pitch_type_group"] = history["row_id"].map(labels)
    valid["e22_pitch_type_group"] = valid["row_id"].map(labels)
    prior = float(candidate_priors(history, season)[args.prior_mode])
    states_before, final_state = season_end_state(history)
    train_priors = prior_before_each_season(history)
    train_e14, _ = build_e14_features(history, states_before, train_priors, prior, k=E14_K)
    valid_e14, _ = build_e14_features(valid, {season: final_state}, {season: prior}, prior, k=E14_K)

    role_before, role_final = role_states_before_each_season(history)
    role_before[season] = role_final
    train_e16, _ = build_e16_features(history, role_before)
    valid_e16, _ = build_e16_features(valid, role_before)

    stage1 = make_stage1()
    labeled = history.dropna(subset=["e22_pitch_type_group"])
    stage1.fit(labeled[E22_FEATURES], labeled["e22_pitch_type_group"].astype(str))

    train_e22 = pd.DataFrame(
        align_group_probabilities(stage1, history),
        columns=E22_PROB_FEATURES,
        index=history.index,
    )
    valid_e22 = pd.DataFrame(
        align_group_probabilities(stage1, valid),
        columns=E22_PROB_FEATURES,
        index=valid.index,
    )
    del stage1
    gc.collect()

    s4_train = pd.concat([history[BASE_FEATURES], train_e14], axis=1)
    s4_valid = pd.concat([valid[BASE_FEATURES], valid_e14], axis=1)
    e16_train = pd.concat([s4_train, train_e16], axis=1)
    e16_valid = pd.concat([s4_valid, valid_e16], axis=1)
    e22_train = pd.concat([s4_train, train_e22], axis=1)
    e22_valid = pd.concat([s4_valid, valid_e22], axis=1)
    train_y = history[TARGET].to_numpy(dtype=np.int8, copy=False)
    valid_y = valid[TARGET].to_numpy(dtype=np.int8, copy=False)

    predictions: dict[str, np.ndarray] = {}
    fit_details: dict[str, Any] = {}
    for key, train_x, valid_x, factories in (
        ("s4", s4_train, s4_valid, [(make_base_linear, "linear"), (make_base_hgb, "hgb")]),
        ("e16", e16_train, e16_valid, [(make_e16_linear, "linear"), (make_e16_hgb, "hgb")]),
        ("e22", e22_train, e22_valid, [(make_base_linear, "linear"), (make_base_hgb, "hgb")]),
    ):
        component = {}
        for factory, name in factories:
            prediction, details = fit_predict(
                f"{season}/ensemble/{key}/{name}", factory, train_x, train_y, valid_x
            )
            component[name] = prediction
            fit_details[f"{key}_{name}"] = details
        predictions[key] = 0.9 * component["linear"] + 0.1 * component["hgb"]
        del component

    # S7 explicit mixture: group-specific target models on S4 columns.
    labeled_mask = history["e22_pitch_type_group"].notna().to_numpy(dtype=bool)
    mixture_components = {"linear": [], "hgb": []}
    for name, factory in (("linear", make_base_linear), ("hgb", make_base_hgb)):
        group_predictions = []
        for group in GROUPS:
            group_mask = labeled_mask & history["e22_pitch_type_group"].eq(group).to_numpy(
                dtype=bool, na_value=False
            )
            prediction, details = fit_predict(
                f"{season}/ensemble/mixture/{name}/{group}",
                factory,
                s4_train.loc[group_mask],
                train_y[group_mask],
                s4_valid,
            )
            group_predictions.append(prediction)
            fit_details[f"mixture_{name}_{group}"] = details
        conditional = np.column_stack(group_predictions)
        mixture_components[name] = np.sum(valid_e22.to_numpy(dtype=np.float64) * conditional, axis=1)
        del conditional, group_predictions
    predictions["s7"] = 0.9 * mixture_components["linear"] + 0.1 * mixture_components["hgb"]

    base = metric(valid_y, predictions["s4"])
    weights = {
        "M0_s4": {"s4": 1.0, "e16": 0.0, "e22": 0.0, "s7": 0.0},
        "M1_conservative": {"s4": 0.80, "e16": 0.05, "e22": 0.10, "s7": 0.05},
        "M2_soft_focus": {"s4": 0.70, "e16": 0.05, "e22": 0.20, "s7": 0.05},
        "M3_mixture_focus": {"s4": 0.70, "e16": 0.05, "e22": 0.10, "s7": 0.15},
        "M4_balanced": {"s4": 0.60, "e16": 0.10, "e22": 0.20, "s7": 0.10},
        "M5_no_e16": {"s4": 0.70, "e16": 0.00, "e22": 0.20, "s7": 0.10},
    }
    ensemble_metrics: dict[str, Any] = {}
    for name, allocation in weights.items():
        blended = sum(allocation[key] * predictions[key] for key in allocation)
        summary = metric(valid_y, blended)
        ensemble_metrics[name] = {
            "weights": allocation,
            "summary": summary,
            "brier_delta_vs_s4": float(summary["brier"] - base["brier"]),
            "score_delta_vs_s4": float(summary["competition_score"] - base["competition_score"]),
        }
    print(
        f"[{season}] S4 Brier={base['brier']:.8f}; "
        + ", ".join(
            f"{name}={item['brier_delta_vs_s4']:+.8f}"
            for name, item in ensemble_metrics.items()
            if name != "M0_s4"
        ),
        flush=True,
    )
    result = {
        "validation_season": season,
        "history_rows": len(history),
        "valid_rows": len(valid),
        "baseline_s4": base,
        "ensembles": ensemble_metrics,
        "stage1_labeled_rows": int(len(labeled)),
        "fit_details": fit_details,
        "elapsed_seconds": time.perf_counter() - started,
    }
    del history, valid, train_e14, valid_e14, train_e16, valid_e16, train_e22, valid_e22
    del s4_train, s4_valid, e16_train, e16_valid, e22_train, e22_valid, predictions
    gc.collect()
    return result


def main() -> None:
    args = parse_args()
    started = time.perf_counter()
    labels = load_group_labels()
    frame = attach_row_ids(load_train(args.data), args.data)
    folds = [run_fold(frame, season, labels, args) for season in sorted(args.validation_seasons)]
    del frame, labels
    gc.collect()
    names = list(folds[0]["ensembles"])
    aggregate: dict[str, Any] = {}
    for name in names:
        deltas = [float(fold["ensembles"][name]["brier_delta_vs_s4"]) for fold in folds]
        scores = [float(fold["ensembles"][name]["score_delta_vs_s4"]) for fold in folds]
        wins = sum(delta < 0.0 for delta in deltas)
        aggregate[name] = {
            "weights": folds[0]["ensembles"][name]["weights"],
            "wins_vs_s4": wins,
            "mean_brier_delta_vs_s4": float(np.mean(deltas)),
            "worst_brier_delta_vs_s4": float(np.max(deltas)),
            "mean_score_delta_vs_s4": float(np.mean(scores)),
            "gate_pass_vs_s4": bool(wins >= 2 and np.max(deltas) <= 0.0005),
        }
    payload = {
        "metadata": {
            "created_at_utc": pd.Timestamp.utcnow().isoformat(),
            "data": str(args.data),
            "validation_seasons": sorted(args.validation_seasons),
            "prior_mode": args.prior_mode,
            "protocol": "fixed weights selected before fold; S4/S5/S6/S7 predictions rebuilt on each outer history",
            "row_independent_inference": True,
            "candidate_sources": ["S4", "S5", "S6", "S7"],
            "versions": {
                "python": platform.python_version(),
                "numpy": np.__version__,
                "pandas": pd.__version__,
            },
            "command": " ".join(sys.argv),
            "elapsed_seconds": time.perf_counter() - started,
        },
        "aggregate": aggregate,
        "folds": folds,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.output_dir / "final_ensemble_rolling.json"
    csv_path = args.output_dir / "final_ensemble_rolling.csv"
    json_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8"
    )
    rows = []
    for fold in folds:
        for name, item in fold["ensembles"].items():
            rows.append(
                {
                    "validation_season": fold["validation_season"],
                    "ensemble": name,
                    "brier": item["summary"]["brier"],
                    "brier_delta_vs_s4": item["brier_delta_vs_s4"],
                    "score_delta_vs_s4": item["score_delta_vs_s4"],
                }
            )
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    print(json.dumps(aggregate, ensure_ascii=False, indent=2), flush=True)
    print(f"Saved {json_path} and {csv_path}.", flush=True)


if __name__ == "__main__":
    main()
