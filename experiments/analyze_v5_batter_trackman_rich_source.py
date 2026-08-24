#!/usr/bin/env python3
"""Locked source gate for V5 batter TrackMan profiles."""

from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.stats import paired_bootstrap_brier_ci


PRED = ROOT / "experiments/results/predictions"
TRAIN = ROOT / "open/data/train.csv"
REPORT = ROOT / "experiments/results/v5_batter_trackman_rich_source_gate.json"
PREREG = ROOT / "experiments/params/v5_batter_trackman_rich_preregister.json"
YEARS = (2020, 2021)


def load(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        return {key: np.asarray(archive[key]) for key in archive.files}


def raw_score(y: np.ndarray, prediction: np.ndarray, mask: np.ndarray) -> float:
    target = np.asarray(y[mask], dtype=np.float64)
    values = np.clip(np.asarray(prediction[mask], dtype=np.float64), 1e-6, 1 - 1e-6)
    reference = float(target.mean() * (1.0 - target.mean()))
    return 100_000.0 * (
        1.0 - float(np.mean((values - target) ** 2)) / reference
    )


def interval(
    y: np.ndarray,
    parent: np.ndarray,
    candidate: np.ndarray,
    clusters: np.ndarray,
    mask: np.ndarray,
    seed: int,
) -> dict:
    target = y[mask]
    reference = float(target.mean() * (1.0 - target.mean()))
    return paired_bootstrap_brier_ci(
        target,
        parent[mask],
        candidate[mask],
        iterations=2000,
        seed=seed,
        clusters=clusters[mask],
        reference_brier=reference,
    )


def main() -> None:
    prereg = json.loads(PREREG.read_text(encoding="utf-8"))
    game_type = pd.read_csv(TRAIN, usecols=["game_type"])["game_type"].to_numpy()
    artifacts = {}
    for year in YEARS:
        parent = load(PRED / f"v5_exact_c_multiseed_source_{year}.npz")
        candidate = load(PRED / f"v5_batter_trackman_rich_source_{year}.npz")
        for key in ("y", "row_index", "cluster"):
            if not np.array_equal(parent[key], candidate[key]):
                raise ValueError(f"alignment mismatch {year}/{key}")
        artifacts[year] = {
            "y": parent["y"].astype(np.float64),
            "row_index": parent["row_index"],
            "cluster": parent["cluster"],
            "parent": parent["parent_exact_c"].astype(np.float64),
            "candidate": candidate["catboost_outcome"].astype(np.float64),
        }

    trials = []
    for gamma in prereg["route_and_selection"]["gamma_grid"]:
        gains = {}
        flat = []
        for year in YEARS:
            item = artifacts[year]
            route = game_type[item["row_index"]] == "R"
            prediction = item["parent"].copy()
            prediction[route] += float(gamma) * (
                item["candidate"][route] - item["parent"][route]
            )
            gain = raw_score(item["y"], prediction, route) - raw_score(
                item["y"], item["parent"], route
            )
            gains[str(year)] = float(gain)
            flat.append(gain)
        trials.append({
            "gamma": float(gamma),
            "R_gains": gains,
            "minimum_R_gain": float(min(flat)),
            "median_R_gain": float(np.median(flat)),
        })
    selected = sorted(
        trials,
        key=lambda row: (
            -float(row["minimum_R_gain"]),
            -float(row["median_R_gain"]),
            float(row["gamma"]),
        ),
    )[0]
    gamma = float(selected["gamma"])

    per_year = {}
    gates = []
    selected_predictions = {}
    for position, year in enumerate(YEARS):
        item = artifacts[year]
        all_mask = np.ones(len(item["y"]), dtype=bool)
        r_mask = game_type[item["row_index"]] == "R"
        prediction = item["parent"].copy()
        prediction[r_mask] += gamma * (
            item["candidate"][r_mask] - item["parent"][r_mask]
        )
        prediction = np.clip(prediction, 1e-6, 1 - 1e-6)
        full_gain = raw_score(item["y"], prediction, all_mask) - raw_score(
            item["y"], item["parent"], all_mask
        )
        r_gain = raw_score(item["y"], prediction, r_mask) - raw_score(
            item["y"], item["parent"], r_mask
        )
        full_ci = interval(
            item["y"], item["parent"], prediction, item["cluster"], all_mask,
            20260822 + position,
        )
        r_ci = interval(
            item["y"], item["parent"], prediction, item["cluster"], r_mask,
            20260922 + position,
        )
        row_gate = {
            "full_point": bool(
                full_gain
                >= prereg["source_gate"]["minimum_full_point_gain_each_year"]
            ),
            "R_point": bool(
                r_gain >= prereg["source_gate"]["minimum_R_point_gain_each_year"]
            ),
            "R_ci_lower": bool(
                float(r_ci["score_ci_low"])
                > prereg["source_gate"][
                    "R_pitcher_cluster_95pct_gain_lower_bound_each_year"
                ]
            ),
        }
        gates.extend(row_gate.values())
        per_year[str(year)] = {
            "full_parent_score": raw_score(item["y"], item["parent"], all_mask),
            "full_candidate_score": raw_score(item["y"], prediction, all_mask),
            "full_gain": float(full_gain),
            "R_parent_score": raw_score(item["y"], item["parent"], r_mask),
            "R_candidate_score": raw_score(item["y"], prediction, r_mask),
            "R_gain": float(r_gain),
            "full_interval": full_ci,
            "R_interval": r_ci,
            "gate": row_gate,
        }
        selected_predictions[year] = prediction
        np.savez_compressed(
            PRED / f"v5_batter_trackman_rich_selected_source_{year}.npz",
            y=item["y"],
            row_index=item["row_index"],
            cluster=item["cluster"],
            parent=item["parent"],
            final_prediction=prediction,
        )

    report = {
        "experiment_id": prereg["experiment_id"],
        "protocol": {
            "source_years": list(YEARS),
            "2022_plus_loaded": False,
            "test_rows_read": False,
            "route": "R only; exact parent unchanged on F",
        },
        "gamma_trials": trials,
        "selected": selected,
        "per_year": per_year,
        "source_gate_pass": bool(all(gates)),
        "decision": (
            "advance_to_locked_2022_2023_development"
            if all(gates)
            else "close_before_2022_plus"
        ),
        "goal_status": "active",
        "goal_completion_claimed": False,
    }
    REPORT.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    print(json.dumps({
        "selected": selected,
        "per_year": {
            year: {
                "full_gain": values["full_gain"],
                "R_gain": values["R_gain"],
                "R_ci_low": values["R_interval"]["score_ci_low"],
            }
            for year, values in per_year.items()
        },
        "pass": report["source_gate_pass"],
    }, ensure_ascii=False, indent=2, allow_nan=False))
    print(f"Saved {REPORT}")


if __name__ == "__main__":
    main()
