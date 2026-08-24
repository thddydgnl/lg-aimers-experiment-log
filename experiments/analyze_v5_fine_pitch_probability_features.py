#!/usr/bin/env python3
"""Apply the preregistered source gate to fine-pitch probability features."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.run_v5_h1_residual import (  # noqa: E402
    cluster_bootstrap_score_gain,
)


PREDICTIONS = ROOT / "experiments/results/predictions"
PREREGISTRATION = (
    ROOT / "experiments/params/v5_fine_pitch_probability_features_preregister.json"
)
OUTPUT = ROOT / "experiments/results/v5_fine_pitch_probability_features_gate.json"
YEARS = (2020, 2021)


def load(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        return {key: np.asarray(archive[key]) for key in archive.files}


def metric(y: np.ndarray, prediction: np.ndarray, mask: np.ndarray) -> dict[str, float | int]:
    target = y[mask].astype(np.float64)
    pred = prediction[mask].astype(np.float64)
    rate = float(target.mean())
    brier = float(np.mean(np.square(pred - target)))
    return {
        "rows": int(mask.sum()),
        "target_rate": rate,
        "prediction_mean": float(pred.mean()),
        "brier": brier,
        "score": float(100_000.0 * (1.0 - brier / (rate * (1.0 - rate)))),
    }


def main() -> None:
    if OUTPUT.exists():
        raise FileExistsError(f"result already exists: {OUTPUT}")
    game_type_all = pd.read_csv(
        ROOT / "open/data/train.csv", usecols=["game_type"], low_memory=False
    )["game_type"].to_numpy()
    years: dict[str, object] = {}
    for offset, year in enumerate(YEARS):
        anchor = load(PREDICTIONS / f"v4_m3_c_backtest_{year}_{year}.npz")
        candidate = load(
            PREDICTIONS / f"v5_fine_pitch_probability_features_source_{year}.npz"
        )
        for key in ("y", "row_index", "cluster"):
            if not np.array_equal(anchor[key], candidate[key]):
                raise ValueError(f"alignment mismatch for {year}/{key}")
        game_type = game_type_all[anchor["row_index"]]
        masks = {
            "all": np.ones(len(game_type), dtype=bool),
            "R": game_type == "R",
        }
        anchor_prediction = anchor["catboost_outcome"].astype(np.float64)
        candidate_prediction = candidate["catboost_outcome"].astype(np.float64)
        anchor_metrics = {
            scope: metric(anchor["y"], anchor_prediction, mask)
            for scope, mask in masks.items()
        }
        candidate_metrics = {
            scope: metric(anchor["y"], candidate_prediction, mask)
            for scope, mask in masks.items()
        }
        years[str(year)] = {
            "anchor": anchor_metrics,
            "candidate": candidate_metrics,
            "gains": {
                scope: float(candidate_metrics[scope]["score"] - anchor_metrics[scope]["score"])
                for scope in masks
            },
            "bootstrap_R": cluster_bootstrap_score_gain(
                anchor["y"],
                anchor_prediction,
                candidate_prediction,
                anchor["cluster"].astype(str),
                masks["R"],
                2000,
                592000 + offset,
            ),
        }
    conditions = {
        "minimum_full_gain_each_year": bool(
            all(years[str(year)]["gains"]["all"] >= 5.0 for year in YEARS)
        ),
        "minimum_R_gain_each_year": bool(
            all(years[str(year)]["gains"]["R"] >= 5.0 for year in YEARS)
        ),
        "ci_lower_positive_each_year": bool(
            all(years[str(year)]["bootstrap_R"]["ci_low"] > 0.0 for year in YEARS)
        ),
    }
    gate_pass = all(conditions.values())
    payload = {
        "experiment_id": "V5_FINE_PITCH_PROBABILITY_FEATURES_V1",
        "status": "source_gate_pass" if gate_pass else "failed_source_gate",
        "preregister_sha256": hashlib.sha256(PREREGISTRATION.read_bytes()).hexdigest(),
        "policy": {
            "test_rows_read": False,
            "latest_control_label_season_read": max(YEARS),
            "current_pitch_type_at_inference": False,
        },
        "years": years,
        "conditions": conditions,
        "gate_pass": gate_pass,
        "decision": "open preregistered 2022" if gate_pass else "close without 2022+",
    }
    OUTPUT.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
