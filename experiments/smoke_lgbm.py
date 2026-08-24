#!/usr/bin/env python3
"""Fast end-to-end smoke test for the project's LightGBM wrapper."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Keep this import first: run_v2_rolling intentionally loads the LightGBM DLL
# before pandas on Windows.
from experiments.run_v2_rolling import make_lgbm

import json

import numpy as np
import pandas as pd


def main() -> None:
    rng = np.random.default_rng(2026)
    rows = 2400
    frame = pd.DataFrame(
        {
            "game_type": np.where(np.arange(rows) % 3 == 0, "F", "R"),
            "batter_id": (np.arange(rows) % 71).astype(np.int32),
            "balls_before": (np.arange(rows) % 4).astype(np.int8),
            "asof_pitcher_success_rate": rng.uniform(0.3, 0.7, rows),
        }
    )
    label = (
        frame["asof_pitcher_success_rate"].to_numpy()
        + 0.03 * (frame["game_type"].to_numpy() == "R")
        + rng.normal(0.0, 0.08, rows)
        > 0.5
    ).astype(np.int32)
    features = list(frame.columns)
    model = make_lgbm(
        features,
        {
            "n_estimators": 30,
            "num_leaves": 15,
            "min_child_samples": 30,
            "n_jobs": 2,
        },
    )
    model.fit_time_ordered(
        frame.iloc[:1600],
        label[:1600],
        frame.iloc[1600:2000],
        label[1600:2000],
        refit_full=False,
    )
    prediction = model.predict_proba(frame.iloc[2000:])[:, 1]
    if prediction.shape != (400,) or not np.isfinite(prediction).all():
        raise RuntimeError("LightGBM smoke predictions are invalid")
    print(
        json.dumps(
            {
                "status": "PASSED",
                "rows": int(len(prediction)),
                "best_iteration": int(model.best_iteration_),
                "prediction_min": float(prediction.min()),
                "prediction_max": float(prediction.max()),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
