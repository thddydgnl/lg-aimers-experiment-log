#!/usr/bin/env python3
"""Persist the current V4 champion for all reusable temporal folds."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.analyze_v4_decayed_differentials import (  # noqa: E402
    add_columns,
    add_differential_columns,
    add_expansion_columns,
    champion_predictions,
)
from experiments.analyze_v4_temporal_residual_ridge import (  # noqa: E402
    MEDIAN_OFFSET,
    load_frames,
    score,
)


PREDICTIONS = ROOT / "experiments/results/predictions"
OUTPUT_JSON = ROOT / "experiments/results/v4_champion_pre_stack.json"


def main() -> None:
    frames, artifacts = load_frames()
    add_columns(frames, artifacts)
    add_expansion_columns(frames, artifacts)
    add_differential_columns(frames)
    predictions = champion_predictions(frames, artifacts)
    metrics = {}
    outputs = {}
    for year, prediction in predictions.items():
        metric = score(artifacts[year]["y"], prediction)
        metrics[str(year)] = metric
        path = PREDICTIONS / f"v4_champion_pre_stack_{year}.npz"
        np.savez_compressed(
            path,
            y=artifacts[year]["y"],
            row_index=artifacts[year]["row_index"],
            cluster=artifacts[year]["cluster"],
            m3=artifacts[year]["m3"],
            champion=prediction,
        )
        outputs[str(year)] = str(path.relative_to(ROOT))
        print(f"[{year}] local={metric['raw_competition_score']:.6f}", flush=True)
    report = {
        "protocol": {
            "official_train_only": True,
            "test_rows_read": False,
            "leaderboard_values_used": False,
            "row_independent": True,
        },
        "metrics": metrics,
        "expected_lb_2024": float(
            metrics["2024"]["raw_competition_score"] + MEDIAN_OFFSET
        ),
        "artifacts": outputs,
    }
    OUTPUT_JSON.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"Saved {OUTPUT_JSON}")


if __name__ == "__main__":
    main()
