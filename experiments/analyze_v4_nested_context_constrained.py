#!/usr/bin/env python3
"""Run the nested-context expansion with conservative nonnegative weights."""

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments import analyze_v4_nested_context_expansion as experiment  # noqa: E402


experiment.WEIGHT_GRID = (
    0.0,
    0.05,
    0.10,
    0.20,
    0.35,
    0.50,
    0.65,
    0.80,
    1.00,
    1.25,
)
experiment.OUTPUT_JSON = (
    experiment.ROOT / "experiments/results/v4_nested_context_constrained.json"
)
experiment.OUTPUT_NPZ = (
    experiment.ROOT
    / "experiments/results/predictions/v4_nested_context_constrained_2024.npz"
)


if __name__ == "__main__":
    experiment.main()
