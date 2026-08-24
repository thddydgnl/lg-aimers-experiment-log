#!/usr/bin/env python3
"""Persist the frozen V3 M3 OOF anchor with one consistent artifact name."""

from __future__ import annotations

from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
PRED = ROOT / "experiments/results/predictions"
WEIGHTS = {"A": 0.501443851662535, "C": 0.27016033407769313, "B": 0.22839581425977187}
SOURCES = {
    2022: {
        "A": "v3_sparse_a_backtest_2022.npz",
        "B": "v3_sparse_b_backtest_2022.npz",
        "C": "v3_sparse_c_backtest_2022.npz",
    },
    2023: {
        "A": "v3_sparse_a_backtest_2023.npz",
        "B": "v3_sparse_b_backtest_2023.npz",
        "C": "v3_sparse_c_backtest_2023.npz",
    },
    2024: {
        "A": "v3_outcome_trackmanrich_overall_components120_e14k50_batter80_middle100_2024.npz",
        "B": "v3_outcome_batter80_middle100_hgroups500_2024.npz",
        "C": "v3_outcome_trackmanrich_overall_e14k50_batter80_middle100_2024.npz",
    },
}


def load(name: str) -> dict[str, np.ndarray]:
    with np.load(PRED / name) as archive:
        return {key: np.asarray(archive[key]) for key in archive.files}


def main() -> None:
    for year, sources in SOURCES.items():
        artifacts = {name: load(filename) for name, filename in sources.items()}
        reference = artifacts["A"]
        for name, artifact in artifacts.items():
            for key in ("y", "row_index", "cluster"):
                if not np.array_equal(reference[key], artifact[key]):
                    raise ValueError(f"{year}/{name} alignment mismatch: {key}")
        raw = sum(
            WEIGHTS[name] * artifacts[name]["catboost_outcome"].astype(np.float64)
            for name in WEIGHTS
        )
        prediction = np.clip(0.5 + 1.05 * (raw - 0.5) - 0.006, 1e-6, 1.0 - 1e-6)
        path = PRED / f"v3_sparse_m3_frozen_{year}.npz"
        np.savez_compressed(
            path,
            y=reference["y"],
            row_index=reference["row_index"],
            cluster=reference["cluster"],
            final_prediction=prediction,
        )
        print(path)


if __name__ == "__main__":
    main()
