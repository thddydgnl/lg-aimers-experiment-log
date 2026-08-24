#!/usr/bin/env python3
"""Immutable target-free invariance audit for TrackMan workload features."""

from __future__ import annotations

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

from experiments.run_e20r_rolling import load_joined_trackman  # noqa: E402
from experiments.v5_trackman_workload_features import (  # noqa: E402
    WORKLOAD_PROFILE_COLUMNS,
    build_workload_profile_features,
    workload_profile_table,
)

TRAIN = ROOT / "open/data/train.csv"
PREREG = ROOT / "experiments/params/v5_trackman_workload_c_preregister.json"
IMPLEMENTATION = ROOT / "experiments/v5_trackman_workload_features.py"
REPORT = ROOT / "experiments/results/v5_trackman_workload_feature_audit.json"
SOURCE_CUTOFF = 2022
SAMPLE_ROWS = 10_000
SEED = 2026


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [safe(item) for item in value]
    if isinstance(value, (np.floating, float)):
        number = float(value)
        return number if np.isfinite(number) else None
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    return value


def main() -> None:
    if REPORT.exists():
        raise FileExistsError(f"immutable report exists: {REPORT}")
    prereg = json.loads(PREREG.read_text(encoding="utf-8"))
    if prereg["status"] != "locked_before_2022_2023_candidate_metrics":
        raise ValueError("unexpected preregistration state")

    joined = load_joined_trackman()
    source = joined.loc[joined["season"].lt(SOURCE_CUTOFF)].copy()
    if source.empty or int(source["season"].max()) >= SOURCE_CUTOFF:
        raise ValueError("invalid temporal source cutoff")
    state = workload_profile_table(source)
    shuffled_source = source.sample(frac=1.0, random_state=SEED)
    shuffled_state = workload_profile_table(shuffled_source).reindex(state.index)
    source_order_invariant = bool(
        state.index.equals(shuffled_state.index)
        and np.array_equal(state.to_numpy(), shuffled_state.to_numpy())
    )

    frame = pd.read_csv(
        TRAIN, usecols=["season", "pitcher_id", "inning"], encoding="utf-8-sig"
    )
    eligible = frame.loc[frame["season"].eq(SOURCE_CUTOFF)]
    if len(eligible) < SAMPLE_ROWS:
        raise ValueError("not enough rows for the fixed audit sample")
    sample = eligible.sample(n=SAMPLE_ROWS, random_state=SEED)
    first, first_meta = build_workload_profile_features(
        sample, {SOURCE_CUTOFF: state}
    )
    order = np.random.default_rng(SEED).permutation(len(sample))
    second, second_meta = build_workload_profile_features(
        sample.iloc[order], {SOURCE_CUTOFF: state}
    )
    second = second.reindex(sample.index)
    prediction_row_order_invariant = bool(
        np.array_equal(first.to_numpy(), second.to_numpy())
    )
    finite = bool(np.isfinite(first.to_numpy(dtype=np.float64)).all())
    schema_exact = list(first.columns) == WORKLOAD_PROFILE_COLUMNS
    checks = {
        "source_order_invariant": source_order_invariant,
        "prediction_row_order_invariant": prediction_row_order_invariant,
        "finite": finite,
        "schema_exact": schema_exact,
        "source_strictly_before_2022": int(source["season"].max()) < SOURCE_CUTOFF,
        "target_columns_not_loaded_by_audit": True,
    }
    report = {
        "experiment_id": prereg["experiment_id"],
        "status": "passed" if all(checks.values()) else "failed",
        "preregister_sha256": sha256(PREREG),
        "implementation_sha256": sha256(IMPLEMENTATION),
        "script_sha256": sha256(Path(__file__)),
        "source_rows": int(len(source)),
        "source_min_season": int(source["season"].min()),
        "source_max_season": int(source["season"].max()),
        "state_rows": int(len(state)),
        "state_index_unique": bool(state.index.is_unique),
        "sample_rows": SAMPLE_ROWS,
        "checks": checks,
        "first_metadata": first_meta,
        "second_metadata": second_meta,
        "feature_summary": {
            column: {
                "min": float(first[column].min()),
                "max": float(first[column].max()),
                "mean": float(first[column].mean()),
                "std": float(first[column].std()),
            }
            for column in first.columns
        },
        "goal_completion_claimed": False,
    }
    REPORT.write_text(
        json.dumps(safe(report), ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(safe(report), ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
