#!/usr/bin/env python3
"""Restore the pre-expansion tagged pitch-type candidate as a locked artifact."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.analyze_v4_temporal_residual_ridge import (  # noqa: E402
    MEDIAN_OFFSET,
    REQUIRED_LOCAL,
    json_safe,
    score,
)
from experiments.v4_current_ensemble import PREDICTIONS  # noqa: E402


SOURCE_REPORT = ROOT / "experiments/results/v4_pitchtype_failure_prior.json"
OUTPUT_REPORT = ROOT / "experiments/results/v4_pitchtype_failure_tagged_locked.json"
YEARS = (2022, 2023, 2024)


def main() -> None:
    source_report = json.loads(SOURCE_REPORT.read_text(encoding="utf-8"))
    source_order = list(source_report["selected_by_source"])
    if "tagged" not in source_order:
        raise ValueError("Expanded pitch-type report has no tagged source")
    tagged_index = source_order.index("tagged")
    source_metrics: dict[str, dict[str, dict[str, float | int]]] = {
        source: {} for source in source_order
    }
    locked_metrics: dict[str, dict[str, float | int]] = {}
    fold_artifacts: dict[str, str] = {}
    for year in YEARS:
        source_path = PREDICTIONS / f"v4_pitchtype_failure_prior_{year}.npz"
        with np.load(source_path) as archive:
            y = np.asarray(archive["y"])
            row_index = np.asarray(archive["row_index"])
            cluster = np.asarray(archive["cluster"])
            champion = np.asarray(archive["champion"], dtype=np.float64)
            directions = np.asarray(archive["source_directions"], dtype=np.float64)
        for source_index, source in enumerate(source_order):
            source_prediction = np.clip(
                champion + directions[:, source_index], 0.0, 1.0
            )
            source_metrics[source][str(year)] = score(y, source_prediction)
        tagged_direction = directions[:, tagged_index]
        locked = np.clip(champion + tagged_direction, 0.0, 1.0)
        locked_metrics[str(year)] = score(y, locked)
        output_path = (
            PREDICTIONS / f"v4_pitchtype_failure_tagged_locked_{year}.npz"
        )
        np.savez_compressed(
            output_path,
            y=y,
            row_index=row_index,
            cluster=cluster,
            champion=champion,
            tagged_direction=tagged_direction,
            tagged_locked=locked,
        )
        fold_artifacts[str(year)] = str(output_path.relative_to(ROOT))

    base_2024 = source_report["base_metrics"]["2024"]["raw_competition_score"]
    local_2024 = float(locked_metrics["2024"]["raw_competition_score"])
    report = {
        "protocol": {
            "status": "locked before multi-source expansion",
            "selection_folds": [2022, 2023],
            "confirmation_fold": 2024,
            "official_train_and_trackman_only": True,
            "test_rows_read": False,
            "current_or_validation_pitch_type_used": False,
            "row_independent": True,
        },
        "locked_source": "tagged",
        "locked_config": source_report["selected_by_source"]["tagged"],
        "source_diagnostic_metrics": source_metrics,
        "multi_source_expansion": {
            "status": "rejected",
            "selected_joint": source_report["selected_joint"],
            "confirmation_2024": source_report["confirmation_2024"],
        },
        "locked_metrics": locked_metrics,
        "locked_2024": {
            "gain_over_pre_pitchtype_champion": local_2024 - float(base_2024),
            "local_score": local_2024,
            "expected_lb_median": local_2024 + MEDIAN_OFFSET,
            "required_local_score": REQUIRED_LOCAL,
            "crosses_required_local_score": local_2024 > REQUIRED_LOCAL,
        },
        "fold_artifacts": fold_artifacts,
    }
    OUTPUT_REPORT.write_text(
        json.dumps(json_safe(report), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "locked_2024": report["locked_2024"],
                "source_2024_scores": {
                    source: metrics["2024"]["raw_competition_score"]
                    for source, metrics in source_metrics.items()
                },
            },
            ensure_ascii=False,
            indent=2,
        ),
        flush=True,
    )
    print(f"Saved {OUTPUT_REPORT}", flush=True)


if __name__ == "__main__":
    main()
