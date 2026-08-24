#!/usr/bin/env python3
"""Target-free stability audit of the fixed 2019 TrackMan archetype basis."""

from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.run_e20r_rolling import load_joined_trackman, rich_profile_table
from experiments.v5_trackman_archetype_features import (
    ARCHETYPE_CELL,
    PC_COLUMNS,
    fit_archetype_basis,
    transform_profile_table,
)


OUTPUT = ROOT / "experiments/results/v5_trackman_archetype_stability.json"


def main() -> None:
    joined = load_joined_trackman()
    basis = fit_archetype_basis(joined)
    transformed = {}
    cutoffs = {}
    for cutoff in range(2020, 2026):
        profile = rich_profile_table(
            joined.loc[
                joined["game_type"].eq("R") & joined["season"].lt(cutoff)
            ]
        )
        representation = transform_profile_table(profile, basis)
        transformed[cutoff] = representation
        cutoffs[str(cutoff)] = {
            "history_seasons": sorted(
                int(value)
                for value in joined.loc[joined["season"].lt(cutoff), "season"].unique()
            ),
            "pitchers": int(len(representation)),
            "cluster_counts": {
                str(key): int(value)
                for key, value in representation[ARCHETYPE_CELL]
                .value_counts()
                .sort_index()
                .items()
            },
            "median_distance": float(
                representation["e82_archetype_distance"].median()
            ),
            "median_margin": float(
                representation["e82_archetype_margin"].median()
            ),
        }

    transitions = {}
    for previous, current in zip(range(2020, 2025), range(2021, 2026)):
        common = transformed[previous].index.intersection(transformed[current].index)
        left = transformed[previous].loc[common]
        right = transformed[current].loc[common]
        difference = (
            left.loc[:, PC_COLUMNS].to_numpy(dtype=np.float64)
            - right.loc[:, PC_COLUMNS].to_numpy(dtype=np.float64)
        )
        drift = np.sqrt(np.sum(np.square(difference), axis=1))
        transitions[f"{previous}_to_{current}"] = {
            "common_pitchers": int(len(common)),
            "cluster_agreement": float(
                np.mean(
                    left[ARCHETYPE_CELL].astype(str).to_numpy()
                    == right[ARCHETYPE_CELL].astype(str).to_numpy()
                )
            ),
            "pc_drift_median": float(np.median(drift)),
            "pc_drift_p90": float(np.quantile(drift, 0.9)),
        }

    report = {
        "experiment_id": "V5_TRACKMAN_ARCHETYPE_C_V1",
        "mode": "target_free_representation_audit",
        "control_success_accessed": False,
        "basis_source_year": 2019,
        "basis_source_pitchers": basis.source_pitchers,
        "source_feature_count": len(basis.source_feature_names),
        "pca_components": len(PC_COLUMNS),
        "explained_variance_ratio": basis.pca.explained_variance_ratio_.tolist(),
        "explained_variance_total": float(basis.pca.explained_variance_ratio_.sum()),
        "cutoffs": cutoffs,
        "transitions": transitions,
    }
    if OUTPUT.exists():
        raise FileExistsError(f"Audit already exists: {OUTPUT}")
    OUTPUT.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({
        "basis_source_pitchers": report["basis_source_pitchers"],
        "explained_variance_total": report["explained_variance_total"],
        "transitions": transitions,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
