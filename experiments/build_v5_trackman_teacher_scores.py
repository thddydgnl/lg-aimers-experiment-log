#!/usr/bin/env python3
"""Build strictly out-of-time historical TrackMan teacher scores for V5.

Each source season is already completed when its physical measurements are
converted into soft scores.  The teacher is fit only on earlier seasons, and
the source season's labels are used for diagnostics only, never for fitting or
for profile construction.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Importing this module loads LightGBM before pandas/sklearn on Windows.
from experiments.run_v5_privileged_trackman_teacher_feasibility import (
    CONTROL_FEATURES,
    FULL_FEATURES,
    TARGET,
    clustered_interval,
    fit_predict,
    load_joined_trackman,
    metrics,
)
import numpy as np


PREREG = ROOT / "experiments/params/v5_privileged_trackman_distill_c_preregister.json"
FEASIBILITY = ROOT / "experiments/results/v5_privileged_trackman_teacher_feasibility.json"
OUTPUT_NPZ = ROOT / "experiments/results/v5_trackman_teacher_scores_dev.npz"
OUTPUT_JSON = ROOT / "experiments/results/v5_trackman_teacher_scores_dev.json"
SOURCE_YEARS = (2020, 2021, 2022)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    prereg = json.loads(PREREG.read_text(encoding="utf-8"))
    feasibility = json.loads(FEASIBILITY.read_text(encoding="utf-8"))
    if prereg["status"] != "locked_before_teacher_score_generation":
        raise ValueError("Unexpected preregistration status")
    if feasibility["status"] != "passed_proceed_to_history_only_distillation":
        raise ValueError("Feasibility prerequisite did not pass")
    if OUTPUT_NPZ.exists() or OUTPUT_JSON.exists():
        raise FileExistsError(
            "Teacher-score artifact already exists; preserve it instead of overwriting"
        )

    joined = load_joined_trackman()
    rows = joined.loc[
        joined["game_type"].eq("R")
        & joined["season"].le(max(SOURCE_YEARS))
        & joined[TARGET].notna()
    ].copy()
    del joined

    arrays: dict[str, list[np.ndarray]] = {
        "row_id": [],
        "season": [],
        "pitcher_id": [],
        "pitch_type_group": [],
        "balls_before": [],
        "strikes_before": [],
        "physics_teacher": [],
        "control_teacher": [],
    }
    folds: dict[str, dict[str, object]] = {}
    for year in SOURCE_YEARS:
        history = rows.loc[rows["season"].lt(year)].copy()
        source = rows.loc[rows["season"].eq(year)].copy()
        if history.empty or source.empty:
            raise ValueError(f"Empty teacher-score fold for {year}")
        print(
            f"teacher score source={year}: train={len(history):,} source={len(source):,}",
            flush=True,
        )
        control = fit_predict(history, source, CONTROL_FEATURES)
        full = fit_predict(history, source, FULL_FEATURES)
        y = source[TARGET].to_numpy(dtype=np.float64)
        interval = clustered_interval(
            y,
            control,
            full,
            source["pitcher_id"].to_numpy(),
            seed=20262821 + year,
        )
        folds[str(year)] = {
            "history_seasons": sorted(
                int(value) for value in history["season"].unique()
            ),
            "train_rows": int(len(history)),
            "source_rows": int(len(source)),
            "source_labels_used_for_fit_or_stored_profile": False,
            "control": metrics(y, control),
            "physics_teacher": metrics(y, full),
            "paired_normalized_brier_gain": interval,
        }
        arrays["row_id"].append(source["row_id"].astype(str).to_numpy(dtype=str))
        arrays["season"].append(source["season"].to_numpy(dtype=np.int16))
        arrays["pitcher_id"].append(source["pitcher_id"].to_numpy(dtype=np.int64))
        arrays["pitch_type_group"].append(
            source["pitch_type_group"].fillna("other").astype(str).to_numpy(dtype=str)
        )
        arrays["balls_before"].append(source["balls_before"].to_numpy(dtype=np.int8))
        arrays["strikes_before"].append(source["strikes_before"].to_numpy(dtype=np.int8))
        arrays["physics_teacher"].append(full.astype(np.float32))
        arrays["control_teacher"].append(control.astype(np.float32))
        print(
            f"  paired gain={interval['point']:.3f} "
            f"CI=[{interval['lower_95']:.3f}, {interval['upper_95']:.3f}]",
            flush=True,
        )

    combined = {key: np.concatenate(parts) for key, parts in arrays.items()}
    if len(np.unique(combined["row_id"])) != len(combined["row_id"]):
        raise AssertionError("Teacher-score row_id values are not unique")
    OUTPUT_NPZ.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(OUTPUT_NPZ, **combined)
    payload = {
        "experiment_id": prereg["experiment_id"],
        "status": "development_teacher_scores_built",
        "protocol": {
            "official_data_only": True,
            "source_year_labels_used_for_teacher_fit": False,
            "source_year_labels_stored_or_used_in_profiles": False,
            "current_or_validation_trackman_at_inference": False,
            "source_2023_generated": False,
            "2024_candidate_scored": False,
            "test_rows_read": False,
        },
        "source_years": list(SOURCE_YEARS),
        "rows": int(len(combined["season"])),
        "folds": folds,
        "preregister": str(PREREG.relative_to(ROOT)),
        "preregister_sha256": sha256(PREREG),
        "feasibility_sha256": sha256(FEASIBILITY),
        "artifact": str(OUTPUT_NPZ.relative_to(ROOT)),
        "artifact_sha256": sha256(OUTPUT_NPZ),
    }
    OUTPUT_JSON.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"wrote {OUTPUT_NPZ} and {OUTPUT_JSON}", flush=True)


if __name__ == "__main__":
    main()
