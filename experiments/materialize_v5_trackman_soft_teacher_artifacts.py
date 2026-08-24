#!/usr/bin/env python3
"""Materialize runner-compatible soft-label artifacts for the V5 student."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
PREREG = ROOT / "experiments/params/v5_privileged_trackman_soft_student_preregister.json"
SOURCE = ROOT / "experiments/results/v5_trackman_teacher_scores_dev.npz"
PREDICTIONS = ROOT / "experiments/results/predictions"
OUTPUT = ROOT / "experiments/results/v5_trackman_soft_teacher_artifacts.json"
YEARS = (2020, 2021, 2022)
STEM = "v5_trackman_physics_soft_teacher"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    prereg = json.loads(PREREG.read_text(encoding="utf-8"))
    if prereg["status"] != "locked_before_student_artifact_materialization":
        raise ValueError("Unexpected preregistration status")
    if OUTPUT.exists():
        raise FileExistsError("Soft-teacher manifest already exists; do not overwrite")
    with np.load(SOURCE, allow_pickle=False) as archive:
        source = {key: np.asarray(archive[key]) for key in archive.files}
    train = pd.read_csv(
        ROOT / "open/data/train.csv",
        usecols=["row_id", "control_success"],
        encoding="utf-8-sig",
        low_memory=False,
    )
    PREDICTIONS.mkdir(parents=True, exist_ok=True)
    artifacts: dict[str, dict[str, object]] = {}
    for year in YEARS:
        mask = source["season"].astype(np.int16) == year
        row_ids = source["row_id"][mask].astype(str)
        # Official row IDs are one-indexed and preserve the original train order.
        row_index = np.asarray(
            [int(value.rsplit("_", 1)[1]) - 1 for value in row_ids],
            dtype=np.int64,
        )
        if not np.array_equal(train.iloc[row_index]["row_id"].astype(str).to_numpy(), row_ids):
            raise ValueError(f"row_id/index mapping mismatch for {year}")
        y = train.iloc[row_index]["control_success"].to_numpy(dtype=np.int8)
        full = source["physics_teacher"][mask].astype(np.float64)
        control = source["control_teacher"][mask].astype(np.float64)
        centered = np.clip(0.5 + full - float(np.mean(full)), 1e-6, 1.0 - 1e-6)
        path = PREDICTIONS / f"{STEM}_{year}.npz"
        if path.exists():
            raise FileExistsError(f"Refusing to overwrite {path}")
        np.savez_compressed(
            path,
            y=y,
            row_index=row_index,
            cluster=source["pitcher_id"][mask].astype(np.int64),
            teacher_full_centered=centered.astype(np.float32),
            physics_teacher=full.astype(np.float32),
            control_teacher=control.astype(np.float32),
        )
        artifacts[str(year)] = {
            "path": str(path.relative_to(ROOT)),
            "sha256": sha256(path),
            "rows": int(mask.sum()),
            "soft_target_mean": float(np.mean(centered)),
            "soft_target_std": float(np.std(centered)),
            "source_label_used_for_student_fit": False,
            "source_label_role": "runner alignment integrity only",
        }
    payload = {
        "experiment_id": prereg["experiment_id"],
        "status": "development_soft_teacher_artifacts_materialized",
        "protocol": {
            "teacher_alpha": 0.0,
            "source_labels_used_for_student_fit": False,
            "source_2023_materialized": False,
            "2024_student_run": False,
            "test_rows_read": False,
        },
        "preregister": str(PREREG.relative_to(ROOT)),
        "preregister_sha256": sha256(PREREG),
        "source": str(SOURCE.relative_to(ROOT)),
        "source_sha256": sha256(SOURCE),
        "artifacts": artifacts,
    }
    OUTPUT.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"wrote {OUTPUT}", flush=True)


if __name__ == "__main__":
    main()
