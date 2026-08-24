#!/usr/bin/env python3
"""Prepare the synthetic 2025 validation shell used for full-history refits.

The five official sample-test rows are appended only as validation rows.  Their
dummy targets are never used for fitting or model selection.  The frozen V3
package supplies the deployment anchor prediction for those same rows.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
TRAIN = ROOT / "open/data/train.csv"
TEST = ROOT / "open/data/test.csv"
SAMPLE = ROOT / "open/data/sample_submission.csv"
COMBINED = ROOT / "experiments/_cache/v4_full_refit_2025.csv"
ANCHOR = ROOT / "experiments/results/predictions/v3_sparse_m3_frozen_2025.npz"
PACKAGE = ROOT / "submission/dist/V3_sparse_m3_1103.zip"
WORK = ROOT / "submission/work/v4_anchor_probe"


def main() -> None:
    train = pd.read_csv(TRAIN, encoding="utf-8-sig")
    test = pd.read_csv(TEST, encoding="utf-8-sig")
    sample = pd.read_csv(SAMPLE, encoding="utf-8-sig")
    if list(test["row_id"].astype(str)) != list(sample["row_id"].astype(str)):
        raise ValueError("Official test/sample row order mismatch")
    if set(test["season"].astype(int)) != {2025}:
        raise ValueError("Expected only 2025 sample-test rows")
    shell = test.copy()
    # Both classes keep diagnostics (AUC/reference Brier) finite.  These labels
    # belong only to the held-out five-row export probe and never enter fitting.
    probe_target = np.arange(len(shell), dtype=np.int8) % 2
    shell["control_success"] = probe_target
    shell = shell[train.columns]
    combined = pd.concat([train, shell], ignore_index=True)
    COMBINED.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(COMBINED, index=False, encoding="utf-8-sig")

    if WORK.exists():
        shutil.rmtree(WORK)
    WORK.mkdir(parents=True)
    with zipfile.ZipFile(PACKAGE) as archive:
        archive.extractall(WORK)
    data_dir = WORK / "data"
    data_dir.mkdir()
    shutil.copyfile(TEST, data_dir / "test.csv")
    shutil.copyfile(SAMPLE, data_dir / "sample_submission.csv")
    subprocess.run([sys.executable, "script.py"], cwd=WORK, check=True)
    prediction_frame = pd.read_csv(WORK / "output/submission.csv")
    if list(prediction_frame["row_id"].astype(str)) != list(test["row_id"].astype(str)):
        raise ValueError("V3 anchor output row mismatch")
    prediction = prediction_frame["control_success"].to_numpy(dtype=np.float64)
    row_index = np.arange(len(train), len(combined), dtype=np.int64)
    ANCHOR.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        ANCHOR,
        y=probe_target,
        row_index=row_index,
        cluster=np.asarray(test["pitcher_id"].astype(str), dtype=np.str_),
        final_prediction=prediction,
    )
    record = {
        "combined_csv": str(COMBINED.relative_to(ROOT)),
        "train_rows": int(len(train)),
        "validation_shell_rows": int(len(test)),
        "validation_shell_season": 2025,
        "dummy_target_used_for_training": False,
        "anchor_artifact": str(ANCHOR.relative_to(ROOT)),
        "anchor_prediction": prediction.tolist(),
    }
    record_path = ROOT / "submission/records/V4_full_refit_preparation.json"
    record_path.parent.mkdir(parents=True, exist_ok=True)
    record_path.write_text(json.dumps(record, indent=2), encoding="utf-8")
    print(json.dumps(record, indent=2))


if __name__ == "__main__":
    main()
