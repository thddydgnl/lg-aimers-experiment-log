#!/usr/bin/env python3
"""Rule-safety and environment smoke test for the isolated TabICLv2 runtime.

This script uses synthetic rows only.  It verifies that the same query row gets
the same prediction alone, in a batch, after shuffling the batch, and after a
duplicate row is appended.  Those checks are mandatory before TabICL can be
used on an official rolling fold because evaluation rows must not interact.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
from pathlib import Path
import sys
import time

# On Windows, load PyTorch's CUDA DLLs before NumPy/pandas/sklearn import
# their native runtimes.  Reversing this order can raise WinError 1114 even
# though the same PyTorch installation is healthy.
import torch
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification


ROOT = Path(__file__).resolve().parents[1]
ISOLATED_SITE = ROOT / "experiments" / "_tabicl_site"
if str(ISOLATED_SITE) not in sys.path:
    sys.path.insert(0, str(ISOLATED_SITE))

from tabicl import TabICLClassifier  # noqa: E402


CHECKPOINT = ROOT / "experiments" / "_cache" / "tabicl" / "tabiclv2_classifier.ckpt"
REPORT = ROOT / "experiments" / "results" / "v5_tabicl_environment_smoke.json"
FIXED_QUERY_ROWS = 256


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fixed_query_predict(
    model: TabICLClassifier,
    frame: pd.DataFrame,
    frozen_pad_row: pd.DataFrame,
    query_rows: int = FIXED_QUERY_ROWS,
) -> np.ndarray:
    """Predict in fixed-size query blocks using only a frozen train-row pad.

    TabICLv2's mathematical attention mask restricts query keys/values to the
    training context, but different query lengths can still select different
    GPU kernels and produce materially different floating-point results.  A
    fixed shape removes that numerical dependency.  The discarded padding row
    is copied from frozen training data, never from another evaluation row.
    """
    outputs: list[np.ndarray] = []
    for start in range(0, len(frame), query_rows):
        chunk = frame.iloc[start : start + query_rows].reset_index(drop=True)
        real_rows = len(chunk)
        if real_rows < query_rows:
            padding = pd.concat(
                [frozen_pad_row] * (query_rows - real_rows), ignore_index=True
            )
            chunk = pd.concat([chunk, padding], ignore_index=True)
        outputs.append(
            model.predict_proba(chunk)[:real_rows, 1].astype(np.float64)
        )
    return np.concatenate(outputs) if outputs else np.empty(0, dtype=np.float64)


def main() -> None:
    rng = np.random.default_rng(2026)
    x, y = make_classification(
        n_samples=1_280,
        n_features=12,
        n_informative=8,
        n_redundant=2,
        class_sep=0.8,
        flip_y=0.03,
        random_state=2026,
    )
    frame = pd.DataFrame(x, columns=[f"x{i}" for i in range(x.shape[1])])
    frame["cat_a"] = pd.Series(np.where(frame["x0"] > 0, "right", "left"), dtype="string")
    frame["cat_b"] = pd.Series(np.floor(frame["x1"] * 2).clip(-3, 3).astype(int), dtype="string")
    train_x = frame.iloc[:1_024].reset_index(drop=True)
    train_y = y[:1_024]
    test_x = frame.iloc[1_024:1_041].reset_index(drop=True)

    CHECKPOINT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(2026)
    torch.cuda.reset_peak_memory_stats()

    model = TabICLClassifier(
        n_estimators=1,
        batch_size=1,
        kv_cache=True,
        model_path=CHECKPOINT,
        allow_auto_download=True,
        checkpoint_version="tabicl-classifier-v2-20260212.ckpt",
        device="cuda",
        use_amp="auto",
        use_fa3=False,
        offload_mode="auto",
        random_state=2026,
        n_jobs=6,
        verbose=False,
    )

    fit_started = time.perf_counter()
    model.fit(train_x, train_y)
    fit_seconds = time.perf_counter() - fit_started

    predict_started = time.perf_counter()
    batch = model.predict_proba(test_x)[:, 1].astype(np.float64)
    batch_seconds = time.perf_counter() - predict_started

    single = np.array(
        [model.predict_proba(test_x.iloc[[i]])[0, 1] for i in range(len(test_x))],
        dtype=np.float64,
    )
    permutation = rng.permutation(len(test_x))
    shuffled = model.predict_proba(test_x.iloc[permutation].reset_index(drop=True))[:, 1]
    shuffled_restored = np.empty_like(shuffled, dtype=np.float64)
    shuffled_restored[permutation] = shuffled
    duplicated = model.predict_proba(
        pd.concat([test_x, test_x.iloc[[0, 3, 3]]], ignore_index=True)
    )[:, 1]

    frozen_pad_row = train_x.iloc[[0]].copy()
    fixed_batch = fixed_query_predict(model, test_x, frozen_pad_row)
    fixed_single = np.array(
        [
            fixed_query_predict(model, test_x.iloc[[i]], frozen_pad_row)[0]
            for i in range(len(test_x))
        ],
        dtype=np.float64,
    )
    fixed_shuffled = fixed_query_predict(
        model, test_x.iloc[permutation].reset_index(drop=True), frozen_pad_row
    )
    fixed_shuffled_restored = np.empty_like(fixed_shuffled)
    fixed_shuffled_restored[permutation] = fixed_shuffled
    fixed_duplicated = fixed_query_predict(
        model,
        pd.concat([test_x, test_x.iloc[[0, 3, 3]]], ignore_index=True),
        frozen_pad_row,
    )

    deltas = {
        "single_vs_batch": float(np.max(np.abs(single - batch))),
        "shuffle_vs_batch": float(np.max(np.abs(shuffled_restored - batch))),
        "duplicate_vs_batch": float(np.max(np.abs(duplicated[: len(batch)] - batch))),
        "duplicate_copy_row0": float(abs(duplicated[len(batch)] - batch[0])),
        "duplicate_copy_row3_a": float(abs(duplicated[len(batch) + 1] - batch[3])),
        "duplicate_copy_row3_b": float(abs(duplicated[len(batch) + 2] - batch[3])),
    }
    fixed_deltas = {
        "single_vs_batch": float(np.max(np.abs(fixed_single - fixed_batch))),
        "shuffle_vs_batch": float(
            np.max(np.abs(fixed_shuffled_restored - fixed_batch))
        ),
        "duplicate_vs_batch": float(
            np.max(np.abs(fixed_duplicated[: len(fixed_batch)] - fixed_batch))
        ),
        "duplicate_copy_row0": float(
            abs(fixed_duplicated[len(fixed_batch)] - fixed_batch[0])
        ),
        "duplicate_copy_row3_a": float(
            abs(fixed_duplicated[len(fixed_batch) + 1] - fixed_batch[3])
        ),
        "duplicate_copy_row3_b": float(
            abs(fixed_duplicated[len(fixed_batch) + 2] - fixed_batch[3])
        ),
    }
    tolerance = 2e-6
    direct_passed = all(value <= tolerance for value in deltas.values())
    fixed_passed = all(value <= tolerance for value in fixed_deltas.values())

    report = {
        "experiment_id": "V5_TABICLV2_ENVIRONMENT_SMOKE_V1",
        "status": (
            "passed_fixed_query_wrapper"
            if fixed_passed
            else "failed_row_independence"
        ),
        "synthetic_only": True,
        "checkpoint": {
            "path": str(CHECKPOINT.relative_to(ROOT)),
            "bytes": CHECKPOINT.stat().st_size,
            "sha256": sha256(CHECKPOINT),
            "version": "tabicl-classifier-v2-20260212.ckpt",
            "source_repo": "jingang/TabICL",
        },
        "runtime": {
            "tabicl": importlib.metadata.version("tabicl"),
            "torch": torch.__version__,
            "cuda": torch.cuda.is_available(),
            "gpu": torch.cuda.get_device_name(0),
            "fit_seconds": fit_seconds,
            "batch_predict_seconds": batch_seconds,
            "peak_cuda_bytes": int(torch.cuda.max_memory_allocated()),
        },
        "model": {
            "n_estimators": 1,
            "batch_size": 1,
            "kv_cache": True,
            "train_rows": len(train_x),
            "test_rows": len(test_x),
            "features": train_x.shape[1],
            "random_state": 2026,
            "fixed_query_rows": FIXED_QUERY_ROWS,
            "frozen_pad_source": "first training feature row",
        },
        "row_independence_tolerance": tolerance,
        "direct_api": {
            "passed": direct_passed,
            "max_abs_deltas": deltas,
            "eligible_for_deployment": False,
        },
        "fixed_query_wrapper": {
            "passed": fixed_passed,
            "max_abs_deltas": fixed_deltas,
            "uses_other_evaluation_rows_for_padding": False,
        },
        "prediction_summary": {
            "mean": float(batch.mean()),
            "std": float(batch.std()),
            "min": float(batch.min()),
            "max": float(batch.max()),
        },
    }
    REPORT.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    if not fixed_passed:
        raise SystemExit("TabICL row-independence smoke failed")


if __name__ == "__main__":
    main()
