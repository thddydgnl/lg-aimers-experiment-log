#!/usr/bin/env python3
"""Audit every paired 2022/2023 probability artifact without touching 2024."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
PRED = ROOT / "experiments/results/predictions"
TRAIN = ROOT / "open/data/train.csv"
PREREG = ROOT / "experiments/params/v5_artifact_inventory_audit_preregister.json"
REPORT = ROOT / "experiments/results/v5_artifact_inventory_audit.json"
YEARS = (2022, 2023)
PARENT = {year: PRED / f"v3_sparse_c_backtest_{year}.npz" for year in YEARS}
ANCHORS = {
    "exact_parent_C": PARENT,
    "honest_r_identity": {
        year: PRED / f"v5_honest_m3_r_identity_{year}.npz" for year in YEARS
    },
    "honest_r_grid": {
        year: PRED / f"v5_honest_m3_r_grid_{year}.npz" for year in YEARS
    },
}


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def score(y: np.ndarray, prediction: np.ndarray, mask: np.ndarray) -> float:
    yy = y[mask].astype(np.float64, copy=False)
    pp = prediction[mask].astype(np.float64, copy=False)
    rate = float(yy.mean())
    return float(100000.0 * (1.0 - np.mean(np.square(pp - yy)) / (rate * (1.0 - rate))))


def safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [safe(v) for v in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def load_reference() -> dict[int, dict[str, Any]]:
    game_type = pd.read_csv(TRAIN, usecols=["game_type"])["game_type"]
    result: dict[int, dict[str, Any]] = {}
    for year in YEARS:
        with np.load(PARENT[year], allow_pickle=False) as data:
            y = data["y"].astype(np.int8)
            row_index = data["row_index"].astype(np.int64)
            parent = data["catboost_outcome"].astype(np.float64)
        regular = game_type.iloc[row_index].astype(str).eq("R").to_numpy()
        anchors: dict[str, np.ndarray] = {"exact_parent_C": parent}
        for name in ("honest_r_identity", "honest_r_grid"):
            with np.load(ANCHORS[name][year], allow_pickle=False) as data:
                if not np.array_equal(data["y"].astype(np.int8), y):
                    raise ValueError(f"{year} {name}: target mismatch")
                if not np.array_equal(data["row_index"].astype(np.int64), row_index):
                    raise ValueError(f"{year} {name}: row mismatch")
                anchors[name] = data["final_prediction"].astype(np.float64)
        result[year] = {
            "y": y,
            "row_index": row_index,
            "parent": parent,
            "regular": regular,
            "anchors": anchors,
        }
    return result


def paired_stems() -> list[str]:
    by_year = {
        year: {path.name[: -len(f"_{year}.npz")] for path in PRED.glob(f"*_{year}.npz")}
        for year in YEARS
    }
    return sorted(by_year[2022] & by_year[2023])


def probability_arrays(path: Path, reference: dict[str, Any]) -> dict[str, np.ndarray]:
    output: dict[str, np.ndarray] = {}
    # Some legacy artifacts contain object-valued diagnostic metadata.  These
    # files are local/trusted; load them, then retain only numeric 1-D arrays.
    with np.load(path, allow_pickle=True) as data:
        if "y" not in data.files or "row_index" not in data.files:
            return output
        if not np.array_equal(data["y"].astype(np.int8), reference["y"]):
            return output
        if not np.array_equal(data["row_index"].astype(np.int64), reference["row_index"]):
            return output
        for key in data.files:
            value = data[key]
            if value.ndim != 1 or len(value) != len(reference["y"]):
                continue
            if value.dtype.kind not in "fc":
                continue
            array = value.astype(np.float64)
            if not np.all(np.isfinite(array)):
                continue
            if float(array.min()) < -1e-9 or float(array.max()) > 1.0 + 1e-9:
                continue
            # Exact labels and bookkeeping rates are retained in the audit only
            # if their key is explicit; the later provenance review must reject
            # them.  Here, reject a byte-for-byte target immediately.
            if np.array_equal(array, reference["y"].astype(np.float64)):
                continue
            output[key] = np.clip(array, 0.0, 1.0)
    return output


def main() -> None:
    if REPORT.exists():
        raise FileExistsError(f"immutable report exists: {REPORT}")
    prereg = json.loads(PREREG.read_text(encoding="utf-8"))
    if prereg["status"] != "locked_before_inventory_scores":
        raise ValueError("unexpected preregistration status")
    reference = load_reference()
    trials: list[dict[str, Any]] = []
    skipped: dict[str, int] = {}
    for stem in paired_stems():
        arrays = {
            year: probability_arrays(PRED / f"{stem}_{year}.npz", reference[year])
            for year in YEARS
        }
        common = sorted(set(arrays[2022]) & set(arrays[2023]))
        if not common:
            skipped["no_common_aligned_probability_key"] = skipped.get(
                "no_common_aligned_probability_key", 0
            ) + 1
            continue
        for key in common:
            for gamma_value in prereg["gamma_grid"]:
                gamma = float(gamma_value)
                cells: dict[str, Any] = {}
                full_gains: list[float] = []
                r_gains: list[float] = []
                for year in YEARS:
                    fold = reference[year]
                    regular = fold["regular"]
                    candidate = fold["parent"].copy()
                    candidate[regular] = (
                        (1.0 - gamma) * fold["parent"][regular]
                        + gamma * arrays[year][key][regular]
                    )
                    cells[str(year)] = {}
                    for anchor_name, anchor in fold["anchors"].items():
                        full_mask = np.ones(len(fold["y"]), dtype=bool)
                        full_gain = score(fold["y"], candidate, full_mask) - score(
                            fold["y"], anchor, full_mask
                        )
                        r_gain = score(fold["y"], candidate, regular) - score(
                            fold["y"], anchor, regular
                        )
                        cells[str(year)][anchor_name] = {
                            "full_gain": full_gain,
                            "R_gain": r_gain,
                        }
                        full_gains.append(full_gain)
                        r_gains.append(r_gain)
                trials.append(
                    {
                        "stem": stem,
                        "key": key,
                        "gamma": gamma,
                        "minimum_full_gain": min(full_gains),
                        "minimum_R_gain": min(r_gains),
                        "mean_full_gain": float(np.mean(full_gains)),
                        "cells": cells,
                    }
                )
    trials.sort(
        key=lambda item: (
            item["minimum_full_gain"], item["minimum_R_gain"], item["mean_full_gain"]
        ),
        reverse=True,
    )
    report = {
        "experiment_id": prereg["experiment_id"],
        "status": "audit_complete_not_candidate_selection",
        "preregister_sha256": digest(PREREG),
        "script_sha256": digest(Path(__file__)),
        "years_read": list(YEARS),
        "years_not_read": [2024],
        "paired_stem_count": len(paired_stems()),
        "trial_count": len(trials),
        "skipped": skipped,
        "top_trials": trials[:200],
        "warning": "Every top row requires manual generator/provenance audit; diagnostic, oracle, validation-selected, and target-derived arrays are ineligible.",
        "goal_status": "active",
        "goal_completion_claimed": False,
    }
    REPORT.write_text(json.dumps(safe(report), ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(safe({**{k: report[k] for k in ("status", "paired_stem_count", "trial_count", "skipped")}, "top_trials": trials[:25]}), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
