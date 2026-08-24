#!/usr/bin/env python3
"""Locked 2022/2023 source evaluation of adaptive state-space shrinkage."""

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

from experiments.run_v5_h1_residual import cluster_bootstrap_score_gain  # noqa: E402
from experiments.v5_adaptive_state_space import (  # noqa: E402
    build_adaptive_state_probability,
)

TRAIN = ROOT / "open/data/train.csv"
PREREG = ROOT / "experiments/params/v5_adaptive_state_space_preregister.json"
REPORT = ROOT / "experiments/results/v5_adaptive_state_space_source.json"
PREDICTIONS = ROOT / "experiments/results/predictions"
YEARS = (2022, 2023)
ANCHORS = {
    2022: PREDICTIONS / "v3_sparse_c_backtest_2022.npz",
    2023: PREDICTIONS / "v3_sparse_c_backtest_2023.npz",
}


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


def metrics(y: np.ndarray, prediction: np.ndarray, mask: np.ndarray) -> dict[str, float]:
    yy = y[mask].astype(np.float64)
    pp = prediction[mask].astype(np.float64)
    rate = float(yy.mean())
    brier = float(np.mean(np.square(pp - yy)))
    score = 100000.0 * (1.0 - brier / (rate * (1.0 - rate)))
    return {
        "rows": int(mask.sum()),
        "target_rate": rate,
        "prediction_mean": float(pp.mean()),
        "prediction_std": float(pp.std()),
        "brier": brier,
        "score": score,
    }


def evaluate(
    y: np.ndarray,
    parent: np.ndarray,
    candidate: np.ndarray,
    game_type: np.ndarray,
) -> dict[str, Any]:
    routes = {
        "full": np.ones(len(y), dtype=bool),
        "R": game_type == "R",
    }
    output: dict[str, Any] = {}
    for name, mask in routes.items():
        p = metrics(y, parent, mask)
        c = metrics(y, candidate, mask)
        output[name] = {"parent": p, "candidate": c, "gain": c["score"] - p["score"]}
    return output


def main() -> None:
    if REPORT.exists():
        raise FileExistsError(f"immutable report exists: {REPORT}")
    prereg = json.loads(PREREG.read_text(encoding="utf-8"))
    if prereg["status"] != "locked_before_2022_2023_source_metrics":
        raise ValueError("unexpected preregistration state")
    anchors: dict[int, dict[str, np.ndarray]] = {}
    maximum_row = 0
    for year, path in ANCHORS.items():
        with np.load(path, allow_pickle=False) as archive:
            anchors[year] = {key: np.asarray(archive[key]) for key in archive.files}
        maximum_row = max(maximum_row, int(anchors[year]["row_index"].max()))
    columns = [
        "season", "game_type", "pitcher_id", "asof_pitcher_n",
        "asof_pitcher_success_rate", "control_success",
    ]
    frame = pd.read_csv(TRAIN, usecols=columns, nrows=maximum_row + 1)
    if int(frame["season"].max()) != 2023:
        raise ValueError("source reader crossed the locked 2023 boundary")

    state: dict[int, np.ndarray] = {}
    state_meta: dict[str, Any] = {}
    folds: dict[int, dict[str, Any]] = {}
    for year in YEARS:
        anchor = anchors[year]
        row_index = anchor["row_index"].astype(np.int64)
        valid = frame.loc[row_index]
        if not valid["season"].eq(year).all():
            raise ValueError(f"{year}: anchor season mismatch")
        if not np.array_equal(
            valid["control_success"].to_numpy(dtype=np.int8),
            anchor["y"].astype(np.int8),
        ):
            raise ValueError(f"{year}: anchor target mismatch")
        probability, metadata = build_adaptive_state_probability(frame, year)
        expected_index = frame.index[frame["season"].eq(year)].to_numpy(dtype=np.int64)
        if not np.array_equal(expected_index, row_index):
            raise ValueError(f"{year}: state/anchor row order mismatch")
        state[year] = probability
        state_meta[str(year)] = metadata
        folds[year] = {
            "y": anchor["y"].astype(np.int8),
            "parent": anchor["catboost_outcome"].astype(np.float64),
            "cluster": anchor["cluster"].astype(str),
            "game_type": valid["game_type"].astype(str).to_numpy(),
            "row_index": row_index,
        }

    trials: list[dict[str, Any]] = []
    for gamma_value in prereg["source_protocol"]["blend_grid"]:
        gamma = float(gamma_value)
        years: dict[str, Any] = {}
        for year in YEARS:
            fold = folds[year]
            regular = fold["game_type"] == "R"
            candidate = fold["parent"].copy()
            candidate[regular] = (
                (1.0 - gamma) * fold["parent"][regular]
                + gamma * state[year][regular]
            )
            candidate = np.clip(candidate, 1e-6, 1.0 - 1e-6)
            years[str(year)] = evaluate(
                fold["y"], fold["parent"], candidate, fold["game_type"]
            )
        trials.append({
            "gamma": gamma,
            "minimum_R_gain": float(min(years[str(y)]["R"]["gain"] for y in YEARS)),
            "minimum_full_gain": float(min(years[str(y)]["full"]["gain"] for y in YEARS)),
            "mean_R_gain": float(np.mean([years[str(y)]["R"]["gain"] for y in YEARS])),
            "years": years,
        })
    selected = max(trials, key=lambda item: (
        item["minimum_R_gain"], item["minimum_full_gain"],
        item["mean_R_gain"], -item["gamma"],
    ))

    intervals: dict[str, Any] = {}
    artifacts: dict[str, str] = {}
    for offset, year in enumerate(YEARS):
        fold = folds[year]
        regular = fold["game_type"] == "R"
        candidate = fold["parent"].copy()
        gamma = float(selected["gamma"])
        candidate[regular] = (
            (1.0 - gamma) * fold["parent"][regular]
            + gamma * state[year][regular]
        )
        intervals[str(year)] = {}
        for name, mask in {
            "R": regular,
            "full": np.ones(len(regular), dtype=bool),
        }.items():
            intervals[str(year)][name] = cluster_bootstrap_score_gain(
                fold["y"], fold["parent"], candidate, fold["cluster"],
                mask, 2000, 882200 + 10 * offset + (0 if name == "R" else 1),
            )
        path = PREDICTIONS / f"v5_adaptive_state_space_source_{year}.npz"
        if path.exists():
            raise FileExistsError(f"immutable artifact exists: {path}")
        np.savez_compressed(
            path,
            y=fold["y"], row_index=fold["row_index"], cluster=fold["cluster"],
            parent=fold["parent"].astype(np.float32),
            adaptive_state=state[year].astype(np.float32),
            final_prediction=candidate.astype(np.float32),
        )
        artifacts[str(year)] = str(path.relative_to(ROOT))

    gate = prereg["source_protocol"]["gate"]
    checks: dict[str, bool] = {}
    for year in YEARS:
        result = selected["years"][str(year)]
        checks[f"{year}_R_gain"] = bool(
            result["R"]["gain"] >= float(gate["minimum_R_gain_each_year"])
        )
        checks[f"{year}_full_gain"] = bool(
            result["full"]["gain"] >= float(gate["minimum_full_gain_each_year"])
        )
        checks[f"{year}_R_ci"] = bool(intervals[str(year)]["R"]["ci_low"] > 0.0)
        checks[f"{year}_full_ci"] = bool(
            intervals[str(year)]["full"]["ci_low"] > 0.0
        )
    passed = bool(all(checks.values()))
    report = {
        "experiment_id": prereg["experiment_id"],
        "status": "source_pass" if passed else "source_failed",
        "preregister_sha256": sha256(PREREG),
        "implementation_sha256": sha256(
            ROOT / "experiments/v5_adaptive_state_space.py"
        ),
        "script_sha256": sha256(Path(__file__)),
        "years_read": list(YEARS),
        "confirmation_2024_read": False,
        "state_metadata": state_meta,
        "selected": selected,
        "intervals": intervals,
        "gate": {"requirements": gate, "checks": checks, "pass": passed},
        "prediction_artifacts": artifacts,
        "goal_status": "active",
        "goal_completion_claimed": False,
    }
    REPORT.write_text(
        json.dumps(safe(report), ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(safe({
        "status": report["status"], "state_metadata": state_meta,
        "selected": selected, "intervals": intervals, "gate": report["gate"],
    }), ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
