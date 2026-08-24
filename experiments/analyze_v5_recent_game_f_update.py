#!/usr/bin/env python3
"""Immutable source gate for a direct previous-game F update."""

from __future__ import annotations

from fractions import Fraction
from math import gcd
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.analyze_v5_dense_pitchtype_moe import digest, load, safe, score
from experiments.run_v5_h1_residual import cluster_bootstrap_score_gain

TRAIN = ROOT / "open/data/train.csv"
PRED = ROOT / "experiments/results/predictions"
PREREG = ROOT / "experiments/params/v5_recent_game_f_update_preregister.json"
REPORT = ROOT / "experiments/results/v5_recent_game_f_update_source.json"
YEARS = (2020, 2021)


def load_source_rows() -> pd.DataFrame:
    columns = ["season", "game_type", "control_success"]
    for horizon in (1, 3, 5):
        columns.extend([
            f"asof_pitcher_prev{horizon}_game_success_rate",
            f"asof_pitcher_prev{horizon}_game_middle_rate",
        ])
    pieces: list[pd.DataFrame] = []
    offset = 0
    for chunk in pd.read_csv(TRAIN, usecols=columns, chunksize=250_000):
        chunk.index = np.arange(offset, offset + len(chunk), dtype=np.int64)
        offset += len(chunk)
        part = chunk.loc[chunk["season"].le(max(YEARS))]
        if len(part):
            pieces.append(part)
        if int(chunk["season"].min()) > max(YEARS):
            break
    frame = pd.concat(pieces)
    if int(frame["season"].max()) != max(YEARS):
        raise ValueError("source loader did not stop at 2021")
    return frame


def denominator(value: float, maximum: int, tolerance: float) -> int:
    if not np.isfinite(value):
        return 0
    if value <= 0.0 or value >= 1.0:
        return 1
    candidate = Fraction(float(value)).limit_denominator(maximum)
    return int(candidate.denominator) if abs(float(candidate) - float(value)) <= tolerance else 0


def decode(rows: pd.DataFrame, horizon: int, maximum: int, tolerance: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    success = pd.to_numeric(rows[f"asof_pitcher_prev{horizon}_game_success_rate"], errors="coerce").to_numpy(dtype=np.float64)
    middle = pd.to_numeric(rows[f"asof_pitcher_prev{horizon}_game_middle_rate"], errors="coerce").to_numpy(dtype=np.float64)
    success_den = np.fromiter((denominator(v, maximum, tolerance) for v in success), dtype=np.int32, count=len(rows))
    middle_den = np.fromiter((denominator(v, maximum, tolerance) for v in middle), dtype=np.int32, count=len(rows))
    common = np.fromiter(
        (int(a // gcd(int(a), int(b)) * b) if a > 0 and b > 0 else 0 for a, b in zip(success_den, middle_den)),
        dtype=np.int32, count=len(rows),
    )
    valid = (common > 0) & (common <= maximum)
    common = np.where(valid, common, 0).astype(np.float64)
    successes = np.where(valid, np.rint(np.nan_to_num(success, nan=0.0) * common), 0.0)
    return common, successes, valid


def metrics(
    artifact: dict[str, np.ndarray], parent: np.ndarray, candidate: np.ndarray,
    masks: dict[str, np.ndarray], bootstrap: int, seed: int,
) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for route_index, (name, mask) in enumerate(masks.items()):
        base = score(artifact["y"], parent, mask)
        cand = score(artifact["y"], candidate, mask)
        ci = cluster_bootstrap_score_gain(
            artifact["y"], parent, candidate, artifact["cluster"], mask,
            iterations=bootstrap, seed=seed + 1000 * route_index,
        )
        out[name] = {"parent": base, "candidate": cand, "gain": float(cand["score"] - base["score"]), "pitcher_cluster_95_ci": ci}
    return out


def main() -> None:
    if REPORT.exists():
        raise FileExistsError(f"immutable report exists: {REPORT}")
    prereg = json.loads(PREREG.read_text(encoding="utf-8"))
    frame = load_source_rows()
    folds: dict[int, dict[str, Any]] = {}
    for year in YEARS:
        path = PRED / f"v4_m3_c_backtest_{year}_{year}.npz"
        artifact = load(path)
        rows = frame.loc[artifact["row_index"].astype(np.int64)]
        if not rows["season"].eq(year).all() or not np.array_equal(rows["control_success"].to_numpy(dtype=np.int8), artifact["y"].astype(np.int8)):
            raise ValueError(f"alignment mismatch: {year}")
        regular = rows["game_type"].astype(str).eq("R").to_numpy()
        decoded: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
        for horizon in prereg["decoder"]["horizons"]:
            maximum = int(prereg["decoder"]["maximum_denominator"][str(horizon)])
            decoded[int(horizon)] = decode(rows, int(horizon), maximum, float(prereg["decoder"]["rounding_tolerance"]))
        folds[year] = {
            "artifact": artifact, "rows": rows, "regular": regular,
            "masks": {"full": np.ones(len(rows), dtype=bool), "R": regular, "F": ~regular},
            "parent": artifact["catboost_outcome"].astype(np.float64),
            "decoded": decoded, "path": path,
        }

    bootstrap = int(prereg["source_selection"]["bootstrap_iterations"])
    trials: list[dict[str, Any]] = []
    cache: dict[tuple[int, float, int], np.ndarray] = {}
    for horizon in (int(v) for v in prereg["decoder"]["horizons"]):
        for k in (float(v) for v in prereg["k_grid"]):
            years: dict[str, Any] = {}
            coverage: dict[str, float] = {}
            for year in YEARS:
                fold = folds[year]
                n, s, valid = fold["decoded"][horizon]
                update = (s + k * fold["parent"]) / (n + k)
                update = np.where(valid, update, fold["parent"])
                candidate = np.clip(np.where(fold["regular"], fold["parent"], update), 1e-6, 1.0 - 1e-6)
                cache[(horizon, k, year)] = candidate
                years[str(year)] = metrics(
                    fold["artifact"], fold["parent"], candidate, fold["masks"],
                    bootstrap, 8350000 + horizon * 100000 + int(k) * 10 + year,
                )
                coverage[str(year)] = float(valid[~fold["regular"]].mean())
            trials.append({
                "horizon": horizon, "k": k, "coverage_F": coverage,
                "minimum_F_gain": float(min(years[str(y)]["F"]["gain"] for y in YEARS)),
                "minimum_full_gain": float(min(years[str(y)]["full"]["gain"] for y in YEARS)),
                "mean_F_gain": float(np.mean([years[str(y)]["F"]["gain"] for y in YEARS])),
                "years": years,
            })
    selected = max(trials, key=lambda x: (x["minimum_F_gain"], x["minimum_full_gain"], x["mean_F_gain"], x["k"], -x["horizon"]))
    gate = prereg["source_selection"]["gate"]
    checks: dict[str, Any] = {}
    passed = True
    for year in YEARS:
        routes = selected["years"][str(year)]
        local = {
            "F_point": routes["F"]["gain"] >= float(gate["minimum_F_gain_each_year"]),
            "full_point": routes["full"]["gain"] >= float(gate["minimum_full_gain_each_year"]),
            "F_ci": routes["F"]["pitcher_cluster_95_ci"]["ci_low"] > 0.0,
            "full_ci": routes["full"]["pitcher_cluster_95_ci"]["ci_low"] > 0.0,
            "coverage": selected["coverage_F"][str(year)] >= float(gate["decoded_coverage_each_year_at_least"]),
        }
        checks[str(year)] = local
        passed = passed and all(local.values())

    artifacts: dict[str, Any] = {}
    for year in YEARS:
        fold = folds[year]
        output = PRED / f"v5_recent_game_f_update_source_{year}.npz"
        if output.exists():
            raise FileExistsError(f"immutable artifact exists: {output}")
        n, s, valid = fold["decoded"][int(selected["horizon"])]
        candidate = cache[(int(selected["horizon"]), float(selected["k"]), year)]
        np.savez_compressed(
            output, y=fold["artifact"]["y"].astype(np.int8), row_index=fold["artifact"]["row_index"].astype(np.int64),
            cluster=fold["artifact"]["cluster"], parent_exact_c=fold["parent"], decoded_n=n,
            decoded_successes=s, decoded_valid=valid.astype(np.int8), final_prediction=candidate,
        )
        artifacts[str(year)] = {"path": str(output.relative_to(ROOT)), "sha256": digest(output)}
    report = {
        "experiment_id": prereg["experiment_id"], "status": "source_pass" if passed else "source_failed",
        "preregister_sha256": digest(PREREG), "script_sha256": digest(Path(__file__)),
        "years_read": list(YEARS), "years_not_read": [2022, 2023, 2024], "trials": trials,
        "selected": selected, "source_gate": {"requirements": gate, "checks": checks, "pass": bool(passed)},
        "artifacts": artifacts, "goal_status": "active", "goal_completion_claimed": False,
    }
    REPORT.write_text(json.dumps(safe(report), ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(safe({"status": report["status"], "selected": selected, "checks": checks}), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
