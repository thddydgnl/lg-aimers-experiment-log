#!/usr/bin/env python3
"""Immutable 2022/2023 evaluation of the locked direct R-season update."""

from __future__ import annotations

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
RESULTS = ROOT / "experiments/results"
PRED = RESULTS / "predictions"
LOCK = ROOT / "experiments/params/v5_direct_season_update_r_lock.json"
CONTRACT = ROOT / "experiments/params/v5_validation_contract_v2.json"
SOURCE = RESULTS / "v5_direct_season_update_source.json"
REPORT = RESULTS / "v5_direct_season_update_dev.json"
YEARS = (2022, 2023)
ANCHORS = {
    "exact_c": ("v3_sparse_c_backtest_{year}.npz", "catboost_outcome"),
    "honest_identity": ("v5_honest_m3_r_identity_{year}.npz", "final_prediction"),
    "honest_grid": ("v5_honest_m3_r_grid_{year}.npz", "final_prediction"),
}


def load_rows() -> pd.DataFrame:
    columns = [
        "season", "game_type", "pitcher_id", "asof_pitcher_n",
        "asof_pitcher_success_rate", "control_success",
    ]
    pieces: list[pd.DataFrame] = []
    offset = 0
    for chunk in pd.read_csv(TRAIN, usecols=columns, chunksize=250_000):
        chunk.index = np.arange(offset, offset + len(chunk), dtype=np.int64)
        offset += len(chunk)
        selected = chunk.loc[chunk["season"].le(max(YEARS))]
        if len(selected):
            pieces.append(selected)
        if int(chunk["season"].min()) > max(YEARS):
            break
    frame = pd.concat(pieces, axis=0)
    if int(frame["season"].max()) != max(YEARS):
        raise ValueError("development loader did not end at 2023")
    return frame


def states_before(frame: pd.DataFrame) -> dict[int, dict[int, tuple[int, int]]]:
    before: dict[int, dict[int, tuple[int, int]]] = {}
    state: dict[int, tuple[int, int]] = {}
    for year in sorted(frame["season"].astype(int).unique()):
        before[int(year)] = dict(state)
        last = frame.loc[frame["season"].eq(year)].groupby(
            "pitcher_id", sort=False, observed=True
        ).tail(1)
        for row in last.itertuples(index=False):
            n = int(row.asof_pitcher_n or 0)
            rate = 0.0 if pd.isna(row.asof_pitcher_success_rate) else float(row.asof_pitcher_success_rate)
            state[int(row.pitcher_id)] = (
                n + 1, int(np.rint(rate * n)) + int(row.control_success)
            )
    return before


def current_state(
    rows: pd.DataFrame, year: int, before: dict[int, dict[int, tuple[int, int]]]
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    frozen = before[year]
    pitchers = rows["pitcher_id"].to_numpy(dtype=np.int64)
    n_end = np.fromiter((frozen.get(int(p), (0, 0))[0] for p in pitchers), dtype=np.int64)
    s_end = np.fromiter((frozen.get(int(p), (0, 0))[1] for p in pitchers), dtype=np.int64)
    n_asof = rows["asof_pitcher_n"].fillna(0).to_numpy(dtype=np.int64)
    career = rows["asof_pitcher_success_rate"].fillna(0.5).to_numpy(dtype=np.float64)
    s_asof = np.rint(career * n_asof).astype(np.int64)
    n = n_asof - n_end
    s = s_asof - s_end
    invalid = (n < 0) | (s < 0) | (s > n)
    return (
        np.where(invalid, 0, n).astype(np.float64),
        np.where(invalid, 0, s).astype(np.float64),
        invalid,
    )


def evaluate(
    y: np.ndarray, anchor: np.ndarray, candidate: np.ndarray,
    cluster: np.ndarray, masks: dict[str, np.ndarray], seed: int,
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for route_index, (name, mask) in enumerate(masks.items()):
        base = score(y, anchor, mask)
        cand = score(y, candidate, mask)
        ci = cluster_bootstrap_score_gain(
            y, anchor, candidate, cluster, mask, iterations=2000,
            seed=seed + 1000 * route_index,
        )
        result[name] = {
            "anchor": base, "candidate": cand,
            "gain": float(cand["score"] - base["score"]),
            "pitcher_cluster_95_ci": ci,
        }
    return result


def main() -> None:
    if REPORT.exists():
        raise FileExistsError(f"immutable report exists: {REPORT}")
    lock = json.loads(LOCK.read_text(encoding="utf-8"))
    contract = json.loads(CONTRACT.read_text(encoding="utf-8"))
    if digest(SOURCE) != lock["source_discovery"]["report_sha256"]:
        raise ValueError("source report changed after the R-only lock")
    recipe = lock["locked_recipe"]
    if float(recipe["R_k"]) != 500.0 or recipe["route"] != "game_type R only":
        raise ValueError("analyzer disagrees with locked recipe")
    frame = load_rows()
    before = states_before(frame)
    years: dict[str, Any] = {}
    artifacts: dict[str, Any] = {}
    full_gains: list[float] = []
    same_parent_checks: dict[str, Any] = {}
    input_hashes: dict[str, Any] = {}
    for year in YEARS:
        anchor_artifacts: dict[str, dict[str, np.ndarray]] = {}
        anchor_paths: dict[str, Path] = {}
        for name, (template, _) in ANCHORS.items():
            path = PRED / template.format(year=year)
            anchor_paths[name] = path
            anchor_artifacts[name] = load(path)
        reference = anchor_artifacts["exact_c"]
        for name, artifact in anchor_artifacts.items():
            for key in ("y", "row_index", "cluster"):
                if not np.array_equal(reference[key], artifact[key]):
                    raise ValueError(f"alignment mismatch: {year}/{name}/{key}")
        indices = reference["row_index"].astype(np.int64)
        rows = frame.loc[indices]
        if not rows["season"].eq(year).all():
            raise ValueError(f"season mismatch: {year}")
        if not np.array_equal(rows["control_success"].to_numpy(dtype=np.int8), reference["y"].astype(np.int8)):
            raise ValueError(f"target mismatch: {year}")
        regular = rows["game_type"].astype(str).eq("R").to_numpy()
        masks = {"full": np.ones(len(rows), dtype=bool), "R": regular, "F": ~regular}
        n, s, invalid = current_state(rows, year, before)
        parent = reference["catboost_outcome"].astype(np.float64)
        updated = (s + 500.0 * parent) / (n + 500.0)
        updated = np.where(invalid, parent, updated)
        candidate = np.clip(np.where(regular, updated, parent), 1e-6, 1.0 - 1e-6)
        comparisons: dict[str, Any] = {}
        for anchor_index, (name, (_, key)) in enumerate(ANCHORS.items()):
            anchor = anchor_artifacts[name][key].astype(np.float64)
            comparisons[name] = evaluate(
                reference["y"], anchor, candidate, reference["cluster"], masks,
                8240000 + 10000 * year + 100 * anchor_index,
            )
            full_gains.append(float(comparisons[name]["full"]["gain"]))
        same = comparisons["exact_c"]["R"]
        same_parent_checks[str(year)] = {
            "point_positive": same["gain"] > 0.0,
            "cluster_ci_lower_positive": same["pitcher_cluster_95_ci"]["ci_low"] > 0.0,
        }
        years[str(year)] = {
            "rows": int(len(rows)), "R_rows": int(regular.sum()),
            "invalid_rows": int(invalid.sum()), "n_current_median": float(np.median(n)),
            "comparisons": comparisons,
        }
        output = PRED / f"v5_direct_season_update_dev_{year}.npz"
        if output.exists():
            raise FileExistsError(f"immutable artifact exists: {output}")
        np.savez_compressed(
            output, y=reference["y"].astype(np.int8), row_index=indices,
            cluster=reference["cluster"], parent_exact_c=parent,
            n_current=n, s_current=s, final_prediction=candidate,
        )
        artifacts[str(year)] = {"path": str(output.relative_to(ROOT)), "sha256": digest(output)}
        input_hashes[str(year)] = {name: digest(path) for name, path in anchor_paths.items()}

    g_dev = float(min(full_gains))
    threshold = float(lock["development_protocol"]["minimum_G_dev_strictly_greater_than"])
    same_parent_pass = all(all(item.values()) for item in same_parent_checks.values())
    development_pass = bool(same_parent_pass and g_dev > threshold)
    report = {
        "experiment_id": lock["experiment_id"],
        "status": "development_pass" if development_pass else "development_failed",
        "lock_sha256": digest(LOCK), "contract_sha256": digest(CONTRACT),
        "years_read": list(YEARS), "confirmation_2024_read": False,
        "rows_loaded_through_year": int(frame["season"].max()),
        "input_sha256": input_hashes, "years": years,
        "gates": {
            "same_parent_R": {"years": same_parent_checks, "pass": same_parent_pass},
            "G_dev": {"minimum_full_gain": g_dev, "required_strictly_greater_than": threshold, "pass": g_dev > threshold},
            "development_pass": development_pass,
        },
        "artifacts": artifacts,
        "confirmation_2024_authorized": development_pass,
        "goal_status": "active", "goal_completion_claimed": False,
    }
    REPORT.write_text(json.dumps(safe(report), ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(safe({"status": report["status"], "gates": report["gates"], "R_and_full": {
        str(y): {"R": years[str(y)]["comparisons"]["exact_c"]["R"], "full": years[str(y)]["comparisons"]["exact_c"]["full"]} for y in YEARS
    }}), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
