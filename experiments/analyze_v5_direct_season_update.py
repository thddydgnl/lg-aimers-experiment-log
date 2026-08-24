#!/usr/bin/env python3
"""Immutable 2020/2021 gate for a direct current-season Bayesian update."""

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
PRED = ROOT / "experiments/results/predictions"
PREREG = ROOT / "experiments/params/v5_direct_season_update_preregister.json"
REPORT = ROOT / "experiments/results/v5_direct_season_update_source.json"
YEARS = (2020, 2021)
FORMULAS = ("anchor_posterior", "preseason_delta", "career_delta")


def load_source_rows() -> pd.DataFrame:
    """Read no target or feature row from a season later than 2021."""
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
        raise ValueError("source loader did not end at 2021")
    return frame


def states_before(frame: pd.DataFrame) -> dict[int, dict[int, tuple[int, int]]]:
    before: dict[int, dict[int, tuple[int, int]]] = {}
    state: dict[int, tuple[int, int]] = {}
    for year in sorted(frame["season"].astype(int).unique()):
        before[int(year)] = dict(state)
        block = frame.loc[frame["season"].eq(year)]
        last = block.groupby("pitcher_id", sort=False, observed=True).tail(1)
        for row in last.itertuples(index=False):
            n = int(row.asof_pitcher_n or 0)
            rate = 0.0 if pd.isna(row.asof_pitcher_success_rate) else float(row.asof_pitcher_success_rate)
            state[int(row.pitcher_id)] = (
                n + 1,
                int(np.rint(rate * n)) + int(row.control_success),
            )
    return before


def reconstruct(
    rows: pd.DataFrame, year: int, before: dict[int, dict[int, tuple[int, int]]]
) -> dict[str, np.ndarray]:
    frozen = before[year]
    pitchers = rows["pitcher_id"].to_numpy(dtype=np.int64)
    n_end = np.fromiter((frozen.get(int(p), (0, 0))[0] for p in pitchers), dtype=np.int64)
    s_end = np.fromiter((frozen.get(int(p), (0, 0))[1] for p in pitchers), dtype=np.int64)
    n_asof = rows["asof_pitcher_n"].fillna(0).to_numpy(dtype=np.int64)
    career = rows["asof_pitcher_success_rate"].fillna(0.5).to_numpy(dtype=np.float64)
    s_asof = np.rint(career * n_asof).astype(np.int64)
    n_cur = n_asof - n_end
    s_cur = s_asof - s_end
    invalid = (n_cur < 0) | (s_cur < 0) | (s_cur > n_cur)
    n_cur = np.where(invalid, 0, n_cur).astype(np.float64)
    s_cur = np.where(invalid, 0, s_cur).astype(np.float64)
    preseason = np.divide(
        s_end,
        n_end,
        out=np.full(len(rows), float(rows["control_success"].mean()), dtype=np.float64),
        where=n_end > 0,
    )
    raw = np.divide(s_cur, n_cur, out=preseason.copy(), where=n_cur > 0)
    return {
        "n": n_cur, "s": s_cur, "preseason": preseason, "raw": raw,
        "career": career, "invalid": invalid,
    }


def direction(name: str, parent: np.ndarray, state: dict[str, np.ndarray], k: float) -> np.ndarray:
    n = state["n"]
    reliability = n / (n + k)
    if name == "anchor_posterior":
        value = (state["s"] + k * parent) / (n + k)
    elif name == "preseason_delta":
        value = parent + reliability * (state["raw"] - state["preseason"])
    elif name == "career_delta":
        value = parent + reliability * (state["raw"] - state["career"])
    else:
        raise KeyError(name)
    value = np.where(state["invalid"], parent, value)
    return np.clip(value, 1e-6, 1.0 - 1e-6)


def route_prediction(
    name: str, parent: np.ndarray, state: dict[str, np.ndarray], regular: np.ndarray, k_r: float
) -> np.ndarray:
    pred_r = direction(name, parent, state, k_r)
    pred_f = direction(name, parent, state, 20.0)
    return np.where(regular, pred_r, pred_f)


def route_metrics(
    artifact: dict[str, np.ndarray], parent: np.ndarray, candidate: np.ndarray,
    masks: dict[str, np.ndarray], bootstrap: int, seed: int,
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for route_index, (route, mask) in enumerate(masks.items()):
        base = score(artifact["y"], parent, mask)
        cand = score(artifact["y"], candidate, mask)
        interval = cluster_bootstrap_score_gain(
            artifact["y"], parent, candidate, artifact["cluster"], mask,
            iterations=bootstrap, seed=seed + 1000 * route_index,
        )
        result[route] = {
            "parent": base, "candidate": cand,
            "gain": float(cand["score"] - base["score"]),
            "pitcher_cluster_95_ci": interval,
        }
    return result


def main() -> None:
    if REPORT.exists():
        raise FileExistsError(f"immutable report exists: {REPORT}")
    prereg = json.loads(PREREG.read_text(encoding="utf-8"))
    frame = load_source_rows()
    before = states_before(frame)
    folds: dict[int, dict[str, Any]] = {}
    for year in YEARS:
        path = PRED / f"v4_m3_c_backtest_{year}_{year}.npz"
        artifact = load(path)
        indices = artifact["row_index"].astype(np.int64)
        rows = frame.loc[indices]
        if not np.array_equal(rows["control_success"].to_numpy(dtype=np.int8), artifact["y"].astype(np.int8)):
            raise ValueError(f"target alignment mismatch: {year}")
        if not rows["season"].eq(year).all():
            raise ValueError(f"season alignment mismatch: {year}")
        regular = rows["game_type"].astype(str).eq("R").to_numpy()
        parent = artifact["catboost_outcome"].astype(np.float64)
        folds[year] = {
            "artifact": artifact, "parent": parent,
            "state": reconstruct(rows, year, before), "regular": regular,
            "masks": {"full": np.ones(len(rows), dtype=bool), "R": regular, "F": ~regular},
            "path": path,
        }

    bootstrap = int(prereg["source_selection"]["bootstrap_iterations"])
    trials: list[dict[str, Any]] = []
    cache: dict[tuple[str, float, int], np.ndarray] = {}
    for formula_index, formula in enumerate(FORMULAS):
        for k in (float(v) for v in prereg["k_grid"]):
            years: dict[str, Any] = {}
            for year in YEARS:
                fold = folds[year]
                candidate = route_prediction(formula, fold["parent"], fold["state"], fold["regular"], k)
                cache[(formula, k, year)] = candidate
                years[str(year)] = route_metrics(
                    fold["artifact"], fold["parent"], candidate, fold["masks"],
                    bootstrap, 8120000 + formula_index * 100000 + int(k) * 10 + year,
                )
            trials.append({
                "formula": formula, "formula_order": formula_index, "k_R": k, "k_F": 20.0,
                "minimum_R_gain": float(min(years[str(y)]["R"]["gain"] for y in YEARS)),
                "minimum_full_gain": float(min(years[str(y)]["full"]["gain"] for y in YEARS)),
                "mean_R_gain": float(np.mean([years[str(y)]["R"]["gain"] for y in YEARS])),
                "years": years,
            })
    selected = max(trials, key=lambda x: (
        x["minimum_R_gain"], x["minimum_full_gain"], x["mean_R_gain"], x["k_R"], -x["formula_order"]
    ))
    gate = prereg["source_selection"]["gate"]
    checks: dict[str, Any] = {}
    passed = True
    for year in YEARS:
        routes = selected["years"][str(year)]
        year_checks = {
            "R_point": routes["R"]["gain"] >= float(gate["minimum_R_gain_each_year"]),
            "full_point": routes["full"]["gain"] >= float(gate["minimum_full_gain_each_year"]),
            "R_ci": routes["R"]["pitcher_cluster_95_ci"]["ci_low"] > 0.0,
            "full_ci": routes["full"]["pitcher_cluster_95_ci"]["ci_low"] > 0.0,
            "R_nonnegative": routes["R"]["gain"] >= 0.0,
            "F_nonnegative": routes["F"]["gain"] >= 0.0,
        }
        checks[str(year)] = year_checks
        passed = passed and all(year_checks.values())

    artifacts: dict[str, Any] = {}
    for year in YEARS:
        fold = folds[year]
        output = PRED / f"v5_direct_season_update_source_{year}.npz"
        if output.exists():
            raise FileExistsError(f"immutable artifact exists: {output}")
        candidate = cache[(selected["formula"], selected["k_R"], year)]
        np.savez_compressed(
            output, y=fold["artifact"]["y"].astype(np.int8),
            row_index=fold["artifact"]["row_index"].astype(np.int64),
            cluster=fold["artifact"]["cluster"], parent_exact_c=fold["parent"],
            final_prediction=candidate, n_current=fold["state"]["n"],
            s_current=fold["state"]["s"], invalid=fold["state"]["invalid"].astype(np.int8),
        )
        artifacts[str(year)] = {"path": str(output.relative_to(ROOT)), "sha256": digest(output)}

    report = {
        "experiment_id": prereg["experiment_id"],
        "status": "source_pass" if passed else "source_failed",
        "preregister_sha256": digest(PREREG), "script_sha256": digest(Path(__file__)),
        "years_read": list(YEARS), "years_not_read": [2022, 2023, 2024],
        "source_rows_loaded": int(len(frame)),
        "input_sha256": {str(y): digest(folds[y]["path"]) for y in YEARS},
        "trials": trials, "selected": selected,
        "source_gate": {"requirements": gate, "checks": checks, "pass": bool(passed)},
        "artifacts": artifacts, "goal_status": "active", "goal_completion_claimed": False,
    }
    REPORT.write_text(json.dumps(safe(report), ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(safe({"status": report["status"], "selected": selected, "checks": checks}), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
