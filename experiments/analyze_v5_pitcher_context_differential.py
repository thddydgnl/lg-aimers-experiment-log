#!/usr/bin/env python3
"""Source-only transfer audit for effect-coded pitcher context differentials."""

from __future__ import annotations

import hashlib
import itertools
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.stats import paired_bootstrap_brier_ci  # noqa: E402

TRAIN = ROOT / "open" / "data" / "train.csv"
PRED = ROOT / "experiments" / "results" / "predictions"
PREREG = ROOT / "experiments" / "params" / "v5_pitcher_context_differential_preregister.json"
REPORT = ROOT / "experiments" / "results" / "v5_pitcher_context_differential_source.json"
PARENTS = {
    2020: PRED / "v4_m3_c_backtest_2020_2020.npz",
    2021: PRED / "v4_m3_c_backtest_2021_2021.npz",
}
CONTEXTS = ("batter_hand", "two_strike", "runners")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def score(y: np.ndarray, prediction: np.ndarray) -> float:
    reference = float(y.mean() * (1.0 - y.mean()))
    brier = float(np.mean(np.square(prediction - y)))
    return max(0.0, 100_000.0 * (1.0 - brier / reference))


def add_contexts(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    result["batter_hand"] = result["batter_hand"].astype(str)
    result["two_strike"] = result["strikes_before"].eq(2).astype(np.int8)
    result["runners"] = result["num_runners_on"].gt(0).astype(np.int8)
    return result


def load_source() -> pd.DataFrame:
    with np.load(PARENTS[2021], allow_pickle=False) as z:
        last_index = int(z["row_index"].max())
    frame = pd.read_csv(
        TRAIN,
        usecols=[
            "season",
            "game_type",
            "pitcher_id",
            "batter_hand",
            "strikes_before",
            "num_runners_on",
            "control_success",
        ],
        nrows=last_index + 1,
    )
    if set(frame["season"].unique()) != {2019, 2020, 2021}:
        raise ValueError("Source audit parsed a label after 2021")
    return add_contexts(frame)


def load_parent(year: int, frame: pd.DataFrame) -> dict[str, np.ndarray]:
    with np.load(PARENTS[year], allow_pickle=False) as z:
        row_index = z["row_index"].astype(np.int64)
        result = {
            "row_index": row_index,
            "y": z["y"].astype(np.float64),
            "cluster": z["cluster"].astype(str),
            "parent": z["catboost_outcome"].astype(np.float64),
        }
    view = frame.iloc[row_index]
    if not view["season"].eq(year).all():
        raise ValueError(f"{year}: parent season mismatch")
    if not np.array_equal(
        view["control_success"].to_numpy(dtype=np.int8),
        result["y"].astype(np.int8),
    ):
        raise ValueError(f"{year}: parent target mismatch")
    result["regular"] = view["game_type"].eq("R").to_numpy()
    return result


def effect_direction(
    history: pd.DataFrame,
    valid: pd.DataFrame,
    context: str,
    k: float,
) -> np.ndarray:
    history = history.loc[history["game_type"].eq("R")]
    pitcher = history.groupby("pitcher_id", sort=False, observed=True)[
        "control_success"
    ].agg(["sum", "count"])
    pitcher_mean = pitcher["sum"] / pitcher["count"]
    cells = history.groupby(
        ["pitcher_id", context], sort=False, observed=True
    )["control_success"].agg(["sum", "count"])
    base = cells.index.get_level_values("pitcher_id").map(pitcher_mean)
    raw = (cells["sum"].to_numpy() + k * base.to_numpy()) / (
        cells["count"].to_numpy() + k
    ) - base.to_numpy()
    table = cells.reset_index()[["pitcher_id", context, "count"]].copy()
    table["raw"] = raw
    weighted_sum = (table["raw"] * table["count"]).groupby(table["pitcher_id"]).transform("sum")
    total_count = table["count"].groupby(table["pitcher_id"]).transform("sum")
    table["effect"] = table["raw"] - weighted_sum / total_count
    mapping = {
        (int(row.pitcher_id), str(getattr(row, context))): float(row.effect)
        for row in table.itertuples(index=False)
    }
    return np.asarray(
        [
            mapping.get((int(pitcher_id), str(value)), 0.0)
            for pitcher_id, value in zip(
                valid["pitcher_id"].to_numpy(),
                valid[context].to_numpy(),
                strict=True,
            )
        ],
        dtype=np.float64,
    )


def main() -> None:
    prereg = json.loads(PREREG.read_text(encoding="utf-8"))
    frame = load_source()
    parents = {year: load_parent(year, frame) for year in (2020, 2021)}
    directions: dict[tuple[int, str, int, str], np.ndarray] = {}
    windows = prereg["source_grid"]["history_windows"]
    ks = [int(value) for value in prereg["source_grid"]["shrinkage_k"]]
    for year in (2020, 2021):
        valid = frame.iloc[parents[year]["row_index"]]
        for window in windows:
            history = frame.loc[frame["season"].lt(year)]
            if window != "all":
                history = history.loc[history["season"].ge(year - int(window))]
            for k in ks:
                for context in CONTEXTS:
                    directions[(year, str(window), k, context)] = effect_direction(
                        history, valid, context, float(k)
                    )

    candidates = []
    subsets = [tuple(value) for value in prereg["source_grid"]["context_subsets"]]
    for window, k, subset, gamma in itertools.product(
        windows, ks, subsets, prereg["source_grid"]["gammas"]
    ):
        years = {}
        for year in (2020, 2021):
            fold = parents[year]
            direction = sum(
                (directions[(year, str(window), k, context)] for context in subset),
                start=np.zeros(len(fold["y"]), dtype=np.float64),
            )
            prediction = fold["parent"].copy()
            regular = fold["regular"]
            prediction[regular] = np.clip(
                prediction[regular] + float(gamma) * direction[regular],
                1e-6,
                1.0 - 1e-6,
            )
            years[str(year)] = {
                "full_gain": score(fold["y"], prediction)
                - score(fold["y"], fold["parent"]),
                "r_gain": score(fold["y"][regular], prediction[regular])
                - score(fold["y"][regular], fold["parent"][regular]),
            }
        candidates.append(
            {
                "window": window,
                "k": k,
                "contexts": list(subset),
                "gamma": float(gamma),
                "min_full_gain": min(value["full_gain"] for value in years.values()),
                "min_r_gain": min(value["r_gain"] for value in years.values()),
                "mean_full_gain": float(
                    np.mean([value["full_gain"] for value in years.values()])
                ),
                "years": years,
            }
        )
    selected = max(
        candidates,
        key=lambda item: (
            item["min_full_gain"],
            item["min_r_gain"],
            item["mean_full_gain"],
            -len(item["contexts"]),
            -item["gamma"],
            item["k"],
            item["window"] == "all",
        ),
    )
    intervals = {}
    for offset, year in enumerate((2020, 2021)):
        fold = parents[year]
        direction = sum(
            (
                directions[
                    (year, str(selected["window"]), selected["k"], context)
                ]
                for context in selected["contexts"]
            ),
            start=np.zeros(len(fold["y"]), dtype=np.float64),
        )
        regular = fold["regular"]
        prediction = fold["parent"].copy()
        prediction[regular] = np.clip(
            prediction[regular] + selected["gamma"] * direction[regular],
            1e-6,
            1.0 - 1e-6,
        )
        intervals[str(year)] = paired_bootstrap_brier_ci(
            fold["y"][regular],
            fold["parent"][regular],
            prediction[regular],
            iterations=2000,
            seed=52700 + offset,
            clusters=fold["cluster"][regular],
        )
    gate = prereg["source_gate"]
    conditions = {
        "minimum_full_gain": bool(
            selected["min_full_gain"] >= float(gate["minimum_full_gain_each_year"])
        ),
        "minimum_r_gain": bool(
            selected["min_r_gain"] >= float(gate["minimum_r_gain_each_year"])
        ),
        "ci_lower_positive_each_year": bool(
            all(value["score_ci_low"] > 0.0 for value in intervals.values())
        ),
    }
    passed = bool(all(conditions.values()))
    report = {
        "experiment_id": prereg["experiment_id"],
        "status": "passed_source_gate" if passed else "failed_source_gate",
        "preregister_sha256": sha256(PREREG),
        "policy": {
            "test_rows_read": False,
            "latest_label_season_read": 2021,
            "row_independent": True,
        },
        "candidate_count": len(candidates),
        "selected": selected,
        "selected_r_cluster_intervals": intervals,
        "conditions": conditions,
        "gate_pass": passed,
        "decision": "freeze before 2022" if passed else "close without 2022+",
        "top_candidates": sorted(
            candidates,
            key=lambda item: (item["min_full_gain"], item["min_r_gain"]),
            reverse=True,
        )[:25],
        "artifact_hashes": {
            str(path.relative_to(ROOT)).replace("\\", "/"): sha256(path)
            for path in [TRAIN, PARENTS[2020], PARENTS[2021]]
        },
    }
    REPORT.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({
        "status": report["status"],
        "selected": selected,
        "intervals": intervals,
        "conditions": conditions,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
