#!/usr/bin/env python3
"""Immutable source gate for low-cardinality official game-state lookups."""

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

from experiments.analyze_v5_dense_pitchtype_moe import load, safe, score  # noqa: E402
from experiments.run_v5_h1_residual import cluster_bootstrap_score_gain  # noqa: E402


YEARS = (2020, 2021)
TRAIN = ROOT / "open/data/train.csv"
PRED = ROOT / "experiments/results/predictions"
PREREG = ROOT / "experiments/params/v5_expectancy_state_lookup_preregister.json"
REPORT = ROOT / "experiments/results/v5_expectancy_state_lookup_source.json"
ARTIFACT_DIR = ROOT / "experiments/results/predictions"
PARENT_PATHS = {
    2020: PRED / "v4_m3_c_backtest_2020_2020.npz",
    2021: PRED / "v4_m3_c_backtest_2021_2021.npz",
}


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def lookup(table: pd.DataFrame, rows: pd.DataFrame, keys: list[str], column: str) -> np.ndarray:
    if len(keys) == 1:
        index = pd.Index(rows[keys[0]].to_numpy(), name=keys[0])
    else:
        index = pd.MultiIndex.from_frame(rows[keys])
    return table[column].reindex(index).fillna(0.0).to_numpy(dtype=np.float64)


def nested_direction(
    history: pd.DataFrame,
    target: pd.DataFrame,
    parent_keys: list[str],
    child_keys: list[str],
    parent_strength: float,
    child_strength: float,
) -> tuple[np.ndarray, dict[str, Any]]:
    global_rate = float(history["control_success"].mean())
    parent = history.groupby(parent_keys, sort=False, observed=True)[
        "control_success"
    ].agg(["sum", "size"])
    parent["rate"] = (
        parent["sum"] + parent_strength * global_rate
    ) / (parent["size"] + parent_strength)

    child = history.groupby(child_keys, sort=False, observed=True)[
        "control_success"
    ].agg(["sum", "size"])
    child_frame = child.reset_index()
    if len(parent_keys) == 1:
        child_parent_index = pd.Index(
            child_frame[parent_keys[0]].to_numpy(), name=parent_keys[0]
        )
    else:
        child_parent_index = pd.MultiIndex.from_frame(child_frame[parent_keys])
    child_parent_rate = parent["rate"].reindex(child_parent_index).fillna(
        global_rate
    ).to_numpy(dtype=np.float64)
    child_frame["direction"] = (
        child_frame["sum"].to_numpy(dtype=np.float64)
        + child_strength * child_parent_rate
    ) / (
        child_frame["size"].to_numpy(dtype=np.float64) + child_strength
    ) - child_parent_rate
    child_table = child_frame.set_index(child_keys)
    direction = lookup(child_table, target, child_keys, "direction")
    target_parent = lookup(parent, target, parent_keys, "rate")
    return direction, {
        "history_global_rate": global_rate,
        "parent_cells": int(len(parent)),
        "child_cells": int(len(child)),
        "target_seen_rate": float(np.mean(np.abs(direction) > 0.0)),
        "direction_mean": float(direction.mean()),
        "direction_std": float(direction.std()),
        "direction_max_abs": float(np.max(np.abs(direction))),
        "target_parent_mean": float(target_parent.mean()),
    }


def simple_metrics(
    y: np.ndarray, parent: np.ndarray, candidate: np.ndarray, regular: np.ndarray
) -> dict[str, dict[str, float]]:
    output: dict[str, dict[str, float]] = {}
    for route, mask in {
        "full": np.ones(len(y), dtype=bool),
        "R": regular,
        "F": ~regular,
    }.items():
        before = score(y, parent, mask)
        after = score(y, candidate, mask)
        output[route] = {
            "parent_score": float(before["score"]),
            "candidate_score": float(after["score"]),
            "gain": float(after["score"] - before["score"]),
        }
    return output


def detailed_metrics(
    y: np.ndarray,
    parent: np.ndarray,
    candidate: np.ndarray,
    cluster: np.ndarray,
    regular: np.ndarray,
    seed: int,
) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for offset, (route, mask) in enumerate(
        {
            "full": np.ones(len(y), dtype=bool),
            "R": regular,
            "F": ~regular,
        }.items()
    ):
        before = score(y, parent, mask)
        after = score(y, candidate, mask)
        interval = cluster_bootstrap_score_gain(
            y,
            parent,
            candidate,
            cluster,
            mask,
            iterations=2000,
            seed=seed + 1000 * offset,
        )
        output[route] = {
            "parent": before,
            "candidate": after,
            "gain": float(after["score"] - before["score"]),
            "pitcher_cluster_95_ci": interval,
        }
    return output


def main() -> None:
    if REPORT.exists():
        raise FileExistsError(f"immutable report exists: {REPORT}")
    prereg = json.loads(PREREG.read_text(encoding="utf-8"))
    if prereg["status"] != "locked_before_source_metrics":
        raise ValueError("unexpected preregistration status")

    source = prereg["source_protocol"]
    specs = source["specifications"]
    needed = {
        "season",
        "game_type",
        "pitcher_id",
        "balls_before",
        "strikes_before",
        "outs_before",
        "base_state",
        "inning",
        "score_diff_pitcher_team",
        "home_win_expectancy",
        "away_win_expectancy",
        "li",
        "control_success",
    }
    # The train file is chronological.  Reading through the final 2021 index
    # makes it impossible for this source program to touch a 2022+ label.
    frame = pd.read_csv(TRAIN, usecols=sorted(needed), nrows=728_588)
    if set(frame["season"].unique()) != {2019, 2020, 2021}:
        raise ValueError("source reader did not stop exactly at 2021")
    frame["inning_bucket"] = np.minimum(frame["inning"].to_numpy(), 10).astype(
        np.int8
    )
    frame["score_diff_bucket"] = np.clip(
        frame["score_diff_pitcher_team"].to_numpy(), -5, 5
    ).astype(np.int8)

    folds: dict[int, dict[str, Any]] = {}
    input_hashes: dict[str, str] = {}
    for year in YEARS:
        artifact = load(PARENT_PATHS[year])
        row_index = artifact["row_index"].astype(np.int64)
        rows = frame.loc[row_index]
        y = artifact["y"].astype(np.int8)
        if not rows["season"].eq(year).all():
            raise ValueError(f"{year}: row season mismatch")
        if not np.array_equal(rows["control_success"].to_numpy(dtype=np.int8), y):
            raise ValueError(f"{year}: target mismatch")
        folds[year] = {
            "rows": rows,
            "y": y,
            "parent": artifact["catboost_outcome"].astype(np.float64),
            "cluster": artifact["cluster"],
            "regular": rows["game_type"].astype(str).eq("R").to_numpy(),
            "history": frame.loc[
                frame["season"].lt(year) & frame["game_type"].eq("R")
            ],
        }
        input_hashes[str(year)] = digest(PARENT_PATHS[year])

    directions: dict[tuple[str, float, int], np.ndarray] = {}
    diagnostics: dict[str, Any] = {}
    trials: list[dict[str, Any]] = []
    for spec_name, spec in specs.items():
        parent_keys = list(spec["parent"])
        child_keys = list(spec["child"])
        for strength_value in source["child_strength_grid"]:
            strength = float(strength_value)
            year_directions: dict[int, np.ndarray] = {}
            diagnostic: dict[str, Any] = {}
            for year in YEARS:
                direction, meta = nested_direction(
                    folds[year]["history"],
                    folds[year]["rows"],
                    parent_keys,
                    child_keys,
                    float(source["parent_strength"]),
                    strength,
                )
                direction[~folds[year]["regular"]] = 0.0
                year_directions[year] = direction
                directions[(spec_name, strength, year)] = direction
                diagnostic[str(year)] = meta
            diagnostics[f"{spec_name}|{strength:g}"] = diagnostic
            for gamma_value in source["gamma_grid"]:
                gamma = float(gamma_value)
                metrics: dict[str, Any] = {}
                for year in YEARS:
                    candidate = np.clip(
                        folds[year]["parent"] + gamma * year_directions[year],
                        1e-6,
                        1.0 - 1e-6,
                    )
                    metrics[str(year)] = simple_metrics(
                        folds[year]["y"],
                        folds[year]["parent"],
                        candidate,
                        folds[year]["regular"],
                    )
                trials.append(
                    {
                        "specification": spec_name,
                        "child_strength": strength,
                        "gamma": gamma,
                        "minimum_full_gain": float(
                            min(metrics[str(year)]["full"]["gain"] for year in YEARS)
                        ),
                        "minimum_R_gain": float(
                            min(metrics[str(year)]["R"]["gain"] for year in YEARS)
                        ),
                        "mean_full_gain": float(
                            np.mean(
                                [metrics[str(year)]["full"]["gain"] for year in YEARS]
                            )
                        ),
                        "years": metrics,
                    }
                )
    selected = max(
        trials,
        key=lambda item: (
            item["minimum_full_gain"],
            item["minimum_R_gain"],
            item["mean_full_gain"],
            -item["gamma"],
            item["child_strength"],
            item["specification"],
        ),
    )

    selected_metrics: dict[str, Any] = {}
    artifacts: dict[str, Any] = {}
    spec_name = str(selected["specification"])
    strength = float(selected["child_strength"])
    gamma = float(selected["gamma"])
    for offset, year in enumerate(YEARS):
        direction = directions[(spec_name, strength, year)]
        candidate = np.clip(
            folds[year]["parent"] + gamma * direction, 1e-6, 1.0 - 1e-6
        )
        selected_metrics[str(year)] = detailed_metrics(
            folds[year]["y"],
            folds[year]["parent"],
            candidate,
            folds[year]["cluster"],
            folds[year]["regular"],
            8227000 + 10000 * offset,
        )
        output = ARTIFACT_DIR / f"v5_expectancy_state_lookup_selected_source_{year}.npz"
        if output.exists():
            raise FileExistsError(f"immutable artifact exists: {output}")
        np.savez_compressed(
            output,
            y=folds[year]["y"],
            row_index=folds[year]["rows"].index.to_numpy(dtype=np.int64),
            cluster=folds[year]["cluster"],
            parent=folds[year]["parent"].astype(np.float32),
            lookup_direction=direction.astype(np.float32),
            final_prediction=candidate.astype(np.float32),
        )
        artifacts[str(year)] = {
            "path": str(output.relative_to(ROOT)),
            "sha256": digest(output),
        }

    gate = source["advance_gate"]
    checks: dict[str, bool] = {}
    for year in YEARS:
        result = selected_metrics[str(year)]
        checks[f"{year}_full_gain"] = bool(
            result["full"]["gain"] >= float(gate["minimum_full_gain_each_year"])
        )
        checks[f"{year}_R_gain"] = bool(
            result["R"]["gain"] >= float(gate["minimum_R_gain_each_year"])
        )
        checks[f"{year}_full_ci"] = bool(
            result["full"]["pitcher_cluster_95_ci"]["ci_low"]
            > float(gate["full_pitcher_cluster_95_ci_low_each_year"])
        )
        checks[f"{year}_R_ci"] = bool(
            result["R"]["pitcher_cluster_95_ci"]["ci_low"]
            > float(gate["R_pitcher_cluster_95_ci_low_each_year"])
        )
    passed = bool(all(checks.values()))
    report = {
        "experiment_id": prereg["experiment_id"],
        "status": "source_pass" if passed else "source_failed",
        "preregister_sha256": digest(PREREG),
        "script_sha256": digest(Path(__file__)),
        "years_read": list(YEARS),
        "years_not_read": [2022, 2023, 2024],
        "raw_train_max_season": int(frame["season"].max()),
        "selection": selected,
        "selected_metrics": selected_metrics,
        "top_trials": sorted(
            trials,
            key=lambda item: (
                item["minimum_full_gain"],
                item["minimum_R_gain"],
                item["mean_full_gain"],
            ),
            reverse=True,
        )[:30],
        "diagnostics": diagnostics,
        "gate": {"requirements": gate, "checks": checks, "pass": passed},
        "input_sha256": input_hashes,
        "artifacts": artifacts,
        "goal_status": "active",
        "goal_completion_claimed": False,
    }
    REPORT.write_text(
        json.dumps(safe(report), ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(
        json.dumps(
            safe(
                {
                    "status": report["status"],
                    "selection": selected,
                    "selected_metrics": selected_metrics,
                    "gate": report["gate"],
                }
            ),
            ensure_ascii=False,
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
