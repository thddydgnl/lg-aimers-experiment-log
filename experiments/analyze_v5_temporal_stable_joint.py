#!/usr/bin/env python3
"""Immutable two-year source gate for temporal-stable direct features."""

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

from experiments.analyze_v5_dense_pitchtype_moe import (  # noqa: E402
    evaluate,
    load,
    safe,
)


RESULTS = ROOT / "experiments/results"
PREDICTIONS = RESULTS / "predictions"
PREREG = ROOT / "experiments/params/v5_temporal_stable_joint_preregister.json"
REPORT = RESULTS / "v5_temporal_stable_joint_source_gate.json"
TRAIN = ROOT / "open/data/train.csv"
YEARS = (2020, 2021)
KEY = "catboost_outcome"
STAGES = {
    2020: "v5_temporal_stable_joint_source2020",
    2021: "v5_temporal_stable_joint_source2021",
}


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def semantic_audit(metadata: dict[str, Any], year: int) -> dict[str, Any]:
    feature = metadata["folds"][0]["temporal_stable_joint"]
    train = feature["train"]
    valid = feature["valid"]
    expected_strengths = {
        "pitcher": 100.0,
        "hand": 38.0,
        "pressure_hand": 30.0,
    }
    train_seasons = train["seasons"]
    valid_season = valid["seasons"][str(year)]
    checks = {
        "enabled": bool(feature["enabled"]),
        "strengths": train["strengths"] == expected_strengths
        and valid["strengths"] == expected_strengths,
        "feature_count": int(train["feature_count"]) == 18
        and int(valid["feature_count"]) == 18,
        "same_columns": train["feature_columns"] == valid["feature_columns"],
        "e96_columns_present": all(
            column in metadata["folds"][0]["feature_columns"]
            for column in train["feature_columns"]
        ),
        "train_self_exclusion": all(
            int(details["self_excluded_rows"]) > 0
            and int(details["source_season"]) == int(season)
            for season, details in train_seasons.items()
        ),
        "validation_latest_completed": int(valid_season["source_season"])
        == year - 1,
        "validation_no_self_exclusion": int(valid_season["self_excluded_rows"])
        == 0,
        "row_independent_validation": bool(valid["row_independent_validation"]),
        "source_scope_R": train["source_scope"] == "R"
        and valid["source_scope"] == "R",
    }
    return {
        "pass": bool(all(checks.values())),
        "checks": checks,
        "strengths": train["strengths"],
        "feature_columns": train["feature_columns"],
        "train_seasons": train_seasons,
        "validation_season": valid_season,
    }


def main() -> None:
    if REPORT.exists():
        raise FileExistsError(f"immutable report already exists: {REPORT}")
    prereg = json.loads(PREREG.read_text(encoding="utf-8"))
    game_types = pd.read_csv(TRAIN, usecols=["game_type"])["game_type"].astype(str)
    folds: dict[int, dict[str, Any]] = {}
    for year in YEARS:
        candidate_path = (
            PREDICTIONS / f"{STAGES[year]}_{year}.npz"
        )
        parent_path = (
            PREDICTIONS / f"v4_m3_c_backtest_{year}_{year}.npz"
        )
        metadata_path = RESULTS / f"{STAGES[year]}.json"
        candidate = load(candidate_path)
        parent_artifact = load(parent_path)
        for align_key in ("y", "row_index", "cluster"):
            if not np.array_equal(candidate[align_key], parent_artifact[align_key]):
                raise ValueError(f"alignment mismatch: {year}/{align_key}")
        types = game_types.iloc[
            candidate["row_index"].astype(np.int64)
        ].to_numpy(dtype=str)
        regular = types == "R"
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        folds[year] = {
            "artifact": candidate,
            "direction": candidate[KEY].astype(np.float64),
            "parent": parent_artifact[KEY].astype(np.float64),
            "regular": regular,
            "masks": {
                "full": np.ones(len(regular), dtype=bool),
                "R": regular,
            },
            "semantic": semantic_audit(metadata, year),
            "paths": {
                "candidate": candidate_path,
                "parent": parent_path,
                "metadata": metadata_path,
            },
        }

    semantic_pass = bool(all(folds[year]["semantic"]["pass"] for year in YEARS))
    trials: list[dict[str, Any]] = []
    if semantic_pass:
        iterations = int(
            prereg["source_protocol"]["bootstrap_replicates"]
        )
        for gamma in prereg["source_protocol"]["blend_gamma_grid"]:
            per_year: dict[str, Any] = {}
            for year in YEARS:
                fold = folds[year]
                per_year[str(year)] = evaluate(
                    fold["artifact"],
                    fold["parent"],
                    fold["direction"],
                    fold["regular"],
                    fold["masks"],
                    float(gamma),
                    iterations,
                    int(prereg["source_protocol"]["bootstrap_seed"])
                    + year * 1000
                    + int(float(gamma) * 100),
                )
            r_gains = [
                per_year[str(year)]["routes"]["R"]["gain"] for year in YEARS
            ]
            full_gains = [
                per_year[str(year)]["routes"]["full"]["gain"]
                for year in YEARS
            ]
            r_ci_lows = [
                per_year[str(year)]["routes"]["R"][
                    "pitcher_cluster_95_ci"
                ]["ci_low"]
                for year in YEARS
            ]
            trials.append(
                {
                    "gamma": float(gamma),
                    "minimum_R_gain": float(min(r_gains)),
                    "mean_R_gain": float(np.mean(r_gains)),
                    "minimum_R_CI_lower": float(min(r_ci_lows)),
                    "minimum_full_gain": float(min(full_gains)),
                    "years": per_year,
                }
            )
    selected = (
        max(
            trials,
            key=lambda item: (
                item["minimum_R_gain"],
                item["mean_R_gain"],
                item["minimum_R_CI_lower"],
                -item["gamma"],
            ),
        )
        if trials
        else None
    )

    gate = prereg["source_protocol"]["gate"]
    checks: dict[str, bool] = {
        "semantic": semantic_pass,
        "selected": selected is not None,
    }
    if selected is not None:
        for year in YEARS:
            routes = selected["years"][str(year)]["routes"]
            checks[f"{year}_R_gain"] = bool(
                routes["R"]["gain"]
                >= float(gate["minimum_R_raw_gain_each_year"])
            )
            checks[f"{year}_R_CI"] = bool(
                routes["R"]["pitcher_cluster_95_ci"]["ci_low"]
                > float(gate["minimum_R_cluster_CI_lower_each_year"])
            )
            checks[f"{year}_full_gain"] = bool(
                routes["full"]["gain"]
                > float(gate["minimum_routed_full_raw_gain_each_year"])
            )
    passed = bool(all(checks.values()))
    report = {
        "experiment_id": prereg["experiment_id"],
        "status": "source_pass" if passed else "source_failed",
        "preregister_sha256": digest(PREREG),
        "script_sha256": digest(Path(__file__)),
        "years_read": list(YEARS),
        "years_not_read": [2022, 2023, 2024],
        "semantic": {
            str(year): folds[year]["semantic"] for year in YEARS
        },
        "artifacts": {
            str(year): {
                name: {
                    "path": str(path.relative_to(ROOT)),
                    "sha256": digest(path),
                }
                for name, path in folds[year]["paths"].items()
            }
            for year in YEARS
        },
        "trials": trials,
        "selected": selected,
        "source_gate": {
            "requirements": gate,
            "checks": checks,
            "passed": passed,
            "decision": (
                "freeze gamma and advance"
                if passed
                else "close without generating or reading 2022+ candidate predictions"
            ),
        },
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
                    "selected": selected,
                    "checks": checks,
                    "semantic": report["semantic"],
                }
            ),
            ensure_ascii=False,
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
