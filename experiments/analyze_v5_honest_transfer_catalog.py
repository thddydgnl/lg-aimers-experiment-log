#!/usr/bin/env python3
"""Re-screen existing OOF directions against target-blind historical anchors.

This is a discovery catalog only.  A survivor can still be contaminated by
post-hoc recipe selection or inconsistent prediction-key semantics and must be
retrained from a new preregistration before it can become confirmation
evidence.
"""

from __future__ import annotations

import json
from pathlib import Path
import re

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
PREDICTIONS = ROOT / "experiments/results/predictions"
OUTPUT = ROOT / "experiments/results/v5_honest_transfer_catalog.json"
TRAIN = ROOT / "open/data/train.csv"
YEARS = (2022, 2023, 2024)
DEVELOPMENT_YEARS = (2022, 2023)
ANCHORS = {
    "r_identity": "v5_honest_m3_r_identity",
    "r_grid": "v5_honest_m3_r_grid",
}
GAMMAS = (0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.75, 1.00)
YEAR_PATTERN = re.compile(r"^(?P<stem>.+)_(?P<year>2022|2023|2024)\.npz$")
NON_PREDICTION_KEYS = {
    "y",
    "target",
    "row_index",
    "cluster",
    "game_type_r",
    "game_type_f",
    "route_r",
    "route_f",
}


def load(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        return {key: np.asarray(archive[key]) for key in archive.files}


def raw_score(y: np.ndarray, prediction: np.ndarray, mask: np.ndarray) -> float:
    target = np.asarray(y[mask], dtype=np.float64)
    pred = np.clip(np.asarray(prediction[mask], dtype=np.float64), 1e-6, 1.0 - 1e-6)
    rate = float(target.mean())
    reference = rate * (1.0 - rate)
    brier = float(np.mean(np.square(pred - target)))
    return 100_000.0 * (1.0 - brier / reference)


def probability_like(values: np.ndarray, expected_rows: int) -> bool:
    return bool(
        values.ndim == 1
        and len(values) == expected_rows
        and np.issubdtype(values.dtype, np.number)
        and np.isfinite(values).all()
        and float(values.min()) >= 0.0
        and float(values.max()) <= 1.0
        and 0.10 <= float(values.mean()) <= 0.90
        and float(values.std()) >= 0.002
    )


def route_prediction(
    anchor: np.ndarray,
    candidate: np.ndarray,
    gamma: float,
    route: np.ndarray,
) -> np.ndarray:
    prediction = np.asarray(anchor, dtype=np.float64).copy()
    prediction[route] += gamma * (
        np.asarray(candidate[route], dtype=np.float64) - prediction[route]
    )
    return np.clip(prediction, 1e-6, 1.0 - 1e-6)


def main() -> None:
    anchors = {
        name: {
            year: load(PREDICTIONS / f"{stem}_{year}.npz") for year in YEARS
        }
        for name, stem in ANCHORS.items()
    }
    reference = anchors["r_identity"]
    all_game_types = pd.read_csv(TRAIN, usecols=["game_type"])["game_type"].to_numpy()
    masks = {
        year: {
            "all": np.ones(len(reference[year]["y"]), dtype=bool),
            "R": all_game_types[reference[year]["row_index"]] == "R",
        }
        for year in YEARS
    }

    paths_by_stem: dict[str, dict[int, Path]] = {}
    for path in PREDICTIONS.glob("*.npz"):
        match = YEAR_PATTERN.match(path.name)
        if match:
            paths_by_stem.setdefault(match.group("stem"), {})[
                int(match.group("year"))
            ] = path
    common_stems = sorted(
        stem
        for stem, paths in paths_by_stem.items()
        if all(year in paths for year in YEARS) and stem not in ANCHORS.values()
    )

    rows: list[dict[str, object]] = []
    failures: list[dict[str, str]] = []
    for stem in common_stems:
        try:
            artifacts = {year: load(paths_by_stem[stem][year]) for year in YEARS}
            for year in YEARS:
                for key in ("y", "row_index", "cluster"):
                    if not np.array_equal(artifacts[year].get(key), reference[year][key]):
                        raise ValueError(f"{key} mismatch in {year}")
            common_keys = set.intersection(*(set(artifacts[year]) for year in YEARS))
            for key in sorted(common_keys):
                if key.lower() in NON_PREDICTION_KEYS:
                    continue
                if any(
                    np.array_equal(artifacts[year][key], reference[year]["y"])
                    for year in YEARS
                ):
                    continue
                if not all(
                    probability_like(artifacts[year][key], len(reference[year]["y"]))
                    for year in YEARS
                ):
                    continue

                gamma_trials: list[dict[str, object]] = []
                for gamma in GAMMAS:
                    development: dict[str, dict[str, float]] = {}
                    gains: list[float] = []
                    for anchor_name in ANCHORS:
                        development[anchor_name] = {}
                        for year in DEVELOPMENT_YEARS:
                            anchor = anchors[anchor_name][year]["final_prediction"]
                            candidate = artifacts[year][key]
                            prediction = route_prediction(anchor, candidate, gamma, masks[year]["R"])
                            gain = raw_score(
                                reference[year]["y"], prediction, masks[year]["R"]
                            ) - raw_score(reference[year]["y"], anchor, masks[year]["R"])
                            development[anchor_name][str(year)] = gain
                            gains.append(gain)
                    gamma_trials.append(
                        {
                            "gamma": gamma,
                            "development_R_gains": development,
                            "minimum_development_R_gain": float(min(gains)),
                            "median_development_R_gain": float(np.median(gains)),
                        }
                    )
                selected = sorted(
                    gamma_trials,
                    key=lambda trial: (
                        -float(trial["minimum_development_R_gain"]),
                        -float(trial["median_development_R_gain"]),
                        float(trial["gamma"]),
                    ),
                )[0]
                gamma = float(selected["gamma"])
                per_anchor: dict[str, dict[str, dict[str, float]]] = {}
                all_r_gains: list[float] = []
                for anchor_name in ANCHORS:
                    per_anchor[anchor_name] = {}
                    for year in YEARS:
                        anchor = anchors[anchor_name][year]["final_prediction"]
                        prediction = route_prediction(
                            anchor, artifacts[year][key], gamma, masks[year]["R"]
                        )
                        r_gain = raw_score(
                            reference[year]["y"], prediction, masks[year]["R"]
                        ) - raw_score(reference[year]["y"], anchor, masks[year]["R"])
                        full_gain = raw_score(
                            reference[year]["y"], prediction, masks[year]["all"]
                        ) - raw_score(reference[year]["y"], anchor, masks[year]["all"])
                        per_anchor[anchor_name][str(year)] = {
                            "R_gain": r_gain,
                            "full_gain": full_gain,
                        }
                        all_r_gains.append(r_gain)
                rows.append(
                    {
                        "stem": stem,
                        "key": key,
                        "route": "R_only",
                        "gamma_selection": "maximize minimum 2022/2023 R gain across both honest anchors",
                        "selected_gamma": gamma,
                        "selected_development": selected,
                        "per_anchor": per_anchor,
                        "minimum_R_gain_all_anchor_year_cells": float(min(all_r_gains)),
                        "minimum_2024_R_gain_across_anchors": float(
                            min(per_anchor[name]["2024"]["R_gain"] for name in ANCHORS)
                        ),
                        "median_R_gain_all_anchor_year_cells": float(np.median(all_r_gains)),
                        "development_positive": bool(
                            float(selected["minimum_development_R_gain"]) > 0.0
                        ),
                        "forward_positive_both_anchors": bool(min(all_r_gains) > 0.0),
                    }
                )
        except Exception as exc:
            failures.append(
                {
                    "stem": stem,
                    "exception": type(exc).__name__,
                    "message": str(exc),
                }
            )

    ranked = sorted(
        rows,
        key=lambda row: (
            bool(row["forward_positive_both_anchors"]),
            float(row["minimum_R_gain_all_anchor_year_cells"]),
            float(row["median_R_gain_all_anchor_year_cells"]),
        ),
        reverse=True,
    )
    report = {
        "protocol": {
            "official_train_only": True,
            "test_rows_read": False,
            "discovery_only": True,
            "development_years": list(DEVELOPMENT_YEARS),
            "unopened_for_gamma_selection": 2024,
            "anchors": ANCHORS,
            "gamma_grid": list(GAMMAS),
            "route": "R_only_F_anchor_unchanged",
            "warning": (
                "Artifact recipes and prediction-key semantics are not trusted by this numerical "
                "screen. Every survivor requires provenance review and a fresh preregistered rerun."
            ),
        },
        "common_stage_count": len(common_stems),
        "candidate_direction_count": len(rows),
        "development_positive_count": sum(bool(row["development_positive"]) for row in rows),
        "forward_positive_both_anchors_count": sum(
            bool(row["forward_positive_both_anchors"]) for row in rows
        ),
        "ranked": ranked,
        "load_failures": failures,
    }
    OUTPUT.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    summary = {
        "common_stage_count": report["common_stage_count"],
        "candidate_direction_count": report["candidate_direction_count"],
        "development_positive_count": report["development_positive_count"],
        "forward_positive_both_anchors_count": report[
            "forward_positive_both_anchors_count"
        ],
        "top": [
            {
                "stem": row["stem"],
                "key": row["key"],
                "gamma": row["selected_gamma"],
                "min_all": row["minimum_R_gain_all_anchor_year_cells"],
                "min_2024": row["minimum_2024_R_gain_across_anchors"],
            }
            for row in ranked[:25]
        ],
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False))
    print(f"Saved {OUTPUT}")


if __name__ == "__main__":
    main()
