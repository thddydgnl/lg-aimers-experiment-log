#!/usr/bin/env python3
"""Screen existing OOF predictions for genuinely forward-stable directions.

The coefficient is fitted on 2022 only and transferred unchanged to 2023 and
2024.  This is a discovery audit, not a V5 completion result: some source
artifacts may themselves have been chosen after looking at 2024 and must be
retrained from a locked recipe before they can become candidates.
"""

from __future__ import annotations

import json
from pathlib import Path
import re

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
PREDICTIONS = ROOT / "experiments/results/predictions"
OUTPUT = ROOT / "experiments/results/v5_transfer_catalog.json"
TRAIN = ROOT / "open/data/train.csv"
YEARS = (2022, 2023, 2024)
REFERENCE_STEM = "v3_sparse_m3_frozen"
REFERENCE_KEY = "final_prediction"
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
    with np.load(path) as archive:
        return {key: np.asarray(archive[key]) for key in archive.files}


def raw_score(y: np.ndarray, prediction: np.ndarray, mask: np.ndarray) -> float:
    target = np.asarray(y[mask], dtype=np.float64)
    pred = np.clip(np.asarray(prediction[mask], dtype=np.float64), 1e-6, 1 - 1e-6)
    rate = float(target.mean())
    reference = rate * (1.0 - rate)
    brier = float(np.mean(np.square(pred - target)))
    return 100000.0 * (1.0 - brier / reference)


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


def fit_gamma(
    y: np.ndarray,
    anchor: np.ndarray,
    candidate: np.ndarray,
    mask: np.ndarray,
) -> tuple[float, float]:
    direction = candidate[mask] - anchor[mask]
    residual = y[mask] - anchor[mask]
    denominator = float(direction @ direction)
    raw = float(direction @ residual / denominator) if denominator else 0.0
    return raw, float(np.clip(raw, 0.0, 1.0))


def apply_route(
    anchor: np.ndarray,
    candidate: np.ndarray,
    gamma: float,
    route: np.ndarray,
) -> np.ndarray:
    output = np.asarray(anchor, dtype=np.float64).copy()
    output[route] += gamma * (candidate[route] - anchor[route])
    return np.clip(output, 1e-6, 1 - 1e-6)


def main() -> None:
    references = {
        year: load(PREDICTIONS / f"{REFERENCE_STEM}_{year}.npz")
        for year in YEARS
    }
    game_type = pd.read_csv(TRAIN, usecols=["game_type"])["game_type"].to_numpy()
    masks = {
        year: {
            "all": np.ones(len(references[year]["y"]), dtype=bool),
            "R": game_type[references[year]["row_index"]] == "R",
            "F": game_type[references[year]["row_index"]] == "F",
        }
        for year in YEARS
    }

    files_by_stem: dict[str, dict[int, Path]] = {}
    for path in PREDICTIONS.glob("*.npz"):
        match = YEAR_PATTERN.match(path.name)
        if not match:
            continue
        stem = match.group("stem")
        year = int(match.group("year"))
        files_by_stem.setdefault(stem, {})[year] = path

    common_stems = sorted(
        stem for stem, paths in files_by_stem.items() if all(year in paths for year in YEARS)
    )
    rows: list[dict[str, object]] = []
    failures: list[dict[str, str]] = []
    for stem in common_stems:
        if stem == REFERENCE_STEM:
            continue
        try:
            artifacts = {year: load(files_by_stem[stem][year]) for year in YEARS}
            for year in YEARS:
                if not np.array_equal(
                    artifacts[year].get("row_index"), references[year]["row_index"]
                ):
                    raise ValueError(f"row_index mismatch in {year}")
                if not np.array_equal(artifacts[year].get("y"), references[year]["y"]):
                    raise ValueError(f"target mismatch in {year}")

            common_keys = set.intersection(*(set(artifacts[year]) for year in YEARS))
            for key in sorted(common_keys):
                if key.lower() in NON_PREDICTION_KEYS:
                    continue
                if any(
                    np.array_equal(artifacts[year][key], references[year]["y"])
                    for year in YEARS
                ):
                    continue
                if not all(
                    probability_like(artifacts[year][key], len(references[year]["y"]))
                    for year in YEARS
                ):
                    continue
                for route_name in ("all", "R"):
                    raw_gamma, gamma = fit_gamma(
                        references[2022]["y"],
                        references[2022][REFERENCE_KEY],
                        artifacts[2022][key],
                        masks[2022][route_name],
                    )
                    scores: dict[str, dict[str, float]] = {}
                    for year in YEARS:
                        prediction = apply_route(
                            references[year][REFERENCE_KEY],
                            artifacts[year][key],
                            gamma,
                            masks[year][route_name],
                        )
                        full_mask = masks[year]["all"]
                        r_mask = masks[year]["R"]
                        anchor = references[year][REFERENCE_KEY]
                        y = references[year]["y"]
                        scores[str(year)] = {
                            "full_score": raw_score(y, prediction, full_mask),
                            "full_gain_vs_v3": (
                                raw_score(y, prediction, full_mask)
                                - raw_score(y, anchor, full_mask)
                            ),
                            "r_score": raw_score(y, prediction, r_mask),
                            "r_gain_vs_v3": (
                                raw_score(y, prediction, r_mask)
                                - raw_score(y, anchor, r_mask)
                            ),
                        }
                    gains = [scores[str(year)]["r_gain_vs_v3"] for year in YEARS]
                    rows.append(
                        {
                            "stem": stem,
                            "key": key,
                            "route": route_name,
                            "gamma_fit_2022_raw": raw_gamma,
                            "gamma_fit_2022": gamma,
                            "scores": scores,
                            "r_gain_median_2022_2024": float(np.median(gains)),
                            "r_gain_min_2022_2024": float(np.min(gains)),
                            "same_positive_direction_all_years": bool(
                                gamma > 0 and all(gain > 0 for gain in gains)
                            ),
                            "forward_gate": bool(
                                gamma > 0
                                and scores["2022"]["r_gain_vs_v3"] > 0
                                and scores["2023"]["r_gain_vs_v3"] > 0
                                and scores["2024"]["r_gain_vs_v3"] > 0
                            ),
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
            bool(row["forward_gate"]),
            float(row["r_gain_min_2022_2024"]),
            float(row["r_gain_median_2022_2024"]),
        ),
        reverse=True,
    )
    report = {
        "protocol": {
            "coefficient_fit_year": 2022,
            "coefficient_bounds": [0.0, 1.0],
            "transfer_years": [2023, 2024],
            "f_regime_note": "Ranking uses R gains; 2023 F is a documented measurement break.",
            "official_train_only": True,
            "test_rows_read": False,
            "discovery_only": True,
            "warning": (
                "Source recipes may have been selected with 2024. Any survivor must be "
                "retrained from a newly locked recipe before V5 confirmation."
            ),
        },
        "common_stage_count": len(common_stems),
        "candidate_direction_count": len(rows),
        "forward_gate_count": sum(bool(row["forward_gate"]) for row in rows),
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
        "forward_gate_count": report["forward_gate_count"],
        "top": [
            {
                "stem": row["stem"],
                "key": row["key"],
                "route": row["route"],
                "gamma": row["gamma_fit_2022"],
                "gains_r": {
                    year: row["scores"][year]["r_gain_vs_v3"]
                    for year in ("2022", "2023", "2024")
                },
            }
            for row in ranked[:20]
        ],
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False))
    print(f"Saved {OUTPUT}")


if __name__ == "__main__":
    main()
