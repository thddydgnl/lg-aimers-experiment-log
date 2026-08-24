#!/usr/bin/env python3
"""Immutable source gate for full-context privileged marginalization."""

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

from experiments.analyze_v5_dense_pitchtype_moe import (  # noqa: E402
    digest,
    evaluate,
    load,
    safe,
    score,
)

RESULTS = ROOT / "experiments/results"
PRED = RESULTS / "predictions"
PREREG = (
    ROOT
    / "experiments/params/v5_contextual_privileged_marginalization_preregister.json"
)
SIGNAL_REPORT = (
    RESULTS / "v5_contextual_privileged_marginalization_signal_source.json"
)
REPORT = RESULTS / "v5_contextual_privileged_marginalization_source.json"
TRAIN = ROOT / "open/data/train.csv"
YEARS = (2020, 2021)


def main() -> None:
    if REPORT.exists():
        raise FileExistsError(f"immutable result already exists: {REPORT}")
    prereg = json.loads(PREREG.read_text(encoding="utf-8"))
    signal_report = json.loads(SIGNAL_REPORT.read_text(encoding="utf-8"))
    if signal_report["status"] != prereg["semantic_gate"][
        "signal_source_status"
    ]:
        raise ValueError("contextual marginalization signal source did not pass")
    if signal_report["downstream_control_metrics_read"]:
        raise ValueError("signal source claims downstream metric access")
    game_types = pd.read_csv(TRAIN, usecols=["game_type"])["game_type"].astype(str)
    folds: dict[int, dict[str, Any]] = {}
    for year in YEARS:
        signal_path = (
            RESULTS / f"v5_contextual_privileged_marginalization_signal_{year}.npz"
        )
        parent_path = PRED / f"v4_m3_c_backtest_{year}_{year}.npz"
        signals = load(signal_path)
        parent_artifact = load(parent_path)
        if not np.array_equal(signals["row_index"], parent_artifact["row_index"]):
            raise ValueError(f"signal/parent alignment mismatch: {year}")
        regular = (
            game_types.iloc[parent_artifact["row_index"].astype(np.int64)].to_numpy(
                dtype=str
            )
            == "R"
        )
        parent = parent_artifact["catboost_outcome"].astype(np.float64)
        folds[year] = {
            "artifact": parent_artifact,
            "parent": parent,
            "signals": {
                direction: signals[direction].astype(np.float64)
                for direction in prereg["directions"]
            },
            "source_level": signals["source_level"].astype(np.int8),
            "regular": regular,
            "masks": {"full": np.ones(len(parent), dtype=bool), "R": regular},
            "paths": {"signal": signal_path, "parent": parent_path},
        }

    trials: list[dict[str, Any]] = []
    prediction_cache: dict[tuple[str, int, float], np.ndarray] = {}
    bootstrap = int(prereg["source_protocol"]["bootstrap_iterations"])
    for direction_index, direction_name in enumerate(prereg["directions"]):
        for gamma in prereg["gamma_grid"]:
            gamma = float(gamma)
            years: dict[str, Any] = {}
            for year in YEARS:
                fold = folds[year]
                direction = (
                    fold["parent"] + fold["signals"][direction_name]
                    if direction_name == "physics_delta"
                    else fold["signals"][direction_name]
                )
                result = evaluate(
                    fold["artifact"],
                    fold["parent"],
                    direction,
                    fold["regular"],
                    fold["masks"],
                    gamma,
                    bootstrap,
                    6910000 + 100000 * direction_index + 10000 * year
                    + int(gamma * 100),
                )
                prediction = fold["parent"].copy()
                prediction[fold["regular"]] += gamma * (
                    direction[fold["regular"]] - fold["parent"][fold["regular"]]
                )
                prediction = np.clip(prediction, 1e-6, 1.0 - 1e-6)
                prediction_cache[(direction_name, year, gamma)] = prediction
                subsets: dict[str, Any] = {}
                for subset_name, subset_mask in (
                    (
                        "native_R",
                        fold["regular"] & (fold["source_level"] == 0),
                    ),
                    (
                        "fallback_R",
                        fold["regular"] & (fold["source_level"] > 0),
                    ),
                ):
                    parent_metrics = score(
                        fold["artifact"]["y"], fold["parent"], subset_mask
                    )
                    candidate_metrics = score(
                        fold["artifact"]["y"], prediction, subset_mask
                    )
                    subsets[subset_name] = {
                        "parent": parent_metrics,
                        "candidate": candidate_metrics,
                        "gain": float(
                            candidate_metrics["score"] - parent_metrics["score"]
                        ),
                    }
                result["subsets"] = subsets
                years[str(year)] = result
            r_gains = [years[str(y)]["routes"]["R"]["gain"] for y in YEARS]
            full_gains = [
                years[str(y)]["routes"]["full"]["gain"] for y in YEARS
            ]
            trials.append(
                {
                    "direction": direction_name,
                    "direction_tiebreak_index": direction_index,
                    "gamma": gamma,
                    "minimum_R_gain": float(min(r_gains)),
                    "minimum_full_gain": float(min(full_gains)),
                    "mean_R_gain": float(np.mean(r_gains)),
                    "years": years,
                }
            )
    selected = max(
        trials,
        key=lambda item: (
            item["minimum_R_gain"],
            item["minimum_full_gain"],
            item["mean_R_gain"],
            -item["gamma"],
            -item["direction_tiebreak_index"],
        ),
    )
    gate = prereg["source_protocol"]["gate"]
    checks: list[bool] = []
    per_year_checks: dict[str, Any] = {}
    for year in YEARS:
        result = selected["years"][str(year)]
        routes = result["routes"]
        year_checks = {
            "R_point": routes["R"]["gain"]
            >= float(gate["minimum_R_gain_each_year"]),
            "full_point": routes["full"]["gain"]
            >= float(gate["minimum_full_gain_each_year"]),
            "R_ci": routes["R"]["pitcher_cluster_95_ci"]["ci_low"] > 0.0,
            "full_ci": routes["full"]["pitcher_cluster_95_ci"]["ci_low"] > 0.0,
            "native_R": result["subsets"]["native_R"]["gain"] >= 0.0,
            "fallback_R": result["subsets"]["fallback_R"]["gain"] >= 0.0,
        }
        checks.extend(year_checks.values())
        per_year_checks[str(year)] = year_checks
    passed = bool(all(checks))

    artifacts: dict[str, Any] = {}
    for year in YEARS:
        fold = folds[year]
        output = PRED / f"v5_contextual_privileged_marginalization_source_{year}.npz"
        if output.exists():
            raise FileExistsError(f"immutable artifact exists: {output}")
        final_prediction = prediction_cache[
            (selected["direction"], year, selected["gamma"])
        ]
        np.savez_compressed(
            output,
            y=fold["artifact"]["y"].astype(np.int8),
            row_index=fold["artifact"]["row_index"].astype(np.int64),
            cluster=fold["artifact"]["cluster"],
            parent_exact_c=fold["parent"],
            source_level=fold["source_level"],
            selected_direction=fold["signals"][selected["direction"]],
            final_prediction=final_prediction,
        )
        artifacts[str(year)] = {
            "path": str(output.relative_to(ROOT)), "sha256": digest(output)
        }

    report: dict[str, Any] = {
        "experiment_id": prereg["experiment_id"],
        "status": "source_pass" if passed else "source_failed",
        "signal_report_sha256": digest(SIGNAL_REPORT),
        "preregister_sha256": digest(PREREG),
        "script_sha256": digest(Path(__file__)),
        "years_read": list(YEARS),
        "years_not_read": [2022, 2023, 2024],
        "input_sha256": {
            str(year): {
                name: digest(path) for name, path in folds[year]["paths"].items()
            }
            for year in YEARS
        },
        "trials": trials,
        "selected": selected,
        "source_gate": {
            "requirements": gate,
            "per_year_checks": per_year_checks,
            "pass": passed,
            "decision": (
                "freeze direction/gamma and advance to 2022/2023"
                if passed
                else "close without generating 2022+ candidate"
            ),
        },
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
                    "selected": selected,
                    "checks": per_year_checks,
                }
            ),
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
