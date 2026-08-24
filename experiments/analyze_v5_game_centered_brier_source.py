#!/usr/bin/env python3
"""Immutable 2020/2021 source gate for game-centered Brier learning."""

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
PREREG = ROOT / "experiments/params/v5_game_centered_brier_preregister.json"
MODEL_PARAMS = ROOT / "experiments/params/v5_game_centered_brier_model.json"
ENGINE = ROOT / "experiments/run_v2_rolling.py"
REPORT = ROOT / "experiments/results/v5_game_centered_brier_source_gate.json"


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def route_metrics(
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
        base = score(y, parent, mask)
        cand = score(y, candidate, mask)
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
            "parent": base,
            "candidate": cand,
            "gain": float(cand["score"] - base["score"]),
            "pitcher_cluster_95_ci": interval,
        }
    return output


def main() -> None:
    if REPORT.exists():
        raise FileExistsError(f"immutable report exists: {REPORT}")
    prereg = json.loads(PREREG.read_text(encoding="utf-8"))
    if prereg["status"] != "locked_before_source_metrics":
        raise ValueError("unexpected preregistration status")

    folds: dict[int, dict[str, np.ndarray]] = {}
    maximum_row = 0
    input_hashes: dict[str, Any] = {}
    for year in YEARS:
        parent_path = PRED / f"v4_m3_c_backtest_{year}_{year}.npz"
        centered_path = PRED / f"v5_game_centered_brier_source_{year}.npz"
        parent_artifact = load(parent_path)
        centered_artifact = load(centered_path)
        for key in ("y", "row_index", "cluster"):
            if not np.array_equal(parent_artifact[key], centered_artifact[key]):
                raise ValueError(f"{year}: {key} alignment mismatch")
        row_index = parent_artifact["row_index"].astype(np.int64)
        maximum_row = max(maximum_row, int(row_index.max()))
        folds[year] = {
            "y": parent_artifact["y"].astype(np.int8),
            "row_index": row_index,
            "cluster": parent_artifact["cluster"],
            "parent": parent_artifact["catboost_outcome"].astype(np.float64),
            "deviation": (
                centered_artifact["catboost_game_centered_brier"].astype(np.float64)
                - 0.5
            ),
        }
        input_hashes[str(year)] = {
            "parent": digest(parent_path),
            "centered": digest(centered_path),
        }

    frame = pd.read_csv(
        TRAIN,
        usecols=["season", "game_type", "control_success"],
        nrows=maximum_row + 1,
    )
    if int(frame["season"].max()) != max(YEARS):
        raise ValueError("source reader crossed the locked 2021 boundary")
    for year in YEARS:
        fold = folds[year]
        rows = frame.loc[fold["row_index"]]
        if not rows["season"].eq(year).all():
            raise ValueError(f"{year}: season mismatch")
        if not np.array_equal(
            rows["control_success"].to_numpy(dtype=np.int8), fold["y"]
        ):
            raise ValueError(f"{year}: target mismatch")
        fold["regular"] = rows["game_type"].astype(str).eq("R").to_numpy()

    trials: list[dict[str, Any]] = []
    for gamma_value in prereg["source_protocol"]["blend_grid"]:
        gamma = float(gamma_value)
        metrics: dict[str, Any] = {}
        for year in YEARS:
            fold = folds[year]
            candidate = fold["parent"].copy()
            regular = fold["regular"].astype(bool)
            candidate[regular] = np.clip(
                fold["parent"][regular] + gamma * fold["deviation"][regular],
                1e-6,
                1.0 - 1e-6,
            )
            full = np.ones(len(fold["y"]), dtype=bool)
            metrics[str(year)] = {
                "full": {
                    "gain": float(
                        score(fold["y"], candidate, full)["score"]
                        - score(fold["y"], fold["parent"], full)["score"]
                    )
                },
                "R": {
                    "gain": float(
                        score(fold["y"], candidate, regular)["score"]
                        - score(fold["y"], fold["parent"], regular)["score"]
                    )
                },
            }
        trials.append(
            {
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
        ),
    )

    selected_metrics: dict[str, Any] = {}
    artifacts: dict[str, Any] = {}
    for offset, year in enumerate(YEARS):
        fold = folds[year]
        regular = fold["regular"].astype(bool)
        candidate = fold["parent"].copy()
        candidate[regular] = np.clip(
            fold["parent"][regular]
            + float(selected["gamma"]) * fold["deviation"][regular],
            1e-6,
            1.0 - 1e-6,
        )
        selected_metrics[str(year)] = route_metrics(
            fold["y"],
            fold["parent"],
            candidate,
            fold["cluster"],
            regular,
            seed=8226000 + 10000 * offset,
        )
        output = PRED / f"v5_game_centered_brier_selected_source_{year}.npz"
        if output.exists():
            raise FileExistsError(f"immutable artifact exists: {output}")
        np.savez_compressed(
            output,
            y=fold["y"],
            row_index=fold["row_index"],
            cluster=fold["cluster"],
            parent=fold["parent"].astype(np.float32),
            centered_deviation=fold["deviation"].astype(np.float32),
            final_prediction=candidate.astype(np.float32),
        )
        artifacts[str(year)] = {
            "path": str(output.relative_to(ROOT)),
            "sha256": digest(output),
        }

    gate = prereg["source_protocol"]["advance_gate"]
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
        "model_params_sha256": digest(MODEL_PARAMS),
        "engine_sha256": digest(ENGINE),
        "script_sha256": digest(Path(__file__)),
        "years_read": list(YEARS),
        "years_not_read": [2022, 2023, 2024],
        "selection": selected,
        "selected_metrics": selected_metrics,
        "trials": trials,
        "input_sha256": input_hashes,
        "gate": {"requirements": gate, "checks": checks, "pass": passed},
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
