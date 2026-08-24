#!/usr/bin/env python3
"""Select and confirm the preregistered conditional-history direction."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.run_v5_h1_residual import (  # noqa: E402
    cluster_bootstrap_score_gain,
    metrics,
)


PRED = ROOT / "experiments/results/predictions"
TRAIN = ROOT / "open/data/train.csv"
PREREG = ROOT / "experiments/params/v5_h2_conditional_history_preregister.json"
SELECTION = ROOT / "experiments/results/v5_h2_conditional_history_selection.json"
CONFIRMATION = ROOT / "experiments/results/v5_h2_conditional_history_confirmation.json"
ANCHOR_STEM = "v3_sparse_m3_frozen"
KEY = "catboost_outcome"
V3_ACTUAL_LB = 1090.9100565103


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("select", "confirm"), required=True)
    parser.add_argument("--bootstrap", type=int, default=1000)
    return parser.parse_args()


def load(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        return {key: np.asarray(archive[key]) for key in archive.files}


def load_fold(year: int, candidate_stem: str) -> dict[str, np.ndarray]:
    anchor = load(PRED / f"{ANCHOR_STEM}_{year}.npz")
    candidate = load(PRED / f"{candidate_stem}_{year}.npz")
    for key in ("y", "row_index", "cluster"):
        if not np.array_equal(anchor[key], candidate[key]):
            raise ValueError(f"alignment mismatch {candidate_stem}/{year}/{key}")
    return {
        "y": anchor["y"].astype(np.int8),
        "row_index": anchor["row_index"].astype(np.int64),
        "cluster": anchor["cluster"].astype(str),
        "anchor": anchor["final_prediction"].astype(np.float64),
        "candidate": candidate[KEY].astype(np.float64),
    }


def routed_prediction(
    fold: dict[str, np.ndarray], game_type: np.ndarray, gamma: float
) -> np.ndarray:
    prediction = fold["anchor"].copy()
    route = game_type == "R"
    prediction[route] += gamma * (
        fold["candidate"][route] - fold["anchor"][route]
    )
    return np.clip(prediction, 1e-6, 1.0 - 1e-6)


def evaluate(
    fold: dict[str, np.ndarray],
    game_type: np.ndarray,
    gamma: float,
    bootstrap: int,
    seed: int,
) -> dict:
    prediction = routed_prediction(fold, game_type, gamma)
    score_metrics = metrics(fold["y"], fold["anchor"], prediction, game_type)
    return {
        "metrics": score_metrics,
        "bootstrap_R": cluster_bootstrap_score_gain(
            fold["y"], fold["anchor"], prediction, fold["cluster"],
            game_type == "R", bootstrap, seed,
        ),
        "bootstrap_all": cluster_bootstrap_score_gain(
            fold["y"], fold["anchor"], prediction, fold["cluster"],
            np.ones(len(game_type), dtype=bool), bootstrap, seed + 10000,
        ),
    }


def game_types(fold: dict[str, np.ndarray], all_types: np.ndarray) -> np.ndarray:
    return all_types[fold["row_index"]]


def run_selection(args: argparse.Namespace, prereg: dict, all_types: np.ndarray) -> None:
    if SELECTION.exists():
        raise FileExistsError(f"immutable selection already exists: {SELECTION}")
    trials = []
    for variant_name, variant in sorted(prereg["variants"].items()):
        folds = {
            year: load_fold(year, variant["development_stem"])
            for year in (2022, 2023)
        }
        for gamma in prereg["selection"]["gamma_grid"]:
            per_fold = {
                str(year): evaluate(
                    folds[year], game_types(folds[year], all_types), float(gamma),
                    args.bootstrap,
                    82000 + year + int(round(float(gamma) * 100))
                    + 1000 * tuple(sorted(prereg["variants"])).index(variant_name),
                )
                for year in (2022, 2023)
            }
            r_gains = [
                per_fold[str(year)]["metrics"]["R"]["gain"]
                for year in (2022, 2023)
            ]
            eligible = bool(
                all(gain > 0.0 for gain in r_gains)
                and all(
                    per_fold[str(year)]["bootstrap_R"]["ci_low"] > 0.0
                    for year in (2022, 2023)
                )
            )
            trials.append({
                "variant": variant_name,
                "development_stem": variant["development_stem"],
                "gamma": float(gamma),
                "min_R_gain": float(min(r_gains)),
                "median_R_gain": float(np.median(r_gains)),
                "folds": per_fold,
                "eligible": eligible,
            })
    eligible = [trial for trial in trials if trial["eligible"]]
    selected = None
    if eligible:
        selected = sorted(
            eligible,
            key=lambda trial: (
                -trial["min_R_gain"], trial["gamma"], trial["variant"]
            ),
        )[0]
    report = {
        "experiment_id": prereg["experiment_id"],
        "mode": "development_selection",
        "preregister_sha256": hashlib.sha256(PREREG.read_bytes()).hexdigest(),
        "years_read": [2022, 2023],
        "confirmation_year_read": False,
        "trials": trials,
        "selected": selected,
        "status": "locked" if selected is not None else "failed_no_eligible_candidate",
    }
    SELECTION.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    if selected is not None:
        variant = prereg["variants"][selected["variant"]]
        gamma = float(selected["gamma"])
        for year in (2022, 2023):
            fold = load_fold(year, variant["development_stem"])
            prediction = routed_prediction(
                fold, game_types(fold, all_types), gamma
            )
            np.savez_compressed(
                PRED / f"v5_h2_conditional_history_locked_{year}.npz",
                y=fold["y"], row_index=fold["row_index"],
                cluster=fold["cluster"], anchor=fold["anchor"],
                conditional_history=fold["candidate"],
                final_prediction=prediction,
            )
    print(json.dumps({"status": report["status"], "selected": selected}, indent=2))
    print(f"Saved {SELECTION}")


def run_confirmation(
    args: argparse.Namespace, prereg: dict, all_types: np.ndarray
) -> None:
    if CONFIRMATION.exists():
        raise FileExistsError(f"immutable confirmation already exists: {CONFIRMATION}")
    selection = json.loads(SELECTION.read_text(encoding="utf-8"))
    prereg_hash = hashlib.sha256(PREREG.read_bytes()).hexdigest()
    if selection.get("status") != "locked" or selection.get("selected") is None:
        raise ValueError("no eligible locked selection")
    if selection.get("preregister_sha256") != prereg_hash:
        raise ValueError("preregister changed after selection")
    selected = selection["selected"]
    variant = prereg["variants"][selected["variant"]]
    gamma = float(selected["gamma"])
    fold = load_fold(2024, variant["confirmation_stem"])
    result = evaluate(
        fold, game_types(fold, all_types), gamma, args.bootstrap, 962024
    )
    prediction = routed_prediction(fold, game_types(fold, all_types), gamma)
    np.savez_compressed(
        PRED / "v5_h2_conditional_history_locked_2024.npz",
        y=fold["y"], row_index=fold["row_index"], cluster=fold["cluster"],
        anchor=fold["anchor"], conditional_history=fold["candidate"],
        final_prediction=prediction,
    )
    development_full_gains = [
        selected["folds"][str(year)]["metrics"]["all"]["gain"]
        for year in (2022, 2023)
    ]
    g_dev = float(np.median(development_full_gains))
    g_confirm = float(result["metrics"]["all"]["gain"])
    g_ci = float(result["bootstrap_all"]["ci_low"])
    g_robust = min(g_dev, g_confirm, g_ci)
    expected_lower = V3_ACTUAL_LB + 0.75 * max(0.0, g_robust)
    report = {
        "experiment_id": prereg["experiment_id"],
        "mode": "locked_confirmation",
        "preregister_sha256": prereg_hash,
        "selected_variant": selected["variant"],
        "confirmation_stem": variant["confirmation_stem"],
        "gamma": gamma,
        "route": "R_only_F_unchanged",
        "development": selected,
        "confirmation_2024": result,
        "conservative_expected_score": {
            "actual_v3_anchor": V3_ACTUAL_LB,
            "G_dev_full": g_dev,
            "G_confirm_full": g_confirm,
            "G_ci_full": g_ci,
            "G_robust": g_robust,
            "haircut": 0.75,
            "expected_lb_lower": expected_lower,
            "passes_1190": bool(expected_lower > 1190.0),
        },
    }
    CONFIRMATION.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    print(json.dumps(report["conservative_expected_score"], indent=2))
    print(f"Saved {CONFIRMATION}")


def main() -> None:
    args = parse_args()
    prereg = json.loads(PREREG.read_text(encoding="utf-8"))
    all_types = pd.read_csv(TRAIN, usecols=["game_type"])["game_type"].to_numpy()
    if args.mode == "select":
        run_selection(args, prereg, all_types)
    else:
        run_confirmation(args, prereg, all_types)


if __name__ == "__main__":
    main()
