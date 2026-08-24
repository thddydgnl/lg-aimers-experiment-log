#!/usr/bin/env python3
"""Evaluate a two-season sign-consensus hierarchy of exact-C residuals."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.analyze_v5_trackman_game_repeatability_source import (  # noqa: E402
    digest,
    load,
    safe,
    score,
)
from experiments.run_v5_h1_residual import cluster_bootstrap_score_gain  # noqa: E402


PRED = ROOT / "experiments/results/predictions"
PREREG = ROOT / "experiments/params/v5_two_season_residual_consensus_preregister.json"
REPORT = ROOT / "experiments/results/v5_two_season_residual_consensus_dev.json"
TRAIN = ROOT / "open/data/train.csv"
TARGET_YEARS = (2022, 2023)
EXACT = {
    2020: (PRED / "v4_m3_c_backtest_2020_2020.npz", "catboost_outcome"),
    2021: (PRED / "v4_m3_c_backtest_2021_2021.npz", "catboost_outcome"),
    2022: (PRED / "v3_sparse_c_backtest_2022.npz", "catboost_outcome"),
    2023: (PRED / "v3_sparse_c_backtest_2023.npz", "catboost_outcome"),
}
ANCHORS = {
    "exact_c": (None, "catboost_outcome"),
    "honest_identity": ("v5_honest_m3_r_identity_{year}.npz", "final_prediction"),
    "honest_grid": ("v5_honest_m3_r_grid_{year}.npz", "final_prediction"),
}
STRENGTHS = {"pitcher": 100.0, "hand": 38.0, "pressure_hand": 30.0}


def pressure_state(frame: pd.DataFrame) -> np.ndarray:
    balls = frame["balls_before"].to_numpy(dtype=np.int8, copy=False)
    strikes = frame["strikes_before"].to_numpy(dtype=np.int8, copy=False)
    output = np.zeros(len(frame), dtype=np.int8)
    output[(balls == 3) & (strikes < 2)] = 1
    output[(balls < 3) & (strikes == 2)] = 2
    output[(balls == 3) & (strikes == 2)] = 3
    return output


def aligned_fold(
    year: int, raw: pd.DataFrame
) -> dict[str, Any]:
    path, key = EXACT[year]
    artifact = load(path)
    rows = artifact["row_index"].astype(np.int64)
    frame = raw.iloc[rows].reset_index(drop=True).copy()
    frame["pressure_state"] = pressure_state(frame)
    if not np.array_equal(
        artifact["y"].astype(np.int8), frame["control_success"].to_numpy(dtype=np.int8)
    ):
        raise ValueError(f"{year} target alignment failed")
    return {
        "year": year,
        "path": path,
        "artifact": artifact,
        "frame": frame,
        "y": artifact["y"].astype(np.int8),
        "parent": artifact[key].astype(np.float64),
        "cluster": artifact["cluster"],
        "regular": frame["game_type"].eq("R").to_numpy(),
    }


def effect_tables(fold: dict[str, Any]) -> dict[str, pd.Series]:
    mask = fold["regular"]
    source = fold["frame"].loc[
        mask, ["pitcher_id", "batter_hand", "pressure_state"]
    ].copy()
    residual = fold["y"][mask].astype(np.float64) - fold["parent"][mask]
    residual = residual - float(residual.mean())
    source["residual"] = residual

    pitcher_stats = source.groupby("pitcher_id", observed=True)["residual"].agg(
        ["count", "sum"]
    )
    pitcher = pitcher_stats["sum"] / (
        pitcher_stats["count"] + STRENGTHS["pitcher"]
    )
    pitcher.name = "effect"

    hand_keys = ["pitcher_id", "batter_hand"]
    hand_stats = source.groupby(hand_keys, observed=True)["residual"].agg(
        ["count", "sum"]
    )
    hand_parent = pitcher.reindex(
        hand_stats.index.get_level_values("pitcher_id")
    ).to_numpy(dtype=np.float64)
    hand = pd.Series(
        (
            hand_stats["sum"].to_numpy(dtype=np.float64)
            + STRENGTHS["hand"] * hand_parent
        )
        / (hand_stats["count"].to_numpy(dtype=np.float64) + STRENGTHS["hand"]),
        index=hand_stats.index,
        name="effect",
    )

    deep_keys = ["pitcher_id", "pressure_state", "batter_hand"]
    deep_stats = source.groupby(deep_keys, observed=True)["residual"].agg(
        ["count", "sum"]
    )
    deep_parent_index = pd.MultiIndex.from_arrays(
        [
            deep_stats.index.get_level_values("pitcher_id"),
            deep_stats.index.get_level_values("batter_hand"),
        ],
        names=hand_keys,
    )
    deep_parent = hand.reindex(deep_parent_index).to_numpy(dtype=np.float64)
    deep = pd.Series(
        (
            deep_stats["sum"].to_numpy(dtype=np.float64)
            + STRENGTHS["pressure_hand"] * deep_parent
        )
        / (
            deep_stats["count"].to_numpy(dtype=np.float64)
            + STRENGTHS["pressure_hand"]
        ),
        index=deep_stats.index,
        name="effect",
    )
    return {"pitcher": pitcher, "hand": hand, "deep": deep}


def lookup_effect(
    tables: dict[str, pd.Series], target: pd.DataFrame
) -> tuple[np.ndarray, np.ndarray, dict[str, int]]:
    pitcher_ids = target["pitcher_id"].to_numpy(dtype=np.int64, copy=False)
    pitcher_raw = tables["pitcher"].reindex(pitcher_ids).to_numpy(dtype=np.float64)
    known = np.isfinite(pitcher_raw)
    pitcher = np.nan_to_num(pitcher_raw, nan=0.0)

    hand_index = pd.MultiIndex.from_arrays(
        [pitcher_ids, target["batter_hand"].to_numpy(dtype=np.int8, copy=False)],
        names=["pitcher_id", "batter_hand"],
    )
    hand_raw = tables["hand"].reindex(hand_index).to_numpy(dtype=np.float64)
    hand_known = np.isfinite(hand_raw)
    hand = np.where(hand_known, hand_raw, pitcher)

    deep_index = pd.MultiIndex.from_arrays(
        [
            pitcher_ids,
            target["pressure_state"].to_numpy(dtype=np.int8, copy=False),
            target["batter_hand"].to_numpy(dtype=np.int8, copy=False),
        ],
        names=["pitcher_id", "pressure_state", "batter_hand"],
    )
    deep_raw = tables["deep"].reindex(deep_index).to_numpy(dtype=np.float64)
    deep_known = np.isfinite(deep_raw)
    effect = np.where(deep_known, deep_raw, hand)
    return effect, known, {
        "pitcher_known": int(known.sum()),
        "hand_known": int(hand_known.sum()),
        "deep_known": int(deep_known.sum()),
    }


def build_component(
    target: dict[str, Any], sources: tuple[dict[str, Any], dict[str, Any]]
) -> tuple[np.ndarray, dict[str, Any]]:
    effects = []
    knowns = []
    coverage = []
    for source in sources:
        effect, known, detail = lookup_effect(effect_tables(source), target["frame"])
        effects.append(effect)
        knowns.append(known)
        coverage.append({"source_year": source["year"], **detail})
    first, second = effects
    agree = knowns[0] & knowns[1] & ((first * second) > 0.0) & target["regular"]
    consensus = np.zeros(len(first), dtype=np.float64)
    consensus[agree] = np.sign(first[agree]) * np.minimum(
        np.abs(first[agree]), np.abs(second[agree])
    )
    component = target["parent"].copy()
    component[target["regular"]] = np.clip(
        component[target["regular"]] + consensus[target["regular"]],
        1e-6,
        1.0 - 1e-6,
    )
    return component, {
        "source_coverage": coverage,
        "consensus_rows": int(agree.sum()),
        "consensus_fraction_R": float(agree.sum() / max(1, target["regular"].sum())),
        "consensus_abs_q50": float(np.median(np.abs(consensus[agree]))) if agree.any() else 0.0,
        "consensus_abs_q90": float(np.quantile(np.abs(consensus[agree]), 0.9)) if agree.any() else 0.0,
        "consensus_mean": float(consensus[agree].mean()) if agree.any() else 0.0,
    }


def anchor_predictions(year: int, target: dict[str, Any]) -> dict[str, np.ndarray]:
    output = {"exact_c": target["parent"]}
    for name, (pattern, key) in ANCHORS.items():
        if name == "exact_c":
            continue
        path = PRED / pattern.format(year=year)
        artifact = load(path)
        for align_key in ("y", "row_index", "cluster"):
            if not np.array_equal(target["artifact"][align_key], artifact[align_key]):
                raise ValueError(f"{year}/{name} alignment mismatch: {align_key}")
        output[name] = artifact[key].astype(np.float64)
    return output


def blend(parent: np.ndarray, component: np.ndarray, regular: np.ndarray, gamma: float) -> np.ndarray:
    prediction = parent.copy()
    prediction[regular] = (
        (1.0 - gamma) * parent[regular] + gamma * component[regular]
    )
    return np.clip(prediction, 1e-6, 1.0 - 1e-6)


def main() -> None:
    if REPORT.exists():
        raise FileExistsError(f"immutable report exists: {REPORT}")
    prereg = json.loads(PREREG.read_text(encoding="utf-8"))
    if prereg["status"] != "locked_before_candidate_scores":
        raise ValueError("unexpected preregistration status")
    raw = pd.read_csv(
        TRAIN,
        usecols=[
            "game_type", "balls_before", "strikes_before", "pitcher_id",
            "batter_hand", "control_success",
        ],
        low_memory=False,
    )
    folds = {year: aligned_fold(year, raw) for year in (2020, 2021, 2022, 2023)}
    components = {
        2022: build_component(folds[2022], (folds[2020], folds[2021])),
        2023: build_component(folds[2023], (folds[2021], folds[2022])),
    }
    anchors = {year: anchor_predictions(year, folds[year]) for year in TARGET_YEARS}

    gamma_trials = []
    source = folds[2022]
    full = np.ones(len(source["y"]), dtype=bool)
    for gamma_value in prereg["effect_recipe"]["gamma_grid"]:
        gamma = float(gamma_value)
        candidate = blend(
            source["parent"], components[2022][0], source["regular"], gamma
        )
        full_gains = {
            name: score(source["y"], candidate, full)["score"]
            - score(source["y"], anchor, full)["score"]
            for name, anchor in anchors[2022].items()
        }
        same_r_gain = score(source["y"], candidate, source["regular"])["score"] - score(
            source["y"], source["parent"], source["regular"]
        )["score"]
        gamma_trials.append(
            {
                "gamma": gamma,
                "full_gains": full_gains,
                "minimum_full_gain": float(min(full_gains.values())),
                "exact_C_R_gain": same_r_gain,
            }
        )
    selected = max(
        gamma_trials,
        key=lambda row: (row["minimum_full_gain"], row["exact_C_R_gain"], -row["gamma"]),
    )
    gamma = float(selected["gamma"])

    results: dict[str, Any] = {}
    all_full_gains: list[float] = []
    same_parent_checks: list[bool] = []
    for year in TARGET_YEARS:
        target = folds[year]
        candidate = blend(
            target["parent"], components[year][0], target["regular"], gamma
        )
        comparisons: dict[str, Any] = {}
        for anchor_name, anchor in anchors[year].items():
            routes: dict[str, Any] = {}
            for route, mask in (
                ("full", np.ones(len(target["y"]), dtype=bool)),
                ("R", target["regular"]),
            ):
                anchor_metrics = score(target["y"], anchor, mask)
                candidate_metrics = score(target["y"], candidate, mask)
                interval = cluster_bootstrap_score_gain(
                    target["y"], anchor, candidate, target["cluster"], mask,
                    iterations=2000,
                    seed=12000 + year + (0 if route == "full" else 100)
                    + list(ANCHORS).index(anchor_name) * 1000,
                )
                gain = candidate_metrics["score"] - anchor_metrics["score"]
                routes[route] = {
                    "anchor": anchor_metrics,
                    "candidate": candidate_metrics,
                    "gain": gain,
                    "pitcher_cluster_95_ci": interval,
                }
                if route == "full":
                    all_full_gains.append(gain)
                if anchor_name == "exact_c" and route == "R":
                    same_parent_checks.extend([gain > 0.0, interval["ci_low"] > 0.0])
            comparisons[anchor_name] = routes
        output_path = PRED / f"v5_two_season_residual_consensus_{year}.npz"
        np.savez_compressed(
            output_path,
            y=target["y"],
            row_index=target["artifact"]["row_index"],
            cluster=target["cluster"],
            parent_exact_c=target["parent"].astype(np.float32),
            consensus_component=components[year][0].astype(np.float32),
            final_prediction=candidate.astype(np.float32),
        )
        results[str(year)] = {
            "component_diagnostics": components[year][1],
            "comparisons": comparisons,
            "artifact": str(output_path.relative_to(ROOT)),
            "artifact_sha256": digest(output_path),
        }

    required = float(
        prereg["validation"]["advance_gate"]
        ["minimum_full_gain_across_both_years_and_all_three_anchors"]
    )
    g_dev = float(min(all_full_gains))
    passed = bool(all(same_parent_checks) and g_dev >= required)
    report = {
        "experiment_id": prereg["experiment_id"],
        "status": "development_passed" if passed else "development_failed",
        "preregister_sha256": digest(PREREG),
        "analysis_code_sha256": digest(Path(__file__)),
        "input_hashes": {
            str(year): digest(EXACT[year][0]) for year in (2020, 2021, 2022, 2023)
        },
        "gamma_trials_2022": gamma_trials,
        "selected_gamma": gamma,
        "results": results,
        "gate": {
            "same_parent_R_point_and_CI_all_pass": bool(all(same_parent_checks)),
            "G_dev": g_dev,
            "required_G_dev": required,
            "passed": passed,
            "decision": (
                "write immutable 2024 lock"
                if passed
                else "close without reading 2024 for this recipe"
            ),
        },
        "confirmation_2024_read": False,
        "leaderboard_values_used": False,
        "goal_status": "active",
        "goal_completion_claimed": False,
    }
    REPORT.write_text(
        json.dumps(safe(report), ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(safe({
        "status": report["status"],
        "selected_gamma": gamma,
        "component_diagnostics": {
            str(year): results[str(year)]["component_diagnostics"] for year in TARGET_YEARS
        },
        "same_parent": {
            str(year): results[str(year)]["comparisons"]["exact_c"] for year in TARGET_YEARS
        },
        "gate": report["gate"],
    }), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
