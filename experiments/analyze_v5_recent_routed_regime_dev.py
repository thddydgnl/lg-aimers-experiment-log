#!/usr/bin/env python3
"""Pre-2024 gate for the locked regime-aware recent routed recipe."""

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

RESULTS = ROOT / "experiments/results"
PRED = RESULTS / "predictions"
TRAIN = ROOT / "open/data/train.csv"
LOCK = ROOT / "experiments/params/v5_recent_routed_regime_lock.json"
CONTRACT = ROOT / "experiments/params/v5_validation_contract_v3_regime_aware.json"
REPORT = RESULTS / "v5_recent_routed_regime_dev.json"
YEARS = (2022, 2023)
ANCHORS = {
    "exact_c": ("v3_sparse_c_backtest_{year}.npz", "catboost_outcome"),
    "honest_identity": ("v5_honest_m3_r_identity_{year}.npz", "final_prediction"),
    "honest_grid": ("v5_honest_m3_r_grid_{year}.npz", "final_prediction"),
}


def evaluate(
    y: np.ndarray, anchor: np.ndarray, candidate: np.ndarray,
    cluster: np.ndarray, masks: dict[str, np.ndarray], seed: int,
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for route_index, (name, mask) in enumerate(masks.items()):
        base = score(y, anchor, mask)
        cand = score(y, candidate, mask)
        ci = cluster_bootstrap_score_gain(
            y, anchor, candidate, cluster, mask, iterations=2000,
            seed=seed + 1000 * route_index,
        )
        result[name] = {"anchor": base, "candidate": cand, "gain": float(cand["score"] - base["score"]), "pitcher_cluster_95_ci": ci}
    return result


def main() -> None:
    if REPORT.exists():
        raise FileExistsError(f"immutable report exists: {REPORT}")
    lock = json.loads(LOCK.read_text(encoding="utf-8"))
    contract = json.loads(CONTRACT.read_text(encoding="utf-8"))
    for item in lock["pre_2024_evidence"].values():
        path = ROOT / item["path"]
        if digest(path) != item["sha256"]:
            raise ValueError(f"pre-2024 evidence changed: {path}")
    game_types = pd.read_csv(TRAIN, usecols=["game_type"])["game_type"].astype(str)
    years: dict[str, Any] = {}
    artifacts: dict[str, Any] = {}
    r_checks: dict[str, Any] = {}
    for year in YEARS:
        dev_path = PRED / f"v5_recent_routed_dev_{year}.npz"
        dev = load(dev_path)
        anchors: dict[str, dict[str, np.ndarray]] = {}
        anchor_paths: dict[str, Path] = {}
        for name, (template, _) in ANCHORS.items():
            path = PRED / template.format(year=year)
            anchors[name] = load(path)
            anchor_paths[name] = path
            for key in ("y", "row_index", "cluster"):
                if not np.array_equal(dev[key], anchors[name][key]):
                    raise ValueError(f"alignment mismatch: {year}/{name}/{key}")
        parent = dev["parent_exact_c"].astype(np.float64)
        fine = parent + 0.5 * (dev["fine_raw"].astype(np.float64) - parent)
        auto = parent + 0.25 * (dev["auto_raw"].astype(np.float64) - parent)
        r_prediction = 0.6 * fine + 0.4 * auto
        f_prediction = (
            (dev["decoded_successes"].astype(np.float64) + 100.0 * parent)
            / (dev["decoded_n"].astype(np.float64) + 100.0)
        )
        f_prediction = np.where(dev["decoded_valid"].astype(bool), f_prediction, parent)
        regular = game_types.iloc[dev["row_index"].astype(np.int64)].to_numpy(dtype=str) == "R"
        masks = {"full": np.ones(len(regular), dtype=bool), "R": regular, "F": ~regular}
        if year <= 2022:
            f_prediction = parent
        candidate = np.clip(np.where(regular, r_prediction, f_prediction), 1e-6, 1.0 - 1e-6)
        comparisons: dict[str, Any] = {}
        for anchor_index, (name, (_, key)) in enumerate(ANCHORS.items()):
            comparisons[name] = evaluate(
                dev["y"], anchors[name][key].astype(np.float64), candidate,
                dev["cluster"], masks, 8710000 + 10000 * year + 100 * anchor_index,
            )
        same = comparisons["exact_c"]
        r_checks[str(year)] = {
            "point_positive": same["R"]["gain"] > 0.0,
            "ci_lower_positive": same["R"]["pitcher_cluster_95_ci"]["ci_low"] > 0.0,
        }
        years[str(year)] = {"comparisons": comparisons, "F_route": "parent" if year <= 2022 else "recent_game_update"}
        output = PRED / f"v5_recent_routed_regime_dev_{year}.npz"
        if output.exists():
            raise FileExistsError(f"immutable artifact exists: {output}")
        np.savez_compressed(
            output, y=dev["y"].astype(np.int8), row_index=dev["row_index"].astype(np.int64),
            cluster=dev["cluster"], parent_exact_c=parent, final_prediction=candidate,
        )
        artifacts[str(year)] = {"path": str(output.relative_to(ROOT)), "sha256": digest(output)}

    threshold = float(contract["non_relaxation_checks"]["required_raw_gain_unchanged"])
    legacy = years["2022"]["comparisons"]["exact_c"]["full"]
    legacy_pass = legacy["gain"] > 0.0 and legacy["pitcher_cluster_95_ci"]["ci_low"] > 0.0
    postbreak_comparisons = years["2023"]["comparisons"]
    f_post = postbreak_comparisons["exact_c"]["F"]
    f_post_pass = f_post["gain"] > 0.0 and f_post["pitcher_cluster_95_ci"]["ci_low"] > 0.0
    dev_points = [item["full"]["gain"] for item in postbreak_comparisons.values()]
    dev_ci_lows = [item["full"]["pitcher_cluster_95_ci"]["ci_low"] for item in postbreak_comparisons.values()]
    g_dev = float(min([*dev_points, *dev_ci_lows]))
    postbreak_pass = g_dev > threshold
    r_pass = all(all(item.values()) for item in r_checks.values())
    passed = bool(r_pass and legacy_pass and f_post_pass and postbreak_pass)
    report = {
        "experiment_id": lock["experiment_id"], "status": "development_pass" if passed else "development_failed",
        "lock_sha256": digest(LOCK), "contract_sha256": digest(CONTRACT),
        "years_read": list(YEARS), "confirmation_2024_read": False, "years": years,
        "gates": {
            "R_same_parent": {"years": r_checks, "pass": r_pass},
            "legacy_2022_full_guard": {"metrics": legacy, "pass": legacy_pass},
            "postbreak_2023_F": {"metrics": f_post, "pass": f_post_pass},
            "postbreak_G_dev": {
                "full_points": dev_points, "full_ci_lowers": dev_ci_lows,
                "G_dev": g_dev, "required_strictly_greater_than": threshold,
                "pass": postbreak_pass,
            },
            "development_pass": passed,
        },
        "artifacts": artifacts, "confirmation_2024_authorized": passed,
        "goal_status": "active", "goal_completion_claimed": False,
    }
    REPORT.write_text(json.dumps(safe(report), ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(safe({"status": report["status"], "gates": report["gates"]}), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
