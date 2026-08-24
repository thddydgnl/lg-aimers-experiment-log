#!/usr/bin/env python3
"""Gate a fixed equal blend of seed-bagged C and dense pitch MoE."""

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
    PRED,
    digest,
    load,
    safe,
    score,
)
from experiments.run_v5_h1_residual import (  # noqa: E402
    cluster_bootstrap_score_gain,
)


PREREG = (
    ROOT
    / "experiments/params/v5_dense_seed_low_complexity_ensemble_preregister.json"
)
SEED_REPORT = ROOT / "experiments/results/v5_exact_c_multiseed_source.json"
DENSE_REPORT = ROOT / "experiments/results/v5_dense_pitchtype_moe_source_gate_v2.json"
REPORT = (
    ROOT / "experiments/results/v5_dense_seed_low_complexity_ensemble_source.json"
)
TRAIN = ROOT / "open/data/train.csv"
YEARS = (2020, 2021)
PARENT = {
    2020: "v4_m3_c_backtest_2020_2020.npz",
    2021: "v4_m3_c_backtest_2021_2021.npz",
}
DENSE_STAGE = {
    2020: "v5_dense_pitchtype_moe_source2020",
    2021: "v5_dense_pitchtype_moe_source2021",
}
DENSE_KEY = "catboost_dense_pitchtype_moe"


def main() -> None:
    if REPORT.exists():
        raise FileExistsError(f"immutable result already exists: {REPORT}")
    prereg = json.loads(PREREG.read_text(encoding="utf-8"))
    seed_report = json.loads(SEED_REPORT.read_text(encoding="utf-8"))
    dense_report = json.loads(DENSE_REPORT.read_text(encoding="utf-8"))
    if seed_report["source_gate"]["passed"]:
        raise ValueError("unexpected seed-bag source status")
    if dense_report["status"] != "source_failed":
        raise ValueError("unexpected dense-MoE source status")
    dense_gamma = float(dense_report["selected"]["gamma"])
    if dense_gamma != 0.5:
        raise ValueError("frozen dense-MoE gamma is not 0.5")
    weights = {
        name: float(config["weight"])
        for name, config in prereg["frozen_components"].items()
    }
    if weights != {"seed_bag": 0.5, "dense_moe": 0.5}:
        raise ValueError("equal ensemble weights changed")

    all_types = pd.read_csv(TRAIN, usecols=["game_type"])["game_type"].astype(str)
    iterations = int(prereg["bootstrap_iterations"])
    years: dict[str, Any] = {}
    checks: list[bool] = []
    minimum_full = float(prereg["source_gate"]["minimum_full_gain_each_year"])
    minimum_r = float(prereg["source_gate"]["minimum_r_gain_each_year"])

    for year in YEARS:
        parent_path = PRED / PARENT[year]
        parent = load(parent_path)
        seed_path = PRED / f"v5_exact_c_multiseed_source_{year}.npz"
        seed = load(seed_path)
        dense_path = PRED / f"{DENSE_STAGE[year]}_{year}.npz"
        dense = load(dense_path)
        for artifact_name, artifact in (("seed", seed), ("dense", dense)):
            for key in ("y", "row_index", "cluster"):
                if not np.array_equal(parent[key], artifact[key]):
                    raise ValueError(
                        f"alignment mismatch: {year}/{artifact_name}/{key}"
                    )
        parent_p = parent["catboost_outcome"].astype(np.float64)
        seed_p = seed["candidate_uniform_three_seed"].astype(np.float64)
        types = all_types.iloc[parent["row_index"].astype(np.int64)].to_numpy(dtype=str)
        regular = types == "R"
        dense_p = parent_p.copy()
        dense_raw = dense[DENSE_KEY].astype(np.float64)
        dense_p[regular] += dense_gamma * (dense_raw[regular] - parent_p[regular])
        dense_p = np.clip(dense_p, 1e-6, 1.0 - 1e-6)
        ensemble = np.clip(
            weights["seed_bag"] * seed_p + weights["dense_moe"] * dense_p,
            1e-6,
            1.0 - 1e-6,
        )
        masks = {
            "full": np.ones(len(parent_p), dtype=bool),
            "R": regular,
        }
        route_results: dict[str, Any] = {}
        for route_index, (route, mask) in enumerate(masks.items()):
            parent_metrics = score(parent["y"], parent_p, mask)
            candidate_metrics = score(parent["y"], ensemble, mask)
            interval = cluster_bootstrap_score_gain(
                parent["y"], parent_p, ensemble, parent["cluster"], mask,
                iterations=iterations,
                seed=910000 + 10000 * year + 1000 * route_index,
            )
            gain = candidate_metrics["score"] - parent_metrics["score"]
            if abs(gain - interval["point"]) > 1e-8:
                raise AssertionError(f"score/CI mismatch: {year}/{route}")
            threshold = minimum_full if route == "full" else minimum_r
            point_pass = bool(gain >= threshold)
            ci_pass = bool(interval["ci_low"] > 0.0)
            checks.extend((point_pass, ci_pass))
            route_results[route] = {
                "parent": parent_metrics,
                "candidate": candidate_metrics,
                "gain": gain,
                "pitcher_cluster_95_ci": interval,
                "passes_point_gate": point_pass,
                "passes_ci_gate": ci_pass,
            }
        direction_seed = seed_p - parent_p
        direction_dense = dense_p - parent_p
        output = PRED / f"v5_dense_seed_low_complexity_ensemble_source_{year}.npz"
        if output.exists():
            raise FileExistsError(f"immutable artifact exists: {output}")
        np.savez_compressed(
            output,
            y=parent["y"].astype(np.int8),
            row_index=parent["row_index"].astype(np.int64),
            cluster=parent["cluster"],
            parent_exact_c=parent_p,
            seed_bag=seed_p,
            dense_moe_frozen=dense_p,
            final_equal_ensemble=ensemble,
        )
        years[str(year)] = {
            "routes": route_results,
            "direction_correlation": {
                "full": float(np.corrcoef(direction_seed, direction_dense)[0, 1]),
                "R": float(
                    np.corrcoef(
                        direction_seed[regular], direction_dense[regular]
                    )[0, 1]
                ),
            },
            "artifacts": {
                "parent": str(parent_path.relative_to(ROOT)),
                "seed_bag": str(seed_path.relative_to(ROOT)),
                "dense_moe": str(dense_path.relative_to(ROOT)),
                "ensemble": str(output.relative_to(ROOT)),
                "ensemble_sha256": digest(output),
            },
        }

    passed = bool(all(checks))
    report = {
        "experiment_id": prereg["experiment_id"],
        "status": "source_pass" if passed else "source_failed",
        "preregister_sha256": digest(PREREG),
        "script_sha256": digest(Path(__file__)),
        "frozen_components": {
            "seed_report": {
                "path": str(SEED_REPORT.relative_to(ROOT)),
                "sha256": digest(SEED_REPORT),
            },
            "dense_report": {
                "path": str(DENSE_REPORT.relative_to(ROOT)),
                "sha256": digest(DENSE_REPORT),
                "gamma": dense_gamma,
            },
            "weights": weights,
        },
        "years_read": list(YEARS),
        "years_not_read": [2022, 2023, 2024],
        "years": years,
        "source_gate": {
            "minimum_full_gain_each_year": minimum_full,
            "minimum_R_gain_each_year": minimum_r,
            "ci_lower_positive_each_year": True,
            "passed": passed,
            "decision": (
                "freeze and advance to 2022/2023"
                if passed
                else "close without reading 2022+ ensemble labels"
            ),
        },
    }
    REPORT.write_text(
        json.dumps(safe(report), ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "status": report["status"],
                "per_year": {
                    str(year): {
                        route: {
                            "gain": years[str(year)]["routes"][route]["gain"],
                            "ci_low": years[str(year)]["routes"][route][
                                "pitcher_cluster_95_ci"
                            ]["ci_low"],
                        }
                        for route in ("full", "R")
                    }
                    for year in YEARS
                },
                "direction_correlation": {
                    str(year): years[str(year)]["direction_correlation"]
                    for year in YEARS
                },
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
