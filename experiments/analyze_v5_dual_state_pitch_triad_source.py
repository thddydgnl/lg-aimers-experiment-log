#!/usr/bin/env python3
"""Reproduce and audit the locked 2020/2021 dual-state pitch triad search."""

from __future__ import annotations

from itertools import combinations
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
    load,
    safe,
    score,
)
from experiments.run_v5_h1_residual import (  # noqa: E402
    cluster_bootstrap_score_gain,
    load_anchor,
)


RESULTS = ROOT / "experiments/results"
PRED = RESULTS / "predictions"
TRAIN = ROOT / "open/data/train.csv"
PREREG = ROOT / "experiments/params/v5_dual_state_pitch_triad_preregister.json"
REPORT = RESULTS / "v5_dual_state_pitch_triad_source.json"
YEARS = (2020, 2021)
DENOMINATOR = 40
BOOTSTRAP_ITERATIONS = 2000
TOP_K = 20
EXPECTED_NAMES = ("direct_update", "expanded_auto", "current_numeric")
EXPECTED_WEIGHTS = np.asarray([0.55, 0.25, 0.20], dtype=np.float64)


def gain_from_gram(
    weights: np.ndarray,
    gram: np.ndarray,
    base_mse: float,
    reference: float,
) -> float:
    return float(100000.0 * (base_mse - weights @ gram @ weights) / reference)


def route_metrics(
    y: np.ndarray,
    parent: np.ndarray,
    candidate: np.ndarray,
    cluster: np.ndarray,
    masks: dict[str, np.ndarray],
    seed: int,
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for route_index, (route, mask) in enumerate(masks.items()):
        parent_score = score(y, parent, mask)
        candidate_score = score(y, candidate, mask)
        interval = cluster_bootstrap_score_gain(
            y,
            parent,
            candidate,
            cluster,
            mask,
            iterations=BOOTSTRAP_ITERATIONS,
            seed=seed + 1000 * route_index,
        )
        point = float(candidate_score["score"] - parent_score["score"])
        if abs(point - float(interval["point"])) > 1e-8:
            raise AssertionError(f"score/CI point mismatch: {route}")
        result[route] = {
            "parent": parent_score,
            "candidate": candidate_score,
            "gain": point,
            "pitcher_cluster_95_ci": interval,
        }
    return result


def main() -> None:
    if REPORT.exists():
        raise FileExistsError(f"immutable result already exists: {REPORT}")
    prereg = json.loads(PREREG.read_text(encoding="utf-8"))
    catalog = prereg["source_search"]["catalog"]
    names = tuple(catalog)
    if DENOMINATOR != round(1.0 / float(prereg["source_search"]["weight_step"])):
        raise AssertionError("weight grid disagrees with preregistration")
    all_types = pd.read_csv(TRAIN, usecols=["game_type"])["game_type"].astype(str)

    folds: dict[int, dict[str, Any]] = {}
    input_hashes: dict[str, dict[str, str]] = {}
    semantic_checks: dict[str, dict[str, bool]] = {}
    for year in YEARS:
        parent_artifact = load_anchor(year)
        row_index = parent_artifact["row_index"].astype(np.int64)
        y = parent_artifact["y"].astype(np.int8)
        cluster = parent_artifact["cluster"]
        parent = parent_artifact["final_prediction"].astype(np.float64)
        game_type = all_types.iloc[row_index].to_numpy(dtype=str)
        r_mask = game_type == "R"
        arrays: list[np.ndarray] = []
        paths: dict[str, Path] = {}
        year_checks: dict[str, bool] = {}
        for name, (template, key) in catalog.items():
            path = PRED / template.format(year=year)
            artifact = load(path)
            paths[name] = path
            year_checks[f"{name}_alignment"] = bool(
                np.array_equal(artifact["y"], y)
                and np.array_equal(artifact["row_index"], row_index)
                and np.array_equal(artifact["cluster"], cluster)
            )
            prediction = artifact[key].astype(np.float64)
            year_checks[f"{name}_finite"] = bool(np.isfinite(prediction).all())
            year_checks[f"{name}_open_probability"] = bool(
                np.all((prediction > 0.0) & (prediction < 1.0))
            )
            arrays.append(prediction)
        if not all(year_checks.values()):
            raise AssertionError(f"source semantic/alignment failure: {year}")
        matrix = np.column_stack(arrays)
        errors_r = matrix[r_mask] - y[r_mask, None].astype(np.float64)
        gram_r = errors_r.T @ errors_r / float(r_mask.sum())
        parent_error_r = parent[r_mask] - y[r_mask]
        parent_mse_r = float(np.mean(np.square(parent_error_r)))
        r_rate = float(y[r_mask].mean())
        full_rate = float(y.mean())
        reference_r = max(r_rate * (1.0 - r_rate), 1e-12)
        reference_full = max(full_rate * (1.0 - full_rate), 1e-12)
        folds[year] = {
            "y": y,
            "row_index": row_index,
            "cluster": cluster,
            "parent": parent,
            "matrix": matrix,
            "r_mask": r_mask,
            "gram_r": gram_r,
            "parent_mse_r": parent_mse_r,
            "reference_r": reference_r,
            "routed_full_gain_multiplier": float(
                r_mask.mean() * reference_r / reference_full
            ),
        }
        semantic_checks[str(year)] = year_checks
        input_hashes[str(year)] = {
            "parent_reconstruction": "deterministic load_anchor",
            **{name: digest(path) for name, path in paths.items()},
        }

    trials: list[dict[str, Any]] = []
    for indices in combinations(range(len(names)), 3):
        component_names = tuple(names[index] for index in indices)
        for first in range(1, DENOMINATOR - 1):
            for second in range(1, DENOMINATOR - first):
                third = DENOMINATOR - first - second
                if third <= 0:
                    continue
                local_weights = np.asarray(
                    [first, second, third], dtype=np.float64
                ) / DENOMINATOR
                global_weights = np.zeros(len(names), dtype=np.float64)
                global_weights[list(indices)] = local_weights
                r_gains = {
                    str(year): gain_from_gram(
                        global_weights,
                        folds[year]["gram_r"],
                        folds[year]["parent_mse_r"],
                        folds[year]["reference_r"],
                    )
                    for year in YEARS
                }
                routed_full_gains = {
                    str(year): float(
                        r_gains[str(year)]
                        * folds[year]["routed_full_gain_multiplier"]
                    )
                    for year in YEARS
                }
                trials.append({
                    "names": component_names,
                    "weights": local_weights,
                    "minimum_R_gain": float(min(r_gains.values())),
                    "mean_R_gain": float(np.mean(list(r_gains.values()))),
                    "minimum_routed_full_gain": float(
                        min(routed_full_gains.values())
                    ),
                    "R_gain": r_gains,
                    "routed_full_gain": routed_full_gains,
                })

    ranked = sorted(
        trials,
        key=lambda item: (
            item["minimum_R_gain"],
            item["mean_R_gain"],
            item["minimum_routed_full_gain"],
            tuple(item["names"]),
            tuple(float(value) for value in item["weights"]),
        ),
        reverse=True,
    )
    selected = ranked[0]
    if tuple(selected["names"]) != EXPECTED_NAMES or not np.allclose(
        selected["weights"], EXPECTED_WEIGHTS, atol=1e-12, rtol=0.0
    ):
        raise AssertionError(
            f"source selection disagrees with lock: {selected['names']} / "
            f"{selected['weights']}"
        )
    locked = prereg["locked_recipe"]["components"]
    if tuple(item["name"] for item in locked) != EXPECTED_NAMES or not np.allclose(
        [item["weight"] for item in locked], EXPECTED_WEIGHTS,
        atol=1e-12, rtol=0.0,
    ):
        raise AssertionError("locked JSON recipe disagrees with source selection")

    global_weights = np.zeros(len(names), dtype=np.float64)
    for name, weight in zip(EXPECTED_NAMES, EXPECTED_WEIGHTS):
        global_weights[names.index(name)] = float(weight)
    metrics: dict[str, Any] = {}
    artifacts: dict[str, Any] = {}
    for year in YEARS:
        fold = folds[year]
        raw = np.clip(fold["matrix"] @ global_weights, 1e-6, 1.0 - 1e-6)
        routed = np.where(fold["r_mask"], raw, fold["parent"])
        masks = {
            "full": np.ones(len(raw), dtype=bool),
            "R": fold["r_mask"],
            "F": ~fold["r_mask"],
        }
        metrics[str(year)] = route_metrics(
            fold["y"], fold["parent"], routed, fold["cluster"], masks,
            seed=4810000 + 10000 * year,
        )
        output = PRED / f"v5_dual_state_pitch_triad_source_{year}.npz"
        if output.exists():
            raise FileExistsError(f"immutable artifact already exists: {output}")
        np.savez_compressed(
            output,
            y=fold["y"],
            row_index=fold["row_index"],
            cluster=fold["cluster"],
            parent_m3=fold["parent"],
            direct_update=fold["matrix"][:, names.index("direct_update")],
            expanded_auto=fold["matrix"][:, names.index("expanded_auto")],
            current_numeric=fold["matrix"][:, names.index("current_numeric")],
            raw_mixture=raw,
            final_prediction=routed,
        )
        artifacts[str(year)] = {
            "path": str(output.relative_to(ROOT)),
            "sha256": digest(output),
        }

    required = float(
        prereg["confirmation_and_completion_gate"][
            "required_G_robust_strictly_greater_than"
        ]
    )
    discovery_gate = prereg["source_search"]["discovery_gate"]
    source_pass = bool(
        all(all(checks.values()) for checks in semantic_checks.values())
        and all(
            metrics[str(year)]["R"]["gain"]
            >= float(discovery_gate["minimum_R_point_gain_each_year"])
            and metrics[str(year)]["R"]["pitcher_cluster_95_ci"]["ci_low"] > 0.0
            and metrics[str(year)]["full"]["pitcher_cluster_95_ci"]["ci_low"] > 0.0
            for year in YEARS
        )
    )
    minimum_full_point = float(
        min(metrics[str(year)]["full"]["gain"] for year in YEARS)
    )
    report = {
        "experiment_id": prereg["experiment_id"],
        "status": "source_lock_passed" if source_pass else "source_failed",
        "evidence_role": "retrospective discovery only; not Goal confirmation",
        "preregister_sha256": digest(PREREG),
        "script_sha256": digest(Path(__file__)),
        "years_read_by_this_script": list(YEARS),
        "years_not_read_by_this_script": [2022, 2023, 2024],
        "prior_development_knowledge_disclosure": prereg[
            "evidence_role_and_disclosure"
        ],
        "search": {
            "catalog": names,
            "weight_step": 1.0 / DENOMINATOR,
            "exactly_positive_components": 3,
            "trial_count": len(trials),
            "ranking": prereg["source_search"]["ranking"],
            "top_20": ranked[:TOP_K],
        },
        "selected": selected,
        "selected_bootstrap_diagnostics": metrics,
        "semantic_checks": semantic_checks,
        "input_sha256": input_hashes,
        "source_gate": {
            "requirements": discovery_gate,
            "pass": source_pass,
        },
        "final_threshold_audit": {
            "minimum_routed_full_point_gain": minimum_full_point,
            "required_raw_gain": required,
            "crosses_required_raw_gain": minimum_full_point > required,
            "goal_completion_allowed_from_source": False,
        },
        "artifacts": artifacts,
        "goal_status": "active",
        "goal_completion_claimed": False,
    }
    REPORT.write_text(
        json.dumps(safe(report), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(safe({
        "status": report["status"],
        "selected": selected,
        "bootstrap": metrics,
        "final_threshold_audit": report["final_threshold_audit"],
    }), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
