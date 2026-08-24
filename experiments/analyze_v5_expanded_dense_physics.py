#!/usr/bin/env python3
"""Immutable source gate for expanded-profile dense physics MoE."""

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
from experiments.run_v5_h1_residual import (  # noqa: E402
    cluster_bootstrap_score_gain,
)

RESULTS = ROOT / "experiments/results"
PRED = RESULTS / "predictions"
PREREG = ROOT / "experiments/params/v5_expanded_dense_physics_preregister.json"
PROFILE_REPORT = RESULTS / "v5_expanded_trackman_profiles_source.json"
PREDECESSOR_REPORT = RESULTS / "v5_dense_physics_pitchtype_moe_source_gate.json"
REPORT = RESULTS / "v5_expanded_dense_physics_source.json"
TRAIN = ROOT / "open/data/train.csv"
YEARS = (2020, 2021)
KEY = "catboost_dense_pitchtype_moe"
GAMMA = 0.5


def main() -> None:
    if REPORT.exists():
        raise FileExistsError(f"immutable result already exists: {REPORT}")
    prereg = json.loads(PREREG.read_text(encoding="utf-8"))
    profile_report = json.loads(PROFILE_REPORT.read_text(encoding="utf-8"))
    predecessor_report = json.loads(PREDECESSOR_REPORT.read_text(encoding="utf-8"))
    if float(preregistered_gamma := prereg["frozen_predecessor"]["gamma"]) != GAMMA:
        raise ValueError(f"frozen gamma changed: {preregistered_gamma}")
    if profile_report["status"] != "source_pass":
        raise ValueError("expanded profile source did not pass")
    if float(predecessor_report["selected"]["gamma"]) != GAMMA:
        raise ValueError("predecessor selected gamma changed")

    game_types = pd.read_csv(TRAIN, usecols=["game_type"])["game_type"].astype(str)
    folds: dict[int, dict[str, Any]] = {}
    semantic_checks: list[bool] = []
    semantic_gate = prereg["semantic_gate"]

    for year in YEARS:
        stage = f"v5_expanded_dense_physics_source{year}"
        candidate_path = PRED / f"{stage}_{year}.npz"
        parent_path = PRED / f"v4_m3_c_backtest_{year}_{year}.npz"
        predecessor_path = PRED / f"v5_dense_physics_pitchtype_moe_source_{year}.npz"
        metadata_path = RESULTS / f"{stage}.json"
        candidate = load(candidate_path)
        parent_artifact = load(parent_path)
        predecessor = load(predecessor_path)
        for name, artifact in (("candidate", candidate), ("predecessor", predecessor)):
            for key in ("y", "row_index", "cluster"):
                if not np.array_equal(artifact[key], parent_artifact[key]):
                    raise ValueError(f"alignment mismatch: {year}/{name}/{key}")
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        meta = metadata["metadata"]
        fold = metadata["folds"][0]
        details = fold["fit_details"][KEY]
        expanded = fold["expanded_trackman_profiles"]
        trackman = fold["trackman"]
        old_stage = f"v5_dense_physics_pitchtype_moe_source{year}"
        old_metadata_path = RESULTS / f"{old_stage}.json"
        old_metadata = json.loads(old_metadata_path.read_text(encoding="utf-8"))
        old_details = old_metadata["folds"][0]["fit_details"][KEY]
        audited = predecessor_report["semantic"][str(year)]
        identity_purities = [
            value["minimum_purity"]
            for value in expanded["identity"].values()
            if value["minimum_purity"] is not None
        ]
        semantic = {
            "history_dense_label_coverage": float(
                details["history_dense_label_coverage"]
            ),
            "history_trackman_group_agreement": float(
                audited["history_trackman_group_agreement"]
            ),
            "expanded_profile_row_factor": float(expanded["row_expansion_factor"]),
            "expanded_major_rows": int(expanded["expanded_major_rows"]),
            "trackman_source_rows": int(trackman["source_rows"]),
            "minimum_identity_purity": float(min(identity_purities)),
            "current_pitch_group_used_at_inference": bool(
                details["current_pitch_group_used_at_inference"]
            ),
            "row_independent_routing": bool(details["row_independent_routing"]),
            "current_validation_trackman_used": bool(
                expanded["current_validation_trackman_used"]
            ),
            "candidate_selector": {
                "log_loss": float(details["diagnostic_selector_log_loss"]),
                "top1_accuracy": float(
                    details["diagnostic_selector_top1_accuracy"]
                ),
            },
            "predecessor_selector": {
                "log_loss": float(old_details["diagnostic_selector_log_loss"]),
                "top1_accuracy": float(
                    old_details["diagnostic_selector_top1_accuracy"]
                ),
            },
            "recipe_exact": bool(
                meta["models"] == [prereg["candidate"]["model"]]
                and meta["features"] == prereg["candidate"]["features"]
                and fold["dropped_features"] == prereg["candidate"]["drop_features"]
                and meta["outcome_scheme"] == prereg["candidate"]["outcome_scheme"]
                and meta["inner_validation"] == prereg["candidate"]["inner_validation"]
            ),
        }
        semantic["pass"] = bool(
            semantic["history_dense_label_coverage"]
            >= float(semantic_gate["minimum_history_dense_label_coverage_each_year"])
            and semantic["history_trackman_group_agreement"]
            >= float(
                semantic_gate[
                    "minimum_history_trackman_group_agreement_each_year"
                ]
            )
            and semantic["expanded_profile_row_factor"]
            >= float(
                semantic_gate["minimum_profile_row_expansion_factor_each_year"]
            )
            and semantic["minimum_identity_purity"]
            >= float(semantic_gate["minimum_identity_purity"])
            and semantic["expanded_major_rows"] == semantic["trackman_source_rows"]
            and not semantic["current_pitch_group_used_at_inference"]
            and semantic["row_independent_routing"]
            and not semantic["current_validation_trackman_used"]
            and semantic["recipe_exact"]
        )
        semantic_checks.append(semantic["pass"])
        regular = (
            game_types.iloc[candidate["row_index"].astype(np.int64)].to_numpy(
                dtype=str
            )
            == "R"
        )
        parent = parent_artifact["catboost_outcome"].astype(np.float64)
        raw = candidate[KEY].astype(np.float64)
        expanded_final = parent.copy()
        expanded_final[regular] += GAMMA * (raw[regular] - parent[regular])
        expanded_final = np.clip(expanded_final, 1e-6, 1.0 - 1e-6)
        folds[year] = {
            "artifact": candidate,
            "parent": parent,
            "raw": raw,
            "expanded_final": expanded_final,
            "predecessor_final": predecessor["final_prediction"].astype(np.float64),
            "regular": regular,
            "masks": {"full": np.ones(len(raw), dtype=bool), "R": regular},
            "semantic": semantic,
            "paths": {
                "candidate": candidate_path,
                "parent": parent_path,
                "predecessor": predecessor_path,
                "metadata": metadata_path,
                "predecessor_metadata": old_metadata_path,
            },
        }

    results: dict[str, Any] = {}
    exact_gate = prereg["source_protocol"]["exact_parent_gate"]
    increment_gate = prereg["source_protocol"]["predecessor_increment_gate"]
    checks = list(semantic_checks)
    for year in YEARS:
        fold = folds[year]
        exact = evaluate(
            fold["artifact"], fold["parent"], fold["raw"], fold["regular"],
            fold["masks"], GAMMA,
            int(prereg["source_protocol"]["bootstrap_iterations"]),
            5310000 + 10000 * year,
        )
        increment: dict[str, Any] = {}
        for route_index, (route, mask) in enumerate(fold["masks"].items()):
            old_metrics = score(
                fold["artifact"]["y"], fold["predecessor_final"], mask
            )
            new_metrics = score(
                fold["artifact"]["y"], fold["expanded_final"], mask
            )
            interval = cluster_bootstrap_score_gain(
                fold["artifact"]["y"],
                fold["predecessor_final"],
                fold["expanded_final"],
                fold["artifact"]["cluster"],
                mask,
                iterations=int(prereg["source_protocol"]["bootstrap_iterations"]),
                seed=5320000 + 10000 * year + 1000 * route_index,
            )
            increment[route] = {
                "predecessor": old_metrics,
                "candidate": new_metrics,
                "gain": float(new_metrics["score"] - old_metrics["score"]),
                "pitcher_cluster_95_ci": interval,
            }
        selector = fold["semantic"]
        selector_checks = {
            "log_loss_lower": selector["candidate_selector"]["log_loss"]
            < selector["predecessor_selector"]["log_loss"],
            "top1_not_lower": selector["candidate_selector"]["top1_accuracy"]
            >= selector["predecessor_selector"]["top1_accuracy"],
        }
        routes = exact["routes"]
        year_checks = {
            "exact_full_point": routes["full"]["gain"]
            >= float(exact_gate["minimum_full_gain_each_year"]),
            "exact_R_point": routes["R"]["gain"]
            >= float(exact_gate["minimum_R_gain_each_year"]),
            "exact_full_ci": routes["full"]["pitcher_cluster_95_ci"]["ci_low"]
            > 0.0,
            "exact_R_ci": routes["R"]["pitcher_cluster_95_ci"]["ci_low"] > 0.0,
            "increment_full": increment["full"]["gain"]
            >= float(
                increment_gate[
                    "minimum_full_gain_over_frozen_predecessor_each_year"
                ]
            ),
            "increment_R": increment["R"]["gain"]
            >= float(
                increment_gate[
                    "minimum_R_gain_over_frozen_predecessor_each_year"
                ]
            ),
            **selector_checks,
        }
        checks.extend(year_checks.values())
        results[str(year)] = {
            "semantic": fold["semantic"],
            "vs_exact_parent": exact,
            "vs_frozen_predecessor": increment,
            "checks": year_checks,
        }

    passed = bool(all(checks))
    artifacts: dict[str, Any] = {}
    for year in YEARS:
        fold = folds[year]
        output = PRED / f"v5_expanded_dense_physics_source_{year}.npz"
        if output.exists():
            raise FileExistsError(f"immutable artifact exists: {output}")
        np.savez_compressed(
            output,
            y=fold["artifact"]["y"].astype(np.int8),
            row_index=fold["artifact"]["row_index"].astype(np.int64),
            cluster=fold["artifact"]["cluster"],
            parent_exact_c=fold["parent"],
            expanded_dense_physics_raw=fold["raw"],
            final_prediction=fold["expanded_final"],
        )
        artifacts[str(year)] = {
            "path": str(output.relative_to(ROOT)), "sha256": digest(output)
        }

    report = {
        "experiment_id": prereg["experiment_id"],
        "status": "source_pass" if passed else "source_failed",
        "preregister_sha256": digest(PREREG),
        "profile_report_sha256": digest(PROFILE_REPORT),
        "predecessor_report_sha256": digest(PREDECESSOR_REPORT),
        "script_sha256": digest(Path(__file__)),
        "years_read": list(YEARS),
        "years_not_read": [2022, 2023, 2024],
        "gamma": GAMMA,
        "years": results,
        "input_sha256": {
            str(year): {
                name: digest(path) for name, path in folds[year]["paths"].items()
            }
            for year in YEARS
        },
        "source_gate": {
            "exact_parent": exact_gate,
            "predecessor_increment": increment_gate,
            "pass": passed,
            "decision": (
                "advance locked expanded recipe to 2022/2023"
                if passed
                else "close without new 2022+ candidate"
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
                    "years": {
                        year: {
                            "exact": {
                                route: results[year]["vs_exact_parent"]["routes"][route][
                                    "gain"
                                ]
                                for route in ("full", "R")
                            },
                            "increment": {
                                route: results[year]["vs_frozen_predecessor"][route][
                                    "gain"
                                ]
                                for route in ("full", "R")
                            },
                            "checks": results[year]["checks"],
                        }
                        for year in results
                    },
                }
            ),
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
