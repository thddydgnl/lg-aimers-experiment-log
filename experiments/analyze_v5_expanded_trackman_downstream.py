#!/usr/bin/env python3
"""Immutable source gate for expanded TrackMan profile downstream arms."""

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
)

RESULTS = ROOT / "experiments/results"
PRED = RESULTS / "predictions"
PREREG = (
    ROOT / "experiments/params/v5_expanded_trackman_downstream_preregister.json"
)
PROFILE_REPORT = RESULTS / "v5_expanded_trackman_profiles_source.json"
REPORT = RESULTS / "v5_expanded_trackman_downstream_source.json"
TRAIN = ROOT / "open/data/train.csv"
PARAMS = ROOT / "experiments/params/v3_outcome_seed2026.json"
YEARS = (2020, 2021)
KEY = "catboost_outcome"
EXPECTED_PARAMS = {
    "iterations": 500,
    "depth": 6,
    "learning_rate": 0.05,
    "l2_leaf_reg": 12.0,
    "random_seed": 2026,
}


def main() -> None:
    if REPORT.exists():
        raise FileExistsError(f"immutable result already exists: {REPORT}")
    prereg = json.loads(PREREG.read_text(encoding="utf-8"))
    profile_report = json.loads(PROFILE_REPORT.read_text(encoding="utf-8"))
    params = json.loads(PARAMS.read_text(encoding="utf-8"))
    if params != EXPECTED_PARAMS:
        raise ValueError("frozen booster parameters changed")
    if profile_report["status"] != prereg["semantic_gate"][
        "expanded_profile_source_status"
    ]:
        raise ValueError("expanded TrackMan source did not pass its target-free gate")
    if profile_report["downstream_control_metrics_read"]:
        raise ValueError("source report claims downstream metrics were already read")

    game_types = pd.read_csv(TRAIN, usecols=["game_type"])["game_type"].astype(str)
    arms: dict[str, dict[int, dict[str, Any]]] = {}
    semantic_checks: list[bool] = []
    shared = prereg["shared_recipe"]
    semantic_gate = prereg["semantic_gate"]

    for arm_name, arm_config in prereg["arms"].items():
        arms[arm_name] = {}
        for year in YEARS:
            stage = f'{arm_config["stage_stem"]}{year}'
            candidate_path = PRED / f"{stage}_{year}.npz"
            parent_path = PRED / f"v4_m3_c_backtest_{year}_{year}.npz"
            metadata_path = RESULTS / f"{stage}.json"
            candidate = load(candidate_path)
            parent_artifact = load(parent_path)
            for align_key in ("y", "row_index", "cluster"):
                if not np.array_equal(
                    candidate[align_key], parent_artifact[align_key]
                ):
                    raise ValueError(
                        f"alignment mismatch: {arm_name}/{year}/{align_key}"
                    )
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            meta = metadata["metadata"]
            fold = metadata["folds"][0]
            expanded = fold["expanded_trackman_profiles"]
            trackman = fold["trackman"]
            source_fold = profile_report["folds"][str(year)]["metadata"]
            feature_columns = fold["feature_columns"]
            prefix_counts = {
                prefix: sum(column.startswith(prefix) for column in feature_columns)
                for prefix in arm_config["required_trackman_prefixes"]
            }
            expected_trackman_flags = {
                "rich": True,
                "stability": arm_name == "command_bundle",
                "group_stability": arm_name == "command_bundle",
                "trend": arm_name == "command_bundle",
                "platoon": arm_name == "command_bundle",
                "count": arm_name == "command_bundle",
            }
            actual_trackman_flags = {
                name: bool(trackman[name]) for name in expected_trackman_flags
            }
            identity_purities = [
                value["minimum_purity"]
                for value in expanded["identity"].values()
                if value["minimum_purity"] is not None
            ]
            semantic = {
                "metadata_stage": meta["stage"],
                "metadata_models": meta["models"],
                "metadata_features": meta["features"],
                "metadata_outcome_scheme": meta["outcome_scheme"],
                "metadata_inner_validation": meta["inner_validation"],
                "metadata_booster_params": meta["booster_params"],
                "metadata_booster_device": meta["booster_device"],
                "dropped_features_exact": fold["dropped_features"]
                == shared["drop_features"],
                "expanded_enabled": bool(expanded["enabled"]),
                "row_expansion_factor": float(expanded["row_expansion_factor"]),
                "expanded_major_rows": int(expanded["expanded_major_rows"]),
                "trackman_source_rows": int(trackman["source_rows"]),
                "mapped_pitchers": int(expanded["mapped_pitchers"]),
                "minimum_identity_purity": float(min(identity_purities)),
                "profile_aggregation_only": bool(
                    expanded["profile_aggregation_only"]
                ),
                "target_columns_read": bool(expanded["target_columns_read"]),
                "current_validation_trackman_used": bool(
                    expanded["current_validation_trackman_used"]
                ),
                "external_data_used": bool(expanded["external_data_used"]),
                "unmatched_game_claimed_as_main_row": bool(
                    expanded["unmatched_game_claimed_as_main_row"]
                ),
                "trackman_flags": actual_trackman_flags,
                "required_feature_prefix_counts": prefix_counts,
                "source_report_match": {
                    "metadata_exact": {
                        key: value
                        for key, value in expanded.items()
                        if key != "enabled"
                    }
                    == source_fold,
                    "expanded_major_rows": int(expanded["expanded_major_rows"])
                    == int(source_fold["expanded_major_rows"]),
                    "row_expansion_factor": abs(
                        float(expanded["row_expansion_factor"])
                        - float(source_fold["row_expansion_factor"])
                    )
                    < 1e-12,
                },
            }
            semantic["pass"] = bool(
                meta["stage"] == stage
                and meta["models"] == [shared["model"]]
                and meta["features"] == arm_config["features"]
                and meta["outcome_scheme"] == shared["outcome_scheme"]
                and meta["inner_validation"] == shared["inner_validation"]
                and meta["booster_params"] == EXPECTED_PARAMS
                and semantic["dropped_features_exact"]
                and semantic["expanded_enabled"]
                and semantic["row_expansion_factor"]
                >= float(semantic_gate["minimum_row_expansion_factor_each_year"])
                and semantic["mapped_pitchers"]
                >= int(semantic_gate["minimum_mapped_pitchers"])
                and semantic["minimum_identity_purity"]
                >= float(semantic_gate["minimum_identity_purity"])
                and semantic["profile_aggregation_only"]
                and not semantic["target_columns_read"]
                and not semantic["current_validation_trackman_used"]
                and not semantic["external_data_used"]
                and not semantic["unmatched_game_claimed_as_main_row"]
                and semantic["expanded_major_rows"]
                == semantic["trackman_source_rows"]
                and actual_trackman_flags == expected_trackman_flags
                and all(count > 0 for count in prefix_counts.values())
                and all(semantic["source_report_match"].values())
            )
            semantic_checks.append(semantic["pass"])
            row_index = candidate["row_index"].astype(np.int64)
            regular = game_types.iloc[row_index].to_numpy(dtype=str) == "R"
            parent = parent_artifact[KEY].astype(np.float64)
            raw = candidate[KEY].astype(np.float64)
            arms[arm_name][year] = {
                "candidate_artifact": candidate,
                "parent": parent,
                "raw": raw,
                "regular": regular,
                "masks": {
                    "full": np.ones(len(raw), dtype=bool),
                    "R": regular,
                },
                "semantic": semantic,
                "paths": {
                    "candidate": candidate_path,
                    "parent": parent_path,
                    "metadata": metadata_path,
                },
            }

    trials: list[dict[str, Any]] = []
    prediction_cache: dict[tuple[str, int, float], np.ndarray] = {}
    if all(semantic_checks):
        for arm_index, arm_name in enumerate(prereg["arms"]):
            for gamma in prereg["source_protocol"]["top_level_blend_grid"]:
                gamma = float(gamma)
                years: dict[str, Any] = {}
                for year in YEARS:
                    fold = arms[arm_name][year]
                    years[str(year)] = evaluate(
                        fold["candidate_artifact"],
                        fold["parent"],
                        fold["raw"],
                        fold["regular"],
                        fold["masks"],
                        gamma,
                        int(prereg["source_protocol"]["bootstrap_iterations"]),
                        4670000 + 100000 * arm_index + 10000 * year
                        + int(gamma * 100),
                    )
                    prediction = fold["parent"].copy()
                    prediction[fold["regular"]] += gamma * (
                        fold["raw"][fold["regular"]]
                        - fold["parent"][fold["regular"]]
                    )
                    prediction_cache[(arm_name, year, gamma)] = np.clip(
                        prediction, 1e-6, 1.0 - 1e-6
                    )
                r_gains = [years[str(y)]["routes"]["R"]["gain"] for y in YEARS]
                full_gains = [
                    years[str(y)]["routes"]["full"]["gain"] for y in YEARS
                ]
                trials.append(
                    {
                        "arm": arm_name,
                        "arm_tiebreak_index": arm_index,
                        "gamma": gamma,
                        "minimum_R_gain": float(min(r_gains)),
                        "minimum_full_gain": float(min(full_gains)),
                        "mean_R_gain": float(np.mean(r_gains)),
                        "years": years,
                    }
                )
    selected = (
        max(
            trials,
            key=lambda item: (
                item["minimum_R_gain"],
                item["minimum_full_gain"],
                item["mean_R_gain"],
                -item["gamma"],
                -item["arm_tiebreak_index"],
            ),
        )
        if trials
        else None
    )

    gate = prereg["source_protocol"]["gate"]
    checks = [all(semantic_checks), selected is not None]
    if selected is not None:
        for year in YEARS:
            routes = selected["years"][str(year)]["routes"]
            checks.extend(
                [
                    routes["R"]["gain"]
                    >= float(gate["minimum_R_gain_each_year"]),
                    routes["full"]["gain"]
                    >= float(gate["minimum_full_gain_each_year"]),
                    routes["R"]["pitcher_cluster_95_ci"]["ci_low"] > 0.0,
                    routes["full"]["pitcher_cluster_95_ci"]["ci_low"] > 0.0,
                ]
            )
    passed = bool(all(checks))

    artifacts: dict[str, Any] = {}
    if selected is not None:
        for year in YEARS:
            fold = arms[selected["arm"]][year]
            output = PRED / f"v5_expanded_trackman_selected_source_{year}.npz"
            if output.exists():
                raise FileExistsError(f"immutable artifact exists: {output}")
            final_prediction = prediction_cache[
                (selected["arm"], year, selected["gamma"])
            ]
            np.savez_compressed(
                output,
                y=fold["candidate_artifact"]["y"].astype(np.int8),
                row_index=fold["candidate_artifact"]["row_index"].astype(np.int64),
                cluster=fold["candidate_artifact"]["cluster"],
                parent_exact_c=fold["parent"],
                expanded_trackman_raw=fold["raw"],
                final_prediction=final_prediction,
            )
            artifacts[str(year)] = {
                "path": str(output.relative_to(ROOT)),
                "sha256": digest(output),
            }

    report = {
        "experiment_id": prereg["experiment_id"],
        "status": "source_pass" if passed else "source_failed",
        "profile_source": {
            "path": str(PROFILE_REPORT.relative_to(ROOT)),
            "sha256": digest(PROFILE_REPORT),
        },
        "preregister_sha256": digest(PREREG),
        "params_sha256": digest(PARAMS),
        "script_sha256": digest(Path(__file__)),
        "years_read": list(YEARS),
        "years_not_read": [2022, 2023, 2024],
        "semantic": {
            arm_name: {
                str(year): arms[arm_name][year]["semantic"] for year in YEARS
            }
            for arm_name in arms
        },
        "input_sha256": {
            arm_name: {
                str(year): {
                    name: digest(path)
                    for name, path in arms[arm_name][year]["paths"].items()
                }
                for year in YEARS
            }
            for arm_name in arms
        },
        "trials": trials,
        "selected": selected,
        "source_gate": {
            "requirements": gate,
            "pass": passed,
            "decision": (
                "freeze selected arm and gamma; advance to 2022/2023"
                if passed
                else "close direction without reading 2022+ labels"
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
                    "semantic_pass": all(semantic_checks),
                }
            ),
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
