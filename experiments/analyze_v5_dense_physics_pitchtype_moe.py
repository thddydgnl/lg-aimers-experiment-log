#!/usr/bin/env python3
"""Source gate for dense pitch experts with matching TrackMan physics."""

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
    PREFIX,
    digest,
    evaluate,
    load,
    safe,
)


PREREG = ROOT / "experiments/params/v5_dense_physics_pitchtype_moe_preregister.json"
SEMANTIC_SOURCE = (
    ROOT / "experiments/results/v5_counter_reconstructed_pitch_hierarchy_source.json"
)
REPORT = ROOT / "experiments/results/v5_dense_physics_pitchtype_moe_source_gate.json"
TRAIN = ROOT / "open/data/train.csv"
YEARS = (2020, 2021)
PARENT = {
    2020: "v4_m3_c_backtest_2020_2020.npz",
    2021: "v4_m3_c_backtest_2021_2021.npz",
}
STAGES = {
    2020: "v5_dense_physics_pitchtype_moe_source2020",
    2021: "v5_dense_physics_pitchtype_moe_source2021",
}
KEY = "catboost_dense_pitchtype_moe"


def main() -> None:
    if REPORT.exists():
        raise FileExistsError(f"immutable result already exists: {REPORT}")
    prereg = json.loads(PREREG.read_text(encoding="utf-8"))
    semantic_source = json.loads(SEMANTIC_SOURCE.read_text(encoding="utf-8"))
    all_types = pd.read_csv(TRAIN, usecols=["game_type"])["game_type"].astype(str)
    folds: dict[int, dict[str, Any]] = {}
    semantic_pass = True
    for year in YEARS:
        parent_path = PRED / PARENT[year]
        candidate_path = PRED / f"{STAGES[year]}_{year}.npz"
        parent_artifact = load(parent_path)
        candidate = load(candidate_path)
        for key in ("y", "row_index", "cluster"):
            if not np.array_equal(parent_artifact[key], candidate[key]):
                raise ValueError(f"alignment mismatch: {year}/{key}")
        types = all_types.iloc[
            candidate["row_index"].astype(np.int64)
        ].to_numpy(dtype=str)
        regular = types == "R"
        stage_report = json.loads(
            (
                ROOT / f"experiments/results/{STAGES[year]}.json"
            ).read_text(encoding="utf-8")
        )
        details = stage_report["folds"][0]["fit_details"][KEY]
        audited = semantic_source["semantic_audit"][str(year)]
        coverage = float(details["history_dense_label_coverage"])
        agreement = float(audited["history_trackman_agreement"])
        if abs(coverage - float(audited["history_coverage"])) > 1e-12:
            raise ValueError(f"semantic coverage mismatch: {year}")
        fold_semantic = bool(
            coverage
            >= float(
                prereg["semantic_gate"][
                    "minimum_history_label_coverage_each_year"
                ]
            )
            and agreement
            >= float(
                prereg["semantic_gate"][
                    "minimum_trackman_group_agreement_each_year"
                ]
            )
            and bool(details["group_physics_only"])
        )
        semantic_pass &= fold_semantic
        folds[year] = {
            "candidate": candidate,
            "parent": parent_artifact["catboost_outcome"].astype(np.float64),
            "regular": regular,
            "masks": {
                "full": np.ones(len(candidate["y"]), dtype=bool),
                "R": regular,
            },
            "paths": {"parent": parent_path, "candidate": candidate_path},
            "semantic": {
                "history_dense_label_coverage": coverage,
                "history_trackman_group_agreement": agreement,
                "history_trackman_comparison_rows": int(
                    audited["history_trackman_comparison_rows"]
                ),
                "group_physics_only": bool(details["group_physics_only"]),
                "selector_top1_accuracy": float(
                    details["diagnostic_selector_top1_accuracy"]
                ),
                "selector_log_loss": float(
                    details["diagnostic_selector_log_loss"]
                ),
                "expert_feature_counts": {
                    group: int(len(columns))
                    for group, columns in details[
                        "expert_feature_columns"
                    ].items()
                },
                "semantic_gate_pass": fold_semantic,
            },
        }
    if not semantic_pass:
        report = {
            "experiment_id": prereg["experiment_id"],
            "status": "failed_semantic_gate",
            "preregister_sha256": digest(PREREG),
            "semantic": {str(year): folds[year]["semantic"] for year in YEARS},
            "control_metrics_computed": False,
            "years_not_read": [2022, 2023, 2024],
        }
        REPORT.write_text(
            json.dumps(safe(report), ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(json.dumps(safe(report), ensure_ascii=False, indent=2))
        return

    iterations = int(prereg["bootstrap_iterations"])
    trials: list[dict[str, Any]] = []
    prediction_cache: dict[tuple[int, float], np.ndarray] = {}
    for gamma in prereg["candidate"]["top_level_blend_grid"]:
        years: dict[str, Any] = {}
        for year in YEARS:
            fold = folds[year]
            evaluated = evaluate(
                fold["candidate"],
                fold["parent"],
                fold["candidate"][KEY].astype(np.float64),
                fold["regular"],
                fold["masks"],
                float(gamma),
                iterations,
                1110000 + 10000 * year + int(float(gamma) * 100),
            )
            prediction = fold["parent"].copy()
            route = fold["regular"]
            prediction[route] += float(gamma) * (
                fold["candidate"][KEY][route] - fold["parent"][route]
            )
            prediction_cache[(year, float(gamma))] = np.clip(
                prediction, 1e-6, 1.0 - 1e-6
            )
            years[str(year)] = evaluated
        full_gains = [
            years[str(year)]["routes"]["full"]["gain"] for year in YEARS
        ]
        r_gains = [
            years[str(year)]["routes"]["R"]["gain"] for year in YEARS
        ]
        trials.append(
            {
                "gamma": float(gamma),
                "minimum_full_gain": float(min(full_gains)),
                "minimum_R_gain": float(min(r_gains)),
                "mean_full_gain": float(np.mean(full_gains)),
                "years": years,
            }
        )
    selected = max(
        trials,
        key=lambda item: (
            item["minimum_full_gain"], item["minimum_R_gain"], -item["gamma"]
        ),
    )
    minimum_full = float(prereg["source_gate"]["minimum_full_gain_each_year"])
    minimum_r = float(prereg["source_gate"]["minimum_r_gain_each_year"])
    checks: list[bool] = []
    for year in YEARS:
        routes = selected["years"][str(year)]["routes"]
        checks.extend(
            (
                routes["full"]["gain"] >= minimum_full,
                routes["R"]["gain"] >= minimum_r,
                routes["full"]["pitcher_cluster_95_ci"]["ci_low"] > 0.0,
                routes["R"]["pitcher_cluster_95_ci"]["ci_low"] > 0.0,
            )
        )
    passed = bool(all(checks))
    artifacts: dict[str, Any] = {}
    for year in YEARS:
        output = PRED / f"v5_dense_physics_pitchtype_moe_source_{year}.npz"
        if output.exists():
            raise FileExistsError(f"immutable artifact exists: {output}")
        fold = folds[year]
        np.savez_compressed(
            output,
            y=fold["candidate"]["y"].astype(np.int8),
            row_index=fold["candidate"]["row_index"].astype(np.int64),
            cluster=fold["candidate"]["cluster"],
            parent_exact_c=fold["parent"],
            dense_physics_moe_raw=fold["candidate"][KEY].astype(np.float64),
            final_prediction=prediction_cache[(year, selected["gamma"])],
        )
        artifacts[str(year)] = {
            "path": str(output.relative_to(ROOT)),
            "sha256": digest(output),
            "parent": str(fold["paths"]["parent"].relative_to(ROOT)),
            "raw_candidate": str(
                fold["paths"]["candidate"].relative_to(ROOT)
            ),
        }

    oracle: dict[str, Any] = {}
    for year in YEARS:
        fold = folds[year]
        available = fold["candidate"][
            f"{PREFIX}diagnostic_true_group_available"
        ].astype(bool)
        oracle[str(year)] = evaluate(
            fold["candidate"],
            fold["parent"],
            fold["candidate"][f"{PREFIX}diagnostic_true_group_oracle"].astype(
                np.float64
            ),
            fold["regular"] & available,
            fold["masks"],
            1.0,
            iterations,
            1210000 + 10000 * year,
        )
        oracle[str(year)]["goal_gate_eligible"] = False
        oracle[str(year)]["exclusion_reason"] = (
            "uses next-row reconstructed current validation pitch group"
        )
    report = {
        "experiment_id": prereg["experiment_id"],
        "status": "source_pass" if passed else "source_failed",
        "preregister_sha256": digest(PREREG),
        "script_sha256": digest(Path(__file__)),
        "semantic_source_sha256": digest(SEMANTIC_SOURCE),
        "years_read": list(YEARS),
        "years_not_read": [2022, 2023, 2024],
        "semantic": {str(year): folds[year]["semantic"] for year in YEARS},
        "trials": trials,
        "selected": selected,
        "diagnostic_true_group_oracle_excluded_from_goal_gate": oracle,
        "artifacts": artifacts,
        "source_gate": {
            "minimum_full_gain_each_year": minimum_full,
            "minimum_R_gain_each_year": minimum_r,
            "ci_lower_positive_each_year": True,
            "passed": passed,
            "decision": (
                "freeze and advance to 2022/2023"
                if passed
                else "close without reading 2022+ candidate labels"
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
                "selected_gamma": selected["gamma"],
                "minimum_full_gain": selected["minimum_full_gain"],
                "minimum_R_gain": selected["minimum_R_gain"],
                "per_year": {
                    str(year): {
                        route: {
                            "gain": selected["years"][str(year)]["routes"][route]["gain"],
                            "ci_low": selected["years"][str(year)]["routes"][route][
                                "pitcher_cluster_95_ci"
                            ]["ci_low"],
                        }
                        for route in ("full", "R")
                    }
                    for year in YEARS
                },
                "selector": {
                    str(year): folds[year]["semantic"] for year in YEARS
                },
                "oracle_full_gain": {
                    str(year): oracle[str(year)]["routes"]["full"]["gain"]
                    for year in YEARS
                },
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
