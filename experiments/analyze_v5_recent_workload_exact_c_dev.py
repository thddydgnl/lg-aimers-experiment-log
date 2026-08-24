#!/usr/bin/env python3
"""Immutable 2022/2023 development gate for the locked workload-decoder C recipe."""

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

from experiments.run_v5_h1_residual import cluster_bootstrap_score_gain  # noqa: E402


RESULTS = ROOT / "experiments/results"
PREDICTIONS = RESULTS / "predictions"
TRAIN = ROOT / "open/data/train.csv"
PREREG = ROOT / "experiments/params/v5_recent_workload_exact_c_preregister.json"
SOURCE_LOCK = ROOT / "experiments/params/v5_recent_workload_decoder_lock.json"
SOURCE_REPORT = RESULTS / "v5_recent_workload_decoder_source.json"
CONTRACT = ROOT / "experiments/params/v5_validation_contract_v2.json"
STAGE_REPORT = RESULTS / "v5_recent_workload_exact_c_dev2223.json"
REPORT = RESULTS / "v5_recent_workload_exact_c_dev_gate.json"

YEARS = (2022, 2023)
STAGE = "v5_recent_workload_exact_c_dev2223"
MODEL_KEY = "catboost_outcome"
FEATURES = [
    "base", "e14", "platoon", "hand_matchup", "e14_hand_cells",
    "e14_count_cells", "e14_type_count_cells", "trackman_rich",
    "batter_e14", "batter_middle_e14", "recent_denominators",
    "recent_workload_decoder",
]
ANCHORS = {
    "exact_c": ("v3_sparse_c_backtest_{year}.npz", "catboost_outcome"),
    "lower_only_c": (
        "v5_recent_denominator_c_dev2223_{year}.npz", "catboost_outcome"
    ),
    "honest_identity": ("v5_honest_m3_r_identity_{year}.npz", "final_prediction"),
    "honest_grid": ("v5_honest_m3_r_grid_{year}.npz", "final_prediction"),
}


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        return {key: np.asarray(archive[key]) for key in archive.files}


def safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return safe(value.tolist())
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return value


def score(y: np.ndarray, prediction: np.ndarray, mask: np.ndarray) -> dict[str, float]:
    target = y[mask].astype(np.float64)
    estimate = prediction[mask].astype(np.float64)
    rate = float(target.mean())
    reference = max(rate * (1.0 - rate), 1e-12)
    brier = float(np.mean(np.square(estimate - target)))
    return {
        "rows": int(mask.sum()),
        "target_rate": rate,
        "prediction_mean": float(estimate.mean()),
        "prediction_std": float(estimate.std()),
        "brier": brier,
        "score": float(100000.0 * (1.0 - brier / reference)),
    }


def candidate_prediction(
    parent: np.ndarray,
    decoder: np.ndarray,
    regular: np.ndarray,
    gamma: float,
) -> np.ndarray:
    result = parent.astype(np.float64).copy()
    result[regular] = np.clip(
        parent[regular] + gamma * (decoder[regular] - parent[regular]),
        1e-6,
        1.0 - 1e-6,
    )
    return result


def audit_source() -> dict[str, Any]:
    source = json.loads(SOURCE_REPORT.read_text(encoding="utf-8"))
    lock = json.loads(SOURCE_LOCK.read_text(encoding="utf-8"))
    checks = {
        "source_pass": source["status"] == "source_pass",
        "source_gate_pass": bool(source["source_gate_pass"]),
        "source_years_only": source["years_read"] == [2020, 2021],
        "development_and_confirmation_unread": source["years_not_read"] == [2022, 2023, 2024],
        "no_control_success": int(source["control_success_columns_read"]) == 0,
        "no_test": int(source["test_rows_read"]) == 0,
        "lock_requires_source_pass": lock["source_status_required"] == "source_pass",
        "lock_no_control_success": bool(lock["no_control_success_used_for_lock"]),
    }
    return {
        "checks": checks,
        "pass": all(checks.values()),
        "source_sha256": digest(SOURCE_REPORT),
        "lock_sha256": digest(SOURCE_LOCK),
    }


def audit_stage() -> dict[str, Any]:
    report = json.loads(STAGE_REPORT.read_text(encoding="utf-8"))
    metadata = report["metadata"]
    checks: dict[str, bool] = {
        "stage": metadata["stage"] == STAGE,
        "single_locked_model": metadata["models"] == [MODEL_KEY],
        "locked_features": metadata["features"] == FEATURES,
        "years": metadata["validation_seasons"] == list(YEARS),
        "inner_validation_none": metadata["inner_validation"] == "none",
        "outcome_scheme": metadata["outcome_scheme"] == "reverse_any",
        "row_independent": bool(metadata["row_independent_inference"]),
        "gpu": metadata["booster_device"] == "gpu",
        "not_smoke": not bool(metadata.get("smoke_test", False)),
    }
    folds: dict[str, Any] = {}
    seen_years: list[int] = []
    for fold in report["folds"]:
        year = int(fold["validation_season"])
        seen_years.append(year)
        decoder = fold["recent_workload_decoder"]
        fold_checks = {
            "decoder_enabled": bool(decoder["enabled"]),
            "outer_year": int(decoder["outer_year"]) == year,
            "row_order_duplicate_invariance": bool(
                decoder["row_order_duplicate_invariance"]
            ),
            "row_order_duplicate_exact": float(
                decoder["row_order_duplicate_max_abs_difference"]
            ) == 0.0,
            "no_control_success": not bool(decoder["control_success_used_by_decoder"]),
            "no_other_validation_rows": not bool(
                decoder["other_validation_rows_used"]
            ),
            "lower_decoder_enabled": bool(fold["recent_denominators"]["enabled"]),
            "lower_decoder_row_local": bool(
                fold["recent_denominators"]["current_row_only"]
            ),
            "pitcher_id_dropped": "pitcher_id" in fold["dropped_features"],
        }
        folds[str(year)] = {
            "checks": fold_checks,
            "pass": all(fold_checks.values()),
            "history_appearances": int(decoder["history_appearances"]),
            "feature_count": len(decoder["feature_columns"]),
        }
    checks["fold_years"] = seen_years == list(YEARS)
    checks["all_fold_audits"] = all(item["pass"] for item in folds.values())
    return {
        "checks": checks,
        "folds": folds,
        "pass": all(checks.values()),
        "stage_sha256": digest(STAGE_REPORT),
        "command": metadata["command"],
    }


def aligned_inputs() -> tuple[dict[int, dict[str, Any]], pd.DataFrame, dict[str, Any]]:
    years: dict[int, dict[str, Any]] = {}
    hashes: dict[str, Any] = {}
    max_row_index = -1
    for year in YEARS:
        raw_path = PREDICTIONS / f"{STAGE}_{year}.npz"
        raw = load_npz(raw_path)
        anchors: dict[str, dict[str, np.ndarray]] = {}
        year_hashes = {"decoder_c": digest(raw_path)}
        for name, (template, _) in ANCHORS.items():
            path = PREDICTIONS / template.format(year=year)
            anchors[name] = load_npz(path)
            year_hashes[name] = digest(path)
        reference = anchors["exact_c"]
        for group, artifacts in (("decoder", {"decoder_c": raw}), ("anchor", anchors)):
            for name, artifact in artifacts.items():
                for key in ("y", "row_index", "cluster"):
                    if not np.array_equal(reference[key], artifact[key]):
                        raise ValueError(
                            f"alignment mismatch {year}/{group}/{name}/{key}"
                        )
        indices = reference["row_index"].astype(np.int64)
        max_row_index = max(max_row_index, int(indices.max()))
        years[year] = {"decoder": raw, "anchors": anchors, "indices": indices}
        hashes[str(year)] = year_hashes

    # The CSV is chronological. Reading exactly through the largest 2023 artifact
    # index prevents even parsing a 2024 confirmation label during development.
    frame = pd.read_csv(
        TRAIN,
        usecols=["season", "game_type", "control_success"],
        nrows=max_row_index + 1,
    )
    if int(frame.index.max()) != max_row_index or int(frame["season"].max()) > 2023:
        raise ValueError("development loader crossed the locked 2023 boundary")
    return years, frame, hashes


def point_comparison(
    y: np.ndarray,
    anchor: np.ndarray,
    candidate: np.ndarray,
    masks: dict[str, np.ndarray],
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for name, mask in masks.items():
        base = score(y, anchor, mask)
        trial = score(y, candidate, mask)
        result[name] = {
            "anchor": base,
            "candidate": trial,
            "gain": float(trial["score"] - base["score"]),
        }
    return result


def main() -> None:
    if REPORT.exists():
        raise FileExistsError(f"immutable report exists: {REPORT}")
    prereg = json.loads(PREREG.read_text(encoding="utf-8"))
    contract = json.loads(CONTRACT.read_text(encoding="utf-8"))
    source_audit = audit_source()
    stage_audit = audit_stage()
    inputs, frame, input_hashes = aligned_inputs()

    gamma_grid = [float(value) for value in prereg["candidate"]["blend_grid"]]
    if gamma_grid != [0.25, 0.5, 0.75, 1.0]:
        raise ValueError("locked gamma grid changed")
    point_results: dict[str, Any] = {}
    materialized: dict[tuple[int, float], np.ndarray] = {}
    masks_by_year: dict[int, dict[str, np.ndarray]] = {}
    for gamma in gamma_grid:
        gamma_full_gains: list[float] = []
        gamma_r_gains: list[float] = []
        years_result: dict[str, Any] = {}
        for year in YEARS:
            item = inputs[year]
            reference = item["anchors"]["exact_c"]
            rows = frame.loc[item["indices"]]
            if not rows["season"].eq(year).all():
                raise ValueError(f"season mismatch in {year}")
            if not np.array_equal(
                rows["control_success"].to_numpy(dtype=np.int8),
                reference["y"].astype(np.int8),
            ):
                raise ValueError(f"target mismatch in {year}")
            regular = rows["game_type"].astype(str).eq("R").to_numpy()
            masks = {
                "full": np.ones(len(rows), dtype=bool),
                "R": regular,
                "F": ~regular,
            }
            masks_by_year[year] = masks
            parent = reference["catboost_outcome"].astype(np.float64)
            decoder = item["decoder"][MODEL_KEY].astype(np.float64)
            candidate = candidate_prediction(parent, decoder, regular, gamma)
            materialized[(year, gamma)] = candidate
            if not np.array_equal(candidate[~regular], parent[~regular]):
                raise AssertionError(f"F changed for gamma={gamma}, year={year}")
            comparisons: dict[str, Any] = {}
            for anchor_name, (_, key) in ANCHORS.items():
                anchor = item["anchors"][anchor_name][key].astype(np.float64)
                comparison = point_comparison(
                    reference["y"], anchor, candidate, masks
                )
                comparisons[anchor_name] = comparison
                gamma_full_gains.append(float(comparison["full"]["gain"]))
                gamma_r_gains.append(float(comparison["R"]["gain"]))
            years_result[str(year)] = {"comparisons": comparisons}
        point_results[str(gamma)] = {
            "years": years_result,
            "objective": {
                "minimum_full_gain": float(min(gamma_full_gains)),
                "minimum_R_gain": float(min(gamma_r_gains)),
            },
        }

    selected_gamma = max(
        gamma_grid,
        key=lambda gamma: (
            point_results[str(gamma)]["objective"]["minimum_full_gain"],
            point_results[str(gamma)]["objective"]["minimum_R_gain"],
            -gamma,
        ),
    )
    selected: dict[str, Any] = {"gamma": selected_gamma, "years": {}}
    all_full_ci_positive = True
    lower_only_r_positive = True
    lower_only_r_ci_positive = True
    f_exact = True
    selected_full_gains: list[float] = []
    output_artifacts: dict[str, Any] = {}
    for year_index, year in enumerate(YEARS):
        item = inputs[year]
        reference = item["anchors"]["exact_c"]
        parent = reference["catboost_outcome"].astype(np.float64)
        decoder = item["decoder"][MODEL_KEY].astype(np.float64)
        candidate = materialized[(year, selected_gamma)]
        masks = masks_by_year[year]
        f_exact = f_exact and np.array_equal(candidate[masks["F"]], parent[masks["F"]])
        comparisons: dict[str, Any] = {}
        for anchor_index, (anchor_name, (_, key)) in enumerate(ANCHORS.items()):
            anchor = item["anchors"][anchor_name][key].astype(np.float64)
            routes = point_comparison(reference["y"], anchor, candidate, masks)
            for route_index, route in enumerate(("full", "R")):
                interval = cluster_bootstrap_score_gain(
                    reference["y"], anchor, candidate, reference["cluster"],
                    masks[route], iterations=int(prereg["development"]["bootstrap_iterations"]),
                    seed=10100000 + 10000 * year_index + 100 * anchor_index + route_index,
                )
                routes[route]["pitcher_cluster_95_ci"] = interval
            comparisons[anchor_name] = routes
            selected_full_gains.append(float(routes["full"]["gain"]))
            all_full_ci_positive = all_full_ci_positive and (
                float(routes["full"]["pitcher_cluster_95_ci"]["ci_low"]) > 0.0
            )
        ablation = comparisons["lower_only_c"]["R"]
        lower_only_r_positive = lower_only_r_positive and float(ablation["gain"]) > 0.0
        lower_only_r_ci_positive = lower_only_r_ci_positive and (
            float(ablation["pitcher_cluster_95_ci"]["ci_low"]) > 0.0
        )
        selected["years"][str(year)] = {
            "rows": int(len(reference["y"])),
            "R_rows": int(masks["R"].sum()),
            "F_rows": int(masks["F"].sum()),
            "comparisons": comparisons,
        }
        output = PREDICTIONS / f"v5_recent_workload_exact_c_dev_selected_{year}.npz"
        if output.exists():
            raise FileExistsError(f"immutable artifact exists: {output}")
        np.savez_compressed(
            output,
            y=reference["y"].astype(np.int8),
            row_index=item["indices"],
            cluster=reference["cluster"],
            exact_c=parent,
            decoder_c=decoder,
            final_prediction=candidate,
            gamma=np.asarray([selected_gamma], dtype=np.float64),
        )
        output_artifacts[str(year)] = {
            "path": str(output.relative_to(ROOT)),
            "sha256": digest(output),
        }

    g_dev = float(min(selected_full_gains))
    threshold = float(
        prereg["development"]["required"][
            "G_dev_minimum_full_gain_strictly_greater_than"
        ]
    )
    gates = {
        "source_audit": source_audit["pass"],
        "stage_audit": stage_audit["pass"],
        "F_preserved_exactly": f_exact,
        "decoder_row_order_and_duplicate_invariance": all(
            fold["checks"]["row_order_duplicate_invariance"]
            for fold in stage_audit["folds"].values()
        ),
        "same_parent_lower_only_R_point_each_year_positive": lower_only_r_positive,
        "same_parent_lower_only_R_ci_lower_each_year_positive": lower_only_r_ci_positive,
        "full_ci_lower_against_every_anchor_each_year_positive": all_full_ci_positive,
        "G_dev": {
            "minimum_full_gain": g_dev,
            "required_strictly_greater_than": threshold,
            "pass": g_dev > threshold,
        },
    }
    passed = bool(
        gates["source_audit"]
        and gates["stage_audit"]
        and gates["F_preserved_exactly"]
        and gates["decoder_row_order_and_duplicate_invariance"]
        and gates["same_parent_lower_only_R_point_each_year_positive"]
        and gates["same_parent_lower_only_R_ci_lower_each_year_positive"]
        and gates["full_ci_lower_against_every_anchor_each_year_positive"]
        and gates["G_dev"]["pass"]
    )
    gates["development_pass"] = passed
    report = {
        "experiment_id": prereg["experiment_id"],
        "status": "development_passed" if passed else "development_failed",
        "preregister_sha256": digest(PREREG),
        "contract_sha256": digest(CONTRACT),
        "source_audit": source_audit,
        "stage_audit": stage_audit,
        "years_read": list(YEARS),
        "confirmation_2024_read": False,
        "selection_rule": prereg["candidate"]["selection"],
        "point_grid": point_results,
        "selected": selected,
        "gates": gates,
        "input_sha256": input_hashes,
        "artifacts": output_artifacts,
        "confirmation_2024_authorized": passed,
        "goal_status": "active",
        "goal_completion_claimed": False,
    }
    REPORT.write_text(
        json.dumps(safe(report), ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(safe({
        "status": report["status"],
        "selected_gamma": selected_gamma,
        "gates": gates,
        "same_parent_R": {
            str(year): selected["years"][str(year)]["comparisons"]["lower_only_c"]["R"]
            for year in YEARS
        },
    }), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
