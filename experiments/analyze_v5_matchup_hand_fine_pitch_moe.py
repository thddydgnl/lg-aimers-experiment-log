#!/usr/bin/env python3
"""Immutable source gate for the locked hand/count selector fine-pitch MoE."""

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

from experiments.analyze_v5_dense_pitchtype_moe import (  # noqa: E402
    evaluate,
    load,
    safe,
)

RESULTS = ROOT / "experiments/results"
PRED = RESULTS / "predictions"
PREREG = ROOT / "experiments/params/v5_matchup_hand_fine_pitch_moe_preregister.json"
LOCK = ROOT / "experiments/params/v5_expanded_matchup_pitch_selector_lock.json"
SELECTOR_REPORT = RESULTS / "v5_expanded_matchup_pitch_selector_source.json"
REPORT = RESULTS / "v5_matchup_hand_fine_pitch_moe_source_gate.json"
STAGE = "v5_matchup_hand_fine_moe_source"
YEARS = (2020, 2021)
KEY = "catboost_fine_pitch_moe"


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def main() -> None:
    if REPORT.exists():
        raise FileExistsError(f"immutable report exists: {REPORT}")
    prereg = json.loads(PREREG.read_text(encoding="utf-8"))
    lock = json.loads(LOCK.read_text(encoding="utf-8"))
    selector_source = json.loads(SELECTOR_REPORT.read_text(encoding="utf-8"))
    if selector_source["status"] != "selector_source_pass":
        raise ValueError("locked selector source evidence did not pass")
    if digest(SELECTOR_REPORT) != lock["source_evidence"]["report_sha256"]:
        raise ValueError("selector report changed after lock")

    game_types = pd.read_csv(
        ROOT / "open/data/train.csv", usecols=["game_type"]
    )["game_type"].astype(str)
    metadata_path = RESULTS / f"{STAGE}.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    fold_metadata = {
        int(item["validation_season"]): item for item in metadata["folds"]
    }
    folds: dict[int, dict[str, Any]] = {}
    semantic_pass = True
    source_expected = selector_source["selected"]["years"]
    for year in YEARS:
        candidate_path = PRED / f"{STAGE}_{year}.npz"
        parent_path = PRED / f"v4_m3_c_backtest_{year}_{year}.npz"
        candidate = load(candidate_path)
        parent_artifact = load(parent_path)
        for key in ("y", "row_index", "cluster"):
            if not np.array_equal(candidate[key], parent_artifact[key]):
                raise ValueError(f"alignment mismatch: {year}/{key}")
        details = fold_metadata[year]["fit_details"][KEY]
        latent = fold_metadata[year]["fine_pitch_latent"]
        expected = source_expected[str(year)]
        observed_baseline = latent["selector_baseline_R"]
        observed_candidate = latent["selector_candidate_R"]
        reproduction_delta = {
            "baseline_log_loss": float(
                observed_baseline["log_loss"] - expected["baseline"]["log_loss"]
            ),
            "candidate_log_loss": float(
                observed_candidate["log_loss"] - expected["candidate"]["log_loss"]
            ),
            "baseline_top1": float(
                observed_baseline["top1_accuracy"]
                - expected["baseline"]["top1_accuracy"]
            ),
            "candidate_top1": float(
                observed_candidate["top1_accuracy"]
                - expected["candidate"]["top1_accuracy"]
            ),
        }
        expert_features = details["expert_feature_columns"]
        semantic = {
            "candidate_id": latent["candidate_id"],
            "history_game_type": latent["history_game_type"],
            "selector_baseline_R": observed_baseline,
            "selector_candidate_R": observed_candidate,
            "selector_reproduction_max_abs": float(
                max(abs(value) for value in reproduction_delta.values())
            ),
            "selector_reproduction_deltas": reproduction_delta,
            "selected_batter_identity_used": bool(
                latent["profile"]["selected_batter_identity_used"]
            ),
            "batter_hand_used": bool(latent["profile"]["batter_hand_used"]),
            "expanded_trackman": latent["expanded_trackman"],
            "training_e92_values": latent["training_e92_values"],
            "training_e92_consumed_by_control_experts": bool(
                latent["training_e92_consumed_by_control_experts"]
            ),
            "expert_count": int(details["expert_count"]),
            "expert_e92_feature_count": int(
                sum(column.startswith("e92_") for column in expert_features)
            ),
            "current_pitch_type_used_at_inference": bool(
                details["current_pitch_type_used_at_inference"]
            ),
            "current_pitch_trackman_used_at_inference": bool(
                details["current_pitch_trackman_used_at_inference"]
            ),
            "row_independent_inference": bool(details["row_independent_inference"]),
        }
        semantic["pass"] = bool(
            semantic["candidate_id"]
            == lock["selected_recipe"]["candidate_id"]
            and semantic["history_game_type"] == "R"
            and semantic["selector_reproduction_max_abs"] <= 1e-5
            and not semantic["selected_batter_identity_used"]
            and semantic["batter_hand_used"]
            and semantic["expanded_trackman"]["profile_aggregation_only"]
            and not semantic["expanded_trackman"]["target_columns_read"]
            and not semantic["training_e92_consumed_by_control_experts"]
            and semantic["expert_count"] == int(prereg["candidate"]["expert_count"])
            and semantic["expert_e92_feature_count"] == 0
            and not semantic["current_pitch_type_used_at_inference"]
            and not semantic["current_pitch_trackman_used_at_inference"]
            and semantic["row_independent_inference"]
        )
        semantic_pass &= semantic["pass"]
        game_type = game_types.iloc[
            candidate["row_index"].astype(np.int64)
        ].to_numpy(dtype=str)
        parent = parent_artifact["catboost_outcome"].astype(np.float64)
        raw = candidate[KEY].astype(np.float64)
        routed_raw = np.where(game_type == "R", raw, parent)
        folds[year] = {
            "candidate": candidate,
            "parent": parent,
            "raw": raw,
            "routed_raw": routed_raw,
            "masks": {"full": np.ones(len(raw), dtype=bool), "R": game_type == "R"},
            "semantic": semantic,
            "paths": {"candidate": candidate_path, "parent": parent_path},
        }

    trials: list[dict[str, Any]] = []
    cache: dict[tuple[int, float], np.ndarray] = {}
    if semantic_pass:
        for gamma in prereg["source_protocol"]["top_level_blend_grid"]:
            years: dict[str, Any] = {}
            for year in YEARS:
                fold = folds[year]
                years[str(year)] = evaluate(
                    fold["candidate"],
                    fold["parent"],
                    fold["routed_raw"],
                    fold["masks"]["full"],
                    fold["masks"],
                    float(gamma),
                    int(prereg["source_protocol"]["bootstrap_iterations"]),
                    7310000 + 10000 * year + int(float(gamma) * 100),
                )
                cache[(year, float(gamma))] = np.clip(
                    fold["parent"]
                    + float(gamma) * (fold["routed_raw"] - fold["parent"]),
                    1e-6,
                    1.0 - 1e-6,
                )
            r_gains = [years[str(year)]["routes"]["R"]["gain"] for year in YEARS]
            full_gains = [
                years[str(year)]["routes"]["full"]["gain"] for year in YEARS
            ]
            trials.append(
                {
                    "gamma": float(gamma),
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
            ),
        )
        if trials
        else None
    )
    gate = prereg["source_protocol"]["gate"]
    checks: dict[str, bool] = {
        "semantic_and_selector_reproduction": semantic_pass,
        "selected": selected is not None,
    }
    if selected is not None:
        for year in YEARS:
            routes = selected["years"][str(year)]["routes"]
            checks[f"{year}_R_gain"] = bool(
                routes["R"]["gain"] >= float(gate["minimum_R_gain_each_year"])
            )
            checks[f"{year}_full_gain"] = bool(
                routes["full"]["gain"] >= float(gate["minimum_full_gain_each_year"])
            )
            checks[f"{year}_R_ci"] = bool(
                routes["R"]["pitcher_cluster_95_ci"]["ci_low"] > 0.0
            )
            checks[f"{year}_full_ci"] = bool(
                routes["full"]["pitcher_cluster_95_ci"]["ci_low"] > 0.0
            )
    passed = bool(all(checks.values()))
    artifacts: dict[str, Any] = {}
    if selected is not None:
        for year in YEARS:
            output = PRED / f"v5_matchup_hand_fine_pitch_moe_source_{year}.npz"
            if output.exists():
                raise FileExistsError(f"immutable artifact exists: {output}")
            fold = folds[year]
            np.savez_compressed(
                output,
                y=fold["candidate"]["y"].astype(np.int8),
                row_index=fold["candidate"]["row_index"].astype(np.int64),
                cluster=fold["candidate"]["cluster"],
                parent_exact_c=fold["parent"],
                matchup_hand_moe_raw=fold["raw"],
                routed_raw=fold["routed_raw"],
                final_prediction=cache[(year, selected["gamma"])],
            )
            artifacts[str(year)] = {
                "path": str(output.relative_to(ROOT)),
                "sha256": digest(output),
            }
    report = {
        "experiment_id": prereg["experiment_id"],
        "status": "source_pass" if passed else "source_failed",
        "selector_lock_sha256": digest(LOCK),
        "selector_report_sha256": digest(SELECTOR_REPORT),
        "preregister_sha256": digest(PREREG),
        "script_sha256": digest(Path(__file__)),
        "years_read": list(YEARS),
        "years_not_read": [2022, 2023, 2024],
        "semantic": {str(year): folds[year]["semantic"] for year in YEARS},
        "input_sha256": {
            str(year): {
                name: digest(path) for name, path in folds[year]["paths"].items()
            }
            for year in YEARS
        },
        "trials": trials,
        "selected": selected,
        "source_gate": {"requirements": gate, "checks": checks, "pass": passed},
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
                    "semantic": report["semantic"],
                    "selected": selected,
                    "checks": checks,
                }
            ),
            ensure_ascii=False,
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
