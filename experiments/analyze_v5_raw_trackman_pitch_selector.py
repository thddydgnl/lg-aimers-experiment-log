#!/usr/bin/env python3
"""Source-only raw-TrackMan fine-pitch selector audit."""

from __future__ import annotations

import gc
import hashlib
import json
import os
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from eda.run_structural_eda import linkage_section, load_trackman, load_train  # noqa: E402
from experiments.analyze_v4_pitchtype_failure_prior import (  # noqa: E402
    normalize_fine_pitch_type,
)
from experiments.analyze_v5_expanded_matchup_pitch_selector import (  # noqa: E402
    blend,
    selector_metrics,
)
from experiments.analyze_v5_fine_pitchtype_latent import FINE_TYPES  # noqa: E402
from experiments.v5_expanded_trackman_profiles import (  # noqa: E402
    _best_map,
    build_expanded_trackman_profile_source,
)

PREREG = ROOT / "experiments/params/v5_raw_trackman_pitch_selector_preregister.json"
LOCK = ROOT / "experiments/params/v5_expanded_matchup_pitch_selector_lock.json"
REPORT = ROOT / "experiments/results/v5_raw_trackman_pitch_selector_source.json"
PREDICTIONS = ROOT / "experiments/results/predictions"
YEARS = (2020, 2021)


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [safe(item) for item in value]
    if isinstance(value, (np.floating, float)):
        number = float(value)
        return number if np.isfinite(number) else None
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    return value


def mode_map(
    rows: pd.DataFrame, left: str, right: str, minimum_purity: float
) -> tuple[dict[Any, Any], dict[str, Any]]:
    """Recover a deterministic many-to-one-safe main-to-TrackMan mapping."""
    return _best_map(rows, left, right, minimum_purity, False)


def text(values: pd.Series) -> pd.Series:
    return values.astype("string").fillna("__unknown__").astype(str)


def build_features(frame: pd.DataFrame, raw: bool) -> tuple[pd.DataFrame, list[str]]:
    numeric = [
        "game_month", "game_dayofweek", "inning", "balls_before",
        "strikes_before", "outs_before",
    ]
    output = frame[numeric].copy()
    output["pitcher_tm"] = text(frame["pitcher_tm"])
    output["batter_tm"] = text(frame["batter_tm"])
    output["pitcher_team_code"] = text(frame["pitcher_team_code"])
    output["batter_team_code"] = text(frame["batter_team_code"])
    output["batter_hand"] = text(frame["batter_hand"])
    top = text(frame["top_bottom"])
    if raw:
        top = top.str.slice(0, 1).str.upper()
    output["top_bottom"] = top
    count = text(frame["balls_before"]) + "-" + text(frame["strikes_before"])
    output["count_state"] = count
    output["hand_matchup"] = text(frame["batter_hand"]) + "|" + text(frame["top_bottom"])
    output["pitcher_count"] = output["pitcher_tm"] + "|" + count
    output["pitcher_hand_count"] = (
        output["pitcher_tm"] + "|" + output["batter_hand"] + "|" + count
    )
    output["pitcher_batter"] = output["pitcher_tm"] + "|" + output["batter_tm"]
    output["pitcher_batter_count"] = output["pitcher_batter"] + "|" + count
    output["batter_count"] = output["batter_tm"] + "|" + count
    output["team_matchup"] = (
        output["pitcher_team_code"] + "|" + output["batter_team_code"]
    )
    output["pitcher_inning"] = output["pitcher_tm"] + "|" + text(frame["inning"])
    categorical = [column for column in output.columns if column not in numeric]
    return output, categorical


def align_classes(raw_probability: np.ndarray, classes: list[str]) -> np.ndarray:
    result = np.zeros((len(raw_probability), len(FINE_TYPES)), dtype=np.float64)
    for source_index, label in enumerate(classes):
        if label in FINE_TYPES:
            result[:, FINE_TYPES.index(label)] = raw_probability[:, source_index]
    denominator = result.sum(axis=1)
    missing = denominator <= 0.0
    result[missing] = 1.0 / len(FINE_TYPES)
    denominator[missing] = 1.0
    result /= denominator[:, None]
    return result


def main() -> None:
    if REPORT.exists():
        raise FileExistsError(f"immutable report exists: {REPORT}")
    prereg = json.loads(PREREG.read_text(encoding="utf-8"))
    lock = json.loads(LOCK.read_text(encoding="utf-8"))
    if prereg["status"] != "locked_before_source_selector_metrics":
        raise ValueError("unexpected preregistration state")
    if lock["selected_recipe"]["candidate_id"] != "geometric_count_hand__cb0.5":
        raise ValueError("locked parent selector changed")

    started = time.perf_counter()
    train = load_train()
    trackman, game_ids, _ = load_trackman()
    linkage_meta, joined = linkage_section(train, trackman, len(game_ids))
    labels = (
        joined[["row_id", "auto_pitch_type"]]
        .drop_duplicates("row_id")
        .assign(fine_auto=lambda value: normalize_fine_pitch_type(value["auto_pitch_type"]))
        .set_index("row_id")["fine_auto"]
    )
    train["fine_auto"] = train["row_id"].map(labels)

    params = prereg["raw_model"]
    folds: dict[int, dict[str, Any]] = {}
    candidate_probabilities: dict[tuple[int, float], np.ndarray] = {}
    metadata: dict[str, Any] = {}
    for year in YEARS:
        valid = train.loc[
            train["season"].eq(year) & train["game_type"].eq("R")
        ].copy()
        exact = joined.loc[
            joined["season"].lt(year) & joined["game_type"].eq("R")
        ].copy()
        allowed = sorted(int(value) for value in exact["season"].unique())
        major, major_meta = build_expanded_trackman_profile_source(
            joined,
            trackman,
            allowed,
            float(params["identity_minimum_purity"]),
        )
        pitcher_map, pitcher_meta = _best_map(
            exact, "pitcher_id", "pitcher_trackman_id",
            float(params["identity_minimum_purity"]), True,
        )
        batter_map, batter_meta = _best_map(
            exact, "batter_id", "batter_trackman_id",
            float(params["identity_minimum_purity"]), True,
        )
        pitcher_team_map, pitcher_team_meta = mode_map(
            exact, "pitcher_team_id", "pitcher_team",
            float(params["identity_minimum_purity"]),
        )
        batter_team_map, batter_team_meta = mode_map(
            exact, "batter_team_id", "batter_team",
            float(params["identity_minimum_purity"]),
        )

        raw_frame = major.copy()
        raw_frame["pitcher_tm"] = raw_frame["pitcher_trackman_id"]
        raw_frame["batter_tm"] = raw_frame["batter_trackman_id"]
        raw_frame["pitcher_team_code"] = raw_frame["pitcher_team"]
        raw_frame["batter_team_code"] = raw_frame["batter_team"]
        raw_frame["fine_auto"] = normalize_fine_pitch_type(raw_frame["auto_pitch_type"])
        raw_frame = raw_frame.loc[raw_frame["fine_auto"].notna()].copy()

        valid_frame = valid.copy()
        valid_frame["pitcher_tm"] = valid_frame["pitcher_id"].map(pitcher_map)
        valid_frame["batter_tm"] = valid_frame["batter_id"].map(batter_map)
        valid_frame["pitcher_team_code"] = valid_frame["pitcher_team_id"].map(
            pitcher_team_map
        )
        valid_frame["batter_team_code"] = valid_frame["batter_team_id"].map(
            batter_team_map
        )

        raw_x, categorical = build_features(raw_frame, True)
        valid_x, valid_categorical = build_features(valid_frame, False)
        if categorical != valid_categorical:
            raise AssertionError("raw/main selector schema mismatch")
        model = CatBoostClassifier(
            loss_function=str(params["loss_function"]),
            iterations=int(params["iterations"]),
            depth=int(params["depth"]),
            learning_rate=float(params["learning_rate"]),
            l2_leaf_reg=float(params["l2_leaf_reg"]),
            random_seed=int(params["random_seed"]) + year,
            allow_writing_files=bool(params["allow_writing_files"]),
            thread_count=6,
            task_type=(
                "GPU"
                if os.environ.get("V2_BOOSTER_DEVICE", "cpu").lower() == "gpu"
                else "CPU"
            ),
        )
        fit_started = time.perf_counter()
        model.fit(
            raw_x,
            raw_frame["fine_auto"].astype(str),
            cat_features=categorical,
            verbose=False,
        )
        raw_probability = align_classes(
            np.asarray(model.predict_proba(valid_x), dtype=np.float64),
            [str(value) for value in model.classes_],
        )
        fit_seconds = float(time.perf_counter() - fit_started)

        expert_path = PREDICTIONS / f"v5_matchup_hand_fine_moe_source_{year}.npz"
        with np.load(expert_path, allow_pickle=False) as archive:
            row_index = np.asarray(archive["row_index"], dtype=np.int64)
            selector_all = np.column_stack([
                np.asarray(
                    archive[f"catboost_fine_pitch_moe__selector_{label.lower()}"],
                    dtype=np.float64,
                )
                for label in FINE_TYPES
            ])
        season_index = train.index[train["season"].eq(year)].to_numpy(dtype=np.int64)
        if not np.array_equal(row_index, season_index):
            raise ValueError(f"{year}: frozen expert row order mismatch")
        regular_mask = train.loc[row_index, "game_type"].eq("R").to_numpy(dtype=bool)
        locked_probability = selector_all[regular_mask]
        if not np.array_equal(row_index[regular_mask], valid.index.to_numpy(dtype=np.int64)):
            raise ValueError(f"{year}: R row order mismatch")

        matched = valid["fine_auto"].notna().to_numpy(dtype=bool)
        truth = valid.loc[matched, "fine_auto"].astype(str).to_numpy()
        truth_index = np.asarray(
            [FINE_TYPES.index(value) for value in truth], dtype=np.int16
        )
        locked_metrics = selector_metrics(locked_probability[matched], truth_index)
        raw_metrics = selector_metrics(raw_probability[matched], truth_index)
        folds[year] = {
            "truth_index": truth_index,
            "matched": matched,
            "locked_metrics": locked_metrics,
            "raw_metrics": raw_metrics,
        }
        for weight in prereg["source_protocol"]["raw_model_weight_grid"]:
            candidate_probabilities[(year, float(weight))] = blend(
                raw_probability, locked_probability, float(weight)
            )
        metadata[str(year)] = {
            "allowed_history_seasons": allowed,
            "raw_training_rows": int(len(raw_frame)),
            "valid_R_rows": int(len(valid)),
            "matched_selector_rows": int(matched.sum()),
            "mapped_validation_pitcher_fraction": float(
                valid_frame["pitcher_tm"].notna().mean()
            ),
            "mapped_validation_batter_fraction": float(
                valid_frame["batter_tm"].notna().mean()
            ),
            "fit_seconds": fit_seconds,
            "classes": [str(value) for value in model.classes_],
            "expanded_source": major_meta,
            "identity": {
                "pitcher": pitcher_meta,
                "batter": batter_meta,
                "pitcher_team": pitcher_team_meta,
                "batter_team": batter_team_meta,
            },
        }
        del model, raw_x, valid_x, raw_frame, valid_frame, major, exact
        gc.collect()

    trials: list[dict[str, Any]] = []
    for raw_weight in prereg["source_protocol"]["raw_model_weight_grid"]:
        weight = float(raw_weight)
        years: dict[str, Any] = {}
        for year in YEARS:
            fold = folds[year]
            metrics = selector_metrics(
                candidate_probabilities[(year, weight)][fold["matched"]],
                fold["truth_index"],
            )
            locked_metrics = fold["locked_metrics"]
            years[str(year)] = {
                "locked": locked_metrics,
                "raw_only": fold["raw_metrics"],
                "candidate": metrics,
                "log_loss_improvement": float(
                    locked_metrics["log_loss"] - metrics["log_loss"]
                ),
                "top1_improvement": float(
                    metrics["top1_accuracy"] - locked_metrics["top1_accuracy"]
                ),
            }
        trials.append({
            "raw_model_weight": weight,
            "worst_log_loss": float(
                max(years[str(year)]["candidate"]["log_loss"] for year in YEARS)
            ),
            "mean_log_loss": float(np.mean([
                years[str(year)]["candidate"]["log_loss"] for year in YEARS
            ])),
            "minimum_top1_improvement": float(min(
                years[str(year)]["top1_improvement"] for year in YEARS
            )),
            "years": years,
        })
    selected = min(trials, key=lambda item: (
        item["worst_log_loss"], item["mean_log_loss"],
        -item["minimum_top1_improvement"], item["raw_model_weight"],
    ))

    gate = prereg["source_protocol"]["gate"]
    checks: dict[str, bool] = {}
    artifacts: dict[str, str] = {}
    for year in YEARS:
        result = selected["years"][str(year)]
        checks[f"{year}_logloss"] = bool(
            result["log_loss_improvement"]
            >= float(gate["minimum_log_loss_improvement_over_locked_each_year"])
        )
        checks[f"{year}_top1"] = bool(
            result["top1_improvement"]
            >= float(gate["minimum_top1_improvement_over_locked_each_year"])
        )
        checks[f"{year}_raw_rows"] = bool(
            metadata[str(year)]["raw_training_rows"]
            >= int(gate["minimum_raw_training_rows_each_year"])
        )
        checks[f"{year}_pitcher_mapping"] = bool(
            metadata[str(year)]["mapped_validation_pitcher_fraction"]
            >= float(gate["minimum_mapped_validation_pitcher_fraction_each_year"])
        )
        probability = candidate_probabilities[
            (year, float(selected["raw_model_weight"]))
        ]
        path = PREDICTIONS / f"v5_raw_trackman_selector_source_{year}.npz"
        if path.exists():
            raise FileExistsError(f"immutable artifact exists: {path}")
        valid_index = train.index[
            train["season"].eq(year) & train["game_type"].eq("R")
        ].to_numpy(dtype=np.int64)
        np.savez_compressed(
            path,
            row_index=valid_index,
            selected_probability=probability.astype(np.float32),
            raw_model_weight=np.asarray([selected["raw_model_weight"]]),
        )
        artifacts[str(year)] = str(path.relative_to(ROOT))

    passed = bool(all(checks.values()))
    report = {
        "experiment_id": prereg["experiment_id"],
        "status": "selector_source_pass" if passed else "selector_source_failed",
        "preregister_sha256": digest(PREREG),
        "script_sha256": digest(Path(__file__)),
        "parent_lock_sha256": digest(LOCK),
        "years_read": list(YEARS),
        "years_not_read": [2022, 2023, 2024],
        "control_success_read_for_selection": False,
        "test_rows_read": False,
        "linkage": {
            key: linkage_meta[key] for key in (
                "aligned_rows", "unambiguous_one_to_one", "elementwise_state_agreement"
            )
        },
        "metadata": metadata,
        "trial_count": len(trials),
        "selected": selected,
        "trials": trials,
        "gate": {"requirements": gate, "checks": checks, "pass": passed},
        "prediction_artifacts": artifacts,
        "elapsed_seconds": float(time.perf_counter() - started),
        "goal_status": "active",
        "goal_completion_claimed": False,
    }
    REPORT.write_text(
        json.dumps(safe(report), ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(safe({
        "status": report["status"],
        "metadata": metadata,
        "selected": selected,
        "gate": report["gate"],
        "elapsed_seconds": report["elapsed_seconds"],
    }), ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    os.environ.setdefault("V2_BOOSTER_DEVICE", "gpu")
    main()
