#!/usr/bin/env python3
"""Source-only audit of full-TrackMan matchup pitch-selection profiles."""

from __future__ import annotations

import gc
import hashlib
import json
import os
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from eda.run_structural_eda import (  # noqa: E402
    linkage_section,
    load_trackman,
    load_train,
)
from experiments.analyze_v4_pitchtype_failure_prior import (  # noqa: E402
    normalize_fine_pitch_type,
)
from experiments.analyze_v5_fine_pitchtype_latent import (  # noqa: E402
    FINE_TYPES,
    fit_selector,
)
from experiments.v5_expanded_trackman_profiles import (  # noqa: E402
    _best_map,
    build_expanded_trackman_profile_source,
)

PREREG = ROOT / "experiments/params/v5_expanded_matchup_pitch_selector_preregister.json"
REPORT = ROOT / "experiments/results/v5_expanded_matchup_pitch_selector_source.json"
YEARS = (2020, 2021)


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
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


def count_table(rows: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    table = rows.groupby(
        [*keys, "fine_auto"], sort=False, observed=True, dropna=False
    ).size().unstack("fine_auto", fill_value=0)
    return table.reindex(columns=FINE_TYPES, fill_value=0).astype(np.float64)


def map_counts(table: pd.DataFrame, query: pd.DataFrame, keys: list[str]) -> np.ndarray:
    if len(keys) == 1:
        index = pd.Index(query[keys[0]].to_numpy(), name=table.index.name)
    else:
        index = pd.MultiIndex.from_frame(query[keys])
        index.names = table.index.names
    values = table.reindex(index).to_numpy(dtype=np.float64)
    missing = np.isnan(values).all(axis=1)
    values[missing] = 0.0
    return values


def smooth(counts: np.ndarray, prior: np.ndarray, k: float) -> np.ndarray:
    total = counts.sum(axis=1)
    return (counts + k * prior) / (total[:, None] + k)


def geometric(*probabilities: np.ndarray) -> np.ndarray:
    result = np.mean(
        [np.log(np.clip(value, 1e-12, 1.0)) for value in probabilities],
        axis=0,
    )
    result = np.exp(result)
    return result / result.sum(axis=1, keepdims=True)


def blend(catboost: np.ndarray, profile: np.ndarray, weight: float) -> np.ndarray:
    if weight <= 0.0:
        return profile
    if weight >= 1.0:
        return catboost
    result = np.exp(
        weight * np.log(np.clip(catboost, 1e-12, 1.0))
        + (1.0 - weight) * np.log(np.clip(profile, 1e-12, 1.0))
    )
    return result / result.sum(axis=1, keepdims=True)


def selector_metrics(probability: np.ndarray, truth_index: np.ndarray) -> dict[str, float]:
    chosen = probability[np.arange(len(truth_index)), truth_index]
    return {
        "log_loss": float(-np.mean(np.log(np.clip(chosen, 1e-12, 1.0)))),
        "top1_accuracy": float(np.mean(probability.argmax(axis=1) == truth_index)),
    }


def main() -> None:
    if REPORT.exists():
        raise FileExistsError(f"immutable report exists: {REPORT}")
    prereg = json.loads(PREREG.read_text(encoding="utf-8"))
    train = load_train()
    trackman, game_ids, _ = load_trackman()
    linkage_meta, joined = linkage_section(train, trackman, len(game_ids))
    label_map = (
        joined[["row_id", "auto_pitch_type"]]
        .drop_duplicates("row_id")
        .assign(fine_auto=lambda frame: normalize_fine_pitch_type(frame["auto_pitch_type"]))
        .set_index("row_id")["fine_auto"]
    )
    train["fine_auto"] = train["row_id"].map(label_map)

    smoothing = prereg["fixed_smoothing"]
    folds: dict[int, dict[str, Any]] = {}
    mapping: dict[str, Any] = {}
    for year in YEARS:
        history = train.loc[
            train["season"].lt(year) & train["game_type"].eq("R")
        ].copy()
        valid = train.loc[
            train["season"].eq(year) & train["game_type"].eq("R")
        ].copy()
        baseline_all, baseline_meta = fit_selector(history, valid, "auto", year)
        matched = valid["fine_auto"].notna().to_numpy(dtype=bool)
        truth = valid.loc[matched, "fine_auto"].astype(str).to_numpy()
        truth_index = np.asarray(
            [FINE_TYPES.index(value) for value in truth], dtype=np.int16
        )
        query = valid.loc[matched].copy()
        baseline = baseline_all[matched]

        exact = joined.loc[
            joined["season"].lt(year) & joined["game_type"].eq("R")
        ].copy()
        major, major_meta = build_expanded_trackman_profile_source(
            joined, trackman, sorted(int(v) for v in history["season"].unique()),
            float(prereg["source_protocol"]["minimum_identity_purity"]),
        )
        batter_map, batter_meta = _best_map(
            exact, "batter_id", "batter_trackman_id",
            float(prereg["source_protocol"]["minimum_identity_purity"]), True,
        )
        inverse_batter = {
            trackman_id: int(batter_id)
            for batter_id, trackman_id in batter_map.items()
        }
        major["batter_id"] = major["batter_trackman_id"].map(inverse_batter)
        major["fine_auto"] = normalize_fine_pitch_type(major["auto_pitch_type"])
        major = major.loc[major["fine_auto"].notna()].copy()

        global_counts = (
            major["fine_auto"].value_counts().reindex(FINE_TYPES).fillna(0.0)
            .to_numpy(dtype=np.float64)
        )
        global_prior = global_counts / global_counts.sum()
        global_matrix = np.broadcast_to(global_prior, (len(query), len(FINE_TYPES)))
        pitcher = smooth(
            map_counts(count_table(major, ["pitcher_id"]), query, ["pitcher_id"]),
            global_matrix, float(smoothing["pitcher_k"]),
        )
        pitcher_count = smooth(
            map_counts(
                count_table(major, ["pitcher_id", "balls_before", "strikes_before"]),
                query, ["pitcher_id", "balls_before", "strikes_before"],
            ), pitcher, float(smoothing["pitcher_count_k"]),
        )
        pitcher_hand = smooth(
            map_counts(
                count_table(major, ["pitcher_id", "batter_hand"]),
                query, ["pitcher_id", "batter_hand"],
            ), pitcher, float(smoothing["pitcher_hand_k"]),
        )
        pitcher_hand_count = smooth(
            map_counts(
                count_table(
                    major,
                    ["pitcher_id", "batter_hand", "balls_before", "strikes_before"],
                ),
                query,
                ["pitcher_id", "batter_hand", "balls_before", "strikes_before"],
            ), pitcher_hand, float(smoothing["pitcher_hand_count_k"]),
        )

        mapped_major = major.loc[major["batter_id"].notna()].copy()
        mapped_major["batter_id"] = mapped_major["batter_id"].astype(np.int64)
        pitcher_batter = smooth(
            map_counts(
                count_table(mapped_major, ["pitcher_id", "batter_id"]),
                query, ["pitcher_id", "batter_id"],
            ), pitcher_hand, float(smoothing["pitcher_batter_k"]),
        )
        pitcher_batter_count = smooth(
            map_counts(
                count_table(
                    mapped_major,
                    ["pitcher_id", "batter_id", "balls_before", "strikes_before"],
                ),
                query,
                ["pitcher_id", "batter_id", "balls_before", "strikes_before"],
            ), geometric(pitcher_hand_count, pitcher_batter),
            float(smoothing["pitcher_batter_count_k"]),
        )
        profiles = {
            "pitcher_count": pitcher_count,
            "pitcher_hand_count": pitcher_hand_count,
            "geometric_count_hand": geometric(pitcher_count, pitcher_hand_count),
            "pitcher_batter": pitcher_batter,
            "pitcher_batter_count": pitcher_batter_count,
            "geometric_hand_matchup": geometric(
                pitcher_hand_count, pitcher_batter_count
            ),
        }
        folds[year] = {
            "truth_index": truth_index,
            "baseline": baseline,
            "baseline_metrics": selector_metrics(baseline, truth_index),
            "profiles": profiles,
        }
        mapping[str(year)] = {
            "baseline": baseline_meta,
            "expanded": major_meta,
            "batter_identity": batter_meta,
            "mapped_batter_trackman_rows": int(major["batter_id"].notna().sum()),
            "mapped_batter_fraction": float(major["batter_id"].notna().mean()),
            "matched_validation_rows": int(matched.sum()),
        }
        del history, valid, exact, major, mapped_major, baseline_all
        gc.collect()

    trials: list[dict[str, Any]] = []
    for profile_name in prereg["candidate_profiles"]:
        for catboost_weight in prereg["catboost_geometric_weights"]:
            years: dict[str, Any] = {}
            for year in YEARS:
                fold = folds[year]
                candidate = blend(
                    fold["baseline"], fold["profiles"][profile_name],
                    float(catboost_weight),
                )
                metrics = selector_metrics(candidate, fold["truth_index"])
                baseline_metrics = fold["baseline_metrics"]
                years[str(year)] = {
                    "baseline": baseline_metrics,
                    "candidate": metrics,
                    "log_loss_improvement": float(
                        baseline_metrics["log_loss"] - metrics["log_loss"]
                    ),
                    "top1_improvement": float(
                        metrics["top1_accuracy"] - baseline_metrics["top1_accuracy"]
                    ),
                }
            trials.append({
                "candidate_id": f"{profile_name}__cb{float(catboost_weight):g}",
                "profile": profile_name,
                "catboost_geometric_weight": float(catboost_weight),
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
        -item["minimum_top1_improvement"], item["candidate_id"],
    ))
    gate = prereg["source_protocol"]["gate"]
    checks: dict[str, bool] = {}
    for year in YEARS:
        result = selected["years"][str(year)]
        checks[f"{year}_logloss"] = bool(
            result["log_loss_improvement"]
            >= float(gate["minimum_log_loss_improvement_each_year"])
        )
        checks[f"{year}_top1"] = bool(
            result["top1_improvement"]
            >= float(gate["minimum_top1_improvement_each_year"])
        )
        checks[f"{year}_mapped_batter_rows"] = bool(
            mapping[str(year)]["mapped_batter_trackman_rows"]
            >= int(gate["minimum_mapped_batter_rows_each_year"])
        )
    passed = bool(all(checks.values()))
    report = {
        "experiment_id": prereg["experiment_id"],
        "status": "selector_source_pass" if passed else "selector_source_failed",
        "preregister_sha256": digest(PREREG),
        "script_sha256": digest(Path(__file__)),
        "years_read": list(YEARS),
        "years_not_read": [2022, 2023, 2024],
        "control_success_read_for_selection": False,
        "test_rows_read": False,
        "linkage": {
            key: linkage_meta[key] for key in (
                "aligned_rows", "unambiguous_one_to_one", "elementwise_state_agreement"
            )
        },
        "mapping": mapping,
        "trial_count": len(trials),
        "selected": selected,
        "top_trials": sorted(
            trials,
            key=lambda item: (
                item["worst_log_loss"], item["mean_log_loss"], item["candidate_id"]
            ),
        )[:10],
        "gate": {"requirements": gate, "checks": checks, "pass": passed},
        "goal_status": "active",
        "goal_completion_claimed": False,
    }
    REPORT.write_text(
        json.dumps(safe(report), ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(safe({
        "status": report["status"], "mapping": mapping,
        "selected": selected, "gate": report["gate"],
    }), ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    os.environ.setdefault("V2_BOOSTER_DEVICE", "gpu")
    main()
