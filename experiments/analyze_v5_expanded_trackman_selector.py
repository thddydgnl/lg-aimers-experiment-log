#!/usr/bin/env python3
"""Source-only selector audit using all history-mapped official TrackMan rows."""

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
    linkage_section, load_trackman, load_train,
)
from experiments.analyze_v4_pitchtype_failure_prior import (  # noqa: E402
    normalize_fine_pitch_type,
)
from experiments.analyze_v5_fine_pitchtype_latent import (  # noqa: E402
    FINE_TYPES, fit_selector,
)

PREREG = ROOT / "experiments/params/v5_expanded_trackman_selector_preregister.json"
OUTPUT = ROOT / "experiments/results/v5_expanded_trackman_selector_source.json"
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
        numeric = float(value)
        return numeric if np.isfinite(numeric) else None
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    return value


def history_identity_map(
    joined: pd.DataFrame, year: int, minimum_purity: float,
) -> tuple[dict[Any, int], dict[str, Any]]:
    pairs = joined.loc[
        joined["season"].lt(year), ["pitcher_id", "pitcher_trackman_id"]
    ].dropna()
    counts = pairs.groupby(
        ["pitcher_id", "pitcher_trackman_id"], sort=False, observed=True
    ).size().rename("n").reset_index()
    totals = counts.groupby("pitcher_id", observed=True)["n"].transform("sum")
    counts["purity"] = counts["n"] / totals
    best = counts.sort_values(
        ["pitcher_id", "n"], ascending=[True, False], kind="stable"
    ).drop_duplicates("pitcher_id")
    best = best.loc[best["purity"].ge(minimum_purity)].copy()
    # A TrackMan identity may map to only one anonymous pitcher.  Retain the
    # strongest evidence if a rare collision survives the purity filter.
    best = best.sort_values(
        ["pitcher_trackman_id", "n"], ascending=[True, False], kind="stable"
    ).drop_duplicates("pitcher_trackman_id")
    inverse = {
        trackman_id: int(pitcher_id)
        for pitcher_id, trackman_id in zip(
            best["pitcher_id"], best["pitcher_trackman_id"]
        )
    }
    return inverse, {
        "history_joined_rows": int(len(pairs)),
        "mapped_anonymous_pitchers": int(best["pitcher_id"].nunique()),
        "mapped_trackman_pitchers": int(best["pitcher_trackman_id"].nunique()),
        "minimum_retained_purity": float(best["purity"].min()),
        "mean_retained_purity": float(best["purity"].mean()),
        "collisions_removed": int(
            counts.loc[counts["purity"].ge(minimum_purity), "pitcher_trackman_id"].duplicated().sum()
        ),
    }


def count_matrix(
    rows: pd.DataFrame, keys: list[str], label: str,
) -> pd.DataFrame:
    table = rows.groupby(
        [*keys, label], sort=False, observed=True, dropna=False
    ).size().unstack(label, fill_value=0)
    return table.reindex(columns=FINE_TYPES, fill_value=0).astype(np.float64)


def map_table(table: pd.DataFrame, query: pd.DataFrame, keys: list[str]) -> np.ndarray:
    if len(keys) == 1:
        index = pd.Index(query[keys[0]].to_numpy(), name=table.index.name)
    else:
        index = pd.MultiIndex.from_frame(query[keys])
        index.names = table.index.names
    return table.reindex(index).to_numpy(dtype=np.float64)


def selector_metrics(
    probabilities: np.ndarray, truth_index: np.ndarray,
) -> dict[str, float]:
    chosen = probabilities[np.arange(len(truth_index)), truth_index]
    return {
        "log_loss": float(-np.mean(np.log(np.clip(chosen, 1e-12, 1.0)))),
        "top1_accuracy": float(np.mean(probabilities.argmax(axis=1) == truth_index)),
    }


def main() -> None:
    if OUTPUT.exists():
        raise FileExistsError(f"immutable result already exists: {OUTPUT}")
    prereg = json.loads(PREREG.read_text(encoding="utf-8"))
    train = load_train()
    trackman, game_ids, _ = load_trackman()
    linkage, joined = linkage_section(train, trackman, len(game_ids))
    label_map = (
        joined[["row_id", "auto_pitch_type"]]
        .drop_duplicates("row_id")
        .assign(fine_auto=lambda frame: normalize_fine_pitch_type(frame["auto_pitch_type"]))
        .set_index("row_id")["fine_auto"]
    )
    train["fine_auto"] = train["row_id"].map(label_map)

    fold_data: dict[int, dict[str, Any]] = {}
    baseline_meta: dict[str, Any] = {}
    profile_cache: dict[tuple[int, float, float, float], np.ndarray] = {}
    mapping_meta: dict[str, Any] = {}
    minimum_purity = float(
        prereg["selection_protocol"]["minimum_identity_purity"]
    )

    for year in YEARS:
        history = train.loc[
            train["season"].lt(year) & train["game_type"].eq("R")
        ].copy()
        valid = train.loc[
            train["season"].eq(year) & train["game_type"].eq("R")
        ].copy()
        catboost_probabilities, diagnostics = fit_selector(
            history, valid, "auto", year
        )
        baseline_meta[str(year)] = diagnostics
        matched = valid["fine_auto"].notna().to_numpy(dtype=bool)
        truth = valid.loc[matched, "fine_auto"].astype(str).to_numpy()
        truth_index = np.array(
            [FINE_TYPES.index(value) for value in truth], dtype=np.int16
        )

        identity_map, identity_meta = history_identity_map(
            joined, year, minimum_purity
        )
        expanded = trackman.loc[trackman["season"].lt(year)].copy()
        expanded["pitcher_id"] = expanded["pitcher_trackman_id"].map(identity_map)
        expanded = expanded.loc[expanded["pitcher_id"].notna()].copy()
        expanded["pitcher_id"] = expanded["pitcher_id"].astype(np.int64)
        expanded["fine_auto"] = normalize_fine_pitch_type(
            expanded["auto_pitch_type"]
        )
        expanded = expanded.loc[expanded["fine_auto"].notna()].copy()
        matched_history_rows = int(
            joined.loc[
                joined["season"].lt(year) & joined["auto_pitch_type"].notna()
            ].shape[0]
        )
        expansion_factor = float(len(expanded) / max(1, matched_history_rows))
        identity_meta.update({
            "expanded_trackman_rows": int(len(expanded)),
            "matched_joined_history_rows": matched_history_rows,
            "expansion_factor": expansion_factor,
            "expanded_unique_games": int(expanded["trackman_game_id"].nunique()),
        })
        mapping_meta[str(year)] = identity_meta

        global_counts = (
            expanded["fine_auto"].value_counts().reindex(FINE_TYPES).fillna(0.0)
            .to_numpy(dtype=np.float64)
        )
        global_prior = global_counts / global_counts.sum()
        pitcher_counts = count_matrix(expanded, ["pitcher_id"], "fine_auto")
        count_counts = count_matrix(
            expanded, ["pitcher_id", "balls_before", "strikes_before"],
            "fine_auto",
        )
        valid_matched = valid.loc[matched]
        pitcher_raw = map_table(pitcher_counts, valid_matched, ["pitcher_id"])
        pitcher_missing = np.isnan(pitcher_raw).all(axis=1)
        pitcher_raw[pitcher_missing] = 0.0
        count_raw = map_table(
            count_counts, valid_matched,
            ["pitcher_id", "balls_before", "strikes_before"],
        )
        count_missing = np.isnan(count_raw).all(axis=1)
        count_raw[count_missing] = 0.0
        cb_matched = catboost_probabilities[matched]

        for pitcher_k in prereg["profile_grid"]["pitcher_k"]:
            pitcher_total = pitcher_raw.sum(axis=1)
            pitcher_probability = (
                pitcher_raw + float(pitcher_k) * global_prior[None, :]
            ) / (pitcher_total[:, None] + float(pitcher_k))
            for count_k in prereg["profile_grid"]["pitcher_count_k"]:
                count_total = count_raw.sum(axis=1)
                count_probability = (
                    count_raw + float(count_k) * pitcher_probability
                ) / (count_total[:, None] + float(count_k))
                for count_weight in prereg["profile_grid"]["count_weight"]:
                    profile = (
                        (1.0 - float(count_weight)) * pitcher_probability
                        + float(count_weight) * count_probability
                    )
                    profile_cache[(
                        year, float(pitcher_k), float(count_k), float(count_weight)
                    )] = profile
        fold_data[year] = {
            "truth_index": truth_index,
            "catboost": cb_matched,
            "baseline": selector_metrics(cb_matched, truth_index),
        }
        del history, valid, expanded, pitcher_counts, count_counts
        gc.collect()

    trials: list[dict[str, Any]] = []
    for pitcher_k in prereg["profile_grid"]["pitcher_k"]:
        for count_k in prereg["profile_grid"]["pitcher_count_k"]:
            for count_weight in prereg["profile_grid"]["count_weight"]:
                for catboost_weight in prereg["profile_grid"]["catboost_geometric_weight"]:
                    years: dict[str, Any] = {}
                    for year in YEARS:
                        fold = fold_data[year]
                        profile = profile_cache[(
                            year, float(pitcher_k), float(count_k), float(count_weight)
                        )]
                        alpha = float(catboost_weight)
                        blended = np.exp(
                            alpha * np.log(np.clip(fold["catboost"], 1e-12, 1.0))
                            + (1.0 - alpha) * np.log(np.clip(profile, 1e-12, 1.0))
                        )
                        blended /= blended.sum(axis=1, keepdims=True)
                        candidate = selector_metrics(blended, fold["truth_index"])
                        years[str(year)] = {
                            "baseline": fold["baseline"],
                            "candidate": candidate,
                            "log_loss_improvement": float(
                                fold["baseline"]["log_loss"] - candidate["log_loss"]
                            ),
                            "top1_improvement": float(
                                candidate["top1_accuracy"]
                                - fold["baseline"]["top1_accuracy"]
                            ),
                        }
                    loglosses = [years[str(y)]["candidate"]["log_loss"] for y in YEARS]
                    top1_improvements = [years[str(y)]["top1_improvement"] for y in YEARS]
                    candidate_id = (
                        f"pk{pitcher_k:g}_ck{count_k:g}_cw{count_weight:g}_"
                        f"cb{catboost_weight:g}"
                    )
                    trials.append({
                        "candidate_id": candidate_id,
                        "pitcher_k": float(pitcher_k),
                        "pitcher_count_k": float(count_k),
                        "count_weight": float(count_weight),
                        "catboost_geometric_weight": float(catboost_weight),
                        "worst_log_loss": float(max(loglosses)),
                        "mean_log_loss": float(np.mean(loglosses)),
                        "minimum_top1_improvement": float(min(top1_improvements)),
                        "years": years,
                    })
    selected = min(
        trials,
        key=lambda item: (
            item["worst_log_loss"], item["mean_log_loss"],
            -item["minimum_top1_improvement"], item["candidate_id"],
        ),
    )
    gate = prereg["selection_protocol"]["gate"]
    checks: dict[str, bool] = {}
    for year in YEARS:
        metrics = selected["years"][str(year)]
        checks[f"{year}_expansion"] = bool(
            mapping_meta[str(year)]["expansion_factor"]
            >= float(gate["minimum_expansion_factor_each_year"])
        )
        checks[f"{year}_logloss"] = bool(
            metrics["log_loss_improvement"]
            >= float(gate["minimum_log_loss_improvement_each_year"])
        )
        checks[f"{year}_top1"] = bool(
            metrics["top1_improvement"]
            >= float(gate["minimum_top1_improvement_each_year"])
        )
    passed = bool(all(checks.values()))
    report = {
        "experiment_id": prereg["experiment_id"],
        "status": "selector_source_pass" if passed else "selector_source_failed",
        "preregister_sha256": digest(PREREG),
        "script_sha256": digest(Path(__file__)),
        "years_read": list(YEARS),
        "years_not_read": [2022, 2023, 2024],
        "competition_control_target_used": False,
        "test_rows_read": False,
        "linkage_summary": {
            key: linkage[key] for key in (
                "train_games", "trackman_games", "unambiguous_one_to_one",
                "aligned_rows", "elementwise_state_agreement",
            )
        },
        "mapping": mapping_meta,
        "baseline_selector": baseline_meta,
        "trial_count": len(trials),
        "selected": selected,
        "gate": {"requirements": gate, "checks": checks, "pass": passed},
        "goal_status": "active",
        "goal_completion_claimed": False,
    }
    OUTPUT.write_text(
        json.dumps(safe(report), ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(safe({
        "status": report["status"], "mapping": mapping_meta,
        "selected": selected, "gate": report["gate"],
    }), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    os.environ.setdefault("V2_BOOSTER_DEVICE", "gpu")
    main()
