#!/usr/bin/env python3
"""Source-only selective fine-pitch correction with explicit abstention."""

from __future__ import annotations

import gc
import hashlib
import json
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.analyze_v5_fine_pitchtype_latent import (  # noqa: E402
    FINE_TYPES,
    LABEL_SOURCES,
    PREDICTIONS,
    SOURCE_YEARS,
    TARGET,
    build_control_matrices,
    evaluate,
    fit_selector,
    json_safe,
    load_anchor,
    load_fine_labels,
    load_main_frame,
)
from experiments.run_v5_h1_residual import (  # noqa: E402
    cluster_bootstrap_score_gain,
)


PREREG = (
    ROOT
    / "experiments/params/v5_selective_fine_pitch_abstention_preregister.json"
)
REPORT = (
    ROOT / "experiments/results/v5_selective_fine_pitch_abstention_source.json"
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def agreement_diagnostics(
    valid_r: Any,
    tagged_probability: np.ndarray,
    auto_probability: np.ndarray,
    thresholds: list[float],
) -> dict[str, Any]:
    tagged_top = tagged_probability.argmax(axis=1)
    auto_top = auto_probability.argmax(axis=1)
    agreement = tagged_top == auto_top
    confidence = np.minimum(
        tagged_probability.max(axis=1), auto_probability.max(axis=1)
    )
    output: dict[str, Any] = {}
    for threshold in thresholds:
        eligible = agreement & (confidence >= threshold)
        row: dict[str, Any] = {
            "eligible_rows": int(eligible.sum()),
            "eligible_fraction": float(eligible.mean()),
            "mean_confidence_eligible": (
                float(confidence[eligible].mean()) if eligible.any() else None
            ),
        }
        for source in LABEL_SOURCES:
            label_column = f"fine_{source}"
            matched = valid_r[label_column].notna().to_numpy(dtype=bool)
            diagnostic_mask = eligible & matched
            truth = valid_r.loc[diagnostic_mask, label_column].astype(str).to_numpy()
            truth_index = np.asarray(
                [FINE_TYPES.index(value) for value in truth], dtype=np.int16
            )
            row[f"{source}_diagnostic_rows"] = int(diagnostic_mask.sum())
            row[f"{source}_top1_accuracy"] = (
                float(np.mean(tagged_top[diagnostic_mask] == truth_index))
                if diagnostic_mask.any()
                else None
            )
        output[str(threshold)] = row
    return output


def main() -> None:
    if REPORT.exists():
        raise FileExistsError(
            "Preserve the immutable selective-abstention report instead of overwriting"
        )
    prereg = json.loads(PREREG.read_text(encoding="utf-8"))
    if prereg["status"] != "preregistered_before_source_metrics":
        raise ValueError("Unexpected preregistration status")
    started = time.perf_counter()

    labels, linkage_meta = load_fine_labels()
    frame = load_main_frame(labels)
    del labels
    gc.collect()

    thresholds = [
        float(value) for value in prereg["source_grid"]["confidence_thresholds"]
    ]
    gammas = [float(value) for value in prereg["source_grid"]["gammas"]]
    outcome_ks = [
        float(value)
        for value in prereg["control_profile"]["outcome_shrinkage_k"]
    ]
    repertoire_ks = [
        float(value)
        for value in prereg["control_profile"]["repertoire_shrinkage_k"]
    ]

    folds: dict[int, dict[str, Any]] = {}
    selector_diagnostics: dict[str, Any] = {}
    agreement_meta: dict[str, Any] = {}
    direction_cache: dict[tuple[int, str, float, float, float], np.ndarray] = {}
    route_cache: dict[tuple[int, float], np.ndarray] = {}

    for year in SOURCE_YEARS:
        anchor = load_anchor(year)
        valid = frame.iloc[anchor["row_index"]].copy()
        if not valid["season"].eq(year).all():
            raise ValueError(f"{year}: anchor season mismatch")
        if not np.array_equal(
            valid[TARGET].to_numpy(dtype=np.int8), anchor["y"].astype(np.int8)
        ):
            raise ValueError(f"{year}: anchor target mismatch")
        history = frame.loc[
            (frame["season"] < year) & frame["game_type"].eq("R")
        ].copy()
        r_mask = valid["game_type"].eq("R").to_numpy(dtype=bool)
        valid_r = valid.loc[r_mask].copy()

        probabilities: dict[str, np.ndarray] = {}
        for source in LABEL_SOURCES:
            probability, diagnostics = fit_selector(history, valid_r, source, year)
            probabilities[source] = probability
            selector_diagnostics[f"{year}_{source}"] = diagnostics
        tagged_top = probabilities["tagged"].argmax(axis=1)
        auto_top = probabilities["auto"].argmax(axis=1)
        agreement = tagged_top == auto_top
        confidence = np.minimum(
            probabilities["tagged"].max(axis=1),
            probabilities["auto"].max(axis=1),
        )
        for threshold in thresholds:
            route_cache[(year, threshold)] = agreement & (confidence >= threshold)
        agreement_meta[str(year)] = agreement_diagnostics(
            valid_r,
            probabilities["tagged"],
            probabilities["auto"],
            thresholds,
        )

        for profile_source in LABEL_SOURCES:
            for outcome_k in outcome_ks:
                for repertoire_k in repertoire_ks:
                    q_matrix, mix_matrix, _ = build_control_matrices(
                        history,
                        valid_r,
                        profile_source,
                        outcome_k,
                        repertoire_k,
                    )
                    baseline_expected = np.sum(mix_matrix * q_matrix, axis=1)
                    hard_direction = (
                        q_matrix[np.arange(len(valid_r)), tagged_top]
                        - baseline_expected
                    )
                    for threshold in thresholds:
                        direction_cache[
                            (
                                year,
                                profile_source,
                                outcome_k,
                                repertoire_k,
                                threshold,
                            )
                        ] = np.where(
                            route_cache[(year, threshold)], hard_direction, 0.0
                        )
        folds[year] = {
            "anchor": anchor,
            "game_type": valid["game_type"].astype(str).to_numpy(),
            "r_mask": r_mask,
        }
        del history, valid, valid_r, probabilities
        gc.collect()

    candidates: list[dict[str, Any]] = []
    for profile_source in LABEL_SOURCES:
        for outcome_k in outcome_ks:
            for repertoire_k in repertoire_ks:
                for threshold in thresholds:
                    for gamma in gammas:
                        years: dict[str, Any] = {}
                        coverage: dict[str, Any] = {}
                        for year in SOURCE_YEARS:
                            fold = folds[year]
                            anchor = fold["anchor"]
                            base = anchor["catboost_outcome"].astype(np.float64)
                            candidate = base.copy()
                            direction = direction_cache[
                                (
                                    year,
                                    profile_source,
                                    outcome_k,
                                    repertoire_k,
                                    threshold,
                                )
                            ]
                            candidate[fold["r_mask"]] = np.clip(
                                candidate[fold["r_mask"]] + gamma * direction,
                                1e-6,
                                1.0 - 1e-6,
                            )
                            years[str(year)] = evaluate(
                                anchor["y"], base, candidate, fold["game_type"]
                            )
                            route = route_cache[(year, threshold)]
                            coverage[str(year)] = {
                                "eligible_r_rows": int(route.sum()),
                                "eligible_r_fraction": float(route.mean()),
                            }
                        full_gains = [
                            years[str(year)]["gains"]["all"]
                            for year in SOURCE_YEARS
                        ]
                        r_gains = [
                            years[str(year)]["gains"]["R"]
                            for year in SOURCE_YEARS
                        ]
                        candidates.append(
                            {
                                "profile_source": profile_source,
                                "outcome_k": outcome_k,
                                "repertoire_k": repertoire_k,
                                "confidence_threshold": threshold,
                                "gamma": gamma,
                                "min_full_gain": float(min(full_gains)),
                                "min_r_gain": float(min(r_gains)),
                                "mean_full_gain": float(np.mean(full_gains)),
                                "coverage": coverage,
                                "years": years,
                            }
                        )
    candidates.sort(
        key=lambda row: (
            row["min_full_gain"],
            row["min_r_gain"],
            row["mean_full_gain"],
            row["confidence_threshold"],
            -row["gamma"],
            row["outcome_k"],
            row["repertoire_k"],
            row["profile_source"] == "tagged",
        ),
        reverse=True,
    )
    selected = candidates[0]

    intervals: dict[str, Any] = {}
    artifacts: dict[str, str] = {}
    for offset, year in enumerate(SOURCE_YEARS):
        fold = folds[year]
        anchor = fold["anchor"]
        base = anchor["catboost_outcome"].astype(np.float64)
        direction = direction_cache[
            (
                year,
                selected["profile_source"],
                selected["outcome_k"],
                selected["repertoire_k"],
                selected["confidence_threshold"],
            )
        ]
        candidate = base.copy()
        candidate[fold["r_mask"]] = np.clip(
            candidate[fold["r_mask"]] + selected["gamma"] * direction,
            1e-6,
            1.0 - 1e-6,
        )
        intervals[str(year)] = cluster_bootstrap_score_gain(
            anchor["y"],
            base,
            candidate,
            anchor["cluster"].astype(str),
            fold["r_mask"],
            2000,
            591100 + offset,
        )
        path = (
            PREDICTIONS
            / f"v5_selective_fine_pitch_abstention_source_{year}.npz"
        )
        if path.exists():
            raise FileExistsError(f"Preserve existing prediction artifact: {path}")
        np.savez_compressed(
            path,
            y=anchor["y"],
            row_index=anchor["row_index"],
            cluster=anchor["cluster"],
            base=base.astype(np.float32),
            correction_direction=direction.astype(np.float32),
            final_prediction=candidate.astype(np.float32),
        )
        artifacts[str(year)] = str(path.relative_to(ROOT))

    gate = prereg["source_gate"]
    conditions = {
        "minimum_full_gain_each_year": bool(
            selected["min_full_gain"]
            >= float(gate["minimum_full_gain_each_year"])
        ),
        "minimum_r_gain_each_year": bool(
            selected["min_r_gain"] >= float(gate["minimum_r_gain_each_year"])
        ),
        "r_cluster_ci_lower_positive_each_year": bool(
            all(value["ci_low"] > 0.0 for value in intervals.values())
        ),
    }
    passed = bool(all(conditions.values()))
    payload = {
        "experiment_id": prereg["experiment_id"],
        "status": "passed_source_gate" if passed else "failed_source_gate",
        "preregister": str(PREREG.relative_to(ROOT)),
        "preregister_sha256": sha256(PREREG),
        "policy": {
            "official_data_only": True,
            "test_rows_read": False,
            "latest_control_label_season_used_for_metrics": 2021,
            "current_pitch_type_in_deployable_prediction": False,
            "row_independent": True,
            "automatic_submission": False,
        },
        "linkage": linkage_meta,
        "selector_diagnostics": selector_diagnostics,
        "agreement_diagnostics": agreement_meta,
        "candidate_count": len(candidates),
        "selected": selected,
        "selected_r_cluster_intervals": intervals,
        "conditions": conditions,
        "gate_pass": passed,
        "decision": "freeze before 2022" if passed else "close without 2022+",
        "top_candidates": candidates[:20],
        "prediction_artifacts": artifacts,
        "artifact_hashes": {
            "preregister": sha256(PREREG),
            **{
                f"anchor_{year}": sha256(
                    PREDICTIONS / f"v4_m3_c_backtest_{year}_{year}.npz"
                )
                for year in SOURCE_YEARS
            },
        },
        "elapsed_seconds": float(time.perf_counter() - started),
    }
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(
        json.dumps(json_safe(payload), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(
        json.dumps(
            json_safe(
                {
                    "status": payload["status"],
                    "selected": selected,
                    "intervals": intervals,
                    "conditions": conditions,
                    "elapsed_seconds": payload["elapsed_seconds"],
                }
            ),
            ensure_ascii=False,
            indent=2,
        ),
        flush=True,
    )
    print(f"wrote {REPORT}", flush=True)


if __name__ == "__main__":
    main()
