#!/usr/bin/env python3
"""Privileged source-only ceiling for R/F current-state deconvolution.

The R-only state below uses earlier labels from the same validation season and
is therefore deliberately non-deployable.  It is never written as a prediction
artifact.  The only purpose of this script is to decide whether a separate,
row-local estimator of that state has goal-scale headroom.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.analyze_v5_fine_pitchtype_latent import (  # noqa: E402
    PREDICTIONS,
    SOURCE_YEARS,
    TARGET,
    evaluate,
    json_safe,
    load_anchor,
)
from experiments.run_e14_rolling import (  # noqa: E402
    build_e14_features,
    season_end_state,
)
from experiments.run_v5_h1_residual import (  # noqa: E402
    cluster_bootstrap_score_gain,
)


TRAIN = ROOT / "open/data/train.csv"
PREREG = (
    ROOT
    / "experiments/params/v5_game_type_state_deconvolution_oracle_preregister.json"
)
REPORT = (
    ROOT / "experiments/results/v5_game_type_state_deconvolution_oracle_source.json"
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_source() -> pd.DataFrame:
    with np.load(
        PREDICTIONS / "v4_m3_c_backtest_2021_2021.npz", allow_pickle=False
    ) as archive:
        final_index = int(np.max(archive["row_index"]))
    frame = pd.read_csv(
        TRAIN,
        usecols=[
            "season",
            "game_type",
            "pitcher_id",
            "asof_pitcher_n",
            "asof_pitcher_success_rate",
            TARGET,
        ],
        nrows=final_index + 1,
    )
    if set(frame["season"].unique()) != {2019, 2020, 2021}:
        raise ValueError("Source oracle read a label after 2021")
    return frame


def prior_row_states(frame: pd.DataFrame) -> pd.DataFrame:
    """Return privileged sequence states immediately before every source row."""
    keys = ["season", "pitcher_id"]
    all_count = frame.groupby(keys, sort=False, observed=True).cumcount()
    all_success = frame.groupby(keys, sort=False, observed=True)[TARGET].cumsum()
    all_success = all_success - frame[TARGET]

    is_r = frame["game_type"].eq("R").astype(np.int64)
    r_success_now = is_r * frame[TARGET].astype(np.int64)
    r_count = is_r.groupby(
        [frame["season"], frame["pitcher_id"]], sort=False, observed=True
    ).cumsum() - is_r
    r_success = r_success_now.groupby(
        [frame["season"], frame["pitcher_id"]], sort=False, observed=True
    ).cumsum() - r_success_now
    return pd.DataFrame(
        {
            "sequence_all_n": all_count.to_numpy(dtype=np.int64),
            "sequence_all_s": all_success.to_numpy(dtype=np.int64),
            "oracle_r_n": r_count.to_numpy(dtype=np.int64),
            "oracle_r_s": r_success.to_numpy(dtype=np.int64),
        },
        index=frame.index,
    )


def source_r_prior(frame: pd.DataFrame, year: int) -> float:
    mask = frame["season"].lt(year) & frame["game_type"].eq("R")
    if not mask.any():
        raise ValueError(f"{year}: no completed-season R prior")
    return float(frame.loc[mask, TARGET].mean())


def apply_direction(
    base: np.ndarray,
    direction: np.ndarray,
    regular: np.ndarray,
    gamma: float,
) -> np.ndarray:
    candidate = base.astype(np.float64).copy()
    candidate[regular] = np.clip(
        candidate[regular] + gamma * direction[regular], 1e-6, 1.0 - 1e-6
    )
    return candidate


def main() -> None:
    if REPORT.exists():
        raise FileExistsError(f"Preserve immutable oracle report: {REPORT}")
    prereg = json.loads(PREREG.read_text(encoding="utf-8"))
    if prereg["status"] != "preregistered_before_source_metrics":
        raise ValueError("Unexpected preregistration status")
    if prereg["role"] != "privileged_diagnostic_only":
        raise ValueError("Oracle role was not frozen as diagnostic-only")

    started = time.perf_counter()
    frame = load_source()
    sequence = prior_row_states(frame)
    states_before, _ = season_end_state(frame)
    folds: dict[int, dict[str, Any]] = {}
    semantic: dict[str, Any] = {}

    required_n = float(
        prereg["semantic_audit"]["required_aggregate_n_match_rate"]
    )
    required_s = float(
        prereg["semantic_audit"]["required_aggregate_success_match_rate"]
    )
    semantic_ok = True
    for year in SOURCE_YEARS:
        anchor = load_anchor(year)
        valid = frame.iloc[anchor["row_index"]].copy()
        valid_sequence = sequence.iloc[anchor["row_index"]]
        if not valid["season"].eq(year).all():
            raise ValueError(f"{year}: anchor season mismatch")
        if not np.array_equal(
            valid[TARGET].to_numpy(dtype=np.int8), anchor["y"].astype(np.int8)
        ):
            raise ValueError(f"{year}: anchor target mismatch")

        prior = source_r_prior(frame, year)
        aggregate, aggregate_meta = build_e14_features(
            valid,
            states_before,
            {year: prior},
            prior,
            k=50.0,
        )
        e14_n = aggregate["e14_n_season"].to_numpy(dtype=np.int64)
        e14_s = aggregate["e14_s_season"].to_numpy(dtype=np.int64)
        sequence_n = valid_sequence["sequence_all_n"].to_numpy(dtype=np.int64)
        sequence_s = valid_sequence["sequence_all_s"].to_numpy(dtype=np.int64)
        n_match = float(np.mean(e14_n == sequence_n))
        s_match = float(np.mean(e14_s == sequence_s))
        fold_semantic_ok = n_match >= required_n and s_match >= required_s
        semantic_ok &= fold_semantic_ok
        regular = valid["game_type"].eq("R").to_numpy(dtype=bool)
        oracle_r_n = valid_sequence["oracle_r_n"].to_numpy(dtype=np.int64)
        oracle_r_s = valid_sequence["oracle_r_s"].to_numpy(dtype=np.int64)
        contaminated = regular & ((oracle_r_n != e14_n) | (oracle_r_s != e14_s))
        semantic[str(year)] = {
            "rows": int(len(valid)),
            "aggregate_n_match_rate": n_match,
            "aggregate_success_match_rate": s_match,
            "semantic_gate_pass": bool(fold_semantic_ok),
            "e14_metadata": aggregate_meta,
            "r_rows": int(regular.sum()),
            "r_rows_with_prior_f_state": int(contaminated.sum()),
            "r_prior_f_state_rate": float(contaminated.sum() / regular.sum()),
            "mean_aggregate_current_season_n_on_r": float(e14_n[regular].mean()),
            "mean_r_only_current_season_n_on_r": float(oracle_r_n[regular].mean()),
        }
        folds[year] = {
            "anchor": anchor,
            "game_type": valid["game_type"].astype(str).to_numpy(),
            "regular": regular,
            "aggregate_n": e14_n,
            "aggregate_s": e14_s,
            "oracle_r_n": oracle_r_n,
            "oracle_r_s": oracle_r_s,
            "prior": prior,
        }

    if not semantic_ok:
        report = {
            "experiment_id": prereg["experiment_id"],
            "status": "invalid_semantic_audit",
            "role": prereg["role"],
            "preregister": str(PREREG.relative_to(ROOT)),
            "preregister_sha256": sha256(PREREG),
            "semantic_audit": semantic,
            "metrics_computed": False,
            "oracle_prediction_artifact_saved": False,
            "decision": "close without metrics or 2022+ labels",
            "elapsed_seconds": float(time.perf_counter() - started),
        }
        REPORT.write_text(
            json.dumps(json_safe(report), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        print(json.dumps(json_safe(report), indent=2, ensure_ascii=False))
        return

    directions: dict[tuple[int, int], np.ndarray] = {}
    for year in SOURCE_YEARS:
        fold = folds[year]
        n_all = fold["aggregate_n"].astype(np.float64)
        s_all = fold["aggregate_s"].astype(np.float64)
        n_r = fold["oracle_r_n"].astype(np.float64)
        s_r = fold["oracle_r_s"].astype(np.float64)
        prior = float(fold["prior"])
        for k_value in prereg["source_grid"]["eb_k"]:
            k = float(k_value)
            aggregate_rate = (s_all + k * prior) / (n_all + k)
            oracle_r_rate = (s_r + k * prior) / (n_r + k)
            directions[(year, int(k_value))] = oracle_r_rate - aggregate_rate

    candidates: list[dict[str, Any]] = []
    for k_value in prereg["source_grid"]["eb_k"]:
        for gamma_value in prereg["source_grid"]["gammas"]:
            k = int(k_value)
            gamma = float(gamma_value)
            years: dict[str, Any] = {}
            for year in SOURCE_YEARS:
                fold = folds[year]
                anchor = fold["anchor"]
                base = anchor["catboost_outcome"].astype(np.float64)
                candidate = apply_direction(
                    base, directions[(year, k)], fold["regular"], gamma
                )
                years[str(year)] = evaluate(
                    anchor["y"], base, candidate, fold["game_type"]
                )
            full_gain = [
                years[str(year)]["gains"]["all"] for year in SOURCE_YEARS
            ]
            r_gain = [years[str(year)]["gains"]["R"] for year in SOURCE_YEARS]
            candidates.append(
                {
                    "k": k,
                    "gamma": gamma,
                    "min_full_gain": float(min(full_gain)),
                    "min_r_gain": float(min(r_gain)),
                    "mean_full_gain": float(np.mean(full_gain)),
                    "years": years,
                }
            )
    candidates.sort(
        key=lambda row: (
            row["min_full_gain"],
            row["min_r_gain"],
            row["mean_full_gain"],
            row["k"],
            -row["gamma"],
        ),
        reverse=True,
    )
    selected = candidates[0]

    intervals: dict[str, Any] = {}
    direction_meta: dict[str, Any] = {}
    for offset, year in enumerate(SOURCE_YEARS):
        fold = folds[year]
        anchor = fold["anchor"]
        base = anchor["catboost_outcome"].astype(np.float64)
        direction = directions[(year, int(selected["k"]))]
        candidate = apply_direction(
            base, direction, fold["regular"], float(selected["gamma"])
        )
        intervals[str(year)] = cluster_bootstrap_score_gain(
            anchor["y"],
            base,
            candidate,
            anchor["cluster"],
            fold["regular"],
            iterations=2000,
            seed=53100 + offset,
        )
        active = fold["regular"] & (np.abs(direction) > 1e-15)
        direction_meta[str(year)] = {
            "active_r_rows": int(active.sum()),
            "active_r_rate": float(active.sum() / fold["regular"].sum()),
            "mean_absolute_direction_on_r": float(
                np.mean(np.abs(direction[fold["regular"]]))
            ),
            "direction_std_on_r": float(np.std(direction[fold["regular"]])),
        }

    gate = prereg["headroom_gate"]
    conditions = {
        "semantic_audit": bool(semantic_ok),
        "minimum_full_gain_each_year": bool(
            selected["min_full_gain"]
            >= float(gate["required_minimum_full_gain_each_year"])
        ),
        "minimum_r_gain_each_year": bool(
            selected["min_r_gain"]
            >= float(gate["required_minimum_r_gain_each_year"])
        ),
        "r_cluster_ci_lower_positive_each_year": bool(
            all(value["ci_low"] > 0.0 for value in intervals.values())
        ),
    }
    passed = bool(all(conditions.values()))
    report = {
        "experiment_id": prereg["experiment_id"],
        "status": "passed_headroom_gate" if passed else "failed_headroom_gate",
        "role": "privileged_diagnostic_only",
        "preregister": str(PREREG.relative_to(ROOT)),
        "preregister_sha256": sha256(PREREG),
        "policy": {
            "official_data_only": True,
            "test_rows_read": False,
            "latest_label_season_read": 2021,
            "same_season_prior_validation_labels_used": True,
            "row_independent": False,
            "eligible_as_candidate": False,
            "eligible_for_expected_lb": False,
            "eligible_for_submission": False,
            "automatic_submission": False,
        },
        "semantic_audit": semantic,
        "candidate_count": int(len(candidates)),
        "selected_privileged_oracle": selected,
        "selected_r_pitcher_cluster_intervals": intervals,
        "direction_metadata": direction_meta,
        "conditions": conditions,
        "headroom_gate_pass": passed,
        "oracle_prediction_artifact_saved": False,
        "decision": (
            "preregister a separate legal row-local state estimator"
            if passed
            else "close R/F state-deconvolution without 2022+ labels"
        ),
        "top_candidates": candidates[:20],
        "elapsed_seconds": float(time.perf_counter() - started),
    }
    REPORT.write_text(
        json.dumps(json_safe(report), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(json.dumps(json_safe(report), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
