#!/usr/bin/env python3
"""Combine immutable dense experts with the independently locked selector."""

from __future__ import annotations

import gc
import json
import os
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np


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
from experiments.analyze_v5_fine_pitchtype_latent import (  # noqa: E402
    SOURCE_YEARS,
    TARGET,
    load_anchor,
)
from experiments.analyze_v5_row_local_pitchmix_selector import (  # noqa: E402
    GROUPS,
    derive_coarse_labels,
    fit_selector,
    history_context_probabilities,
    load_source,
    normalize,
    pitchmix_states_before_each_season,
    probability_metrics,
    state_features,
)


PREREG = ROOT / "experiments/params/v5_dense_moe_locked_selector_preregister.json"
ROW_SELECTOR_PREREG = (
    ROOT / "experiments/params/v5_row_local_pitchmix_selector_preregister.json"
)
ROW_SELECTOR_REPORT = (
    ROOT / "experiments/results/v5_row_local_pitchmix_selector_source.json"
)
DENSE_REPORT = ROOT / "experiments/results/v5_dense_pitchtype_moe_source_gate_v2.json"
REPORT = ROOT / "experiments/results/v5_dense_moe_locked_selector_source_v2.json"
STAGES = {
    2020: "v5_dense_pitchtype_moe_source2020",
    2021: "v5_dense_pitchtype_moe_source2021",
}
KEY = "catboost_dense_pitchtype_moe"


def main() -> None:
    if REPORT.exists():
        raise FileExistsError(f"immutable result already exists: {REPORT}")
    started = time.perf_counter()
    prereg = json.loads(PREREG.read_text(encoding="utf-8"))
    row_prereg = json.loads(ROW_SELECTOR_PREREG.read_text(encoding="utf-8"))
    frozen_report = json.loads(ROW_SELECTOR_REPORT.read_text(encoding="utf-8"))
    dense_report = json.loads(DENSE_REPORT.read_text(encoding="utf-8"))
    frozen = frozen_report["selected_selector"]
    registered = prereg["frozen_inputs"]["selector"]
    for key in (
        "state_k",
        "context_variant",
        "context_k",
        "tilt_lambda",
        "state_catboost_weight",
    ):
        if frozen[key] != registered[key]:
            raise ValueError(f"locked selector prereg mismatch: {key}")
    if dense_report["status"] != "source_failed":
        raise ValueError("unexpected immutable dense-expert source status")

    frame = load_source()
    state_ks = [
        int(value) for value in row_prereg["row_local_state"]["state_k"]
    ]
    reproduction_tolerance = float(
        prereg["reproduction_gate"][
            "maximum_absolute_selector_log_loss_difference_from_frozen_report"
        ]
    )
    folds: dict[int, dict[str, Any]] = {}
    reproduction_pass = True

    for year in SOURCE_YEARS:
        anchor = load_anchor(year)
        candidate_path = PRED / f"{STAGES[year]}_{year}.npz"
        candidate = load(candidate_path)
        for key in ("y", "row_index", "cluster"):
            if not np.array_equal(anchor[key], candidate[key]):
                raise ValueError(f"dense expert alignment mismatch: {year}/{key}")
        valid = frame.iloc[anchor["row_index"]].copy()
        if not valid["season"].eq(year).all():
            raise ValueError(f"selector validation season mismatch: {year}")
        if not np.array_equal(
            valid[TARGET].to_numpy(dtype=np.int8), anchor["y"].astype(np.int8)
        ):
            raise ValueError(f"selector target alignment mismatch: {year}")
        history_all = frame.loc[frame["season"].lt(year)].copy()
        history_all["coarse_reconstructed"] = derive_coarse_labels(history_all)
        valid["coarse_reconstructed"] = derive_coarse_labels(valid)
        states_before, final_state = pitchmix_states_before_each_season(history_all)
        history_state, _ = state_features(history_all, states_before, state_ks)
        valid_state, _ = state_features(valid, {year: final_state}, state_ks)
        history_r = history_all.loc[history_all["game_type"].eq("R")].copy()
        history_state_r = history_state.loc[history_r.index]
        regular = valid["game_type"].eq("R").to_numpy(dtype=bool)
        valid_r = valid.loc[regular].copy()
        valid_state_r = valid_state.loc[valid_r.index]

        # Reproduce the frozen experiment's exact GPU training order.  Its
        # state selector was fitted immediately after the same-year base
        # selector; the V1 combiner omitted that preceding fit and retained a
        # small 2020 GPU reduction drift.  The base probabilities are not used
        # by the candidate.
        base_probability, base_model_meta = fit_selector(
            history_r,
            valid_r,
            None,
            None,
            year,
            "base",
            row_prereg,
        )
        state_probability, model_meta = fit_selector(
            history_r,
            valid_r,
            history_state_r,
            valid_state_r,
            year,
            "state",
            row_prereg,
        )
        context, pitcher, context_meta = history_context_probabilities(
            history_r,
            valid_r,
            str(registered["context_variant"]),
            float(registered["context_k"]),
        )
        state_probability_structured = np.column_stack(
            [
                valid_state_r[
                    f"pmx_state_p_{group}_k{int(registered['state_k'])}"
                ].to_numpy(dtype=np.float64)
                for group in GROUPS
            ]
        )
        ratio = np.divide(
            context,
            pitcher,
            out=np.ones_like(context),
            where=pitcher > 1e-12,
        )
        structured = normalize(
            state_probability_structured
            * np.power(ratio, float(registered["tilt_lambda"]))
        )
        weight = float(registered["state_catboost_weight"])
        locked_probability = normalize(
            weight * state_probability + (1.0 - weight) * structured
        )
        selector_metric = probability_metrics(
            locked_probability, valid_r["coarse_reconstructed"]
        )
        expected_metric = frozen["years"][str(year)]
        difference = abs(
            float(selector_metric["log_loss"])
            - float(expected_metric["log_loss"])
        )
        fold_reproduction_pass = bool(difference <= reproduction_tolerance)
        reproduction_pass &= fold_reproduction_pass

        probability_path = (
            PRED / f"v5_dense_moe_locked_selector_source_v2_{year}.npz"
        )
        if probability_path.exists():
            raise FileExistsError(
                f"immutable selector artifact already exists: {probability_path}"
            )
        np.savez_compressed(
            probability_path,
            row_index=anchor["row_index"].astype(np.int64),
            regular_row_index=anchor["row_index"][regular].astype(np.int64),
            p_fastball=locked_probability[:, 0],
            p_breaking=locked_probability[:, 1],
            p_offspeed=locked_probability[:, 2],
        )

        expert_matrix = np.column_stack(
            [candidate[f"{PREFIX}expert_{group}"] for group in GROUPS]
        ).astype(np.float64)
        dense_moe = anchor["catboost_outcome"].astype(np.float64).copy()
        dense_moe[regular] = np.sum(
            locked_probability * expert_matrix[regular], axis=1
        )
        folds[year] = {
            "anchor": anchor,
            "candidate": candidate,
            "parent": anchor["catboost_outcome"].astype(np.float64),
            "dense_moe": dense_moe,
            "regular": regular,
            "masks": {
                "full": np.ones(len(anchor["y"]), dtype=bool),
                "R": regular,
            },
            "candidate_path": candidate_path,
            "selector_path": probability_path,
            "selector_reproduction": {
                "expected": expected_metric,
                "reproduced": selector_metric,
                "absolute_log_loss_difference": difference,
                "tolerance": reproduction_tolerance,
                "passed": fold_reproduction_pass,
                "model_metadata": model_meta,
                "preceding_base_model_metadata": base_model_meta,
                "context_metadata": context_meta,
            },
        }
        del (
            history_all,
            history_state,
            valid_state,
            history_r,
            history_state_r,
            valid_r,
            valid_state_r,
            state_probability,
            base_probability,
            structured,
            expert_matrix,
        )
        gc.collect()

    if not reproduction_pass:
        report = {
            "experiment_id": prereg["experiment_id"],
            "status": "failed_selector_reproduction_gate",
            "preregister_sha256": digest(PREREG),
            "frozen_selector_report_sha256": digest(ROW_SELECTOR_REPORT),
            "selector_reproduction": {
                str(year): folds[year]["selector_reproduction"]
                for year in SOURCE_YEARS
            },
            "control_metrics_computed": False,
            "years_not_read": [2022, 2023, 2024],
        }
        REPORT.write_text(
            json.dumps(safe(report), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(json.dumps(safe(report), ensure_ascii=False, indent=2))
        return

    bootstrap = int(prereg["bootstrap_iterations"])
    trials: list[dict[str, Any]] = []
    for gamma in prereg["candidate"]["top_level_blend_grid"]:
        years: dict[str, Any] = {}
        for year in SOURCE_YEARS:
            fold = folds[year]
            years[str(year)] = evaluate(
                fold["candidate"],
                fold["parent"],
                fold["dense_moe"],
                fold["regular"],
                fold["masks"],
                float(gamma),
                bootstrap,
                710000 + 10000 * year + int(float(gamma) * 100),
            )
        full_gains = [
            years[str(year)]["routes"]["full"]["gain"]
            for year in SOURCE_YEARS
        ]
        r_gains = [
            years[str(year)]["routes"]["R"]["gain"]
            for year in SOURCE_YEARS
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
    for year in SOURCE_YEARS:
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
    report = {
        "experiment_id": prereg["experiment_id"],
        "status": "source_pass" if passed else "source_failed",
        "preregister_sha256": digest(PREREG),
        "script_sha256": digest(Path(__file__)),
        "frozen_selector": {
            "report": str(ROW_SELECTOR_REPORT.relative_to(ROOT)),
            "report_sha256": digest(ROW_SELECTOR_REPORT),
            "preregister_sha256": digest(ROW_SELECTOR_PREREG),
            "recipe": registered,
        },
        "immutable_dense_experts": {
            "report": str(DENSE_REPORT.relative_to(ROOT)),
            "report_sha256": digest(DENSE_REPORT),
        },
        "years_read": list(SOURCE_YEARS),
        "years_not_read": [2022, 2023, 2024],
        "selector_reproduction": {
            str(year): folds[year]["selector_reproduction"]
            for year in SOURCE_YEARS
        },
        "artifacts": {
            str(year): {
                "dense_experts": str(
                    folds[year]["candidate_path"].relative_to(ROOT)
                ),
                "locked_selector": str(
                    folds[year]["selector_path"].relative_to(ROOT)
                ),
                "locked_selector_sha256": digest(folds[year]["selector_path"]),
            }
            for year in SOURCE_YEARS
        },
        "trials": trials,
        "selected": selected,
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
        "elapsed_seconds": float(time.perf_counter() - started),
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
                    for year in SOURCE_YEARS
                },
                "selector_reproduction": {
                    str(year): folds[year]["selector_reproduction"][
                        "absolute_log_loss_difference"
                    ]
                    for year in SOURCE_YEARS
                },
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    os.environ.setdefault("V2_BOOSTER_DEVICE", "gpu")
    main()
