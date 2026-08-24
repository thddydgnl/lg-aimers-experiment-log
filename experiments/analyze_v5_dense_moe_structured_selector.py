#!/usr/bin/env python3
"""Gate immutable dense experts with a deterministic structured selector."""

from __future__ import annotations

import gc
import json
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
    history_context_probabilities,
    load_source,
    normalize,
    pitchmix_states_before_each_season,
    probability_metrics,
    state_features,
)


PREREG = (
    ROOT
    / "experiments/params/v5_dense_moe_structured_selector_preregister.json"
)
ROW_SELECTOR_REPORT = (
    ROOT / "experiments/results/v5_row_local_pitchmix_selector_source.json"
)
DENSE_REPORT = ROOT / "experiments/results/v5_dense_pitchtype_moe_source_gate_v2.json"
REPORT = ROOT / "experiments/results/v5_dense_moe_structured_selector_source.json"
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
    selector_config = prereg["selector"]
    frozen_selector = json.loads(
        ROW_SELECTOR_REPORT.read_text(encoding="utf-8")
    )["selected_selector"]
    for key in ("state_k", "context_variant", "context_k", "tilt_lambda"):
        if selector_config[key] != frozen_selector[key]:
            raise ValueError(f"structured selector differs from frozen recipe: {key}")

    frame = load_source()
    folds: dict[int, dict[str, Any]] = {}
    for year in SOURCE_YEARS:
        anchor = load_anchor(year)
        candidate_path = PRED / f"{STAGES[year]}_{year}.npz"
        candidate = load(candidate_path)
        for key in ("y", "row_index", "cluster"):
            if not np.array_equal(anchor[key], candidate[key]):
                raise ValueError(f"dense expert alignment mismatch: {year}/{key}")
        valid = frame.iloc[anchor["row_index"]].copy()
        if not valid["season"].eq(year).all():
            raise ValueError(f"validation season mismatch: {year}")
        if not np.array_equal(
            valid[TARGET].to_numpy(dtype=np.int8), anchor["y"].astype(np.int8)
        ):
            raise ValueError(f"target alignment mismatch: {year}")
        history_all = frame.loc[frame["season"].lt(year)].copy()
        history_all["coarse_reconstructed"] = derive_coarse_labels(history_all)
        valid["coarse_reconstructed"] = derive_coarse_labels(valid)
        states_before, final_state = pitchmix_states_before_each_season(history_all)
        valid_state, state_meta = state_features(
            valid, {year: final_state}, [int(selector_config["state_k"])]
        )
        history_r = history_all.loc[history_all["game_type"].eq("R")].copy()
        regular = valid["game_type"].eq("R").to_numpy(dtype=bool)
        valid_r = valid.loc[regular].copy()
        valid_state_r = valid_state.loc[valid_r.index]
        context, pitcher, context_meta = history_context_probabilities(
            history_r,
            valid_r,
            str(selector_config["context_variant"]),
            float(selector_config["context_k"]),
        )
        state_probability = np.column_stack(
            [
                valid_state_r[
                    f"pmx_state_p_{group}_k{int(selector_config['state_k'])}"
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
            state_probability
            * np.power(ratio, float(selector_config["tilt_lambda"]))
        )
        selector_metrics = probability_metrics(
            structured, valid_r["coarse_reconstructed"]
        )
        selector_path = (
            PRED / f"v5_dense_moe_structured_selector_source_{year}.npz"
        )
        if selector_path.exists():
            raise FileExistsError(f"immutable selector artifact exists: {selector_path}")
        np.savez_compressed(
            selector_path,
            row_index=anchor["row_index"].astype(np.int64),
            regular_row_index=anchor["row_index"][regular].astype(np.int64),
            p_fastball=structured[:, 0],
            p_breaking=structured[:, 1],
            p_offspeed=structured[:, 2],
        )
        expert_matrix = np.column_stack(
            [candidate[f"{PREFIX}expert_{group}"] for group in GROUPS]
        ).astype(np.float64)
        dense_moe = anchor["catboost_outcome"].astype(np.float64).copy()
        dense_moe[regular] = np.sum(structured * expert_matrix[regular], axis=1)
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
            "selector_path": selector_path,
            "selector_metrics": selector_metrics,
            "state_metadata": state_meta,
            "context_metadata": context_meta,
        }
        del history_all, history_r, valid_state, valid_state_r, expert_matrix
        gc.collect()

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
                810000 + 10000 * year + int(float(gamma) * 100),
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
        "frozen_selector_report": {
            "path": str(ROW_SELECTOR_REPORT.relative_to(ROOT)),
            "sha256": digest(ROW_SELECTOR_REPORT),
        },
        "immutable_dense_expert_report": {
            "path": str(DENSE_REPORT.relative_to(ROOT)),
            "sha256": digest(DENSE_REPORT),
        },
        "years_read": list(SOURCE_YEARS),
        "years_not_read": [2022, 2023, 2024],
        "selector": {
            str(year): {
                "metrics": folds[year]["selector_metrics"],
                "state_metadata": folds[year]["state_metadata"],
                "context_metadata": folds[year]["context_metadata"],
                "artifact": str(folds[year]["selector_path"].relative_to(ROOT)),
                "artifact_sha256": digest(folds[year]["selector_path"]),
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
                "selector_metrics": {
                    str(year): folds[year]["selector_metrics"]
                    for year in SOURCE_YEARS
                },
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
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
