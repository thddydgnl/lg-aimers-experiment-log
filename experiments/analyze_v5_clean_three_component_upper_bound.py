#!/usr/bin/env python3
"""Exact 2022/2023 grid upper bound for simple existing-model mixtures.

The script deliberately has no 2024 path.  It is a development-only catalog:
even a survivor must be retrained from a fresh recipe and locked before the
confirmation fold is opened.
"""

from __future__ import annotations

from itertools import combinations
import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
PRED = ROOT / "experiments/results/predictions"
OUTPUT = ROOT / "experiments/results/v5_clean_three_component_upper_bound.json"
TRAIN = ROOT / "open/data/train.csv"
YEARS = (2022, 2023)
STEP = 0.025
REQUIRED_GAIN = 132.11992465293324

# Every entry below is either one fitted model or one fixed, reproducible
# row-local recipe.  Known signed/many-arm meta stacks are intentionally absent.
# A per-year stem is used where the old experiment named identical recipes
# differently across support folds.
CANDIDATES: dict[str, dict[int, tuple[str, str]]] = {
    "exact_c": {
        2022: ("v3_sparse_c_backtest", "catboost_outcome"),
        2023: ("v3_sparse_c_backtest", "catboost_outcome"),
    },
    "v3_a": {
        2022: ("v3_sparse_a_backtest", "catboost_outcome"),
        2023: ("v3_sparse_a_backtest", "catboost_outcome"),
    },
    "v3_b": {
        2022: ("v3_sparse_b_backtest", "catboost_outcome"),
        2023: ("v3_sparse_b_backtest", "catboost_outcome"),
    },
    "tabm_successcall": {
        year: ("v4_tabm_enhanced_successcall_all", "tabm_outcome") for year in YEARS
    },
    "tabm_enhanced": {
        year: ("v4_tabm_enhanced_all", "tabm_outcome") for year in YEARS
    },
    "tabm_seed42": {
        year: ("v4_tabm_enhanced_seed42_all", "tabm_outcome") for year in YEARS
    },
    "tabm_rfit": {
        year: ("v4_tabm_enhanced_rfit_all", "tabm_outcome") for year in YEARS
    },
    "tabm_alltype": {
        year: ("v4_tabm_enhanced_alltype_all", "tabm_outcome") for year in YEARS
    },
    "numeric_current_context_level_tm": {
        year: ("v4_numeric_cat_current_context_level_tmctx_seed42", "catboost_numeric")
        for year in YEARS
    },
    "numeric_current_context_tm": {
        year: ("v4_numeric_cat_current_context_tmctx_seed42", "catboost_numeric")
        for year in YEARS
    },
    "numeric_current_tm": {
        year: ("v4_numeric_cat_current_tmctx_seed42", "catboost_numeric")
        for year in YEARS
    },
    "numeric_current_context_level_no_tm": {
        year: ("v4_numeric_cat_current_context_level_notm_seed42", "catboost_numeric")
        for year in YEARS
    },
    "numeric_no_current_tm": {
        year: ("v4_numeric_cat_nocurrent_tmctx_seed42", "catboost_numeric")
        for year in YEARS
    },
    "outcome_all_call": {
        year: ("v4_outcome_all_call_components", "catboost_outcome") for year in YEARS
    },
    "outcome_a_components": {
        year: ("v4_outcome_a_components", "catboost_outcome") for year in YEARS
    },
    "outcome_c_current_context_level": {
        year: ("v4_outcome_c_current_context_level", "catboost_outcome")
        for year in YEARS
    },
    "failure_decomposition": {
        year: ("v4_failure_decomp_blend", "all_failure") for year in YEARS
    },
    "component_reweight": {
        year: ("v4_outcome_component_reweight", "component_reweight") for year in YEARS
    },
    "pitchtype_failure": {
        year: ("v4_pitchtype_failure_prior", "pitchtype_failure") for year in YEARS
    },
    "tabm_binary_brier": {
        year: ("v4_tabm_binary_brier_enhanced_all", "tabm") for year in YEARS
    },
    "v2_linear": {
        year: ("v2_linear_tuned", "linear") for year in YEARS
    },
    "v2_hgb": {
        year: ("v2_base", "hgb") for year in YEARS
    },
    "v2_catboost": {
        year: ("v2_catboost", "catboost") for year in YEARS
    },
    "v2_hgb_pitcher_te_platoon": {
        year: ("v2_hgb_pitcher_te_platoon", "hgb") for year in YEARS
    },
    "v3_lgbm_regular": {
        year: ("v3_lgbm_regular", "lgbm") for year in YEARS
    },
    "neural_resnet_candidate": {
        year: ("v4_neural_resnet_delta", "candidate") for year in YEARS
    },
    "joint_neural_conservative": {
        year: ("v4_joint_neural_conservative", "conservative") for year in YEARS
    },
    "current_state_binary": {
        2022: ("v4_current_state_binary_support22", "catboost"),
        2023: ("v4_current_state_binary_support23", "catboost"),
    },
    "current_state_binary_tuned": {
        2022: ("v4_current_state_binary_tuned_backtest", "catboost"),
        2023: ("v4_current_state_binary_tuned_backtest", "catboost"),
    },
    "current_state_c": {
        2022: ("v4_current_state_c_support2223", "catboost_outcome"),
        2023: ("v4_current_state_c_support2223", "catboost_outcome"),
    },
    "adaptive_state": {
        year: ("v5_adaptive_state_space_source", "final_prediction") for year in YEARS
    },
    "direct_season_update": {
        year: ("v5_direct_season_update_dev", "final_prediction") for year in YEARS
    },
    "dynamic_state_e14": {
        year: ("v5_dynamic_state_e14_dev2223", "catboost_state_residual")
        for year in YEARS
    },
    "dynamic_state_hier": {
        year: ("v5_dynamic_state_hier_dev2223", "catboost_state_residual")
        for year in YEARS
    },
    "hgb_state_context": {
        year: ("v5_hgb_state_context_dev2223", "hgb") for year in YEARS
    },
    "monotone_state_brier": {
        year: ("v5_monotone_state_brier_v1_dev2223", "candidate") for year in YEARS
    },
    "temporal_state_offset_gate": {
        year: ("v5_temporal_state_offset_gate_source", "temporal_state_offset_gate")
        for year in YEARS
    },
}

ANCHORS = {
    "exact_c": None,
    "honest_identity": "v5_honest_m3_r_identity",
    "honest_grid": "v5_honest_m3_r_grid",
}


def load_archive(stem: str, year: int) -> dict[str, np.ndarray]:
    path = PRED / f"{stem}_{year}.npz"
    with np.load(path, allow_pickle=False) as archive:
        result = {}
        for key in archive.files:
            try:
                result[key] = np.asarray(archive[key])
            except ValueError as exc:
                if "Object arrays cannot be loaded" not in str(exc):
                    raise
        return result


def score(y: np.ndarray, prediction: np.ndarray) -> float:
    y64 = np.asarray(y, dtype=np.float64)
    p64 = np.clip(np.asarray(prediction, dtype=np.float64), 1e-6, 1.0 - 1e-6)
    reference = float(y64.mean() * (1.0 - y64.mean()))
    return 100_000.0 * (1.0 - float(np.mean((p64 - y64) ** 2)) / reference)


def weight_grid() -> np.ndarray:
    denominator = int(round(1.0 / STEP))
    rows = []
    for first in range(denominator + 1):
        for second in range(denominator - first + 1):
            third = denominator - first - second
            rows.append((first, second, third))
    return np.asarray(rows, dtype=np.float64) / denominator


def canonical_solution(names: tuple[str, str, str], weights: np.ndarray) -> tuple:
    active = tuple(
        (name, round(float(weight), 10))
        for name, weight in zip(names, weights)
        if weight > 1e-12
    )
    return active


def main() -> None:
    game_type = pd.read_csv(TRAIN, usecols=["game_type"])["game_type"].to_numpy()
    exact = {
        year: load_archive(CANDIDATES["exact_c"][year][0], year)
        for year in YEARS
    }
    masks = {
        year: game_type[exact[year]["row_index"]] == "R" for year in YEARS
    }

    anchors: dict[int, dict[str, np.ndarray]] = {year: {} for year in YEARS}
    for year in YEARS:
        anchors[year]["exact_c"] = exact[year]["catboost_outcome"].astype(np.float64)
        for name, stem in ANCHORS.items():
            if stem is None:
                continue
            artifact = load_archive(stem, year)
            for key in ("y", "row_index", "cluster"):
                if not np.array_equal(artifact[key], exact[year][key]):
                    raise ValueError(f"{name} {year} {key} alignment mismatch")
            anchors[year][name] = artifact["final_prediction"].astype(np.float64)

    names = list(CANDIDATES)
    matrices: dict[int, np.ndarray] = {}
    for year in YEARS:
        columns = []
        for name in names:
            stem, key = CANDIDATES[name][year]
            artifact = load_archive(stem, year)
            for align_key in ("y", "row_index"):
                if not np.array_equal(artifact[align_key], exact[year][align_key]):
                    raise ValueError(f"{name} {year} {align_key} alignment mismatch")
            values = np.asarray(artifact[key], dtype=np.float64)
            if values.ndim != 1 or not np.isfinite(values).all():
                raise ValueError(f"{name} {year} is not a finite prediction vector")
            if float(values.min()) < 0.0 or float(values.max()) > 1.0:
                raise ValueError(f"{name} {year} lies outside [0, 1]")
            routed = anchors[year]["exact_c"].copy()
            routed[masks[year]] = values[masks[year]]
            columns.append(routed)
        matrices[year] = np.column_stack(columns)

    anchor_scores = {
        year: {
            name: score(exact[year]["y"], prediction)
            for name, prediction in anchors[year].items()
        }
        for year in YEARS
    }
    strongest_anchor_score = {
        year: max(anchor_scores[year].values()) for year in YEARS
    }

    quadratics = {}
    for year in YEARS:
        matrix = matrices[year]
        y = exact[year]["y"].astype(np.float64)
        quadratics[year] = {
            "gram": matrix.T @ matrix / len(y),
            "cross": matrix.T @ y / len(y),
            "yy": float(np.mean(y)),
            "reference": float(y.mean() * (1.0 - y.mean())),
        }

    grid = weight_grid()
    best_by_recipe: dict[tuple, dict[str, object]] = {}
    for indices in combinations(range(len(names)), 3):
        recipe_names = tuple(names[index] for index in indices)
        per_year_scores = []
        for year in YEARS:
            stats = quadratics[year]
            sub_gram = stats["gram"][np.ix_(indices, indices)]
            sub_cross = stats["cross"][list(indices)]
            mse = (
                np.einsum("bi,ij,bj->b", grid, sub_gram, grid)
                - 2.0 * (grid @ sub_cross)
                + stats["yy"]
            )
            scores = 100_000.0 * (1.0 - mse / stats["reference"])
            per_year_scores.append(scores)
        score_matrix = np.column_stack(per_year_scores)
        gain_matrix = score_matrix - np.asarray(
            [strongest_anchor_score[year] for year in YEARS], dtype=np.float64
        )
        robust = gain_matrix.min(axis=1)
        median = np.median(gain_matrix, axis=1)
        order = np.lexsort((-median, -robust))
        selected_index = int(order[0])
        weights = grid[selected_index]
        canonical = canonical_solution(recipe_names, weights)
        row = {
            "components": [
                {"name": name, "weight": float(weight)} for name, weight in canonical
            ],
            "component_count": len(canonical),
            "scores": {
                str(year): float(score_matrix[selected_index, position])
                for position, year in enumerate(YEARS)
            },
            "gains_vs_strongest_anchor": {
                str(year): float(gain_matrix[selected_index, position])
                for position, year in enumerate(YEARS)
            },
            "minimum_full_gain": float(robust[selected_index]),
            "median_full_gain": float(median[selected_index]),
        }
        previous = best_by_recipe.get(canonical)
        if previous is None or (
            row["minimum_full_gain"], row["median_full_gain"]
        ) > (previous["minimum_full_gain"], previous["median_full_gain"]):
            best_by_recipe[canonical] = row

    ranked = sorted(
        best_by_recipe.values(),
        key=lambda row: (
            float(row["minimum_full_gain"]),
            float(row["median_full_gain"]),
            -int(row["component_count"]),
        ),
        reverse=True,
    )
    best = ranked[0]
    report = {
        "protocol": {
            "development_years": list(YEARS),
            "confirmation_year_loaded": False,
            "test_rows_read": False,
            "route": "R mixture; exact-C unchanged on F",
            "weight_step": STEP,
            "candidate_count": len(names),
            "candidate_names": names,
            "strongest_anchor_per_year": strongest_anchor_score,
            "anchor_scores": anchor_scores,
            "warning": "Discovery upper bound only; any survivor requires a fresh preregistered rerun before 2024 confirmation.",
        },
        "required_minimum_full_gain": REQUIRED_GAIN,
        "best": best,
        "passes_required_gain": bool(float(best["minimum_full_gain"]) > REQUIRED_GAIN),
        "top_100": ranked[:100],
    }
    OUTPUT.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    print(json.dumps({
        "candidate_count": len(names),
        "unique_recipe_count": len(ranked),
        "best": best,
        "required": REQUIRED_GAIN,
        "passes": report["passes_required_gain"],
    }, ensure_ascii=False, indent=2, allow_nan=False))
    print(f"Saved {OUTPUT}")


if __name__ == "__main__":
    main()
