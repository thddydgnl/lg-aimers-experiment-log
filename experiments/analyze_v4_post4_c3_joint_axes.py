#!/usr/bin/env python3
"""Fit post4 source and C3 axes separately under a historical-transfer gate."""

from __future__ import annotations

import itertools
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import lsq_linear


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.analyze_v4_oof_residual_differentials import (  # noqa: E402
    CONTEXTS,
    apply_table,
    context_values,
    differential_table,
    source_artifact,
)
from experiments.analyze_v4_post4_c3_source import post4  # noqa: E402
from experiments.analyze_v4_temporal_residual_ridge import (  # noqa: E402
    MEDIAN_OFFSET,
    REQUIRED_LOCAL,
    json_safe,
    score,
)
from experiments.finalize_v4_oof_direction_locked import nested_base  # noqa: E402


PRED = ROOT / "experiments/results/predictions"
REPORT = ROOT / "experiments/results/v4_post4_c3_joint_axes.json"
MODEL_KEY = "catboost_numeric"
YEARS = (2020, 2021, 2022, 2023, 2024)
MAIN = "post4_source"


def load(path: Path) -> dict[str, np.ndarray]:
    with np.load(path) as archive:
        return {key: np.asarray(archive[key]) for key in archive.files}


def raw_score(y: np.ndarray, prediction: np.ndarray) -> float:
    return float(score(y, np.clip(prediction, 0.0, 1.0))["raw_competition_score"])


def fit_bounded(
    columns: list[np.ndarray], residual: np.ndarray, names: tuple[str, ...]
) -> np.ndarray:
    design = np.column_stack(columns)
    lower = np.zeros(len(names), dtype=np.float64)
    upper = np.asarray([1.0 if name == MAIN else 2.0 for name in names])
    result = lsq_linear(
        design,
        residual,
        bounds=(lower, upper),
        method="bvls",
        tol=1e-10,
        max_iter=500,
    )
    if not result.success:
        raise RuntimeError(f"bounded least squares failed: {result.message}")
    return result.x.astype(np.float64)


def coefficient_stability(left: np.ndarray, right: np.ndarray) -> dict[str, object]:
    left_norm = float(np.linalg.norm(left))
    right_norm = float(np.linalg.norm(right))
    cosine = (
        float(np.dot(left, right) / (left_norm * right_norm))
        if left_norm > 1e-12 and right_norm > 1e-12
        else 0.0
    )
    norm_ratio = right_norm / left_norm if left_norm > 1e-12 else float("inf")
    stable = bool(cosine >= 0.75 and 0.5 <= norm_ratio <= 2.0)
    return {"cosine": cosine, "norm_ratio": norm_ratio, "stable": stable}


def main() -> None:
    raw = pd.read_csv(
        ROOT / "open/data/train.csv",
        usecols=[
            "season", "pitcher_id", "pitcher_hand", "batter_hand",
            "balls_before", "strikes_before", "num_runners_on",
            "control_success",
        ],
        encoding="utf-8-sig",
        low_memory=False,
    )
    source = {year: source_artifact(year) for year in YEARS}
    frames = {
        year: raw.iloc[source[year]["row_index"].astype(np.int64)].reset_index(drop=True)
        for year in YEARS
    }
    y = {year: source[year]["y"].astype(np.float64) for year in YEARS}
    model = {year: source[year][MODEL_KEY].astype(np.float64) for year in YEARS}
    post_model = {
        year: np.clip(
            model[year] + post4(raw.loc[raw["season"] < year], frames[year]),
            0.0,
            1.0,
        )
        for year in YEARS
    }

    c3_axes: dict[int, dict[str, np.ndarray]] = {}
    for target in (2022, 2023, 2024):
        history_years = (target - 2, target - 1)
        history_pitcher = np.concatenate(
            [frames[year]["pitcher_id"].to_numpy(dtype=np.int64) for year in history_years]
        )
        history_residual = np.concatenate(
            [y[year] - post_model[year] for year in history_years]
        )
        history_context = {name: [] for name in CONTEXTS}
        for year in history_years:
            values = context_values(frames[year])
            for name in CONTEXTS:
                history_context[name].append(values[name])
        target_context = context_values(frames[target])
        target_pitcher = frames[target]["pitcher_id"].to_numpy(dtype=np.int64)
        c3_axes[target] = {}
        for name, k in CONTEXTS.items():
            table = differential_table(
                history_pitcher,
                np.concatenate(history_context[name]),
                history_residual,
                k,
            )
            c3_axes[target][name] = apply_table(
                table, target_pitcher, target_context[name]
            )

    accepted = {
        year: load(PRED / f"v4_routed_tabm_stack_locked_{year}.npz")
        for year in (2022, 2023, 2024)
    }
    accepted_prediction = {
        year: accepted[year]["routed_tabm_stack"].astype(np.float64)
        for year in accepted
    }
    route_r = {year: accepted[year]["game_type_r"].astype(bool) for year in accepted}
    accepted_score = {
        year: raw_score(y[year], accepted_prediction[year]) for year in accepted
    }
    all_columns = {
        year: {
            MAIN: np.where(
                route_r[year], post_model[year] - accepted_prediction[year], 0.0
            ),
            **{
                name: np.where(route_r[year], c3_axes[year][name], 0.0)
                for name in CONTEXTS
            },
        }
        for year in accepted
    }

    subsets: list[tuple[str, ...]] = []
    axis_names = tuple(CONTEXTS)
    for size in range(len(axis_names) + 1):
        for subset in itertools.combinations(axis_names, size):
            subsets.append((MAIN, *subset))

    screens: dict[str, dict[str, object]] = {}
    for names in subsets:
        candidate = "__".join(names)
        coeff22 = fit_bounded(
            [all_columns[2022][name] for name in names],
            y[2022] - accepted_prediction[2022],
            names,
        )
        coeff23 = fit_bounded(
            [all_columns[2023][name] for name in names],
            y[2023] - accepted_prediction[2023],
            names,
        )
        pred22 = accepted_prediction[2022] + sum(
            coefficient * all_columns[2022][name]
            for name, coefficient in zip(names, coeff22)
        )
        pred23_transfer = accepted_prediction[2023] + sum(
            coefficient * all_columns[2023][name]
            for name, coefficient in zip(names, coeff22)
        )
        gain22 = raw_score(y[2022], pred22) - accepted_score[2022]
        transfer23 = raw_score(y[2023], pred23_transfer) - accepted_score[2023]
        stability = coefficient_stability(coeff22, coeff23)
        screens[candidate] = {
            "columns": list(names),
            "coefficients_fit_2022": dict(zip(names, coeff22.tolist())),
            "coefficients_fit_2023_accepted": dict(zip(names, coeff23.tolist())),
            "gain_fit_2022": gain22,
            "transfer_gain_2023": transfer23,
            "stability": stability,
            "passes_gate": bool(
                gain22 > 0.05 and transfer23 > 0.05 and stability["stable"]
            ),
        }

    eligible = [name for name, row in screens.items() if row["passes_gate"]]
    selected = max(
        eligible,
        key=lambda name: (
            min(
                float(screens[name]["gain_fit_2022"]),
                float(screens[name]["transfer_gain_2023"]),
            ),
            -len(screens[name]["columns"]),
        ),
        default="none",
    )

    nested_artifact = {
        year: load(PRED / f"v4_nested_direction_stack_{year}.npz")
        for year in (2023, 2024)
    }
    base = {year: nested_base(year, nested_artifact[year]) for year in nested_artifact}
    base_score = {year: raw_score(y[year], base[year]) for year in base}
    confirmation: dict[str, object] | None = None
    final_prediction: dict[int, np.ndarray] | None = None
    selected_coefficients: dict[str, float] = {}
    if selected != "none":
        names = tuple(screens[selected]["columns"])
        coefficients = fit_bounded(
            [all_columns[2023][name] for name in names],
            y[2023] - base[2023],
            names,
        )
        selected_coefficients = dict(zip(names, coefficients.tolist()))
        arm_prediction = {
            year: np.clip(
                base[year]
                + sum(
                    coefficient * all_columns[year][name]
                    for name, coefficient in zip(names, coefficients)
                ),
                0.0,
                1.0,
            )
            for year in base
        }
        tab = {
            year: load(PRED / f"v4_tabtransformer_seed_ensemble_{year}.npz")
            for year in base
        }
        tab_direction = {
            year: tab[year]["direction_tabtransformer_seed_average"].astype(np.float64)
            for year in base
        }
        tab_denominator = float(np.dot(tab_direction[2023], tab_direction[2023]))
        tab_gamma_raw = float(
            np.dot(tab_direction[2023], y[2023] - arm_prediction[2023])
            / tab_denominator
        )
        tab_gamma = float(np.clip(tab_gamma_raw, -1.0, 1.0))
        final_prediction = {
            year: np.clip(
                arm_prediction[year] + tab_gamma * tab_direction[year], 0.0, 1.0
            )
            for year in base
        }
        confirmation = {
            "coefficients_fit_2023_current_base": selected_coefficients,
            "tab_gamma_fit_2023_raw": tab_gamma_raw,
            "tab_gamma_fit_2023": tab_gamma,
            "arm_scores": {
                year: raw_score(y[year], arm_prediction[year]) for year in base
            },
            "final_scores": {
                year: raw_score(y[year], final_prediction[year]) for year in base
            },
        }

    artifact_paths: dict[int, str] = {}
    for year in (2023, 2024):
        artifact_path = PRED / f"v4_post4_c3_joint_axes_{year}.npz"
        payload: dict[str, np.ndarray] = {
            "y": y[year],
            "row_index": accepted[year]["row_index"],
            "cluster": accepted[year]["cluster"],
            "game_type_r": accepted[year]["game_type_r"],
            "base": base[year],
        }
        for name in (MAIN, *CONTEXTS):
            payload[f"direction_{name}"] = all_columns[year][name]
        if final_prediction is not None:
            payload["selected_prediction_plus_tabtransformer"] = final_prediction[year]
        np.savez_compressed(artifact_path, **payload)
        artifact_paths[year] = str(artifact_path.relative_to(ROOT))

    final_score = (
        raw_score(y[2024], final_prediction[2024])
        if final_prediction is not None
        else base_score[2024]
    )
    report = {
        "protocol": {
            "official_train_only": True,
            "external_model_artifacts_used": False,
            "test_rows_read": False,
            "fit": "bounded least squares; post4_source [0,1], each C3 axis [0,2]",
            "candidate_family_predeclared": "post4 source plus every C3 axis subset",
            "screen_fit_year": 2022,
            "screen_transfer_year": 2023,
            "selection_rule": "stable candidate maximizing min historical gain",
            "coefficient_refit_year": 2023,
            "confirmation_year": 2024,
            "selection_does_not_read_2024_labels": True,
        },
        "screens": screens,
        "eligible": eligible,
        "selected": selected,
        "selected_coefficients": selected_coefficients,
        "base_scores": base_score,
        "confirmation": confirmation,
        "final_2024_score": final_score,
        "expected_lb_median": final_score + MEDIAN_OFFSET,
        "required_local_score": REQUIRED_LOCAL,
        "crosses_required_local_score": final_score > REQUIRED_LOCAL,
        "prediction_artifacts": artifact_paths,
        "warning": "2024 is a diagnostic confirmation and was not used for selection.",
    }
    REPORT.write_text(
        json.dumps(json_safe(report), ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(json_safe(report), ensure_ascii=False, indent=2), flush=True)
    print(f"Saved {REPORT}", flush=True)


if __name__ == "__main__":
    main()
