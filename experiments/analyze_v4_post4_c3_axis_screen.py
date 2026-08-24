#!/usr/bin/env python3
"""Screen fixed post4 plus every predeclared C3 axis subset.

Candidate admission uses only the 2022 fit fold and its unchanged 2023
transfer.  The chosen arm is then refit against the current 2023 stack and
2024 is read once as a diagnostic confirmation.  All tables are rebuilt from
official training rows and strictly earlier-season OOF residuals.
"""

from __future__ import annotations

import itertools
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


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
REPORT = ROOT / "experiments/results/v4_post4_c3_axis_screen.json"
MODEL_KEY = "catboost_numeric"
YEARS = (2020, 2021, 2022, 2023, 2024)


def load(path: Path) -> dict[str, np.ndarray]:
    with np.load(path) as archive:
        return {key: np.asarray(archive[key]) for key in archive.files}


def raw_score(y: np.ndarray, prediction: np.ndarray) -> float:
    return float(score(y, np.clip(prediction, 0.0, 1.0))["raw_competition_score"])


def fit_scalar(direction: np.ndarray, residual: np.ndarray) -> tuple[float, float]:
    denominator = float(np.dot(direction, direction))
    raw = float(np.dot(direction, residual) / denominator) if denominator else 0.0
    return raw, float(np.clip(raw, -1.0, 1.0))


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
    post = {
        year: post4(raw.loc[raw["season"] < year], frames[year]) for year in YEARS
    }
    post_model = {
        year: np.clip(model[year] + post[year], 0.0, 1.0) for year in YEARS
    }

    axes: dict[int, dict[str, np.ndarray]] = {}
    for target in (2022, 2023, 2024):
        history_years = (target - 2, target - 1)
        pitcher = np.concatenate(
            [frames[year]["pitcher_id"].to_numpy(dtype=np.int64) for year in history_years]
        )
        residual = np.concatenate([y[year] - post_model[year] for year in history_years])
        history_context = {name: [] for name in CONTEXTS}
        for year in history_years:
            values = context_values(frames[year])
            for name in CONTEXTS:
                history_context[name].append(values[name])
        target_context = context_values(frames[target])
        target_pitcher = frames[target]["pitcher_id"].to_numpy(dtype=np.int64)
        axes[target] = {}
        for name, k in CONTEXTS.items():
            table = differential_table(
                pitcher,
                np.concatenate(history_context[name]),
                residual,
                k,
            )
            axes[target][name] = apply_table(
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

    axis_names = tuple(CONTEXTS)
    subsets: list[tuple[str, ...]] = [tuple()]
    for size in range(1, len(axis_names) + 1):
        subsets.extend(itertools.combinations(axis_names, size))

    screen: dict[str, dict[str, object]] = {}
    directions: dict[str, dict[int, np.ndarray]] = {}
    for subset in subsets:
        name = "post4" if not subset else "post4__" + "__".join(subset)
        predictions = {
            year: np.clip(
                post_model[year] + sum((axes[year][axis] for axis in subset), start=np.zeros(len(y[year]))),
                0.0,
                1.0,
            )
            for year in accepted
        }
        direction = {
            year: np.where(
                route_r[year],
                predictions[year] - accepted_prediction[year],
                0.0,
            )
            for year in accepted
        }
        gamma22_raw, gamma22 = fit_scalar(
            direction[2022], y[2022] - accepted_prediction[2022]
        )
        gamma23_raw, gamma23 = fit_scalar(
            direction[2023], y[2023] - accepted_prediction[2023]
        )
        gain22 = raw_score(
            y[2022], accepted_prediction[2022] + gamma22 * direction[2022]
        ) - accepted_score[2022]
        transfer23 = raw_score(
            y[2023], accepted_prediction[2023] + gamma22 * direction[2023]
        ) - accepted_score[2023]
        ratio = abs(gamma23 / gamma22) if abs(gamma22) > 1e-12 else float("inf")
        stable = bool(gamma22 * gamma23 > 0 and 0.5 <= ratio <= 2.0)
        screen[name] = {
            "axes": list(subset),
            "gamma_fit_2022_raw": gamma22_raw,
            "gamma_fit_2022": gamma22,
            "gain_fit_2022": gain22,
            "transfer_gain_2023": transfer23,
            "gamma_fit_2023_raw": gamma23_raw,
            "gamma_fit_2023": gamma23,
            "gamma_abs_ratio": ratio,
            "coefficient_stable": stable,
            "passes_gate": bool(gain22 > 0.05 and transfer23 > 0.05 and stable),
        }
        directions[name] = direction

    eligible = [name for name, row in screen.items() if row["passes_gate"]]
    # This rule is fixed before looking at 2024: maximize the weaker historical gain.
    selected = max(
        eligible,
        key=lambda name: min(
            float(screen[name]["gain_fit_2022"]),
            float(screen[name]["transfer_gain_2023"]),
        ),
        default="none",
    )

    nested_artifact = {
        year: load(PRED / f"v4_nested_direction_stack_{year}.npz")
        for year in (2023, 2024)
    }
    base = {year: nested_base(year, nested_artifact[year]) for year in nested_artifact}
    base_score = {year: raw_score(y[year], base[year]) for year in base}
    confirmations: dict[str, dict[str, object]] = {}
    final_prediction: dict[int, np.ndarray] | None = None
    selected_gamma = 0.0
    if selected != "none":
        selected_gamma_raw, selected_gamma = fit_scalar(
            directions[selected][2023], y[2023] - base[2023]
        )
        arm_prediction = {
            year: np.clip(
                base[year] + selected_gamma * directions[selected][year], 0.0, 1.0
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
        tab_gamma_raw, tab_gamma = fit_scalar(
            tab_direction[2023], y[2023] - arm_prediction[2023]
        )
        final_prediction = {
            year: np.clip(
                arm_prediction[year] + tab_gamma * tab_direction[year], 0.0, 1.0
            )
            for year in base
        }
        confirmations[selected] = {
            "arm_gamma_fit_2023_raw": selected_gamma_raw,
            "arm_gamma_fit_2023": selected_gamma,
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
        artifact_path = PRED / f"v4_post4_c3_axis_screen_{year}.npz"
        payload: dict[str, np.ndarray] = {
            "y": y[year],
            "row_index": accepted[year]["row_index"],
            "cluster": accepted[year]["cluster"],
            "game_type_r": accepted[year]["game_type_r"],
            "base": base[year],
        }
        if final_prediction is not None:
            payload["selected_prediction_plus_tabtransformer"] = final_prediction[year]
            payload["selected_direction"] = directions[selected][year]
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
            "candidate_family_predeclared": "all 2^3 fixed C3 axis subsets after post4",
            "screen_fit_year": 2022,
            "screen_transfer_year": 2023,
            "selection_rule": "max min(gain_2022, unchanged_transfer_gain_2023)",
            "coefficient_refit_year": 2023,
            "confirmation_year": 2024,
            "selection_does_not_read_2024_labels": True,
        },
        "screen": screen,
        "eligible": eligible,
        "selected": selected,
        "selected_gamma_fit_2023": selected_gamma,
        "base_scores": base_score,
        "confirmation": confirmations.get(selected),
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
