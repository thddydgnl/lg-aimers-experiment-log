#!/usr/bin/env python3
"""Blend independently trained failure experts into the current 2024 stack.

The two blend coefficients are fit on the documented 2024 development fold.
The unchanged coefficients must keep 2022 Brier deterioration below 0.0005.
All predictions are outer-OOF and use only seasons strictly before each fold.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import minimize


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.analyze_v4_temporal_residual_ridge import (  # noqa: E402
    MEDIAN_OFFSET,
    REQUIRED_LOCAL,
    json_safe,
    score,
)


PRED = ROOT / "experiments/results/predictions"
REPORT = ROOT / "experiments/results/v4_failure_decomp_blend.json"
MAX_2022_BRIER_WORSENING = 0.0005


def load(path: Path) -> dict[str, np.ndarray]:
    with np.load(path) as archive:
        return {key: np.asarray(archive[key]) for key in archive.files}


def logit(probability: np.ndarray) -> np.ndarray:
    clipped = np.clip(probability, 1e-6, 1.0 - 1e-6)
    return np.log(clipped / (1.0 - clipped))


def sigmoid(value: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(value, -30.0, 30.0)))


def metrics(y: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    return score(y, np.clip(prediction, 0.0, 1.0))


def blend(
    base: np.ndarray,
    all_failure: np.ndarray,
    no_middle: np.ndarray,
    route: np.ndarray,
    weights: np.ndarray,
) -> np.ndarray:
    result = logit(base)
    result[route] = (
        (1.0 - float(weights.sum())) * result[route]
        + float(weights[0]) * logit(all_failure[route])
        + float(weights[1]) * logit(no_middle[route])
    )
    return sigmoid(result)


def main() -> None:
    accepted22 = load(PRED / "v4_routed_tabm_stack_locked_2022.npz")
    current24 = load(PRED / "v4_post4_c3_axis_screen_2024.npz")
    failure = {
        year: load(PRED / f"v4_failure_decomp_current_primary_{year}.npz")
        for year in (2022, 2024)
    }
    anchors = {2022: accepted22, 2024: current24}
    for year in (2022, 2024):
        if not np.array_equal(anchors[year]["row_index"], failure[year]["row_index"]):
            raise ValueError(f"row_index mismatch for {year}")

    y = {year: anchors[year]["y"].astype(np.float64) for year in anchors}
    base = {
        2022: accepted22["routed_tabm_stack"].astype(np.float64),
        2024: current24["selected_prediction_plus_tabtransformer"].astype(np.float64),
    }
    route_r = {
        2022: accepted22["game_type_r"].astype(bool),
        2024: current24["game_type_r"].astype(bool),
    }
    expert = {
        year: {
            "all_failure": failure[year]["catboost_failure_decomp"].astype(np.float64),
            "no_middle": failure[year][
                "catboost_failure_decomp__p_no_middle"
            ].astype(np.float64),
        }
        for year in failure
    }
    base_metrics = {year: metrics(y[year], base[year]) for year in base}

    candidates: list[dict[str, object]] = []
    predictions: dict[str, dict[int, np.ndarray]] = {}
    for route_name in ("all", "R"):
        routes = {
            year: (
                np.ones(len(y[year]), dtype=bool)
                if route_name == "all"
                else route_r[year]
            )
            for year in y
        }

        def objective(weights: np.ndarray) -> float:
            prediction = blend(
                base[2024], expert[2024]["all_failure"],
                expert[2024]["no_middle"], routes[2024], weights,
            )
            return float(np.mean(np.square(y[2024] - prediction)))

        optimized = minimize(
            objective,
            x0=np.asarray([0.06, 0.015]),
            method="SLSQP",
            bounds=((0.0, 0.30), (0.0, 0.20)),
            constraints=({"type": "ineq", "fun": lambda value: 0.30 - value.sum()},),
            options={"ftol": 1e-14, "maxiter": 500},
        )
        if not optimized.success:
            raise RuntimeError(optimized.message)
        for fit_name, weights in (
            ("public_fixed", np.asarray([0.06, 0.015])),
            ("primary_optimized", optimized.x.astype(np.float64)),
        ):
            name = f"{route_name}__{fit_name}"
            predictions[name] = {
                year: blend(
                    base[year], expert[year]["all_failure"],
                    expert[year]["no_middle"], routes[year], weights,
                )
                for year in y
            }
            candidate_metrics = {
                year: metrics(y[year], predictions[name][year]) for year in y
            }
            brier_delta22 = float(
                candidate_metrics[2022]["brier"] - base_metrics[2022]["brier"]
            )
            candidates.append({
                "name": name,
                "route": route_name,
                "weights": {
                    "all_failure": float(weights[0]),
                    "no_middle": float(weights[1]),
                    "base": float(1.0 - weights.sum()),
                },
                "metrics": candidate_metrics,
                "gain_2024": float(
                    candidate_metrics[2024]["raw_competition_score"]
                    - base_metrics[2024]["raw_competition_score"]
                ),
                "brier_delta_2022": brier_delta22,
                "passes_2022_safety": brier_delta22 <= MAX_2022_BRIER_WORSENING,
            })

    eligible = [
        row for row in candidates
        if row["passes_2022_safety"] and float(row["gain_2024"]) > 0.0
    ]
    selected = max(eligible, key=lambda row: float(row["gain_2024"]))
    selected_name = str(selected["name"])

    artifacts: dict[int, str] = {}
    for year in (2022, 2024):
        path = PRED / f"v4_failure_decomp_blend_{year}.npz"
        np.savez_compressed(
            path,
            y=y[year],
            row_index=anchors[year]["row_index"],
            cluster=anchors[year]["cluster"],
            base=base[year],
            final_prediction=predictions[selected_name][year],
            all_failure=expert[year]["all_failure"],
            no_middle=expert[year]["no_middle"],
        )
        artifacts[year] = str(path.relative_to(ROOT))

    final_score = float(selected["metrics"][2024]["raw_competition_score"])
    report = {
        "protocol": {
            "official_train_only": True,
            "outer_oof_training": "season strictly before target",
            "primary_development_fold": 2024,
            "support_fold": 2022,
            "max_2022_brier_worsening": MAX_2022_BRIER_WORSENING,
            "coefficient_fit": "two nonnegative logit weights, sum <= 0.30",
        },
        "base_metrics": base_metrics,
        "candidates": candidates,
        "selected": selected,
        "final_2024_score": final_score,
        "expected_lb_median": final_score + MEDIAN_OFFSET,
        "required_local_score": REQUIRED_LOCAL,
        "crosses_required_local_score": final_score > REQUIRED_LOCAL,
        "prediction_artifacts": artifacts,
        "warning": "2024 is the documented development/meta-fit fold.",
    }
    REPORT.write_text(
        json.dumps(json_safe(report), ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(json_safe(report), ensure_ascii=False, indent=2), flush=True)
    print(f"Saved {REPORT}", flush=True)


if __name__ == "__main__":
    main()
