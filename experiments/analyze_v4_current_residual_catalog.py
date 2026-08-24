#!/usr/bin/env python3
"""Forward-select one historically transferred OOF direction on the latest stack."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np


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
CATALOG = ROOT / "experiments/results/v4_oof_direction_catalog.json"
REPORT = ROOT / "experiments/results/v4_current_residual_catalog.json"


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
    catalog = json.loads(CATALOG.read_text(encoding="utf-8"))
    admitted = catalog["diverse_selected"]
    accepted = {
        year: load(PRED / f"v4_routed_tabm_stack_locked_{year}.npz")
        for year in (2023, 2024)
    }
    accepted_prediction = {
        year: accepted[year]["routed_tabm_stack"].astype(np.float64)
        for year in accepted
    }
    y = {year: accepted[year]["y"].astype(np.float64) for year in accepted}
    route_r = {year: accepted[year]["game_type_r"].astype(bool) for year in accepted}
    latest = {
        year: load(PRED / f"v4_post4_c3_axis_screen_{year}.npz")
        for year in (2023, 2024)
    }
    base = {
        year: latest[year]["selected_prediction_plus_tabtransformer"].astype(np.float64)
        for year in latest
    }
    base_scores = {year: raw_score(y[year], base[year]) for year in base}

    candidates: dict[str, dict[str, object]] = {}
    directions: dict[str, dict[int, np.ndarray]] = {}
    for row in admitted:
        name = str(row["name"])
        if row["route"] != "R" or not row["coefficient_stable"]:
            continue
        artifacts = {
            year: load(PRED / f"{row['stem']}_{year}.npz") for year in (2023, 2024)
        }
        for year in artifacts:
            if not np.array_equal(artifacts[year]["row_index"], accepted[year]["row_index"]):
                raise ValueError(f"row_index mismatch for {name}/{year}")
        values = {
            year: np.where(
                route_r[year],
                artifacts[year][row["key"]].astype(np.float64)
                - accepted_prediction[year],
                0.0,
            )
            for year in artifacts
        }
        gamma_raw, gamma = fit_scalar(values[2023], y[2023] - base[2023])
        prediction23 = np.clip(base[2023] + gamma * values[2023], 0.0, 1.0)
        gain23 = raw_score(y[2023], prediction23) - base_scores[2023]
        prior_gamma = float(row["gamma_fit_2023_accepted"])
        same_sign = bool(gamma * prior_gamma > 0.0)
        ratio = abs(gamma / prior_gamma) if abs(prior_gamma) > 1e-12 else float("inf")
        candidates[name] = {
            "stem": row["stem"],
            "key": row["key"],
            "historical_gain_fit_2022": row["gain_fit_2022"],
            "historical_transfer_gain_2023": row["transfer_gain_2023"],
            "gamma_fit_2023_accepted": prior_gamma,
            "gamma_fit_2023_current_raw": gamma_raw,
            "gamma_fit_2023_current": gamma,
            "marginal_gain_2023": gain23,
            "marginal_to_prior_gamma_ratio": ratio,
            "same_sign": same_sign,
            "passes_forward_gate": bool(
                same_sign and 0.05 <= ratio <= 4.0 and gain23 > 0.05
            ),
        }
        directions[name] = values

    eligible = [name for name, row in candidates.items() if row["passes_forward_gate"]]
    selected = max(
        eligible,
        key=lambda name: float(candidates[name]["marginal_gain_2023"]),
        default="none",
    )
    final_prediction = base.copy()
    selected_gamma = 0.0
    if selected != "none":
        selected_gamma = float(candidates[selected]["gamma_fit_2023_current"])
        final_prediction = {
            year: np.clip(
                base[year] + selected_gamma * directions[selected][year], 0.0, 1.0
            )
            for year in base
        }
    final_scores = {year: raw_score(y[year], final_prediction[year]) for year in base}

    artifacts_out: dict[int, str] = {}
    for year in (2023, 2024):
        path = PRED / f"v4_current_residual_catalog_{year}.npz"
        payload: dict[str, np.ndarray] = {
            "y": y[year],
            "row_index": accepted[year]["row_index"],
            "cluster": accepted[year]["cluster"],
            "base": base[year],
            "final_prediction": final_prediction[year],
        }
        if selected != "none":
            payload["selected_direction"] = directions[selected][year]
        np.savez_compressed(path, **payload)
        artifacts_out[year] = str(path.relative_to(ROOT))

    report = {
        "protocol": {
            "official_train_only": True,
            "test_rows_read": False,
            "candidate_admission": "positive 2022 fit and unchanged 2023 transfer with stable coefficient",
            "forward_selection_year": 2023,
            "confirmation_year": 2024,
            "one_direction_only": True,
            "selection_does_not_read_2024_labels": True,
        },
        "base_scores": base_scores,
        "candidates": candidates,
        "eligible": eligible,
        "selected": selected,
        "selected_gamma": selected_gamma,
        "final_scores": final_scores,
        "gains": {year: final_scores[year] - base_scores[year] for year in base},
        "expected_lb_median": final_scores[2024] + MEDIAN_OFFSET,
        "required_local_score": REQUIRED_LOCAL,
        "crosses_required_local_score": final_scores[2024] > REQUIRED_LOCAL,
        "prediction_artifacts": artifacts_out,
        "warning": "2024 is diagnostic confirmation and was not used for selection.",
    }
    REPORT.write_text(
        json.dumps(json_safe(report), ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(json_safe(report), ensure_ascii=False, indent=2), flush=True)
    print(f"Saved {REPORT}", flush=True)


if __name__ == "__main__":
    main()
