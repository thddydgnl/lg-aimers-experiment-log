#!/usr/bin/env python3
"""Screen the complete three-fold OOF prediction catalog without 2024 selection.

Each direction is fitted on 2022 and must improve untouched 2023.  Only those
transferred candidates are refit against the current 2023 stack and confirmed
on 2024.  This turns previously generated official-data OOF artifacts into a
leakage-safe diversity search rather than selecting them on the final fold.
"""

from __future__ import annotations

import glob
import json
import os
import sys
from collections import defaultdict
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
BASE_REPORT = ROOT / "experiments/results/v4_nested_direction_stack.json"
REPORT = ROOT / "experiments/results/v4_oof_direction_catalog.json"
META_KEYS = {"y", "row_index", "cluster", "weights", "source_scales"}
GAMMA_LIMIT = 1.0


def load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path) as archive:
        return {key: np.asarray(archive[key]) for key in archive.files}


def raw_score(y: np.ndarray, prediction: np.ndarray) -> float:
    return float(score(y, np.clip(prediction, 0.0, 1.0))["raw_competition_score"])


def fit_scalar(direction: np.ndarray, residual: np.ndarray) -> tuple[float, float]:
    denominator = float(np.dot(direction, direction))
    raw = float(np.dot(direction, residual) / denominator) if denominator else 0.0
    return raw, float(np.clip(raw, -GAMMA_LIMIT, GAMMA_LIMIT))


def discover_stems() -> list[str]:
    years: dict[str, set[int]] = defaultdict(set)
    for path in glob.glob(str(PRED / "*.npz")):
        name = os.path.basename(path)
        for year in (2022, 2023, 2024):
            suffix = f"_{year}.npz"
            if name.endswith(suffix):
                years[name[: -len(suffix)]].add(year)
    return sorted(stem for stem, values in years.items() if len(values) == 3)


def eligible_keys(artifact: dict[str, np.ndarray], rows: int) -> list[str]:
    result: list[str] = []
    for key, values in artifact.items():
        if key in META_KEYS or key.startswith("game_type"):
            continue
        array = np.asarray(values)
        if array.ndim != 1 or len(array) != rows:
            continue
        if array.dtype.kind not in "fiu" or not np.isfinite(array).all():
            continue
        # Probability-like predictions and corrections are both useful; large
        # ID/count vectors are not model outputs.
        if float(np.nanmax(np.abs(array))) > 5.0:
            continue
        result.append(key)
    return result


def current_base() -> tuple[dict[int, np.ndarray], dict[int, dict[str, np.ndarray]]]:
    report = json.loads(BASE_REPORT.read_text(encoding="utf-8"))
    stacks = {
        year: load_npz(PRED / f"v4_nested_direction_stack_{year}.npz")
        for year in (2023, 2024)
    }
    coefficients = report["candidates"]["base_plus_nested_axes_joint"]["coefficients"]
    base = {
        2023: stacks[2023]["base"].astype(np.float64)
        + sum(
            float(coefficients[name]) * stacks[2023][f"direction_{name}"]
            for name in coefficients
        ),
        2024: stacks[2024]["base_plus_nested_axes_joint"].astype(np.float64),
    }
    return base, stacks


def standardized_ridge(
    design: np.ndarray, residual: np.ndarray, penalty: float
) -> np.ndarray:
    scale = np.sqrt(np.mean(np.square(design), axis=0))
    scale = np.where(scale > 1e-12, scale, 1.0)
    x = design / scale
    beta = np.linalg.solve(
        x.T @ x + len(x) * penalty * np.eye(x.shape[1]), x.T @ residual
    )
    return beta / scale


def main() -> None:
    accepted = {
        year: load_npz(PRED / f"v4_routed_tabm_stack_locked_{year}.npz")
        for year in (2022, 2023, 2024)
    }
    accepted_pred = {
        year: accepted[year]["routed_tabm_stack"].astype(np.float64)
        for year in accepted
    }
    y = {year: accepted[year]["y"].astype(np.float64) for year in accepted}
    route_r = {year: accepted[year]["game_type_r"].astype(bool) for year in accepted}
    accepted_scores = {
        year: raw_score(y[year], accepted_pred[year]) for year in accepted
    }
    base, stacks = current_base()
    for year in (2023, 2024):
        if not np.array_equal(stacks[year]["row_index"], accepted[year]["row_index"]):
            raise ValueError(f"Current base alignment mismatch for {year}")
    base_scores = {year: raw_score(y[year], base[year]) for year in (2023, 2024)}

    screened: list[dict[str, object]] = []
    directions: dict[str, dict[int, np.ndarray]] = {}
    failures: list[dict[str, str]] = []
    stems = discover_stems()
    for stem in stems:
        try:
            artifacts = {
                year: load_npz(PRED / f"{stem}_{year}.npz")
                for year in (2022, 2023, 2024)
            }
            for year in artifacts:
                if not np.array_equal(artifacts[year]["row_index"], accepted[year]["row_index"]):
                    raise ValueError(f"row_index mismatch in {year}")
            keys = set(eligible_keys(artifacts[2022], len(y[2022])))
            keys &= set(eligible_keys(artifacts[2023], len(y[2023])))
            keys &= set(eligible_keys(artifacts[2024], len(y[2024])))
            for key in sorted(keys):
                raw_direction = {
                    year: artifacts[year][key].astype(np.float64) - accepted_pred[year]
                    for year in artifacts
                }
                for route in ("all", "R", "F"):
                    routed = {
                        year: (
                            raw_direction[year]
                            if route == "all"
                            else np.where(
                                route_r[year] if route == "R" else ~route_r[year],
                                raw_direction[year],
                                0.0,
                            )
                        )
                        for year in raw_direction
                    }
                    raw_gamma, gamma = fit_scalar(
                        routed[2022], y[2022] - accepted_pred[2022]
                    )
                    p22 = accepted_pred[2022] + gamma * routed[2022]
                    p23 = accepted_pred[2023] + gamma * routed[2023]
                    gain22 = raw_score(y[2022], p22) - accepted_scores[2022]
                    gain23 = raw_score(y[2023], p23) - accepted_scores[2023]
                    raw_gamma23, gamma23 = fit_scalar(
                        routed[2023], y[2023] - accepted_pred[2023]
                    )
                    ratio = (
                        abs(gamma23 / gamma)
                        if abs(gamma) > 1e-8 else float("inf")
                    )
                    coefficient_stable = bool(
                        gamma * gamma23 > 0.0 and 0.5 <= ratio <= 2.0
                    )
                    name = f"{stem}::{key}::{route}"
                    row = {
                        "name": name,
                        "stem": stem,
                        "key": key,
                        "route": route,
                        "gamma_fit_2022_raw": raw_gamma,
                        "gamma_fit_2022": gamma,
                        "gain_fit_2022": gain22,
                        "transfer_gain_2023": gain23,
                        "gamma_fit_2023_accepted_raw": raw_gamma23,
                        "gamma_fit_2023_accepted": gamma23,
                        "gamma_abs_ratio_2023_to_2022": ratio,
                        "coefficient_stable": coefficient_stable,
                    }
                    screened.append(row)
                    if gain23 > 0.05:
                        directions[name] = routed
        except Exception as exc:
            failures.append(
                {"stem": stem, "exception": type(exc).__name__, "message": str(exc)}
            )

    ranked = sorted(
        screened,
        key=lambda row: float(row["transfer_gain_2023"]),
        reverse=True,
    )
    transferred = [row for row in ranked if row["name"] in directions]

    # Greedy diversity on the selection fold.  Route/stem duplicates often
    # encode the same prediction; keep at most one when |correlation| >= .97.
    selected: list[dict[str, object]] = []
    # F changed labeling/regime sharply in 2023 and then reverted in 2024.
    # That behavior is documented independently of this catalog, so restrict
    # the actionable diversity screen to R-only directions.
    actionable_transferred = [
        row
        for row in transferred
        if row["route"] == "R" and bool(row["coefficient_stable"])
    ]
    for row in actionable_transferred:
        candidate = directions[str(row["name"])][2023]
        keep = True
        for prior in selected:
            existing = directions[str(prior["name"])][2023]
            if candidate.std() <= 1e-12 or existing.std() <= 1e-12:
                keep = False
                break
            correlation = float(np.corrcoef(candidate, existing)[0, 1])
            if abs(correlation) >= 0.99:
                keep = False
                break
        if keep:
            selected.append(row)
        if len(selected) >= 20:
            break

    confirmations: dict[str, dict[str, float]] = {}
    payload: dict[str, np.ndarray] = {
        "y": y[2024],
        "row_index": accepted[2024]["row_index"],
        "cluster": accepted[2024]["cluster"],
        "base": base[2024],
    }
    for row in selected:
        name = str(row["name"])
        direction = directions[name]
        raw_gamma, gamma = fit_scalar(direction[2023], y[2023] - base[2023])
        p23 = base[2023] + gamma * direction[2023]
        p24 = base[2024] + gamma * direction[2024]
        s23 = raw_score(y[2023], p23)
        s24 = raw_score(y[2024], p24)
        safe_name = f"candidate_{len(confirmations):02d}"
        confirmations[name] = {
            "gamma_fit_2023_raw": raw_gamma,
            "gamma_fit_2023": gamma,
            "selection_gain_over_current": s23 - base_scores[2023],
            "confirmation_gain_over_current": s24 - base_scores[2024],
            "confirmation_score": s24,
            "expected_lb_median": s24 + MEDIAN_OFFSET,
        }
        payload[safe_name] = np.clip(p24, 0.0, 1.0)
        payload[f"direction_{len(confirmations)-1:02d}"] = direction[2024]

    joint_results: dict[str, dict[str, object]] = {}
    if selected:
        matrix23 = np.column_stack(
            [directions[str(row["name"])][2023] for row in selected]
        )
        matrix24 = np.column_stack(
            [directions[str(row["name"])][2024] for row in selected]
        )
        specifications = {
            "lstsq": np.linalg.lstsq(matrix23, y[2023] - base[2023], rcond=None)[0]
        }
        for penalty in (1e-2, 1e-1, 1.0, 10.0):
            specifications[f"ridge_{penalty:g}"] = standardized_ridge(
                matrix23, y[2023] - base[2023], penalty
            )
        for label, coefficients in specifications.items():
            p23 = base[2023] + matrix23 @ coefficients
            p24 = base[2024] + matrix24 @ coefficients
            s23 = raw_score(y[2023], p23)
            s24 = raw_score(y[2024], p24)
            joint_results[label] = {
                "coefficients": {
                    str(row["name"]): float(value)
                    for row, value in zip(selected, coefficients)
                },
                "selection_gain_over_current": s23 - base_scores[2023],
                "confirmation_gain_over_current": s24 - base_scores[2024],
                "confirmation_score": s24,
                "expected_lb_median": s24 + MEDIAN_OFFSET,
            }
            payload[f"joint_{label}"] = np.clip(p24, 0.0, 1.0)

    all_confirmations: list[tuple[str, dict[str, object]]] = [
        *confirmations.items(),
        *((f"joint::{name}", values) for name, values in joint_results.items()),
    ]
    best = max(
        all_confirmations,
        key=lambda item: float(item[1]["confirmation_score"]),
        default=("current_base", {"confirmation_score": base_scores[2024],
                                  "confirmation_gain_over_current": 0.0,
                                  "expected_lb_median": base_scores[2024] + MEDIAN_OFFSET}),
    )
    output = PRED / "v4_oof_direction_catalog_2024.npz"
    np.savez_compressed(output, **payload)
    report = {
        "protocol": {
            "official_train_only": True,
            "test_rows_read": False,
            "candidate_screen_fit_year": 2022,
            "candidate_screen_transfer_year": 2023,
            "meta_refit_year": 2023,
            "confirmation_year": 2024,
            "gamma_bounds": [-GAMMA_LIMIT, GAMMA_LIMIT],
            "selection_does_not_read_2024_labels": True,
        },
        "catalog": {
            "three_fold_stems": len(stems),
            "screened_directions": len(screened),
            "positive_transfers": len(transferred),
            "actionable_r_transfers": len(actionable_transferred),
            "failures": failures,
        },
        "base_scores": base_scores,
        "top_screened": ranked[:100],
        "diverse_selected": selected,
        "individual_confirmations": confirmations,
        "joint_confirmations": joint_results,
        "best_observed_confirmation_diagnostic": {
            "name": best[0],
            **best[1],
            "required_local_score": REQUIRED_LOCAL,
            "crosses_required_local_score": (
                float(best[1]["confirmation_score"]) > REQUIRED_LOCAL
            ),
            "warning": "2024 ranking is diagnostic, not a selection rule",
        },
        "prediction_artifact": str(output.relative_to(ROOT)),
    }
    REPORT.write_text(
        json.dumps(json_safe(report), ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(
        json.dumps(
            json_safe(
                {
                    "catalog": report["catalog"],
                    "selected": selected,
                    "individual_confirmations": confirmations,
                    "joint_confirmations": joint_results,
                    "best": report["best_observed_confirmation_diagnostic"],
                }
            ),
            ensure_ascii=False,
            indent=2,
        ),
        flush=True,
    )
    print(f"Saved {REPORT}", flush=True)


if __name__ == "__main__":
    main()
