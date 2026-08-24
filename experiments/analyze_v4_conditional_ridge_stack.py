#!/usr/bin/env python3
"""Test the preselected conditional Ridge directions on the V4 hybrid."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.analyze_v4_temporal_residual_ridge import (  # noqa: E402
    Config,
    MEDIAN_OFFSET,
    REQUIRED_LOCAL,
    json_safe,
    load_frames,
    score,
    transfer_data,
)
from experiments.analyze_v4_temporal_residual_models import (  # noqa: E402
    add_raw_columns,
    build_data,
    model_recipes,
)


PRED = ROOT / "experiments/results/predictions"
SOURCE_REPORT = ROOT / "experiments/results/v4_temporal_residual_ridge.json"
REPORT = ROOT / "experiments/results/v4_conditional_ridge_stack.json"


def load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path) as archive:
        return {key: np.asarray(archive[key]) for key in archive.files}


def raw_score(y: np.ndarray, prediction: np.ndarray) -> float:
    return float(score(y, np.clip(prediction, 0.0, 1.0))["raw_competition_score"])


def scalar_fit(direction: np.ndarray, residual: np.ndarray) -> float:
    denominator = float(np.dot(direction, direction))
    return float(np.dot(direction, residual) / denominator) if denominator else 0.0


def main() -> None:
    source_report = json.loads(SOURCE_REPORT.read_text(encoding="utf-8"))
    selected = source_report["selection"]["best_by_training_mode"]
    frames, artifacts = load_frames()
    hybrid = {
        year: load_npz(PRED / f"v4_public_residual_postprocess_{year}.npz")
        for year in (2023, 2024)
    }
    for year in (2023, 2024):
        if not np.array_equal(hybrid[year]["row_index"], artifacts[year]["row_index"]):
            raise ValueError(f"Artifact alignment mismatch for {year}")

    directions: dict[int, dict[str, np.ndarray]] = {2023: {}, 2024: {}}
    for mode in ("loo", "full"):
        config = Config(**selected[mode]["config"])
        for year, source_year in ((2023, 2022), (2024, 2023)):
            data = transfer_data(
                frames[source_year], frames[year], artifacts[source_year]["m3"], config
            )
            model = make_pipeline(
                StandardScaler(), Ridge(alpha=config.alpha, fit_intercept=True)
            )
            model.fit(data["x_source"], data["residual"])
            correction = np.asarray(model.predict(data["x_target"]), dtype=np.float64)
            direction = np.zeros(len(artifacts[year]["y"]), dtype=np.float64)
            direction[data["target_core"]] = config.gamma * correction
            directions[year][mode] = direction
            print(
                f"[{mode}] {source_year}->{year}: rows={len(correction)} "
                f"std={direction.std():.6f}",
                flush=True,
            )
        directions[2023][f"config_{mode}"] = np.asarray([], dtype=np.float64)

    for year in (2023, 2024):
        directions[year]["consensus"] = 0.5 * (
            directions[year]["loo"] + directions[year]["full"]
        )

    # The HGB family representative was independently fixed by the same two
    # historical transfers.  Rebuild only this one lightweight direction.
    add_raw_columns(frames, artifacts)
    hgb_recipe = next(
        recipe for recipe in model_recipes() if recipe.name == "hgb_l15_leaf1000_full"
    )
    for year, source_year in ((2023, 2022), (2024, 2023)):
        hgb_data = build_data(frames, artifacts, source_year, year, "full")
        hgb = hgb_recipe.factory()
        hgb.fit(hgb_data["x_source"], hgb_data["residual"])
        correction = np.asarray(hgb.predict(hgb_data["x_target"]), dtype=np.float64)
        direction = np.zeros(len(artifacts[year]["y"]), dtype=np.float64)
        direction[hgb_data["target_core"]] = correction
        directions[year]["hgb"] = direction

    y = {year: hybrid[year]["y"].astype(np.float64) for year in (2023, 2024)}
    base = {
        year: hybrid[year]["split_r_selected_f_fixed"].astype(np.float64)
        for year in (2023, 2024)
    }
    base_score = {year: raw_score(y[year], base[year]) for year in (2023, 2024)}
    candidates: dict[str, dict[str, object]] = {}
    payload: dict[str, np.ndarray] = {
        "y": y[2024],
        "row_index": hybrid[2024]["row_index"],
        "cluster": hybrid[2024]["cluster"],
        "hybrid": base[2024],
    }

    for name in ("loo", "full", "consensus", "hgb"):
        gamma = scalar_fit(directions[2023][name], y[2023] - base[2023])
        p23 = base[2023] + gamma * directions[2023][name]
        p24 = base[2024] + gamma * directions[2024][name]
        s23 = raw_score(y[2023], p23)
        s24 = raw_score(y[2024], p24)
        candidate_name = f"hybrid_plus_conditional_{name}"
        candidates[candidate_name] = {
            "selected_scalar": gamma,
            "selection_gain": s23 - base_score[2023],
            "confirmation_gain": s24 - base_score[2024],
            "confirmation_score": s24,
            "expected_lb_median": s24 + MEDIAN_OFFSET,
        }
        payload[candidate_name] = np.clip(p24, 0.0, 1.0)

    for joint_name, direction_names in (
        ("conditional_joint", ("loo", "full")),
        ("consensus_hgb_joint", ("consensus", "hgb")),
        ("conditional_hgb_joint", ("loo", "full", "hgb")),
    ):
        x23 = np.column_stack([directions[2023][name] for name in direction_names])
        x24 = np.column_stack([directions[2024][name] for name in direction_names])
        coefficients = np.linalg.lstsq(x23, y[2023] - base[2023], rcond=None)[0]
        p23 = base[2023] + x23 @ coefficients
        p24 = base[2024] + x24 @ coefficients
        s23 = raw_score(y[2023], p23)
        s24 = raw_score(y[2024], p24)
        candidate_name = f"hybrid_plus_{joint_name}"
        candidates[candidate_name] = {
            "coefficients": {
                name: float(value) for name, value in zip(direction_names, coefficients)
            },
            "selection_gain": s23 - base_score[2023],
            "confirmation_gain": s24 - base_score[2024],
            "confirmation_score": s24,
            "expected_lb_median": s24 + MEDIAN_OFFSET,
        }
        payload[candidate_name] = np.clip(p24, 0.0, 1.0)

    best = max(candidates.items(), key=lambda item: float(item[1]["confirmation_score"]))
    output = PRED / "v4_conditional_ridge_stack_2024.npz"
    np.savez_compressed(
        output,
        **payload,
        direction_loo=directions[2024]["loo"],
        direction_full=directions[2024]["full"],
        direction_consensus=directions[2024]["consensus"],
        direction_hgb=directions[2024]["hgb"],
    )
    selection_output = PRED / "v4_conditional_ridge_stack_2023.npz"
    np.savez_compressed(
        selection_output,
        y=y[2023],
        row_index=hybrid[2023]["row_index"],
        cluster=hybrid[2023]["cluster"],
        hybrid=base[2023],
        direction_loo=directions[2023]["loo"],
        direction_full=directions[2023]["full"],
        direction_consensus=directions[2023]["consensus"],
        direction_hgb=directions[2023]["hgb"],
    )
    report = {
        "protocol": {
            "official_train_only": True,
            "test_rows_read": False,
            "configs_preselected_on_2021_to_2022_and_2022_to_2023": True,
            "scalar_selection_year": 2023,
            "confirmation_year": 2024,
            "route": "R_CORE only",
            "row_independent_inference": True,
        },
        "selected_configs": selected,
        "baseline_scores": base_score,
        "candidates": candidates,
        "best_observed_confirmation_diagnostic": {
            "name": best[0],
            **best[1],
            "required_local_score": REQUIRED_LOCAL,
            "crosses_required_local_score": (
                float(best[1]["confirmation_score"]) > REQUIRED_LOCAL
            ),
            "warning": "2024 ranking is diagnostic, not a selection rule",
        },
        "prediction_artifacts": {
            "2023": str(selection_output.relative_to(ROOT)),
            "2024": str(output.relative_to(ROOT)),
        },
    }
    REPORT.write_text(
        json.dumps(json_safe(report), ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(json_safe(report["best_observed_confirmation_diagnostic"]), indent=2))
    print(f"Saved {REPORT}")


if __name__ == "__main__":
    main()
