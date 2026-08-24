#!/usr/bin/env python3
"""Persist the preselected neural ResNet residual delta for 2022-2024."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.analyze_v4_neural_residual import (  # noqa: E402
    Recipe,
    build_feature_data,
    fit_correction,
)
from experiments.analyze_v4_temporal_residual_models import (  # noqa: E402
    add_raw_columns,
)
from experiments.analyze_v4_temporal_residual_ridge import (  # noqa: E402
    load_frames,
    score,
)


PREDICTIONS = ROOT / "experiments/results/predictions"
OUTPUT_JSON = ROOT / "experiments/results/v4_neural_resnet_oof.json"
GAMMA = -0.02
RECIPE = Recipe(
    "aug_resnet_mse",
    "augmented",
    "resnet",
    "mse",
    width=128,
    blocks=3,
)


def main() -> None:
    frames, artifacts = load_frames()
    add_raw_columns(frames, artifacts)
    transitions = ((2021, 2022), (2022, 2023), (2023, 2024))
    report = {
        "protocol": {
            "official_train_only": True,
            "test_rows_read": False,
            "leaderboard_values_used": False,
            "recipe_and_gamma_preselected": True,
            "gamma": GAMMA,
        },
        "recipe": RECIPE.__dict__,
        "folds": {},
    }
    for source, target in transitions:
        data = build_feature_data(
            frames, artifacts, source, target, RECIPE.feature_set,
            RECIPE.training_mode
        )
        correction, elapsed = fit_correction(
            RECIPE, data, artifacts[source]["m3"], artifacts[target]["m3"]
        )
        delta = np.zeros(len(artifacts[target]["y"]), dtype=np.float64)
        delta[data["target_core"]] = GAMMA * correction
        champion_path = PREDICTIONS / f"v4_champion_pre_stack_{target}.npz"
        with np.load(champion_path) as archive:
            champion = np.asarray(archive["champion"], dtype=np.float64)
        candidate = np.clip(champion + delta, 0.0, 1.0)
        base_metric = score(artifacts[target]["y"], champion)
        metric = score(artifacts[target]["y"], candidate)
        path = PREDICTIONS / f"v4_neural_resnet_delta_{target}.npz"
        np.savez_compressed(
            path,
            y=artifacts[target]["y"],
            row_index=artifacts[target]["row_index"],
            cluster=artifacts[target]["cluster"],
            neural_delta=delta,
            champion=champion,
            candidate=candidate,
        )
        report["folds"][str(target)] = {
            "source": source,
            "fit_seconds": elapsed,
            "baseline": base_metric,
            "candidate": metric,
            "gain": float(
                metric["raw_competition_score"]
                - base_metric["raw_competition_score"]
            ),
            "delta_mean": float(delta.mean()),
            "delta_std": float(delta.std()),
            "artifact": str(path.relative_to(ROOT)),
        }
        print(f"[{source}->{target}] gain={report['folds'][str(target)]['gain']:+.4f} "
              f"local={metric['raw_competition_score']:.4f}", flush=True)
    OUTPUT_JSON.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"Saved {OUTPUT_JSON}")


if __name__ == "__main__":
    main()
