#!/usr/bin/env python3
"""Apply the preregistered 2021 source gate to hybrid LUPI targets."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.stats import paired_bootstrap_brier_ci  # noqa: E402

PRED = ROOT / "experiments" / "results" / "predictions"
RESULT = ROOT / "experiments" / "results" / "v5_lupi_hybrid_source_selection.json"
PREREG = ROOT / "experiments" / "params" / "v5_lupi_hybrid_denoising_preregister.json"
TRAIN = ROOT / "open" / "data" / "train.csv"
STAGES = {
    0.0: "v5_lupi_hybrid_source21_a0",
    0.5: "v5_lupi_hybrid_source21_a50",
    1.0: "v5_lupi_hybrid_source21_a100",
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def score(y: np.ndarray, prediction: np.ndarray) -> float:
    reference = float(y.mean() * (1.0 - y.mean()))
    brier = float(np.mean(np.square(prediction - y)))
    return max(0.0, 100_000.0 * (1.0 - brier / reference))


def main() -> None:
    prereg = json.loads(PREREG.read_text(encoding="utf-8"))
    with np.load(PRED / "v4_m3_c_backtest_2021_2021.npz", allow_pickle=False) as z:
        y = z["y"].astype(np.float64)
        row_index = z["row_index"].astype(np.int64)
        cluster = z["cluster"].astype(str)
        parent = z["catboost_outcome"].astype(np.float64)
    game_type = pd.read_csv(TRAIN, usecols=["game_type"])["game_type"]
    regular = game_type.iloc[row_index].astype(str).eq("R").to_numpy()
    models: dict[float, np.ndarray] = {}
    artifact_hashes = {}
    for alpha, stage in STAGES.items():
        path = PRED / f"{stage}_2021.npz"
        with np.load(path, allow_pickle=False) as z:
            for key, expected in (("y", y), ("row_index", row_index), ("cluster", cluster)):
                if not np.array_equal(z[key], expected):
                    raise ValueError(f"{stage}: alignment mismatch for {key}")
            models[alpha] = z["catboost_teacher"].astype(np.float64)
        artifact_hashes[str(path.relative_to(ROOT)).replace("\\", "/")] = sha256(path)

    candidates = []
    alpha1_by_weight = {}
    for weight in prereg["source_gate"]["blend_weights"]:
        hard = parent.copy()
        hard[regular] = (1.0 - weight) * parent[regular] + weight * models[1.0][regular]
        alpha1_by_weight[float(weight)] = hard
    for alpha in prereg["source_gate"]["target_mix_alphas"]:
        alpha = float(alpha)
        for weight in prereg["source_gate"]["blend_weights"]:
            weight = float(weight)
            prediction = parent.copy()
            prediction[regular] = (
                (1.0 - weight) * parent[regular] + weight * models[alpha][regular]
            )
            hard = alpha1_by_weight[weight]
            candidates.append(
                {
                    "alpha": alpha,
                    "weight": weight,
                    "r_gain": score(y[regular], prediction[regular])
                    - score(y[regular], parent[regular]),
                    "full_gain": score(y, prediction) - score(y, parent),
                    "gain_over_alpha1_same_weight_R": score(
                        y[regular], prediction[regular]
                    )
                    - score(y[regular], hard[regular]),
                    "prediction": prediction,
                }
            )
    selected = max(
        candidates,
        key=lambda item: (
            item["r_gain"], item["full_gain"], item["alpha"], -item["weight"]
        ),
    )
    interval = paired_bootstrap_brier_ci(
        y[regular],
        parent[regular],
        selected["prediction"][regular],
        iterations=2000,
        seed=52601,
        clusters=cluster[regular],
    )
    requirements = {
        "r_gain_at_least_25": bool(selected["r_gain"] >= 25.0),
        "r_cluster_ci_lower_positive": bool(interval["score_ci_low"] > 0.0),
        "routed_full_gain_positive": bool(selected["full_gain"] > 0.0),
        "beats_alpha1_at_same_weight": bool(
            selected["gain_over_alpha1_same_weight_R"] > 0.0
        ),
    }
    passed = bool(all(requirements.values()))
    clean_candidates = [
        {key: value for key, value in item.items() if key != "prediction"}
        for item in candidates
    ]
    selected_clean = {key: value for key, value in selected.items() if key != "prediction"}
    report = {
        "experiment_id": prereg["experiment_id"],
        "status": "passed_source_gate" if passed else "failed_source_gate",
        "preregister_sha256": sha256(PREREG),
        "source_year": 2021,
        "years_not_read_for_this_axis": [2022, 2023, 2024],
        "parent": {
            "full_score": score(y, parent),
            "r_score": score(y[regular], parent[regular]),
        },
        "standalone": {
            str(alpha): {
                "full_score": score(y, prediction),
                "r_score": score(y[regular], prediction[regular]),
            }
            for alpha, prediction in models.items()
        },
        "candidates": clean_candidates,
        "selected": selected_clean,
        "selected_r_cluster_interval": interval,
        "requirements": requirements,
        "gate_pass": passed,
        "decision": (
            "freeze for 2022 development" if passed else "close without 2022+"
        ),
        "artifact_hashes": artifact_hashes,
    }
    RESULT.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({
        "status": report["status"],
        "selected": selected_clean,
        "interval": interval,
        "requirements": requirements,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
