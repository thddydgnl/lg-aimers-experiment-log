#!/usr/bin/env python3
"""Multi-task residual ResNets with official failure-component auxiliaries."""

from __future__ import annotations

import itertools
import json
import sys
import time
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.analyze_v4_neural_residual import (  # noqa: E402
    GAMMAS,
    ResidualBlock,
    standardize,
)
from experiments.analyze_v4_pitchtype_failure_prior import (  # noqa: E402
    derive_failure_components,
)
from experiments.analyze_v4_temporal_residual_models import (  # noqa: E402
    add_raw_columns,
    build_data,
)
from experiments.analyze_v4_temporal_residual_ridge import (  # noqa: E402
    MEDIAN_OFFSET,
    REQUIRED_LOCAL,
    json_safe,
    load_frames,
    score,
)
from experiments.v4_current_ensemble import PREDICTIONS  # noqa: E402


OUTPUT_JSON = ROOT / "experiments/results/v4_neural_multitask_residual.json"
OUTPUT_NPZ = PREDICTIONS / "v4_neural_multitask_residual_2024.npz"
TRANSITIONS = ((2021, 2022), (2022, 2023))
CONFIRMATION = (2023, 2024)
AUX_WEIGHTS = (0.025, 0.05, 0.10, 0.20)


class MultiTaskResNet(nn.Module):
    def __init__(self, inputs: int, width: int = 128, blocks: int = 3,
                 dropout: float = 0.12):
        super().__init__()
        self.stem = nn.Linear(inputs, width)
        self.blocks = nn.Sequential(
            *(ResidualBlock(width, dropout) for _ in range(blocks))
        )
        self.norm = nn.LayerNorm(width)
        self.residual_head = nn.Linear(width, 1)
        self.auxiliary_head = nn.Linear(width, 4)
        nn.init.zeros_(self.residual_head.weight)
        nn.init.zeros_(self.residual_head.bias)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        hidden = self.norm(self.blocks(self.stem(x)))
        return self.residual_head(hidden).squeeze(1), self.auxiliary_head(hidden)


def load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path) as archive:
        return {key: np.asarray(archive[key]) for key in archive.files}


def load_locked_base(year: int) -> dict[str, np.ndarray]:
    return load_npz(
        PREDICTIONS / f"v4_pitchtype_failure_tagged_locked_{year}.npz"
    )


def auxiliary_targets(
    artifacts: dict[int, dict[str, np.ndarray]],
) -> tuple[dict[int, np.ndarray], dict[int, np.ndarray]]:
    columns = [
        "row_id",
        "pitcher_id",
        "asof_pitcher_n",
        "asof_pitcher_reverse_rate",
        "asof_pitcher_middle_rate",
        "control_success",
    ]
    full = pd.read_csv(
        ROOT / "open/data/train.csv",
        usecols=columns,
        encoding="utf-8-sig",
        low_memory=False,
    )
    components = derive_failure_components(full)
    target_matrix = components[["success", "reverse", "middle", "wayoff"]].to_numpy(
        dtype=np.float32
    )
    valid = components["component_valid"].to_numpy(dtype=bool)
    targets_by_year = {}
    valid_by_year = {}
    for year, artifact in artifacts.items():
        index = np.asarray(artifact["row_index"], dtype=np.int64)
        targets_by_year[year] = np.nan_to_num(target_matrix[index], nan=0.0)
        valid_by_year[year] = valid[index]
    return targets_by_year, valid_by_year


def train_predict(
    data: dict[str, Any],
    auxiliary: np.ndarray,
    auxiliary_valid: np.ndarray,
    auxiliary_weight: float,
    seed: int = 2026,
) -> tuple[np.ndarray, float]:
    started = time.perf_counter()
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    source_x, target_x = standardize(data["x_source"], data["x_target"])
    residual = np.asarray(data["residual"], dtype=np.float32)
    auxiliary = np.asarray(auxiliary, dtype=np.float32)
    auxiliary_valid = np.asarray(auxiliary_valid, dtype=bool)
    model = MultiTaskResNet(source_x.shape[1]).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=8e-4, weight_decay=5e-4
    )
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    batch_size = 4096
    rng = np.random.default_rng(seed)
    model.train()
    for epoch in range(18):
        order = rng.permutation(len(source_x))
        total = 0.0
        for start in range(0, len(order), batch_size):
            index = order[start:start + batch_size]
            xb = torch.from_numpy(source_x[index]).to(device)
            residual_target = torch.from_numpy(residual[index]).to(device)
            auxiliary_target = torch.from_numpy(auxiliary[index]).to(device)
            valid_target = torch.from_numpy(auxiliary_valid[index]).to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=device.type == "cuda",
            ):
                residual_output, auxiliary_logits = model(xb)
                main_loss = F.mse_loss(residual_output, residual_target)
                if bool(valid_target.any()):
                    auxiliary_loss = F.binary_cross_entropy_with_logits(
                        auxiliary_logits[valid_target],
                        auxiliary_target[valid_target],
                    )
                else:
                    auxiliary_loss = auxiliary_logits.sum() * 0.0
                loss = main_loss + auxiliary_weight * auxiliary_loss
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            scaler.step(optimizer)
            scaler.update()
            total += float(loss.detach()) * len(index)
        if epoch in (0, 17):
            print(
                f"    aux={auxiliary_weight:g} epoch={epoch + 1}/18 "
                f"loss={total / len(source_x):.6f}",
                flush=True,
            )
    model.eval()
    outputs: list[np.ndarray] = []
    with torch.inference_mode():
        for start in range(0, len(target_x), 16384):
            xb = torch.from_numpy(target_x[start:start + 16384]).to(device)
            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=device.type == "cuda",
            ):
                residual_output, _ = model(xb)
            outputs.append(residual_output.float().cpu().numpy())
    correction = np.clip(np.concatenate(outputs).astype(np.float64), -0.20, 0.20)
    del model, optimizer
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return correction, time.perf_counter() - started


def apply_correction(
    baseline: np.ndarray,
    target_core: np.ndarray,
    correction: np.ndarray,
    gamma: float,
) -> np.ndarray:
    prediction = np.asarray(baseline, dtype=np.float64).copy()
    prediction[target_core] = np.clip(
        prediction[target_core] + gamma * correction, 0.0, 1.0
    )
    return prediction


def main() -> None:
    frames, artifacts = load_frames()
    add_raw_columns(frames, artifacts)
    bases = {year: load_locked_base(year) for year in (2022, 2023, 2024)}
    for year, base in bases.items():
        if not np.array_equal(base["row_index"], artifacts[year]["row_index"]):
            raise ValueError(f"Locked-base alignment mismatch for {year}")
    auxiliary, auxiliary_valid = auxiliary_targets(artifacts)

    transition_data = {
        (source, target): build_data(frames, artifacts, source, target, "loo")
        for source, target in (*TRANSITIONS, CONFIRMATION)
    }
    corrections: dict[tuple[float, int, int], np.ndarray] = {}
    selection_rows: list[dict[str, Any]] = []
    for auxiliary_weight in AUX_WEIGHTS:
        elapsed = 0.0
        for source, target in TRANSITIONS:
            data = transition_data[(source, target)]
            source_core = data["source_core"]
            correction, seconds = train_predict(
                data,
                auxiliary[source][source_core],
                auxiliary_valid[source][source_core],
                auxiliary_weight,
            )
            corrections[(auxiliary_weight, source, target)] = correction
            elapsed += seconds
        gamma_trials = []
        for gamma in GAMMAS:
            gains = {}
            for source, target in TRANSITIONS:
                data = transition_data[(source, target)]
                prediction = apply_correction(
                    bases[target]["tagged_locked"],
                    data["target_core"],
                    corrections[(auxiliary_weight, source, target)],
                    gamma,
                )
                candidate = score(artifacts[target]["y"], prediction)
                baseline = score(
                    artifacts[target]["y"], bases[target]["tagged_locked"]
                )
                gains[str(target)] = float(
                    candidate["raw_competition_score"]
                    - baseline["raw_competition_score"]
                )
            gamma_trials.append(
                {
                    "gamma": gamma,
                    "gains": gains,
                    "robust_min_gain": float(min(gains.values())),
                    "mean_gain": float(np.mean(list(gains.values()))),
                }
            )
        selected_gamma = max(
            gamma_trials,
            key=lambda row: (row["robust_min_gain"], row["mean_gain"]),
        )
        selection_rows.append(
            {
                "auxiliary_weight": auxiliary_weight,
                "selected": selected_gamma,
                "top_gamma_trials": sorted(
                    gamma_trials,
                    key=lambda row: (row["robust_min_gain"], row["mean_gain"]),
                    reverse=True,
                )[:10],
                "fit_seconds": elapsed,
            }
        )
        print(
            f"  aux={auxiliary_weight:g} gamma={selected_gamma['gamma']:+.3f} "
            f"min={selected_gamma['robust_min_gain']:+.4f}",
            flush=True,
        )

    selected = max(
        selection_rows,
        key=lambda row: (
            row["selected"]["robust_min_gain"],
            row["selected"]["mean_gain"],
        ),
    )
    auxiliary_weight = float(selected["auxiliary_weight"])
    gamma = float(selected["selected"]["gamma"])
    source, target = CONFIRMATION
    confirmation_data = transition_data[CONFIRMATION]
    confirmation_correction, confirmation_seconds = train_predict(
        confirmation_data,
        auxiliary[source][confirmation_data["source_core"]],
        auxiliary_valid[source][confirmation_data["source_core"]],
        auxiliary_weight,
    )
    prediction = apply_correction(
        bases[target]["tagged_locked"],
        confirmation_data["target_core"],
        confirmation_correction,
        gamma,
    )
    baseline_metrics = score(
        artifacts[target]["y"], bases[target]["tagged_locked"]
    )
    confirmation_metrics = score(artifacts[target]["y"], prediction)
    gain = float(
        confirmation_metrics["raw_competition_score"]
        - baseline_metrics["raw_competition_score"]
    )
    np.savez_compressed(
        OUTPUT_NPZ,
        y=artifacts[target]["y"],
        row_index=artifacts[target]["row_index"],
        cluster=artifacts[target]["cluster"],
        base=bases[target]["tagged_locked"],
        target_core=confirmation_data["target_core"],
        raw_correction=confirmation_correction,
        gamma=np.asarray(gamma),
        neural_multitask=prediction,
    )
    report = {
        "protocol": {
            "official_train_only": True,
            "test_rows_read": False,
            "failure_auxiliaries_recovered_from_official_asof_counters": True,
            "selection_transitions": [list(value) for value in TRANSITIONS],
            "confirmation_transition": list(CONFIRMATION),
            "row_independent_inference": True,
        },
        "recipes": selection_rows,
        "selected": selected,
        "confirmation_fit_seconds": confirmation_seconds,
        "confirmation_2024": {
            "gain": gain,
            "metrics": confirmation_metrics,
            "expected_lb_median": (
                confirmation_metrics["raw_competition_score"] + MEDIAN_OFFSET
            ),
            "required_local_score": REQUIRED_LOCAL,
            "crosses_required_local_score": bool(
                confirmation_metrics["raw_competition_score"] > REQUIRED_LOCAL
            ),
        },
        "prediction_artifact": str(OUTPUT_NPZ.relative_to(ROOT)),
    }
    OUTPUT_JSON.write_text(
        json.dumps(json_safe(report), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "selected_auxiliary_weight": auxiliary_weight,
                "selected_gamma": gamma,
                "selection_gains": selected["selected"]["gains"],
                "confirmation_gain": gain,
                "score_2024": confirmation_metrics["raw_competition_score"],
                "expected_lb_median": (
                    confirmation_metrics["raw_competition_score"] + MEDIAN_OFFSET
                ),
            },
            ensure_ascii=False,
            indent=2,
        ),
        flush=True,
    )
    print(f"Saved {OUTPUT_JSON}", flush=True)
    print(f"Saved {OUTPUT_NPZ}", flush=True)


if __name__ == "__main__":
    main()
