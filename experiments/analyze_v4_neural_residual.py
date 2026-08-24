#!/usr/bin/env python3
"""Season-transfer neural residual experiments for the V4 ensemble.

The networks are trained on one completed season and applied to the next
season.  Architecture and correction strength are selected only by the worst
gain over 2021->2022 and 2022->2023.  The selected recipes are then refit on
2023 and confirmed once on 2024.  Only official train rows and frozen OOF
prediction artifacts are read.
"""

from __future__ import annotations

import json
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.analyze_v4_temporal_residual_models import (  # noqa: E402
    add_raw_columns,
    build_data,
)
from experiments.analyze_v4_temporal_residual_ridge import (  # noqa: E402
    MEDIAN_OFFSET,
    M3_WEIGHTS,
    REQUIRED_LOCAL,
    Config,
    json_safe,
    load_frames,
    score,
    transfer_data,
)


PREDICTIONS = ROOT / "experiments/results/predictions"
OUTPUT_JSON = ROOT / "experiments/results/v4_neural_residual.json"
OUTPUT_NPZ = PREDICTIONS / "v4_neural_residual_2024.npz"
TRANSITIONS = ((2021, 2022), (2022, 2023))
GAMMAS = (-1.00, -0.75, -0.55, -0.40, -0.30, -0.20, -0.15, -0.10,
          -0.08, -0.05, -0.03, -0.02, 0.0, 0.02, 0.03, 0.05, 0.08,
          0.10, 0.15, 0.20, 0.30, 0.40, 0.55, 0.75, 1.00)
CONTEXT_WEIGHT = 0.15
LEVEL_WEIGHT = 0.50
STABILITY_C_WEIGHT = 1.05
STABILITY_B_WEIGHT = 0.925


@dataclass(frozen=True)
class Recipe:
    name: str
    feature_set: str
    architecture: str
    loss: str
    training_mode: str = "loo"
    width: int = 128
    blocks: int = 2
    dropout: float = 0.12
    epochs: int = 18
    learning_rate: float = 8e-4
    weight_decay: float = 5e-4
    seeds: tuple[int, ...] = (2026,)


def recipes() -> list[Recipe]:
    return [
        Recipe("stable_mlp_mse", "stable", "mlp", "mse", width=96),
        Recipe("aug_mlp_mse", "augmented", "mlp", "mse", width=128),
        Recipe("aug_resnet_mse", "augmented", "resnet", "mse", width=128,
               blocks=3),
        Recipe("aug_gated_mse", "augmented", "gated", "mse", width=128,
               blocks=3),
        Recipe("aug_resnet_bce", "augmented", "resnet", "bce", width=128,
               blocks=3),
        Recipe("aug_resnet_mse_seed3", "augmented", "resnet", "mse",
               width=128, blocks=3, seeds=(2026, 7, 42)),
    ]


def load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path) as archive:
        return {key: np.asarray(archive[key]) for key in archive.files}


def aligned(reference: dict[str, np.ndarray], candidate: dict[str, np.ndarray],
            label: str) -> None:
    for key in ("y", "row_index"):
        if not np.array_equal(reference[key], candidate[key]):
            raise ValueError(f"Artifact alignment mismatch for {label}/{key}")


def current_ensemble(season: int,
                     artifact: dict[str, np.ndarray]) -> np.ndarray:
    """Reconstruct the pre-neural V4 ensemble with preselected weights."""
    residual = load_npz(PREDICTIONS / f"v4_residual_ensemble_{season}.npz")
    aligned(artifact, residual, f"residual/{season}")
    numeric: dict[str, np.ndarray] = {}
    for key, stem in (
        ("base", "v4_numeric_cat_current_tmctx_seed42"),
        ("context", "v4_numeric_cat_current_context_tmctx_seed42"),
        ("level", "v4_numeric_cat_current_context_level_tmctx_seed42"),
    ):
        item = load_npz(PREDICTIONS / f"{stem}_{season}.npz")
        aligned(artifact, item, f"{stem}/{season}")
        numeric[key] = np.asarray(item["catboost_numeric"], dtype=np.float64)

    c_stem = ("v4_outcome_c_trackman_stability_backtest" if season < 2024
              else "v4_outcome_c_trackman_stability")
    b_stem = ("v4_outcome_b_trackman_stability_backtest" if season < 2024
              else "v4_outcome_b_trackman_stability")
    c_item = load_npz(PREDICTIONS / f"{c_stem}_{season}.npz")
    b_item = load_npz(PREDICTIONS / f"{b_stem}_{season}.npz")
    aligned(artifact, c_item, f"{c_stem}/{season}")
    aligned(artifact, b_item, f"{b_stem}/{season}")

    context_delta = numeric["context"] - numeric["base"]
    level_delta = numeric["level"] - numeric["context"]
    c_delta = (np.asarray(c_item["catboost_outcome"], dtype=np.float64)
               - np.asarray(artifact["component_C"], dtype=np.float64))
    b_delta = (np.asarray(b_item["catboost_outcome"], dtype=np.float64)
               - np.asarray(artifact["component_B"], dtype=np.float64))
    prediction = (
        np.asarray(residual["residual_ensemble"], dtype=np.float64)
        + CONTEXT_WEIGHT * context_delta
        + LEVEL_WEIGHT * level_delta
        + 1.05 * STABILITY_C_WEIGHT * M3_WEIGHTS["C"] * c_delta
        + 1.05 * STABILITY_B_WEIGHT * M3_WEIGHTS["B"] * b_delta
    )
    return np.clip(prediction, 0.0, 1.0)


def build_feature_data(frames: dict[int, Any],
                       artifacts: dict[int, dict[str, np.ndarray]],
                       source: int, target: int,
                       feature_set: str, mode: str) -> dict[str, Any]:
    if feature_set == "stable":
        config = Config("r_all", 800.0, 800.0, 1600.0, 1.0, 1.0, mode)
        return transfer_data(frames[source], frames[target],
                             artifacts[source]["m3"], config)
    if feature_set == "augmented":
        return build_data(frames, artifacts, source, target, mode)
    raise ValueError(f"Unknown feature set: {feature_set}")


class MLP(nn.Module):
    def __init__(self, inputs: int, width: int, blocks: int, dropout: float):
        super().__init__()
        layers: list[nn.Module] = []
        dimension = inputs
        for index in range(blocks):
            out = width if index < blocks - 1 else max(32, width // 2)
            layers.extend((nn.Linear(dimension, out), nn.SiLU(),
                           nn.LayerNorm(out), nn.Dropout(dropout)))
            dimension = out
        self.body = nn.Sequential(*layers)
        self.head = nn.Linear(dimension, 1)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.body(x)).squeeze(1)


class ResidualBlock(nn.Module):
    def __init__(self, width: int, dropout: float):
        super().__init__()
        self.norm = nn.LayerNorm(width)
        self.layers = nn.Sequential(
            nn.Linear(width, width * 2), nn.SiLU(), nn.Dropout(dropout),
            nn.Linear(width * 2, width), nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.layers(self.norm(x))


class ResNet(nn.Module):
    def __init__(self, inputs: int, width: int, blocks: int, dropout: float):
        super().__init__()
        self.stem = nn.Linear(inputs, width)
        self.blocks = nn.Sequential(
            *(ResidualBlock(width, dropout) for _ in range(blocks))
        )
        self.norm = nn.LayerNorm(width)
        self.head = nn.Linear(width, 1)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.norm(self.blocks(self.stem(x)))).squeeze(1)


class GatedBlock(nn.Module):
    def __init__(self, width: int, dropout: float):
        super().__init__()
        self.norm = nn.LayerNorm(width)
        self.gate = nn.Linear(width, width * 2)
        self.out = nn.Sequential(nn.Linear(width, width), nn.Dropout(dropout))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        left, right = self.gate(self.norm(x)).chunk(2, dim=1)
        return x + self.out(F.silu(left) * torch.sigmoid(right))


class GatedNet(nn.Module):
    def __init__(self, inputs: int, width: int, blocks: int, dropout: float):
        super().__init__()
        self.stem = nn.Linear(inputs, width)
        self.blocks = nn.Sequential(
            *(GatedBlock(width, dropout) for _ in range(blocks))
        )
        self.norm = nn.LayerNorm(width)
        self.head = nn.Linear(width, 1)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.norm(self.blocks(self.stem(x)))).squeeze(1)


def make_model(recipe: Recipe, inputs: int) -> nn.Module:
    classes = {"mlp": MLP, "resnet": ResNet, "gated": GatedNet}
    return classes[recipe.architecture](inputs, recipe.width, recipe.blocks,
                                        recipe.dropout)


def standardize(source: np.ndarray,
                target: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    source = np.asarray(source, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    source = np.where(np.isfinite(source), source, np.nan)
    target = np.where(np.isfinite(target), target, np.nan)
    center = np.nanmedian(source, axis=0)
    center[~np.isfinite(center)] = 0.0
    source = np.where(np.isfinite(source), source, center)
    target = np.where(np.isfinite(target), target, center)
    scale = np.std(source, axis=0)
    scale[~np.isfinite(scale) | (scale < 1e-6)] = 1.0
    source = np.clip((source - center) / scale, -12.0, 12.0)
    target = np.clip((target - center) / scale, -12.0, 12.0)
    return source.astype(np.float32), target.astype(np.float32)


def train_once(recipe: Recipe, data: dict[str, Any],
               source_baseline: np.ndarray, target_baseline: np.ndarray,
               seed: int) -> np.ndarray:
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x_source, x_target = standardize(data["x_source"], data["x_target"])
    source_core = data["source_core"]
    target_core = data["target_core"]
    y = np.asarray(data["residual"], dtype=np.float32)
    source_base = np.asarray(source_baseline[source_core], dtype=np.float32)
    target_base = np.asarray(target_baseline[target_core], dtype=np.float32)
    model = make_model(recipe, x_source.shape[1]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=recipe.learning_rate,
                                  weight_decay=recipe.weight_decay)
    batch_size = 4096
    amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    rng = np.random.default_rng(seed)
    model.train()
    for epoch in range(recipe.epochs):
        order = rng.permutation(len(x_source))
        total = 0.0
        for start in range(0, len(order), batch_size):
            index = order[start:start + batch_size]
            xb = torch.from_numpy(x_source[index]).to(device)
            residual = torch.from_numpy(y[index]).to(device)
            base = torch.from_numpy(source_base[index]).to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.float16,
                                enabled=amp):
                output = model(xb)
                if recipe.loss == "mse":
                    loss = F.mse_loss(output, residual)
                elif recipe.loss == "bce":
                    label = torch.clamp(base + residual, 0.0, 1.0)
                    base_logit = torch.logit(base.clamp(1e-5, 1.0 - 1e-5))
                    combined_logit = base_logit + output
                    probability = torch.sigmoid(combined_logit)
                    loss = (F.binary_cross_entropy_with_logits(
                                combined_logit, label)
                            + 0.25 * F.mse_loss(probability, label)
                            + 1e-3 * output.square().mean())
                else:
                    raise ValueError(recipe.loss)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            scaler.step(optimizer)
            scaler.update()
            total += float(loss.detach()) * len(index)
        if epoch in (0, recipe.epochs - 1):
            print(f"    seed={seed} epoch={epoch + 1}/{recipe.epochs} "
                  f"loss={total / len(x_source):.6f}", flush=True)

    model.eval()
    outputs: list[np.ndarray] = []
    with torch.inference_mode():
        for start in range(0, len(x_target), 16384):
            xb = torch.from_numpy(x_target[start:start + 16384]).to(device)
            with torch.autocast(device_type=device.type, dtype=torch.float16,
                                enabled=amp):
                output = model(xb)
            outputs.append(output.float().cpu().numpy())
    raw = np.concatenate(outputs).astype(np.float64)
    if recipe.loss == "bce":
        base_logit = np.log(np.clip(target_base, 1e-5, 1.0 - 1e-5)
                            / np.clip(1.0 - target_base, 1e-5, 1.0))
        probability = 1.0 / (1.0 + np.exp(-np.clip(base_logit + raw, -20, 20)))
        correction = probability - target_base
    else:
        correction = raw
    del model, optimizer
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return np.clip(correction, -0.20, 0.20)


def fit_correction(recipe: Recipe, data: dict[str, Any],
                   source_baseline: np.ndarray,
                   target_baseline: np.ndarray) -> tuple[np.ndarray, float]:
    started = time.perf_counter()
    members = [train_once(recipe, data, source_baseline, target_baseline, seed)
               for seed in recipe.seeds]
    return np.mean(members, axis=0), time.perf_counter() - started


def corrected(baseline: np.ndarray, core: np.ndarray,
              correction: np.ndarray, gamma: float) -> np.ndarray:
    result = np.asarray(baseline, dtype=np.float64).copy()
    result[core] = np.clip(result[core] + gamma * correction, 0.0, 1.0)
    return result


def main() -> None:
    frames, artifacts = load_frames()
    add_raw_columns(frames, artifacts)
    current = {season: current_ensemble(season, artifacts[season])
               for season in (2022, 2023, 2024)}
    baselines = {season: score(artifacts[season]["y"], prediction)
                 for season, prediction in current.items()}
    all_data: dict[tuple[str, str, int, int], dict[str, Any]] = {}
    for recipe in recipes():
        key = (recipe.feature_set, recipe.training_mode)
        for source, target in (*TRANSITIONS, (2023, 2024)):
            full_key = (*key, source, target)
            if full_key not in all_data:
                all_data[full_key] = build_feature_data(
                    frames, artifacts, source, target, *key
                )

    selections: list[dict[str, Any]] = []
    correction_cache: dict[tuple[str, int, int], np.ndarray] = {}
    for index, recipe in enumerate(recipes(), start=1):
        print(f"[{index}/{len(recipes())}] {recipe.name}", flush=True)
        transition_seconds = 0.0
        for source, target in TRANSITIONS:
            data = all_data[(recipe.feature_set, recipe.training_mode,
                             source, target)]
            correction, elapsed = fit_correction(
                recipe, data, artifacts[source]["m3"], artifacts[target]["m3"]
            )
            correction_cache[(recipe.name, source, target)] = correction
            transition_seconds += elapsed
        trials: list[dict[str, Any]] = []
        for gamma in GAMMAS:
            gains: dict[str, float] = {}
            metrics: dict[str, Any] = {}
            for source, target in TRANSITIONS:
                data = all_data[(recipe.feature_set, recipe.training_mode,
                                 source, target)]
                pred = corrected(current[target], data["target_core"],
                                 correction_cache[(recipe.name, source, target)],
                                 gamma)
                metric = score(artifacts[target]["y"], pred)
                gains[str(target)] = float(
                    metric["raw_competition_score"]
                    - baselines[target]["raw_competition_score"]
                )
                metrics[str(target)] = metric
            trials.append({
                "gamma": gamma,
                "gains": gains,
                "robust_min_gain": float(min(gains.values())),
                "mean_gain": float(np.mean(list(gains.values()))),
                "metrics": metrics,
            })
        selected = max(trials,
                       key=lambda row: (row["robust_min_gain"], row["mean_gain"]))
        row = {
            "recipe": recipe.__dict__,
            "selected_gamma": selected["gamma"],
            "robust_min_gain": selected["robust_min_gain"],
            "mean_gain": selected["mean_gain"],
            "selection": selected,
            "top_trials": sorted(
                trials,
                key=lambda item: (item["robust_min_gain"], item["mean_gain"]),
                reverse=True,
            )[:8],
            "selection_fit_seconds": transition_seconds,
        }
        selections.append(row)
        print(f"  selected gamma={selected['gamma']:.2f} "
              f"min={selected['robust_min_gain']:+.4f} "
              f"mean={selected['mean_gain']:+.4f} "
              f"gains={selected['gains']}", flush=True)

    ranked = sorted(selections,
                    key=lambda row: (row["robust_min_gain"], row["mean_gain"]),
                    reverse=True)
    lookup = {recipe.name: recipe for recipe in recipes()}
    confirm_names = [row["recipe"]["name"] for row in ranked[:3]]
    confirmations: dict[str, Any] = {}
    predictions: dict[str, np.ndarray] = {}
    for name in confirm_names:
        recipe = lookup[name]
        selected = next(row for row in ranked if row["recipe"]["name"] == name)
        data = all_data[(recipe.feature_set, recipe.training_mode, 2023, 2024)]
        correction, elapsed = fit_correction(
            recipe, data, artifacts[2023]["m3"], artifacts[2024]["m3"]
        )
        prediction = corrected(current[2024], data["target_core"], correction,
                               float(selected["selected_gamma"]))
        metric = score(artifacts[2024]["y"], prediction)
        gain = float(metric["raw_competition_score"]
                     - baselines[2024]["raw_competition_score"])
        confirmations[name] = {
            "metrics": metric,
            "gain": gain,
            "expected_lb_median": float(
                metric["raw_competition_score"] + MEDIAN_OFFSET
            ),
            "crosses_required_local_score": bool(
                metric["raw_competition_score"] > REQUIRED_LOCAL
            ),
            "fit_seconds": elapsed,
            "correction_mean": float(correction.mean()),
            "correction_std": float(correction.std()),
            "correction_max_abs": float(np.max(np.abs(correction))),
        }
        predictions[name] = prediction
        print(f"[confirm] {name}: gain={gain:+.4f} "
              f"local={metric['raw_competition_score']:.4f}", flush=True)

    primary = ranked[0]["recipe"]["name"]
    payload: dict[str, np.ndarray] = {
        "y": artifacts[2024]["y"],
        "row_index": artifacts[2024]["row_index"],
        "cluster": artifacts[2024]["cluster"],
        "m3": artifacts[2024]["m3"],
        "current_ensemble": current[2024],
        "neural_residual": predictions[primary],
    }
    for name, prediction in predictions.items():
        payload[f"candidate_{name}"] = prediction
    np.savez_compressed(OUTPUT_NPZ, **payload)
    report = {
        "protocol": {
            "official_train_only": True,
            "test_rows_read": False,
            "leaderboard_values_used": False,
            "row_independent_inference": True,
            "selection": "maximize worst gain on 2021->2022 and 2022->2023",
            "confirmation": "refit on 2023 and apply once to 2024",
            "neural_device": "cuda" if torch.cuda.is_available() else "cpu",
            "torch_version": torch.__version__,
        },
        "fixed_estimator": {
            "median_offset": MEDIAN_OFFSET,
            "required_local_score": REQUIRED_LOCAL,
            "target_lb": 1190.0,
        },
        "current_ensemble_weights": {
            "context": CONTEXT_WEIGHT,
            "level": LEVEL_WEIGHT,
            "trackman_stability_c": STABILITY_C_WEIGHT,
            "trackman_stability_b": STABILITY_B_WEIGHT,
        },
        "baselines": baselines,
        "ranked_selection": ranked,
        "primary_name": primary,
        "confirmations_2024": confirmations,
        "prediction_artifact": str(OUTPUT_NPZ.relative_to(ROOT)),
    }
    OUTPUT_JSON.write_text(json.dumps(json_safe(report), ensure_ascii=False,
                                      indent=2), encoding="utf-8")
    print(json.dumps({
        "primary": primary,
        "selection_min_gain": ranked[0]["robust_min_gain"],
        "selection_mean_gain": ranked[0]["mean_gain"],
        "confirmation": confirmations[primary],
    }, ensure_ascii=False, indent=2))
    print(f"Saved {OUTPUT_JSON}")
    print(f"Saved {OUTPUT_NPZ}")


if __name__ == "__main__":
    main()
