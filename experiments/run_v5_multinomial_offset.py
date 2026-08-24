#!/usr/bin/env python3
"""Strictly temporal multinomial current-state offset experiment.

The model starts from four row-local probabilities reconstructed from the
official cumulative pitcher counters.  A small additive model may only adjust
those logits with stable, current-row context.  It is trained on regular-season
rows strictly before each validation season and never reads another validation
row while producing a prediction.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
from pathlib import Path
import random
import sys
import time
from typing import Any

import numpy as np
# Load PyTorch's native runtime before pandas/sklearn/LightGBM.  Loading it
# after the 1.5M-row frame and the other Windows native runtimes intermittently
# fails with WinError 1114 even though the same installation imports cleanly.
import torch
from torch import nn
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.run_e14_rolling import build_e14_features  # noqa: E402
from experiments.run_v2_rolling import (  # noqa: E402
    build_component_features,
    build_hierarchical_entity_features,
    candidate_priors_before_each_season,
    component_states_before_each_season,
    derive_control_outcome_labels,
    entity_season_end_state,
)


TRAIN = ROOT / "open/data/train.csv"
PREREG = ROOT / "experiments/params/v5_multinomial_offset_preregister.json"
RESULTS = ROOT / "experiments/results"
PREDICTIONS = RESULTS / "predictions"
CLASS_NAMES = ("success", "reverse", "middle", "wide")
CATEGORICAL = (
    "count_state",
    "hand_matchup",
    "game_month_cat",
    "inning_band",
    "top_bottom_cat",
    "outs_cat",
    "base_state_cat",
    "runner_count_cat",
    "score_diff_bin_cat",
)
RAW_CONTINUOUS = (
    "asof_pitcher_success_rate",
    "asof_pitcher_reverse_rate",
    "asof_pitcher_middle_rate",
    "asof_pitcher_ball_rate",
    "asof_pitcher_strike_rate",
    "asof_pitcher_prev1_game_success_rate",
    "asof_pitcher_prev3_game_success_rate",
    "asof_pitcher_prev5_game_success_rate",
    "asof_pitcher_prev1_game_middle_rate",
    "asof_pitcher_prev3_game_middle_rate",
    "asof_pitcher_prev5_game_middle_rate",
    "asof_batter_success_rate",
    "asof_batter_middle_rate",
    "asof_pitcher_fastball_rate",
    "asof_pitcher_breaking_rate",
    "asof_pitcher_offspeed_rate",
    "li",
    "score_diff_pitcher_team",
    "num_runners_on",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", required=True)
    parser.add_argument("--validation-seasons", nargs="+", type=int, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    return parser.parse_args()


def score(y: np.ndarray, prediction: np.ndarray) -> dict[str, float | int]:
    y = np.asarray(y, dtype=np.float64)
    prediction = np.asarray(prediction, dtype=np.float64)
    target_rate = float(y.mean())
    reference = target_rate * (1.0 - target_rate)
    brier = float(np.mean(np.square(y - prediction)))
    raw = 100000.0 * (1.0 - brier / reference)
    return {
        "rows": int(len(y)),
        "target_rate": target_rate,
        "prediction_mean": float(prediction.mean()),
        "prediction_std": float(prediction.std()),
        "brier": brier,
        "raw_competition_score": float(raw),
        "competition_score": float(max(0.0, raw)),
    }


def contextual_frame(frame: pd.DataFrame) -> pd.DataFrame:
    result = pd.DataFrame(index=frame.index)
    result["count_state"] = (
        frame["balls_before"].astype(str) + "-" + frame["strikes_before"].astype(str)
    )
    result["hand_matchup"] = (
        frame["pitcher_hand"].astype(str) + "-" + frame["batter_hand"].astype(str)
    )
    result["game_month_cat"] = frame["game_month"].astype(str)
    inning = pd.to_numeric(frame["inning"], errors="coerce").fillna(0).to_numpy()
    result["inning_band"] = np.select(
        [inning <= 3, inning <= 6, inning <= 9],
        ["01_03", "04_06", "07_09"],
        default="10_plus",
    )
    result["top_bottom_cat"] = frame["top_bottom"].astype("string").fillna("missing")
    result["outs_cat"] = frame["outs_before"].astype(str)
    result["base_state_cat"] = frame["base_state"].astype("string").fillna("missing")
    result["runner_count_cat"] = frame["num_runners_on"].astype(str)
    score_diff = pd.to_numeric(
        frame["score_diff_pitcher_team"], errors="coerce"
    ).fillna(0.0).to_numpy()
    result["score_diff_bin_cat"] = np.clip(score_diff, -3, 3).astype(int).astype(str)
    return result


def state_probabilities(
    frame: pd.DataFrame,
    states_before: dict[int, dict[int, tuple[int, int]]],
    priors: dict[int, float],
    fallback_prior: float,
    component_states: dict[int, dict[int, tuple[int, ...]]],
    component_priors: dict[int, dict[str, float]],
    fallback_component_priors: dict[str, float],
) -> tuple[np.ndarray, pd.DataFrame, dict[str, Any]]:
    hierarchical, hierarchical_meta = build_hierarchical_entity_features(
        frame,
        states_before,
        priors,
        fallback_prior,
        "pitcher_id",
        "asof_pitcher_n",
        "asof_pitcher_success_rate",
        "v5m",
        history_k=1000.0,
        current_ks=(80.0,),
    )
    components, component_meta = build_component_features(
        frame,
        component_states,
        component_priors,
        fallback_component_priors,
        80.0,
    )
    success = np.clip(
        hierarchical["v5m_posterior_k80"].to_numpy(dtype=np.float64),
        1e-4,
        1.0 - 1e-4,
    )
    failure_parts = np.column_stack(
        [
            np.clip(
                components["e31_reverse_rate_season"].to_numpy(dtype=np.float64),
                1e-5,
                None,
            ),
            np.clip(
                components["e31_middle_rate_season"].to_numpy(dtype=np.float64),
                1e-5,
                None,
            ),
            np.clip(
                1.0
                - success
                - components["e31_reverse_rate_season"].to_numpy(dtype=np.float64)
                - components["e31_middle_rate_season"].to_numpy(dtype=np.float64),
                1e-5,
                None,
            ),
        ]
    )
    failure_parts /= failure_parts.sum(axis=1, keepdims=True)
    probabilities = np.column_stack(
        [success, (1.0 - success)[:, None] * failure_parts]
    )
    probabilities = np.clip(probabilities, 1e-6, 1.0)
    probabilities /= probabilities.sum(axis=1, keepdims=True)
    derived = pd.DataFrame(index=frame.index)
    current_n = np.expm1(
        hierarchical["v5m_history_log_n"].to_numpy(dtype=np.float64)
    )
    # The exact current-season count is available from the E14 reconstruction.
    e14, e14_meta = build_e14_features(
        frame, states_before, priors, fallback_prior, k=80.0
    )
    season_n = e14["e14_n_season"].to_numpy(dtype=np.float64)
    derived["log_current_pitcher_n"] = np.log1p(season_n)
    derived["current_reliability_k80"] = season_n / (season_n + 80.0)
    derived["log_completed_pitcher_n"] = np.log1p(current_n)
    for index, class_name in enumerate(CLASS_NAMES):
        derived[f"log_offset_{class_name}"] = np.log(probabilities[:, index])
    return probabilities, derived, {
        "hierarchical": hierarchical_meta,
        "components": component_meta,
        "e14": e14_meta,
    }


def encode_context(
    train_context: pd.DataFrame,
    valid_context: pd.DataFrame,
    fit_mask: np.ndarray,
) -> tuple[list[np.ndarray], list[np.ndarray], list[int], dict[str, Any]]:
    train_arrays: list[np.ndarray] = []
    valid_arrays: list[np.ndarray] = []
    cardinalities: list[int] = []
    metadata: dict[str, Any] = {}
    for column in CATEGORICAL:
        values = train_context.loc[fit_mask, column].astype(str)
        categories = sorted(values.unique().tolist())
        mapping = {value: index + 1 for index, value in enumerate(categories)}
        train_encoded = (
            train_context.loc[fit_mask, column].astype(str).map(mapping).fillna(0)
        ).to_numpy(dtype=np.int64)
        valid_encoded = (
            valid_context[column].astype(str).map(mapping).fillna(0)
        ).to_numpy(dtype=np.int64)
        train_arrays.append(train_encoded)
        valid_arrays.append(valid_encoded)
        cardinalities.append(len(categories) + 1)
        metadata[column] = {
            "known_categories": len(categories),
            "valid_unknown_rows": int(np.sum(valid_encoded == 0)),
        }
    return train_arrays, valid_arrays, cardinalities, metadata


def continuous_context(
    train: pd.DataFrame,
    valid: pd.DataFrame,
    train_derived: pd.DataFrame,
    valid_derived: pd.DataFrame,
    fit_mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    train_values = pd.concat(
        [train.loc[:, list(RAW_CONTINUOUS)], train_derived], axis=1
    ).loc[fit_mask]
    valid_values = pd.concat(
        [valid.loc[:, list(RAW_CONTINUOUS)], valid_derived], axis=1
    )
    train_values = train_values.apply(pd.to_numeric, errors="coerce")
    valid_values = valid_values.apply(pd.to_numeric, errors="coerce")
    medians = train_values.median(axis=0)
    train_values = train_values.fillna(medians)
    valid_values = valid_values.fillna(medians)
    means = train_values.mean(axis=0)
    scales = train_values.std(axis=0).replace(0.0, 1.0).fillna(1.0)
    train_array = ((train_values - means) / scales).to_numpy(dtype=np.float32)
    valid_array = ((valid_values - means) / scales).to_numpy(dtype=np.float32)
    return train_array, valid_array, {
        "columns": list(train_values.columns),
        "means": {key: float(value) for key, value in means.items()},
        "scales": {key: float(value) for key, value in scales.items()},
    }


def fit_predict(
    base_train: np.ndarray,
    base_valid: np.ndarray,
    train_categories: list[np.ndarray],
    valid_categories: list[np.ndarray],
    cardinalities: list[int],
    train_continuous: np.ndarray,
    valid_continuous: np.ndarray,
    targets: np.ndarray,
    recipe: dict[str, Any],
    requested_device: str,
) -> tuple[np.ndarray, dict[str, Any]]:
    seed = int(recipe["random_seed"])
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    device = torch.device(
        "cuda" if requested_device == "cuda" and torch.cuda.is_available() else "cpu"
    )

    class OffsetModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.embeddings = nn.ModuleList(
                [nn.Embedding(size, len(CLASS_NAMES), padding_idx=0) for size in cardinalities]
            )
            self.linear = nn.Linear(train_continuous.shape[1], len(CLASS_NAMES), bias=True)
            for embedding in self.embeddings:
                nn.init.zeros_(embedding.weight)
            nn.init.zeros_(self.linear.weight)
            nn.init.zeros_(self.linear.bias)

        def forward(
            self,
            base_log: torch.Tensor,
            categorical_values: list[torch.Tensor],
            continuous_values: torch.Tensor,
        ) -> torch.Tensor:
            adjustment = self.linear(continuous_values)
            for embedding, values in zip(self.embeddings, categorical_values):
                adjustment = adjustment + embedding(values)
            return base_log + adjustment

    model = OffsetModel().to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(recipe["learning_rate"]),
        weight_decay=float(recipe["weight_decay"]),
    )
    loss_function = nn.CrossEntropyLoss()
    base_tensor = torch.from_numpy(np.log(base_train).astype(np.float32))
    continuous_tensor = torch.from_numpy(train_continuous)
    category_tensors = [torch.from_numpy(values) for values in train_categories]
    target_tensor = torch.from_numpy(targets.astype(np.int64))
    batch_size = int(recipe["batch_size"])
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    history: list[dict[str, float | int]] = []
    started = time.perf_counter()
    for epoch in range(int(recipe["epochs"])):
        model.train()
        order = torch.randperm(len(targets), generator=generator)
        loss_sum = 0.0
        for start in range(0, len(targets), batch_size):
            index = order[start : start + batch_size]
            base_batch = base_tensor[index].to(device, non_blocking=True)
            continuous_batch = continuous_tensor[index].to(device, non_blocking=True)
            categorical_batch = [
                values[index].to(device, non_blocking=True) for values in category_tensors
            ]
            target_batch = target_tensor[index].to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            logits = model(base_batch, categorical_batch, continuous_batch)
            loss = loss_function(logits, target_batch)
            loss.backward()
            optimizer.step()
            loss_sum += float(loss.detach().cpu()) * len(index)
        epoch_loss = loss_sum / len(targets)
        history.append({"epoch": epoch + 1, "cross_entropy": epoch_loss})
        print(f"  epoch {epoch + 1}: cross_entropy={epoch_loss:.7f}", flush=True)

    model.eval()
    predictions: list[np.ndarray] = []
    valid_base_tensor = torch.from_numpy(np.log(base_valid).astype(np.float32))
    valid_continuous_tensor = torch.from_numpy(valid_continuous)
    valid_category_tensors = [torch.from_numpy(values) for values in valid_categories]
    with torch.no_grad():
        for start in range(0, len(base_valid), batch_size):
            stop = min(start + batch_size, len(base_valid))
            logits = model(
                valid_base_tensor[start:stop].to(device),
                [values[start:stop].to(device) for values in valid_category_tensors],
                valid_continuous_tensor[start:stop].to(device),
            )
            probabilities = torch.softmax(logits, dim=1).cpu().numpy()
            predictions.append(probabilities)
    probability = np.concatenate(predictions, axis=0).astype(np.float64)
    details = {
        "backend": "pytorch_additive_multinomial_offset",
        "device": str(device),
        "training_history": history,
        "fit_predict_seconds": time.perf_counter() - started,
        "parameter_count": int(sum(parameter.numel() for parameter in model.parameters())),
        "prediction_class_means": {
            name: float(probability[:, index].mean())
            for index, name in enumerate(CLASS_NAMES)
        },
    }
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return probability[:, 0], details


def run_fold(
    frame: pd.DataFrame,
    validation_season: int,
    recipe: dict[str, Any],
    device: str,
) -> dict[str, Any]:
    started = time.perf_counter()
    source = frame.loc[frame["season"] <= validation_season].copy()
    history = source.loc[source["season"] < validation_season].copy()
    valid = source.loc[source["season"] == validation_season].copy()
    states_before, _ = entity_season_end_state(
        source,
        "pitcher_id",
        "asof_pitcher_n",
        "asof_pitcher_success_rate",
    )
    priors = candidate_priors_before_each_season(source, "r_recent3")
    fallback_prior = float(priors[validation_season])
    component_states, component_priors, _, _ = component_states_before_each_season(
        source
    )
    fallback_component_priors = dict(component_priors[validation_season])
    train_base, train_derived, train_state_meta = state_probabilities(
        history,
        states_before,
        priors,
        fallback_prior,
        component_states,
        component_priors,
        fallback_component_priors,
    )
    valid_base, valid_derived, valid_state_meta = state_probabilities(
        valid,
        states_before,
        priors,
        fallback_prior,
        component_states,
        component_priors,
        fallback_component_priors,
    )
    outcome = derive_control_outcome_labels(history, "reverse_any")
    class_index = {name: index for index, name in enumerate(CLASS_NAMES)}
    encoded_target = outcome.map(class_index)
    fit_mask = (
        history["game_type"].eq("R") & encoded_target.notna()
    ).to_numpy(dtype=bool)
    targets = encoded_target.loc[fit_mask].to_numpy(dtype=np.int64)
    train_context = contextual_frame(history)
    valid_context = contextual_frame(valid)
    train_categories, valid_categories, cardinalities, category_meta = encode_context(
        train_context, valid_context, fit_mask
    )
    train_continuous, valid_continuous, continuous_meta = continuous_context(
        history,
        valid,
        train_derived,
        valid_derived,
        fit_mask,
    )
    prediction, model_meta = fit_predict(
        train_base[fit_mask],
        valid_base,
        train_categories,
        valid_categories,
        cardinalities,
        train_continuous,
        valid_continuous,
        targets,
        recipe,
        device,
    )
    prediction = np.clip(prediction, 1e-6, 1.0 - 1e-6)
    y = valid["control_success"].to_numpy(dtype=np.int8)
    types = valid["game_type"].astype(str).to_numpy()
    metrics = {
        "all": score(y, prediction),
        "R": score(y[types == "R"], prediction[types == "R"]),
        "F": score(y[types == "F"], prediction[types == "F"]),
    }
    PREDICTIONS.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        PREDICTIONS / f"{recipe['stage']}_{validation_season}.npz",
        y=y,
        row_index=valid.index.to_numpy(dtype=np.int64),
        cluster=np.asarray(valid["pitcher_id"].astype(str).to_numpy(), dtype=np.str_),
        multinomial_offset=prediction,
    )
    print(
        f"[{validation_season}] R score={metrics['R']['competition_score']:.3f}, "
        f"all score={metrics['all']['competition_score']:.3f}",
        flush=True,
    )
    return {
        "validation_season": validation_season,
        "history_rows": int(len(history)),
        "fit_rows_R_with_outcome": int(fit_mask.sum()),
        "valid_rows": int(len(valid)),
        "outcome_counts": {
            name: int(np.sum(targets == index))
            for index, name in enumerate(CLASS_NAMES)
        },
        "metrics": metrics,
        "training_state": train_state_meta,
        "validation_state": valid_state_meta,
        "categorical_encoding": category_meta,
        "continuous_encoding": continuous_meta,
        "model": model_meta,
        "elapsed_seconds": time.perf_counter() - started,
    }


def main() -> None:
    args = parse_args()
    prereg = json.loads(PREREG.read_text(encoding="utf-8"))
    recipe = dict(prereg["fixed_recipe"])
    recipe["stage"] = args.stage
    output = RESULTS / f"{args.stage}.json"
    if output.exists():
        raise FileExistsError(f"immutable result already exists: {output}")
    if any(year not in (2022, 2023, 2024) for year in args.validation_seasons):
        raise ValueError("this preregistered experiment supports only 2022-2024")
    if 2024 in args.validation_seasons and args.stage != recipe["confirmation_stem"]:
        raise ValueError("2024 may only be generated under the fixed confirmation stem")
    if any(year in (2022, 2023) for year in args.validation_seasons) and args.stage != recipe["development_stem"]:
        raise ValueError("development folds require the preregistered development stem")
    frame = pd.read_csv(TRAIN)
    if not np.array_equal(frame.index.to_numpy(), np.arange(len(frame))):
        raise ValueError("training row index must remain the canonical file order")
    folds: list[dict[str, Any]] = []
    for validation_season in args.validation_seasons:
        folds.append(run_fold(frame, validation_season, recipe, args.device))
        gc.collect()
    report = {
        "experiment_id": prereg["experiment_id"],
        "stage": args.stage,
        "preregister_sha256": hashlib.sha256(PREREG.read_bytes()).hexdigest(),
        "official_train_only": True,
        "test_rows_read": False,
        "row_independent_inference": True,
        "recipe": recipe,
        "folds": folds,
    }
    RESULTS.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    print(f"Saved {output}")


if __name__ == "__main__":
    main()
