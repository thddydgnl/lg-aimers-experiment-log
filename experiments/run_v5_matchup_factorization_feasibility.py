#!/usr/bin/env python3
"""Out-of-time additive-vs-low-rank pitcher/batter feasibility diagnostic."""

from __future__ import annotations

# Load torch before pandas/sklearn/native boosters on this Windows runtime.
import torch

import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


ROOT = Path(__file__).resolve().parents[1]
PREREG = ROOT / "experiments/params/v5_matchup_factorization_feasibility_preregister.json"
OUTPUT = ROOT / "experiments/results/v5_matchup_factorization_feasibility.json"
PREDICTIONS = ROOT / "experiments/results/predictions"
TARGET_YEARS = (2021, 2022, 2023)
TARGET = "control_success"
ID_COLUMNS = ("pitcher_id", "batter_id")
NUMERIC_RAW = (
    "inning",
    "outs_before",
    "num_runners_on",
    "score_diff_pitcher_team",
    "li",
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
)
COUNT_RAW = ("asof_pitcher_n", "asof_batter_n", "asof_pitcher_pitchmix_n")
USECOLS = sorted(
    {
        "season",
        "game_type",
        "balls_before",
        "strikes_before",
        "pitcher_hand",
        "batter_hand",
        "pitcher_team_id",
        "batter_team_id",
        "top_bottom",
        *ID_COLUMNS,
        *NUMERIC_RAW,
        *COUNT_RAW,
        TARGET,
    }
)
EPOCHS = 8
BATCH_SIZE = 16384
LEARNING_RATE = 0.01
WEIGHT_DECAY = 1e-4
SEED = 20260821


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def encode_known(train: pd.Series, valid: pd.Series) -> tuple[np.ndarray, np.ndarray, int]:
    categories = np.sort(train.dropna().unique())
    train_values = train.to_numpy()
    valid_values = valid.to_numpy()
    train_codes = np.searchsorted(categories, train_values) + 1
    valid_positions = np.searchsorted(categories, valid_values)
    valid_codes = np.zeros(len(valid_values), dtype=np.int64)
    in_bounds = valid_positions < len(categories)
    matched = np.zeros(len(valid_values), dtype=bool)
    matched[in_bounds] = categories[valid_positions[in_bounds]] == valid_values[in_bounds]
    valid_codes[matched] = valid_positions[matched] + 1
    return train_codes.astype(np.int64), valid_codes, int(len(categories) + 1)


def numeric_matrix(
    train: pd.DataFrame, valid: pd.DataFrame
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    train_parts: list[np.ndarray] = []
    valid_parts: list[np.ndarray] = []
    names: list[str] = []
    for name in NUMERIC_RAW:
        train_parts.append(pd.to_numeric(train[name], errors="coerce").to_numpy(dtype=np.float64))
        valid_parts.append(pd.to_numeric(valid[name], errors="coerce").to_numpy(dtype=np.float64))
        names.append(name)
    for name in COUNT_RAW:
        train_parts.append(np.log1p(pd.to_numeric(train[name], errors="coerce").clip(lower=0).to_numpy(dtype=np.float64)))
        valid_parts.append(np.log1p(pd.to_numeric(valid[name], errors="coerce").clip(lower=0).to_numpy(dtype=np.float64)))
        names.append(f"log1p_{name}")
    train_x = np.column_stack(train_parts)
    valid_x = np.column_stack(valid_parts)
    train_x[~np.isfinite(train_x)] = np.nan
    valid_x[~np.isfinite(valid_x)] = np.nan
    median = np.nanmedian(train_x, axis=0)
    median[~np.isfinite(median)] = 0.0
    train_missing = np.where(np.isnan(train_x))
    valid_missing = np.where(np.isnan(valid_x))
    train_x[train_missing] = median[train_missing[1]]
    valid_x[valid_missing] = median[valid_missing[1]]
    mean = train_x.mean(axis=0)
    scale = train_x.std(axis=0)
    scale[~np.isfinite(scale) | (scale < 1e-6)] = 1.0
    train_x = np.clip((train_x - mean) / scale, -10.0, 10.0).astype(np.float32)
    valid_x = np.clip((valid_x - mean) / scale, -10.0, 10.0).astype(np.float32)
    return train_x, valid_x, names


def context_codes(frame: pd.DataFrame) -> np.ndarray:
    count = frame["balls_before"].to_numpy(dtype=np.int64) * 3 + frame["strikes_before"].to_numpy(dtype=np.int64)
    hand = (frame["pitcher_hand"].to_numpy(dtype=np.int64) - 1) * 2 + (frame["batter_hand"].to_numpy(dtype=np.int64) - 1)
    pitcher_team = frame["pitcher_team_id"].to_numpy(dtype=np.int64)
    batter_team = frame["batter_team_id"].to_numpy(dtype=np.int64)
    top_bottom = frame["top_bottom"].astype(str).map({"T": 1, "B": 2}).fillna(0).to_numpy(dtype=np.int64)
    return np.column_stack([count + 1, hand + 1, pitcher_team + 1, batter_team + 1, top_bottom])


class MatchupModel(torch.nn.Module):
    def __init__(
        self,
        pitcher_cardinality: int,
        batter_cardinality: int,
        numeric_count: int,
        rank: int,
        prior: float,
    ) -> None:
        super().__init__()
        self.rank = rank
        self.pitcher_bias = torch.nn.Embedding(pitcher_cardinality, 1, padding_idx=0)
        self.batter_bias = torch.nn.Embedding(batter_cardinality, 1, padding_idx=0)
        self.context = torch.nn.ModuleList(
            [
                torch.nn.Embedding(13, 1, padding_idx=0),
                torch.nn.Embedding(6, 1, padding_idx=0),
                torch.nn.Embedding(32, 1, padding_idx=0),
                torch.nn.Embedding(32, 1, padding_idx=0),
                torch.nn.Embedding(3, 1, padding_idx=0),
            ]
        )
        self.numeric = torch.nn.Linear(numeric_count, 1, bias=False)
        self.global_bias = torch.nn.Parameter(
            torch.tensor(math.log(prior / (1.0 - prior)), dtype=torch.float32)
        )
        if rank > 0:
            self.pitcher_latent = torch.nn.Embedding(
                pitcher_cardinality, rank, padding_idx=0
            )
            self.batter_latent = torch.nn.Embedding(
                batter_cardinality, rank, padding_idx=0
            )
            torch.nn.init.normal_(self.pitcher_latent.weight, std=0.02)
            torch.nn.init.normal_(self.batter_latent.weight, std=0.02)
        for embedding in [self.pitcher_bias, self.batter_bias, *self.context]:
            torch.nn.init.zeros_(embedding.weight)
        torch.nn.init.zeros_(self.numeric.weight)

    def forward(
        self,
        pitcher: torch.Tensor,
        batter: torch.Tensor,
        context: torch.Tensor,
        numeric: torch.Tensor,
    ) -> torch.Tensor:
        result = self.global_bias + self.pitcher_bias(pitcher).squeeze(1)
        result = result + self.batter_bias(batter).squeeze(1)
        result = result + self.numeric(numeric).squeeze(1)
        for index, embedding in enumerate(self.context):
            result = result + embedding(context[:, index]).squeeze(1)
        if self.rank > 0:
            result = result + (
                self.pitcher_latent(pitcher) * self.batter_latent(batter)
            ).sum(dim=1) / math.sqrt(float(self.rank))
        return result


def fit_predict(
    arrays: dict[str, np.ndarray],
    valid_arrays: dict[str, np.ndarray],
    cardinalities: tuple[int, int],
    rank: int,
    prior: float,
    device: torch.device,
) -> tuple[np.ndarray, list[float]]:
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)
    model = MatchupModel(
        cardinalities[0], cardinalities[1], arrays["numeric"].shape[1], rank, prior
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )
    tensors = {
        name: torch.from_numpy(np.ascontiguousarray(value)).to(device)
        for name, value in arrays.items()
    }
    history: list[float] = []
    n = len(arrays["y"])
    generator = torch.Generator(device=device)
    generator.manual_seed(SEED)
    model.train()
    for _epoch in range(EPOCHS):
        permutation = torch.randperm(n, generator=generator, device=device)
        total = 0.0
        for start in range(0, n, BATCH_SIZE):
            index = permutation[start : start + BATCH_SIZE]
            optimizer.zero_grad(set_to_none=True)
            logits = model(
                tensors["pitcher"][index],
                tensors["batter"][index],
                tensors["context"][index],
                tensors["numeric"][index],
            )
            loss = torch.nn.functional.binary_cross_entropy_with_logits(
                logits, tensors["y"][index]
            )
            loss.backward()
            optimizer.step()
            total += float(loss.detach()) * int(len(index))
        history.append(total / n)
    model.eval()
    predictions: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(valid_arrays["pitcher"]), 65536):
            stop = start + 65536
            pitcher = torch.from_numpy(valid_arrays["pitcher"][start:stop]).to(device)
            batter = torch.from_numpy(valid_arrays["batter"][start:stop]).to(device)
            context = torch.from_numpy(valid_arrays["context"][start:stop]).to(device)
            numeric = torch.from_numpy(valid_arrays["numeric"][start:stop]).to(device)
            prediction = torch.sigmoid(model(pitcher, batter, context, numeric))
            predictions.append(prediction.cpu().numpy().astype(np.float64))
    del model, optimizer, tensors
    torch.cuda.empty_cache()
    return np.concatenate(predictions), history


def gain_interval(
    y: np.ndarray,
    parent: np.ndarray,
    candidate: np.ndarray,
    cluster: np.ndarray,
    seed: int,
) -> dict[str, float | int]:
    paired = np.square(y - parent) - np.square(y - candidate)
    rate = float(np.mean(y))
    scale = 100_000.0 / (rate * (1.0 - rate))
    grouped = pd.DataFrame({"cluster": cluster, "paired": paired}).groupby(
        "cluster", sort=False, observed=True
    )["paired"].agg(["sum", "count"])
    sums = grouped["sum"].to_numpy(dtype=np.float64)
    counts = grouped["count"].to_numpy(dtype=np.int64)
    rng = np.random.default_rng(seed)
    draws = np.empty(1000, dtype=np.float64)
    for index in range(1000):
        sampled = rng.integers(0, len(grouped), size=len(grouped))
        draws[index] = scale * float(sums[sampled].sum() / counts[sampled].sum())
    return {
        "point_gain": scale * float(np.mean(paired)),
        "lower_95": float(np.quantile(draws, 0.025)),
        "upper_95": float(np.quantile(draws, 0.975)),
        "clusters": int(len(grouped)),
        "replicates": 1000,
    }


def main() -> None:
    prereg = json.loads(PREREG.read_text(encoding="utf-8"))
    if prereg["status"] != "locked_before_execution":
        raise ValueError("Preregister status changed")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("This preregistered diagnostic requires CUDA")
    full = pd.read_csv(
        ROOT / "open/data/train.csv",
        usecols=USECOLS,
        encoding="utf-8-sig",
        low_memory=False,
    )
    folds: dict[str, Any] = {}
    PREDICTIONS.mkdir(parents=True, exist_ok=True)
    for year in TARGET_YEARS:
        history = full.loc[full["season"].lt(year) & full["game_type"].eq("R")].copy()
        valid = full.loc[full["season"].eq(year) & full["game_type"].eq("R")].copy()
        pitcher_train, pitcher_valid, pitcher_cardinality = encode_known(
            history["pitcher_id"], valid["pitcher_id"]
        )
        batter_train, batter_valid, batter_cardinality = encode_known(
            history["batter_id"], valid["batter_id"]
        )
        numeric_train, numeric_valid, numeric_names = numeric_matrix(history, valid)
        arrays = {
            "pitcher": pitcher_train,
            "batter": batter_train,
            "context": context_codes(history).astype(np.int64),
            "numeric": numeric_train,
            "y": history[TARGET].to_numpy(dtype=np.float32),
        }
        valid_arrays = {
            "pitcher": pitcher_valid,
            "batter": batter_valid,
            "context": context_codes(valid).astype(np.int64),
            "numeric": numeric_valid,
        }
        prior = float(arrays["y"].mean())
        print(
            f"matchup {year}: train={len(history):,} valid={len(valid):,} "
            f"pitchers={pitcher_cardinality-1} batters={batter_cardinality-1}",
            flush=True,
        )
        additive, additive_history = fit_predict(
            arrays, valid_arrays, (pitcher_cardinality, batter_cardinality), 0, prior, device
        )
        factor, factor_history = fit_predict(
            arrays, valid_arrays, (pitcher_cardinality, batter_cardinality), 4, prior, device
        )
        y = valid[TARGET].to_numpy(dtype=np.float64)
        gain = gain_interval(
            y, additive, factor, valid["pitcher_id"].to_numpy(), SEED + year
        )
        auc_additive = float(roc_auc_score(y, additive))
        auc_factor = float(roc_auc_score(y, factor))
        path = PREDICTIONS / f"v5_matchup_factorization_feasibility_{year}.npz"
        if path.exists():
            raise FileExistsError(f"Refusing to overwrite {path}")
        np.savez_compressed(
            path,
            y=y.astype(np.int8),
            row_index=valid.index.to_numpy(dtype=np.int64),
            cluster=valid["pitcher_id"].to_numpy(dtype=np.int64),
            additive=additive,
            rank4=factor,
        )
        folds[str(year)] = {
            "train_rows": int(len(history)),
            "valid_rows": int(len(valid)),
            "history_seasons": sorted(int(value) for value in history["season"].unique()),
            "known_pitcher_rate": float(np.mean(pitcher_valid > 0)),
            "known_batter_rate": float(np.mean(batter_valid > 0)),
            "additive_brier": float(np.mean(np.square(y - additive))),
            "rank4_brier": float(np.mean(np.square(y - factor))),
            "additive_auc": auc_additive,
            "rank4_auc": auc_factor,
            "auc_delta": auc_factor - auc_additive,
            "paired_gain": gain,
            "additive_training_loss": additive_history,
            "rank4_training_loss": factor_history,
            "numeric_features": numeric_names,
            "artifact": str(path.relative_to(ROOT)),
            "artifact_sha256": sha256(path),
        }
        print(
            f"  gain={gain['point_gain']:.3f} "
            f"CI=[{gain['lower_95']:.3f}, {gain['upper_95']:.3f}] "
            f"auc_delta={auc_factor-auc_additive:.6f}",
            flush=True,
        )

    point_all = all(folds[str(year)]["paired_gain"]["point_gain"] > 0 for year in TARGET_YEARS)
    lower_count = sum(folds[str(year)]["paired_gain"]["lower_95"] > 0 for year in TARGET_YEARS)
    auc_all = all(folds[str(year)]["auc_delta"] > 0 for year in TARGET_YEARS)
    passed = point_all and lower_count >= 2 and auc_all
    payload = {
        "experiment_id": prereg["experiment_id"],
        "status": "passed_proceed_to_candidate" if passed else "failed_reject_without_2024",
        "protocol": {
            "official_data_only": True,
            "regular_season_only": True,
            "2024_labels_used": False,
            "2024_model_run": False,
            "test_rows_read": False,
        },
        "preregister": str(PREREG.relative_to(ROOT)),
        "preregister_sha256": sha256(PREREG),
        "device": str(device),
        "folds": folds,
        "gate": {
            "positive_point_all_three": bool(point_all),
            "positive_lower_years": int(lower_count),
            "positive_auc_delta_all_three": bool(auc_all),
            "passed": bool(passed),
        },
        "next_action": (
            "Preregister rank-4 candidate blend on 2022/2023."
            if passed
            else "Reject factorization without 2024."
        ),
    }
    OUTPUT.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"wrote {OUTPUT}", flush=True)


if __name__ == "__main__":
    main()
