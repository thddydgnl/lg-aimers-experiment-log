#!/usr/bin/env python3
"""Evaluate history-only pitch-type selection and failure-component priors.

The current/validation pitch type is deliberately unavailable to this model.
For each validation row, historical TrackMan rows estimate a pitcher's pitch
mix in the row's pre-pitch context.  Separately, aligned historical outcomes
estimate command residuals by pitcher and pitch type.  Their expected-value
difference from the pitcher's overall repertoire is a row-independent prior.
"""

from __future__ import annotations

import itertools
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.analyze_v4_temporal_residual_ridge import (  # noqa: E402
    MEDIAN_OFFSET,
    REQUIRED_LOCAL,
    json_safe,
    load_frames,
    score,
)
from experiments.run_e20r_rolling import load_joined_trackman  # noqa: E402
from experiments.v4_current_ensemble import PREDICTIONS  # noqa: E402


OUTPUT_JSON = ROOT / "experiments/results/v4_pitchtype_failure_prior.json"
OUTPUT_NPZ = PREDICTIONS / "v4_pitchtype_failure_prior_2024.npz"
YEARS = (2022, 2023, 2024)
SIGNAL_NAMES = ("success", "reverse", "middle", "wayoff")
FINE_TYPES = (
    "Fastball",
    "Slider",
    "Curveball",
    "ChangeUp",
    "Splitter",
    "Sinker",
    "Cutter",
    "Other",
)
COARSE_TYPES = ("fastball", "breaking", "offspeed", "other")


@dataclass(frozen=True)
class Config:
    pitch_source: str
    lookback: int | None
    outcome_k: float
    selection_k: float
    context: str

    @property
    def name(self) -> str:
        lookback = "all" if self.lookback is None else str(self.lookback)
        return (
            f"{self.pitch_source}_lb{lookback}_ok{self.outcome_k:g}_"
            f"sk{self.selection_k:g}_{self.context}"
        )

    @property
    def pitch_types(self) -> tuple[str, ...]:
        return COARSE_TYPES if self.pitch_source == "group" else FINE_TYPES


def load_champion(season: int) -> dict[str, np.ndarray]:
    path = PREDICTIONS / f"v4_joint_neural_conservative_{season}.npz"
    with np.load(path) as archive:
        return {key: np.asarray(archive[key]) for key in archive.files}


def derive_failure_components(frame: pd.DataFrame) -> pd.DataFrame:
    """Recover per-pitch failure increments from official as-of counters."""
    n = pd.to_numeric(frame["asof_pitcher_n"], errors="coerce").fillna(0.0)
    reverse = np.rint(
        pd.to_numeric(frame["asof_pitcher_reverse_rate"], errors="coerce")
        .fillna(0.0)
        .to_numpy(dtype=np.float64)
        * n.to_numpy(dtype=np.float64)
    )
    middle = np.rint(
        pd.to_numeric(frame["asof_pitcher_middle_rate"], errors="coerce")
        .fillna(0.0)
        .to_numpy(dtype=np.float64)
        * n.to_numpy(dtype=np.float64)
    )
    work = pd.DataFrame(
        {
            "pitcher_id": frame["pitcher_id"].to_numpy(),
            "n": n.to_numpy(dtype=np.float64),
            "reverse_count": reverse,
            "middle_count": middle,
        },
        index=frame.index,
    )
    grouped = work.groupby("pitcher_id", sort=False, observed=True)
    next_n = grouped["n"].shift(-1)
    reverse_event = grouped["reverse_count"].shift(-1) - work["reverse_count"]
    middle_event = grouped["middle_count"].shift(-1) - work["middle_count"]
    valid = (
        (next_n - work["n"]).eq(1.0)
        & reverse_event.isin((0.0, 1.0))
        & middle_event.isin((0.0, 1.0))
    )
    success = pd.to_numeric(frame["control_success"], errors="coerce")
    result = pd.DataFrame(
        {
            "row_id": frame["row_id"].astype(str),
            "success": success.astype(np.float32),
            "reverse": reverse_event.astype(np.float32),
            "middle": middle_event.astype(np.float32),
            "wayoff": (
                success.eq(0.0) & reverse_event.eq(0.0) & middle_event.eq(0.0)
            ).astype(np.float32),
            "component_valid": valid.to_numpy(dtype=bool),
        }
    )
    for name in SIGNAL_NAMES:
        result.loc[~valid, name] = np.nan
    return result


def normalize_fine_pitch_type(values: pd.Series) -> pd.Series:
    normalized = values.astype("string").replace(
        {
            "Changeup": "ChangeUp",
            "Four-Seam": "Fastball",
            "SInker": "Sinker",
        }
    )
    return normalized.where(normalized.isin(FINE_TYPES[:-1]), "Other")


def prepare_history(train: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    components = derive_failure_components(train)
    joined = load_joined_trackman()
    history = joined.loc[joined["game_type"].eq("R")].copy()
    history["row_id"] = history["row_id"].astype(str)
    history = history.merge(
        components,
        on="row_id",
        how="left",
        validate="one_to_one",
    )
    history["pitch_type_tagged"] = normalize_fine_pitch_type(
        history["tagged_pitch_type"]
    )
    history["pitch_type_auto"] = normalize_fine_pitch_type(
        history["auto_pitch_type"]
    )
    history["pitch_type_group"] = (
        history["pitch_type_group"]
        .astype("string")
        .where(history["pitch_type_group"].isin(COARSE_TYPES), "other")
    )
    history["count_state"] = (
        pd.to_numeric(history["balls_before"], errors="coerce").fillna(-1)
        .astype(np.int16)
        * 3
        + pd.to_numeric(history["strikes_before"], errors="coerce")
        .fillna(-1)
        .astype(np.int16)
    )
    usable = history[list(SIGNAL_NAMES)].notna().all(axis=1)
    history = history.loc[usable].reset_index(drop=True)
    meta = {
        "aligned_regular_rows": int(len(joined.loc[joined["game_type"].eq("R")])),
        "component_usable_rows": int(len(history)),
        "component_usable_rate": float(usable.mean()),
        "pitch_type_counts": {
            source: {
                str(key): int(value)
                for key, value in history[f"pitch_type_{source}"].value_counts().items()
            }
            for source in ("tagged", "auto", "group")
        },
    }
    return history, meta


def wide_counts(
    frame: pd.DataFrame,
    keys: list[str],
    pitch_types: tuple[str, ...],
) -> pd.DataFrame:
    table = (
        frame.groupby([*keys, "pitch_type"], sort=False, observed=True)
        .size()
        .unstack(fill_value=0)
        .reindex(columns=pitch_types, fill_value=0)
    )
    table.columns = [f"n_{name}" for name in pitch_types]
    return table.reset_index()


def make_signal_matrix(
    history: pd.DataFrame,
    rows: pd.DataFrame,
    config: Config,
) -> np.ndarray:
    """Build four expected-value deltas using history strictly before a fold."""
    if history.empty:
        return np.zeros((len(rows), len(SIGNAL_NAMES)), dtype=np.float64)
    work = history.copy()
    work["pitch_type"] = work[f"pitch_type_{config.pitch_source}"]
    pitch_types = config.pitch_types
    centered_columns: list[str] = []
    for name in SIGNAL_NAMES:
        centered = f"rel_{name}"
        work[centered] = (
            work[name]
            - work.groupby("season", sort=False, observed=True)[name].transform("mean")
        )
        centered_columns.append(centered)

    type_means = (
        work.groupby("pitch_type", sort=False, observed=True)[centered_columns]
        .mean()
        .reindex(pitch_types)
        .fillna(0.0)
    )
    outcome_stats = work.groupby(
        ["pitcher_id", "pitch_type"], sort=False, observed=True
    )[centered_columns].agg(["sum", "count"])
    outcome = outcome_stats.index.to_frame(index=False)
    for name, centered in zip(SIGNAL_NAMES, centered_columns):
        total = outcome_stats[(centered, "sum")].to_numpy(dtype=np.float64)
        count = outcome_stats[(centered, "count")].to_numpy(dtype=np.float64)
        prior = outcome["pitch_type"].map(type_means[centered]).to_numpy(
            dtype=np.float64
        )
        outcome[f"v_{name}"] = (
            total + config.outcome_k * prior
        ) / (count + config.outcome_k)

    global_mix = (
        work["pitch_type"].value_counts(normalize=True).reindex(pitch_types).fillna(0.0)
    )
    overall = wide_counts(work, ["pitcher_id"], pitch_types)
    count_columns = [f"n_{name}" for name in pitch_types]
    total = overall[count_columns].sum(axis=1).to_numpy(dtype=np.float64)
    for pitch_type in pitch_types:
        overall[f"p_{pitch_type}"] = (
            overall[f"n_{pitch_type}"].to_numpy(dtype=np.float64)
            + config.selection_k * float(global_mix[pitch_type])
        ) / (total + config.selection_k)
    probability_columns = [f"p_{name}" for name in pitch_types]

    context_columns = ["count_state"]
    if config.context == "hand_count":
        context_columns = ["batter_hand", "count_state"]
    elif config.context != "count":
        raise ValueError(f"Unknown context: {config.context}")
    keys = ["pitcher_id", *context_columns]
    contextual = wide_counts(work, keys, pitch_types)
    contextual_total = contextual[count_columns].sum(axis=1).to_numpy(
        dtype=np.float64
    )
    contextual = contextual.merge(
        overall[["pitcher_id", *probability_columns]],
        on="pitcher_id",
        how="left",
        validate="many_to_one",
        suffixes=("", "_overall"),
    )
    for pitch_type in pitch_types:
        contextual[f"p_{pitch_type}"] = (
            contextual[f"n_{pitch_type}"].to_numpy(dtype=np.float64)
            + config.selection_k
            * contextual[f"p_{pitch_type}"].to_numpy(dtype=np.float64)
        ) / (contextual_total + config.selection_k)
    contextual = contextual[[*keys, *probability_columns]]

    outcome_wide = outcome.pivot(
        index="pitcher_id", columns="pitch_type", values=[f"v_{n}" for n in SIGNAL_NAMES]
    )
    for name in SIGNAL_NAMES:
        for pitch_type in pitch_types:
            column = (f"v_{name}", pitch_type)
            if column not in outcome_wide.columns:
                outcome_wide[column] = float(type_means.loc[pitch_type, f"rel_{name}"])
    outcome_wide = outcome_wide.reindex(
        columns=pd.MultiIndex.from_product(
            [[f"v_{name}" for name in SIGNAL_NAMES], pitch_types]
        )
    )
    outcome_wide.columns = [
        f"{value_name}_{pitch_type}"
        for value_name, pitch_type in outcome_wide.columns
    ]
    outcome_wide = outcome_wide.reset_index()

    base = overall[["pitcher_id", *probability_columns]].merge(
        outcome_wide,
        on="pitcher_id",
        how="left",
        validate="one_to_one",
    )
    for name in SIGNAL_NAMES:
        base[f"base_{name}"] = sum(
            base[f"p_{pitch_type}"] * base[f"v_{name}_{pitch_type}"]
            for pitch_type in pitch_types
        )

    query = rows[["pitcher_id", "batter_hand", "balls_before", "strikes_before"]].copy()
    query["count_state"] = (
        pd.to_numeric(query["balls_before"], errors="coerce").fillna(-1).astype(np.int16)
        * 3
        + pd.to_numeric(query["strikes_before"], errors="coerce")
        .fillna(-1)
        .astype(np.int16)
    )
    query["_order"] = np.arange(len(query), dtype=np.int64)
    query = query.merge(
        contextual,
        on=keys,
        how="left",
        validate="many_to_one",
    )
    query = query.merge(
        overall[["pitcher_id", *probability_columns]],
        on="pitcher_id",
        how="left",
        validate="many_to_one",
        suffixes=("", "_overall"),
    )
    for pitch_type in pitch_types:
        query[f"p_{pitch_type}"] = query[f"p_{pitch_type}"].fillna(
            query[f"p_{pitch_type}_overall"]
        ).fillna(float(global_mix[pitch_type]))
    query = query.merge(
        outcome_wide,
        on="pitcher_id",
        how="left",
        validate="many_to_one",
    )
    query = query.merge(
        base[["pitcher_id", *[f"base_{name}" for name in SIGNAL_NAMES]]],
        on="pitcher_id",
        how="left",
        validate="many_to_one",
    )
    for name in SIGNAL_NAMES:
        for pitch_type in pitch_types:
            query[f"v_{name}_{pitch_type}"] = query[
                f"v_{name}_{pitch_type}"
            ].fillna(float(type_means.loc[pitch_type, f"rel_{name}"]))
        context_expected = sum(
            query[f"p_{pitch_type}"] * query[f"v_{name}_{pitch_type}"]
            for pitch_type in pitch_types
        )
        query[f"signal_{name}"] = (
            context_expected - query[f"base_{name}"].fillna(0.0)
        )
    query = query.sort_values("_order")
    matrix = query[[f"signal_{name}" for name in SIGNAL_NAMES]].to_numpy(
        dtype=np.float64
    )
    return np.nan_to_num(matrix, nan=0.0, posinf=0.0, neginf=0.0)


def brier_gain(
    y: np.ndarray,
    baseline: np.ndarray,
    correction: np.ndarray,
) -> float:
    residual = np.asarray(y, dtype=np.float64) - np.asarray(baseline, dtype=np.float64)
    delta = np.asarray(correction, dtype=np.float64)
    improvement = (
        2.0 * float(np.dot(residual, delta)) - float(np.dot(delta, delta))
    ) / float(len(residual))
    rate = float(np.mean(y))
    return 100_000.0 * improvement / (rate * (1.0 - rate))


def solve_weights(
    x: np.ndarray,
    y: np.ndarray,
    baseline: np.ndarray,
    ridge: float,
) -> np.ndarray:
    residual = np.asarray(y, dtype=np.float64) - np.asarray(baseline, dtype=np.float64)
    xtx = x.T @ x
    scale = max(float(np.trace(xtx)) / max(1, x.shape[1]), 1e-12)
    return np.linalg.solve(
        xtx + ridge * scale * np.eye(x.shape[1], dtype=np.float64),
        x.T @ residual,
    )


def candidate_weights(
    signals: dict[int, np.ndarray],
    targets: dict[int, np.ndarray],
    champions: dict[int, np.ndarray],
) -> list[tuple[str, np.ndarray]]:
    candidates: list[tuple[str, np.ndarray]] = []
    for columns, label in (
        (np.arange(4), "all"),
        (np.arange(1, 4), "failures"),
        (np.asarray([0]), "success"),
    ):
        for ridge in (0.01, 0.1, 1.0, 10.0):
            fold_solutions = []
            for year in (2022, 2023):
                fold_solutions.append(
                    solve_weights(
                        signals[year][:, columns],
                        targets[year],
                        champions[year],
                        ridge,
                    )
                )
            for aggregate_name, aggregate in (
                ("median", np.median(np.stack(fold_solutions), axis=0)),
                ("mean", np.mean(np.stack(fold_solutions), axis=0)),
            ):
                for shrink in (0.25, 0.50, 0.75, 1.00):
                    weights = np.zeros(4, dtype=np.float64)
                    weights[columns] = shrink * aggregate
                    candidates.append(
                        (
                            f"{label}_{aggregate_name}_r{ridge:g}_s{shrink:g}",
                            weights,
                        )
                    )
    for index, name in enumerate(SIGNAL_NAMES):
        for value in (-1.0, -0.75, -0.5, -0.25, 0.25, 0.5, 0.75, 1.0):
            weights = np.zeros(4, dtype=np.float64)
            weights[index] = value
            candidates.append((f"only_{name}_{value:g}", weights))
    return candidates


def main() -> None:
    frames, artifacts = load_frames()
    champions_artifacts = {year: load_champion(year) for year in YEARS}
    champions: dict[int, np.ndarray] = {}
    targets: dict[int, np.ndarray] = {}
    for year in YEARS:
        if not np.array_equal(
            artifacts[year]["row_index"], champions_artifacts[year]["row_index"]
        ):
            raise ValueError(f"Champion alignment mismatch for {year}")
        targets[year] = np.asarray(artifacts[year]["y"], dtype=np.float64)
        champions[year] = np.asarray(
            champions_artifacts[year]["conservative"], dtype=np.float64
        )

    train_columns = [
        "row_id",
        "pitcher_id",
        "asof_pitcher_n",
        "asof_pitcher_reverse_rate",
        "asof_pitcher_middle_rate",
        "control_success",
    ]
    train = pd.read_csv(
        ROOT / "open/data/train.csv",
        usecols=train_columns,
        encoding="utf-8-sig",
        low_memory=False,
    )
    history, history_meta = prepare_history(train)
    del train

    configs = [
        Config(pitch_source, lookback, outcome_k, selection_k, context)
        for pitch_source, lookback, outcome_k, selection_k, context in itertools.product(
            ("tagged", "auto", "group"),
            (None, 3, 1),
            (10.0, 50.0),
            (100.0, 300.0),
            ("count", "hand_count"),
        )
    ]
    selection_rows: list[dict[str, Any]] = []
    selected_payload: tuple[
        Config, np.ndarray, dict[int, np.ndarray], dict[str, Any]
    ] | None = None
    best_by_source: dict[
        str, tuple[Config, np.ndarray, dict[int, np.ndarray], dict[str, Any]]
    ] = {}
    for config_index, config in enumerate(configs, start=1):
        fold_signals: dict[int, np.ndarray] = {}
        for year in YEARS:
            cutoff_history = history.loc[history["season"].lt(year)]
            if config.lookback is not None:
                cutoff_history = cutoff_history.loc[
                    cutoff_history["season"].ge(year - config.lookback)
                ]
            signal = make_signal_matrix(cutoff_history, frames[year], config)
            signal[~frames[year]["game_type"].eq("R").to_numpy(), :] = 0.0
            fold_signals[year] = signal
        for candidate_name, weights in candidate_weights(
            fold_signals, targets, champions
        ):
            gains = {
                str(year): brier_gain(
                    targets[year], champions[year], fold_signals[year] @ weights
                )
                for year in (2022, 2023)
            }
            row = {
                "config": config.name,
                "candidate": candidate_name,
                "weights": {
                    name: float(value) for name, value in zip(SIGNAL_NAMES, weights)
                },
                "gains": gains,
                "robust_min_gain": float(min(gains.values())),
                "mean_gain": float(np.mean(list(gains.values()))),
                "correction_max_abs_2022_2023": float(
                    max(
                        np.max(np.abs(fold_signals[year] @ weights))
                        for year in (2022, 2023)
                    )
                ),
            }
            selection_rows.append(row)
            key = (row["robust_min_gain"], row["mean_gain"])
            if selected_payload is None or key > (
                selected_payload[3]["robust_min_gain"],
                selected_payload[3]["mean_gain"],
            ):
                selected_payload = (config, weights.copy(), fold_signals, row)
            source_payload = best_by_source.get(config.pitch_source)
            if source_payload is None or key > (
                source_payload[3]["robust_min_gain"],
                source_payload[3]["mean_gain"],
            ):
                best_by_source[config.pitch_source] = (
                    config,
                    weights.copy(),
                    fold_signals,
                    row,
                )
        print(
            f"[{config_index:02d}/{len(configs)}] {config.name}",
            flush=True,
        )

    if selected_payload is None:
        raise RuntimeError("No pitch-type candidates were evaluated")
    source_order = tuple(
        source for source in ("tagged", "auto", "group") if source in best_by_source
    )
    source_directions = {
        year: np.column_stack(
            [
                best_by_source[source][2][year] @ best_by_source[source][1]
                for source in source_order
            ]
        )
        for year in YEARS
    }
    joint_trials: list[dict[str, Any]] = []
    for scales in itertools.product((0.0, 0.25, 0.50, 0.75, 1.00), repeat=len(source_order)):
        scale_array = np.asarray(scales, dtype=np.float64)
        gains = {
            str(year): brier_gain(
                targets[year],
                champions[year],
                source_directions[year] @ scale_array,
            )
            for year in (2022, 2023)
        }
        joint_trials.append(
            {
                "source_scales": {
                    source: float(value)
                    for source, value in zip(source_order, scale_array)
                },
                "gains": gains,
                "robust_min_gain": float(min(gains.values())),
                "mean_gain": float(np.mean(list(gains.values()))),
            }
        )
    selected_joint = max(
        joint_trials,
        key=lambda row: (row["robust_min_gain"], row["mean_gain"]),
    )
    selected_scales = np.asarray(
        [selected_joint["source_scales"][source] for source in source_order],
        dtype=np.float64,
    )
    correction = source_directions[2024] @ selected_scales
    prediction = np.clip(champions[2024] + correction, 0.0, 1.0)
    base_metrics = {str(year): score(targets[year], champions[year]) for year in YEARS}
    confirmation_metrics = score(targets[2024], prediction)
    confirmation_gain = float(
        confirmation_metrics["raw_competition_score"]
        - base_metrics["2024"]["raw_competition_score"]
    )

    fold_artifacts: dict[str, str] = {}
    for year in YEARS:
        fold_correction = source_directions[year] @ selected_scales
        fold_prediction = np.clip(champions[year] + fold_correction, 0.0, 1.0)
        path = PREDICTIONS / f"v4_pitchtype_failure_prior_{year}.npz"
        np.savez_compressed(
            path,
            y=targets[year].astype(np.int8),
            row_index=artifacts[year]["row_index"],
            cluster=artifacts[year]["cluster"],
            champion=champions[year],
            source_directions=source_directions[year],
            source_scales=selected_scales,
            correction=fold_correction,
            pitchtype_failure=fold_prediction,
        )
        fold_artifacts[str(year)] = str(path.relative_to(ROOT))
    report = {
        "protocol": {
            "official_train_and_trackman_only": True,
            "external_data_used": False,
            "test_rows_read": False,
            "leaderboard_values_used_for_selection": False,
            "current_or_validation_pitch_type_used": False,
            "row_independent": True,
            "history_cutoff": "season strictly before validation season",
            "selection_folds": [2022, 2023],
            "confirmation_fold": 2024,
        },
        "fixed_estimator": {
            "median_offset": MEDIAN_OFFSET,
            "required_local_score": REQUIRED_LOCAL,
            "target_lb": 1190.0,
        },
        "history": history_meta,
        "signal_names": list(SIGNAL_NAMES),
        "pitch_types": {
            "tagged": list(FINE_TYPES),
            "auto": list(FINE_TYPES),
            "group": list(COARSE_TYPES),
        },
        "config_count": len(configs),
        "selected_by_source": {
            source: {
                "config": asdict(best_by_source[source][0]),
                "selection": best_by_source[source][3],
            }
            for source in source_order
        },
        "selected_joint": selected_joint,
        "top_joint_trials": sorted(
            joint_trials,
            key=lambda row: (row["robust_min_gain"], row["mean_gain"]),
            reverse=True,
        )[:50],
        "top_selection_trials": sorted(
            selection_rows,
            key=lambda row: (row["robust_min_gain"], row["mean_gain"]),
            reverse=True,
        )[:100],
        "base_metrics": base_metrics,
        "confirmation_2024": {
            "metrics": confirmation_metrics,
            "gain": confirmation_gain,
            "expected_lb_median": float(
                confirmation_metrics["raw_competition_score"] + MEDIAN_OFFSET
            ),
            "crosses_required_local_score": bool(
                confirmation_metrics["raw_competition_score"] > REQUIRED_LOCAL
            ),
            "correction_mean": float(correction.mean()),
            "correction_std": float(correction.std()),
            "correction_max_abs": float(np.max(np.abs(correction))),
        },
        "fold_artifacts": fold_artifacts,
        "prediction_artifact": str(OUTPUT_NPZ.relative_to(ROOT)),
    }
    OUTPUT_JSON.write_text(
        json.dumps(json_safe(report), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "selected_by_source": {
                    source: best_by_source[source][0].name for source in source_order
                },
                "source_scales": selected_joint["source_scales"],
                "selection_gains": selected_joint["gains"],
                "confirmation_gain": confirmation_gain,
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
