#!/usr/bin/env python3
"""Rebuild the public v20 gated hierarchical residual recipe, leakage-safely.

The public hyperparameters are treated as a fixed method description, never as
an external prediction artifact.  A target fold's tables use strictly earlier
OOF residuals.  Selection is performed on 2023 and 2024 remains confirmation.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


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
REPORT = ROOT / "experiments/results/v4_gated_hier_residual.json"
CONTEXT_SPECS = (
    ("count", ("count_state",), 0.25),
    ("count_hands", ("count_state", "pitcher_hand", "batter_hand"), 0.25),
    ("base", ("base_state",), 0.50),
)


def load(path: Path) -> dict[str, np.ndarray]:
    with np.load(path) as archive:
        return {key: np.asarray(archive[key]) for key in archive.files}


def sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(z, -30.0, 30.0)))


def logit(p: np.ndarray) -> np.ndarray:
    values = np.clip(p, 1e-6, 1.0 - 1e-6)
    return np.log(values / (1.0 - values))


def raw_score(y: np.ndarray, prediction: np.ndarray) -> float:
    return float(score(y, np.clip(prediction, 0.0, 1.0))["raw_competition_score"])


def add_keys(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    result["count_state"] = (
        result["balls_before"].to_numpy(dtype=np.int64) * 3
        + result["strikes_before"].to_numpy(dtype=np.int64)
    ).astype(np.int8)
    result["base_state"] = result["base_state"].astype(str)
    result["game_type"] = result["game_type"].astype(str)
    return result


def group_offset_table(
    frame: pd.DataFrame, prediction: np.ndarray, y: np.ndarray, keys: tuple[str, ...]
) -> pd.DataFrame:
    work = frame[list(keys)].copy()
    work["z"] = logit(prediction)
    work["y"] = y
    rows: list[dict[str, object]] = []
    grouper: str | list[str] = keys[0] if len(keys) == 1 else list(keys)
    for values, group in work.groupby(grouper, sort=True, observed=True):
        group_values = values if isinstance(values, tuple) else (values,)
        z = group["z"].to_numpy(dtype=np.float64)
        target = float(group["y"].mean())
        delta = 0.0
        for _ in range(30):
            probabilities = sigmoid(z + delta)
            error = float(probabilities.mean()) - target
            if abs(error) < 1e-13:
                break
            delta -= error / max(float(np.mean(probabilities * (1.0 - probabilities))), 1e-12)
        rows.append({**dict(zip(keys, group_values)), "offset": delta, "n": len(group)})
    return pd.DataFrame(rows)


def lookup(frame: pd.DataFrame, table: pd.DataFrame, keys: tuple[str, ...]) -> np.ndarray:
    left = frame[list(keys)].copy()
    left["_order"] = np.arange(len(left), dtype=np.int64)
    merged = left.merge(table, on=list(keys), how="left", sort=False).sort_values("_order")
    return merged["offset"].fillna(0.0).to_numpy(dtype=np.float64)


def residual_table(
    frame: pd.DataFrame,
    z_work: np.ndarray,
    y: np.ndarray,
    keys: tuple[str, ...],
    shrink: float,
) -> pd.DataFrame:
    probability = sigmoid(z_work)
    work = frame[list(keys)].copy()
    work["residual"] = y - probability
    work["info"] = probability * (1.0 - probability)
    table = work.groupby(list(keys), sort=True, observed=True).agg(
        residual=("residual", "sum"),
        info=("info", "sum"),
        n=("info", "size"),
    ).reset_index()
    table["offset"] = table["residual"] / (
        table["info"] + shrink * float(work["info"].mean())
    )
    table["offset"] = table["offset"].clip(-2.0, 2.0)
    return table[[*keys, "offset", "n"]]


def recipe_direction(
    source_frame: pd.DataFrame,
    source_prediction: np.ndarray,
    source_y: np.ndarray,
    target_frame: pd.DataFrame,
    target_prediction: np.ndarray,
    source_route: str,
) -> tuple[np.ndarray, dict[str, object]]:
    source_mask = np.ones(len(source_frame), dtype=bool)
    if source_route == "R":
        source_mask = source_frame["game_type"].eq("R").to_numpy()
    source = source_frame.loc[source_mask].reset_index(drop=True)
    prediction = source_prediction[source_mask]
    labels = source_y[source_mask]
    target_r = target_frame["game_type"].eq("R").to_numpy()

    source_correction = np.zeros(len(source), dtype=np.float64)
    target_correction = np.zeros(len(target_frame), dtype=np.float64)
    table_rows: dict[str, int] = {}
    for name, keys, weight in CONTEXT_SPECS:
        table = group_offset_table(source, prediction, labels, keys)
        source_correction += weight * lookup(source, table, keys)
        target_correction += weight * lookup(target_frame, table, keys)
        table_rows[name] = len(table)

    source_z = logit(prediction) + source_correction
    primary_keys = ("pitcher_id", "batter_hand")
    primary = residual_table(source, source_z, labels, primary_keys, 800.0)
    source_primary = 0.15 * lookup(source, primary, primary_keys)
    target_primary = 0.15 * lookup(target_frame, primary, primary_keys)
    source_primary_gate = (
        (source["pitcher_hand"].to_numpy(dtype=np.int64) == 2)
        | source["game_type"].eq("F").to_numpy()
    )
    target_primary_gate = (
        (target_frame["pitcher_hand"].to_numpy(dtype=np.int64) == 2)
        | target_frame["game_type"].eq("F").to_numpy()
    )
    source_z += source_primary * source_primary_gate
    target_correction += target_primary * target_primary_gate

    secondary_keys = ("pitcher_id", "count_state")
    secondary = residual_table(source, source_z, labels, secondary_keys, 1600.0)
    target_secondary = 0.40 * lookup(target_frame, secondary, secondary_keys)
    target_secondary_gate = (
        (target_frame["pitcher_hand"].to_numpy(dtype=np.int64) == 1)
        & target_r
    )
    target_correction += target_secondary * target_secondary_gate

    # The hidden test sample is regular-season.  Route this experimental arm
    # to R only so the 2023 F label regime cannot drive selection.
    corrected = sigmoid(logit(target_prediction) + target_correction)
    direction = np.where(target_r, corrected - target_prediction, 0.0)
    return direction, {
        "source_rows": int(source_mask.sum()),
        "context_table_rows": table_rows,
        "primary_rows": len(primary),
        "secondary_rows": len(secondary),
        "direction_std_r": float(np.std(direction[target_r])),
        "direction_max_abs": float(np.max(np.abs(direction))),
    }


def fit_scalar(direction: np.ndarray, residual: np.ndarray) -> tuple[float, float]:
    denominator = float(np.dot(direction, direction))
    raw = float(np.dot(direction, residual) / denominator) if denominator else 0.0
    return raw, float(np.clip(raw, -2.0, 2.0))


def main() -> None:
    train = add_keys(pd.read_csv(
        ROOT / "open/data/train.csv",
        usecols=[
            "season", "pitcher_id", "pitcher_hand", "batter_hand",
            "balls_before", "strikes_before", "base_state", "game_type",
        ],
        encoding="utf-8-sig",
        low_memory=False,
    ))
    accepted = {
        year: load(PRED / f"v4_routed_tabm_stack_locked_{year}.npz")
        for year in (2022, 2023, 2024)
    }
    frames = {
        year: train.iloc[accepted[year]["row_index"].astype(np.int64)].reset_index(drop=True)
        for year in accepted
    }
    y = {year: accepted[year]["y"].astype(np.float64) for year in accepted}
    accepted_prediction = {
        year: accepted[year]["routed_tabm_stack"].astype(np.float64)
        for year in accepted
    }

    latest = {
        year: load(PRED / f"v4_post4_c3_axis_screen_{year}.npz")
        for year in (2023, 2024)
    }
    base = {
        year: latest[year]["selected_prediction_plus_tabtransformer"].astype(np.float64)
        for year in latest
    }
    base_scores = {year: raw_score(y[year], base[year]) for year in base}

    directions: dict[str, dict[int, np.ndarray]] = {}
    diagnostics: dict[str, dict[int, dict[str, object]]] = {}
    for route in ("all", "R"):
        directions[route] = {}
        diagnostics[route] = {}
        direction23, diag23 = recipe_direction(
            frames[2022], accepted_prediction[2022], y[2022],
            frames[2023], accepted_prediction[2023], route,
        )
        source_frame24 = pd.concat([frames[2022], frames[2023]], ignore_index=True)
        source_prediction24 = np.concatenate(
            [accepted_prediction[2022], accepted_prediction[2023]]
        )
        source_y24 = np.concatenate([y[2022], y[2023]])
        direction24, diag24 = recipe_direction(
            source_frame24, source_prediction24, source_y24,
            frames[2024], accepted_prediction[2024], route,
        )
        directions[route][2023] = direction23
        directions[route][2024] = direction24
        diagnostics[route][2023] = diag23
        diagnostics[route][2024] = diag24

    screens: dict[str, dict[str, object]] = {}
    for route, values in directions.items():
        gamma_raw, gamma = fit_scalar(values[2023], y[2023] - base[2023])
        prediction23 = np.clip(base[2023] + gamma * values[2023], 0.0, 1.0)
        gain23 = raw_score(y[2023], prediction23) - base_scores[2023]
        screens[route] = {
            "gamma_fit_2023_raw": gamma_raw,
            "gamma_fit_2023": gamma,
            "gain_fit_2023": gain23,
            "passes_gate": bool(gamma > 0.0 and gain23 > 0.05),
            "diagnostics": diagnostics[route],
        }
    eligible = [name for name, row in screens.items() if row["passes_gate"]]
    selected = max(
        eligible, key=lambda name: float(screens[name]["gain_fit_2023"]), default="none"
    )

    final_prediction = base.copy()
    selected_gamma = 0.0
    if selected != "none":
        selected_gamma = float(screens[selected]["gamma_fit_2023"])
        final_prediction = {
            year: np.clip(
                base[year] + selected_gamma * directions[selected][year], 0.0, 1.0
            )
            for year in base
        }
    final_scores = {year: raw_score(y[year], final_prediction[year]) for year in base}

    artifact_paths: dict[int, str] = {}
    for year in (2023, 2024):
        path = PRED / f"v4_gated_hier_residual_{year}.npz"
        payload: dict[str, np.ndarray] = {
            "y": y[year],
            "row_index": accepted[year]["row_index"],
            "cluster": accepted[year]["cluster"],
            "base": base[year],
            "final_prediction": final_prediction[year],
        }
        for route in directions:
            payload[f"direction_source_{route}"] = directions[route][year]
        np.savez_compressed(path, **payload)
        artifact_paths[year] = str(path.relative_to(ROOT))

    report = {
        "protocol": {
            "official_train_only": True,
            "external_prediction_artifacts_used": False,
            "public_method_only": "v19 context plus v20 gated hierarchical residual recipe",
            "selection_target": 2023,
            "selection_source": [2022],
            "confirmation_target": 2024,
            "confirmation_sources": [2022, 2023],
            "test_rows_read": False,
            "apply_route": "R only",
            "row_independent_inference": True,
        },
        "fixed_recipe": {
            "context": {name: {"keys": list(keys), "weight": weight}
                        for name, keys, weight in CONTEXT_SPECS},
            "primary": {"keys": ["pitcher_id", "batter_hand"], "k": 800.0,
                        "weight": 0.15, "gate": "pitcher_hand2_or_F"},
            "secondary": {"keys": ["pitcher_id", "count_state"], "k": 1600.0,
                          "weight": 0.40, "gate": "pitcher_hand1_and_R"},
        },
        "screens": screens,
        "selected_source_route": selected,
        "selected_gamma": selected_gamma,
        "base_scores": base_scores,
        "final_scores": final_scores,
        "gains": {year: final_scores[year] - base_scores[year] for year in base},
        "expected_lb_median": final_scores[2024] + MEDIAN_OFFSET,
        "required_local_score": REQUIRED_LOCAL,
        "crosses_required_local_score": final_scores[2024] > REQUIRED_LOCAL,
        "prediction_artifacts": artifact_paths,
        "warning": "2024 is diagnostic confirmation and was not used for selection.",
    }
    REPORT.write_text(
        json.dumps(json_safe(report), ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(json_safe(report), ensure_ascii=False, indent=2), flush=True)
    print(f"Saved {REPORT}", flush=True)


if __name__ == "__main__":
    main()
