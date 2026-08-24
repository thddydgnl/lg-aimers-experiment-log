#!/usr/bin/env python3
"""Reproduce the fixed post4 -> residual-C3 order on local OOF predictions."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.analyze_v4_oof_residual_differentials import (  # noqa: E402
    CONTEXTS,
    apply_table,
    context_values,
    differential_table,
    source_artifact,
)
from experiments.finalize_v4_oof_direction_locked import nested_base  # noqa: E402
from experiments.analyze_v4_temporal_residual_ridge import (  # noqa: E402
    MEDIAN_OFFSET,
    REQUIRED_LOCAL,
    json_safe,
    score,
)


PRED = ROOT / "experiments/results/predictions"
REPORT = ROOT / "experiments/results/v4_post4_c3_source.json"
MODEL_KEY = "catboost_numeric"
TARGETS = (2020, 2021, 2022, 2023, 2024)
POST_K = (300.0, 2000.0, 800.0, 2000.0)
POST_WEIGHT = (0.20, 0.825, 0.280, 0.45)


def load(path: Path) -> dict[str, np.ndarray]:
    with np.load(path) as archive:
        return {key: np.asarray(archive[key]) for key in archive.files}


def raw_score(y: np.ndarray, p: np.ndarray) -> float:
    return float(score(y, np.clip(p, 0.0, 1.0))["raw_competition_score"])


def fit_scalar(direction: np.ndarray, residual: np.ndarray) -> tuple[float, float]:
    denominator = float(np.dot(direction, direction))
    raw = float(np.dot(direction, residual) / denominator) if denominator else 0.0
    return raw, float(np.clip(raw, -1.0, 1.0))


def nested_deviation(
    parent: np.ndarray, child: np.ndarray, y: np.ndarray, k: float
) -> tuple[np.ndarray, np.ndarray]:
    order = np.argsort(child, kind="stable")
    ys, parents, children = y[order], parent[order], child[order]
    unique, starts = np.unique(children, return_index=True)
    counts = np.diff(np.append(starts, len(children)))
    cell_mean = np.add.reduceat(ys, starts) / counts
    cell_parent = parents[starts]
    parent_order = np.argsort(parent, kind="stable")
    py, pp = y[parent_order], parent[parent_order]
    parent_unique, parent_starts = np.unique(pp, return_index=True)
    parent_counts = np.diff(np.append(parent_starts, len(pp)))
    parent_mean = np.add.reduceat(py, parent_starts) / parent_counts
    deviation = cell_mean - parent_mean[np.searchsorted(parent_unique, cell_parent)]
    return unique, counts * deviation / (counts + k)


def lookup(unique: np.ndarray, values: np.ndarray, keys: np.ndarray) -> np.ndarray:
    index = np.clip(np.searchsorted(unique, keys), 0, len(unique) - 1)
    valid = unique[index] == keys
    result = np.zeros(len(keys), dtype=np.float64)
    result[valid] = values[index[valid]]
    return result


def group_keys(frame: pd.DataFrame) -> list[tuple[np.ndarray, np.ndarray]]:
    pitcher = frame["pitcher_id"].to_numpy(dtype=np.int64)
    batter_hand = frame["batter_hand"].to_numpy(dtype=np.int64)
    balls = frame["balls_before"].to_numpy(dtype=np.int64)
    strikes = frame["strikes_before"].to_numpy(dtype=np.int64)
    runner = (frame["num_runners_on"].to_numpy(dtype=np.int64) > 0).astype(np.int64)
    platoon = pitcher * 10 + batter_hand
    advantage = platoon * 10 + (strikes > balls).astype(np.int64)
    return [
        (pitcher, platoon),
        (platoon, advantage),
        (advantage, advantage * 100 + (balls * 4 + strikes)),
        (platoon, platoon * 10 + runner),
    ]


def post4(history: pd.DataFrame, target: pd.DataFrame) -> np.ndarray:
    history_axes = group_keys(history)
    target_axes = group_keys(target)
    labels = history["control_success"].to_numpy(dtype=np.float64)
    parts = []
    for (parent, child), (_, target_child), k in zip(
        history_axes, target_axes, POST_K
    ):
        parts.append(lookup(*nested_deviation(parent, child, labels, k), target_child))
    return np.column_stack(parts) @ np.asarray(POST_WEIGHT, dtype=np.float64)


def main() -> None:
    raw = pd.read_csv(
        ROOT / "open/data/train.csv",
        usecols=["season", "pitcher_id", "pitcher_hand", "batter_hand",
                 "balls_before", "strikes_before", "num_runners_on", "control_success"],
        encoding="utf-8-sig", low_memory=False,
    )
    artifacts = {year: source_artifact(year) for year in TARGETS}
    frames = {
        year: raw.iloc[artifacts[year]["row_index"].astype(np.int64)].reset_index(drop=True)
        for year in TARGETS
    }
    y = {year: artifacts[year]["y"].astype(np.float64) for year in TARGETS}
    model = {year: artifacts[year][MODEL_KEY].astype(np.float64) for year in TARGETS}
    post = {
        year: post4(raw.loc[raw["season"] < year], frames[year]) for year in TARGETS
    }
    post_model = {year: np.clip(model[year] + post[year], 0.0, 1.0) for year in TARGETS}

    c3_axes: dict[int, dict[str, np.ndarray]] = {}
    c3 = {}
    for target in (2022, 2023, 2024):
        sources = (target - 2, target - 1)
        pitcher = np.concatenate(
            [frames[year]["pitcher_id"].to_numpy(dtype=np.int64) for year in sources]
        )
        residual = np.concatenate([y[year] - post_model[year] for year in sources])
        source_context = {name: [] for name in CONTEXTS}
        for year in sources:
            values = context_values(frames[year])
            for name in CONTEXTS:
                source_context[name].append(values[name])
        target_context = context_values(frames[target])
        target_pitcher = frames[target]["pitcher_id"].to_numpy(dtype=np.int64)
        c3_axes[target] = {}
        for name, k in CONTEXTS.items():
            table = differential_table(
                pitcher, np.concatenate(source_context[name]), residual, k
            )
            c3_axes[target][name] = apply_table(
                table, target_pitcher, target_context[name]
            )
        c3[target] = sum(c3_axes[target].values())

    accepted = {
        year: load(PRED / f"v4_routed_tabm_stack_locked_{year}.npz")
        for year in (2022, 2023, 2024)
    }
    accepted_prediction = {
        year: accepted[year]["routed_tabm_stack"].astype(np.float64)
        for year in accepted
    }
    route_r = {year: accepted[year]["game_type_r"].astype(bool) for year in accepted}
    accepted_scores = {
        year: raw_score(y[year], accepted_prediction[year]) for year in accepted
    }

    variants = {
        "source_plus_post4": {year: post_model[year] for year in accepted},
        "source_plus_post4_c3": {
            year: np.clip(post_model[year] + c3[year], 0.0, 1.0) for year in accepted
        },
    }
    screens = {}
    directions = {}
    for name, predictions in variants.items():
        values = {
            year: np.where(
                route_r[year], predictions[year] - accepted_prediction[year], 0.0
            )
            for year in accepted
        }
        gamma22_raw, gamma22 = fit_scalar(
            values[2022], y[2022] - accepted_prediction[2022]
        )
        gamma23_raw, gamma23 = fit_scalar(
            values[2023], y[2023] - accepted_prediction[2023]
        )
        gain22 = raw_score(
            y[2022], accepted_prediction[2022] + gamma22 * values[2022]
        ) - accepted_scores[2022]
        transfer23 = raw_score(
            y[2023], accepted_prediction[2023] + gamma22 * values[2023]
        ) - accepted_scores[2023]
        ratio = abs(gamma23 / gamma22) if abs(gamma22) > 1e-9 else float("inf")
        screens[name] = {
            "gamma_fit_2022_raw": gamma22_raw, "gamma_fit_2022": gamma22,
            "gain_fit_2022": gain22, "transfer_gain_2023": transfer23,
            "gamma_fit_2023_raw": gamma23_raw, "gamma_fit_2023": gamma23,
            "gamma_abs_ratio": ratio,
            "coefficient_stable": bool(gamma22 * gamma23 > 0 and 0.5 <= ratio <= 2.0),
        }
        directions[name] = values

    eligible = [
        name for name, row in screens.items()
        if row["gain_fit_2022"] > 0.05 and row["transfer_gain_2023"] > 0.05
        and row["coefficient_stable"]
    ]
    selected = max(
        eligible, key=lambda name: float(screens[name]["transfer_gain_2023"]),
        default="none",
    )
    nested_artifacts = {
        year: load(PRED / f"v4_nested_direction_stack_{year}.npz")
        for year in (2023, 2024)
    }
    base = {year: nested_base(year, nested_artifacts[year]) for year in nested_artifacts}
    base_scores = {year: raw_score(y[year], base[year]) for year in base}
    confirmations = {}
    payload = {
        "y": y[2024], "row_index": accepted[2024]["row_index"],
        "cluster": accepted[2024]["cluster"], "base": base[2024],
    }
    for name in eligible:
        values = directions[name]
        gamma_raw, gamma = fit_scalar(values[2023], y[2023] - base[2023])
        prediction = {
            year: np.clip(base[year] + gamma * values[year], 0.0, 1.0)
            for year in base
        }
        scores = {year: raw_score(y[year], prediction[year]) for year in prediction}
        confirmations[name] = {
            "gamma_fit_2023_raw": gamma_raw, "gamma_fit_2023": gamma,
            "scores": scores,
            "gains_over_base": {year: scores[year] - base_scores[year] for year in base},
            "expected_lb_median": scores[2024] + MEDIAN_OFFSET,
        }
        payload[f"candidate_{name}"] = prediction[2024]
        payload[f"direction_{name}"] = values[2024]

    # Add the already admitted TabTransformer only to the historically selected arm.
    stacked = None
    if selected != "none":
        selected_values = directions[selected]
        _, selected_gamma = fit_scalar(selected_values[2023], y[2023] - base[2023])
        selected_prediction = {
            year: np.clip(base[year] + selected_gamma * selected_values[year], 0.0, 1.0)
            for year in base
        }
        tab = {
            year: load(PRED / f"v4_tabtransformer_seed_ensemble_{year}.npz")
            for year in base
        }
        tab_direction = {
            year: tab[year]["direction_tabtransformer_seed_average"].astype(np.float64)
            for year in base
        }
        tab_gamma_raw, tab_gamma = fit_scalar(
            tab_direction[2023], y[2023] - selected_prediction[2023]
        )
        prediction = {
            year: np.clip(selected_prediction[year] + tab_gamma * tab_direction[year], 0.0, 1.0)
            for year in base
        }
        scores = {year: raw_score(y[year], prediction[year]) for year in prediction}
        stacked = {
            "selected_source": selected,
            "tab_gamma_fit_2023_raw": tab_gamma_raw,
            "tab_gamma_fit_2023": tab_gamma,
            "scores": scores,
            "expected_lb_median": scores[2024] + MEDIAN_OFFSET,
            "required_local_score": REQUIRED_LOCAL,
            "crosses_required_local_score": scores[2024] > REQUIRED_LOCAL,
        }
        payload["selected_plus_tabtransformer"] = prediction[2024]

    output = PRED / "v4_post4_c3_source_2024.npz"
    np.savez_compressed(output, **payload)
    report = {
        "protocol": {
            "official_train_only": True,
            "external_model_artifacts_used": False,
            "test_rows_read": False,
            "fixed_post4_precedes_residual_c3": True,
            "screen_fit_year": 2022,
            "screen_transfer_year": 2023,
            "coefficient_refit_year": 2023,
            "confirmation_year": 2024,
            "selection_does_not_read_2024_labels": True,
        },
        "post4": {"k": POST_K, "weights": POST_WEIGHT},
        "c3": CONTEXTS,
        "source_scores": {
            year: {
                "raw": raw_score(y[year], model[year]),
                "post4": raw_score(y[year], post_model[year]),
                "post4_c3": raw_score(y[year], post_model[year] + c3[year]),
            }
            for year in (2022, 2023, 2024)
        },
        "screens": screens,
        "selected_source": selected,
        "base_scores": base_scores,
        "confirmations": confirmations,
        "selected_plus_tabtransformer": stacked,
        "prediction_artifact": str(output.relative_to(ROOT)),
        "warning": "2024 values are diagnostic confirmations, not selection inputs.",
    }
    REPORT.write_text(json.dumps(json_safe(report), ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(json_safe(report), ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
