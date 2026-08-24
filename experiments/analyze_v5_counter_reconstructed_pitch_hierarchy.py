#!/usr/bin/env python3
"""Source screen for dense counter-reconstructed pitch-group supervision."""

from __future__ import annotations

import gc
import hashlib
import json
import os
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.analyze_v4_pitchtype_failure_prior import (  # noqa: E402
    normalize_fine_pitch_type,
)
from experiments.analyze_v5_fine_pitchtype_latent import (  # noqa: E402
    FINE_TYPES,
    PREDICTIONS,
    SOURCE_YEARS,
    TARGET,
    evaluate,
    fit_selector,
    json_safe,
    load_anchor,
    prepare_catboost,
)
from experiments.run_baselines import (  # noqa: E402
    FEATURES as BASE_FEATURES,
    RANDOM_SEED,
)
from experiments.run_e20r_rolling import load_joined_trackman  # noqa: E402
from experiments.run_v5_h1_residual import (  # noqa: E402
    cluster_bootstrap_score_gain,
)


TRAIN = ROOT / "open/data/train.csv"
PREREG = (
    ROOT
    / "experiments/params/v5_counter_reconstructed_pitch_hierarchy_preregister.json"
)
REPORT = (
    ROOT
    / "experiments/results/v5_counter_reconstructed_pitch_hierarchy_source.json"
)
GROUPS = ("fastball", "breaking", "offspeed", "other")
PITCHMIX_COLUMNS = (
    "asof_pitcher_fastball_rate",
    "asof_pitcher_breaking_rate",
    "asof_pitcher_offspeed_rate",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_source() -> tuple[pd.DataFrame, dict[str, Any]]:
    with np.load(
        PREDICTIONS / "v4_m3_c_backtest_2021_2021.npz", allow_pickle=False
    ) as archive:
        last_index = int(np.max(archive["row_index"]))
    columns = list(
        dict.fromkeys(["row_id", *BASE_FEATURES, TARGET])
    )
    frame = pd.read_csv(TRAIN, usecols=columns, nrows=last_index + 1)
    if set(frame["season"].unique()) != {2019, 2020, 2021}:
        raise ValueError("Dense pitch hierarchy read a control label after 2021")
    frame["row_id"] = frame["row_id"].astype(str)

    joined = load_joined_trackman()
    joined = joined.loc[joined["season"].le(2021)].copy()
    joined["row_id"] = joined["row_id"].astype(str)
    labels = joined[
        ["row_id", "pitch_type_group", "tagged_pitch_type", "auto_pitch_type"]
    ].drop_duplicates("row_id", keep="first")
    labels["coarse_trackman"] = (
        labels["pitch_type_group"]
        .astype("string")
        .where(labels["pitch_type_group"].isin(GROUPS), "other")
    )
    labels["fine_tagged"] = normalize_fine_pitch_type(labels["tagged_pitch_type"])
    labels["fine_auto"] = normalize_fine_pitch_type(labels["auto_pitch_type"])
    label_map = labels.set_index("row_id")
    for column in ("coarse_trackman", "fine_tagged", "fine_auto"):
        frame[column] = frame["row_id"].map(label_map[column])
    metadata = {
        "source_rows": int(len(frame)),
        "source_trackman_rows": int(frame["coarse_trackman"].notna().sum()),
        "trackman_group_counts": {
            str(key): int(value)
            for key, value in frame["coarse_trackman"].value_counts().items()
        },
    }
    del joined, labels, label_map
    gc.collect()
    return frame, metadata


def derive_coarse_labels(frame: pd.DataFrame) -> pd.Series:
    """Recover each historical pitch group from the next as-of counter row."""
    n = (
        pd.to_numeric(frame["asof_pitcher_pitchmix_n"], errors="coerce")
        .fillna(0)
        .to_numpy(dtype=np.int64)
    )
    component_counts = np.column_stack(
        [
            np.rint(
                pd.to_numeric(frame[column], errors="coerce")
                .fillna(0.0)
                .to_numpy(dtype=np.float64)
                * n
            ).astype(np.int64)
            for column in PITCHMIX_COLUMNS
        ]
    )
    work = pd.DataFrame(
        {
            "pitcher_id": frame["pitcher_id"].to_numpy(),
            "n": n,
            "fastball": component_counts[:, 0],
            "breaking": component_counts[:, 1],
            "offspeed": component_counts[:, 2],
        },
        index=frame.index,
    )
    grouped = work.groupby("pitcher_id", sort=False, observed=True)
    next_n = grouped["n"].shift(-1)
    deltas = pd.DataFrame(
        {
            group: grouped[group].shift(-1) - work[group]
            for group in GROUPS[:3]
        },
        index=frame.index,
    )
    delta_sum = deltas.sum(axis=1)
    valid = next_n.eq(work["n"] + 1)
    for group in GROUPS[:3]:
        valid &= deltas[group].isin((0.0, 1.0))
    valid &= delta_sum.isin((0.0, 1.0))
    labels = pd.Series(pd.NA, index=frame.index, dtype="string")
    for group in GROUPS[:3]:
        labels.loc[valid & deltas[group].eq(1.0)] = group
    labels.loc[valid & delta_sum.eq(0.0)] = "other"
    return labels


def fit_coarse_selector(
    history: pd.DataFrame, valid_r: pd.DataFrame, year: int, prereg: dict[str, Any]
) -> tuple[np.ndarray, dict[str, Any]]:
    labeled = history.loc[history["coarse_reconstructed"].notna()].copy()
    train_x, categorical = prepare_catboost(labeled)
    valid_x, valid_categorical = prepare_catboost(valid_r)
    if categorical != valid_categorical:
        raise AssertionError("Coarse selector categorical schema changed")
    settings = prereg["selector"]
    model = CatBoostClassifier(
        loss_function="MultiClass",
        iterations=int(settings["iterations"]),
        depth=int(settings["depth"]),
        learning_rate=float(settings["learning_rate"]),
        l2_leaf_reg=float(settings["l2_leaf_reg"]),
        random_seed=RANDOM_SEED + 700 + year,
        allow_writing_files=False,
        thread_count=6,
        task_type=(
            "GPU"
            if os.environ.get("V2_BOOSTER_DEVICE", "cpu").lower() == "gpu"
            else "CPU"
        ),
    )
    started = time.perf_counter()
    model.fit(
        train_x,
        labeled["coarse_reconstructed"].astype(str),
        cat_features=categorical,
        verbose=False,
    )
    raw = np.asarray(model.predict_proba(valid_x), dtype=np.float64)
    probabilities = np.zeros((len(valid_r), len(GROUPS)), dtype=np.float64)
    classes = [str(value) for value in model.classes_]
    for source_index, label in enumerate(classes):
        if label in GROUPS:
            probabilities[:, GROUPS.index(label)] = raw[:, source_index]
    denominator = probabilities.sum(axis=1)
    invalid = denominator <= 0.0
    probabilities[invalid] = 1.0 / len(GROUPS)
    denominator[invalid] = 1.0
    probabilities /= denominator[:, None]

    diagnostics: dict[str, Any] = {
        "history_labeled_rows": int(len(labeled)),
        "valid_rows": int(len(valid_r)),
        "classes": classes,
        "fit_seconds": float(time.perf_counter() - started),
        "probability_mean": probabilities.mean(axis=0).tolist(),
        "current_pitch_group_used_for_prediction": False,
    }
    for truth_column, prefix in (
        ("coarse_reconstructed", "reconstructed"),
        ("coarse_trackman", "trackman"),
    ):
        matched = valid_r[truth_column].notna().to_numpy(dtype=bool)
        truth = valid_r.loc[matched, truth_column].astype(str).to_numpy()
        truth_index = np.asarray(
            [GROUPS.index(value) for value in truth], dtype=np.int16
        )
        selected = probabilities[matched]
        chosen = selected[np.arange(len(selected)), truth_index]
        diagnostics[f"{prefix}_matched_rows"] = int(matched.sum())
        diagnostics[f"{prefix}_top1_accuracy"] = float(
            np.mean(selected.argmax(axis=1) == truth_index)
        )
        diagnostics[f"{prefix}_log_loss"] = float(
            -np.mean(np.log(np.clip(chosen, 1e-12, 1.0)))
        )
    del model, train_x, valid_x, raw, labeled
    gc.collect()
    return probabilities, diagnostics


def build_control_matrices(
    history: pd.DataFrame,
    valid_r: pd.DataFrame,
    label_column: str,
    groups: tuple[str, ...],
    outcome_k: float,
    repertoire_k: float,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    work = history.loc[history[label_column].notna()].copy()
    season_mean = work.groupby("season", observed=True)[TARGET].transform("mean")
    work["centered_control"] = work[TARGET].astype(float) - season_mean
    type_prior = (
        work.groupby(label_column, observed=True)["centered_control"]
        .mean()
        .reindex(groups)
        .fillna(0.0)
    )
    stats = work.groupby(
        ["pitcher_id", label_column], sort=False, observed=True
    )["centered_control"].agg(["sum", "count"])
    stats = stats.reset_index()
    stats["value"] = (
        stats["sum"].to_numpy(dtype=np.float64)
        + outcome_k * stats[label_column].map(type_prior).to_numpy(dtype=np.float64)
    ) / (stats["count"].to_numpy(dtype=np.float64) + outcome_k)
    values = stats.pivot(index="pitcher_id", columns=label_column, values="value")
    values = values.reindex(columns=groups)
    for group in groups:
        values[group] = values[group].fillna(float(type_prior[group]))

    counts = (
        work.groupby(["pitcher_id", label_column], sort=False, observed=True)
        .size()
        .unstack(fill_value=0)
        .reindex(columns=groups, fill_value=0)
    )
    global_mix = (
        work[label_column]
        .value_counts(normalize=True)
        .reindex(groups)
        .fillna(0.0)
        .to_numpy(dtype=np.float64)
    )
    count_matrix = counts.to_numpy(dtype=np.float64)
    total = count_matrix.sum(axis=1)
    mix = (count_matrix + repertoire_k * global_mix[None, :]) / (
        total[:, None] + repertoire_k
    )
    mix_table = pd.DataFrame(mix, index=counts.index, columns=groups)

    pitchers = valid_r["pitcher_id"].to_numpy()
    q_matrix = values.reindex(pitchers).to_numpy(dtype=np.float64)
    mix_matrix = mix_table.reindex(pitchers).to_numpy(dtype=np.float64)
    unseen = np.isnan(mix_matrix).all(axis=1)
    if unseen.any():
        mix_matrix[unseen] = global_mix
    for column, group in enumerate(groups):
        missing_q = np.isnan(q_matrix[:, column])
        q_matrix[missing_q, column] = float(type_prior[group])
        missing_mix = np.isnan(mix_matrix[:, column])
        mix_matrix[missing_mix, column] = global_mix[column]
    mix_matrix /= np.maximum(mix_matrix.sum(axis=1, keepdims=True), 1e-12)
    return q_matrix, mix_matrix, {
        "history_rows": int(len(work)),
        "state_pitchers": int(len(values)),
        "valid_unseen_pitchers": int(unseen.sum()),
        "global_mix": {
            group: float(global_mix[index]) for index, group in enumerate(groups)
        },
    }


def learned_fine_mapping(history: pd.DataFrame, source: str) -> dict[str, str]:
    fine_column = f"fine_{source}"
    usable = history.loc[
        history[fine_column].notna() & history["coarse_trackman"].notna(),
        [fine_column, "coarse_trackman"],
    ]
    mapping: dict[str, str] = {}
    for fine_type in FINE_TYPES:
        counts = usable.loc[usable[fine_column].eq(fine_type), "coarse_trackman"].value_counts()
        mapping[fine_type] = str(counts.index[0]) if len(counts) else "other"
    return mapping


def project_fine_probabilities(
    fine: np.ndarray, coarse: np.ndarray, mapping: dict[str, str]
) -> np.ndarray:
    result = np.zeros_like(fine, dtype=np.float64)
    for group_index, group in enumerate(GROUPS):
        columns = [
            index
            for index, fine_type in enumerate(FINE_TYPES)
            if mapping[fine_type] == group
        ]
        if not columns:
            continue
        conditional = fine[:, columns]
        denominator = conditional.sum(axis=1, keepdims=True)
        conditional = np.divide(
            conditional,
            denominator,
            out=np.full_like(conditional, 1.0 / len(columns)),
            where=denominator > 0.0,
        )
        result[:, columns] = conditional * coarse[:, [group_index]]
    denominator = result.sum(axis=1, keepdims=True)
    missing = denominator[:, 0] <= 0.0
    result[missing] = fine[missing]
    denominator[missing] = result[missing].sum(axis=1, keepdims=True)
    result /= np.maximum(denominator, 1e-12)
    return result


def direction_record(
    anchor: dict[str, np.ndarray],
    base: np.ndarray,
    regular: np.ndarray,
    delta_r: np.ndarray,
) -> dict[str, Any]:
    """Cache the exact quadratic Brier change for one prediction direction."""
    y_all = anchor["y"].astype(np.float64)
    y_r = y_all[regular]
    base_r = base[regular].astype(np.float64)
    delta = np.asarray(delta_r, dtype=np.float64)
    error = base_r - y_r
    rate_all = float(y_all.mean())
    rate_r = float(y_r.mean())
    return {
        "base_r": base_r,
        "y_r": y_r,
        "delta": delta,
        "linear": float(2.0 * np.mean(error * delta)),
        "quadratic": float(np.mean(delta * delta)),
        "r_share": float(len(y_r) / len(y_all)),
        "reference_all": float(rate_all * (1.0 - rate_all)),
        "reference_r": float(rate_r * (1.0 - rate_r)),
    }


def gains_from_record(record: dict[str, Any], scale: float) -> tuple[float, float]:
    """Evaluate a scale without rescanning rows unless probability clipping binds."""
    base_r = record["base_r"]
    delta = record["delta"]
    shifted = base_r + scale * delta
    if float(shifted.min()) >= 1e-6 and float(shifted.max()) <= 1.0 - 1e-6:
        brier_delta_r = (
            scale * record["linear"] + scale * scale * record["quadratic"]
        )
    else:
        candidate_r = np.clip(shifted, 1e-6, 1.0 - 1e-6)
        brier_delta_r = float(
            np.mean((candidate_r - record["y_r"]) ** 2)
            - np.mean((base_r - record["y_r"]) ** 2)
        )
    r_gain = -100_000.0 * brier_delta_r / record["reference_r"]
    full_gain = (
        -100_000.0
        * record["r_share"]
        * brier_delta_r
        / record["reference_all"]
    )
    return float(full_gain), float(r_gain)


def main() -> None:
    if REPORT.exists():
        raise FileExistsError(f"Preserve immutable hierarchy report: {REPORT}")
    prereg = json.loads(PREREG.read_text(encoding="utf-8"))
    if prereg["status"] != "preregistered_before_source_metrics":
        raise ValueError("Unexpected preregistration status")
    started = time.perf_counter()
    frame, linkage_meta = load_source()
    folds: dict[int, dict[str, Any]] = {}
    semantic: dict[str, Any] = {}
    semantic_ok = True

    for year in SOURCE_YEARS:
        anchor = load_anchor(year)
        valid = frame.iloc[anchor["row_index"]].copy()
        if not valid["season"].eq(year).all():
            raise ValueError(f"{year}: anchor season mismatch")
        if not np.array_equal(
            valid[TARGET].to_numpy(dtype=np.int8), anchor["y"].astype(np.int8)
        ):
            raise ValueError(f"{year}: anchor target mismatch")
        history_all = frame.loc[frame["season"].lt(year)].copy()
        history_all["coarse_reconstructed"] = derive_coarse_labels(history_all)
        valid["coarse_reconstructed"] = derive_coarse_labels(valid)
        history_r = history_all.loc[history_all["game_type"].eq("R")].copy()
        regular = valid["game_type"].eq("R").to_numpy(dtype=bool)
        valid_r = valid.loc[regular].copy()

        history_coverage = float(history_r["coarse_reconstructed"].notna().mean())
        agreement_mask = (
            history_r["coarse_reconstructed"].notna()
            & history_r["coarse_trackman"].notna()
        )
        agreement = float(
            history_r.loc[agreement_mask, "coarse_reconstructed"].eq(
                history_r.loc[agreement_mask, "coarse_trackman"]
            ).mean()
        )
        fold_ok = (
            history_coverage
            >= float(prereg["coarse_label_reconstruction"]["required_history_coverage"])
            and agreement
            >= float(prereg["coarse_label_reconstruction"]["required_trackman_agreement"])
        )
        semantic_ok &= fold_ok
        semantic[str(year)] = {
            "history_r_rows": int(len(history_r)),
            "history_reconstructed_rows": int(
                history_r["coarse_reconstructed"].notna().sum()
            ),
            "history_coverage": history_coverage,
            "history_trackman_comparison_rows": int(agreement_mask.sum()),
            "history_trackman_agreement": agreement,
            "valid_r_reconstructed_coverage": float(
                valid_r["coarse_reconstructed"].notna().mean()
            ),
            "semantic_gate_pass": bool(fold_ok),
            "history_reconstructed_counts": {
                str(key): int(value)
                for key, value in history_r["coarse_reconstructed"].value_counts().items()
            },
        }
        folds[year] = {
            "anchor": anchor,
            "valid": valid,
            "valid_r": valid_r,
            "history_r": history_r,
            "regular": regular,
            "game_type": valid["game_type"].astype(str).to_numpy(),
        }
        del history_all

    if not semantic_ok:
        payload = {
            "experiment_id": prereg["experiment_id"],
            "status": "failed_semantic_gate",
            "preregister": str(PREREG.relative_to(ROOT)),
            "preregister_sha256": sha256(PREREG),
            "linkage": linkage_meta,
            "semantic_audit": semantic,
            "target_metrics_computed": False,
            "test_rows_read": False,
            "decision": "close without target metrics or 2022+ labels",
            "elapsed_seconds": float(time.perf_counter() - started),
        }
        REPORT.write_text(
            json.dumps(json_safe(payload), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(json.dumps(json_safe(payload), ensure_ascii=False, indent=2))
        return

    coarse_probabilities: dict[int, np.ndarray] = {}
    coarse_selector_meta: dict[str, Any] = {}
    fine_probabilities: dict[tuple[int, str], np.ndarray] = {}
    fine_selector_meta: dict[str, Any] = {}
    mappings: dict[tuple[int, str], dict[str, str]] = {}
    coarse_states: dict[tuple[int, int, int], tuple[np.ndarray, np.ndarray]] = {}
    fine_states: dict[
        tuple[int, str, int, int], tuple[np.ndarray, np.ndarray]
    ] = {}
    state_meta: dict[str, Any] = {}

    outcome_ks = [int(value) for value in prereg["source_grid"]["outcome_k"]]
    repertoire_ks = [int(value) for value in prereg["source_grid"]["repertoire_k"]]
    sources = [str(value) for value in prereg["source_grid"]["fine_label_sources"]]
    for year in SOURCE_YEARS:
        fold = folds[year]
        history_r = fold["history_r"]
        valid_r = fold["valid_r"]
        coarse_probabilities[year], coarse_selector_meta[str(year)] = (
            fit_coarse_selector(history_r, valid_r, year, prereg)
        )
        for outcome_k in outcome_ks:
            for repertoire_k in repertoire_ks:
                q, mix, meta = build_control_matrices(
                    history_r,
                    valid_r,
                    "coarse_reconstructed",
                    GROUPS,
                    float(outcome_k),
                    float(repertoire_k),
                )
                coarse_states[(year, outcome_k, repertoire_k)] = (q, mix)
                state_meta[f"{year}:coarse:k{outcome_k}:r{repertoire_k}"] = meta
        for source in sources:
            probabilities, meta = fit_selector(history_r, valid_r, source, year)
            fine_probabilities[(year, source)] = probabilities
            fine_selector_meta[f"{year}_{source}"] = meta
            mappings[(year, source)] = learned_fine_mapping(history_r, source)
            for outcome_k in outcome_ks:
                for repertoire_k in repertoire_ks:
                    q, mix, meta = build_control_matrices(
                        history_r,
                        valid_r,
                        f"fine_{source}",
                        FINE_TYPES,
                        float(outcome_k),
                        float(repertoire_k),
                    )
                    fine_states[(year, source, outcome_k, repertoire_k)] = (q, mix)
                    state_meta[
                        f"{year}:{source}:k{outcome_k}:r{repertoire_k}"
                    ] = meta

    selector_weights = [
        float(value) for value in prereg["source_grid"]["selector_weights"]
    ]
    alphas = [float(value) for value in prereg["source_grid"]["hierarchy_alphas"]]
    gammas = [float(value) for value in prereg["source_grid"]["gammas"]]
    trials: list[dict[str, Any]] = []

    coarse_direction_cache: dict[tuple[int, int, int], np.ndarray] = {}
    coarse_record_cache: dict[tuple[int, int, int], dict[str, Any]] = {}
    for year in SOURCE_YEARS:
        fold = folds[year]
        base = fold["anchor"]["catboost_outcome"].astype(np.float64)
        for outcome_k in outcome_ks:
            for repertoire_k in repertoire_ks:
                q, mix = coarse_states[(year, outcome_k, repertoire_k)]
                delta = np.sum((coarse_probabilities[year] - mix) * q, axis=1)
                key = (year, outcome_k, repertoire_k)
                coarse_direction_cache[key] = delta
                coarse_record_cache[key] = direction_record(
                    fold["anchor"], base, fold["regular"], delta
                )

    for outcome_k in outcome_ks:
        for repertoire_k in repertoire_ks:
            for selector_weight in selector_weights:
                for gamma in gammas:
                    years: dict[str, Any] = {}
                    for year in SOURCE_YEARS:
                        full_gain, r_gain = gains_from_record(
                            coarse_record_cache[(year, outcome_k, repertoire_k)],
                            selector_weight * gamma,
                        )
                        years[str(year)] = {
                            "full_gain": full_gain,
                            "R_gain": r_gain,
                        }
                    trials.append(
                        {
                            "family": "dense_coarse",
                            "outcome_k": outcome_k,
                            "repertoire_k": repertoire_k,
                            "selector_weight": selector_weight,
                            "hierarchy_alpha": 0.0,
                            "gamma": gamma,
                            "effective_scale": selector_weight * gamma,
                            "min_full_gain": float(
                                min(value["full_gain"] for value in years.values())
                            ),
                            "min_R_gain": float(
                                min(value["R_gain"] for value in years.values())
                            ),
                            "mean_full_gain": float(
                                np.mean([value["full_gain"] for value in years.values()])
                            ),
                            "years": years,
                        }
                    )

    projected_cache: dict[tuple[int, str], np.ndarray] = {}
    for year in SOURCE_YEARS:
        for source in sources:
            projected_cache[(year, source)] = project_fine_probabilities(
                fine_probabilities[(year, source)],
                coarse_probabilities[year],
                mappings[(year, source)],
            )

    hierarchy_direction_cache: dict[
        tuple[int, str, int, int, float], np.ndarray
    ] = {}
    hierarchy_record_cache: dict[
        tuple[int, str, int, int, float], dict[str, Any]
    ] = {}
    for year in SOURCE_YEARS:
        fold = folds[year]
        base = fold["anchor"]["catboost_outcome"].astype(np.float64)
        for source in sources:
            raw = fine_probabilities[(year, source)]
            projected = projected_cache[(year, source)]
            for outcome_k in outcome_ks:
                for repertoire_k in repertoire_ks:
                    q, mix = fine_states[
                        (year, source, outcome_k, repertoire_k)
                    ]
                    for alpha in alphas:
                        hierarchical = (1.0 - alpha) * raw + alpha * projected
                        delta = np.sum((hierarchical - mix) * q, axis=1)
                        key = (year, source, outcome_k, repertoire_k, alpha)
                        hierarchy_direction_cache[key] = delta
                        hierarchy_record_cache[key] = direction_record(
                            fold["anchor"], base, fold["regular"], delta
                        )
    for source in sources:
        for outcome_k in outcome_ks:
            for repertoire_k in repertoire_ks:
                for alpha in alphas:
                    for selector_weight in selector_weights:
                        for gamma in gammas:
                            years: dict[str, Any] = {}
                            for year in SOURCE_YEARS:
                                full_gain, r_gain = gains_from_record(
                                    hierarchy_record_cache[
                                        (
                                            year,
                                            source,
                                            outcome_k,
                                            repertoire_k,
                                            alpha,
                                        )
                                    ],
                                    selector_weight * gamma,
                                )
                                years[str(year)] = {
                                    "full_gain": full_gain,
                                    "R_gain": r_gain,
                                }
                            trials.append(
                                {
                                    "family": "hierarchical_fine",
                                    "source": source,
                                    "outcome_k": outcome_k,
                                    "repertoire_k": repertoire_k,
                                    "selector_weight": selector_weight,
                                    "hierarchy_alpha": alpha,
                                    "gamma": gamma,
                                    "effective_scale": selector_weight * gamma,
                                    "min_full_gain": float(
                                        min(
                                            value["full_gain"]
                                            for value in years.values()
                                        )
                                    ),
                                    "min_R_gain": float(
                                        min(value["R_gain"] for value in years.values())
                                    ),
                                    "mean_full_gain": float(
                                        np.mean(
                                            [
                                                value["full_gain"]
                                                for value in years.values()
                                            ]
                                        )
                                    ),
                                    "years": years,
                                }
                            )

    trials.sort(
        key=lambda row: (
            row["min_full_gain"],
            row["min_R_gain"],
            row["mean_full_gain"],
            row["family"] == "dense_coarse",
            row["outcome_k"],
            row["repertoire_k"],
            -row["gamma"],
        ),
        reverse=True,
    )
    selected = trials[0]

    def selected_delta(year: int) -> np.ndarray:
        if selected["family"] == "dense_coarse":
            return coarse_direction_cache[
                (year, selected["outcome_k"], selected["repertoire_k"])
            ]
        source = selected["source"]
        return hierarchy_direction_cache[
            (
                year,
                source,
                selected["outcome_k"],
                selected["repertoire_k"],
                selected["hierarchy_alpha"],
            )
        ]

    intervals: dict[str, Any] = {}
    selected_details: dict[str, Any] = {}
    for offset, year in enumerate(SOURCE_YEARS):
        fold = folds[year]
        anchor = fold["anchor"]
        base = anchor["catboost_outcome"].astype(np.float64)
        candidate = base.copy()
        candidate[fold["regular"]] = np.clip(
            candidate[fold["regular"]]
            + selected["effective_scale"] * selected_delta(year),
            1e-6,
            1.0 - 1e-6,
        )
        intervals[str(year)] = cluster_bootstrap_score_gain(
            anchor["y"],
            base,
            candidate,
            anchor["cluster"].astype(str),
            fold["regular"],
            2000,
            53400 + offset,
        )
        selected_details[str(year)] = evaluate(
            anchor["y"], base, candidate, fold["game_type"]
        )

    oracle_record_cache: dict[
        tuple[int, str, int, int], tuple[dict[str, Any], int]
    ] = {}
    for family in ("dense_coarse", "fine_tagged", "fine_auto"):
        for outcome_k in outcome_ks:
            for repertoire_k in repertoire_ks:
                for year in SOURCE_YEARS:
                    fold = folds[year]
                    valid_r = fold["valid_r"]
                    if family == "dense_coarse":
                        q, mix = coarse_states[(year, outcome_k, repertoire_k)]
                        labels = valid_r["coarse_reconstructed"]
                        truth_groups = GROUPS
                    else:
                        source = family.removeprefix("fine_")
                        q, mix = fine_states[
                            (year, source, outcome_k, repertoire_k)
                        ]
                        labels = valid_r[f"fine_{source}"]
                        truth_groups = FINE_TYPES
                    matched = labels.notna().to_numpy(dtype=bool)
                    truth_index = np.asarray(
                        [truth_groups.index(value) for value in labels.loc[matched]],
                        dtype=np.int16,
                    )
                    oracle_delta = np.zeros(len(valid_r), dtype=np.float64)
                    baseline_expected = np.sum(mix * q, axis=1)
                    oracle_delta[matched] = (
                        q[matched, truth_index] - baseline_expected[matched]
                    )
                    base = fold["anchor"]["catboost_outcome"].astype(np.float64)
                    oracle_record_cache[(year, family, outcome_k, repertoire_k)] = (
                        direction_record(
                            fold["anchor"], base, fold["regular"], oracle_delta
                        ),
                        int(matched.sum()),
                    )

    oracle_trials: list[dict[str, Any]] = []
    for family in ("dense_coarse", "fine_tagged", "fine_auto"):
        for outcome_k in outcome_ks:
            for repertoire_k in repertoire_ks:
                for gamma in gammas:
                    years: dict[str, Any] = {}
                    for year in SOURCE_YEARS:
                        record, matched_rows = oracle_record_cache[
                            (year, family, outcome_k, repertoire_k)
                        ]
                        full_gain, r_gain = gains_from_record(record, gamma)
                        years[str(year)] = {
                            "full_gain": full_gain,
                            "R_gain": r_gain,
                            "oracle_matched_r_rows": matched_rows,
                        }
                    oracle_trials.append(
                        {
                            "family": family,
                            "outcome_k": outcome_k,
                            "repertoire_k": repertoire_k,
                            "gamma": gamma,
                            "min_full_gain": float(
                                min(value["full_gain"] for value in years.values())
                            ),
                            "min_R_gain": float(
                                min(value["R_gain"] for value in years.values())
                            ),
                            "mean_full_gain": float(
                                np.mean([value["full_gain"] for value in years.values()])
                            ),
                            "years": years,
                        }
                    )
    oracle_trials.sort(
        key=lambda row: (row["min_full_gain"], row["min_R_gain"]), reverse=True
    )
    oracle_selected = oracle_trials[0]

    gate = prereg["source_gate"]
    conditions = {
        "semantic_gate": bool(semantic_ok),
        "minimum_full_gain_each_year": bool(
            selected["min_full_gain"] >= float(gate["minimum_full_gain_each_year"])
        ),
        "minimum_R_gain_each_year": bool(
            selected["min_R_gain"] >= float(gate["minimum_r_gain_each_year"])
        ),
        "R_cluster_ci_lower_positive_each_year": bool(
            all(value["ci_low"] > 0.0 for value in intervals.values())
        ),
        "goal_scale_oracle_headroom": bool(
            oracle_selected["min_full_gain"]
            >= float(
                prereg["diagnostic_oracle"][
                    "required_minimum_full_gain_to_confirm_goal_scale_headroom"
                ]
            )
        ),
    }
    passed = bool(all(conditions.values()))
    payload = {
        "experiment_id": prereg["experiment_id"],
        "status": "passed_source_gate" if passed else "failed_source_gate",
        "preregister": str(PREREG.relative_to(ROOT)),
        "preregister_sha256": sha256(PREREG),
        "policy": {
            "official_data_only": True,
            "test_rows_read": False,
            "latest_control_label_season_used_for_metrics": 2021,
            "current_pitch_group_used_in_candidate": False,
            "validation_reconstructed_label_used_in_candidate": False,
            "row_independent_inference": True,
            "automatic_submission": False,
        },
        "linkage": linkage_meta,
        "semantic_audit": semantic,
        "coarse_selector_diagnostics": coarse_selector_meta,
        "fine_selector_diagnostics": fine_selector_meta,
        "learned_fine_to_coarse_mappings": {
            f"{year}_{source}": mapping
            for (year, source), mapping in mappings.items()
        },
        "candidate_count": int(len(trials)),
        "selected": selected,
        "selected_detailed_metrics": selected_details,
        "selected_R_pitcher_cluster_intervals": intervals,
        "diagnostic_oracle": {
            "eligible_as_candidate": False,
            "eligible_for_expected_lb": False,
            "selected": oracle_selected,
            "candidate_count": int(len(oracle_trials)),
        },
        "conditions": conditions,
        "gate_pass": passed,
        "decision": (
            "freeze selected hierarchy recipe before 2022"
            if passed
            else "close without 2022+ labels"
        ),
        "top_candidates": trials[:30],
        "top_oracle_candidates": oracle_trials[:15],
        "state_metadata": state_meta,
        "elapsed_seconds": float(time.perf_counter() - started),
    }
    REPORT.write_text(
        json.dumps(json_safe(payload), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(
        json.dumps(
            json_safe(
                {
                    "status": payload["status"],
                    "semantic": semantic,
                    "coarse_selector": coarse_selector_meta,
                    "selected": selected,
                    "selected_intervals": intervals,
                    "oracle_selected": oracle_selected,
                    "conditions": conditions,
                    "elapsed_seconds": payload["elapsed_seconds"],
                }
            ),
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
