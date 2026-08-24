"""Deployable conditional integration of historical TrackMan physics teachers."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys
from typing import Any

# Load LightGBM before pandas/sklearn on this Windows runtime.
from lightgbm import LGBMClassifier
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from eda.run_structural_eda import load_trackman  # noqa: E402
from experiments.run_v5_privileged_trackman_teacher_feasibility import (  # noqa: E402
    CATEGORICAL,
    CONTROL_FEATURES,
    FULL_FEATURES,
    MODEL_PARAMS,
    PHYSICS,
    TARGET,
    encode_features,
)

TRAIN = ROOT / "open/data/train.csv"
RESULTS = ROOT / "experiments/results"
PREREG = (
    ROOT
    / "experiments/params/v5_conditional_physics_integration_signal_preregister.json"
)
LINKAGE_REPORT = RESULTS / "v5_partial_trackman_linkage_source.json"
REPORT = RESULTS / "v5_conditional_physics_integration_signal_source.json"
YEARS = (2020, 2021)
MAIN_COLUMNS = [
    "row_id",
    "season",
    "game_type",
    "balls_before",
    "strikes_before",
    "outs_before",
    "pitcher_id",
    "pitcher_team_id",
    "pitcher_hand",
    "batter_hand",
]
QUERY_KEYS = [
    "pitcher_id",
    "pitcher_team_id",
    "balls_before",
    "strikes_before",
    "outs_before",
    "pitcher_hand",
    "batter_hand",
]
REPERTOIRE_COLUMNS = [
    "pitch_type_group",
    "tagged_pitch_type",
    "auto_pitch_type",
    *PHYSICS,
]


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def load_main() -> tuple[pd.DataFrame, pd.Series]:
    features = pd.read_csv(
        TRAIN, usecols=MAIN_COLUMNS, dtype={"row_id": "string"}
    )
    labels = pd.read_csv(TRAIN, usecols=[TARGET])[TARGET].astype(np.int8)
    return features, labels


def augmented_history(
    year: int,
    main: pd.DataFrame,
    labels: pd.Series,
    raw_trackman: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    mapping_path = (
        RESULTS / "predictions"
        / f"v5_partial_trackman_linkage_history_to_{year}.npz"
    )
    with np.load(mapping_path, allow_pickle=False) as archive:
        mapping = pd.DataFrame(
            {
                "row_id": np.asarray(archive["row_id"]).astype(str),
                "trackman_id": np.asarray(archive["trackman_id"]),
                "is_partial": np.asarray(archive["is_partial"], dtype=np.int8),
            }
        )
        allowed = [
            int(value) for value in np.asarray(archive["allowed_seasons"])
        ]
    expected = list(range(2019, year))
    if allowed != expected:
        raise ValueError(f"linkage cutoff mismatch: {year}/{allowed}")
    if mapping["row_id"].duplicated().any() or mapping["trackman_id"].duplicated().any():
        raise ValueError(f"non-unique linkage rows for {year}")
    main_history = main.loc[
        main["season"].lt(year) & main["game_type"].eq("R")
    ].copy()
    main_history[TARGET] = labels.loc[main_history.index].to_numpy(dtype=np.int8)
    raw_payload = raw_trackman[["trackman_id", *REPERTOIRE_COLUMNS]].copy()
    if raw_payload["trackman_id"].duplicated().any():
        raise ValueError("raw TrackMan ID is not unique")
    history = mapping.merge(
        main_history,
        on="row_id",
        how="inner",
        validate="one_to_one",
    ).merge(
        raw_payload,
        on="trackman_id",
        how="inner",
        validate="one_to_one",
    )
    history = history.sort_values(
        ["season", "row_id"], kind="stable"
    ).reset_index(drop=True)
    metadata = {
        "allowed_history_seasons": allowed,
        "mapping_rows": int(len(mapping)),
        "joined_training_rows": int(len(history)),
        "partial_rows": int(history["is_partial"].sum()),
        "row_alignment_fraction": float(len(history) / max(len(mapping), 1)),
        "validation_labels_used_for_fit_or_integration": False,
        "current_validation_trackman_used": False,
    }
    return history, metadata


def deterministic_sample(indices: np.ndarray, maximum: int) -> np.ndarray:
    if len(indices) <= maximum:
        return indices
    positions = np.linspace(0, len(indices) - 1, maximum, dtype=np.int64)
    return indices[positions]


def build_integration_rows(
    history: pd.DataFrame,
    valid: pd.DataFrame,
    maximum: int,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    unique = valid[QUERY_KEYS].drop_duplicates().reset_index(drop=True)
    unique["query_id"] = np.arange(len(unique), dtype=np.int64)
    query_index = pd.MultiIndex.from_frame(unique[QUERY_KEYS])
    query_index.names = QUERY_KEYS
    row_query = pd.Series(
        unique["query_id"].to_numpy(), index=query_index
    ).reindex(pd.MultiIndex.from_frame(valid[QUERY_KEYS])).to_numpy(dtype=np.int64)

    pitcher_groups = {
        key: np.asarray(value, dtype=np.int64)
        for key, value in history.groupby("pitcher_id", sort=False).indices.items()
    }
    team_hand_groups = {
        key: np.asarray(value, dtype=np.int64)
        for key, value in history.groupby(
            ["pitcher_team_id", "pitcher_hand"], sort=False
        ).indices.items()
    }
    hand_groups = {
        key: np.asarray(value, dtype=np.int64)
        for key, value in history.groupby("pitcher_hand", sort=False).indices.items()
    }
    selected_parts: list[np.ndarray] = []
    query_parts: list[np.ndarray] = []
    source_level = np.empty(len(unique), dtype=np.int8)
    samples_per_query = np.empty(len(unique), dtype=np.int16)
    for row in unique.itertuples(index=False):
        pitcher = row.pitcher_id
        team_hand = (row.pitcher_team_id, row.pitcher_hand)
        if pitcher in pitcher_groups:
            indices = pitcher_groups[pitcher]
            level = 0
        elif team_hand in team_hand_groups:
            indices = team_hand_groups[team_hand]
            level = 1
        elif row.pitcher_hand in hand_groups:
            indices = hand_groups[row.pitcher_hand]
            level = 2
        else:
            raise ValueError(f"no repertoire fallback for query {row.query_id}")
        chosen = deterministic_sample(indices, maximum)
        selected_parts.append(chosen)
        query_parts.append(np.full(len(chosen), row.query_id, dtype=np.int64))
        source_level[row.query_id] = level
        samples_per_query[row.query_id] = len(chosen)
    selected = np.concatenate(selected_parts)
    synthetic_query = np.concatenate(query_parts)
    synthetic = history.iloc[selected][REPERTOIRE_COLUMNS].reset_index(drop=True)
    context = unique.iloc[synthetic_query]
    for column in (
        "balls_before",
        "strikes_before",
        "outs_before",
        "pitcher_hand",
        "batter_hand",
    ):
        synthetic[column] = context[column].to_numpy()
    synthetic["query_id"] = synthetic_query
    row_source_level = source_level[row_query]
    metadata = {
        "unique_context_queries": int(len(unique)),
        "synthetic_rows": int(len(synthetic)),
        "samples_per_query_min": int(samples_per_query.min()),
        "samples_per_query_mean": float(samples_per_query.mean()),
        "samples_per_query_max": int(samples_per_query.max()),
        "query_source_levels": {
            "pitcher": int(np.sum(source_level == 0)),
            "team_hand": int(np.sum(source_level == 1)),
            "hand": int(np.sum(source_level == 2)),
        },
        "validation_row_source_coverage": {
            "pitcher": float(np.mean(row_source_level == 0)),
            "team_hand": float(np.mean(row_source_level == 1)),
            "hand": float(np.mean(row_source_level == 2)),
            "total": 1.0,
        },
    }
    return synthetic, synthetic_query, row_query, row_source_level, metadata


def fit_predict(
    history: pd.DataFrame,
    synthetic: pd.DataFrame,
    features: tuple[str, ...],
) -> np.ndarray:
    train_x, synthetic_x, categorical = encode_features(
        history, synthetic, features
    )
    model = LGBMClassifier(**MODEL_PARAMS)
    model.fit(
        train_x,
        np.ascontiguousarray(history[TARGET].to_numpy(dtype=np.float32)),
        categorical_feature=categorical,
    )
    return model.predict_proba(synthetic_x)[:, 1].astype(np.float64)


def integrate(
    predictions: np.ndarray,
    synthetic_query: np.ndarray,
    row_query: np.ndarray,
    query_count: int,
) -> np.ndarray:
    sums = np.bincount(
        synthetic_query, weights=predictions, minlength=query_count
    )
    counts = np.bincount(synthetic_query, minlength=query_count)
    query_values = sums / np.maximum(counts, 1)
    return query_values[row_query]


def statistics(values: np.ndarray) -> dict[str, float]:
    return {
        "mean": float(values.mean()),
        "std": float(values.std()),
        "minimum": float(values.min()),
        "q01": float(np.quantile(values, 0.01)),
        "q50": float(np.quantile(values, 0.50)),
        "q99": float(np.quantile(values, 0.99)),
        "maximum": float(values.max()),
    }


def main() -> None:
    if REPORT.exists():
        raise FileExistsError(f"immutable result already exists: {REPORT}")
    prereg = json.loads(PREREG.read_text(encoding="utf-8"))
    linkage_report = json.loads(LINKAGE_REPORT.read_text(encoding="utf-8"))
    if linkage_report["status"] != "source_pass":
        raise ValueError("partial linkage source is not locked/pass")
    main_frame, labels = load_main()
    raw_trackman, _, _ = load_trackman()
    maximum = int(
        prereg["integration"]["maximum_repertoire_pitches_per_context"]
    )
    folds: dict[str, Any] = {}
    artifacts: dict[str, Any] = {}
    checks: list[bool] = []
    for year in YEARS:
        history, history_meta = augmented_history(
            year, main_frame, labels, raw_trackman
        )
        valid = main_frame.loc[main_frame["season"].eq(year)].copy()
        synthetic, synthetic_query, row_query, source_level, integration_meta = (
            build_integration_rows(history, valid, maximum)
        )
        print(
            f"conditional integration {year}: history={len(history):,}, "
            f"queries={integration_meta['unique_context_queries']:,}, "
            f"synthetic={len(synthetic):,}",
            flush=True,
        )
        control_prediction = fit_predict(history, synthetic, CONTROL_FEATURES)
        full_prediction = fit_predict(history, synthetic, FULL_FEATURES)
        query_count = int(integration_meta["unique_context_queries"])
        integrated_control = integrate(
            control_prediction, synthetic_query, row_query, query_count
        )
        integrated_full = integrate(
            full_prediction, synthetic_query, row_query, query_count
        )
        physics_delta = integrated_full - integrated_control
        signals = {
            "integrated_full": integrated_full,
            "integrated_control": integrated_control,
            "physics_delta": physics_delta,
        }
        signal_stats = {
            name: statistics(values) for name, values in signals.items()
        }
        minimum_rows = int(
            prereg["semantic_gate"][
                f"minimum_augmented_history_rows_{year}"
            ]
        )
        coverage = integration_meta["validation_row_source_coverage"]
        fold_checks = {
            "history_rows": len(history) >= minimum_rows,
            "native_pitcher_coverage": coverage["pitcher"]
            >= float(
                prereg["semantic_gate"][
                    "minimum_native_pitcher_validation_coverage_each_year"
                ]
            ),
            "total_repertoire_coverage": coverage["total"]
            >= float(
                prereg["semantic_gate"][
                    "minimum_total_repertoire_coverage_each_year"
                ]
            ),
            "integrated_full_variation": signal_stats["integrated_full"]["std"]
            >= float(
                prereg["semantic_gate"][
                    "minimum_integrated_full_standard_deviation"
                ]
            ),
            "physics_delta_variation": signal_stats["physics_delta"]["std"]
            >= float(
                prereg["semantic_gate"][
                    "minimum_physics_delta_standard_deviation"
                ]
            ),
            "validation_labels_used_for_fit_or_integration": False,
            "current_validation_trackman_used": False,
            "row_independent_inference": True,
        }
        checks.extend(
            [
                fold_checks["history_rows"],
                fold_checks["native_pitcher_coverage"],
                fold_checks["total_repertoire_coverage"],
                fold_checks["integrated_full_variation"],
                fold_checks["physics_delta_variation"],
                not fold_checks["validation_labels_used_for_fit_or_integration"],
                not fold_checks["current_validation_trackman_used"],
                fold_checks["row_independent_inference"],
            ]
        )
        output = RESULTS / f"v5_conditional_physics_integration_signal_{year}.npz"
        if output.exists():
            raise FileExistsError(f"immutable artifact exists: {output}")
        np.savez_compressed(
            output,
            row_index=valid.index.to_numpy(dtype=np.int64),
            integrated_full=integrated_full,
            integrated_control=integrated_control,
            physics_delta=physics_delta,
            source_level=source_level.astype(np.int8),
        )
        matrix = np.column_stack(list(signals.values())).astype(np.float64)
        folds[str(year)] = {
            "history": history_meta,
            "integration": integration_meta,
            "signals": signal_stats,
            "signal_matrix_sha256": hashlib.sha256(matrix.tobytes()).hexdigest(),
            "checks": fold_checks,
        }
        artifacts[str(year)] = {
            "path": str(output.relative_to(ROOT)), "sha256": digest(output)
        }
        del history, valid, synthetic, control_prediction, full_prediction

    passed = bool(all(checks))
    report = {
        "experiment_id": prereg["experiment_id"],
        "status": "source_pass" if passed else "source_failed",
        "preregister_sha256": digest(PREREG),
        "implementation_sha256": digest(Path(__file__)),
        "linkage_report_sha256": digest(LINKAGE_REPORT),
        "input_sha256": {
            "train": digest(TRAIN),
            "trackman_history": digest(ROOT / "open/data/trackman_history.csv"),
            "linkage_2020": digest(
                RESULTS / "predictions"
                / "v5_partial_trackman_linkage_history_to_2020.npz"
            ),
            "linkage_2021": digest(
                RESULTS / "predictions"
                / "v5_partial_trackman_linkage_history_to_2021.npz"
            ),
        },
        "model_params": MODEL_PARAMS,
        "folds": folds,
        "semantic_gate_pass": passed,
        "artifacts": artifacts,
        "downstream_control_metrics_read": False,
        "goal_status": "active",
        "goal_completion_claimed": False,
    }
    REPORT.write_text(
        json.dumps(safe(report), ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(
        json.dumps(
            safe(
                {
                    "status": report["status"],
                    "folds": folds,
                }
            ),
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
