"""Marginalize a full-context privileged teacher over historical pitch physics."""

from __future__ import annotations

import gc
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
    MODEL_PARAMS,
    PHYSICS,
    TARGET,
)
from experiments.v5_conditional_physics_integration import (  # noqa: E402
    RESULTS,
    TRAIN,
    augmented_history,
    deterministic_sample,
    digest,
    safe,
    statistics,
)

PREREG = (
    ROOT
    / "experiments/params/v5_contextual_privileged_marginalization_signal_preregister.json"
)
LINKAGE_REPORT = RESULTS / "v5_partial_trackman_linkage_source.json"
REPORT = RESULTS / "v5_contextual_privileged_marginalization_signal_source.json"
YEARS = (2020, 2021)
PITCH_CATEGORICAL = ("pitch_type_group", "tagged_pitch_type", "auto_pitch_type")
MAIN_CATEGORICAL = (
    "top_bottom",
    "game_type",
    "base_state",
    "pitcher_id",
    "batter_id",
    "pitcher_hand",
    "batter_hand",
    "pitcher_team_id",
    "batter_team_id",
)
REPERTOIRE_COLUMNS = [*PITCH_CATEGORICAL, *PHYSICS]


def feature_contract() -> tuple[list[str], list[str], list[str], list[str]]:
    columns = pd.read_csv(TRAIN, nrows=0).columns.tolist()
    main_features = [
        column for column in columns if column not in ("row_id", TARGET)
    ]
    if len(main_features) != 47:
        raise ValueError(f"expected 47 official main features, got {len(main_features)}")
    control_features = [*main_features, *PITCH_CATEGORICAL]
    full_features = [*control_features, *PHYSICS]
    categorical = [
        column
        for column in full_features
        if column in (*MAIN_CATEGORICAL, *PITCH_CATEGORICAL)
    ]
    return main_features, control_features, full_features, categorical


def load_main(main_features: list[str]) -> tuple[pd.DataFrame, pd.Series]:
    frame = pd.read_csv(
        TRAIN,
        usecols=["row_id", *main_features],
        dtype={"row_id": "string"},
    )
    labels = pd.read_csv(TRAIN, usecols=[TARGET])[TARGET].astype(np.int8)
    return frame, labels


def build_category_maps(
    history: pd.DataFrame,
    valid: pd.DataFrame,
    categorical: list[str],
) -> dict[str, dict[str, int]]:
    mappings: dict[str, dict[str, int]] = {}
    for column in categorical:
        parts = [history[column].astype("string")]
        if column in valid.columns:
            parts.append(valid[column].astype("string"))
        values = pd.concat(parts, ignore_index=True).fillna("__MISSING__")
        categories = sorted(values.unique().tolist())
        mappings[column] = {value: index for index, value in enumerate(categories)}
    return mappings


def encode_frame(
    frame: pd.DataFrame,
    features: list[str],
    mappings: dict[str, dict[str, int]],
) -> pd.DataFrame:
    output = pd.DataFrame(index=frame.index)
    for column in features:
        if column in mappings:
            mapping = mappings[column]
            values = (
                frame[column]
                .astype("string")
                .fillna("__MISSING__")
                .map(mapping)
                .fillna(mapping.get("__MISSING__", 0))
                .to_numpy(dtype=np.int32)
            )
            output[column] = pd.Categorical(
                values, categories=np.arange(len(mapping), dtype=np.int32)
            )
        else:
            output[column] = pd.to_numeric(
                frame[column], errors="coerce"
            ).astype(np.float32)
    return output


def fit_teachers(
    history: pd.DataFrame,
    valid: pd.DataFrame,
    control_features: list[str],
    full_features: list[str],
    categorical: list[str],
) -> tuple[LGBMClassifier, LGBMClassifier, dict[str, dict[str, int]]]:
    mappings = build_category_maps(history, valid, categorical)
    train_x = encode_frame(history, full_features, mappings)
    target = np.ascontiguousarray(history[TARGET].to_numpy(dtype=np.float32))
    control_model = LGBMClassifier(**MODEL_PARAMS)
    control_model.fit(
        train_x[control_features],
        target,
        categorical_feature=[
            column for column in control_features if column in categorical
        ],
    )
    full_model = LGBMClassifier(**MODEL_PARAMS)
    full_model.fit(
        train_x[full_features],
        target,
        categorical_feature=categorical,
    )
    del train_x, target
    gc.collect()
    return control_model, full_model, mappings


def sample_tables(
    history: pd.DataFrame,
    maximum: int,
) -> tuple[
    dict[Any, np.ndarray],
    dict[tuple[Any, Any], np.ndarray],
    dict[Any, np.ndarray],
]:
    pitcher = {
        key: deterministic_sample(np.asarray(value, dtype=np.int64), maximum)
        for key, value in history.groupby("pitcher_id", sort=False).indices.items()
    }
    team_hand = {
        key: deterministic_sample(np.asarray(value, dtype=np.int64), maximum)
        for key, value in history.groupby(
            ["pitcher_team_id", "pitcher_hand"], sort=False
        ).indices.items()
    }
    hand = {
        key: deterministic_sample(np.asarray(value, dtype=np.int64), maximum)
        for key, value in history.groupby("pitcher_hand", sort=False).indices.items()
    }
    return pitcher, team_hand, hand


def marginalize(
    history: pd.DataFrame,
    valid: pd.DataFrame,
    main_features: list[str],
    control_features: list[str],
    full_features: list[str],
    categorical: list[str],
    control_model: LGBMClassifier,
    full_model: LGBMClassifier,
    mappings: dict[str, dict[str, int]],
    maximum: int,
    batch_rows: int,
) -> tuple[dict[str, np.ndarray], np.ndarray, dict[str, Any]]:
    pitcher_samples, team_hand_samples, hand_samples = sample_tables(
        history, maximum
    )
    integrated_control = np.empty(len(valid), dtype=np.float64)
    integrated_full = np.empty(len(valid), dtype=np.float64)
    source_level = np.empty(len(valid), dtype=np.int8)
    total_synthetic = 0
    sample_counts: list[int] = []
    valid_reset = valid.reset_index(drop=True)
    for start in range(0, len(valid_reset), batch_rows):
        stop = min(start + batch_rows, len(valid_reset))
        batch = valid_reset.iloc[start:stop]
        selected_parts: list[np.ndarray] = []
        levels = np.empty(len(batch), dtype=np.int8)
        counts = np.empty(len(batch), dtype=np.int16)
        for local_index, row in enumerate(batch.itertuples(index=False)):
            if row.pitcher_id in pitcher_samples:
                selected = pitcher_samples[row.pitcher_id]
                level = 0
            elif (row.pitcher_team_id, row.pitcher_hand) in team_hand_samples:
                selected = team_hand_samples[(row.pitcher_team_id, row.pitcher_hand)]
                level = 1
            elif row.pitcher_hand in hand_samples:
                selected = hand_samples[row.pitcher_hand]
                level = 2
            else:
                raise ValueError(f"no repertoire fallback at row {start + local_index}")
            selected_parts.append(selected)
            levels[local_index] = level
            counts[local_index] = len(selected)
        selected_rows = np.concatenate(selected_parts)
        repeats = np.repeat(np.arange(len(batch), dtype=np.int64), counts)
        current = batch.iloc[repeats][main_features].reset_index(drop=True)
        repertoire = history.iloc[selected_rows][REPERTOIRE_COLUMNS].reset_index(
            drop=True
        )
        synthetic = pd.concat([current, repertoire], axis=1)
        encoded = encode_frame(synthetic, full_features, mappings)
        control_prediction = control_model.predict_proba(
            encoded[control_features]
        )[:, 1].astype(np.float64)
        full_prediction = full_model.predict_proba(
            encoded[full_features]
        )[:, 1].astype(np.float64)
        offsets = np.concatenate(
            [np.asarray([0], dtype=np.int64), np.cumsum(counts[:-1], dtype=np.int64)]
        )
        integrated_control[start:stop] = np.add.reduceat(
            control_prediction, offsets
        ) / counts
        integrated_full[start:stop] = np.add.reduceat(
            full_prediction, offsets
        ) / counts
        source_level[start:stop] = levels
        total_synthetic += len(synthetic)
        sample_counts.extend(int(value) for value in counts)
        del current, repertoire, synthetic, encoded, control_prediction, full_prediction
        gc.collect()
        if start == 0 or stop == len(valid_reset) or stop % 40000 == 0:
            print(
                f"  marginalized {stop:,}/{len(valid_reset):,} rows",
                flush=True,
            )
    signals = {
        "integrated_full": integrated_full,
        "integrated_control": integrated_control,
        "physics_delta": integrated_full - integrated_control,
    }
    metadata = {
        "synthetic_rows": int(total_synthetic),
        "samples_per_row_min": int(min(sample_counts)),
        "samples_per_row_mean": float(np.mean(sample_counts)),
        "samples_per_row_max": int(max(sample_counts)),
        "validation_row_source_coverage": {
            "pitcher": float(np.mean(source_level == 0)),
            "team_hand": float(np.mean(source_level == 1)),
            "hand": float(np.mean(source_level == 2)),
            "total": 1.0,
        },
    }
    return signals, source_level, metadata


def main() -> None:
    if REPORT.exists():
        raise FileExistsError(f"immutable result already exists: {REPORT}")
    prereg = json.loads(PREREG.read_text(encoding="utf-8"))
    linkage_report = json.loads(LINKAGE_REPORT.read_text(encoding="utf-8"))
    if linkage_report["status"] != "source_pass":
        raise ValueError("partial linkage evidence did not pass")
    main_features, control_features, full_features, categorical = feature_contract()
    main_frame, labels = load_main(main_features)
    raw_trackman, _, _ = load_trackman()
    maximum = int(prereg["marginalization"]["repertoire_sample_size"])
    batch_rows = int(prereg["marginalization"]["batch_rows"])
    folds: dict[str, Any] = {}
    artifacts: dict[str, Any] = {}
    checks: list[bool] = []
    for year in YEARS:
        history, history_meta = augmented_history(
            year, main_frame, labels, raw_trackman
        )
        valid = main_frame.loc[main_frame["season"].eq(year)].copy()
        print(
            f"contextual privileged marginalization {year}: "
            f"history={len(history):,}, valid={len(valid):,}",
            flush=True,
        )
        control_model, full_model, mappings = fit_teachers(
            history,
            valid,
            control_features,
            full_features,
            categorical,
        )
        signals, source_level, marginal_meta = marginalize(
            history,
            valid,
            main_features,
            control_features,
            full_features,
            categorical,
            control_model,
            full_model,
            mappings,
            maximum,
            batch_rows,
        )
        signal_stats = {
            name: statistics(values) for name, values in signals.items()
        }
        coverage = marginal_meta["validation_row_source_coverage"]
        minimum_rows = int(
            prereg["semantic_gate"][
                f"minimum_augmented_history_rows_{year}"
            ]
        )
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
            "validation_labels_used_for_fit_or_marginalization": False,
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
                not fold_checks["validation_labels_used_for_fit_or_marginalization"],
                not fold_checks["current_validation_trackman_used"],
                fold_checks["row_independent_inference"],
            ]
        )
        output = RESULTS / f"v5_contextual_privileged_marginalization_signal_{year}.npz"
        if output.exists():
            raise FileExistsError(f"immutable artifact exists: {output}")
        np.savez_compressed(
            output,
            row_index=valid.index.to_numpy(dtype=np.int64),
            integrated_full=signals["integrated_full"],
            integrated_control=signals["integrated_control"],
            physics_delta=signals["physics_delta"],
            source_level=source_level,
        )
        matrix = np.column_stack(list(signals.values())).astype(np.float64)
        folds[str(year)] = {
            "history": history_meta,
            "marginalization": marginal_meta,
            "signals": signal_stats,
            "feature_contract": {
                "main_feature_count": len(main_features),
                "control_feature_count": len(control_features),
                "full_feature_count": len(full_features),
                "categorical_feature_count": len(categorical),
                "main_features": main_features,
            },
            "category_cardinality": {
                column: len(mapping) for column, mapping in mappings.items()
            },
            "signal_matrix_sha256": hashlib.sha256(matrix.tobytes()).hexdigest(),
            "checks": fold_checks,
        }
        artifacts[str(year)] = {
            "path": str(output.relative_to(ROOT)), "sha256": digest(output)
        }
        del history, valid, control_model, full_model, mappings, signals
        gc.collect()

    passed = bool(all(checks))
    report: dict[str, Any] = {
        "experiment_id": prereg["experiment_id"],
        "status": "source_pass" if passed else "source_failed",
        "preregister_sha256": digest(PREREG),
        "implementation_sha256": digest(Path(__file__)),
        "linkage_report_sha256": digest(LINKAGE_REPORT),
        "input_sha256": {
            "train": digest(TRAIN),
            "trackman_history": digest(ROOT / "open/data/trackman_history.csv"),
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
