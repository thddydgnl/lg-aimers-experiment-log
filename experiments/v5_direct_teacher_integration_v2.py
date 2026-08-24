"""Hierarchical-fallback direct historical physics-teacher signals."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.v5_direct_teacher_integration import (  # noqa: E402
    OOF_2019,
    OOT_DEV,
    SIGNALS,
    TRAIN,
    digest,
    historical_teacher_scores,
    load_target_free_train,
    safe,
    signal_statistics,
)
from experiments.v5_trackman_teacher_profiles import (  # noqa: E402
    build_teacher_profile_features,
    teacher_profile_table,
)

PREREG = ROOT / "experiments/params/v5_direct_teacher_signal_v2_preregister.json"
PREDECESSOR = ROOT / "experiments/results/v5_direct_teacher_signal_source.json"
RESULTS = ROOT / "experiments/results"
REPORT = RESULTS / "v5_direct_teacher_signal_v2_source.json"
YEARS = (2020, 2021)


def _group_signal(
    frame: pd.DataFrame,
    keys: list[str],
    prior: pd.Series | float,
    k: float,
) -> pd.Series:
    grouped = frame.groupby(keys, observed=True)["teacher_full_centered"].agg(
        ["sum", "size"]
    )
    if isinstance(prior, pd.Series):
        if len(keys) == 1:
            aligned = prior.reindex(grouped.index).fillna(0.0).to_numpy(
                dtype=np.float64
            )
        elif prior.index.name in keys:
            aligned = prior.reindex(
                grouped.index.get_level_values(prior.index.name)
            ).fillna(0.0).to_numpy(dtype=np.float64)
        else:
            parent_index = grouped.index.droplevel(-1)
            aligned = prior.reindex(parent_index).fillna(0.0).to_numpy(
                dtype=np.float64
            )
    else:
        aligned = np.full(len(grouped), float(prior), dtype=np.float64)
    return pd.Series(
        (grouped["sum"].to_numpy(dtype=np.float64) + k * aligned)
        / (grouped["size"].to_numpy(dtype=np.float64) + k),
        index=grouped.index,
    )


def _lookup(
    query: pd.DataFrame,
    table: pd.Series,
    keys: list[str],
) -> np.ndarray:
    if len(keys) == 1:
        return table.reindex(query[keys[0]].to_numpy()).to_numpy(dtype=np.float64)
    index = pd.MultiIndex.from_frame(query[keys])
    index.names = keys
    return table.reindex(index).to_numpy(dtype=np.float64)


def hierarchical_fallbacks(
    history: pd.DataFrame,
    valid: pd.DataFrame,
    smoothing: dict[str, float],
) -> dict[str, np.ndarray]:
    hand = _group_signal(
        history, ["pitcher_hand"], 0.0, smoothing["hand_k"]
    )
    team_hand = _group_signal(
        history,
        ["pitcher_team_id", "pitcher_hand"],
        hand,
        smoothing["team_hand_k"],
    )
    latest = int(history["season"].max())
    recent_history = history.loc[history["season"].eq(latest)]
    recent_grouped = recent_history.groupby(
        ["pitcher_team_id", "pitcher_hand"], observed=True
    )["teacher_full_centered"].agg(["sum", "size"])
    recent_prior = team_hand.reindex(recent_grouped.index).fillna(0.0).to_numpy(
        dtype=np.float64
    )
    recent_table = pd.Series(
        (
            recent_grouped["sum"].to_numpy(dtype=np.float64)
            + smoothing["team_hand_recent_k"] * recent_prior
        )
        / (
            recent_grouped["size"].to_numpy(dtype=np.float64)
            + smoothing["team_hand_recent_k"]
        ),
        index=recent_grouped.index,
    )
    count_keys = [
        "pitcher_team_id",
        "pitcher_hand",
        "balls_before",
        "strikes_before",
    ]
    count_grouped = history.groupby(count_keys, observed=True)[
        "teacher_full_centered"
    ].agg(["sum", "size"])
    count_parent_index = count_grouped.index.droplevel([-2, -1])
    count_prior = team_hand.reindex(count_parent_index).fillna(0.0).to_numpy(
        dtype=np.float64
    )
    count_table = pd.Series(
        (
            count_grouped["sum"].to_numpy(dtype=np.float64)
            + smoothing["team_hand_count_k"] * count_prior
        )
        / (
            count_grouped["size"].to_numpy(dtype=np.float64)
            + smoothing["team_hand_count_k"]
        ),
        index=count_grouped.index,
    )
    hand_values = _lookup(valid, hand, ["pitcher_hand"])
    team_values = _lookup(
        valid, team_hand, ["pitcher_team_id", "pitcher_hand"]
    )
    team_values = np.where(np.isfinite(team_values), team_values, hand_values)
    recent_values = _lookup(
        valid, recent_table, ["pitcher_team_id", "pitcher_hand"]
    )
    recent_values = np.where(
        np.isfinite(recent_values), recent_values, team_values
    )
    context_values = _lookup(valid, count_table, count_keys)
    context_values = np.where(
        np.isfinite(context_values), context_values, team_values
    )
    return {
        "all": np.nan_to_num(team_values, nan=0.0),
        "recent": np.nan_to_num(recent_values, nan=0.0),
        "context": np.nan_to_num(context_values, nan=0.0),
    }


def main() -> None:
    if REPORT.exists():
        raise FileExistsError(f"immutable result already exists: {REPORT}")
    prereg = json.loads(PREREG.read_text(encoding="utf-8"))
    predecessor = json.loads(PREDECESSOR.read_text(encoding="utf-8"))
    if predecessor["downstream_control_metrics_read"]:
        raise ValueError("predecessor unexpectedly read downstream metrics")
    frame = load_target_free_train()
    scores, source_meta = historical_teacher_scores(frame)
    identity = frame.set_index("row_id")[["pitcher_team_id", "pitcher_hand"]]
    scores = scores.join(identity, on="row_id", validate="many_to_one")
    if scores[["pitcher_team_id", "pitcher_hand"]].isna().any().any():
        raise ValueError("teacher source is missing team/hand identity")
    smoothing = {
        key: float(value) for key, value in prereg["fixed_smoothing"].items()
    }
    folds: dict[str, Any] = {}
    artifacts: dict[str, Any] = {}
    checks: list[bool] = [
        source_meta["source_rows"]["2019"]
        >= int(prereg["semantic_gate"]["minimum_source_rows_2019"]),
        source_meta["source_rows"]["2020"]
        >= int(prereg["semantic_gate"]["minimum_source_rows_2020"]),
        not source_meta["target_column_loaded"],
    ]
    for year in YEARS:
        history = scores.loc[scores["season"].lt(year)].copy()
        expected_seasons = prereg["source_cutoffs"][f"{year}_validation"]
        observed_seasons = sorted(
            int(value) for value in history["season"].unique()
        )
        if observed_seasons != expected_seasons:
            raise ValueError(f"cutoff mismatch: {year}/{observed_seasons}")
        state = teacher_profile_table(history)
        valid = frame.loc[frame["season"].eq(year)].copy()
        native, native_meta = build_teacher_profile_features(valid, {year: state})
        native_known = native["e80_teacher_unseen"].eq(0).to_numpy()
        fallbacks = hierarchical_fallbacks(history, valid, smoothing)
        values: dict[str, np.ndarray] = {}
        for name, column in SIGNALS.items():
            native_values = native[column].to_numpy(dtype=np.float64)
            values[name] = np.where(
                native_known, native_values, fallbacks[name]
            )
        stats = {name: signal_statistics(value) for name, value in values.items()}
        nonzero_coverage = min(
            item["nonzero_coverage"] for item in stats.values()
        )
        minimum_std = min(item["std"] for item in stats.values())
        native_coverage = float(native_known.mean())
        fallback_rows = ~native_known
        fallback_nonzero = {
            name: float(np.mean(np.abs(value[fallback_rows]) > 1e-12))
            for name, value in values.items()
        }
        fold_checks = {
            "coverage": nonzero_coverage
            >= float(
                prereg["semantic_gate"][
                    "minimum_nonzero_validation_coverage_each_year"
                ]
            ),
            "variation": minimum_std
            >= float(
                prereg["semantic_gate"]["minimum_signal_standard_deviation"]
            ),
            "native_coverage": native_coverage
            >= float(
                prereg["semantic_gate"][
                    "minimum_pitcher_native_coverage_each_year"
                ]
            ),
            "cutoff": observed_seasons == expected_seasons,
            "validation_target_column_read": False,
            "current_validation_trackman_used": False,
        }
        checks.extend(
            [
                fold_checks["coverage"],
                fold_checks["variation"],
                fold_checks["native_coverage"],
                fold_checks["cutoff"],
                not fold_checks["validation_target_column_read"],
                not fold_checks["current_validation_trackman_used"],
            ]
        )
        output = RESULTS / f"v5_direct_teacher_signal_v2_{year}.npz"
        if output.exists():
            raise FileExistsError(f"immutable artifact exists: {output}")
        np.savez_compressed(
            output,
            row_index=valid.index.to_numpy(dtype=np.int64),
            signal_all=values["all"],
            signal_recent=values["recent"],
            signal_context=values["context"],
            native_pitcher=native_known.astype(np.int8),
        )
        matrix = np.column_stack(list(values.values())).astype(np.float64)
        folds[str(year)] = {
            "allowed_history_seasons": observed_seasons,
            "history_teacher_rows": int(len(history)),
            "validation_rows": int(len(valid)),
            "native_pitcher_coverage": native_coverage,
            "fallback_rows": int(fallback_rows.sum()),
            "fallback_nonzero_coverage": fallback_nonzero,
            "signals": stats,
            "native_feature_metadata": native_meta,
            "signal_matrix_sha256": hashlib.sha256(matrix.tobytes()).hexdigest(),
            "checks": fold_checks,
        }
        artifacts[str(year)] = {
            "path": str(output.relative_to(ROOT)), "sha256": digest(output)
        }

    passed = bool(all(checks))
    report: dict[str, Any] = {
        "experiment_id": prereg["experiment_id"],
        "status": "source_pass" if passed else "source_failed",
        "predecessor_sha256": digest(PREDECESSOR),
        "preregister_sha256": digest(PREREG),
        "implementation_sha256": digest(Path(__file__)),
        "input_sha256": {
            "train": digest(TRAIN),
            "trackman_history": digest(ROOT / "open/data/trackman_history.csv"),
            "oof_2019": digest(OOF_2019),
            "oot_dev": digest(OOT_DEV),
        },
        "historical_source": source_meta,
        "smoothing": smoothing,
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
