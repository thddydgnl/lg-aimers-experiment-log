"""Target-free construction of direct historical physics-teacher signals."""

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

from eda.run_structural_eda import (  # noqa: E402
    linkage_section,
    load_trackman,
    state_code,
)
from experiments.v5_trackman_teacher_profiles import (  # noqa: E402
    build_teacher_profile_features,
    teacher_profile_table,
)

TRAIN = ROOT / "open/data/train.csv"
OOF_2019 = ROOT / "experiments/results/v5_physics_adjusted_command_2019_oof.npz"
OOT_DEV = ROOT / "experiments/results/v5_trackman_teacher_scores_dev.npz"
PREREG = ROOT / "experiments/params/v5_direct_teacher_signal_preregister.json"
RESULTS = ROOT / "experiments/results"
REPORT = RESULTS / "v5_direct_teacher_signal_source.json"
YEARS = (2020, 2021)
SIGNALS = {
    "all": "e80_teacher_full_all",
    "recent": "e80_teacher_full_recent",
    "context": "e80_teacher_full_context",
}
TARGET_FREE_COLUMNS = [
    "row_id",
    "season",
    "game_month",
    "game_dayofweek",
    "inning",
    "top_bottom",
    "game_type",
    "balls_before",
    "strikes_before",
    "outs_before",
    "run_total_before",
    "pitcher_id",
    "batter_id",
    "pitcher_hand",
    "batter_hand",
    "pitcher_team_id",
    "batter_team_id",
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


def load_target_free_train() -> pd.DataFrame:
    frame = pd.read_csv(TRAIN, usecols=TARGET_FREE_COLUMNS)
    low_team = np.minimum(
        frame["pitcher_team_id"], frame["batter_team_id"]
    ).to_numpy()
    high_team = np.maximum(
        frame["pitcher_team_id"], frame["batter_team_id"]
    ).to_numpy()
    game_key = np.stack(
        [
            frame["season"],
            frame["game_month"],
            frame["game_dayofweek"],
            low_team,
            high_team,
        ],
        axis=1,
    )
    half = frame["top_bottom"].eq("B").to_numpy(dtype=np.int64)
    progress = frame["inning"].to_numpy(dtype=np.int64) * 2 + half
    runs = frame["run_total_before"].to_numpy(dtype=np.int64)
    boundary = np.concatenate(
        [
            [True],
            np.any(game_key[1:] != game_key[:-1], axis=1)
            | (progress[1:] < progress[:-1])
            | (runs[1:] < runs[:-1]),
        ]
    )
    frame["gid"] = boundary.cumsum() - 1
    frame["half"] = half
    frame["state"] = state_code(
        frame["inning"].to_numpy(),
        half,
        frame["balls_before"].to_numpy(),
        frame["strikes_before"].to_numpy(),
        frame["outs_before"].to_numpy(),
    )
    return frame


def historical_teacher_scores(
    target_free_train: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    trackman, game_ids, _ = load_trackman()
    linkage, joined = linkage_section(
        target_free_train, trackman, len(game_ids)
    )
    linkage_columns = [
        "row_id",
        "season",
        "pitcher_id",
        "pitch_type_group",
        "balls_before",
        "strikes_before",
    ]
    exact = joined[linkage_columns].copy()
    with np.load(OOF_2019, allow_pickle=False) as archive:
        oof = pd.DataFrame(
            {
                "row_id": np.asarray(archive["row_id"]).astype(str),
                "physics_teacher": np.asarray(
                    archive["physics_teacher"], dtype=np.float64
                ),
            }
        )
    source_2019 = oof.merge(
        exact.loc[exact["season"].eq(2019)],
        on="row_id",
        how="inner",
        validate="one_to_one",
    )
    with np.load(OOT_DEV, allow_pickle=False) as archive:
        dev = pd.DataFrame(
            {
                "row_id": np.asarray(archive["row_id"]).astype(str),
                "season": np.asarray(archive["season"], dtype=np.int16),
                "pitcher_id": np.asarray(archive["pitcher_id"], dtype=np.int64),
                "pitch_type_group": np.asarray(
                    archive["pitch_type_group"]
                ).astype(str),
                "balls_before": np.asarray(
                    archive["balls_before"], dtype=np.int8
                ),
                "strikes_before": np.asarray(
                    archive["strikes_before"], dtype=np.int8
                ),
                "physics_teacher": np.asarray(
                    archive["physics_teacher"], dtype=np.float64
                ),
            }
        )
    source_2020 = dev.loc[dev["season"].eq(2020)].copy()
    columns = [*linkage_columns, "physics_teacher"]
    scores = pd.concat(
        [source_2019[columns], source_2020[columns]], ignore_index=True
    )
    if scores["row_id"].duplicated().any():
        raise ValueError("historical teacher row_id is not unique")
    scores["pitch_type_group"] = scores["pitch_type_group"].where(
        scores["pitch_type_group"].isin(
            ["fastball", "breaking", "offspeed", "other"]
        ),
        "other",
    )
    scores["teacher_full_centered"] = scores["physics_teacher"] - scores.groupby(
        "season", observed=True
    )["physics_teacher"].transform("mean")
    scores["teacher_delta_centered"] = 0.0
    metadata = {
        "exact_linkage_rows": int(len(joined)),
        "exact_linkage_state_agreement": float(
            linkage["elementwise_state_agreement"]
        ),
        "source_rows": {
            "2019": int(len(source_2019)),
            "2020": int(len(source_2020)),
        },
        "oof_2019_rows": int(len(oof)),
        "oof_2019_alignment": float(len(source_2019) / max(len(oof), 1)),
        "row_id_unique": True,
        "target_column_loaded": False,
    }
    return scores, metadata


def signal_statistics(values: np.ndarray) -> dict[str, float]:
    return {
        "mean": float(values.mean()),
        "std": float(values.std()),
        "minimum": float(values.min()),
        "q01": float(np.quantile(values, 0.01)),
        "q50": float(np.quantile(values, 0.50)),
        "q99": float(np.quantile(values, 0.99)),
        "maximum": float(values.max()),
        "nonzero_coverage": float(np.mean(np.abs(values) > 1e-12)),
    }


def main() -> None:
    if REPORT.exists():
        raise FileExistsError(f"immutable result already exists: {REPORT}")
    prereg = json.loads(PREREG.read_text(encoding="utf-8"))
    frame = load_target_free_train()
    scores, source_meta = historical_teacher_scores(frame)
    folds: dict[str, Any] = {}
    artifacts: dict[str, Any] = {}
    checks: list[bool] = [
        source_meta["source_rows"]["2019"]
        >= int(prereg["semantic_gate"]["minimum_source_rows_2019"]),
        source_meta["source_rows"]["2020"]
        >= int(prereg["semantic_gate"]["minimum_source_rows_2020"]),
        source_meta["row_id_unique"],
        not source_meta["target_column_loaded"],
    ]
    for year in YEARS:
        history_scores = scores.loc[scores["season"].lt(year)].copy()
        expected_seasons = prereg["source_cutoffs"][f"{year}_validation"]
        observed_seasons = sorted(
            int(value) for value in history_scores["season"].unique()
        )
        if observed_seasons != expected_seasons:
            raise ValueError(
                f"teacher cutoff mismatch: {year}/{observed_seasons}"
            )
        state = teacher_profile_table(history_scores)
        valid = frame.loc[frame["season"].eq(year)].copy()
        features, feature_meta = build_teacher_profile_features(
            valid, {year: state}
        )
        signal_values = {
            name: features[column].to_numpy(dtype=np.float64)
            for name, column in SIGNALS.items()
        }
        stats = {
            name: signal_statistics(values)
            for name, values in signal_values.items()
        }
        minimum_coverage = min(
            item["nonzero_coverage"] for item in stats.values()
        )
        minimum_std = min(item["std"] for item in stats.values())
        fold_checks = {
            "coverage": minimum_coverage
            >= float(
                prereg["semantic_gate"][
                    "minimum_nonzero_validation_coverage_each_year"
                ]
            ),
            "variation": minimum_std
            >= float(
                prereg["semantic_gate"]["minimum_signal_standard_deviation"]
            ),
            "cutoff": observed_seasons == expected_seasons,
            "current_validation_trackman_used": False,
            "validation_target_column_read": False,
        }
        checks.extend(
            [
                fold_checks["coverage"],
                fold_checks["variation"],
                fold_checks["cutoff"],
                not fold_checks["current_validation_trackman_used"],
                not fold_checks["validation_target_column_read"],
            ]
        )
        output = RESULTS / f"v5_direct_teacher_signal_{year}.npz"
        if output.exists():
            raise FileExistsError(f"immutable artifact exists: {output}")
        row_index = valid.index.to_numpy(dtype=np.int64)
        np.savez_compressed(
            output,
            row_index=row_index,
            signal_all=signal_values["all"],
            signal_recent=signal_values["recent"],
            signal_context=signal_values["context"],
        )
        matrix = np.column_stack(list(signal_values.values())).astype(np.float64)
        folds[str(year)] = {
            "allowed_history_seasons": observed_seasons,
            "history_teacher_rows": int(len(history_scores)),
            "validation_rows": int(len(valid)),
            "signals": stats,
            "feature_metadata": feature_meta,
            "signal_matrix_sha256": hashlib.sha256(matrix.tobytes()).hexdigest(),
            "checks": fold_checks,
        }
        artifacts[str(year)] = {
            "path": str(output.relative_to(ROOT)), "sha256": digest(output)
        }

    passed = bool(all(checks))
    report = {
        "experiment_id": prereg["experiment_id"],
        "status": "source_pass" if passed else "source_failed",
        "preregister_sha256": digest(PREREG),
        "script_sha256": digest(Path(__file__)),
        "input_sha256": {
            "train": digest(TRAIN),
            "trackman_history": digest(ROOT / "open/data/trackman_history.csv"),
            "oof_2019": digest(OOF_2019),
            "oot_dev": digest(OOT_DEV),
        },
        "historical_source": source_meta,
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
                    "historical_source": source_meta,
                    "folds": folds,
                }
            ),
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
