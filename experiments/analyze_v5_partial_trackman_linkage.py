#!/usr/bin/env python3
"""Immutable target-free source gate for partial TrackMan linkage."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.run_e20r_rolling import load_joined_and_raw_trackman
from experiments.v5_partial_trackman_linkage import (
    build_augmented_trackman_linkage,
    load_main_linkage_frame,
)


PREREG = ROOT / "experiments/params/v5_partial_trackman_linkage_preregister.json"
REPORT = ROOT / "experiments/results/v5_partial_trackman_linkage_source.json"
ARTIFACT_DIR = ROOT / "experiments/results/predictions"


def digest(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(block)
    return hasher.hexdigest()


def safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [safe(item) for item in value]
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        numeric = float(value)
        return numeric if np.isfinite(numeric) else None
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    return value


def main() -> None:
    if REPORT.exists():
        raise FileExistsError(f"immutable report already exists: {REPORT}")
    prereg = json.loads(PREREG.read_text(encoding="utf-8"))
    main_frame = load_main_linkage_frame()
    exact, raw = load_joined_and_raw_trackman()
    semantic_gate = prereg["semantic_gate"]
    folds: dict[str, Any] = {}
    checks: list[bool] = []
    artifacts: dict[str, Any] = {}
    exact_row_ids = set(exact["row_id"].astype(str))
    for validation_year, allowed in ((2020, [2019]), (2021, [2019, 2020])):
        augmented, metadata = build_augmented_trackman_linkage(
            main_frame, exact, raw, allowed
        )
        calibration = metadata["known_exact_calibration"]
        identity_purities = [
            details["minimum_purity"]
            for details in metadata["identity"].values()
            if details["minimum_purity"] is not None
        ]
        fold_checks = {
            "known_exact_precision": calibration["precision"]
            == float(semantic_gate["known_exact_game_precision_at_locked_threshold"]),
            "known_exact_sample": calibration["passing_games"]
            >= int(semantic_gate["minimum_known_exact_games_passing_each_cutoff"]),
            "identity_purity": min(identity_purities)
            >= float(semantic_gate["minimum_identity_purity"]),
            "row_expansion": metadata["joined_row_expansion_factor"]
            >= float(semantic_gate["minimum_joined_row_expansion_factor_each_cutoff"]),
            "partial_games": metadata["partial_games"]
            >= int(semantic_gate["minimum_partial_games_each_cutoff"]),
            "unique_main_rows": metadata["duplicate_main_row_ids"]
            == int(semantic_gate["duplicate_main_row_ids"]),
            "target_free_match": not metadata["control_target_used_for_matching"],
            "no_current_validation_trackman": not metadata[
                "current_validation_trackman_used"
            ],
            "no_current_pitch_payload_key": not metadata[
                "current_pitch_type_or_physics_used_as_match_key"
            ],
        }
        checks.extend(fold_checks.values())
        artifact = (
            ARTIFACT_DIR
            / f"v5_partial_trackman_linkage_history_to_{validation_year}.npz"
        )
        if artifact.exists():
            raise FileExistsError(f"immutable artifact already exists: {artifact}")
        row_ids = augmented["row_id"].astype(str).to_numpy(dtype=str)
        np.savez_compressed(
            artifact,
            row_id=row_ids,
            trackman_id=augmented["trackman_id"].to_numpy(dtype=np.int64),
            is_partial=np.asarray(
                [value not in exact_row_ids for value in row_ids], dtype=np.int8
            ),
            allowed_seasons=np.asarray(allowed, dtype=np.int16),
        )
        artifacts[str(validation_year)] = {
            "path": str(artifact.relative_to(ROOT)),
            "sha256": digest(artifact),
            "rows": int(len(augmented)),
        }
        compact_metadata = dict(metadata)
        compact_metadata["row_alignment_records_sha256"] = hashlib.sha256(
            json.dumps(
                safe(compact_metadata.pop("row_alignment_records")),
                sort_keys=True,
            ).encode("utf-8")
        ).hexdigest()
        folds[str(validation_year)] = {
            "allowed_history_seasons": allowed,
            "checks": fold_checks,
            "metadata": compact_metadata,
        }
    report = {
        "experiment_id": prereg["experiment_id"],
        "status": "source_pass" if all(checks) else "source_failed",
        "preregister_sha256": digest(PREREG),
        "implementation_sha256": digest(
            ROOT / "experiments/v5_partial_trackman_linkage.py"
        ),
        "script_sha256": digest(Path(__file__)),
        "input_sha256": {
            "train": digest(ROOT / "open/data/train.csv"),
            "trackman_history": digest(ROOT / "open/data/trackman_history.csv"),
        },
        "control_target_loaded_for_matching": False,
        "validation_or_test_trackman_used": False,
        "folds": folds,
        "artifacts": artifacts,
        "source_gate_pass": bool(all(checks)),
        "downstream_control_metrics_read": False,
        "goal_status": "active",
        "goal_completion_claimed": False,
    }
    REPORT.write_text(
        json.dumps(safe(report), ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(safe(report), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
