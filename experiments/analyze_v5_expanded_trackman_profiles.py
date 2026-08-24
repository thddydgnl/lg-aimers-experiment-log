#!/usr/bin/env python3
"""Immutable target-free gate for full-log TrackMan profile sources."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.run_e20r_rolling import load_joined_and_raw_trackman  # noqa: E402
from experiments.v5_expanded_trackman_profiles import (  # noqa: E402
    build_expanded_trackman_profile_source,
)

PREREG = ROOT / "experiments/params/v5_expanded_trackman_profiles_preregister.json"
REPORT = ROOT / "experiments/results/v5_expanded_trackman_profiles_source.json"


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def main() -> None:
    if REPORT.exists():
        raise FileExistsError(f"immutable report already exists: {REPORT}")
    prereg = json.loads(PREREG.read_text(encoding="utf-8"))
    exact, raw = load_joined_and_raw_trackman()
    gate = prereg["semantic_gate"]
    folds = {}
    checks = []
    for validation_year, allowed in ((2020, [2019]), (2021, [2019, 2020])):
        source, metadata = build_expanded_trackman_profile_source(
            exact, raw, allowed
        )
        purities = [
            value["minimum_purity"] for value in metadata["identity"].values()
        ]
        fold_checks = {
            "row_expansion": metadata["row_expansion_factor"]
            >= float(gate["minimum_row_expansion_factor_each_cutoff"]),
            "mapped_pitchers": metadata["mapped_pitchers"]
            >= int(gate["minimum_mapped_pitchers_each_cutoff"]),
            "identity_purity": min(purities)
            >= float(gate["minimum_identity_purity"]),
            "major_teams": metadata["major_team_code_count"]
            >= int(gate["minimum_major_team_codes"]),
            "batter_hand_complete": metadata["unmapped_batter_hand_rows"]
            == int(gate["unmapped_batter_hand_rows"]),
            "target_free": not metadata["target_columns_read"],
            "no_validation_trackman": not metadata[
                "current_validation_trackman_used"
            ],
            "aggregation_only": metadata["profile_aggregation_only"]
            and not metadata["unmatched_game_claimed_as_main_row"],
        }
        checks.extend(fold_checks.values())
        folds[str(validation_year)] = {
            "checks": fold_checks,
            "metadata": metadata,
            "profile_row_fingerprint": hashlib.sha256(
                source[[
                    "trackman_id", "pitcher_id", "season",
                    "pitcher_trackman_id",
                ]].to_csv(index=False).encode("utf-8")
            ).hexdigest(),
        }
    report = {
        "experiment_id": prereg["experiment_id"],
        "status": "source_pass" if all(checks) else "source_failed",
        "preregister_sha256": digest(PREREG),
        "implementation_sha256": digest(
            ROOT / "experiments/v5_expanded_trackman_profiles.py"
        ),
        "script_sha256": digest(Path(__file__)),
        "input_sha256": {
            "train": digest(ROOT / "open/data/train.csv"),
            "trackman_history": digest(ROOT / "open/data/trackman_history.csv"),
        },
        "folds": folds,
        "source_gate_pass": bool(all(checks)),
        "downstream_control_metrics_read": False,
        "goal_status": "active",
        "goal_completion_claimed": False,
    }
    REPORT.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

