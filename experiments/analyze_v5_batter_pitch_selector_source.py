#!/usr/bin/env python3
"""Target-free source gate for batter TrackMan pitch-selector features."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "experiments/results"
PREREG = ROOT / "experiments/params/v5_batter_pitch_selector_preregister.json"
TRAINING = RESULTS / "v5_batter_pitch_selector_source.json"
REPORT = RESULTS / "v5_batter_pitch_selector_source_gate.json"
YEARS = (2020, 2021)
KEY = "catboost_dense_pitchtype_moe"


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
    return value


def main() -> None:
    if REPORT.exists():
        raise FileExistsError(f"immutable result already exists: {REPORT}")
    prereg = json.loads(PREREG.read_text(encoding="utf-8"))
    training = json.loads(TRAINING.read_text(encoding="utf-8"))
    folds = {
        int(item["validation_season"]): item for item in training["folds"]
    }
    minimum_ll = float(
        prereg["source_selection"]["selector_proxy_requirement"][
            "minimum_log_loss_improvement_vs_immutable_dense_selector_each_year"
        ]
    )
    minimum_accuracy = float(
        prereg["source_selection"]["selector_proxy_requirement"][
            "minimum_top1_accuracy_change_each_year"
        ]
    )
    proxy: dict[str, Any] = {}
    all_pass = True
    for year in YEARS:
        baseline_path = RESULTS / f"v5_dense_pitchtype_moe_source{year}.json"
        baseline = json.loads(baseline_path.read_text(encoding="utf-8"))["folds"][0]
        old = baseline["fit_details"][KEY]
        new = folds[year]["fit_details"][KEY]
        prefixes = list(new.get("selector_only_prefixes", []))
        selector_only = list(new.get("selector_only_features", []))
        expert_columns = new["expert_feature_columns"]
        isolation = bool(
            prefixes == prereg["locked_model"]["selector_only_prefixes"]
            and len(selector_only) == 42
            and all(
                not any(column.startswith(tuple(prefixes)) for column in columns)
                for columns in expert_columns.values()
            )
        )
        ll_improvement = float(
            old["diagnostic_selector_log_loss"]
            - new["diagnostic_selector_log_loss"]
        )
        accuracy_change = float(
            new["diagnostic_selector_top1_accuracy"]
            - old["diagnostic_selector_top1_accuracy"]
        )
        passed = bool(
            isolation
            and ll_improvement >= minimum_ll
            and accuracy_change >= minimum_accuracy
        )
        all_pass &= passed
        proxy[str(year)] = {
            "immutable_baseline_report": str(baseline_path.relative_to(ROOT)),
            "baseline_log_loss": float(old["diagnostic_selector_log_loss"]),
            "candidate_log_loss": float(new["diagnostic_selector_log_loss"]),
            "log_loss_improvement": ll_improvement,
            "required_log_loss_improvement": minimum_ll,
            "baseline_top1_accuracy": float(
                old["diagnostic_selector_top1_accuracy"]
            ),
            "candidate_top1_accuracy": float(
                new["diagnostic_selector_top1_accuracy"]
            ),
            "top1_accuracy_change": accuracy_change,
            "required_top1_accuracy_change": minimum_accuracy,
            "selector_only_feature_count": len(selector_only),
            "control_expert_feature_count": {
                name: len(columns) for name, columns in expert_columns.items()
            },
            "selector_expert_isolation_pass": isolation,
            "proxy_gate_pass": passed,
        }
    report = {
        "experiment_id": prereg["experiment_id"],
        "status": (
            "selector_proxy_pass_requires_control_analysis"
            if all_pass
            else "source_failed_selector_proxy"
        ),
        "preregister_sha256": digest(PREREG),
        "implementation_sha256": digest(ROOT / "experiments/run_v2_rolling.py"),
        "analyzer_sha256": digest(Path(__file__)),
        "training_report": str(TRAINING.relative_to(ROOT)),
        "training_report_sha256": digest(TRAINING),
        "years_read": list(YEARS),
        "years_not_read": [2022, 2023, 2024],
        "selector_proxy": proxy,
        "selector_proxy_passed": all_pass,
        "control_blend_selection_run": False,
        "reason_control_blend_selection_not_run": (
            None
            if all_pass
            else "The preregistered target-free selector proxy failed before control-target blend selection."
        ),
        "decision": (
            "run the separately locked control blend analysis"
            if all_pass
            else "close without reading 2022+ candidate labels"
        ),
        "goal_status": "active",
        "goal_completion_claimed": False,
    }
    REPORT.write_text(
        json.dumps(safe(report), ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
