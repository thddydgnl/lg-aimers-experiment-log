#!/usr/bin/env python3
"""Refit and export every model used by V4_compact_supported_1193.

The research commands are read from their immutable result metadata, then only
the outer validation season/data/output plumbing is changed.  Baseline scoring
arguments are removed because they do not affect fitting.  Completed exports
are skipped, so an interrupted run resumes model by model.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "experiments/results"
COMBINED = ROOT / "experiments/_cache/v4_full_refit_2025.csv"
WORK = ROOT / "submission/work/v4_full_refit"
EXPORT = WORK / "models"
RUN_RESULTS = WORK / "results"
RUN_PREDICTIONS = ROOT / "experiments/results/predictions"
REGISTRY = WORK / "registry.json"

REMOVE_OPTIONS = {
    "--stage",
    "--data",
    "--validation-seasons",
    "--baseline-stage",
    "--baseline-key",
    "--baseline-models",
    "--bootstrap",
    "--output-dir",
    "--save-predictions",
    "--max-history-rows",
    "--max-valid-rows",
}


def metadata(stage: str) -> dict:
    path = RESULTS / f"{stage}.json"
    return json.loads(path.read_text(encoding="utf-8"))["metadata"]


def research_tokens(stage: str) -> list[str]:
    if stage == "v3_catboost_platoon_cfg01":
        return (
            "--stage v3_catboost_platoon_cfg01 --models catboost "
            "--features base e14 platoon --validation-seasons 2024 "
            "--inner-validation regular --params "
            "experiments/params/v3_catboost_platoon_cfg01_frozen.json"
        ).split()
    tokens = metadata(stage)["command"].split()
    try:
        return tokens[tokens.index("--stage") :]
    except ValueError as error:
        raise ValueError(f"No --stage in command for {stage}") from error


def remove_options(tokens: list[str], options: set[str]) -> list[str]:
    result: list[str] = []
    index = 0
    while index < len(tokens):
        token = tokens[index]
        if token not in options:
            result.append(token)
            index += 1
            continue
        index += 1
        while index < len(tokens) and not tokens[index].startswith("--"):
            index += 1
    return result


def command_for(original_stage: str, full_stage: str, teacher: bool = False) -> list[str]:
    tokens = research_tokens(original_stage)
    removals = set(REMOVE_OPTIONS)
    if teacher:
        removals.add("--teacher-years")
    tokens = remove_options(tokens, removals)
    tokens.extend(
        [
            "--stage", full_stage,
            "--data", str(COMBINED),
            "--validation-seasons", "2025",
            "--bootstrap", "20",
            "--output-dir", str(RUN_RESULTS),
            "--save-predictions", str(RUN_PREDICTIONS),
        ]
    )
    if teacher:
        tokens.extend(["--teacher-years", "2022", "2023", "2024"])
    return [sys.executable, str(ROOT / "experiments/run_v2_rolling.py"), *tokens]


def exported_spec(full_stage: str) -> Path | None:
    matches = sorted(EXPORT.glob(f"*{full_stage}*.json"))
    if len(matches) > 1:
        raise RuntimeError(f"Ambiguous exports for {full_stage}: {matches}")
    if not matches:
        return None
    spec = json.loads(matches[0].read_text(encoding="utf-8"))
    if not (EXPORT / spec["model_file"]).is_file():
        return None
    return matches[0]


def run_one(original_stage: str, full_stage: str, teacher: bool = False) -> Path:
    existing = exported_spec(full_stage)
    if existing is not None:
        print(f"[skip] {full_stage}: {existing.name}", flush=True)
        return existing
    command = command_for(original_stage, full_stage, teacher)
    print(f"[run] {full_stage} <- {original_stage}", flush=True)
    print("      " + subprocess.list2cmdline(command), flush=True)
    environment = os.environ.copy()
    environment.update(
        {
            "V2_EXPORT_MODEL_DIR": str(EXPORT),
            "V2_BOOSTER_DEVICE": "gpu",
            "PYTHONUNBUFFERED": "1",
        }
    )
    subprocess.run(command, cwd=ROOT, env=environment, check=True)
    result = exported_spec(full_stage)
    if result is None:
        raise RuntimeError(f"No exported model after {full_stage}")
    return result


def main() -> None:
    if not COMBINED.is_file():
        raise FileNotFoundError(
            f"Run submission/prepare_v4_full_refit.py first: {COMBINED}"
        )
    EXPORT.mkdir(parents=True, exist_ok=True)
    RUN_RESULTS.mkdir(parents=True, exist_ok=True)
    report = json.loads(
        (RESULTS / "v4_compact_supported_ensemble.json").read_text(encoding="utf-8")
    )
    registry: dict = {
        "candidate": report["candidate"],
        "target_season": 2025,
        "data": str(COMBINED.relative_to(ROOT)),
        "arms": [],
    }
    for index, arm in enumerate(report["arms"], start=1):
        full_stage = f"v4_full_arm{index:02d}"
        spec = run_one(arm["stage"], full_stage)
        registry["arms"].append(
            {
                **arm,
                "full_stage": full_stage,
                "export_spec": str(spec.relative_to(WORK)),
            }
        )
        REGISTRY.write_text(json.dumps(registry, indent=2), encoding="utf-8")

    student_spec = run_one(
        "v4_teacher_residual_centered_r_primary24",
        "v4_full_student",
        teacher=True,
    )
    registry["student"] = {
        "research_stage": "v4_teacher_residual_centered_r_primary24",
        "full_stage": "v4_full_student",
        "export_spec": str(student_spec.relative_to(WORK)),
        "deployment_formula": "clip(v3_anchor + regressor_output - 0.5)",
    }
    REGISTRY.write_text(json.dumps(registry, indent=2), encoding="utf-8")
    print(f"Saved {REGISTRY}", flush=True)


if __name__ == "__main__":
    main()
