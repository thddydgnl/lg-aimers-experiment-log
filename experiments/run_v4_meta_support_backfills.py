#!/usr/bin/env python3
"""Backfill exact 2022/2023 OOF predictions for the compact CatBoost stack."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "experiments/results"
PREDICTIONS = RESULTS / "predictions"
STAGES = (
    "v4_outcome_context_publicparam_primary24",
    "v3_outcome_e14k50_batter80_middle100_dropseason",
    "v4_regime_expert_f_all",
    "v4_count_expert_0_2",
    "v3_outcome_rev_count",
    "v4_outcome_ova",
    "v4_current_state_c",
    "v3_outcome_rev_e14multi",
    "v4_recent_form",
    "v4_outcome_balance_latest_type",
    "v3_outcome_e14k50_batter80",
    "v3_outcome_batter80_middle500",
    "v4_outcome_trackman_count_k200",
    "v3_catboost_platoon_cfg01",
    "v3_outcome_trackman_w2_e14k50_batter80_middle100",
    "v3_outcome_trackman_e14k80_batter80_middle100",
)


def replace_option(tokens: list[str], option: str, values: list[str]) -> None:
    if option not in tokens:
        tokens.extend([option, *values])
        return
    start = tokens.index(option)
    end = start + 1
    while end < len(tokens) and not tokens[end].startswith("--"):
        end += 1
    tokens[start:end] = [option, *values]


def remove_option(tokens: list[str], option: str) -> None:
    if option not in tokens:
        return
    start = tokens.index(option)
    end = start + 1
    while end < len(tokens) and not tokens[end].startswith("--"):
        end += 1
    del tokens[start:end]


def arguments(stage: str) -> list[str]:
    report = json.loads((RESULTS / f"{stage}.json").read_text(encoding="utf-8"))
    command = str(report["metadata"]["command"])
    if stage == "v3_catboost_platoon_cfg01":
        tokens = [
            "experiments/run_v2_rolling.py", "--stage", stage,
            "--models", "catboost", "--features", "base", "e14", "platoon",
            "--validation-seasons", "2022", "2023", "--inner-validation", "regular",
            "--params", "experiments/params/v3_catboost_platoon_cfg01_frozen.json",
            "--bootstrap", "20",
        ]
    else:
        tokens = command.split()
    tokens[0] = str(ROOT / "experiments/run_v2_rolling.py")
    replace_option(tokens, "--stage", [f"{stage}_support2223"])
    replace_option(tokens, "--validation-seasons", ["2022", "2023"])
    replace_option(tokens, "--bootstrap", ["20"])
    remove_option(tokens, "--baseline-stage")
    remove_option(tokens, "--baseline-key")
    return tokens


def main() -> None:
    environment = dict(os.environ)
    environment["V2_BOOSTER_DEVICE"] = "gpu"
    for index, stage in enumerate(STAGES, start=1):
        output_stage = f"{stage}_support2223"
        expected = [PREDICTIONS / f"{output_stage}_{year}.npz" for year in (2022, 2023)]
        if all(path.is_file() for path in expected):
            print(f"[{index:02d}/{len(STAGES)}] skip {stage}", flush=True)
            continue
        command = [sys.executable, "-u", *arguments(stage)]
        print(f"[{index:02d}/{len(STAGES)}] run {stage}", flush=True)
        subprocess.run(command, cwd=ROOT, env=environment, check=True)
    print("Backfill complete", flush=True)


if __name__ == "__main__":
    main()
