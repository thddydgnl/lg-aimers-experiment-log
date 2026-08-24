#!/usr/bin/env python3
"""Grid-search a booster by driving run_v2_rolling once per configuration.

Shelling out costs a few seconds of CSV reload per config but reuses the tested
rolling harness verbatim, so the search cannot drift from the protocol the gate
assumes.  The winner is re-run under a stable stage name and its parameters are
written to experiments/params/<model>_best.json for downstream stages.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "experiments" / "results"
PARAM_DIR = ROOT / "experiments" / "params"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model", default="lgbm", choices=("linear", "hgb", "lgbm", "catboost")
    )
    parser.add_argument("--features", nargs="+", default=["base", "e14"])
    parser.add_argument("--validation-seasons", nargs="+", default=["2022", "2023", "2024"])
    parser.add_argument("--baseline-stage", default="v2_base")
    parser.add_argument("--baseline-key", default="blend")
    parser.add_argument("--grid", type=Path, default=PARAM_DIR / "lgbm_grid.json")
    parser.add_argument("--output", type=Path, default=RESULTS / "v2_lgbm_search.json")
    parser.add_argument("--tuned-stage", default=None,
                        help="Stage name for the winner (default: v2_<model>_tuned).")
    parser.add_argument(
        "--stage-prefix", default=None,
        help="Unique prefix for trial artifacts; defaults to v2_<model>.",
    )
    parser.add_argument(
        "--best-params", type=Path, default=None,
        help="Where to save the winning parameters without overwriting an older search.",
    )
    parser.add_argument("--primary-season", type=int, default=2024)
    parser.add_argument(
        "--inner-validation", choices=("all", "regular", "none"), default="all",
        help="Forwarded to run_v2_rolling.py for booster iteration selection.",
    )
    parser.add_argument("--max-history-rows", type=int, default=None)
    parser.add_argument("--max-valid-rows", type=int, default=None)
    parser.add_argument("--k-pitcher", type=float, default=200.0)
    parser.add_argument("--k-platoon", type=float, default=200.0)
    parser.add_argument("--drop-features", nargs="+", default=None)
    return parser.parse_args()


def utf8_env() -> dict[str, str]:
    return {**os.environ, "PYTHONUTF8": "1", "PYTHONIOENCODING": "utf-8"}


def run_rolling(stage: str, model: str, features: list[str], seasons: list[str],
                params_path: Path | None, baseline_stage: str, baseline_key: str,
                args: argparse.Namespace) -> dict:
    argv = [
        sys.executable, str(ROOT / "experiments" / "run_v2_rolling.py"),
        "--stage", stage,
        "--models", model,
        "--features", *features,
        "--validation-seasons", *seasons,
        "--baseline-stage", baseline_stage,
        "--baseline-key", baseline_key,
        "--inner-validation", args.inner_validation,
        "--output-dir", str(RESULTS),
        "--save-predictions", str(RESULTS / "predictions"),
        "--k-pitcher", str(args.k_pitcher),
        "--k-platoon", str(args.k_platoon),
    ]
    if params_path is not None:
        argv += ["--params", str(params_path)]
    if args.max_history_rows:
        argv += ["--max-history-rows", str(args.max_history_rows)]
    if args.max_valid_rows:
        argv += ["--max-valid-rows", str(args.max_valid_rows)]
    if args.drop_features:
        argv += ["--drop-features", *args.drop_features]
    process = subprocess.run(argv, cwd=ROOT, env=utf8_env(), text=True,
                             encoding="utf-8", errors="replace")
    if process.returncode != 0:
        raise RuntimeError(f"rolling stage {stage} failed with exit {process.returncode}")
    return json.loads((RESULTS / f"{stage}.json").read_text(encoding="utf-8"))


def main() -> None:
    args = parse_args()
    stage_prefix = args.stage_prefix or f"v2_{args.model}"
    tuned_stage = args.tuned_stage or f"v2_{args.model}_tuned"
    grid = json.loads(args.grid.read_text(encoding="utf-8"))
    if not isinstance(grid, list) or not grid:
        raise SystemExit(f"{args.grid} 는 비어 있지 않은 리스트여야 합니다.")

    PARAM_DIR.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    trials = []
    for index, params in enumerate(grid):
        stage = f"{stage_prefix}_cfg{index:02d}"
        params_path = PARAM_DIR / f"_search_{stage_prefix}_{index:02d}.json"
        params_path.write_text(json.dumps(params, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\n=== [{index + 1}/{len(grid)}] {stage}  {params} ===", flush=True)
        payload = run_rolling(stage, args.model, args.features, args.validation_seasons,
                              params_path, args.baseline_stage, args.baseline_key, args)
        gate = payload["gates"][args.model]
        iterations = [
            fold.get("fit_details", {}).get(args.model, {}).get("n_iter")
            for fold in payload.get("folds", [])
        ]
        iterations = [int(value) for value in iterations if value is not None]
        primary_fold = next(
            fold for fold in payload.get("folds", [])
            if int(fold["validation_season"]) == int(args.primary_season)
        )
        primary_prediction_std = primary_fold.get("fit_details", {}).get(
            args.model, {}
        ).get("prediction_std")
        trials.append({
            "index": index,
            "stage": stage,
            "params": params,
            "primary_season": gate["primary_season"],
            "primary_score": gate["primary_score"],
            "mean_score": gate["mean_score"],
            "primary_point": gate["primary_point"],
            "primary_ci": gate["primary_ci"],
            "gate_pass": gate["gate_pass"],
            "selected_iterations": iterations,
            "primary_prediction_std": primary_prediction_std,
        })
        print(f"  -> primary({gate['primary_season']})={gate['primary_score']:,.1f} "
              f"mean={gate['mean_score']:,.1f} std={primary_prediction_std:.5f} "
              f"pass={gate['gate_pass']}", flush=True)

    # Selection: primary fold score. The 2023 fold is the one-off futures-league
    # regime break (EDA 20.2) and never decides a candidate.
    best = max(trials, key=lambda row: row["primary_score"])
    final_params = dict(best["params"])
    # With chronological inner validation, the configured maximum is part of
    # the stochastic selection/refit recipe (especially for GPU CatBoost).
    # Replacing it by the observed best iteration and then early-stopping a
    # second time can produce a different refit.  Collapse to a fixed count
    # only when no inner validation is requested.
    if (
        best["selected_iterations"]
        and args.model in ("lgbm", "catboost")
        and args.inner_validation == "none"
    ):
        iteration_key = "n_estimators" if args.model == "lgbm" else "iterations"
        final_params[iteration_key] = int(np.median(best["selected_iterations"]))
    best_path = args.best_params or (PARAM_DIR / f"{stage_prefix}_best.json")
    best_path.write_text(json.dumps(final_params, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n최적 설정 (primary fold 기준): {best['stage']}  {final_params}", flush=True)
    print(f"  primary={best['primary_score']:,.1f}  mean={best['mean_score']:,.1f}", flush=True)

    print(f"\n=== 최적 설정을 {tuned_stage} 로 재실행 ===", flush=True)
    tuned = run_rolling(tuned_stage, args.model, args.features, args.validation_seasons,
                        best_path, args.baseline_stage, args.baseline_key, args)

    payload = {
        "metadata": {
            "model": args.model,
            "features": args.features,
            "grid": str(args.grid),
            "baseline_stage": args.baseline_stage,
            "tuned_stage": tuned_stage,
            "stage_prefix": stage_prefix,
            "inner_validation": args.inner_validation,
            "k_pitcher": args.k_pitcher,
            "k_platoon": args.k_platoon,
            "drop_features": args.drop_features or [],
            "selection_rule": "primary fold competition score (2023 recorded, never decisive)",
            "selection_inference": (
                "exploratory: the primary development fold selects the configuration; "
                "its CI is not an independent confirmation"
            ),
            "elapsed_seconds": time.perf_counter() - started,
        },
        "best": {**best, "packaging_params": final_params},
        "best_params_path": str(best_path),
        "tuned_gate": tuned["gates"][args.model],
        "trials": trials,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    for leftover in PARAM_DIR.glob(f"_search_{stage_prefix}_*.json"):
        leftover.unlink()
    print(f"Saved {args.output} and {best_path}.", flush=True)


if __name__ == "__main__":
    main()
