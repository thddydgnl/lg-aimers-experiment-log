#!/usr/bin/env python3
"""Turn the ensemble search result into actual submission ZIPs.

Reads experiments/results/v2_ensemble.json, recovers each component's training
configuration from its own stage JSON, then builds:

  1. one standalone ZIP per component with a non-trivial weight, and
  2. one blended ZIP carrying every component at its searched weight.

Component weights are flattened: a component that is itself a Linear/HGB blend
contributes each of its models at `component_weight * internal_weight`, so the
packaged blend reproduces exactly what the search scored.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "experiments" / "results"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ensemble", type=Path, default=RESULTS / "v2_ensemble.json")
    parser.add_argument("--results-dir", type=Path, default=RESULTS)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "submission/dist")
    parser.add_argument("--prefix", default="V")
    parser.add_argument("--min-weight", type=float, default=0.05)
    parser.add_argument("--skip-components", action="store_true",
                        help="Build only the blended ZIP.")
    parser.add_argument("--logit-shifts", nargs="*", type=float, default=[0.0],
                        help="Extra blended ZIPs, one per global logit offset (plan 8.1).")
    return parser.parse_args()


def utf8_env() -> dict[str, str]:
    return {**os.environ, "PYTHONUTF8": "1", "PYTHONIOENCODING": "utf-8"}


def stage_config(results_dir: Path, stage: str) -> dict:
    path = results_dir / f"{stage}.json"
    if not path.is_file():
        raise FileNotFoundError(f"stage 결과가 없습니다: {path}")
    return json.loads(path.read_text(encoding="utf-8"))["metadata"]


def component_models(meta: dict, key: str) -> list[dict]:
    """Flatten one saved prediction key into concrete (family, internal weight) pairs."""
    models = list(meta["models"])
    blend = meta.get("blend")
    if key == "blend":
        if not blend:
            raise ValueError(f"stage {meta['stage']} has no blend weights")
        internal = [float(w) for w in blend]
    elif key in models:
        internal = [1.0 if name == key else 0.0 for name in models]
    else:
        raise ValueError(f"stage {meta['stage']} has no prediction '{key}'")
    return [
        {"family": name, "internal": weight,
         "features": list(meta["features"]), "params": meta.get("booster_params")}
        for name, weight in zip(models, internal) if weight > 0
    ]


def build(spec: dict, candidate: str, output_dir: Path) -> bool:
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False, encoding="utf-8") as handle:
        json.dump(spec, handle, ensure_ascii=False, indent=2)
        spec_path = Path(handle.name)
    try:
        print(f"\n=== {candidate} ===", flush=True)
        process = subprocess.run(
            [sys.executable, str(ROOT / "submission" / "build_v2_candidate.py"),
             "--candidate", candidate, "--spec", str(spec_path),
             "--logit-shift", str(spec.get("logit_shift", 0.0)),
             "--output-dir", str(output_dir)],
            cwd=ROOT, env=utf8_env(), text=True, encoding="utf-8", errors="replace",
        )
        return process.returncode == 0
    finally:
        spec_path.unlink(missing_ok=True)


def main() -> None:
    args = parse_args()
    if not args.ensemble.is_file():
        raise SystemExit(f"{args.ensemble} 이 없습니다. search_ensemble.py 를 먼저 실행하세요.")
    payload = json.loads(args.ensemble.read_text(encoding="utf-8"))
    weights = payload["weights"]
    if not weights:
        raise SystemExit("앙상블 가중치가 비어 있습니다.")

    print("앙상블 성분:", flush=True)
    for name, weight in weights.items():
        print(f"  {weight:5.2f}  {name}", flush=True)

    flattened: list[dict] = []
    for name, weight in weights.items():
        if weight < args.min_weight:
            continue
        stage, _, key = name.partition(":")
        meta = stage_config(args.results_dir, stage)
        for model in component_models(meta, key or "blend"):
            flattened.append({
                "family": model["family"],
                "weight": float(weight) * model["internal"],
                "features": model["features"],
                "params": model["params"],
                "source": name,
            })

    total = sum(item["weight"] for item in flattened)
    for item in flattened:
        item["weight"] /= total  # renormalise after dropping sub-threshold components

    groups = sorted({g for item in flattened for g in item["features"]},
                    key=["base", "e14", "platoon", "pitcher_te"].index)
    built, failed = [], []
    package_entries: list[dict] = []

    if not args.skip_components:
        for name, weight in weights.items():
            if weight < args.min_weight:
                continue
            stage, _, key = name.partition(":")
            meta = stage_config(args.results_dir, stage)
            models = component_models(meta, key or "blend")
            candidate = f"{args.prefix}_{stage.replace('v2_', '')}"
            spec = {"features": list(meta["features"]),
                    "models": [{"family": m["family"], "weight": m["internal"],
                                "features": m["features"], "params": m["params"]} for m in models]}
            success = build(spec, candidate, args.output_dir)
            (built if success else failed).append(candidate)
            if success:
                package_entries.append({
                    "candidate": candidate,
                    "kind": "component",
                    "source": name,
                    "weight_in_selected_ensemble": float(weight),
                    "expected_scores": payload.get("component_scores", {}).get(name, {}),
                    "family": payload.get("component_families", {}).get(name),
                })

    for shift in args.logit_shifts:
        suffix = "" if shift == 0.0 else f"_shift{shift:+g}".replace("+", "p").replace("-", "m")
        candidate = f"{args.prefix}_ensemble{suffix}"
        spec = {"features": groups, "logit_shift": float(shift),
                "models": [{"family": i["family"], "weight": i["weight"],
                            "features": i["features"], "params": i["params"]} for i in flattened]}
        success = build(spec, candidate, args.output_dir)
        (built if success else failed).append(candidate)
        if success:
            package_entries.append({
                "candidate": candidate,
                "kind": "ensemble",
                "source": str(args.ensemble),
                "logit_shift": float(shift),
                "expected_scores": ({
                    season: item.get("ensemble_score")
                    for season, item in payload.get("per_season", {}).items()
                } if shift == 0.0 else {}),
                "score_note": (
                    "selected ensemble development-fold scores"
                    if shift == 0.0 else
                    "global logit-shift sensitivity candidate; not assigned the unshifted score"
                ),
                "gate_pass": payload.get("gate", {}).get("gate_pass"),
            })

    record_path = ROOT / "submission" / "records" / "v2_package_index.json"
    record_path.parent.mkdir(parents=True, exist_ok=True)
    record_path.write_text(json.dumps({
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "ensemble": str(args.ensemble),
        "selection_is_exploratory": True,
        "candidates": package_entries,
    }, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"\n빌드 성공 {len(built)}개: {built}", flush=True)
    print(f"패키지 근거: {record_path}", flush=True)
    if failed:
        print(f"빌드 실패 {len(failed)}개: {failed}", flush=True)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
