#!/usr/bin/env python3
"""Search fixed ensemble weights over saved fold predictions. No retraining.

v1's ensemble gained only ~1e-6 because S4-S7 were all feature variants on the
same Linear+HGB pair, so their errors were nearly collinear.  This searches over
genuinely different families (linear, hgb, lgbm, catboost, platoon variants) and
reports the paired bootstrap CI, so a blend has to earn its place.

Weights are chosen on the primary and secondary folds only.  The 2023 fold is
scored and reported but never decides, per EDA 20.2.
"""

from __future__ import annotations

import argparse
import itertools
import json
import re
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.stats import aggregate_gate, paired_bootstrap_brier_ci  # noqa: E402

RESULTS = ROOT / "experiments" / "results"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, default=RESULTS / "predictions")
    parser.add_argument("--validation-seasons", nargs="+", type=int, default=[2022, 2023, 2024])
    parser.add_argument("--baseline-stage", default="v2_base")
    parser.add_argument("--baseline-key", default="blend")
    parser.add_argument("--components", nargs="+", default=None,
                        help="stage:key entries. Default: auto-discover every saved stage.")
    parser.add_argument("--primary-season", type=int, default=2024)
    parser.add_argument("--secondary-season", type=int, default=2022)
    parser.add_argument("--step", type=float, default=0.05, help="Weight grid resolution.")
    parser.add_argument("--max-components", type=int, default=4)
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--output", type=Path, default=RESULTS / "v2_ensemble.json")
    return parser.parse_args()


def load_fold(
    directory: Path, season: int
) -> tuple[dict[str, np.ndarray], np.ndarray, np.ndarray]:
    predictions: dict[str, np.ndarray] = {}
    y: np.ndarray | None = None
    index: np.ndarray | None = None
    clusters: np.ndarray | None = None
    for path in sorted(directory.glob(f"*_{season}.npz")):
        stage = path.name[: -len(f"_{season}.npz")]
        data = np.load(path)
        if y is None:
            y = np.asarray(data["y"], dtype=np.float64)
            index = np.asarray(data["row_index"])
            clusters = np.asarray(data["cluster"] if "cluster" in data else index)
        else:
            incoming_index = np.asarray(data["row_index"])
            incoming_y = np.asarray(data["y"], dtype=np.float64)
            incoming_clusters = np.asarray(
                data["cluster"] if "cluster" in data else incoming_index
            )
            if (
                not np.array_equal(index, incoming_index)
                or not np.array_equal(y, incoming_y)
                or not np.array_equal(clusters, incoming_clusters)
            ):
                print(f"  경고: {path.name} 의 행/target/cluster 구성이 달라 건너뜁니다.", flush=True)
                continue
        for key in data.files:
            if key in ("y", "row_index", "cluster"):
                continue
            predictions[f"{stage}:{key}"] = np.asarray(data[key], dtype=np.float64)
    if y is None:
        raise FileNotFoundError(f"{directory} 에 season {season} 예측이 없습니다.")
    assert clusters is not None
    return predictions, y, clusters


def component_family(name: str, results_dir: Path) -> str | None:
    stage, _, key = name.partition(":")
    if re.search(r"_cfg\d+$", stage) or "smoke" in stage.lower():
        return None
    result_path = results_dir / f"{stage}.json"
    if not result_path.is_file():
        return None
    metadata = json.loads(result_path.read_text(encoding="utf-8")).get("metadata", {})
    if metadata.get("smoke_test"):
        return None
    models = list(metadata.get("models", []))
    if key in models and key in {"linear", "hgb", "lgbm", "catboost"}:
        return key
    return None


def auto_components(
    common: set[str],
    folds: dict[int, tuple[dict[str, np.ndarray], np.ndarray, np.ndarray]],
    primary_season: int,
    max_components: int,
    results_dir: Path,
) -> tuple[list[str], dict[str, str]]:
    """Choose at most one prediction per genuinely different model family."""
    primary_pred, primary_y, _ = folds[primary_season]
    by_family: dict[str, tuple[float, str]] = {}
    for name in sorted(common):
        family = component_family(name, results_dir)
        if family is None:
            continue
        value = score(primary_y, primary_pred[name])
        if family not in by_family or value > by_family[family][0]:
            by_family[family] = (value, name)

    ranked = sorted(by_family.items(), key=lambda item: item[1][0], reverse=True)
    selected: list[str] = []
    families: dict[str, str] = {}
    for family, (_, name) in ranked:
        # Exact/near-exact duplicate predictions add grid dimensions but no diversity.
        if any(
            all(
                np.allclose(folds[s][0][name], folds[s][0][prior], rtol=0.0, atol=1e-12)
                for s in folds
            )
            for prior in selected
        ):
            continue
        selected.append(name)
        families[name] = family
        if len(selected) >= max_components:
            break
    return selected, families


def score(y: np.ndarray, prediction: np.ndarray) -> float:
    rate = float(y.mean())
    reference = rate * (1.0 - rate)
    brier = float(np.mean(np.square(prediction - y)))
    return float(max(0.0, 100_000.0 * (1.0 - brier / reference)))


def weight_grid(count: int, step: float):
    steps = int(round(1.0 / step))
    for combo in itertools.product(range(steps + 1), repeat=count - 1):
        used = sum(combo)
        if used > steps:
            continue
        yield tuple(value / steps for value in (*combo, steps - used))


def main() -> None:
    args = parse_args()
    seasons = sorted(args.validation_seasons)
    folds = {season: load_fold(args.predictions, season) for season in seasons}

    common = set.intersection(*(set(pred) for pred, _, _ in folds.values()))
    baseline_name = f"{args.baseline_stage}:{args.baseline_key}"
    if baseline_name not in common:
        raise SystemExit(f"baseline {baseline_name} 이 모든 fold에 없습니다. 있는 것: {sorted(common)}")

    if args.components:
        components = [c for c in args.components if c in common]
        missing = [c for c in args.components if c not in common]
        if missing:
            raise SystemExit(f"다음 성분이 모든 fold에 없습니다: {missing}")
        component_families = {
            name: component_family(name, args.predictions.parent) or "explicit"
            for name in components
        }
    else:
        components, component_families = auto_components(
            common, folds, args.primary_season, args.max_components,
            args.predictions.parent,
        )
    if not components:
        raise SystemExit("앙상블에 사용할 비-smoke 모델 성분이 없습니다.")

    print(f"성분 {len(components)}개: {components}", flush=True)
    for name in components:
        line = "  ".join(
            f"{season}:{score(folds[season][1], folds[season][0][name]):7.1f}" for season in seasons
        )
        print(f"  {name:<34} {line}", flush=True)

    decisive = [s for s in (args.primary_season, args.secondary_season) if s in folds]
    best = None
    for weights in weight_grid(len(components), args.step):
        total = 0.0
        for season in decisive:
            predictions, y, _ = folds[season]
            blended = sum(w * predictions[n] for w, n in zip(weights, components))
            total += float(np.mean(np.square(blended - y)))
        mean_brier = total / len(decisive)
        if best is None or mean_brier < best[0]:
            best = (mean_brier, weights)
    assert best is not None
    _, weights = best

    allocation = {name: float(w) for name, w in zip(components, weights) if w > 0}
    print(f"\n최적 가중치 ({'/'.join(str(s) for s in decisive)} 평균 Brier 최소화):", flush=True)
    for name, weight in allocation.items():
        print(f"  {weight:5.2f}  {name}", flush=True)

    intervals: dict[int, dict] = {}
    per_season: dict[int, dict] = {}
    for season in seasons:
        predictions, y, clusters = folds[season]
        blended = sum(w * predictions[n] for n, w in allocation.items())
        baseline = predictions[baseline_name]
        interval = paired_bootstrap_brier_ci(
            y, baseline, blended, iterations=args.bootstrap, clusters=clusters
        )
        intervals[season] = interval
        per_season[season] = {
            "ensemble_score": score(y, blended),
            "baseline_score": score(y, baseline),
            "vs_baseline": interval,
        }
        print(f"  {season}: ensemble={per_season[season]['ensemble_score']:,.1f} "
              f"baseline={per_season[season]['baseline_score']:,.1f} "
              f"CI=[{interval['ci_low']:+.2e}, {interval['ci_high']:+.2e}]"
              f"{' SIG' if interval['significant'] else ''}", flush=True)

    gate = aggregate_gate(intervals, args.primary_season, args.secondary_season)
    component_scores = {
        name: {
            str(season): score(folds[season][1], folds[season][0][name])
            for season in seasons
        }
        for name in components
    }
    payload = {
        "metadata": {
            "predictions_dir": str(args.predictions),
            "baseline": baseline_name,
            "validation_seasons": seasons,
            "decisive_seasons": decisive,
            "step": args.step,
            "selection_rule": "mean Brier over decisive folds; 2023 recorded only",
            "diversity_rule": "one strongest non-smoke/non-search-trial component per model family",
            "post_selection_intervals": "exploratory; weights and intervals use the same development folds",
        },
        "components": components,
        "component_families": component_families,
        "component_scores": component_scores,
        "weights": allocation,
        "per_season": per_season,
        "gate": gate,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\ngate_pass={gate['gate_pass']}\nSaved {args.output}.", flush=True)


if __name__ == "__main__":
    main()
