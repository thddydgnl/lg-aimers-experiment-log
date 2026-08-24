#!/usr/bin/env python3
"""Resume the immutable source route search preregistered for pitch-gated TabM."""

from __future__ import annotations

from itertools import combinations, product
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.analyze_v5_dense_pitchtype_moe import (  # noqa: E402
    digest,
    load,
    safe,
    score,
)
from experiments.analyze_v5_direct_season_update_dev import (  # noqa: E402
    current_state,
    states_before,
)
from experiments.run_v5_h1_residual import (  # noqa: E402
    cluster_bootstrap_score_gain,
)


TRAIN = ROOT / "open/data/train.csv"
PRED = ROOT / "experiments/results/predictions"
PREREG = ROOT / "experiments/params/v5_pitch_gated_stable_route_preregister.json"
REPORT = ROOT / "experiments/results/v5_pitch_gated_stable_route_source.json"
LOCK = ROOT / "experiments/params/v5_pitch_gated_stable_route_source_lock.json"
YEARS = (2020, 2021)


def load_rows() -> pd.DataFrame:
    columns = [
        "season", "game_type", "pitcher_id", "pitcher_hand", "batter_hand",
        "balls_before", "strikes_before", "asof_pitcher_n",
        "asof_pitcher_success_rate", "control_success",
    ]
    parts: list[pd.DataFrame] = []
    offset = 0
    for chunk in pd.read_csv(TRAIN, usecols=columns, chunksize=250_000):
        chunk.index = np.arange(offset, offset + len(chunk), dtype=np.int64)
        offset += len(chunk)
        selected = chunk.loc[chunk["season"].le(max(YEARS))]
        if len(selected):
            parts.append(selected)
        if int(chunk["season"].min()) > max(YEARS):
            break
    frame = pd.concat(parts, axis=0)
    if int(frame["season"].max()) != max(YEARS):
        raise AssertionError("source loader crossed or missed 2021")
    return frame


def atom_masks(rows: pd.DataFrame, n_current: np.ndarray) -> dict[str, dict[str, np.ndarray]]:
    balls = rows["balls_before"].to_numpy(dtype=np.int8)
    strikes = rows["strikes_before"].to_numpy(dtype=np.int8)
    pitcher_hand = rows["pitcher_hand"].to_numpy(dtype=np.int8)
    batter_hand = rows["batter_hand"].to_numpy(dtype=np.int8)
    full = (balls == 3) & (strikes == 2)
    three_ball = (balls == 3) & (strikes < 2)
    two_strike = (balls < 3) & (strikes == 2)
    normal = ~(full | three_ball | two_strike)
    return {
        "pressure": {
            "normal": normal,
            "two_strike": two_strike,
            "three_ball": three_ball,
            "full": full,
        },
        "count_shape": {
            "balls_gt_strikes": balls > strikes,
            "balls_eq_strikes": balls == strikes,
            "balls_lt_strikes": balls < strikes,
        },
        "hand": {
            "same_hand": pitcher_hand == batter_hand,
            "opposite_hand": pitcher_hand != batter_hand,
            "batter_left": batter_hand == 2,
            "batter_right": batter_hand == 1,
        },
        "matchup": {
            "same_hand": pitcher_hand == batter_hand,
            "opposite_hand": pitcher_hand != batter_hand,
        },
        "batter_side": {
            "batter_left": batter_hand == 2,
            "batter_right": batter_hand == 1,
        },
        "season_evidence": {
            "n_eq_0": n_current == 0,
            "n_1_49": (n_current >= 1) & (n_current < 50),
            "n_50_199": (n_current >= 50) & (n_current < 200),
            "n_200_499": (n_current >= 200) & (n_current < 500),
            "n_ge_500": n_current >= 500,
        },
    }


def route_specs() -> list[dict[str, Any]]:
    family_names = {
        "pressure": ["normal", "two_strike", "three_ball", "full"],
        "count_shape": ["balls_gt_strikes", "balls_eq_strikes", "balls_lt_strikes"],
        "hand": ["same_hand", "opposite_hand", "batter_left", "batter_right"],
        "season_evidence": ["n_eq_0", "n_1_49", "n_50_199", "n_200_499", "n_ge_500"],
    }
    specs: dict[str, dict[str, Any]] = {}
    for family, names in family_names.items():
        for name in names:
            route_id = f"atom:{family}:{name}"
            specs[route_id] = {
                "route_id": route_id, "kind": "atom", "leaves": 1,
                "terms": [[family, name]],
            }
    context_atoms = [
        (family, name)
        for family in ("pressure", "count_shape")
        for name in family_names[family]
    ]
    for (family, name), hand_name in product(context_atoms, family_names["hand"]):
        route_id = f"and:{family}:{name}&hand:{hand_name}"
        specs[route_id] = {
            "route_id": route_id, "kind": "conjunction", "leaves": 1,
            "terms": [[family, name], ["hand", hand_name]],
        }
    for (family, name), n_name in product(context_atoms, family_names["season_evidence"]):
        route_id = f"and:{family}:{name}&season_evidence:{n_name}"
        specs[route_id] = {
            "route_id": route_id, "kind": "conjunction", "leaves": 1,
            "terms": [[family, name], ["season_evidence", n_name]],
        }
    mutually_exclusive = {
        "pressure": family_names["pressure"],
        "count_shape": family_names["count_shape"],
        "season_evidence": family_names["season_evidence"],
        "matchup": ["same_hand", "opposite_hand"],
        "batter_side": ["batter_left", "batter_right"],
    }
    for family, names in mutually_exclusive.items():
        for size in (2, 3):
            for selected in combinations(names, size):
                route_id = f"union:{family}:" + "+".join(selected)
                specs[route_id] = {
                    "route_id": route_id, "kind": "union", "leaves": size,
                    "terms": [[family, name] for name in selected],
                }
    return [specs[key] for key in sorted(specs)]


def apply_spec(spec: dict[str, Any], atoms: dict[str, dict[str, np.ndarray]]) -> np.ndarray:
    terms = [atoms[family][name] for family, name in spec["terms"]]
    if spec["kind"] == "conjunction":
        result = terms[0].copy()
        for term in terms[1:]:
            result &= term
        return result
    result = np.zeros(len(terms[0]), dtype=bool)
    for term in terms:
        result |= term
    return result


class FastClusterBootstrap:
    def __init__(
        self, y: np.ndarray, parent: np.ndarray, cluster: np.ndarray,
        regular: np.ndarray, iterations: int, seed: int,
    ) -> None:
        self.mask = regular
        local_cluster = cluster[regular].astype(str)
        codes, uniques = pd.factorize(local_cluster, sort=False)
        self.codes = codes.astype(np.int32)
        self.cluster_count = len(uniques)
        yy = y[regular].astype(np.float64)
        self.y = yy
        self.parent = parent[regular]
        self.n_group = np.bincount(self.codes, minlength=self.cluster_count).astype(np.float64)
        self.y_group = np.bincount(
            self.codes, weights=yy, minlength=self.cluster_count
        ).astype(np.float64)
        rng = np.random.default_rng(seed)
        self.draw = rng.integers(
            0, self.cluster_count,
            size=(iterations, self.cluster_count), dtype=np.int32,
        )
        self.sample_n = self.n_group[self.draw].sum(axis=1)
        self.sample_y = self.y_group[self.draw].sum(axis=1)
        rate = self.sample_y / self.sample_n
        self.reference = np.maximum(rate * (1.0 - rate), 1e-12)
        self.iterations = iterations

    def interval(self, candidate: np.ndarray) -> dict[str, float]:
        cand = candidate[self.mask]
        advantage = np.square(self.parent - self.y) - np.square(cand - self.y)
        group_advantage = np.bincount(
            self.codes, weights=advantage, minlength=self.cluster_count
        ).astype(np.float64)
        sampled_advantage = group_advantage[self.draw].sum(axis=1)
        gains = 100000.0 * (sampled_advantage / self.sample_n) / self.reference
        point = 100000.0 * advantage.mean() / max(
            float(self.y.mean() * (1.0 - self.y.mean())), 1e-12
        )
        return {
            "point": float(point),
            "ci_low": float(np.quantile(gains, 0.025)),
            "ci_high": float(np.quantile(gains, 0.975)),
            "bootstrap_std": float(gains.std(ddof=1)),
            "iterations": int(self.iterations),
            "cluster_count": int(self.cluster_count),
        }


def main() -> None:
    if REPORT.exists() or LOCK.exists():
        raise FileExistsError("immutable source report or lock already exists")
    prereg = json.loads(PREREG.read_text(encoding="utf-8"))
    frame = load_rows()
    before = states_before(frame)
    gamma = float(prereg["immutable_inputs"]["inner_gamma"])
    iterations = int(prereg["selection"]["bootstrap_replicates"])
    seed = int(prereg["selection"]["bootstrap_seed"])

    folds: dict[int, dict[str, Any]] = {}
    for year in YEARS:
        parent_path = PRED / f"v4_m3_c_backtest_{year}_{year}.npz"
        challenger_path = PRED / f"v5_tabm_pitch_gated_source{year}_{year}.npz"
        parent_artifact = load(parent_path)
        challenger_artifact = load(challenger_path)
        for key in ("y", "row_index", "cluster"):
            if not np.array_equal(parent_artifact[key], challenger_artifact[key]):
                raise AssertionError(f"parent/challenger mismatch {year}: {key}")
        rows = frame.loc[parent_artifact["row_index"].astype(np.int64)]
        if not rows["season"].eq(year).all():
            raise AssertionError(f"season mismatch: {year}")
        if not np.array_equal(
            rows["control_success"].to_numpy(dtype=np.int8),
            parent_artifact["y"].astype(np.int8),
        ):
            raise AssertionError(f"target mismatch: {year}")
        n_current, _, invalid = current_state(rows, year, before)
        if invalid.any():
            raise AssertionError(f"unexpected invalid current state: {year}")
        regular = rows["game_type"].astype(str).eq("R").to_numpy()
        parent = parent_artifact["catboost_outcome"].astype(np.float64)
        challenger = challenger_artifact["tabm_pitch_gated"].astype(np.float64)
        inner = np.clip((1.0 - gamma) * parent + gamma * challenger, 1e-6, 1.0 - 1e-6)
        folds[year] = {
            "artifact": parent_artifact,
            "rows": rows,
            "regular": regular,
            "parent": parent,
            "challenger": challenger,
            "inner": inner,
            "atoms": atom_masks(rows, n_current),
            "n_current": n_current,
            "parent_path": parent_path,
            "challenger_path": challenger_path,
            "bootstrap": FastClusterBootstrap(
                parent_artifact["y"], parent, parent_artifact["cluster"],
                regular, iterations, seed + year,
            ),
        }

    minimum_coverage = float(prereg["constraints"]["minimum_R_coverage_each_source"])
    trials: list[dict[str, Any]] = []
    candidates: dict[tuple[str, int], tuple[np.ndarray, np.ndarray]] = {}
    for spec in route_specs():
        years: dict[str, Any] = {}
        eligible = True
        for year in YEARS:
            fold = folds[year]
            route = apply_spec(spec, fold["atoms"]) & fold["regular"]
            coverage = float(route.sum() / fold["regular"].sum())
            if coverage < minimum_coverage:
                eligible = False
            candidate = fold["parent"].copy()
            candidate[route] = fold["inner"][route]
            candidates[(spec["route_id"], year)] = (candidate, route)
            r_parent = score(fold["artifact"]["y"], fold["parent"], fold["regular"])
            r_candidate = score(fold["artifact"]["y"], candidate, fold["regular"])
            full_mask = np.ones(len(candidate), dtype=bool)
            full_parent = score(fold["artifact"]["y"], fold["parent"], full_mask)
            full_candidate = score(fold["artifact"]["y"], candidate, full_mask)
            interval = fold["bootstrap"].interval(candidate)
            years[str(year)] = {
                "coverage_R": coverage,
                "routed_rows": int(route.sum()),
                "R_gain": float(r_candidate["score"] - r_parent["score"]),
                "R_pitcher_cluster_95_ci": interval,
                "full_gain": float(full_candidate["score"] - full_parent["score"]),
            }
        if not eligible:
            continue
        trials.append({
            **spec,
            "minimum_R_gain": float(min(years[str(y)]["R_gain"] for y in YEARS)),
            "minimum_R_CI_lower": float(min(
                years[str(y)]["R_pitcher_cluster_95_ci"]["ci_low"] for y in YEARS
            )),
            "minimum_coverage_R": float(min(
                years[str(y)]["coverage_R"] for y in YEARS
            )),
            "years": years,
        })
    if not trials:
        raise AssertionError("no route met the preregistered coverage floor")
    selected = max(
        trials,
        key=lambda trial: (
            trial["minimum_R_gain"], trial["minimum_R_CI_lower"],
            -int(trial["leaves"]), trial["minimum_coverage_R"],
            # max() requires reversing lexicographic preference explicitly below
        ),
    )
    tied_key = (
        selected["minimum_R_gain"], selected["minimum_R_CI_lower"],
        -int(selected["leaves"]), selected["minimum_coverage_R"],
    )
    tied = [
        trial for trial in trials
        if (
            trial["minimum_R_gain"], trial["minimum_R_CI_lower"],
            -int(trial["leaves"]), trial["minimum_coverage_R"],
        ) == tied_key
    ]
    selected = min(tied, key=lambda trial: trial["route_id"])

    checks: dict[str, Any] = {}
    detailed_metrics: dict[str, Any] = {}
    passed = True
    for year in YEARS:
        fold = folds[year]
        candidate, route = candidates[(selected["route_id"], year)]
        masks = {
            "full": np.ones(len(candidate), dtype=bool),
            "R": fold["regular"],
            "F": ~fold["regular"],
        }
        metrics: dict[str, Any] = {}
        for route_index, (name, mask) in enumerate(masks.items()):
            parent_score = score(fold["artifact"]["y"], fold["parent"], mask)
            candidate_score = score(fold["artifact"]["y"], candidate, mask)
            interval = cluster_bootstrap_score_gain(
                fold["artifact"]["y"], fold["parent"], candidate,
                fold["artifact"]["cluster"], mask, iterations=iterations,
                seed=seed + year * 10000 + route_index * 1000,
            )
            metrics[name] = {
                "parent": parent_score, "candidate": candidate_score,
                "gain": float(candidate_score["score"] - parent_score["score"]),
                "pitcher_cluster_95_ci": interval,
            }
        detailed_metrics[str(year)] = metrics
        local = {
            "R_point_at_least_50": metrics["R"]["gain"] >= 50.0,
            "R_ci_lower_positive": metrics["R"]["pitcher_cluster_95_ci"]["ci_low"] > 0.0,
            "full_point_positive": metrics["full"]["gain"] > 0.0,
            "F_unchanged": abs(metrics["F"]["gain"]) <= 1e-12,
            "coverage_at_least_0p15": selected["years"][str(year)]["coverage_R"] >= minimum_coverage,
        }
        checks[str(year)] = local
        passed = passed and all(local.values())

    artifacts: dict[str, Any] = {}
    for year in YEARS:
        fold = folds[year]
        candidate, route = candidates[(selected["route_id"], year)]
        output = PRED / f"v5_pitch_gated_stable_route_source_{year}.npz"
        if output.exists():
            raise FileExistsError(f"immutable artifact exists: {output}")
        np.savez_compressed(
            output,
            y=fold["artifact"]["y"].astype(np.int8),
            row_index=fold["artifact"]["row_index"].astype(np.int64),
            cluster=fold["artifact"]["cluster"],
            parent_exact_c=fold["parent"],
            pitch_gated=fold["challenger"],
            route_mask=route.astype(np.int8),
            final_prediction=candidate,
        )
        artifacts[str(year)] = {
            "path": str(output.relative_to(ROOT)), "sha256": digest(output),
        }

    report = {
        "experiment_id": prereg["experiment_id"],
        "status": "source_pass" if passed else "source_failed_closed",
        "preregister_sha256": digest(PREREG),
        "script_sha256": digest(Path(__file__)),
        "years_read": list(YEARS),
        "years_not_read_by_this_script": [2022, 2023, 2024],
        "route_count_total": len(route_specs()),
        "route_count_coverage_eligible": len(trials),
        "trials": trials,
        "selected": selected,
        "selected_detailed_metrics": detailed_metrics,
        "source_gate": {"checks": checks, "pass": bool(passed)},
        "artifacts": artifacts,
        "goal_status": "active",
        "goal_completion_claimed": False,
    }
    REPORT.write_text(
        json.dumps(safe(report), ensure_ascii=False, indent=2), encoding="utf-8"
    )
    lock = {
        "experiment_id": prereg["experiment_id"],
        "status": "source_locked" if passed else "source_failed_closed",
        "preregister_sha256": digest(PREREG),
        "source_report_sha256": digest(REPORT),
        "source_script_sha256": digest(Path(__file__)),
        "selected_route": {
            key: selected[key] for key in ("route_id", "kind", "leaves", "terms")
        },
        "inner_gamma": gamma,
        "advance_to_2022_2023": bool(passed),
        "no_recipe_change_after_lock": True,
        "goal_completion_claimed": False,
    }
    LOCK.write_text(
        json.dumps(safe(lock), ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(safe({
        "status": report["status"], "selected": selected,
        "detailed_metrics": detailed_metrics, "checks": checks,
    }), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
