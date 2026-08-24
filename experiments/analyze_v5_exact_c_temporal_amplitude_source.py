#!/usr/bin/env python3
"""Immutable source gate for the preregistered exact-C temporal amplitude."""

from __future__ import annotations

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
from experiments.run_v5_h1_residual import (  # noqa: E402
    cluster_bootstrap_score_gain,
)


TRAIN = ROOT / "open/data/train.csv"
PRED = ROOT / "experiments/results/predictions"
PREREG = ROOT / "experiments/params/v5_exact_c_temporal_amplitude_preregister.json"
REPORT = ROOT / "experiments/results/v5_exact_c_temporal_amplitude_source.json"
LOCK = ROOT / "experiments/params/v5_exact_c_temporal_amplitude_source_lock.json"
YEARS = (2020, 2021)


def load_rows() -> pd.DataFrame:
    columns = ["season", "game_type", "control_success"]
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
        raise AssertionError("source loader crossed or missed the 2021 boundary")
    return frame


def route_metrics(
    artifact: dict[str, np.ndarray], parent: np.ndarray, candidate: np.ndarray,
    regular: np.ndarray, seed: int, bootstrap: int,
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    masks = {
        "full": np.ones(len(parent), dtype=bool),
        "R": regular,
        "F": ~regular,
    }
    for route_index, (route, mask) in enumerate(masks.items()):
        parent_score = score(artifact["y"], parent, mask)
        candidate_score = score(artifact["y"], candidate, mask)
        interval = cluster_bootstrap_score_gain(
            artifact["y"], parent, candidate, artifact["cluster"], mask,
            iterations=bootstrap, seed=seed + 1000 * route_index,
        )
        result[route] = {
            "parent": parent_score,
            "candidate": candidate_score,
            "gain": float(candidate_score["score"] - parent_score["score"]),
            "pitcher_cluster_95_ci": interval,
        }
    return result


def main() -> None:
    if REPORT.exists() or LOCK.exists():
        raise FileExistsError("immutable source report or lock already exists")
    prereg = json.loads(PREREG.read_text(encoding="utf-8"))
    frame = load_rows()
    bootstrap = int(prereg["selection"]["bootstrap_replicates"])
    seed = int(prereg["selection"]["bootstrap_seed"])

    folds: dict[int, dict[str, Any]] = {}
    for year in YEARS:
        path = PRED / f"v4_m3_c_backtest_{year}_{year}.npz"
        artifact = load(path)
        rows = frame.loc[artifact["row_index"].astype(np.int64)]
        if not rows["season"].eq(year).all():
            raise AssertionError(f"season mismatch: {year}")
        if not np.array_equal(
            rows["control_success"].to_numpy(dtype=np.int8),
            artifact["y"].astype(np.int8),
        ):
            raise AssertionError(f"target mismatch: {year}")
        regular = rows["game_type"].astype(str).eq("R").to_numpy()
        center_rows = frame.loc[
            frame["season"].eq(year - 1)
            & frame["game_type"].astype(str).eq("R"),
            "control_success",
        ]
        if center_rows.empty:
            raise AssertionError(f"missing previous-season R center: {year}")
        folds[year] = {
            "artifact": artifact,
            "rows": rows,
            "regular": regular,
            "center": float(center_rows.mean()),
            "parent": artifact["catboost_outcome"].astype(np.float64),
            "path": path,
        }

    trials: list[dict[str, Any]] = []
    cache: dict[tuple[float, int], np.ndarray] = {}
    for alpha_value in prereg["formula"]["alpha_grid"]:
        alpha = float(alpha_value)
        years: dict[str, Any] = {}
        for year in YEARS:
            fold = folds[year]
            parent = fold["parent"]
            candidate = parent.copy()
            regular = fold["regular"]
            candidate[regular] = np.clip(
                fold["center"] + alpha * (parent[regular] - fold["center"]),
                1e-6, 1.0 - 1e-6,
            )
            cache[(alpha, year)] = candidate
            years[str(year)] = route_metrics(
                fold["artifact"], parent, candidate, regular,
                seed + int(round(alpha * 1000)) * 10000 + year, bootstrap,
            )
        trials.append({
            "alpha": alpha,
            "minimum_R_gain": float(min(years[str(y)]["R"]["gain"] for y in YEARS)),
            "minimum_R_CI_lower": float(min(
                years[str(y)]["R"]["pitcher_cluster_95_ci"]["ci_low"]
                for y in YEARS
            )),
            "minimum_full_gain": float(min(
                years[str(y)]["full"]["gain"] for y in YEARS
            )),
            "years": years,
        })

    selected = max(
        trials,
        key=lambda trial: (
            trial["minimum_R_gain"], trial["minimum_R_CI_lower"],
            -abs(trial["alpha"] - 1.0), -trial["alpha"],
        ),
    )
    checks: dict[str, Any] = {}
    passed = True
    for year in YEARS:
        metrics = selected["years"][str(year)]
        local = {
            "R_point_at_least_50": metrics["R"]["gain"] >= 50.0,
            "R_ci_lower_positive": (
                metrics["R"]["pitcher_cluster_95_ci"]["ci_low"] > 0.0
            ),
            "full_point_positive": metrics["full"]["gain"] > 0.0,
            "F_unchanged": abs(metrics["F"]["gain"]) <= 1e-12,
        }
        checks[str(year)] = local
        passed = passed and all(local.values())

    artifacts: dict[str, Any] = {}
    for year in YEARS:
        fold = folds[year]
        output = PRED / f"v5_exact_c_temporal_amplitude_source_{year}.npz"
        if output.exists():
            raise FileExistsError(f"immutable artifact exists: {output}")
        candidate = cache[(float(selected["alpha"]), year)]
        np.savez_compressed(
            output,
            y=fold["artifact"]["y"].astype(np.int8),
            row_index=fold["artifact"]["row_index"].astype(np.int64),
            cluster=fold["artifact"]["cluster"],
            parent_exact_c=fold["parent"],
            final_prediction=candidate,
        )
        artifacts[str(year)] = {
            "path": str(output.relative_to(ROOT)),
            "sha256": digest(output),
        }

    report = {
        "experiment_id": prereg["experiment_id"],
        "status": "source_pass" if passed else "source_failed_closed",
        "preregister_sha256": digest(PREREG),
        "script_sha256": digest(Path(__file__)),
        "years_read": list(YEARS),
        "years_not_read_by_this_script": [2022, 2023, 2024],
        "centers": {str(year): folds[year]["center"] for year in YEARS},
        "trials": trials,
        "selected": selected,
        "source_gate": {"checks": checks, "pass": bool(passed)},
        "artifacts": artifacts,
        "goal_status": "active",
        "goal_completion_claimed": False,
    }
    REPORT.write_text(
        json.dumps(safe(report), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    lock = {
        "experiment_id": prereg["experiment_id"],
        "status": "source_locked" if passed else "source_failed_closed",
        "preregister_sha256": digest(PREREG),
        "source_report_sha256": digest(REPORT),
        "source_script_sha256": digest(Path(__file__)),
        "selected_alpha": float(selected["alpha"]),
        "advance_to_2022_2023": bool(passed),
        "no_recipe_change_after_lock": True,
        "goal_completion_claimed": False,
    }
    LOCK.write_text(
        json.dumps(safe(lock), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(safe({
        "status": report["status"], "selected": selected,
        "checks": checks,
    }), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
