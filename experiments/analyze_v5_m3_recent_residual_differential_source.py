#!/usr/bin/env python3
"""Source-only M3 residual differential audit; 2024 is deliberately unopened."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.analyze_v4_temporal_residual_ridge import (  # noqa: E402
    json_safe,
    load_frames,
    score,
)
from experiments.stats import paired_bootstrap_brier_ci  # noqa: E402


TRAIN = ROOT / "open/data/train.csv"
PRED = ROOT / "experiments/results/predictions"
PREREG = ROOT / "experiments/params/v5_m3_recent_residual_differential_preregister.json"
REPORT = ROOT / "experiments/results/v5_m3_recent_residual_differential_source.json"
LOCK = ROOT / "experiments/params/v5_m3_recent_residual_differential_source_lock.json"
TARGETS = (2022, 2023)
SOURCES = {2022: (2020, 2021), 2023: (2021, 2022)}
CONTEXTS = {
    "same_hand": 1000.0,
    "two_strikes": 1000.0,
    "runner_present": 2000.0,
}
ANCHOR_FILES = {
    "published_v3_sparse_m3": "v3_sparse_m3_frozen_{year}.npz",
    "v5_honest_m3_r_identity": "v5_honest_m3_r_identity_{year}.npz",
    "v5_honest_m3_r_grid": "v5_honest_m3_r_grid_{year}.npz",
}
THRESHOLD = 132.11992465293324


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        return {key: np.asarray(archive[key]) for key in archive.files}


def raw_score(y: np.ndarray, prediction: np.ndarray) -> float:
    return float(score(y, prediction)["raw_competition_score"])


def contrast_table(
    pitcher: np.ndarray,
    context: np.ndarray,
    residual: np.ndarray,
    k: float,
) -> pd.Series:
    grouped = pd.DataFrame(
        {"pitcher": pitcher, "context": context, "residual": residual}
    ).groupby(["pitcher", "context"], observed=True)["residual"].agg(["mean", "size"])
    means = grouped["mean"].unstack("context")
    sizes = grouped["size"].unstack("context").fillna(0.0)
    for value in (0, 1):
        if value not in means:
            means[value] = np.nan
            sizes[value] = 0.0
    n0, n1 = sizes[0], sizes[1]
    n_eff = n0 * n1 / (n0 + n1).replace(0.0, np.nan)
    return ((means[1] - means[0]) * n_eff / (n_eff + k)).dropna()


def apply_contrast(
    table: pd.Series,
    pitcher: np.ndarray,
    context: np.ndarray,
) -> np.ndarray:
    mapped = pd.Series(pitcher).map(table).fillna(0.0).to_numpy(np.float64)
    return mapped * np.where(context == 1, 0.5, -0.5)


def scope_metrics(
    y: np.ndarray,
    baseline: np.ndarray,
    candidate: np.ndarray,
    cluster: np.ndarray,
    mask: np.ndarray,
    seed: int,
) -> dict[str, object]:
    interval = paired_bootstrap_brier_ci(
        y[mask], baseline[mask], candidate[mask],
        iterations=2000, seed=seed, clusters=cluster[mask],
    )
    return {
        "rows": int(mask.sum()),
        "baseline_score": raw_score(y[mask], baseline[mask]),
        "candidate_score": raw_score(y[mask], candidate[mask]),
        "point_gain": raw_score(y[mask], candidate[mask]) - raw_score(y[mask], baseline[mask]),
        "pitcher_cluster_95_ci": interval,
    }


def main() -> None:
    prereg = json.loads(PREREG.read_text(encoding="utf-8"))
    frames, artifacts = load_frames()
    raw = pd.read_csv(
        TRAIN,
        usecols=[
            "season", "game_type", "pitcher_id", "pitcher_hand", "batter_hand",
            "strikes_before", "num_runners_on", "control_success",
        ],
        encoding="utf-8-sig",
        low_memory=False,
    )
    panels: dict[int, pd.DataFrame] = {}
    for year, artifact in artifacts.items():
        panel = raw.iloc[np.asarray(artifact["row_index"], dtype=np.int64)].reset_index(drop=True)
        if not panel["season"].eq(year).all():
            raise ValueError(f"{year}: row alignment failed")
        if not np.array_equal(panel["control_success"].to_numpy(np.int8), artifact["y"].astype(np.int8)):
            raise ValueError(f"{year}: target alignment failed")
        panel["same_hand"] = (
            panel["pitcher_hand"].astype(str) == panel["batter_hand"].astype(str)
        ).astype(np.int8)
        panel["two_strikes"] = panel["strikes_before"].eq(2).astype(np.int8)
        panel["runner_present"] = panel["num_runners_on"].gt(0).astype(np.int8)
        panels[year] = panel

    anchors: dict[int, dict[str, np.ndarray]] = {}
    for year in TARGETS:
        anchors[year] = {}
        for name, pattern in ANCHOR_FILES.items():
            item = load_npz(PRED / pattern.format(year=year))
            for key in ("y", "row_index", "cluster"):
                if not np.array_equal(item[key], artifacts[year][key]):
                    raise ValueError(f"{year}/{name}: {key} alignment failed")
            anchors[year][name] = item["final_prediction"].astype(np.float64)

    directions: dict[int, np.ndarray] = {}
    table_meta: dict[str, object] = {}
    for target in TARGETS:
        source_years = SOURCES[target]
        source_mask = [panels[year]["game_type"].eq("R").to_numpy() for year in source_years]
        source_pitcher = np.concatenate([
            panels[year].loc[source_mask[index], "pitcher_id"].to_numpy(np.int64)
            for index, year in enumerate(source_years)
        ])
        source_residual = np.concatenate([
            (artifacts[year]["y"].astype(np.float64) - artifacts[year]["m3"].astype(np.float64))[
                source_mask[index]
            ]
            for index, year in enumerate(source_years)
        ])
        target_frame = panels[target]
        target_pitcher = target_frame["pitcher_id"].to_numpy(np.int64)
        direction = np.zeros(len(target_frame), dtype=np.float64)
        context_meta: dict[str, object] = {}
        for name, k in CONTEXTS.items():
            source_context = np.concatenate([
                panels[year].loc[source_mask[index], name].to_numpy(np.int8)
                for index, year in enumerate(source_years)
            ])
            table = contrast_table(source_pitcher, source_context, source_residual, k)
            part = apply_contrast(
                table,
                target_pitcher,
                target_frame[name].to_numpy(np.int8),
            )
            direction += part
            context_meta[name] = {
                "k": k,
                "pitcher_cells": int(len(table)),
                "target_nonzero_rate": float(np.mean(part != 0.0)),
                "direction_std": float(part.std()),
            }
        regular = target_frame["game_type"].eq("R").to_numpy()
        direction[~regular] = 0.0
        directions[target] = direction
        table_meta[str(target)] = {
            "source_seasons": source_years,
            "source_R_rows": int(sum(mask.sum() for mask in source_mask)),
            "target_R_rows": int(regular.sum()),
            "total_direction_std": float(direction.std()),
            "contexts": context_meta,
        }

    comparisons: dict[str, object] = {}
    g_values: list[float] = []
    same_parent_r_pass = True
    for year_offset, year in enumerate(TARGETS):
        fold = artifacts[year]
        y = fold["y"].astype(np.float64)
        cluster = fold["cluster"].astype(str)
        regular = panels[year]["game_type"].eq("R").to_numpy()
        comparisons[str(year)] = {}
        for anchor_offset, (anchor_name, baseline) in enumerate(anchors[year].items()):
            candidate = np.clip(baseline + directions[year], 1e-6, 1.0 - 1e-6)
            full = scope_metrics(
                y, baseline, candidate, cluster, np.ones(len(y), dtype=bool),
                58100 + 100 * year_offset + anchor_offset,
            )
            r_only = scope_metrics(
                y, baseline, candidate, cluster, regular,
                59100 + 100 * year_offset + anchor_offset,
            )
            comparisons[str(year)][anchor_name] = {"full": full, "R": r_only}
            g_values.extend([
                float(full["point_gain"]),
                float(full["pitcher_cluster_95_ci"]["score_ci_low"]),
            ])
            if anchor_name == "published_v3_sparse_m3":
                same_parent_r_pass = same_parent_r_pass and (
                    float(r_only["point_gain"]) > 0.0
                    and float(r_only["pitcher_cluster_95_ci"]["score_ci_low"]) > 0.0
                )

    g_dev = float(min(g_values))
    gates = {
        "same_parent_R_point_and_ci_positive_each_year": bool(same_parent_r_pass),
        "all_anchor_full_point_and_ci_above_threshold": bool(g_dev > THRESHOLD),
    }
    passed = bool(all(gates.values()))
    report = {
        "experiment_id": prereg["experiment_id"],
        "status": "passed_source_gate" if passed else "source_failed_closed",
        "preregister_sha256": sha256(PREREG),
        "implementation_sha256": sha256(Path(__file__)),
        "policy": {
            "official_train_only": True,
            "test_rows_read": False,
            "latest_target_label_read": 2023,
            "confirmation_2024_read": False,
            "leaderboard_derived_scale_used": False,
            "row_independent": True,
        },
        "recipe": prereg["recipe"],
        "table_meta": table_meta,
        "comparisons": comparisons,
        "g_dev": g_dev,
        "required_gain": THRESHOLD,
        "expected_lb_lower_if_no_later_failure": 1090.9100565103 + 0.75 * max(0.0, g_dev),
        "gates": gates,
        "gate_pass": passed,
        "decision": "lock before one-shot 2024" if passed else "close without 2024",
    }
    REPORT.write_text(json.dumps(json_safe(report), ensure_ascii=False, indent=2), encoding="utf-8")
    if passed:
        lock = {
            "experiment_id": prereg["experiment_id"],
            "status": "immutable_before_2024",
            "preregister_sha256": sha256(PREREG),
            "implementation_sha256": sha256(Path(__file__)),
            "recipe": prereg["recipe"],
            "source_report_sha256": sha256(REPORT),
        }
        LOCK.write_text(json.dumps(lock, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(json_safe({
        "status": report["status"],
        "g_dev": g_dev,
        "gates": gates,
        "point_gains": {
            year: {
                anchor: {
                    scope: row[scope]["point_gain"]
                    for scope in ("full", "R")
                }
                for anchor, row in comparisons[year].items()
            }
            for year in comparisons
        },
    }), ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
