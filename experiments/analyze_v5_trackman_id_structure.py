#!/usr/bin/env python3
"""Source-only screen for shared structure in official TrackMan player IDs."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.analyze_v5_fine_pitchtype_latent import (  # noqa: E402
    PREDICTIONS,
    SOURCE_YEARS,
    TARGET,
    evaluate,
    json_safe,
    load_anchor,
)
from experiments.run_e20r_rolling import load_joined_trackman  # noqa: E402
from experiments.run_v5_h1_residual import (  # noqa: E402
    cluster_bootstrap_score_gain,
)


TRAIN = ROOT / "open/data/train.csv"
PREREG = ROOT / "experiments/params/v5_trackman_id_structure_preregister.json"
REPORT = ROOT / "experiments/results/v5_trackman_id_structure_source.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def source_frame() -> pd.DataFrame:
    with np.load(
        PREDICTIONS / "v4_m3_c_backtest_2021_2021.npz", allow_pickle=False
    ) as archive:
        last_index = int(np.max(archive["row_index"]))
    columns = ["season", "game_type", "pitcher_id", "batter_id", TARGET]
    frame = pd.read_csv(TRAIN, usecols=columns, nrows=last_index + 1)
    if set(frame["season"].unique()) != {2019, 2020, 2021}:
        raise ValueError("Source frame read a label after 2021")
    return frame


def majority_map(
    joined: pd.DataFrame, anonymous: str, trackman: str
) -> tuple[pd.Series, dict[str, Any]]:
    usable = joined.loc[
        joined[anonymous].notna() & joined[trackman].notna(), [anonymous, trackman]
    ].copy()
    counts = usable.groupby([anonymous, trackman], observed=True).size().rename("n")
    table = counts.reset_index().sort_values(
        [anonymous, "n", trackman], ascending=[True, False, True], kind="stable"
    )
    best = table.drop_duplicates(anonymous, keep="first").set_index(anonymous)
    totals = usable.groupby(anonymous, observed=True).size().reindex(best.index)
    purity = best["n"] / totals
    mapping = best[trackman].map(lambda value: str(int(value)))
    return mapping, {
        "linked_rows": int(len(usable)),
        "mapped_entities": int(len(mapping)),
        "mean_purity": float(purity.mean()),
        "minimum_purity": float(purity.min()),
        "entities_purity_ge_099": int(purity.ge(0.99).sum()),
    }


def id_group(values: pd.Series, description: str) -> pd.Series:
    text = values.astype("string")
    if description.endswith("_length"):
        return text.str.len().astype("Int64").astype("string")
    if description.endswith("_prefix1"):
        return text.str[:1]
    if description.endswith("_prefix2"):
        return text.str[:2]
    if description.endswith("_suffix1"):
        return text.str[-1:]
    if description.endswith("_suffix2"):
        return text.str[-2:]
    raise ValueError(f"Unknown ID group description: {description}")


def group_direction(
    history: pd.DataFrame,
    valid: pd.DataFrame,
    mapped_history: pd.Series,
    mapped_valid: pd.Series,
    description: str,
    k: float,
) -> tuple[np.ndarray, dict[str, Any]]:
    season_mean = history.groupby("season", observed=True)[TARGET].transform("mean")
    work = pd.DataFrame(
        {
            "signal": history[TARGET].to_numpy(dtype=np.float64)
            - season_mean.to_numpy(dtype=np.float64),
            "group": id_group(mapped_history, description),
        },
        index=history.index,
    ).dropna(subset=["group"])
    stats = work.groupby("group", observed=True)["signal"].agg(["sum", "count"])
    value = stats["sum"] / (stats["count"] + float(k))
    valid_group = id_group(mapped_valid, description)
    direction = valid_group.map(value).fillna(0.0).to_numpy(dtype=np.float64)
    return direction, {
        "history_mapped_rows": int(len(work)),
        "history_groups": int(len(stats)),
        "valid_mapped_rows": int(mapped_valid.notna().sum()),
        "valid_nonzero_direction_rows": int(np.count_nonzero(direction)),
    }


def main() -> None:
    if REPORT.exists():
        raise FileExistsError("Preserve the immutable ID-structure source report")
    prereg = json.loads(PREREG.read_text(encoding="utf-8"))
    if prereg["status"] != "preregistered_before_source_metrics":
        raise ValueError("Unexpected preregistration status")
    started = time.perf_counter()
    frame = source_frame()
    joined = load_joined_trackman()

    groups = list(prereg["candidate_groups"])
    ks = [int(value) for value in prereg["source_grid"]["shrinkage_k"]]
    gammas = [float(value) for value in prereg["source_grid"]["gammas"]]
    folds: dict[int, dict[str, Any]] = {}
    direction_cache: dict[tuple[int, str, int], np.ndarray] = {}
    mapping_meta: dict[str, Any] = {}
    direction_meta: dict[str, Any] = {}

    for year in SOURCE_YEARS:
        anchor = load_anchor(year)
        valid = frame.iloc[anchor["row_index"]].copy()
        if not valid["season"].eq(year).all():
            raise ValueError(f"{year}: anchor season mismatch")
        if not np.array_equal(
            valid[TARGET].to_numpy(dtype=np.int8), anchor["y"].astype(np.int8)
        ):
            raise ValueError(f"{year}: anchor target mismatch")
        history = frame.loc[
            (frame["season"] < year) & frame["game_type"].eq("R")
        ].copy()
        linked_history = joined.loc[joined["season"] < year]
        pitcher_map, pitcher_meta = majority_map(
            linked_history, "pitcher_id", "pitcher_trackman_id"
        )
        batter_map, batter_meta = majority_map(
            linked_history, "batter_id", "batter_trackman_id"
        )
        mapping_meta[str(year)] = {
            "source_seasons": sorted(
                int(value) for value in linked_history["season"].unique()
            ),
            "pitcher": pitcher_meta,
            "batter": batter_meta,
        }
        mapped_history = {
            "pitcher": history["pitcher_id"].map(pitcher_map),
            "batter": history["batter_id"].map(batter_map),
        }
        mapped_valid = {
            "pitcher": valid["pitcher_id"].map(pitcher_map),
            "batter": valid["batter_id"].map(batter_map),
        }
        for description in groups:
            entity = "pitcher" if description.startswith("pitcher_") else "batter"
            for k in ks:
                direction, meta = group_direction(
                    history,
                    valid,
                    mapped_history[entity],
                    mapped_valid[entity],
                    description,
                    float(k),
                )
                direction_cache[(year, description, k)] = direction
                direction_meta[f"{year}:{description}:k{k}"] = meta
        folds[year] = {
            "anchor": anchor,
            "game_type": valid["game_type"].astype(str).to_numpy(),
            "r_mask": valid["game_type"].eq("R").to_numpy(dtype=bool),
        }

    candidates: list[dict[str, Any]] = []
    for description in groups:
        for k in ks:
            for gamma in gammas:
                years: dict[str, Any] = {}
                for year in SOURCE_YEARS:
                    fold = folds[year]
                    anchor = fold["anchor"]
                    base = anchor["catboost_outcome"].astype(np.float64)
                    candidate = base.copy()
                    direction = direction_cache[(year, description, k)]
                    regular = fold["r_mask"]
                    candidate[regular] = np.clip(
                        candidate[regular] + gamma * direction[regular],
                        1e-6,
                        1.0 - 1e-6,
                    )
                    years[str(year)] = evaluate(
                        anchor["y"], base, candidate, fold["game_type"]
                    )
                full = [years[str(year)]["gains"]["all"] for year in SOURCE_YEARS]
                regular_gain = [
                    years[str(year)]["gains"]["R"] for year in SOURCE_YEARS
                ]
                candidates.append(
                    {
                        "group": description,
                        "k": k,
                        "gamma": gamma,
                        "min_full_gain": float(min(full)),
                        "min_r_gain": float(min(regular_gain)),
                        "mean_full_gain": float(np.mean(full)),
                        "years": years,
                    }
                )
    candidates.sort(
        key=lambda row: (
            row["min_full_gain"],
            row["min_r_gain"],
            row["mean_full_gain"],
            row["k"],
            -row["gamma"],
            -len(row["group"]),
        ),
        reverse=True,
    )
    selected = candidates[0]

    intervals: dict[str, Any] = {}
    artifacts: dict[str, str] = {}
    for offset, year in enumerate(SOURCE_YEARS):
        fold = folds[year]
        anchor = fold["anchor"]
        base = anchor["catboost_outcome"].astype(np.float64)
        direction = direction_cache[(year, selected["group"], selected["k"])]
        regular = fold["r_mask"]
        candidate = base.copy()
        candidate[regular] = np.clip(
            candidate[regular] + selected["gamma"] * direction[regular],
            1e-6,
            1.0 - 1e-6,
        )
        intervals[str(year)] = cluster_bootstrap_score_gain(
            anchor["y"],
            base,
            candidate,
            anchor["cluster"].astype(str),
            regular,
            2000,
            592200 + offset,
        )
        path = PREDICTIONS / f"v5_trackman_id_structure_source_{year}.npz"
        if path.exists():
            raise FileExistsError(f"Preserve existing prediction artifact: {path}")
        np.savez_compressed(
            path,
            y=anchor["y"],
            row_index=anchor["row_index"],
            cluster=anchor["cluster"],
            base=base.astype(np.float32),
            id_group_direction=direction.astype(np.float32),
            final_prediction=candidate.astype(np.float32),
        )
        artifacts[str(year)] = str(path.relative_to(ROOT))

    gate = prereg["source_gate"]
    conditions = {
        "minimum_full_gain_each_year": bool(
            selected["min_full_gain"]
            >= float(gate["minimum_full_gain_each_year"])
        ),
        "minimum_r_gain_each_year": bool(
            selected["min_r_gain"] >= float(gate["minimum_r_gain_each_year"])
        ),
        "r_cluster_ci_lower_positive_each_year": bool(
            all(value["ci_low"] > 0.0 for value in intervals.values())
        ),
    }
    passed = bool(all(conditions.values()))
    payload = {
        "experiment_id": prereg["experiment_id"],
        "status": "passed_source_gate" if passed else "failed_source_gate",
        "preregister": str(PREREG.relative_to(ROOT)),
        "preregister_sha256": sha256(PREREG),
        "policy": {
            "official_data_only": True,
            "external_identity_lookup": False,
            "test_rows_read": False,
            "latest_control_label_season_used_for_metrics": 2021,
            "current_or_validation_trackman_at_inference": False,
            "row_independent": True,
            "automatic_submission": False,
        },
        "mapping": mapping_meta,
        "direction_metadata": direction_meta,
        "candidate_count": len(candidates),
        "selected": selected,
        "selected_r_cluster_intervals": intervals,
        "conditions": conditions,
        "gate_pass": passed,
        "decision": "retrain exact-C with frozen ID feature" if passed else "close without 2022+",
        "top_candidates": candidates[:20],
        "prediction_artifacts": artifacts,
        "artifact_hashes": {
            "preregister": sha256(PREREG),
            **{
                f"anchor_{year}": sha256(
                    PREDICTIONS / f"v4_m3_c_backtest_{year}_{year}.npz"
                )
                for year in SOURCE_YEARS
            },
        },
        "elapsed_seconds": float(time.perf_counter() - started),
    }
    REPORT.write_text(
        json.dumps(json_safe(payload), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(
        json.dumps(
            json_safe(
                {
                    "status": payload["status"],
                    "selected": selected,
                    "intervals": intervals,
                    "conditions": conditions,
                    "elapsed_seconds": payload["elapsed_seconds"],
                }
            ),
            ensure_ascii=False,
            indent=2,
        ),
        flush=True,
    )
    print(f"wrote {REPORT}", flush=True)


if __name__ == "__main__":
    main()
