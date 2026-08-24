#!/usr/bin/env python3
"""Source-only audit of whether row_id carries portable predictive semantics.

The audit deliberately never reads test.csv and restricts train.csv to the
2019--2021 prefix.  It separates two questions:

1. Is the suffix anything more than file position?
2. Even if file position is predictive in train, does a fixed correction
   learned in one source season improve the next source season?

The preregistered deployment gate requires both semantic portability and
cross-season performance.  A train-order artifact cannot pass that gate.
"""

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

from experiments.stats import paired_bootstrap_brier_ci  # noqa: E402


DATA = ROOT / "open" / "data" / "train.csv"
DESCRIPTION = ROOT / "open" / "data_description.md"
RESULTS = ROOT / "experiments" / "results"
PREDICTIONS = RESULTS / "predictions"
PREREG = ROOT / "experiments" / "params" / "v5_row_id_source_audit_preregister.json"
REPORT = RESULTS / "v5_row_id_source_audit.json"
C_PATHS = {
    2020: PREDICTIONS / "v4_m3_c_backtest_2020_2020.npz",
    2021: PREDICTIONS / "v4_m3_c_backtest_2021_2021.npz",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def score(y: np.ndarray, prediction: np.ndarray) -> float:
    y = np.asarray(y, dtype=np.float64)
    prediction = np.asarray(prediction, dtype=np.float64)
    reference = float(y.mean() * (1.0 - y.mean()))
    brier = float(np.mean(np.square(prediction - y)))
    return max(0.0, 100_000.0 * (1.0 - brier / reference))


def safe_corr(left: np.ndarray, right: np.ndarray) -> float | None:
    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    keep = np.isfinite(left) & np.isfinite(right)
    if keep.sum() < 3 or left[keep].std() == 0.0 or right[keep].std() == 0.0:
        return None
    return float(np.corrcoef(left[keep], right[keep])[0, 1])


def load_source_prefix() -> pd.DataFrame:
    # The end of the 2021 exact-C fold is the last permitted row.  nrows keeps
    # pandas from even parsing a 2022 label during this source-only audit.
    with np.load(C_PATHS[2021], allow_pickle=False) as fold:
        last_source_index = int(fold["row_index"].max())
    frame = pd.read_csv(
        DATA,
        usecols=[
            "row_id",
            "season",
            "game_month",
            "game_type",
            "pitcher_id",
            "control_success",
        ],
        nrows=last_source_index + 1,
    )
    if set(frame["season"].unique()) != {2019, 2020, 2021}:
        raise ValueError("Source prefix unexpectedly contains a non-source season")
    suffix = frame["row_id"].str.extract(r"(\d+)$", expand=False)
    if suffix.isna().any():
        raise ValueError("row_id without a numeric suffix")
    frame["row_suffix"] = suffix.astype(np.int64)
    starts = frame.groupby("season", sort=False)["row_suffix"].transform("min")
    frame["season_position"] = frame["row_suffix"] - starts + 1
    return frame


def load_parent(year: int, frame: pd.DataFrame) -> dict[str, np.ndarray]:
    with np.load(C_PATHS[year], allow_pickle=False) as fold:
        row_index = fold["row_index"].astype(np.int64)
        y = fold["y"].astype(np.float64)
        parent = fold["catboost_outcome"].astype(np.float64)
        cluster = fold["cluster"].astype(str)
    view = frame.iloc[row_index]
    if not np.array_equal(view["control_success"].to_numpy(dtype=np.int8), y.astype(np.int8)):
        raise ValueError(f"{year}: exact-C artifact is not aligned with train.csv")
    if not view["season"].eq(year).all():
        raise ValueError(f"{year}: exact-C artifact contains another season")
    return {
        "row_index": row_index,
        "y": y,
        "parent": parent,
        "cluster": cluster,
        "position": view["season_position"].to_numpy(dtype=np.int64),
        "game_type": view["game_type"].astype(str).to_numpy(),
        "regular": view["game_type"].astype(str).eq("R").to_numpy(),
    }


def source_response(
    source: pd.DataFrame,
    source_parent: np.ndarray | None,
) -> np.ndarray:
    y = source["control_success"].to_numpy(dtype=np.float64)
    if source_parent is not None:
        if len(source_parent) != len(source):
            raise ValueError("Source parent is not aligned")
        return y - source_parent
    centers = source.groupby("game_type")["control_success"].transform("mean")
    return y - centers.to_numpy(dtype=np.float64)


def fit_position_map(
    position: np.ndarray,
    game_type: np.ndarray,
    response: np.ndarray,
    width: int,
    prior: int,
    scope: str,
) -> dict[tuple[str, int] | int, float]:
    block = ((np.asarray(position, dtype=np.int64) - 1) // width).astype(np.int64)
    if scope == "global":
        key_frame = pd.DataFrame({"block": block, "response": response})
        stats = key_frame.groupby("block", sort=True)["response"].agg(["sum", "count"])
        return {
            int(index): float(row["sum"] / (row["count"] + prior))
            for index, row in stats.iterrows()
        }
    if scope == "game_type":
        key_frame = pd.DataFrame(
            {"game_type": game_type.astype(str), "block": block, "response": response}
        )
        stats = key_frame.groupby(["game_type", "block"], sort=True)["response"].agg(
            ["sum", "count"]
        )
        return {
            (str(index[0]), int(index[1])): float(
                row["sum"] / (row["count"] + prior)
            )
            for index, row in stats.iterrows()
        }
    raise ValueError(f"Unknown scope: {scope}")


def apply_position_map(
    mapping: dict,
    position: np.ndarray,
    game_type: np.ndarray,
    width: int,
    scope: str,
) -> np.ndarray:
    block = ((np.asarray(position, dtype=np.int64) - 1) // width).astype(np.int64)
    if scope == "global":
        return np.asarray([mapping.get(int(value), 0.0) for value in block])
    return np.asarray(
        [
            mapping.get((str(kind), int(value)), 0.0)
            for kind, value in zip(game_type, block, strict=True)
        ],
        dtype=np.float64,
    )


def transfer_candidate(
    source: pd.DataFrame,
    source_parent: np.ndarray | None,
    target: dict[str, np.ndarray],
    width: int,
    prior: int,
    scope: str,
    alpha: float,
) -> np.ndarray:
    response = source_response(source, source_parent)
    mapping = fit_position_map(
        source["season_position"].to_numpy(dtype=np.int64),
        source["game_type"].astype(str).to_numpy(),
        response,
        width,
        prior,
        scope,
    )
    correction = apply_position_map(
        mapping,
        target["position"],
        target["game_type"],
        width,
        scope,
    )
    return np.clip(target["parent"] + alpha * correction, 1e-6, 1.0 - 1e-6)


def metrics(target: dict[str, np.ndarray], candidate: np.ndarray) -> dict[str, float]:
    regular = target["regular"]
    parent_full = score(target["y"], target["parent"])
    candidate_full = score(target["y"], candidate)
    parent_r = score(target["y"][regular], target["parent"][regular])
    candidate_r = score(target["y"][regular], candidate[regular])
    return {
        "parent_full_score": parent_full,
        "candidate_full_score": candidate_full,
        "full_gain": candidate_full - parent_full,
        "parent_r_score": parent_r,
        "candidate_r_score": candidate_r,
        "r_gain": candidate_r - parent_r,
        "candidate_mean": float(candidate.mean()),
        "candidate_std": float(candidate.std()),
    }


def fixed_block_profile(frame: pd.DataFrame, width: int = 25000) -> dict:
    profiles: dict[int, pd.Series] = {}
    output: dict[str, dict] = {}
    for year, year_frame in frame.groupby("season", sort=True):
        block = ((year_frame["season_position"] - 1) // width).astype(int)
        centered = year_frame["control_success"] - year_frame["control_success"].mean()
        profile = centered.groupby(block).mean()
        profiles[int(year)] = profile
        output[str(year)] = {
            "rows": int(len(year_frame)),
            "suffix_min": int(year_frame["row_suffix"].min()),
            "suffix_max": int(year_frame["row_suffix"].max()),
            "position_max": int(year_frame["season_position"].max()),
            "month_position_correlation": safe_corr(
                year_frame["season_position"].to_numpy(),
                year_frame["game_month"].to_numpy(),
            ),
            "target_position_correlation": safe_corr(
                year_frame["season_position"].to_numpy(),
                year_frame["control_success"].to_numpy(),
            ),
            "centered_target_by_fixed_block": {
                str(int(index)): float(value) for index, value in profile.items()
            },
        }
    pair_correlations = {}
    for left, right in ((2019, 2020), (2020, 2021)):
        joined = pd.concat(
            [profiles[left].rename("left"), profiles[right].rename("right")], axis=1
        ).dropna()
        pair_correlations[f"{left}_to_{right}"] = {
            "common_blocks": int(len(joined)),
            "correlation": safe_corr(joined["left"].to_numpy(), joined["right"].to_numpy()),
        }
    return {
        "fixed_block_width": width,
        "years": output,
        "adjacent_year_centered_target_correlations": pair_correlations,
    }


def main() -> None:
    prereg = json.loads(PREREG.read_text(encoding="utf-8"))
    frame = load_source_prefix()
    parents = {year: load_parent(year, frame) for year in (2020, 2021)}

    expected_position = np.arange(1, len(frame) + 1, dtype=np.int64)
    suffix_is_file_position = bool(
        np.array_equal(frame["row_suffix"].to_numpy(), expected_position)
    )
    per_year_contiguous = {
        str(int(year)): bool(
            int(group["row_suffix"].max() - group["row_suffix"].min() + 1)
            == len(group)
        )
        for year, group in frame.groupby("season", sort=True)
    }

    source_2019 = frame.loc[frame["season"].eq(2019)].copy()
    source_2020 = frame.iloc[parents[2020]["row_index"]].copy()
    source_parent_2020 = parents[2020]["parent"]

    grid = prereg["source_protocol"]
    candidates: list[dict] = []
    for width in grid["fixed_block_widths"]:
        for prior in grid["shrinkage_priors"]:
            for scope in grid["scopes"]:
                for alpha in grid["alphas"]:
                    prediction_2020 = transfer_candidate(
                        source_2019,
                        None,
                        parents[2020],
                        int(width),
                        int(prior),
                        str(scope),
                        float(alpha),
                    )
                    prediction_2021 = transfer_candidate(
                        source_2020,
                        source_parent_2020,
                        parents[2021],
                        int(width),
                        int(prior),
                        str(scope),
                        float(alpha),
                    )
                    year_metrics = {
                        "2020": metrics(parents[2020], prediction_2020),
                        "2021": metrics(parents[2021], prediction_2021),
                    }
                    candidates.append(
                        {
                            "width": int(width),
                            "prior": int(prior),
                            "scope": str(scope),
                            "alpha": float(alpha),
                            "min_full_gain": float(
                                min(value["full_gain"] for value in year_metrics.values())
                            ),
                            "min_r_gain": float(
                                min(value["r_gain"] for value in year_metrics.values())
                            ),
                            "mean_full_gain": float(
                                np.mean(
                                    [value["full_gain"] for value in year_metrics.values()]
                                )
                            ),
                            "years": year_metrics,
                        }
                    )

    selected = max(
        candidates,
        key=lambda item: (
            item["min_full_gain"],
            item["min_r_gain"],
            item["mean_full_gain"],
            -item["alpha"],
            item["prior"],
            item["width"],
            item["scope"] == "global",
        ),
    )

    selected_predictions = {
        2020: transfer_candidate(
            source_2019,
            None,
            parents[2020],
            selected["width"],
            selected["prior"],
            selected["scope"],
            selected["alpha"],
        ),
        2021: transfer_candidate(
            source_2020,
            source_parent_2020,
            parents[2021],
            selected["width"],
            selected["prior"],
            selected["scope"],
            selected["alpha"],
        ),
    }
    intervals: dict[str, dict] = {}
    for offset, year in enumerate((2020, 2021)):
        target = parents[year]
        candidate = selected_predictions[year]
        full = paired_bootstrap_brier_ci(
            target["y"],
            target["parent"],
            candidate,
            iterations=int(grid["bootstrap"]["iterations"]),
            seed=int(grid["bootstrap"]["seed_base"]) + offset,
            clusters=target["cluster"],
        )
        regular = target["regular"]
        r_only = paired_bootstrap_brier_ci(
            target["y"][regular],
            target["parent"][regular],
            candidate[regular],
            iterations=int(grid["bootstrap"]["iterations"]),
            seed=int(grid["bootstrap"]["seed_base"]) + 100 + offset,
            clusters=target["cluster"][regular],
        )
        intervals[str(year)] = {"full": full, "R": r_only}

    semantic_conditions = {
        "suffix_not_exact_csv_position": not suffix_is_file_position,
        "documented_ordered_semantics": False,
        "per_year_suffix_is_contiguous": per_year_contiguous,
        "documentation_evidence": (
            "open/data_description.md: row_id is a unique sample identifier used "
            "to match the submission; no time/order semantics are documented."
        ),
    }
    gate_spec = prereg["source_gate"]
    performance_conditions = {
        "minimum_full_gain_each_transfer": bool(
            selected["min_full_gain"]
            >= float(gate_spec["minimum_full_gain_each_transfer"])
        ),
        "minimum_r_gain_each_transfer": bool(
            selected["min_r_gain"] >= float(gate_spec["minimum_r_gain_each_transfer"])
        ),
        "full_cluster_ci_lower_positive_each_transfer": bool(
            all(intervals[str(year)]["full"]["score_ci_low"] > 0.0 for year in (2020, 2021))
        ),
        "r_cluster_ci_lower_positive_each_transfer": bool(
            all(intervals[str(year)]["R"]["score_ci_low"] > 0.0 for year in (2020, 2021))
        ),
    }
    semantic_pass = bool(
        semantic_conditions["suffix_not_exact_csv_position"]
        and semantic_conditions["documented_ordered_semantics"]
    )
    performance_pass = bool(all(performance_conditions.values()))
    gate_pass = bool(semantic_pass and performance_pass)

    report = {
        "experiment_id": prereg["experiment_id"],
        "status": "passed" if gate_pass else "failed_source_gate",
        "preregistration": {
            "path": str(PREREG.relative_to(ROOT)).replace("\\", "/"),
            "sha256": sha256(PREREG),
        },
        "policy_audit": {
            "test_csv_read": False,
            "latest_label_season_read": 2021,
            "validation_or_test_aggregation_used_for_deployment_feature": False,
            "row_local_deployment_requirement": True,
        },
        "source_rows": int(len(frame)),
        "structure": {
            "numeric_suffix_equals_one_based_csv_position": suffix_is_file_position,
            "per_year_suffix_contiguous": per_year_contiguous,
            "profile": fixed_block_profile(frame),
        },
        "grid_candidate_count": len(candidates),
        "selected": selected,
        "selected_cluster_intervals": intervals,
        "gate": {
            "semantic_conditions": semantic_conditions,
            "semantic_pass": semantic_pass,
            "performance_conditions": performance_conditions,
            "performance_pass": performance_pass,
            "gate_pass": gate_pass,
            "decision": (
                "preregister a 2022 development experiment"
                if gate_pass
                else "close row_id axis without reading 2022+ metrics"
            ),
        },
        "artifacts": {
            str(path.relative_to(ROOT)).replace("\\", "/"): sha256(path)
            for path in [DATA, DESCRIPTION, C_PATHS[2020], C_PATHS[2021]]
        },
    }
    REPORT.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report["gate"], indent=2, ensure_ascii=False))
    print(json.dumps({"selected": selected}, indent=2, ensure_ascii=False))
    print(f"Wrote {REPORT}")


if __name__ == "__main__":
    main()
