#!/usr/bin/env python3
"""Source gate for dense counter-labelled pitch-group outcome experts."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.run_v5_h1_residual import (  # noqa: E402
    cluster_bootstrap_score_gain,
)


PRED = ROOT / "experiments/results/predictions"
RESULTS = ROOT / "experiments/results"
TRAIN = ROOT / "open/data/train.csv"
PREREG = ROOT / "experiments/params/v5_dense_pitchtype_moe_preregister.json"
REPORT = RESULTS / "v5_dense_pitchtype_moe_source_gate_v2.json"
SEMANTIC_SOURCE = (
    RESULTS / "v5_counter_reconstructed_pitch_hierarchy_source.json"
)
YEARS = (2020, 2021)
PARENT = {
    2020: "v4_m3_c_backtest_2020_2020.npz",
    2021: "v4_m3_c_backtest_2021_2021.npz",
}
STAGES = {
    2020: "v5_dense_pitchtype_moe_source2020",
    2021: "v5_dense_pitchtype_moe_source2021",
}
KEY = "catboost_dense_pitchtype_moe"
PREFIX = f"{KEY}__"


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def load(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        return {key: np.asarray(archive[key]) for key in archive.files}


def score(y: np.ndarray, p: np.ndarray, mask: np.ndarray) -> dict[str, float]:
    target = y[mask].astype(np.float64)
    prediction = p[mask].astype(np.float64)
    rate = float(target.mean())
    reference = max(rate * (1.0 - rate), 1e-12)
    brier = float(np.mean(np.square(prediction - target)))
    return {
        "rows": int(mask.sum()),
        "target_rate": rate,
        "prediction_mean": float(prediction.mean()),
        "prediction_std": float(prediction.std()),
        "brier": brier,
        "score": float(100000.0 * (1.0 - brier / reference)),
    }


def evaluate(
    artifact: dict[str, np.ndarray],
    parent: np.ndarray,
    direction: np.ndarray,
    route: np.ndarray,
    masks: dict[str, np.ndarray],
    gamma: float,
    bootstrap: int,
    seed: int,
) -> dict[str, Any]:
    candidate = parent.astype(np.float64).copy()
    candidate[route] += gamma * (direction[route] - parent[route])
    candidate = np.clip(candidate, 1e-6, 1.0 - 1e-6)
    result: dict[str, Any] = {
        "gamma": float(gamma),
        "route_rows": int(route.sum()),
        "routes": {},
    }
    for index, (name, mask) in enumerate(masks.items()):
        parent_metrics = score(artifact["y"], parent, mask)
        candidate_metrics = score(artifact["y"], candidate, mask)
        interval = cluster_bootstrap_score_gain(
            artifact["y"], parent, candidate, artifact["cluster"], mask,
            iterations=bootstrap, seed=seed + 1000 * index,
        )
        gain = candidate_metrics["score"] - parent_metrics["score"]
        if abs(gain - interval["point"]) > 1e-8:
            raise AssertionError(f"score/CI point mismatch: {name}")
        result["routes"][name] = {
            "parent": parent_metrics,
            "candidate": candidate_metrics,
            "gain": gain,
            "pitcher_cluster_95_ci": interval,
        }
    return result


def safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def main() -> None:
    if REPORT.exists():
        raise FileExistsError(f"immutable result already exists: {REPORT}")
    prereg = json.loads(PREREG.read_text(encoding="utf-8"))
    semantic_source = json.loads(SEMANTIC_SOURCE.read_text(encoding="utf-8"))
    all_types = pd.read_csv(TRAIN, usecols=["game_type"])["game_type"].astype(str)
    bootstrap = int(prereg["bootstrap_iterations"])
    folds: dict[int, dict[str, Any]] = {}
    semantic_pass = True

    for year in YEARS:
        candidate_path = PRED / f"{STAGES[year]}_{year}.npz"
        parent_path = PRED / PARENT[year]
        candidate = load(candidate_path)
        parent_artifact = load(parent_path)
        for align_key in ("y", "row_index", "cluster"):
            if not np.array_equal(candidate[align_key], parent_artifact[align_key]):
                raise ValueError(f"alignment mismatch: {year}/{align_key}")
        types = all_types.iloc[candidate["row_index"].astype(np.int64)].to_numpy(dtype=str)
        regular = types == "R"
        metadata = json.loads(
            (RESULTS / f"{STAGES[year]}.json").read_text(encoding="utf-8")
        )
        details = metadata["folds"][0]["fit_details"][KEY]
        coverage = float(details["history_dense_label_coverage"])
        audited_semantic = semantic_source["semantic_audit"][str(year)]
        if abs(coverage - float(audited_semantic["history_coverage"])) > 1e-12:
            raise ValueError(f"dense-label coverage audit mismatch: {year}")
        if int(details["history_dense_label_rows"]) != int(
            audited_semantic["history_reconstructed_rows"]
        ):
            raise ValueError(f"dense-label row audit mismatch: {year}")
        agreement = float(audited_semantic["history_trackman_agreement"])
        fold_semantic = bool(
            coverage
            >= float(
                prereg["semantic_gate"][
                    "minimum_history_label_coverage_each_year"
                ]
            )
            and agreement is not None
            and agreement
            >= float(
                prereg["semantic_gate"][
                    "minimum_trackman_group_agreement_each_year"
                ]
            )
        )
        semantic_pass &= fold_semantic
        folds[year] = {
            "candidate": candidate,
            "parent": parent_artifact["catboost_outcome"].astype(np.float64),
            "types": types,
            "regular": regular,
            "masks": {
                "full": np.ones(len(candidate["y"]), dtype=bool),
                "R": regular,
            },
            "candidate_path": candidate_path,
            "parent_path": parent_path,
            "semantic": {
                "history_regular_rows": int(details["history_regular_rows"]),
                "history_dense_label_rows": int(details["history_dense_label_rows"]),
                "history_dense_label_coverage": coverage,
                "history_trackman_group_agreement": agreement,
                "history_trackman_comparison_rows": int(
                    audited_semantic["history_trackman_comparison_rows"]
                ),
                "trackman_agreement_source": str(
                    SEMANTIC_SOURCE.relative_to(ROOT)
                ),
                "selector_validation_dense_rows": int(
                    details["diagnostic_validation_dense_rows"]
                ),
                "selector_top1_accuracy": float(
                    details["diagnostic_selector_top1_accuracy"]
                ),
                "selector_log_loss": float(
                    details["diagnostic_selector_log_loss"]
                ),
                "semantic_gate_pass": fold_semantic,
            },
        }

    if not semantic_pass:
        report = {
            "experiment_id": prereg["experiment_id"],
            "status": "failed_semantic_gate",
            "preregister_sha256": digest(PREREG),
            "semantic": {str(year): folds[year]["semantic"] for year in YEARS},
            "control_metrics_computed": False,
            "years_read": list(YEARS),
            "years_not_read": [2022, 2023, 2024],
        }
        REPORT.write_text(
            json.dumps(safe(report), ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(json.dumps(safe(report), ensure_ascii=False, indent=2))
        return

    trials: list[dict[str, Any]] = []
    for gamma in prereg["candidate"]["top_level_blend_grid"]:
        years: dict[str, Any] = {}
        for year in YEARS:
            fold = folds[year]
            years[str(year)] = evaluate(
                fold["candidate"],
                fold["parent"],
                fold["candidate"][KEY].astype(np.float64),
                fold["regular"],
                fold["masks"],
                float(gamma),
                bootstrap,
                510000 + 10000 * year + int(float(gamma) * 100),
            )
        full_gains = [years[str(year)]["routes"]["full"]["gain"] for year in YEARS]
        r_gains = [years[str(year)]["routes"]["R"]["gain"] for year in YEARS]
        trials.append(
            {
                "gamma": float(gamma),
                "minimum_full_gain": float(min(full_gains)),
                "minimum_R_gain": float(min(r_gains)),
                "mean_full_gain": float(np.mean(full_gains)),
                "years": years,
            }
        )
    selected = max(
        trials,
        key=lambda item: (
            item["minimum_full_gain"], item["minimum_R_gain"], -item["gamma"]
        ),
    )

    oracle: dict[str, Any] = {}
    for year in YEARS:
        fold = folds[year]
        available = fold["candidate"][
            f"{PREFIX}diagnostic_true_group_available"
        ].astype(bool)
        oracle_route = fold["regular"] & available
        oracle[str(year)] = evaluate(
            fold["candidate"],
            fold["parent"],
            fold["candidate"][f"{PREFIX}diagnostic_true_group_oracle"].astype(
                np.float64
            ),
            oracle_route,
            fold["masks"],
            1.0,
            bootstrap,
            610000 + 10000 * year,
        )
        oracle[str(year)]["goal_gate_eligible"] = False
        oracle[str(year)]["exclusion_reason"] = (
            "uses next-row reconstructed current validation pitch group"
        )

    threshold_full = float(
        prereg["source_gate"]["minimum_full_gain_each_year"]
    )
    threshold_r = float(prereg["source_gate"]["minimum_r_gain_each_year"])
    checks: list[bool] = []
    for year in YEARS:
        routes = selected["years"][str(year)]["routes"]
        checks.extend(
            (
                routes["full"]["gain"] >= threshold_full,
                routes["R"]["gain"] >= threshold_r,
                routes["full"]["pitcher_cluster_95_ci"]["ci_low"] > 0.0,
                routes["R"]["pitcher_cluster_95_ci"]["ci_low"] > 0.0,
            )
        )
    passed = bool(all(checks))
    report = {
        "experiment_id": prereg["experiment_id"],
        "status": "source_pass" if passed else "source_failed",
        "preregister_sha256": digest(PREREG),
        "script_sha256": digest(Path(__file__)),
        "semantic_source": {
            "path": str(SEMANTIC_SOURCE.relative_to(ROOT)),
            "sha256": digest(SEMANTIC_SOURCE),
            "note": (
                "Immutable official-TrackMan agreement audit for the identical "
                "dense counter-label derivation; V1 analyzer report is retained."
            ),
        },
        "years_read": list(YEARS),
        "years_not_read": [2022, 2023, 2024],
        "semantic": {str(year): folds[year]["semantic"] for year in YEARS},
        "artifacts": {
            str(year): {
                "candidate": str(folds[year]["candidate_path"].relative_to(ROOT)),
                "candidate_sha256": digest(folds[year]["candidate_path"]),
                "parent": str(folds[year]["parent_path"].relative_to(ROOT)),
            }
            for year in YEARS
        },
        "trials": trials,
        "selected": selected,
        "diagnostic_true_group_oracle_excluded_from_goal_gate": oracle,
        "source_gate": {
            "minimum_full_gain_each_year": threshold_full,
            "minimum_R_gain_each_year": threshold_r,
            "ci_lower_positive_each_year": True,
            "passed": passed,
            "decision": (
                "freeze and advance to 2022/2023"
                if passed
                else "close without reading 2022+ candidate labels"
            ),
        },
    }
    REPORT.write_text(
        json.dumps(safe(report), ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "status": report["status"],
                "selected_gamma": selected["gamma"],
                "minimum_full_gain": selected["minimum_full_gain"],
                "minimum_R_gain": selected["minimum_R_gain"],
                "per_year": {
                    str(year): {
                        route: {
                            "gain": selected["years"][str(year)]["routes"][route]["gain"],
                            "ci_low": selected["years"][str(year)]["routes"][route][
                                "pitcher_cluster_95_ci"
                            ]["ci_low"],
                        }
                        for route in ("full", "R")
                    }
                    for year in YEARS
                },
                "semantic": report["semantic"],
                "oracle_full_gain": {
                    str(year): oracle[str(year)]["routes"]["full"]["gain"]
                    for year in YEARS
                },
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
