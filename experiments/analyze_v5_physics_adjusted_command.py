#!/usr/bin/env python3
"""Source-only audit of physics-adjusted completed-history command skill."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Import this before pandas/sklearn so LightGBM owns the first OpenMP runtime on
# Windows.  The model and feature constants are the preregistered teacher recipe.
from experiments.run_v5_privileged_trackman_teacher_feasibility import (  # noqa: E402
    FULL_FEATURES,
    TARGET,
    fit_predict,
)

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.model_selection import GroupKFold  # noqa: E402

from eda.run_structural_eda import linkage_section, state_code  # noqa: E402
from experiments.stats import paired_bootstrap_brier_ci  # noqa: E402


TRAIN = ROOT / "open" / "data" / "train.csv"
TRACKMAN = ROOT / "open" / "data" / "trackman_history.csv"
PREDICTIONS = ROOT / "experiments" / "results" / "predictions"
TEACHER_SCORES = ROOT / "experiments" / "results" / "v5_trackman_teacher_scores_dev.npz"
PREREG = ROOT / "experiments" / "params" / "v5_physics_adjusted_command_preregister.json"
REPORT = ROOT / "experiments" / "results" / "v5_physics_adjusted_command_source.json"
OOF_2019 = ROOT / "experiments" / "results" / "v5_physics_adjusted_command_2019_oof.npz"
PARENTS = {
    2020: PREDICTIONS / "v4_m3_c_backtest_2020_2020.npz",
    2021: PREDICTIONS / "v4_m3_c_backtest_2021_2021.npz",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def score(y: np.ndarray, prediction: np.ndarray) -> float:
    rate = float(np.mean(y))
    reference = rate * (1.0 - rate)
    brier = float(np.mean(np.square(np.asarray(prediction) - y)))
    return max(0.0, 100_000.0 * (1.0 - brier / reference))


def prepare_train_structure(train: pd.DataFrame) -> pd.DataFrame:
    """Reconstruct game/state keys on a season-prefix without later labels."""
    train = train.copy()
    lo = np.minimum(train["pitcher_team_id"], train["batter_team_id"]).to_numpy()
    hi = np.maximum(train["pitcher_team_id"], train["batter_team_id"]).to_numpy()
    key = np.stack(
        [
            train["season"],
            train["game_month"],
            train["game_dayofweek"],
            lo,
            hi,
        ],
        axis=1,
    )
    half = train["top_bottom"].eq("B").to_numpy(dtype=np.int64)
    progress = train["inning"].to_numpy() * 2 + half
    runs = train["run_total_before"].to_numpy()
    boundary = np.concatenate(
        [
            np.asarray([True]),
            np.any(key[1:] != key[:-1], axis=1)
            | (progress[1:] < progress[:-1])
            | (runs[1:] < runs[:-1]),
        ]
    )
    train["gid"] = boundary.cumsum() - 1
    train["half"] = half
    train["state"] = state_code(
        train["inning"].to_numpy(),
        half,
        train["balls_before"].to_numpy(),
        train["strikes_before"].to_numpy(),
        train["outs_before"].to_numpy(),
    )
    return train


def load_2019_joined() -> pd.DataFrame:
    """Link only the 2019 prefixes of the two official files."""
    with np.load(PARENTS[2020], allow_pickle=False) as archive:
        first_2020 = int(np.min(archive["row_index"]))
    train = pd.read_csv(TRAIN, nrows=first_2020, low_memory=False)
    if set(train["season"].unique()) != {2019}:
        raise ValueError("2019 train prefix boundary changed")
    train = prepare_train_structure(train)

    season_column = pd.read_csv(TRACKMAN, usecols=["season"], low_memory=False)[
        "season"
    ].to_numpy(dtype=np.int16)
    later = np.flatnonzero(season_column > 2019)
    if not len(later):
        raise ValueError("Could not locate the 2019 TrackMan prefix")
    first_2020_trackman = int(later[0])
    trackman = pd.read_csv(TRACKMAN, nrows=first_2020_trackman, low_memory=False)
    if set(trackman["season"].unique()) != {2019}:
        raise ValueError("2019 TrackMan prefix boundary changed")
    clean = trackman.loc[
        trackman["balls_before"].le(3)
        & trackman["strikes_before"].le(2)
        & trackman["outs_before"].le(2)
    ].copy()
    clean = clean.sort_values(
        ["trackman_game_id", "pitch_no"], kind="stable"
    ).reset_index(drop=True)
    codes, game_ids = pd.factorize(clean["trackman_game_id"])
    clean["g"] = codes
    clean["state"] = state_code(
        clean["inning"].to_numpy(),
        clean["top_bottom"].eq("Bottom").to_numpy(dtype=np.int64),
        clean["balls_before"].to_numpy(),
        clean["strikes_before"].to_numpy(),
        clean["outs_before"].to_numpy(),
    )
    clean["venue"] = clean["trackman_game_id"].str.split("-").str[1]
    _, joined = linkage_section(train, clean, len(game_ids))
    regular = joined.loc[
        joined["game_type"].eq("R") & joined[TARGET].notna()
    ].copy()
    if regular.empty or not regular["season"].eq(2019).all():
        raise ValueError("Empty or contaminated 2019 joined source")
    print(
        f"2019 official matched-R source: {len(regular):,} rows, "
        f"{regular['pitcher_id'].nunique():,} pitchers",
        flush=True,
    )
    return regular


def build_or_load_2019_oof(joined: pd.DataFrame) -> np.ndarray:
    row_id = joined["row_id"].astype(str).to_numpy(dtype=str)
    if OOF_2019.is_file():
        with np.load(OOF_2019, allow_pickle=False) as archive:
            if not np.array_equal(archive["row_id"].astype(str), row_id):
                raise ValueError("Cached 2019 OOF row alignment changed")
            prediction = archive["physics_teacher"].astype(np.float64)
        if not np.isfinite(prediction).all():
            raise ValueError("Cached 2019 OOF prediction is non-finite")
        print(f"loaded immutable 2019 OOF artifact {OOF_2019.name}", flush=True)
        return prediction

    groups = joined["pitcher_id"].to_numpy(dtype=np.int64)
    splitter = GroupKFold(n_splits=5)
    prediction = np.full(len(joined), np.nan, dtype=np.float64)
    placeholder = np.zeros(len(joined), dtype=np.int8)
    for fold, (fit_index, valid_index) in enumerate(
        splitter.split(joined, placeholder, groups), start=1
    ):
        overlap = np.intersect1d(groups[fit_index], groups[valid_index])
        if len(overlap):
            raise AssertionError("Pitcher leaked across the 2019 group fold")
        print(
            f"2019 physics OOF fold {fold}/5: "
            f"fit={len(fit_index):,} valid={len(valid_index):,}",
            flush=True,
        )
        prediction[valid_index] = fit_predict(
            joined.iloc[fit_index], joined.iloc[valid_index], FULL_FEATURES
        )
    if not np.isfinite(prediction).all():
        raise ValueError("Incomplete 2019 OOF prediction")
    OOF_2019.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        OOF_2019,
        row_id=row_id,
        physics_teacher=prediction.astype(np.float32),
    )
    print(f"wrote immutable source artifact {OOF_2019}", flush=True)
    return prediction


def load_source_frame() -> pd.DataFrame:
    with np.load(PARENTS[2021], allow_pickle=False) as archive:
        last_2021 = int(np.max(archive["row_index"]))
    columns = [
        "row_id",
        "season",
        "game_type",
        "pitcher_id",
        TARGET,
    ]
    frame = pd.read_csv(
        TRAIN,
        usecols=columns,
        nrows=last_2021 + 1,
        low_memory=False,
    )
    if set(frame["season"].unique()) != {2019, 2020, 2021}:
        raise ValueError("Source frame read a label after 2021")
    return frame


def load_parent(year: int, frame: pd.DataFrame) -> dict[str, np.ndarray]:
    with np.load(PARENTS[year], allow_pickle=False) as archive:
        result = {
            "row_index": archive["row_index"].astype(np.int64),
            "y": archive["y"].astype(np.float64),
            "cluster": archive["cluster"].astype(str),
            "parent": archive["catboost_outcome"].astype(np.float64),
        }
    view = frame.iloc[result["row_index"]]
    if not view["season"].eq(year).all():
        raise ValueError(f"{year}: parent season mismatch")
    if not np.array_equal(
        view[TARGET].to_numpy(dtype=np.int8), result["y"].astype(np.int8)
    ):
        raise ValueError(f"{year}: parent target mismatch")
    result["regular"] = view["game_type"].eq("R").to_numpy(dtype=bool)
    result["pitcher_id"] = view["pitcher_id"].to_numpy(dtype=np.int64)
    return result


def build_residual_rows(
    joined_2019: pd.DataFrame,
    prediction_2019: np.ndarray,
    frame: pd.DataFrame,
) -> pd.DataFrame:
    source_2019 = joined_2019[["row_id", "season", "pitcher_id", TARGET]].copy()
    source_2019["physics_teacher"] = prediction_2019

    with np.load(TEACHER_SCORES, allow_pickle=False) as archive:
        mask = archive["season"].astype(np.int16) == 2020
        source_2020 = pd.DataFrame(
            {
                "row_id": archive["row_id"][mask].astype(str),
                "season": archive["season"][mask].astype(np.int16),
                "pitcher_id_artifact": archive["pitcher_id"][mask].astype(np.int64),
                "physics_teacher": archive["physics_teacher"][mask].astype(
                    np.float64
                ),
            }
        )
    labels = frame.loc[
        frame["season"].eq(2020), ["row_id", "pitcher_id", "game_type", TARGET]
    ]
    source_2020 = source_2020.merge(
        labels, on="row_id", how="left", validate="one_to_one"
    )
    if source_2020[TARGET].isna().any():
        raise ValueError("Missing 2020 source labels for teacher scores")
    if not source_2020["game_type"].eq("R").all():
        raise ValueError("Teacher-score artifact unexpectedly contains F rows")
    if not np.array_equal(
        source_2020["pitcher_id"].to_numpy(dtype=np.int64),
        source_2020["pitcher_id_artifact"].to_numpy(dtype=np.int64),
    ):
        raise ValueError("2020 teacher-score pitcher alignment mismatch")
    source_2020 = source_2020[
        ["row_id", "season", "pitcher_id", TARGET, "physics_teacher"]
    ]
    residual = pd.concat([source_2019, source_2020], ignore_index=True)
    residual["raw_residual"] = (
        residual[TARGET].to_numpy(dtype=np.float64)
        - residual["physics_teacher"].to_numpy(dtype=np.float64)
    )
    residual["centered_residual"] = residual["raw_residual"] - residual.groupby(
        "season", observed=True
    )["raw_residual"].transform("mean")
    if not np.isfinite(residual["centered_residual"]).all():
        raise ValueError("Non-finite physics-adjusted residual")
    return residual


def profile_direction(
    residual: pd.DataFrame,
    year: int,
    pitcher_id: np.ndarray,
    window: str,
    k: float,
) -> np.ndarray:
    history = residual.loc[residual["season"].lt(year)]
    if window == "recent_completed_season":
        history = history.loc[history["season"].eq(int(history["season"].max()))]
    elif window != "all":
        raise ValueError(f"Unknown history window: {window}")
    grouped = history.groupby("pitcher_id", sort=False, observed=True)[
        "centered_residual"
    ].agg(["sum", "count"])
    profile = grouped["sum"] / (grouped["count"] + float(k))
    return (
        pd.Series(pitcher_id).map(profile).fillna(0.0).to_numpy(dtype=np.float64)
    )


def persistence_diagnostic(residual: pd.DataFrame) -> dict[str, Any]:
    table = residual.groupby(["season", "pitcher_id"], observed=True)[
        "centered_residual"
    ].agg(["mean", "count"])
    left = table.loc[2019].rename(columns=lambda name: f"{name}_2019")
    right = table.loc[2020].rename(columns=lambda name: f"{name}_2020")
    common = left.join(right, how="inner")
    eligible = common.loc[
        common["count_2019"].ge(50) & common["count_2020"].ge(50)
    ]
    return {
        "common_pitchers": int(len(common)),
        "eligible_pitchers_n_ge_50_both": int(len(eligible)),
        "unweighted_correlation": float(
            eligible["mean_2019"].corr(eligible["mean_2020"])
        )
        if len(eligible) >= 3
        else None,
    }


def selected_prediction(
    fold: dict[str, np.ndarray], direction: np.ndarray, gamma: float
) -> np.ndarray:
    prediction = fold["parent"].copy()
    regular = fold["regular"]
    prediction[regular] = np.clip(
        prediction[regular] + float(gamma) * direction[regular],
        1e-6,
        1.0 - 1e-6,
    )
    return prediction


def main() -> None:
    prereg = json.loads(PREREG.read_text(encoding="utf-8"))
    if prereg["status"] != "preregistered_before_source_metrics":
        raise ValueError("Unexpected preregistration status")
    if REPORT.exists():
        raise FileExistsError("Preserve the immutable source report instead of overwriting")

    joined_2019 = load_2019_joined()
    prediction_2019 = build_or_load_2019_oof(joined_2019)
    frame = load_source_frame()
    parents = {year: load_parent(year, frame) for year in (2020, 2021)}
    residual = build_residual_rows(joined_2019, prediction_2019, frame)
    print(
        "physics-adjusted source rows: "
        + ", ".join(
            f"{int(year)}={int(count):,}"
            for year, count in residual.groupby("season").size().items()
        ),
        flush=True,
    )

    directions: dict[tuple[int, str, int], np.ndarray] = {}
    windows = prereg["residual_profile"]["history_windows"]
    ks = [int(value) for value in prereg["source_grid"]["shrinkage_k"]]
    gammas = [float(value) for value in prereg["source_grid"]["gammas"]]
    for year in (2020, 2021):
        fold = parents[year]
        for window in windows:
            for k in ks:
                directions[(year, window, k)] = profile_direction(
                    residual,
                    year,
                    fold["pitcher_id"],
                    window,
                    float(k),
                )

    candidates: list[dict[str, Any]] = []
    for window in windows:
        for k in ks:
            for gamma in gammas:
                years: dict[str, Any] = {}
                for year in (2020, 2021):
                    fold = parents[year]
                    prediction = selected_prediction(
                        fold, directions[(year, window, k)], gamma
                    )
                    regular = fold["regular"]
                    years[str(year)] = {
                        "full_gain": score(fold["y"], prediction)
                        - score(fold["y"], fold["parent"]),
                        "r_gain": score(fold["y"][regular], prediction[regular])
                        - score(
                            fold["y"][regular], fold["parent"][regular]
                        ),
                    }
                candidates.append(
                    {
                        "window": window,
                        "k": int(k),
                        "gamma": float(gamma),
                        "min_full_gain": min(
                            value["full_gain"] for value in years.values()
                        ),
                        "min_r_gain": min(
                            value["r_gain"] for value in years.values()
                        ),
                        "mean_full_gain": float(
                            np.mean(
                                [value["full_gain"] for value in years.values()]
                            )
                        ),
                        "years": years,
                    }
                )
    selected = max(
        candidates,
        key=lambda item: (
            item["min_full_gain"],
            item["min_r_gain"],
            item["mean_full_gain"],
            -item["gamma"],
            item["k"],
            item["window"] == "all",
        ),
    )

    intervals: dict[str, Any] = {}
    prediction_artifacts: dict[str, str] = {}
    for offset, year in enumerate((2020, 2021)):
        fold = parents[year]
        direction = directions[(year, selected["window"], selected["k"])]
        prediction = selected_prediction(fold, direction, selected["gamma"])
        regular = fold["regular"]
        intervals[str(year)] = paired_bootstrap_brier_ci(
            fold["y"][regular],
            fold["parent"][regular],
            prediction[regular],
            iterations=2000,
            seed=52820 + offset,
            clusters=fold["cluster"][regular],
        )
        artifact = (
            PREDICTIONS / f"v5_physics_adjusted_command_source_{year}.npz"
        )
        if artifact.exists():
            raise FileExistsError(f"Preserve existing prediction artifact: {artifact}")
        np.savez_compressed(
            artifact,
            y=fold["y"].astype(np.float32),
            row_index=fold["row_index"],
            cluster=fold["cluster"],
            regular=regular,
            parent_prediction=fold["parent"].astype(np.float32),
            command_profile=direction.astype(np.float32),
            final_prediction=prediction.astype(np.float32),
        )
        prediction_artifacts[str(year)] = str(artifact.relative_to(ROOT))

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
            all(value["score_ci_low"] > 0.0 for value in intervals.values())
        ),
    }
    passed = bool(all(conditions.values()))
    diagnostics = {
        "source_rows": {
            str(int(year)): int(count)
            for year, count in residual.groupby("season").size().items()
        },
        "source_pitchers": {
            str(int(year)): int(count)
            for year, count in residual.groupby("season")["pitcher_id"]
            .nunique()
            .items()
        },
        "raw_residual_mean_by_season": {
            str(int(year)): float(value)
            for year, value in residual.groupby("season")["raw_residual"]
            .mean()
            .items()
        },
        "persistence": persistence_diagnostic(residual),
    }
    report = {
        "experiment_id": prereg["experiment_id"],
        "status": "passed_source_gate" if passed else "failed_source_gate",
        "preregister": str(PREREG.relative_to(ROOT)),
        "preregister_sha256": sha256(PREREG),
        "policy": {
            "official_data_only": True,
            "test_rows_read": False,
            "latest_label_season_used_for_metrics": 2021,
            "current_or_validation_trackman_at_inference": False,
            "row_independent": True,
            "automatic_submission": False,
        },
        "artifacts": {
            "teacher_scores": str(TEACHER_SCORES.relative_to(ROOT)),
            "teacher_scores_sha256": sha256(TEACHER_SCORES),
            "oof_2019": str(OOF_2019.relative_to(ROOT)),
            "oof_2019_sha256": sha256(OOF_2019),
            "predictions": prediction_artifacts,
        },
        "candidate_count": len(candidates),
        "diagnostics": diagnostics,
        "selected": selected,
        "selected_r_cluster_intervals": intervals,
        "conditions": conditions,
        "gate_pass": passed,
        "decision": "freeze before 2022" if passed else "close without 2022+",
        "top_candidates": sorted(
            candidates,
            key=lambda item: (item["min_full_gain"], item["min_r_gain"]),
            reverse=True,
        )[:10],
    }
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(selected, ensure_ascii=False, indent=2), flush=True)
    print(json.dumps(intervals, ensure_ascii=False, indent=2), flush=True)
    print(json.dumps(conditions, ensure_ascii=False, indent=2), flush=True)
    print(f"wrote {REPORT}", flush=True)


if __name__ == "__main__":
    main()
