"""Cutoff-correct profile features distilled from historical TrackMan teachers."""

from __future__ import annotations

import hashlib
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SCORES = ROOT / "experiments/results/v5_trackman_teacher_scores_dev.npz"
GROUPS = ("fastball", "breaking", "offspeed", "other")
LEVEL_K = 200.0
RECENT_K = 100.0
GROUP_K = 100.0
SELECTION_K = 100.0
FEATURE_COLUMNS = [
    "e80_teacher_full_all",
    "e80_teacher_full_recent",
    "e80_teacher_full_context",
    "e80_teacher_delta_all",
    "e80_teacher_delta_recent",
    "e80_teacher_delta_context",
    "e80_teacher_profile_n_log",
    "e80_teacher_recent_n_log",
    "e80_teacher_unseen",
]


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


@lru_cache(maxsize=1)
def load_teacher_scores() -> pd.DataFrame:
    if not SCORES.is_file():
        raise FileNotFoundError(
            f"Missing {SCORES}; run build_v5_trackman_teacher_scores.py first"
        )
    with np.load(SCORES, allow_pickle=False) as archive:
        frame = pd.DataFrame({key: np.asarray(archive[key]) for key in archive.files})
    frame["pitch_type_group"] = (
        frame["pitch_type_group"].astype("string").where(
            frame["pitch_type_group"].isin(GROUPS), "other"
        )
    )
    frame["teacher_full_centered"] = (
        frame["physics_teacher"]
        - frame.groupby("season", observed=True)["physics_teacher"].transform("mean")
    )
    frame["teacher_delta_centered"] = (
        frame["physics_teacher"] - frame["control_teacher"]
    )
    frame["teacher_delta_centered"] -= frame.groupby(
        "season", observed=True
    )["teacher_delta_centered"].transform("mean")
    return frame


def _empty_state() -> dict[str, Any]:
    return {
        "pitcher": pd.DataFrame(),
        "context_mix": pd.DataFrame(),
        "source_seasons": [],
        "source_rows": 0,
    }


def teacher_profile_table(rows: pd.DataFrame) -> dict[str, Any]:
    if rows.empty:
        return _empty_state()
    work = rows.copy()
    signals = {
        "full": "teacher_full_centered",
        "delta": "teacher_delta_centered",
    }
    pitcher_index = pd.Index(
        sorted(work["pitcher_id"].unique()), name="pitcher_id"
    )
    result = pd.DataFrame(index=pitcher_index)
    pitcher_count = work.groupby("pitcher_id", observed=True).size().reindex(
        pitcher_index, fill_value=0
    ).astype(np.float64)
    result["e80_teacher_profile_n_log"] = np.log1p(pitcher_count)
    latest_season = int(work["season"].max())
    recent_rows = work.loc[work["season"].eq(latest_season)]
    recent_count = recent_rows.groupby("pitcher_id", observed=True).size().reindex(
        pitcher_index, fill_value=0
    ).astype(np.float64)
    result["e80_teacher_recent_n_log"] = np.log1p(recent_count)

    for short, column in signals.items():
        total = work.groupby("pitcher_id", observed=True)[column].sum().reindex(
            pitcher_index, fill_value=0.0
        )
        level = total / (pitcher_count + LEVEL_K)
        result[f"e80_teacher_{short}_all"] = level
        recent_total = recent_rows.groupby("pitcher_id", observed=True)[column].sum().reindex(
            pitcher_index, fill_value=0.0
        )
        recent = (recent_total + RECENT_K * level) / (recent_count + RECENT_K)
        result[f"e80_teacher_{short}_recent"] = recent

    group_count = (
        work.groupby(["pitcher_id", "pitch_type_group"], observed=True)
        .size()
        .unstack(fill_value=0)
        .reindex(index=pitcher_index, columns=GROUPS, fill_value=0)
        .astype(np.float64)
    )
    global_mix = (
        work["pitch_type_group"].value_counts(normalize=True).reindex(GROUPS).fillna(0.0)
    )
    overall_mix = pd.DataFrame(index=pitcher_index)
    for group in GROUPS:
        overall_mix[f"p_{group}"] = (
            group_count[group] + SELECTION_K * float(global_mix[group])
        ) / (pitcher_count + SELECTION_K)
    for short, column in signals.items():
        group_sum = (
            work.groupby(["pitcher_id", "pitch_type_group"], observed=True)[column]
            .sum()
            .unstack(fill_value=0.0)
            .reindex(index=pitcher_index, columns=GROUPS, fill_value=0.0)
        )
        for group in GROUPS:
            result[f"v_{short}_{group}"] = (
                group_sum[group]
                + GROUP_K * result[f"e80_teacher_{short}_all"]
            ) / (group_count[group] + GROUP_K)

    context_keys = ["pitcher_id", "balls_before", "strikes_before"]
    context_count = (
        work.groupby([*context_keys, "pitch_type_group"], observed=True)
        .size()
        .unstack(fill_value=0)
        .reindex(columns=GROUPS, fill_value=0)
        .astype(np.float64)
    )
    context_total = context_count.sum(axis=1).to_numpy(dtype=np.float64)
    context_pitchers = context_count.index.get_level_values("pitcher_id")
    prior_mix = overall_mix.reindex(context_pitchers).to_numpy(dtype=np.float64)
    probabilities = (
        context_count.to_numpy(dtype=np.float64) + SELECTION_K * prior_mix
    ) / (context_total[:, None] + SELECTION_K)
    context_mix = pd.DataFrame(
        probabilities,
        index=context_count.index,
        columns=[f"p_{group}" for group in GROUPS],
    )
    result = result.join(overall_mix)
    result["e80_teacher_unseen"] = 0.0
    return {
        "pitcher": result,
        "context_mix": context_mix,
        "source_seasons": sorted(int(value) for value in work["season"].unique()),
        "source_rows": int(len(work)),
    }


def teacher_profile_states_before_each_season(
    _joined: pd.DataFrame,
    seasons: list[int],
    window: int | None = None,
) -> tuple[dict[int, dict[str, Any]], dict[str, Any]]:
    scores = load_teacher_scores()
    before: dict[int, dict[str, Any]] = {}
    for season in sorted(seasons):
        subset = scores.loc[scores["season"].lt(season)]
        if window is not None:
            subset = subset.loc[subset["season"].ge(season - window)]
        before[season] = teacher_profile_table(subset)
    if seasons:
        cutoff = max(seasons) + 1
        subset = scores.loc[scores["season"].lt(cutoff)]
        if window is not None:
            subset = subset.loc[subset["season"].ge(cutoff - window)]
        final = teacher_profile_table(subset)
    else:
        final = _empty_state()
    return before, final


def build_teacher_profile_features(
    frame: pd.DataFrame,
    profiles_before: dict[int, dict[str, Any]],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    values = np.zeros((len(frame), len(FEATURE_COLUMNS)), dtype=np.float32)
    unseen_index = FEATURE_COLUMNS.index("e80_teacher_unseen")
    values[:, unseen_index] = 1.0
    seasons = frame["season"].to_numpy(dtype=np.int16, copy=False)
    for season in sorted(set(int(value) for value in seasons)):
        mask = seasons == season
        indices = np.flatnonzero(mask)
        state = profiles_before.get(season, _empty_state())
        pitcher_table = state["pitcher"]
        if pitcher_table.empty:
            continue
        query = frame.loc[mask, ["pitcher_id", "balls_before", "strikes_before"]]
        pitchers = query["pitcher_id"].to_numpy(dtype=np.int64)
        lookup = pitcher_table.reindex(pitchers)
        known = lookup["e80_teacher_unseen"].notna().to_numpy(dtype=bool)
        for name in (
            "e80_teacher_full_all",
            "e80_teacher_full_recent",
            "e80_teacher_delta_all",
            "e80_teacher_delta_recent",
            "e80_teacher_profile_n_log",
            "e80_teacher_recent_n_log",
        ):
            column_index = FEATURE_COLUMNS.index(name)
            column = lookup[name].fillna(0.0).to_numpy(dtype=np.float64)
            values[indices, column_index] = column.astype(np.float32)

        keys = pd.MultiIndex.from_arrays(
            [
                pitchers,
                query["balls_before"].to_numpy(dtype=np.int16),
                query["strikes_before"].to_numpy(dtype=np.int16),
            ],
            names=["pitcher_id", "balls_before", "strikes_before"],
        )
        context_mix = state["context_mix"].reindex(keys)
        overall_mix = lookup[[f"p_{group}" for group in GROUPS]]
        probabilities = context_mix.to_numpy(dtype=np.float64)
        fallback = overall_mix.to_numpy(dtype=np.float64)
        probabilities = np.where(np.isfinite(probabilities), probabilities, fallback)
        probabilities = np.nan_to_num(probabilities, nan=0.0)
        for short in ("full", "delta"):
            group_values = lookup[
                [f"v_{short}_{group}" for group in GROUPS]
            ].to_numpy(dtype=np.float64)
            expected = np.sum(probabilities * np.nan_to_num(group_values), axis=1)
            values[
                indices, FEATURE_COLUMNS.index(f"e80_teacher_{short}_context")
            ] = expected.astype(np.float32)
        values[indices[known], unseen_index] = 0.0

    result = pd.DataFrame(values, columns=FEATURE_COLUMNS, index=frame.index)
    source_years = sorted(
        {
            source
            for state in profiles_before.values()
            for source in state.get("source_seasons", [])
        }
    )
    return result, {
        "feature_count": len(FEATURE_COLUMNS),
        "features": FEATURE_COLUMNS,
        "known_rows": int((result["e80_teacher_unseen"] == 0).sum()),
        "unseen_rows": int((result["e80_teacher_unseen"] > 0).sum()),
        "source_years": source_years,
        "score_artifact": str(SCORES.relative_to(ROOT)),
        "score_artifact_sha256": file_sha256(SCORES),
        "cutoff": "out-of-time teacher-scored completed seasons strictly before row season",
        "current_or_validation_trackman_used": False,
    }
