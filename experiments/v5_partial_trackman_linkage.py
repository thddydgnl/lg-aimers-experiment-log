#!/usr/bin/env python3
"""Conservative target-free partial linkage to official TrackMan history."""

from __future__ import annotations

from collections import Counter
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from eda.run_structural_eda import state_code


ROOT = Path(__file__).resolve().parents[1]
TRAIN_PATH = ROOT / "open/data/train.csv"

MAIN_COLUMNS = [
    "row_id", "season", "game_month", "game_dayofweek", "inning",
    "top_bottom", "game_type", "balls_before", "strikes_before",
    "outs_before", "run_total_before", "pitcher_id", "batter_id",
    "pitcher_hand", "batter_hand", "pitcher_team_id", "batter_team_id",
]

RIGHT_DROP_COLUMNS = [
    "g", "state", "season", "game_month", "game_dayofweek", "inning",
    "top_bottom", "balls_before", "strikes_before", "outs_before",
    "pitcher_hand", "batter_hand",
]


def load_main_linkage_frame(path: Path = TRAIN_PATH) -> pd.DataFrame:
    """Load only target-free columns and reproduce the official game boundary."""
    frame = pd.read_csv(
        path, usecols=MAIN_COLUMNS, encoding="utf-8-sig",
        dtype={"row_id": "string"},
    )
    low = np.minimum(
        frame["pitcher_team_id"].to_numpy(),
        frame["batter_team_id"].to_numpy(),
    )
    high = np.maximum(
        frame["pitcher_team_id"].to_numpy(),
        frame["batter_team_id"].to_numpy(),
    )
    key = np.stack(
        [
            frame["season"].to_numpy(), frame["game_month"].to_numpy(),
            frame["game_dayofweek"].to_numpy(), low, high,
        ],
        axis=1,
    )
    half = frame["top_bottom"].eq("B").to_numpy(dtype=np.int8)
    progress = frame["inning"].to_numpy() * 2 + half
    runs = frame["run_total_before"].to_numpy()
    boundary = np.concatenate(
        [
            [True],
            np.any(key[1:] != key[:-1], axis=1)
            | (progress[1:] < progress[:-1])
            | (runs[1:] < runs[:-1]),
        ]
    )
    frame["gid"] = boundary.cumsum() - 1
    frame["half"] = half
    frame["state"] = state_code(
        frame["inning"].to_numpy(), half,
        frame["balls_before"].to_numpy(),
        frame["strikes_before"].to_numpy(),
        frame["outs_before"].to_numpy(),
    )
    return frame


def _recover_map(
    rows: pd.DataFrame,
    left: str,
    right: str,
    minimum_purity: float,
    injective: bool,
) -> tuple[dict[Any, Any], dict[str, Any]]:
    pairs = rows[[left, right]].dropna()
    counts = (
        pairs.groupby([left, right], sort=False, observed=True)
        .size().rename("n").reset_index()
    )
    if counts.empty:
        return {}, {
            "entities": 0, "minimum_purity": None,
            "mean_purity": None, "injective": True,
        }
    totals = counts.groupby(left, observed=True)["n"].transform("sum")
    counts["purity"] = counts["n"] / totals
    best = counts.sort_values(
        [left, "n", "purity"], ascending=[True, False, False], kind="stable"
    ).drop_duplicates(left)
    best = best.loc[best["purity"].ge(minimum_purity)].copy()
    if injective:
        best = best.sort_values(
            [right, "n", "purity"],
            ascending=[True, False, False], kind="stable",
        ).drop_duplicates(right)
    mapping = dict(zip(best[left], best[right]))
    return mapping, {
        "entities": int(len(best)),
        "minimum_purity": float(best["purity"].min()) if len(best) else None,
        "mean_purity": float(best["purity"].mean()) if len(best) else None,
        "injective": bool(len(set(mapping.values())) == len(mapping)),
    }


def _jaccard(left: set[Any], right: set[Any]) -> float:
    return float(len(left & right) / max(1, len(left | right)))


def _counter_dice(left: Counter, right: Counter) -> float:
    overlap = sum((left & right).values())
    return float(2.0 * overlap / max(1, sum(left.values()) + sum(right.values())))


def _main_descriptor(
    rows: pd.DataFrame,
    pitcher_map: dict[Any, Any],
    batter_map: dict[Any, Any],
    team_map: dict[Any, Any],
) -> dict[str, Any] | None:
    raw_teams = set(rows["pitcher_team_id"]) | set(rows["batter_team_id"])
    teams = {team_map.get(value) for value in raw_teams}
    if None in teams or len(teams) != 2:
        return None
    sequence: list[tuple[Any, Any, Any]] = []
    mapped_sequence: list[tuple[int, Any, Any]] = []
    pitchers: set[Any] = set()
    batters: set[Any] = set()
    for row in rows.itertuples(index=False):
        pitcher = pitcher_map.get(row.pitcher_id)
        batter = batter_map.get(row.batter_id)
        if pitcher is None or batter is None:
            sequence.append((int(row.state), f"p?{row.pitcher_id}", f"b?{row.batter_id}"))
            continue
        key = (int(row.state), pitcher, batter)
        sequence.append(key)
        mapped_sequence.append(key)
        pitchers.add(pitcher)
        batters.add(batter)
    if not mapped_sequence:
        return None
    return {
        "key": (
            int(rows["season"].iloc[0]), int(rows["game_month"].iloc[0]),
            int(rows["game_dayofweek"].iloc[0]), tuple(sorted(teams)),
        ),
        "sequence": sequence,
        "counter": Counter(mapped_sequence),
        "pitchers": pitchers,
        "batters": batters,
        "mapped_rows": len(mapped_sequence),
    }


def _trackman_descriptor(rows: pd.DataFrame) -> dict[str, Any] | None:
    teams = set(rows["pitcher_team"].astype(str)) | set(
        rows["batter_team"].astype(str)
    )
    if len(teams) != 2:
        return None
    sequence = [
        (int(row.state), row.pitcher_trackman_id, row.batter_trackman_id)
        for row in rows.itertuples(index=False)
    ]
    return {
        "key": (
            int(rows["season"].iloc[0]), int(rows["game_month"].iloc[0]),
            int(rows["game_dayofweek"].iloc[0]), tuple(sorted(teams)),
        ),
        "sequence": sequence,
        "counter": Counter(sequence),
        "pitchers": set(rows["pitcher_trackman_id"].dropna()),
        "batters": set(rows["batter_trackman_id"].dropna()),
    }


def _candidate_table(
    main_descriptors: dict[int, dict[str, Any]],
    trackman_descriptors: dict[int, dict[str, Any]],
) -> pd.DataFrame:
    trackman_by_key: dict[tuple[Any, ...], list[int]] = {}
    for game, descriptor in trackman_descriptors.items():
        trackman_by_key.setdefault(descriptor["key"], []).append(game)
    records: list[dict[str, Any]] = []
    for gid, main in main_descriptors.items():
        for game in trackman_by_key.get(main["key"], []):
            trackman = trackman_descriptors[game]
            dice = _counter_dice(main["counter"], trackman["counter"])
            pitcher_jaccard = _jaccard(main["pitchers"], trackman["pitchers"])
            batter_jaccard = _jaccard(main["batters"], trackman["batters"])
            score = 0.75 * dice + 0.15 * pitcher_jaccard + 0.10 * batter_jaccard
            records.append(
                {
                    "gid": gid, "g": game, "score": score,
                    "tuple_dice": dice,
                    "pitcher_jaccard": pitcher_jaccard,
                    "batter_jaccard": batter_jaccard,
                }
            )
    if not records:
        return pd.DataFrame(
            columns=[
                "gid", "g", "score", "tuple_dice", "pitcher_jaccard",
                "batter_jaccard", "forward_rank", "reverse_rank",
                "forward_margin", "reverse_margin",
            ]
        )
    result = pd.DataFrame(records)
    result["forward_rank"] = result.groupby("gid")["score"].rank(
        method="first", ascending=False
    ).astype(np.int16)
    result["reverse_rank"] = result.groupby("g")["score"].rank(
        method="first", ascending=False
    ).astype(np.int16)

    def margins(key: str) -> dict[Any, float]:
        output: dict[Any, float] = {}
        for value, group in result.groupby(key, sort=False):
            ordered = np.sort(group["score"].to_numpy(dtype=np.float64))[::-1]
            output[value] = float(
                ordered[0] - (ordered[1] if len(ordered) > 1 else 0.0)
            )
        return output

    result["forward_margin"] = result["gid"].map(margins("gid"))
    result["reverse_margin"] = result["g"].map(margins("g"))
    return result


def build_augmented_trackman_linkage(
    main: pd.DataFrame,
    exact_joined: pd.DataFrame,
    raw_trackman: pd.DataFrame,
    allowed_seasons: list[int],
    minimum_purity: float = 0.99,
    minimum_score: float = 0.90,
    minimum_tuple_dice: float = 0.90,
    minimum_margin: float = 0.05,
    minimum_sequence_ratio: float = 0.90,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Return exact history plus conservative partial row alignments."""
    allowed = sorted(int(value) for value in allowed_seasons)
    exact = exact_joined.loc[exact_joined["season"].isin(allowed)].copy()
    main_history = main.loc[
        main["season"].isin(allowed) & main["game_type"].eq("R")
    ].copy()
    trackman_history = raw_trackman.loc[
        raw_trackman["season"].isin(allowed)
    ].copy()

    pitcher_map, pitcher_meta = _recover_map(
        exact, "pitcher_id", "pitcher_trackman_id", minimum_purity, True
    )
    batter_map, batter_meta = _recover_map(
        exact, "batter_id", "batter_trackman_id", minimum_purity, True
    )
    # Futures/minor games legitimately use several TrackMan club aliases for
    # one parent organization.  Partial recovery is deliberately R-only, so
    # its team key must be learned from exact regular-season games as well.
    exact_regular = exact.loc[exact["game_type"].eq("R")]
    pitcher_team_map, pitcher_team_meta = _recover_map(
        exact_regular, "pitcher_team_id", "pitcher_team", minimum_purity, True
    )
    batter_team_map, batter_team_meta = _recover_map(
        exact_regular, "batter_team_id", "batter_team", minimum_purity, True
    )
    team_map = dict(pitcher_team_map)
    for key, value in batter_team_map.items():
        if key in team_map and team_map[key] != value:
            raise ValueError(f"inconsistent team map for {key}: {team_map[key]}/{value}")
        team_map[key] = value

    main_groups = {
        int(gid): rows.reset_index(drop=True)
        for gid, rows in main_history.groupby("gid", sort=False)
    }
    trackman_groups = {
        int(game): rows.sort_values("pitch_no", kind="stable").reset_index(drop=True)
        for game, rows in trackman_history.groupby("g", sort=False)
    }
    main_descriptors = {
        gid: descriptor
        for gid, rows in main_groups.items()
        if (
            descriptor := _main_descriptor(
                rows, pitcher_map, batter_map, team_map
            )
        ) is not None
    }
    trackman_descriptors = {
        game: descriptor
        for game, rows in trackman_groups.items()
        if (descriptor := _trackman_descriptor(rows)) is not None
    }
    candidates = _candidate_table(main_descriptors, trackman_descriptors)
    true_pairs = exact[["gid", "g"]].drop_duplicates()
    true_map = dict(zip(true_pairs["gid"], true_pairs["g"]))
    locked = candidates.loc[
        candidates["forward_rank"].eq(1)
        & candidates["reverse_rank"].eq(1)
        & candidates["score"].ge(minimum_score)
        & candidates["tuple_dice"].ge(minimum_tuple_dice)
        & candidates["forward_margin"].ge(minimum_margin)
        & candidates["reverse_margin"].ge(minimum_margin)
    ].copy()
    calibration = locked.loc[locked["gid"].isin(true_map)].copy()
    calibration_correct = calibration.apply(
        lambda row: true_map[int(row["gid"])] == int(row["g"]), axis=1
    ) if len(calibration) else pd.Series(dtype=bool)

    exact_gids = set(int(value) for value in true_pairs["gid"])
    exact_games = set(int(value) for value in true_pairs["g"])
    recovered = locked.loc[
        ~locked["gid"].isin(exact_gids) & ~locked["g"].isin(exact_games)
    ].copy()
    partial_parts: list[pd.DataFrame] = []
    alignment_records: list[dict[str, Any]] = []
    for pair in recovered.itertuples(index=False):
        gid = int(pair.gid)
        game = int(pair.g)
        left = main_groups[gid]
        right = trackman_groups[game]
        left_sequence = main_descriptors[gid]["sequence"]
        right_sequence = trackman_descriptors[game]["sequence"]
        matcher = SequenceMatcher(
            None, left_sequence, right_sequence, autojunk=False
        )
        ratio = float(matcher.ratio())
        if ratio < minimum_sequence_ratio:
            continue
        left_positions: list[int] = []
        right_positions: list[int] = []
        for block in matcher.get_matching_blocks():
            if block.size <= 0:
                continue
            left_positions.extend(range(block.a, block.a + block.size))
            right_positions.extend(range(block.b, block.b + block.size))
        if not left_positions:
            continue
        left_aligned = left.iloc[left_positions].reset_index(drop=True)
        right_aligned = right.iloc[right_positions].reset_index(drop=True)
        if len(left_aligned) != len(right_aligned):
            raise AssertionError("partial alignment length mismatch")
        right_payload = right_aligned.drop(
            columns=[
                column for column in RIGHT_DROP_COLUMNS
                if column in right_aligned.columns
            ]
        )
        partial = pd.concat([left_aligned, right_payload], axis=1)
        partial_parts.append(partial)
        alignment_records.append(
            {
                "gid": gid, "g": game, "score": float(pair.score),
                "tuple_dice": float(pair.tuple_dice),
                "sequence_ratio": ratio,
                "main_rows": int(len(left)),
                "trackman_rows": int(len(right)),
                "aligned_rows": int(len(partial)),
            }
        )
    partial = (
        pd.concat(partial_parts, ignore_index=True, sort=False)
        if partial_parts else exact.iloc[:0].copy()
    )
    common_columns = list(dict.fromkeys([*exact.columns, *partial.columns]))
    augmented = pd.concat(
        [exact.reindex(columns=common_columns), partial.reindex(columns=common_columns)],
        ignore_index=True,
    )
    duplicate_row_ids = int(augmented["row_id"].astype(str).duplicated().sum())
    duplicate_trackman_ids = int(augmented["trackman_id"].duplicated().sum())
    if duplicate_row_ids or duplicate_trackman_ids:
        raise AssertionError(
            f"non-unique augmented rows: main={duplicate_row_ids}, "
            f"trackman={duplicate_trackman_ids}"
        )
    sequence_ratios = [row["sequence_ratio"] for row in alignment_records]
    metadata = {
        "allowed_history_seasons": allowed,
        "identity": {
            "pitcher": pitcher_meta, "batter": batter_meta,
            "pitcher_team": pitcher_team_meta,
            "batter_team": batter_team_meta,
        },
        "thresholds": {
            "minimum_purity": minimum_purity,
            "minimum_score": minimum_score,
            "minimum_tuple_dice": minimum_tuple_dice,
            "minimum_forward_margin": minimum_margin,
            "minimum_reverse_margin": minimum_margin,
            "minimum_sequence_ratio": minimum_sequence_ratio,
        },
        "candidate_pairs": int(len(candidates)),
        "locked_mutual_pairs": int(len(locked)),
        "known_exact_calibration": {
            "passing_games": int(len(calibration)),
            "correct_games": int(calibration_correct.sum()),
            "precision": float(calibration_correct.mean())
            if len(calibration_correct) else None,
        },
        "exact_games": int(len(true_pairs)),
        "exact_rows": int(len(exact)),
        "partial_games": int(len(alignment_records)),
        "partial_aligned_rows": int(len(partial)),
        "augmented_rows": int(len(augmented)),
        "joined_row_expansion_factor": float(len(augmented) / max(1, len(exact))),
        "minimum_partial_sequence_ratio": float(min(sequence_ratios))
        if sequence_ratios else None,
        "duplicate_main_row_ids": duplicate_row_ids,
        "duplicate_trackman_ids": duplicate_trackman_ids,
        "main_regular_games_considered": int(len(main_groups)),
        "main_games_with_complete_candidate_key": int(len(main_descriptors)),
        "trackman_games_with_complete_candidate_key": int(len(trackman_descriptors)),
        "matching_columns": [
            "season", "game_month", "game_dayofweek", "team pair",
            "state", "mapped pitcher", "mapped batter",
        ],
        "current_pitch_type_or_physics_used_as_match_key": False,
        "control_target_used_for_matching": False,
        "current_validation_trackman_used": False,
        "row_alignment_records": alignment_records,
    }
    return augmented, metadata
