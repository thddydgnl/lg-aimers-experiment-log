"""Row-independent inference entry point for the sparse V3 outcome ensemble."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd


ID_COL = "row_id"
TARGET_COL = "control_success"
DATA_DIR = Path("data")
MODEL_DIR = Path("model")
OUTPUT_PATH = Path("output") / "submission.csv"

BASE_FEATURES = [
    "season", "game_month", "game_dayofweek", "inning", "top_bottom",
    "game_type", "balls_before", "strikes_before", "outs_before",
    "run_top_before", "run_bot_before", "run_total_before", "score_diff_home",
    "score_diff_pitcher_team", "runner_on_1b", "runner_on_2b", "runner_on_3b",
    "num_runners_on", "base_state", "home_win_expectancy",
    "away_win_expectancy", "li", "pitcher_id", "batter_id", "pitcher_hand",
    "batter_hand", "pitcher_team_id", "batter_team_id", "asof_pitcher_n",
    "asof_pitcher_success_rate", "asof_pitcher_reverse_rate",
    "asof_pitcher_middle_rate", "asof_pitcher_ball_rate",
    "asof_pitcher_strike_rate", "asof_pitcher_prev1_game_success_rate",
    "asof_pitcher_prev3_game_success_rate", "asof_pitcher_prev5_game_success_rate",
    "asof_pitcher_prev1_game_middle_rate", "asof_pitcher_prev3_game_middle_rate",
    "asof_pitcher_prev5_game_middle_rate", "asof_batter_n",
    "asof_batter_success_rate", "asof_batter_middle_rate",
    "asof_pitcher_pitchmix_n", "asof_pitcher_fastball_rate",
    "asof_pitcher_breaking_rate", "asof_pitcher_offspeed_rate",
]

COMPONENT_COLUMNS = {
    "reverse": "asof_pitcher_reverse_rate",
    "middle": "asof_pitcher_middle_rate",
    "ball": "asof_pitcher_ball_rate",
    "strike": "asof_pitcher_strike_rate",
}

HISTORY_SPECS = {
    "history_month_rate": (("game_type", "game_month"), "e54_month"),
    "history_count_rate": (
        ("game_type", "balls_before", "strikes_before"), "e55_count"
    ),
    "history_hand_rate": (
        ("game_type", "pitcher_hand", "batter_hand"), "e56_hand"
    ),
    "history_inning_rate": (("game_type", "inning"), "e57_inning"),
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verified_path(name: str, expected: str) -> Path:
    path = MODEL_DIR / name
    actual = sha256_file(path)
    if actual.lower() != expected.lower():
        raise ValueError(f"SHA-256 mismatch for {name}: {actual} != {expected}")
    return path


def load_assets() -> tuple[dict, dict]:
    manifest = json.loads((MODEL_DIR / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("schema_version") != 3:
        raise ValueError("V3 manifest required")
    if manifest.get("row_independent_inference") is not True:
        raise ValueError("Manifest does not certify row-independent inference")
    models = manifest.get("models", [])
    if not models or abs(sum(float(item["weight"]) for item in models) - 1.0) > 1e-9:
        raise ValueError("Invalid model weights")
    state_path = verified_path(
        manifest["state_file"], manifest["state_sha256"]
    )
    return manifest, joblib.load(state_path)


def numeric_ids(series: pd.Series) -> np.ndarray:
    return pd.to_numeric(series, errors="coerce").fillna(-1).to_numpy(dtype=np.int64)


def frozen_rows(entities: np.ndarray, state: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rows = [state.get(int(entity)) for entity in entities]
    n_end = np.asarray([0 if row is None else int(row[0]) for row in rows], dtype=np.int64)
    s_end = np.asarray([0 if row is None else int(row[1]) for row in rows], dtype=np.int64)
    unseen = np.asarray([1 if row is None else 0 for row in rows], dtype=np.int8)
    return n_end, s_end, unseen


def build_entity_success(
    frame: pd.DataFrame,
    state: dict,
    prior: float,
    k: float,
    entity_column: str,
    n_column: str,
    rate_column: str,
    prefix: str,
) -> pd.DataFrame:
    entities = numeric_ids(frame[entity_column])
    n_end, s_end, unseen = frozen_rows(entities, state)
    n_asof = pd.to_numeric(frame[n_column], errors="coerce").fillna(0).to_numpy(
        dtype=np.int64
    )
    career = pd.to_numeric(frame[rate_column], errors="coerce").fillna(prior).to_numpy(
        dtype=np.float64
    )
    s_asof = np.rint(career * n_asof).astype(np.int64)
    n_delta = n_asof - n_end
    s_delta = s_asof - s_end
    invalid = (n_delta < 0) | (s_delta < 0) | (s_delta > n_delta)
    safe_n = np.where(invalid, 0, n_delta)
    safe_s = np.where(invalid, 0, s_delta)
    rate = (safe_s + k * prior) / (safe_n + k)
    if prefix == "e14":
        names = {
            "n": "e14_n_season", "s": "e14_s_season",
            "log": "e14_log_n_season", "rate": "e14_rate_season",
            "delta": "e14_rate_delta", "zero": "e14_n_season_zero",
            "unseen": "e14_pitcher_unseen", "invalid": "e14_counter_invalid",
        }
    else:
        names = {
            "n": f"{prefix}_n_season", "s": f"{prefix}_s_season",
            "log": f"{prefix}_log_n_season", "rate": f"{prefix}_rate_season",
            "delta": f"{prefix}_rate_delta", "zero": f"{prefix}_n_season_zero",
            "unseen": f"{prefix}_unseen", "invalid": f"{prefix}_counter_invalid",
        }
    return pd.DataFrame(
        {
            names["n"]: safe_n.astype(np.int32),
            names["s"]: safe_s.astype(np.int32),
            names["log"]: np.log1p(safe_n).astype(np.float32),
            names["rate"]: rate.astype(np.float32),
            names["delta"]: (rate - career).astype(np.float32),
            names["zero"]: (safe_n == 0).astype(np.int8),
            names["unseen"]: unseen,
            names["invalid"]: invalid.astype(np.int8),
        },
        index=frame.index,
    )


def build_platoon(frame: pd.DataFrame, state: dict) -> pd.DataFrame:
    prior = float(state["prior"])
    pitcher = pd.to_numeric(frame["pitcher_id"], errors="coerce").astype("Int64")
    pitcher_key = pitcher.astype("string").fillna("__missing__")
    batter_key = pd.to_numeric(frame["batter_hand"], errors="coerce").astype(
        "Int64"
    ).astype("string").fillna("__missing__")
    cell = pitcher_key.astype(str) + "|" + batter_key.astype(str)
    rate = pitcher_key.map(state["pitcher_rate"])
    delta = cell.map(state["platoon_delta"])
    count = cell.map(state["platoon_n"])
    return pd.DataFrame(
        {
            "e30_pitcher_rate": pd.to_numeric(rate, errors="coerce").fillna(prior).to_numpy(np.float32),
            "e30_platoon_delta": pd.to_numeric(delta, errors="coerce").fillna(0.0).to_numpy(np.float32),
            "e30_platoon_n_log": np.log1p(
                pd.to_numeric(count, errors="coerce").fillna(0.0).to_numpy(np.float64)
            ).astype(np.float32),
            "e30_platoon_unseen": delta.isna().to_numpy(dtype=np.int8),
        },
        index=frame.index,
    )


def build_hand_matchup(frame: pd.DataFrame) -> pd.DataFrame:
    pitcher = pd.to_numeric(frame["pitcher_hand"], errors="coerce").astype("Int64")
    batter = pd.to_numeric(frame["batter_hand"], errors="coerce").astype("Int64")
    cell = pitcher.astype("string").fillna("__missing__") + "|" + batter.astype(
        "string"
    ).fillna("__missing__")
    same = (pitcher == batter).fillna(False).astype(np.int8)
    return pd.DataFrame(
        {"c36_pitcher_hand_batter_hand": cell, "c36_same_hand": same.to_numpy()},
        index=frame.index,
    )


def build_e14_interactions(frame: pd.DataFrame, e14: pd.DataFrame) -> pd.DataFrame:
    pitcher = pd.to_numeric(frame["pitcher_hand"], errors="coerce").fillna(-1).to_numpy()
    batter = pd.to_numeric(frame["batter_hand"], errors="coerce").fillna(-1).to_numpy()
    balls = pd.to_numeric(frame["balls_before"], errors="coerce").fillna(-1).to_numpy()
    strikes = pd.to_numeric(frame["strikes_before"], errors="coerce").fillna(-1).to_numpy()
    types = frame["game_type"].astype(str).to_numpy()
    rate = e14["e14_rate_season"].to_numpy(dtype=np.float32, copy=False)
    values: dict[str, np.ndarray] = {}
    for pitcher_hand in (1, 2):
        for batter_hand in (1, 2):
            mask = (pitcher == pitcher_hand) & (batter == batter_hand)
            values[f"c39_e14_ph{pitcher_hand}_bh{batter_hand}"] = rate * mask.astype(np.float32)
    for game_type in (None, "R", "F"):
        for ball_count in range(4):
            for strike_count in range(3):
                mask = (balls == ball_count) & (strikes == strike_count)
                if game_type is None:
                    prefix = "c48_e14"
                else:
                    mask &= types == game_type
                    prefix = f"c48_e14_{game_type.lower()}"
                values[f"{prefix}_b{ball_count}s{strike_count}"] = rate * mask.astype(np.float32)
    return pd.DataFrame(values, index=frame.index)


def build_generic_component(
    frame: pd.DataFrame,
    state: dict,
    priors: dict,
    entity_column: str,
    n_column: str,
    component_columns: dict[str, str],
    prefix: str,
    k: float,
) -> pd.DataFrame:
    entities = numeric_ids(frame[entity_column])
    rows = [state.get(int(entity)) for entity in entities]
    n_end = np.asarray([0 if row is None else int(row[0]) for row in rows], dtype=np.int64)
    unseen = np.asarray([1 if row is None else 0 for row in rows], dtype=np.int8)
    component_end = np.zeros((len(frame), len(component_columns)), dtype=np.int64)
    for position, row in enumerate(rows):
        if row is not None:
            component_end[position] = row[1:]
    n_asof = pd.to_numeric(frame[n_column], errors="coerce").fillna(0).to_numpy(np.int64)
    n_delta = n_asof - n_end
    values: dict[str, np.ndarray] = {
        f"{prefix}_n_season": np.maximum(n_delta, 0).astype(np.int32),
        f"{prefix}_log_n_season": np.log1p(np.maximum(n_delta, 0)).astype(np.float32),
        f"{prefix}_unseen": unseen,
    }
    invalid_total = n_delta < 0
    for index, (name, column) in enumerate(component_columns.items()):
        prior = float(priors[name])
        career = pd.to_numeric(frame[column], errors="coerce").fillna(prior).to_numpy(np.float64)
        count_asof = np.rint(career * n_asof).astype(np.int64)
        count_delta = count_asof - component_end[:, index]
        invalid = (n_delta < 0) | (count_delta < 0) | (count_delta > n_delta)
        invalid_total |= invalid
        safe_n = np.where(invalid, 0, n_delta)
        safe_count = np.where(invalid, 0, count_delta)
        rate = (safe_count + k * prior) / (safe_n + k)
        values[f"{prefix}_{name}_rate_season"] = rate.astype(np.float32)
        values[f"{prefix}_{name}_rate_delta"] = (rate - career).astype(np.float32)
    values[f"{prefix}_counter_invalid"] = invalid_total.astype(np.int8)
    return pd.DataFrame(values, index=frame.index)


def build_pitcher_components(frame: pd.DataFrame, state: dict, priors: dict, k: float) -> pd.DataFrame:
    entities = numeric_ids(frame["pitcher_id"])
    rows = [state.get(int(entity)) for entity in entities]
    n_end = np.asarray([0 if row is None else int(row[0]) for row in rows], dtype=np.int64)
    component_end = np.zeros((len(frame), len(COMPONENT_COLUMNS)), dtype=np.int64)
    for position, row in enumerate(rows):
        if row is not None:
            component_end[position] = row[1:]
    n_asof = pd.to_numeric(frame["asof_pitcher_n"], errors="coerce").fillna(0).to_numpy(np.int64)
    n_delta = n_asof - n_end
    invalid_total = np.zeros(len(frame), dtype=bool)
    values: dict[str, np.ndarray] = {}
    for index, (name, column) in enumerate(COMPONENT_COLUMNS.items()):
        prior = float(priors[name])
        career = pd.to_numeric(frame[column], errors="coerce").fillna(prior).to_numpy(np.float64)
        count_asof = np.rint(career * n_asof).astype(np.int64)
        count_delta = count_asof - component_end[:, index]
        invalid = (n_delta < 0) | (count_delta < 0) | (count_delta > n_delta)
        invalid_total |= invalid
        safe_n = np.where(invalid, 0, n_delta)
        safe_count = np.where(invalid, 0, count_delta)
        raw = np.divide(safe_count, safe_n, out=np.full(len(frame), prior), where=safe_n > 0)
        smooth = (safe_count + k * prior) / (safe_n + k)
        values[f"e31_{name}_rate_season"] = smooth.astype(np.float32)
        values[f"e31_{name}_raw_season"] = raw.astype(np.float32)
        values[f"e31_{name}_delta_career"] = (smooth - career).astype(np.float32)
    values["e31_component_invalid"] = invalid_total.astype(np.int8)
    return pd.DataFrame(values, index=frame.index)


def build_trackman(frame: pd.DataFrame, profile: pd.DataFrame) -> pd.DataFrame:
    entities = numeric_ids(frame["pitcher_id"])
    lookup = profile.reindex(entities)
    known = lookup["e58_profile_unseen"].notna().to_numpy(dtype=bool)
    values = lookup.to_numpy(dtype=np.float32)
    unseen_index = list(profile.columns).index("e58_profile_unseen")
    values[~known] = np.nan
    values[~known, unseen_index] = 1.0
    values[known, unseen_index] = 0.0
    return pd.DataFrame(values, columns=profile.columns, index=frame.index)


def build_history_groups(frame: pd.DataFrame, state: dict) -> pd.DataFrame:
    parts = []
    for name, (columns, prefix) in HISTORY_SPECS.items():
        item = state[name]
        table = item["table"]
        if len(columns) == 1:
            aligned = table.reindex(frame[columns[0]].to_numpy())
        else:
            key = pd.MultiIndex.from_frame(frame[list(columns)])
            aligned = table.reindex(key)
        count = aligned["count"].fillna(0.0).to_numpy(dtype=np.float64)
        success = aligned["sum"].fillna(0.0).to_numpy(dtype=np.float64)
        prior = float(item["prior"])
        k = float(item["k"])
        rate = (success + k * prior) / (count + k)
        parts.append(
            pd.DataFrame(
                {
                    f"{prefix}_rate": rate.astype(np.float32),
                    f"{prefix}_delta": (rate - prior).astype(np.float32),
                    f"{prefix}_n_log": np.log1p(count).astype(np.float32),
                    f"{prefix}_unseen": (count <= 0).astype(np.int8),
                },
                index=frame.index,
            )
        )
    return pd.concat(parts, axis=1)


def build_features(frame: pd.DataFrame, manifest: dict, state: dict) -> pd.DataFrame:
    missing = [column for column in BASE_FEATURES if column not in frame.columns]
    if missing:
        raise ValueError(f"test.csv missing columns: {missing}")
    prior = float(state["prior"])
    e14 = build_entity_success(
        frame, state["e14"], prior, 50.0, "pitcher_id", "asof_pitcher_n",
        "asof_pitcher_success_rate", "e14",
    )
    batter = build_entity_success(
        frame, state["batter"], prior, 80.0, "batter_id", "asof_batter_n",
        "asof_batter_success_rate", "e49_batter",
    )
    batter_middle = build_generic_component(
        frame, state["batter_middle"], state["batter_middle_priors"],
        "batter_id", "asof_batter_n", {"middle": "asof_batter_middle_rate"},
        "e52_batter", 100.0,
    )
    parts = [
        frame[BASE_FEATURES], e14, build_platoon(frame, state["platoon"]),
        build_trackman(frame, state["trackman_profile"]),
        build_pitcher_components(
            frame, state["components"], state["component_priors"], 120.0
        ),
        build_hand_matchup(frame), build_e14_interactions(frame, e14), batter,
        batter_middle, build_history_groups(frame, state["history_groups"]),
    ]
    return pd.concat(parts, axis=1)


def prepare_categorical(matrix: pd.DataFrame, spec: dict) -> pd.DataFrame:
    result = matrix.loc[:, spec["model_features"]].copy()
    for column in spec.get("categorical", []):
        result[column] = result[column].astype("string").fillna("__missing__").astype(str)
    return result


def predict(frame: pd.DataFrame, manifest: dict, state: dict) -> np.ndarray:
    matrix = build_features(frame, manifest, state)
    total = np.zeros(len(frame), dtype=np.float64)
    for spec in manifest["models"]:
        model = joblib.load(verified_path(spec["file"], spec["sha256"]))
        raw = np.asarray(model.predict_proba(prepare_categorical(matrix, spec)), dtype=np.float64)
        probability = raw[:, spec["success_indices"]].sum(axis=1)
        total += float(spec["weight"]) * probability
    calibration = manifest["calibration"]
    total = 0.5 + float(calibration["slope"]) * (total - 0.5) + float(
        calibration["offset"]
    )
    if not np.isfinite(total).all():
        raise ValueError("Predictions contain NaN or infinity")
    return np.clip(total, 1e-6, 1.0 - 1e-6)


def main() -> None:
    test = pd.read_csv(DATA_DIR / "test.csv", encoding="utf-8-sig")
    sample = pd.read_csv(DATA_DIR / "sample_submission.csv", encoding="utf-8-sig")
    manifest, state = load_assets()
    probability = predict(test, manifest, state)
    mapping = dict(zip(test[ID_COL], probability))
    output = sample[[ID_COL]].copy()
    output[TARGET_COL] = output[ID_COL].map(mapping)
    if output[TARGET_COL].isna().any():
        raise ValueError("sample_submission.csv contains row_id absent from test.csv")
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(OUTPUT_PATH, index=False, encoding="utf-8")
    print(f"wrote {len(output):,} rows to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
