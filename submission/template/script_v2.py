"""DACON inference entry point for v2 candidates (schema_version 2).

Supports the base feature set, two frozen encoders (E14 season-to-date
counters, B1' pitcher x batter_hand platoon split), B2's model-owned pitcher
TargetEncoder input, and any weighted blend of
joblib-serialised estimators, including LightGBM and CatBoost wrappers.

Row independence
    Every derived feature is a dictionary lookup keyed by values on the row
    itself.  No groupby, no rolling, no statistic computed across evaluation
    rows.  The self-check at the end of `predict` re-runs the complete pipeline
    on single-row frames and compares the resulting probabilities against the
    batch result, so it verifies the actual end-to-end property rather than
    comparing a function to itself.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ID_COL = "row_id"
TARGET_COL = "control_success"
DATA_DIR = Path("data")
MODEL_DIR = Path("model")
OUTPUT_PATH = Path("output") / "submission.csv"
SELF_CHECK_ROWS = 3
SELF_CHECK_TOLERANCE = 1e-12


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_csv(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(path)
    return pd.read_csv(path, encoding="utf-8-sig")


def load_manifest() -> dict:
    manifest = json.loads((MODEL_DIR / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("schema_version") != 2:
        raise ValueError("script_v2.py requires manifest schema_version 2.")
    if manifest.get("row_independent_inference") is not True:
        raise ValueError("The manifest does not certify row-independent inference.")
    models = manifest.get("models", [])
    if not models:
        raise ValueError("The manifest lists no models.")
    total = sum(float(item["weight"]) for item in models)
    if abs(total - 1.0) > 1e-9:
        raise ValueError(f"Model weights must sum to 1.0; got {total!r}.")
    for item in models:
        if "file" not in item or "sha256" not in item:
            raise ValueError("Every model entry needs 'file' and 'sha256'.")
    return manifest


def verified_path(name: str, expected: str) -> Path:
    path = MODEL_DIR / name
    actual = sha256_file(path)
    if actual.lower() != expected.lower():
        raise ValueError(f"SHA-256 mismatch for {name}: {actual} != {expected}")
    return path


def load_state(spec: dict, file_key: str, hash_key: str) -> dict:
    path = verified_path(spec[file_key], spec[hash_key])
    return json.loads(path.read_text(encoding="utf-8"))


def pitcher_key(series: pd.Series) -> pd.Series:
    """Match the training-time key: str(int(pitcher_id)), '__missing__' for NaN."""
    numeric = pd.to_numeric(series, errors="coerce")
    keys = numeric.astype("Int64").astype("string")
    return keys.fillna("__missing__")


def build_e14(frame: pd.DataFrame, spec: dict, state: dict) -> pd.DataFrame:
    prior = float(spec["prior"])
    k = float(spec["k"])
    keys = pitcher_key(frame["pitcher_id"])
    lookup = keys.map(lambda key: state.get(key))

    n_end = np.array([0 if item is None else int(item[0]) for item in lookup], dtype=np.int64)
    s_end = np.array([0 if item is None else int(item[1]) for item in lookup], dtype=np.int64)
    unseen = np.array([1 if item is None else 0 for item in lookup], dtype=np.int8)

    n_asof = pd.to_numeric(frame["asof_pitcher_n"], errors="coerce").fillna(0).to_numpy(np.int64)
    career = pd.to_numeric(frame["asof_pitcher_success_rate"], errors="coerce").fillna(prior)
    career_rate = career.to_numpy(np.float64)
    s_asof = np.rint(career_rate * n_asof).astype(np.int64)

    n_delta = n_asof - n_end
    s_delta = s_asof - s_end
    invalid = (n_delta < 0) | (s_delta < 0) | (s_delta > n_delta)
    safe_n = np.where(invalid, 0, n_delta)
    safe_s = np.where(invalid, 0, s_delta)
    rate_season = (safe_s + k * prior) / (safe_n + k)
    return pd.DataFrame(
        {
            "e14_n_season": safe_n.astype(np.int32),
            "e14_s_season": safe_s.astype(np.int32),
            "e14_log_n_season": np.log1p(safe_n).astype(np.float32),
            "e14_rate_season": rate_season.astype(np.float32),
            "e14_rate_delta": (rate_season - career_rate).astype(np.float32),
            "e14_n_season_zero": (safe_n == 0).astype(np.int8),
            "e14_pitcher_unseen": unseen,
            "e14_counter_invalid": invalid.astype(np.int8),
        },
        index=frame.index,
    )


def build_platoon(frame: pd.DataFrame, state: dict) -> pd.DataFrame:
    prior = float(state["prior"])
    pitcher_rate = state["pitcher_rate"]
    platoon_delta = state["platoon_delta"]
    platoon_n = state["platoon_n"]

    keys = pitcher_key(frame["pitcher_id"])
    cell = keys.astype(str) + "|" + frame["batter_hand"].astype(str)
    rate = keys.map(pitcher_rate)
    delta = cell.map(platoon_delta)
    count = cell.map(platoon_n)
    return pd.DataFrame(
        {
            "e30_pitcher_rate": pd.to_numeric(rate, errors="coerce").fillna(prior).to_numpy(np.float32),
            "e30_platoon_delta": pd.to_numeric(delta, errors="coerce").fillna(0.0).to_numpy(np.float32),
            "e30_platoon_n_log": np.log1p(
                pd.to_numeric(count, errors="coerce").fillna(0.0).to_numpy(np.float64)
            ).astype(np.float32),
            "e30_platoon_unseen": delta.isna().to_numpy().astype(np.int8),
        },
        index=frame.index,
    )


def build_pitcher_te(frame: pd.DataFrame) -> pd.DataFrame:
    """Recreate the raw identity column consumed by the fitted TargetEncoder."""
    return pd.DataFrame(
        {"b2_pitcher_id": pitcher_key(frame["pitcher_id"])}, index=frame.index
    )


def build_features(frame: pd.DataFrame, manifest: dict, states: dict) -> pd.DataFrame:
    base_features = list(manifest["base_features"])
    missing = [name for name in base_features if name not in frame.columns]
    if missing:
        raise ValueError(f"test.csv is missing required columns: {missing}")

    parts = [frame.loc[:, base_features]]
    if "e14" in manifest:
        parts.append(build_e14(frame, manifest["e14"], states["e14"]))
    if "platoon" in manifest:
        parts.append(build_platoon(frame, states["platoon"]))
    if "pitcher_te" in manifest:
        parts.append(build_pitcher_te(frame))
    matrix = pd.concat(parts, axis=1)

    expected = list(manifest["features"])
    if list(matrix.columns) != expected:
        raise AssertionError(
            f"Feature order drifted from the manifest.\n  got: {list(matrix.columns)}\n"
            f"  expected: {expected}"
        )
    return matrix


def blend(matrix: pd.DataFrame, manifest: dict) -> np.ndarray:
    total = np.zeros(len(matrix), dtype=np.float64)
    for spec in manifest["models"]:
        path = verified_path(spec["file"], spec["sha256"])
        model = joblib.load(path)
        columns = list(spec.get("model_features", manifest["features"]))
        model_input = matrix.loc[:, columns].copy()
        preprocessing = spec.get("booster_preprocessing")
        if preprocessing:
            for column in preprocessing.get("categorical", []):
                if column not in model_input.columns:
                    continue
                if preprocessing["backend"] == "lgbm":
                    categories = preprocessing.get("categories", {}).get(column, [])
                    model_input[column] = pd.Categorical(
                        model_input[column].astype(str), categories=categories
                    )
                else:
                    model_input[column] = (
                        model_input[column].astype("string")
                        .fillna("__missing__").astype(str)
                    )
        probability = np.asarray(
            model.predict_proba(model_input)[:, 1], dtype=np.float64
        )
        total += float(spec["weight"]) * probability
        del model
    return total


def apply_shift(probability: np.ndarray, manifest: dict) -> np.ndarray:
    """Optional global logit offset (plan section 8.1). A constant, applied identically."""
    shift = float(manifest.get("logit_shift", 0.0))
    if shift == 0.0:
        return probability
    clipped = np.clip(probability, 1e-6, 1.0 - 1e-6)
    odds = np.log(clipped / (1.0 - clipped)) + shift
    return 1.0 / (1.0 + np.exp(-odds))


def finalize(probability: np.ndarray) -> np.ndarray:
    if not np.isfinite(probability).all():
        raise ValueError("Predictions contain NaN or infinity.")
    outside = int(((probability < 0.0) | (probability > 1.0)).sum())
    if outside:
        print(f"WARNING: clipping {outside} predictions outside [0, 1].", flush=True)
    return np.clip(probability, 1e-6, 1.0 - 1e-6)


def predict(frame: pd.DataFrame, manifest: dict) -> np.ndarray:
    states: dict = {}
    if "e14" in manifest:
        states["e14"] = load_state(manifest["e14"], "state_file", "state_sha256")
    if "platoon" in manifest:
        states["platoon"] = load_state(manifest["platoon"], "state_file", "state_sha256")

    matrix = build_features(frame, manifest, states)
    probability = finalize(apply_shift(blend(matrix, manifest), manifest))

    # Genuine row-independence self-check: push single-row frames through the
    # entire pipeline and compare the resulting probabilities to the batch run.
    for position in range(min(SELF_CHECK_ROWS, len(frame))):
        single = frame.iloc[[position]]
        alone = finalize(
            apply_shift(blend(build_features(single, manifest, states), manifest), manifest)
        )
        difference = abs(float(alone[0]) - float(probability[position]))
        if difference > SELF_CHECK_TOLERANCE:
            raise AssertionError(
                f"Row {position} changed with batch composition by {difference:.3e}; "
                "the model is not row-independent."
            )
    return probability


def validate_ids(test: pd.DataFrame, sample: pd.DataFrame) -> None:
    if ID_COL not in test.columns:
        raise ValueError(f"test.csv is missing {ID_COL}")
    if test[ID_COL].isna().any():
        raise ValueError("test.csv row_id must be non-null.")
    if list(sample.columns) != [ID_COL, TARGET_COL]:
        raise ValueError(f"Unexpected sample columns: {list(sample.columns)}")
    if sample[ID_COL].isna().any() or sample[ID_COL].duplicated().any():
        raise ValueError("sample_submission row_id must be non-null and unique.")
    known = set(test[ID_COL].tolist())
    missing = [value for value in sample[ID_COL] if value not in known]
    if missing:
        raise ValueError(f"sample_submission has {len(missing)} IDs absent from test.csv")


def align(sample: pd.DataFrame, ids: pd.Series, probability: np.ndarray) -> pd.DataFrame:
    by_id: dict[object, float] = {}
    for row_id, value in zip(ids.tolist(), probability.tolist()):
        previous = by_id.get(row_id)
        if previous is not None and abs(previous - value) >= 1e-12:
            raise ValueError(f"Duplicate row_id with inconsistent predictions: {row_id}")
        by_id[row_id] = value
    result = sample.copy()
    result[TARGET_COL] = [by_id[row_id] for row_id in result[ID_COL]]
    return result


def main() -> None:
    os.environ.setdefault("OMP_NUM_THREADS", "6")
    os.environ.setdefault("MKL_NUM_THREADS", "6")
    started = time.perf_counter()
    manifest = load_manifest()
    test = load_csv(DATA_DIR / "test.csv")
    sample = load_csv(DATA_DIR / "sample_submission.csv")
    validate_ids(test, sample)
    submission = align(sample, test[ID_COL], predict(test, manifest))
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(OUTPUT_PATH, index=False, encoding="utf-8")
    print(
        f"Saved {OUTPUT_PATH}: rows={len(submission):,}, "
        f"mean={submission[TARGET_COL].mean():.8f}, "
        f"elapsed={time.perf_counter() - started:.2f}s",
        flush=True,
    )


if __name__ == "__main__":
    main()
