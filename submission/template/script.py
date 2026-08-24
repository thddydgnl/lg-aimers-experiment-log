"""DACON inference entry point for row-independent ensemble submissions."""

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
    path = MODEL_DIR / "manifest.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("row_independent_inference") is not True:
        raise ValueError("The model manifest does not certify row-independent inference.")
    models = manifest.get("models", [])
    if not models or abs(sum(float(item["weight"]) for item in models) - 1.0) > 1e-12:
        raise ValueError("Model weights must be present and sum to one.")
    if manifest.get("ensemble_mixture"):
        for item in models:
            kind = item.get("kind", "regular")
            if kind == "regular":
                if "file" not in item or "sha256" not in item:
                    raise ValueError("Regular ensemble component hashes are required.")
            elif kind == "e22_mixture":
                components = item.get("components", [])
                if len(components) != len(manifest["ensemble_mixture"]["groups"]):
                    raise ValueError("E22 mixture must provide one component per group.")
                if any("file" not in component or "sha256" not in component for component in components):
                    raise ValueError("E22 mixture component hashes are required.")
            else:
                raise ValueError(f"Unknown ensemble component kind: {kind}")
    elif manifest.get("e22_mixture"):
        for family in models:
            components = family.get("components", [])
            if len(components) != len(manifest["e22_mixture"]["groups"]):
                raise ValueError("E22 mixture must provide one component per group.")
            if any("file" not in component or "sha256" not in component for component in components):
                raise ValueError("E22 mixture component hashes are required.")
    elif any("file" not in item or "sha256" not in item for item in models):
        raise ValueError("Model file hashes are required.")
    return manifest


def build_row_features(row: dict, features: list[str]) -> dict:
    """Pure reference feature function: one input row produces one feature row."""
    return {feature: row[feature] for feature in features}


def pitcher_key(value: object) -> str:
    if pd.isna(value):
        return "__missing__"
    return str(int(value))


def e14_row_features(row: dict, specification: dict, states: dict[str, list[int]]) -> dict:
    prior = float(specification["prior"])
    k = float(specification["k"])
    n_asof = 0 if pd.isna(row["asof_pitcher_n"]) else int(row["asof_pitcher_n"])
    career_rate = (
        prior
        if pd.isna(row["asof_pitcher_success_rate"])
        else float(row["asof_pitcher_success_rate"])
    )
    state = states.get(pitcher_key(row["pitcher_id"]))
    unseen = 0 if state is not None else 1
    n_end, s_end = state if state is not None else (0, 0)
    s_asof = int(np.rint(career_rate * n_asof))
    n_delta = n_asof - int(n_end)
    s_delta = s_asof - int(s_end)
    invalid = int(n_delta < 0 or s_delta < 0 or s_delta > n_delta)
    n_season = 0 if invalid else n_delta
    s_season = 0 if invalid else s_delta
    rate_season = (s_season + k * prior) / (n_season + k)
    return {
        "e14_n_season": n_season,
        "e14_s_season": s_season,
        "e14_log_n_season": float(np.log1p(n_season)),
        "e14_rate_season": rate_season,
        "e14_rate_delta": rate_season - career_rate,
        "e14_n_season_zero": int(n_season == 0),
        "e14_pitcher_unseen": unseen,
        "e14_counter_invalid": invalid,
    }


def e16_team_key(value: object) -> str:
    if pd.isna(value):
        return "__missing__"
    try:
        return str(int(value))
    except (TypeError, ValueError):
        return str(value)


def e16_row_features(row: dict, specification: dict, states: dict[str, list[int]]) -> dict:
    top_bottom = row.get("top_bottom")
    pitcher_team = row.get("pitcher_team_id")
    batter_team = row.get("batter_team_id")
    home_team = pitcher_team if top_bottom == "T" else batter_team
    state = states.get(pitcher_key(row["pitcher_id"]))
    if state is None:
        start_ratio = 0.5
        games_log = 0.0
        unseen = 1
    else:
        starts, games = int(state[0]), int(state[1])
        start_ratio = starts / games if games > 0 else 0.5
        games_log = float(np.log1p(games))
        unseen = 0
    return {
        "e16_home_team_id": e16_team_key(home_team),
        "e16_role_start_ratio": start_ratio,
        "e16_role_games_log": games_log,
        "e16_role_unseen": unseen,
    }


def align_e22_probabilities(model: object, frame: pd.DataFrame, specification: dict) -> pd.DataFrame:
    source_features = list(specification["source_features"])
    groups = list(specification["groups"])
    probability_features = list(specification["features"])
    missing = [feature for feature in source_features if feature not in frame.columns]
    if missing:
        raise ValueError(f"Missing E22 stage-1 features: {missing}")
    raw = np.asarray(model.predict_proba(frame[source_features]), dtype=np.float64)
    classes = list(model.named_steps["clf"].classes_)
    values = np.zeros((len(frame), len(groups)), dtype=np.float64)
    for source, label in enumerate(classes):
        if label in groups:
            values[:, groups.index(label)] = raw[:, source]
    row_sum = values.sum(axis=1)
    missing_rows = row_sum <= 0.0
    if missing_rows.any():
        values[missing_rows] = 1.0 / len(groups)
        row_sum[missing_rows] = 1.0
    values /= row_sum[:, None]
    return pd.DataFrame(values.astype(np.float32), columns=probability_features, index=frame.index)


def build_features(
    frame: pd.DataFrame, features: list[str], manifest: dict
) -> pd.DataFrame:
    """Vectorised form of build_row_features; it never reads other test rows."""
    e14_specification = manifest.get("e14")
    e16_specification = manifest.get("e16")
    e22_specification = manifest.get("e22")
    base_features = list(manifest.get("base_features", features))
    e14_features = list(e14_specification["features"]) if e14_specification else []
    e16_features = list(e16_specification["features"]) if e16_specification else []
    e22_features = list(e22_specification["features"]) if e22_specification else []
    required_input = base_features
    missing = [feature for feature in required_input if feature not in frame.columns]
    if missing:
        raise ValueError(f"Missing model features: {missing}")
    result = frame.loc[:, base_features].copy()
    if e14_specification or e16_specification:
        derived_features: list[str] = []
        e14_states: dict[str, list[int]] = {}
        e16_states: dict[str, list[int]] = {}
        if e14_specification:
            derived_features.extend(e14_features)
            state_path = MODEL_DIR / e14_specification["state_file"]
            if sha256_file(state_path).lower() != e14_specification["state_sha256"].lower():
                raise ValueError(f"E14 state SHA-256 mismatch: {state_path}")
            e14_states = json.loads(state_path.read_text(encoding="utf-8"))
        if e16_specification:
            derived_features.extend(e16_features)
            role_path = MODEL_DIR / e16_specification["role_state_file"]
            if sha256_file(role_path).lower() != e16_specification["role_state_sha256"].lower():
                raise ValueError(f"E16 role state SHA-256 mismatch: {role_path}")
            e16_states = json.loads(role_path.read_text(encoding="utf-8"))
        derived = []
        for row in frame.to_dict(orient="records"):
            row_features: dict[str, object] = {}
            if e14_specification:
                row_features.update(e14_row_features(row, e14_specification, e14_states))
            if e16_specification:
                row_features.update(e16_row_features(row, e16_specification, e16_states))
            derived.append(row_features)
        derived_frame = pd.DataFrame(derived, index=frame.index).loc[:, derived_features]
        result = pd.concat([result, derived_frame], axis=1)
        if list(result.columns) != base_features + derived_features:
            raise AssertionError("Manifest feature order does not match derived output.")
        for index in range(min(5, len(frame))):
            row = frame.iloc[index].to_dict()
            reference: dict[str, object] = {}
            if e14_specification:
                reference.update(e14_row_features(row, e14_specification, e14_states))
            if e16_specification:
                reference.update(e16_row_features(row, e16_specification, e16_states))
            for feature in derived_features:
                left, right = result.iloc[index][feature], reference[feature]
                if isinstance(right, str):
                    if str(left) != right:
                        raise AssertionError(
                            f"Derived scalar/vector feature mismatch: row={index}, {feature}"
                        )
                elif not np.isclose(left, right, equal_nan=True, atol=1e-12):
                    raise AssertionError(
                        f"Derived scalar/vector feature mismatch: row={index}, {feature}"
                    )
        if e22_specification:
            stage1_path = MODEL_DIR / e22_specification["stage1_model_file"]
            if sha256_file(stage1_path).lower() != e22_specification["stage1_model_sha256"].lower():
                raise ValueError(f"E22 stage-1 model SHA-256 mismatch: {stage1_path}")
            stage1_model = joblib.load(stage1_path)
            e22_frame = align_e22_probabilities(stage1_model, frame, e22_specification)
            result = pd.concat([result, e22_frame], axis=1)
            if list(result.columns) != features:
                raise AssertionError("Manifest feature order does not match E22 output.")
            scalar = align_e22_probabilities(
                stage1_model, frame.iloc[: min(5, len(frame))], e22_specification
            )
            for index in range(len(scalar)):
                for feature in e22_features:
                    if not np.isclose(
                        result.iloc[index][feature], scalar.iloc[index][feature], atol=1e-12
                    ):
                        raise AssertionError(
                            f"E22 scalar/vector feature mismatch: row={index}, {feature}"
                        )
        return result

    if e22_specification and not (e14_specification or e16_specification):
        stage1_path = MODEL_DIR / e22_specification["stage1_model_file"]
        if sha256_file(stage1_path).lower() != e22_specification["stage1_model_sha256"].lower():
            raise ValueError(f"E22 stage-1 model SHA-256 mismatch: {stage1_path}")
        stage1_model = joblib.load(stage1_path)
        e22_frame = align_e22_probabilities(stage1_model, frame, e22_specification)
        result = pd.concat([result, e22_frame], axis=1)
        if list(result.columns) != features:
            raise AssertionError("Manifest feature order does not match E22 output.")
        # Recompute a few rows through the same model path as the scalar check.
        scalar = align_e22_probabilities(
            stage1_model, frame.iloc[: min(5, len(frame))], e22_specification
        )
        for index in range(len(scalar)):
            for feature in e22_features:
                if not np.isclose(
                    result.iloc[index][feature], scalar.iloc[index][feature], atol=1e-12
                ):
                    raise AssertionError(
                        f"E22 scalar/vector feature mismatch: row={index}, {feature}"
                    )
        return result

    for index in range(min(5, len(frame))):
        reference = build_row_features(frame.iloc[index].to_dict(), features)
        for feature in features:
            left, right = result.iloc[index][feature], reference[feature]
            if pd.isna(left) and pd.isna(right):
                continue
            if left != right:
                raise AssertionError(f"Scalar/vector feature mismatch: row={index}, {feature}")
    return result


def validate_ids(test: pd.DataFrame, sample: pd.DataFrame) -> None:
    if ID_COL not in test.columns:
        raise ValueError(f"test.csv is missing {ID_COL}")
    if test[ID_COL].isna().any():
        raise ValueError("test.csv row_id must be non-null.")
    if list(sample.columns) != [ID_COL, TARGET_COL]:
        raise ValueError(f"Unexpected sample columns: {list(sample.columns)}")
    if sample[ID_COL].isna().any() or sample[ID_COL].duplicated().any():
        raise ValueError("sample_submission row_id must be non-null and unique.")
    test_ids = set(test[ID_COL].tolist())
    missing = [row_id for row_id in sample[ID_COL] if row_id not in test_ids]
    if missing:
        raise ValueError(f"sample_submission contains {len(missing)} IDs absent from test.csv")


def predict(frame: pd.DataFrame, manifest: dict) -> np.ndarray:
    features = list(manifest["features"])
    matrix = build_features(frame, features, manifest)
    blended = np.zeros(len(frame), dtype=np.float64)
    ensemble_mixture = manifest.get("ensemble_mixture")
    mixture = manifest.get("e22_mixture")
    if ensemble_mixture:
        group_features = list(ensemble_mixture["probability_features"])
        probabilities = matrix.loc[:, group_features].to_numpy(dtype=np.float64)
        probabilities = probabilities / probabilities.sum(axis=1, keepdims=True)
        for specification in manifest["models"]:
            kind = specification.get("kind", "regular")
            if kind == "regular":
                path = MODEL_DIR / specification["file"]
                actual_hash = sha256_file(path)
                if actual_hash.lower() != specification["sha256"].lower():
                    raise ValueError(f"Model SHA-256 mismatch: {path}")
                model = joblib.load(path)
                model_features = list(specification.get("model_features", features))
                values = np.asarray(
                    model.predict_proba(matrix.loc[:, model_features])[:, 1], dtype=np.float64
                )
                blended += float(specification["weight"]) * values
            elif kind == "e22_mixture":
                model_features = list(specification["model_features"])
                component_predictions = []
                for component in specification["components"]:
                    path = MODEL_DIR / component["file"]
                    actual_hash = sha256_file(path)
                    if actual_hash.lower() != component["sha256"].lower():
                        raise ValueError(f"Model SHA-256 mismatch: {path}")
                    model = joblib.load(path)
                    component_predictions.append(
                        np.asarray(model.predict_proba(matrix.loc[:, model_features])[:, 1], dtype=np.float64)
                    )
                conditional = np.column_stack(component_predictions)
                blended += float(specification["weight"]) * np.sum(
                    probabilities * conditional, axis=1
                )
        if not np.isfinite(blended).all():
            raise ValueError("Predictions contain NaN or infinity.")
        outside = int(((blended < 0.0) | (blended > 1.0)).sum())
        if outside:
            print(f"WARNING: clipping {outside} finite predictions outside [0, 1].", flush=True)
            blended = np.clip(blended, 1e-6, 1.0 - 1e-6)
        return blended
    if mixture:
        group_features = list(mixture["probability_features"])
        model_features = list(mixture["model_features"])
        probabilities = matrix.loc[:, group_features].to_numpy(dtype=np.float64)
        probabilities = probabilities / probabilities.sum(axis=1, keepdims=True)
        for family in manifest["models"]:
            component_predictions = []
            for component in family["components"]:
                path = MODEL_DIR / component["file"]
                actual_hash = sha256_file(path)
                if actual_hash.lower() != component["sha256"].lower():
                    raise ValueError(f"Model SHA-256 mismatch: {path}")
                model = joblib.load(path)
                component_predictions.append(
                    np.asarray(model.predict_proba(matrix.loc[:, model_features])[:, 1], dtype=np.float64)
                )
            conditional = np.column_stack(component_predictions)
            blended += float(family["weight"]) * np.sum(probabilities * conditional, axis=1)
        if not np.isfinite(blended).all():
            raise ValueError("Predictions contain NaN or infinity.")
        outside = int(((blended < 0.0) | (blended > 1.0)).sum())
        if outside:
            print(f"WARNING: clipping {outside} finite predictions outside [0, 1].", flush=True)
            blended = np.clip(blended, 1e-6, 1.0 - 1e-6)
        return blended
    for specification in manifest["models"]:
        path = MODEL_DIR / specification["file"]
        actual_hash = sha256_file(path)
        if actual_hash.lower() != specification["sha256"].lower():
            raise ValueError(f"Model SHA-256 mismatch: {path}")
        model = joblib.load(path)
        values = np.asarray(model.predict_proba(matrix)[:, 1], dtype=np.float64)
        blended += float(specification["weight"]) * values
    if not np.isfinite(blended).all():
        raise ValueError("Predictions contain NaN or infinity.")
    outside = int(((blended < 0.0) | (blended > 1.0)).sum())
    if outside:
        print(f"WARNING: clipping {outside} finite predictions outside [0, 1].", flush=True)
        blended = np.clip(blended, 1e-6, 1.0 - 1e-6)
    return blended


def align_predictions(
    sample: pd.DataFrame, test_ids: pd.Series, predictions: np.ndarray
) -> pd.DataFrame:
    prediction_by_id: dict[object, float] = {}
    for row_id, prediction in zip(test_ids.tolist(), predictions.tolist()):
        previous = prediction_by_id.get(row_id)
        if previous is not None and abs(previous - prediction) >= 1e-12:
            raise ValueError(f"Duplicate row_id has inconsistent predictions: {row_id}")
        prediction_by_id[row_id] = prediction
    result = sample.copy()
    result[TARGET_COL] = [prediction_by_id[row_id] for row_id in result[ID_COL]]
    return result


def main() -> None:
    os.environ.setdefault("OMP_NUM_THREADS", "6")
    os.environ.setdefault("MKL_NUM_THREADS", "6")
    started = time.perf_counter()
    manifest = load_manifest()
    test = load_csv(DATA_DIR / "test.csv")
    sample = load_csv(DATA_DIR / "sample_submission.csv")
    validate_ids(test, sample)
    predictions = predict(test, manifest)
    submission = align_predictions(sample, test[ID_COL], predictions)
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
