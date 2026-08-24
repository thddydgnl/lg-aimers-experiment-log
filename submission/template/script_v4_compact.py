"""Row-independent inference for the compact supported V4 ensemble."""

from __future__ import annotations

import gc
import hashlib
import json
from pathlib import Path
import sys

import joblib
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, CatBoostRegressor

RUNTIME_LIB = Path(__file__).resolve().parent / "model" / "runtime_lib"
if str(RUNTIME_LIB) not in sys.path:
    sys.path.insert(0, str(RUNTIME_LIB))

import v3_runtime
from experiments.run_baselines import FEATURES as BASE_FEATURES, dtype_map
from experiments.run_e14_rolling import build_e14_features
from experiments.run_v2_rolling import (
    build_component_features,
    build_count_state_feature,
    build_current_state_full_features,
    build_current_state_interaction_features,
    build_e14_count_cell_features,
    build_e14_hand_cell_features,
    build_e14_multi_features,
    build_entity_season_features,
    build_generic_component_features,
    build_hand_matchup_features,
    build_outcome_context_features,
    build_platoon_frame,
    build_recent_form_features,
)
from experiments.run_e20r_rolling import (
    build_profile_features,
    build_rich_profile_features,
    build_stability_profile_features,
    build_trackman_count_features,
    build_trackman_platoon_features,
)


ID_COL = "row_id"
TARGET_COL = "control_success"
DATA_DIR = Path("data")
MODEL_DIR = Path("model")
OUTPUT_PATH = Path("output") / "submission.csv"
PITCHER_COMPONENTS = {
    "reverse": "asof_pitcher_reverse_rate",
    "middle": "asof_pitcher_middle_rate",
    "ball": "asof_pitcher_ball_rate",
    "strike": "asof_pitcher_strike_rate",
}
BATTER_MIDDLE = {"middle": "asof_batter_middle_rate"}
PITCHMIX = {
    "fastball": "asof_pitcher_fastball_rate",
    "breaking": "asof_pitcher_breaking_rate",
    "offspeed": "asof_pitcher_offspeed_rate",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verified_path(relative: str, expected: str) -> Path:
    path = MODEL_DIR / relative
    actual = sha256_file(path)
    if actual.lower() != expected.lower():
        raise ValueError(f"SHA-256 mismatch for {relative}: {actual} != {expected}")
    return path


def season_mapping(frame: pd.DataFrame, value) -> dict[int, object]:
    return {int(season): value for season in frame["season"].unique()}


class FeatureFactory:
    """Build each recipe from the current row plus frozen 2019-2024 state."""

    def __init__(self, frame: pd.DataFrame, state: dict):
        missing = [column for column in BASE_FEATURES if column not in frame.columns]
        if missing:
            raise ValueError(f"test.csv missing columns: {missing}")
        # Full refits used the optimized training loader.  Reapply its exact
        # dtypes so float thresholds see the same float32 values at inference.
        self.frame = frame.copy()
        for column, dtype in dtype_map().items():
            if column in self.frame.columns:
                self.frame[column] = self.frame[column].astype(dtype)
        self.state = state
        self.prior = float(state["prior"])
        self.cache: dict[tuple, pd.DataFrame] = {}

    def e14(self, k: float) -> pd.DataFrame:
        key = ("e14", float(k))
        if key not in self.cache:
            self.cache[key], _ = build_e14_features(
                self.frame,
                season_mapping(self.frame, self.state["e14"]),
                season_mapping(self.frame, self.prior),
                self.prior,
                k=float(k),
            )
        return self.cache[key]

    def platoon(self, k: float) -> pd.DataFrame:
        key = ("platoon", float(k))
        if key not in self.cache:
            frozen = self.state[f"platoon_{int(k)}"]
            self.cache[key] = build_platoon_frame(
                self.frame, season_mapping(self.frame, frozen), frozen
            )
        return self.cache[key]

    def batter(self, k: float) -> pd.DataFrame:
        key = ("batter", float(k))
        if key not in self.cache:
            self.cache[key], _ = build_entity_season_features(
                self.frame,
                season_mapping(self.frame, self.state["batter"]),
                season_mapping(self.frame, self.prior),
                self.prior,
                "batter_id",
                "asof_batter_n",
                "asof_batter_success_rate",
                "e49_batter",
                float(k),
            )
        return self.cache[key]

    def generic(
        self,
        name: str,
        entity: str,
        n_column: str,
        columns: dict[str, str],
        prefix: str,
        k: float,
        include_raw: bool,
    ) -> pd.DataFrame:
        key = (name, float(k), bool(include_raw))
        if key not in self.cache:
            frozen = self.state[name]
            priors = self.state[f"{name}_priors"]
            self.cache[key], _ = build_generic_component_features(
                self.frame,
                season_mapping(self.frame, frozen),
                season_mapping(self.frame, priors),
                priors,
                entity,
                n_column,
                columns,
                prefix,
                float(k),
                include_raw=include_raw,
            )
        return self.cache[key]

    def trackman(self, name: str) -> pd.DataFrame:
        key = ("trackman", name)
        if key in self.cache:
            return self.cache[key]
        frozen = self.state[f"trackman_{name}"]
        mapping = season_mapping(self.frame, frozen)
        builders = {
            "simple": build_profile_features,
            "simple_w2": build_profile_features,
            "rich": build_rich_profile_features,
            "stability": build_stability_profile_features,
            "platoon": build_trackman_platoon_features,
            "count": build_trackman_count_features,
        }
        self.cache[key], _ = builders[name](self.frame, mapping)
        return self.cache[key]

    def components(self) -> pd.DataFrame:
        key = ("components",)
        if key not in self.cache:
            priors = self.state["components_priors"]
            self.cache[key], _ = build_component_features(
                self.frame,
                season_mapping(self.frame, self.state["components"]),
                season_mapping(self.frame, priors),
                priors,
                120.0,
            )
        return self.cache[key]

    def outcome_context(self) -> pd.DataFrame:
        key = ("outcome_context",)
        if key not in self.cache:
            self.cache[key], _ = build_outcome_context_features(
                self.frame,
                self.state["outcome_source"],
                self.state["outcome_labels"],
                200.0,
            )
        return self.cache[key]

    def current_state(
        self,
        e14: pd.DataFrame,
        batter: pd.DataFrame,
        middle: pd.DataFrame,
        include_context: bool,
        include_level: bool,
    ) -> pd.DataFrame:
        key = ("current", include_context, include_level)
        if key not in self.cache:
            pitchmix = self.generic(
                "pitchmix", "pitcher_id", "asof_pitcher_pitchmix_n",
                PITCHMIX, "e53_pitchmix", 100.0, True,
            )
            current = build_current_state_full_features(
                e14, self.components(), batter, pd.concat([middle, pitchmix], axis=1)
            )
            if include_context or include_level:
                interactions = build_current_state_interaction_features(
                    self.frame, current, include_context, include_level
                )
                current = pd.concat([current, interactions], axis=1)
            self.cache[key] = current
        return self.cache[key]

    def matrix(self, recipe: dict, required: list[str]) -> pd.DataFrame:
        features = set(recipe["features"])
        e14_needed = bool(
            features
            & {
                "e14", "e14_multi", "e14_hand_cells", "e14_count_cells",
                "e14_type_count_cells", "recent_form", "current_state_full",
                "current_state_context", "current_state_level",
            }
        )
        e14 = self.e14(recipe["e14_k"]) if e14_needed else None
        parts: list[pd.DataFrame] = [self.frame[BASE_FEATURES]]
        if "e14" in features:
            parts.append(e14)
        if "e14_multi" in features:
            multi_key = ("e14_multi", float(recipe["e14_k"]))
            if multi_key not in self.cache:
                self.cache[multi_key] = build_e14_multi_features(
                    self.frame,
                    e14,
                    season_mapping(self.frame, self.prior),
                    self.prior,
                )
            parts.append(self.cache[multi_key])
        if "platoon" in features:
            parts.append(self.platoon(recipe["platoon_k"]))
        trackman_parts = []
        if "trackman" in features:
            trackman_parts.append(
                self.trackman("simple_w2" if recipe.get("trackman_window") == 2 else "simple")
            )
        if "trackman_rich" in features:
            trackman_parts.append(self.trackman("rich"))
        if "trackman_stability" in features:
            trackman_parts.append(self.trackman("stability"))
        if "trackman_platoon" in features:
            trackman_parts.append(self.trackman("platoon"))
        if "trackman_count" in features:
            trackman_parts.append(self.trackman("count"))
        if trackman_parts:
            parts.append(pd.concat(trackman_parts, axis=1))
        if "hand_matchup" in features:
            parts.append(build_hand_matchup_features(self.frame))
        if "count_state" in features:
            parts.append(build_count_state_feature(self.frame))
        interactions = []
        if "e14_hand_cells" in features:
            interactions.append(build_e14_hand_cell_features(self.frame, e14))
        if "e14_count_cells" in features:
            interactions.append(build_e14_count_cell_features(self.frame, e14, False))
        if "e14_type_count_cells" in features:
            interactions.append(build_e14_count_cell_features(self.frame, e14, True))
        if interactions:
            parts.append(pd.concat(interactions, axis=1))

        current_needed = bool(
            features & {"current_state_full", "current_state_context", "current_state_level"}
        )
        batter_needed = "batter_e14" in features or current_needed
        middle_needed = "batter_middle_e14" in features or current_needed
        batter = self.batter(recipe["batter_k"]) if batter_needed else None
        middle = (
            self.generic(
                "batter_middle", "batter_id", "asof_batter_n",
                BATTER_MIDDLE, "e52_batter", recipe["batter_middle_k"],
                current_needed,
            )
            if middle_needed else None
        )
        if "batter_e14" in features:
            parts.append(batter)
        if "batter_middle_e14" in features:
            parts.append(middle)
        # In the research harness, requesting current-state reconstruction
        # builds both auxiliary blocks.  When batter_middle_e14 is also an
        # explicit feature, the whole auxiliary frame (middle + pitch mix) is
        # appended before the reconstructed e70 state.
        if current_needed and (
            "batter_middle_e14" in features or "pitchmix_e14" in features
        ):
            parts.append(
                self.generic(
                    "pitchmix", "pitcher_id", "asof_pitcher_pitchmix_n",
                    PITCHMIX, "e53_pitchmix", 100.0, True,
                )
            )
        if current_needed:
            parts.append(
                self.current_state(
                    e14,
                    batter,
                    middle,
                    "current_state_context" in features,
                    "current_state_level" in features,
                )
            )
        if "outcome_context" in features:
            parts.append(self.outcome_context())
        if "recent_form" in features:
            recent_key = ("recent", float(recipe["e14_k"]))
            if recent_key not in self.cache:
                self.cache[recent_key] = build_recent_form_features(
                    self.frame, e14, False
                )
            parts.append(self.cache[recent_key])
        matrix = pd.concat(parts, axis=1)
        missing = [column for column in required if column not in matrix.columns]
        if missing:
            raise ValueError(f"Recipe {recipe['name']} missing features: {missing}")
        return matrix.loc[:, required].copy()


def prepare(matrix: pd.DataFrame, spec: dict) -> pd.DataFrame:
    for column in spec.get("ordinal_encoding_columns", []):
        values = matrix[column].astype("string").fillna("__missing__").astype(str)
        categories = spec.get("ordinal_categories", {}).get(column, [])
        matrix[column] = pd.Categorical(values, categories=categories).codes.astype(
            np.int32
        )
    for column in spec.get("categorical", []):
        matrix[column] = (
            matrix[column].astype("string").fillna("__missing__").astype(str)
        )
    return matrix


def model_prediction(
    factory: FeatureFactory, item: dict, manifest: dict
) -> np.ndarray:
    spec = json.loads(
        verified_path(item["spec_file"], item["spec_sha256"]).read_text(
            encoding="utf-8"
        )
    )
    model_path = verified_path(spec["model_file"], item["model_sha256"])
    matrix = prepare(
        factory.matrix(item["recipe"], spec["feature_columns"]), spec
    )
    if spec["kind"] == "regressor":
        model = CatBoostRegressor()
        model.load_model(model_path)
        raw = np.asarray(model.predict(matrix), dtype=np.float64)
        if spec.get("clip_prediction"):
            raw = np.clip(raw, *spec["clip_prediction"])
        prediction = raw
    else:
        model = CatBoostClassifier()
        model.load_model(model_path)
        probabilities = np.asarray(model.predict_proba(matrix), dtype=np.float64)
        weights = spec.get("class_weight_vector")
        if weights is not None:
            probabilities = probabilities / np.asarray(weights, dtype=np.float64)[None, :]
            probabilities /= probabilities.sum(axis=1, keepdims=True)
        indices = (
            spec["success_indices"]
            if spec["kind"] == "outcome_classifier"
            else spec["positive_indices"]
        )
        prediction = probabilities[:, indices].sum(axis=1)
        del probabilities
    del model, matrix
    gc.collect()
    if not np.isfinite(prediction).all():
        raise ValueError(f"Non-finite prediction: {item['name']}")
    return prediction


def load_assets() -> tuple[dict, dict, dict, dict]:
    manifest = json.loads((MODEL_DIR / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("schema_version") != 4:
        raise ValueError("V4 manifest required")
    if manifest.get("row_independent_inference") is not True:
        raise ValueError("Manifest does not certify row independence")
    state = joblib.load(
        verified_path(manifest["state_file"], manifest["state_sha256"])
    )
    v3_runtime.MODEL_DIR = MODEL_DIR / "v3"
    v3_manifest, v3_state = v3_runtime.load_assets()
    return manifest, state, v3_manifest, v3_state


def predict(frame: pd.DataFrame, assets=None) -> np.ndarray:
    if assets is None:
        assets = load_assets()
    manifest, state, v3_manifest, v3_state = assets
    anchor = v3_runtime.predict(frame, v3_manifest, v3_state)
    factory = FeatureFactory(frame, state)
    student_raw = model_prediction(factory, manifest["student"], manifest)
    student = np.clip(anchor + student_raw - 0.5, 1e-6, 1.0 - 1e-6)
    final = student.copy()
    for arm in manifest["arms"]:
        arm_prediction = model_prediction(factory, arm, manifest)
        final += float(arm["coefficient"]) * (arm_prediction - anchor)
    final = np.clip(final, 1e-6, 1.0 - 1e-6)
    if not np.isfinite(final).all():
        raise ValueError("Predictions contain NaN or infinity")
    return final


def main() -> None:
    test = pd.read_csv(DATA_DIR / "test.csv", encoding="utf-8-sig")
    sample = pd.read_csv(DATA_DIR / "sample_submission.csv", encoding="utf-8-sig")
    probability = predict(test)
    mapping = dict(zip(test[ID_COL].astype(str), probability))
    output = sample[[ID_COL]].copy()
    output[TARGET_COL] = output[ID_COL].astype(str).map(mapping)
    if output[TARGET_COL].isna().any():
        raise ValueError("sample_submission.csv contains row_id absent from test.csv")
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(OUTPUT_PATH, index=False, encoding="utf-8")
    print(f"wrote {len(output):,} rows to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
