#!/usr/bin/env python3
"""Build, parity-check, and freeze V4_compact_supported_1193.zip."""

from __future__ import annotations

import gc
import importlib.util
import json
import platform
import shutil
import sys
import tempfile
import time
import zipfile
from pathlib import Path

import joblib
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.run_baselines import TARGET, load_train  # noqa: E402
from experiments.run_e14_rolling import prior_before_each_season, season_end_state  # noqa: E402
from experiments.run_e15_pseudo_forward import candidate_priors  # noqa: E402
from experiments.run_e20r_rolling import (  # noqa: E402
    load_joined_trackman,
    profile_states_before_each_season,
    rich_profile_states_before_each_season,
    stability_profile_states_before_each_season,
    trackman_count_states_before_each_season,
    trackman_platoon_states_before_each_season,
)
from experiments.run_v2_rolling import (  # noqa: E402
    OUTCOME_CONTEXT_SPECS,
    component_states_before_each_season,
    derive_control_outcome_labels,
    entity_season_end_state,
    generic_component_states_before_each_season,
    platoon_states_before_each_season,
)
from submission.build_submission import deterministic_zip, sha256_file, write_json  # noqa: E402


TRAIN = ROOT / "open/data/train.csv"
TEST = ROOT / "open/data/test.csv"
SAMPLE = ROOT / "open/data/sample_submission.csv"
REGISTRY = ROOT / "submission/work/v4_full_refit/registry.json"
MODEL_SOURCE = ROOT / "submission/work/v4_full_refit/models"
RESEARCH_REPORT = ROOT / "experiments/results/v4_compact_supported_ensemble.json"
PREDICTIONS = ROOT / "experiments/results/predictions"
V3_PACKAGE = ROOT / "submission/dist/V3_sparse_m3_1103.zip"
RUNTIME = ROOT / "submission/template/script_v4_compact.py"
V3_RUNTIME = ROOT / "submission/template/script_v3.py"
OUTPUT = ROOT / "submission/dist/V4_compact_supported_1193.zip"
RECORD = ROOT / "submission/records/V4_compact_supported_1193_build.json"
STATE_CACHE = ROOT / "submission/work/v4_compact_state.joblib"

RUNTIME_MODULES = [
    "experiments/run_baselines.py",
    "experiments/run_e14_rolling.py",
    "experiments/run_e15_pseudo_forward.py",
    "experiments/run_v2_rolling.py",
    "experiments/run_e20r_rolling.py",
    "experiments/stats.py",
    "eda/run_structural_eda.py",
]

# Only parameters that change inference-time feature values are recorded here.
RECIPE_OVERRIDES = {
    1: {"e14_k": 50, "platoon_k": 50, "batter_k": 80, "batter_middle_k": 100},
    2: {"e14_k": 50, "platoon_k": 50, "batter_k": 80, "batter_middle_k": 100},
    3: {"e14_k": 50, "platoon_k": 50, "batter_k": 80, "batter_middle_k": 100},
    4: {"e14_k": 50, "platoon_k": 50, "batter_k": 80, "batter_middle_k": 100},
    5: {"e14_k": 120, "platoon_k": 50},
    6: {"e14_k": 50, "platoon_k": 50, "batter_k": 80, "batter_middle_k": 100},
    7: {"e14_k": 50, "platoon_k": 50, "batter_k": 80, "batter_middle_k": 100},
    8: {"e14_k": 50, "platoon_k": 50, "batter_k": 80, "batter_middle_k": 100},
    9: {"e14_k": 120, "platoon_k": 50},
    10: {"e14_k": 50, "platoon_k": 50, "batter_k": 80, "batter_middle_k": 100},
    11: {"e14_k": 50, "platoon_k": 50, "batter_k": 80, "batter_middle_k": 100},
    12: {"e14_k": 50, "platoon_k": 50, "batter_k": 80},
    13: {"e14_k": 120, "platoon_k": 50, "batter_k": 80, "batter_middle_k": 500},
    14: {"e14_k": 50, "platoon_k": 50, "batter_k": 80, "batter_middle_k": 100},
    15: {},
    16: {"e14_k": 120, "platoon_k": 200},
    17: {"e14_k": 50, "platoon_k": 50, "batter_k": 80, "batter_middle_k": 100, "trackman_window": 2},
    18: {"e14_k": 80, "platoon_k": 50, "batter_k": 80, "batter_middle_k": 100},
}
STUDENT_OVERRIDE = {
    "e14_k": 50,
    "platoon_k": 50,
    "batter_k": 80,
    "batter_middle_k": 100,
}


def recipe(name: str, features: list[str], override: dict) -> dict:
    result = {
        "name": name,
        "features": features,
        "e14_k": 120.0,
        "platoon_k": 200.0,
        "batter_k": 200.0,
        "batter_middle_k": 100.0,
        "trackman_window": None,
    }
    result.update(override)
    return result


def build_state(frame: pd.DataFrame) -> dict:
    print("[state] player counters", flush=True)
    prior = float(candidate_priors(frame, 2025)["r_recent3"])
    _, e14_final = season_end_state(frame)
    priors = prior_before_each_season(frame)
    _, platoon_50 = platoon_states_before_each_season(frame, priors, 50.0, 50.0)
    _, platoon_200 = platoon_states_before_each_season(frame, priors, 200.0, 200.0)
    _, batter_final = entity_season_end_state(
        frame, "batter_id", "asof_batter_n", "asof_batter_success_rate"
    )
    _, _, middle_final, middle_priors = generic_component_states_before_each_season(
        frame,
        "batter_id",
        "asof_batter_n",
        {"middle": "asof_batter_middle_rate"},
    )
    _, _, pitchmix_final, pitchmix_priors = generic_component_states_before_each_season(
        frame,
        "pitcher_id",
        "asof_pitcher_pitchmix_n",
        {
            "fastball": "asof_pitcher_fastball_rate",
            "breaking": "asof_pitcher_breaking_rate",
            "offspeed": "asof_pitcher_offspeed_rate",
        },
    )
    _, _, component_final, component_priors = component_states_before_each_season(frame)

    print("[state] detailed 2024 outcome source", flush=True)
    outcome_labels = derive_control_outcome_labels(frame, "component15")
    outcome_columns = sorted(
        {
            "season",
            TARGET,
            *[column for columns in OUTCOME_CONTEXT_SPECS.values() for column in columns],
        }
    )
    outcome_mask = frame["season"].eq(2024)
    outcome_source = frame.loc[outcome_mask, outcome_columns].copy()
    outcome_labels = outcome_labels.loc[outcome_mask].copy()

    print("[state] target-free TrackMan profiles", flush=True)
    joined = load_joined_trackman()
    seasons = sorted(int(value) for value in joined["season"].unique())
    _, simple = profile_states_before_each_season(joined, seasons)
    _, simple_w2 = profile_states_before_each_season(joined, seasons, window=2)
    _, rich = rich_profile_states_before_each_season(joined, seasons)
    _, stability = stability_profile_states_before_each_season(joined, seasons)
    _, trackman_platoon = trackman_platoon_states_before_each_season(
        joined, seasons, k=200.0
    )
    _, trackman_count = trackman_count_states_before_each_season(
        joined, seasons, k=200.0
    )
    del joined
    gc.collect()
    return {
        "prior": prior,
        "e14": e14_final,
        "platoon_50": platoon_50,
        "platoon_200": platoon_200,
        "batter": batter_final,
        "batter_middle": middle_final,
        "batter_middle_priors": middle_priors,
        "pitchmix": pitchmix_final,
        "pitchmix_priors": pitchmix_priors,
        "components": component_final,
        "components_priors": component_priors,
        "trackman_simple": simple,
        "trackman_simple_w2": simple_w2,
        "trackman_rich": rich,
        "trackman_stability": stability,
        "trackman_platoon": trackman_platoon,
        "trackman_count": trackman_count,
        "outcome_source": outcome_source,
        "outcome_labels": outcome_labels,
    }


def copy_runtime_modules(stage: Path) -> None:
    runtime_root = stage / "model/runtime_lib"
    (runtime_root / "experiments").mkdir(parents=True, exist_ok=True)
    (runtime_root / "eda").mkdir(parents=True, exist_ok=True)
    (runtime_root / "experiments/__init__.py").write_text("", encoding="utf-8")
    (runtime_root / "eda/__init__.py").write_text("", encoding="utf-8")
    for relative in RUNTIME_MODULES:
        source = ROOT / relative
        destination = runtime_root / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)


def import_runtime(stage: Path):
    sys.path.insert(0, str(stage))
    try:
        spec = importlib.util.spec_from_file_location("v4_package_runtime", stage / "script.py")
        if spec is None or spec.loader is None:
            raise ImportError(stage / "script.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        module.MODEL_DIR = stage / "model"
        module.v3_runtime.MODEL_DIR = stage / "model/v3"
        return module
    finally:
        sys.path.pop(0)


def expected_sample(report: dict, registry: dict) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    with np.load(PREDICTIONS / "v3_sparse_m3_frozen_2025.npz") as archive:
        anchor = np.asarray(archive["final_prediction"], dtype=np.float64)
    with np.load(PREDICTIONS / "v4_full_student_2025.npz") as archive:
        student = np.asarray(archive["catboost_teacher"], dtype=np.float64)
    final = student.copy()
    arms: dict[str, np.ndarray] = {}
    for index, item in enumerate(registry["arms"], start=1):
        with np.load(PREDICTIONS / f"v4_full_arm{index:02d}_2025.npz") as archive:
            prediction = np.asarray(archive[item["key"]], dtype=np.float64)
        arms[item["full_stage"]] = prediction
        final += float(item["coefficient"]) * (prediction - anchor)
    return np.clip(final, 1e-6, 1.0 - 1e-6), {
        "anchor": anchor,
        "student": student,
        **arms,
    }


def main() -> None:
    started = time.perf_counter()
    registry = json.loads(REGISTRY.read_text(encoding="utf-8"))
    report = json.loads(RESEARCH_REPORT.read_text(encoding="utf-8"))
    if STATE_CACHE.is_file():
        print(f"[state] reuse {STATE_CACHE}", flush=True)
        state = joblib.load(STATE_CACHE)
    else:
        frame = load_train(TRAIN)
        state = build_state(frame)
        STATE_CACHE.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(state, STATE_CACHE, compress=3)
        del frame
        gc.collect()

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    RECORD.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="v4_compact_", dir=OUTPUT.parent) as temporary:
        stage = Path(temporary)
        model_dir = stage / "model"
        model_dir.mkdir()
        shutil.copyfile(RUNTIME, stage / "script.py")
        copy_runtime_modules(stage)
        shutil.copyfile(V3_RUNTIME, model_dir / "runtime_lib/v3_runtime.py")
        (stage / "requirements.txt").write_text(
            "catboost==1.2.8\nscikit-learn==1.8.0\njoblib==1.5.3\n",
            encoding="utf-8",
        )
        state_path = model_dir / "state.joblib"
        joblib.dump(state, state_path, compress=3)

        v3_dir = model_dir / "v3"
        v3_dir.mkdir()
        with zipfile.ZipFile(V3_PACKAGE) as archive:
            for member in archive.infolist():
                if member.filename.startswith("model/") and not member.is_dir():
                    target = v3_dir / Path(member.filename).name
                    target.write_bytes(archive.read(member))

        manifest_arms = []
        for index, item in enumerate(registry["arms"], start=1):
            source_spec = MODEL_SOURCE / Path(item["export_spec"]).name
            spec_data = json.loads(source_spec.read_text(encoding="utf-8"))
            source_model = MODEL_SOURCE / spec_data["model_file"]
            shutil.copyfile(source_spec, model_dir / source_spec.name)
            shutil.copyfile(source_model, model_dir / source_model.name)
            original_features = json.loads(
                (ROOT / f"experiments/results/{item['stage']}.json").read_text(
                    encoding="utf-8"
                )
            )["metadata"]["features"]
            manifest_arms.append(
                {
                    "name": item["full_stage"],
                    "research_stage": item["stage"],
                    "coefficient": float(item["coefficient"]),
                    "spec_file": source_spec.name,
                    "spec_sha256": sha256_file(source_spec),
                    "model_sha256": sha256_file(source_model),
                    "recipe": recipe(
                        item["full_stage"], original_features, RECIPE_OVERRIDES[index]
                    ),
                }
            )

        student_item = registry["student"]
        student_spec = MODEL_SOURCE / Path(student_item["export_spec"]).name
        student_data = json.loads(student_spec.read_text(encoding="utf-8"))
        student_model = MODEL_SOURCE / student_data["model_file"]
        shutil.copyfile(student_spec, model_dir / student_spec.name)
        shutil.copyfile(student_model, model_dir / student_model.name)
        student_features = json.loads(
            (ROOT / "experiments/results/v4_teacher_residual_centered_r_primary24.json")
            .read_text(encoding="utf-8")
        )["metadata"]["features"]
        manifest_student = {
            "name": "v4_full_student",
            "research_stage": "v4_teacher_residual_centered_r_primary24",
            "spec_file": student_spec.name,
            "spec_sha256": sha256_file(student_spec),
            "model_sha256": sha256_file(student_model),
            "recipe": recipe("v4_full_student", student_features, STUDENT_OVERRIDE),
        }

        manifest = {
            "schema_version": 4,
            "candidate": "V4_compact_supported_1193",
            "description": "V3 anchor + supported teacher residual + 18 bounded OOT directions",
            "data_cutoff": "season <= 2024",
            "target_season": 2025,
            "official_data_only": True,
            "external_api_usage": False,
            "test_aggregate_usage": False,
            "row_independent_inference": True,
            "state_file": "state.joblib",
            "state_sha256": sha256_file(state_path),
            "student": manifest_student,
            "arms": manifest_arms,
            "ensemble_formula": "clip(student + sum(coef * (arm - v3_anchor)))",
            "expected_score": report["expected_score"],
            "validation": report["folds"],
            "training_rows": 1475092,
            "training_data_sha256": sha256_file(TRAIN),
            "training_versions": {
                "python": platform.python_version(),
                "numpy": np.__version__,
                "pandas": pd.__version__,
                "joblib": joblib.__version__,
            },
        }
        write_json(model_dir / "manifest.json", manifest)

        print("[parity] five official sample rows", flush=True)
        runtime = import_runtime(stage)
        sample = pd.read_csv(TEST, encoding="utf-8-sig")
        assets = runtime.load_assets()
        factory = runtime.FeatureFactory(sample, assets[1])
        expected_final, expected_parts = expected_sample(report, registry)
        component_deltas = {}
        anchor_actual = runtime.v3_runtime.predict(sample, assets[2], assets[3])
        component_deltas["anchor"] = float(
            np.max(np.abs(anchor_actual - expected_parts["anchor"]))
        )
        student_raw = runtime.model_prediction(factory, manifest_student, manifest)
        student_actual = np.clip(anchor_actual + student_raw - 0.5, 1e-6, 1 - 1e-6)
        component_deltas["student"] = float(
            np.max(np.abs(student_actual - expected_parts["student"]))
        )
        for arm in manifest_arms:
            actual = runtime.model_prediction(factory, arm, manifest)
            component_deltas[arm["name"]] = float(
                np.max(np.abs(actual - expected_parts[arm["name"]]))
            )
        actual_final = runtime.predict(sample, assets)
        final_delta = float(np.max(np.abs(actual_final - expected_final)))
        maximum_component_delta = max(component_deltas.values())
        print(json.dumps(component_deltas, indent=2), flush=True)
        print(f"final_max_abs_delta={final_delta:.12g}", flush=True)
        if maximum_component_delta > 2e-6 or final_delta > 2e-6:
            worst = max(component_deltas, key=component_deltas.get)
            raise AssertionError(
                f"Runtime parity failed: component {worst}={component_deltas[worst]}, "
                f"final={final_delta}"
            )
        manifest["sample_parity"] = {
            "max_component_abs_delta": maximum_component_delta,
            "final_max_abs_delta": final_delta,
            "component_abs_delta": component_deltas,
        }
        write_json(model_dir / "manifest.json", manifest)
        # Local parity imports can create bytecode caches inside this temporary
        # staging tree.  They are not runtime assets and top-level extras are
        # rejected by the submission structure gate.
        for cache_dir in stage.rglob("__pycache__"):
            shutil.rmtree(cache_dir)
        deterministic_zip(stage, OUTPUT)

    build_record = {
        "candidate": manifest["candidate"],
        "zip": str(OUTPUT.relative_to(ROOT)),
        "zip_sha256": sha256_file(OUTPUT),
        "zip_bytes": OUTPUT.stat().st_size,
        "expected_score": manifest["expected_score"],
        "sample_parity": manifest["sample_parity"],
        "model_count": 22,
        "elapsed_seconds": time.perf_counter() - started,
    }
    write_json(RECORD, build_record)
    print(json.dumps(build_record, indent=2), flush=True)


if __name__ == "__main__":
    main()
