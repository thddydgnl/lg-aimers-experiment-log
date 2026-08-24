#!/usr/bin/env python3
"""Fit the selected sparse V3 outcome models and build verified ZIP candidates."""

from __future__ import annotations

import argparse
import gc
import importlib.util
import json
import platform
import shutil
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.run_baselines import FEATURES as BASE_FEATURES, TARGET, load_train  # noqa: E402
from experiments.run_e14_rolling import (  # noqa: E402
    build_e14_features,
    prior_before_each_season,
    season_end_state,
)
from experiments.run_e15_pseudo_forward import candidate_priors  # noqa: E402
from experiments.run_e20r_rolling import (  # noqa: E402
    RICH_PROFILE_COLUMNS,
    RICH_PITCH_GROUPS,
    RICH_TRACKMAN_COLUMNS,
    build_rich_profile_features,
    load_joined_trackman,
    rich_profile_states_before_each_season,
)
from experiments.run_v2_rolling import (  # noqa: E402
    BOOSTER_CATEGORICAL,
    CategoricalFrameModel,
    COMPONENT_RATE_COLUMNS,
    HISTORICAL_GROUP_RATE_SPECS,
    assemble,
    build_component_features,
    build_e14_count_cell_features,
    build_e14_hand_cell_features,
    build_entity_season_features,
    build_generic_component_features,
    build_hand_matchup_features,
    build_historical_group_rate_features,
    build_platoon_frame,
    component_states_before_each_season,
    derive_control_outcome_labels,
    entity_season_end_state,
    generic_component_states_before_each_season,
    platoon_states_before_each_season,
)
from submission.build_submission import (  # noqa: E402
    common_metadata,
    deterministic_zip,
    sha256_file,
    write_json,
)

TEMPLATE = ROOT / "submission" / "template" / "script_v3.py"
PARAMS = {
    "loss_function": "MultiClass",
    "iterations": 500,
    "depth": 6,
    "learning_rate": 0.05,
    "l2_leaf_reg": 12.0,
    "random_seed": 2026,
    "allow_writing_files": False,
    "thread_count": 6,
    "task_type": "GPU",
}
HISTORY_NAMES = list(HISTORICAL_GROUP_RATE_SPECS)
CALIBRATION = {
    "kind": "centered_affine",
    "slope": 1.05,
    "offset": -0.006,
    "formula": "clip(0.5 + slope * (p - 0.5) + offset)",
    "selection_grid": {"slope": [1.0, 1.05, 1.1], "offset": [-0.004, -0.006, -0.008]},
}
WEIGHTS = {
    "V3_sparse_m2_1100": {
        "A": 0.6293619759116473,
        "B": 0.37063802408835267,
    },
    "V3_sparse_m3_1103": {
        "A": 0.501443851662535,
        "C": 0.27016033407769313,
        "B": 0.22839581425977187,
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=ROOT / "open/data/train.csv")
    parser.add_argument("--target-season", type=int, default=2025)
    parser.add_argument("--candidates", nargs="+", choices=sorted(WEIGHTS), default=sorted(WEIGHTS))
    parser.add_argument("--output-dir", type=Path, default=ROOT / "submission/dist")
    parser.add_argument("--record-dir", type=Path, default=ROOT / "submission/records")
    return parser.parse_args()


def rich_keep_columns() -> list[str]:
    return [
        "e58_profile_n_log",
        *[
            f"e58_{metric}_{stat}"
            for metric in RICH_TRACKMAN_COLUMNS
            for stat in ("mean", "sd")
        ],
        "e58_profile_unseen",
    ]


def build_history_state(frame: pd.DataFrame, k: float) -> dict[str, dict[str, Any]]:
    prior = float(frame[TARGET].mean())
    result: dict[str, dict[str, Any]] = {}
    for name, columns in HISTORICAL_GROUP_RATE_SPECS.items():
        table = frame.groupby(list(columns), sort=False, observed=True)[TARGET].agg(
            ["sum", "count"]
        )
        result[name] = {
            "columns": list(columns),
            "table": table,
            "prior": prior,
            "k": float(k),
        }
    return result


def base_assembly(
    frame: pd.DataFrame,
    e14: pd.DataFrame,
    platoon: pd.DataFrame,
    hand: pd.DataFrame,
    interactions: pd.DataFrame,
    trackman: pd.DataFrame | None,
    components: pd.DataFrame | None,
) -> pd.DataFrame:
    return assemble(
        frame,
        e14,
        None,
        platoon,
        None,
        trackman,
        None,
        components,
        None,
        None,
        None,
        hand,
        None,
        None,
        None,
        interactions,
        None,
        None,
        None,
    )


def matrix_for(
    key: str,
    frame: pd.DataFrame,
    common: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    if key == "A":
        matrix = base_assembly(
            frame, common["e14"], common["platoon"], common["hand"],
            common["interactions"], common["trackman"], common["components"],
        )
        matrix = pd.concat([matrix, common["batter"], common["batter_middle"]], axis=1)
    elif key == "C":
        matrix = base_assembly(
            frame, common["e14"], common["platoon"], common["hand"],
            common["interactions"], common["trackman"], None,
        )
        matrix = pd.concat([matrix, common["batter"], common["batter_middle"]], axis=1)
    elif key == "B":
        matrix = base_assembly(
            frame, common["e14"], common["platoon"], common["hand"],
            common["interactions"], None, None,
        )
        matrix = pd.concat(
            [
                matrix,
                common["batter"],
                common["batter_middle"],
                common["history_groups"],
            ],
            axis=1,
        )
    else:
        raise ValueError(key)
    return matrix.drop(columns=["pitcher_id"])


def fit_component(
    key: str,
    matrix: pd.DataFrame,
    labels: pd.Series,
    model_dir: Path,
) -> dict[str, Any]:
    usable = labels.notna().to_numpy(dtype=bool)
    categorical = [column for column in BOOSTER_CATEGORICAL if column in matrix.columns]
    wrapper = CategoricalFrameModel(
        CatBoostClassifier(**PARAMS), categorical, "catboost"
    )
    started = time.perf_counter()
    wrapper.fit(
        matrix.loc[usable], labels.loc[usable].astype(str).to_numpy()
    )
    filename = f"model_{key}.joblib"
    joblib.dump(wrapper.estimator, model_dir / filename, compress=3)
    classes = [str(value) for value in wrapper.estimator.classes_]
    success_indices = [
        index
        for index, value in enumerate(classes)
        if value == "success" or value.startswith("success|")
    ]
    if not success_indices:
        raise ValueError(f"No success class in {classes}")
    fit_seconds = time.perf_counter() - started
    result = {
        "key": key,
        "family": "catboost_outcome",
        "file": filename,
        "sha256": sha256_file(model_dir / filename),
        "bytes": (model_dir / filename).stat().st_size,
        "model_features": list(matrix.columns),
        "categorical": categorical,
        "outcome_classes": classes,
        "success_indices": success_indices,
        "params": PARAMS,
        "fit_rows": int(usable.sum()),
    }
    print(
        f"[{key}] {len(matrix):,} x {matrix.shape[1]} in "
        f"{fit_seconds:.1f}s, {result['bytes'] / 2**20:.2f} MiB",
        flush=True,
    )
    del wrapper, matrix
    gc.collect()
    return result


def import_runtime() -> Any:
    spec = importlib.util.spec_from_file_location("v3_runtime_parity", TEMPLATE)
    if spec is None or spec.loader is None:
        raise ImportError(TEMPLATE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def assert_sample_parity(
    frame: pd.DataFrame,
    state: dict,
    expected: dict[str, pd.DataFrame],
) -> dict[str, float]:
    runtime = import_runtime()
    actual = runtime.build_features(frame, {}, state)
    deltas: dict[str, float] = {}
    for key, matrix in expected.items():
        columns = list(matrix.columns)
        left = matrix[columns]
        right = actual[columns]
        maximum = 0.0
        for column in columns:
            if pd.api.types.is_numeric_dtype(left[column]):
                a = pd.to_numeric(left[column], errors="coerce").to_numpy(dtype=float)
                b = pd.to_numeric(right[column], errors="coerce").to_numpy(dtype=float)
                difference = np.abs(a - b)
                finite = np.isfinite(difference)
                if finite.any():
                    maximum = max(maximum, float(difference[finite].max()))
                if not np.array_equal(np.isnan(a), np.isnan(b)):
                    raise AssertionError(f"NaN mask mismatch for {key}:{column}")
            elif left[column].astype(str).tolist() != right[column].astype(str).tolist():
                raise AssertionError(f"Categorical mismatch for {key}:{column}")
        if maximum > 1e-6:
            raise AssertionError(f"Runtime parity failed for {key}: {maximum}")
        deltas[key] = maximum
    return deltas


def main() -> None:
    args = parse_args()
    started = time.perf_counter()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.record_dir.mkdir(parents=True, exist_ok=True)
    frame = load_train(args.data)
    seasons = sorted(int(value) for value in frame["season"].unique())
    if seasons[-1] != 2024 or args.target_season != 2025:
        raise ValueError(f"Expected 2019-2024 -> 2025; got {seasons} -> {args.target_season}")
    prior = float(candidate_priors(frame, args.target_season)["r_recent3"])
    train_priors = prior_before_each_season(frame)

    e14_before, e14_final = season_end_state(frame)
    e14, _ = build_e14_features(frame, e14_before, train_priors, prior, k=50.0)
    platoon_before, platoon_final = platoon_states_before_each_season(
        frame, train_priors, 50.0, 50.0
    )
    platoon = build_platoon_frame(frame, platoon_before, platoon_final)
    hand = build_hand_matchup_features(frame)
    interactions = pd.concat(
        [
            build_e14_hand_cell_features(frame, e14),
            build_e14_count_cell_features(frame, e14, False),
            build_e14_count_cell_features(frame, e14, True),
        ],
        axis=1,
    )

    batter_before, batter_final = entity_season_end_state(
        frame, "batter_id", "asof_batter_n", "asof_batter_success_rate"
    )
    batter, _ = build_entity_season_features(
        frame, batter_before, train_priors, prior, "batter_id", "asof_batter_n",
        "asof_batter_success_rate", "e49_batter", 80.0,
    )
    middle_columns = {"middle": "asof_batter_middle_rate"}
    middle_before, middle_priors, middle_final, middle_final_priors = (
        generic_component_states_before_each_season(
            frame, "batter_id", "asof_batter_n", middle_columns
        )
    )
    batter_middle, _ = build_generic_component_features(
        frame, middle_before, middle_priors, middle_final_priors, "batter_id",
        "asof_batter_n", middle_columns, "e52_batter", 100.0,
    )

    component_before, component_priors, component_final, component_final_priors = (
        component_states_before_each_season(frame)
    )
    components, _ = build_component_features(
        frame, component_before, component_priors, component_final_priors, 120.0
    )
    history_groups, _ = build_historical_group_rate_features(
        frame, frame, HISTORY_NAMES, 500.0, None, prior
    )

    joined = load_joined_trackman()
    profile_seasons = sorted(int(value) for value in joined["season"].unique())
    profile_before, profile_final_full = rich_profile_states_before_each_season(
        joined, profile_seasons
    )
    trackman_full, _ = build_rich_profile_features(frame, profile_before)
    keep_trackman = rich_keep_columns()
    trackman = trackman_full[keep_trackman].copy()
    profile_final = profile_final_full[keep_trackman].copy()
    del joined, trackman_full, profile_before
    gc.collect()

    common = {
        "e14": e14,
        "platoon": platoon,
        "hand": hand,
        "interactions": interactions,
        "trackman": trackman,
        "components": components,
        "batter": batter,
        "batter_middle": batter_middle,
        "history_groups": history_groups,
    }
    labels = derive_control_outcome_labels(frame, "reverse_any")
    history_state = build_history_state(frame, 500.0)
    state = {
        "prior": prior,
        "e14": e14_final,
        "platoon": platoon_final,
        "batter": batter_final,
        "batter_middle": middle_final,
        "batter_middle_priors": middle_final_priors,
        "components": component_final,
        "component_priors": component_final_priors,
        "trackman_profile": profile_final,
        "history_groups": history_state,
    }

    needed = set().union(*(WEIGHTS[name] for name in args.candidates))
    with tempfile.TemporaryDirectory(prefix="v3_sparse_build_", dir=args.output_dir) as temporary:
        work = Path(temporary)
        shared_model_dir = work / "shared_model"
        shared_model_dir.mkdir()
        state_path = shared_model_dir / "state.joblib"
        joblib.dump(state, state_path, compress=3)
        state_hash = sha256_file(state_path)

        model_specs: dict[str, dict[str, Any]] = {}
        for key in ("A", "B", "C"):
            if key not in needed:
                continue
            model_specs[key] = fit_component(
                key, matrix_for(key, frame, common), labels, shared_model_dir
            )

        sample = pd.read_csv(ROOT / "open/data/test.csv", encoding="utf-8-sig")
        sample_e14, _ = build_e14_features(
            sample, {2025: e14_final}, {2025: prior}, prior, k=50.0
        )
        sample_platoon = build_platoon_frame(
            sample, {2025: platoon_final}, platoon_final
        )
        sample_hand = build_hand_matchup_features(sample)
        sample_interactions = pd.concat(
            [
                build_e14_hand_cell_features(sample, sample_e14),
                build_e14_count_cell_features(sample, sample_e14, False),
                build_e14_count_cell_features(sample, sample_e14, True),
            ],
            axis=1,
        )
        sample_batter, _ = build_entity_season_features(
            sample, {2025: batter_final}, {2025: prior}, prior, "batter_id",
            "asof_batter_n", "asof_batter_success_rate", "e49_batter", 80.0,
        )
        sample_middle, _ = build_generic_component_features(
            sample, {2025: middle_final}, {2025: middle_final_priors},
            middle_final_priors, "batter_id", "asof_batter_n", middle_columns,
            "e52_batter", 100.0,
        )
        sample_components, _ = build_component_features(
            sample, {2025: component_final}, {2025: component_final_priors},
            component_final_priors, 120.0,
        )
        sample_trackman, _ = build_rich_profile_features(
            sample, {2025: profile_final_full}
        )
        sample_groups, _ = build_historical_group_rate_features(
            sample, frame, HISTORY_NAMES, 500.0, None, prior
        )
        sample_common = {
            "e14": sample_e14,
            "platoon": sample_platoon,
            "hand": sample_hand,
            "interactions": sample_interactions,
            "trackman": sample_trackman[keep_trackman],
            "components": sample_components,
            "batter": sample_batter,
            "batter_middle": sample_middle,
            "history_groups": sample_groups,
        }
        parity_expected = {
            key: matrix_for(key, sample, sample_common) for key in sorted(needed)
        }
        parity = assert_sample_parity(sample, state, parity_expected)
        print(f"runtime parity: {parity}", flush=True)

        outputs = []
        for candidate in args.candidates:
            stage = work / candidate
            model_dir = stage / "model"
            model_dir.mkdir(parents=True)
            shutil.copyfile(TEMPLATE, stage / "script.py")
            (stage / "requirements.txt").write_text(
                "catboost==1.2.8\n", encoding="utf-8"
            )
            shutil.copyfile(state_path, model_dir / "state.joblib")
            chosen_models = []
            for key, weight in WEIGHTS[candidate].items():
                spec = dict(model_specs[key])
                spec["weight"] = float(weight)
                shutil.copyfile(
                    shared_model_dir / spec["file"], model_dir / spec["file"]
                )
                chosen_models.append(spec)
            manifest = {
                "candidate": candidate,
                "schema_version": 3,
                "description": "Sparse V3 auxiliary-outcome CatBoost ensemble",
                "data_cutoff": "season <= 2024",
                "target_season": 2025,
                "training_seasons": seasons,
                "training_rows": int(len(frame)),
                "training_data_sha256": sha256_file(args.data.resolve()),
                "row_independent_inference": True,
                "state_file": "state.joblib",
                "state_sha256": state_hash,
                "models": chosen_models,
                "calibration": CALIBRATION,
                "feature_protocol": {
                    "training": "season-wise OOF frozen encoders",
                    "inference": "current row plus state frozen after 2024",
                    "test_aggregate_usage": False,
                    "trackman": "matched official 2019-2024 regular history only",
                },
                "reproducibility": {
                    "artifact_policy": "freeze and verify the final ZIP by SHA-256",
                    "gpu_refit_probe_rows": 98340,
                    "gpu_refit_prediction_max_abs_delta": 1.0448960696685106e-08,
                    "gpu_refit_prediction_mean_abs_delta": 3.823735424506269e-09,
                    "frozen_state_byte_identical_across_refits": True,
                    "note": "CatBoost GPU reductions are numerically, not byte, deterministic",
                },
                "validation": {
                    "2022": 2440.2549 if len(chosen_models) == 2 else 2445.2770,
                    "2023": 0.0,
                    "2024": 960.5052 if len(chosen_models) == 2 else 963.5501,
                    "expected_lb": 1100.6527 if len(chosen_models) == 2 else 1103.6977,
                    "expected_formula_offset": 140.1475834416,
                },
                "runtime_parity_max_delta": parity,
                "training_versions": {
                    "python": platform.python_version(),
                    "numpy": np.__version__,
                    "pandas": pd.__version__,
                    "joblib": joblib.__version__,
                    "catboost": __import__("catboost").__version__,
                },
                "training_code_sha256": {
                    "run_v2_rolling": sha256_file(ROOT / "experiments/run_v2_rolling.py"),
                    "run_e20r_rolling": sha256_file(ROOT / "experiments/run_e20r_rolling.py"),
                    "builder": sha256_file(Path(__file__).resolve()),
                    "runtime": sha256_file(TEMPLATE),
                },
            }
            write_json(model_dir / "manifest.json", manifest)
            output = args.output_dir / f"{candidate}.zip"
            deterministic_zip(stage, output)
            metadata = common_metadata(candidate, output, started)
            metadata.update(
                {
                    "models": list(WEIGHTS[candidate]),
                    "weights": WEIGHTS[candidate],
                    "calibration": CALIBRATION,
                    "validation": manifest["validation"],
                    "runtime_parity_max_delta": parity,
                }
            )
            write_json(args.record_dir / f"{candidate}_build.json", metadata)
            outputs.append((candidate, output, metadata["zip_sha256"]))

    for candidate, output, digest in outputs:
        print(f"Built {candidate}: {output} ({digest})", flush=True)


if __name__ == "__main__":
    main()
