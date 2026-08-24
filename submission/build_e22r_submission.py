#!/usr/bin/env python3
"""Build the unsubmitted S6 E22R soft pitch-group candidate."""

from __future__ import annotations

import json
import platform
import shutil
import sys
import tempfile
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import scipy
import sklearn

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.run_baselines import FEATURES as BASE_FEATURES  # noqa: E402
from experiments.run_baselines import TARGET, load_train  # noqa: E402
from experiments.run_e14_rolling import (  # noqa: E402
    E14_FEATURES,
    E14_K,
    build_e14_features,
    make_hgb,
    make_linear,
    prior_before_each_season,
    season_end_state,
)
from experiments.run_e15_pseudo_forward import candidate_priors  # noqa: E402
from experiments.run_e22r_probs_rolling import (  # noqa: E402
    E22_FEATURES,
    E22_PROB_FEATURES,
    GROUPS,
    align_group_probabilities,
    load_group_labels,
    make_stage1,
)
from submission.build_submission import (  # noqa: E402
    common_metadata,
    deterministic_zip,
    sha256_file,
    write_json,
)


TRAIN_PATH = ROOT / "open/data/train.csv"
TEMPLATE_DIR = Path(__file__).resolve().parent / "template"
OUTPUT_DIR = Path(__file__).resolve().parent / "dist"
RECORD_DIR = Path(__file__).resolve().parent / "records"


def fit_and_save(label: str, factory, train_x: pd.DataFrame, train_y: np.ndarray, path: Path) -> dict:
    print(f"[{label}] fitting {len(train_x):,} rows x {train_x.shape[1]} features", flush=True)
    started = time.perf_counter()
    model = factory(list(train_x.columns))
    model.fit(train_x, train_y)
    fit_seconds = time.perf_counter() - started
    joblib.dump(model, path, compress=3)
    del model
    return {
        "file": path.name,
        "sha256": sha256_file(path),
        "bytes": path.stat().st_size,
        "fit_seconds": fit_seconds,
    }


def main() -> None:
    started = time.perf_counter()
    train_path = TRAIN_PATH.resolve()
    labels = load_group_labels()
    train = load_train(train_path)
    row_ids = pd.read_csv(train_path, usecols=["row_id"], dtype="string", encoding="utf-8-sig")[
        "row_id"
    ]
    if len(row_ids) != len(train):
        raise AssertionError("row_id length does not match training frame")
    train.insert(0, "row_id", row_ids.to_numpy())
    train["e22_pitch_type_group"] = train["row_id"].map(labels)
    labeled = train.dropna(subset=["e22_pitch_type_group"])
    if labeled.empty:
        raise ValueError("No matched pitch-group labels available for E22 stage 1")

    prior = float(candidate_priors(train, 2025)["r_recent3"])
    states_before, final_state = season_end_state(train)
    priors = prior_before_each_season(train)
    train_e14, e14_meta = build_e14_features(train, states_before, priors, prior, k=E14_K)

    print(f"[S6/e22/stage1] fitting {len(labeled):,} labeled rows", flush=True)
    stage1 = make_stage1()
    stage1_started = time.perf_counter()
    stage1.fit(labeled[E22_FEATURES], labeled["e22_pitch_type_group"].astype(str))
    stage1_fit_seconds = time.perf_counter() - stage1_started
    train_e22 = pd.DataFrame(
        align_group_probabilities(stage1, train),
        columns=E22_PROB_FEATURES,
        index=train.index,
    )
    train_features = pd.concat([train[BASE_FEATURES], train_e14, train_e22], axis=1)
    train_target = train[TARGET].to_numpy(dtype=np.int8, copy=False)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="s6_e22r_build_", dir=OUTPUT_DIR) as temporary:
        stage = Path(temporary)
        model_dir = stage / "model"
        model_dir.mkdir()
        shutil.copyfile(TEMPLATE_DIR / "script.py", stage / "script.py")
        shutil.copyfile(TEMPLATE_DIR / "requirements.txt", stage / "requirements.txt")

        state_payload = {
            str(int(pitcher)): [int(values[0]), int(values[1])]
            for pitcher, values in sorted(final_state.items())
        }
        state_path = model_dir / "e14_state.json"
        state_path.write_text(
            json.dumps(state_payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")),
            encoding="utf-8",
        )
        state_hash = sha256_file(state_path)
        stage1_path = model_dir / "e22_stage1.joblib"
        joblib.dump(stage1, stage1_path, compress=3)
        stage1_hash = sha256_file(stage1_path)
        linear_record = fit_and_save(
            "S6/e22/linear", make_linear, train_features, train_target, model_dir / "linear_sgd.joblib"
        )
        hgb_record = fit_and_save(
            "S6/e22/hgb", make_hgb, train_features, train_target, model_dir / "hgb.joblib"
        )
        model_specs = [
            {
                "file": linear_record["file"],
                "sha256": linear_record["sha256"],
                "bytes": linear_record["bytes"],
                "weight": 0.9,
            },
            {
                "file": hgb_record["file"],
                "sha256": hgb_record["sha256"],
                "bytes": hgb_record["bytes"],
                "weight": 0.1,
            },
        ]
        manifest = {
            "candidate": "S6",
            "description": "Full-data S4 + E22R soft pitch-group probabilities, Linear 90% + HGB 10%, r_recent3 prior",
            "build_spec_version": 1,
            "data_cutoff": "season <= 2024",
            "training_seasons": sorted(int(value) for value in train["season"].unique()),
            "training_rows": len(train),
            "training_data_sha256": sha256_file(train_path),
            "training_code_sha256": {
                "baseline": sha256_file(ROOT / "experiments/run_baselines.py"),
                "e14": sha256_file(ROOT / "experiments/run_e14_rolling.py"),
                "e22r": sha256_file(ROOT / "experiments/run_e22r_probs_rolling.py"),
            },
            "feature_cutoff": "E14 state through 2024; E22 stage-1 trained on matched historical labels only",
            "prior_source": "E15 r_recent3 prior calculated from seasons <= 2024",
            "k_source": "fixed EDA/theoretical E14 value 120; not tuned on validation targets",
            "row_independent_inference": True,
            "base_features": BASE_FEATURES,
            "features": BASE_FEATURES + E14_FEATURES + E22_PROB_FEATURES,
            "e14": {
                "features": E14_FEATURES,
                "state_file": state_path.name,
                "state_sha256": state_hash,
                "prior": prior,
                "k": E14_K,
                "state_cutoff": "after season 2024",
            },
            "e22": {
                "features": E22_PROB_FEATURES,
                "source_features": E22_FEATURES,
                "groups": GROUPS,
                "stage1_model_file": stage1_path.name,
                "stage1_model_sha256": stage1_hash,
                "stage1_label_source": "matched historical pitch_type_group only",
                "uses_current_trackman": False,
            },
            "models": model_specs,
            "training_versions": {
                "python": platform.python_version(),
                "numpy": np.__version__,
                "pandas": pd.__version__,
                "scipy": scipy.__version__,
                "scikit_learn": sklearn.__version__,
                "joblib": joblib.__version__,
            },
        }
        write_json(model_dir / "manifest.json", manifest)
        output = OUTPUT_DIR / "S6_e22r_probs.zip"
        deterministic_zip(stage, output)

    metadata = common_metadata("S6", output, started)
    metadata.update(
        {
            "description": manifest["description"],
            "train_path": str(train_path),
            "train_sha256": manifest["training_data_sha256"],
            "training_rows": len(train),
            "training_seasons": manifest["training_seasons"],
            "prior": prior,
            "prior_mode": "r_recent3",
            "k": E14_K,
            "e14_training_metadata": e14_meta,
            "e22_stage1_fit_seconds": stage1_fit_seconds,
            "e22_labeled_rows": len(labeled),
            "models": [
                {**linear_record, "weight": 0.9},
                {**hgb_record, "weight": 0.1},
            ],
            "rolling_reference": {
                "result": "experiments/results/archive/E22R_probs_v1/e22r_probs_rolling.json",
                "prior_mode": "r_recent3",
                "mean_brier_delta": -0.000004120888850545068,
                "e22r_wins": "3/3",
                "gate_pass": True,
            },
        }
    )
    RECORD_DIR.mkdir(parents=True, exist_ok=True)
    write_json(RECORD_DIR / "S6_build.json", metadata)
    print(f"Built S6: {output} ({metadata['zip_sha256']})", flush=True)


if __name__ == "__main__":
    main()
