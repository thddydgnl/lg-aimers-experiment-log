#!/usr/bin/env python3
"""Build full-data E14 candidates and the optional E16 context candidate."""

from __future__ import annotations

import hashlib
import json
import platform
import argparse
import shutil
import sys
import tempfile
import time
from datetime import datetime, timezone
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
from experiments.run_e16_rolling import (  # noqa: E402
    E16_FEATURES,
    build_e16_features,
    make_e16_hgb,
    make_e16_linear,
    role_states_before_each_season,
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-id", choices=("S3", "S4", "S5"), default="S3")
    parser.add_argument(
        "--prior-mode", choices=("all_history", "r_recent3"), default="all_history"
    )
    return parser.parse_args()


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
    args = parse_args()
    started = time.perf_counter()
    train_path = TRAIN_PATH.resolve()
    train = load_train(train_path)
    prior_candidates = candidate_priors(train, 2025)
    prior = (
        float(train[TARGET].mean())
        if args.prior_mode == "all_history"
        else float(prior_candidates[args.prior_mode])
    )
    states_before, final_state = season_end_state(train)
    priors = prior_before_each_season(train)
    train_e14, e14_meta = build_e14_features(
        train, states_before, priors, prior, k=E14_K
    )
    use_e16 = args.candidate_id == "S5"
    if use_e16:
        role_before, role_final = role_states_before_each_season(train)
        role_before[2025] = role_final
        train_e16, e16_meta = build_e16_features(train, role_before)
        train_features = pd.concat([train[BASE_FEATURES], train_e14, train_e16], axis=1)
    else:
        role_final = None
        e16_meta = None
        train_features = pd.concat([train[BASE_FEATURES], train_e14], axis=1)
    train_target = train[TARGET].to_numpy(dtype=np.int8, copy=False)
    output_dir = OUTPUT_DIR
    record_dir = RECORD_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(
        prefix=f"{args.candidate_id.lower()}_build_", dir=output_dir
    ) as temporary:
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
        role_state_hash = None
        if use_e16:
            role_state_payload = {
                str(int(pitcher)): [int(values[0]), int(values[1])]
                for pitcher, values in sorted(role_final.items())
            }
            role_state_path = model_dir / "e16_role_state.json"
            role_state_path.write_text(
                json.dumps(
                    role_state_payload,
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                ),
                encoding="utf-8",
            )
            role_state_hash = sha256_file(role_state_path)
        linear_factory = make_e16_linear if use_e16 else make_linear
        hgb_factory = make_e16_hgb if use_e16 else make_hgb
        linear_record = fit_and_save(
            f"{args.candidate_id}/{'e16' if use_e16 else 'e14'}/linear",
            linear_factory,
            train_features,
            train_target,
            model_dir / "linear_sgd.joblib",
        )
        hgb_record = fit_and_save(
            f"{args.candidate_id}/{'e16' if use_e16 else 'e14'}/hgb",
            hgb_factory,
            train_features,
            train_target,
            model_dir / "hgb.joblib",
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
            "candidate": args.candidate_id,
            "description": (
                "Full-data Linear 90% + HGB 10% with E14 season-to-date pitcher features"
                + (" and E16 frozen role/home-team context" if use_e16 else "")
                + f" and {args.prior_mode} prior"
            ),
            "build_spec_version": 1,
            "data_cutoff": "season <= 2024",
            "training_seasons": sorted(int(value) for value in train["season"].unique()),
            "training_rows": len(train),
            "training_data_sha256": sha256_file(train_path),
            "training_code_sha256": {
                "baseline": sha256_file(ROOT / "experiments/run_baselines.py"),
                "e14": sha256_file(ROOT / "experiments/run_e14_rolling.py"),
                **(
                    {"e16": sha256_file(ROOT / "experiments/run_e16_rolling.py")}
                    if use_e16
                    else {}
                ),
            },
            "feature_cutoff": (
                "asof_pitcher_* minus frozen state through 2024; E16 role map through 2024"
                if use_e16
                else "asof_pitcher_* minus frozen state through 2024"
            ),
            "prior_source": f"E15 prior mode: {args.prior_mode}; calculated from seasons <= 2024",
            "k_source": "fixed EDA/theoretical E14 value 120; not tuned on validation targets",
            "row_independent_inference": True,
            "base_features": BASE_FEATURES,
            "features": BASE_FEATURES + E14_FEATURES + (E16_FEATURES if use_e16 else []),
            "e14": {
                "features": E14_FEATURES,
                "state_file": state_path.name,
                "state_sha256": state_hash,
                "prior": prior,
                "k": E14_K,
                "state_cutoff": "after season 2024",
            },
            **(
                {
                    "e16": {
                        "features": E16_FEATURES,
                        "role_state_file": "e16_role_state.json",
                        "role_state_sha256": role_state_hash,
                        "state_cutoff": "after season 2024",
                        "home_team_formula": "top_bottom == 'T' ? pitcher_team_id : batter_team_id",
                    }
                }
                if use_e16
                else {}
            ),
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
        output_name = {
            "S3": "S3_e14.zip",
            "S4": "S4_e14_r_recent3.zip",
            "S5": "S5_e16.zip",
        }[args.candidate_id]
        output = output_dir / output_name
        deterministic_zip(stage, output)

    metadata = common_metadata(args.candidate_id, output, started)
    metadata.update(
        {
            "description": manifest["description"],
            "train_path": str(train_path),
            "train_sha256": manifest["training_data_sha256"],
            "training_rows": len(train),
            "training_seasons": manifest["training_seasons"],
            "prior": prior,
            "prior_mode": args.prior_mode,
            "k": E14_K,
            "e14_training_metadata": e14_meta,
            "e16_training_metadata": e16_meta,
            "models": [
                {**linear_record, "weight": 0.9},
                {**hgb_record, "weight": 0.1},
            ],
            "rolling_reference": {
                "result": (
                    "experiments/results/archive/E16_v1/e16_rolling.json"
                    if use_e16
                    else (
                        "experiments/results/archive/E14_v1/e14_rolling.json"
                        if args.prior_mode == "all_history"
                        else "experiments/results/archive/E15_r_recent3_v1/e14_rolling.json"
                    )
                ),
                "prior_mode": args.prior_mode,
                "mean_brier_delta": (
                    0.00000033922651087222217
                    if use_e16
                    else (
                        -0.00033351923970253994
                        if args.prior_mode == "all_history"
                        else -0.00037670581948266263
                    )
                ),
                "e14_wins": "2/3" if use_e16 else "3/3",
                "gate_pass": True,
            },
        }
    )
    record_dir.mkdir(parents=True, exist_ok=True)
    write_json(record_dir / f"{args.candidate_id}_build.json", metadata)
    print(f"Built {args.candidate_id}: {output} ({metadata['zip_sha256']})", flush=True)


if __name__ == "__main__":
    main()
