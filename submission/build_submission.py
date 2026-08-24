#!/usr/bin/env python3
"""Build reproducible S1 and S2 DACON submission ZIP files."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import shutil
import sys
import tempfile
import time
import zipfile
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

from experiments.run_baselines import (  # noqa: E402
    FEATURES,
    SEASON,
    TARGET,
    load_train,
    make_hist_gradient_boosting,
    make_linear_sgd,
)


S1_SOURCE = ROOT / "open" / "baseline_submit.zip"
TRAIN_PATH = ROOT / "open" / "data" / "train.csv"
TEMPLATE_DIR = Path(__file__).resolve().parent / "template"
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "dist"
DEFAULT_RECORD_DIR = Path(__file__).resolve().parent / "records"
FIXED_ZIP_TIME = (2026, 8, 17, 0, 0, 0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", choices=("s1", "s2", "all"), default="all")
    parser.add_argument("--train", type=Path, default=TRAIN_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--record-dir", type=Path, default=DEFAULT_RECORD_DIR)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )


def deterministic_zip(source: Path, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for path in sorted(item for item in source.rglob("*") if item.is_file()):
            relative = path.relative_to(source).as_posix()
            info = zipfile.ZipInfo(relative, FIXED_ZIP_TIME)
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o644 << 16
            archive.writestr(info, path.read_bytes(), compress_type=zipfile.ZIP_DEFLATED)


def common_metadata(candidate: str, output: Path, started: float) -> dict:
    return {
        "candidate": candidate,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "output": str(output),
        "zip_sha256": sha256_file(output),
        "zip_bytes": output.stat().st_size,
        "elapsed_seconds": time.perf_counter() - started,
        "versions": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scipy": scipy.__version__,
            "scikit_learn": sklearn.__version__,
            "joblib": joblib.__version__,
        },
        "command": " ".join(sys.argv),
    }


def build_s1(output_dir: Path, record_dir: Path) -> Path:
    started = time.perf_counter()
    if not S1_SOURCE.is_file():
        raise FileNotFoundError(S1_SOURCE)
    output = output_dir / "S1_official_frozen_rf.zip"
    output.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(S1_SOURCE, output)
    if sha256_file(output) != sha256_file(S1_SOURCE):
        raise AssertionError("S1 copy is not byte-identical to the official ZIP.")
    metadata = common_metadata("S1", output, started)
    metadata.update(
        {
            "source": str(S1_SOURCE),
            "source_sha256": sha256_file(S1_SOURCE),
            "byte_identical_to_source": True,
            "model": "official frozen rf.pkl",
            "training_cutoff": "unknown (official frozen artifact)",
        }
    )
    write_json(record_dir / "S1_build.json", metadata)
    print(f"Built S1: {output} ({metadata['zip_sha256']})", flush=True)
    return output


def fit_and_save(label: str, model, train: pd.DataFrame, path: Path) -> dict:
    print(f"[{label}] fitting on {len(train):,} rows...", flush=True)
    started = time.perf_counter()
    model.fit(train[FEATURES], train[TARGET])
    fit_seconds = time.perf_counter() - started
    joblib.dump(model, path, compress=3)
    record = {
        "file": path.name,
        "sha256": sha256_file(path),
        "bytes": path.stat().st_size,
        "fit_seconds": fit_seconds,
    }
    print(
        f"[{label}] saved {path.name}: {record['bytes'] / 2**20:.2f} MiB, "
        f"fit={fit_seconds:.1f}s",
        flush=True,
    )
    return record


def build_s2(train_path: Path, output_dir: Path, record_dir: Path) -> Path:
    started = time.perf_counter()
    train_path = train_path.resolve()
    train = load_train(train_path)
    seasons = sorted(int(value) for value in train[SEASON].unique())
    if not seasons or seasons[-1] > 2024:
        raise ValueError(f"S2 training must stop at 2024; found seasons={seasons}")
    output_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="s2_build_", dir=output_dir) as temporary:
        stage = Path(temporary)
        model_dir = stage / "model"
        model_dir.mkdir()
        shutil.copyfile(TEMPLATE_DIR / "script.py", stage / "script.py")
        shutil.copyfile(TEMPLATE_DIR / "requirements.txt", stage / "requirements.txt")

        linear_record = fit_and_save(
            "linear", make_linear_sgd(), train, model_dir / "linear_sgd.joblib"
        )
        hgb_record = fit_and_save(
            "hgb", make_hist_gradient_boosting(), train, model_dir / "hgb.joblib"
        )
        train_hash = sha256_file(train_path)
        model_specifications = [
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
            "candidate": "S2",
            "description": "Full-data Linear SGD 90% + HGB 10%",
            "build_spec_version": 1,
            "data_cutoff": "season <= 2024",
            "training_seasons": seasons,
            "training_rows": len(train),
            "training_data_sha256": train_hash,
            "training_code_sha256": sha256_file(ROOT / "experiments" / "run_baselines.py"),
            "feature_cutoff": "All asof_* values are supplied point-in-time features",
            "prior_source": "not applicable",
            "k_source": "not applicable",
            "row_independent_inference": True,
            "features": FEATURES,
            "models": model_specifications,
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
        output = output_dir / "S2_linear90_hgb10.zip"
        deterministic_zip(stage, output)

    metadata = common_metadata("S2", output, started)
    metadata.update(
        {
            "description": manifest["description"],
            "train_path": str(train_path),
            "train_sha256": train_hash,
            "training_seasons": seasons,
            "training_rows": len(train),
            "models": [
                {**linear_record, "weight": 0.9},
                {**hgb_record, "weight": 0.1},
            ],
            "requirements_strategy": "competition preinstalled packages; no pip downloads",
        }
    )
    write_json(record_dir / "S2_build.json", metadata)
    print(f"Built S2: {output} ({metadata['zip_sha256']})", flush=True)
    return output


def main() -> None:
    args = parse_args()
    if args.candidate in ("s1", "all"):
        build_s1(args.output_dir, args.record_dir)
    if args.candidate in ("s2", "all"):
        build_s2(args.train, args.output_dir, args.record_dir)


if __name__ == "__main__":
    main()
