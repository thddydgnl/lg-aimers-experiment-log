#!/usr/bin/env python3
"""Run the mandatory local gate against a DACON submission ZIP."""

from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
import zipfile
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath

import numpy as np
import pandas as pd


ID_COL = "row_id"
TARGET_COL = "control_success"
REQUIRED_ROOT_FILES = {"script.py", "requirements.txt"}
ALLOWED_ROOTS = {"script.py", "requirements.txt", "model"}
FORBIDDEN_INFERENCE_PATTERNS = {
    "groupby": r"\.groupby\s*\(",
    "rolling": r"\.rolling\s*\(",
    "expanding": r"\.expanding\s*\(",
    "model fit": r"\.fit(?:_transform)?\s*\(",
    "remote requests": r"\b(?:requests|urllib|httpx|socket)\b",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("zip_path", type=Path)
    parser.add_argument("--data-dir", type=Path, default=Path("open/data"))
    parser.add_argument("--mimic-source", type=Path, default=Path("open/data/train.csv"))
    parser.add_argument("--mimic-rows", type=int, default=245_789)
    parser.add_argument("--max-seconds", type=float, default=180.0)
    parser.add_argument("--max-memory-gb", type=float, default=20.0)
    parser.add_argument("--skip-mimic", action="store_true")
    parser.add_argument("--report", type=Path, default=None)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def inspect_zip(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(path)
    if path.stat().st_size > 10 * 1024**3:
        raise ValueError("ZIP exceeds the 10 GiB competition limit.")
    with zipfile.ZipFile(path) as archive:
        members = archive.infolist()
        if not members:
            raise ValueError("ZIP is empty.")
        total_uncompressed = 0
        roots: set[str] = set()
        names: set[str] = set()
        for member in members:
            pure = PurePosixPath(member.filename)
            if pure.is_absolute() or ".." in pure.parts:
                raise ValueError(f"Unsafe ZIP path: {member.filename}")
            if not pure.parts:
                continue
            roots.add(pure.parts[0])
            names.add(member.filename.rstrip("/"))
            total_uncompressed += member.file_size
        missing = REQUIRED_ROOT_FILES - names
        if missing:
            raise ValueError(f"Missing required root files: {sorted(missing)}")
        if "model" not in roots:
            raise ValueError("ZIP does not contain model/ at its root.")
        extra = roots - ALLOWED_ROOTS
        if extra:
            raise ValueError(f"Unexpected top-level ZIP entries: {sorted(extra)}")
        if total_uncompressed > 32 * 1024**3:
            raise ValueError("Uncompressed contents exceed 32 GiB.")
        script = archive.read("script.py").decode("utf-8-sig")
        requirements = archive.read("requirements.txt").decode("utf-8-sig")
    violations = [
        name
        for name, pattern in FORBIDDEN_INFERENCE_PATTERNS.items()
        if re.search(pattern, script, flags=re.IGNORECASE)
    ]
    if violations:
        raise ValueError(f"Static inference scan failed: {violations}")
    required_path_fragments = ("data", "test.csv", "sample_submission.csv", "output")
    if any(fragment not in script for fragment in required_path_fragments):
        raise ValueError("script.py does not visibly use the required relative data/output paths.")
    return {
        "zip_bytes": path.stat().st_size,
        "uncompressed_bytes": total_uncompressed,
        "member_count": len(members),
        "roots": sorted(roots),
        "requirements": requirements.strip(),
        "static_scan": "passed",
    }


def current_working_set(process: subprocess.Popen) -> int | None:
    if os.name != "nt" or not hasattr(process, "_handle"):
        return None

    class ProcessMemoryCounters(ctypes.Structure):
        _fields_ = [
            ("cb", ctypes.c_ulong),
            ("PageFaultCount", ctypes.c_ulong),
            ("PeakWorkingSetSize", ctypes.c_size_t),
            ("WorkingSetSize", ctypes.c_size_t),
            ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
            ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
            ("PagefileUsage", ctypes.c_size_t),
            ("PeakPagefileUsage", ctypes.c_size_t),
        ]

    counters = ProcessMemoryCounters()
    counters.cb = ctypes.sizeof(counters)
    success = ctypes.windll.psapi.GetProcessMemoryInfo(
        int(process._handle), ctypes.byref(counters), counters.cb
    )
    return int(counters.WorkingSetSize) if success else None


def execute(stage: Path, label: str, timeout: float) -> dict:
    output = stage / "output" / "submission.csv"
    if output.exists():
        output.unlink()
    environment = os.environ.copy()
    environment["PYTHONUTF8"] = "1"
    # On Windows, venv's python.exe is a small redirector.  Launching the base
    # executable with the active interpreter paths keeps the same packages while
    # making RSS monitoring observe the real inference process rather than the
    # short-lived redirector.
    executable = sys.executable
    if os.name == "nt" and getattr(sys, "_base_executable", None):
        executable = sys._base_executable
        active_paths = [entry for entry in sys.path if entry]
        inherited = environment.get("PYTHONPATH")
        if inherited:
            active_paths.append(inherited)
        environment["PYTHONPATH"] = os.pathsep.join(active_paths)
    started = time.perf_counter()
    process = subprocess.Popen(
        [executable, "script.py"],
        cwd=stage,
        env=environment,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    peak_working_set = 0
    while process.poll() is None:
        elapsed = time.perf_counter() - started
        if elapsed > timeout:
            process.kill()
            stdout, _ = process.communicate()
            raise TimeoutError(f"{label} exceeded {timeout:.1f}s.\n{stdout}")
        current = current_working_set(process)
        if current is not None:
            peak_working_set = max(peak_working_set, current)
        time.sleep(0.05)
    stdout, _ = process.communicate()
    elapsed = time.perf_counter() - started
    if process.returncode != 0:
        raise RuntimeError(f"{label} failed with exit {process.returncode}.\n{stdout}")
    if not output.is_file():
        raise FileNotFoundError(f"{label} did not create output/submission.csv")
    return {
        "label": label,
        "seconds": elapsed,
        "peak_working_set_bytes": peak_working_set or None,
        "stdout": stdout.strip(),
    }


def validate_output(path: Path, sample: pd.DataFrame) -> pd.DataFrame:
    output = pd.read_csv(path, encoding="utf-8-sig")
    if list(output.columns) != [ID_COL, TARGET_COL]:
        raise ValueError(f"Invalid output columns: {list(output.columns)}")
    if len(output) != len(sample):
        raise ValueError(f"Output rows {len(output):,} != expected {len(sample):,}")
    if output[ID_COL].tolist() != sample[ID_COL].tolist():
        raise ValueError("Output row_id order differs from sample_submission.csv.")
    values = pd.to_numeric(output[TARGET_COL], errors="coerce").to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("Output predictions contain NaN or infinity.")
    if ((values < 0.0) | (values > 1.0)).any():
        raise ValueError("Output predictions are outside [0, 1].")
    return output


def write_case_data(stage: Path, test: pd.DataFrame, sample: pd.DataFrame) -> None:
    data_dir = stage / "data"
    data_dir.mkdir(exist_ok=True)
    test.to_csv(data_dir / "test.csv", index=False, encoding="utf-8")
    sample.to_csv(data_dir / "sample_submission.csv", index=False, encoding="utf-8")


def compare(reference: pd.DataFrame, candidate: pd.DataFrame) -> float:
    left = reference.set_index(ID_COL)[TARGET_COL].sort_index()
    right = candidate.set_index(ID_COL)[TARGET_COL].sort_index()
    if left.index.tolist() != right.index.tolist():
        raise ValueError("Invariant test ID sets differ.")
    return float(np.max(np.abs(left.to_numpy(dtype=float) - right.to_numpy(dtype=float))))


def run_invariance_gate(
    stage: Path, test: pd.DataFrame, sample: pd.DataFrame, timeout: float
) -> tuple[list[dict], float, pd.DataFrame]:
    runs: list[dict] = []
    write_case_data(stage, test, sample)
    runs.append(execute(stage, "sample batch", timeout))
    reference = validate_output(stage / "output" / "submission.csv", sample)
    maximum_delta = 0.0

    shuffled = test.sample(frac=1.0, random_state=42).reset_index(drop=True)
    write_case_data(stage, shuffled, sample)
    runs.append(execute(stage, "shuffled batch", timeout))
    candidate = validate_output(stage / "output" / "submission.csv", sample)
    maximum_delta = max(maximum_delta, compare(reference, candidate))

    duplicated = pd.concat([test, test.iloc[[0]]], ignore_index=True)
    write_case_data(stage, duplicated, sample)
    runs.append(execute(stage, "duplicate insertion", timeout))
    candidate = validate_output(stage / "output" / "submission.csv", sample)
    maximum_delta = max(maximum_delta, compare(reference, candidate))

    for index in range(len(sample)):
        row_id = sample.iloc[index][ID_COL]
        one_test = test.loc[test[ID_COL] == row_id].iloc[[0]].copy()
        one_sample = sample.iloc[[index]].copy()
        write_case_data(stage, one_test, one_sample)
        runs.append(execute(stage, f"single row {index + 1}", timeout))
        candidate = validate_output(stage / "output" / "submission.csv", one_sample)
        expected = reference.iloc[[index]].copy()
        maximum_delta = max(maximum_delta, compare(expected, candidate))

    if maximum_delta >= 1e-12:
        raise AssertionError(f"Row invariance delta {maximum_delta:.3e} >= 1e-12")
    return runs, maximum_delta, reference


def make_mimic(
    source_path: Path, test_columns: list[str], rows: int
) -> tuple[pd.DataFrame, pd.DataFrame]:
    source = pd.read_csv(source_path, encoding="utf-8-sig")
    missing = [column for column in test_columns if column not in source.columns]
    if missing:
        raise ValueError(f"Mimic source is missing test columns: {missing}")
    if len(source) < rows:
        raise ValueError(f"Mimic source has only {len(source):,} rows; requested {rows:,}")
    mimic = source.sample(n=rows, random_state=42).loc[:, test_columns].reset_index(drop=True)
    mimic[ID_COL] = [f"MIMIC_{index:06d}" for index in range(rows)]
    sample = pd.DataFrame({ID_COL: mimic[ID_COL], TARGET_COL: np.full(rows, 0.5)})
    return mimic, sample


def extract_safely(zip_path: Path, destination: Path) -> None:
    with zipfile.ZipFile(zip_path) as archive:
        destination_resolved = destination.resolve()
        for member in archive.infolist():
            target = (destination / member.filename).resolve()
            if destination_resolved != target and destination_resolved not in target.parents:
                raise ValueError(f"Unsafe ZIP extraction path: {member.filename}")
        archive.extractall(destination)


def main() -> None:
    args = parse_args()
    zip_path = args.zip_path.resolve()
    data_dir = args.data_dir.resolve()
    test = pd.read_csv(data_dir / "test.csv", encoding="utf-8-sig")
    sample = pd.read_csv(data_dir / "sample_submission.csv", encoding="utf-8-sig")
    report = {
        "verified_at_utc": datetime.now(timezone.utc).isoformat(),
        "zip_path": str(zip_path),
        "zip_sha256": sha256_file(zip_path),
        "python": sys.version,
        "checks": inspect_zip(zip_path),
    }
    with tempfile.TemporaryDirectory(prefix="submission_verify_") as temporary:
        stage = Path(temporary)
        extract_safely(zip_path, stage)
        runs, maximum_delta, reference = run_invariance_gate(
            stage, test, sample, args.max_seconds
        )
        report["sample"] = {
            "rows": len(reference),
            "prediction_mean": float(reference[TARGET_COL].mean()),
            "prediction_min": float(reference[TARGET_COL].min()),
            "prediction_max": float(reference[TARGET_COL].max()),
            "invariance_max_abs_delta": maximum_delta,
            "runs": runs,
        }
        if not args.skip_mimic:
            mimic, mimic_sample = make_mimic(
                args.mimic_source.resolve(), test.columns.tolist(), args.mimic_rows
            )
            write_case_data(stage, mimic, mimic_sample)
            mimic_run = execute(stage, f"{args.mimic_rows:,}-row mimic", args.max_seconds)
            mimic_output = validate_output(
                stage / "output" / "submission.csv", mimic_sample
            )
            peak = mimic_run["peak_working_set_bytes"]
            if peak is not None and peak > args.max_memory_gb * 1024**3:
                raise MemoryError(
                    f"Peak working set {peak / 1024**3:.2f} GiB exceeds "
                    f"{args.max_memory_gb:.2f} GiB."
                )
            report["mimic"] = {
                "rows": len(mimic_output),
                "prediction_mean": float(mimic_output[TARGET_COL].mean()),
                "prediction_min": float(mimic_output[TARGET_COL].min()),
                "prediction_max": float(mimic_output[TARGET_COL].max()),
                "run": mimic_run,
                "time_limit_seconds": args.max_seconds,
                "memory_limit_gb": args.max_memory_gb,
            }

    report["status"] = "PASSED"
    report_path = args.report or zip_path.with_suffix(".verification.json")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    print(
        f"PASSED {zip_path.name}: SHA-256={report['zip_sha256']}, "
        f"invariance={report['sample']['invariance_max_abs_delta']:.3e}",
        flush=True,
    )
    if "mimic" in report:
        run = report["mimic"]["run"]
        peak = run["peak_working_set_bytes"]
        peak_text = "unavailable" if peak is None else f"{peak / 1024**3:.2f} GiB"
        print(
            f"Mimic: rows={report['mimic']['rows']:,}, "
            f"time={run['seconds']:.2f}s, peak={peak_text}",
            flush=True,
        )
    print(f"Report: {report_path}", flush=True)


if __name__ == "__main__":
    main()
