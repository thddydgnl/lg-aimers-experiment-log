#!/usr/bin/env python3
"""Run heavy pipeline stages on Kaggle's free compute, then bring results home.

WHY THIS EXISTS
    The local box is a Ryzen 5 5600 (6C/12T).  Kaggle's CPU is not faster, so the
    win is not raw speed - it is (a) a free NVIDIA GPU, which CatBoost actually
    uses well via task_type="GPU", (b) 12-hour unattended sessions, and (c) a
    second machine running while the local one keeps working.  LightGBM on this
    dataset is fine locally; CatBoost is the stage worth offloading.

RULE WARNING - READ BEFORE FIRST USE
    COMPETITION.md section 9.3: competition data may be used only for this
    competition, and may not be transmitted, copied, or redistributed to
    non-participants.  Uploading train.csv to Kaggle means putting competition
    data on a third-party service.

    Therefore this module refuses to create anything public.  Datasets and
    kernels are forced private, and `--i-understand-rule-9-3` must be passed once
    to acknowledge the transfer.  Never flip these to public: a public Kaggle
    dataset of this data, or a public notebook of this code, is a clear rule
    violation and the organisers have been removing scores for less.

    If you are not comfortable with that reading, run everything locally.  The
    pipeline works without Kaggle; only the CatBoost stages are slower.

SETUP (once)
    pip install kaggle
    Kaggle -> Account -> API -> "Create New Token"  downloads kaggle.json
    Windows: move it to %USERPROFILE%\\.kaggle\\kaggle.json
    python kaggle_offload/offload.py --setup --i-understand-rule-9-3
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
HERE = Path(__file__).resolve().parent
CONFIG_PATH = HERE / "config.json"

DATA_SLUG = "lgaimers-data"
CODE_SLUG = "lgaimers-code"
DATA_FILES = ["train.csv", "test.csv", "sample_submission.csv", "trackman_history.csv"]
CODE_DIRS = ["experiments", "submission"]
CODE_SKIP = {"__pycache__", "archive", "dist", "artifacts", "logs", "_smoke", "predictions"}
POLL_SECONDS = 60
DEFAULT_TIMEOUT_SECONDS = 11 * 3600


# --------------------------------------------------------------------------- #
# CLI plumbing
# --------------------------------------------------------------------------- #
def kaggle(*args: str, check: bool = True, capture: bool = True) -> subprocess.CompletedProcess:
    command = [sys.executable, "-m", "kaggle", *args]
    environment = {**os.environ, "PYTHONUTF8": "1", "PYTHONIOENCODING": "utf-8"}
    result = subprocess.run(
        command,
        cwd=ROOT,
        text=True,
        encoding="utf-8",
        errors="replace",
        capture_output=capture,
        env=environment,
    )
    if check and result.returncode != 0:
        detail = (result.stdout or "") + (result.stderr or "")
        raise RuntimeError(f"kaggle {' '.join(args)} failed:\n{detail.strip()}")
    return result


def load_config() -> dict:
    if not CONFIG_PATH.is_file():
        raise RuntimeError(
            "kaggle_offload/config.json 이 없습니다. 먼저 실행하세요:\n"
            "  python kaggle_offload/offload.py --setup --i-understand-rule-9-3"
        )
    return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))


def resolve_username() -> str:
    for candidate in (
        Path(os.environ.get("KAGGLE_CONFIG_DIR", "")) / "kaggle.json" if os.environ.get("KAGGLE_CONFIG_DIR") else None,
        Path.home() / ".kaggle" / "kaggle.json",
    ):
        if candidate and candidate.is_file():
            return json.loads(candidate.read_text(encoding="utf-8"))["username"]
    if os.environ.get("KAGGLE_USERNAME"):
        return os.environ["KAGGLE_USERNAME"]
    raise RuntimeError(
        "kaggle.json 을 찾지 못했습니다. Kaggle > Account > API > Create New Token 후 "
        r"%USERPROFILE%\.kaggle\kaggle.json 에 두세요."
    )


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


# --------------------------------------------------------------------------- #
# Private dataset sync
# --------------------------------------------------------------------------- #
def push_dataset(stage_dir: Path, owner: str, slug: str, title: str) -> str:
    reference = f"{owner}/{slug}"
    write_json(
        stage_dir / "dataset-metadata.json",
        {"title": title, "id": reference, "licenses": [{"name": "other"}]},
    )
    existing = kaggle("datasets", "list", "-m", "-s", slug, check=False)
    if reference in (existing.stdout or ""):
        print(f"  updating {reference} ...", flush=True)
        kaggle("datasets", "version", "-p", str(stage_dir), "-m",
               f"pipeline sync {datetime.now(timezone.utc).isoformat()}",
               "--dir-mode", "zip", capture=False)
    else:
        print(f"  creating private {reference} ...", flush=True)
        # Kaggle datasets are private by default; we never pass --public.
        kaggle("datasets", "create", "-p", str(stage_dir), "--dir-mode", "zip", capture=False)
    return reference


def sync_data(owner: str) -> str:
    source = ROOT / "open" / "data"
    with tempfile.TemporaryDirectory(prefix="kag_data_") as temporary:
        stage_dir = Path(temporary)
        for name in DATA_FILES:
            path = source / name
            if not path.is_file():
                raise FileNotFoundError(path)
            shutil.copy2(path, stage_dir / name)
        return push_dataset(stage_dir, owner, DATA_SLUG, "LG Aimers private data")


def sync_code(owner: str) -> str:
    with tempfile.TemporaryDirectory(prefix="kag_code_") as temporary:
        stage_dir = Path(temporary)
        for directory in CODE_DIRS:
            source = ROOT / directory
            if not source.is_dir():
                continue
            shutil.copytree(
                source,
                stage_dir / directory,
                ignore=shutil.ignore_patterns(*CODE_SKIP, "*.pyc", "*.joblib", "*.zip"),
            )
        (stage_dir / "experiments" / "params").mkdir(parents=True, exist_ok=True)
        return push_dataset(stage_dir, owner, CODE_SLUG, "LG Aimers private code")


# --------------------------------------------------------------------------- #
# Kernel execution
# --------------------------------------------------------------------------- #
KERNEL_TEMPLATE = '''"""Auto-generated Kaggle runner. Do not edit here; edit kaggle_offload/offload.py."""
import os, shutil, subprocess, sys, json
from pathlib import Path

INPUT = Path("/kaggle/input")
WORK = Path("/kaggle/working")
REPO = WORK / "repo"

# Rebuild the repo layout the scripts expect: writable code, read-only data.
if REPO.exists():
    shutil.rmtree(REPO)
REPO.mkdir(parents=True)
for name in ("experiments", "submission"):
    source = INPUT / {code_slug!r} / name
    if source.is_dir():
        shutil.copytree(source, REPO / name)
(REPO / "open").mkdir(exist_ok=True)
data_dir = REPO / "open" / "data"
data_dir.mkdir(exist_ok=True)
for item in (INPUT / {data_slug!r}).glob("*.csv"):
    os.symlink(item, data_dir / item.name)

argv = {argv!r}
argv = [a.replace({root_marker!r}, str(REPO)) for a in argv]

env = dict(
    os.environ,
    PYTHONUTF8="1",
    PYTHONIOENCODING="utf-8",
    PYTHONPATH=str(REPO),
    V2_BOOSTER_DEVICE={device_value!r},
)
print("RUNNING:", " ".join(argv), flush=True)
code = subprocess.call([sys.executable, *argv], cwd=str(REPO), env=env)
print("EXIT:", code, flush=True)

# Everything the pipeline needs to pull back lives directly in /kaggle/working.
out = WORK / "results"
if out.exists():
    shutil.rmtree(out)
shutil.copytree(REPO / "experiments" / "results", out, dirs_exist_ok=True)
out_params = WORK / "params"
if out_params.exists():
    shutil.rmtree(out_params)
shutil.copytree(REPO / "experiments" / "params", out_params, dirs_exist_ok=True)
(WORK / "exit_code.json").write_text(json.dumps({{"returncode": code}}), encoding="utf-8")
if code != 0:
    raise SystemExit(code)
'''


def slugify(key: str) -> str:
    return re.sub(r"[^a-z0-9-]+", "-", f"lgaimers-{key}".lower()).strip("-")[:48]


def run_stage_on_kaggle(
    key: str,
    argv: list[str],
    device: str,
    log_dir: Path,
    timeout_seconds: int = DEFAULT_TIMEOUT_SECONDS,
) -> dict:
    config = load_config()
    owner = config["username"]
    slug = slugify(key)
    reference = f"{owner}/{slug}"
    started = time.perf_counter()
    log_dir.mkdir(parents=True, exist_ok=True)

    print("  syncing private datasets ...", flush=True)
    data_ref = sync_data(owner)
    code_ref = sync_code(owner)

    with tempfile.TemporaryDirectory(prefix="kag_kernel_") as temporary:
        stage_dir = Path(temporary)
        script = KERNEL_TEMPLATE.format(
            code_slug=CODE_SLUG,
            data_slug=DATA_SLUG,
            argv=argv,
            root_marker=str(ROOT),
            device_value=device,
        )
        (stage_dir / "run.py").write_text(script, encoding="utf-8")
        write_json(
            stage_dir / "kernel-metadata.json",
            {
                "id": reference,
                "title": slug,
                "code_file": "run.py",
                "language": "python",
                "kernel_type": "script",
                "is_private": True,          # never make this public - rule 9.3
                "enable_gpu": device == "gpu",
                "enable_internet": False,
                "dataset_sources": [data_ref, code_ref],
                "competition_sources": [],
                "kernel_sources": [],
            },
        )
        print(f"  pushing {reference} (gpu={device == 'gpu'}) ...", flush=True)
        kaggle("kernels", "push", "-p", str(stage_dir), capture=False)

    deadline = time.time() + timeout_seconds
    status = "unknown"
    while time.time() < deadline:
        time.sleep(POLL_SECONDS)
        result = kaggle("kernels", "status", reference, check=False)
        blob = ((result.stdout or "") + (result.stderr or "")).lower()
        if "complete" in blob:
            status = "complete"
            break
        if "error" in blob or "cancel" in blob:
            status = "error"
            break
        elapsed = (time.perf_counter() - started) / 60
        print(f"  [{elapsed:5.1f}m] {blob.strip()[:90]}", flush=True)
    else:
        status = "timeout"

    output_dir = ROOT / "experiments" / "kaggle_output" / key
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    kaggle("kernels", "output", reference, "-p", str(output_dir), check=False, capture=False)

    returncode = 1
    exit_marker = output_dir / "exit_code.json"
    if exit_marker.is_file():
        returncode = int(json.loads(exit_marker.read_text(encoding="utf-8"))["returncode"])

    pulled = output_dir / "results"
    if pulled.is_dir():
        shutil.copytree(pulled, ROOT / "experiments" / "results", dirs_exist_ok=True)
        print(f"  merged Kaggle results into experiments/results", flush=True)

    pulled_params = output_dir / "params"
    if pulled_params.is_dir():
        shutil.copytree(
            pulled_params, ROOT / "experiments" / "params", dirs_exist_ok=True
        )
        print("  merged Kaggle params into experiments/params", flush=True)

    log_path = log_dir / f"{key}.kaggle.log"
    log_source = output_dir / f"{slug}.log"
    if log_source.is_file():
        shutil.copy2(log_source, log_path)

    return {
        "returncode": 0 if (status == "complete" and returncode == 0) else 1,
        "elapsed_seconds": time.perf_counter() - started,
        "where": "kaggle",
        "kernel": reference,
        "kernel_status": status,
        "device": device,
        "log": str(log_path),
        "output_dir": str(output_dir),
    }


# --------------------------------------------------------------------------- #
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--setup", action="store_true", help="Verify the CLI and write config.json.")
    parser.add_argument("--sync", action="store_true", help="Push the private data/code datasets now.")
    parser.add_argument("--i-understand-rule-9-3", dest="acknowledged", action="store_true",
                        help="Acknowledge that competition data will be copied to Kaggle (private).")
    args = parser.parse_args()

    if args.setup or args.sync:
        if not args.acknowledged:
            raise SystemExit(
                "\n대회 데이터를 외부 서비스(Kaggle)에 복사하게 됩니다.\n"
                "COMPETITION.md §9.3 을 읽고, 데이터셋·노트북을 절대 public 으로 바꾸지 않겠다면\n"
                "--i-understand-rule-9-3 을 함께 지정하세요.\n"
            )
        version = kaggle("--version", check=False)
        if version.returncode != 0:
            raise SystemExit("kaggle CLI 가 없습니다:  pip install kaggle")
        username = resolve_username()
        write_json(CONFIG_PATH, {
            "username": username,
            "acknowledged_rule_9_3_at_utc": datetime.now(timezone.utc).isoformat(),
            "data_dataset": f"{username}/{DATA_SLUG}",
            "code_dataset": f"{username}/{CODE_SLUG}",
            "visibility": "private",
        })
        print(f"OK — kaggle user: {username}", flush=True)
        print(f"설정 저장: {CONFIG_PATH}", flush=True)
        if args.sync:
            sync_data(username)
            sync_code(username)
            print("데이터·코드 동기화 완료 (둘 다 private).", flush=True)
        return

    parser.print_help()


if __name__ == "__main__":
    main()
