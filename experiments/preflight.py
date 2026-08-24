#!/usr/bin/env python3
"""Fail fast on a broken environment instead of dying an hour into training.

Pinned versions are load-bearing here: numpy/scipy/scikit-learn/joblib must match
requirements-baseline.txt or the frozen S1-S8 artifacts stop reproducing, and
that reproducibility is what Phase 3 code verification checks.  So optional
extras are installed under `--constraint requirements-baseline.txt`, and the S4
archive is re-verified immediately afterwards.  If that check fails, roll back.
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MPL_CACHE = ROOT / "experiments" / "_cache" / "matplotlib"
MPL_CACHE.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPL_CACHE))
CONSTRAINT = ROOT / "requirements-baseline.txt"
S4_ZIP = ROOT / "submission" / "archive" / "S4" / "S4.zip"

PINNED = {
    "numpy": "1.26.4",
    "pandas": "2.0.3",
    "scipy": "1.15.3",
    "sklearn": "1.8.0",
    "joblib": "1.5.3",
}
OPTIONAL = {
    "lightgbm": "lightgbm==4.7.0",   # sklearn 1.8-compatible Track C runtime
    "catboost": "catboost==1.2.8",   # stages c3, c3b - optional, install has failed before
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--install", action="store_true",
                        help="pip install missing optional packages under the constraint file.")
    parser.add_argument("--require", nargs="*", default=["lightgbm"],
                        help="Optional packages whose absence is fatal.")
    parser.add_argument("--skip-reverify", action="store_true",
                        help="Skip the S4 reproducibility re-check after installing.")
    return parser.parse_args()


def utf8_env() -> dict[str, str]:
    return {
        **os.environ,
        "PYTHONUTF8": "1",
        "PYTHONIOENCODING": "utf-8",
        "MPLCONFIGDIR": str(MPL_CACHE),
    }


def version_of(module: str) -> str | None:
    try:
        return importlib.import_module(module).__version__
    except Exception:  # noqa: BLE001 - any import failure means "not usable"
        return None


def pip(*args: str) -> int:
    print(f"  $ pip {' '.join(args)}", flush=True)
    return subprocess.run(
        [sys.executable, "-m", "pip", "--disable-pip-version-check", *args],
        cwd=ROOT, env=utf8_env(),
    ).returncode


def main() -> None:
    args = parse_args()
    report: dict = {"python": sys.version.split()[0], "executable": sys.executable}
    problems: list[str] = []

    print(f"Python {report['python']}  ({sys.executable})", flush=True)
    if not sys.version.startswith("3.11"):
        problems.append(f"Python 3.11이 아닙니다: {report['python']} — .venv 를 쓰고 있는지 확인하세요.")

    print("\n고정 버전 확인:", flush=True)
    report["pinned"] = {}
    for module, expected in PINNED.items():
        found = version_of(module)
        report["pinned"][module] = found
        mark = "OK " if found == expected else "!! "
        print(f"  {mark}{module:<14} {found or '없음'}  (기대 {expected})", flush=True)
        if found != expected:
            problems.append(
                f"{module} 버전이 {found} 입니다. {expected} 이어야 S1~S8 재현성이 유지됩니다."
            )

    print("\n선택 패키지:", flush=True)
    installed_anything = False
    report["optional"] = {}
    for module, requirement in OPTIONAL.items():
        found = version_of(module)
        if found is None and args.install and module in args.require:
            print(f"  {module} 설치 시도 ...", flush=True)
            if pip("install", "--constraint", str(CONSTRAINT), requirement) == 0:
                found = version_of(module)
                installed_anything = found is not None
            else:
                print(f"  {module} 설치 실패", flush=True)
        report["optional"][module] = found
        state = found or "없음"
        required = module in args.require
        mark = "OK " if found else ("!! " if required else "-- ")
        print(f"  {mark}{module:<14} {state}", flush=True)
        if found is None and required:
            problems.append(
                f"{module} 이 없습니다. 실행:\n"
                f'      & .\\.venv\\Scripts\\python.exe -m pip install --constraint '
                f'requirements-baseline.txt "{requirement}"\n'
                f"    또는 preflight 를 --install 과 함께 실행하세요."
            )

    if report["optional"].get("lightgbm") is not None:
        print("\nLightGBM project-wrapper smoke test:", flush=True)
        smoke = subprocess.run(
            [sys.executable, str(ROOT / "experiments" / "smoke_lgbm.py")],
            cwd=ROOT,
            env=utf8_env(),
            capture_output=True,
            text=True,
        )
        report["lightgbm_smoke"] = smoke.returncode == 0
        if smoke.stdout.strip():
            print(f"  {smoke.stdout.strip()}", flush=True)
        if smoke.returncode != 0:
            detail = smoke.stderr.strip() or "unknown smoke-test error"
            problems.append(f"LightGBM project-wrapper smoke test failed: {detail}")

    if installed_anything:
        print("\n의존성 정합성 확인:", flush=True)
        if pip("check") != 0:
            problems.append("pip check 실패 — 방금 설치가 기존 패키지를 깼습니다. 롤백하세요.")
        for module, expected in PINNED.items():
            if version_of(module) != expected:
                problems.append(
                    f"설치 후 {module} 이 {version_of(module)} 로 바뀌었습니다. "
                    f"즉시 롤백: pip install {module}=={expected}"
                )

        if not args.skip_reverify and S4_ZIP.is_file() and not problems:
            print("\nS4 재현성 재확인 (필수):", flush=True)
            code = subprocess.run(
                [sys.executable, str(ROOT / "submission" / "verify_submission.py"),
                 str(S4_ZIP), "--skip-mimic"],
                cwd=ROOT, env=utf8_env(),
            ).returncode
            report["s4_reverified"] = code == 0
            if code != 0:
                problems.append(
                    "S4 ZIP 재검증에 실패했습니다. 방금 설치를 롤백하고 다시 확인하세요."
                )

    (ROOT / "experiments" / "results").mkdir(parents=True, exist_ok=True)
    (ROOT / "experiments" / "results" / "preflight.json").write_text(
        json.dumps({**report, "problems": problems}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    if problems:
        print("\n" + "=" * 70, flush=True)
        print("환경 점검 실패:", flush=True)
        for problem in problems:
            print(f"  - {problem}", flush=True)
        print("=" * 70, flush=True)
        raise SystemExit(1)
    print("\n환경 점검 통과.", flush=True)


if __name__ == "__main__":
    main()
