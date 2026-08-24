#!/usr/bin/env python3
"""Track A: emit the reweighted blend candidates and gate every one of them.

The S4 archive already contains both trained models, so this stage never trains
anything - it rewrites two numbers in model/manifest.json.  Each output ZIP is
then put through the full submission gate, because the thing we are willing to
take risk on is the modelling, never the rules.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from submission.reweight_candidate import reweight  # noqa: E402

# (candidate name, linear weight).  Evidence: CALIBRATION_ENSEMBLE_REPORT.md §4,
# where HGB raw beats the shipped 90:10 by 175.9 points on the 2024 fold.
CANDIDATES = [
    ("S9_s4_hgb50", 0.5),
    ("S10_s4_hgb80", 0.2),
    ("S11_s4_hgb100", 0.0),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=ROOT / "submission/archive/S4/S4.zip")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "submission/dist")
    parser.add_argument("--record-dir", type=Path, default=ROOT / "submission/records")
    parser.add_argument("--skip-verify", action="store_true",
                        help="Build without running the gate. Never use before submitting.")
    return parser.parse_args()


def verify(zip_path: Path) -> dict:
    started = time.perf_counter()
    process = subprocess.run(
        [sys.executable, str(ROOT / "submission" / "verify_submission.py"), str(zip_path)],
        cwd=ROOT,
        text=True,
        encoding="utf-8",
        errors="replace",
        capture_output=True,
        env={**_utf8_env()},
    )
    output = (process.stdout or "") + (process.stderr or "")
    print(output.strip(), flush=True)
    return {
        "passed": process.returncode == 0,
        "seconds": time.perf_counter() - started,
        "stdout": output.strip()[-2000:],
    }


def _utf8_env() -> dict[str, str]:
    import os

    environment = os.environ.copy()
    environment["PYTHONUTF8"] = "1"
    environment["PYTHONIOENCODING"] = "utf-8"
    return environment


def main() -> None:
    args = parse_args()
    if not args.source.is_file():
        raise SystemExit(f"소스 ZIP이 없습니다: {args.source}")

    report = {"source": str(args.source), "candidates": []}
    failures = []
    for name, linear_weight in CANDIDATES:
        print(f"\n--- {name}: linear={linear_weight:g}, hgb={1 - linear_weight:g} ---", flush=True)
        zip_path = reweight(args.source, linear_weight, name, args.output_dir, args.record_dir)
        entry = {
            "candidate": name,
            "linear_weight": linear_weight,
            "hgb_weight": 1.0 - linear_weight,
            "zip": str(zip_path),
        }
        if not args.skip_verify:
            entry["gate"] = verify(zip_path)
            if not entry["gate"]["passed"]:
                failures.append(name)
        report["candidates"].append(entry)

    out = args.output_dir / "blend_sweep.json"
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nSaved {out}", flush=True)

    if failures:
        raise SystemExit(f"게이트 실패: {failures} — 이 후보는 절대 제출하지 마세요.")
    print(f"{len(CANDIDATES)}개 후보 모두 게이트 통과.", flush=True)


if __name__ == "__main__":
    main()
