#!/usr/bin/env python3
"""Final stage: gate every built ZIP and emit one ordered submission queue.

The pipeline never submits.  It ends here, with a checked list the human works
through by hand.  Ordering matters because the daily cap is 5 and the DACON rule
is that your best score is kept, so the queue is sorted by expected value: the
strongest candidate goes first on each day, and nothing unverified is listed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DIST = ROOT / "submission" / "dist"
DAILY_LIMIT = 5

# Prior LB results, so the queue can show what a candidate must beat.
KNOWN_SCORES = {
    "S1": 549.5119345223, "S2": 527.6161010151, "S3": 662.3418227385,
    "S4": 689.2244587204, "S5": 688.1692139081, "S6": 687.2564723096,
    "S8": 689.3999289563,
}
CHAMPION = 689.3999289563


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dist", type=Path, default=DIST)
    parser.add_argument("--ensemble", type=Path, default=ROOT / "experiments/results/v2_ensemble.json")
    parser.add_argument("--blend-sweep", type=Path, default=DIST / "blend_sweep.json")
    parser.add_argument(
        "--package-index", type=Path,
        default=ROOT / "submission" / "records" / "v2_package_index.json",
    )
    parser.add_argument("--top", type=int, default=8)
    parser.add_argument("--output", type=Path, default=ROOT / "submission" / "SUBMIT_QUEUE.md")
    parser.add_argument("--reverify", action="store_true",
                        help="Re-run the gate even for ZIPs that already have a report.")
    return parser.parse_args()


def utf8_env() -> dict[str, str]:
    return {**os.environ, "PYTHONUTF8": "1", "PYTHONIOENCODING": "utf-8"}


def verify(zip_path: Path) -> dict:
    process = subprocess.run(
        [sys.executable, str(ROOT / "submission" / "verify_submission.py"), str(zip_path)],
        cwd=ROOT, env=utf8_env(), text=True, encoding="utf-8",
        errors="replace", capture_output=True,
    )
    return {"passed": process.returncode == 0,
            "output": ((process.stdout or "") + (process.stderr or "")).strip()[-1500:]}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def collect(args: argparse.Namespace) -> list[dict]:
    entries: list[dict] = []
    for zip_path in sorted(args.dist.glob("*.zip")):
        report_path = zip_path.with_suffix(".verification.json")
        current_sha = sha256_file(zip_path)
        report_sha = ""
        if report_path.is_file():
            try:
                report_sha = json.loads(report_path.read_text(encoding="utf-8")).get(
                    "zip_sha256", ""
                )
            except (json.JSONDecodeError, OSError):
                report_sha = ""
        if args.reverify or not report_path.is_file() or report_sha != current_sha:
            print(f"  게이트 실행: {zip_path.name}", flush=True)
            outcome = verify(zip_path)
            if not outcome["passed"]:
                entries.append({"name": zip_path.stem, "zip": zip_path, "passed": False,
                                "detail": outcome["output"]})
                continue
        if not report_path.is_file():
            continue
        report = json.loads(report_path.read_text(encoding="utf-8"))
        if report.get("zip_sha256") != current_sha:
            entries.append({
                "name": zip_path.stem, "zip": zip_path, "passed": False,
                "detail": "verification report SHA-256 does not match current ZIP",
            })
            continue
        mimic = report.get("mimic", {})
        entries.append({
            "name": zip_path.stem,
            "zip": zip_path,
            "passed": report.get("status") == "PASSED",
            "sha256": report.get("zip_sha256", ""),
            "bytes": zip_path.stat().st_size,
            "invariance": report.get("sample", {}).get("invariance_max_abs_delta"),
            "mimic_seconds": mimic.get("run", {}).get("seconds"),
            "mimic_mean": mimic.get("prediction_mean"),
        })
    return entries


def attach_evidence(entries: list[dict], args: argparse.Namespace) -> None:
    evidence: dict[str, str] = {}
    priority: dict[str, float] = {}

    if args.blend_sweep.is_file():
        sweep = json.loads(args.blend_sweep.read_text(encoding="utf-8"))
        for item in sweep.get("candidates", []):
            name = item["candidate"]
            weight = item["linear_weight"]
            evidence[name] = (
                f"Track A · linear {weight:g}/hgb {1 - weight:g}"
                " · manifest-only 민감도 후보(독립 rolling 점수 없음)"
            )
            priority[name] = 900.0 + (1.0 - weight)

    if args.package_index.is_file():
        package = json.loads(args.package_index.read_text(encoding="utf-8"))
        for item in package.get("candidates", []):
            name = item["candidate"]
            primary = item.get("expected_scores", {}).get("2024")
            if primary is not None:
                family = item.get("family") or item.get("kind", "v2")
                evidence[name] = (
                    f"V2 {family} · 2024 개발 fold {float(primary):,.1f} "
                    "· post-selection 탐색값"
                )
                priority[name] = 2000.0 + float(primary)
            else:
                shift = item.get("logit_shift")
                evidence[name] = (
                    f"V2 base-rate shift {float(shift):+g} · 점수 미할당 민감도 후보"
                    if shift is not None else "V2 후보 · 개발 fold 근거 없음"
                )
                priority[name] = 1100.0

    if args.ensemble.is_file():
        payload = json.loads(args.ensemble.read_text(encoding="utf-8"))
        gate = payload.get("gate", {})
        primary = payload.get("per_season", {}).get(str(gate.get("primary_season", 2024)), {})
        blurb = (f"Track F 앙상블 · 2024 개발 fold {primary.get('ensemble_score', 0):,.1f} "
                 f"· gate_pass={gate.get('gate_pass')}")
        for entry in entries:
            if "ensemble" in entry["name"].lower() or "final" in entry["name"].lower():
                if entry["name"] not in evidence:
                    evidence[entry["name"]] = blurb
                    priority[entry["name"]] = (
                        2000.0 + float(primary.get("ensemble_score", 0))
                        if gate.get("gate_pass") else 500.0
                    )

    for entry in entries:
        entry["evidence"] = evidence.get(entry["name"], "근거 미기록 — 제출 전 확인")
        base = priority.get(entry["name"], 100.0)
        if entry["name"].split("_")[0] in KNOWN_SCORES:
            base = -1.0  # already submitted; keep it out of the way
        entry["priority"] = base


def render(entries: list[dict], args: argparse.Namespace) -> str:
    passed = [e for e in entries if e.get("passed")]
    failed = [e for e in entries if not e.get("passed")]
    passed.sort(key=lambda e: -e["priority"])
    queue = [e for e in passed if e["priority"] > 0][: args.top]

    lines = [
        "# 제출 큐",
        "",
        f"> 생성: **{datetime.now(timezone.utc).astimezone().strftime('%Y-%m-%d %H:%M %Z')}**  ",
        f"> 현재 LB 챔피언: **`{CHAMPION}`** (S8)  ",
        "> 리더보드 제출 마감: **2026-09-01 10:00 KST**  ",
        "> 일일 한도 **5회**. 최고 점수가 유지되므로 실패해도 챔피언 점수는 잃지 않는다.",
        "",
        "## 제출 순서",
        "",
        "높은 기대값부터 배치했다. 하루 5개씩 끊어 올린다.",
        "",
    ]
    if not queue:
        lines += ["아직 제출할 후보가 없다. `python experiments/pipeline.py --run` 을 먼저 완료한다.", ""]
    for day, start in enumerate(range(0, len(queue), DAILY_LIMIT), start=1):
        batch = queue[start : start + DAILY_LIMIT]
        lines += [f"### {day}일차 ({len(batch)}건)", "",
                  "| # | 후보 | 근거 | 크기 | 추론 | SHA-256 |",
                  "| ---: | --- | --- | ---: | ---: | --- |"]
        for offset, entry in enumerate(batch, start=1):
            seconds = f"{entry['mimic_seconds']:.1f}s" if entry.get("mimic_seconds") else "—"
            lines.append(
                f"| {start + offset} | `{entry['name']}.zip` | {entry['evidence']} | "
                f"{entry['bytes'] / 2**20:.2f}MB | {seconds} | `{(entry.get('sha256') or '')[:16]}…` |"
            )
        lines.append("")

    lines += ["## 게이트 결과", "",
              "| 후보 | 상태 | 불변성 최대 차이 | 245,789행 시간 | 예측 평균 |",
              "| --- | --- | ---: | ---: | ---: |"]
    for entry in passed:
        delta = entry.get("invariance")
        seconds = entry.get("mimic_seconds")
        mean = entry.get("mimic_mean")
        cells = [
            f"`{entry['name']}`",
            "**PASSED**",
            f"`{delta:.3e}`" if delta is not None else "—",
            f"{seconds:.2f}초" if seconds is not None else "—",
            f"`{mean:.8f}`" if mean is not None else "—",
        ]
        lines.append("| " + " | ".join(cells) + " |")
    for entry in failed:
        lines.append(f"| `{entry['name']}` | **FAILED — 제출 금지** | — | — | — |")
    lines += ["", "## 업로드 절차", "",
              "1. DACON 제출 탭에서 위 순서대로 ZIP을 올린다. 파일명은 `submit.zip` 으로 바꿔도 되고 그대로 둬도 된다.",
              "2. 각 제출의 서버 실행 결과와 점수를 `SUBMISSION_LOG.md` 에 기록한다.",
              "3. 챔피언이 갱신되면 해당 ZIP·모델·설정·SHA-256을 `submission/archive/` 로 옮긴다.",
              "",
              "**제출은 자동화하지 않는다.** 업로드는 사람이 한다.", ""]
    if failed:
        lines += ["> ⚠️ 게이트 실패 후보가 있다. 절대 올리지 말고 원인을 먼저 고친다.", ""]
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    args.dist.mkdir(parents=True, exist_ok=True)
    print("dist/ 의 ZIP을 수집하고 게이트를 확인합니다 ...", flush=True)
    entries = collect(args)
    if not entries:
        print("ZIP이 없습니다. 먼저 후보를 빌드하세요.", flush=True)
    attach_evidence(entries, args)
    args.output.write_text(render(entries, args), encoding="utf-8")
    passed = sum(1 for e in entries if e.get("passed"))
    print(f"후보 {len(entries)}개 중 {passed}개 통과.", flush=True)
    print(f"Saved {args.output}", flush=True)
    if any(not e.get("passed") for e in entries):
        raise SystemExit("게이트 실패 후보가 있습니다. SUBMIT_QUEUE.md 를 확인하세요.")


if __name__ == "__main__":
    main()
