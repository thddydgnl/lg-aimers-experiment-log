#!/usr/bin/env python3
"""Immutably archive one candidate ZIP and its evidence bundle."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
from datetime import datetime, timezone
from pathlib import Path


SUBMISSION_DIR = Path(__file__).resolve().parent
DEFAULT_ARCHIVE_ROOT = SUBMISSION_DIR / "archive"
REGISTRY_PATH = SUBMISSION_DIR / "CANDIDATE_REGISTRY.md"
CANDIDATE_PATTERN = re.compile(r"^S[0-9]+(?:[A-Z][A-Z0-9_-]*)?$")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("candidate_id", help="Immutable ID such as S3 or S14")
    parser.add_argument("zip_path", type=Path)
    parser.add_argument("--build-record", type=Path, default=None)
    parser.add_argument("--verification-report", type=Path, default=None)
    parser.add_argument("--local-metric", default="")
    parser.add_argument("--lb-score", default="pending")
    parser.add_argument("--notes", default="")
    parser.add_argument("--archive-root", type=Path, default=DEFAULT_ARCHIVE_ROOT)
    return parser.parse_args()


def copy_immutable(source: Path, destination: Path) -> None:
    if not source.is_file():
        raise FileNotFoundError(source)
    if destination.exists():
        if sha256_file(source) != sha256_file(destination):
            raise FileExistsError(
                f"Refusing to overwrite different artifact: {destination}"
            )
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)


def append_registry(record: dict) -> None:
    if not REGISTRY_PATH.exists():
        REGISTRY_PATH.write_text(
            "# 후보 산출물 보관대장\n\n"
            "| ID | ZIP SHA-256 | 로컬 지표 | LB 점수 | 보관 경로 | 상태 |\n"
            "| --- | --- | --- | ---: | --- | --- |\n",
            encoding="utf-8",
        )
    text = REGISTRY_PATH.read_text(encoding="utf-8")
    marker = f"| {record['candidate_id']} |"
    row = (
        f"| {record['candidate_id']} | `{record['zip_sha256']}` | "
        f"{record['local_metric'] or '미기록'} | {record['lb_score']} | "
        f"`{record['archive_path']}` | 보존 완료 |\n"
    )
    if marker in text:
        existing = [line for line in text.splitlines() if line.startswith(marker)]
        if existing and existing[0] != row.rstrip("\n"):
            raise ValueError(
                f"Registry already contains a different record for {record['candidate_id']}"
            )
        return
    header_end = text.find("| --- | --- | --- | ---: | --- | --- |\n")
    if header_end < 0:
        with REGISTRY_PATH.open("a", encoding="utf-8") as stream:
            stream.write(row)
        return
    insert_at = header_end + len("| --- | --- | --- | ---: | --- | --- |\n")
    REGISTRY_PATH.write_text(
        text[:insert_at] + row + text[insert_at:], encoding="utf-8"
    )


def main() -> None:
    args = parse_args()
    candidate_id = args.candidate_id.upper()
    if not CANDIDATE_PATTERN.fullmatch(candidate_id):
        raise ValueError("candidate_id must look like S3, S4, or S14A")
    zip_path = args.zip_path.resolve()
    archive_dir = (args.archive_root / candidate_id).resolve()
    archive_dir.mkdir(parents=True, exist_ok=True)
    archived_zip = archive_dir / f"{candidate_id}.zip"
    copy_immutable(zip_path, archived_zip)

    evidence = {
        "build_record": args.build_record.resolve() if args.build_record else None,
        "verification_report": (
            args.verification_report.resolve() if args.verification_report else None
        ),
    }
    for label, source in evidence.items():
        if source is not None:
            destination = archive_dir / source.name
            copy_immutable(source, destination)
            evidence[label] = str(destination.relative_to(SUBMISSION_DIR.parent))

    manifest = {
        "candidate_id": candidate_id,
        "archived_at_utc": datetime.now(timezone.utc).isoformat(),
        "zip_sha256": sha256_file(archived_zip),
        "zip_bytes": archived_zip.stat().st_size,
        "archive_path": str(archived_zip.relative_to(SUBMISSION_DIR.parent)),
        "local_metric": args.local_metric,
        "lb_score": args.lb_score,
        "notes": args.notes,
        "evidence": evidence,
    }
    manifest_path = archive_dir / "candidate_manifest.json"
    if manifest_path.exists():
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        if existing.get("zip_sha256") != manifest["zip_sha256"]:
            raise FileExistsError(f"Existing manifest differs: {manifest_path}")
        manifest = existing
    else:
        manifest_path.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    append_registry(manifest)
    print(
        f"Archived {candidate_id}: {manifest['archive_path']} "
        f"({manifest['zip_sha256']})",
        flush=True,
    )


if __name__ == "__main__":
    main()
