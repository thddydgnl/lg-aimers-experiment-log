"""Create a stable manifest for files intended for Git."""

from __future__ import annotations

import csv
import hashlib
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUTPUTS = {
    Path("catalog/included_files.csv"),
    Path("catalog/validation_report.json"),
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    records: list[dict[str, object]] = []
    for path in sorted(ROOT.rglob("*")):
        if not path.is_file() or ".git" in path.parts:
            continue
        relative = path.relative_to(ROOT)
        if relative in OUTPUTS:
            continue
        records.append(
            {
                "path": relative.as_posix(),
                "size_bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
        )
    output = ROOT / "catalog" / "included_files.csv"
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=("path", "size_bytes", "sha256"))
        writer.writeheader()
        writer.writerows(records)
    total = sum(int(row["size_bytes"]) for row in records)
    print(f"Manifested {len(records)} files ({total / 1024 / 1024:.2f} MiB)")


if __name__ == "__main__":
    main()

