"""Fail if the compact handoff contains forbidden or suspicious files."""

from __future__ import annotations

import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "catalog" / "validation_report.json"
FORBIDDEN_PARTS = {
    ".venv",
    "__pycache__",
    "artifacts",
    "_cache",
    "_tabicl_site",
    "predictions",
    "external_repos",
}
FORBIDDEN_SUFFIXES = {
    ".npy",
    ".npz",
    ".pkl",
    ".pickle",
    ".joblib",
    ".cbm",
    ".ckpt",
    ".zip",
    ".zst",
    ".pem",
    ".key",
}
SECRET_PATTERNS = {
    "github_token": re.compile(r"(?:ghp_|github_pat_)[A-Za-z0-9_]{20,}"),
    "aws_access_key": re.compile(r"AKIA[0-9A-Z]{16}"),
    "private_key": re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----"),
    "openai_key": re.compile(r"sk-(?:proj-)?[A-Za-z0-9_-]{20,}"),
}
TEXT_SUFFIXES = {".py", ".json", ".csv", ".md", ".txt", ".yaml", ".yml", ".toml", ".ps1", ".sh"}
LINK_CHECK_EXCLUSIONS = {
    Path("docs/ORIGINAL_AGENT_GUIDE.md"),
    Path("docs/ORIGINAL_PROJECT_README.md"),
}


def main() -> None:
    files = [path for path in ROOT.rglob("*") if path.is_file() and ".git" not in path.parts]
    forbidden: list[str] = []
    large: list[str] = []
    secrets: list[dict[str, str]] = []
    local_paths: list[str] = []
    broken_links: list[dict[str, str]] = []
    for path in files:
        relative = path.relative_to(ROOT)
        if any(part in FORBIDDEN_PARTS for part in relative.parts) or path.suffix.lower() in FORBIDDEN_SUFFIXES:
            forbidden.append(relative.as_posix())
        if path.stat().st_size > 25 * 1024 * 1024:
            large.append(relative.as_posix())
        if path.suffix.lower() not in TEXT_SUFFIXES or relative == Path("tools/validate_repo.py"):
            continue
        text = path.read_text(encoding="utf-8", errors="ignore")
        if re.search(r"[A-Za-z]:\\Users\\[^\\\s]+", text, flags=re.IGNORECASE):
            local_paths.append(relative.as_posix())
        for name, pattern in SECRET_PATTERNS.items():
            if pattern.search(text):
                secrets.append({"path": relative.as_posix(), "pattern": name})
        if path.suffix.lower() == ".md" and relative not in LINK_CHECK_EXCLUSIONS:
            for target in re.findall(r"\[[^\]]+\]\(([^)]+)\)", text):
                target = target.strip().strip("<>").split("#", 1)[0]
                if not target or re.match(r"^[a-z]+://", target, flags=re.IGNORECASE):
                    continue
                if target.lower().endswith(".zip"):
                    continue
                resolved = (path.parent / target).resolve()
                if not resolved.exists():
                    broken_links.append({"path": relative.as_posix(), "target": target})
    report = {
        "status": "PASS" if not (forbidden or large or secrets or local_paths or broken_links) else "FAIL",
        "file_count": len(files),
        "forbidden_files": forbidden,
        "files_over_25_mib": large,
        "possible_secrets": secrets,
        "personal_absolute_paths": local_paths,
        "broken_local_markdown_links": broken_links,
    }
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if report["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
