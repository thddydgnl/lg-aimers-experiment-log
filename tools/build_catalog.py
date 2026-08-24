"""Build machine-readable experiment catalogs from the Markdown registry."""

from __future__ import annotations

import csv
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REGISTRY = ROOT / "experiments" / "EXPERIMENT_REGISTRY.md"
OUT_DIR = ROOT / "catalog"


def split_row(line: str) -> list[str]:
    return [cell.strip() for cell in line.strip().strip("|").split("|")]


def phase_for(experiment_id: str) -> str:
    upper = experiment_id.upper()
    for phase in ("V5", "V4", "V3", "V2"):
        if upper.startswith(phase):
            return phase
    if upper.startswith("E") or upper.startswith("FINAL_ENSEMBLE"):
        return "EARLY"
    return "OTHER"


def parse_registry() -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    group = "primary_registry"
    for line in REGISTRY.read_text(encoding="utf-8").splitlines():
        if line.startswith("## V5 recent immutable results"):
            group = "v5_recent_immutable"
            continue
        if not line.startswith("|") or line.startswith("| ---"):
            continue
        cells = split_row(line)
        if len(cells) != 5 or cells[0] in {"실험 ID", "Experiment"}:
            continue
        experiment_id, change, result, evidence, decision = cells
        records.append(
            {
                "sequence": len(records) + 1,
                "group": group,
                "phase": phase_for(experiment_id),
                "experiment_id": experiment_id,
                "change": change,
                "result": result,
                "evidence": evidence,
                "decision": decision,
            }
        )
    return records


def main() -> None:
    records = parse_registry()
    if not records:
        raise SystemExit("No experiment records were parsed")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fields = list(records[0])
    with (OUT_DIR / "experiments.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(records)
    (OUT_DIR / "experiments.json").write_text(
        json.dumps(records, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Wrote {len(records)} experiment records")


if __name__ == "__main__":
    main()

