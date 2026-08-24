#!/usr/bin/env python3
"""Rebuild an archived candidate ZIP with new Linear/HGB blend weights.

No retraining: the source ZIP's model artifacts are copied byte-for-byte and only
the manifest weights change.  Source model hashes are re-verified first, so a
corrupted archive cannot silently produce a new candidate.
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
import time
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from submission.build_submission import (  # noqa: E402
    common_metadata,
    deterministic_zip,
    sha256_file,
    write_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=ROOT / "submission/archive/S4/S4.zip")
    parser.add_argument("--linear-weight", type=float, required=True)
    parser.add_argument("--candidate", required=True, help="e.g. S9_s4_hgb50")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "submission/dist")
    parser.add_argument("--record-dir", type=Path, default=ROOT / "submission/records")
    return parser.parse_args()


def reweight(source: Path, linear_weight: float, candidate: str,
             output_dir: Path, record_dir: Path) -> Path:
    started = time.perf_counter()
    if not 0.0 <= linear_weight <= 1.0:
        raise ValueError(f"--linear-weight must lie in [0, 1]; got {linear_weight}")
    hgb_weight = 1.0 - linear_weight
    if abs(linear_weight + hgb_weight - 1.0) > 1e-12:
        raise ValueError("Weights do not sum to 1 within the manifest tolerance.")

    output_dir.mkdir(parents=True, exist_ok=True)
    source_hash = sha256_file(source)
    with tempfile.TemporaryDirectory(prefix="reweight_", dir=output_dir) as temporary:
        stage = Path(temporary)
        with zipfile.ZipFile(source) as archive:
            archive.extractall(stage)

        manifest_path = stage / "model" / "manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        models = manifest["models"]
        if len(models) != 2:
            raise ValueError(f"Expected a 2-model blend; found {len(models)}.")
        for item in models:
            actual = sha256_file(stage / "model" / item["file"])
            if actual.lower() != item["sha256"].lower():
                raise ValueError(f"Source model hash mismatch: {item['file']}")

        linear = next(item for item in models if "linear" in item["file"])
        hgb = next(item for item in models if "hgb" in item["file"])
        previous = {"linear": linear["weight"], "hgb": hgb["weight"]}
        linear["weight"] = linear_weight
        hgb["weight"] = hgb_weight

        manifest["candidate"] = candidate
        manifest["description"] = (
            f"{manifest.get('description', '')} | reweighted "
            f"linear={linear_weight:g}, hgb={hgb_weight:g}"
        ).strip(" |")
        manifest["reweighted_from"] = {
            "source": str(source),
            "source_zip_sha256": source_hash,
            "previous_weights": previous,
            "rationale": "CALIBRATION_ENSEMBLE_REPORT.md section 4, 2024 fold",
        }
        write_json(manifest_path, manifest)

        output = output_dir / f"{candidate}.zip"
        deterministic_zip(stage, output)

    metadata = common_metadata(candidate, output, started)
    metadata.update(
        {
            "description": manifest["description"],
            "source": str(source),
            "source_zip_sha256": source_hash,
            "previous_weights": previous,
            "new_weights": {"linear": linear_weight, "hgb": hgb_weight},
            "retrained": False,
        }
    )
    write_json(record_dir / f"{candidate}_build.json", metadata)
    print(f"Built {candidate}: {output} ({metadata['zip_sha256']})", flush=True)
    return output


def main() -> None:
    args = parse_args()
    reweight(args.source, args.linear_weight, args.candidate, args.output_dir, args.record_dir)


if __name__ == "__main__":
    main()
