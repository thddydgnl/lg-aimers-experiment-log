#!/usr/bin/env python3
"""Package the fixed M3 ensemble from independently archived S4--S7 assets."""

from __future__ import annotations

import hashlib
import json
import platform
import shutil
import sys
import tempfile
import time
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from submission.build_submission import common_metadata, deterministic_zip, sha256_file, write_json  # noqa: E402


OUTPUT_DIR = ROOT / "submission" / "dist"
RECORD_DIR = ROOT / "submission" / "records"
TEMPLATE_DIR = ROOT / "submission" / "template"
SOURCES = {
    "S4": ROOT / "submission" / "archive" / "S4" / "S4.zip",
    "S5": ROOT / "submission" / "archive" / "S5" / "S5.zip",
    "S6": ROOT / "submission" / "archive" / "S6" / "S6.zip",
    "S7": ROOT / "submission" / "archive" / "S7" / "S7.zip",
}
WEIGHTS = {"S4": 0.70, "S5": 0.05, "S6": 0.10}
S7_WEIGHT = 0.15
MODEL_BLEND = {"linear": 0.90, "hgb": 0.10}


def read_manifest(archive: Path) -> dict:
    with zipfile.ZipFile(archive) as source:
        return json.loads(source.read("model/manifest.json"))


def copy_from_archive(archive: Path, member: str, destination: Path) -> str:
    with zipfile.ZipFile(archive) as source:
        data = source.read(member)
    destination.write_bytes(data)
    return hashlib.sha256(data).hexdigest()


def main() -> None:
    output_dir = OUTPUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)
    manifests = {candidate: read_manifest(path) for candidate, path in SOURCES.items()}
    with tempfile.TemporaryDirectory(prefix="s8_final_m3_build_", dir=output_dir) as temporary:
        stage = Path(temporary)
        model_dir = stage / "model"
        model_dir.mkdir()
        shutil.copyfile(TEMPLATE_DIR / "script.py", stage / "script.py")
        shutil.copyfile(TEMPLATE_DIR / "requirements.txt", stage / "requirements.txt")

        model_specs = []
        for candidate in ("S4", "S5", "S6"):
            archive = SOURCES[candidate]
            for family in ("linear", "hgb"):
                filename = f"{candidate.lower()}_{family}.joblib"
                source_hash = copy_from_archive(
                    archive, f"model/{'linear_sgd' if family == 'linear' else 'hgb'}.joblib", model_dir / filename
                )
                model_specs.append(
                    {
                        "kind": "regular",
                        "source_candidate": candidate,
                        "family": family,
                        "file": filename,
                        "sha256": source_hash,
                        "bytes": (model_dir / filename).stat().st_size,
                        "weight": WEIGHTS[candidate] * MODEL_BLEND[family],
                        "source_features": manifests[candidate]["features"],
                        "model_features": manifests[candidate]["features"],
                    }
                )

        # S7 is represented as two mixture families so its internal 90:10
        # Linear/HGB blend remains exact while the outer M3 weight is 0.15.
        for family, internal_weight in (("linear", 0.90), ("hgb", 0.10)):
            source_family = next(
                item for item in manifests["S7"]["models"] if item["family"] == family
            )
            components = []
            for component in source_family["components"]:
                filename = f"s7_{component['file']}"
                source_hash = copy_from_archive(
                    SOURCES["S7"], f"model/{component['file']}", model_dir / filename
                )
                components.append(
                    {
                        "group": component["group"],
                        "file": filename,
                        "sha256": source_hash,
                        "bytes": (model_dir / filename).stat().st_size,
                    }
                )
            model_specs.append(
                {
                    "kind": "e22_mixture",
                    "source_candidate": "S7",
                    "family": family,
                    "weight": S7_WEIGHT * internal_weight,
                    "components": components,
                    "model_features": manifests["S7"]["e22_mixture"]["model_features"],
                }
            )

        copy_from_archive(SOURCES["S4"], "model/e14_state.json", model_dir / "e14_state.json")
        copy_from_archive(SOURCES["S5"], "model/e16_role_state.json", model_dir / "e16_role_state.json")
        copy_from_archive(SOURCES["S6"], "model/e22_stage1.joblib", model_dir / "e22_stage1.joblib")

        e14 = dict(manifests["S4"]["e14"])
        e14["state_file"] = "e14_state.json"
        e16 = dict(manifests["S5"]["e16"])
        e16["role_state_file"] = "e16_role_state.json"
        e22 = dict(manifests["S6"]["e22"])
        e22["stage1_model_file"] = "e22_stage1.joblib"
        base_features = list(manifests["S4"]["base_features"])
        features = base_features + list(e14["features"]) + list(e16["features"]) + list(e22["features"])
        manifest = {
            "candidate": "S8",
            "description": "Final M3 ensemble: 0.70 S4 + 0.05 S5(E16) + 0.10 S6(E22R soft) + 0.15 S7(E22R mixture)",
            "build_spec_version": 1,
            "data_cutoff": "season <= 2024",
            "training_seasons": manifests["S4"]["training_seasons"],
            "training_rows": manifests["S4"]["training_rows"],
            "training_data_sha256": manifests["S4"]["training_data_sha256"],
            "training_code_sha256": {
                "baseline": manifests["S4"]["training_code_sha256"]["baseline"],
                "e14": manifests["S4"]["training_code_sha256"]["e14"],
                "e16": manifests["S5"]["training_code_sha256"]["e16"],
                "e22r": manifests["S6"]["training_code_sha256"]["e22r"],
                "e22r_mixture": manifests["S7"]["training_code_sha256"]["e22r"],
            },
            "feature_cutoff": "E14 state through 2024; E16 role state through 2024; E22 stage-1 historical labels only",
            "prior_source": "E15 r_recent3 prior; inherited identically by S4/S5/S6/S7",
            "k_source": "fixed EDA/theoretical E14 value 120",
            "row_independent_inference": True,
            "base_features": base_features,
            "features": features,
            "e14": e14,
            "e16": e16,
            "e22": e22,
            "ensemble_mixture": {
                "groups": manifests["S7"]["e22_mixture"]["groups"],
                "probability_features": manifests["S7"]["e22_mixture"]["probability_features"],
            },
            "ensemble": {
                "candidate_weights": {**WEIGHTS, "S7": S7_WEIGHT},
                "internal_model_weights": MODEL_BLEND,
                "rolling_reference": "experiments/results/archive/FINAL_ENSEMBLE_M3_v1/final_ensemble_rolling.json",
                "mean_brier_delta_vs_s4": -0.000005657343267897093,
                "worst_brier_delta_vs_s4": 0.000021784690996756728,
                "wins_vs_s4": "2/3",
                "gate_pass": True,
            },
            "models": model_specs,
            "training_versions": manifests["S4"]["training_versions"],
        }
        write_json(model_dir / "manifest.json", manifest)
        output = output_dir / "S8_final_m3.zip"
        deterministic_zip(stage, output)

    metadata = common_metadata("S8", output, time.perf_counter())
    metadata.update(
        {
            "description": manifest["description"],
            "source_archives": {candidate: str(path) for candidate, path in SOURCES.items()},
            "source_zip_sha256": {candidate: sha256_file(path) for candidate, path in SOURCES.items()},
            "candidate_weights": {**WEIGHTS, "S7": S7_WEIGHT},
            "internal_model_weights": MODEL_BLEND,
            "rolling_reference": manifest["ensemble"],
        }
    )
    RECORD_DIR.mkdir(parents=True, exist_ok=True)
    write_json(RECORD_DIR / "S8_build.json", metadata)
    print(f"Built S8: {output} ({metadata['zip_sha256']})", flush=True)


if __name__ == "__main__":
    main()
