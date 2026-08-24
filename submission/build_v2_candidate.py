#!/usr/bin/env python3
"""Train a v2 configuration on 2019-2024 and package it as a submission ZIP.

Feature/encoder cutoffs mirror the rolling protocol exactly:

* E14 counters and the platoon encoder are frozen *after* the 2024 season, which
  is what an unseen 2025 row must be scored against.
* The training matrix uses season-wise out-of-fold encodings, so no training row
  ever sees an encoder fitted on itself.
* Everything the inference script needs is a frozen lookup table shipped inside
  model/, hashed in the manifest, and verified at load time.
"""

from __future__ import annotations

import argparse
import json
import platform
import shutil
import sys
import tempfile
import time
import zipfile
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.run_baselines import FEATURES as BASE_FEATURES, SEASON, TARGET, load_train  # noqa: E402
from experiments.run_e14_rolling import (  # noqa: E402
    E14_FEATURES, E14_K, build_e14_features, prior_before_each_season, season_end_state,
)
from experiments.run_e15_pseudo_forward import candidate_priors  # noqa: E402
from experiments.run_v2_rolling import (  # noqa: E402
    CategoricalFrameModel, PITCHER_TE_FEATURES, PLATOON_FEATURES,
    build_pitcher_te_features, build_platoon_frame, model_factory,
    platoon_states_before_each_season,
)
from submission.build_submission import common_metadata, deterministic_zip, sha256_file, write_json  # noqa: E402

TEMPLATE = ROOT / "submission" / "template"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", required=True, help="e.g. S12_lgbm_platoon")
    parser.add_argument("--models", nargs="+", default=None,
                        help="Model families in blend order, e.g. lgbm linear")
    parser.add_argument("--weights", nargs="+", type=float, default=None)
    parser.add_argument("--features", nargs="+", default=["base", "e14"],
                        choices=("base", "e14", "platoon", "pitcher_te"))
    parser.add_argument("--params", type=Path, default=None, help="Booster params JSON.")
    parser.add_argument("--spec", type=Path, default=None,
                        help="JSON spec for a mixed blend; overrides --models/--weights/--features. "
                             'Shape: {"features": [...], "models": [{"family","weight","features",'
                             '"params"}]}. Each model is fitted on its own feature subset of the '
                             "union matrix, so components may differ.")
    parser.add_argument("--prior-mode", default="r_recent3")
    parser.add_argument(
        "--inner-validation", choices=("none", "all", "regular"), default="none",
        help=(
            "LightGBM/CatBoost iteration selection. V3 candidates use 'regular' so "
            "rolling validation and the final build share one chronological recipe."
        ),
    )
    parser.add_argument("--logit-shift", type=float, default=0.0,
                        help="Global logit offset (plan section 8.1). 0 disables it.")
    parser.add_argument("--data", type=Path, default=ROOT / "open/data/train.csv")
    parser.add_argument("--target-season", type=int, default=2025)
    parser.add_argument(
        "--history-window", type=int, default=None,
        help="Fit model rows from the latest N seasons; state assets still use all history.",
    )
    parser.add_argument("--f-regime-start", type=int, default=None)
    parser.add_argument("--season-decay", type=float, default=1.0)
    parser.add_argument("--f-pre-regime-weight", type=float, default=0.0)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "submission/dist")
    parser.add_argument("--record-dir", type=Path, default=ROOT / "submission/records")
    return parser.parse_args()


def reuse_unshifted_ensemble(
    args: argparse.Namespace, components: list[dict], started: float
) -> bool:
    """Derive a logit-shift variant without refitting identical models.

    A global logit shift is applied by ``script_v2.py`` only after the frozen
    component probabilities have been blended.  Reusing the already-built
    unshifted archive is therefore exactly equivalent to fitting the same
    deterministic component specifications again.  The strict specification
    comparison below prevents this shortcut from being used for any candidate
    whose models, weights, feature groups, or parameters differ.
    """
    if args.logit_shift == 0.0 or "_shift" not in args.candidate:
        return False

    base_candidate = args.candidate.split("_shift", 1)[0]
    base_zip = args.output_dir / f"{base_candidate}.zip"
    if not base_zip.is_file():
        return False

    with zipfile.ZipFile(base_zip) as archive:
        base_manifest = json.loads(
            archive.read("model/manifest.json").decode("utf-8")
        )

    expected = [
        {
            "family": item["family"],
            "weight": float(item["weight"]),
            "feature_groups": list(item.get("features", args.features)),
            "params": item.get("params"),
        }
        for item in components
    ]
    actual = [
        {
            "family": item["family"],
            "weight": float(item["weight"]),
            "feature_groups": list(item.get("feature_groups", [])),
            "params": item.get("params"),
        }
        for item in base_manifest.get("models", [])
    ]
    if (
        float(base_manifest.get("logit_shift", 0.0)) != 0.0
        or base_manifest.get("inner_validation", "none") != args.inner_validation
        or base_manifest.get("history_window") != args.history_window
        or base_manifest.get("f_regime_start") != args.f_regime_start
        or float(base_manifest.get("season_decay", 1.0)) != args.season_decay
        or float(base_manifest.get("f_pre_regime_weight", 0.0)) != args.f_pre_regime_weight
        or actual != expected
    ):
        raise ValueError(
            f"Cannot reuse {base_zip.name}: its frozen component specification differs."
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="v2_shift_", dir=args.output_dir) as temporary:
        stage = Path(temporary)
        with zipfile.ZipFile(base_zip) as archive:
            archive.extractall(stage)
        manifest_path = stage / "model" / "manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["candidate"] = args.candidate
        manifest["logit_shift"] = float(args.logit_shift)
        manifest["description"] = (
            f"{'+'.join(args.models)} blend {args.weights} on "
            f"{'+'.join(args.features)} features, logit shift {args.logit_shift:+g}"
        )
        write_json(manifest_path, manifest)
        output = args.output_dir / f"{args.candidate}.zip"
        deterministic_zip(stage, output)

    metadata = common_metadata(args.candidate, output, started)
    metadata.update({
        "description": manifest["description"],
        "models": args.models,
        "weights": args.weights,
        "features": args.features,
        "prior": manifest.get("e14", {}).get("prior"),
        "logit_shift": args.logit_shift,
        "inner_validation": args.inner_validation,
        "history_window": args.history_window,
        "f_regime_start": args.f_regime_start,
        "season_decay": args.season_decay,
        "f_pre_regime_weight": args.f_pre_regime_weight,
        "booster_params": None,
        "reused_from": str(base_zip),
        "reused_from_sha256": sha256_file(base_zip),
        "reuse_equivalence": "identical frozen models; manifest-only global logit shift",
    })
    write_json(args.record_dir / f"{args.candidate}_build.json", metadata)
    print(
        f"Reused {base_zip.name} for {args.candidate}: {output} "
        f"({metadata['zip_sha256']})",
        flush=True,
    )
    return True


def main() -> None:
    args = parse_args()
    started = time.perf_counter()

    if args.spec is not None:
        spec = json.loads(args.spec.read_text(encoding="utf-8"))
        args.inner_validation = spec.get("inner_validation", args.inner_validation)
        args.history_window = spec.get("history_window", args.history_window)
        args.f_regime_start = spec.get("f_regime_start", args.f_regime_start)
        args.season_decay = float(spec.get("season_decay", args.season_decay))
        args.f_pre_regime_weight = float(
            spec.get("f_pre_regime_weight", args.f_pre_regime_weight)
        )
        components = spec["models"]
        args.features = list(spec.get("features") or sorted(
            {f for item in components for f in item.get("features", ["base"])},
            key=["base", "e14", "platoon", "pitcher_te"].index,
        ))
        args.models = [item["family"] for item in components]
        args.weights = [float(item["weight"]) for item in components]
    else:
        if not args.models or not args.weights:
            raise SystemExit("--models/--weights 또는 --spec 중 하나는 필요합니다.")
        components = [
            {"family": name, "weight": float(weight), "features": list(args.features)}
            for name, weight in zip(args.models, args.weights)
        ]
        if args.params is not None:
            shared = json.loads(args.params.read_text(encoding="utf-8"))
            for item in components:
                item.setdefault("params", shared)

    if len(args.models) != len(args.weights):
        raise SystemExit("--models 와 --weights 의 개수가 같아야 합니다.")
    if abs(sum(args.weights) - 1.0) > 1e-9:
        raise SystemExit(f"--weights 합이 1.0 이어야 합니다. 현재 {sum(args.weights)}")

    if reuse_unshifted_ensemble(args, components, started):
        return

    params = json.loads(args.params.read_text(encoding="utf-8")) if args.params else None
    frame = load_train(args.data)
    seasons = sorted(int(value) for value in frame[SEASON].unique())
    if seasons[-1] > 2024:
        raise SystemExit(f"학습은 2024까지여야 합니다. 발견된 시즌: {seasons}")

    prior = float(candidate_priors(frame, args.target_season)[args.prior_mode])
    print(f"prior({args.prior_mode}) = {prior:.8f}", flush=True)

    use_e14 = "e14" in args.features
    use_platoon = "platoon" in args.features
    use_pitcher_te = "pitcher_te" in args.features
    parts = [frame[BASE_FEATURES]]
    manifest_extra: dict = {}
    state_files: dict[str, dict] = {}

    if use_e14:
        states_before, final_state = season_end_state(frame)
        train_priors = prior_before_each_season(frame)
        e14_train, _ = build_e14_features(frame, states_before, train_priors, prior, k=E14_K)
        parts.append(e14_train)
        state_files["e14_state.json"] = {
            str(pitcher): [int(n), int(s)] for pitcher, (n, s) in final_state.items()
        }
        manifest_extra["e14"] = {
            "features": list(E14_FEATURES),
            "state_file": "e14_state.json",
            "prior": prior,
            "k": float(E14_K),
            "state_cutoff": f"after season {seasons[-1]}",
        }
        print(f"E14 state: {len(final_state):,} pitchers", flush=True)

    if use_platoon:
        priors_by_season = prior_before_each_season(frame)
        platoon_before, platoon_final = platoon_states_before_each_season(
            frame, priors_by_season, 200.0, 200.0
        )
        parts.append(build_platoon_frame(frame, platoon_before, platoon_final))
        state_files["platoon_state.json"] = platoon_final
        manifest_extra["platoon"] = {
            "features": list(PLATOON_FEATURES),
            "state_file": "platoon_state.json",
            "k_pitcher": platoon_final["k_pitcher"],
            "k_platoon": platoon_final["k_platoon"],
            "state_cutoff": f"after season {seasons[-1]}",
        }
        print(f"Platoon state: {len(platoon_final['pitcher_rate']):,} pitchers, "
              f"{len(platoon_final['platoon_delta']):,} cells", flush=True)

    if use_pitcher_te:
        parts.append(build_pitcher_te_features(frame))
        manifest_extra["pitcher_te"] = {
            "features": list(PITCHER_TE_FEATURES),
            "implementation": "sklearn TargetEncoder cross-fit inside the fitted model pipeline",
        }

    train_x = pd.concat(parts, axis=1)
    train_y = frame[TARGET].to_numpy(dtype=np.int8, copy=False)
    fit_mask = np.ones(len(frame), dtype=bool)
    if args.history_window is not None:
        if args.history_window < 1:
            raise ValueError("--history-window must be >= 1")
        fit_mask &= frame[SEASON].to_numpy() >= args.target_season - args.history_window
    if not 0.0 < args.season_decay <= 1.0:
        raise ValueError("--season-decay must be in (0, 1]")
    if not 0.0 <= args.f_pre_regime_weight <= 1.0:
        raise ValueError("--f-pre-regime-weight must be in [0, 1]")
    if (
        args.f_regime_start is not None
        and args.target_season > args.f_regime_start
        and args.f_pre_regime_weight == 0.0
    ):
        fit_mask &= (
            frame["game_type"].ne("F").to_numpy()
            | frame[SEASON].ge(args.f_regime_start).to_numpy()
        )
    fit_x = train_x.loc[fit_mask]
    fit_y = train_y[fit_mask]
    fit_frame = frame.loc[fit_mask]
    fit_weight: np.ndarray | None = None
    if args.season_decay != 1.0 or (
        args.f_regime_start is not None and args.f_pre_regime_weight != 1.0
    ):
        ages = (
            args.target_season - 1 - fit_frame[SEASON].to_numpy(dtype=np.int16)
        ).astype(np.float64)
        fit_weight = np.power(args.season_decay, ages)
        if args.f_regime_start is not None and args.target_season > args.f_regime_start:
            pre_f = (
                fit_frame["game_type"].eq("F").to_numpy(dtype=bool, na_value=False)
                & fit_frame[SEASON].lt(args.f_regime_start).to_numpy(
                    dtype=bool, na_value=False
                )
            )
            fit_weight[pre_f] *= args.f_pre_regime_weight
    features = list(train_x.columns)
    print(
        f"Training matrix: assets={len(train_x):,}, fit={len(fit_x):,} rows x "
        f"{len(features)} features",
        flush=True,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="v2_build_", dir=args.output_dir) as temporary:
        stage = Path(temporary)
        model_dir = stage / "model"
        model_dir.mkdir()
        shutil.copyfile(TEMPLATE / "script_v2.py", stage / "script.py")
        shutil.copyfile(TEMPLATE / "requirements.txt", stage / "requirements.txt")
        if "catboost" in args.models:
            with (stage / "requirements.txt").open("a", encoding="utf-8") as stream:
                stream.write("catboost==1.2.8\n")

        for name, payload in state_files.items():
            write_json(model_dir / name, payload)
            manifest_key = "e14" if name.startswith("e14") else "platoon"
            manifest_extra[manifest_key]["state_sha256"] = sha256_file(model_dir / name)

        group_columns = {
            "base": list(BASE_FEATURES),
            "e14": list(E14_FEATURES),
            "platoon": list(PLATOON_FEATURES),
            "pitcher_te": list(PITCHER_TE_FEATURES),
        }
        specs = []
        for index, component in enumerate(components):
            name = component["family"]
            weight = float(component["weight"])
            groups = list(component.get("features", args.features))
            # Preserve the union matrix's column order so the model sees exactly
            # the layout the manifest promises at inference time.
            wanted = {c for group in groups for c in group_columns[group]}
            columns = [c for c in features if c in wanted]
            component_params = component.get("params", params)
            print(f"[{index}:{name}] fitting weight={weight:g} on {len(columns)} features ...", flush=True)
            fit_started = time.perf_counter()
            model = model_factory(name, component_params)(columns)
            selected_iteration = None
            if isinstance(model, CategoricalFrameModel) and args.inner_validation != "none":
                inner_season = max(seasons)
                season_values = fit_frame[SEASON].to_numpy()
                earlier = season_values < inner_season
                inner_valid = season_values == inner_season
                if args.inner_validation == "regular":
                    inner_valid &= fit_frame["game_type"].astype(str).to_numpy() == "R"
                if not np.any(earlier) or not np.any(inner_valid):
                    raise ValueError(
                        f"Empty build inner split: season={inner_season}, "
                        f"mode={args.inner_validation}"
                    )
                component_x = fit_x.loc[:, columns]
                model.fit_time_ordered(
                    component_x.loc[earlier], fit_y[earlier],
                    component_x.loc[inner_valid], fit_y[inner_valid],
                    refit_full=True, refit_X=component_x, refit_y=fit_y,
                    sample_weight=(fit_weight[earlier] if fit_weight is not None else None),
                    eval_sample_weight=(fit_weight[inner_valid] if fit_weight is not None else None),
                    refit_sample_weight=fit_weight,
                )
                selected_iteration = model.best_iteration_
            else:
                if isinstance(model, CategoricalFrameModel):
                    model.fit(fit_x.loc[:, columns], fit_y, sample_weight=fit_weight)
                elif fit_weight is not None:
                    model.fit(fit_x.loc[:, columns], fit_y, clf__sample_weight=fit_weight)
                else:
                    model.fit(fit_x.loc[:, columns], fit_y)
            filename = f"{index:02d}_{name}.joblib"
            portable_model = model
            booster_preprocessing = None
            if isinstance(model, CategoricalFrameModel):
                portable_model = model.estimator
                booster_preprocessing = {
                    "backend": model.backend,
                    "categorical": list(model.categorical),
                    "categories": {
                        column: values.astype(str).tolist()
                        for column, values in model.categories_.items()
                    },
                }
            joblib.dump(portable_model, model_dir / filename, compress=3)
            model_spec = {
                "family": name,
                "file": filename,
                "sha256": sha256_file(model_dir / filename),
                "bytes": (model_dir / filename).stat().st_size,
                "weight": weight,
                "feature_groups": groups,
                "params": component_params,
                "model_features": columns,
                "selected_iteration": selected_iteration,
                "inner_validation": args.inner_validation,
            }
            if booster_preprocessing is not None:
                model_spec["booster_preprocessing"] = booster_preprocessing
            specs.append(model_spec)
            print(f"[{index}:{name}] saved {filename} "
                  f"({specs[-1]['bytes'] / 2**20:.2f} MiB, {time.perf_counter() - fit_started:.1f}s)",
                  flush=True)
            del model

        manifest = {
            "candidate": args.candidate,
            "schema_version": 2,
            "description": (
                f"{'+'.join(args.models)} blend {args.weights} on "
                f"{'+'.join(args.features)} features"
                + (f", logit shift {args.logit_shift:+g}" if args.logit_shift else "")
            ),
            "data_cutoff": f"season <= {seasons[-1]}",
            "training_seasons": seasons,
            "training_rows": int(len(frame)),
            "model_fit_rows": int(fit_mask.sum()),
            "model_fit_seasons": sorted(int(value) for value in fit_frame[SEASON].unique()),
            "training_data_sha256": sha256_file(args.data.resolve()),
            "training_code_sha256": {
                "run_v2_rolling": sha256_file(ROOT / "experiments" / "run_v2_rolling.py"),
                "build_v2_candidate": sha256_file(Path(__file__).resolve()),
            },
            "prior_source": f"E15 {args.prior_mode}",
            "row_independent_inference": True,
            "base_features": list(BASE_FEATURES),
            "features": features,
            "logit_shift": float(args.logit_shift),
            "inner_validation": args.inner_validation,
            "history_window": args.history_window,
            "f_regime_start": args.f_regime_start,
            "season_decay": args.season_decay,
            "f_pre_regime_weight": args.f_pre_regime_weight,
            "effective_fit_rows": (
                float(np.square(fit_weight.sum()) / np.square(fit_weight).sum())
                if fit_weight is not None else float(fit_mask.sum())
            ),
            "models": specs,
            **manifest_extra,
            "training_versions": {
                "python": platform.python_version(),
                "numpy": np.__version__,
                "pandas": pd.__version__,
                "joblib": joblib.__version__,
            },
        }
        for family in set(args.models):
            try:
                module = __import__({"lgbm": "lightgbm", "catboost": "catboost"}[family])
                manifest["training_versions"][family] = module.__version__
            except (KeyError, ImportError):
                pass
        write_json(model_dir / "manifest.json", manifest)
        output = args.output_dir / f"{args.candidate}.zip"
        deterministic_zip(stage, output)

    metadata = common_metadata(args.candidate, output, started)
    metadata.update({
        "description": manifest["description"],
        "models": args.models,
        "weights": args.weights,
        "features": args.features,
        "prior": prior,
        "logit_shift": args.logit_shift,
        "inner_validation": args.inner_validation,
        "history_window": args.history_window,
        "f_regime_start": args.f_regime_start,
        "season_decay": args.season_decay,
        "f_pre_regime_weight": args.f_pre_regime_weight,
        "booster_params": params,
    })
    write_json(args.record_dir / f"{args.candidate}_build.json", metadata)
    print(f"\nBuilt {args.candidate}: {output} ({metadata['zip_sha256']})", flush=True)
    print("다음: python submission/verify_submission.py "
          f"submission/dist/{args.candidate}.zip", flush=True)


if __name__ == "__main__":
    main()
