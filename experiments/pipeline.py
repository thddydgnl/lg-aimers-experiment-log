#!/usr/bin/env python3
"""Run the whole v2 plan end to end, unattended, and hand back a submit queue.

Design points that follow from EXPERIMENT_PLAN_V2.md:

* Track C (boosters) runs before Track B feature work.  Track A's ceiling is
  roughly 850-880, which is already below the ~900 a plain GBDT reaches, so the
  blend rewrite is kept as a free 30-minute side quest rather than a day of work.
* Nothing is submitted mid-run.  Candidate selection uses the 2024 fold plus a
  paired bootstrap CI (experiments/stats.py), never leaderboard feedback, so the
  pipeline can decide everything offline and emit one batch at the end.
* Every stage checkpoints into state.json, so a crash, a reboot, or a Kaggle
  offload can resume without recomputing finished work.

    python experiments/pipeline.py --run            # start or resume
    python experiments/pipeline.py --status         # what is done
    python experiments/pipeline.py --run --from c1  # redo from a stage
    python experiments/pipeline.py --run --kaggle   # offload heavy stages
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

PYTHON = Path(sys.executable)
RESULTS = ROOT / "experiments" / "results"
PREDICTIONS = RESULTS / "predictions"
STATE_PATH = ROOT / "experiments" / "pipeline_state.json"
PARAM_DIR = ROOT / "experiments" / "params"


@dataclass
class Stage:
    key: str
    title: str
    kind: str                      # rolling | script | builder
    argv: list[str] = field(default_factory=list)
    offload: str | None = None     # None | "cpu" | "gpu"
    optional: bool = False         # a failure here does not stop the pipeline
    note: str = ""


def rolling(stage: str, *extra: str) -> list[str]:
    return [
        str(ROOT / "experiments" / "run_v2_rolling.py"),
        "--stage", stage,
        "--output-dir", str(RESULTS),
        "--save-predictions", str(PREDICTIONS),
        *extra,
    ]


def build_stages(seasons: list[str]) -> list[Stage]:
    season_args = ["--validation-seasons", *seasons]
    return [
        Stage(
            key="preflight",
            title="환경 점검 — 고정 버전 확인, lightgbm 설치, S4 재현성 재확인",
            kind="script",
            argv=[str(ROOT / "experiments" / "preflight.py"), "--install"],
            note="여기서 막히면 30분 뒤 c1에서 실패하는 것보다 낫다.",
        ),
        Stage(
            key="probe",
            title="EDA 상한 재측정 (계획서 §1.6 재현)",
            kind="script",
            argv=[
                str(ROOT / "experiments" / "probe_grouping_ceilings.py"),
                "--output", str(RESULTS / "probe_grouping_ceilings.json"),
            ],
            note="빠름. Track B 우선순위의 근거를 다시 찍어 둔다.",
        ),
        Stage(
            key="a_blend",
            title="Track A — 블렌드 가중치 재조정 (재학습 0)",
            kind="script",
            argv=[
                str(ROOT / "submission" / "sweep_blend_candidates.py"),
                "--source", str(ROOT / "submission" / "archive" / "S4" / "S4.zip"),
            ],
            note="S4 ZIP의 manifest weight만 바꿔 S9~S11을 만든다. 30분.",
        ),
        Stage(
            key="base",
            title="기준선 — S4 구성 재현 (Linear 90 + HGB 10, base+e14)",
            kind="rolling",
            argv=rolling(
                "v2_base", "--models", "linear", "hgb", "--blend", "0.9", "0.1",
                "--features", "base", "e14", *season_args,
            ),
            note="이후 모든 단계가 이 예측을 baseline으로 비교한다.",
        ),
        Stage(
            key="c1",
            title="Track C-1 — LightGBM 기본값 (base+e14)",
            kind="rolling",
            argv=rolling(
                "v2_lgbm_base", "--models", "lgbm", "--features", "base", "e14",
                "--baseline-stage", "v2_base", *season_args,
            ),
            offload="cpu",
            note="Track C를 앞으로 당긴 첫 단계. pitcher_id를 native categorical로 투입.",
        ),
        Stage(
            key="c2",
            title="Track C-2 — LightGBM 하이퍼파라미터 탐색",
            kind="script",
            argv=[
                str(ROOT / "experiments" / "search_booster.py"),
                "--model", "lgbm",
                "--features", "base", "e14",
                "--baseline-stage", "v2_base",
                "--grid", str(PARAM_DIR / "lgbm_grid.json"),
                "--output", str(RESULTS / "v2_lgbm_search.json"),
                *season_args,
            ],
            offload="cpu",
            note="가장 무거운 단계. Kaggle offload 1순위.",
        ),
        Stage(
            key="b1",
            title="Track B1′ — 투수 × 타자 손 platoon split",
            kind="rolling",
            argv=rolling(
                "v2_lgbm_platoon", "--models", "lgbm",
                "--features", "base", "e14", "platoon",
                "--params", str(PARAM_DIR / "lgbm_best.json"),
                "--baseline-stage", "v2_lgbm_tuned", "--baseline-key", "lgbm",
                *season_args,
            ),
            note="2024 실측 +135~165점. LightGBM이 스스로 만들지 못하는 조합.",
        ),
        Stage(
            key="b2",
            title="Track B2 — HGB pitcher_id cross-fitted TargetEncoder",
            kind="rolling",
            argv=rolling(
                "v2_hgb_pitcher_te", "--models", "hgb",
                "--features", "base", "e14", "pitcher_te",
                "--baseline-stage", "v2_base", *season_args,
            ),
            note="792개 투수 ID를 5-fold cross-fitted TargetEncoder로 HGB에 복원한다.",
        ),
        Stage(
            key="b2b",
            title="Track B2 ablation — HGB pitcher TE + platoon",
            kind="rolling",
            argv=rolling(
                "v2_hgb_pitcher_te_platoon", "--models", "hgb",
                "--features", "base", "e14", "pitcher_te", "platoon",
                "--baseline-stage", "v2_hgb_pitcher_te", "--baseline-key", "hgb",
                *season_args,
            ),
            note="B1′ 주효과 중복 여부를 직접 측정한다.",
        ),
        Stage(
            key="b3_linear",
            title="Track B3 — Linear alpha/eta0 sweep",
            kind="script",
            argv=[
                str(ROOT / "experiments" / "search_booster.py"),
                "--model", "linear", "--features", "base", "e14",
                "--baseline-stage", "v2_base",
                "--grid", str(PARAM_DIR / "linear_grid.json"),
                "--output", str(RESULTS / "v2_linear_search.json"),
                "--tuned-stage", "v2_linear_tuned", *season_args,
            ],
            note="예측 표준편차와 2024 점수를 함께 기록해 과소적합을 판정한다.",
        ),
        Stage(
            key="b3_hgb",
            title="Track B3 — HGB capacity/l2 sweep",
            kind="script",
            argv=[
                str(ROOT / "experiments" / "search_booster.py"),
                "--model", "hgb", "--features", "base", "e14", "pitcher_te",
                "--baseline-stage", "v2_base",
                "--grid", str(PARAM_DIR / "hgb_grid.json"),
                "--output", str(RESULTS / "v2_hgb_search.json"),
                "--tuned-stage", "v2_hgb_tuned", *season_args,
            ],
            note="투수 TE를 유지한 채 leaf 수·L2·반복 수를 탐색한다.",
        ),
        Stage(
            key="b4",
            title="Track B4 — 결측 indicator + 공선 정리 (선형 성분)",
            kind="rolling",
            argv=rolling(
                "v2_linear_b4", "--models", "linear", "--features", "base", "e14",
                "--params", str(PARAM_DIR / "linear_b4.json"),
                "--baseline-stage", "v2_base", "--baseline-key", "linear",
                *season_args,
            ),
            optional=True,
            note="EDA §7.1·§9 권고를 구현해 앙상블의 선형 성분을 개선한다.",
        ),
        Stage(
            key="c3",
            title="Track C-3 — CatBoost (ordered CTR + 범주 조합 자동 생성)",
            kind="rolling",
            argv=rolling(
                "v2_catboost", "--models", "catboost", "--features", "base", "e14",
                "--baseline-stage", "v2_lgbm_tuned", "--baseline-key", "lgbm",
                *season_args,
            ),
            offload="gpu",
            optional=True,
            note="1.2.8 설치·CPU smoke 완료. 전량 GPU 이득이 큰 단계 → Kaggle 1순위.",
        ),
        Stage(
            key="c3b",
            title="Track C-3b — CatBoost + platoon (자동 조합 대조군)",
            kind="rolling",
            argv=rolling(
                "v2_catboost_platoon", "--models", "catboost",
                "--features", "base", "e14", "platoon",
                "--baseline-stage", "v2_catboost", "--baseline-key", "catboost",
                *season_args,
            ),
            offload="gpu",
            optional=True,
            note="CatBoost의 자동 조합이 B1′을 대체하는지 판정한다.",
        ),
        Stage(
            key="f1",
            title="Track F1 — 앙상블 성분 조합과 가중치",
            kind="script",
            argv=[
                str(ROOT / "experiments" / "search_ensemble.py"),
                "--predictions", str(PREDICTIONS),
                "--baseline-stage", "v2_base",
                "--output", str(RESULTS / "v2_ensemble.json"),
                *season_args,
            ],
            note="저장된 fold 예측만 사용하므로 재학습 없음. 수 초.",
        ),
        Stage(
            key="package",
            title="후보 ZIP 빌드 — 성분별 + 앙상블 + base rate shift 변형",
            kind="builder",
            argv=[
                str(ROOT / "submission" / "build_from_ensemble.py"),
                "--ensemble", str(RESULTS / "v2_ensemble.json"),
                "--output-dir", str(ROOT / "submission" / "dist"),
                # 0.0 = 무보정, -0.032 ≈ -0.8%p, -0.064 ≈ -1.6%p (계획서 §8.1).
                "--logit-shifts", "0.0", "-0.032", "-0.064",
            ],
            note="전체 데이터 재학습이 들어가므로 가장 오래 걸린다.",
        ),
        Stage(
            key="final",
            title="최종 — 전수 게이트 + 제출 큐 생성",
            kind="builder",
            argv=[
                str(ROOT / "submission" / "make_submit_queue.py"),
                "--ensemble", str(RESULTS / "v2_ensemble.json"),
                "--top", "10",
            ],
            note="여기까지 끝나면 사람이 업로드만 하면 된다.",
        ),
    ]


def load_state() -> dict[str, Any]:
    if STATE_PATH.is_file():
        return json.loads(STATE_PATH.read_text(encoding="utf-8"))
    return {"created_at_utc": datetime.now(timezone.utc).isoformat(), "stages": {}}


def save_state(state: dict[str, Any]) -> None:
    state["updated_at_utc"] = datetime.now(timezone.utc).isoformat()
    temporary = STATE_PATH.with_suffix(".tmp")
    temporary.write_text(
        json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    temporary.replace(STATE_PATH)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def adoptable(stage: Stage, seasons: list[str]) -> tuple[bool, str]:
    """Validate old artifacts before turning them into a checkpoint."""
    if stage.key == "preflight":
        path = RESULTS / "preflight.json"
        if not path.is_file():
            return False, "preflight.json missing"
        report = json.loads(path.read_text(encoding="utf-8"))
        required = {
            "numpy": "1.26.4", "pandas": "2.0.3", "scipy": "1.15.3",
            "sklearn": "1.8.0", "joblib": "1.5.3",
        }
        actual = {
            module: importlib.import_module(module).__version__
            for module in required
        }
        actual_lgbm = importlib.import_module("lightgbm").__version__
        complete = (
            not report.get("problems")
            and report.get("pinned") == required
            and report.get("optional", {}).get("lightgbm") == "4.7.0"
            and actual == required
            and actual_lgbm == "4.7.0"
        )
        return complete, str(path)
    if stage.key == "probe":
        path = RESULTS / "probe_grouping_ceilings.json"
        return path.is_file(), str(path)
    if stage.key == "a_blend":
        sweep_path = ROOT / "submission" / "dist" / "blend_sweep.json"
        if not sweep_path.is_file():
            return False, "blend_sweep.json missing"
        sweep = json.loads(sweep_path.read_text(encoding="utf-8"))
        for item in sweep.get("candidates", []):
            zip_path = Path(item["zip"])
            report_path = zip_path.with_suffix(".verification.json")
            if not zip_path.is_file() or not report_path.is_file():
                return False, f"missing ZIP/report for {zip_path.name}"
            report = json.loads(report_path.read_text(encoding="utf-8"))
            if report.get("status") != "PASSED" or report.get("zip_sha256") != sha256_file(zip_path):
                return False, f"stale or failed verification for {zip_path.name}"
        if len(sweep.get("candidates", [])) < 3:
            return False, "blend sweep has fewer than three candidates"
        return True, str(sweep_path)
    if stage.kind == "rolling":
        try:
            stage_name = stage.argv[stage.argv.index("--stage") + 1]
        except (ValueError, IndexError):
            return False, "rolling stage name missing"
        result = RESULTS / f"{stage_name}.json"
        csv = RESULTS / f"{stage_name}.csv"
        predictions = [PREDICTIONS / f"{stage_name}_{season}.npz" for season in seasons]
        if not result.is_file() or not csv.is_file() or not all(path.is_file() for path in predictions):
            return False, f"incomplete artifacts for {stage_name}"
        payload = json.loads(result.read_text(encoding="utf-8"))
        metadata = payload.get("metadata", {})

        def option_values(option: str) -> list[str]:
            if option not in stage.argv:
                return []
            start = stage.argv.index(option) + 1
            values = []
            for value in stage.argv[start:]:
                if value.startswith("--"):
                    break
                values.append(value)
            return values

        complete = (
            metadata.get("stage") == stage_name
            and metadata.get("models") == option_values("--models")
            and metadata.get("features") == option_values("--features")
            and metadata.get("validation_seasons") == sorted(int(value) for value in seasons)
            and not metadata.get("smoke_test")
        )
        if complete:
            for path in predictions:
                with np.load(path) as stored:
                    if not {"y", "row_index", "cluster"}.issubset(stored.files):
                        complete = False
                        break
        return complete, str(result)
    return False, "no safe adoption rule"


def artifact_patterns(stage_key: str) -> list[tuple[Path, str]]:
    mapping: dict[str, list[tuple[Path, str]]] = {
        "preflight": [(RESULTS, "preflight.json")],
        "probe": [(RESULTS, "probe_grouping_ceilings.json")],
        "a_blend": [(ROOT / "submission" / "dist", "blend_sweep.json")],
        "base": [(RESULTS, "v2_base.*"), (PREDICTIONS, "v2_base_*.npz")],
        "c1": [(RESULTS, "v2_lgbm_base.*"), (PREDICTIONS, "v2_lgbm_base_*.npz")],
        "c2": [
            (RESULTS, "v2_lgbm_cfg*"), (RESULTS, "v2_lgbm_tuned.*"),
            (RESULTS, "v2_lgbm_search.json"), (PREDICTIONS, "v2_lgbm_cfg*.npz"),
            (PREDICTIONS, "v2_lgbm_tuned_*.npz"), (PARAM_DIR, "lgbm_best.json"),
        ],
        "b1": [(RESULTS, "v2_lgbm_platoon.*"), (PREDICTIONS, "v2_lgbm_platoon_*.npz")],
        "b2": [(RESULTS, "v2_hgb_pitcher_te.*"), (PREDICTIONS, "v2_hgb_pitcher_te_*.npz")],
        "b2b": [(RESULTS, "v2_hgb_pitcher_te_platoon.*"), (PREDICTIONS, "v2_hgb_pitcher_te_platoon_*.npz")],
        "b3_linear": [
            (RESULTS, "v2_linear_cfg*"), (RESULTS, "v2_linear_tuned.*"),
            (RESULTS, "v2_linear_search.json"), (PREDICTIONS, "v2_linear_cfg*.npz"),
            (PREDICTIONS, "v2_linear_tuned_*.npz"), (PARAM_DIR, "linear_best.json"),
        ],
        "b3_hgb": [
            (RESULTS, "v2_hgb_cfg*"), (RESULTS, "v2_hgb_tuned.*"),
            (RESULTS, "v2_hgb_search.json"), (PREDICTIONS, "v2_hgb_cfg*.npz"),
            (PREDICTIONS, "v2_hgb_tuned_*.npz"), (PARAM_DIR, "hgb_best.json"),
        ],
        "b4": [(RESULTS, "v2_linear_b4.*"), (PREDICTIONS, "v2_linear_b4_*.npz")],
        "c3": [(RESULTS, "v2_catboost.*"), (PREDICTIONS, "v2_catboost_*.npz")],
        "c3b": [(RESULTS, "v2_catboost_platoon.*"), (PREDICTIONS, "v2_catboost_platoon_*.npz")],
        "f1": [(RESULTS, "v2_ensemble.json")],
        "package": [
            (ROOT / "submission" / "dist", "V*.zip"),
            (ROOT / "submission" / "dist", "V*.verification.json"),
            (ROOT / "submission" / "records", "V*_build.json"),
            (ROOT / "submission" / "records", "v2_package_index.json"),
        ],
        "final": [(ROOT / "submission", "SUBMIT_QUEUE.md")],
    }
    return mapping.get(stage_key, [])


def archive_rerun_artifacts(stage_keys: list[str]) -> Path | None:
    found: list[Path] = []
    for key in stage_keys:
        for directory, pattern in artifact_patterns(key):
            found.extend(path for path in directory.glob(pattern) if path.is_file())
    found = sorted(set(found))
    if not found:
        return None
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    destination = RESULTS / "archive" / f"pipeline_rerun_{stamp}"
    for source in found:
        relative = source.relative_to(ROOT)
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(source), str(target))
    return destination


def run_local(stage: Stage, log_dir: Path) -> dict[str, Any]:
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"{stage.key}.log"
    started = time.perf_counter()
    print(f"\n{'=' * 72}\n[{stage.key}] {stage.title}\n{'=' * 72}", flush=True)
    if stage.note:
        print(f"  {stage.note}", flush=True)
    with log_path.open("w", encoding="utf-8") as log:
        process = subprocess.Popen(
            [str(PYTHON), *stage.argv],
            cwd=ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            env={**_utf8_env()},
        )
        assert process.stdout is not None
        for line in process.stdout:
            sys.stdout.write(line)
            log.write(line)
        process.wait()
    elapsed = time.perf_counter() - started
    return {
        "returncode": process.returncode,
        "elapsed_seconds": elapsed,
        "log": str(log_path),
        "where": "local",
    }


def _utf8_env() -> dict[str, str]:
    import os

    environment = os.environ.copy()
    environment["PYTHONUTF8"] = "1"
    environment["PYTHONIOENCODING"] = "utf-8"
    matplotlib_cache = ROOT / "experiments" / "_cache" / "matplotlib"
    matplotlib_cache.mkdir(parents=True, exist_ok=True)
    environment["MPLCONFIGDIR"] = str(matplotlib_cache)
    return environment


def run_offloaded(stage: Stage, log_dir: Path) -> dict[str, Any]:
    from kaggle_offload.offload import run_stage_on_kaggle  # noqa: E402

    print(f"\n{'=' * 72}\n[{stage.key}] {stage.title}  → Kaggle ({stage.offload})\n{'=' * 72}", flush=True)
    return run_stage_on_kaggle(stage.key, stage.argv, stage.offload or "cpu", log_dir)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="store_true", help="Start or resume the pipeline.")
    parser.add_argument("--status", action="store_true", help="Print stage status and exit.")
    parser.add_argument("--from", dest="from_stage", default=None,
                        help="Re-run from this stage, discarding later checkpoints.")
    parser.add_argument("--only", nargs="+", default=None, help="Run only these stages.")
    parser.add_argument("--skip", nargs="+", default=[], help="Skip these stages.")
    parser.add_argument("--adopt-existing", action="store_true",
                        help="Checkpoint only artifacts that pass a stage-specific integrity check.")
    parser.add_argument("--kaggle", action="store_true",
                        help="Send stages marked offload=cpu/gpu to Kaggle.")
    parser.add_argument("--validation-seasons", nargs="+", default=["2022", "2023", "2024"])
    parser.add_argument("--log-dir", type=Path, default=ROOT / "experiments" / "logs")
    args = parser.parse_args()

    stages = build_stages(list(args.validation_seasons))
    state = load_state()

    if args.status or not args.run:
        print(f"{'stage':<10} {'status':<10} {'elapsed':>10}  title")
        print("-" * 78)
        for stage in stages:
            record = state["stages"].get(stage.key)
            status = "pending"
            elapsed = ""
            if record:
                status = record.get("status", "?")
                if record.get("elapsed_seconds"):
                    elapsed = f"{record['elapsed_seconds']:.0f}s"
            print(f"{stage.key:<10} {status:<10} {elapsed:>10}  {stage.title}")
        if not args.run:
            print("\n--run 을 붙여야 실제로 실행됩니다.")
        return

    if args.from_stage:
        keys = [stage.key for stage in stages]
        if args.from_stage not in keys:
            raise SystemExit(f"Unknown stage: {args.from_stage}. Known: {keys}")
        rerun_keys = keys[keys.index(args.from_stage):]
        archived = archive_rerun_artifacts(rerun_keys)
        if archived:
            print(f"기존 결과 보관: {archived}", flush=True)
        for key in rerun_keys:
            state["stages"].pop(key, None)
        save_state(state)

    PARAM_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS.mkdir(parents=True, exist_ok=True)
    PREDICTIONS.mkdir(parents=True, exist_ok=True)

    overall = time.perf_counter()
    for stage in stages:
        if args.only and stage.key not in args.only:
            continue
        if stage.key in args.skip:
            print(f"[{stage.key}] skipped by request", flush=True)
            state["stages"][stage.key] = {
                "status": "skipped",
                "reason": "explicit --skip",
                "finished_at_utc": datetime.now(timezone.utc).isoformat(),
            }
            save_state(state)
            continue
        record = state["stages"].get(stage.key)
        if record and record.get("status") == "done":
            print(f"[{stage.key}] already done ({record.get('elapsed_seconds', 0):.0f}s) — skipping", flush=True)
            continue

        if args.adopt_existing:
            adopted, evidence = adoptable(stage, list(args.validation_seasons))
            if adopted:
                print(f"[{stage.key}] validated existing artifact — adopted", flush=True)
                state["stages"][stage.key] = {
                    "status": "done", "adopted": True, "evidence": evidence,
                    "finished_at_utc": datetime.now(timezone.utc).isoformat(),
                }
                save_state(state)
                continue

        use_kaggle = bool(args.kaggle and stage.offload)
        try:
            outcome = run_offloaded(stage, args.log_dir) if use_kaggle else run_local(stage, args.log_dir)
        except Exception as error:  # noqa: BLE001 - the pipeline must survive one bad stage
            outcome = {"returncode": 1, "error": repr(error), "where": "kaggle" if use_kaggle else "local"}

        ok = outcome.get("returncode") == 0
        outcome["status"] = "done" if ok else ("skipped" if stage.optional else "failed")
        outcome["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
        state["stages"][stage.key] = outcome
        save_state(state)

        if not ok:
            if stage.optional:
                print(f"\n[{stage.key}] 실패했지만 optional 단계이므로 계속합니다: "
                      f"{outcome.get('error', 'exit ' + str(outcome.get('returncode')))}\n", flush=True)
                continue
            print(f"\n[{stage.key}] 실패. 파이프라인을 멈춥니다.", flush=True)
            print(f"  로그: {outcome.get('log', '(없음)')}", flush=True)
            print(f"  고친 뒤: python experiments/pipeline.py --run   (이어서 재개됩니다)", flush=True)
            raise SystemExit(1)

    total = time.perf_counter() - overall
    print(f"\n{'=' * 72}", flush=True)
    incomplete = [
        stage.key for stage in stages
        if state["stages"].get(stage.key, {}).get("status") not in ("done", "skipped")
    ]
    if incomplete:
        print(f"선택 단계 완료 — {total / 60:.1f}분", flush=True)
        print(f"전체 파이프라인 미완료: {', '.join(incomplete)}", flush=True)
        return
    print(f"전체 파이프라인 완료 — {total / 60:.1f}분", flush=True)
    queue = ROOT / "submission" / "SUBMIT_QUEUE.md"
    if queue.is_file():
        print(f"제출 큐: {queue}", flush=True)
        print(shutil.get_terminal_size().columns * "-", flush=True)
        print(queue.read_text(encoding="utf-8"), flush=True)
    else:
        print("제출 큐가 생성되지 않았습니다. final 단계 로그를 확인하세요.", flush=True)


if __name__ == "__main__":
    main()
