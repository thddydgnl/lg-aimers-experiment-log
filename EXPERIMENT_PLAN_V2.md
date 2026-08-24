# 통합 실험 계획서 v2 — 용량 전환

> 작성일: **2026-08-18 KST**
> 대상 기간: **2026-08-18 밤 ~ 2026-09-01 10:00 (리더보드 제출 마감)**
> 선행 문서: [`EXPERIMENT_PLAN.md`](EXPERIMENT_PLAN.md) (S1~S8 이력, 그대로 보존) · [`COMPETITION.md`](COMPETITION.md) · [`LOCAL_ENVIRONMENT.md`](LOCAL_ENVIRONMENT.md)
> 이 문서는 v1을 **대체하지 않고 이어받는다.** v1은 S1~S8까지의 기록이고, v2는 그 이후의 실행 계획이다.

> **수치 스냅샷:** 아래 커트라인 약 `1,100`, 1등 `1,196.04861`, 남은 제출 예산은
> 2026-08-18 당시 값이다. 로그인 후 보이는 라이브 순위는 제출 전에 사람이 다시 갱신한다.
> **구현 동기화:** 2026-08-20에 B2/B3/B4, 시간 순 booster early stopping,
> pitcher-cluster bootstrap, family-diverse ensemble, checkpoint 무결성 로직을 코드에 반영했다.
> LightGBM 4.5.0과 CatBoost 1.2.8도 baseline constraint 아래 설치하고 S4 재검증을 통과했다.
> **실행 완료:** 2026-08-20 로컬 연속 파이프라인 17/17 단계 `done`(112.1분).
> 신규 V2 ZIP 6개와 제출 큐 전체 17개가 전수 게이트·SHA 검사를 통과했다. 상세 결과는
> [`experiments/EXPERIMENT_REGISTRY.md`](experiments/EXPERIMENT_REGISTRY.md), 제출 순서는
> [`submission/SUBMIT_QUEUE.md`](submission/SUBMIT_QUEUE.md)에 기록했다.
> **현재 환경 주의:** 위 `4.5.0`은 V2 실행 당시 기록이다. scikit-learn 1.8 호환 문제를
> 수정한 현재 `.venv`와 신규 빌드 템플릿은 LightGBM `4.7.0`을 사용한다.
> **후속 실제 결과:** V3 sparse M3가 2026-08-20 실제 LB `1090.9100565103`을 기록했다.
> 아래 `현재 LB`와 제출 예산은 V2 계획 시작 당시 스냅샷으로 읽는다.

---

## 0. 한 장 요약

| 항목 | 값 |
| --- | --- |
| V2 시작 당시 LB | S8 `689.3999289563` |
| 현재 LB | V3 sparse M3 **`1090.9100565103`** (`57386`) |
| Phase 3 커트라인 추정 | 약 `1,100` (93등 기준) |
| LB 1등 | `1,196.04861` |
| **필요 개선폭** | **약 `+410점` = Brier `-1.02e-3`** |
| 참고: E14+E15가 준 전체 | `+161.6점` = Brier `-4.03e-4` |
| 남은 제출 예산 | 약 `65~70회` (8/19~9/1) |

**필요 개선폭은 E14+E15가 준 것 전체의 2.5배다.** 피처 추가로 메울 수 있는 폭이 아니며, 모델 용량과 선택 기준을 바꿔야 한다.

### 전략 세 줄

1. **선택 기준을 max-min에서 max-expected로 바꾼다.** 이 대회는 최고 점수가 유지되므로 `689.40`은 잃을 수 없고, 공격적 제출의 하방 위험은 정확히 0이다. v1의 "최악 fold ≥ 0" 제약이 지금까지의 모든 보수적 결정을 만들었다.
2. **이미 학습된 HGB의 가중치를 올리는 것부터 한다.** 재학습 없이 manifest 숫자 두 개다. 2024 fold 기준 `+175.9점`의 근거가 이미 레포 안에 있다.
3. **그 다음 platoon split과 모델 용량을 준다.** `투수 × 타자 손` 조건부 인코딩(2024 실측 `+135~165점`)·투수 정체성·정규화 완화·LightGBM. 볼카운트 교차는 실측 `30.7점`으로 기대에서 제외한다.

> **개정 이력.** 2026-08-18 초안 이후 §1.6의 다섯 가지 측정으로 Track B를 재구성했다. 초안의 B1(볼카운트 교차)은 기각했고, 초안에 없던 `투수 × 타자 손` platoon split을 B1′로 최우선에 두었다.

---

## 1. 전환의 근거

### 1.1 최고 점수가 유지된다 — 위험은 공짜다

[`COMPETITION.md` §7.2](COMPETITION.md)와 [`EXPERIMENT_PLAN.md` §1.1](EXPERIMENT_PLAN.md)에 이미 확인해 기록해 둔 사실이다.

> 최고 점수가 리더보드에 표시되고 Public Score가 실시간 최고 점수로 갱신된다.
> 마지막 제출이 이전 최고 점수를 덮어쓰지 않는다.
> Private Score = 대회 종료 시점의 Public Score

따라서 새 제출의 결과가 `300`이든 `900`이든 **LB에 기록된 `689.3999289563`은 그대로 남는다.** 손실 가능성이 없는 선택지에 대해 분산을 회피할 이유가 없다.

v1은 이 규칙을 문서에 적어두고도 모델 선택은 최악 fold 기준으로 했다. 이것이 레포에서 가장 큰 전략적 모순이며, v2의 출발점이다.

### 1.2 버려진 175.9점

[`experiments/CALIBRATION_ENSEMBLE_REPORT.md` §4](experiments/CALIBRATION_ENSEMBLE_REPORT.md)의 실측이다. 괄호는 대회식 환산 점수다.

| 전략 | 2022 | 2023 | **2024** | 평균 Brier | 양의 fold |
| --- | ---: | ---: | ---: | ---: | ---: |
| **HGB raw** | **2,228.5** | 0.0 | **585.6** | 0.248568 | 2/3 |
| 50% Linear + 50% HGB | 2,033.6 | 0.0 | 564.9 | **0.247815** | 2/3 |
| 80% Linear + 20% HGB | 1,701.1 | 17.6 | 460.0 | 0.247846 | 3/3 |
| **90:10 ← v1 채택** | 1,554.3 | 105.4 | **409.7** | 0.247937 | **3/3** |
| Linear raw | 1,389.6 | 170.5 | 351.7 | 0.248068 | 3/3 |

- 2024 fold에서 HGB raw가 90:10보다 **`+175.9점`** (Brier `-4.39e-4`)
- 2022 fold에서 **`+674.2점`**
- 채택 이유는 오직 `3/3 양의 fold` 하나였다

2024는 2025에 가장 가까운 체제다. 이 한 칸의 175.9점을 최악 fold 방어를 위해 포기했다.

### 1.3 2023 fold의 0점은 재현될 사건이 아니다

[`experiments/CALIBRATION_ENSEMBLE_REPORT.md` §7](experiments/CALIBRATION_ENSEMBLE_REPORT.md):

| `game_type=F` 실제 성공률 | 2022 | 2023 | 2024 |
| --- | ---: | ---: | ---: |
| | **70.87%** | 47.29% | 45.93% |

2023 fold는 **F가 70.87%였던 이상 체제(2022)까지 학습하고 47.29%를 맞히는 문제**였다. 유연한 모델이 무너지는 것이 정상이다. F는 2023~2024에 이미 안정화됐고, 2025는 2022→2023식 단절보다 2023→2024식 연속에 가깝다.

**결론: 2023 fold는 일회성 체제 단절 사건으로 분류하고, 후보 채택 기준에서 동등 가중을 주지 않는다.** 기록은 계속 남긴다.

### 1.4 v1 게이트는 노이즈를 통과시켰다

[`experiments/run_e14_rolling.py:498`](experiments/run_e14_rolling.py)

```python
"gate_pass": bool(wins >= 2 and np.max(deltas) <= 0.0005)
```

E16·E22R·최종 앙상블이 다룬 효과는 `1e-6 ~ 1e-5`인데 임계값은 `5e-4`다. 200배 크므로 항상 참이고, 게이트에 남는 것은 `3 fold 중 2승`뿐이다. 귀무가설에서 이 조건이 통과할 확률은 **50%**다.

실증이 레포 안에 있다.

| | 로컬 평균 Brier delta | v1 게이트 | 실제 LB |
| --- | ---: | :---: | --- |
| E16 (S5) | **`+0.000000339`** (악화) | **PASS** | `688.17` < S4 `689.22` — **악화 확인** |
| 최종 M3 (S8) | `-0.000005657` (`≈ +2.27점` 예측) | PASS | `689.40`, S4 대비 **`+0.18점`** — 예측의 1/5 |

**S5~S8 구간 전체가 측정 한계 안이다.** §6에서 게이트를 재설계한다.

### 1.5 저용량 모델은 모든 피처 실험을 0으로 만든다

기각된 E10·E11·E20R은 전부 `Linear(alpha=0.3) 90% + HGB 10%` 위에서 평가됐다. 모델이 신호를 쓸 수 없는 상태에서 나온 "개선 없음"은 **피처에 신호가 없다는 증거가 아니라 모델에 용량이 없다는 증거**일 수 있다.

과소적합의 직접 증거 — 2024 fold 예측 표준편차:

| 모델 | 예측 std | 2024 점수 |
| --- | ---: | ---: |
| Linear SGD | `0.0297` | 351.7 |
| HGB | `0.0410` | 585.6 |
| 실제 필요한 분산 | 더 큼 | — |

`SGDClassifier(alpha=0.3, learning_rate="constant", eta0=0.001, max_iter=100)`([`experiments/run_baselines.py:416`](experiments/run_baselines.py))은 계수가 prior 근처에서 거의 움직이지 않는 설정이다. 주석은 "Brier가 과확신을 벌하므로 의도적"이라 하지만, 과확신의 정답은 적합을 망가뜨리는 것이 아니라 calibration이며 그 코드는 이미 `run_temporal_calibration_ensemble.py`에 있다.

**따라서 Track D(기각 실험 재평가)는 Track B·C(용량) 이후에 배치한다.**

### 1.6 초안 이후 추가 측정 — Track B를 재구성한 근거

초안의 가정 세 가지를 2024 데이터로 직접 검증했다. 재현 스크립트는 [§14](#14-1-6절-측정-재현)에 있다.

#### 검증 1 — 볼카운트 교차는 거의 값이 없다 (초안 B1 기각)

2024를 무작위 반으로 나눠 한쪽에서 EB 그룹 평균을 추정하고 다른 쪽에서 채점한 정직한 상한이다.

| 그룹 | 셀 수 | 2024 split-half 점수 |
| --- | ---: | ---: |
| `count_state` | 12 | **30.7** |
| `game_type` | 2 | 47.2 |
| `pitcher_hand × batter_hand` | 4 | **104.1** |
| `count × hand` | 48 | 105.9 |

`count_state`는 12셀 전체가 `30.7점`이고 hand와 교차해도 `104.1 → 105.9`로 2점만 더한다. `game_type` 하나보다 못하다.

원인은 라벨 정의에 있다. 실패가 **포수 요구 방향 기준**이므로 0-2에서 의도적으로 크게 뺀 공도 요구대로 들어갔다면 성공이다. 카운트가 만드는 투구 의도 차이가 라벨에서 이미 상쇄된다. EDA §5.1의 `49.96~53.41%`(3.45%p) 범위가 그 결과였다.

**조치: 초안 B1을 기대 목록에서 제외한다.** 비용이 없으므로 다른 실험의 부수 컬럼으로만 유지한다.

#### 검증 2 — `투수 × 타자 손` platoon split이 가장 큰 미실행 항목이다

EDA §22.2가 `투수 646` vs `투수 × 타자 손 800`으로 이미 정량화했다. 2024에서 재현하고 한계 기여까지 측정했다.

| 그룹 | k=0 | k=50 | k=200 | k=500 |
| --- | ---: | ---: | ---: | ---: |
| `pitcher_id` | 514.5 | 643.4 | **652.7** | 587.0 |
| `pitcher_id × batter_hand` | 545.9 | **814.1** | 779.1 | 622.5 |

투수 효과를 이미 가진 상태에서 platoon **잔차**만 더했을 때:

| 구성 | 점수 | 기준 대비 |
| --- | ---: | ---: |
| `pitcher(k=200)` 단독 | 652.7 | — |
| `+ platoon-residual(k=50)` | 788.6 | `+135.9` |
| **`+ platoon-residual(k=200)`** | **817.3** | **`+164.6`** |
| `+ platoon-residual(k=500)` | 788.0 | `+135.4` |

수축 설정을 바꿔도 `+135~165`가 안정적이다. EDA §22.2 표에서 **투수 단독을 이기는 유일한 상호작용**이다 — 투수×카운트 `339`, 투수×타자 raw pair `287`은 둘 다 더 나쁘다.

`batter_hand`를 전 실험 스크립트에 grep한 결과 **상호작용으로 만들어진 적이 한 번도 없다.** 전부 단순 피처 목록의 한 줄이다.

**GBDT가 대신 찾아주지 않는다.** HGB는 `pitcher_id`를 drop하고([`experiments/run_baselines.py:111`](experiments/run_baselines.py)), 넣더라도 792×2 조건부 평균을 `min_samples_leaf=100`으로 안정적으로 학습하지 못한다. 명시적 EB 인코딩이 필요하다.

> **감쇠 주의.** 위 숫자는 2024 **내부** split-half다. EDA §22.2는 시즌을 넘기면 투수 신호가 `646 → 250~340`으로 절반 이하가 된다고 실측했다. 같은 감쇠를 적용하면 2025 전이분은 **`+60~80점`** 정도로 보아야 한다. 확정은 rolling fold 측정으로 한다.

#### 검증 3 — base rate는 이미 대부분 해결돼 있다 (초안의 Track F 배치가 옳았다)

EDA §22.3의 "2.4%p 폭 = 최대 230점"을 근거로 이 항목을 Track A로 올리려 했으나, 측정 결과 그 폭은 E14가 이미 대부분 흡수한다.

| fold | S2 예측평균 편향 | **S4 (E14+E15) 편향** |
| ---: | ---: | ---: |
| 2022 | `−0.54%p` | `−0.75%p` |
| 2023 | `+1.35%p` | **`+0.83%p`** |
| 2024 | `+1.13%p` | **`+0.82%p`** |

잔여 편향 `+0.82%p`의 비용은 `0.0082² / 0.2498 × 100000 ≈ 27점`이다. 230점이 아니다.

EDA도 §22.3 말미에 *"§21의 시즌 내 피처가 이 문제를 부분적으로 자동 해결한다… 이중 보정이 되지 않는지 반드시 검사해야 한다"*고 적어 두었다.

**조치: Track F 배치를 유지하되, 저비용 하향 shift 1회만 D2로 앞당긴다.**

#### 검증 4 — 기대치를 낮출 두 항목

| 항목 | 근거 | 조치 |
| --- | --- | --- |
| **E14R** | EDA §8.3에서 `asof_pitcher_success_rate`와 `asof_pitcher_reverse_rate`의 상관이 **`−0.811`**. 시즌 내 버전도 중복이 클 것이다. 단독 `225점` ≠ E14 위의 한계 `225점` | Track D 유지, 기대치 하향 |
| **E20R** | 초안은 "저용량 모델 탓"이라 했으나 사양 확인 결과 `e20_rel_speed_sd`·`e20_rel_side_sd` 등 EDA §19.7이 지목한 산포 피처를 **이미 포함**하고 있었다. 제대로 만들어졌고 제대로 기각됐다. 투수 수준 Trackman 요약은 `asof_pitcher_success_rate`(같은 투수의 제구 결과를 직접 측정)와 중복된다 | Track D 최하위로 강등 |

#### 검증 5 — Track A와 상호작용 피처는 가산적이지 않다

**명시적 상호작용 피처는 주로 선형 모델을 구제한다.** Track A로 무게가 GBDT로 옮겨가면 `hand_pair`·`count_state`의 값은 줄어든다. 트리가 알아서 찾기 때문이다.

**예외는 platoon split이다.** 검증 2의 이유로 트리가 만들지 못하므로 Track A와 독립적으로 더해진다. 이것이 B1′을 최우선에 두는 이유다.

---

## 2. 트랙 구성

| 트랙 | 내용 | 재학습 | 기대 | 파이프라인 단계 |
| --- | --- | :---: | :---: | --- |
| **A** | 블렌드 가중치 재조정 | **불필요** | **매우 높음** | `a_blend` |
| **C** | **LightGBM → CatBoost** | 필요 | **높음** | `c1` `c2` `c3` `c3b` |
| **B** | platoon split · 투수 TE · 결측 indicator · 정규화 완화 | 필요 | 높음 | `b1` `b4` |
| **D** | 기각 실험 재평가 (E14R·E10·E11) | 필요 | 중하 | (백로그) |
| **E** | 검증 프로토콜 v2 | — | 필수 | 전 단계 내장 |
| **F** | 앙상블 · base rate shift | 필요 | 중 | `f1` `package` |

> **C가 B보다 먼저다 (초안에서 변경).** Track A의 천장은 약 `850~880`인데, 평범한 GBDT가
> 이미 `~900`에 도달한다는 관측이 있다. A에 이틀을 쓸 이유가 없어졌고, A는 재학습이 없는
> 30분짜리 작업이므로 파이프라인 앞머리에 그대로 두되 **부스터를 곧바로 이어서 돌린다.**
>
> **CatBoost를 성분으로 추가한다.** ordered target statistics가 투수 정체성을 원칙적으로
> 인코딩하고, `max_ctr_complexity` 기반 범주 조합 자동 생성이 §1.6 검증 2의
> `투수 × 타자 손`(+135~165점)을 **스스로 만들 수 있다.** LightGBM은 범주를 정렬해
> 분할점만 찾으므로 조합을 만들지 않는다. `c3b`가 이 대체 가능성을 판정하는 대조군이다.
> 다만 v1에서 설치가 실패한 이력이 있어 `optional` 단계로 두었다 — 실패해도 파이프라인은
> 계속 진행한다.

---

## 3. Track A — 블렌드 가중치 재조정 (재학습 0)

### 3.1 근거

`submission/archive/S4/S4.zip` 내부:

```text
model/e14_state.json      14,523 B
model/hgb.joblib         442,887 B   weight: 0.1
model/linear_sgd.joblib   36,818 B   weight: 0.9
model/manifest.json        4,723 B
requirements.txt             132 B
script.py                  8,414 B
```

**두 모델이 이미 학습돼 들어 있다.** `model/manifest.json`의 `models[].weight` 두 값만 바꾸면 된다. [`submission/template/script.py:43`](submission/template/script.py)이 합이 1.0인지만 검사한다.

학습·피처 재계산·상태 파일 변경이 전혀 없으므로 **E14 상태, prior, 행 독립성 성질이 모두 그대로 보존된다.**

### 3.2 신규 스크립트

`submission/reweight_candidate.py`를 새로 만든다. 기존 헬퍼를 그대로 재사용한다.

```python
#!/usr/bin/env python3
"""Rebuild an archived candidate ZIP with new Linear/HGB blend weights.

No retraining: the source ZIP's model artifacts are copied byte-for-byte and
only manifest weights change.  Source model hashes are re-verified first.
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


def main() -> None:
    args = parse_args()
    started = time.perf_counter()
    linear_weight = float(args.linear_weight)
    if not 0.0 <= linear_weight <= 1.0:
        raise ValueError(f"--linear-weight must lie in [0, 1]; got {linear_weight}")
    hgb_weight = 1.0 - linear_weight
    if abs(linear_weight + hgb_weight - 1.0) > 1e-12:
        raise ValueError("Weights do not sum to 1 within the manifest tolerance.")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    source_hash = sha256_file(args.source)
    with tempfile.TemporaryDirectory(prefix="reweight_", dir=args.output_dir) as temporary:
        stage = Path(temporary)
        with zipfile.ZipFile(args.source) as archive:
            archive.extractall(stage)

        manifest_path = stage / "model" / "manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        models = manifest["models"]
        if len(models) != 2:
            raise ValueError(f"Expected a 2-model blend; found {len(models)}.")

        # Re-verify the copied artifacts before trusting them.
        for item in models:
            actual = sha256_file(stage / "model" / item["file"])
            if actual.lower() != item["sha256"].lower():
                raise ValueError(f"Source model hash mismatch: {item['file']}")

        linear = next(item for item in models if "linear" in item["file"])
        hgb = next(item for item in models if "hgb" in item["file"])
        previous = {"linear": linear["weight"], "hgb": hgb["weight"]}
        linear["weight"] = linear_weight
        hgb["weight"] = hgb_weight

        manifest["candidate"] = args.candidate
        manifest["description"] = (
            f"{manifest.get('description', '')} | reweighted "
            f"linear={linear_weight:g}, hgb={hgb_weight:g}"
        ).strip(" |")
        manifest["reweighted_from"] = {
            "source": str(args.source),
            "source_zip_sha256": source_hash,
            "previous_weights": previous,
            "rationale": "CALIBRATION_ENSEMBLE_REPORT.md section 4, 2024 fold",
        }
        write_json(manifest_path, manifest)

        output = args.output_dir / f"{args.candidate}.zip"
        deterministic_zip(stage, output)

    metadata = common_metadata(args.candidate, output, started)
    metadata.update(
        {
            "description": manifest["description"],
            "source": str(args.source),
            "source_zip_sha256": source_hash,
            "previous_weights": previous,
            "new_weights": {"linear": linear_weight, "hgb": hgb_weight},
            "retrained": False,
        }
    )
    write_json(args.record_dir / f"{args.candidate}_build.json", metadata)
    print(f"Built {args.candidate}: {output} ({metadata['zip_sha256']})", flush=True)


if __name__ == "__main__":
    main()
```

### 3.3 실행 (오늘 밤)

```powershell
$env:PYTHONUTF8 = '1'

& .\.venv\Scripts\python.exe submission\reweight_candidate.py `
  --linear-weight 0.5 --candidate S9_s4_hgb50
& .\.venv\Scripts\python.exe submission\reweight_candidate.py `
  --linear-weight 0.2 --candidate S10_s4_hgb80
& .\.venv\Scripts\python.exe submission\reweight_candidate.py `
  --linear-weight 0.0 --candidate S11_s4_hgb100

foreach ($Name in 'S9_s4_hgb50', 'S10_s4_hgb80', 'S11_s4_hgb100') {
  & .\.venv\Scripts\python.exe submission\verify_submission.py `
    "submission\dist\$Name.zip"
}
```

`verify_submission.py`는 반드시 3개 모두 통과시킨다. **위험을 감수할 대상은 모델링이지 규정이 아니다.**

### 3.4 로컬 사전 확인 (선택, 30분)

제출 전에 2024 fold에서 S4 피처 위의 가중치별 점수를 직접 뽑아두면 LB 결과 해석이 쉬워진다. `run_e14_rolling.py`의 `run_fold`가 이미 linear/hgb 예측을 따로 갖고 있으므로, 블렌드 비율만 순회하는 짧은 스크립트로 충분하다.

`experiments/run_blend_sweep.py`:

- `--validation-seasons 2022 2023 2024`
- linear 가중치 `1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.0` 순회
- S4 피처(E14 + `r_recent3` prior) 기준
- fold별 Brier·환산 점수와 §6의 부트스트랩 CI를 함께 저장

### 3.5 제출과 기대치

| 제출 | linear / hgb | 2024 fold 근거(S2 기반) |
| --- | --- | ---: |
| S9 | 0.5 / 0.5 | 564.9 |
| S10 | 0.2 / 0.8 | (보간) |
| S11 | 0.0 / 1.0 | 585.6 |

**기대 LB: 대략 `850~880`.** 다만 175.9점 델타는 S2 기반 측정이고 S4에는 이미 E14/E15가 얹혀 있어 선형으로 더해지지 않을 수 있다. **예측하지 말고 측정한다.** 하방은 0이다.

---

## 4. Track B — 모델 용량과 platoon split

네 항목을 **하나의 rolling 스크립트**(`experiments/run_v2_rolling.py`)와
`experiments/search_booster.py`에서 ablation으로 함께 돌린다. S4 피처 구성을 기준선으로 둔다.

| | 항목 | 2024 실측 근거 | 우선순위 |
| --- | --- | ---: | :---: |
| **B1′** | `투수 × 타자 손` platoon EB 인코딩 | **`+135~165`** (전이 후 `+60~80` 추정) | **최우선** |
| B2 | `pitcher_id` TargetEncoder | EDA 상한 `646` | 높음 |
| B4 | 결측 indicator · 공선 정리 | EDA §7.1·§9 권고, 미측정 | 중 (무료) |
| B3 | 정규화 완화 | 예측 std `0.0297` vs HGB `0.0410` | 중 |
| ~~B1~~ | ~~`count_state`~~ | `30.7` — **기각** | 부수 컬럼만 |

### 4.1 B1′ — `투수 × 타자 손` platoon split ← 신규 최우선

**근거.** [§1.6 검증 2](#검증-2--투수--타자-손-platoon-split이-가장-큰-미실행-항목이다)에서 한계 기여 `+135~165점`을 실측했다. EDA §22.2 표에서 투수 단독(`646`)을 이기는 **유일한** 상호작용이며(`800`), 한 번도 구현된 적이 없다.

**왜 트리가 대신 못 하는가.** `HGB_DROPPED`가 `pitcher_id`를 제거하고, 넣더라도 792×2 조건부 평균은 `min_samples_leaf=100`에서 안정적으로 학습되지 않는다. 명시적 인코딩이 필요하다.

**구현.** 잔차 형태가 측정에서 가장 안정적이었다(`k=200`에서 `+164.6`). 투수 주효과를 먼저 잡고 platoon **잔차**만 별도로 수축한다.

```python
E30_PLATOON_FEATURES = [
    "e30_pitcher_rate",         # EB(k=200) pitcher main effect
    "e30_platoon_delta",        # EB(k=200) residual for pitcher x batter_hand
    "e30_platoon_n_log",        # log1p(n) of the pitcher-hand cell
    "e30_platoon_unseen",       # 1 when the pitcher-hand cell is absent
]


def build_platoon_state(history: pd.DataFrame, prior: float,
                        k_pitcher: float = 200.0, k_platoon: float = 200.0) -> dict:
    """Fit pitcher and platoon-residual encoders on outer history only.

    Returned state is a plain lookup dict; inference reads it per row and never
    touches another evaluation row.
    """
    g = history.groupby("pitcher_id")["control_success"].agg(["sum", "size"])
    pitcher_rate = (g["sum"] + k_pitcher * prior) / (g["size"] + k_pitcher)

    frame = history[["pitcher_id", "batter_hand", "control_success"]].copy()
    frame["base"] = frame["pitcher_id"].map(pitcher_rate).fillna(prior)
    frame["resid"] = frame["control_success"] - frame["base"]
    h = frame.groupby(["pitcher_id", "batter_hand"])["resid"].agg(["sum", "size"])
    platoon_delta = h["sum"] / (h["size"] + k_platoon)   # shrink toward 0

    return {
        "prior": float(prior),
        "k_pitcher": float(k_pitcher),
        "k_platoon": float(k_platoon),
        "pitcher_rate": {str(i): float(v) for i, v in pitcher_rate.items()},
        "platoon_delta": {f"{i}|{h_}": float(v) for (i, h_), v in platoon_delta.items()},
        "platoon_n": {f"{i}|{h_}": int(v) for (i, h_), v in h["size"].items()},
    }
```

**누수 방지 규칙.** E14와 동일하다.

- rolling fold에서는 **outer history(`season < Y`)로만** 인코더를 적합한다
- 제출 ZIP에서는 `season <= 2024` 전체로 적합해 `model/platoon_state.json`으로 동결하고 SHA-256을 manifest에 기록한다
- 추론 시에는 사전 조회만 하므로 행 독립성이 자명하다 — E14 상태 파일과 같은 취급

**측정 항목.** 다음을 fold별로 기록한다.

- `k_pitcher × k_platoon` 격자: `{50, 200, 500} × {50, 200, 500}`
- `e30_platoon_unseen` 비율 (2025 신규 투수 대비)
- **가장 중요:** 시즌 넘김 감쇠. 2024 내부 `+164.6`이 rolling fold에서 얼마나 남는지가 이 항목의 실제 값이다

### 4.2 B2 — 투수 정체성을 부스팅에 투입

**근거.** [`eda/EDA_REPORT.md` §22.2](eda/EDA_REPORT.md)는 투수 정체성의 시즌 내 신호 총량을 `646점`(투수 × 타자 손이면 `800점`)으로 실측했다. 그런데 [`experiments/run_baselines.py:111`](experiments/run_baselines.py):

```python
HGB_DROPPED = ["pitcher_id", "batter_id"]
```

**부스팅이 신호의 거의 전부인 축을 아예 보지 못한다.** 이것이 HGB가 저평가된 주요 원인이며, 앙상블이 Linear에 의존할 수밖에 없었던 구조적 이유다.

**제약.** sklearn `HistGradientBoostingClassifier`의 native categorical은 **255 카테고리 한도**다. 투수는 792명이라 그대로는 못 넣는다.

**해법 — `TargetEncoder` (sklearn 1.8.0 보유).**

```python
from sklearn.preprocessing import TargetEncoder

TargetEncoder(target_type="binary", smooth="auto", cv=5,
              shuffle=True, random_state=RANDOM_SEED)
```

- Pipeline 안에서 `fit` 시 **cross-fitted** 인코딩이 적용되어 학습 누수를 막는다
- `transform`(추론) 시에는 전체 학습셋 인코딩을 쓰므로 **행 독립적**이다
- 규정 확인: [`COMPETITION.md` §6 허용 예](COMPETITION.md)에 *"학습 데이터만으로 만든 선수별 prior, smoothing 통계, target encoding"* 이 명시적으로 허용돼 있다

**적용 범위.** `pitcher_id`만 넣는다. `batter_id`는 EDA가 `38점`(사실상 0)으로 실측했으므로 계속 drop한다.

**B1′과의 관계.** 둘 다 투수 정체성을 모델에 넣는 수단이지만 겹치지 않는다. B2는 투수 **주효과**를 트리에 주고, B1′은 트리가 만들 수 없는 **platoon 잔차**를 준다. B1′의 `e30_pitcher_rate`가 B2의 TE와 중복되므로, ablation에서 `B1′ 단독` / `B2 단독` / `B1′+B2`를 모두 비교하고 중복이 확인되면 B1′의 주효과 항을 뺀다.

**대안.** LightGBM은 고카디널리티 범주를 native로 처리하므로 Track C로 넘어가면 TE 없이 `pitcher_id`를 직접 넣을 수 있다. B2는 Track C 착수 전의 저비용 선행 실험으로 본다. **단 B1′은 LightGBM으로 가도 계속 필요하다.**

### 4.3 B3 — 정규화 완화

| 항목 | 현재 | 스윕 |
| --- | --- | --- |
| `SGDClassifier.alpha` | `0.3` | `0.3, 0.03, 0.003, 0.0003` |
| `eta0` | `0.001` 고정 | `0.001, 0.01`, 또는 `learning_rate="optimal"` |
| 대안 모델 | — | `LogisticRegression(solver="saga", C∈{0.01,0.1,1,10})` |
| HGB `max_iter` / `max_leaf_nodes` / `l2` | `250 / 31 / 5.0` | `500 / 63,127 / 0.1,1,5` |

각 조합에서 **예측 표준편차를 함께 기록한다.** 현재 Linear `0.0297`, HGB `0.0410`이며, 이 값이 올라가면서 2024 점수가 함께 오르면 과소적합 진단이 확정된다.

### 4.4 B4 — EDA가 권고했는데 구현되지 않은 무료 항목 ← 신규

두 항목 모두 EDA 제1부의 명시적 권고이며 지금까지 코드에 반영되지 않았다. 비용이 거의 없다.

#### 결측 indicator (EDA §7.1)

[`experiments/run_baselines.py:391`](experiments/run_baselines.py)의 선형 파이프라인은 `SimpleImputer(strategy="median")`을 쓰고 `add_indicator`가 없다. 그런데 EDA 실측은 다음과 같다.

| 컬럼군 | 결측률 | 결측 행 성공률 | 비결측 행 성공률 |
| --- | ---: | ---: | ---: |
| 최근 1/3/5경기 성공률·middle rate 6개 | `1.98%` | **`55.08%`** | `52.32%` |

**`2.76%p` 차이를 선형 모델이 통째로 버리고 있다.** EDA는 *"단순 평균 대치만 하지 말고 결측 indicator와 표본 수를 함께 둔다"*고 적었다. HGB는 NaN을 native로 처리하므로 이 손실은 선형 쪽에만 있다.

```python
SimpleImputer(strategy="median", add_indicator=True)
```

`add_indicator=True`는 serialized Linear pipeline의 numeric transformer 내부에서
`MissingIndicator` 컬럼을 뒤에 덧붙인다. 따라서 raw manifest의 `features` 순서는 바뀌지
않고, 학습·추론이 동일 joblib pipeline을 사용해 내부 순서를 보존한다.

#### 공선 정리 (EDA §9)

전수 검사에서 다음이 모든 행에서 성립한다.

```text
num_runners_on   = runner_on_1b + runner_on_2b + runner_on_3b
base_state       <- runner flag 3개로 완전 결정
run_total_before = run_top_before + run_bot_before
score_diff_home  <- 두 점수로 결정
score_diff_pitcher_team <- top_bottom + score_diff_home
asof_pitcher_pitchmix_n = asof_pitcher_n
fastball + breaking + offspeed = 1
```

`alpha=0.3`의 강한 L2에서 공선 피처끼리 신호를 나눠 갖고 **함께 수축한다.** EDA는 *"선형 모델: 한 관계에서 기준 피처 하나를 제거하여 수치 안정성 확보"*를 권고했다. 트리 쪽은 유지한 버전과 제거한 버전을 모두 비교한다.

관계당 하나씩 제거하는 후보: `num_runners_on`, `run_total_before`, `score_diff_pitcher_team`, `asof_pitcher_pitchmix_n`, `asof_pitcher_offspeed_rate`.

### 4.5 Track B 실행

```powershell
$env:PYTHONUTF8 = '1'
& .\.venv\Scripts\python.exe experiments\pipeline.py --run `
  --only b1 b2 b2b b3_linear b3_hgb b4
```

기준선은 S4 구성(E14 + `r_recent3`), 각 ablation은 한 번에 하나씩만 바꾼다. `count_state`는 §1.6 검증 1로 기각했지만 비용이 없으므로 대조군으로만 남긴다 — **개선이 나와도 채택 근거로 쓰지 않는다.**

---

## 5. Track C — LightGBM

### 5.1 근거

147만 행 · 50여 개 피처 · 고카디널리티 범주가 있는 정형 이진 문제다. 상위권 100팀이 도달한 `1,100+`는 **튜닝된 GBDT의 통상적 결과**로 보는 것이 가장 자연스럽다. 이 레포는 sklearn `HistGradientBoosting` **기본값 고정**만 써 봤고 탐색 흔적이 없다.

LightGBM의 이점:

- 고카디널리티 범주 native 처리 → `pitcher_id` 직접 투입 (TE 불필요)
- `num_leaves` 등으로 용량을 넓게 조절
- 학습이 sklearn HGB보다 빠름 → 탐색 반복 수 확보

### 5.2 로컬 환경 구성 — 기존 `.venv`를 깨뜨리지 않는다

`verify_submission.py`는 `sys.executable`, 즉 `.venv`의 파이썬으로 ZIP을 실행한다. 따라서 lightgbm은 **`.venv` 안에** 있어야 한다. 다만 무심코 설치하면 numpy/scipy/sklearn이 함께 올라가 S1~S8 재현성이 깨진다.

```powershell
# 1) 핵심 버전을 제약으로 고정한 채 설치
& .\.venv\Scripts\python.exe -m pip install `
  --disable-pip-version-check `
  --constraint requirements-baseline.txt `
  "lightgbm==4.7.0"

# 2) 의존성 정합성 확인
& .\.venv\Scripts\python.exe -m pip check

# 3) 핵심 버전이 그대로인지 확인
& .\.venv\Scripts\python.exe -c `
  "import numpy,scipy,sklearn,pandas;print(numpy.__version__,scipy.__version__,sklearn.__version__,pandas.__version__)"

# 4) 기존 후보가 여전히 재현되는지 확인 (필수)
& .\.venv\Scripts\python.exe submission\verify_submission.py `
  submission\archive\S4\S4.zip
```

3번이 `1.26.4 1.15.3 1.8.0 2.0.3`가 아니거나 4번이 실패하면 **즉시 롤백**하고 별도 `.venv_lgbm`으로 학습만 하되, ZIP 검증은 lightgbm을 넣은 전용 venv를 따로 만들어 수행한다.

> CatBoost는 v1에서 `.venv_catboost` 설치에 실패했지만 2026-08-20 공식 wheel로
> `1.2.8` 설치를 완료했다. 다만 전량 학습·GPU offload는 아직 검증 전이므로 c3/c3b는
> optional을 유지하고, 런타임 실패 시 해당 계열을 앙상블에서 제외한다.

### 5.3 평가 서버 반영

`submission/template/requirements.txt`에 정확한 핀을 추가한다.

```text
lightgbm==4.7.0
```

- 설치 한도 10분 안에 충분히 들어간다 (wheel 수 MB)
- **joblib 피클은 버전이 맞아야 로드된다.** 로컬과 서버 버전을 반드시 동일하게 핀한다
- 대안으로 `booster.save_model()` 텍스트 포맷을 쓰면 버전 민감도가 낮아진다 — 안정성을 원하면 이쪽을 권장한다

**설치만 먼저 검증한다.** 최소 크기 ZIP(더미 모델 + lightgbm import만 하는 `script.py`)을 하나 올려 설치가 통과하는지 확인한다. **설치 오류는 일일 제출 예산을 차감하지 않는다**([`COMPETITION.md` §8.5](COMPETITION.md)).

### 5.4 탐색 범위

```python
params = {
    "objective": "binary",
    "learning_rate": 0.03,
    "num_leaves": [63, 127, 255],
    "min_child_samples": [100, 500, 2000],
    "feature_fraction": [0.7, 0.9],
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "lambda_l2": [1.0, 10.0, 100.0],
    "n_estimators": 3000,          # early stopping으로 실제 값 결정
    "n_jobs": 6,
    "random_state": RANDOM_SEED,
}
categorical_feature = [
    "pitcher_id", "batter_id", "pitcher_team_id", "batter_team_id",
    "base_state", "game_type", "top_bottom", "count_state",
]
```

**early stopping의 내부 분할은 시간 순으로 한다.** sklearn HGB의 `validation_fraction=0.1`은 시간 정렬 데이터를 무작위로 자른다. LightGBM에서는 history의 마지막 시즌을 명시적 `eval_set`으로 준다.

**주 지표는 Brier다.** logloss로 학습하되 조기 종료와 선택은 Brier로 한다.

### 5.5 추론 예산

245,789행 LightGBM 추론은 수 초다. 현재 최대가 S8의 `49.15초`이고 한도가 10분이므로 여유가 크다. `verify_submission.py`의 mimic으로 매번 실측한다.

---

## 6. Track E — 검증 프로토콜 v2

### 6.1 기준 교체

| | v1 | **v2** |
| --- | --- | --- |
| 주 지표 | 3-fold 평균 Brier + 최악 fold ≥ 0 | **2024 fold 환산 점수** |
| 보조 | — | 2022 fold |
| 2023 fold | 동등 가중 | **기록만.** F 체제 단절 일회성 사건 |
| 통과 조건 | `wins>=2 and worst<=5e-4` | **2024 fold 개선 + paired bootstrap 95% CI가 0을 넘지 않을 것** |
| 폐기 | | `worst fold >= 0` 제약, `wins>=2` 단독 통과 |

**2024는 개발 fold가 된다.** 여기서 하이퍼파라미터를 고르면 2024 점수는 더 이상 비편향 추정치가 아니다. 이를 명시적으로 받아들이고 **최종 심판은 LB로 둔다.** 최고 점수가 유지되는 규칙 덕분에 이 선택이 안전하다.

### 6.2 paired bootstrap 유틸리티

`experiments/stats.py`에 구현했다. 같은 투수의 타석들이 독립이라는 가정을 피하기 위해
행이 아니라 **validation season의 `pitcher_id` cluster 전체를 복원추출**한다.

```python
def paired_bootstrap_brier_ci(
    y: np.ndarray,
    baseline: np.ndarray,
    candidate: np.ndarray,
    iterations: int = 2000,
    seed: int = RANDOM_SEED,
    clusters: np.ndarray | None = None,
    confidence: float = 0.95,
) -> dict:
    """Cluster-paired CI; negative favours the candidate."""
```

2024를 설정 선택과 평가에 함께 쓰므로 탐색 뒤 CI는 **탐색적(exploratory)** 으로 기록한다.
`aggregate_gate`도 `confirmatory: false`를 명시하며 최종 심판은 LB다.

### 6.3 계속 유지할 것

**제출 게이트는 완화하지 않는다.** `verify_submission.py`의 단일행·셔플·중복·배치 불변성, SHA-256 체인, 245,789행 모사, 시간·메모리 한도는 전부 그대로다. Phase 3는 상위권이 **코드 검증**을 통과해야 하므로 이 인프라는 진출 이후 오히려 더 중요해진다.

---

## 7. Track D — 기각 실험 재평가

Track B·C로 용량을 키운 **뒤에** 실행한다.

| ID | 내용 | v1 결과 | 기대 | 판단 |
| --- | --- | --- | :---: | --- |
| **E14R** | 투수 시즌 내 **reverse rate** 복원 | **미구현** | **중하** | EDA §21.5 단독 `225점`. 단 §8.3에서 success와 reverse의 상관이 **`−0.811`**이므로 E14 위의 한계 기여는 그보다 훨씬 작다 |
| E10 | 2022 이전 정규경기 F 배제 | 2/3, `+4.30e-4` | 중하 | 체제 단절 대응 자체는 유효한 방향. 용량 확대 후 재측정 |
| E11 | `game_type × pitcher_hand` EB | 1/3, `+1.15e-4` | 낮음 | B1′이 더 나은 형태의 같은 아이디어. B1′ 결과를 보고 판단 |
| ~~E20R~~ | Trackman 투수 프로파일 6종 | 1/3, `+1.31e-6` | **최하위** | **강등.** 초안은 "저용량 모델 탓"이라 했으나 사양 확인 결과 `e20_rel_speed_sd`·`e20_rel_side_sd` 등 EDA §19.7이 지목한 산포 피처를 **이미 포함**했다. 제대로 만들어졌고 제대로 기각됐다 |

> **E20R 재평가를 강등한 이유.** 투수 수준 Trackman 요약(평균 구속·회전수·무브먼트·산포)은 결국 "이 투수가 어떤 투수인가"를 말한다. 그런데 `asof_pitcher_success_rate`는 **같은 투수의 제구 결과를 직접 측정**한 값이다. 물리량은 그 결과의 원인 중 일부일 뿐이므로, 결과 지표가 이미 있는 상태에서 원인 지표의 한계 정보량은 작다. 이것이 `r = -0.171`(투수 간 집계 상관)이라는 큰 수치에도 불구하고 rolling에서 `+1.31e-6`이 나온 이유로 보인다. 남은 가능성은 **cold-start 투수 구간**뿐이므로, 재평가한다면 `asof_pitcher_n`이 작은 세그먼트로 한정해 측정한다.

### 7.1 E14R 구현 노트 (기대치 하향 반영)

> **먼저 중복부터 측정한다.** EDA §8.3의 통산 상관 `−0.811`이 시즌 내 버전에서도 유지되는지 확인하고, 유지되면 E14R을 만들지 않는다. `e14_rate_season`과 `e14r_reverse_rate_season`의 상관 한 줄이면 판정된다. 이 확인 없이 구현하면 반나절을 중복 피처에 쓴다.


E14와 동일한 메커니즘이다. `asof_pitcher_reverse_rate × asof_pitcher_n`으로 통산 reverse 횟수를 복원하고, 시즌 시작 시점 상태와 차분해 시즌 내 reverse 횟수를 얻는다.

**한 가지 차이.** [`experiments/run_e14_rolling.py:115`](experiments/run_e14_rolling.py)의 `season_end_state`는 마지막 행의 `control_success` 라벨로 카운터를 1 전진시킨다. reverse는 행 단위 라벨이 데이터에 없으므로 이 전진을 할 수 없다.

**대응:** 마지막 행 전진 없이 `(n_end, reverse_end)`를 그대로 쓴다. 투수-시즌당 최대 1구 오차이고 `n`은 보통 수백~수천이므로 무시할 수 있다. E14와 동일하게 `e14r_counter_invalid` 가드를 둔다.

`middle_rate`·`wide_rate`는 EDA가 `0점`으로 실측했으므로 만들지 않는다.

---

## 8. Track F — 앙상블·calibration 재설계

D8 이후, 개별 모델이 안정된 뒤에 착수한다.

1. **성분 재구성.** Linear / HGB / LightGBM / (가능하면 CatBoost) 4종. 성분이 서로 다른 모델 계열이어야 앙상블 이득이 생긴다. v1의 S4~S7은 전부 같은 Linear+HGB 위의 피처 변형이라 상관이 지나치게 높았고, 그래서 이득이 `1e-6`대에 머물렀다.
2. **가중치 선택.** 단일 시즌 최적 가중치는 v1에서 `3.3% → 100%`로 튀며 실패했다([`experiments/CALIBRATION_ENSEMBLE_REPORT.md` §5](experiments/CALIBRATION_ENSEMBLE_REPORT.md)). 2022·2024 두 fold의 평균 Brier를 최소화하되 성분별 하한 없이 탐색하고, §6.2의 CI로 유의성을 확인한다.
3. **calibration 재시도.** v1은 직전 한 시즌만으로 학습해 실패했다. 여러 rolling fold의 OOF 예측을 모아 isotonic/Platt을 다시 설계한다.

### 8.1 base rate 잔여 보정 — D2로 앞당기는 저비용 1회

[§1.6 검증 3](#검증-3--base-rate는-이미-대부분-해결돼-있다-초안의-track-f-배치가-옳았다)에서 잔여 편향을 실측했다. **E14가 이미 대부분 흡수해서 `+0.82%p`, 약 `27점`만 남아 있다.** 초안이 EDA §22.3의 "230점"을 근거로 이 항목을 앞세우려 한 것은 과대평가였다.

그럼에도 `27점`은 재학습이 필요 없고 구현이 한 줄이므로 **D2에 저비용 1회만 시도한다.**

**방법.** 이미 구현된 `logit_intercept` 보정을 재사용하되, **적합 방식을 바꾼다.**

| | v1 (실패) | v2 |
| --- | --- | --- |
| 절편 학습 | 직전 시즌 예측으로 적합 | **적합하지 않는다** |
| 목표 평균 | 직전 시즌 실측 평균 | rolling fold에서 관측된 **잔여 편향 `+0.82%p`** 를 상쇄하는 고정값 |
| 결과 | refit 모델과 스케일 불일치 → 이중 보정 | 모델 출력 평균을 `−0.8%p` 이동 |

```python
from scipy.optimize import brentq
from scipy.special import expit, logit


def shift_to_target_mean(p: np.ndarray, target_mean: float) -> float:
    """Solve for the single logit offset that moves mean(p) to target_mean.

    A global constant applied identically to every row: row-independent by
    construction, and the invariance gate covers it.
    """
    q = np.clip(p, 1e-6, 1 - 1e-6)
    return brentq(lambda d: expit(logit(q) + d).mean() - target_mean, -3.0, 3.0)
```

**후보 shift.** 세 수준을 만들고 각각 별도 후보로 제출한다. 최고 점수만 남으므로 하방이 없다.

| 후보 | shift | 근거 |
| --- | ---: | --- |
| 무보정 | `0.0%p` | 기준 |
| 보수 | `−0.8%p` | 2023·2024 fold에서 관측된 잔여 편향 |
| 공격 | `−1.6%p` | 잔여 편향 + 2025 추가 하락분 |

> **이중 보정 금지.** EDA §22.3 말미의 경고대로, E14 시즌 내 피처가 이미 자동 보정 중이다. **rolling fold에서 잔여 편향을 다시 측정한 뒤**에만 shift를 정하고, 원시 시즌 추세(`−1.2%p/년`)를 그대로 빼지 않는다.
>
> **LB 탐침은 하지 않는다.** 위 세 후보는 학습 데이터에서 도출한 정상적인 모델 변형이다. 반면 LB 점수를 되먹여 2025 평균을 역산하는 이분 탐색은 [`COMPETITION.md` §9.1](COMPETITION.md)의 "평가 데이터셋 유출 시도" 판정 위험이 있다. **3회 이상의 shift 후보를 만들지 않는다.**

---

## 9. 실행 — 연속 파이프라인

일자별 제출 루프를 폐기하고 **한 번에 끝까지 돌린 뒤 사람이 일괄 제출**하는 구조로 바꿨다.
근거는 §6.1이다. 후보 선택 기준이 이미 2024 fold + 부트스트랩 CI이지 LB 피드백이 아니므로,
중간 제출은 의사결정에 기여하지 않으면서 시간만 쓴다.

```powershell
$env:PYTHONUTF8 = '1'
& .\.venv\Scripts\python.exe experiments\pipeline.py --status   # 진행 상황
& .\.venv\Scripts\python.exe experiments\pipeline.py --run      # 시작 / 이어서 진행
& .\.venv\Scripts\python.exe experiments\pipeline.py --run --kaggle   # 무거운 단계 원격 실행
```

파이프라인은 `experiments/pipeline_state.json`에 체크포인트를 남긴다. **중단·재부팅·실패
후 같은 `--run` 명령으로 재개**되며 완료된 단계는 다시 돌지 않는다.

### 9.1 단계

| key | 내용 | 예상 시간 | 실패 시 | offload |
| --- | --- | ---: | --- | :---: |
| `preflight` | 고정 버전 확인 · lightgbm 설치 · S4 재현성 재확인 | 5분 | 중단 | — |
| `probe` | §1.6 수치 재현 | 1분 | 중단 | — |
| `a_blend` | **Track A** — S9~S11 생성 + 게이트 | 5분 | 중단 | — |
| `base` | 기준선 재현 (Linear 90 + HGB 10) | 15분 | 중단 | — |
| `c1` | **Track C-1** — LightGBM 기본값 | 30분~1시간 | 중단 | cpu |
| `c2` | **Track C-2** — 그리드 12설정 탐색 | **3~8시간** | 중단 | cpu |
| `b1` | **Track B1′** — platoon split | 1시간 | 중단 | — |
| `b2` | **Track B2** — HGB 투수 cross-fitted TargetEncoder | 30분 | 중단 | — |
| `b2b` | B1′+B2 중복 ablation | 30분 | 중단 | — |
| `b3_linear` | **Track B3** — Linear alpha·eta0 탐색 | 1~2시간 | 중단 | — |
| `b3_hgb` | **Track B3** — HGB capacity·L2 탐색 | 2~4시간 | 중단 | — |
| `b4` | Track B4 — 결측 indicator·공선 정리 | 20분 | 계속 | — |
| `c3` | **Track C-3** — CatBoost | 2~6시간 | 계속 | **gpu** |
| `c3b` | Track C-3b — CatBoost + platoon 대조군 | 2~6시간 | 계속 | **gpu** |
| `f1` | Track F1 — 앙상블 가중치 (재학습 없음) | 1분 | 중단 | — |
| `package` | 후보 ZIP 빌드 (전체 재학습 + shift 3종) | 1~3시간 | 중단 | — |
| `final` | 전수 게이트 + `SUBMIT_QUEUE.md` | 30분 | 중단 | — |

전체 로컬 실행 예상 **12~24시간**. `--kaggle`로 `c2`·`c3`·`c3b`를 병렬 원격 실행하면 크게 줄어든다.

### 9.2 단계 간 연결

각 rolling 단계는 fold별 검증 예측을 `experiments/results/predictions/*.npz`로 저장하고,
다음 단계는 `--baseline-stage`로 그것을 직접 baseline 삼아 비교한다. 앙상블 탐색(`f1`)은
저장된 예측만 읽으므로 **재학습이 전혀 없다.**

```
base ──┬─> c1 ──> c2 ──> b1 ──────────────────────────────┐
       ├─> b2 ─> b2b ─> b3_hgb ──────────────────────────┤
       ├─> b3_linear ─> b4 ──────────────────────────────┤
       └─> c3 ─> c3b ────────────────────────────────────┤
                                                            └─> f1 ─> package ─> final
```

### 9.3 끝나면

`submission/SUBMIT_QUEUE.md`에 **게이트를 통과한 후보만** 기대값 순으로, 하루 5건씩 묶여
나온다. 업로드는 사람이 한다 — 파이프라인은 제출하지 않고 제출 코드도 만들지 않는다.

각 제출 후 서버 실행 결과와 점수를 [`submission/SUBMISSION_LOG.md`](submission/SUBMISSION_LOG.md)에 기록한다.

### 9.4 참고 — 폐기한 일자별 계획

> 아래는 §9로 대체된 초안의 일자별 계획이다. 제출 예산 배분의 참고로만 남긴다.
> 2026-08-18에 이미 5회(S8·S4·S6·S3·S5)를 제출했으므로 그날의 한도는 소진됐다.

| 일자 | 작업 | 제출 |
| --- | --- | --- |
| **D0 (8/18 밤)** | `reweight_candidate.py` 작성 · S9/S10/S11 빌드 · 3개 모두 `verify_submission` 통과 | 0 (대기) |
| **D1 (8/19)** | S9·S10·S11 제출. `stats.py` 부트스트랩 추가 후 과거 실험에 소급 적용. **B1′ platoon rolling 착수** | **3** |
| **D2 (8/20)** | B1′ rolling 결과 → ZIP 빌드·제출. **§8.1 base rate shift 3후보 중 2개 제출.** LightGBM 설치 및 `.venv` 무결성 확인 + 설치 검증용 최소 ZIP | **3** (+설치 테스트) |
| **D3 (8/21)** | B2·B4 ablation rolling. LightGBM rolling 1차 (기본 파라미터 + native categorical). 제출 | **2** |
| **D4 (8/22)** | LightGBM 하이퍼파라미터 탐색 1차 + B1′ 결합. 제출 | **2** |
| **D5 (8/23)** | 탐색 2차. B3 정규화 스윕. **E14R 중복 확인 한 줄**(§7.1) | **2** |
| **D6 (8/24)** | E14R이 중복 판정을 통과했을 때만 rolling · 제출 | **2** |
| **D7 (8/25)** | E10 재평가 결과 반영 · 제출 | **2** |
| **D8 (8/26)** | Track F 앙상블 성분 구성. *팀 병합 마감 23:59 — 해당 시 오늘 처리* | **2** |
| **D9 (8/27)** | 앙상블 가중치 rolling · 제출 | **3** |
| **D10 (8/28)** | calibration 재설계 (다중 fold OOF) · 제출 | **3** |
| **D11 (8/29)** | 상위 후보 3종 전체 재검증 · 제출 | **3** |
| **D12 (8/30)** | 예비: 미탐색 방향 또는 실패 복구 | **3** |
| **D13 (8/31)** | 최종 후보 동결. 전 후보 재검증·아카이브·해시 기록 | **2** |
| **D14 (9/1 오전)** | 예비만. **10:00 마감 — 09:00까지 모든 작업 종료** | 0~2 |

누계 계획 제출 약 `31회` / 가용 약 `65~70회`. **여유가 크므로 확신이 서지 않는 후보도 올린다.** 하방이 0이기 때문이다.

---

## 10. 리스크 레지스터

| # | 리스크 | 영향 | 대응 |
| ---: | --- | --- | --- |
| 1 | HGB 중심 모델이 2025에서 붕괴 (2023형 체제 단절) | LB 낮은 점수 | **대응 불필요.** 최고 점수 유지 규칙으로 `689.40`이 보존된다 |
| 2 | `.venv`에 lightgbm 설치하다 numpy/sklearn 손상 | S1~S8 재현 불가 | `--constraint requirements-baseline.txt` 설치 → `pip check` → S4 재검증. 실패 시 즉시 롤백 |
| 3 | 서버에서 lightgbm 설치 실패 | 제출 불가 | 설치 오류는 **예산 차감 없음.** 최소 ZIP으로 먼저 설치만 검증 |
| 4 | joblib 피클 버전 불일치로 서버 로드 실패 | 제출 오류(예산 차감) | requirements.txt에 정확 핀. 또는 `booster.save_model()` 텍스트 포맷 사용 |
| 5 | 추론 10분 초과 | 제출 오류(예산 차감) | `verify_submission.py` mimic으로 매번 실측. 현재 최대 49초로 여유 큼 |
| 6 | TargetEncoder가 규정 위반이라는 오해 | 불필요한 자기검열 | [`COMPETITION.md` §6](COMPETITION.md) 허용 예에 target encoding이 명시돼 있음. 학습 데이터로만 fit |
| 7 | 2024 fold 과적합 | 로컬-LB 괴리 | 이미 알고 수용한 선택(§6.1). LB가 심판. 최고 점수 유지로 안전 |
| 8 | 시간 부족으로 Track C 미완 | 목표 미달 | Track A만으로도 상당 폭 기대. A → B → C 순서를 지켜 조기 이득을 먼저 확보 |
| 9 | 행 독립성 위반으로 실격 | **치명적** | 게이트 완화 금지(§6.3). 모든 제출에 `verify_submission.py` 전 항목 통과 |
| 10 | Phase 3 코드 검증 탈락 | 진출 무효 | 학습 코드·환경·해시 기록을 계속 유지. 이미 잘 하고 있는 부분 |

---

## 11. 실행 체크리스트

### 오늘 밤

- [x] `submission/reweight_candidate.py` 작성
- [x] S9(0.5/0.5) · S10(0.2/0.8) · S11(0.0/1.0) 빌드
- [x] 3개 모두 `verify_submission.py` **PASSED**, 불변성 delta `< 1e-12`
- [x] `submission/records/S*_build.json` 기록
- [x] S9~S11 ZIP·증거를 `submission/archive/`에 독립 보존
- [x] LightGBM 4.5.0 constraint 설치 · `pip check` · S4 동일 SHA 재검증
- [x] CatBoost 1.2.8 constraint 설치 · CPU early-stopping smoke · S4 재검증
- [ ] 오늘 잔여 제출 횟수 확인

### 매 제출 전 (전부 유지)

- [ ] ZIP 최상위가 `model/`, `script.py`, `requirements.txt`뿐
- [ ] 단일행·셔플·중복·배치 불변성 통과
- [ ] 245,789행 모사 통과, 시간·RAM 한도 내
- [ ] 출력 컬럼 `row_id`, `control_success`, 확률 `[0,1]`
- [ ] ZIP SHA-256을 `SUBMISSION_LOG.md`에 기록
- [ ] 오프라인 실행 가능 (원격 호출 없음)

### 후보 채택 기준 (v2)

- [x] 2024 fold 환산 점수 개선 — 선택 ensemble `696.5` vs 기준 `547.2`
- [x] paired bootstrap 95% CI 상한 `< 0` — `-2.665e-4`
- [x] 2022 fold에서 큰 악화 없음 — ensemble `2324.3` vs 기준 `1658.0`
- [x] 2023 fold 결과는 기록만 — 채택 여부를 결정하지 않음

### 문서 갱신

- [x] `README.md`에 이 문서 링크 추가
- [x] `SUBMISSION_LOG.md`에 S9 이후 이력 추가
- [x] `EXPERIMENT_REGISTRY.md`에 V2 Track B/C/F 및 package/final 결과 등록
- [x] 부정 결과도 계속 보존 — 설정별 JSON·CSV·fold 예측·로그 유지

---

## 12. 성공 기준

### 12.1 항목별 기대 누계

§1.6의 측정을 반영한 회계다. **측정 가능한 항목을 다 더해도 격차가 남는다**는 점이 Track C를 유지하는 근거다.

| 항목 | 근거 | 기대 |
| --- | --- | ---: |
| Track A 블렌드 재조정 | 2024 fold 실측 `+175.9` | **`+176`** |
| B1′ platoon split | 2024 실측 `+135~165`, 시즌 감쇠 적용 | **`+60~80`** |
| §8.1 base rate 잔여 보정 | S4 잔여 편향 `+0.82%p` 실측 | **`+27`** |
| B2·B4 (투수 TE·결측 indicator·공선 정리) | EDA 권고, 미측정 | 소폭 |
| ~~count_state~~ | 실측 `30.7`, GBDT 한계 `~0` | **`~0`** |
| E14R / E20R | 중복성 근거 | 작음 |
| **소계** | | **`+265~285`** |
| **필요 개선폭** | | **`+410`** |
| **미설명 잔차 → Track C** | | **`+130` 이상** |

### 12.2 단계별 판정

| 단계 | 목표 | 판정 시점 |
| --- | ---: | --- |
| Track A | `> 800` | D1 |
| Track A + B1′ + base rate | `> 900` | D2~D3 |
| Track C | `> 1,050` | D4~D6 |
| Track D~F | **`> 1,100`** (Phase 3 커트라인) | D9~D13 |

Track A가 기대에 못 미치면(`< 750`) 진단이 틀린 것이므로 **즉시 재검토한다.** 그 경우 다음을 의심한다.

1. 2025의 체제가 2024와 크게 다르다 → base rate·drift 대응(Track F calibration)을 앞당긴다
2. 상위권이 쓰는 것이 GBDT 튜닝이 아닌 다른 무엇이다 → Trackman 활용(E20R 계열)을 앞당긴다
3. E14 상태 파일이 2025 test에서 예상과 다르게 동작한다 → mimic을 2025 형태로 재구성해 `e14_counter_invalid` 비율을 실측한다

---

## 13. 부수적 정리 (여유 시간에만)

우선순위 최하위다. 점수에 기여하지 않는다.

- Git 저장소 초기화 (현재 Git CLI 미설치, `.gitignore`가 무효 상태)
- `LG Aimers/LG Aimers` 중첩 경로, `__MACOSX`, 상위의 빈 `experiments/results/archive/E10_v1` 정리
- `submission/build_*.py` 5종의 공통 로직 통합
- `submission/template/script.py`의 3중 분기 중복 정리 — `row_derived` 스펙 도입 시 자연히 해소
- `script.py:200-228`의 스칼라/벡터 대조는 같은 코드 경로를 자기 자신과 비교하는 항등식이다. `row_derived` 도입 시 진짜 벡터화 경로 대 스칼라 경로로 교체한다
- 깨진 `.venv_catboost` 제거

---

## 14. §1.6 측정 재현

§1.6의 모든 수치는 [`experiments/probe_grouping_ceilings.py`](experiments/probe_grouping_ceilings.py) 한 파일에서 나온다. 진단용 probe이며 제출 후보나 아티팩트를 만들지 않는다.

```powershell
$env:PYTHONUTF8 = '1'
& .\.venv\Scripts\python.exe experiments\probe_grouping_ceilings.py `
  --output experiments\results\probe_grouping_ceilings.json
```

현재 PC 실행 결과는 [`experiments/results/probe_grouping_ceilings.json`](experiments/results/probe_grouping_ceilings.json)에 보존했다.

| 출력 키 | §1.6 대응 |
| --- | --- |
| `check1_situational_ceilings_k50` | 검증 1 — `count_state` `30.75`, `hand_pair` `104.10` |
| `check2_identity_ceilings` | 검증 2 상단 — `pitcher_id` `652.66`, `× batter_hand` `814.07` |
| `check2_platoon_marginal` | 검증 2 하단 — platoon 잔차 한계 기여 `+135.9 ~ +164.6` |
| `check3_regular_season_rate` | 검증 3 — R 시즌 추세와 외삽 후보 |

### 14.1 split-half 프로토콜

2024를 시드 고정 무작위 반으로 나눠 **한쪽에서 EB 그룹 평균을 추정하고 다른 쪽에서 채점한다.** in-sample 과적합이 없으므로 "완벽한 시즌 내 그룹 추정이 낼 수 있는 총량"의 정직한 상한이다. EDA §22.2와 같은 프로토콜이며 그 표를 재현한다.

### 14.2 이 수치들의 한계

1. **시즌 내 상한이지 전이 가능량이 아니다.** EDA §22.2가 실측한 대로 시즌을 넘기면 투수 신호는 `646 → 250~340`으로 절반 이하가 된다. B1′의 실제 이득은 rolling fold로만 확정된다.
2. **단독 그룹 상한이지 한계 기여가 아니다.** 검증 1의 `104.10`은 모델이 이미 가진 것 위의 증분이 아니다. 현재 모델은 `pitcher_hand`·`batter_hand`를 각각 one-hot으로 갖고 있고 트리는 2×2 교차를 스스로 찾는다. 그래서 §1.6 검증 5의 비가산성 주의가 필요하다.
3. **`check2_platoon_marginal`만 한계 기여다.** 투수 주효과를 먼저 적합하고 잔차만 추가했으므로 §4.1이 구현할 형태와 일치한다. B1′의 근거로 쓸 수 있는 것은 이 표뿐이다.
4. **시드 하나다.** 채택 판단 전에 시드 몇 개로 안정성을 확인한다.
