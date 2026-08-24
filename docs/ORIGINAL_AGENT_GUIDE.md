# 에이전트 작업 지시서

이 저장소를 자율 실행하는 코딩 에이전트를 위한 문서다. 현재 사람이 읽는 활성 계획은
[`EXPERIMENT_PLAN_V5.md`](EXPERIMENT_PLAN_V5.md)에 있다. V3/V4와 V2 완료 이력은 각각
[`EXPERIMENT_PLAN_V3.md`](EXPERIMENT_PLAN_V3.md),
[`EXPERIMENT_PLAN_V2.md`](EXPERIMENT_PLAN_V2.md)에 있다. 이 문서는 **무엇을 실행하고
무엇을 절대 하지 말아야 하는지**만 다룬다.

---

## 0. 한 줄 목표

실제 리더보드 챔피언은 `1090.9100565103`(`V3_sparse_m3_1103`, 2026-08-20)이다.
`V4_compact_supported_1193`은 2024 로컬 `1052.9439576`, 이전 예상 `1193.0915411`이었지만
사용자 보고 실제 LB는 **약 `1005`**였다. 정확한 소수점·제출 ID는 아직 기록 대기다.

현재 목표는 **V5의 사전 고정한 다중 시간축 계약에서 보수적 예상 LB 하한 `1,190` 초과,
또는 실제 LB `1,190` 초과**다. 기존 `2024 local + 140.1476`은 폐기됐고, 2024 한 fold에
맞춘 signed stack은 실제 LB 없이는 완료 근거가 될 수 없다. 실제 제출은 자동화하지 않는다.
마감은 **2026-09-01 10:00 KST**.

추가 감사에서 `v3_sparse_m3_frozen`의 과거 2022·2023 예측에도 2024에서 고른 가중치와
affine 보정이 역적용된 것이 확인됐다. 이 artifact는 V3 재현용일 뿐 V5 개발 기준점으로
쓰지 않는다. 구조 ablation은 동일 부모 recipe와 비교하고, 앙상블은 목표 연도 직전
시즌에서만 가중치·보정을 정한 `v5_honest_m3_r_identity`와
`v5_honest_m3_r_grid`를 모두 사용한다. 상세 계약은
`experiments/params/v5_validation_contract_v2.json`이다.

V4 ZIP은 실패 감사용으로 보존하며 다음 제출 후보가 아니다. 새 후보는 V5 검증 계약과
전수 ZIP 게이트를 모두 통과한 뒤에만 `SUBMIT_QUEUE.md`에 올린다.

---

## 1. 시작 명령

기존 파이프라인은 V2 이력 재현용이며 현재 V5 Goal을 완료하지 않는다. V5 시작 감사는
다음과 같다.

```powershell
$env:PYTHONUTF8 = '1'
& .\.venv\Scripts\python.exe experiments\audit_v4_failure.py
```

기존 파이프라인 상태·재현 명령은 아래와 같다.

```powershell
$env:PYTHONUTF8 = '1'
& .\.venv\Scripts\python.exe experiments\pipeline.py --status   # 현재 진행 상황
& .\.venv\Scripts\python.exe experiments\pipeline.py --run      # 시작 또는 이어서 진행
```

파이프라인은 **중단되어도 이어서 재개된다.** 상태는 `experiments/pipeline_state.json`에
있고, 완료된 단계는 다시 실행하지 않는다. 실패하면 그 단계에서 멈추고, 고친 뒤 같은
`--run` 명령으로 재개한다.

특정 단계부터 다시: `--from c1` · 특정 단계만: `--only b1` · 건너뛰기: `--skip c3 c3b`  
기존 산출물 채택: `--run --only probe a_blend --adopt-existing` (단계별 무결성 검사를 통과해야 함)

`--from`은 이후 단계의 결과·예측을 `experiments/results/archive/pipeline_rerun_<UTC>/`에
먼저 옮긴 뒤 체크포인트를 지운다. `--skip`도 `skipped` 상태로 저장된다.

---

## 2. 절대 하지 말 것

1. **제출하지 않는다.** 파이프라인은 ZIP과 `submission/SUBMIT_QUEUE.md`까지만 만든다.
   DACON 업로드는 사람이 한다. 자동 업로드 코드를 작성하지 않는다.
2. **`submission/verify_submission.py` 게이트를 완화하거나 우회하지 않는다.**
   단일행·셔플·중복·배치 불변성, 245,789행 모사, SHA-256 검증은 전부 유지한다.
   게이트에 실패한 ZIP은 `SUBMIT_QUEUE.md`에 올리지 않는다. 실격 위험이 실재한다.
3. **평가 행 독립성을 깨지 않는다.** 어떤 행의 예측도 다른 평가 행에 의존해서는 안 된다.
   추론 코드에 `groupby`, `rolling`, `expanding`, `fit`, test 전체 통계가 들어가면 안 된다.
   모든 파생 피처는 **그 행의 값으로 키를 만든 사전 조회**여야 한다.
4. **외부 데이터를 쓰지 않는다.** `open/` 안의 파일만 사용한다.
5. **미래 정보를 쓰지 않는다.** 인코더·상태·prior는 항상 해당 시점 **이전** 데이터로만
   만든다. rolling fold에서는 `season < Y`, 제출 빌드에서는 `season <= 2024`.
6. **Kaggle 데이터셋·노트북을 public으로 바꾸지 않는다.** §6 참조.
7. **기존 결과를 덮어쓰지 않는다.** 재실행 시 `experiments/results/archive/<ID>/`에 먼저
   복사한다. 부정 결과도 보존한다.

---

## 3. 파이프라인 단계

| key | 내용 | 소요(예상) | 실패 시 |
| --- | --- | --- | --- |
| `preflight` | 고정 버전·LightGBM·S4 재현성 확인 | 5분 | 중단 |
| `probe` | 계획서 §1.6 수치 재현 | 1분 | 중단 |
| `a_blend` | Track A — S4 manifest weight만 변경해 S9~S11 생성 | 5분 | 중단 |
| `base` | 기준선 재현 (Linear 90 + HGB 10, base+e14) | 15분 | 중단 |
| `c1` | LightGBM 기본값 | 30분~1시간 | 중단 |
| `c2` | LightGBM 그리드 탐색 (12개 설정) | **3~8시간** | 중단 |
| `b1` | 투수 × 타자 손 platoon split | 1시간 | 중단 |
| `b2` | HGB + cross-fitted 투수 TargetEncoder | 30분 | 중단 |
| `b2b` | HGB 투수 TE + platoon 중복 ablation | 30분 | 중단 |
| `b3_linear` | Linear alpha·eta0 탐색 | 1~2시간 | 중단 |
| `b3_hgb` | HGB leaf·L2·반복 수 탐색 | 2~4시간 | 중단 |
| `b4` | 결측 indicator·공선 정리 | 20분 | **계속** (optional) |
| `c3` | CatBoost | 2~6시간 | **계속** (optional) |
| `c3b` | CatBoost + platoon | 2~6시간 | **계속** (optional) |
| `f1` | 앙상블 가중치 탐색 (저장된 예측만 사용) | 1분 | 중단 |
| `package` | 후보 ZIP 빌드 (전체 데이터 재학습) | 1~3시간 | 중단 |
| `final` | 전수 게이트 + 제출 큐 | 30분 | 중단 |

`optional=True` 단계는 실패해도 파이프라인이 계속 진행한다. CatBoost는 v1에서 설치가
실패한 이력이 있으므로 optional로 두었다.

---

## 4. 우선순위 근거 (바꾸기 전에 읽을 것)

이 순서는 측정에 기반한다. 임의로 바꾸지 않는다.

- **Track C(부스터)가 Track B(피처)보다 먼저다.** Track A의 천장이 약 850~880인데
  평범한 GBDT가 이미 ~900에 도달한다는 관측이 있다. 계획서 §12.1의 회계에서
  측정 가능한 비-GBDT 항목을 다 더해도 `+265~285`이고 필요한 폭은 `+410`이다.
- **`count_state`(볼카운트 교차)는 기각됐다.** 2024 split-half 실측 `30.7점`. 만들지 말 것.
  근거: `experiments/probe_grouping_ceilings.py`, 계획서 §1.6 검증 1.
- **`투수 × 타자 손` platoon split은 실측 `+135~165점`이다.** 최우선 피처 항목.
  단 시즌 넘김 감쇠 후 실제 전이분은 `+60~80` 추정이며, `b1` 단계가 이를 확정한다.
- **2023 fold는 채택 판단에 쓰지 않는다.** 2군 리그 라벨 체제 단절이라는 일회성 사건이다
  (EDA §20.2). 기록만 한다. 주 지표는 **2024 fold**, 보조는 2022다.
- **게이트는 `experiments/stats.py`의 pitcher-cluster paired bootstrap CI다.** v1의
  `wins>=2 and worst<=5e-4`는 측정 대상보다 200배 큰 임계값이라 노이즈를 통과시켰다.
  2024는 개발 fold이므로 하이퍼파라미터 선택 뒤 CI는 confirmatory가 아니라 exploratory다.

---

## 5. 코드 지도

```
experiments/
  pipeline.py                  ← 오케스트레이터. 여기서 시작한다
  audit_v4_failure.py          ← V4 예상/실제 역전과 cross-fold 계수 불안정 감사
  run_v2_rolling.py            ← 모든 rolling 실험의 단일 진입점
  stats.py                     ← paired bootstrap 게이트
  search_booster.py            ← 하이퍼파라미터 그리드 탐색
  search_ensemble.py           ← 저장된 예측으로 가중치 탐색 (재학습 없음)
  probe_grouping_ceilings.py   ← 계획서 §1.6 수치 재현
  params/*_grid.json           ← LightGBM·Linear·HGB 탐색 그리드
  params/linear_b4.json        ← 결측 indicator·공선 제거 설정
  results/                     ← 단계별 JSON/CSV
  results/predictions/*.npz    ← fold별 검증 예측. 앙상블이 이것만 읽는다
  pipeline_state.json          ← 체크포인트

submission/
  reweight_candidate.py        ← manifest weight만 변경 (재학습 0)
  sweep_blend_candidates.py    ← Track A 일괄 실행 + 게이트
  build_v2_candidate.py        ← 설정 하나를 전체 데이터로 학습해 ZIP 생성
  build_from_ensemble.py       ← 앙상블 결과를 ZIP들로 변환
  make_submit_queue.py         ← 전수 게이트 + SUBMIT_QUEUE.md 생성
  verify_submission.py         ← 제출 게이트. 절대 수정하지 않는다
  template/script_v2.py        ← v2 추론 코드 (schema_version 2)
  template/script.py           ← v1 추론 코드 (S1~S8용, 건드리지 않는다)
  template/script_v4_compact.py ← V4 22모델 행 독립 추론 코드
  prepare_v4_full_refit.py      ← 공식 2019~2024 전체 재학습 입력 준비
  run_v4_full_refits.py         ← V4 student·18 arm 전체 재학습/export
  build_v4_compact.py           ← V4 deterministic ZIP 빌드·sample parity
  dist/                        ← 생성된 ZIP
```

일반 rolling 실험은 `run_v2_rolling.py`의 `--models` / `--features` 조합으로 실행한다.
전체 재학습·패키징처럼 역할이 다른 단계만 `submission/`의 전용 스크립트를 사용한다.
기존 산출물 이름은 재사용하지 않고 V4처럼 새 stage/candidate ID로 보존한다.

---

## 6. Kaggle offload (선택)

로컬 CPU는 Ryzen 5 5600(6C/12T)이고 Kaggle CPU는 더 빠르지 않다. 실익은
**CatBoost GPU**, 12시간 무인 세션, 로컬과의 병렬 실행이다.

```powershell
pip install kaggle
# Kaggle > Account > API > Create New Token  →  %USERPROFILE%\.kaggle\kaggle.json
& .\.venv\Scripts\python.exe kaggle_offload\offload.py --setup --sync --i-understand-rule-9-3
& .\.venv\Scripts\python.exe experiments\pipeline.py --run --kaggle
```

`--kaggle`을 주면 `offload` 표시가 있는 단계(`c1`, `c2`, `c3`, `c3b`)만 원격 실행된다.

> **규정 경고.** 이것은 대회 데이터를 제3자 서비스에 복사하는 행위다.
> `COMPETITION.md §9.3`은 비참가자에게의 전송·복제·재배포를 금지한다.
> 그래서 `offload.py`는 데이터셋과 노트북을 **강제로 private**으로 만들고,
> `--i-understand-rule-9-3` 없이는 동작하지 않는다.
> **절대 public으로 바꾸지 않는다.** 확신이 없으면 Kaggle 없이 로컬로만 실행한다 —
> 파이프라인은 CatBoost가 느려질 뿐 정상 동작한다.

---

## 7. 완료 조건

V4의 패키지 검증은 성공했지만 성능 Goal은 실패했다. V5 Goal은 다음을 모두 만족할 때만
완료한다.

- [x] V4 예상 `1193.0915` → 실제 약 `1005` 실패를 감사하고 기존 offset 폐기
- [x] V5 다중 시간축·저복잡도·실제 앵커 기반 판정식을 확인 결과 전에 고정
- [x] 2024 선택값이 역적용된 과거 anchor를 폐기하고 one-year-ahead honest anchor 생성
- [ ] 잠긴 단일 recipe 또는 최대 3개 비음수 앙상블이 비단절 개발 fold와 2024에서 재현
- [ ] V3 actual을 기준으로 한 보수적 예상 LB 하한 `> 1190`, 또는 실제 LB `> 1190`
- [ ] 공식 데이터만 사용하고 test 집계·행간 정보·외부 API를 쓰지 않음
- [ ] 전체 재학습과 research/ZIP sample parity 통과
- [ ] ZIP 구조·SHA·불변성·245,789행 시간·RAM 검증 `PASSED`
- [ ] 실제 제출 시 LB·제출 ID·서버 실행시간 기록

고차원 signed stack, 2024 in-fold 점수, 중앙 예상값만으로 Goal을 완료하지 않는다.

---

## 8. 환경

| 항목 | 값 |
| --- | --- |
| OS / 셸 | Windows 11, PowerShell 5.1 |
| Python | `.venv\Scripts\python.exe` (3.11.9) |
| 고정 버전 | numpy 1.26.4, pandas 2.0.3, scipy 1.15.3, scikit-learn 1.8.0, joblib 1.5.3 |
| 현재 추가 | lightgbm 4.7.0, catboost 1.2.8 (`pip check`·project-wrapper smoke 완료, 2026-08-20) |
| 평가 서버 | Ubuntu 22.04.5, Python 3.11.15, 6 vCPU, 28GB RAM, NVIDIA L4 |

**`.venv`에 패키지를 추가할 때는 반드시 제약을 걸고, 직후에 재현성을 확인한다.**

```powershell
& .\.venv\Scripts\python.exe -m pip install --constraint requirements-baseline.txt "lightgbm==4.7.0"
& .\.venv\Scripts\python.exe -m pip check
& .\.venv\Scripts\python.exe submission\verify_submission.py submission\archive\S4\S4.zip
```

마지막 명령이 실패하면 즉시 롤백한다. numpy/scipy/scikit-learn 버전이 바뀌면
S1~S8의 재현성이 깨지고, 그것은 Phase 3 코드 검증에서 문제가 된다.

`submission/template/requirements.txt`에는 **평가 서버에서 필요한 패키지만** 정확한
버전으로 넣는다. joblib 피클은 버전이 맞아야 로드되므로 로컬과 서버 버전을 일치시킨다.

---

## 9. 막혔을 때

- **단계가 실패한다** → `experiments/logs/<stage>.log`를 읽는다. 고친 뒤 `--run`으로 재개.
- **lightgbm/catboost가 없다** → §8의 제약 설치. CatBoost가 안 되면 `--skip c3 c3b`.
- **메모리가 부족하다** → `--max-history-rows`로 먼저 스모크 테스트한 뒤 전량 실행.
- **결과가 이전과 다르다** → `experiments/results/archive/`의 보존본과 비교한다.
- **판단이 필요하다** → 임의로 결정하지 말고 계획서 §12.2의 판정 기준을 따르고,
  그래도 모호하면 멈추고 사람에게 묻는다.
