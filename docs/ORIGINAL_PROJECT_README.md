# LG Aimers 9기 해커톤 실험 저장소

DACON의 **투구 제구 성공 확률 예측 AI 온라인 해커톤** 참가를 위한 실험 저장소다.

- **[에이전트 작업 지시서](AGENTS.md)**: 자율 실행 에이전트용. 실행 명령, 금지 사항, 완료 조건 ← **자동 실행 시작점**
- **[통합 실험 계획서 V5 — 실제 전이 가능한 1,190점](EXPERIMENT_PLAN_V5.md)**: V4 예상 1193 → 실제 약 1005 실패 감사, 다중 시간축·저복잡도·실제 앵커 기반 새 완료 계약 ← **현재 활성 계획**
- **[V5 honest-anchor 감사](experiments/results/v5_anchor_honesty_audit.json)**: 2024에서 고른 V3 가중치·보정의 과거 fold 역적용을 발견해 폐기하고, 직전 시즌만 쓰는 2022~2024 기준점 생성
- **[통합 실험 계획서 v3/v4](EXPERIMENT_PLAN_V3.md)**: outcome·Trackman·딥러닝·supported ensemble과 V4 패키징 이력
- **[통합 실험 계획서 v2 — 용량 전환](EXPERIMENT_PLAN_V2.md)**: 블렌드 재조정 → LightGBM/CatBoost → platoon split → 앙상블까지의 연속 파이프라인, 검증 프로토콜 v2 ← **사람이 읽는 시작점**

```powershell
$env:PYTHONUTF8 = '1'
& .\.venv\Scripts\python.exe experiments\pipeline.py --status   # 진행 상황
& .\.venv\Scripts\python.exe experiments\pipeline.py --run      # 끝까지 실행 (중단 시 재개 가능)
```

끝나면 [`submission/SUBMIT_QUEUE.md`](submission/SUBMIT_QUEUE.md)에 게이트를 통과한 후보가 제출 순서대로 나온다. **업로드는 사람이 한다.**

- [통합 실험 계획서 v1](EXPERIMENT_PLAN.md): S1~S8까지의 일자별 계획, 제출 예산 설계, 제출 게이트, 리스크 레지스터 (이력으로 보존)
- [현재 로컬 개발 환경](LOCAL_ENVIRONMENT.md): Windows 11·PowerShell 실행 명령, 설치 버전, 현재 PC 실측 시간, 평가 서버와의 차이
- [대회 상세 정리](COMPETITION.md): 문제 정의, 일정, 평가식, 규칙, 제출 환경, 중요 공지, 체크리스트
- [데이터 상세 설명](open/data_description.md): 파일 구조와 전체 컬럼 정의
- [상세 EDA 보고서](eda/EDA_REPORT.md): **제1부** Target drift·결측·분포·정합성 / **제2부** 행 순서와 경기 복원, Trackman 정확 연결, 시즌 내 누적 복원, 성능 상한
- [기본 베이스라인 보고서](experiments/BASELINE_REPORT.md): 2022~2024 rolling validation, 모델 비교, 세그먼트 오류, 재현 방법
- [시간 순 Calibration·Ensemble 보고서](experiments/CALIBRATION_ENSEMBLE_REPORT.md): frozen/refit 보정, Linear-HGB 가중치, 안정성 분석
- [제출 이력과 재현 명령](submission/SUBMISSION_LOG.md): S1~S11·V3 ZIP 해시, 불변성·24.6만 행 게이트, DACON 점수
- [후보 산출물 보관대장](submission/CANDIDATE_REGISTRY.md): S1~S11·V3 ZIP·모델·설정·검증 보고서·SHA-256
- [실험 산출물 보관대장](experiments/EXPERIMENT_REGISTRY.md): rolling 결과와 기각 실험까지 ID별 보존
- [관련 연구·프로젝트와 실험 로드맵](research/RELATED_WORK_AND_EXPERIMENT_ROADMAP.md): 제구·의도·Trackman·선수 효과·drift·calibration 문헌 / 실력 추정·Stuff+·행위자 분해 방법론을 이 데이터로 실측 / 우선순위 실험 명세
- [유사 EDA 구조 데이터의 방법론 조사](research/ANALOGOUS_DATA_METHODS_AND_EXPERIMENTS.md): CTR·추천·의료/신용·부정거래·확률 예보 / 패널 계량경제·심리측정·재식별·누출 분류학·quantification·측정 체제 단절에서 가져온 방법과 실험 설계
- [공식 대회 페이지](https://dacon.io/competitions/official/236743/overview/description)

## 현재 상태

| 항목 | 값 |
| --- | --- |
| 현재 Goal | V5 활성 — 보수적 예상 LB 하한 또는 실제 LB `>1190` 전까지 계속 |
| V4 결과 | `V4_compact_supported_1193`: 로컬 `1052.9440`, 이전 예상 `1193.0915` → 사용자 보고 실제 **약 `1005`**; exact/ID 기록 대기 |
| 다음 제출 | 없음 — V5 계약을 통과한 새 후보만 큐에 등록 |
| 최우선 제출 결과 | `V3_sparse_m3_1103`: 예상 `1103.6977` → 실제 **`1090.9100565103`** |
| 백업 제출 결과 | `V3_sparse_m2_1100`: 예상 `1100.6527` → 실제 `1088.5196116458` |
| LB 챔피언 | `V3_sparse_m3_1103`, **`1090.9100565103`** |
| S4 rolling 3-fold 평균 Brier | `0.24756026` (3/3 fold 개선) |
| S8 M3 rolling 3-fold 평균 Brier | `0.24755460` (S4 대비 평균 `-0.000005657`, 2/3 fold 개선) |
| 로컬 환경 | Windows 11, Ryzen 5 5600, RAM 31.9GB, Python 3.11.9 `.venv` 검증 완료 |
| 후보 보존 | S1~S11, V3 M2·M3, V4 1193 ZIP·빌드 기록·manifest·검증 보고서·SHA-256 보존 |
| 현재 환경 | LightGBM 4.7.0·CatBoost 1.2.8, wrapper smoke·`pip check` 완료 |
| V3 검증 | M3 245,789행 6.29초/0.85GiB, M2 5.19초/0.83GiB, 불변성 오차 0 |
| V4 검증 | 22모델, 245,789행 37.60초/1.50GiB, 불변성 0, sample parity `2.22e-16`, SHA 결정성 통과 |
| 2026-08-20 신규 제출 | M3 `1090.9100565103`, M2 `1088.5196116458`, V ensemble `906.8719072396`, V base·S11 `879.8414124135` |

V3에서는 공식 과거 데이터로 복원한 보조 outcome, 동결 Trackman 물리 프로파일,
타자·역사 그룹률을 CatBoost로 학습하고 잔차가 다른 세 성분을 희소 앙상블했다.
M3는 2024·2022 paired gate와 보수 S8 앵커 기준을 모두 통과했다. 실제 제출에서도
M3가 `1090.9100565103`으로 새 챔피언이 되었고 M2가 `1088.5196116458`로 뒤를 이었다.
예상값보다 각각 `12.7876`, `12.1331`점 낮아 실제 1,100에는 도달하지 못했다.

이후 목표 예상점수를 1,190으로 높여 MLP, DeepFM, TabTransformer, TabM, RealMLP와
neural residual을 포함한 시간순 실험을 수행했다. 딥러닝 단독 최고는 TabM 계열
`916.9032`, deep OOF stack은 `1008.4905`였다. TabM 포함 meta stack도 목표를 넘었지만,
최종적으로 더 높은 로컬 `1052.9440`과 간단한 CPU 재현성을 보인 22개 CatBoost 조합을
`submission/dist/V4_compact_supported_1193.zip`으로 동결했다. 그러나 사용자 보고 실제
LB는 약 `1005`로 V3보다 크게 낮았다. 18개 signed 계수 중 2022와 2024에서 방향이 같은
것은 5개뿐이었고, 이전 fold에서 맞춘 계수는 2024로 전이되지 않았다. 따라서 기존 고정
offset과 V4 완료 판정을 폐기하고 V5 검증 계약으로 전환했다.

## 스크립트

현재 PC의 프로젝트 루트에서 PowerShell로 실행한다. 가상환경을 활성화하지 않고 실행 파일을 직접 지정하는 것이 기준이다.

```powershell
$env:PYTHONUTF8 = '1'

& .\.venv\Scripts\python.exe eda\run_eda.py
& .\.venv\Scripts\python.exe eda\run_structural_eda.py
& .\.venv\Scripts\python.exe experiments\run_baselines.py `
  --validation-season 2024 --save-models
& .\.venv\Scripts\python.exe `
  experiments\run_temporal_calibration_ensemble.py `
  --validation-seasons 2022 2023 2024
& .\.venv\Scripts\python.exe submission\build_submission.py --candidate all
& .\.venv\Scripts\python.exe submission\verify_submission.py `
  submission\dist\S2_linear90_hgb10.zip
```

설치부터 다시 시작해야 하거나 현재 PC와 DACON Ubuntu 서버의 차이를 확인하려면 [`LOCAL_ENVIRONMENT.md`](LOCAL_ENVIRONMENT.md)를 먼저 본다.

대회 규칙과 일정은 변경될 수 있으므로 제출 전 공식 페이지와 중요 공지를 다시 확인한다.
