# 통합 실험 계획서 V5 — 실제 전이 가능한 1,190점

> 시작: 2026-08-21 KST  
> 실제 LB 챔피언: `1090.9100565103` (`V3_sparse_m3_1103`)  
> 실패 앵커: `V4_compact_supported_1193`, 개발 예상 `1193.0915` → 사용자 보고 실제 약 `1005`  
> 목표: **대회 허용 범위 안에서 보수적 예상 LB 하한 또는 실제 LB가 1,190을 초과할 때까지 개선**

V5는 V4의 로컬 점수를 더 높이는 계획이 아니다. V4는 2024 한 개발 fold에서 18개 signed
coefficient를 맞춰 `+89.39점`을 얻었지만 실제 LB에서는 V3보다 약 `85.91점` 낮았다.
따라서 기존 `2024 로컬 + 140.1476` 환산식과 고차원 in-fold stacking을 폐기하고,
**교차 시즌 전이와 실제 제출 앵커**를 중심으로 평가 계약을 다시 만든다.

정확한 V4 점수·제출 ID·서버 시간은 사용자가 제공하는 즉시
[`submission/records/leaderboard_v4_user_report.json`](submission/records/leaderboard_v4_user_report.json)에
기록한다. 그 전에는 `1005.0`을 반올림된 감사값으로만 사용한다.

---

## 1. V4 실패에서 고정한 사실

[`experiments/results/v4_failure_audit.json`](experiments/results/v4_failure_audit.json)의 재현 결과다.

| 항목 | 값 |
| --- | ---: |
| 2024 V3 로컬 | `963.5501` |
| 2024 V4 로컬 | `1052.9440` |
| 로컬 delta | `+89.3938` |
| 실제 V3 LB | `1090.9101` |
| 실제 V4 LB | 약 `1005` |
| 실제 delta | 약 `-85.9101` |
| delta 방향 오류 | 약 `-175.3039` |
| 예전 예상식 오차 | 약 `-188.0915` |

추가 감사 결과:

- 18개 arm 중 2022와 2024에서 단변량 최적 방향의 부호가 같은 것은 **5개뿐**이다.
- 2022에서 맞춘 같은 18개 계수는 2024에 옮기면 V3 대비 `-66.39점`이다.
- 2022+2023 pooled 계수는 2024에서 `-319.48점`이다.
- 2023 계수는 2024에서 `-628.19점`이다.
- student 단독의 2024 이득은 `+40.66점`, 18-arm correction의 추가 이득은 약
  `+48.73점`이지만 둘 다 독립 확인이 아니었다.

결론은 패키지 오류가 아니라 **선택·검증 오류**다. sample parity, SHA, 불변성, 실행시간이
완벽해도 모델 성능의 외부 전이를 보장하지 않는다.

---

## 2. 절대 경계

다음은 Goal 권한으로도 허용하지 않는다.

1. `open/` 밖의 외부 데이터, 외부 API, 원격 추론.
2. 다른 평가 행, test 전체 평균·빈도·순서·배치 통계 사용.
3. 평가 행 사이의 `groupby`, `rolling`, `expanding`, 온라인 fit.
4. 미래 시즌 또는 검증 target으로 encoder·prior·iteration을 만드는 행위.
5. 리더보드 점수로 평가 target 평균이나 개별 라벨 통계를 역추정하는 행위.
6. `verify_submission.py` 완화·우회, 자동 DACON 제출.
7. 기존 결과·ZIP·OOF 예측 덮어쓰기.

허용되는 입력은 각 평가 행 자체의 값과 공식 train에서 미리 동결한 모델·사전뿐이다.

---

## 3. V5 검증 계약

> **V5.1 정정(2026-08-21):** 첫 V5 후보들을 감사하는 과정에서
> `v3_sparse_m3_frozen`의 2022·2023 예측에도 2024에서 고른 M3 가중치와 affine 보정이
> 역적용됐음을 확인했다. 이 파일은 제출된 V3의 재현물로는 유효하지만 과거 개발 비교의
> one-year-ahead 기준점으로는 유효하지 않다. 이후 판정은
> [`v5_validation_contract_v2.json`](experiments/params/v5_validation_contract_v2.json)과
> [`v5_anchor_honesty_audit.json`](experiments/results/v5_anchor_honesty_audit.json)을 따른다.

### 3.1 역할이 다른 시간 fold

- 개발·학습: 2020~2023의 strictly-forward OOF. R과 F를 분리한다.
- `F`: 2023 라벨 체제 단절 전후를 같은 분포로 취급하지 않는다. 2023 이전 F는 2025 F
  성능 개선 근거로 쓰지 않는다.
- 잠금 확인: 한 V5 실험 recipe를 JSON으로 먼저 저장한 뒤 2024를 실행한다.
- 2024 결과를 보고 recipe를 바꾸면 새 ID로 다시 잠그며, 이전 2024 결과는 개발 사용으로
  강등한다. 같은 결과를 확인 fold라고 부르지 않는다.
- 실제 LB만 완전히 독립적인 최종 확인이다.

각 목표 연도 `Y`의 앙상블 기준점은 `Y-1`의 라벨과 OOF 성분만으로 비음수 M3 가중치와
보정을 정한 뒤 `Y`에 그대로 적용한다. 주 기준은 `R`에서 맞춘 다음 두 앵커다.

- `v5_honest_m3_r_identity`: 직전 시즌 가중치, affine 보정 없음
- `v5_honest_m3_r_grid`: 직전 시즌에서만 원래 V3 사전 격자 중 보정값 선택

구조 피처 ablation은 이 앵커보다 먼저 **동일한 부모 recipe**와 비교한다. 모델·seed·학습
행·나머지 피처가 같은 부모보다 2022·2023에서 모두 좋아지지 않으면 2024를 열 후보가 아니다.

### 3.2 후보 복잡도 제한

예상점수만으로 Goal 완료 자격을 얻는 후보는 다음 중 하나여야 한다.

- 단일 재현 recipe, 또는
- 사전 고정된 비음수 가중치 최대 3개. 가중치 합은 1이며 같은 fold의 target으로 다시
  맞추지 않는다.

signed residual correction, 4개 이상 자유 가중치, 조건부 gate를 같은 확인 fold에 맞춘
후보는 **실제 LB가 나오기 전에는 Goal 완료 자격이 없다.** 연구 결과로는 보존한다.

### 3.3 V3 실제 점수 기반 보수적 예상식

절대 로컬 점수에 offset을 더하지 않는다. V3와 동일한 outcome/CatBoost 계열의 저복잡도
후보만 다음 상대식의 적용 대상이다.

```text
G_dev      = 2022·2023의 동일 부모 및 두 honest anchor 대비 full-score delta 최솟값
G_confirm  = 잠금 뒤 2024의 동일 부모 및 두 honest anchor 대비 point delta 최솟값
G_ci       = 같은 2024 비교들의 pitcher-cluster 95% score-delta 하한 최솟값
G_robust   = min(G_dev, G_confirm, G_ci)

보수적 예상 LB 하한 = 1090.9100565103 + 0.75 × max(0, G_robust)
```

`0.75`는 M2→M3의 실제 delta 전이비보다 보수적인 haircut이며 V4 같은 고복잡도 후보에는
적용하지 않는다. `1190`을 넘으려면 이 엄격한 공통 하한이 **`132.1199점`보다 커야 한다.**
V5.1은 기존 기준을 완화한 것이 아니라, 첫 후보 감사에서 발견된 미래 선택값 혼입을 제거해
더 엄격하게 만든 정정이다.

Goal 완료에는 다음 중 하나가 필요하다.

- 위 **보수적 예상 LB 하한 `> 1190`**, 모든 계약·패키지 게이트 통과, 또는
- 실제 DACON LB `> 1190`.

중앙 예상, in-fold 최고점, 기존 `+140.1476`, 단일 bootstrap point만으로는 완료하지 않는다.

---

## 4. 획기적 개선 축

### H1 — 행 단위 동적 실력 추정

CatBoost가 모든 상태를 알아서 조합하게 두는 대신, 공식 누적 카운터에서 복원한 현재 시즌
성공 수·실패 형태·표본 수를 **계층 Beta-Binomial/state-space posterior**로 만든다.

- current-season 성공률, 통산률, 직전 1/3/5경기율의 서로 다른 시간 척도
- 투수별 장기 prior + 리그/경기유형 prior + 표본수 기반 신뢰도
- posterior mean뿐 아니라 posterior variance와 변화량
- `n`이 커질수록 현재 시즌 정보의 비중이 단조 증가하는 reliability gate
- 모든 값은 해당 행과 train 동결 사전만 사용

핵심 검증은 단순 CatBoost feature 추가가 아니라, posterior 단독·V3 blend·잔차 모델을
각각 잠가 비교하는 것이다.

### H2 — Brier 직접 최적화와 형태 제약

compact behavioral state 위에서 squared-error/Brier를 직접 최적화한다.

- HGB/LightGBM regression 또는 강한 ridge/GAM
- current-season 성공률·표본수에 monotonic/reliability 제약
- 확률 범위는 학습 후 임의 calibration이 아니라 모델 계약 안에서 보장
- 수백 개 raw feature 대신 사전 정의한 작은 상태 벡터 사용

### H3 — 안정 신호와 최신 신호의 분리

한 모델에 장기·단기 정보를 모두 넣지 않고 application/stable expert와 behavioral/recent
expert를 별도 학습한다. 결합은 이전 시즌 OOF로만 정한 1개 reliability 함수 또는 최대
3개 비음수 고정 가중치만 허용한다.

### H4 — outcome 구조의 계층화

V3의 `reverse_any`를 버리지 않고 성공 확률과 실패 형태를 계층적으로 분리한다.

```text
P(success)
P(reverse | failure), P(middle | failure), P(wide | failure)
```

다중분류 한 번보다 클래스 간 calibration이 안정적인지 다중 forward fold에서 확인한다.

### H5 — 실제 전이 가능한 단순 앙상블

모델 수가 아니라 fold별 방향 일치성을 최적화한다.

- 후보별 2020~2024 score delta, 부호, residual correlation을 먼저 계산
- 단일 fold gain이 큰 후보보다 모든 비단절 fold에서 같은 방향인 후보 우선
- family당 최대 1개, 전체 최대 3개, 비음수 가중치
- 가중치는 확인 fold를 열기 전에 저장

---

## 5. 실행 순서

1. **P0 감사:** V4 실패 감사와 문서·대장 갱신.
2. **P1 기반 재생성:** 직전 시즌만 쓰는 honest M3 anchor와 compact state의 strict OOF 확보.
3. **P2 H1:** 계층 posterior·reliability 모델 구현 및 잠금 backtest.
4. **P3 H2/H4:** Brier 직접 모델과 계층 outcome을 독립 비교.
5. **P4 H3/H5:** 안정/최신 expert와 저복잡도 앙상블 사전 고정.
6. **P5 확인:** 잠긴 후보만 2024 확인 및 보수적 예상 하한 계산.
7. **P6 패키징:** 전체 재학습, sample parity, 불변성, 245,789행, SHA 게이트.
8. **P7 수동 제출:** 사용자가 업로드하고 실제 LB·ID·시간을 기록.

실험이 실패하면 결과를 보존하고 다음 구조적 가설로 이동한다. 파이프라인이 끝났거나 로컬
중앙값이 높다는 이유로 Goal을 완료하지 않는다.

---

## 6. 현재 상태

- [x] V4 실패를 Goal 완료 오류로 인정
- [x] 기존 고정 offset 폐기
- [x] V4 cross-fold coefficient 감사 스크립트·결과 생성
- [x] V5 완료 게이트 사전 고정
- [x] 2024 선택값이 역적용된 과거 V3 anchor 오류 감사
- [x] 직전 시즌만 쓰는 honest M3 anchor 2022~2024 생성
- [x] H1 계층 posterior 1차 잠금 실험(기각)
- [x] 최근 분모·조건부 이력 구조 실험(정정된 부모 게이트에서 기각)
- [ ] H2/H4 차기 구조 실험
- [ ] 저복잡도 최종 후보 확인
- [ ] 보수적 예상 하한 또는 실제 LB `> 1190`
- [ ] 최종 ZIP 전수 게이트

Goal은 마지막 두 성능·패키지 조건을 모두 만족하기 전까지 active 상태로 유지한다.
