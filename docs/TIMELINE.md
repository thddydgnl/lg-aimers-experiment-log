# 실험 타임라인

원본 작업 폴더에는 Git 이력이 없으므로 이 타임라인은 결과 JSON 내부 기록, 파이프라인 상태, 제출 기록, 계획서, 파일 수정 시각을 순서대로 대조해 재구성했다. 같은 시각대의 세부 실행 순서는 일부 추정일 수 있으며, 상세 행 순서는 [`catalog/experiments.csv`](../catalog/experiments.csv)에 보존한다.

## 1. 기준선과 초기 개선 — 2026-08-17~18

- 공식 데이터 구조와 평가 지표를 분석하고 baseline Linear/HGB 계열을 구축했다.
- E14는 시즌 내 투수 성공 누적을 복원해 3개 rolling fold 모두 개선했다.
- E15는 최근 3시즌 R prior를 추가해 초기 S4 기준선을 만들었다.
- E10·E11 등 recency/EB 확장은 평균 또는 최악 fold가 악화되어 폐기했다.
- E16, E22R 계열은 단독 챔피언이 아니라 앙상블 다양성 성분으로 보존했다.

## 2. V2 재현 파이프라인과 부스터 탐색 — 2026-08-19~20

- 중단 후 재개 가능한 파이프라인과 결과 보존 규칙을 구축했다.
- LightGBM, Linear, HGB, CatBoost의 기본값·그리드·platoon·TargetEncoder 조합을 rolling 평가했다.
- `pitcher TargetEncoder + platoon`은 통계 게이트를 통과했지만, 다수 단독 모델과 용량 확대는 기대 성능에 못 미쳤다.
- family-diverse ensemble은 실제 LB `906.8719072396`을 기록했다.
- S11/V_base는 실제 LB `879.8414124135`로 기준 제출 역할을 했다.

## 3. V3 outcome·TrackMan·희소 앙상블 — 2026-08-20

- 공식 누적 카운터 차이로 보조 outcome을 복원하고 CatBoost outcome family를 확장했다.
- rich TrackMan, outcome component, history group이 서로 다른 잔차를 제공했다.
- XGBoost, raw ID, 과도한 기간창·감쇠·구성요소 탐색 등 부정 결과도 보존했다.
- Sparse M2는 실제 LB `1088.5196116458`, Sparse M3는 `1090.9100565103`을 기록했다.
- V3 Sparse M3가 현재 실제 리더보드 챔피언이다.

## 4. V4 deep/OOF/meta stack과 실패 감사 — 2026-08-21

- MLP, DeepFM, TabTransformer, TabM, RealMLP 및 다양한 residual/OOF stack을 실험했다.
- Supported Meta Stack과 Compact Supported 18은 2024 로컬에서 약 `1052~1053`, 고정 예상에서 `1192~1193`을 기록했다.
- 22개 모델 전체 재학습, runtime parity, 불변성 및 ZIP 검증은 통과했다.
- 그러나 V4 Compact의 실제 LB는 사용자 보고 약 `1005`로 V3보다 크게 낮았다.
- 사후 감사에서 18개 arm 중 2022↔2024 방향이 일치한 것은 5개뿐이었고, 기존 offset 및 single-fold signed stack을 폐기했다.

## 5. V5 honest validation program — 2026-08-21~22

- V3 과거 fold에 2024 선택 가중치와 affine 보정이 역적용된 문제를 확인했다.
- 직전 시즌만 사용해 가중치·보정을 고르는 one-year-ahead honest anchor와 다중 시간축 계약을 고정했다.
- conditional history, dynamic state, TrackMan distillation, pitch selector/MoE, temporal adaptation, neural/tabular transfer, workload, pairwise rank 등 다수 가설을 source→development→confirmation 순으로 검증했다.
- 많은 후보가 일부 연도에서 양의 신호를 보였지만 연도 전이, CI 하한, 세 anchor 최악값 또는 2024 confirmation에서 실패했다.
- V5의 중요한 성과는 새로운 제출 후보보다 과적합과 전이 실패를 조기에 차단하는 검증 체계를 만든 것이다.

## 6. 현재 상태 — 2026-08-24 정리 기준

- 실제 LB 챔피언: V3 Sparse M3 `1090.9100565103`
- V4: 패키지 검증 성공, 성능 실패
- V5: 목표 `>1190` 미달, 실험 기록과 부정 결과 보존
- 향후 후보는 V5 계약과 전수 ZIP 게이트를 모두 통과하기 전에는 제출 큐에 올리지 않는다.

