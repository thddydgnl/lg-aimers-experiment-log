# 결과 요약과 교훈

## 효과가 확인된 방향

- E14/E15 계열의 공식 누적 상태 복원과 최근 R prior는 초기 기준선을 안정적으로 개선했다.
- 투수 TargetEncoder와 platoon 조합은 단독 피처 실험 중 통계적으로 유효한 결과를 만들었다.
- V3의 outcome 복원, rich TrackMan, history group은 서로 다른 잔차를 제공했고 희소 앙상블에서 실제 LB 개선으로 이어졌다.
- 전체 재학습, runtime parity, 행 독립성, SHA와 대규모 모사 검증 체계는 V3/V4 패키징에서 정상 작동했다.

## 실패했거나 제한적이었던 방향

- 단순 LightGBM 튜닝, 모델 용량 확대, 일부 CatBoost/platoon 조합은 기준선 대비 충분한 개선을 만들지 못했다.
- deep model은 일부 로컬 신호가 있었지만 단독 후보 또는 안정적 시간 전이 성분이 되지 못했다.
- V4 signed stack은 2024 개발 fold에 강하게 맞았지만 cross-fold 계수 방향이 불안정했고 실제 LB에서 실패했다.
- 현재 투구의 숨은 구종·물리 정보를 예측하거나 distill하는 V5 계열은 진단 oracle과 합법 배포 모델 사이의 큰 격차를 줄이지 못했다.
- V5의 많은 후보는 특정 source 연도에서는 개선됐지만 다른 연도, CI 하한, honest anchor 또는 2024 confirmation에서 역전됐다.

## 가장 중요한 교훈

1. 로컬 점수가 높아도 실제 LB 앵커와 시간 전이 검증이 없으면 제출 근거가 아니다.
2. 가중치·보정값은 목표 연도 결과를 본 뒤 과거 fold에 역적용하면 안 된다.
3. 단일 개발 fold의 signed stack보다 직전 시즌에서 잠근 저복잡도·비음수 조합이 안전하다.
4. 실패한 실험도 다음 탐색 범위를 줄이는 자산이므로 삭제하지 않는다.
5. 패키지 정확성과 모델 성능은 별개의 게이트다. V4는 패키지는 통과했지만 성능 목표에는 실패했다.

## 해석 시 주의

- 2023 fold는 리그·라벨 체제 단절 영향이 있어 주 선택 지표로 사용하지 않는다.
- V5 source 결과와 oracle 결과는 제출 가능한 성능을 의미하지 않는다.
- `예상 LB`는 실제 LB가 아니며 V4 이후 보수적으로 취급한다.
- 세부 수치와 판정은 [`experiments/EXPERIMENT_REGISTRY.md`](../experiments/EXPERIMENT_REGISTRY.md) 및 연결된 JSON/CSV를 우선한다.

