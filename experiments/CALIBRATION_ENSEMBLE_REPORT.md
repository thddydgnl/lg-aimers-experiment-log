# 시간 순 Calibration·Linear-HGB Ensemble 실험 보고서

> 실행일: 2026-08-17  
> 선행 실험: [기본 베이스라인](BASELINE_REPORT.md)  
> 현재 PC 재검증: Windows PowerShell 환경은 [`../LOCAL_ENVIRONMENT.md`](../LOCAL_ENVIRONMENT.md)를 따른다.  
> 결론: 현 단계의 기본 후보는 **보정하지 않은 refit Linear 90% + HGB 10%**다.

## 1. 결론

### 1.1 권장 전략

`Linear 90% + HGB 10%` 고정 ensemble이 성능과 안정성의 균형이 가장 좋았다.

- 3-fold 평균 Brier: `0.24793696`
- 2022·2023·2024 모두 양의 skill: `3/3`
- 최악 fold인 2023 환산 점수: `105.4`
- Linear 단독 대비 평균 Brier: `0.24806782 → 0.24793696`
- Linear 단독 대비 평균 환산 점수: `637.3 → 689.8`

`80:20`은 평균 Brier `0.24784630`으로 더 좋지만, 2023 점수가 `17.6`에 불과해 작은 추가 drift에도 0점이 될 수 있다. `50:50`은 평균 Brier `0.24781548`로 가장 낮지만 2023에서 음의 skill이므로 안정적 기본선으로 채택하지 않는다.

### 1.2 Calibration 결론

직전 시즌 하나로 학습한 calibration은 최종 refit 모델에 일관된 이득을 주지 못했다.

- refit Linear raw: 평균 Brier `0.24806782`
- refit Linear + logit intercept: `0.24814147` — 악화
- refit HGB raw: `0.24856794`
- refit HGB + logit intercept: `0.24855133` — 평균은 미세 개선하지만 2023 음의 skill 유지
- frozen Linear는 logit/affine 보정으로 개선됐지만 최신 시즌을 기본 모델 학습에서 제외하는 손실 때문에 refit raw보다 나빴다.

따라서 현재 제출 후보에는 별도의 calibration을 적용하지 않는다. 보정은 여러 시간 fold에서 만든 OOF 예측으로 다시 설계해야 한다.

## 2. 누수 방지 설계

외부 검증 시즌을 `Y`라 할 때 다음 순서로 실행했다.

```text
season < Y-1 ── base model 학습 ──> Y-1 예측
                                           │
                                           ├─ calibrator 학습
                                           └─ Linear-HGB 가중치 선택

frozen: season < Y-1 모델을 그대로 Y에 적용
refit : season < Y로 base model 재학습 후, 이전 calibrator/가중치를 Y에 적용
```

| 외부 검증 | 기본 모델 history | Calibration | Refit 학습 |
| ---: | --- | ---: | --- |
| 2022 | 2019~2020 | 2021 | 2019~2021 |
| 2023 | 2019~2021 | 2022 | 2019~2022 |
| 2024 | 2019~2022 | 2023 | 2019~2023 |

외부 검증 시즌 Target이나 외부 검증 행 전체 통계는 calibrator, ensemble 가중치 또는 모델 입력에 사용하지 않았다. 각 행은 독립적으로 예측할 수 있다.

두 프로토콜의 의미는 다음과 같다.

- `frozen`: calibrator가 본 모델과 실제 적용 모델의 확률 스케일이 같다. 대신 가장 최근 시즌을 기본 모델 학습에 쓰지 못한다.
- `refit-transfer`: 가장 최근 시즌까지 학습할 수 있지만 재학습으로 확률 분포가 바뀐 모델에 과거 calibrator를 옮겨야 한다.

## 3. 실험 범위

각 프로토콜에서 다음을 비교했다.

### 3.1 Calibration

- `logit_intercept`: logit에 절편만 더해 직전 시즌 평균을 맞춘다.
- `affine_brier`: `clip(intercept + slope × p)`를 Brier 최소제곱으로 학습한다.
- `platt`: `logit(p)`에 logistic slope와 intercept를 학습한다.
- `isotonic`: 단조 비모수 보정이다.

### 3.2 Ensemble

- 고정 `50:50`, `80:20`, `90:10`
- 직전 시즌 Brier를 최소화하는 Linear 가중치
- 가중 ensemble 후 calibration
- 두 모델을 각각 calibration한 후 가중 ensemble

fold당 44개, 총 132개 전략을 평가했다. 주 지표는 Brier와 clipping 전 `raw_skill`이며, AUC·log loss·R/F·cold-start 세그먼트도 함께 저장했다.

## 4. Ensemble 결과

괄호 안은 검증 시즌별 대회식 환산 점수다.

| 전략 | 2022 Brier | 2023 Brier | 2024 Brier | 평균 Brier | 최악 raw skill | 양의 fold |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| HGB raw | **0.243611** (2,228.5) | 0.253749 (0.0) | **0.248344** (585.6) | 0.248568 | -0.014995 | 2/3 |
| 50% Linear + 50% HGB | 0.244097 (2,033.6) | 0.250954 (0.0) | 0.248396 (564.9) | **0.247815** | -0.003816 | 2/3 |
| 80% Linear + 20% HGB | 0.244925 (1,701.1) | 0.249956 (17.6) | 0.248658 (460.0) | 0.247846 | 0.000176 | 3/3 |
| **90% Linear + 10% HGB** | 0.245291 (1,554.3) | 0.249737 (105.4) | 0.248783 (409.7) | 0.247937 | **0.001054** | **3/3** |
| Linear raw | 0.245701 (1,389.6) | **0.249574** (170.5) | 0.248928 (351.7) | 0.248068 | **0.001705** | **3/3** |
| 직전 시즌 최적 가중치 | 0.243852 (2,131.9) | 0.253521 (0.0) | 0.248928 (351.7) | 0.248767 | -0.014084 | 2/3 |

평균만 최적화하면 50:50 또는 약 60:40이 유리하지만, 2023 체제 변화에서 음의 skill이 된다. 90:10은 HGB의 좋은 비선형 신호를 일부 가져오면서 Linear의 수축 안정성을 대부분 보존한다.

이 `90:10`은 동일 rolling fold에서 선택한 개발용 설정이다. 완전히 독립적인 최종 성능 추정치로 해석해서는 안 된다.

## 5. 직전 시즌 최적 가중치가 실패한 이유

| 외부 검증 | 가중치 선택 시즌 | 선택된 Linear 가중치 | 다음 시즌 결과 |
| ---: | ---: | ---: | --- |
| 2022 | 2021 | 36.25% | HGB 중심 구성이 유리 |
| 2023 | 2022 | **3.30%** | 거의 HGB가 되어 F 체제 변화에서 붕괴 |
| 2024 | 2023 | **100.00%** | HGB 회복 이득을 전혀 사용하지 못함 |

가중치가 `3.3% → 100%`로 한 해 만에 경계값 사이를 이동했다. 직전 시즌 하나의 Brier 최적점은 모델들의 오차가 비슷할 때 매우 불안정하며, 다음 시즌 regime을 예측하는 값도 아니다.

앞으로 동적 가중치를 쓴다면 최소한 다음 제약이 필요하다.

- Linear 하한 `80~90%`
- 여러 rolling fold의 평균과 최악 skill을 함께 최적화
- `game_type=F`처럼 drift가 큰 세그먼트에 별도 안정성 제약

## 6. Calibration 결과

### 6.1 주요 전략 비교

| 전략 | 평균 Brier | 최악 raw skill | 양의 fold | 해석 |
| --- | ---: | ---: | ---: | --- |
| refit Linear raw | **0.248068** | **0.001705** | 3/3 | calibration 기준선 |
| refit Linear + logit intercept | 0.248141 | 0.000897 | 3/3 | 2024 개선, 2022·2023 악화 |
| frozen Linear raw | 0.248443 | **0.002139** | 3/3 | 안정적이지만 최근 학습 데이터 손실 |
| frozen Linear + logit intercept | 0.248195 | 0.001539 | 3/3 | frozen 내부에서는 확실히 개선 |
| frozen Linear + affine Brier | 0.248185 | 0.000217 | 3/3 | 평균은 좋지만 최악 여유 작음 |
| refit HGB raw | 0.248568 | -0.014995 | 2/3 | 2023 붕괴 |
| refit HGB + logit intercept | 0.248551 | -0.014511 | 2/3 | 붕괴를 해결하지 못함 |

Isotonic은 calibration 시즌 안에서는 강하지만 다음 regime에서 불안정했다. frozen Linear isotonic의 평균 Brier는 `0.248166`이지만 최악 raw skill이 `-0.000584`여서 채택하지 않았다.

### 6.2 Refit-transfer의 확률 스케일 불일치

logit 절편 보정 전후의 외부 검증 예측 평균이다.

| 모델·검증 | raw 평균 | 보정 평균 | 실제 성공률 | 결과 |
| --- | ---: | ---: | ---: | --- |
| Linear 2022 | 52.25% | 54.20% | 52.89% | 과보정 |
| Linear 2023 | 51.26% | 51.90% | 50.00% | 과보정 |
| Linear 2024 | 49.75% | 48.49% | 48.61% | 개선 |
| HGB 2022 | 53.24% | 53.72% | 52.89% | 악화 |
| HGB 2023 | 52.18% | 51.83% | 50.00% | 불충분 |
| HGB 2024 | 49.68% | 47.47% | 48.61% | 반대 방향 과보정 |

예를 들어 HGB calibrator는 2023에서 과대예측한 이전 HGB에 맞춰 강한 하향 절편을 배웠다. 하지만 2023까지 포함해 refit한 HGB는 이미 2024 평균을 `49.68%`까지 낮췄고, 같은 절편을 적용하자 `47.47%`로 이중 보정됐다.

순위 정보가 바뀌지 않는 logit 절편은 AUC를 개선할 수 없으며, 실제로 refit Linear의 평균 AUC는 raw와 보정 모두 `0.545224`다. 이번 문제는 단순 calibration뿐 아니라 regime-aware 학습이 필요하다.

## 7. `game_type=F` 잔여 위험

| 검증 시즌 | Linear F Brier | HGB F Brier | 90:10 F Brier | F 실제 성공률 | 90:10 예측 평균 |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 2022 | 0.220751 | **0.206899** | 0.218180 | 70.87% | 60.64% |
| 2023 | **0.255652** | 0.294783 | 0.257763 | 47.29% | 56.42% |
| 2024 | 0.251079 | **0.247844** | 0.250542 | 45.93% | 50.65% |

90:10은 전체 점수의 안정성을 높이지만 F의 regime shift 자체를 해결하지는 못했다. 세 시즌 모두 F 구간만 다시 계산한 skill은 0 이하이며, 2023에는 실제보다 `9.13%p`, 2024에는 `4.72%p` 높게 예측했다.

## 8. 현재 모델링 결정

다음 실험의 기준선을 아래처럼 고정한다.

1. **주 후보:** refit Linear 90% + refit HGB 10%, calibration 없음
2. **안전 기준:** refit Linear 100%, calibration 없음
3. **공격적 비교군:** refit HGB 100%와 50:50 ensemble
4. 직전 한 시즌 최적 가중치와 refit-transfer isotonic/Platt은 사용하지 않음

다음 개선 우선순위는 다음과 같다.

1. `season × game_type=F`와 최근 시즌 가중치를 넣은 regime-aware 기본 모델
2. 여러 시간 fold의 OOF 예측을 함께 사용한 calibration
3. Linear 가중치 하한을 둔 constrained/gated ensemble
4. R/F 별 모델 또는 calibration을 만들되 최악 fold 제약으로 검증

## 9. 재현 방법과 결과 파일

```powershell
$env:PYTHONUTF8 = '1'
& .\.venv\Scripts\python.exe `
  experiments\run_temporal_calibration_ensemble.py `
  --validation-seasons 2022 2023 2024
```

전체 실행은 현재 Windows PC에서 `249.4초` 걸렸다. rolling 132행과 aggregate 44행의 전략 키가 모두 일치했고, 저장 결과 대비 최대 차이는 환산 점수 `4.8e-5`로 수치 오차 수준이다. 챔피언 평균 Brier는 `0.24793696494`, 2024 환산 점수는 `409.7038`로 다시 확인됐다.

주요 파일:

- `run_temporal_calibration_ensemble.py` — 전체 실행 코드
- `results/calibration_ensemble_valid_2022.{json,csv}`
- `results/calibration_ensemble_valid_2023.{json,csv}`
- `results/calibration_ensemble_valid_2024.{json,csv}`
- `results/calibration_ensemble_rolling.csv` — 132개 fold-전략 행
- `results/calibration_ensemble_aggregate.csv` — 전략별 rolling 요약

JSON에는 calibrator 파라미터, isotonic knot, 직전 시즌 가중치, 전체·세그먼트 지표, 학습/예측 시간이 포함돼 있다.
