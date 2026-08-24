# 유사 EDA 구조 데이터의 방법론 조사와 실험 설계

> 작성일: 2026-08-17 (§1~§19), 2026-08-17 2차 (§20~§29)  
> 목적: 야구 도메인이 아니라 **현재 데이터의 통계적 구조와 평가 방식이 비슷한 문제**에서 검증된 방법을 찾아, 이 대회에서 실행할 수 있는 실험으로 번역한다.  
> 선행 문서: [EDA 보고서](../eda/EDA_REPORT.md), [기본 베이스라인](../experiments/BASELINE_REPORT.md), [Calibration·Ensemble 실험](../experiments/CALIBRATION_ENSEMBLE_REPORT.md), [야구 도메인 관련 연구](RELATED_WORK_AND_EXPERIMENT_ROADMAP.md)
> 현재 실행 환경·자원 판단: [`LOCAL_ENVIRONMENT.md`](../LOCAL_ENVIRONMENT.md), [`EXPERIMENT_PLAN.md` §13](../EXPERIMENT_PLAN.md#13-필요-자원과-실행-환경)

> **개정 안내.** §1~§19는 [EDA 제2부](../eda/EDA_REPORT.md#제2부-데이터를-생성한-구조)의 구조 복원 이전에 작성됐다. 그 결과 §10(불확실한 TrackMan 연결)과 `A30`/`A31`의 전제가 무너졌고, 동시에 §1의 다섯 부류로 설명되지 않던 새 문제 유형 일곱 가지가 드러났다. **§20이 무엇이 바뀌었는지 정리하고, §21~§27이 새로 추가된 유사 분야다.**

## 1. 결론부터

이 데이터와 가장 가까운 비야구 문제는 하나가 아니라 다음 다섯 부류의 교집합이다.

1. **CTR·광고 클릭 확률 예측**  
   대규모 이진 확률 예측, 시간 순 train/test, 익명 고카디널리티 범주, count/rate 피처, 작은 신호라는 점이 가장 가깝다.
2. **추천 시스템의 user-item 상호작용**  
   투수-타자라는 두 종류 ID, 희소한 pair, 신규 ID와 cold-start가 같은 구조다.
3. **의료·신용 위험 예측**  
   미래 시점으로 갈수록 base rate와 calibration이 변하고, 단순 재보정·부분 수정·전체 재학습 중 무엇을 할지 선택해야 한다.
4. **부정거래·스트리밍 분류**  
   장기적으로 안정적인 패턴과 최근 체제 패턴을 별도 전문가 모델로 학습해 결합하는 접근을 제공한다.
5. **확률 예보와 행정 데이터 record linkage**  
   Brier Score를 reliability와 resolution으로 분리하는 평가법, 불확실한 보조 데이터 연결을 확률로 전파하는 방법을 제공한다.

이 문헌들을 현재 EDA에 적용하면 우선순위는 다음과 같다.

1. 시간 fold마다 **calibration intercept·slope, Brier reliability·resolution, R/F·cold-start 성능**을 먼저 계측한다.
2. `asof_* rate + n`에 **시간 감쇠 계층적 empirical Bayes 수축**을 적용한다.
3. 선수 ID 효과를 그대로 외우지 말고 **문맥 모델 + 표본 수로 gating한 ID residual**로 분해한다.
4. Linear와 HGB 사이의 빈 공간을 채우는 **GA2M/EBM**, **GBDT leaf → Logistic**, **Factorization Machine**을 시험한다.
5. 전체 기간 모델과 최근 기간 모델을 **stable/recent expert**로 분리하고, 여러 시간 fold의 평균과 최악 성능을 함께 보며 결합한다.
6. 기본 모델을 고정한 뒤에만 여러 시간 OOF를 이용한 **identity/intercept/affine-logit/beta calibration**을 비교한다.
7. TrackMan은 저신뢰 hard join 대신 **확률적 ID 매핑 + soft aggregate + 불확실성 피처**로 사용한다.

가장 중요한 금지선도 명확하다.

- 일반적인 domain adaptation 논문이 쓰는 **평가 test 전체 분포 기반 importance weighting·calibration**은 이 대회에서 사용하지 않는다.
- validation 시즌 안에서 앞 행을 뒤 행의 history로 쓰지 않는다. 평가 행 독립성을 재현하려면 validation도 모든 행을 train cutoff에서 동결된 자산으로 예측해야 한다.
- 현재 투구의 TrackMan 행을 찾거나, 낮은 신뢰도의 ID 대응을 하나로 확정하지 않는다.

## 2. 조사 범위와 증거 등급

### 2.1 범위

이번 조사는 야구 자체의 제구·투구 연구를 반복하지 않고 다음 키워드군을 대상으로 했다.

- chronological CTR prediction, high-cardinality categorical, sparse interaction
- cold-start recommendation, hybrid metadata embedding, factorization
- out-of-time risk validation, calibration drift, dynamic model updating
- fraud concept drift, sliding window, stable/recent ensemble
- Brier decomposition, proper scoring rule, calibrated ensemble
- probabilistic record linkage, linkage uncertainty propagation
- point-in-time feature join, leakage prevention
- robust tabular model, GA2M/EBM, CatBoost, GBDT vs tabular deep learning

“관련 연구 전부”를 문자 그대로 보장할 수는 없다. 대신 **현재 EDA의 각 실패 모드에 직접 대응하는 대표 원 논문·공식 데이터·공식 구현**을 우선 확인했고, 단순 블로그나 leaderboard 회고는 근거에서 제외했다.

### 2.2 증거 등급

| 등급 | 의미 | 이 문서에서의 사용 |
| --- | --- | --- |
| A | 원 논문, 학회 proceedings, 공식 데이터·공식 문서 | 방법 채택의 주 근거 |
| B | 동료심사 응용 연구·체계적 문헌고찰 | 시간 drift·업데이트 전략 보강 |
| C | 공개 구현·프로젝트 | 구현 가능성 참고 |
| 제외 | 개인 블로그, 출처 불명 성능표, 임의 leaderboard 해설 | 채택 근거로 사용하지 않음 |

논문에서 좋았던 방법도 이 대회에서 자동으로 좋다고 간주하지 않는다. 최종 증거는 언제나 **동일한 누출 방지 조건의 rolling Brier와 최악 fold 안정성**이다.

## 3. 현재 데이터를 도메인 독립적으로 다시 정의하기

### 3.1 데이터 서명

| 특성 | 현재 데이터에서 관측된 값 | 일반적인 문제 유형 |
| --- | --- | --- |
| 목표 | 0/1 사건의 확률, 평균 `52.38%` | CTR, 위험도, 확률 예보 |
| 평가 | Brier Skill Score | 확률 calibration + resolution |
| 크기 | train `1,475,092`행 | 대규모 tabular/광고 로그 |
| 시간 | 2019~2024 train → 2025 test | out-of-time validation |
| target drift | `56.47% → 48.61%` | prior/base-rate drift |
| subgroup regime | F `70.87%`(2022) → `47.29%`(2023) | group-specific concept drift |
| 범주 | pitcher 792, batter 830 등 | high-cardinality entity ID |
| 상호작용 | pitcher × batter × hand × count | user-item-context 추천 |
| 누적 통계 | `asof_* rate`, `n`, 최근 1/3/5 경기 | CTR history, risk history |
| 저표본 과신 | 과거 rate 극단값이 현재 결과에서 평균으로 회귀 | empirical-Bayes shrinkage 문제 |
| 결측 | 신규 선수·history 부족과 결부 | informative missingness/cold-start |
| 보조 로그 | TrackMan 179만 행, 직접 1:1 join 불가 | event log + entity resolution |
| 추론 제약 | test 각 행 독립 예측 | stateless scoring, no transduction |
| 신호 크기 | Linear가 안정적이고 HGB가 일부 fold에서 붕괴 | low signal, model variance 위험 |

### 3.2 가장 가까운 유사 문제 지도

| 유사 분야 | 매우 닮은 점 | 다른 점 | 가져올 핵심 |
| --- | --- | --- | --- |
| Criteo/Avazu CTR | 시간 순 이진 확률, count+범주, 고카디널리티, 작은 개선 | CTR은 더 희소하고 feature space가 더 큼 | sparse Linear, tree→LR, freshness, calibration |
| 추천 시스템 | 두 종류 entity와 희소 interaction, cold-start | 랭킹보다 단일 사건 확률을 예측 | FM, metadata fallback, ID residual gating |
| 의료 위험 | 시간에 따른 calibration drift, 위험 확률 | 설명 가능성·의사결정 비용이 더 중요 | 업데이트 단계, intercept/slope 진단 |
| 신용 위험 | out-of-time 검증, 확률 안정성 | class imbalance와 규제가 다름 | OOT 검증, 보수적 업데이트 |
| 부정거래 | concept drift, 오래된/최근 지식 결합 | 보통 극심한 불균형·지연 라벨 | stable/recent experts, sliding window |
| 기상 확률 예보 | Brier와 skill, calibration | 범주 ID가 거의 없음 | reliability-resolution 분해 |
| record linkage | 불완전한 ID 대응과 보조 데이터 결합 | 목표가 매칭 자체인 경우가 많음 | hard match 회피, posterior/불확실성 전파 |
| feature store | 시간 로그를 as-of로 집계 | 모델 방법론이 아니라 데이터 시스템 | point-in-time join, TTL/cutoff 계약 |

## 4. CTR·광고 클릭 예측에서 가져올 것

### 4.1 왜 가장 가까운가

[Criteo Display Advertising Challenge](https://www.kaggle.com/c/criteo-display-ad-challenge/data)는 7일의 광고 노출을 시간 순으로 제공하고 다음 날 사건을 test로 사용했다. 각 행은 클릭 여부, 13개 정수형 피처 대부분은 count, 26개는 해시된 범주형이며 결측도 있다. [Criteo AI Lab](https://ailab.criteo.com/ressources/)은 이 데이터를 공식 연구 데이터로 제공한다.

현재 문제와 공통되는 핵심은 다음이다.

- 미래 기간의 사건 확률을 맞힌다.
- 익명 ID와 문맥 범주가 많다.
- 과거 빈도·누적량이 중요한 동시에 freshness가 중요하다.
- 정확도보다 확률 손실의 작은 차이가 중요하다.
- 복잡한 모델이 평균은 좋아도 시간 drift에서 calibration이 쉽게 무너질 수 있다.

### 4.2 FTRL-Proximal: sparse Linear는 약한 기준선이 아니다

Google의 [Ad Click Prediction: a View from the Trenches](https://research.google.com/pubs/archive/41159.pdf)는 대규모 CTR에서 정규화 Logistic과 FTRL-Proximal을 사용하며, sparse 계수, per-coordinate 학습률, online update, calibration과 feature 관리까지 함께 다룬다.

현재 데이터에 대한 의미:

- Linear가 HGB보다 안정적인 것은 이상한 결과가 아니다. sparse/weak-signal 확률 문제에서 강한 정규화 Linear는 매우 강한 기준선이다.
- pitcher/batter/team/hand/count의 제한된 cross를 one-hot 또는 hashing한 Logistic/FTRL을 별도 후보로 둘 수 있다.
- 다만 현재 ID 수는 수백 수준이라 feature hashing은 필수가 아니다. 해시 충돌과 해석 손실을 감수할 이유가 있는지는 실제 메모리로 판단한다.
- 온라인 갱신 자체는 2025 label이 없고 test 행 독립 규칙이 있으므로 사용할 수 없다. 대신 과거 시즌에서 progressive validation을 재현하는 데만 쓴다.

### 4.3 GBDT leaf → Logistic: 비선형 구간을 선형 확률 모델에 전달

Meta의 [Practical Lessons from Predicting Clicks on Ads at Facebook](https://ai.meta.com/research/publications/practical-lessons-from-predicting-clicks-on-ads-at-facebook/)은 decision tree가 만든 leaf를 sparse 범주 피처로 바꾼 뒤 Logistic에 넣는 방식이 tree와 Logistic 각각보다 3% 이상 좋았다고 보고했다. 또한 user/ad의 역사 피처가 가장 중요했고 freshness와 학습률 조정은 그 다음이었다.

이 결과가 현재 `Linear 90% + HGB 10%`보다 한 단계 더 나아가는 이유:

- 단순 확률 평균은 두 모델의 마지막 출력만 섞는다.
- leaf→Logistic은 HGB가 발견한 `season × game_type`, rate×n, count×hand 같은 비선형 partition을 Linear가 다시 강하게 수축해 조합한다.
- Logistic이 최종 확률 scale을 학습하므로 tree 확률의 과신을 줄일 가능성이 있다.

그러나 tree와 Logistic을 같은 표본으로 학습하면 leaf 선택 과적합을 그대로 전달할 수 있다. 첫 실험은 보수적으로 다음처럼 분리한다.

```text
과거 block A ── HGB 학습 ──> leaf encoder 고정
최근 block B ── leaf + 원본 Linear 피처 ──> Logistic 학습
미래 season C ── 동결된 HGB와 Logistic으로 평가
```

최종 2025 후보라면 예를 들어 HGB는 2019~2023, Logistic은 2024로 학습하는 방식이다. 2024를 HGB가 쓰지 못하는 손실과 leaf 결합 이득을 반드시 비교한다.

### 4.4 Wide & Deep: memorization과 generalization의 분리

[Wide & Deep](https://research.google/pubs/wide-deep-learning-for-recommender-systems/)은 cross-product가 있는 Linear의 memorization과 embedding 기반 deep model의 generalization을 함께 학습한다. 저자들은 sparse하고 rank가 높은 상호작용에서는 embedding이 지나치게 일반화할 수도 있다고 지적한다.

현재 문제에 바로 번역하면:

- wide: hand, count, season/regime, `game_type`, 안정적인 명시적 interaction
- entity residual: pitcher/batter ID 또는 저차원 factor
- continuous: EB로 수축한 `asof_*`, 최근-장기 차이, `log1p(n)`
- fallback: 신규 ID에서는 wide/context 부분만 남긴다.

대형 deep network부터 할 필요는 없다. 같은 구조를 **regularized Linear + 작은 FM/MLP residual**로 먼저 구현하는 편이 비용과 과적합 위험이 낮다.

### 4.5 CTR 문헌이 제안하는 실험 우선순위

| 방법 | 현재 적용 | 우선순위 | 주요 위험 |
| --- | --- | ---: | --- |
| 정규화 sparse Logistic | 명시적 소수 cross 추가 | 높음 | cross 폭발 |
| recency/freshness | 시즌 감쇠·최근 창 | 매우 높음 | 최근 한 시즌 과적합 |
| tree leaf→Logistic | HGB partition을 sparse 피처화 | 높음 | 두 단계 temporal split 손실 |
| Wide + entity residual | 전역 안정성 + ID 효과 | 매우 높음 | ID cold-start |
| FTRL/feature hashing | cross가 매우 커질 때 | 낮음~중간 | 불필요한 복잡도·충돌 |

## 5. 추천 시스템에서 가져올 것

### 5.1 투수-타자는 user-item-context 구조다

추천 문제의 전형적인 입력은 `user × item × context`다. 현재 데이터는 이를 다음처럼 읽을 수 있다.

```text
user 역할     = pitcher_id
item 역할     = batter_id
context       = hand, count, inning, game_type, season, score/base state
binary reward = control_success
```

관측된 pitcher-batter pair는 희소하고, 미래에는 처음 보는 pair 또는 선수 자체가 나온다. raw pair target encoding이 위험한 이유와 factorization이 필요한 이유가 정확히 같다.

### 5.2 Factorization Machine: 희소 pair를 저차원으로 공유

[Factorization Machines](https://doi.org/10.1109/ICDM.2010.127)은 희소한 범주 상호작용을 각 pair별 독립 계수가 아니라 저차원 latent vector의 내적으로 표현한다.

이진 확률 모델은 다음 형태로 만들 수 있다.

```text
logit(p) = w0 + Σ wi xi + Σ(i<j) <vi, vj> xi xj
```

현재 데이터에서 직접 기대할 수 있는 효과:

- 한 번도 보지 못한 pitcher-batter **pair**라도 각각의 factor를 통해 상호작용을 추정한다.
- pitcher×hand, batter×pitcher-hand, entity×count, entity×game_type 신호를 공유한다.
- raw pair target encoding보다 파라미터 수가 작고 강하게 수축된다.

한계도 분명하다.

- 완전히 처음 보는 pitcher/batter ID에는 factor가 없다.
- 시간 drift를 자동 해결하지 않는다.
- factor dimension이 커지면 작은 신호를 쉽게 외운다.

따라서 dimension `4/8/16`, 강한 L2, recent weight, unknown fallback과 함께 시험하고 cold-start Brier를 별도로 본다.

[Field-aware Factorization Machines](https://www.csie.ntu.edu.tw/~cjlin/papers/ffm.pdf)는 같은 feature라도 상대 feature의 field에 따라 다른 latent vector를 사용한다. 예를 들어 pitcher ID가 batter field와 상호작용할 때와 count field와 상호작용할 때 서로 다른 표현을 쓴다. Criteo·Avazu 같은 CTR 문제에서 FM보다 강한 결과를 보였지만 파라미터와 연산량이 늘고 epoch에 민감해 과적합하기 쉽다. 따라서 기본 FM이 유효한 신호를 보인 뒤 `pitcher/batter/context` 세 수준의 작은 FFM만 후속 ablation으로 시험한다.

### 5.3 metadata embedding: 신규 ID는 속성으로 후퇴한다

[LightFM의 Metadata Embeddings](https://arxiv.org/abs/1507.08439)은 user/item 표현을 해당 개체의 metadata feature embedding 합으로 만든다. interaction이 적거나 cold-start인 경우 pure collaborative model보다 좋고, 데이터가 충분할 때도 경쟁력을 유지했다고 보고한다. [OFF-Set](https://arxiv.org/abs/1308.1792)도 지속적인 cold-start 환경에서 알려진 속성을 latent space로 매핑하고 이진 reward를 온라인 학습한다.

이를 현재 데이터에 그대로 복제하기보다 다음과 같이 단순화하는 것이 안전하다.

```text
player representation
  = shared(hand, team, asof profile, experience bucket)
  + reliability_gate(n) × player_ID_residual
```

권장 gate:

```text
g(n; k) = n / (n + k)
```

- `n=0`이면 ID residual은 0이고 hand/team/context만 남는다.
- `n`이 커질수록 ID 고유 효과가 점진적으로 활성화된다.
- `k`는 20~1000 grid에서 inner temporal fold로 선택한다.
- pitcher와 batter의 `k`는 따로 둔다.

이는 LightFM 논문의 정확한 모델이 아니라 그 핵심 원리인 **ID 정보와 side information의 결합**을 현재 규칙에 맞게 단순화한 제안이다.

### 5.4 high-cardinality target statistics

[Micci-Barreca 2001](https://doi.org/10.1145/507533.507538)은 고카디널리티 범주를 전체 평균과 범주 평균의 empirical-Bayes식 혼합으로 수치화한다. [CatBoost](https://proceedings.neurips.cc/paper/2018/hash/14491b756b3a51daac41c24863285549-Abstract.html)는 target leakage로 생기는 prediction shift를 줄이기 위해 ordered target statistics와 ordered boosting을 제안한다.

현재 데이터에서는 두 가지 추가 제약이 필요하다.

1. CatBoost 내부 permutation은 **외부 시즌 분리**를 대신하지 않는다.
2. validation/test 행은 서로 history를 갱신하지 않는다. category statistics는 train cutoff에서 동결한다.

권장 계층은 다음과 같다.

```text
league/global
  └─ season trend or regime
      └─ game_type × hand
          └─ pitcher or batter
              └─ 제한된 entity × context
```

`pitcher × batter` raw target mean은 만들지 않는다. 필요한 pair 신호는 FM 또는 매우 강하게 정규화한 latent interaction으로 처리한다.

## 6. 의료·신용 위험 예측에서 가져올 것

### 6.1 업데이트 전에 어떤 drift인지 진단한다

[임상 예측 모델 평가·업데이트 지침의 체계적 검토](https://pmc.ncbi.nlm.nih.gov/articles/PMC9742671/)는 calibration-in-the-large, calibration slope를 평가하고, 차이가 작을 때는 recalibration, predictor effect가 달라질 때는 부분/전체 revision을 권한다.

모델 예측 `p`에 대해 각 validation season에서 다음 회귀를 적합한다.

```text
logit P(Y=1) = a + b × logit(p)
```

해석:

| 현상 | 관측 | 먼저 시험할 조치 |
| --- | --- | --- |
| 전체 확률이 일괄적으로 높거나 낮음 | `a != 0`, `b ≈ 1` | intercept-only 보정 |
| 확률이 너무 극단적/평평함 | `b < 1` / `b > 1` | affine-logit, shrinkage |
| 특정 변수·그룹에서만 residual 변화 | 계수·segment calibration 변화 | 부분 revision, interaction |
| ranking과 calibration 모두 붕괴 | Brier·AUC·group 성능 동시 악화 | recent refit, expert model |

이 표는 drift 원인을 완전히 식별하는 인과 검사가 아니다. 다만 모든 변수를 다시 학습하기 전에 **가장 작은 수정으로 해결되는 문제인지** 구분하는 실용적 진단이다.

### 6.2 보수적인 업데이트 사다리

임상 문헌의 공통적인 업데이트 순서는 다음과 같다.

1. 원 모델 유지
2. intercept만 수정
3. intercept + slope 수정
4. 일부 predictor effect 또는 residual 수정
5. 전체 재학습
6. 새 피처를 추가한 model extension

[비모수적 model drift 업데이트 연구](https://pmc.ncbi.nlm.nih.gov/articles/PMC6857513/)와 [동적 업데이트 전략 비교](https://pmc.ncbi.nlm.nih.gov/articles/PMC8647501/)도 업데이트 데이터가 적을 때 복잡한 refit이 항상 낫지 않으며, 빈번한 업데이트에서는 recalibration이 안정적일 수 있음을 보인다.

현재 실험에 대한 의미:

- 직전 한 시즌 calibration이 실패했다고 calibration 전체를 버릴 이유는 없다.
- 먼저 fold별 `a`, `b`가 같은 방향인지 본다.
- 일관된 intercept drift만 있으면 base model 전체를 바꾸기보다 강하게 수축한 intercept를 시험한다.
- 2023 F처럼 특정 그룹의 조건부 관계가 급변하면 global intercept로 해결하려 하지 않는다.
- calibrator가 본 frozen model과 최종 refit model의 score scale이 달라지는 문제를 별도 검증한다.

### 6.3 dynamic calibration과 adaptive window

[Detection of Calibration Drift](https://pmc.ncbi.nlm.nih.gov/articles/PMC8627243/)는 시간에 따라 calibration curve를 갱신하고 ADWIN 계열 adaptive sliding window로 drift를 탐지해 업데이트에 적합한 최근 구간을 찾는다. [Dynamic prediction model review](https://pmc.ncbi.nlm.nih.gov/articles/PMC6460710/)는 sliding window, decay factor, Bayesian/빈도주의 업데이트를 검토하면서 window와 갑작스러운 변화의 선택이 여전히 핵심 문제라고 정리한다.

2025 test label을 순차적으로 받을 수 없으므로 online 업데이트를 직접 쓸 수는 없다. 대신 과거를 다음처럼 pseudo-deployment로 사용한다.

```text
2019~2020 fit → 2021 월/블록별 calibration monitor
2019~2021 fit → 2022 monitor
2019~2022 fit → 2023 monitor
2019~2023 fit → 2024 monitor
```

이 분석으로 고정 2년/3년 window와 exponential half-life 후보를 줄일 수 있다. test를 보지 않고 train 내부에서 window를 정한다는 점이 중요하다.

### 6.4 out-of-time validation

신용평가 연구에서도 개발 표본과 미래 out-of-time 표본의 적합도가 달라지는 문제가 반복된다. [Development and Validation of Credit-Scoring Models](https://kiefer.economics.cornell.edu/WP9_2007.pdf)는 in-time과 out-of-time validation을 분리해 모델 형태의 시간 일반화를 검토한다.

현재 문제에서는 random CV가 아니라 다음 rolling OOT가 주 평가다.

```text
≤2021 → 2022
≤2022 → 2023
≤2023 → 2024
```

그리고 2023은 단순한 한 fold가 아니라 F 체제 급변을 포함하는 stress fold로 본다.

## 7. 부정거래·스트리밍 분류에서 가져올 것

### 7.1 오래된 지식과 최근 지식을 별도 모델로 둔다

[Credit Card Fraud Detection and Concept-Drift Adaptation](https://dalpozz.github.io/static/pdf/IJCNN2015_final.pdf)은 concept drift와 지연된 label 환경에서 서로 다른 정보원을 학습한 두 classifier를 결합하는 전략을 다룬다. 문제의 class imbalance는 현재와 다르지만, **서로 다른 시간 규모의 전문가를 분리**한다는 원리는 직접 적용할 수 있다.

현재 후보:

- stable expert: 2019~cutoff 전체, 강한 regularization Linear
- recent expert: 최근 2년 또는 감쇠 가중 HGB/Linear
- regime expert: F만 별도 학습하는 대신 global model의 F residual을 강하게 수축
- 결합: 고정 convex blend 또는 logit stack

```text
p = α p_stable + (1-α) p_recent
```

또는

```text
logit(p) = c0 + c1 logit(p_stable) + c2 logit(p_recent)
```

가중치를 직전 한 시즌에서 고르지 않는다. 이미 기존 실험에서 Linear 가중치가 `3.3% → 100%`로 튀었다. 여러 inner rolling fold의 평균 Brier와 최악 fold를 함께 최적화하고 stable expert의 하한을 둔다.

### 7.2 recent-only가 항상 정답은 아니다

- 최근 window는 새 regime에는 민감하지만 선수 표본과 드문 상황을 잃는다.
- 전체 history는 희소 그룹을 안정화하지만 사라진 regime을 오래 기억한다.
- 따라서 한 모델에 모든 역할을 강요하기보다 두 모델의 bias-variance를 분리하는 편이 자연스럽다.

현재 `Linear 90% + HGB 10%`도 이 철학의 초기 형태다. 다음 단계는 알고리즘 종류뿐 아니라 **시간 window가 다른 모델의 다양성**을 추가하는 것이다.

## 8. 확률 예보·Brier 연구에서 가져올 것

### 8.1 Brier는 calibration만 보는 지표가 아니다

[Murphy 1973](https://doi.org/10.1175/1520-0450(1973)012%3C0595:ANVPOT%3E2.0.CO;2)은 Brier Score를 다음처럼 분해한다.

```text
Brier = reliability - resolution + uncertainty
```

- reliability: 예측 확률과 실제 빈도의 불일치. 낮을수록 좋다.
- resolution: 서로 다른 사건 위험을 다른 확률로 분리하는 능력. 높을수록 좋다.
- uncertainty: 표본의 base rate가 정하는 항으로 동일 fold 내 모델 간에는 같다.

[Gneiting & Raftery 2007](https://doi.org/10.1198/016214506000001437)는 Brier 같은 strictly proper scoring rule이 정직한 확률 예측을 유도함을 정리한다.

현재 실험에서 필요한 질문은 “Brier가 좋아졌는가” 하나가 아니다.

- calibration을 좋아지게 했지만 예측을 0.5 근처로 지나치게 압축해 resolution을 잃지 않았는가?
- HGB가 resolution을 얻었지만 2023 F에서 reliability를 크게 잃지 않았는가?
- 앙상블이 어느 성분을 개선했는가?

### 8.2 필수 확률 진단

모든 후보에 다음을 저장한다.

- Brier, fold base rate, raw Brier skill
- calibration intercept와 slope
- 예측 평균, 표준편차, 1/5/50/95/99 분위수
- equal-frequency reliability table과 표본 수
- Murphy decomposition 또는 bias-corrected proxy
- 전체, R/F, hand matchup, count, `n` bucket, 신규 ID별 Brier
- 행별 squared-error 차이의 paired block bootstrap

ECE 하나만 사용하지 않는다. bin 수에 민감하고 작은 개선을 불안정하게 평가할 수 있다.

### 8.3 calibration 후보

[Beta calibration](https://proceedings.mlr.press/v54/kull17a.html)은 identity map을 포함하면서 일반 logistic calibration보다 유연하고, isotonic보다 파라미터 수가 작다. 후보 순서는 다음으로 제한한다.

1. identity
2. logit intercept only
3. affine-logit: `a + b logit(p)`
4. beta calibration
5. R/F 또는 cold-start intercept를 global 쪽으로 강하게 수축

[Multicalibration](https://proceedings.mlr.press/v80/hebert-johnson18a.html)은 겹치는 여러 subgroup에서 calibration을 맞추는 이론을 제공한다. 다만 현재는 subgroup 수가 많고 시간 fold가 3개뿐이므로 자유로운 multicalibration을 바로 적용하지 않는다. 먼저 소수의 사전 정의 그룹에 대한 regularized intercept만 시험한다.

### 8.4 covariate-shift calibration은 연구 가치와 대회 적합성이 다르다

[Unsupervised Calibration under Covariate Shift](https://arxiv.org/abs/2006.16405)와 [Calibrated Prediction with Covariate Shift](https://proceedings.mlr.press/v108/park20b/park20b.pdf)는 label 없는 target 입력 분포를 이용한 importance weighting을 제안한다. calibration이 작은 covariate shift에도 깨질 수 있다는 진단은 중요하다.

그러나 이 대회에서는 다음을 하지 않는다.

- test 전체의 feature 분포와 train을 비교해 weight 생성
- test score histogram으로 calibrator 수정
- test 안의 ID 빈도로 confidence 또는 fallback 변경

이는 평가 행 독립 예측 규칙과 직접 충돌한다. 이 논문들은 **왜 과거 fold에서 calibration transfer를 엄격히 시험해야 하는지**에 대한 근거로만 사용한다.

## 9. worst-group 강건화와 앙상블

### 9.1 평균 성능만 최적화하지 않는다

[Group DRO](https://arxiv.org/abs/1911.08731)는 평균 위험이 아니라 사전에 정의한 그룹 중 최악 위험을 줄이는 방법을 다루며, worst-group 일반화에는 강한 regularization과 early stopping이 중요하다고 보고한다.

현재 그룹 후보:

- validation season
- `season × game_type`
- cold pitcher / warm pitcher
- cold batter / warm batter
- pitcher/batter hand matchup
- `asof_*_n` 구간

그룹을 무한히 쪼개지 않는다. 최소 행 수를 정하고 작은 그룹 Brier는 전체 평균 쪽으로 수축하거나 confidence interval을 함께 표시한다.

### 9.2 첫 적용은 training DRO보다 robust model selection

이 문제에서 바로 group-DRO 학습을 적용하면 과거의 오래된 F regime을 과도하게 맞출 수 있다. 첫 단계는 다음 목적함수로 후보와 앙상블을 고르는 것이다.

```text
J = mean_fold_Brier
    + λ1 × max(0, worst_fold_Brier - mean_fold_Brier)
    + λ2 × max(0, worst_key_group_regret)
```

- `λ1`, `λ2`도 외부 fold가 아니라 inner fold에서만 선택한다.
- 2023/2024 F, cold-start가 악화된 평균 개선은 champion으로 채택하지 않는다.
- 이후에만 group loss exponentiated reweighting을 Linear/EBM에 시험한다.

### 9.3 제한된 ensemble selection

[Caruana et al. 2004](https://doi.org/10.1145/1015330.1015432)는 여러 모델 library에서 validation metric을 개선하는 모델을 순차적으로 추가하는 ensemble selection을 제안했다.

현재는 후보 수가 작고 winner's curse가 크므로 다음처럼 제한한다.

- 모델 family당 1~3개 seed/recipe만 library에 남긴다.
- 모든 prediction은 같은 rolling OOF protocol로 생성한다.
- weight는 `w ≥ 0`, `Σw=1` simplex로 제한한다.
- 평균 Brier뿐 아니라 worst-fold penalty를 포함한다.
- 유사 모델의 residual correlation이 높으면 하나만 남긴다.
- 90% Linear + 10% HGB를 고정 기준으로 둔다.

## 10. 불확실한 TrackMan 연결에서 가져올 것

> **§20.1에서 폐기됨.** 이 절 전체가 "연결이 불확실하다"는 전제 위에 서 있는데, 그 전제가 틀렸다. 경기 단위 지문 매칭으로 연결은 **결정적으로** 풀린다(투수 731명, 커버리지 99.79%). §10.1~§10.3과 `A30`/`A31`은 실행하지 않는다. **§10.4(point-in-time correctness)만 유효하며, §26.4에서 날짜 단위로 강화됐다.** 매칭 단위를 바꿔 확률 문제를 결정 문제로 만든 원리 자체는 §23에 유사 분야 문헌과 함께 정리했다.

### 10.1 hard join이 왜 위험한가

현재 공통 상황 키만으로 TrackMan 후보 수는 중앙값 5개, 90백분위 15개다. 후보 하나를 임의로 선택하면 TrackMan 피처는 유용한 측정값이 아니라 **매핑 오류가 섞인 measurement error**가 된다.

[Enamorado, Fifield & Imai의 probabilistic record linkage](https://imai.fas.harvard.edu/research/linkage.html)는 고유 식별자가 없고 값이 부정확한 대규모 행정 데이터에서 match 확률을 추정하고, post-merge 분석에 연결 불확실성을 반영한다. [Sadinle의 Bayesian bipartite matching](https://arxiv.org/abs/1601.06630)은 one-to-one 구조와 posterior uncertainty를 다루고, 불확실한 부분을 미해결 상태로 남기는 partial estimate를 제안한다.

핵심 교훈:

- 낮은 confidence를 억지로 한 ID에 할당하지 않는다.
- one-to-one 또는 roster/season consistency 같은 전역 제약을 활용한다.
- match probability와 entropy를 downstream model에 전달한다.
- hard match 결과만 쓴 분석과 uncertainty를 반영한 분석을 분리한다.

### 10.2 soft TrackMan aggregate

메인 투수 `i`의 TrackMan 후보 `j`에 대한 match posterior를 `q_ij`라고 하자. cutoff 이전 TrackMan 요약 `z_j,<t`가 있으면 다음처럼 기대 피처를 만든다.

```text
z_soft(i, t) = Σj q_ij × z_j,<t
```

추가 입력:

- `max_match_probability`
- match entropy
- 후보 수
- top1-top2 probability margin
- unmatched flag
- TrackMan history count

confidence가 낮으면 hand/league prior로 수축한다.

```text
z_final = c(q) × z_soft + (1-c(q)) × z_hand_prior
```

`c(q)`는 posterior max 또는 정규화 entropy로 만들되 inner validation에서 고정한다. 이는 probabilistic linkage의 uncertainty propagation 원리를 현재 예측 피처로 단순화한 제안이다.

### 10.3 비교해야 할 네 가지 arm

1. TrackMan 미사용
2. 사전 정의한 high-confidence hard match만 사용
3. soft posterior aggregate
4. 여러 plausible mapping으로 예측한 확률을 평균하는 multiple-imputation식 ensemble

전체 성능뿐 아니라 matched 선수 동일 행에서의 paired Brier를 비교한다. low-confidence 선수에서만 이득이 나오면 TrackMan 후보를 기각한다.

### 10.4 point-in-time correctness

[Feast 공식 quickstart](https://github.com/feast-dev/feast/blob/master/docs/getting-started/quickstart.md)는 각 event timestamp를 상한으로 두고 그 시점 이전의 최신 피처만 조인해 미래 누출을 막는 point-in-time join을 설명한다.

현재 TrackMan feature asset도 같은 계약을 가져야 한다.

```text
build_trackman_features(entity, cutoff)
  1. event_time < cutoff만 필터
  2. entity/pitch-group별 강건 통계
  3. low-n hierarchy shrinkage
  4. mapping uncertainty 결합
  5. immutable artifact로 저장
```

- validation 2024는 2024 TrackMan을 보지 않는 버전을 먼저 사용한다.
- 정확한 메인 행 날짜를 신뢰성 있게 복원한 경우에만 같은 시즌의 엄격한 과거 로그를 별도 실험한다.
- 2025 test에는 제공된 2019~2024 TrackMan의 동결 요약만 사용한다.
- 현재 투구의 개별 TrackMan 행을 찾지 않는다.

## 11. 일반 tabular 연구에서 가져올 것

### 11.1 GA2M/EBM: Linear의 안정성과 제한된 비선형성

[Accurate Intelligible Models with Pairwise Interactions](https://www.microsoft.com/en-us/research/wp-content/uploads/2017/06/kdd13.pdf)는 GAM에 선택된 2차 interaction을 추가한 GA2M이 여러 문제에서 full-complexity model과 비슷한 성능을 보일 수 있음을 제시했다.

형태:

```text
logit(p) = β0 + Σj fj(xj) + Σ(j,k in S) fjk(xj, xk)
```

현재 문제에 특히 맞는 이유:

- 단변량 rate 효과는 약하지만 비선형 수축 곡선이 예상된다.
- 중요한 interaction 후보가 EDA로 이미 좁혀져 있다.
- HGB 전체 복잡도보다 variance가 작고, Linear보다 표현력이 크다.
- 각 shape와 heatmap을 확인해 2023 F 같은 비정상 extrapolation을 찾을 수 있다.

첫 interaction 후보:

- `season × game_type`
- `asof_pitcher_success_rate × log1p(asof_pitcher_n)`
- `asof_batter_success_rate × log1p(asof_batter_n)`
- `pitcher_hand × batter_hand`
- `balls_before × strikes_before`
- count × recent-career delta
- game_type × recent-career delta

raw pitcher/batter ID는 GA2M에 바로 넣지 않고 cutoff-only EB 수치 또는 별도 residual로 처리한다.

### 11.2 CatBoost는 여전히 높은 우선순위다

CatBoost는 고카디널리티 범주와 interaction에 강하고 ordered statistics로 target leakage 문제를 줄인다. 다만 다음을 지킨다.

- outer season은 완전히 분리한다.
- validation target이 CatBoost statistic에 들어가지 않는지 검사한다.
- 여러 seed와 depth 5~8 정도의 보수적인 grid를 쓴다.
- model selection은 Logloss가 아니라 Brier와 worst-fold로 한다.
- unknown ID와 `n=0`을 별도 세그먼트로 평가한다.

### 11.3 대형 tabular deep model은 후순위다

[Revisiting Deep Learning Models for Tabular Data](https://papers.nips.cc/paper_files/paper/2021/hash/9d86d83f925f2149e9edb0ac3b49229c-Abstract.html)는 ResNet·FT-Transformer를 강한 deep baseline으로 제시하지만 GBDT와 비교해 보편적으로 우월한 해법은 없다고 결론낸다. [Why do tree-based models still outperform deep learning on tabular data?](https://proceedings.neurips.cc/paper_files/paper/2022/file/0378c7692da36807bdec87ab043cdadc-Paper-Datasets_and_Benchmarks.pdf)도 여러 중간 크기 tabular 문제에서 tree의 강한 inductive bias를 분석한다.

현재 순서:

1. EB + Linear/HGB
2. GA2M/EBM
3. CatBoost
4. FM 또는 작은 metadata embedding
5. 이들이 안정적으로 개선된 뒤 FT-Transformer/DeepFM

신호가 작고 calibration이 중요한 문제에서 deep model의 seed variance와 확률 과신을 먼저 감수할 이유가 없다.

### 11.4 참고할 공식 구현 프로젝트

| 프로젝트 | 무엇을 제공하는가 | 이 저장소에서의 역할 | 제출 시 주의 |
| --- | --- | --- | --- |
| [InterpretML](https://github.com/interpretml/interpret) | EBM/GA2M 구현과 shape·interaction 설명 | A12의 1차 구현 | 평가 서버 dependency·모델 크기 확인 |
| [CatBoost](https://github.com/catboost/catboost) | ordered categorical boosting | 기존 E12와 고카디널리티 비교 | native library와 추론 시간 확인 |
| [LightFM](https://github.com/lyst/lightfm) | metadata를 결합한 hybrid factorization | A11 구조 참고 | 기본 loss가 ranking 중심이므로 그대로 제출 모델로 쓰지 않음 |
| [LIBFFM](https://www.csie.ntu.edu.tw/~cjlin/libffm/) | L2 Logistic FFM, disk learning | A14 후속 FFM 비교 | C++ binary 동봉·플랫폼 호환성 점검 |
| [Vowpal Wabbit](https://github.com/VowpalWabbit/vowpal_wabbit) | hashing, online sparse learning, feature interaction | A40 FTRL/hash 후보 | online update는 사용하지 않고 frozen model만 추론 |
| [rtdl revisiting models](https://github.com/yandex-research/rtdl-revisiting-models) | FT-Transformer·ResNet 재현 코드 | A41 최종 deep baseline | PyTorch 모델 크기·CPU 추론 시간 확인 |
| [Feast](https://github.com/feast-dev/feast) | point-in-time feature retrieval 개념·구현 | TrackMan asset 계약 참고 | 제출 dependency로 넣을 필요는 없음 |

프로젝트를 사용하기 전 라이선스와 평가 서버 패키지 조건을 확인한다. 특히 LightFM은 추천 ranking용 구현이므로 논문의 cold-start 표현 원리를 가져오되, 현재 Brier 목적의 binary probability model은 별도로 학습하는 편이 맞다.

## 12. EDA 특성 → 방법 매핑

| EDA에서 보인 현상 | 유사 분야의 방법 | 현재 구현 | 우선순위 |
| --- | --- | --- | ---: |
| 연도별 target 하락 | clinical updating, CTR freshness | half-life/window, intercept/slope 진단 | P0 |
| 2023 F 급변 | group robustness, partial revision | `season×game_type`, F residual, worst-group 제약 | P0 |
| low-n rate 과신 | empirical Bayes | posterior mean/variance, hierarchy | P0 |
| 최근율과 누적율 중복 | dynamic risk, multi-timescale model | recent-career delta, stable/recent experts | P1 |
| pitcher/batter ID 효과 | high-cardinality CTR | ordered TE, CatBoost | P1 |
| 희소 pitcher-batter pair | recommender FM | rank 4/8/16 factor interaction | P1 |
| 신규 선수 | hybrid recommendation | metadata fallback + ID gate | P1 |
| Linear 안정·HGB 불안정 | GBDT→LR, GA2M | leaf Logistic, EBM | P1 |
| 모델별 오차 다양성 | ensemble selection | constrained OOF simplex | P2 |
| group별 calibration | clinical/multicalibration | regularized segment intercept | P2 |
| TrackMan 다대다 후보 | probabilistic linkage | soft aggregate + entropy | P2 |
| TrackMan 시간 로그 | feature store | point-in-time artifact | P2 |
| 매우 큰 sparse cross | CTR FTRL/hashing | 필요할 때만 hashed Logistic | P3 |
| deep 상호작용 가능성 | DeepFM/FT-Transformer | 마지막 ablation | P4 |

## 13. 새로 추가할 구체적 실험

아래 `Axx`는 유사 구조 문헌에서 추가된 실험이다. 기존 야구 문헌 로드맵의 `Exx`와 함께 사용한다.

### P0 — 평가와 drift 진단

#### A00 — temporal probability audit

목적: Brier 차이가 calibration, resolution, 특정 그룹 붕괴 중 무엇인지 분리한다.

외부 fold:

```text
≤2021 → 2022
≤2022 → 2023
≤2023 → 2024
```

각 모델·fold에 저장:

- Brier/raw skill/AUC/logloss
- calibration `a`, `b`
- reliability-resolution-uncertainty
- prediction distribution
- R/F, hand, count, pitcher/batter n bucket, unseen ID
- 2023 F와 2024 F의 예측 평균-실제 평균 차이
- paired season-block bootstrap

채택 조건: 후속 모든 실험이 동일한 JSON/CSV schema와 행 독립성 테스트를 통과해야 한다.

#### A01 — shift taxonomy report

각 fold에서 다음을 비교한다.

- label/base-rate shift: 실제 평균과 예측 평균
- calibration scale shift: `a`, `b`
- covariate shift proxy: train-only 연도 사이의 numeric/category distribution 변화
- conditional shift proxy: 고정 bin/segment 내 residual 변화
- model-family sensitivity: Linear, HGB, CatBoost/EBM의 동일 그룹 regret

주의: 2025 test 분포는 사용하지 않는다. 진단명은 인과적 확정이 아니라 실험 선택을 위한 proxy로 기록한다.

### P1 — 가장 유망한 모델·피처

#### A10 — decay-weighted hierarchical empirical Bayes

기본식:

```text
α0 = k μ
β0 = k (1-μ)
p_EB = (successes + α0) / (n + k)
Var(p|data) = αβ / ((α+β)^2(α+β+1))
```

비교:

- 원본 `asof_* rate`
- global prior EB
- `game_type × hand` prior EB
- season/regime-aware prior EB
- 최근 시즌에 exponential decay를 둔 custom history EB

grid:

- `k`: 20, 50, 100, 250, 500, 1000
- half-life: 0.5, 1, 2, 3년
- recent window: 1, 2, 3시즌

피처:

- posterior mean, variance, credible-width proxy
- 원본 rate와 EB rate 차이
- recent-career delta
- `log1p(n)`, missing/cold flag

누출 방지:

- train 행의 custom target statistics는 해당 행 이전 history만 사용한다.
- validation 전체는 train cutoff에서 동결하고 validation 앞 행으로 갱신하지 않는다.
- 최종 2025 artifact는 2019~2024 train label만 사용한다.

#### A11 — reliability-gated player residual

모델:

```text
logit(p) = f_context(x)
         + g(np; kp) × pitcher_residual[p]
         + g(nb; kb) × batter_residual[b]
         + optional_factor_interaction
```

```text
g(n; k) = n / (n+k)
```

비교 arm:

1. context only
2. raw one-hot ID + L2
3. gated ID residual
4. gated ID + hand/team/asof-profile shared component
5. 4번 + rank-4/8 pair factor

cold-start 검증:

- `n=0`
- validation에 처음 등장한 ID
- warm ID지만 처음 보는 pitcher-batter pair
- team 이동/season 변화가 있는 ID proxy

채택 조건: 전체 평균뿐 아니라 cold-start에서 context-only보다 나빠지지 않아야 한다.

#### A12 — GA2M/EBM

입력:

- 중복 제거 numeric/context
- A10 EB 피처
- low-cardinality 범주
- raw high-cardinality ID 제외 또는 A11 residual과 분리

비교:

- main effects only
- EDA 지정 interaction 6~12개
- inner fold에서 자동 선택한 interaction 10/20개

규제:

- 낮은 learning rate, bagging
- min samples/bin과 smoothing 강화
- season extrapolation shape 검사

성공 기준: Linear 대비 `≥0.00010` 평균 Brier 개선 또는 HGB와 낮은 residual correlation을 보이면서 3/3 양의 skill.

#### A13 — temporally split GBDT leaf + Logistic

첫 구현:

- HGB depth/leaf를 작게 제한해 과거 block A에서 학습
- leaf index를 one-hot sparse feature로 변환
- 최근 block B에서 원본 Linear 피처 + leaf로 Ridge Logistic 학습
- 다음 season C에서 평가

ablation:

- leaf only
- original Linear + leaf
- HGB probability + Linear
- 기존 90:10 probability blend

필수 진단:

- A→B→C 분할로 잃는 최근 HGB 학습량
- unseen leaf 없음 여부
- leaf cardinality와 sparse memory
- probability mean/slope transfer

#### A14 — shallow Factorization Machine

필드:

- pitcher, batter, pitcher team, batter team
- pitcher hand, batter hand
- count state, game_type, season/regime, inning band
- A10의 standardized numeric

grid:

- factor rank: 4, 8, 16
- L2: 강한 범위부터
- ID minimum frequency/UNK: 10, 50, 100
- full history vs recent decay

모델 선택은 Brier로 하며 3 seed 평균·최악값을 기록한다. 완전 신규 ID는 A11 side component 또는 0 residual로 fallback한다.

FM이 안정적으로 개선되면 같은 field 정의로 FFM rank `2/4/8`을 후속 비교한다. FFM은 field 수를 늘릴수록 모델 크기가 급증하므로 pitcher, batter, team, hand/count/regime 정도로만 묶고 early stopping을 inner temporal block에서 수행한다.

#### A15 — stable/recent expert

expert 후보:

- stable Linear: 전체 history 동일/완만한 decay
- recent Linear: 최근 2/3시즌
- recent HGB/EBM: 최근 regime의 nonlinear residual
- optional F residual: global score 위에 강하게 수축한 보정

결합 후보:

- fixed `90:10`, `80:20`, `70:30`
- stable weight 하한 0.7/0.8/0.9
- nonnegative logit stack

선택 목적함수는 평균 Brier + worst-fold penalty다. 단일 직전 시즌에서 가중치를 고르지 않는다.

### P2 — 강건화·앙상블·calibration

#### A20 — constrained ensemble library

library:

- 기존 Linear/HGB
- A12 EBM
- CatBoost
- A13 leaf Logistic
- A14 FM
- TrackMan 모델은 나중에 추가

절차:

1. 동일 rolling OOF 생성
2. residual correlation과 prediction spread 비교
3. family당 중복 후보 제거
4. simplex weight를 inner rolling fold에서 선택
5. worst-fold/group penalty 적용
6. 가장 최근 외부 fold에서 확인

#### A21 — robust model selection / light group reweighting

1단계는 training loss를 바꾸지 않고 robust selection만 한다. 2단계에서만 다음을 Linear/EBM에 시험한다.

- 그룹별 Brier가 큰 train group weight를 제한적으로 증가
- weight cap과 effective sample size 기록
- L2와 early stopping 강화
- group 정의는 season×game_type과 cold bucket까지만

오래된 F regime을 과대가중해 2024가 나빠지면 즉시 중단한다.

#### A22 — multi-domain OOF calibration

calibration train은 동일 recipe의 여러 historical OOF prediction을 사용한다.

비교:

- identity
- intercept-only
- affine-logit
- beta
- global + shrunk R/F intercept

규칙:

- calibrator hyperparameter는 외부 validation target을 보지 않는 inner season에서 선택
- identity 방향 regularization
- frozen과 refit-transfer를 모두 기록
- raw보다 fold 하나라도 음의 skill로 바뀌면 탈락
- calibration이 resolution을 지나치게 줄이지 않는지 분해

### P3 — TrackMan 불확실성

#### A30 — probabilistic pitcher linkage audit

> **폐기됨.** 연결이 결정적으로 풀렸다. §20.1 참고.

산출:

- 후보 pair evidence table
- one-to-one/season/hand/team consistency
- posterior 또는 normalized match score
- top1 probability, margin, entropy, candidate count
- high/medium/low confidence의 사전 고정 경계
- mapping perturbation sensitivity

매핑 label이 없으면 score를 진짜 확률이라고 과장하지 않고 `normalized confidence`로 부른다. 가능한 경우 high-confidence anchor만으로 calibration한다.

#### A31 — hard vs soft TrackMan aggregate

> **폐기됨 → `A54`(§28).** soft aggregate가 필요 없다. 날짜 단위 point-in-time 자산으로 대체한다.

비교:

- no TrackMan
- high-confidence hard only
- soft expected aggregate
- multiple plausible mapping prediction average

모든 통계는 validation cutoff 이전이고, 구종군별 mean/std/MAD/IQR/quantile/count와 uncertainty만 시작한다. soft 방식이 high-confidence hard보다 좋아도 low-confidence 행에서 불안정하면 champion에 넣지 않는다.

### P4 — 근거가 생긴 뒤 실행

#### A40 — FTRL/hashed crosses

명시적 cross 수가 메모리 병목이 될 때만 시행한다.

- signed feature hashing dimension `2^18~2^22`
- pitcher/batter ID, context, 제한된 pair cross
- FTRL 또는 SGD Logistic
- collision sensitivity를 2개 dimension에서 확인

현재 cardinality에서는 one-hot Ridge Logistic이 먼저다.

#### A41 — DeepFM/FT-Transformer

진입 조건:

- A14 FM이 pair factor의 유효성을 보임
- CatBoost/EBM과 다른 OOF residual을 보임
- 3 seed variance를 감당할 계산 예산이 있음
- 패키지·모델 크기·10분 추론 제한을 통과할 전망이 있음

그 전에는 실행 우선순위가 낮다.

[DeepFM](https://www.ijcai.org/Proceedings/2017/239)은 FM의 저차 interaction과 neural network의 고차 interaction을 같은 입력 embedding에서 함께 학습하는 CTR 모델이다. A14에서 저차 factor가 유효하고 A12/CatBoost가 놓치는 반복 가능한 residual이 있을 때만 이 구조를 시험한다.

## 14. 권장 실행 순서

### 1차: 가장 높은 정보가치

1. A00 temporal probability audit
2. A01 shift taxonomy
3. A10 decay hierarchical EB
4. A12 GA2M/EBM

이 단계에서 `Linear 90% + HGB 10%`의 약점이 calibration인지 nonlinear resolution 부족인지 구분할 수 있다.

### 2차: entity와 interaction

5. A11 gated ID residual
6. A13 tree leaf→Logistic
7. A14 shallow FM
8. CatBoost와 위 세 방법의 ablation

### 3차: 시간 규모와 확률 결합

9. A15 stable/recent experts
10. A20 constrained ensemble
11. A21 robust selection
12. base recipe 고정 후 A22 calibration

### 4차: 보조 로그

13. A30 linkage audit
14. A31 soft TrackMan aggregate
15. 유효성 확인 후 기존 문서의 TrackMan 분포·구종군 실험 확장

### 마지막

16. A40 FTRL/hash
17. A41 DeepFM/FT-Transformer

## 15. 공통 채택·중단 기준

현재 안정 기준선:

- Linear 90% + HGB 10%
- rolling 3-fold 평균 Brier `0.24793696`
- 3/3 fold 양의 raw skill

champion 교체 권장 조건:

- 평균 Brier 최소 `0.00010` 개선
- 3/3 fold 양의 raw skill
- 최악 fold 비악화
- 2023·2024 F 과대예측 비악화
- cold-start 비악화
- paired block bootstrap의 개선 방향이 대체로 일관
- test row-independence와 서버 제약 통과

앙상블 후보 보존 조건:

- 단독 평균 개선이 작아도 champion과 residual correlation이 낮음
- 특정 합법적 세그먼트에서 반복적으로 개선
- 확률 분포가 극단적이지 않고 calibration transfer가 안정적

즉시 중단:

- random CV에서만 개선
- validation 행을 순차 history로 사용
- test 전체 분포·빈도·score histogram 사용
- raw pitcher-batter target mean 사용
- 낮은 confidence TrackMan hard mapping에서만 개선
- calibration이 최신 fold 하나를 크게 악화
- seed 하나에서만 이득

## 16. 재현 가능한 실험 기록 schema

모든 `Axx` 실험 결과는 최소 다음 필드를 저장한다.

```text
experiment_id
git_commit / config_hash
feature_asset_version
train_cutoff / validation_season
history_window / decay_half_life
model_family / hyperparameters / seed
overall_brier / raw_skill / auc / logloss
calibration_intercept / calibration_slope
reliability / resolution / uncertainty
segment_metrics
prediction_summary
row_independence_checks
fit_seconds / predict_seconds / peak_memory
```

통계 피처 artifact에는 별도로 다음 metadata를 둔다.

```text
source_files
event_time_column or row_order_contract
strict_cutoff_rule
entity_keys
fallback_hierarchy
smoothing_parameters
mapping_confidence_version
```

## 17. 규칙상 사용하지 않을 방법

| 일반 연구 방법 | 왜 여기서는 사용하지 않는가 |
| --- | --- |
| test covariate importance weighting | test 전체 분포 사용 금지 |
| test score distribution calibration | 평가 행 상호 의존 발생 |
| transductive category frequency | 다른 test 행이 한 행 예측에 영향 |
| validation 안에서 online update | 실제 test에는 label도 순서 계약도 없음 |
| current-pitch TrackMan nearest join | 현재 투구 이후 정보·오매칭 위험 |
| 2025 외부 데이터 adaptation | 외부 데이터 금지 |
| raw pair target encoding | 극심한 희소성·누출·cold-start |
| 단일 시즌 자유 isotonic | regime 전환에서 과적합 가능성 큼 |
| unrestricted group calibration | 작은 그룹 noise와 fold 과적합 |
| 대형 deep model 단독 탐색 | 검증 budget 대비 variance가 큼 |

## 18. 참고 문헌·자료 색인

### CTR·광고

- Criteo, [Display Advertising Challenge data](https://www.kaggle.com/c/criteo-display-ad-challenge/data)
- Criteo AI Lab, [Research datasets](https://ailab.criteo.com/ressources/)
- McMahan et al., 2013, [Ad Click Prediction: a View from the Trenches](https://research.google.com/pubs/archive/41159.pdf)
- He et al., 2014, [Practical Lessons from Predicting Clicks on Ads at Facebook](https://ai.meta.com/research/publications/practical-lessons-from-predicting-clicks-on-ads-at-facebook/)
- Cheng et al., 2016, [Wide & Deep Learning for Recommender Systems](https://research.google/pubs/wide-deep-learning-for-recommender-systems/)

### 추천·고카디널리티

- Rendle, 2010, [Factorization Machines](https://doi.org/10.1109/ICDM.2010.127)
- Juan et al., 2016, [Field-aware Factorization Machines for CTR Prediction](https://www.csie.ntu.edu.tw/~cjlin/papers/ffm.pdf)
- Kula, 2015, [Metadata Embeddings for User and Item Cold-start Recommendations](https://arxiv.org/abs/1507.08439)
- Aharon et al., 2013, [OFF-Set](https://arxiv.org/abs/1308.1792)
- Micci-Barreca, 2001, [High-cardinality categorical preprocessing](https://doi.org/10.1145/507533.507538)
- Prokhorenkova et al., 2018, [CatBoost](https://proceedings.neurips.cc/paper/2018/hash/14491b756b3a51daac41c24863285549-Abstract.html)
- Guo et al., 2017, [DeepFM](https://www.ijcai.org/Proceedings/2017/239)

### 시간 drift·위험 모델

- Binuya et al., 2022, [Methodological guidance for evaluation and updating](https://pmc.ncbi.nlm.nih.gov/articles/PMC9742671/)
- Davis et al., 2020, [Detection of Calibration Drift](https://pmc.ncbi.nlm.nih.gov/articles/PMC8627243/)
- Davis et al., 2019, [A nonparametric updating method](https://pmc.ncbi.nlm.nih.gov/articles/PMC6857513/)
- Schnellinger et al., 2021, [Comparison of dynamic updating strategies](https://pmc.ncbi.nlm.nih.gov/articles/PMC8647501/)
- Jenkins et al., 2018, [Dynamic models to predict health outcomes](https://pmc.ncbi.nlm.nih.gov/articles/PMC6460710/)
- Dal Pozzolo et al., 2015, [Fraud Detection and Concept-Drift Adaptation](https://dalpozz.github.io/static/pdf/IJCNN2015_final.pdf)
- Kiefer & Larson, 2007, [Development and Validation of Credit-Scoring Models](https://kiefer.economics.cornell.edu/WP9_2007.pdf)

### Calibration·Brier·강건성

- Murphy, 1973, [A New Vector Partition of the Probability Score](https://doi.org/10.1175/1520-0450(1973)012%3C0595:ANVPOT%3E2.0.CO;2)
- Gneiting & Raftery, 2007, [Strictly Proper Scoring Rules](https://doi.org/10.1198/016214506000001437)
- Kull et al., 2017, [Beta Calibration](https://proceedings.mlr.press/v54/kull17a.html)
- Hebert-Johnson et al., 2018, [Multicalibration](https://proceedings.mlr.press/v80/hebert-johnson18a.html)
- Pampari & Ermon, 2020, [Unsupervised Calibration under Covariate Shift](https://arxiv.org/abs/2006.16405)
- Park et al., 2020, [Calibrated Prediction with Covariate Shift](https://proceedings.mlr.press/v108/park20b/park20b.pdf)
- Sagawa et al., 2020, [Group DRO](https://arxiv.org/abs/1911.08731)
- Caruana et al., 2004, [Ensemble Selection from Libraries of Models](https://doi.org/10.1145/1015330.1015432)

### Record linkage·point-in-time data

- Enamorado, Fifield & Imai, 2019, [Probabilistic merging of administrative records](https://imai.fas.harvard.edu/research/linkage.html)
- Sadinle, 2017, [Bayesian Estimation of Bipartite Matchings](https://arxiv.org/abs/1601.06630)
- Feast, [Point-in-time correct feature retrieval](https://github.com/feast-dev/feast/blob/master/docs/getting-started/quickstart.md)

### 일반 tabular

- Lou et al., 2013, [Accurate Intelligible Models with Pairwise Interactions](https://www.microsoft.com/en-us/research/wp-content/uploads/2017/06/kdd13.pdf)
- Gorishniy et al., 2021, [Revisiting Deep Learning Models for Tabular Data](https://papers.nips.cc/paper_files/paper/2021/hash/9d86d83f925f2149e9edb0ac3b49229c-Abstract.html)
- Grinsztajn et al., 2022, [Why do tree-based models still outperform deep learning on tabular data?](https://proceedings.neurips.cc/paper_files/paper/2022/file/0378c7692da36807bdec87ab043cdadc-Paper-Datasets_and_Benchmarks.pdf)

## 19. 최종 판단

유사 분야의 공통 결론은 “가장 복잡한 모델”이 아니라 다음 네 가지다.

1. 미래를 흉내 내는 시간 검증이 모델보다 먼저다.
2. 낮은 표본의 history와 ID 효과는 반드시 수축하고 fallback해야 한다.
3. 안정적인 전역 모델과 비선형·최근 모델은 역할을 나눠 결합한다.
4. 확률 모델은 평균 Brier뿐 아니라 시간·그룹별 calibration 붕괴를 막아야 한다.

따라서 이 저장소의 다음 실험은 **A00/A01 → A10 → A12 → A11/A13/A14 → A15/A20 → A22 → A30/A31** 순서가 가장 합리적이다. 문헌의 보고 성능은 가설의 근거일 뿐이고, 채택 기준은 이 저장소의 누출 없는 rolling Brier, 최악 fold, F regime, cold-start 안정성이다.

> §28에서 이 순서를 개정했다.

---

# 제2부: 구조 복원 이후에 추가된 유사 분야

## 20. 무엇이 바뀌었는가

[EDA 제2부](../eda/EDA_REPORT.md#제2부-데이터를-생성한-구조)의 결과로 §1의 문제 지도가 두 방향으로 바뀌었다.

### 20.1 TrackMan은 record linkage 문제가 아니었다 — `A30`/`A31` 폐기

§10은 "공통 상황 키만으로 후보가 중앙값 5개"라는 관측에서 출발해 확률적 record linkage(Enamorado·Fifield·Imai, Sadinle)를 처방했다. 관측은 맞지만 **매칭 단위가 틀렸다.** 투구가 아니라 경기를 매칭하면 문제는 결정적으로 풀린다.

| `A30`/`A31`의 전제 | 실제 |
| --- | --- |
| match posterior `q_ij`와 soft aggregate가 필요 | **1:1 결정적 매핑** |
| entropy, top1-top2 margin, candidate count를 피처로 | 필요 없음 |
| hard vs soft vs multiple-imputation 4개 arm 비교 | arm 1개 |
| high/medium/low confidence 경계 사전 고정 | 미매핑 0.21%용 fallback 하나 |

`A30`(linkage audit), `A31`(hard vs soft aggregate)은 실행하지 않는다. §26.4의 point-in-time 계약만 남는다. 이 변화의 방법론적 교훈은 §23에 별도로 정리했다 — **매칭 단위를 바꾸면 확률 문제가 결정 문제가 될 수 있다**는 것 자체가 유사 분야에서 잘 알려진 현상이다.

### 20.2 §1의 다섯 부류로 설명되지 않는 문제가 드러났다

| 새로 드러난 현상 | §1의 다섯 부류로 설명되는가 | 대응하는 유사 분야 |
| --- | :---: | --- |
| Trackman 물리량의 pooled 상관이 투수 간 상관과 **부호가 반대** | ❌ | **패널 데이터 계량경제학** (§21) |
| 제구 성공률이 100구에서 신뢰도 0.76에 도달 | ❌ | **심리측정학 신뢰도 이론** (§22) |
| 경기 지문으로 익명 데이터가 결정적으로 연결됨 | ❌ | **희소 데이터 재식별** (§23) |
| "시즌 내 복원"이 규정 위반인지 판단해야 함 | ❌ | **누출 분류학** (§24) |
| 2025 평균을 test를 보지 않고 추정해야 함 | 부분적 | **quantification learning** (§25) |
| F의 2023 단절이 실력이 아니라 측정 체제 변화 | 부분적 | **임상 검사 체제 변경·공식 통계 단절** (§26) |
| 개체 이력이 각 행 안에 이미 들어 있음 | 부분적 | **CTR 실시간 개체 피처·행동 스코어카드** (§27) |

## 21. 패널 데이터 계량경제학: between/within과 Mundlak 장치

### 21.1 왜 이 분야인가

[EDA §19.7](../eda/EDA_REPORT.md#197-trackman-물리량-부호가-반대인-두-신호)에서 `rel_speed`의 pooled 상관은 `+0.045`인데 투수 간 상관은 `-0.172`, 투구 내 상관은 `+0.056`이었다. 부호가 반대다.

이는 통계적 이변이 아니라 **군집 데이터에서 공변량이 두 개의 서로 다른 효과를 갖는다는, 이론이 완비된 현상**이다. 계량경제학은 이를 60년 동안 다뤄 왔다.

```text
x_ij = x̄_i  +  (x_ij - x̄_i)
       └군집간┘  └───군집내───┘
```

[Mundlak(1978)](https://doi.org/10.2307/1913646)은 random effect 모형에 **군집 평균 `x̄_i`를 추가 회귀변수로 넣으면** fixed effect 추정량과 같은 within 계수를 얻으면서 between 성분도 별도 계수로 분리된다는 것을 보였다. [Chamberlain(1982)](https://doi.org/10.1016/0304-4076(82)90094-X)이 이를 일반화했고, 생물통계 쪽에서는 [Neuhaus & Kalbfleisch(1998)](https://doi.org/10.2307/2533862)가 군집 데이터의 between/within 공변량 효과를 정확히 이 형태로 정리했다. 다수준 모형 실무에서는 [Enders & Tofighi(2007)](https://doi.org/10.1037/1082-989X.12.2.121)의 group-mean centering 논의가 표준 참고 자료다.

### 21.2 이 대회로의 번역

**진단 규칙(필수).** 군집(투수) 구조가 있는 어떤 수치 피처든 설계 전에 세 상관을 함께 기록한다.

```text
r_pooled  : 전체 행
r_between : 투수 평균끼리
r_within  : 투수 평균을 뺀 잔차끼리
```

`r_pooled`와 `r_between`의 부호가 다르면 pooled 통계로 피처를 만들면 안 된다.

**모형 규칙.** 이 대회는 특수한 제약이 있다. **평가 시점에 쓸 수 있는 것은 between 성분(과거 집계)뿐이고, within 성분은 현재 투구 값이라 금지된다.** 따라서 Mundlak 장치의 절반만 쓴다.

| 성분 | 이 대회에서 | 조치 |
| --- | --- | --- |
| `x̄_i` (투수 평균 Trackman) | **사용 가능** | 명시적 피처로 넣는다 |
| `x_ij - x̄_i` (현재 투구 편차) | **금지·미제공** | 사용하지 않는다 |

즉 우리는 **between 계수만 추정하려는 것**이므로, 학습 데이터에서 pooled 회귀를 돌리면 within 성분이 계수를 오염시킨다. 안전한 구현은 **투수 평균만 피처로 넣고 원 투구 값은 아예 파이프라인에 넣지 않는 것**이다(현재 투구 Trackman은 어차피 금지이므로 자연스럽게 지켜진다). 위험한 것은 정렬된 810,644행을 이용해 실수로 투구 단위 값을 학습에 섞는 경우다.

**같은 진단을 `asof_*`에도 적용한다.** `asof_pitcher_success_rate`도 "투수의 실력"(between)과 "시즌 내 어느 시점인가"(within, 시간에 따라 리그 평균이 내려가므로)를 동시에 담고 있다. [EDA §21.4](../eda/EDA_REPORT.md#214-왜-통산-rate는-0점인데-시즌-내-rate는-492점인가)에서 통산 rate가 0점이고 시즌 내 rate가 492점이었던 것은 이 두 성분이 통산 rate 안에서 섞여 있었기 때문으로 읽을 수 있다.

### 21.3 age-period-cohort 식별 문제

[EDA §6.2](../eda/EDA_REPORT.md#62-경험량과-cold-start)는 "누적 표본이 큰 행은 대체로 후반 시즌에 있으므로 경험 효과를 인과로 읽으면 안 된다"고 경고했다. 이것은 이름이 붙어 있는 문제다.

```text
연령(경험) + 코호트(데뷔 시즌) = 기간(시즌)
```

세 축이 선형 종속이므로 세 효과를 동시에 식별할 수 없다. 신용 위험에서 [Breeden(2007)](https://doi.org/10.1016/j.csda.2007.02.019)이 vintage 데이터의 age-period-cohort 분해로 다루는 문제와 동일하다.

실무적 대응은 **식별을 포기하고 예측만 한다**는 것이다.

- 세 축의 "진짜 효과"를 분리하려 하지 않는다.
- 대신 세 축을 모두 피처로 넣고 정규화로 통제한다: `log1p(asof_pitcher_n)`, `season`, `데뷔 시즌`(train에서 동결 가능한 per-pitcher 상수).
- **예측 목적에서는 교란이 문제가 아니다.** 문제는 2025에 세 축의 조합이 학습 범위 밖으로 나가는 것이다. 특히 트리 모델은 `season=2025`를 외삽하지 못하므로 §25의 base-rate 처리가 필요하다.

## 22. 심리측정학: 신뢰도, 안정화, 감쇠 보정

### 22.1 왜 이 분야인가

[EDA §19.3 (야구 문서)](RELATED_WORK_AND_EXPERIMENT_ROADMAP.md#193-안정화-지점-control-success는-야구-지표-중-매우-빨리-안정된다)에서 제구 성공률의 split-half 신뢰도가 100구에 0.76, 500구에 0.9 이상임을 확인했다. 이 계산은 심리측정학의 [Spearman-Brown 예언 공식](https://doi.org/10.1111/j.2044-8295.1910.tb00206.x)과 [Cronbach(1951)의 알파](https://doi.org/10.1007/BF02310555)에서 그대로 왔다.

CTR·의료 위험 문헌은 "표본이 적으면 수축하라"까지만 말한다. 심리측정학은 **얼마나 수축해야 하는지를 신뢰도로 정량화**한다.

### 22.2 세 가지 직접 적용

**(1) 신뢰도 = 수축 계수.** 표본 `n`인 개체의 관측 rate의 신뢰도는

```text
ρ(n) = n / (n + k)
```

이며, 이는 EB 수축 가중과 **정확히 같은 식**이다. 즉 §19.1(야구 문서)의 분산 성분 `k ≈ 120`은 "120구에서 신뢰도가 0.5가 된다"는 뜻이다. 실측 split-half가 100구에서 0.76을 보인 것과는 다소 차이가 있는데, 이는 실측 쪽이 "많이 던진 투수만" 대상으로 해 실력 분산이 큰 집단이기 때문이다. **두 추정치를 함께 기록하고, 격자 탐색 결과가 이 범위(대략 `k = 60~200`)를 크게 벗어나면 무언가 다른 문제를 보상하고 있는 것으로 간주한다.**

**(2) 감쇠 보정(correction for attenuation).** [Spearman(1904)](https://doi.org/10.2307/1412159)은 잡음이 있는 측정치의 상관이 신뢰도의 제곱근만큼 축소된다는 것을 보였다.

```text
r_관측 = r_참 × √(ρ_x · ρ_y)
```

이 대회에서의 함의: **저표본 투수의 `asof_*` rate는 신뢰도가 낮으므로, 그 피처의 회귀 계수가 자동으로 작아진다.** 하나의 전역 계수를 쓰면 고표본 투수에서는 과소, 저표본 투수에서는 과대 반영된다. 대응은 두 가지다.

- rate를 넣기 전에 신뢰도로 수축해서 **모든 행의 신뢰도를 균질화**한다(EB의 진짜 목적).
- 또는 `rate × log1p(n)` 상호작용을 명시해 계수가 `n`에 따라 변하게 한다(`A10`이 이미 제안).

첫 번째가 더 안정적이며, 두 방법을 동시에 쓰면 중복이다.

**(3) 신뢰도 상한이 곧 예측 상한이다.** 개체 수준 신호가 아무리 강해도 신뢰도 `ρ`를 넘는 상관은 추출할 수 없다. [EDA §22.2](../eda/EDA_REPORT.md#222-정직한-성능-상한)의 split-half 상한 646점은 정확히 이 논리로 얻은 값이다.

### 22.3 주의

- Spearman-Brown은 두 반쪽이 **교환 가능(parallel)**하다고 가정한다. 무작위 절반 분할은 이 가정에 가깝지만, 시즌 전·후반 분할은 아니다(체제가 다르다). 안정화 계산에는 무작위 분할만 쓴다.
- 표본 하한을 올릴수록 신뢰도가 올라가는 것은 **선택 효과**가 섞인 결과다. 절대값이 아니라 순서와 크기 규모만 사용한다.

## 23. 희소 데이터 재식별: 지문 매칭이 왜 통했는가

### 23.1 이 분야가 정확히 같은 문제를 다룬다

경기 하나를 그 안의 모든 투구 상태 `(inning, 초말, B, S, O)`의 multiset으로 요약하니 4,868개 train 경기 중 2,700개가 5,980개 Trackman 경기와 **모호성 없이 1:1로** 붙었다. 이것은 프라이버시 연구에서 잘 정립된 현상이다.

- [Narayanan & Shmatikov(2008)](https://doi.org/10.1109/SP.2008.33)는 익명화된 Netflix Prize 데이터에서 **소수의 평점 기록만으로** 사용자를 IMDb와 대응시켰다. 핵심 정리는 "고차원 희소 데이터에서는 몇 개의 관측점만으로 record가 거의 확실하게 유일해진다"는 것이다.
- [de Montjoye et al.(2013)](https://doi.org/10.1038/srep01376)은 이동통신 위치 데이터에서 **4개의 시공간 점**이 개인의 95%를 유일하게 식별함을 보였다.
- [de Montjoye et al.(2015)](https://doi.org/10.1126/science.1256297)은 신용카드 거래에서 **4개 거래**로 90%가 유일해짐을 보였다.

우리의 경기는 약 300개의 상태 관측점을 갖는다. 유일성은 사실상 보장된다.

### 23.2 방법론적 교훈: 매칭 단위를 올리면 확률 문제가 결정 문제가 된다

§10은 **투구 단위**로 매칭을 시도해 후보 5개를 얻었고, 그래서 확률적 linkage를 처방했다. **경기 단위**로 올리면 후보가 1개가 된다.

이것은 entity resolution 문헌의 표준 통찰이다. 개별 속성이 약한 식별력을 가질 때도, **같은 개체에 속한 속성들의 집합**은 강한 식별력을 갖는다. [Broder(1997)](https://doi.org/10.1109/SEQUEN.1997.666900)의 집합 유사도(MinHash) 계열 기법이 이 원리 위에 서 있다.

> **일반화 가능한 규칙:** 두 테이블을 연결할 때 후보가 많으면, 더 정밀한 키를 찾기 전에 **더 큰 단위로 묶어서 집합 지문을 만들 수 있는지** 먼저 본다. 개별 행이 모호해도 행의 묶음은 유일할 수 있다.

이 규칙은 이 대회 밖에서도 재사용 가능하며, 이 조사에서 얻은 가장 이전성 높은 결론이다.

### 23.3 반대 방향의 교훈: 우리도 재식별당할 수 있다

같은 원리가 **평가 데이터에도 적용된다**는 점이 중요하다. 만약 누군가 2025 KBO 공개 기록으로 test 행을 역식별해 실제 결과를 붙이면 그것은 명백한 외부 데이터 사용이자 평가 데이터 유출 시도이며, [`COMPETITION.md` §9.1](../COMPETITION.md)의 실격 사유다.

이 저장소는 **train 내부의 두 제공 파일을 연결하는 것**만 한다. 두 파일 모두 대회가 제공한 공식 데이터이고, 연결 결과는 학습 자산일 뿐 평가 데이터에 대한 어떤 외부 정보도 도입하지 않는다. 이 구분을 실험 기록에 명시해 둔다.

## 24. 누출 분류학: "시즌 내 복원"은 왜 누출이 아닌가

### 24.1 표준 정의

[Kaufman, Rosset, Perlich(2011)](https://doi.org/10.1145/2020408.2020496)의 KDD 논문은 데이터 마이닝 누출의 표준 분류를 제시한다. 핵심 판정 기준은 하나다.

> **피처의 값이 예측 시점에 합법적으로 이용 가능했는가.**

논문은 특히 **식별자·행 순서에서 오는 누출**(row ID가 시간이나 클래스와 상관된 경우)을 대표 사례로 든다. 이 대회 데이터에는 그 사례가 실제로 존재한다.

### 24.2 이 저장소에서 발견한 세 가지를 이 기준으로 판정한다

| 발견 | 예측 시점 이용 가능? | 다른 test 행 사용? | 판정 |
| --- | :---: | :---: | --- |
| train 행 순서로 경기 복원 | test에서 불가([EDA §17.3](../eda/EDA_REPORT.md#173-파생되는-사실과-한계)) | — | **학습 자산으로만 사용** |
| `row_id` 숫자를 시간 인덱스로 사용 | test는 시간 순이 아님 | — | **근거 없음, 사용 금지** |
| **시즌 내 누적 복원** | **그 행의 컬럼 값 + 학습에서 만든 상수** | **없음** | **합법** |

세 번째가 핵심이다. 판정 근거를 분해하면:

```text
rate_2025(행) = f( 행.asof_pitcher_n,          <- 그 행의 입력
                   행.asof_pitcher_success_rate,<- 그 행의 입력
                   D[행.pitcher_id] )           <- train만으로 만든 고정 사전
```

- **다른 test 행이 등장하지 않는다.** 한 행만 넣어도 같은 값이 나온다.
- **배치 크기·순서에 의존하지 않는다.**
- **미래 정보가 아니다.** `asof_*`는 운영진이 "투구 직전까지"의 정보로 정의해 제공한 공식 입력이다.

우리는 누적을 **새로 계산하는 것이 아니라**, 운영진이 이미 각 행에 넣어 준 누적값에서 학습 시점의 상수를 뺄 뿐이다. 대회 금지 목록의 "test 내부 rolling/expanding/누적"은 **test 행들을 서로 참조해 누적을 만드는 행위**를 가리키며, 성질이 다르다.

### 24.3 그러나 한 걸음만 더 가면 위반이다

같은 도구로 만들 수 있는 **명백한 위반**을 명시해 둔다.

| 하고 싶은 것 | 판정 |
| --- | --- |
| test 행들의 `rate_2025`를 평균 내어 2025 리그 수준 추정 | **위반** — test 전체 분포 사용 |
| test에 등장한 `n_season` 최대값으로 시즌 진행도 정규화 | **위반** — test 전체 통계 |
| test의 투수별 등장 빈도로 fallback 강도 조절 | **위반** — 명시적 금지 항목 |
| test 예측 확률 히스토그램으로 절편 재보정 | **위반** — §8.4에서 이미 금지 |

경계가 얇으므로, `E14` 구현 시 **행 단위 함수 하나로 작성하고 test 데이터프레임 전체를 인자로 받는 코드 경로를 만들지 않는 것**이 가장 안전하다. `A00`의 불변성 테스트(단일 행/배치/셔플/중복)가 이 경계를 자동으로 지키는 게이트다.

## 25. 표적 집합을 보지 않고 사전확률을 옮기는 문제

### 25.1 이 문제에 이름이 있다

[EDA §22.3](../eda/EDA_REPORT.md#223-2025-base-rate를-맞히는-것의-가치)에서 2025 평균을 1%p 틀리면 40점, 2%p면 160점을 잃는다는 것을 확인했다. 후보 외삽값이 `0.462~0.486`으로 2.4%p 벌어져 있으므로 최대 230점이 걸려 있다.

"표적 도메인의 클래스 사전확률을 추정한다"는 문제는 **quantification learning**(또는 prevalence estimation)이라는 독립 연구 분야다. [Forman(2008)](https://doi.org/10.1007/s10618-008-0097-y)이 분류기 출력에서 prevalence를 추정하는 보정법을 정리했고, [González et al.(2017)의 서베이](https://doi.org/10.1145/3117807)가 전체 지형을 정리한다. [Saerens, Latinne, Decaestecker(2002)](https://doi.org/10.1162/089976602753284446)의 EM 기반 사전확률 보정은 이 분야의 표준 기법이다.

### 25.2 그런데 이 대회에서는 그 분야 전체를 쓸 수 없다

quantification의 모든 표준 기법은 **표적 표본 전체**를 본다. Saerens EM은 test 예측 분포를 반복적으로 사용하고, Forman의 ACC/PACC는 test의 예측 클래스 비율을 사용한다. 전부 §24.3의 위반 목록에 해당한다.

| quantification 표준 기법 | 이 대회 |
| --- | --- |
| Classify & Count, ACC, PACC | **금지** (test 예측 분포 사용) |
| Saerens EM 사전확률 보정 | **금지** |
| HDy, 분포 매칭 계열 | **금지** |
| Test-time adaptation ([Tent](https://arxiv.org/abs/2006.10726), [prediction-time BN](https://arxiv.org/abs/2006.10963)) | **금지** (배치 통계 사용) |

**따라서 이 문제는 quantification이 아니라 시계열 외삽 문제로 다뤄야 한다.** 이 점을 명시하는 것이 이 절의 목적이다. 문헌이 풍부하다고 해서 쓸 수 있는 것이 아니다.

### 25.3 남는 합법적 방법

1. **학습 시즌 평균의 시계열 외삽.** 선형 추세, 최근 창 평균, 감쇠 가중. R과 F를 분리하고 F는 2023 단절 이후만 사용한다([EDA §20.3](../eda/EDA_REPORT.md#203-모델링-함의)).
2. **pseudo-forward 검증으로 외삽 오차를 실측한다.** 2022·2023·2024를 각각 가상 미래로 두고 "그 시점까지의 데이터만으로 외삽했다면 몇 %p 틀렸는가"를 기록한다. 이것이 2025 외삽 불확실성의 유일한 실증 추정치다.
3. **개체 수준 우회(§27).** 각 행이 들고 있는 그 투수의 2025 성적이 리그 수준을 간접적으로 담고 있다. 집합을 보지 않고 사전확률 이동을 흡수하는 유일한 경로다.

3번은 quantification 문헌에 없는 접근인데, 이유는 그 분야가 대체로 **개체 이력이 입력에 들어 있지 않은** 문제를 다루기 때문이다. 이 대회의 특수 구조가 만든 기회다.

### 25.4 이중 보정 위험

1번(명시적 외삽)과 3번(개체 수준 자동 흡수)은 **같은 편향을 두 번 고칠 수 있다.** [Calibration 실험 보고서 §6.2](../experiments/CALIBRATION_ENSEMBLE_REPORT.md)가 이미 refit 모델에 과거 calibrator를 옮겨 이중 보정이 일어난 사례를 기록했다. 반드시 세 조합(외삽만 / 개체 피처만 / 둘 다)을 비교하고, 예측 평균이 목표 수준을 지나치지 않는지 확인한다.

## 26. 측정 체제 단절: 임상 검사·공식 통계·신용 vintage

### 26.1 F의 2023 단절은 concept drift가 아니라 measurement change다

[EDA §20.2](../eda/EDA_REPORT.md#202-2023년-급변은-라벨-체제-변화다)에서 확인한 것:

- 동일 투수 70명의 평균 성공률이 `0.7050 → 0.4836` (**-22.1%p**)
- 모든 2군 구장에서 동시 발생
- 같은 기간 R의 within-pitcher 변화는 `-0.4 ~ -2.2%p`

§7이 인용한 부정거래 concept drift 문헌은 "세상이 서서히 변한다"를 다룬다. 이것은 다르다. **측정 도구가 바뀌었다.** 이 구분에 맞는 유사 분야는 따로 있다.

### 26.2 임상 검사법 변경

임상 검사실에서 측정 방법·시약·장비가 바뀌면 같은 환자의 값이 달라진다. 표준 절차는 확립돼 있다.

- 두 방법의 일치도를 [Bland & Altman(1986)](https://doi.org/10.1016/S0140-6736(86)90837-8)의 차이 플롯으로 평가한다.
- 계통 편향이 있으면 Deming/Passing-Bablok 회귀로 **bridging(가교) 방정식**을 추정한다.
- **가교 없이 두 시기 데이터를 그냥 합치지 않는다.**

이 대회에 옮기면:

| 임상 절차 | 이 대회 대응 |
| --- | --- |
| 동일 검체를 두 방법으로 측정 | **불가능** — 같은 투구를 두 체제로 라벨링한 데이터가 없다 |
| bridging 방정식 추정 | 동일 투수의 2022 vs 2023 F 성적으로 근사 가능 |
| 가교 없이 합치지 않는다 | **2022 이전 F를 그대로 학습에 넣지 않는다** |

동일 투수 70명이라는 준-가교 표본이 있으므로, 원한다면 `F_2022 → F_2023` 사상을 추정해 과거 F 데이터를 새 척도로 옮길 수 있다. 다만 표본이 70명이고 사상의 형태(단순 이동인가 척도 변화인가)를 검증할 방법이 제한적이므로, **먼저 시도할 것은 가교가 아니라 배제**다.

### 26.3 공식 통계의 계열 단절

통계청·중앙은행이 조사 방식이나 분류 체계를 바꿀 때의 표준 관행도 같다: 단절 시점에 **새 계열을 시작하거나**, 겹치는 기간의 병행 조사로 역산 계수를 만든다. 겹치는 기간이 없으면 역산하지 않고 단절을 명시한다.

이 대회는 겹치는 기간이 없다. 따라서 **F는 2023부터 새 계열로 취급한다**는 것이 정석이다.

### 26.4 신용 vintage 분석과 point-in-time 계약

§6.4가 인용한 out-of-time 검증에 더해, 신용 위험의 **vintage(코호트) 분석**은 origination 시점별로 성과 곡선을 따로 그린다. [Breeden(2007)](https://doi.org/10.1016/j.csda.2007.02.019)의 age-period-cohort 분해가 그 형식화다(§21.3).

이 대회에서 시즌은 vintage에 해당한다. 실무 규칙:

- 시즌별 성과 곡선을 항상 분리해 본다(이미 `A00`에 포함).
- **정책 변경(=측정 체제 변경) 시점 이전 vintage는 별도 표시**하고, 모델에 넣을지 말지를 명시적 실험으로 정한다.
- §10.4의 point-in-time 계약은 그대로 유효하며, 이제 매칭된 2,700경기의 **실제 날짜**를 쓸 수 있으므로 시즌 단위가 아닌 날짜 단위 cutoff가 가능하다.

```text
build_trackman_features(entity, cutoff_date)   # 시즌이 아니라 날짜
  1. 매칭된 경기의 실제 game_date < cutoff_date 만 필터
  2. 매칭 안 된 경기는 시즌 경계로 보수적 처리
  3. 구종군별 강건 통계 -> low-n 계층 수축
  4. immutable artifact로 저장
```

### 26.5 변화 시점을 자동으로 찾는 도구

지금은 2023 단절을 눈으로 찾았다. 추가 단절이 더 있는지 자동 점검할 가치가 있다. 표준 도구는 [Chow(1960)](https://doi.org/10.2307/1910133)의 구조 변화 검정, [Bai & Perron(1998)](https://doi.org/10.2307/2998540)의 다중 구조 변화 추정, 그리고 온라인 계열의 [PELT(Killick et al. 2012)](https://doi.org/10.1080/01621459.2012.737745)·[Bayesian online changepoint detection(Adams & MacKay 2007)](https://arxiv.org/abs/0710.3742)이다.

적용 방법: 세그먼트별(`game_type` × 팀 × 월) 시계열에 changepoint 검출을 돌려 **2023 F 외에 우리가 놓친 단절이 있는지** 확인한다. 저비용 고가치의 안전 점검이다.

## 27. 개체 수준 실시간 피처: CTR·부정거래·행동 스코어카드

### 27.1 "시즌 내 복원"은 이 분야에서는 표준 관행이다

[EDA §21](../eda/EDA_REPORT.md#21-시즌-내-누적-복원-2025-drift에-접근하는-합법적-경로)의 발견은 야구에서는 새롭지만, **개체 이력이 요청에 함께 실려 오는 시스템**에서는 오래된 표준이다.

| 분야 | 대응하는 피처 | 성격 |
| --- | --- | --- |
| CTR / 광고 | 사용자·광고의 **최근 CTR**, 최근 노출·클릭 카운트 | 요청 시점에 서빙 스토어에서 조회 |
| 부정거래 | 카드·계정의 **velocity 피처**(최근 1시간/1일 거래 수·금액) | 승인 요청에 함께 계산 |
| 신용 | **behavioural scorecard** — 개설 시점 정보(application)가 아니라 계좌의 최근 행동 | 매월 갱신 |
| 의료 | 환자의 **baseline 대비 현재 방문** 값 | 차트에서 조회 |

§4.3에서 인용한 [Meta의 CTR 논문](https://ai.meta.com/research/publications/practical-lessons-from-predicting-clicks-on-ads-at-facebook/)이 "user/ad의 **역사 피처가 가장 중요했다**"고 보고한 것이 정확히 이 지점이다. [DIN(Zhou et al. 2018)](https://doi.org/10.1145/3219819.3219823)과 [DIEN(Zhou et al. 2019)](https://doi.org/10.1609/aaai.v33i01.33015941)은 이 이력을 시퀀스로 확장한 형태다.

핵심 통찰:

> **모델을 표적 도메인에 적응시킬 수 없을 때, 피처를 적응시킨다.** 개체 수준 충분통계를 표적 기간 안에서 계산해 각 행에 실어 보내면, 모델은 고정된 채로도 새 체제를 관측한다.

이 대회의 특수성은 **그 충분통계를 우리가 서빙할 필요조차 없다는 것**이다. 운영진이 이미 `asof_pitcher_n`과 `asof_pitcher_success_rate`에 넣어 두었고, 우리는 학습 시점 상수를 빼서 창(window)만 바꾸면 된다.

### 27.2 신용 위험의 application vs behavioural 구분이 가장 정확한 유비

신용 스코어링은 두 종류의 모형을 명확히 구분한다([Thomas, Edelman, Crook, *Credit Scoring and Its Applications*](https://doi.org/10.1137/1.9781611974560)).

| | application scorecard | **behavioural scorecard** |
| --- | --- | --- |
| 입력 | 개설 시점의 정적 정보 | 계좌의 최근 거래·연체 행동 |
| 신규 고객 | 사용 가능 | **사용 불가** |
| 성능 | 낮음 | 높음 |
| 체제 변화 | 취약 | **상대적으로 강건** |

이 대회에 그대로 대응한다.

| | 이 대회의 application 해당 | 이 대회의 behavioural 해당 |
| --- | --- | --- |
| 입력 | 손, 팀, 역할, Trackman 프로파일, 통산 rate | **시즌 내 rate**, `n_season` |
| 신규 투수 | 사용 가능 | `n_season` 작을 때 약함 |
| 2024 단독 점수 | 통산 rate **0점** | 시즌 내 rate **492점** |

**따라서 두 모형을 나누어 만들고 `n_season`으로 게이팅하는 것이 이 분야의 정석 설계다.**

```text
logit(p) = f_application(손, 팀, 역할, Trackman 프로파일, 상황)
         + g(n_season; k) × f_behavioural(rate_season, rate_season - rate_career)
g(n; k) = n / (n + k)
```

`A11`의 reliability-gated player residual과 같은 게이트 구조인데, 게이트의 대상이 "ID 잔차"가 아니라 **"시즌 내 행동 성분"**이라는 점이 다르다. [EDA §21.3](../eda/EDA_REPORT.md#213-검증-2024를-2025처럼-다뤘을-때)에서 `n_season > 0`인 행이 99.85%, `n_season ≥ 200`이 75.8%이므로 게이트는 대부분의 행에서 열린다.

### 27.3 이 접근의 알려진 실패 모드

CTR·부정거래 실무에서 반복 보고되는 문제를 그대로 옮긴다.

| 실패 모드 | 이 대회에서의 형태 | 대응 |
| --- | --- | --- |
| **피처 서빙 편차(training/serving skew)** | 학습 시 창 정의와 추론 시 창 정의가 다름 | `n_end` 사전을 만드는 코드와 추론 코드를 **같은 함수**로 |
| **신규 개체 절벽** | 2025 데뷔 투수는 `n_season`이 작다 | application 성분으로 부드럽게 후퇴 |
| **초기 구간 편향** | 시즌 초 소표본에서 rate가 극단적 | EB 수축 `k ≈ 120` (§22.2) |
| **이력 피처가 라벨을 역으로 담음** | 투수가 부진하면 등판이 줄어 `n_season`이 작아짐 | `n_season` 자체를 피처로 쓸 때 주의, ablation 필수 |
| **이중 보정** | 명시적 base-rate 외삽과 중복 | §25.4의 3조합 비교 |

네 번째가 가장 미묘하다. `n_season`은 시즌 진행도이면서 동시에 **투수의 사용량 = 코칭스태프의 평가**를 담는다. 이는 정보이기도 하고 편향이기도 하다. 반드시 `n_season` 포함/제외 ablation을 돌린다.

## 28. 추가 실험 명세와 개정된 실행 순서

### A50 — behavioural/application 분리 모형 (신설, 최우선)

- §27.2의 게이트 구조를 `A11`의 확장으로 구현한다.
- arm: (1) application만, (2) behavioural만, (3) 고정 가중 결합, (4) `n_season` 게이트 결합.
- 게이트 `k`는 §22.2의 신뢰도 범위(`60~200`)에서만 탐색한다.
- 세그먼트: `n_season` 구간, 신규 투수, R/F, 시즌 초/중/말(월 대리).
- 야구 문서의 `E14`와 같은 피처를 쓰지만, `E14`는 **피처의 가치**를, `A50`은 **모형 구조**를 검증한다.

### A51 — between/within 진단 게이트 (신설, 필수 절차)

- 모든 수치 피처에 대해 `r_pooled`, `r_between`, `r_within`을 산출해 실험 로그에 저장한다.
- **부호가 다르면 자동 경고**를 띄우고, pooled 통계 기반 피처를 만들지 못하게 한다.
- `A00`의 저장 스키마에 세 필드를 추가한다.
- 비용이 거의 없고 §21.2의 부호 오류를 구조적으로 막는다.

### A52 — base-rate 외삽의 pseudo-forward 검증 (신설, 최우선)

- 2022·2023·2024를 각각 가상 미래로 두고, 그 시점까지의 데이터만으로 여러 외삽 규칙을 적용해 **실제 오차 분포**를 만든다.
- 규칙: 전체 선형 / 최근 3시즌 선형 / 최근 2시즌 평균 / 직전 시즌 유지 / R·F 분리 후 재결합.
- 산출물은 점 추정이 아니라 **오차의 분포**다. 이것이 2025 예측 구간의 근거가 된다.
- §25.2의 quantification 기법은 사용하지 않는다는 것을 기록에 명시한다.

### A53 — 다중 changepoint 점검 (신설, 소규모)

- 세그먼트별 월 단위 성공률 시계열에 PELT 또는 Bai-Perron을 적용한다.
- 목적은 "2023 F 외에 우리가 놓친 단절이 있는가" 하나다.
- 발견되면 해당 세그먼트를 `A21`의 그룹 정의에 추가한다.

### A54 — 날짜 단위 point-in-time Trackman 자산 (`A31` 대체)

- §26.4의 계약대로 매칭된 2,700경기의 실제 날짜를 사용한다.
- 매칭되지 않은 경기는 시즌 경계로 보수적 처리.
- 피처는 야구 문서 `E20R`의 6개로 제한한다.
- 주 평가 지표는 `asof_pitcher_n` 구간별 paired Brier.

### 개정된 실행 순서

```text
1차 (정보가치 최상)
  A00  temporal probability audit  (+ A51 between/within 필드 추가)
  A01  shift taxonomy
  A52  base-rate pseudo-forward 검증
  A50  behavioural/application 분리   <- 신설, 최우선
  A10  decay hierarchical EB (k는 분산 성분·신뢰도로 고정)

2차 (구조와 상호작용)
  A12  GA2M/EBM
  A11  gated ID residual
  A13  tree leaf -> Logistic
  A14  shallow FM
  A53  changepoint 점검 (저비용, 언제든)

3차 (결합과 보정)
  A15  stable/recent experts
  A20  constrained ensemble
  A21  robust selection
  A22  multi-domain OOF calibration

4차 (보조 로그)
  A54  날짜 단위 point-in-time Trackman

폐기
  A30  probabilistic linkage audit   -> §20.1
  A31  hard vs soft aggregate        -> A54로 대체
```

`A40`(FTRL/hash), `A41`(DeepFM/FT-Transformer)은 §13의 판단대로 마지막에 둔다.

### §17에 추가할 금지 방법

| 방법 | 왜 사용하지 않는가 |
| --- | --- |
| quantification (ACC/PACC/Saerens EM/HDy) | test 예측 분포 사용 (§25.2) |
| test-time adaptation (Tent, prediction-time BN) | 배치 통계 사용 |
| test 행들의 `rate_2025` 집계 | test 전체 분포 사용 (§24.3) |
| `row_id` 숫자를 시간 인덱스로 사용 | test는 시간 순이 아님 (§24.2) |
| 2025 KBO 공개 기록으로 test 행 역식별 | 외부 데이터·평가 유출 (§23.3) |
| pooled 상관 기반 Trackman 피처 설계 | 부호가 뒤집힘 (§21.2) |
| 2022년 이전 F를 가교 없이 학습에 포함 | 측정 체제 단절 (§26.2) |

## 29. 추가 참고 문헌

### 패널·다수준 모형

- Mundlak, 1978, [On the Pooling of Time Series and Cross Section Data](https://doi.org/10.2307/1913646)
- Chamberlain, 1982, [Multivariate regression models for panel data](https://doi.org/10.1016/0304-4076(82)90094-X)
- Neuhaus & Kalbfleisch, 1998, [Between- and within-cluster covariate effects](https://doi.org/10.2307/2533862)
- Enders & Tofighi, 2007, [Centering predictor variables in cross-sectional multilevel models](https://doi.org/10.1037/1082-989X.12.2.121)
- Breeden, 2007, [Modeling data with multiple time dimensions](https://doi.org/10.1016/j.csda.2007.02.019) — age-period-cohort

### 신뢰도·측정

- Spearman, 1904, [The Proof and Measurement of Association between Two Things](https://doi.org/10.2307/1412159) — 감쇠 보정
- Spearman, 1910 / Brown, 1910, [예언 공식](https://doi.org/10.1111/j.2044-8295.1910.tb00206.x)
- Cronbach, 1951, [Coefficient alpha and the internal structure of tests](https://doi.org/10.1007/BF02310555)
- Bland & Altman, 1986, [Statistical methods for assessing agreement](https://doi.org/10.1016/S0140-6736(86)90837-8)

### 재식별·연결

- Narayanan & Shmatikov, 2008, [Robust De-anonymization of Large Sparse Datasets](https://doi.org/10.1109/SP.2008.33)
- de Montjoye et al., 2013, [Unique in the Crowd](https://doi.org/10.1038/srep01376)
- de Montjoye et al., 2015, [Unique in the shopping mall](https://doi.org/10.1126/science.1256297)
- Broder, 1997, [On the resemblance and containment of documents](https://doi.org/10.1109/SEQUEN.1997.666900)

### 누출·사전확률 이동·구조 변화

- Kaufman, Rosset, Perlich, 2011, [Leakage in Data Mining](https://doi.org/10.1145/2020408.2020496)
- Forman, 2008, [Quantifying counts and costs via classification](https://doi.org/10.1007/s10618-008-0097-y)
- González et al., 2017, [A Review on Quantification Learning](https://doi.org/10.1145/3117807)
- Saerens, Latinne, Decaestecker, 2002, [Adjusting the outputs of a classifier to new a priori probabilities](https://doi.org/10.1162/089976602753284446)
- Wang et al., 2021, [Tent: Fully Test-Time Adaptation](https://arxiv.org/abs/2006.10726) — 사용 금지 사례
- Nado et al., 2020, [Prediction-Time Batch Normalization](https://arxiv.org/abs/2006.10963) — 사용 금지 사례
- Chow, 1960, [Tests of Equality Between Sets of Coefficients](https://doi.org/10.2307/1910133)
- Bai & Perron, 1998, [Estimating and testing linear models with multiple structural changes](https://doi.org/10.2307/2998540)
- Killick, Fearnhead, Eckley, 2012, [Optimal detection of changepoints (PELT)](https://doi.org/10.1080/01621459.2012.737745)
- Adams & MacKay, 2007, [Bayesian Online Changepoint Detection](https://arxiv.org/abs/0710.3742)

### 개체 수준 실시간 피처

- Zhou et al., 2018, [Deep Interest Network for CTR Prediction](https://doi.org/10.1145/3219819.3219823)
- Zhou et al., 2019, [Deep Interest Evolution Network](https://doi.org/10.1609/aaai.v33i01.33015941)
- Graepel et al., 2010, [Web-Scale Bayesian CTR Prediction (adPredictor)](https://icml.cc/Conferences/2010/papers/901.pdf)
- Thomas, Edelman, Crook, [Credit Scoring and Its Applications](https://doi.org/10.1137/1.9781611974560) — application vs behavioural scorecard
