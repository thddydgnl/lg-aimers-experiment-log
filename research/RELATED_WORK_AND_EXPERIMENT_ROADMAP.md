# 투구 제구 예측 관련 연구·프로젝트 조사와 실험 로드맵

> 대상: LG Aimers 9기 투구 제구 성공 확률 예측 해커톤  
> 조사 기준일: **2026-08-17 KST** (§1~§17), **2026-08-17 2차** (§18~§23)  
> 대회 문제·규칙: [`COMPETITION.md`](../COMPETITION.md)  
> 현재 실험 기준선: [`CALIBRATION_ENSEMBLE_REPORT.md`](../experiments/CALIBRATION_ENSEMBLE_REPORT.md)  
> 구조 분석: [`EDA_REPORT.md` 제2부](../eda/EDA_REPORT.md#제2부-데이터를-생성한-구조)
> 현재 실행 환경·자원 판단: [`LOCAL_ENVIRONMENT.md`](../LOCAL_ENVIRONMENT.md), [`EXPERIMENT_PLAN.md` §13](../EXPERIMENT_PLAN.md#13-필요-자원과-실행-환경)

> **개정 안내.** §1~§17은 데이터의 잠재 구조를 복원하기 전에 작성됐다. [EDA 제2부](../eda/EDA_REPORT.md#제2부-데이터를-생성한-구조)에서 Trackman 연결이 결정적으로 풀리고 시즌 내 누적 복원 경로가 발견되면서 일부 전제가 바뀌었다. **§18이 무엇이 바뀌었는지 먼저 정리한다.** §19~§21은 이 도메인의 "진짜 실력 추정" 방법론을 문헌과 **이 데이터에서의 실측**으로 함께 다루고, §22가 개정된 실험 명세다.

## 1. 결론부터

이 대회와 완전히 같은 공개 문제는 찾지 못했다. 공개 연구는 대체로 현재 투구의 **실제 위치·구종·릴리스 값·포수 미트 위치**를 사용하지만, 이 대회는 그 정보를 모두 금지하고 투구 직전 정보만 허용한다. 따라서 논문의 최고 성능 모델을 그대로 복제하는 것보다, 반복해서 확인되는 원리를 대회용 사전 피처로 바꾸는 것이 중요하다.

가장 가능성이 높은 실험 순서는 다음과 같다.

1. **시간·체제 변화 대응**: 최근 시즌 가중치, 학습 창 길이, `season × game_type`, 2025 기준 base-rate 외삽을 먼저 검증한다.
2. **누적 성공률의 표본 수 기반 수축**: `asof_*_rate`를 그대로 믿지 말고 empirical Bayes, 최근 기록과 누적 기록의 차이, `rate × log1p(n)`을 사용한다.
3. **고카디널리티 범주 모델**: 투수·타자 ID를 CatBoost 또는 시간 순 계층 인코딩으로 사용하되 cold-start와 미래 정보 누출을 통제한다.
4. **Trackman의 평균보다 분포와 일관성**: 과거 구속·무브먼트·릴리스 위치의 평균뿐 아니라 표준편차, 강건 산포, 공분산, 분위수, 최근-장기 변화, 다봉성을 만든다.
5. **잠재 구종군 혼합**: 현재 구종을 맞혀 입력하는 것이 아니라, 사전에 알 수 있는 구종군 확률로 구종군별 제구 확률을 주변화한다.
6. **다양한 모델의 제한적 앙상블**: Linear의 안정성과 CatBoost/HGB의 비선형성을 시간 순 OOF에서만 조합한다.
7. **calibration은 마지막**: 기본 모델을 고정한 뒤 여러 시간 fold의 OOF로 beta 또는 affine-logit 보정을 검증한다. 직전 한 시즌 보정은 이미 불안정했다.

반대로 현재 투구의 lag/이전 행을 만드는 LSTM·Transformer, 실제 구종/위치 추정값을 현재 행에 붙이는 방식, test 내부 집계, 무작위 K-fold, raw 투수-타자 pair target encoding은 우선순위가 낮거나 규정상 사용할 수 없다.

## 2. 조사 범위와 근거 수준

### 2.1 무엇을 조사했는가

다음 여섯 축을 영어·한국어 키워드로 조사했다.

- pitching command/control, intended target, pitch location accuracy
- release parameters, Trackman, pitching kinematics, repeatability
- called-strike probability, pitch framing, actor effects
- pitcher-batter matchup, hierarchical Bayes, player embedding
- pitch selection/sequence, count-dependent strategy
- Brier score, calibration, concept/covariate drift

논문 원문, 저널·학회 페이지, 저자 공개 코드, KBO 공식 공지를 우선 확인했다. 공개 코드가 없는 프로젝트는 방법과 재현성 한계를 함께 적었다.

“전부”를 문자 그대로 보장할 수는 없다. 비공개 구단 연구, 유료 데이터 제품의 내부 모형, 검색 엔진에 노출되지 않은 프로젝트는 확인할 수 없다. 이 문서는 **2026-08-17까지 공개 검색 가능한 자료 중 대회에 직접 또는 방법론적으로 연결되는 연구**를 폭넓게 정리한 조사본이다.

### 2.2 근거 등급

| 등급 | 의미 | 사용 원칙 |
| --- | --- | --- |
| A | 동료평가 논문·공식 기관 자료 | 실험 우선순위의 주 근거 |
| B | 공개 preprint·학술 프로젝트 | 유망 가설로 사용하되 재검증 |
| C | 분석 기사·공개 GitHub 프로젝트 | 구현 아이디어와 부정적 사례 참고 |
| D | 일반 ML 방법론 | 대회 데이터에서 시간 순 검증 후 채택 |

논문에 보고된 정확도·상관·R²는 데이터, 목표, 분할 방식이 다르므로 이 대회의 예상 점수로 해석하지 않는다.

## 3. 대회 문제와 기존 연구의 차이

| 항목 | 기존 제구·위치 연구 | 이 대회에서 가능한 것 |
| --- | --- | --- |
| 목표 | 실제 위치 오차, 타깃-공 거리, called strike, 다음 구종 | `control_success=1` 확률 |
| 현재 실제 위치 | 대부분 사용 | **금지/미제공** |
| 포수 요구 위치 | 일부 COMMANDf/x·CV 연구에서 사용 | **미제공** |
| 현재 실제 구종 | 대부분 사용 | **금지/미제공** |
| 현재 릴리스·무브먼트 | Trackman 연구의 핵심 입력 | **금지**; 2019~2024 과거 요약만 가능 |
| 선수·카운트·손 | 대체로 사용 | 사용 가능 |
| 시퀀스 | 이전 투구를 사용 | 평가 다른 행 사용 금지; 현재 행에 이전 투구 정보 없음 |
| 검증 | 무작위 분할도 흔함 | 2019→2025이므로 시간 순 검증 필수 |
| 평가 | 위치 거리, accuracy, log loss 등 | Brier Skill Score |

이 대회의 실패 정의는 단순한 볼/스트라이크가 아니다. 가운데 위험 코스, 크게 벗어난 공, 포수 요구 방향과 반대인 공이 모두 실패다. 따라서 called-strike 연구는 **상황·선수 효과를 분해하는 방법**만 이전하고, 그 확률을 제구 성공 확률로 동일시해서는 안 된다.

## 4. 가장 직접적인 관련 연구: 의도와 실행의 분리

### 4.1 xCTRL: 개인별 다봉 타깃 추정

Ludwig, Brill, Wyner의 [xCTRL 논문](https://arxiv.org/abs/2508.19184)은 이 대회 목표와 가장 가까운 공개 연구다. 2025년 공개 preprint이며 [저자 코드](https://github.com/mattludwig6/Pitching-Control-Metric)도 있다.

핵심 방법은 다음과 같다.

- 투수 × 시즌 × 구종 × 타자 손 조합별 실제 위치 분포를 Gaussian mixture로 적합한다.
- 한 투수가 여러 코스를 목표로 할 수 있음을 여러 mixture component로 표현한다.
- 관측 위치에서 각 잠재 타깃까지의 거리를 posterior 확률로 가중해 실행 오차를 계산한다.
- component 수는 검증 데이터 likelihood로 정하고, bootstrap과 여러 초기값으로 불확실성과 local optimum을 다룬다.
- count까지 세분화하면 표본 부족이 심해져 전체 분포로 수축하는 방법을 시험했지만, 저자도 count별 결과가 아직 불안정하다고 명시한다.

보고 결과에서 xCTRL의 연도 간 안정성은 fastball 표본에서 Location+보다 높았고, BB/9·WHIP·RE 계열 미래 성과와도 더 강한 관계를 보였다. 다만 의도는 실제 위치의 반복 패턴으로 간접 추정했으며, 진짜 포수 사인과 동일하다고 보장할 수 없다.

대회로 옮길 수 있는 부분:

- 개인별·구종군별 분포와 여러 작전 타깃을 허용한다.
- 표본이 작은 `pitcher × hand × count`는 투수 전체 → 손 → count 순으로 수축한다.
- 평균 하나보다 산포, 공분산, mixture component 수, mixture entropy를 사용한다.
- 현재 위치가 없으므로 xCTRL을 계산한다고 부르지 않고, **과거 실행 일관성 proxy**로만 사용한다.

옮길 수 없는 부분:

- 현재 투구의 실제 위치와 실제 구종이 없으므로 pitch-level target distance를 계산할 수 없다.
- 제공 Trackman에는 plate location과 release angle이 없다.
- 외부 Statcast나 pybaseball 데이터를 모델 입력으로 사용할 수 없다.

### 4.2 고정 타깃 실험과 오차 타원

[Shinya et al.](https://doi.org/10.1080/02640414.2016.1258484)은 대학 투수 18명이 같은 타깃에 각각 100구를 던진 위치를 2차원 확률 분포로 분석했다. 위치 오차를 이변량 정규분포와 95% 오차 타원으로 표현했고, 타원의 방향이 투구 폼·팔 경로와 관련됨을 보였다.

이 결과는 `rel_height`와 `rel_side`를 각각 따로 요약하는 것만으로 부족할 수 있음을 뜻한다. 제공 데이터에서는 다음 값을 후보로 둔다.

- `std(rel_height)`, `std(rel_side)`
- 두 값의 covariance와 correlation
- 공분산 행렬 고유값, 장축/단축 비, 타원 면적 proxy
- 투수 손과 release-side 부호를 표준화한 arm-slot proxy

단, 제공 Trackman의 release point 분포는 실제 plate location 오차가 아니며, 여러 구종·여러 타깃을 섞으면 산포를 과대평가한다. 반드시 구종군과 시즌을 나눠 계산한다.

### 4.3 실제 포수 타깃을 사용한 연구

[UCL 재건술 전후 COMMANDf/x 연구](https://pmc.ncbi.nlm.nih.gov/articles/PMC7747121/)는 카메라로 포수 미트의 초기 위치와 공의 plate location을 함께 측정해 fastball 타깃 오차를 계산했다. 이는 “제구”를 실제 타깃과의 거리로 정의한 직접 사례다. 그러나 비공개 COMMANDf/x 데이터, 수술 전후 비교, fastball 중심 연구라 이 대회에 그대로 재현할 수 없다.

의미는 명확하다. **의도 타깃이 관측되지 않는 데이터에서 위치 산포를 제구 자체로 단정하면 안 된다.** 따라서 Trackman 산포 피처는 target 예측에 실제 추가 이득이 있는지 반드시 시간 순 ablation으로 검증한다.

## 5. Trackman·생체역학 연구가 주는 실험 가설

### 5.1 릴리스 각도가 위치를 강하게 결정한다

다수 연구의 공통 결론은 공이 손을 떠나는 순간의 방향과 그 반복성이 최종 위치에 큰 영향을 준다는 것이다.

- [Kusafuka et al. 2020](https://www.frontiersin.org/journals/sports-and-active-living/articles/10.3389/fspor.2020.00036/full): 숙련 투수 7명, fastball 187구. 수직 위치에는 elevation angle·구속·spin axis, 수평 위치에는 azimuth angle·spin axis·수평 release point가 유의했고, 효과 크기는 투수마다 달랐다.
- [Nasu & Kashino 2021](https://doi.org/10.1080/02640414.2020.1868679): 투수 26명의 four-seam fastball에서 release projection angle의 영향이 가장 컸고 다른 릴리스 파라미터도 기여했다.
- [Moore et al. 2025](https://link.springer.com/article/10.1007/s12283-025-00497-5): NCAA 투수 9,476명, 2,215,013구. 같은 투수를 train/holdout에 섞지 않은 분할에서 DNN의 평균 위치 거리 오차가 0.154m로 선형모형 0.215m, 단순 물리모형 0.421m보다 낮았다. 물리 계산 피처를 빼면 DNN 오차가 17%, 약 2.8cm 증가했다.
- [Pitching kinematics direct/indirect effects 2025](https://www.thieme-connect.de/products/ejournals/html/10.1055/a-2468-5645): 실전 Kinatrax·Trackman 자료에서 수직/수평 release angle과 최종 위치의 매우 강한 관계를 보고했다.
- [Glanzer et al.](https://pubmed.ncbi.nlm.nih.gov/31449438/): 47명, 각 10구의 fastball에서 5개 투구 동작 변동성이 위치 일관성 분산의 58%를 설명했다.
- [Joseph et al. 2021](https://doi.org/10.36959/987/261): 프로 투수 322명의 소수 fastball 표본에서 elastic net·random forest로 동작 평균과 변동성을 분석했다.

대회 Trackman에는 가장 중요한 release angle이 없다. 따라서 현재 이용 가능한 `rel_height`, `rel_side`, `extension`, `rel_speed`, `spin_rate`, `induced_vert_break`, `horz_break`, `zone_speed`는 **불완전한 proxy**다. 이 한계 때문에 Trackman deep model부터 시작할 근거는 약하다.

권장 요약값:

- 중심: mean, median, trimmed mean
- 산포: std, MAD, IQR, p90-p10, coefficient of variation
- 꼬리/품질: p05, p10, p90, p95, 결측률, 이상치율
- 관계: `rel_height × rel_side` covariance, 속도-무브먼트·속도-release point correlation
- 시간: 최근 시즌, 최근 2시즌, 전체 기간, 최근-전체 차이, 가중 선형 slope
- 구종: pitch type group별 값, 투수 전체로 수축한 값, 그룹 간 거리
- 형태: arm-slot proxy `atan2(rel_side, rel_height)`, 공분산 고유값, GMM component 수·entropy

### 5.2 평균보다 반복성과 여러 타깃이 중요하다

[Kirby Index 분석](https://blogs.fangraphs.com/introducing-the-kirby-index-a-new-way-to-quantify-command/)과 [공개 노트북](https://github.com/michaelrosen3/kirby_index)은 release height/side와 release angle의 표준편차로 four-seam 반복성을 측정했다. 기사에서는 release angle을 넣었을 때 위치 예측 R²와 연도 간 안정성이 크게 높아졌다고 보고한다.

하지만 저자가 직접 밝힌 가장 큰 한계는 단일 타깃 가정이다. 여러 코스를 의도한 투수는 완벽히 실행해도 분산이 커 보인다. 따라서 다음 순서로 비교한다.

1. 단순 표준편차
2. 구종군·타자 손·시즌을 나눈 표준편차
3. 한 개와 여러 개 GMM의 검증 likelihood/BIC
4. 전체 산포가 아니라 component 내부 산포의 가중 평균

Kirby Index는 동료평가 논문이 아닌 practitioner 분석이며 MLB four-seam 결과다. 방향성 근거로만 사용한다.

### 5.3 순차 보정 능력

[Kusafuka et al. 2025](https://www.nature.com/articles/s41598-025-97146-5)는 숙련 투수 14명이 30구씩 던진 release angle의 lag-1 자기상관과 2차원 상태 전이를 분석했다. 수평 방향에서 산포가 큰 투수는 적절한 trial-by-trial 보정이 적은 경향을 보였다.

현재 평가 행에는 직전 투구 Trackman이 없고 다른 test 행을 사용할 수 없으므로 이 효과를 실시간 피처로 만들 수 없다. 다만 과거 Trackman에서 투수별 고정 signature로 아래를 만들 수 있다.

- 같은 경기·구종군 내 `rel_height`, `rel_side`, 속도, 무브먼트의 lag-1 autocorrelation
- 절대 1-step 변화량의 중앙값
- 경기 초반→후반 slope와 산포 변화
- 사분면/cluster 상태 전이 entropy

이는 작은 실험실 연구를 멀리 외삽하는 것이므로 낮은 우선순위로 둔다.

## 6. 선수·상황 효과: 수축과 계층 모델

### 6.1 누적 성공률은 표본 수와 함께 봐야 한다

[Brown 2008](https://arxiv.org/abs/0803.3697)은 시즌 전반 타율로 후반 타율을 예측할 때 관측 타율을 그대로 쓰는 naive 방법이 가장 나빴고, empirical/hierarchical Bayes 수축이 더 좋았음을 보였다. [Jensen, McShane, Wyner 2009](https://arxiv.org/abs/0902.1360)도 선수와 시간 사이 정보를 공유하는 계층 모형을 held-out season에서 검증했다.

이 대회의 `asof_pitcher_success_rate`와 `asof_batter_success_rate`는 같은 이항 비율 구조다. 기본 변환은 다음과 같다.

```text
shrunk_rate = (n * observed_rate + k * prior_rate) / (n + k)
```

- `prior_rate`: 반드시 학습 cutoff 이전 자료로 구한 league 또는 `season × game_type × hand` prior
- `k`: inner temporal validation으로 선택하는 prior strength
- `n=0`: prior와 명시적 cold-start flag 사용
- 원본 rate, shrunk rate, `log1p(n)`, posterior variance를 모두 후보로 둔다.

추가 후보:

- 최근 1/3/5경기와 career의 차이 및 기울기
- recent rate도 표본 수가 없으므로 과격한 값을 clipping/수축한 버전
- success, reverse, middle, ball, strike의 log-ratio와 합 제약을 활용한 변환
- `rate × log1p(n)`, `rate × game_type`, `rate × hand matchup`

### 6.2 여러 행위자 효과를 분리한다

[Deshpande & Wyner 2017](https://arxiv.org/abs/1704.00823)의 hierarchical pitch-framing 모형은 위치·카운트뿐 아니라 투수, 타자, 포수, 심판 효과를 동시에 조정한다. [Sports Info Solutions의 10년 회고](https://www.sportsinfosolutions.com/2025/08/28/lessons-from-a-decade-of-strike-zone-runs-saved/)도 포수·심판·투수·타자 효과를 분리하고, 최근 2년 창과 타깃 검출 신뢰도를 사용한다.

이 대회에는 포수·심판 ID가 없지만 다음 계층은 있다.

```text
league/season/game_type
  └─ pitcher team / batter team
       └─ pitcher / batter
            └─ hand matchup / count / base state
```

권장 방식:

- 투수와 타자의 독립 smoothed encoding
- 팀 효과와 시즌별 roster drift
- `pitcher × batter` raw pair 평균 대신 저차원 interaction 또는 유사 선수 수축
- `pitcher_hand × batter_hand`, `balls × strikes`, count × pitcher prior
- 신규 ID는 손·팀·game_type/league prior로 fallback

[SEAM](https://arxiv.org/abs/2005.07742)은 희소한 투수-타자 matchup을 비슷한 synthetic 선수로 보강해 MSE를 줄이는 접근을 보였다. 이 결과도 raw pair encoding보다 선수별 표현과 shrinkage가 안전하다는 근거다.

### 6.3 ID를 쓰는 세 가지 후보

1. **CatBoost native categorical**  
   [CatBoost 논문](https://arxiv.org/abs/1706.09516)의 ordered statistics는 범주 target leakage를 줄이기 위해 설계됐다. 다만 임의 permutation 기반 ordered encoding이 시간 순 안전성을 자동 보장하지는 않는다. 학습/검증 시즌은 분리하고, validation target이 category statistic에 들어가지 않는지 직접 검사한다.

2. **시간 순 계층 target encoding**  
   각 외부 fold에서 train cutoff 안에서만 `pitcher`, `batter`, team, hand/count 조건부 통계를 만들고 nested smoothing한다. 최종 2025 추론용 통계는 2019~2024 train만 사용한다.

3. **저차원 player embedding/factorization**  
   [`batter|pitcher2vec`](https://assets-global.website-files.com/5f1af76ed86d6771ad48324b/5ff4ac5dbbab5b7d59e29438_Statistic-Free%20Talent%20Modeling%20With%20Neural%20Player%20Embeddings.pdf)은 선수 embedding으로 처음 보는 matchup을 일반화했다. 일반 방법론인 [entity embedding](https://arxiv.org/abs/1604.06737)도 후보지만, 이 대회의 신호가 작고 cold-start가 있으므로 CatBoost/계층 인코딩 뒤에 시험한다.

## 7. 카운트·구종 선택·시퀀스 연구

### 7.1 카운트는 잠재 의도를 바꾼다

[Computing an Optimal Pitching Strategy](https://arxiv.org/abs/2110.04321)는 타석을 count state의 확률 게임으로 보고, 의도한 위치→실제 위치 분포와 위치→결과 분포를 분리했다. 3-0 count에서 스트라이크 의도가 높다는 가정으로 투수별 control distribution을 추정하고, sparse한 선수·구종 조합은 신경망 표현으로 보강했다.

대회에서 직접 쓸 수 있는 부분은 다음과 같다.

- count는 단순 숫자가 아니라 12개 상태 범주로 사용한다.
- `balls_before × strikes_before × pitcher prior × hand matchup` interaction을 둔다.
- 현재 구종이 없으므로 구종군 확률을 잠재 변수로 주변화한다.

```text
P(success | x)
  = sum_g P(pitch_group=g | pre-pitch x, history)
          * P(success | g, pre-pitch x, history)
```

현재 구종을 hard classification해 넣으면 틀린 분류의 확신이 전파된다. soft mixture나 구종군별 역사 피처를 나란히 입력하는 편이 안전하다.

### 7.2 다음 구종 예측 연구의 교훈과 한계

관련 연구는 다음과 같다.

- [Bock 2015](https://www.mdpi.com/2075-4663/3/1/40): 선수별 pitch sequence 예측 가능성과 장기 성과의 관계를 분석했다.
- [Healey & Zhao 2017](https://journals.sagepub.com/doi/10.3233/JSA-170103): 연속 투구의 위치·구속·무브먼트 상관을 분석했다.
- [Hoang et al. 2015](https://doi.org/10.1007/978-3-319-25660-3_11): 투수와 count에 따라 feature를 고르는 next-pitch 분류.
- [Sidle & Tran 2018](https://journals.sagepub.com/doi/10.3233/JSA-170171): 개별 투수의 다음 구종 multiclass 예측.
- [Lee 2022](https://journals.sagepub.com/doi/10.3233/JSA-200559): 한 KBO 투수의 구종·위치 34개 joint class를 ensemble DNN으로 예측.
- [Attention LSTM 2022](https://doi.org/10.1109/ICMEW56448.2022.9859411): 선수별 시퀀스에서 다음 구종을 예측.
- [Neural Sabermetrics 2026](https://arxiv.org/abs/2602.07030): 10년 이상 play-by-play를 대규모 sequence model로 학습한 preprint.
- [Pitch-pattern motif 2026](https://arxiv.org/abs/2601.11904): 1,240만 구의 motif와 entropy를 분석했지만 단순 motif와 결과의 관계는 제한적이었다.

보고 accuracy는 클래스 불균형, dominant pitch, 특정 선수, 무작위 분할의 영향을 크게 받는다. 또한 이 대회 test에서는 다른 행과 시퀀스를 만들 수 없다. 따라서 LSTM/Transformer를 현재 투구의 직전 sequence에 적용하는 것은 불가능하다.

허용 가능한 낮은 우선순위 변환은 제공 Trackman만으로 만든 **고정된 과거 성향**이다.

- 투수별 구종군 entropy·집중도
- count/타자 손별 과거 구종군 분포
- 과거 전이행렬의 대각합·entropy·상위 전이 확률
- `pitch_of_pa`별 구종군 변화

이들은 2025 test 행끼리 계산하지 않고 2019~2024 train 자산으로 고정해야 한다.

## 8. KBO의 2024·2025 체제 변화

[KBO 공식 발표](https://www.koreabaseball.com/MediaNews/Notice/View.aspx?bdSe=9852)에 따르면 KBO 리그는 2024시즌부터 ABS를 도입했다. [KBO ABS 연구](https://arxiv.org/abs/2407.15779)는 2021~2024 데이터를 비교해 2024 존이 이전 사람 심판의 존보다 경계가 엄격하고, 더 직사각형이며, 수평으로 좁고 수직으로 다소 높았다고 분석했다. 특히 경계와 구종별 called-strike 패턴이 달라졌고 투수의 코스·구종 선택 변화도 관측했다.

또한 [KBO 2025 공식 규정 발표](https://www.koreabaseball.com/MediaNews/Notice/View.aspx?bdSe=10321)는 다음 변화를 명시한다.

- ABS 상·하단을 각각 신장의 0.6%p, 180cm 타자 기준 약 1cm 하향
- 존 크기는 유지하고 전체 위치만 아래로 이동
- 약 1.2%의 전체 투구 판정에 영향을 줄 것으로 예상
- 2025 KBO 리그에 피치클락 정식 도입

이는 train 2019~2024와 test 2025 사이에 실제 운영 체제 변화가 있음을 보여준다. 그러나 `control_success`는 called strike가 아니고, `game_type=F`의 의미도 공식 데이터 설명에 없다. 특히 F의 큰 변화는 2022→2023에 발생했으므로 2024 ABS로 설명할 수 없다.

따라서 외부 수치를 예측값에 직접 더하거나 빼지 않는다. 문헌은 다음 실험의 **검증 근거**로만 사용한다.

- 모든 기간 / 최근 4·3·2시즌 / exponential decay 학습 비교
- R/F 별 별도 intercept·trend와 공통 slope 비교
- 전역 모델 + 최근 체제 residual model
- 선형 연도 외삽과 최근 평균 수축의 혼합
- 2022, 2023, 2024를 각각 가상 미래로 둔 pseudo-forward 검증
- 2025에 없는 사후 외부 통계나 test 전체 분포는 사용하지 않음

트리 모델은 `season=2025`를 학습 범위 밖 수치로 받아도 선형 외삽하지 못한다. 따라서 HGB/CatBoost에는 명시적인 training-only base-rate trend 또는 Linear residual/blend를 함께 시험할 가치가 크다.

## 9. 확률 예측·Brier·calibration 연구

### 9.1 Brier를 reliability와 resolution으로 나눈다

[Murphy의 Brier decomposition](https://doi.org/10.1175/1520-0450(1973)012%3C0595:ANVPOT%3E2.0.CO;2)은 Brier Score를 불확실성, reliability, resolution으로 해석한다. 이 대회에서는 “Brier가 좋아졌다”만 기록하지 말고 다음을 함께 본다.

- 전체 Brier와 raw Brier skill
- reliability bin의 가중 제곱 오차
- 예측 확률의 분산과 resolution proxy
- R/F, 시즌, count, cold-start, `n` 구간별 calibration

확률의 proper scoring 원칙은 [Gneiting & Raftery 2007](https://sites.stat.washington.edu/raftery/Research/PDF/Gneiting2007jasa.pdf)을 참고한다.

### 9.2 보정은 독립 OOF 예측에서만

[Niculescu-Mizil & Caruana 2005](https://icml.cc/Conferences/2005/proceedings/papers/079_GoodProbabilities_NiculescuMizilCaruana.pdf)는 boosted model의 확률 왜곡과 Platt/isotonic 보정을 비교하고, calibration용 독립 예측이 필요함을 강조한다. Isotonic은 유연하지만 데이터가 적거나 분포가 바뀌면 더 쉽게 과적합한다.

[Beta calibration](https://proceedings.mlr.press/v54/kull17a.html)은 identity mapping을 포함하면서 logistic calibration보다 유연한 선택지다. 후보는 다음으로 제한한다.

- identity
- logit intercept only
- affine logit: `sigmoid(a * logit(p) + b)`
- beta calibration
- 강하게 regularized R/F별 intercept

현재 저장소 실험에서는 직전 한 시즌 calibration이 다음 refit 모델에 안정적으로 이전되지 않았다. 그러므로 여러 과거 validation season의 **동일 학습 프로토콜 OOF 예측**을 모아 calibrator를 학습하고, 가장 최근 한 season을 다시 외부 검증하는 nested 구조가 필요하다.

### 9.3 drift 아래 calibration은 쉽게 깨진다

[Pampari & Ermon 2020](https://arxiv.org/abs/2006.16405)은 작은 covariate shift만으로도 calibration이 깨질 수 있음을 보였다. 논문의 importance-weighting 방식은 target-domain 입력 분포를 사용하지만, 이 대회는 test 전체 분포를 보는 보정을 금지한다.

따라서 사용 가능한 대응은 학습 데이터 내부의 과거 drift를 모사하는 것뿐이다.

- 최근 fold를 더 크게 가중한 OOF calibrator
- 지나치게 유연한 isotonic 제외
- base model refit 전후 확률 분포가 같은지 확인
- fold 하나라도 raw skill이 음수가 되면 보정 후보 탈락

## 10. 공개 프로젝트의 재현 가능성 평가

| 프로젝트 | 공개 내용 | 대회 적용성 | 주의점 |
| --- | --- | --- | --- |
| [Pitching-Control-Metric](https://github.com/mattludwig6/Pitching-Control-Metric) | xCTRL GMM, sparse case 수축, simulation | 분포·GMM 구현 참고 | 현재 위치/구종 필요, pybaseball 외부 데이터 사용 불가 |
| [Kirby Index](https://github.com/michaelrosen3/kirby_index) | release angle 계산, 반복성, 안정화 분석 | 산포·연도 안정성 ablation 참고 | 단일 타깃, MLB four-seam, practitioner |
| [BaseballCV](https://github.com/BaseballCV/BaseballCV) | glove·ball·plate 검출, `CommandAnalyzer`, intended-target notebook | 타깃 검출 방법론 참고 | 대회에는 영상이 없고 외부 모델/데이터 입력 불가 |
| [Notre Dame CZR+ 소개](https://mendoza.nd.edu/analytics/pitchers-performance-data-in-baseball/) | 타깃 거리 + ball/strike RF 결합 | 제구를 거리와 위험도로 나누는 개념 | 연결된 GitHub가 현재 404라 완전 재현 불가 |
| [mlb_pred](https://github.com/ytszkyuta-max/mlb_pred) | XGB/CatBoost/TFT/stacking next-pitch 예측 | cross-year 하락과 범주 모델 참고 | 최근 비심사 프로젝트, lag 피처는 대회에 부적합 |
| [mlb-pitch-classification](https://github.com/dan-rock/mlb-pitch-classification) | 물리량 기반 구종 분류 | 구현 스타일 참고 | 목표·입력이 다르고 현재 물리량 금지 |
| [OpenBiomechanics](https://github.com/drivelineresearch/openbiomechanics) | 공개 투구 생체역학 데이터·코드 | 용어·피처 해석 참고 | 외부 데이터이므로 학습/피처 생성에 사용 금지 |
| [pybaseball](https://github.com/jldbc/pybaseball) | Statcast 수집 | 논문 재현 경로 확인 | 외부 데이터 사용 금지 |

공개 프로젝트의 leaderboard나 README 성능 주장은 논문 수준의 증거로 취급하지 않는다. 이 저장소에 코드를 복사하기 전 라이선스와 평가 규칙도 별도로 확인한다.

## 11. 우선순위별 실험 명세

### P0. 모든 후속 실험의 평가 기반

#### E00 — Brier 진단과 불변성 테스트

- 가설: 작은 전체 Brier 차이가 calibration 개선인지 특정 세그먼트 과적합인지 분리해야 안전한 모델 선택이 가능하다.
- 구현: 전체·R/F·count·hand matchup·cold-start·표본 수 구간 Brier, raw skill, 예측 평균, reliability table, Brier decomposition proxy 저장.
- 통계 안정성: 두 후보의 행별 squared-error 차이를 사용한 paired 비교와 시즌 내 연속 블록 bootstrap을 함께 기록.
- 규정 게이트: 한 행 단독/배치, test shuffle, duplicate 추가 시 동일 예측인지 검사.
- 산출물: 모든 실험이 같은 JSON/CSV schema를 사용.

#### E01 — nested rolling runner

- 외부 fold: `≤2021→2022`, `≤2022→2023`, `≤2023→2024`.
- 모델·피처·calibration 선택은 각 외부 validation target을 보지 않는 inner temporal split에서만 수행.
- 평균 Brier뿐 아니라 최악 fold raw skill과 2023/2024 F를 기록.
- final 2025 학습은 선택이 끝난 하나의 고정 recipe로 2019~2024를 refit.

#### E02 — Trackman ID 매핑 품질 감사

> **폐기됨 → `E02R`(§22).** 매핑이 결정적으로 풀렸으므로 confidence 등급·후보 수·perturbation 감사는 불필요하다. 자세한 내용은 §18.1.

- 매핑별 confidence, 후보 수, train 선수 coverage, 시즌별 일관성, 손·팀 일치율을 산출.
- high/medium/low confidence 경계를 validation 전에 고정.
- 매핑 불가 투수의 fallback과 `trackman_unmatched` flag를 만든다.
- 현재 투구의 Trackman 행을 1:1로 붙이지 않는지 자동 검사.

### P1. 가장 먼저 실행할 모델 실험

#### E10 — recency·regime-aware Linear/HGB

- 가설: 2019~2024 전체 동일 가중치가 2025 base rate와 F 체제를 과대예측한다.
- 비교군:
  - 전체 기간 동일 가중치
  - 최근 4/3/2시즌만
  - 반감기 0.5/1/2/3년 exponential weight
  - 전역 모델 + `season × game_type` residual/intercept
  - 전역 Linear + 최근 HGB residual, 또는 최근 Linear + 전역 HGB
- 중요한 상호작용: `season × game_type`, count × game_type, player prior × season, hand matchup × count.
- 주의: 2025 규정 수치를 수동 offset으로 넣지 않는다. trend 파라미터는 train label만으로 적합한다.
- 기대: **높음**. 현재 가장 큰 오차 원인이 시간 drift와 F 과대예측이다.

#### E11 — empirical-Bayes `asof_*` 수축

- 가설: 표본 수가 작은 rate의 과신을 줄이고 최근/장기 신호를 분리하면 Linear와 tree 모두 개선된다.
- `k` grid: 20, 50, 100, 250, 500, 1000과 data-driven inner-fold 추정.
- prior 계층: 전체 → game_type/season trend → pitcher/batter hand.
- 피처: 원본, shrunk rate, posterior variance, `log1p(n)`, recent-career delta, rate 신뢰도 interaction.
- ablation: pitcher만, batter만, reverse/middle 포함, recent 포함.
- 기대: **높음**. 데이터 구조와 야구 EB 연구가 직접 일치한다.

#### E12 — CatBoost native categorical

- 범주: pitcher/batter/team ID, 양손, count state, base state, inning band, top/bottom, game type, month/day.
- 수치: 원본 + E11 변환.
- grid는 깊이 5~8, learning rate, L2, random strength, bagging, class weight 없음부터 작게 시작.
- loss는 Logloss로 학습하되 model selection은 Brier.
- 3개 이상 seed와 시간/메모리·추론 10분 제한 측정.
- 범주 statistic에 validation season target이 들어가지 않는 단위 테스트 필수.
- 기대: **중상**. 강한 ID 효과와 비선형 상호작용을 포착할 수 있다.

#### E13 — 시간 순 계층 encoding + Linear/HGB

- 투수·타자·팀의 cutoff-only smoothed target statistics를 별도 생성.
- 조건부 통계는 `pitcher × hand`, `pitcher × count band`, `batter × pitcher hand`까지만 시작.
- raw batter-pitcher pair는 만들지 않거나 매우 강하게 수축.
- CatBoost와 신호가 겹치는지 ablation하고, 더 안정적인 쪽만 유지.
- 기대: **중상**, 구현 누출 위험도 높음.

### P2. Trackman 역사 피처

#### E20 — 강건한 분포 요약

- 단위: mapped pitcher × season/window × pitch type group.
- 기본 cutoff: validation/prediction 시즌보다 **엄격히 이전 시즌**의 Trackman만 사용. 메인 행의 정확한 경기 날짜를 안전하게 복원한 경우에만 같은 시즌의 과거 로그 사용을 별도 실험한다.
- 값: mean/median/std/MAD/IQR/p10/p90/missing rate/count.
- 파생: covariance ellipse, arm-slot proxy, speed-movement 관계, 최근-장기 delta와 slope.
- 수축: pitch group 표본이 작으면 pitcher 전체, 그다음 hand/league prior로 수축.
- 테스트: 전체 선수, high-confidence mapping만, matched 선수만의 paired 성능을 모두 기록.
- 기대: **중간**. 현재 Trackman 값이 아니라 장기 능력 proxy라는 한계가 있다.

#### E21 — 다봉성·component 내부 일관성

- 구종군별 1~3 component GMM 또는 가벼운 clustering.
- feature: 선택 component 수, BIC 개선, mixture entropy, 최소 component weight, component 내부 공분산 trace, component 간 거리.
- 표본 수 하한과 여러 seed 설정. 작은 그룹은 GMM을 적합하지 않고 단순 산포로 fallback.
- 기대: **중간 이하**. xCTRL의 핵심 가설이지만 plate location과 release angle이 없다.

#### E22 — 잠재 구종군 mixture

- Trackman에서 pitcher × count × batter hand의 구종군 확률을 cutoff-only로 추정.
- `asof_pitcher_*_rate`를 prior로 사용하고 sparse conditional distribution을 수축.
- 구종군별 물리 요약을 모두 입력하는 방식과 확률 가중 expected feature를 비교.
- hard predicted pitch type은 사용하지 않는다.
- 기대: **중간**. 현재 구종 금지 조건을 지키면서 intent 차이를 표현할 수 있다.

#### E23 — 과거 순차·피로 signature

- 같은 Trackman 경기 안에서 release/속도/무브먼트 lag-1 correlation, absolute step, pitch_no slope, inning별 산포를 요약.
- 2019~2024 고정 선수 특성으로만 사용.
- E20 이후 추가 이득이 없으면 즉시 중단.
- 기대: **낮음**.

### P3. 표현 학습과 앙상블

#### E30 — 얕은 player factorization/embedding

- 투수·타자 embedding + hand/count/game type + E11 수치 피처의 작은 MLP 또는 factorization machine.
- ID별 최소 빈도 아래는 unknown bucket; embedding dimension 4/8/16 비교.
- unseen matchup을 검증하기 위해 validation의 신규 선수/신규 pair 세그먼트를 별도 측정.
- 큰 Transformer보다 먼저 수행하고 3 seed 편차를 기록.
- 기대: **중간 이하**. raw ID 효과는 크지만 CatBoost보다 구현 비용이 높다.

#### E31 — 다양성 기반 제한 앙상블

- 후보: regime-aware Linear, HGB, CatBoost, Trackman model, embedding.
- fold별 prediction residual correlation과 Brier diversity를 먼저 본다.
- 가중치는 0~1, 합 1로 제한하고 한 모델 최소/최대 가중치 grid를 둔다.
- 평균만 최소화하지 않고 `mean Brier + worst-fold penalty`로 선택.
- 현재 90% Linear + 10% HGB를 고정 기준으로 둔다.
- 기대: **중상**, 단 개별 모델 개선 이후 수행.

### P4. 마지막 확률 보정

#### E40 — multi-fold OOF beta/affine calibration

- 같은 recipe로 생성된 과거 OOF prediction만 결합.
- identity, intercept, affine-logit, beta를 비교.
- calibrator parameter를 identity 쪽으로 regularize.
- refit 전후 prediction mean/std/quantile shift를 기록.
- 3/3 fold 양의 skill과 최악 fold 비악화가 아니면 적용하지 않는다.

#### E41 — segment intercept

- R/F 또는 cold-start별 intercept만 강하게 수축해 시험.
- 자유로운 segment isotonic은 사용하지 않는다.
- global calibrator보다 모든 핵심 fold에서 좋아야 채택.

### P5. 당장은 미룰 실험

- full sequence LSTM/Transformer: 현재 test sequence를 사용할 수 없다.
- FT-Transformer/대형 tabular deep model: 먼저 CatBoost와 작은 embedding의 이득을 확인한다.
- 외부 Statcast/공개 생체역학 데이터 사전학습: 대회 외부 데이터 금지와 충돌할 가능성이 높다.
- xCTRL 직접 복제: 현재 위치·구종이 없어 동일 metric이 아니다.
- 단일 시즌 isotonic/Platt: 이미 regime 전이에서 실패했다.
- 더 복잡한 motif/entropy: 최근 대규모 연구도 단순 성과 연관이 제한적이다.

## 12. 실험 채택·중단 기준

현재 안정적 기준선은 Linear 90% + HGB 10%, rolling 평균 Brier 약 `0.24793696`, 3개 fold 모두 양의 raw skill이다.

### 후보 보존 기준

- 코드와 피처가 행 독립성 테스트를 통과한다.
- rolling 3-fold 평균 Brier가 기준선보다 좋아지거나, 독립 앙상블 다양성이 명확하다.
- 2023과 2024 모두 raw skill이 양수다.
- R/F 중 한 구간의 큰 악화를 전체 평균 개선으로 숨기지 않는다.
- cold-start와 신규 선수에서 NaN·극단 확률·fallback 오류가 없다.
- 여러 seed가 있는 모델은 평균과 최악 seed를 함께 기록한다.

### champion 교체 권장 기준

- 평균 Brier가 기준선 대비 최소 `0.00010` 개선
- 3/3 fold 양의 raw skill 유지
- 최악 fold Brier가 기준선보다 의미 있게 악화되지 않음
- 2023·2024 F 과대예측이 줄거나 적어도 악화되지 않음
- 연속 블록 paired bootstrap에서 개선 방향이 대체로 일관됨
- 평가 서버 추론 시간·메모리·패키지 제한 통과

`0.00010`은 절대 법칙이 아니라 작은 validation noise에 대한 작업용 안전 마진이다. 여러 후보를 반복 비교할수록 winner's curse가 커지므로, 미세한 차이는 독립 앙상블 후보로만 보존하고 champion 교체에는 더 엄격하게 사용한다.

### 즉시 중단 기준

- validation target이나 미래 Trackman이 집계에 포함됨
- test 행 순서·배치 크기·다른 test 행에 따라 예측이 달라짐
- 무작위 CV에서만 좋고 시간 순 fold에서 일관되게 나쁨
- mapping confidence가 낮은 선수에서만 Trackman 이득이 발생
- 한 fold의 큰 악화를 다른 fold 하나가 상쇄
- calibration이 base model refit 뒤 확률을 이중 보정

## 13. 권장 실제 실행 순서

1. E00/E01 공통 평가기와 불변성 테스트 완성
2. E10 최근 창·decay·R/F 체제 실험
3. E11 empirical-Bayes `asof_*` 피처
4. E12 CatBoost와 E13 시간 순 계층 인코딩
5. E02 매핑 감사 후 E20 Trackman 분포 피처
6. E21 다봉성, E22 잠재 구종군 mixture ablation
7. 가장 안정적인 Linear/HGB/CatBoost/Trackman 후보로 E31 앙상블
8. 최종 base recipe를 고정한 뒤 E40/E41 calibration
9. 시간이 남고 추가 이득 근거가 있을 때 E30 embedding
10. E23·대형 sequence/deep model은 마지막

이 순서는 도메인 문헌의 화려한 모델보다 현재 EDA에서 확인된 실제 오차인 **시간 drift, F 체제 변화, 희소 선수율, 과신된 누적 비율**을 먼저 해결하도록 설계했다.

## 14. 제출 전 규칙 체크리스트

- [ ] 모델 입력은 제공 train/test의 현재 행과 2019~2024 제공 Trackman에서 학습한 고정 자산뿐이다.
- [ ] 현재 투구의 실제 위치·구종·결과·Trackman 값을 사용하지 않는다.
- [ ] 2025 외부 Trackman·Statcast·KBO 성적을 사용하지 않는다.
- [ ] test 내부 선수/팀/월 빈도, rolling, target encoding, distribution calibration이 없다.
- [ ] target encoding과 Trackman aggregate가 각 validation cutoff 이전 데이터만 사용한다.
- [ ] 한 행/배치/shuffle/duplicate 불변성 테스트를 통과한다.
- [ ] unknown pitcher/batter/team과 `n=0`이 안전하게 fallback된다.
- [ ] 모델 선택과 calibration이 외부 validation target을 재사용하지 않는다.
- [ ] Brier, raw skill, R/F, cold-start 지표와 추론 시간을 함께 저장한다.

## 15. 참고 문헌·자료 색인

### 직접 제구·위치·릴리스

- Ludwig, Brill, Wyner, 2025, [Separating Intent from Execution: xCTRL](https://arxiv.org/abs/2508.19184) — preprint, [code](https://github.com/mattludwig6/Pitching-Control-Metric)
- Shinya et al., 2017, [Pitching form determines probabilistic structure of errors in pitch location](https://doi.org/10.1080/02640414.2016.1258484)
- Kusafuka et al., 2020, [Influence of Release Parameters on Pitch Location](https://www.frontiersin.org/journals/sports-and-active-living/articles/10.3389/fspor.2020.00036/full)
- Nasu & Kashino, 2021, [Impact of each release parameter on pitch location](https://doi.org/10.1080/02640414.2020.1868679)
- Glanzer et al., 2019/2021, [Kinematic variability and pitch-location consistency](https://pubmed.ncbi.nlm.nih.gov/31449438/)
- Joseph et al., 2021, [Kinematic Models for Pitch Location Metrics](https://doi.org/10.36959/987/261)
- Moore et al., 2025, [Context-enhanced DNN for pitch location](https://link.springer.com/article/10.1007/s12283-025-00497-5)
- Kusafuka et al., 2025, [Two-dimensional trial-by-trial error correction](https://www.nature.com/articles/s41598-025-97146-5)
- [UCL reconstruction and COMMANDf/x target accuracy](https://pmc.ncbi.nlm.nih.gov/articles/PMC7747121/)
- Rosen, 2024, [Kirby Index](https://blogs.fangraphs.com/introducing-the-kirby-index-a-new-way-to-quantify-command/) — practitioner, [code](https://github.com/michaelrosen3/kirby_index)

### 의도·전략·선수 효과

- Douglas et al., 2021, [Computing an Optimal Pitching Strategy in a Baseball At-Bat](https://arxiv.org/abs/2110.04321)
- Deshpande & Wyner, 2017, [A Hierarchical Bayesian Model of Pitch Framing](https://arxiv.org/abs/1704.00823)
- Brown, 2008, [In-season prediction with empirical Bayes](https://arxiv.org/abs/0803.3697)
- Jensen, McShane, Wyner, 2009, [Hierarchical Bayesian Modeling of Hitting Performance](https://arxiv.org/abs/0902.1360)
- Wapner, Dalpiaz, Eck, 2022, [SEAM matchup methodology](https://arxiv.org/abs/2005.07742)
- Sports Info Solutions, 2025, [A decade of Strike Zone Runs Saved](https://www.sportsinfosolutions.com/2025/08/28/lessons-from-a-decade-of-strike-zone-runs-saved/) — proprietary/practitioner
- Notre Dame, [Command Zone Rating Plus](https://mendoza.nd.edu/analytics/pitchers-performance-data-in-baseball/) — 연결 코드 현재 404

### 시퀀스·구종 선택

- Bock, 2015, [Pitch Sequence Complexity and Performance](https://www.mdpi.com/2075-4663/3/1/40)
- Healey & Zhao, 2017, [Pitch-to-pitch correlations](https://journals.sagepub.com/doi/10.3233/JSA-170103)
- Hoang et al., 2015, [Dynamic feature selection for pitch prediction](https://doi.org/10.1007/978-3-319-25660-3_11)
- Sidle & Tran, 2018, [Multiclass next-pitch prediction](https://journals.sagepub.com/doi/10.3233/JSA-170171)
- Lee, 2022, [Joint pitch type/location ensemble DNN](https://journals.sagepub.com/doi/10.3233/JSA-200559)
- Yu et al., 2022, [Attention-based next-pitch prediction](https://doi.org/10.1109/ICMEW56448.2022.9859411)
- [Neural Sabermetrics with World Model, 2026](https://arxiv.org/abs/2602.07030) — preprint
- [Pitch-pattern motifs, 2026](https://arxiv.org/abs/2601.11904) — preprint

### KBO 체제·확률 모델링

- KBO, [2024 ABS 도입 공식 발표](https://www.koreabaseball.com/MediaNews/Notice/View.aspx?bdSe=9852)
- Moon et al., [KBO ABS 영향 분석](https://arxiv.org/abs/2407.15779)
- KBO, [2025 ABS 존 하향·피치클락 공식 발표](https://www.koreabaseball.com/MediaNews/Notice/View.aspx?bdSe=10321)
- Niculescu-Mizil & Caruana, 2005, [Predicting Good Probabilities](https://icml.cc/Conferences/2005/proceedings/papers/079_GoodProbabilities_NiculescuMizilCaruana.pdf)
- Kull et al., 2017, [Beta calibration](https://proceedings.mlr.press/v54/kull17a.html)
- Pampari & Ermon, 2020, [Calibration under Covariate Shift](https://arxiv.org/abs/2006.16405)
- Gneiting & Raftery, 2007, [Strictly Proper Scoring Rules](https://sites.stat.washington.edu/raftery/Research/PDF/Gneiting2007jasa.pdf)
- Prokhorenkova et al., 2018, [CatBoost](https://arxiv.org/abs/1706.09516)

## 16. 추가로 확인했지만 직접 우선순위를 낮춘 자료

아래 자료도 조사했지만 현재 입력 제약, 작은 표본, 다른 목표 또는 검증 방식 때문에 핵심 실험의 직접 근거에서는 한 단계 낮췄다.

| 자료 | 확인한 내용 | 우선순위를 낮춘 이유 |
| --- | --- | --- |
| [Machine Learning Applications in Baseball: systematic review](https://www.tandfonline.com/doi/full/10.1080/08839514.2018.1442991) | 145개 자료를 선별해 32개 야구 ML 연구를 분류; next-pitch·성과·선수 평가 연구의 초기 지형 제공 | 2018년 이전의 넓은 야구 ML 리뷰이며 제구 성공 확률에 특화되지 않음 |
| [Individual factors associated with pitching performance: scoping review](https://pmc.ncbi.nlm.nih.gov/articles/PMC7047480/) | 투구 정확도를 포함한 개인 요인과 측정법을 체계적으로 정리 | 예측 대회보다 생체역학·성과 요인 전반의 리뷰 |
| [Using Sensors for Player Development](https://www.mdpi.com/1424-8220/22/21/8488) | 대학 투수 10명의 센서·Trackman과 주관적 command를 비교; 일부 가속도·vertical break 관계 보고 | 표본이 매우 작고 command가 주관 평가이며 현재 센서 값 없음 |
| [Pitching strategy via propensity-score stratification](https://arxiv.org/abs/2208.03492) | NPB 포수 요구 코스와 투구 선택을 propensity score로 조정 | causal 전략 평가가 목표이고 요구 위치·직전 결과가 대회에 없음 |
| [TruMedia catcher framing model](https://trumedianetworks.atlassian.net/wiki/spaces/Baseball/pages/186712078/Catcher+Framing+model) | 위치·타자 손·존·카운트, 시즌/레벨별 재학습과 공간 smoothing 사용 | proprietary called-strike 목표이며 현재 위치 없음 |
| [Analyzing Baseball Data with R: framing](https://beanumber.github.io/abdwr3e/07-framing.html) | GAM으로 위치별 called-strike 확률 표면을 재현 | 교육용 예제이고 현재 위치가 필수 |
| [The Quality of Pitches in MLB](https://www.sfu.ca/~tswartz/papers/pitching.pdf) | count·위치·구종·구속으로 투구 결과 가치를 random forest로 추정 | 현재 투구의 사후 위치·구종·구속을 사용하고 무작위 표본 검증 |
| [Strategic Pitch Location: two-pitch sequences](https://sabr.org/journal/article/strategic-pitch-location-the-role-of-two-pitch-sequences-in-pitching-success/) | 실제 위치 전이행렬과 투수 cluster를 기술 | 평가 행의 실제 이전 위치·현재 위치를 사용할 수 없음 |
| [MONEYBaRL](https://arxiv.org/abs/1407.8392) | count state에서 투구 선택을 MDP/RL로 모델링 | 정책 최적화 목표이며 이 대회 입력·Target과 다름 |
| [Counterfactual Optimization of Pitch Sequences, 2026](https://arxiv.org/abs/2606.17345) | Transformer로 setup/final pitch sequence를 바꾸는 반사실 분석 | 최신 preprint이고 실제 sequence·구종·결과가 필요 |
| [Precision: release-angle command metric](https://medium.com/iowabaseballmanagers/precision-using-release-angles-to-measure-command-8ad13e208226) | release angle 기반 command의 미래 안정성을 practitioner 분석 | 동료평가가 아니고 제공 Trackman에 release angle이 없음 |
| [BaseballCV Command Analyzer](https://github.com/BaseballCV/BaseballCV) | 영상에서 glove·ball·plate를 검출해 intended target과 miss를 계산 | 대회에 영상이 없고 외부 모델 입력 사용 불가 |

이 목록의 “낮은 우선순위”는 연구의 질이 낮다는 뜻이 아니라, **현재 대회의 합법적 입력으로 전환했을 때 기대 정보량이 작다**는 뜻이다.

## 17. 해석상의 마지막 주의

문헌에서 반복적으로 관측된 것은 “릴리스 일관성·개인별 타깃·계층 수축·카운트 맥락이 중요하다”는 사실이다. 이것이 제공 Trackman의 과거 산포가 곧바로 이 대회 Target을 잘 맞힌다는 뜻은 아니다. 제공 Target에는 위험한 가운데 공과 포수 요구 반대 방향도 포함되고, 진짜 요구 위치는 관측되지 않는다.

따라서 최종 판단 기준은 언제나 논문의 보고 수치가 아니라 이 저장소의 **누출 없는 rolling Brier와 최악 fold 안정성**이다.

---

# 제2부: 구조 복원 이후의 도메인 방법론

## 18. 구조 발견에 따른 로드맵 개정

[EDA 제2부](../eda/EDA_REPORT.md#제2부-데이터를-생성한-구조)의 결과가 §1~§17의 전제 네 가지를 바꿨다.

### 18.1 Trackman 연결은 확률 문제가 아니다 — `E02` 폐기

§11과 `E02`는 Trackman ID 대응을 confidence 점수·후보 수·매핑 불확실성이 있는 **추정 문제**로 다뤘다. 실제로는 경기 단위 투구 상태 지문으로 **정확히 풀린다**.

| 항목 | `E02`의 전제 | 실제 |
| --- | --- | --- |
| 매핑 성격 | 확률적, confidence 등급 필요 | **결정적, 1:1** |
| 투수 매핑 | 일부만 high-confidence | 792명 중 **731명**, 순도 평균 0.9984 |
| 커버리지 | 미지 | train 행의 **99.79%** |
| 손 코드 | 추정 | `1=Left`, `2=Right` **증명** |
| fallback | 다단계 신뢰도 게이트 | 미매핑 61명(행의 0.21%)용 단일 fallback |

`E02`의 산출물 중 confidence 경계, mapping perturbation sensitivity, 후보 수 분포는 만들 필요가 없다. **남는 것은 동결 매핑 사전 하나와 cutoff 계약뿐이다.** 개정판은 §22의 `E02R`이다.

이 변화의 실질적 의미는 Trackman 실험의 **기대값이 올라간 것이 아니라 비용이 내려간 것**이다. 물리량이 Target을 얼마나 설명하는지는 여전히 §20에서 보듯 제한적이다.

### 18.2 구종군 지도학습이 가능해졌다 — `E22` 승격

`E22`(잠재 구종군 mixture)는 "현재 구종을 알 수 없으니 `asof_pitcher_*_rate`를 prior로 쓰자"는 우회였다. 이제 정렬된 **810,644행에 실제 `pitch_type_group` 라벨**이 붙는다. 따라서

```text
1단계: (투구 직전 피처) -> P(pitch_type_group) 분류기를 810,644행으로 지도학습
2단계: P(success | x) = Σ_g P(g | x) · P(success | g, x)
```

가 실제 지도학습으로 구현된다. 구종군 효과는 fastball `54.37%` vs breaking `48.50%`로 `5.87%p`이며, 이는 카운트·주자·점수차 어떤 상황 피처보다 크다. `E22`를 **P2에서 P1로 올린다.**

단, 1단계 분류기의 입력에는 현재 투구의 Trackman 값이 들어가면 안 되고, 2단계에서 hard label을 쓰면 안 된다(§7.1의 원칙 그대로).

### 18.3 시즌 내 누적 복원이 최우선 실험이 됐다 — `E14` 신설

`asof_pitcher_n`이 시즌을 넘어 이어지는 통산 카운터이므로, **평가 행 하나만으로 그 투수의 2025 시즌 성적을 복원**할 수 있다. 단일 파생 피처로 2024에서 **492점**(정확한 prior 평균과 결합하면 **600점**)이며, 현재 챔피언 전체 파이프라인(409.7점)을 넘는다.

§13의 실행 순서에서 이 실험을 **E00/E01 다음, E10보다 앞**에 놓는다. 명세는 §22의 `E14`다.

### 18.4 `game_type` F의 정체가 밝혀졌다 — `E10`의 F 처리 구체화

F는 2군(퓨처스) 경기이고, 2023년 급변은 **동일 투수 70명에서 -22.1%p, 모든 2군 구장에서 동시에** 발생한 라벨 체제 변화다. §8이 "F의 큰 변화는 2022→2023에 발생했으므로 2024 ABS로 설명할 수 없다"고 열어 둔 질문에 부분적 답이 됐다. **원인은 여전히 미지지만 성격은 확정됐다: 실력·구성 변화가 아니라 라벨 생성 규칙의 변화다.**

따라서 `E10`의 F 처리는 "체제 변화를 모델이 흡수하는가"가 아니라 다음으로 바뀐다.

- 2022년 이전 F 행은 **다른 라벨 척도**이므로 F 파라미터 추정에서 제외하거나 별도 intercept를 준다.
- F 시계열의 선형 외삽은 무효다(최근 3시즌 선형 외삽 → 2025년 `0.2975`).
- 유효 F 표본은 2023·2024의 **5.6만 구**뿐이다. F 전용 모델은 표본이 얇으므로 전역 모델 + 강하게 수축한 F residual이 안전하다.

## 19. "진짜 실력" 추정: 야구 계량학의 표준 도구를 이 데이터로 검증

야구 계량학이 60년 동안 다듬어 온 핵심 문제는 이 대회의 문제와 정확히 같다. **표본이 유한한 관측 비율에서 다음 시행의 확률을 어떻게 추정하는가.** 이 절은 그 표준 도구 네 가지를 문헌으로 정리하고, **각각을 이 데이터에 직접 계산해** 어느 것이 살아남는지 본다.

### 19.1 회귀 상수는 격자 탐색이 아니라 분산 성분으로 구한다

기존 `E11`은 EB 상수 `k`를 `20, 50, 100, 250, 500, 1000` 격자로 탐색하라고 썼다. 야구 계량학의 표준은 격자가 아니라 **분산 성분 분해**다. [Efron & Morris (1975)](https://doi.org/10.1080/01621459.1975.10479864)가 James-Stein 추정량의 대표 예제로 1970년 타율을 사용한 이래, [The Book (Tango, Lichtman, Dolphin 2007)](http://www.insidethebook.com/) 계열에서 "ballast" 또는 "regression amount"로 정착한 방법이다.

관측 비율의 분산을 참 실력 분산과 이항 표본 분산으로 나눈다.

```text
Var(관측)  = Var(참 실력) + E[ p(1-p) / n ]
k          = p(1-p) / Var(참 실력)
```

R 경기에서 200구 이상 던진 투수만으로 계산한 결과:

| 시즌 | 투수 | 리그 평균 | Var(관측) | Var(이항) | **참 실력 SD** | **k** |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2022 | 202 | 0.5057 | 0.001940 | 0.000240 | **0.0412** | **147** |
| 2023 | 202 | 0.5043 | 0.002521 | 0.000237 | **0.0478** | **109** |
| 2024 | 219 | 0.4907 | 0.002288 | 0.000252 | **0.0451** | **123** |

세 시즌에서 `k ≈ 110~150`으로 안정적이고, 투수 간 참 실력 표준편차는 **약 4.5%p**다. 이 두 숫자가 이 문제의 신호 크기를 규정한다. 투수 간 참 SD가 4.5%p라는 것은, 완벽한 투수 정보를 가져도 예측 확률의 분산이 그 수준을 넘을 수 없다는 뜻이다(§22.2의 상한 646점과 정합적이다).

### 19.2 이론값과 실측 최적값이 어긋나는 이유 — 이 조사에서 가장 실용적인 발견

기존 실험에서 시즌 내 성공률의 최적 `k`는 **50**이었다. 이론값 `~120`과 두 배 이상 차이가 난다. 원인은 **prior 평균**이다.

2024를 평가 시즌으로 두고, prior 평균만 바꿔 가며 같은 `k` 격자를 돌린 결과:

| `k` | prior = 학습 평균 `0.5316` (낡음) | prior = 실제 2024 평균 `0.4861` |
| ---: | ---: | ---: |
| 25 | 477 | 524 |
| **50** | **492** ← 실측 최적 | 575 |
| **100** | 460 | **600** ← 최적 |
| 150 | 414 | 599 |
| 200 | 368 | 591 |
| 300 | 283 | 569 |
| 500 | 146 | 522 |

**prior 평균이 정확하면 최적 `k`가 이론값 `109~147` 구간으로 정확히 이동하고, 최고 점수는 492 → 600으로 오른다.**

해석은 명확하다. prior 평균이 3.2%p 높으면, 수축을 적게 해야(작은 `k`) 그 오차에 덜 끌려간다. 즉 **작은 `k`는 잘못된 평균에 대한 암묵적 보상**이다. 두 문제를 섞으면 둘 다 최적이 되지 않는다.

> **권고: `k`와 base rate를 분리해서 추정한다.**
> 1. `k`는 분산 성분으로 고정한다(`≈120`, 시즌 간 안정적).
> 2. prior 평균은 §22.3의 base-rate 외삽으로 따로 결정한다.
> 3. `k`를 격자 탐색으로 튜닝한다면, 그것이 평균 오차를 보상하고 있는 것은 아닌지 반드시 확인한다.

이 원칙은 `E11`, `E14`, `A10` 전부에 적용된다.

### 19.3 안정화 지점: control success는 야구 지표 중 매우 빨리 안정된다

[Spearman(1910)·Brown(1910)의 예언 공식](https://doi.org/10.1111/j.2044-8295.1910.tb00206.x)을 야구 지표에 적용해 "몇 번의 시행에서 지표가 신뢰할 만해지는가"를 구하는 것은 [Russell Carleton의 stabilization 연구](https://www.baseballprospectus.com/news/article/17659/baseball-therapy-its-a-small-sample-size-after-all/) 이래 표준 절차다. 관례적 기준은 신뢰도 0.5(신호와 잡음이 같아지는 지점)다.

2024 R 경기를 무작위 절반으로 나눠 split-half 상관을 구했다.

| 반쪽 표본 하한 | 투수 수 | split-half r | **Spearman-Brown 신뢰도** |
| ---: | ---: | ---: | ---: |
| 50구 | 244 | +0.617 | **0.763** |
| 100구 | 214 | +0.645 | 0.784 |
| 200구 | 175 | +0.723 | **0.839** |
| 400구 | 102 | +0.841 | 0.913 |
| 800구 | 35 | +0.905 | **0.950** |

**투수당 100구만 있어도 신뢰도가 0.76이다.** 야구에서 이 정도로 빨리 안정되는 지표는 드물다(참고로 타율은 신뢰도 0.5에 약 900타석이 필요하다고 알려져 있다). 이유는 명확하다. 제구 성공률은 타구 결과처럼 상대·수비·운이 개입하지 않고, 투수의 실행 자체만 측정하기 때문이다.

이것이 **§18.3의 시즌 내 복원 피처가 왜 그렇게 강한지 설명한다.** 2024 시즌 내 누적 표본의 중앙값은 506구이고, 위 표에 따르면 그 지점의 신뢰도는 0.9를 넘는다. 즉 시즌 내 성적은 이미 거의 잡음이 없는 실력 측정치다.

주의: 표의 하한을 올릴수록 r이 커지는 것은 표본 크기 효과와 **투수 선택 효과**(많이 던지는 투수는 실력 분포가 다르다)가 섞인 결과다. 절대 수준보다 "빠르게 안정된다"는 방향만 사용한다.

### 19.4 Marcel식 다시즌 가중은 이 데이터에서 실패한다

[Marcel the Monkey](http://www.tangotiger.net/marcel/)는 야구 예측 시스템의 표준 baseline이다. 최근 3시즌에 `5:4:3` 가중을 주고 출전량에 따라 리그 평균으로 회귀시킨다. ZiPS·Steamer 같은 실제 시스템도 이 골격을 공유한다. `E11`이 제안한 "최근 1/3/5경기와 누적 prior의 가중 조합"도 같은 발상이다.

2024를 평가 시즌으로 두고 비교했다.

| 투수 prior 구성 | 최적 EB `k` | 환산 점수 |
| --- | ---: | ---: |
| 전체 시즌 균등 (2019~2023) | 600 | 111 |
| **Marcel 5/4/3** (2023·2022·2021) | 2000 | 246 |
| **2023 한 시즌만** | 600 | **274** |

**최근 한 시즌만 쓰는 것이 Marcel 가중보다 낫다.** 그리고 전체 균등 가중은 절반 이하다.

이는 야구 계량학의 통념과 반대다. 원인은 이 데이터의 drift가 **실력 변화가 아니라 라벨 체제 변화**이기 때문이다([EDA §22.1](../eda/EDA_REPORT.md#221-드리프트는-선수-구성이-아니라-투수-내-변화다): 시즌 간 변화의 대부분이 within-pitcher 성분). 실력이 서서히 변하는 세계에서는 과거 시즌이 표본을 늘려 주지만, **측정 척도 자체가 바뀌는 세계에서는 과거 시즌이 편향을 늘린다.**

실무적 결론:

- 다시즌 가중을 쓰려면 **각 시즌을 그 시즌의 리그 평균으로 중심화한 뒤** 합쳐야 한다(척도 정렬).
- 중심화 없이 원 rate를 가중 평균하는 Marcel 형태는 이 대회에서 사용하지 않는다.
- 시즌 내 정보가 있으면 시즌 간 정보보다 항상 우선한다(§19.3의 신뢰도가 이를 뒷받침한다).

### 19.5 상태공간 모형: 시즌 내 복원의 원리적 일반화

§18.3의 시즌 내 누적은 "현재 시즌 시작 시점에서 리셋하는 단순 평균"이다. 더 일반적인 형태는 잠재 실력이 확률과정을 따르고 각 투구가 그 실력의 베르누이 관측이라는 **상태공간 모형**이다.

```text
θ_t = θ_{t-1} + w_t        (실력의 완만한 변화, w_t ~ N(0, σ²_process))
y_t ~ Bernoulli(σ(θ_t))    (투구 결과)
```

이 구조는 야구 밖에서 더 성숙해 있다. [Glickman(1999)의 Glicko](https://doi.org/10.1111/1467-9876.00159)는 체스 실력을 동적 paired comparison으로 추적하며 불확실성이 시간에 따라 커지는 것을 명시적으로 모형화한다. [TrueSkill(Herbrich et al. 2007)](https://papers.nips.cc/paper/2006/hash/f44ee263952e65b3610b8ba51229d1f9-Abstract.html)은 이를 베이지안 그래프 모형으로 확장했다. 야구 쪽에서는 [Jensen, McShane, Wyner(2009)](https://arxiv.org/abs/0902.1360)가 이미 §6.1에서 인용한 계층 모형에 시간 성분을 넣었다.

이 대회에 옮길 때의 이점과 제약:

| 이점 | 제약 |
| --- | --- |
| 시즌 경계에서 딱 잘리지 않고 부드럽게 잊는다 | 평가 행에서 쓸 수 있는 것은 **그 행이 들고 있는 누적값 두 개**뿐이다 |
| 최근 정보에 자동으로 더 큰 가중 | 투구별 시계열을 test에서 재구성할 수 없다(EDA §17.3) |
| 불확실성(사후 분산)을 피처로 낼 수 있다 | 따라서 완전한 필터링이 아니라 **근사 요약**만 가능 |

현실적 구현은 다음 두 가지다.

1. **감쇠 가중 누적의 근사**: 시즌 내 누적(`n_season`, `s_season`)과 통산 누적(`n`, `s`)의 두 창을, 신뢰도 가중 `n/(n+k)`로 결합한다. 이것이 상태공간의 1차 근사다.
2. **학습 데이터에서만 완전 필터를 돌려 정적 요약을 만든다**: 각 투수의 `σ²_process`(실력 변동성)를 train에서 추정해 **투수별 상수**로 동결하고, 평가 시점에는 그 상수를 shrinkage 강도에 반영한다. 변동이 큰 투수는 최근 정보에 더 크게 의존하게 된다.

2번이 이 대회에서 상태공간 모형을 쓸 수 있는 유일한 합법 경로다. 우선순위는 `E14`가 단순 형태로 이득을 확인한 다음이다.

### 19.6 aging curve는 이 문제에서 우선순위가 낮다

[Bradbury(2009)](https://doi.org/10.1080/02640410902829261)는 야구 선수의 연령별 성과 곡선을 다루며, [delta method 기반 aging curve](https://blogs.fangraphs.com/instagraphs/how-do-baseball-players-age-investigating-the-age-27-theory/)는 생존 편향 때문에 실제 노화를 과소 추정한다는 점이 반복 지적돼 왔다.

이 대회에는 **연령 정보가 없다**. `asof_pitcher_n`이 경력 대리 변수지만, [EDA §6.2](../eda/EDA_REPORT.md#62-경험량과-cold-start)가 보였듯 누적 표본과 시즌이 강하게 교란돼 있어 그대로 쓰면 시간 drift를 노화 효과로 오해한다. 연령 곡선 문헌은 **방법론적 경고**로만 인용하고 별도 실험을 만들지 않는다.

## 20. Stuff+ / Location+ / Command 계열과 이 데이터의 대응

### 20.1 산업 표준의 구조

현재 프로 야구 분석의 사실상 표준은 투구를 세 층으로 나누는 것이다.

| 지표군 | 입력 | 측정 대상 |
| --- | --- | --- |
| **Stuff+** | 구속, 무브먼트, 회전, 릴리스 위치·extension (위치 제외) | 공 자체의 위력 |
| **Location+** | plate location, 카운트, 타자 손 | 던진 곳의 가치 |
| **Pitching+** | 둘의 결합 | 종합 |

이 분해는 [Fangraphs의 Stuff+/Location+/Pitching+ 도입 문서](https://blogs.fangraphs.com/stuff-location-and-pitching-primer/)와 Cameron Grove의 PitchingBot, Driveline 계열 모델에서 공통으로 사용된다. 핵심 아이디어는 **"좋은 공"과 "좋은 위치"는 다른 능력이며 상관도 낮다**는 것이다.

### 20.2 이 대회는 Location+ 과제인데 Stuff 입력만 제공된다

| | Stuff+ | Location+ | 이 대회 |
| --- | :---: | :---: | :---: |
| 현재 구속·무브먼트·릴리스 | 사용 | 미사용 | **금지** (과거 집계만) |
| 현재 plate location | 미사용 | 사용 | **미제공** |
| 예측 대상 | 공의 위력 | 위치의 가치 | **의도 대비 실행 정확도** |

즉 이 대회의 Target은 Location+보다도 더 상위인 **command**(의도한 곳에 던지는 능력)인데, 입력으로는 Stuff 계열의 **과거 요약만** 쓸 수 있다. §4에서 정리한 "의도와 실행의 분리" 문제가 산업 지표 체계에서도 같은 방식으로 나타난다.

### 20.3 그렇다면 Stuff는 Command를 예측하는가 — 실측 결과

산업계의 통념은 "**Stuff와 Command는 서로 다른 능력이며, 강속구 투수일수록 제구가 나쁜 경향이 있다**"는 것이다. 이를 이 데이터에서 직접 확인했다([EDA §19.7](../eda/EDA_REPORT.md#197-trackman-물리량-부호가-반대인-두-신호)).

정렬된 R 경기에서 300구 이상 던진 투수 400명의 **투수 수준** 상관:

| 투수 수준 Trackman 요약 | 제구 성공률과의 r | 통념과 일치? |
| --- | ---: | :---: |
| 평균 구속 | **-0.171** | ✅ 강속구 ↔ 제구 난조 |
| 평균 회전수 | -0.127 | ✅ |
| 평균 수평 무브먼트 | +0.151 | — |
| 평균 릴리스 좌우 (arm slot) | +0.133 | — |
| 평균 extension | +0.088 | — |
| **구속 표준편차** | **-0.095** | ✅ 반복성 가설 |
| **릴리스 좌우 표준편차** | **-0.082** | ✅ 반복성 가설 |
| fastball 사용 비율 | -0.079 | — |

두 가지가 확인된다.

1. **Stuff와 Command의 음의 상관은 이 데이터에서도 성립한다.** 구속 `-0.171`은 제공된 어떤 `asof_*` 단변량 상관(최대 `+0.0843`)보다 크다.
2. **§5.2의 반복성(repeatability) 가설도 성립한다.** Kirby Index가 주장한 "릴리스 산포가 작을수록 command가 좋다"가 구속 SD `-0.095`, 릴리스 좌우 SD `-0.082`로 재현된다. 제공 Trackman에 release angle이 없다는 §5.1의 한계에도 불구하고 방향은 살아남았다.

### 20.4 그러나 pooled 상관을 그대로 쓰면 부호가 뒤집힌다

같은 물리량을 **투구 단위로** 보면 결론이 반대가 된다.

| 물리량 | pooled r | **투수 간 r** | **투구 내 r** |
| --- | ---: | ---: | ---: |
| `rel_speed` | +0.0445 | **-0.1719** | **+0.0563** |
| `zone_speed` | +0.0439 | -0.1672 | +0.0562 |
| `spin_rate` | -0.0014 | -0.1259 | +0.0045 |

pooled 상관 `+0.044`는 부호가 투수 간 성분과 **반대**다. 전형적인 Simpson's paradox이며, 원인은 두 효과가 섞여 있기 때문이다.

- **투수 간**: 강속구 투수는 제구가 나쁘다 (음)
- **투구 내**: 같은 투수가 던진 공 중 빠른 것은 대체로 fastball이고, fastball은 제구 성공률이 높다 (양)

평가 시점에 쓸 수 있는 것은 **투수 간 성분뿐**이다(과거 집계). 따라서 pooled 상관으로 피처 방향을 정하면 정확히 틀린 부호를 학습한다.

> **원칙: 모든 Trackman 파생 피처는 설계 전에 between/within 분해를 먼저 본다.** 이 진단의 일반 이론적 배경(패널 데이터의 Mundlak 장치, 다수준 모형의 group-mean centering)은 [유사 도메인 문서 §21](ANALOGOUS_DATA_METHODS_AND_EXPERIMENTS.md)에서 다룬다.

### 20.5 실무적 결론

- Trackman 물리량은 **투수 수준 프로파일로만** 만든다. 투구 단위 값은 어차피 금지이고 방향도 다르다.
- 가장 유망한 소수 후보를 먼저 본다: 평균 구속, 구속 SD, 평균 수평 무브먼트, 평균 릴리스 좌우, 릴리스 좌우 SD, fastball 비율. §5.1이 나열한 30종 이상의 요약값을 한꺼번에 넣는 것보다 낫다.
- **가장 큰 기대 이득 구간은 저표본 투수다.** §19.3에서 보듯 제구 성공률 자체는 100구면 신뢰도 0.76에 도달하므로, 표본이 충분한 투수에게 Trackman을 더해도 이득이 작을 가능성이 높다. 반대로 `n`이 작은 투수에게는 구속·무브먼트가 훨씬 안정적인 prior다. **`asof_pitcher_n` 구간별 paired 비교가 이 실험의 주 평가 지표여야 한다.**

## 21. 행위자·문맥 효과: 혼합모형 계열과 이 데이터에서 실제로 큰 것

### 21.1 DRA/framing 계열의 교훈

[Baseball Prospectus의 DRA(Deserved Run Average)](https://www.baseballprospectus.com/news/article/26195/prospectus-feature-introducing-deserved-run-average-dra-and-all-its-friends/)와 [Mixed 접근의 catcher framing 모형](https://www.baseballprospectus.com/news/article/25514/moving-beyond-wowy-a-mixed-approach-to-measuring-catcher-framing/)은 §6.2에서 인용한 Deshpande & Wyner와 같은 문제를 산업 규모로 푼다. 투수·타자·포수·심판·구장·상황을 **동시에 random effect로 두고 부분 풀링**한다.

이 대회에는 포수·심판 ID가 없지만, 구조 복원으로 **두 개의 행위자 축이 새로 생겼다**.

### 21.2 새로 쓸 수 있는 축 1: 투수 역할 (선발/구원)

학습 데이터에서 경기를 복원하면 각 투수의 등판당 투구 수와 등판 이닝을 알 수 있고, 이는 **투수별 정적 속성**이므로 동결 자산으로 만들어 평가 행에 붙일 수 있다(합법).

train ≤2023에서 정의(등판당 중앙 투구수 ≥50 & 중앙 진입 이닝 ≤2 → 선발):

| 역할 | 투수 수 | 2024 행 수 | 2024 성공률 |
| --- | ---: | ---: | ---: |
| 선발 | 138 | 74,415 | **50.54%** |
| 구원 | 411 | 102,516 | **47.68%** |
| swing | 5 | 1,276 | 46.00% |
| 미상(신규) | — | 45,290 | 49.40% |

**선발-구원 차이가 2.86%p**로, 카운트·주자·점수차 등 대부분의 상황 피처보다 크다.

다만 주의: 역할만으로 예측하면 환산 점수는 **0**이다. 이유는 §19.2와 같다 — train ≤2023의 역할별 평균이 낡은 수준이라 절대값이 틀린다. **역할은 상대 효과(offset)로 써야지 절대 수준으로 쓰면 안 된다.** 또한 투수 자신의 이력이 이미 역할을 상당 부분 흡수하므로, 순수 증분은 ablation으로만 확인된다. 특히 **신규 투수의 fallback prior**로서 가치가 클 가능성이 높다.

### 21.3 새로 쓸 수 있는 축 2: 구장

§19의 매칭으로 구장을 알 수 있고, 구장은 **홈 팀으로 완전히 결정**되므로 평가 행에서도 유도할 수 있다.

```text
home_team_id = (top_bottom == 'T') ? pitcher_team_id : batter_team_id
```

(`top_bottom=='T'`일 때 투수 팀이 홈이라는 관계는 [EDA §17.3](../eda/EDA_REPORT.md#173-파생되는-사실과-한계)에서 전 행 검증됐다.)

매칭된 R 경기의 구장별 성공률:

| 구장 | 행 수 | 성공률 |
| --- | ---: | ---: |
| Incheon | 78,131 | 50.54% |
| Jamsil | 175,076 | 50.56% |
| Gocheok | 82,248 | 51.23% |
| NCDinosMajors | 83,271 | 51.33% |
| Daejeon | 82,865 | 51.63% |
| Suwon | 82,651 | 51.89% |
| Sajik | 77,688 | 52.22% |
| DaeguPark | 79,447 | **52.77%** |

**최대-최소 차이 2.23%p, 가중 표준편차 0.74%p**다. 야구의 park factor 문헌이 다루는 크기와 비슷하다. 다만 이 차이는 순수 구장 효과가 아니라 홈 팀 투수진 구성과 교란돼 있다. 예측 목적에서는 구분할 필요가 없지만, **`home_team_id`는 이미 존재하는 세 컬럼의 조합이므로 트리 모델은 이미 학습 가능하고 선형 모델만 명시적 추가의 이득을 본다.**

### 21.4 무엇이 실제로 큰지 순서를 다시 매기면

[EDA §22.2](../eda/EDA_REPORT.md#222-정직한-성능-상한)의 split-half 상한과 위 결과를 합치면 이 도메인의 효과 크기 순서는 다음과 같다.

| 축 | 크기 | 실험 우선순위 |
| --- | --- | --- |
| **시즌 내 투수 성적** | 단독 492~600점 | **최상** (`E14`) |
| **투수 정체성 (시즌 내)** | 상한 646점 | 최상 |
| 투수 × 타자 손 | 상한 800점 | 상 |
| 구종군 (fastball vs breaking) | 5.87%p | 상 (`E22`) |
| 투수 간 Trackman 프로파일 | 구속 r=-0.17 | 중 (`E20R`) |
| 투수 역할 (선발/구원) | 2.86%p | 중 |
| 구장 / 홈 팀 | 2.23%p spread | 중하 |
| 카운트 | 3.45%p (0-1 ~ 3-2) | 중하 |
| 좌우 조합 | 4.66%p (P1 한정) | 중하 |
| 홈/원정 | **0.13%p** | **불필요** |
| **타자 정체성** | **상한 38점** | **불필요** |
| **투수 × 타자 raw pair** | 상한 287점 (투수 단독보다 나쁨) | **금지** |

§6.2가 제안한 "투수와 타자의 독립 smoothed encoding"에서 **타자 쪽은 사실상 버려도 된다**. §6.3의 embedding 후보(`E30`)도 타자 축이 비어 있으므로 기대값이 더 낮아진다.

## 22. 개정·추가 실험 명세

### E14 — 시즌 내 누적 복원 (신설, 최우선)

- **가설**: 각 평가 행이 들고 있는 통산 누적값에서 학습 시점의 고정 상수를 빼면, 2025 라벨 없이 2025 체제를 관측할 수 있다.
- **동결 자산**: `pitcher_id -> (n_end, s_end)` 사전. train 마지막 행의 `asof_pitcher_n + 1`, `round(rate × n) + control_success`.
- **파생 피처**:
  - `n_season = asof_pitcher_n - n_end` (시즌 내 투구 수, 시즌 진행도 대리 변수이기도 함)
  - `rate_season = (s_season + k·prior) / (n_season + k)`, `k` 고정 `≈120` (§19.1)
  - `rate_season - rate_career` (체제 변화의 개인 수준 신호)
  - `log1p(n_season)`, `n_season == 0` flag
  - 같은 방식의 `reverse` 버전(단독 225점). `middle`·`wide`·타자 버전은 0점이므로 만들지 않는다([EDA §21.5](../eda/EDA_REPORT.md#215-확장되지-않는-방향)).
- **prior 평균**: §22.3의 base-rate 외삽값을 사용한다. 학습 전체 평균을 쓰면 §19.2대로 `k`가 왜곡된다.
- **필수 검사**:
  - 단일 행/배치/셔플/중복 불변성 (이 피처는 통과해야 정상이다)
  - 2022·2023 fold에서도 재현되는가 (2024 단일 fold 결과다)
  - `n_season` 구간별, R/F별, 신규 투수별 Brier
  - 기존 `asof_*`와의 잔차 상관 — 통산 rate를 대체하는가 보완하는가
- **이중 보정 위험**: 이 피처는 base rate를 부분적으로 자동 교정한다. §22.3의 명시적 외삽과 함께 쓰면 과보정될 수 있다. 두 조합(피처만 / 외삽만 / 둘 다)을 반드시 비교한다.
- **기대**: **매우 높음.** 단일 피처로 현재 챔피언 전체를 넘었다.

### E15 — base rate 외삽 (신설, 최우선)

- **가설**: 2025 평균을 1%p 틀리면 40점, 2%p 틀리면 160점을 잃는다. 현재 모델은 2024를 `1.07~1.51%p` 과대예측했다.
- **후보**: 전체 6시즌 선형 / 최근 3시즌 선형 / 최근 2시즌 / 2024 실측 유지 / R·F 분리 외삽 후 비중 재결합.
- **F 처리**: 2022년 이전 제외. 선형 외삽 금지(2025년 `0.2975`가 나온다). 2023·2024 수준에서 출발.
- **검증 방법**: 2022·2023·2024를 각각 가상 미래로 두고 "그 시점까지의 데이터만으로 외삽했다면 얼마나 틀렸는가"를 측정한다. 이것이 2025 외삽 오차의 유일한 실증적 추정치다.
- **적용 위치**: 모델 출력의 사후 보정이 아니라 **EB prior 평균과 절편**에 넣는다. 사후 보정은 `E14`와 이중이 된다.
- **기대**: **매우 높음.** 모델 교체보다 기대 이득이 크다.

### E02R — Trackman 결정적 매핑 (`E02` 대체)

- 경기 지문 매칭 → 투구 단위 정렬 → 매핑 사전 추출. 구현은 `eda/run_structural_eda.py`에 이미 있다.
- 산출: `pitcher_id -> pitcher_trackman_id`(731), `batter_id -> batter_trackman_id`(780), `pitcher_team_id -> 팀`(13), 매칭 경기의 **실제 날짜** 2,700건.
- 남는 작업은 세 가지뿐이다.
  1. 미매핑 투수 61명(행의 0.21%)용 fallback: 손 × 팀 × 역할 prior.
  2. 팀 매핑의 시즌 의존성 처리(SK→SSG 개명).
  3. **cutoff 계약**: 매칭된 경기의 실제 날짜를 이용해 시즌 단위가 아닌 **날짜 단위 point-in-time** 집계를 만든다.
- 삭제: confidence 등급, posterior, entropy, top1-top2 margin, perturbation sensitivity.

### E20R — Trackman 투수 프로파일 (`E20` 축소)

- `E20`의 30종 이상 요약값 대신 §20.5의 **6개 후보**로 시작한다: 평균 구속, 구속 SD, 평균 수평 무브먼트, 평균 릴리스 좌우, 릴리스 좌우 SD, fastball 비율.
- 모든 요약은 구종군 내부에서 계산한 뒤 결합한다(릴리스 높이 SD의 부호 이상은 구종 혼합 오염이다).
- **주 평가 지표는 전체 Brier가 아니라 `asof_pitcher_n` 구간별 paired Brier**다. 저표본 투수에서 이득이 없으면 기각한다(§20.5).
- between/within 분해를 피처 설계 **전에** 기록한다.

### E22R — 구종군 지도학습 mixture (`E22` 승격)

- 1단계: 정렬된 810,644행으로 `P(pitch_type_group | 투구 직전 피처)` 분류기 학습. 입력에 현재 Trackman 값을 넣지 않는다.
- 2단계: `P(success|x) = Σ_g P(g|x)·P(success|g,x)`. hard label 금지.
- ablation: 구종군 확률 4개를 그냥 피처로 넣는 방식 vs 명시적 주변화.
- 누출 검사: 1단계 분류기의 학습 시즌과 2단계 평가 시즌을 분리한다.

### E16 — 역할·구장 등 복원된 문맥 (신설, 소규모)

- 투수 역할(선발/구원/미상)을 train에서 동결해 붙인다. **절대 수준이 아니라 offset으로** 사용한다(§21.2).
- `home_team_id`를 명시적 피처로 추가한다(선형 모델에서만 이득 예상).
- 두 피처 모두 투수 이력과 중복될 가능성이 높으므로 **ablation으로만 판단**하고, 신규 투수 세그먼트를 따로 본다.
- 기대: 낮음~중간. 비용이 매우 작아 먼저 확인할 가치는 있다.

### 우선순위 재배치

§13의 실행 순서를 다음으로 대체한다.

```text
1. E00/E01  공통 평가기와 불변성 테스트
2. E14      시즌 내 누적 복원          <- 신설, 최우선
3. E15      base rate 외삽             <- 신설, 최우선
4. E11      EB 수축 (k는 분산 성분으로 고정, 격자 탐색 아님)
5. E10      최근 창·decay·R/F 체제
6. E22R     구종군 지도학습 mixture    <- P2에서 승격
7. E12/E13  CatBoost·계층 인코딩
8. E02R     매핑(이미 완료) -> E20R Trackman 프로파일
9. E16      역할·구장
10. E31     제한 앙상블
11. E40/E41 calibration
12. E30 embedding, E21/E23은 마지막
```

`E02`, `E21`(다봉성 GMM), `E23`(순차 signature)은 우선순위를 더 낮춘다. `E21`은 plate location과 release angle이 없다는 §5.1의 한계에 더해, §19.3에서 제구 성공률 자체가 이미 빠르게 안정된다는 것이 확인되어 추가 이득 여지가 좁다.

## 23. 추가 참고 문헌

### 진짜 실력 추정·수축

- Efron & Morris, 1975, [Data Analysis Using Stein's Estimator and Its Generalizations](https://doi.org/10.1080/01621459.1975.10479864) — James-Stein의 대표 예제가 야구 타율이다
- Tango, Lichtman, Dolphin, 2007, [The Book: Playing the Percentages in Baseball](http://www.insidethebook.com/) — regression toward the mean과 ballast 상수
- Tango, [Marcel the Monkey Forecasting System](http://www.tangotiger.net/marcel/) — 예측 시스템의 표준 baseline
- Spearman, 1910 / Brown, 1910, [예언 공식](https://doi.org/10.1111/j.2044-8295.1910.tb00206.x) — split-half 신뢰도 보정
- Carleton, [It's a Small Sample Size After All](https://www.baseballprospectus.com/news/article/17659/baseball-therapy-its-a-small-sample-size-after-all/) — 야구 지표의 안정화 지점 (practitioner)
- Glickman, 1999, [Parameter Estimation in Large Dynamic Paired Comparison Experiments](https://doi.org/10.1111/1467-9876.00159) — Glicko, 동적 실력 추적
- Herbrich, Minka, Graepel, 2007, [TrueSkill](https://papers.nips.cc/paper/2006/hash/f44ee263952e65b3610b8ba51229d1f9-Abstract.html)
- Bradbury, 2009, [Peak athletic performance and ageing](https://doi.org/10.1080/02640410902829261) — aging curve와 생존 편향

### 행위자 분해·산업 표준 지표

- Judge, Pavlidis, Turkenkopf, 2015, [Introducing Deserved Run Average (DRA)](https://www.baseballprospectus.com/news/article/26195/prospectus-feature-introducing-deserved-run-average-dra-and-all-its-friends/) — 혼합모형으로 다중 행위자 분리 (practitioner)
- Judge, Pavlidis, Brooks, 2015, [Moving Beyond WOWY: A Mixed Approach to Measuring Catcher Framing](https://www.baseballprospectus.com/news/article/25514/moving-beyond-wowy-a-mixed-approach-to-measuring-catcher-framing/) (practitioner)
- Fangraphs, [Stuff+, Location+, Pitching+ primer](https://blogs.fangraphs.com/stuff-location-and-pitching-primer/) — Stuff와 Command의 분리 (practitioner)

practitioner 자료는 동료평가 논문이 아니므로 방향성 근거로만 사용하고, 채택 판단은 이 저장소의 rolling Brier로 한다.
