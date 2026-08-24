# LG Aimers experiment log

LG Aimers 야구 투구 제구 예측 과제에서 진행한 실험을 팀원이 빠르게 파악할 수 있도록 정리한 비공개 공유용 저장소다. 성공한 실험뿐 아니라 실패·기각·무효 처리된 실험도 함께 보존한다.

## 현재 결론

- 실제 리더보드 최고 기록은 **V3 Sparse M3 `1090.9100565103`**이다.
- V2의 최종 앙상블은 실제 LB `906.8719072396`이었다.
- V4 Compact Supported는 로컬 예상 `1193.0915411`과 달리 실제 LB가 약 `1005`로 하락했다. 이 실패로 기존 단일-fold offset과 불안정한 signed stack을 폐기했다.
- V5는 one-year-ahead honest anchor와 다중 시간축 검증 계약을 도입했다. 많은 후보가 source/development/confirmation gate에서 기각됐으며, 현재까지 `>1190` 완료 조건을 만족한 후보는 없다.
- 자동 제출은 하지 않는다. DACON 업로드와 실제 LB 기록은 사람이 수행한다.

## 읽는 순서

1. [`docs/TIMELINE.md`](docs/TIMELINE.md) — 전체 진행 순서와 버전별 전환점
2. [`docs/RESULTS_SUMMARY.md`](docs/RESULTS_SUMMARY.md) — 핵심 성과·실패·교훈
3. [`catalog/experiments.csv`](catalog/experiments.csv) — 논리적 실험 전체 목록
4. [`experiments/EXPERIMENT_REGISTRY.md`](experiments/EXPERIMENT_REGISTRY.md) — 상세 결과 원문 대장
5. [`docs/REPRODUCIBILITY.md`](docs/REPRODUCIBILITY.md) — 데이터 배치와 재현 방법

## 주요 제출 및 기준점

| 단계 | 후보 | 로컬/예상 | 실제 LB | 판단 |
| --- | --- | ---: | ---: | --- |
| V2 Track A | S11 / V_base | - | `879.8414124135` | 기준 제출 |
| V2 F1 | V_ensemble | 2024 `696.5` | `906.8719072396` | V2 최고 |
| V3 | Sparse M2 | 예상 `1100.6527` | `1088.5196116458` | 유효 개선 |
| V3 | Sparse M3 | 예상 `1103.6977` | **`1090.9100565103`** | 실제 LB 챔피언 |
| V4 | Compact Supported 18 | 예상 `1193.0915411` | 약 `1005` | 성능 실패, 재제출 금지 |
| V5 | honest validation program | 목표 하한 `>1190` | 미달/미제출 | 진행 기록 보존 |

## 저장소 구성

```text
catalog/       실험 및 파일 카탈로그
docs/          타임라인, 요약, 재현 문서, 원본 인수인계 문서
eda/           EDA 코드와 경량 결과
experiments/   실험 코드, 설정, JSON/CSV 결과
submission/    패키징·검증 코드와 제출 기록
research/      자체 작성 조사 문서와 외부 자료 목록
open/          데이터 설명과 로컬 배치 안내(원본 데이터 제외)
tools/         카탈로그 생성·검증 도구
```

## 포함하지 않는 항목

대회 원본 데이터, `.venv`, 캐시, 외부 저장소 복제본, NPZ 예측 배열, 학습 모델, 체크포인트, 제출 ZIP은 포함하지 않는다. 자세한 기준은 [`docs/INCLUSION_POLICY.md`](docs/INCLUSION_POLICY.md)를 참고한다.

## 빠른 확인

```powershell
python tools\build_catalog.py
python tools\build_manifest.py
python tools\validate_repo.py
```

모델을 다시 학습하기 전에는 [`COMPETITION.md`](COMPETITION.md), [`EXPERIMENT_PLAN_V5.md`](EXPERIMENT_PLAN_V5.md), [`docs/ORIGINAL_AGENT_GUIDE.md`](docs/ORIGINAL_AGENT_GUIDE.md)를 먼저 확인한다.

