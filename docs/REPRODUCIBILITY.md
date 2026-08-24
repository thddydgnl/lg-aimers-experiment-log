# 재현 안내

## 환경

- Windows 11 / PowerShell 5.1
- Python 3.11.9
- 기준 패키지 버전은 [`requirements-baseline.txt`](../requirements-baseline.txt)와 [`LOCAL_ENVIRONMENT.md`](../LOCAL_ENVIRONMENT.md)를 참고한다.
- 추가로 사용한 주요 패키지는 LightGBM 4.7.0과 CatBoost 1.2.8이다.

가상환경 자체는 저장소에 포함하지 않는다.

```powershell
python -m venv .venv
& .\.venv\Scripts\python.exe -m pip install --upgrade pip
& .\.venv\Scripts\python.exe -m pip install --constraint requirements-baseline.txt "lightgbm==4.7.0" "catboost==1.2.8"
& .\.venv\Scripts\python.exe -m pip check
```

개별 제출 템플릿에는 별도 requirements가 있을 수 있으므로 해당 템플릿 파일을 우선한다.

## 데이터

대회 원본은 재배포하지 않는다. 공식 배포본을 내려받아 [`open/DATA_SETUP.md`](../open/DATA_SETUP.md)의 구조대로 `open/data/`에 둔다. 외부 데이터나 외부 API를 사용하지 않는다.

## 결과 확인

기존 실험을 다시 학습하지 않고 기록만 확인하려면 다음 파일을 사용한다.

- 논리적 실험 목록: `catalog/experiments.csv`
- 상세 판정: `experiments/EXPERIMENT_REGISTRY.md`
- 원시 경량 결과: `experiments/results/**/*.json`, `experiments/results/**/*.csv`
- 제출 이력: `submission/SUBMISSION_LOG.md`, `submission/CANDIDATE_REGISTRY.md`

## 실행 전 주의

- 현재 V5 목표는 완료되지 않았다.
- V2 파이프라인은 과거 재현용이며 V5 완료를 의미하지 않는다.
- 재실행 결과로 기존 결과를 덮어쓰지 않는다.
- 평가 행 간 집계, 미래 정보, 외부 데이터 사용을 금지한다.
- DACON 제출은 자동화하지 않는다.

상세 실행 명령은 [`docs/ORIGINAL_AGENT_GUIDE.md`](ORIGINAL_AGENT_GUIDE.md)와 각 계획서를 참고한다.

