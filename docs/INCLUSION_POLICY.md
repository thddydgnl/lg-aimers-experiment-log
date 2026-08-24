# 포함 및 제외 정책

이 저장소는 팀원이 실험 흐름과 근거를 검토하는 데 필요한 텍스트 중심 자료를 보존한다.

## 포함

- 자체 작성 Python/PowerShell/Shell 코드
- 실험 및 제출 설정 JSON/YAML/TOML
- 경량 결과 JSON/CSV
- EDA 보고서와 자체 연구 문서
- 계획서, 실험 대장, 제출 대장
- 재현 환경과 검증 스크립트

## 제외

| 범주 | 예시 | 이유 |
| --- | --- | --- |
| 대회 데이터 | `open/data/*.csv` | 재배포 금지 및 저장소 경량화 |
| 실행 환경 | `.venv/`, `__pycache__/` | requirements로 재생성 가능 |
| 대형 예측 | `*.npz`, `results/predictions/` | Git에 부적합, 원시 수치 요약은 JSON/CSV로 보존 |
| 학습 모델 | `*.pkl`, `*.joblib`, `*.cbm`, `*.ckpt` | 재학습 가능하고 용량이 큼 |
| 패키지 | `submission/dist/`, `*.zip` | 제출 ZIP 재배포 및 중복 방지 |
| 캐시·임시파일 | `_cache/`, `tmp/`, `catboost_info/` | 재생성 가능 |
| 외부 저장소 복제본 | `research/external_repos/` | 제3자 코드 중복·라이선스 위험 |
| 인증정보 | `.env`, `*.pem`, `*.key`, `kaggle.json` | 보안 |

제외된 대형 파일은 파일명과 해시를 모두 옮기는 대신 원본 작업 폴더에서 보존한다. 팀 공유에 꼭 필요한 대형 산출물은 별도 비공개 저장소나 객체 스토리지 사용을 검토한다.

