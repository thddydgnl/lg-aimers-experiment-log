# Catalog

- `experiments.csv` / `experiments.json`: `EXPERIMENT_REGISTRY.md`에서 추출한 논리적 실험 115개. 원문 순서를 유지한다.
- `included_files.csv`: Git에 포함할 파일의 상대경로, 크기, SHA-256.
- `excluded_artifacts.csv`: 원본에는 있으나 이 경량 저장소에서 제외한 범주와 이유.
- `source_copy_audit.json`: 원본 선택 파일과 복사본의 존재·해시 비교 결과.
- `validation_report.json`: 금지 파일, 대용량, 비밀정보, 개인 경로, Markdown 링크 검사 결과.

카탈로그는 탐색을 위한 인덱스다. 수치와 판정이 충돌하면 연결된 원시 JSON/CSV와 `experiments/EXPERIMENT_REGISTRY.md`를 우선한다.

