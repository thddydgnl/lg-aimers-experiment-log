# V5 H1 OOF residual preregistration

`V5_H1_OOF_RESIDUAL_R_V1`의 결과 확인 전 고정 계약은
[`params/v5_h1_residual_preregister.json`](params/v5_h1_residual_preregister.json)에 있다.

2020·2021의 strictly-forward V3 OOF 오차만으로 2022 R 후보를 선택하고, 선택 결과를 파일로
잠근 뒤 2023 전이 감사와 2024 확인을 순서대로 수행한다. 현재 시즌 계층 posterior는 각 행의
공식 as-of 누적값과 해당 시즌 전 고정 상태만 사용한다. F는 2023 측정체계 단절 때문에 이번
실험 ID에서 교정하지 않고 V3 예측을 그대로 유지한다.

이 실험은 V4처럼 같은 fold 정답에 signed coefficient를 맞추지 않는다. 선택 가능한 모델 4개,
보정 강도 4개, 선택 기준과 tie-break까지 2022 결과를 열기 전에 고정했다.
