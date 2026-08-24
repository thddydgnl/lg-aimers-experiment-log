# 현재 로컬 개발 환경

> 확인 시각: **2026-08-21 KST**  
> 범위: 이 저장소를 실제로 실행하는 현재 Windows PC  
> 주의: DACON 평가 서버는 별도 Ubuntu 환경이다. 공식 서버 사양은 [`COMPETITION.md` §8.4](COMPETITION.md#84-평가-서버)와 아래 §6을 따른다.

## 1. 현재 PC 사양

| 항목 | 현재 값 |
| --- | --- |
| OS | Microsoft Windows 11 Pro 64-bit (`10.0.26200`) |
| 셸 | Windows PowerShell `5.1.26100.8875` |
| CPU | AMD Ryzen 5 5600, 6코어 12스레드 |
| RAM | 31.9GB |
| GPU | NVIDIA GeForce RTX 2070 SUPER, 8GB VRAM |
| NVIDIA 드라이버 | `610.88` (CUDA UMD `13.3`) |
| C: 여유 공간 | 약 320GB (확인 시점 값) |
| 로컬 Python | `3.11.9` |
| 프로젝트 가상환경 | `.venv\Scripts\python.exe` |
| Git | CLI 미설치, 현재 폴더도 Git 저장소로 초기화되지 않음 |

sklearn·pandas 베이스라인은 CPU를 사용한다. V3/V4 CatBoost 학습과 V4의 MLP·DeepFM·
TabTransformer·TabM·RealMLP 실험은 RTX 2070 SUPER를 사용했다. 최종 V4 제출 ZIP은
PyTorch 없이 CatBoost CPU 추론만 사용하며 245,789행을 로컬에서 `37.60초/1.50GiB`로
통과했다.

## 2. Python 환경

Windows용 Python 3.11은 사용자 범위에 설치되어 있다.

```text
<LOCALAPPDATA>\Programs\Python\Python311\python.exe
```

프로젝트 전용 `.venv`에는 [`requirements-baseline.txt`](requirements-baseline.txt)의 다음 버전이 설치되어 있고 `pip check`를 통과했다.

| 패키지 | 버전 |
| --- | ---: |
| numpy | 1.26.4 |
| pandas | 2.0.3 |
| scipy | 1.15.3 |
| scikit-learn | 1.8.0 |
| joblib | 1.5.3 |
| lightgbm | 4.7.0 |
| catboost | 1.2.8 |
| torch | 2.11.0+cu128 (딥러닝 실험 전용, 최종 ZIP 미포함) |

`torch.cuda.is_available()`은 `True`, CUDA runtime은 `12.8`, 장치는
`NVIDIA GeForce RTX 2070 SUPER`로 확인했다. `pip check`도 2026-08-21에 다시 통과했다.
평가 서버의 Python은 `3.11.15`지만, Windows 공식 설치본은 로컬에서 `3.11.9`를 사용한다.
최종 ZIP의 핵심 라이브러리 버전은 서버와 같고, 저장된 모델 로드·추론과 전체 재학습
재현을 이 조합에서 확인했다.

`python` 명령은 WindowsApps 실행 별칭을 가리킬 수 있으므로 문서와 자동화에서는 항상 `.venv`의 실행 파일을 명시한다. 가상환경 활성화는 필수가 아니다.

## 3. 새로 환경을 만드는 PowerShell 명령

프로젝트 루트에서 실행한다.

```powershell
winget install --id Python.Python.3.11 --exact --scope user `
  --silent --accept-package-agreements --accept-source-agreements

$Python311 = Join-Path $env:LOCALAPPDATA 'Programs\Python\Python311\python.exe'
& $Python311 -m venv .venv
& .\.venv\Scripts\python.exe -m pip install --disable-pip-version-check `
  -r requirements-baseline.txt
& .\.venv\Scripts\python.exe -m pip check
```

현재 셸에서 한글·이모지 출력까지 UTF-8로 고정하려면 다음을 먼저 실행한다.

```powershell
$env:PYTHONUTF8 = '1'
```

공식 `baseline_submit.zip`의 `script.py`는 결과 저장 후 `✅` 문자를 출력한다. Windows CP949 콘솔에서는 이 마지막 출력만 실패할 수 있으므로 공식 ZIP의 로컬 모사 실행에는 `PYTHONUTF8=1`을 사용한다. Ubuntu 평가 서버에서는 이 문제가 발생하지 않는다.

## 4. 표준 실행 명령

```powershell
$env:PYTHONUTF8 = '1'

# 제1부 스트리밍 EDA
& .\.venv\Scripts\python.exe eda\run_eda.py

# 제2부 구조 EDA
& .\.venv\Scripts\python.exe eda\run_structural_eda.py

# 2024 시간 순 검증과 검증용 모델 저장
& .\.venv\Scripts\python.exe experiments\run_baselines.py `
  --validation-season 2024 --save-models

# 나머지 rolling baseline folds
foreach ($Season in 2022, 2023) {
  & .\.venv\Scripts\python.exe experiments\run_baselines.py `
    --validation-season $Season
}

# Calibration·Ensemble 전체 재현
& .\.venv\Scripts\python.exe `
  experiments\run_temporal_calibration_ensemble.py `
  --validation-seasons 2022 2023 2024
```

위 명령은 기본적으로 기존 `eda/results`, `eda/figures`, `experiments/results`를 갱신한다. 기존 결과를 보존한 비교 실행은 실험 스크립트의 `--output-dir` 옵션을 임시 폴더로 지정한다.

## 5. 현재 PC 실측과 검증 상태

2026-08-17에 현재 PC에서 원본 파일을 바꾸지 않는 임시 출력 경로로 다시 측정했다.

| 작업 | 현재 PC 실측 | 비고 |
| --- | ---: | --- |
| 구조 EDA | `21.8초` | 결과 내용 완전 일치 |
| 제1·2부 EDA 전체 재생성 | `150.1초` | JSON은 OS 경로 구분자만 다르고, SVG는 줄바꿈 정규화 후 14/14 일치 |
| Baseline 2024 fold | 약 `105초` | 데이터 로드 포함, 6개 모델 |
| Baseline 3개 fold 전체 | `278.2초` | 18개 모델-fold 조합 |
| Calibration·Ensemble 전체 | `249.4초` | rolling 132행, aggregate 44행 |
| 실행 중 Python working set 관측 | 약 `1.7~1.8GB` | 계측 peak가 아니라 실행 중 점검값 |

2024 fold의 현재 PC 모델별 시간은 다음과 같다.

| 모델 | 학습 | 253,507행 예측 |
| --- | ---: | ---: |
| Linear | 19.1초 | 1.2초 |
| RF | 54.8초 | 0.6초 |
| HGB | 24.2초 | 1.5초 |

Linear와 HGB는 저장 지표를 수치 오차 수준으로 재현했다. RF는 OS·병렬 실행 차이로 fresh retrain Brier가 저장 결과와 최대 `9.4e-6`, 환산 점수가 최대 `1.28점` 달랐다. 저장된 RF 아티팩트를 다시 예측했을 때는 기존 Brier와 정확히 일치했다. 현재 V3 챔피언도 RF를 사용하지 않으므로 이 차이는 챔피언 재현에 영향을 주지 않는다.

추가 확인 사항:

- 공식 `open.zip` SHA-256: `389D7DD8FBC8F529AB345F2A15F0C83BAD9552EB0ABA2410F7B48A9A3339773F`
- 공식 ZIP 내부 6개 파일과 로컬 `open/` 파일: SHA-256 6/6 일치
- 저장 모델의 단일행·배치·셔플·중복 불변성: 차이 `0` 또는 부동소수점 오차 `1.7e-16` 이하
- 공식 baseline ZIP의 5행 샘플 추론과 결과 스키마: 통과
- `pip check`: 통과
- LightGBM `4.5.0`은 scikit-learn `1.8.0`의 `force_all_finite` API 변경과 맞지 않아
  프로젝트 래퍼가 실패했다. `4.7.0`으로 올리고 Windows DLL import 순서와 eval API를
  수정했다. `experiments/smoke_lgbm.py`의 일반 범주형 학습·시간순 early-stopping 및
  `pip check`를 통과했다.
- CatBoost `1.2.8`도 같은 constraint 아래 설치했고 다시 `pip check`와 S4 재검증을
  통과했다. 작은 rolling smoke fold에서 시간 순 early stopping·전체 history refit까지 확인했다.

## 6. 로컬 PC와 평가 서버를 혼동하지 않는다

| 항목 | 현재 로컬 PC | DACON 평가 서버 |
| --- | --- | --- |
| OS | Windows 11 Pro | Ubuntu 22.04.5 LTS |
| Python | 3.11.9 | 3.11.15 |
| CPU | Ryzen 5 5600, 6C/12T | 6 vCPU |
| RAM | 31.9GB | 28GB |
| GPU | RTX 2070 SUPER 8GB | NVIDIA L4 22.4GiB |
| CUDA | 로컬 드라이버 UMD 13.3 | 12.8 |
| 셸·경로 | PowerShell, `\` | Linux, `/` |
| 인터넷 | 사용 가능 | 패키지 설치 후 비활성화 |

문서의 개발·재현 명령은 Windows PowerShell 기준이다. `submit.zip` 안의 `script.py`는 평가 서버에서 실행되므로 OS에 독립적인 상대 경로와 Python 코드만 사용해야 한다.

## 7. 현재 산출물·미완료 항목

완료된 로컬 산출물:

- `submission/` 빌드·검증 하네스와 행 독립성 게이트를 구현·실행했다.
- S1~S11 ZIP, 각 후보의 `candidate_manifest.json`, 빌드 기록, 검증 보고서, SHA-256을 `submission/archive/S1~S11/`에 독립 보존했다.
- E14/E15/E16/E22R rolling 및 최종 M3 고정 앙상블 비교를 완료했다. E11/E10/E20R은 부정 결과도 ID별로 보존했다.
- S3~S11은 각 후보에 해당하는 245,789행 모사·불변성·시간/RAM 게이트를 통과했다.
  S3~S6·S8·S11은 실제 리더보드 제출까지 완료했고 S7·S9·S10은 미제출이다.
- V3 M3 `1090.9100565103`, M2 `1088.5196116458`, V ensemble `906.8719072396`,
  V base와 S11 `879.8414124135`의 2026-08-20 실제 LB 결과를 기록했다.
- V2 `preflight`·`probe`·`a_blend`는 무결성 검사 후 체크포인트에 채택했다. S9~S11은
  각각 다른 SHA의 ZIP과 증거 묶음으로 보존했으며 아직 LB 점수는 없다.
- V3 outcome CatBoost·공식 Trackman rich profile·역사 그룹률·고정 affine 보정·희소
  앙상블 실험을 완료했고 긍정·부정 JSON/CSV/NPZ를 모두 별도 보존했다.
- 최종 `V3_sparse_m3_1103.zip`과 `V3_sparse_m2_1100.zip`은 245,789행 모사에서 각각
  `6.29초/0.85GiB`, `5.19초/0.83GiB`였고 행 독립성 오차는 `0`이다.

실제 결과 해석:

- V3 두 ZIP의 실제 DACON LB를 기록했다. 예상 `1103.6977`, `1100.6527`보다 각각
  `12.7876`, `12.1331`점 낮았고, 현재 챔피언은 M3 `1090.9100565103`이다.
