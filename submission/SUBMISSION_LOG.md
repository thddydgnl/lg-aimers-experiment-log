# 제출 이력

> ZIP 생성 시점의 해시와 로컬 검증 결과를 연결한다. 실제 DACON 업로드 후 `업로드 시각`, `실행 상태`, `LB 점수`를 추가한다.

중간 후보는 [`CANDIDATE_REGISTRY.md`](CANDIDATE_REGISTRY.md)의 고유 ID별 보관 폴더에 먼저 동결한다. GOAL 종료 후 이 대장에서 제출할 후보를 선택한다.

| ID | 후보 | 변경점 | 로컬 3-fold Brier | ZIP SHA-256 | 게이트 | LB 점수 |
| --- | --- | --- | ---: | --- | --- | ---: |
| S1 | 공식 동결 `rf.pkl` 원본 ZIP | 공식 환경·업로드 경로 앵커 | 해당 없음 | `b35241c921ac2e18ab485946ec2952f13b2c9b76656d871c34eb5dc2a220acb3` | **PASS** | **`549.5119345223`** |
| S2 | Linear 90% + HGB 10%, 2019~2024 전체 재학습 | 자체 코드 LB 앵커 | `0.24793696` | `de5ff1509bbcc6c61d7ab37dde28b01a75802ce44d5e54bc2b99db84af72d6f7` | **PASS** | **`527.6161010151`** |
| S3 | S2 + E14 시즌 내 투수 누적 복원 | `-0.00033352`, 3/3 개선 | `6de6fbadfcb7a6352de3fb44b4c20957291360ea319f7fe39c5720bd034e548a` | **PASS** | **`662.3418227385`** |
| S4 | S3 + E15 `r_recent3` prior | `-0.00037671`, 3/3 개선 | `2e224321ab99a904a55d00669a90bbfd56cc5c7fc40ecf00803069403a678478` | **PASS** | **`689.2244587204`** |
| S5 | S4 + E16 동결 역할·홈팀 문맥 | `+0.000000339`, 2/3 개선 | `c368810fa5be2fb19792d495010ef987280d4d26d4646b5ef21c90f144efaa95` | **PASS** | **`688.1692139081`** |
| S6 | S4 + E22R soft 구종군 확률 4개 | `-0.000004121`, 3/3 개선 | `4f9ef705f03648a19011c579684c15bb9c42e9021ce9a2bd91c5fb8ad4b6891b` | **PASS** | **`687.2564723096`** |
| S7 | S4 + E22R 명시적 구종군 주변화 | `-0.000003409`, 2/3 개선 | `6a59873de5222e4787234718c82f4f0b4df4fabb6d6932cdace848d8d57639df` | **PASS** | 미제출 |
| S8 | 최종 M3: S4 70% + S5 5% + S6 10% + S7 15% | `-0.000005657`, 2/3 개선 | `632f41fd46d0f4b61fcb017881f05fba94e5dd3fbd7dbc0deccf72823141ff5b` | **PASS** | **`689.3999289563`** |
| S9 | S4 모델 재학습 없음, Linear 50% + HGB 50% | 독립 rolling 점수 없음(민감도 후보) | `7a3e010c33a7b1a990103091c220d55b5d1a15f74cd6e2b4756c2fc9bbda8db1` | **PASS** | 미제출 |
| S10 | S4 모델 재학습 없음, Linear 20% + HGB 80% | 독립 rolling 점수 없음(민감도 후보) | `76cb1a78590f94adfc4f4d12f45fa99573182db13768df48935b0da40673eaf8` | **PASS** | 미제출 |
| S11 | S4 모델 재학습 없음, HGB 100% | 독립 rolling 점수 없음(민감도 후보) | `56e4a17e68761ffa1b2a28fd55dd5adb1d52235277f56677f1d2d70abfe2cf17` | **PASS** | **`879.8414124135`** |

> DACON 제출 ID/시각: S1 `52715` (2026-08-17 22:58:52), S2 `52721` (2026-08-17 23:02:02), S8 `54316` (2026-08-18 22:10:26), S4 `54319` (22:10:47), S6 `54320` (22:11:12), S3 `54321` (22:11:44), S5 `54323` (22:12:06). S7은 미제출.

## 2026-08-17 로컬 게이트 결과

| ID | ZIP 크기 | 샘플 예측 평균 | 불변성 최대 차이 | 245,789행 시간 | peak RAM | 상태 |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| S1 | 3.94MB | `0.47835429` | `1.665e-16` | `3.52초` | `0.54GiB` | 업로드 가능 |
| S2 | 0.48MB | `0.47667704` | `0.000e+00` | `5.25초` | `0.75GiB` | 업로드 가능 |
| S3 | 0.48MB | 게이트 보고서 참조 | `0.000e+00` | `7.46초` | `0.91GiB` | 제출 완료, LB `662.3418` |
| S4 | 0.48MB | 게이트 보고서 참조 | `0.000e+00` | `7.46초` | `0.92GiB` | 제출 완료, LB `689.2245` |
| S5 | 0.48MB | 게이트 보고서 참조 | `0.000e+00` | `13.41초` | `0.99GiB` | 제출 완료, LB `688.1692` |
| S6 | 0.52MB | 게이트 보고서 참조 | `0.000e+00` | `12.80초` | `0.92GiB` | 제출 완료, LB `687.2565` |
| S7 | 1.4MB | 게이트 보고서 참조 | `0.000e+00` | `27.08초` | `0.97GiB` | 미제출 |
| S8 | 1.6MB | 게이트 보고서 참조 | `0.000e+00` | `49.15초` | `1.14GiB` | 제출 완료, LB `689.3999` |

- S1 ZIP은 공식 원본과 바이트 단위로 동일하다.
- S2는 전체 재학습을 연속 두 번 실행해 두 번 모두 동일한 최종 ZIP SHA-256을 얻었다.
- S2 학습 데이터 SHA-256: `d2081186b458b49f60b082be480c273135833e15ba59a76d033af28bcf8763ff`
- S2 동결 모델 SHA-256: Linear `9ddafed6edfc040716677135a8429095e3880a391243589b06326375c68940b8`, HGB `514c0f6302c2f23a7c2037d111d7825657a79ff0c81e11459396b7d4c84a72e3`
- 상세 원본은 [`dist/S1_official_frozen_rf.verification.json`](dist/S1_official_frozen_rf.verification.json), [`dist/S2_linear90_hgb10.verification.json`](dist/S2_linear90_hgb10.verification.json), `records/*_build.json`에 있다.

## 2026-08-18 후보 전체 재검증

보존된 archive ZIP을 현재 Windows `.venv`에서 다시 실행했다. 모든 후보가 `PASSED`와 245,789행을 재현했고, 불변성 최대 차이는 S1 `1.665e-16`, S2~S8 `0`이었다. 상세 재검증 보고서는 각 `submission/archive/S*/S*_rerun.verification.json`이다.

| ID | SHA-256 확인 | 245,789행 시간 | peak RAM | 결과 |
| --- | --- | ---: | ---: | --- |
| S1 | `b35241c921ac…acb3` | 2.73초 | 0.54GiB | **PASSED**, delta `1.665e-16` |
| S2 | `de5ff1509bbc…d6f7` | 4.54초 | 0.72GiB | **PASSED** |
| S3 | `6de6fbadfcb7…e548a` | 7.41초 | 0.92GiB | **PASSED** |
| S4 | `2e224321ab99…8478` | 7.47초 | 0.91GiB | **PASSED** |
| S5 | `c368810fa5be…aa95` | 8.93초 | 0.99GiB | **PASSED** |
| S6 | `4f9ef705f036…891b` | 8.42초 | 0.92GiB | **PASSED** |
| S7 | `6a59873de522…39df` | 15.17초 | 0.96GiB | **PASSED** |
| S8 | `632f41fd46d0…ff5b` | 24.10초 | 1.17GiB | **PASSED** |

## DACON 실행 결과

- `2026-08-17` — S1 `549.5119345223`, S2 `527.6161010151` 서버 실행 성공.
- `2026-08-18` — S3 `662.3418227385`, S4 `689.2244587204`, S5 `688.1692139081`, S6 `687.2564723096`, S8 `689.3999289563` 제출 결과를 확인했다.
- 이 항목은 2026-08-18 당시 스냅샷이다. 2026-08-20 V3 제출 뒤 현재 최고는
  `V3_sparse_m3_1103`의 **`1090.9100565103`**이다.

## 2026-08-20 V2 준비 상태

- S9~S11은 `submission/archive/S9~S11/`에 ZIP·빌드 기록·검증 보고서·manifest를 독립 보존했다.
- 게이트 시간은 S9 `8.86초`, S10 `8.52초`, S11 `8.52초`; 세 후보 모두 불변성 차이 `0`이다.
- 이 세 후보의 순서는 S4 모델 가중치 민감도 가설에 따른 것이며, 각 가중치의 독립 rolling
  점수는 없다. 따라서 문서와 제출 큐에서 보간 점수를 실제 측정값처럼 표기하지 않는다.
- V2 당시 LightGBM `4.5.0`·CatBoost `1.2.8` 설치 뒤 `pip check`와 S4 동일 SHA
  재검증을 통과했다. 이후 sklearn 1.8 래퍼 불일치는 LightGBM `4.7.0` 업그레이드와
  Windows import 순서 고정으로 수정했고 프로젝트 smoke test를 통과했다.

## 2026-08-20 V2 전체 파이프라인 완료 — 일부 제출 완료

17개 단계가 모두 `done`으로 끝났고, 신규 V2 ZIP 6개는 각각 245,789행 모사와
불변성·구조·SHA-256 게이트를 통과했다. 아래 점수는 2024 **개발 fold의 탐색적 값**이며
DACON LB 점수가 아니다. shift 후보에는 무보정 앙상블 점수를 재사용해 표기하지 않는다.

| 후보 | 구성 / 2024 개발 fold | ZIP SHA-256 | 245,789행 | peak RAM | 게이트 | LB 점수 |
| --- | --- | --- | ---: | ---: | --- | ---: |
| V_ensemble | HGB 0.75 + Linear 0.15 + CatBoost 0.10 / `696.5` | `5d995abcc0e802930d82d7cd5d6948208da83083ffbf0482475d4e8f4c6ce57a` | 9.97초 | 0.94GiB | **PASS** | **`906.8719072396`** |
| V_base | HGB / `681.9` | `2003aded8c1c8c24249ccff69879b67dbcd165ad0c4ab4327afc26beb3c965fb` | 5.10초 | 0.72GiB | **PASS** | **`879.8414124135`** |
| V_linear_tuned | Linear / `499.3` | `57a6ce31f75521559ae9d8b12778ccafae5ac21f34e95c6869466d2bee7c3686` | 3.63초 | 0.89GiB | **PASS** | pending |
| V_catboost | CatBoost / `33.4` | `8bae80a49fd183600eeb9ddf5be41ae73bde1a801c2b3b151675c702480d7bf6` | 5.56초 | 0.76GiB | **PASS** | pending |
| V_ensemble_shiftm0.032 | 무보정 ensemble + logit `-0.032` / 미할당 | `296e95fd6c75b5ec3849aedeb012fa3d7557663831590392feca76b9ded9f7f9` | 9.20초 | 0.89GiB | **PASS** | pending |
| V_ensemble_shiftm0.064 | 무보정 ensemble + logit `-0.064` / 미할당 | `bd17f046b6a7d8ed89ef9e280dfeb5f1ba81bd54dbfeabfcca2c0fc990050559` | 10.06초 | 0.94GiB | **PASS** | pending |

제출 순서와 기존 후보를 포함한 전체 게이트 표는 [`SUBMIT_QUEUE.md`](SUBMIT_QUEUE.md),
패키지 근거는 [`records/v2_package_index.json`](records/v2_package_index.json)에 있다.

## 2026-08-20 V3 예상 1,100 GOAL — 제출 결과

`예상 LB` 열은 계획 시작 전에 S4·S5·S6·S8로 고정한 환산식의 값이고, 마지막 `LB` 열은
실제 서버 제출 결과다. 2024는 반복 탐색에 사용했으므로 paired bootstrap은 탐색적으로 해석한다.

| 후보 | 구성 / 로컬 점수 | 예상 LB | ZIP SHA-256 | 245,789행 / peak | 게이트 | LB |
| --- | --- | ---: | --- | ---: | --- | ---: |
| `V3_sparse_m3_1103` | A `0.501444` + C `0.270160` + B `0.228396`; 2022 `2445.2773`, 2024 `963.5501` | `1103.6977` (`1101.5328~1108.9599`) | `b62f43c49a9093a60610200d0ee9bdd1afbe7a3eac506dd71a9706585d522bad` | 6.29초 / 0.85GiB | **PASS** | **`1090.9100565103`** |
| `V3_sparse_m2_1100` | A `0.629362` + B `0.370638`; 2022 `2440.2546`, 2024 `960.5052` | `1100.6527` (`1098.4878~1105.9149`) | `0c8826b6181403d365a7a14b8309e656b2be4bde6d39b6970d93de02065f8e27` | 5.19초 / 0.83GiB | **PASS** | **`1088.5196116458`** |

- 두 ZIP 모두 샘플·셔플·중복·단일행 불변성 최대 차이 `0`, runtime feature parity `0`이다.
- M3의 V2 ensemble 대비 2024 paired Brier delta는 `-0.00066703`, 95% CI
  `[-0.00080442,-0.00053854]`; 2022도 유의하게 개선했다.
- GPU CatBoost 독립 재학습은 98,340행에서 최대 `1.045e-8`의 예측 차이가 있어 완전한
  모델 바이트 동일성을 주장하지 않는다. 제출 대상 최종 ZIP은 위 SHA-256으로 동결했다.
- 근거: [`../experiments/results/v3_sparse_ensemble.json`](../experiments/results/v3_sparse_ensemble.json),
  [`records/v3_package_index.json`](records/v3_package_index.json), 각 `*.verification.json`.

제출 세부 기록:

| 파일 | DACON ID | 제출 시각 | 서버 실행 | 예상 대비 |
| --- | ---: | --- | ---: | ---: |
| `V3_sparse_m3_1103.zip` | `57386` | 2026-08-20 21:16:03 | 5초 | `-12.7876392` |
| `V3_sparse_m2_1100.zip` | `57388` | 2026-08-20 21:16:21 | 5초 | `-12.1331327` |
| `V_ensemble.zip` | `57391` | 2026-08-20 21:16:47 | 8초 | `+70.1943485` |
| `V_base.zip` | `57394` | 2026-08-20 21:18:49 | 3초 | `+57.8388749` |
| `S11.zip` | `57395` | 2026-08-20 21:19:11 | 6초 | 미산정 |

현재 실제 LB 챔피언은 `V3_sparse_m3_1103`의 **`1090.9100565103`**이다.
스크린샷에서 전사한 기계 판독 기록은
[`records/leaderboard_2026-08-20.json`](records/leaderboard_2026-08-20.json)에 보존했다.

## 2026-08-21 V4 예상 1,190 GOAL — 실제 전이 실패

아래 예상값은 당시 사용한 `2024 로컬 + 140.1475834416` 환산식이다. 이후 사용자가
실제 LB를 약 `1005`로 보고했으며 정확한 소수점·제출 ID·서버 시간은 추가 기록 대기다.

| 후보 | 구성 / 로컬 점수 | 고정 예상 LB | ZIP SHA-256 | 245,789행 / peak | 게이트 | LB |
| --- | --- | ---: | --- | ---: | --- | ---: |
| `V4_compact_supported_1193` | anchor 3 + student 1 + arm 18; 2022 `2413.7661`, 2024 **`1052.9440`** | **`1193.0915`** | `49708fe3e6a6b4f472e0771ba396f5eacb69ecf5a38d91406fc0049b628754b0` | 37.60초 / 1.50GiB | **PASS** | **약 `1005`** |

- 전체 22모델을 공식 2019~2024 train `1,475,092`행으로 재학습했다.
- research와 ZIP의 sample 최종 parity는 `2.22e-16`, 셔플·중복·단일행 불변성은 `0`이다.
- ZIP은 `26,677,480` bytes이고 동일 입력 재빌드 SHA가 일치했다.
- 딥러닝 계열도 별도 비교했으며 단독 최고 TabM `916.9032`, deep OOF stack 최고
  `1008.4905`, TabM 포함 supported meta stack `1052.0165`를 기록했다.
- 최종 배포본은 더 높은 `1052.9440`과 단순한 CPU 재현성을 가진 CatBoost 조합이다.
- 실제 결과는 이전 예상보다 약 `188.09점`, V3 챔피언보다 약 `85.91점` 낮았다.
- 18개 arm 중 2022와 2024에서 방향이 같은 것은 5개뿐이고, 2022에서 맞춘 계수는
  2024에서 V3 대비 `-66.39점`이었다. 패키지 오류가 아니라 검증·선택 실패로 판정한다.
- 기존 `+140.1476` 환산식은 폐기했으며 V5부터 실제 V3 앵커의 상대 개선 하한만 사용한다.
- 근거: [`records/v4_package_index.json`](records/v4_package_index.json),
  [`../experiments/results/v4_compact_supported_ensemble.json`](../experiments/results/v4_compact_supported_ensemble.json),
  [`dist/V4_compact_supported_1193.verification.json`](dist/V4_compact_supported_1193.verification.json).
- 실패 감사: [`../experiments/results/v4_failure_audit.json`](../experiments/results/v4_failure_audit.json),
  사용자 보고 기록: [`records/leaderboard_v4_user_report.json`](records/leaderboard_v4_user_report.json).

## 공통 실행 명령

```powershell
$env:PYTHONUTF8='1'
& .\.venv\Scripts\python.exe submission\build_submission.py --candidate all
& .\.venv\Scripts\python.exe submission\verify_submission.py submission\dist\S1_official_frozen_rf.zip
& .\.venv\Scripts\python.exe submission\verify_submission.py submission\dist\S2_linear90_hgb10.zip
```

## 기록 원칙

- S1은 [`../open/baseline_submit.zip`](../open/baseline_submit.zip)의 바이트 단위 복사본이어야 한다.
- S2 가중치는 Linear `0.9`, HGB `0.1`이며 calibration은 적용하지 않는다.
- 학습 데이터, 학습 코드, 각 모델 자산의 SHA-256은 `records/*_build.json`과 ZIP 내부 `model/manifest.json`에 기록한다.
- 검증 결과는 ZIP과 같은 이름의 `*.verification.json`에 기록한다.
- 실제 제출은 자동화하지 않는다. DACON 제출 탭에서 ZIP을 업로드한 뒤 서버 로그와 LB 점수를 이 문서에 수동으로 연결한다.
