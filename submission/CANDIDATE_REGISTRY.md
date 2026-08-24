# 후보 산출물 보관대장

GOAL 종료 후 제출 후보를 선택할 수 있도록 각 후보를 고유 ID별로 보존한다. 후보 ZIP은 한 번 보관되면 다른 ZIP으로 덮어쓰지 않는다.

| ID | ZIP SHA-256 | 로컬 지표 | LB 점수 | 보관 경로 | 상태 |
| --- | --- | --- | ---: | --- | --- |
| S11 | `56e4a17e68761ffa1b2a28fd55dd5adb1d52235277f56677f1d2d70abfe2cf17` | manifest-only blend sensitivity; no independent rolling score | `879.8414124135` | `submission\archive\S11\S11.zip` | 제출 완료 (`57395`) |
| S10 | `76cb1a78590f94adfc4f4d12f45fa99573182db13768df48935b0da40673eaf8` | manifest-only blend sensitivity; no independent rolling score | pending | `submission\archive\S10\S10.zip` | 보존 완료 |
| S9 | `7a3e010c33a7b1a990103091c220d55b5d1a15f74cd6e2b4756c2fc9bbda8db1` | manifest-only blend sensitivity; no independent rolling score | pending | `submission\archive\S9\S9.zip` | 보존 완료 |
| S8 | `632f41fd46d0f4b61fcb017881f05fba94e5dd3fbd7dbc0deccf72823141ff5b` | M3: S4=0.70; S5=0.05; S6=0.10; S7=0.15; mean_brier_delta=-0.0000056573; wins=2/3; worst=+0.0000217847; gate=PASS | 689.3999289563 | `submission\archive\S8\S8.zip` | 제출 완료 (54316) |
| S7 | `6a59873de5222e4787234718c82f4f0b4df4fabb6d6932cdace848d8d57639df` | e22r_mixture_mean_brier_delta=-0.0000034093; e22r_mixture_wins=2/3; worst_delta=+0.0001834161; gate=PASS | pending | `submission\archive\S7\S7.zip` | 미제출 |
| S6 | `4f9ef705f03648a19011c579684c15bb9c42e9021ce9a2bd91c5fb8ad4b6891b` | e22r_probs_mean_brier_delta=-0.0000041209; e22r_wins=3/3; worst_delta=-0.0000021380; gate=PASS | 687.2564723096 | `submission\archive\S6\S6.zip` | 제출 완료 (54320) |
| S5 | `c368810fa5be2fb19792d495010ef987280d4d26d4646b5ef21c90f144efaa95` | e16_mean_brier_delta=+0.0000003392; e16_wins=2/3; worst_delta=+0.0000110981; gate=PASS | 688.1692139081 | `submission\archive\S5\S5.zip` | 제출 완료 (54323) |
| S4 | `2e224321ab99a904a55d00669a90bbfd56cc5c7fc40ecf00803069403a678478` | e14+r_recent3_mean_brier_delta=-0.0003767058; e14_wins=3/3 | 689.2244587204 | `submission\archive\S4\S4.zip` | 제출 완료 (54319) |
| S3 | `6de6fbadfcb7a6352de3fb44b4c20957291360ea319f7fe39c5720bd034e548a` | e14_mean_brier_delta=-0.0003335192; e14_wins=3/3 | 662.3418227385 | `submission\archive\S3\S3.zip` | 제출 완료 (54321) |
| S1 | `b35241c921ac2e18ab485946ec2952f13b2c9b76656d871c34eb5dc2a220acb3` | 미기록 | 549.5119345223 | `submission\archive\S1\S1.zip` | 제출 완료 (52715) |
| S2 | `de5ff1509bbcc6c61d7ab37dde28b01a75802ce44d5e54bc2b99db84af72d6f7` | rolling_mean_brier=0.24793696 | 527.6161010151 | `submission\archive\S2\S2.zip` | 제출 완료 (52721) |

2026-08-18 archive ZIP S1~S8을 재검증했고, 2026-08-20 S9~S11을 추가로 독립 보존했다.
S9~S11은 모두 `PASSED`, 불변성 차이 `0`이며 각 ZIP의 현재 SHA와 검증 보고서 SHA가
일치한다. S11은 2026-08-20 제출해 `879.8414124135`를 기록했고, 아직 제출하지 않은
S7·S9·S10은 `pending`으로 유지한다. 제출 시각·DACON 제출 ID는 이 대장과 제출 이력에 연결한다.

## V2 GOAL 패키지 (2026-08-20)

V2 내부 후보명은 LB 제출 전 작업 ID이므로 아직 S 번호를 부여하지 않았다. ZIP은
`submission/dist/`, 빌드 근거는 `submission/records/`, 검증 보고서는 ZIP과 같은 이름으로
분리 보존한다. 실제 제출 후보로 확정할 때 고유 S 번호로 immutable archive에 옮긴다.

| 작업 ID | ZIP SHA-256 | 2024 개발 fold | 보관 경로 | 게이트 | LB |
| --- | --- | ---: | --- | --- | ---: |
| V_ensemble | `5d995abcc0e802930d82d7cd5d6948208da83083ffbf0482475d4e8f4c6ce57a` | `696.5` (exploratory) | `submission/dist/V_ensemble.zip` | **PASSED** | `906.8719072396` (`57391`) |
| V_base | `2003aded8c1c8c24249ccff69879b67dbcd165ad0c4ab4327afc26beb3c965fb` | `681.9` | `submission/dist/V_base.zip` | **PASSED** | `879.8414124135` (`57394`) |
| V_linear_tuned | `57a6ce31f75521559ae9d8b12778ccafae5ac21f34e95c6869466d2bee7c3686` | `499.3` | `submission/dist/V_linear_tuned.zip` | **PASSED** | pending |
| V_catboost | `8bae80a49fd183600eeb9ddf5be41ae73bde1a801c2b3b151675c702480d7bf6` | `33.4` | `submission/dist/V_catboost.zip` | **PASSED** | pending |
| V_ensemble_shiftm0.032 | `296e95fd6c75b5ec3849aedeb012fa3d7557663831590392feca76b9ded9f7f9` | 미할당 | `submission/dist/V_ensemble_shiftm0.032.zip` | **PASSED** | pending |
| V_ensemble_shiftm0.064 | `bd17f046b6a7d8ed89ef9e280dfeb5f1ba81bd54dbfeabfcca2c0fc990050559` | 미할당 | `submission/dist/V_ensemble_shiftm0.064.zip` | **PASSED** | pending |

전체 목록과 제출 순서는 [`SUBMIT_QUEUE.md`](SUBMIT_QUEUE.md), 사양·원천 점수는
[`records/v2_package_index.json`](records/v2_package_index.json)을 단일 근거로 사용한다.

## V3 예상 1,100 GOAL 패키지 (2026-08-20)

| 작업 ID | ZIP SHA-256 | 로컬 지표 | 예상 LB | 실제 LB | 보관 경로 | 상태 |
| --- | --- | --- | ---: | ---: | --- | --- |
| `V3_sparse_m3_1103` | `b62f43c49a9093a60610200d0ee9bdd1afbe7a3eac506dd71a9706585d522bad` | 2022 `2445.2773`; 2024 `963.5501`; paired gate PASS | `1103.6977` | **`1090.9100565103`** | `submission/dist/V3_sparse_m3_1103.zip` | 제출 완료 (`57386`) |
| `V3_sparse_m2_1100` | `0c8826b6181403d365a7a14b8309e656b2be4bde6d39b6970d93de02065f8e27` | 2022 `2440.2546`; 2024 `960.5052`; paired gate PASS | `1100.6527` | `1088.5196116458` | `submission/dist/V3_sparse_m2_1100.zip` | 제출 완료 (`57388`) |

두 후보의 build JSON, verification JSON, OOF 예측과 최종 결과표를 서로 다른 파일로
보존했다. 실제 제출 결과와 ID를 연결했으며 현재 파일은 위 SHA로 동결한다.
단일 근거는 [`records/v3_package_index.json`](records/v3_package_index.json)이다.

## V4 예상 1,190 GOAL 패키지 (2026-08-21)

| 작업 ID | ZIP SHA-256 | 로컬 지표 | 고정 예상 LB | 실제 LB | 보관 경로 | 상태 |
| --- | --- | --- | ---: | ---: | --- | --- |
| `V4_compact_supported_1193` | `49708fe3e6a6b4f472e0771ba396f5eacb69ecf5a38d91406fc0049b628754b0` | 2022 `2413.7661`; 2023 `0`(기록용); 2024 **`1052.9440`** | 이전식 `1193.0915` — 폐기 | **약 `1005`** (exact/ID 대기) | `submission/dist/V4_compact_supported_1193.zip` | **제출 성능 실패·감사 앵커로 보존** |

구성은 V3 anchor 3개, teacher-residual student 1개, centered deviation arm 18개로 총
22모델이다. 연구 예측과 ZIP의 최종 sample 차이는 `2.22e-16`, 행 순서·중복·단일행
불변성 차이는 `0`이었다. 245,789행 모사는 `37.60초`, peak `1.50GiB`로 10분/28GB
한도를 통과했다. 동일 입력 재빌드 SHA도 일치했다. 단일 근거는
[`records/v4_package_index.json`](records/v4_package_index.json)이다.

`1193.0915`는 GOAL 시작 시 고정한 `2024 로컬 + 140.1475834416`의 결과이며 실제 LB
보장값이 아니다. V3 M3의 과대추정이 그대로 반복되는 민감도 시나리오는 약 `1180.30`이다.
실제 점수와 제출 ID는 사용자가 업로드한 뒤에만 기록한다.

## 보관 규칙

- 후보 ID는 `S3`, `S4`, `S14A`처럼 고유하게 부여한다.
- 후보별 폴더에 ZIP, 빌드 기록, 검증 보고서, `candidate_manifest.json`을 함께 보관한다.
- 실제 LB 점수는 사용자가 제출한 뒤 `--lb-score` 또는 [`SUBMISSION_LOG.md`](SUBMISSION_LOG.md)에 기록한다.
- 동일 ID로 다른 SHA-256 파일을 보관하려는 시도는 스크립트가 실패시킨다.

```powershell
$env:PYTHONUTF8='1'
& .\.venv\Scripts\python.exe submission\archive_candidate.py `
  S3 submission\dist\S3_e14.zip `
  --build-record submission\records\S3_build.json `
  --verification-report submission\dist\S3_e14.verification.json `
  --local-metric 'rolling_mean_brier=...' `
  --notes 'E14 point-in-time candidate'
```
