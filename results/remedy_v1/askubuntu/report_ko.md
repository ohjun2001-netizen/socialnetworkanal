# askubuntu: 오류 개선 구조 실험

기존 24시간 후보·6시간 학습 Full LightGBM과 두 요청 모델 및 결합형을 같은 시간 분할에서 비교했습니다.
두 데이터셋의 테스트 오류를 본 뒤 설계했으므로 모두 탐색 결과이며 새 독립 확인 결과가 아닙니다.

## 넓어진 24시간 후보 전체

| model | ap_mean | auc_mean | f1_mean | precision_mean | recall_mean | mcc_mean | brier_mean | test_n | test_positive |
|---|---|---|---|---|---|---|---|---|---|
| Baseline Full LightGBM | 0.1532 | 0.7982 | 0.2185 | 0.1641 | 0.3306 | 0.1870 | 0.0360 | 367855 | 14783 |
| Recency-type experts | 0.1554 | 0.8002 | 0.2187 | 0.1678 | 0.3333 | 0.1890 | 0.0360 | 367855 | 14783 |
| Two-stage activity-choice | 0.1323 | 0.7911 | 0.2030 | 0.1401 | 0.3868 | 0.1783 | 0.0364 | 367855 | 14783 |
| Two-stage recency-type experts | 0.1344 | 0.7928 | 0.2023 | 0.1516 | 0.3251 | 0.1718 | 0.0364 | 367855 | 14783 |

## 동일한 기존 4시간 후보

| model | ap_mean | f1_mean | precision_mean | recall_mean |
|---|---|---|---|---|
| Baseline Full LightGBM | 0.2204 | 0.2770 | 0.1819 | 0.5855 |
| Recency-type experts | 0.2225 | 0.2784 | 0.1846 | 0.5800 |
| Two-stage activity-choice | 0.1896 | 0.2646 | 0.1756 | 0.5499 |
| Two-stage recency-type experts | 0.1912 | 0.2655 | 0.1873 | 0.4703 |

## 기준모델과 AP 차이

| model | baseline | cohort | delta_ap | ci_low | ci_high | n | block_days | blocks | repeats |
|---|---|---|---|---|---|---|---|---|---|
| Two-stage activity-choice | Baseline Full LightGBM | all | -0.0189 | -0.0223 | -0.0156 | 367855 | 7 | 151 | 1000 |
| Recency-type experts | Baseline Full LightGBM | all | 0.0022 | 0.0003 | 0.0041 | 367855 | 7 | 151 | 1000 |
| Two-stage recency-type experts | Baseline Full LightGBM | all | -0.0170 | -0.0204 | -0.0136 | 367855 | 7 | 151 | 1000 |
| Two-stage activity-choice | Baseline Full LightGBM | common_4h | -0.0274 | -0.0338 | -0.0209 | 84311 | 7 | 151 | 1000 |
| Recency-type experts | Baseline Full LightGBM | common_4h | 0.0019 | -0.0022 | 0.0059 | 84311 | 7 | 151 | 1000 |
| Two-stage recency-type experts | Baseline Full LightGBM | common_4h | -0.0262 | -0.0329 | -0.0196 | 84311 | 7 | 151 | 1000 |

## 1단계 사용자 활동 모델 자체의 성능

| fold | ap | auc | f1 | precision | recall | mcc | brier |
|---|---|---|---|---|---|---|---|
| 0 | 0.6611 | 0.8365 | 0.6089 | 0.5729 | 0.6498 | 0.4816 | 0.1235 |
| 1 | 0.6903 | 0.8465 | 0.6332 | 0.6020 | 0.6678 | 0.5102 | 0.1208 |
| 2 | 0.6763 | 0.8459 | 0.6246 | 0.6199 | 0.6294 | 0.5088 | 0.1201 |
| 3 | 0.7268 | 0.8638 | 0.6651 | 0.6373 | 0.6955 | 0.5501 | 0.1147 |
| 4 | 0.7294 | 0.8656 | 0.6713 | 0.6580 | 0.6851 | 0.5587 | 0.1155 |

## 모델 정의

- Two-stage activity-choice: 과거의 사용자 활동 특징만으로 A가 다음 24시간 행동할 확률을 구하고, 활동한 학습 사례로 학습한 A→B 재연결 조건부 점수와 곱합니다.
- Recency-type experts: 마지막 연결이 0~4시간/4~24시간인지와 마지막 행동 유형 a2q/c2a/c2q를 교차한 여섯 모델입니다.
- Two-stage recency-type experts: 활동 모델과 조건부 여섯 전문가를 결합했습니다.
- 미래 활동은 과거 학습 구간의 정답으로만 사용합니다. 테스트 시점에는 과거 특징만 사용하며 실제 미래 활동을 입력하지 않습니다.

## 제한

- 기존 fold별 LightGBM 설정을 고정해서 구조만 비교했습니다. 테스트 점수로 모델·설정·임계값을 고르지 않았습니다.
- F1 임계값은 각 fold의 테스트 이전 보정 구간에서 정했습니다.
- 시드 평균 뒤 같은 후보에 대한 AP를 7일 시간 블록으로 1,000회 재표집했습니다. 시간 의존성이 더 길게 남을 수 있습니다.
- 후보는 과거 24시간에 연결됐던 관계로 제한됩니다. 완전히 새로운 관계 예측 결과가 아닙니다.
- 이 결과로 인과관계, SOTA 또는 새 데이터 성능을 주장하지 않습니다.