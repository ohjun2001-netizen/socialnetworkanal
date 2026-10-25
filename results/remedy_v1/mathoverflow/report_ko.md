# mathoverflow: 오류 개선 구조 실험

기존 24시간 후보·6시간 학습 Full LightGBM과 두 요청 모델 및 결합형을 같은 시간 분할에서 비교했습니다.
두 데이터셋의 테스트 오류를 본 뒤 설계했으므로 모두 탐색 결과이며 새 독립 확인 결과가 아닙니다.

## 넓어진 24시간 후보 전체

| model | ap_mean | auc_mean | f1_mean | precision_mean | recall_mean | mcc_mean | brier_mean | test_n | test_positive |
|---|---|---|---|---|---|---|---|---|---|
| Baseline Full LightGBM | 0.1686 | 0.7379 | 0.2425 | 0.1690 | 0.4376 | 0.1868 | 0.0589 | 133841 | 8881 |
| Recency-type experts | 0.1684 | 0.7364 | 0.2410 | 0.1674 | 0.4404 | 0.1855 | 0.0589 | 133841 | 8881 |
| Two-stage activity-choice | 0.1572 | 0.7292 | 0.2304 | 0.1635 | 0.4036 | 0.1709 | 0.0593 | 133841 | 8881 |
| Two-stage recency-type experts | 0.1565 | 0.7272 | 0.2273 | 0.1572 | 0.4228 | 0.1685 | 0.0593 | 133841 | 8881 |

## 동일한 기존 4시간 후보

| model | ap_mean | f1_mean | precision_mean | recall_mean |
|---|---|---|---|---|
| Baseline Full LightGBM | 0.2267 | 0.2960 | 0.1842 | 0.7588 |
| Recency-type experts | 0.2276 | 0.2935 | 0.1821 | 0.7685 |
| Two-stage activity-choice | 0.2158 | 0.2964 | 0.1954 | 0.6229 |
| Two-stage recency-type experts | 0.2154 | 0.2919 | 0.1912 | 0.6284 |

## 기준모델과 AP 차이

| model | baseline | cohort | delta_ap | ci_low | ci_high | n | block_days | blocks | repeats |
|---|---|---|---|---|---|---|---|---|---|
| Two-stage activity-choice | Baseline Full LightGBM | all | -0.0103 | -0.0135 | -0.0075 | 133841 | 7 | 170 | 1000 |
| Recency-type experts | Baseline Full LightGBM | all | -0.0006 | -0.0026 | 0.0016 | 133841 | 7 | 170 | 1000 |
| Two-stage recency-type experts | Baseline Full LightGBM | all | -0.0106 | -0.0140 | -0.0076 | 133841 | 7 | 170 | 1000 |
| Two-stage activity-choice | Baseline Full LightGBM | common_4h | -0.0088 | -0.0154 | -0.0037 | 31305 | 7 | 170 | 1000 |
| Recency-type experts | Baseline Full LightGBM | common_4h | -0.0007 | -0.0051 | 0.0037 | 31305 | 7 | 170 | 1000 |
| Two-stage recency-type experts | Baseline Full LightGBM | common_4h | -0.0089 | -0.0152 | -0.0035 | 31305 | 7 | 170 | 1000 |

## 1단계 사용자 활동 모델 자체의 성능

| fold | ap | auc | f1 | precision | recall | mcc | brier |
|---|---|---|---|---|---|---|---|
| 0 | 0.5722 | 0.7530 | 0.5777 | 0.4875 | 0.7095 | 0.3544 | 0.1751 |
| 1 | 0.5172 | 0.7384 | 0.5425 | 0.4450 | 0.6948 | 0.3243 | 0.1722 |
| 2 | 0.5165 | 0.7366 | 0.5283 | 0.4499 | 0.6403 | 0.3181 | 0.1691 |
| 3 | 0.5285 | 0.7502 | 0.5401 | 0.4348 | 0.7127 | 0.3323 | 0.1647 |
| 4 | 0.5240 | 0.7442 | 0.5360 | 0.4432 | 0.6781 | 0.3314 | 0.1650 |

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