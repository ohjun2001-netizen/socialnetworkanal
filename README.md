# 24시간 사용자 재연결 예측

직전 4시간에 상호작용한 방향성 사용자 쌍 `A→B`가 다음 24시간 안에 다시 상호작용하는지를 예측하는 소셜 네트워크 분석 팀프로젝트입니다.

- 주 데이터: MathOverflow
- 반복 실험: AskUbuntu에서 동일 절차로 별도 재학습
- 주 모델: 78개 과거 특성을 사용하는 LightGBM
- 비교 모델: 과거 빈도, History HGB, GraphSAGE, TGN 계열 모델
- 주 지표: 실제 후보 분포에서 계산한 Average Precision(AP)

MathOverflow는 설정 탐색의 영향을 받은 회고적 결과입니다. AskUbuntu 역시 제로샷 외부 검증이 아니라 해당 데이터에서 모델을 다시 학습한 반복 실험입니다.

## 예측 문제

각 예측 시점에서 다음과 같이 사례를 만듭니다.

1. 직전 4시간에 관측된 방향성 사용자 쌍을 후보로 선택합니다.
2. 예측 시점 이전 기록만 이용해 빈도, 최근성, 활동량, 방향성 이력과 구조 특성을 계산합니다.
3. 같은 방향의 상호작용이 다음 24시간에 다시 나타나면 양성으로 표시합니다.
4. 시간순 학습·보정·테스트 구간을 사용해 미래 정보 누수를 방지합니다.

따라서 이 연구는 모든 새로운 링크를 예측하는 것이 아니라, 최근 연결된 관계의 재발생을 예측하는 조건부 문제입니다.

## 주요 결과

| 데이터 | 테스트 양성률 | Full LightGBM AP | 무작위 기준 대비 |
|---|---:|---:|---:|
| MathOverflow | 0.1210 | 0.2260 | 약 1.87배 |
| AskUbuntu | 0.0722 | 0.2183 | 약 3.02배 |

수치는 5개 시간 폴드의 평균입니다. AP는 정확도가 아니며 양성률을 무작위 순위의 기준으로 함께 해석해야 합니다. 이 결과는 과거 상호작용에 미래 재연결을 구분하는 신호가 있음을 보여주는 탐색적 증거입니다.

## 설치 및 결과 화면

Python 3.11 환경을 권장합니다.

```powershell
python -m pip install -r requirements-research.txt
python -m streamlit run research_app.py
```

Windows에서는 다음 명령도 사용할 수 있습니다.

```powershell
.\start_app.ps1
```

저장된 결과 화면을 보는 과정에서는 재학습이나 유료 API 호출이 발생하지 않습니다.

## 전체 실험 재현

```powershell
python -m pytest -q tests/test_pipeline.py tests/test_research.py
python -m research.batch
python -m research.finalize
```

전체 학습은 시간이 오래 걸립니다. CUDA를 사용할 수 있으면 신경망 모델은 GPU를 사용하고, LightGBM과 HGB는 CPU를 사용합니다. 원자료는 코드가 내려받으며 `data/`, 학습 모델과 대용량 개별 예측은 Git에 포함하지 않습니다.

## 저장소 구조

```text
research_app.py          저장된 핵심 결과 대시보드
research/                데이터 처리, 특성 생성, 모델, 시간순 평가
mathoverflow_lab/        공통 시간 후보 및 특성 함수
tests/                   핵심 파이프라인 자동 검사
results/research_v2/     대표 집계표와 결과 그림
report/최종보고서.docx   최종 팀프로젝트 보고서
```

## 해석상의 한계

- MathOverflow에서 후보 기간과 특성을 탐색했으므로 확증적 검증으로 볼 수 없습니다.
- AskUbuntu는 별도 재학습 실험이며 완전한 외부 이전 성능을 뜻하지 않습니다.
- GNN과 트리 모델은 입력과 탐색 예산이 달라 모델 자체의 우열을 결론 내릴 수 없습니다.
- 원자료에 게시물 ID가 없어 같은 게시물의 연속 활동과 다른 게시물에서의 재접촉을 구분할 수 없습니다.
- 높은 균형 표본 F1을 실제 후보 전체의 성능으로 해석하지 않습니다.

## 데이터 출처

- [SNAP MathOverflow](https://snap.stanford.edu/data/sx-mathoverflow.html)
- [SNAP AskUbuntu](https://snap.stanford.edu/data/sx-askubuntu.html)
- Paranjape, Benson, Leskovec, *Motifs in Temporal Networks*, WSDM 2017
