# 24시간 사용자 재연결 예측 — research v2

과거 4시간에 상호작용한 방향성 사용자 쌍 A→B가 다음 24시간 안에 다시 상호작용하는지 예측합니다. MathOverflow가 주 데이터이고 AskUbuntu는 별도 학습을 통한 재현 데이터입니다. 질문·답변·댓글 데이터이지 리트윗이나 밈의 인과적 전파 데이터가 아닙니다.

이 폴더는 GitHub 업로드용 경량본입니다. 원자료, 캐시, 학습 모델과 대용량 개별 예측은 제외했고 재현 코드, 설정, 요약 결과, 그림과 최종 보고서를 포함했습니다. 최종 보고서는 `report/팀프로젝트_보고서_사용자재연결예측_보수적최종본.docx`에서 확인할 수 있습니다.

## 결과 보기

PowerShell에서 이 폴더로 이동한 뒤 실행합니다.

```powershell
python -m pip install -r requirements-research.txt
.\start_app.ps1
```

기본 화면은 research_app.py이며 보통 주소는 http://localhost:8501 입니다. 결과 열람은 재학습하거나 유료 API를 호출하지 않습니다.

- 최종_결과_쉽게_설명.txt: 두 데이터의 실제 결과와 해석(전체 완료 후 생성)
- 지표_쉽게_읽기.md: AP, AUC, 정밀도, 재현율, F1, MCC, Brier
- results/research_v2/mathoverflow/report_ko.md: 주 데이터 결과
- results/research_v2/askubuntu/report_ko.md: 별도 데이터 결과
- 각 결과 폴더의 ap_comparison.png, fold_stability.png, pr_and_calibration.png: 그래프

## 구현 내용

추가 확장: **후보 기간 4/12/24시간 × 학습 간격 24/6시간**을 별도 비교합니다. 새 결과 화면의 ‘후보·학습간격 확장’ 탭, 확장실험_결과.md, results/lookback_density_v1을 확인하세요. 기존 최종_결과_쉽게_설명.txt는 원래 v2 실험 보고서로 보존합니다. 확장 설계는 확장실험_설계.md에 있습니다. 확장 재현은 requirements-extensions.txt 설치 후 run_extensions.ps1로 실행합니다.

1. 원본 행동 종류(a2q/c2a/c2q) 복구, 출처·해시·행 수 기록.
2. 과거 정보로 특징 생성, 이후 24시간으로 정답 생성, 불완전한 미래 구간 제외.
3. 5개 시간 테스트 구간 × 3개 시드. 학습/튜닝/보정/테스트 시간순 분리.
4. 이력 HGB, 78개 특징 LightGBM, TGN 계열 메모리 모델, 갱신형 GraphSAGE와 단순 기준선 비교. 결합 모델은 공동 확률 보정이 없어 탐색 기록으로만 남기고 핵심 결론에서는 제외.
5. 행동 종류·시간 감쇠·구조 특징 제거 실험.
6. AskUbuntu 재현 실험, 날짜 블록 신뢰구간, 확률 보정, 저장 예측 재검산.

주 지표는 실제 후보 전체 AP입니다. AP는 정확도가 아니므로 양성률과 강한 기준모델을 함께 제시합니다. 균형 F1을 실제 분포 성능처럼 보고하지 않습니다.

## 데이터

| 데이터 | 원본 이벤트 | 자기 연결 제외 후 | 사용 사용자 |
|---|---:|---:|---:|
| MathOverflow | 506,550 | 390,441 | 24,759 |
| AskUbuntu | 964,437 | 726,661 | 157,222 |

이벤트 수, 후보 쌍-날짜 수, 고유 사용자 수는 다릅니다. 시드 반복으로 표본 수를 부풀리지 않습니다. SNAP 연결 문제로 커밋을 고정한 공개 미러를 사용합니다. 출처·해시는 data_manifest.json에 기록하며 MathOverflow는 기존 원본과 다중집합 일치도 확인했습니다.

## 전체 재현

```powershell
python -m pytest -q
python -m research.batch
python -m research.finalize
```

또는 run_research.ps1을 실행합니다. 코드·규칙이 같으면 완료한 실행을 건너뜁니다. 전체 학습은 시간이 걸립니다. 신경망은 CUDA 사용 가능 시 GPU, 트리는 CPU를 사용합니다. 장비에 맞는 GPU용 PyTorch가 필요합니다. 실제 환경은 run_manifest.json에 기록합니다.

특정 실행: python -m research.run --dataset mathoverflow --fold 0 --seeds 2026

research.report는 15회 완료 후 보고서를 만듭니다. research.finalize는 두 데이터의 전체 완료 후 예측·지표를 재검산하고 모델을 내보냅니다.

## 구조

```text
research_app.py                # 새 결과 화면
research/
  protocol.json               # 실행 규칙
  data.py / features.py       # 원본·과거 특징
  models.py / neural.py       # 트리·GPU 신경망
  evaluation.py               # 시간 분할·지표·보정·부트스트랩
  run.py / batch.py            # 반복 실행·재개·상태
  report.py                   # 표·그림·보고서
  audit.py / export_model.py   # 재검산·모델 검증
  finalize.py                 # 최종 설명서
data/typed/ / data/cache/      # 원본·캐시 (Git 제외)
models/research_v2/            # 검증된 모델 (Git 제외)
results/research_v2/           # 새 결과·예측·그림
tests/                        # 자동 검사
docs/legacy-v1/                # 기존 설명 보관
app.py / mathoverflow_lab/     # v1 화면·기존 코드·공통 함수
```

큰 원본·캐시·모델·압축 예측은 Git에서 제외하되 로컬에 보존합니다. 코드와 작은 결과·그래프·문서는 Git으로 공유할 수 있습니다. Git만 받으면 원본 자동 다운로드로 재현할 수 있습니다.

## 제한

MathOverflow 기존 테스트를 이미 본 후향적 평가입니다. 완전히 손대지 않은 확인용 테스트라고 주장하지 않습니다. AskUbuntu는 별도 학습한 재현 실험이지 제로샷 이전이 아닙니다. TGN은 과제별 구현이며 공식 재현이나 SOTA를 주장하지 않습니다. 신경망 4 epoch와 트리 16회 튜닝의 탐색 예산은 동일하지 않습니다. RESEARCH_PROTOCOL_KO.md에 자세히 설명했습니다.

기존 균형 F1 0.680 등은 v1 결과로, 분할·평가 방식이 달라 새 결과와 직접 비교하지 않습니다. 이전 문서의 ‘테스트를 한 번만 확인’했다는 표현은 현재 연구 전체에 해당하지 않아 정정했습니다.

## 원 출처

확장 실험을 처음 읽는 팀원은 **확장실험_쉽게읽기.txt**부터 확인하세요. 전체 조건은 **확장실험_결과.md**, 완료 검증은 **확장실험_검증.md**에 있습니다.

오류 분석은 **오류분석_쉽게읽기.md**, 그 분석에서 제안한 활동→상대 선택 및 최근성·행동 유형 전문가 실험은 **개선실험_쉽게읽기.md**에 있습니다. 앱의 `오류 개선 구조` 탭에서도 확인할 수 있습니다.

새 데이터에서의 사전 고정 추가 확인은 **SUPERUSER_확인실험_쉽게읽기.md**와 앱의 `Super User 독립 확인` 탭에서 볼 수 있습니다. Super User 원본을 받기 전에 가설과 성공 조건을 고정했고 5개 시간 폴드 × 3개 시드가 모두 끝난 뒤 테스트를 한 번 평가했습니다. 최근성·유형 전문가의 평균 점수는 아주 조금 높았지만 AP 차이의 95% 구간이 0을 포함해 사전등록 성공 조건은 충족하지 못했습니다. 이 결과도 도메인별 재학습이며 완전한 제로샷 외부 검증은 아닙니다.

추가 확인 뒤 진행한 CUDA 상대 선택 실험은 **GPU_랭킹실험_쉽게읽기.md**와 앱의 `GPU 상대 랭킹` 탭에 있습니다. RTX 4070 Ti SUPER에서 15회 학습했지만 RankNet AP 0.1285, NDCG@5 0.7476으로 고정 LightGBM 기준선(AP 0.1393, NDCG@5 0.7556)을 넘지 못했습니다. 이 실험은 후향적 탐색으로 명시하며 실패 결과도 그대로 보존합니다.

- [SNAP MathOverflow](https://snap.stanford.edu/data/sx-mathoverflow.html)
- [SNAP AskUbuntu](https://snap.stanford.edu/data/sx-askubuntu.html)
- [SNAP Super User](https://snap.stanford.edu/data/sx-superuser.html)
- Paranjape, Benson, Leskovec. [Motifs in Temporal Networks](https://snap.stanford.edu/temporal-motifs/), WSDM 2017.

추가 문헌: RESEARCH_PROTOCOL_KO.md.
