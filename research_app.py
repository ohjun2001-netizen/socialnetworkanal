"""Read-only dashboard for the final MathOverflow and AskUbuntu results."""

import json
from pathlib import Path

import pandas as pd
import streamlit as st


ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results" / "research_v2"

st.set_page_config(page_title="24시간 사용자 재연결 예측", layout="wide")
st.title("최근 연결된 두 사용자는 24시간 안에 다시 연결될까?")
st.caption(
    "MathOverflow와 AskUbuntu의 시간순 평가 결과입니다. "
    "저장된 결과를 읽기만 하며 재학습이나 유료 API 호출을 하지 않습니다."
)

with st.sidebar:
    st.header("결과 선택")
    dataset = st.selectbox("데이터셋", ["mathoverflow", "askubuntu"])
    st.info(
        "MathOverflow는 회고적 탐색 결과입니다.\n\n"
        "AskUbuntu는 같은 절차로 별도 재학습한 반복 실험이며 제로샷 검증이 아닙니다."
    )

folder = RESULTS / dataset
summary_path = folder / "summary.json"

if not summary_path.exists():
    st.error(f"결과 파일을 찾을 수 없습니다: {summary_path}")
    st.stop()

result = json.loads(summary_path.read_text(encoding="utf-8"))
models = pd.DataFrame(result["models"])

tabs = st.tabs(["핵심 결과", "평가 지표", "실험 설계", "파일 안내"])

with tabs[0]:
    cards = st.columns(3)
    cards[0].metric("테스트 후보", f"{result['unique_test_candidates']:,}건")
    cards[1].metric("24시간 내 재연결", f"{result['unique_test_positives']:,}건")
    cards[2].metric("양성률", f"{result['test_prevalence']:.2%}")

    st.caption(
        "후보 1건은 특정 예측 시점의 방향성 사용자 쌍입니다. "
        "같은 사용자 쌍이 다른 시점에 다시 후보가 될 수 있습니다."
    )

    wanted = [
        "Pair frequency",
        "History HGB",
        "Full LightGBM",
        "TGN recurrence",
        "Rolling GraphSAGE",
        "Train prevalence",
    ]
    columns = [
        "model",
        "ap_mean",
        "ap_fold_sd",
        "auc_mean",
        "f1_mean",
        "precision_mean",
        "recall_mean",
        "mcc_mean",
        "brier_mean",
    ]
    labels = {
        "model": "모델",
        "ap_mean": "AP ↑",
        "ap_fold_sd": "AP 폴드 SD",
        "auc_mean": "ROC-AUC ↑",
        "f1_mean": "F1 ↑",
        "precision_mean": "정밀도 ↑",
        "recall_mean": "재현율 ↑",
        "mcc_mean": "MCC ↑",
        "brier_mean": "Brier ↓",
    }

    table = models.loc[models.model.isin(wanted), columns].rename(columns=labels)
    st.subheader("모델별 결과")
    st.dataframe(table.round(4), hide_index=True, width="stretch")
    st.info(
        "AP는 정확도가 아닙니다. 실제 양성을 후보 목록의 위쪽에 배치하는 능력을 "
        "나타내며, 각 데이터의 양성률을 무작위 기준으로 함께 봐야 합니다."
    )

    figures = [
        ("ap_comparison.png", "모델별 AP"),
        ("fold_stability.png", "시간 폴드별 안정성"),
        ("pr_and_calibration.png", "정밀도·재현율과 확률 보정"),
        ("improvement_intervals.png", "기준모델 대비 AP 차이"),
    ]
    for filename, title in figures:
        path = folder / filename
        if path.exists():
            st.subheader(title)
            st.image(str(path), width="stretch")

with tabs[1]:
    st.subheader("지표 해석")
    st.markdown(
        """
- **AP**: 불균형 후보에서 실제 재연결을 상위에 배치하는 순위 성능입니다. 무작위 AP의 기준은 양성률입니다.
- **ROC-AUC**: 무작위로 뽑은 양성과 음성 중 양성에 더 높은 점수를 줄 확률로 해석할 수 있습니다.
- **정밀도**: 재연결한다고 분류한 관계 중 실제 재연결 비율입니다.
- **재현율**: 실제 재연결 관계 중 모델이 찾아낸 비율입니다.
- **F1**: 정밀도와 재현율의 조화평균이며 선택한 임계값에 영향을 받습니다.
- **MCC**: 네 가지 혼동행렬 값을 모두 사용하는 불균형 분류 지표입니다.
- **Brier score**: 예측 확률과 실제 정답의 제곱 오차이며 낮을수록 좋습니다.

균형 표본의 점수를 실제 후보 전체의 성능처럼 해석하지 않습니다. 최종 결과는 자연 양성률을 유지한 후보에서 평가했습니다.
"""
    )

with tabs[2]:
    st.subheader("시간 기반 예측 설계")
    st.markdown(
        """
1. 직전 4시간에 상호작용한 방향성 사용자 쌍을 후보로 선택합니다.
2. 예측 시점 이전 정보만 이용해 78개 특성을 계산합니다.
3. 같은 방향의 상호작용이 다음 24시간에 다시 관측되면 양성으로 정의합니다.
4. 학습, 보정, 테스트 구간을 시간순으로 분리합니다.
5. 두 데이터에서 각각 5개 시간 폴드와 3개 학습 시드를 사용합니다.

주 모델은 Full LightGBM입니다. GraphSAGE와 TGN 계열 모델은 입력과 튜닝 예산이 달라 공정한 우열 비교로 해석하지 않습니다. MathOverflow 결과는 설정 탐색의 영향을 받았으며 AskUbuntu는 별도 재학습한 반복 실험입니다.
"""
    )

with tabs[3]:
    st.subheader("저장된 결과 파일")
    st.code(str(folder))
    st.markdown(
        """
- `summary.json`: 앱에 표시되는 최종 집계
- `model_summary.csv`: 모델별 5개 폴드 평균
- `fold_metrics.csv`: 시간 폴드별 지표
- `data_manifest.json`: 데이터 출처와 전처리 기록
- `run_manifest.json`: 실행 환경과 특성 기록
- `*.png`: 발표와 보고서에 사용한 대표 결과 그림
"""
    )
    model_csv = folder / "model_summary.csv"
    if model_csv.exists():
        st.download_button(
            "모델 결과 CSV 다운로드",
            model_csv.read_bytes(),
            file_name=f"{dataset}_model_summary.csv",
            mime="text/csv",
        )

st.warning(
    "이 결과는 논문 수준의 보편적 성능이나 인과관계를 입증하지 않습니다. "
    "선택된 문제 정의에서 과거 관계 기록의 예측 신호를 평가한 학부 팀프로젝트입니다."
)
