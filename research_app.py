"""Read-only research dashboard: opening it never trains or calls a paid API."""
import json
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results" / "research_v2"
st.set_page_config(page_title="24시간 재연결 예측 · 연구 결과", layout="wide")
st.title("최근 연결된 두 사용자는 24시간 안에 다시 연결될까?")
st.caption("MathOverflow / AskUbuntu · 시간순 평가 · 실제 양성 비율 유지 · 결과 열람은 재학습·유료 API 호출 없이 동작합니다.")

with st.sidebar:
    st.header("결과 선택")
    dataset = st.selectbox("데이터셋", ["mathoverflow", "askubuntu"])
    st.button("결과 새로고침")
    st.write("5개 시간 구간 × 3개 학습 시드 = 데이터셋당 15회")
    st.info("MathOverflow: 후향적 평가\n\nAskUbuntu: 해당 데이터로 별도 학습한 재현 실험. 제로샷 이전은 아닙니다.")
    st.markdown("기존 v1 결과는 `python -m streamlit run app.py`로 별도 확인합니다.")

folder = RESULTS / dataset
done = len(list(folder.glob("fold_*/seed_*/complete.json")))
st.progress(min(done / 15, 1), text=f"학습 완료 {done}/15회")
if (RESULTS / "batch_status.json").exists():
    batch = json.loads((RESULTS / "batch_status.json").read_text(encoding="utf-8"))
    st.caption(f"전체 실험 상태: {batch['status']} · 학습 완료 후 보고서·신뢰구간 계산이 이어집니다.")

tabs = st.tabs(["최종 결과", "지표 쉽게 이해하기", "실험 설계", "파일 위치", "후보·학습간격 확장", "오류 개선 구조", "Super User 독립 확인", "GPU 상대 랭킹"])
with tabs[0]:
    path = folder / "summary.json"
    if not path.exists():
        st.warning("아직 전체 실험 보고서가 완성되지 않았습니다. 부분 결과를 최종 성능으로 표시하지 않습니다.")
        st.write("계산이 끝나면 ‘결과 새로고침’을 누르세요.")
    else:
        result = json.loads(path.read_text(encoding="utf-8"))
        if result["completed_runs"] != 15 or not result.get("final_report",False):
            st.warning("부분 보고서입니다. 최종 결과가 아닙니다.")
        stats = st.columns(3)
        stats[0].metric("테스트 후보", f"{result['unique_test_candidates']:,}건")
        stats[1].metric("실제 24시간 내 재연결", f"{result['unique_test_positives']:,}건")
        stats[2].metric("양성률 / 상수 점수 AP", f"{result['test_prevalence']:.2%}")
        st.write("후보 1건은 ‘특정 날짜의 방향성 사용자 쌍’입니다. 고유 사용자 수가 아니며 같은 쌍이 다른 날짜에 다시 등장할 수 있습니다.")
        frame = pd.DataFrame(result["models"])
        wanted = ["History HGB", "Full LightGBM", "TGN recurrence", "Rolling GraphSAGE", "Hybrid ensemble", "Train prevalence"]
        st.subheader("주요 모델 비교")
        st.caption("5개 시간 구간 평균. AP 구간 SD는 구간 간 흔들림이며 95% 신뢰구간이 아닙니다. 테스트 점수로 모델을 다시 선택하지 않습니다.")
        columns = ["model", "ap_mean", "ap_fold_sd", "auc_mean", "f1_mean", "precision_mean", "recall_mean", "mcc_mean", "brier_mean"]
        names = {"model":"모델", "ap_mean":"AP ↑", "ap_fold_sd":"AP 구간 SD", "auc_mean":"ROC-AUC ↑", "f1_mean":"F1 ↑", "precision_mean":"정밀도 ↑", "recall_mean":"재현율 ↑", "mcc_mean":"MCC ↑", "brier_mean":"Brier ↓"}
        st.dataframe(frame.loc[frame.model.isin(wanted), columns].rename(columns=names).round(4), hide_index=True, width="stretch")
        st.info("AP 0.25는 정확도 25%라는 뜻이 아닙니다. 실제 재연결 후보를 높은 순위로 올리는 성능입니다. 다른 데이터의 AP는 양성률과 난도가 달라 단순 비교하면 안 됩니다.")
        for filename, label in [("ap_comparison.png", "모델과 제거 실험 비교"), ("fold_stability.png", "시간이 바뀌어도 성능이 유지되는가?"), ("pr_and_calibration.png", "정밀도–재현율과 확률의 신뢰도")]:
            if (folder / filename).exists():
                st.subheader(label)
                st.image(str(folder / filename), width="stretch")
        st.subheader("기준모델보다 좋아졌다는 근거")
        if (folder / "improvement_intervals.png").exists():
            st.image(str(folder / "improvement_intervals.png"), width="stretch")
        comparison = pd.DataFrame(result["comparisons"])
        st.dataframe(comparison[["model_a","model_b","delta_ap","ci_low","ci_high","block_days"]].round(4), hide_index=True, width="stretch")
        st.caption("같은 테스트 후보를 날짜 블록 단위로 재표집한 AP 차이입니다. 구간 전체가 0보다 높으면 해당 설정에서 개선을 지지합니다. 유형 제거·앙상블 비교는 탐색적 분석입니다.")
        with st.expander("전체 모델 표 / CSV 다운로드"):
            st.dataframe(frame.round(4), hide_index=True)
            st.download_button("모델 결과 CSV", (folder / "model_summary.csv").read_bytes(), file_name=f"{dataset}_model_summary.csv", mime="text/csv")
        st.warning("논문 게재나 SOTA를 보장하는 결과가 아닙니다. TGN recurrence는 과제별 구현이며 공식 TGN 재현으로 주장하지 않습니다. 균형 F1 0.667과 현재 실제 분포 F1을 직접 비교하지 마세요.")

with tabs[1]:
    st.markdown((ROOT / "지표_쉽게_읽기.md").read_text(encoding="utf-8"))
with tabs[2]:
    st.markdown((ROOT / "RESEARCH_PROTOCOL_KO.md").read_text(encoding="utf-8"))
with tabs[3]:
    st.write("프로젝트 폴더")
    st.code(str(ROOT))
    st.write("선택한 데이터의 결과 폴더")
    st.code(str(folder))
    st.markdown("- `report_ko.md`: 결과와 해석\n- `model_summary.csv`: 5구간 평균\n- `paired_block_bootstrap.csv`: 개선량의 95% 구간\n- `fold_N/seed_N/`: 개별 실험 예측·설정·모델\n- `data_manifest.json`: 데이터 출처·해시·전처리\n- `run_manifest.json`: 실행 환경·특징 목록")

with tabs[4]:
    extension = ROOT / "results" / "lookback_density_v1"
    ext_folder = extension / dataset
    st.header("최근 관계 4·12·24시간 × 학습 간격 24·6시간")
    st.write("정답은 계속 다음 24시간 재연결입니다. 검증·테스트는 하루 간격으로 유지하고, 기존 시간 분할 경계도 고정했습니다.")
    ext_path = ext_folder / "summary.json"
    if not ext_path.exists():
        records = len(list(ext_folder.glob("lookback_*/fold_*/seed_*/complete.json")))
        st.info(f"확장 실험 진행 중: {records}/90개 조건·구간·시드 기록 완료. 이전 결과는 다른 탭에서 볼 수 있습니다.")
    else:
        ext = json.loads(ext_path.read_text(encoding="utf-8"))
        ext_table = pd.DataFrame(ext["models"])
        selected_model = st.radio("확장 비교 모델", ["Full LightGBM", "History HGB"], horizontal=True)
        selected = ext_table[ext_table.model==selected_model]
        st.caption("각 조합을 따로 최적화하지 않고 기존 기준 설정을 고정한 민감도 분석입니다. GNN의 기존 결과는 보존하며 이 확장에서는 재학습하지 않았습니다.")
        rename = {"lookback":"후보 기간(h)","train_stride":"학습 간격(h)","test_n":"테스트 후보","test_positive":"양성 건수","prevalence":"양성률",
                  "ap_mean":"AP","auc_mean":"ROC-AUC","f1_mean":"F1","precision_mean":"정밀도","recall_mean":"재현율"}
        st.subheader("① 범위를 넓힌 후보 전체: 얼마나 더 포함했나?")
        st.dataframe(selected[selected.cohort=="all"][["lookback","train_stride","test_n","test_positive","prevalence","ap_mean","auc_mean","f1_mean"]].rename(columns=rename).round(4),hide_index=True,width="stretch")
        st.warning("후보 범위를 바꾸면 양성률·난도도 바뀝니다. 이 표의 AP만으로 예측력이 개선됐다고 단정하지 마세요.")
        st.subheader("② 동일한 기존 4시간 후보: 예측력이 달라졌나?")
        st.dataframe(selected[selected.cohort=="common_4h"][["lookback","train_stride","test_n","ap_mean","auc_mean","f1_mean","precision_mean","recall_mean"]].rename(columns=rename).round(4),hide_index=True,width="stretch")
        for filename in ("lookback_ap.png","candidate_coverage.png"):
            st.image(str(ext_folder/filename),width="stretch")
        with st.expander("LightGBM AP 차이와 탐색적 95% 구간"):
            st.dataframe(pd.DataFrame(ext["comparisons"]).round(4),hide_index=True,width="stretch")
        st.info("6시간 학습은 관찰 시간대와 사례 수를 함께 늘리지만 미래 정답 구간이 겹칩니다. 독립적인 새 원본 데이터가 늘어난 것은 아닙니다. 테스트는 24시간 비중첩으로 유지했습니다.")
        st.caption("기존 테스트 결과를 본 뒤 진행한 탐색적 확장입니다. 다중비교 보정하지 않은 구간이며 최고 테스트 조합을 새 최종 모델로 선정하지 않습니다.")
        st.download_button("확장 결과 CSV",(ext_folder/"model_summary.csv").read_bytes(),file_name=f"{dataset}_lookback_density.csv",mime="text/csv")

with tabs[5]:
    remedy_folder = ROOT / "results" / "remedy_v1" / dataset
    remedy_path = remedy_folder / "summary.json"
    st.header("사용자 활동→상대 선택 · 최근성/행동 유형 전문가")
    st.write("오류 분석에서 나온 두 개선안을 기존 24시간 후보·6시간 학습 모델과 같은 시간 구간에서 비교합니다.")
    if not remedy_path.exists():
        records = len(list(remedy_folder.glob("fold_*/seed_*/complete.json")))
        st.info(f"실험 진행 중: {records}/15개 구간·시드 완료. 전체 검산 전에는 부분 점수를 최종 결과로 표시하지 않습니다.")
    else:
        remedy = json.loads(remedy_path.read_text(encoding="utf-8"))
        table = pd.DataFrame(remedy["models"])
        all_candidates = table[table.cohort=="all"]
        labels = {"Baseline Full LightGBM":"기존 Full LightGBM",
                  "Two-stage activity-choice":"1번: 활동→상대 선택",
                  "Recency-type experts":"2번: 최근성·유형 전문가",
                  "Two-stage recency-type experts":"1+2번 결합"}
        all_candidates = all_candidates.copy();all_candidates["model"] = all_candidates.model.map(labels)
        columns = ["model","ap_mean","auc_mean","f1_mean","precision_mean","recall_mean","mcc_mean","brier_mean","test_n","test_positive"]
        names = {"model":"모델","ap_mean":"AP","auc_mean":"ROC-AUC","f1_mean":"F1","precision_mean":"정밀도",
                 "recall_mean":"재현율","mcc_mean":"MCC","brier_mean":"Brier","test_n":"후보","test_positive":"양성"}
        st.dataframe(all_candidates[columns].rename(columns=names).round(4),hide_index=True,width="stretch")
        for filename in ("remedy_comparison.png","ap_differences.png"):
            st.image(str(remedy_folder/filename),width="stretch")
        st.subheader("기준모델과 AP 차이의 탐색적 95% 구간")
        st.dataframe(pd.DataFrame(remedy["comparisons"]).round(4),hide_index=True,width="stretch")
        st.warning("두 데이터셋의 테스트 오류를 본 뒤 설계한 후향적 탐색 실험입니다. 여기서 가장 높은 모델을 새 미확인 최종 모델로 주장하지 않습니다.")
        st.caption("미래 활동은 과거 학습 구간의 정답으로만 사용합니다. 테스트 예측에는 실제 미래 활동이 들어가지 않습니다. 임계값은 테스트 이전 보정 구간에서 정했습니다.")
        st.download_button("개선 구조 결과 CSV",(remedy_folder/"model_summary.csv").read_bytes(),file_name=f"{dataset}_remedy_models.csv",mime="text/csv")

with tabs[6]:
    confirmation = ROOT / "results" / "confirmation_superuser_v1"
    confirmation_path = confirmation / "summary.json"
    st.header("Super User: 손대지 않은 독립 확인 실험")
    st.write("MathOverflow·AskUbuntu 오류 분석에서 만든 개선안이 새 데이터에서도 재현되는지 확인했습니다. Super User 원본을 받기 전에 가설과 성공 조건을 고정하고, 5개 시간 폴드 × 3개 시드가 모두 끝난 뒤 테스트 지표를 한 번만 계산했습니다.")
    if not confirmation_path.exists():
        records = len(list(confirmation.glob("fold_*/seed_*/complete.json")))
        st.info(f"확인 실험 진행 중: {records}/15개 구간·시드 완료. 전체 완료 전에는 테스트 지표를 표시하지 않습니다.")
    else:
        confirmation_result = json.loads(confirmation_path.read_text(encoding="utf-8"))
        success = bool(confirmation_result["primary_success"])
        if success:
            st.success("사전등록 성공 조건을 충족했습니다.")
        else:
            st.warning("사전등록 성공 조건을 충족하지 못했습니다. 평균은 조금 좋아졌지만, 개선의 95% 신뢰구간이 0을 포함합니다.")
        cards = st.columns(4)
        cards[0].metric("테스트 후보", f"{confirmation_result['models'][0]['test_n']:,}건")
        cards[1].metric("실제 재연결", f"{confirmation_result['models'][0]['test_positive']:,}건")
        cards[2].metric("양성률", f"{confirmation_result['models'][0]['prevalence']:.2%}")
        cards[3].metric("완료 실행", f"{confirmation_result['completed_runs']}/15")
        confirmation_table = pd.DataFrame(confirmation_result["models"])
        confirmation_table = confirmation_table[confirmation_table.cohort == "all"].copy()
        confirmation_labels = {
            "Baseline Full LightGBM": "기준 Full LightGBM",
            "Two-stage activity-choice": "활동→상대 선택",
            "Recency-type experts": "최근성·유형 전문가",
            "Two-stage recency-type experts": "두 개선안 결합",
        }
        confirmation_table["model"] = confirmation_table.model.map(confirmation_labels)
        show_columns = ["model", "ap_mean", "auc_mean", "f1_mean", "precision_mean", "recall_mean", "mcc_mean", "ap_lift_mean"]
        show_names = {"model":"모델", "ap_mean":"AP", "auc_mean":"ROC-AUC", "f1_mean":"F1", "precision_mean":"정밀도", "recall_mean":"재현율", "mcc_mean":"MCC", "ap_lift_mean":"무작위 대비 AP 배수"}
        st.dataframe(confirmation_table[show_columns].rename(columns=show_names).round(4), hide_index=True, width="stretch")
        primary = next(row for row in confirmation_result["comparisons"] if row["primary"])
        st.info(f"주 비교: 최근성·유형 전문가의 AP 차이 {primary['delta_ap']:+.6f}, 95% 구간 [{primary['ci_low']:+.6f}, {primary['ci_high']:+.6f}]. F1 차이 {confirmation_result['primary_f1_delta']:+.6f}.")
        st.write("쉽게 말해 ‘내일 활동할 사용자’를 맞히는 1단계는 잘 됐지만, 그 사용자가 수많은 과거 상대 중 누구와 다시 연결될지를 고르는 2단계가 어려웠습니다. 따라서 두 단계 분리는 설명에는 유용했지만 최종 링크 순위 성능은 오히려 낮아졌습니다.")
        for filename in ("model_comparison.png", "ap_differences.png"):
            if (confirmation / filename).exists():
                st.image(str(confirmation / filename), width="stretch")
        st.caption("AP는 전체 후보의 자연 양성률 3.66%에서 평가했습니다. 기준모델 AP 0.1393은 무작위 순위 기대치의 약 3.81배입니다. 이 실험은 Super User에서 새로 학습한 도메인 내 확인이며 제로샷 이전이나 인과 효과 검증은 아닙니다.")
        st.download_button("Super User 결과 CSV", (confirmation / "model_summary.csv").read_bytes(), file_name="superuser_confirmation_model_summary.csv", mime="text/csv")

with tabs[7]:
    ranking = ROOT / "results" / "ranking_v1"
    ranking_path = ranking / "summary.json"
    st.header("CUDA 사용자별 상대 선택 랭킹")
    st.write("같은 시점에 사용자 A가 다시 만날 수 있는 B·C·D를 한 그룹으로 묶고, 실제 재연결 상대를 위로 올리는 pairwise RankNet을 RTX GPU에서 학습했습니다.")
    st.warning("Super User 독립 확인 결과를 본 뒤 만든 후향적 탐색 실험입니다. 새로운 손대지 않은 확인 결과로 해석하면 안 됩니다.")
    if not ranking_path.exists():
        records = len(list(ranking.glob("fold_*/seed_*/complete.json")))
        st.info(f"CUDA 학습 진행 중: {records}/15회 완료")
    else:
        ranking_result = json.loads(ranking_path.read_text(encoding="utf-8"))
        if ranking_result["exploratory_success_rule_met"]:
            st.success("탐색적 성공 규칙을 충족했습니다.")
        else:
            st.error("탐색적 성공 규칙을 충족하지 못했습니다. CUDA RankNet이 기준 LightGBM보다 낮았습니다.")
        ranking_table = pd.DataFrame(ranking_result["models"])
        ranking_labels = {"Frozen Baseline Full LightGBM":"고정 기준 LightGBM",
                          "CUDA RankNet":"CUDA RankNet",
                          "Fixed 50/50 ensemble":"고정 50/50 앙상블"}
        ranking_table["model"] = ranking_table.model.map(ranking_labels)
        ranking_columns = ["model","ap_mean","auc_mean","f1_mean","recall_at_1","recall_at_5","ndcg_at_5","map_at_5"]
        ranking_names = {"model":"모델","ap_mean":"전체 AP","auc_mean":"ROC-AUC","f1_mean":"F1",
                         "recall_at_1":"상대 Recall@1","recall_at_5":"상대 Recall@5",
                         "ndcg_at_5":"상대 NDCG@5","map_at_5":"상대 MAP@5"}
        st.dataframe(ranking_table[ranking_columns].rename(columns=ranking_names).round(4), hide_index=True, width="stretch")
        ap_row = next(row for row in ranking_result["ap_comparisons"] if row["model"] == "CUDA RankNet")
        ndcg_row = ranking_result["ranknet_ndcg_comparison"]
        st.info(f"RankNet AP 차이 {ap_row['delta_ap']:+.4f} (95% 구간 {ap_row['ci_low']:+.4f}~{ap_row['ci_high']:+.4f}), NDCG@5 차이 {ndcg_row['delta_ndcg_at_5']:+.4f} (95% 구간 {ndcg_row['ci_low']:+.4f}~{ndcg_row['ci_high']:+.4f}).")
        st.write("결론: 상대를 같은 사용자 안에서 비교하는 문제 정의는 유용했지만, 현재 MLP RankNet은 트리 기준선이 포착한 횟수·최근성의 비선형 규칙을 넘지 못했습니다. 앙상블도 손실을 줄였을 뿐 기준선을 이기지는 못했습니다.")
        for filename in ("ranking_comparison.png", "ndcg_difference.png"):
            if (ranking / filename).exists():
                st.image(str(ranking / filename), width="stretch")
        st.caption("15/15 학습 기록에서 device=cuda:0, GPU=RTX 4070 Ti SUPER를 검증했습니다. LightGBM 기준선은 이전 잠긴 예측을 재사용했고 새 RankNet만 GPU에서 학습했습니다.")
        st.download_button("GPU 랭킹 결과 CSV", (ranking / "model_summary.csv").read_bytes(), file_name="superuser_gpu_ranking.csv", mime="text/csv")
