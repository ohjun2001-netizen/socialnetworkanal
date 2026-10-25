"""Aggregate completed folds without selecting or retuning from test results."""
from __future__ import annotations
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import average_precision_score, roc_auc_score, precision_recall_curve

from .data import ROOT
from .evaluation import block_comparison

FOCUS = ["History HGB","Full LightGBM","No types LightGBM","No decay LightGBM",
         "No structure LightGBM","TGN recurrence","Rolling GraphSAGE","Hybrid ensemble","Pair frequency","Train prevalence"]


def markdown_table(frame):
    headers = list(frame.columns)
    lines = ["| " + " | ".join(headers) + " |", "|"+"|".join(["---"]*len(headers))+"|"]
    for row in frame.itertuples(index=False, name=None):
        lines.append("| " + " | ".join(f"{v:.4f}" if isinstance(v, (float, np.floating)) else str(v) for v in row) + " |")
    return "\n".join(lines)


def aggregate(dataset, repeats=1000, require_complete=True):
    output = ROOT / "results" / "research_v2" / dataset
    completed = sorted(output.glob("fold_*/seed_*/complete.json"))
    if require_complete and len(completed) != 15:
        raise RuntimeError(f"{dataset}: {len(completed)}/15 runs complete; no final report yet")
    tables, records = [], []
    for done in completed:
        tables.append(pd.read_csv(done.parent / "metrics.csv"))
        records.append(pd.read_csv(done.parent / "predictions.csv.gz"))
    measurements = pd.concat(tables, ignore_index=True)
    predictions = pd.concat(records, ignore_index=True)
    measurements.to_csv(output / "all_metrics.csv", index=False)
    # Seeds describe training variability; independent days describe data uncertainty.
    per_fold = measurements.groupby(["model","fold"], observed=True).agg(
        ap=("ap","mean"), ap_seed_sd=("ap","std"), auc=("auc","mean"),
        f1=("f1","mean"), precision=("precision","mean"), recall=("recall","mean"),
        mcc=("mcc","mean"), brier=("brier","mean"), p5=("precision_at_5","mean"),
        r5=("recall_at_5","mean"), prevalence=("prevalence","mean"),
    ).reset_index()
    summary = per_fold.groupby("model", observed=True).agg(
        ap_mean=("ap","mean"), ap_fold_sd=("ap","std"), ap_seed_sd_mean=("ap_seed_sd","mean"),
        auc_mean=("auc","mean"), f1_mean=("f1","mean"), precision_mean=("precision","mean"),
        recall_mean=("recall","mean"), mcc_mean=("mcc","mean"), brier_mean=("brier","mean"),
        precision_at_5=("p5","mean"), recall_at_5=("r5","mean"),
    ).reset_index().sort_values("ap_mean",ascending=False)
    per_fold.to_csv(output / "fold_metrics.csv", index=False)
    summary.to_csv(output / "model_summary.csv", index=False)
    keys = ["fold","cutoff","src","dst","label","model"]
    ensemble = predictions.groupby(keys, observed=True, sort=True).score.mean().reset_index()
    ensemble.to_csv(output / "oof_seed_ensemble.csv.gz",index=False,compression="gzip")
    pooled = []
    for name, group in ensemble.groupby("model"):
        ap = average_precision_score(group.label, group.score)
        pooled.append({"model":name,"ap":ap,"auc":roc_auc_score(group.label,group.score),
                       "prevalence":group.label.mean(),"ap_lift":ap/group.label.mean(),
                       "n":len(group),"positive":int(group.label.sum())})
    pd.DataFrame(pooled).sort_values("ap",ascending=False).to_csv(output / "pooled_ensemble_metrics.csv",index=False)
    comparisons = []
    for a,b in (("Full LightGBM","History HGB"),("Full LightGBM","No types LightGBM"),("Hybrid ensemble","Full LightGBM")):
        if a in ensemble.model.unique() and b in ensemble.model.unique():
            comparisons.append(block_comparison(predictions,a,b,repeats=repeats))
    # Sensitivity to a longer dependence window for the pre-specified primary contrast.
    if require_complete:
        for block in (14,28):
            comparisons.append(block_comparison(predictions,"Full LightGBM","History HGB", repeats=repeats,block_days=block))
    pd.DataFrame(comparisons).to_csv(output / "paired_block_bootstrap.csv",index=False)
    _plots(output,summary,per_fold,ensemble)
    _comparison_plot(output, comparisons)
    data_info = json.loads((output / "data_manifest.json").read_text())
    counts = ensemble[ensemble.model=="Full LightGBM"]
    dates = {"first_test_cutoff_utc":str(pd.to_datetime(counts.cutoff.min(),unit="s",utc=True)),
             "last_test_label_end_utc":str(pd.to_datetime(counts.cutoff.max()+86400,unit="s",utc=True))}
    result = {"dataset":dataset,"completed_runs":len(completed),"expected_runs":15,
              "final_report":require_complete,"bootstrap_repeats":repeats,
              "test_dates":dates,
              "data":data_info,"unique_test_candidates":len(counts),"unique_test_positives":int(counts.label.sum()),
              "test_prevalence":float(counts.label.mean()),"models":summary.to_dict("records"),
              "pooled_ensemble":pooled,"comparisons":comparisons,
              "scope":"retrospective" if dataset=="mathoverflow" else "separate-dataset replication, not zero-shot",
              "disclosure":"v1 MathOverflow test was previously examined; no claim of untouched confirmatory holdout"}
    (output/"summary.json").write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding="utf-8")
    table=summary[summary.model.isin(FOCUS)][["model","ap_mean","ap_fold_sd","auc_mean","f1_mean","precision_mean","recall_mean","mcc_mean"]]
    notes = [f"# {dataset} 연구 실험 결과", "", f"완료: {len(completed)}/15회(5개 시간 구간 × 3개 학습 시드).",
             f"중복 시드를 제외한 테스트 후보 {len(counts):,}건, 양성 {int(counts.label.sum()):,}건, 양성률 {counts.label.mean():.2%}.", "",
             f"테스트 관측 범위: {dates['first_test_cutoff_utc']}부터 {dates['last_test_label_end_utc']}까지. 과거 자료 안의 미래 구간을 예측한 것이며 현재 시점 서비스 예측이 아니다.", "",
             "다음 표는 각 구간에서 3개 시드를 평균한 뒤 5개 구간을 평균한 값이다. ap_fold_sd는 구간 간 표준편차이며 신뢰구간이 아니다.", "",
             markdown_table(table), "", "## 사전 지정 비교와 시간 블록 부트스트랩", "",
             "시드 평균 예측을 모은 AP 차이에 대해 날짜 블록을 재표집했다. 95% 구간이 0을 넘는지 확인한다. 7/14/28일 블록 민감도를 함께 제시한다. 이 구간은 반복 관측의 모든 의존성을 제거한다고 보장하지 않는다. 유형 제거와 앙상블 비교는 탐색적 보조 분석이다.", "",
             markdown_table(pd.DataFrame(comparisons)[["model_a","model_b","delta_ap","ci_low","ci_high","block_days"]]), "",
             "## 해석 범위", "", "실제 후보 전체 평가이다. 균형 F1과 직접 비교하지 않는다. AP는 average precision 계산이며, 사다리꼴 PR 면적과 같은 수치라고 가정하지 않는다.",
             "Brier는 확률을 출력하는 모델에만 계산한다. 원시 빈도·CN·AA·Jaccard 점수에는 적용하지 않는다.",
             "TGN recurrence는 GRU 메모리와 시간 이웃 attention을 사용한 과제별 구현이다. 원 논문의 공식 재현 점수나 공개 벤치마크 순위라고 주장할 수 없다.",
             "GraphSAGE는 각 예측 시점의 최근 10개 이웃과 누적 노드 활동으로 갱신한다. 모델 가중치는 테스트에서 갱신하지 않는다.",
             "유형/시간감쇠/구조 제거 실험은 전체 LightGBM과 같은 하이퍼파라미터를 사용하므로 구성요소의 조건부 효과를 측정한다. 각 제거 모델의 최적 성능 탐색은 아니다.",
             "MathOverflow는 이전 실험 결과를 이미 확인한 자료이므로 후향적 연구로 표시한다. AskUbuntu는 같은 과제를 별도 데이터에서 다시 학습·평가한 재현 실험이며 제로샷 이전 실험은 아니다."]
    notes.append("두 데이터 모두 Stack Exchange 계열의 질문·답변 커뮤니티이므로 다른 종류의 네트워크까지 일반화했다고 주장하지 않는다.")
    (output/"report_ko.md").write_text("\n".join(notes),encoding="utf-8")
    print(summary[["model","ap_mean","auc_mean","f1_mean"]].to_string(index=False),flush=True)
    return result


def _plots(output,summary,per_fold,ensemble):
    focus = summary[summary.model.isin(FOCUS)].iloc[::-1]
    fig,ax=plt.subplots(figsize=(10,6))
    ax.barh(focus.model,focus.ap_mean,xerr=focus.ap_fold_sd.fillna(0),color="#217A8B",capsize=3)
    ax.set_xlabel("Average precision (mean +/- SD across time folds)")
    ax.set_title(f"{output.name}: {per_fold.fold.nunique()} temporal folds / natural prevalence")
    baseline = summary.loc[summary.model=="Train prevalence","ap_mean"].iloc[0]
    ax.axvline(baseline,color="#555555",linestyle="--",label="Constant-score AP (mean prevalence)")
    ax.legend(fontsize=8,loc="lower right")
    ax.grid(axis="x",alpha=.2); fig.tight_layout(); fig.savefig(output/"ap_comparison.png",dpi=200); plt.close(fig)
    fig,ax=plt.subplots(figsize=(9,5))
    for name in ("History HGB","Full LightGBM","TGN recurrence","Hybrid ensemble"):
        g=per_fold[per_fold.model==name]
        ax.plot(g.fold+1,g.ap,marker="o",label=name)
    baseline_folds=per_fold[per_fold.model=="Train prevalence"]
    ax.plot(baseline_folds.fold+1,baseline_folds.ap,linestyle="--",color="gray",label="Test prevalence")
    ax.set_xlabel("Chronological test fold");ax.set_ylabel("AP (seed mean)")
    ax.set_xticks(range(1,6));ax.grid(alpha=.2);ax.legend();fig.tight_layout()
    fig.savefig(output/"fold_stability.png",dpi=200);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(12,5))
    calibration_max = .25
    for name in ("History HGB","Full LightGBM","TGN recurrence","Hybrid ensemble"):
        g=ensemble[ensemble.model==name]
        if g.empty:continue
        p,r,_=precision_recall_curve(g.label,g.score)
        axes[0].plot(r,p,label=name)
        observed,predicted=calibration_curve(g.label,g.score,n_bins=8,strategy="quantile")
        axes[1].plot(predicted,observed,marker="o",label=name)
        calibration_max=max(calibration_max,float(observed.max()),float(predicted.max()))
    prevalence=ensemble.loc[ensemble.model=="Full LightGBM","label"].mean()
    axes[0].axhline(prevalence,color="gray",linestyle="--",label="Prevalence")
    axes[0].set(xlabel="Recall",ylabel="Precision",title="OOF seed-ensemble PR curves")
    limit=min(1.,np.ceil((calibration_max+.02)*10)/10)
    axes[1].plot([0,limit],[0,limit],"k--",alpha=.4)
    axes[1].set(xlabel="Predicted probability",ylabel="Observed fraction",title="Calibration (pooled test; zoomed)",xlim=(0,limit),ylim=(0,limit))
    for ax in axes:ax.grid(alpha=.2);ax.legend(fontsize=8)
    fig.tight_layout();fig.savefig(output/"pr_and_calibration.png",dpi=200);plt.close(fig)


def _comparison_plot(output, comparisons):
    rows=[c for c in comparisons if c["block_days"]==7]
    fig,ax=plt.subplots(figsize=(10,4))
    for i,c in enumerate(rows):
        ax.plot([c["ci_low"],c["ci_high"]],[i,i],color="#217A8B",lw=3)
        ax.plot(c["delta_ap"],i,"o",color="#DF6C45")
    ax.axvline(0,color="gray",linestyle="--")
    ax.set_yticks(range(len(rows)),[c["model_a"]+"\nminus "+c["model_b"] for c in rows])
    ax.set_xlabel("Pooled AP difference and paired 95% bootstrap interval")
    ax.set_title(f"{output.name}: 7-day blocks; first contrast is primary")
    ax.invert_yaxis();ax.grid(axis="x",alpha=.2);fig.tight_layout()
    fig.savefig(output/"improvement_intervals.png",dpi=200);plt.close(fig)


if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--dataset",default="mathoverflow")
    p.add_argument("--repeats",type=int,default=1000);p.add_argument("--allow-partial",action="store_true")
    a=p.parse_args();aggregate(a.dataset,a.repeats,not a.allow_partial)
