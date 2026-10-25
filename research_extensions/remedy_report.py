"""Verify and report activity/choice and expert remedies."""
from __future__ import annotations

import json
import hashlib
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from research.data import ROOT
from research.evaluation import metrics
from research.report import markdown_table
from research_extensions.remedy_run import MODELS
from research_extensions.uncertainty import compare

OUT=ROOT/"results"/"remedy_v1"
BASELINE="Baseline Full LightGBM"


def collect(dataset):
    folder=OUT/dataset
    completed=list(folder.glob("fold_*/seed_*/complete.json"))
    if len(completed)!=15:raise RuntimeError(f"{dataset}: expected 15 runs, found {len(completed)}")
    signature=json.loads(completed[0].read_text())["signature"]
    rows=[];activity=[];seed_frames={}
    for seed in (2026,2027,2028):
        model_parts={name:[] for name in [BASELINE,*MODELS]}
        for fold in range(5):
            run=folder/f"fold_{fold}"/f"seed_{seed}"
            complete=json.loads((run/"complete.json").read_text())
            assert complete["signature"]==signature
            p=pd.read_csv(run/"predictions.csv.gz",float_precision="round_trip")
            m=pd.read_csv(run/"metrics.csv",float_precision="round_trip")
            bdir=ROOT/"results"/"lookback_density_v1"/dataset/"lookback_24h_train_6h"/f"fold_{fold}"/f"seed_{seed}"
            bp=pd.read_csv(bdir/"predictions.csv.gz",float_precision="round_trip")
            bp=bp[bp.model=="Full LightGBM"].copy();bp["model"]=BASELINE
            bm=pd.read_csv(bdir/"metrics.csv",float_precision="round_trip")
            bm=bm[bm.model=="Full LightGBM"].copy();bm["model"]=BASELINE
            p=pd.concat([bp,p],ignore_index=True);m=pd.concat([bm,m],ignore_index=True)
            for (name,cohort),measurement in m.set_index(["model","cohort"]).iterrows():
                g=p[p.model==name]
                if cohort=="common_4h":g=g[g.common_4h]
                recomputed=metrics(g.label.to_numpy(),g.score.to_numpy(),float(measurement.threshold),g.cutoff.to_numpy())
                for key in ("ap","auc","f1","precision","recall","mcc","brier"):
                    assert np.isclose(recomputed[key],measurement[key],atol=1e-10),(dataset,fold,seed,name,cohort,key)
                rows.append({"dataset":dataset,"fold":fold,"seed":seed,"model":name,"cohort":cohort,**recomputed})
            for name,g in p.groupby("model"):
                g=g[["fold","cutoff","src","dst","label","common_4h","score"]].copy()
                if g.duplicated(["fold","cutoff","src","dst"]).any():raise AssertionError("duplicate predictions")
                model_parts[name].append(g)
            activity.append(pd.read_csv(run/"activity_metrics.csv"))
        for name,parts in model_parts.items():seed_frames[(seed,name)]=pd.concat(parts,ignore_index=True)
    all_metrics=pd.DataFrame(rows);all_metrics.to_csv(folder/"all_metrics.csv",index=False)
    per_fold=all_metrics.groupby(["model","cohort","fold"]).agg(
        ap=("ap","mean"),auc=("auc","mean"),f1=("f1","mean"),precision=("precision","mean"),
        recall=("recall","mean"),mcc=("mcc","mean"),brier=("brier","mean")).reset_index()
    summary=per_fold.groupby(["model","cohort"]).agg(
        ap_mean=("ap","mean"),ap_fold_sd=("ap","std"),auc_mean=("auc","mean"),f1_mean=("f1","mean"),
        precision_mean=("precision","mean"),recall_mean=("recall","mean"),mcc_mean=("mcc","mean"),
        brier_mean=("brier","mean")).reset_index()
    counts=seed_frames[(2026,BASELINE)].groupby("common_4h").label.agg(["size","sum"])
    summary["test_n"]=summary.cohort.map({"all":int(counts['size'].sum()),"common_4h":int(counts.loc[True,'size'])})
    summary["test_positive"]=summary.cohort.map({"all":int(counts['sum'].sum()),"common_4h":int(counts.loc[True,'sum'])})
    summary.to_csv(folder/"model_summary.csv",index=False);per_fold.to_csv(folder/"fold_metrics.csv",index=False)
    act=pd.concat(activity,ignore_index=True)
    act_summary=act.groupby("fold").agg(ap=("ap","mean"),auc=("auc","mean"),f1=("f1","mean"),
                                        precision=("precision","mean"),recall=("recall","mean"),mcc=("mcc","mean"),brier=("brier","mean")).reset_index()
    act_summary.to_csv(folder/"activity_model_summary.csv",index=False)
    ensembles={}
    for name in [BASELINE,*MODELS]:
        frames=[seed_frames[(seed,name)] for seed in (2026,2027,2028)]
        average=frames[0].copy()
        for frame in frames[1:]:
            pd.testing.assert_frame_equal(average.drop(columns="score"),frame.drop(columns="score"))
            average["score"]+=frame.score.to_numpy()
        average["score"]/=3;ensembles[name]=average
    comparisons=[]
    for cohort in ("all","common_4h"):
        base=ensembles[BASELINE] if cohort=="all" else ensembles[BASELINE][ensembles[BASELINE].common_4h]
        for name in MODELS:
            frame=ensembles[name] if cohort=="all" else ensembles[name][ensembles[name].common_4h]
            comparisons.append({"model":name,"baseline":BASELINE,"cohort":cohort,**compare(frame,base)})
    comparisons=pd.DataFrame(comparisons);comparisons.to_csv(folder/"paired_comparisons.csv",index=False)
    plot(folder,summary,comparisons)
    output={"dataset":dataset,"status":"complete","completed_runs":15,"verified_model_cohort_metrics":len(all_metrics),
            "signature":signature,"models":summary.to_dict("records"),"activity":act_summary.to_dict("records"),
            "comparisons":comparisons.to_dict("records"),"scope":"post-hoc exploratory; both tests inspected before model design"}
    (folder/"summary.json").write_text(json.dumps(output,indent=2),encoding="utf-8")
    write_report(dataset,summary,comparisons,act_summary)
    return output


def plot(folder,summary,comparisons):
    g=summary[summary.cohort=="all"].set_index("model").loc[[BASELINE,*MODELS]].reset_index()
    labels=["Baseline","Activity→choice","Recency/type\nexperts","Combined"]
    fig,axes=plt.subplots(1,2,figsize=(11,4.5))
    for ax,key,title,color in ((axes[0],"ap_mean","Average precision","#217A8B"),(axes[1],"f1_mean","F1 at calibration threshold","#BA654A")):
        bars=ax.bar(labels,g[key],color=color);ax.bar_label(bars,fmt="%.3f",padding=3);ax.set_title(title);ax.set_ylim(0,max(g[key])*1.22);ax.grid(axis="y",alpha=.2)
    fig.suptitle(folder.name+": 24h lookback, 6h training, five time folds / three seeds")
    fig.tight_layout();fig.savefig(folder/"remedy_comparison.png",dpi=170);plt.close(fig)
    c=comparisons[comparisons.cohort=="all"]
    fig,ax=plt.subplots(figsize=(8,4))
    y=np.arange(len(c));err=np.vstack([c.delta_ap-c.ci_low,c.ci_high-c.delta_ap])
    ax.errorbar(c.delta_ap,y,xerr=err,fmt="o",color="#217A8B",capsize=4);ax.axvline(0,color="black",lw=1)
    ax.set_yticks(y,["Activity→choice","Recency/type experts","Combined"]);ax.set_xlabel("AP difference vs frozen baseline (95% paired 7-day block interval)")
    ax.grid(axis="x",alpha=.2);fig.tight_layout();fig.savefig(folder/"ap_differences.png",dpi=170);plt.close(fig)


def write_report(dataset,summary,comparisons,activity):
    folder=OUT/dataset
    table=summary[summary.cohort=="all"][["model","ap_mean","auc_mean","f1_mean","precision_mean","recall_mean","mcc_mean","brier_mean","test_n","test_positive"]]
    common=summary[summary.cohort=="common_4h"][["model","ap_mean","f1_mean","precision_mean","recall_mean"]]
    lines=[f"# {dataset}: 오류 개선 구조 실험","",
           "기존 24시간 후보·6시간 학습 Full LightGBM과 두 요청 모델 및 결합형을 같은 시간 분할에서 비교했습니다.",
           "두 데이터셋의 테스트 오류를 본 뒤 설계했으므로 모두 탐색 결과이며 새 독립 확인 결과가 아닙니다.","",
           "## 넓어진 24시간 후보 전체","",markdown_table(table),"",
           "## 동일한 기존 4시간 후보","",markdown_table(common),"",
           "## 기준모델과 AP 차이","",markdown_table(comparisons),"",
           "## 1단계 사용자 활동 모델 자체의 성능","",markdown_table(activity),"",
           "## 모델 정의","",
           "- Two-stage activity-choice: 과거의 사용자 활동 특징만으로 A가 다음 24시간 행동할 확률을 구하고, 활동한 학습 사례로 학습한 A→B 재연결 조건부 점수와 곱합니다.",
           "- Recency-type experts: 마지막 연결이 0~4시간/4~24시간인지와 마지막 행동 유형 a2q/c2a/c2q를 교차한 여섯 모델입니다.",
           "- Two-stage recency-type experts: 활동 모델과 조건부 여섯 전문가를 결합했습니다.",
           "- 미래 활동은 과거 학습 구간의 정답으로만 사용합니다. 테스트 시점에는 과거 특징만 사용하며 실제 미래 활동을 입력하지 않습니다.","",
           "## 제한","",
           "- 기존 fold별 LightGBM 설정을 고정해서 구조만 비교했습니다. 테스트 점수로 모델·설정·임계값을 고르지 않았습니다.",
           "- F1 임계값은 각 fold의 테스트 이전 보정 구간에서 정했습니다.",
           "- 시드 평균 뒤 같은 후보에 대한 AP를 7일 시간 블록으로 1,000회 재표집했습니다. 시간 의존성이 더 길게 남을 수 있습니다.",
           "- 후보는 과거 24시간에 연결됐던 관계로 제한됩니다. 완전히 새로운 관계 예측 결과가 아닙니다.",
           "- 이 결과로 인과관계, SOTA 또는 새 데이터 성능을 주장하지 않습니다."]
    (folder/"report_ko.md").write_text("\n".join(lines),encoding="utf-8")


if __name__=="__main__":
    results={dataset:collect(dataset) for dataset in ("mathoverflow","askubuntu")}
    (OUT/"summary.json").write_text(json.dumps({"status":"complete","datasets":results},indent=2),encoding="utf-8")
    print("Remedy report complete and verified.")
