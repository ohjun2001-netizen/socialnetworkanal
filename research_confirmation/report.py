"""One-shot final report for the preregistered Super User confirmation."""
from __future__ import annotations

from datetime import datetime,timezone
import hashlib
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from research.data import ROOT
from research.evaluation import metrics
from research.report import markdown_table
from research_extensions.uncertainty import compare
from .run import MODELS

OUT=ROOT/"results"/"confirmation_superuser_v1"


def main():
    final=OUT/"summary.json"
    if final.exists():raise RuntimeError("Final Super User report already exists; refusing to overwrite the first confirmation result")
    protocol=json.loads((ROOT/"research_confirmation"/"superuser_protocol.json").read_text())
    completed=list(OUT.glob("fold_*/seed_*/complete.json"))
    if len(completed)!=15:raise RuntimeError(f"expected 15 locked runs, found {len(completed)}")
    signatures={json.loads(path.read_text())["signature"] for path in completed}
    if len(signatures)!=1:raise AssertionError("run signature mismatch")
    rows=[];activity_rows=[];seed_frames={}
    for seed in protocol["seeds"]:
        model_parts={name:[] for name in MODELS}
        for fold in range(5):
            folder=OUT/f"fold_{fold}"/f"seed_{seed}"
            locked=folder/"locked_test_scores.npz"
            before=hashlib.sha256(locked.read_bytes()).hexdigest()
            z=np.load(locked,allow_pickle=False)
            keys={name:z[name] for name in ("cutoff","src","dst","label","common_4h")}
            for number,name in enumerate(MODELS):
                frame=pd.DataFrame(keys);frame["score"]=z[f"score_{number}"];frame["fold"]=fold
                threshold_value=json.loads((folder/"thresholds.json").read_text())[name]
                for cohort,select in (("all",np.ones(len(frame),bool)),("common_4h",frame.common_4h.to_numpy(bool))):
                    value=metrics(frame.label.to_numpy()[select],frame.score.to_numpy()[select],threshold_value,frame.cutoff.to_numpy()[select])
                    rows.append({"dataset":"superuser","fold":fold,"seed":seed,"model":name,"cohort":cohort,**value})
                model_parts[name].append(frame)
            activity_rows.append({"dataset":"superuser","fold":fold,"seed":seed,
                **metrics(z["activity_label"],z["activity_score"],float(z["activity_threshold"][0]),z["activity_cutoff"])})
            if hashlib.sha256(locked.read_bytes()).hexdigest()!=before:raise AssertionError("locked score file changed during reporting")
        for name,parts in model_parts.items():seed_frames[(seed,name)]=pd.concat(parts,ignore_index=True)
    all_metrics=pd.DataFrame(rows);all_metrics.to_csv(OUT/"all_metrics.csv",index=False)
    per_fold=all_metrics.groupby(["model","cohort","fold"]).agg(
        ap=("ap","mean"),auc=("auc","mean"),f1=("f1","mean"),precision=("precision","mean"),recall=("recall","mean"),
        mcc=("mcc","mean"),brier=("brier","mean"),prevalence=("prevalence","mean"),ap_lift=("ap_lift","mean")).reset_index()
    summary=per_fold.groupby(["model","cohort"]).agg(
        ap_mean=("ap","mean"),ap_fold_sd=("ap","std"),auc_mean=("auc","mean"),f1_mean=("f1","mean"),
        precision_mean=("precision","mean"),recall_mean=("recall","mean"),mcc_mean=("mcc","mean"),brier_mean=("brier","mean"),
        prevalence=("prevalence","mean"),ap_lift_mean=("ap_lift","mean")).reset_index()
    reference=seed_frames[(protocol["seeds"][0],MODELS[0])]
    for cohort,select in (("all",np.ones(len(reference),bool)),("common_4h",reference.common_4h.to_numpy(bool))):
        mask=summary.cohort==cohort;summary.loc[mask,"test_n"]=int(select.sum());summary.loc[mask,"test_positive"]=int(reference.loc[select,"label"].sum())
    summary[["test_n","test_positive"]]=summary[["test_n","test_positive"]].astype(int)
    per_fold.to_csv(OUT/"fold_metrics.csv",index=False);summary.to_csv(OUT/"model_summary.csv",index=False)
    activity=pd.DataFrame(activity_rows)
    activity_summary=activity.groupby("fold").agg(ap=("ap","mean"),auc=("auc","mean"),f1=("f1","mean"),precision=("precision","mean"),
        recall=("recall","mean"),mcc=("mcc","mean"),brier=("brier","mean"),n=("n","max"),positive=("positive","max")).reset_index()
    activity_summary.to_csv(OUT/"activity_model_summary.csv",index=False)
    ensembles={}
    for name in MODELS:
        frames=[seed_frames[(seed,name)] for seed in protocol["seeds"]];average=frames[0].copy()
        for frame in frames[1:]:
            pd.testing.assert_frame_equal(average.drop(columns="score"),frame.drop(columns="score"))
            average["score"]+=frame.score.to_numpy()
        average["score"]/=len(frames);ensembles[name]=average
    comparisons=[]
    for cohort in ("all","common_4h"):
        base=ensembles[MODELS[0]] if cohort=="all" else ensembles[MODELS[0]][ensembles[MODELS[0]].common_4h]
        for name in MODELS[1:]:
            candidate=ensembles[name] if cohort=="all" else ensembles[name][ensembles[name].common_4h]
            comparisons.append({"model":name,"baseline":MODELS[0],"cohort":cohort,"primary":name=="Recency-type experts" and cohort=="all",**compare(candidate,base)})
    comparisons=pd.DataFrame(comparisons);comparisons.to_csv(OUT/"paired_comparisons.csv",index=False)
    primary=comparisons[comparisons.primary].iloc[0]
    all_summary=summary[summary.cohort=="all"].set_index("model")
    f1_delta=float(all_summary.loc["Recency-type experts","f1_mean"]-all_summary.loc[MODELS[0],"f1_mean"])
    success=bool(primary.ci_low>0 and f1_delta>=-0.005)
    draw(summary,comparisons)
    generated=datetime.now(timezone.utc).isoformat()
    output={"dataset":"superuser","status":"complete","first_final_report":True,"generated_at":generated,
            "run_signature":next(iter(signatures)),"protocol_sha256":hashlib.sha256((ROOT/"research_confirmation"/"superuser_protocol.json").read_bytes()).hexdigest(),
            "completed_runs":15,"verified_model_cohort_metrics":len(all_metrics),"models":summary.to_dict("records"),
            "activity":activity_summary.to_dict("records"),"comparisons":comparisons.to_dict("records"),
            "primary_success":success,"primary_ap_delta":float(primary.delta_ap),"primary_ci":[float(primary.ci_low),float(primary.ci_high)],
            "primary_f1_delta":f1_delta,"interpretation":"confirmatory success" if success else "preregistered success rule not met"}
    write_report(output,summary,comparisons,activity_summary)
    temporary=final.with_suffix(".tmp");temporary.write_text(json.dumps(output,indent=2),encoding="utf-8");temporary.replace(final)
    print("FIRST FINAL SUPER USER REPORT COMPLETE",flush=True)
    print(json.dumps({"primary_success":success,"primary_ap_delta":output["primary_ap_delta"],"primary_ci":output["primary_ci"],"primary_f1_delta":f1_delta}),flush=True)


def draw(summary,comparisons):
    g=summary[summary.cohort=="all"].set_index("model").loc[MODELS].reset_index()
    labels=["Baseline","Activity→choice","Recency/type\nexperts","Combined"]
    fig,axes=plt.subplots(1,2,figsize=(11,4.5))
    for ax,key,title,color in ((axes[0],"ap_mean","Average precision","#217A8B"),(axes[1],"f1_mean","F1 at calibration threshold","#BA654A")):
        bars=ax.bar(labels,g[key],color=color);ax.bar_label(bars,fmt="%.3f",padding=3);ax.set_title(title);ax.set_ylim(0,max(g[key])*1.22);ax.grid(axis="y",alpha=.2)
    fig.suptitle("Super User untouched confirmation: five time folds / three seeds")
    fig.tight_layout();fig.savefig(OUT/"model_comparison.png",dpi=170);plt.close(fig)
    c=comparisons[comparisons.cohort=="all"]
    fig,ax=plt.subplots(figsize=(8,4));y=np.arange(len(c));err=np.vstack([c.delta_ap-c.ci_low,c.ci_high-c.delta_ap])
    ax.errorbar(c.delta_ap,y,xerr=err,fmt="o",color="#217A8B",capsize=4);ax.axvline(0,color="black",lw=1)
    ax.set_yticks(y,["Activity→choice","Recency/type experts","Combined"]);ax.set_xlabel("AP difference vs baseline (95% paired 7-day block interval)")
    ax.grid(axis="x",alpha=.2);fig.tight_layout();fig.savefig(OUT/"ap_differences.png",dpi=170);plt.close(fig)


def write_report(output,summary,comparisons,activity):
    table=summary[summary.cohort=="all"][["model","ap_mean","ap_fold_sd","auc_mean","f1_mean","precision_mean","recall_mean","mcc_mean","brier_mean","prevalence","ap_lift_mean","test_n","test_positive"]]
    lines=["# Super User 독립 확인 실험","",f"첫 최종 보고서 생성: {output['generated_at']}","",
           f"사전등록 성공 조건 충족: **{'예' if output['primary_success'] else '아니오'}**","",
           "## 24시간 후보 전체 결과","",markdown_table(table),"","## 기준모델 대비 AP 차이","",markdown_table(comparisons),"",
           "## 사용자 활동 1단계 모델","",markdown_table(activity),"","## 사전등록 판단","",
           f"주 비교인 최근성·유형 전문가의 통합 AP 차이는 {output['primary_ap_delta']:+.6f}, 95% 구간 [{output['primary_ci'][0]:+.6f}, {output['primary_ci'][1]:+.6f}]입니다.",
           f"5구간 평균 F1 차이는 {output['primary_f1_delta']:+.6f}입니다. 성공 조건은 AP 구간 전체가 0보다 높고 F1 차이가 -0.005 이상인 것입니다.","",
           "## 정당성","",
           "- Super User 원본을 받기 전에 프로토콜·주 가설·성공 기준을 파일로 고정했습니다.",
           "- fold별 설정은 훈련·튜닝 구간에서만 정하고, 확률 보정과 F1 임계값은 테스트 이전 보정 구간에서만 정했습니다.",
           "- 15개 실행이 모두 잠긴 뒤 이 보고서에서 처음 테스트 지표를 계산했습니다. 첫 보고서는 덮어쓰지 않습니다.",
           "- 같은 후보의 시드 평균 점수를 7일 시간 블록으로 1,000회 재표집했습니다.",
           "- Super User 내부에서 새로 학습한 확인 실험이며 MathOverflow 모델의 제로샷 이전은 아닙니다.",
           "- 보조 모델과 기존 4시간 후보 비교는 모두 보고하지만 주 성공 판단을 바꾸지 않습니다."]
    (OUT/"report_ko.md").write_text("\n".join(lines),encoding="utf-8")


if __name__=="__main__":main()
