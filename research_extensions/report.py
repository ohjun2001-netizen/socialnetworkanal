"""Verify and report every arm; never pick an arm by its test result."""
import json
import hashlib
import importlib.metadata
from datetime import datetime,timezone
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from research.data import ROOT,load
from research.evaluation import metrics
from research.report import markdown_table
from .uncertainty import compare

OUT=ROOT/"results"/"lookback_density_v1"


def collect(dataset):
    folder=OUT/dataset
    if len(list(folder.glob("lookback_*/fold_*/seed_*/complete.json")))!=90:
        raise RuntimeError(f"{dataset}: not all 90 condition/fold/seed records complete")
    tables=[];ensembles={};counts=[];verified=0
    source=ROOT/"research_extensions"
    source_hashes={name:hashlib.sha256((source/name).read_bytes()).hexdigest() for name in ("features.py","run.py")}
    configs=json.loads((folder/"fixed_configs.json").read_text())
    expected_signature=hashlib.sha256(((source/"protocol.json").read_text()+json.dumps(configs,sort_keys=True)+source_hashes["features.py"]+source_hashes["run.py"]).encode()).hexdigest()
    feature_validation=json.loads((folder/"feature_validation.json").read_text())
    assert feature_validation["source_sha256"]==source_hashes["features.py"]
    for lookback in (4,12,24):
        for stride in (24,6):
            condition=folder/f"lookback_{lookback}h_train_{stride}h"
            parts=[]
            for fold in range(5):
                average=None
                for seed in (2026,2027,2028):
                    run=condition/f"fold_{fold}"/f"seed_{seed}"
                    assert json.loads((run/"complete.json").read_text())["signature"]==expected_signature
                    measurements=pd.read_csv(run/"metrics.csv",float_precision="round_trip")
                    tables.append(measurements)
                    p=pd.read_csv(run/"predictions.csv.gz",float_precision="round_trip")
                    assert not p.duplicated(["model","cutoff","src","dst"]).any()
                    assert (p.cutoff%86400==0).all() and np.isfinite(p.score).all()
                    for (name,cohort),row in measurements.set_index(["model","cohort"]).iterrows():
                        g=p[p.model==name]
                        if cohort=="common_4h":g=g[g.common_4h]
                        np.testing.assert_array_equal(g.prediction,g.score>=row.threshold)
                        recomputed=metrics(g.label.to_numpy(),g.score.to_numpy(),row.threshold)
                        for key in ("ap","auc","f1","precision","recall","mcc","brier"):
                            assert np.isclose(recomputed[key],row[key],atol=1e-10),(dataset,lookback,stride,fold,seed,name,cohort,key)
                        verified+=1
                    g=p[p.model=="Full LightGBM"].drop(columns=["prediction","model","seed"]).reset_index(drop=True)
                    if average is None:average=g
                    else:
                        pd.testing.assert_frame_equal(average.drop(columns="score"),g.drop(columns="score"))
                        average["score"]=average.score.to_numpy()+g.score.to_numpy()
                average["score"]/=3;parts.append(average)
            combined=pd.concat(parts,ignore_index=True)
            assert not combined.duplicated(["cutoff","src","dst"]).any()
            ensembles[(lookback,stride)]=combined
            last_split=json.loads((condition/"fold_4"/"split.json").read_text())["partitions"]["train"]
            for cohort in ("all","common_4h"):
                g=combined if cohort=="all" else combined[combined.common_4h]
                counts.append({"lookback":lookback,"train_stride":stride,"cohort":cohort,"test_n":len(g),"test_positive":int(g.label.sum()),
                               "prevalence":float(g.label.mean()),"last_fold_train_n":last_split["n"],"last_fold_train_positive":last_split["positive"],
                               "last_fold_train_cutoffs":last_split["cutoffs"]})
    all_metrics=pd.concat(tables,ignore_index=True)
    all_metrics.to_csv(folder/"all_metrics.csv",index=False)
    per_fold=all_metrics.groupby(["lookback","train_stride","model","cohort","fold"]).agg(
        ap=("ap","mean"),auc=("auc","mean"),f1=("f1","mean"),precision=("precision","mean"),recall=("recall","mean"),mcc=("mcc","mean"),brier=("brier","mean")).reset_index()
    summary=per_fold.groupby(["lookback","train_stride","model","cohort"]).agg(
        ap_mean=("ap","mean"),ap_fold_sd=("ap","std"),auc_mean=("auc","mean"),f1_mean=("f1","mean"),
        precision_mean=("precision","mean"),recall_mean=("recall","mean"),mcc_mean=("mcc","mean"),brier_mean=("brier","mean")).reset_index()
    summary=summary.merge(pd.DataFrame(counts),on=["lookback","train_stride","cohort"],validate="many_to_one")
    summary.to_csv(folder/"model_summary.csv",index=False);per_fold.to_csv(folder/"fold_metrics.csv",index=False)
    pd.DataFrame(counts).to_csv(folder/"sample_counts.csv",index=False)
    # Changing training density must not change any test candidate or label.
    keys=["fold","cutoff","src","dst","label","common_4h"]
    for lookback in (4,12,24):
        pd.testing.assert_frame_equal(ensembles[(lookback,24)][keys],ensembles[(lookback,6)][keys])
    reference=ensembles[(4,24)]
    for frame in ensembles.values():
        pd.testing.assert_frame_equal(frame.loc[frame.common_4h,keys].reset_index(drop=True),reference[keys])
    # Independently reconstruct labels and past-candidate presence from raw rows.
    events,_=load(dataset);rng=np.random.default_rng(904)
    widest=ensembles[(24,6)]
    sample=widest.iloc[rng.choice(len(widest),min(1000,len(widest)),replace=False)]
    for row in sample.itertuples():
        pair=events[(events.src==row.src)&(events.dst==row.dst)].time
        assert ((pair>=row.cutoff-86400)&(pair<row.cutoff)).any()
        assert int(((pair>=row.cutoff)&(pair<row.cutoff+86400)).any())==row.label
    comparisons=[]
    for lookback in (4,12,24):
        result=compare(ensembles[(lookback,6)],ensembles[(lookback,24)])
        comparisons.append({"comparison":"train_6h_minus_24h","lookback":lookback,"train_stride":None,"cohort":"all",**result})
    for stride in (24,6):
        for lookback in (12,24):
            a=ensembles[(lookback,stride)];b=ensembles[(4,stride)]
            result=compare(a[a.common_4h],b)
            comparisons.append({"comparison":"wider_training_minus_4h_on_common","lookback":lookback,"train_stride":stride,"cohort":"common_4h",**result})
    widest=ensembles[(24,6)]
    result=compare(widest[widest.common_4h],ensembles[(4,24)])
    comparisons.append({"comparison":"combined_24h6h_minus_original_on_common","lookback":24,"train_stride":6,"cohort":"common_4h",**result})
    pd.DataFrame(comparisons).to_csv(folder/"paired_comparisons.csv",index=False)
    plot(folder,summary,pd.DataFrame(counts))
    output={"dataset":dataset,"complete_records":90,"verified_model_cohort_metrics":verified,"raw_label_checks":len(sample),
            "source_sha256":source_hashes,"configuration_signature":expected_signature,"feature_validation":feature_validation,
            "package_versions":{p:importlib.metadata.version(p) for p in ("numba","numpy","pandas","scikit-learn","lightgbm","scipy")},
            "models":summary.to_dict("records"),"counts":counts,"comparisons":comparisons,
            "scope":"retrospective exploratory sensitivity study; unadjusted exploratory intervals",
            "test_cohort_checks":"identical between training strides; identical common 4h cohort across all arms"}
    (folder/"summary.json").write_text(json.dumps(output,indent=2),encoding="utf-8")
    return output


def plot(folder,summary,counts):
    fig,axes=plt.subplots(1,2,figsize=(12,5),sharey=True)
    for ax,cohort,title in zip(axes,("all","common_4h"),("Expanded candidates (prevalence changes)","Identical 4h candidates (matched comparison)")):
        for model,color in (("Full LightGBM","#217A8B"),("History HGB","#BA654A")):
            for stride,style in ((24,"--"),(6,"-")):
                g=summary[(summary.model==model)&(summary.cohort==cohort)&(summary.train_stride==stride)].sort_values("lookback")
                ax.plot(g.lookback,g.ap_mean,marker="o",linestyle=style,color=color,label=f"{model}, train {stride}h")
        ax.set_title(title);ax.set_xlabel("Candidate lookback (hours)");ax.set_xticks([4,12,24]);ax.grid(alpha=.2);ax.legend(fontsize=8)
    axes[0].set_ylabel("Mean AP across five time folds / three seeds")
    fig.suptitle(folder.name+": 24h target, daily non-overlapping evaluation")
    fig.tight_layout();fig.savefig(folder/"lookback_ap.png",dpi=180);plt.close(fig)
    g=counts[(counts.cohort=="all")&(counts.train_stride==24)].sort_values("lookback")
    fig,axes=plt.subplots(1,2,figsize=(10,4))
    for ax,key,title in zip(axes,("test_n","test_positive"),("Test candidate pair-days","Test positive pair-days")):
        bars=ax.bar(g.lookback.astype(str),g[key],color="#217A8B")
        ax.bar_label(bars,labels=[f"{n:,}" for n in g[key]],padding=3);ax.margins(y=.15)
        ax.set_title(title);ax.set_xlabel("Candidate lookback (hours)")
    fig.suptitle(folder.name+": coverage, not duplicated test rows")
    fig.tight_layout();fig.savefig(folder/"candidate_coverage.png",dpi=180);plt.close(fig)


def main():
    results={dataset:collect(dataset) for dataset in ("mathoverflow","askubuntu")}
    lines=["# 후보 기간·학습 간격 확장 결과", "", "## 무엇을 비교했나", "",
           "MathOverflow·AskUbuntu에서 후보 기간 4/12/24시간 × 학습 간격 24/6시간의 6개 조합을 모두 비교했다. 정답은 다음 24시간 동일 방향 상호작용 재발생이며, 튜닝·보정·테스트는 UTC 자정 24시간 간격이다.",
           "5개 시간 경계와 3개 시드를 기존 v2와 동일하게 유지했다. 기존 4h/24h 결과는 재사용했다. LightGBM·이력 HGB만 고정 비교하며 기존 GNN 실험은 보존했다. HGB는 결정적 실행이라 같은 예측을 시드별 표에 재사용했다.",
           "하이퍼파라미터도 각 fold의 기존 4h/24h 기준 설정으로 고정했다. 이 실험은 그 설정 아래의 데이터 구성 효과이며 각 조합을 별도 최적화한 최대 성능 비교가 아니다.", "",
           "표의 AP/AUC/F1은 각 구간의 3시드 평균을 다시 5개 시간 구간에 대해 평균한 값이다. 신뢰구간은 시드 평균 점수를 모은 통합 AP 차이이므로 표의 평균 AP 차이와 수치가 조금 다를 수 있다.", "",
           "양성/음성 복제, SMOTE, 테스트 음성 재표집은 하지 않았다. 6시간 학습 사례는 정답 구간이 겹치므로 독립 표본 수나 새로운 원본 사건 수로 해석하지 않는다. 테스트 24시간 비중첩도 사용자 반복에 의한 모든 의존성을 없애지는 않는다."]
    for dataset,result in results.items():
        table=pd.DataFrame(result["models"]);full=table[table.model=="Full LightGBM"]
        lines += ["",f"## {dataset}","", "### 확장한 후보 전체", "",
                  markdown_table(full[full.cohort=="all"][["lookback","train_stride","test_n","test_positive","prevalence","ap_mean","auc_mean","f1_mean"]]), "",
                  "후보 범위에 따라 양성률과 난도가 바뀐다. 이 표의 AP만으로 어떤 모델이 더 우수해졌다고 단정하지 않는다.", "",
                  "### 동일한 기존 4시간 후보만 비교", "",
                  markdown_table(full[full.cohort=="common_4h"][["lookback","train_stride","test_n","ap_mean","auc_mean","f1_mean","precision_mean","recall_mean"]]), "",
                  "학습 후보를 넓혔을 때 원래의 동일한 예측 대상에서도 변화가 있는지 보는 표다. 판정 임계값은 각 조합의 전체 보정 후보에서 정한 값을 그대로 적용한다.", "",
                  "### 같은 후보의 AP 차이: 탐색적 95% 날짜 블록 구간", "",
                  markdown_table(pd.DataFrame(result["comparisons"])[["comparison","lookback","train_stride","cohort","delta_ap","ci_low","ci_high"]]), "",
                  f"예측 파일에서 {result['verified_model_cohort_metrics']}개 지표 행을 재계산했고, 원본 정답 {result['raw_label_checks']}건을 재검산했다. 학습 간격을 바꿔도 테스트 후보·정답이 동일함을 확인했다."]
    lines += ["", "## 해석과 주의", "",
              "1. 긴 실제 데이터와 별도 커뮤니티 검증은 이미 v2에 반영했다. 이번 확장은 같은 원본에서 후보와 학습 시점을 달리하는 분석이다.",
              "2. 넓은 후보의 양성 증가와, 같은 후보에서의 예측력 증가는 별개의 성과다.",
              "3. 6시간 학습이 도움이 되는지는 같은 lookback의 6h-24h 차이를 본다. 95% 구간이 0을 걸치면 명확한 개선으로 주장하지 않는다.",
              "4. 여러 비교의 구간은 다중비교 보정하지 않은 탐색적 구간이다. 최고 테스트 조합을 골라 새 최종 모델이라고 선언하지 않는다.",
              "5. 두 데이터의 기존 테스트를 이미 본 뒤 요청된 확장이므로 새 독립 확인 실험은 아니다. 결과가 좋아져도 새 기간/새 데이터 확인이 필요하다.",
              "6. RT·MT·RE가 아니라 답변/질문 댓글/답변 댓글 정보를 사용한다. 현재 정답은 특정 종류만이 아니라 동일 방향 쌍의 어떤 종류든 재발생하는 경우다."]
    (ROOT/"확장실험_결과.md").write_text("\n".join(lines),encoding="utf-8")
    easy=["후보 기간·학습 간격 확장 — 쉽게 읽는 결과", "",
          "MathOverflow·AskUbuntu를 유지했습니다. 원본 사건을 복사하거나 가짜 관계를 만들지 않았습니다.",
          "최근 관계 4/12/24시간 × 학습 간격 24/6시간의 여섯 조합을 모두 비교했습니다. 정답과 최종 테스트 간격은 계속 24시간입니다.",
          "아래는 최고 점수 조합을 고른 것이 아니라, 기존 기준과 요청한 가장 넓고 촘촘한 조합을 미리 정한 방식으로 설명한 것입니다.",""]
    for dataset,result in results.items():
        table=pd.DataFrame(result["models"]);table=table[table.model=="Full LightGBM"]
        base=table[(table.lookback==4)&(table.train_stride==24)&(table.cohort=="all")].iloc[0]
        dense=table[(table.lookback==4)&(table.train_stride==6)&(table.cohort=="all")].iloc[0]
        wide=table[(table.lookback==24)&(table.train_stride==6)&(table.cohort=="all")].iloc[0]
        common=table[(table.lookback==24)&(table.train_stride==6)&(table.cohort=="common_4h")].iloc[0]
        contrast=next(c for c in result["comparisons"] if c["comparison"]=="combined_24h6h_minus_original_on_common")
        easy += [f"[{dataset}]",
                 f"1. 실제 평가 범위: 후보 {int(base.test_n):,}→{int(wide.test_n):,}건, 양성 {int(base.test_positive):,}→{int(wide.test_positive):,}건.",
                 f"양성률은 {base.prevalence:.2%}→{wide.prevalence:.2%}. 음성도 늘어납니다. 후보 수는 사용자 수가 아니라 쌍-날짜 수입니다.",
                 f"2. 같은 기존 후보에서 6시간 학습만 적용: AP {base.ap_mean:.4f}→{dense.ap_mean:.4f}, F1 {base.f1_mean:.4f}→{dense.f1_mean:.4f}.",
                 f"3. 같은 기존 후보에서 24시간 후보 학습+6시간 간격을 함께 적용: AP {base.ap_mean:.4f}→{common.ap_mean:.4f}, F1 {base.f1_mean:.4f}→{common.f1_mean:.4f}.",
                 f"이 조합의 공통 후보 통합 AP 차이 {contrast['delta_ap']:+.4f}, 탐색적 95% 구간 [{contrast['ci_low']:+.4f}, {contrast['ci_high']:+.4f}].",
                 "구간 전체가 0보다 높으면 이 분석에서 개선을 지지하지만 새 독립 검증의 확인 결과는 아닙니다. 0을 포함하면 명확한 개선으로 주장하지 않습니다.",
                 f"4. 넓어진 전체 후보의 성능: AP {wide.ap_mean:.4f}, AUC {wide.auc_mean:.4f}, F1 {wide.f1_mean:.4f}, 정밀도 {wide.precision_mean:.2%}, 재현율 {wide.recall_mean:.2%}.",
                 "4번은 원래와 평가 대상이 달라졌으므로 2·3번과 구별해서 읽어야 합니다.",
                 f"5. 마지막 시간 구간의 학습 사례: {int(base.last_fold_train_n):,}→{int(wide.last_fold_train_n):,}행. 6시간 학습 정답은 겹치므로 독립 표본이 그만큼 늘어난 것은 아닙니다.",""]
    easy += ["지표 설명: AP는 실제 재연결을 높은 순위로 올리는 능력이지 정확도가 아닙니다. 정밀도는 재연결로 고른 것 중 맞은 비율, 재현율은 실제 재연결 중 찾아낸 비율, F1은 둘의 조화평균입니다.",
             "AP/AUC/F1 표는 시간 구간 평균이고 신뢰구간은 통합 AP 차이이므로 집계 수치가 조금 다릅니다. 6시간 학습은 시간대 다양성도 바꾸므로 순수한 표본 수 효과라고 단정하지 않습니다.",
             "기존 GNN 결과는 그대로 보존했습니다. 이번 확장은 LightGBM·이력 HGB와 기존 설정을 고정한 데이터 구성 실험입니다.",
             "전체 여섯 조합과 기준모델: 확장실험_결과.md. 그림: results/lookback_density_v1/<데이터>/lookback_ap.png 및 candidate_coverage.png.",
             "앱에서는 '후보·학습간격 확장' 탭을 누르세요. 결과가 좋아졌더라도 기존 테스트를 본 뒤의 탐색 분석이므로 논문 게재나 새로운 데이터 성능을 보장하지 않습니다."]
    (ROOT/"확장실험_쉽게읽기.txt").write_text("\n".join(easy),encoding="utf-8")
    (OUT/"summary.json").write_text(json.dumps({"finished":datetime.now(timezone.utc).isoformat(),"datasets":results,"status":"complete"},indent=2),encoding="utf-8")
    print("All 12 arms verified and reported.",flush=True)


if __name__=="__main__":main()
