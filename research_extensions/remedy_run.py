"""Train the two requested remedies on fixed temporal partitions."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import time

import joblib
import numpy as np
import pandas as pd

from research.data import ROOT, load
from research.evaluation import metrics, threshold
from research.features import NAMES
from research_extensions.features import prepare
from research_extensions.run import masks_for
from research_extensions.remedy_models import (
    ACTIVITY_FEATURES, activity_view, expert_group, expand_activity_scores,
    fit_activity, fit_choice, fit_experts,
)

MODELS = ["Two-stage activity-choice", "Recency-type experts", "Two-stage recency-type experts"]


def dump(path, value):
    path.write_text(json.dumps(value, indent=2, default=lambda v: v.item() if hasattr(v, "item") else str(v)), encoding="utf-8")


def run(dataset: str):
    started = time.perf_counter()
    meta, x = prepare(dataset); y = meta.label.to_numpy(np.uint8)
    events, data_info = load(dataset)
    unique, ax, ay, source_group, pair_activity, first_index = activity_view(meta, x, events)
    groups = expert_group(x)
    base = ROOT / "results" / "research_v2" / dataset
    configs = json.loads((ROOT / "results" / "lookback_density_v1" / dataset / "fixed_configs.json").read_text())
    output = ROOT / "results" / "remedy_v1" / dataset; output.mkdir(parents=True, exist_ok=True)
    source_files = [ROOT/"research_extensions"/name for name in ("remedy_protocol.json","remedy_models.py","remedy_run.py")]
    hashes = {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in source_files}
    signature = hashlib.sha256((json.dumps(hashes,sort_keys=True)+json.dumps(configs,sort_keys=True)).encode()).hexdigest()
    for fold in range(5):
        boundaries = json.loads((base / f"fold_{fold}" / "split.json").read_text())
        masks = masks_for(meta, x, 24, 6, boundaries)
        activity_masks = {key: value[first_index] for key,value in masks.items()}
        for key in masks:
            assert np.array_equal(unique.loc[activity_masks[key],"cutoff"].unique(), meta.loc[masks[key],"cutoff"].unique())
        split = {key:{"pairs":int(mask.sum()),"pair_positive":int(y[mask].sum()),
                      "source_cutoffs":int(activity_masks[key].sum()),"active_source_cutoffs":int(ay[activity_masks[key]].sum())}
                 for key,mask in masks.items()}
        dump(output/f"fold_{fold}_split.json",{"boundaries":boundaries,"partitions":split})
        for seed in (2026,2027,2028):
            destination = output/f"fold_{fold}"/f"seed_{seed}";destination.mkdir(parents=True,exist_ok=True)
            complete=destination/"complete.json"
            if complete.exists() and json.loads(complete.read_text()).get("signature")==signature:
                print(f"resume {dataset}/{fold}/{seed}",flush=True);continue
            begin=time.perf_counter();config=configs[str(fold)]["lightgbm"]
            print(f"train {dataset} fold={fold} seed={seed} pairs={split['train']['pairs']:,}",flush=True)
            activity_model, activity_cal, activity_test, activity_calibrator = fit_activity(ax,ay,activity_masks,config,seed)
            activity_pair_cal=expand_activity_scores(source_group,masks["cal"],activity_masks["cal"],activity_cal)
            activity_pair_test=expand_activity_scores(source_group,masks["test"],activity_masks["test"],activity_test)
            choice_model,choice_cal,choice_test,choice_calibrator=fit_choice(
                x,y,masks["train"],masks["cal"],masks["test"],pair_activity,config,seed)
            direct_models,direct_cal,direct_test,direct_calibrators=fit_experts(
                x,y,groups,masks["train"],masks["cal"],masks["test"],config,seed)
            conditional_models,conditional_cal,conditional_test,conditional_calibrators=fit_experts(
                x,y,groups,masks["train"],masks["cal"],masks["test"],config,seed,pair_activity)
            scores={
                "Two-stage activity-choice":(activity_pair_cal*choice_cal,activity_pair_test*choice_test),
                "Recency-type experts":(direct_cal,direct_test),
                "Two-stage recency-type experts":(activity_pair_cal*conditional_cal,activity_pair_test*conditional_test),
            }
            test_meta=meta.loc[masks["test"],["cutoff","src","dst","label"]].reset_index(drop=True)
            common=x[masks["test"],NAMES.index("hours_since_pair_event")]<=4
            tables=[];predictions=[]
            for name,(cal_score,test_score) in scores.items():
                th=threshold(y[masks["cal"]],cal_score)
                for cohort,select in (("all",np.ones(len(test_score),bool)),("common_4h",common)):
                    tables.append({"dataset":dataset,"fold":fold,"seed":seed,"model":name,"cohort":cohort,
                                   **metrics(test_meta.label.to_numpy()[select],test_score[select],th,test_meta.cutoff.to_numpy()[select])})
                p=test_meta.copy();p["score"]=test_score;p["prediction"]=test_score>=th;p["threshold"]=th
                p["common_4h"]=common;p["model"]=name;p["fold"]=fold;p["seed"]=seed;predictions.append(p)
            activity_th=threshold(ay[activity_masks["cal"]],activity_cal)
            activity_result=metrics(ay[activity_masks["test"]],activity_test,activity_th,
                                    unique.loc[activity_masks["test"],"cutoff"].to_numpy())
            pd.DataFrame(tables).to_csv(destination/"metrics.csv",index=False)
            pd.DataFrame([{"dataset":dataset,"fold":fold,"seed":seed,**activity_result}]).to_csv(destination/"activity_metrics.csv",index=False)
            pd.concat(predictions,ignore_index=True).to_csv(destination/"predictions.csv.gz",index=False,compression="gzip")
            if fold==4 and seed==2026:
                joblib.dump({"activity_model":activity_model,"activity_calibrator":activity_calibrator,
                             "activity_features":ACTIVITY_FEATURES+["log1p_candidate_count"],
                             "choice_model":choice_model,"choice_calibrator":choice_calibrator,
                             "direct_experts":direct_models,"direct_calibrators":direct_calibrators,
                             "conditional_experts":conditional_models,"conditional_calibrators":conditional_calibrators,
                             "thresholds":{row["model"]:row["threshold"] for row in tables if row["cohort"]=="all"},
                             "group_definition":"(<=4h vs 4-24h) x (a2q,c2a,c2q)"},destination/"models.joblib")
            dump(complete,{"signature":signature,"source_sha256":hashes,"elapsed_seconds":time.perf_counter()-begin,
                           "device":"CPU LightGBM; tree training does not use the preserved GPU GNN",
                           "test_was_previously_inspected":True,"completed":datetime.now(timezone.utc).isoformat()})
            print(f"complete {dataset}/{fold}/{seed} {time.perf_counter()-begin:.1f}s",flush=True)
    dump(output/"run_manifest.json",{"dataset":dataset,"signature":signature,"source_sha256":hashes,
         "data":data_info,"activity_source_cutoffs":len(unique),"activity_positive":int(ay.sum()),
         "feature_rows":len(meta),"pair_positive":int(y.sum()),"models":MODELS,
         "elapsed_seconds":time.perf_counter()-started,"status":"complete"})


if __name__=="__main__":
    parser=argparse.ArgumentParser();parser.add_argument("--dataset",choices=["mathoverflow","askubuntu"],required=True)
    run(parser.parse_args().dataset)
