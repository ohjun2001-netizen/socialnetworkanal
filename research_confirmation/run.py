"""Train frozen Super User confirmation models without computing test metrics."""
from __future__ import annotations

import argparse
from datetime import datetime,timezone
import hashlib
import json
import time

import joblib
import numpy as np
import pandas as pd

from research.data import ROOT
from research.evaluation import calibrate,threshold
from research.features import NAMES
from research.models import fit_lgb,lightgbm_config
from research_extensions.remedy_models import (ACTIVITY_FEATURES,activity_view,expert_group,
    expand_activity_scores,fit_activity,fit_choice,fit_experts)
from .features import prepare,masks

MODELS=["Baseline Full LightGBM","Two-stage activity-choice","Recency-type experts","Two-stage recency-type experts"]


def dump(path,value):
    path.write_text(json.dumps(value,indent=2,default=lambda v:v.item() if hasattr(v,"item") else str(v)),encoding="utf-8")


def calibrated_baseline(x,y,partition,config,seed):
    model=fit_lgb(x,y,partition["train"],list(range(x.shape[1])),config,seed)
    raw_cal=model.predict_proba(x[partition["cal"]])[:,1];raw_test=model.predict_proba(x[partition["test"]])[:,1]
    scores,calibrator=calibrate(y[partition["cal"]],raw_cal,{"cal":raw_cal,"test":raw_test})
    return model,scores["cal"],scores["test"],calibrator


def run(fold):
    meta,x,events,_,splits,feature_manifest=prepare();y=meta.label.to_numpy(np.uint8)
    boundary=splits[fold];partition=masks(meta,boundary)
    output=ROOT/"results"/"confirmation_superuser_v1";folder=output/f"fold_{fold}";folder.mkdir(parents=True,exist_ok=True)
    sources=[ROOT/"research_confirmation"/name for name in ("superuser_protocol.json","data.py","features.py","run.py","report.py")]
    sources += [ROOT/"research_extensions"/"remedy_models.py",ROOT/"research"/"models.py",ROOT/"research"/"evaluation.py"]
    hashes={str(path.relative_to(ROOT)):hashlib.sha256(path.read_bytes()).hexdigest() for path in sources}
    signature=hashlib.sha256((json.dumps(hashes,sort_keys=True)+feature_manifest["cache_sha256"]).encode()).hexdigest()
    config_path=folder/"lightgbm_config.json"
    if config_path.exists():config=json.loads(config_path.read_text())
    else:
        print(f"fold={fold}: 16 train/tune Optuna trials (test untouched)",flush=True)
        config,trials=lightgbm_config(x,y,partition,trials=16)
        trials.to_csv(folder/"optuna_trials.csv",index=False);dump(config_path,config)
    unique,ax,ay,source_group,pair_activity,first_index=activity_view(meta,x,events)
    activity_masks={key:value[first_index] for key,value in partition.items()}
    groups=expert_group(x)
    for seed in (2026,2027,2028):
        destination=folder/f"seed_{seed}";destination.mkdir(exist_ok=True)
        complete=destination/"complete.json"
        if complete.exists() and json.loads(complete.read_text()).get("signature")==signature:
            print(f"resume fold={fold} seed={seed}",flush=True);continue
        begin=time.perf_counter();print(f"fold={fold} seed={seed}: train four preregistered models",flush=True)
        baseline_model,baseline_cal,baseline_test,baseline_calibrator=calibrated_baseline(x,y,partition,config,seed)
        activity_model,activity_cal,activity_test,activity_calibrator=fit_activity(ax,ay,activity_masks,config,seed)
        activity_pair_cal=expand_activity_scores(source_group,partition["cal"],activity_masks["cal"],activity_cal)
        activity_pair_test=expand_activity_scores(source_group,partition["test"],activity_masks["test"],activity_test)
        choice_model,choice_cal,choice_test,choice_calibrator=fit_choice(
            x,y,partition["train"],partition["cal"],partition["test"],pair_activity,config,seed)
        direct_models,direct_cal,direct_test,direct_calibrators=fit_experts(
            x,y,groups,partition["train"],partition["cal"],partition["test"],config,seed)
        conditional_models,conditional_cal,conditional_test,conditional_calibrators=fit_experts(
            x,y,groups,partition["train"],partition["cal"],partition["test"],config,seed,pair_activity)
        scores={
            "Baseline Full LightGBM":(baseline_cal,baseline_test),
            "Two-stage activity-choice":(activity_pair_cal*choice_cal,activity_pair_test*choice_test),
            "Recency-type experts":(direct_cal,direct_test),
            "Two-stage recency-type experts":(activity_pair_cal*conditional_cal,activity_pair_test*conditional_test),
        }
        thresholds={name:threshold(y[partition["cal"]],cal) for name,(cal,_) in scores.items()}
        test=meta.loc[partition["test"],["cutoff","src","dst","label"]].reset_index(drop=True)
        arrays={key:test[key].to_numpy() for key in test}
        arrays["common_4h"]=(x[partition["test"],NAMES.index("hours_since_pair_event")]<=4)
        for number,name in enumerate(MODELS):arrays[f"score_{number}"]=scores[name][1]
        activity_threshold=threshold(ay[activity_masks["cal"]],activity_cal)
        arrays["activity_cutoff"]=unique.loc[activity_masks["test"],"cutoff"].to_numpy()
        arrays["activity_src"]=unique.loc[activity_masks["test"],"src"].to_numpy()
        arrays["activity_label"]=ay[activity_masks["test"]]
        arrays["activity_score"]=activity_test
        arrays["activity_threshold"]=np.array([activity_threshold])
        np.savez_compressed(destination/"locked_test_scores.npz",**arrays)
        dump(destination/"thresholds.json",thresholds)
        activity_raw_cal=activity_model.predict_proba(ax[activity_masks["cal"]])[:,1]
        dump(destination/"training_record.json",{
            "fold":fold,"seed":seed,"config":config,
            "partitions":{name:{"pair_rows":int(mask.sum()),"cutoffs":int(meta.loc[mask,"cutoff"].nunique())} for name,mask in partition.items()},
            "activity_train_source_cutoffs":int(activity_masks["train"].sum()),
            "conditional_choice_train_rows":int((partition["train"]&pair_activity.astype(bool)).sum()),
            "expert_train_rows":{str(group):int((partition["train"]&(groups==group)).sum()) for group in range(6)},
            "thresholds_selected_on":"calibration interval only","test_metrics_computed":False})
        if fold==4 and seed==2026:
            joblib.dump({"baseline_model":baseline_model,"baseline_calibrator":baseline_calibrator,
                         "activity_model":activity_model,"activity_calibrator":activity_calibrator,
                         "activity_features":ACTIVITY_FEATURES+["log1p_candidate_count"],
                         "choice_model":choice_model,"choice_calibrator":choice_calibrator,
                         "direct_experts":direct_models,"direct_calibrators":direct_calibrators,
                         "conditional_experts":conditional_models,"conditional_calibrators":conditional_calibrators,
                         "thresholds":thresholds,"feature_names":NAMES},destination/"models.joblib")
        dump(complete,{"signature":signature,"source_sha256":hashes,"feature_cache_sha256":feature_manifest["cache_sha256"],
                       "test_metrics_computed":False,"elapsed_seconds":time.perf_counter()-begin,
                       "completed":datetime.now(timezone.utc).isoformat()})
        print(f"fold={fold} seed={seed}: locked scores in {time.perf_counter()-begin:.1f}s (no test metric)",flush=True)
    dump(folder/"complete.json",{"fold":fold,"signature":signature,"seeds":[2026,2027,2028],
                                 "test_metrics_computed":False,"completed":datetime.now(timezone.utc).isoformat()})


if __name__=="__main__":
    parser=argparse.ArgumentParser();parser.add_argument("--fold",type=int,choices=range(5),required=True)
    run(parser.parse_args().fold)
