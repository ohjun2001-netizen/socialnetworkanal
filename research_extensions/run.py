"""Controlled two-model comparison of lookback and training cutoff density."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import time
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from research.data import ROOT
from research.features import NAMES, prepare as original_prepare
from research.evaluation import calibrate, threshold, metrics, folds
from research.models import fit_lgb, fit_history
from mathoverflow_lab.features import HISTORY_FEATURES
from .features import prepare


def dump(path,value):
    path.write_text(json.dumps(value,indent=2,default=lambda v:v.item() if hasattr(v,"item") else str(v)),encoding="utf-8")


def masks_for(meta,x,lookback,stride,b):
    t=meta.cutoff.to_numpy();eligible=x[:,NAMES.index("hours_since_pair_event")]<=lookback
    midnight=t%86400==0
    return {
        "train":eligible & (t%(stride*3600)==0) & (t+86400<=b["validation_start"]),
        "tune":eligible & midnight & (t>=b["validation_start"]) & (t+86400<=b["calibration_start"]),
        "cal":eligible & midnight & (t>=b["calibration_start"]) & (t+86400<=b["test_start"]),
        "test":eligible & midnight & (t>=b["test_start"]) & (t<b["test_stop"]),
    }


def fixed_configs(dataset):
    """Called serially before parallel conditions, using only baseline tuning data."""
    output=ROOT/"results"/"lookback_density_v1"/dataset
    output.mkdir(parents=True,exist_ok=True)
    path=output/"fixed_configs.json"
    if path.exists():return json.loads(path.read_text())
    m,x,_,_=original_prepare(dataset);y=m.label.to_numpy()
    cols=[NAMES.index(n) for n in HISTORY_FEATURES];configs={}
    for fold,masks,_ in folds(m):
        history=fit_history(x,y,masks,cols,2026)
        tree=json.loads((ROOT/"results"/"research_v2"/dataset/f"fold_{fold}"/"lightgbm_config.json").read_text())
        configs[str(fold)]={"lightgbm":tree,"history":history.get_params()}
    dump(path,configs)
    return configs


def run(dataset,lookback,stride):
    meta,x=prepare(dataset);y=meta.label.to_numpy(np.uint8)
    base=ROOT/"results"/"research_v2"/dataset
    output=ROOT/"results"/"lookback_density_v1"/dataset/f"lookback_{lookback}h_train_{stride}h"
    output.mkdir(parents=True,exist_ok=True)
    configs=fixed_configs(dataset)
    protocol=ROOT/"research_extensions"/"protocol.json"
    source={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in (ROOT/"research_extensions").glob("*.py")}
    signature=hashlib.sha256((protocol.read_text()+json.dumps(configs,sort_keys=True)+source["features.py"]+source["run.py"]).encode()).hexdigest()
    cols=[NAMES.index(n) for n in HISTORY_FEATURES]
    for fold in range(5):
        b=json.loads((base/f"fold_{fold}"/"split.json").read_text())
        masks=masks_for(meta,x,lookback,stride,b)
        for left,right in (("train","tune"),("tune","cal"),("cal","test")):
            assert meta.loc[masks[left],"cutoff"].max()+86400 <= meta.loc[masks[right],"cutoff"].min()
        folder=output/f"fold_{fold}";folder.mkdir(exist_ok=True)
        split={key:{"n":int(mask.sum()),"positive":int(y[mask].sum()),"cutoffs":int(meta.loc[mask,"cutoff"].nunique())} for key,mask in masks.items()}
        dump(folder/"split.json",{"boundaries":b,"partitions":split})
        common=x[masks["test"],NAMES.index("hours_since_pair_event")]<=4
        test_meta=meta.loc[masks["test"],["cutoff","src","dst","label"]].reset_index(drop=True)
        history_cache=None
        for seed in (2026,2027,2028):
            destination=folder/f"seed_{seed}";destination.mkdir(exist_ok=True)
            if (destination/"complete.json").exists() and json.loads((destination/"complete.json").read_text()).get("signature")==signature:
                print(f"resume {dataset}/{lookback}/{stride}/{fold}/{seed}",flush=True);continue
            begin=time.perf_counter();tables=[];predictions=[]
            print(f"train {dataset} lookback={lookback} stride={stride} fold={fold} seed={seed} rows={split['train']['n']}",flush=True)
            if lookback==4 and stride==24:
                reference=pd.read_csv(base/f"fold_{fold}"/f"seed_{seed}"/"predictions.csv.gz",float_precision="round_trip")
                reference_metrics=pd.read_csv(base/f"fold_{fold}"/f"seed_{seed}"/"metrics.csv",float_precision="round_trip").set_index("model")
            for name in ("Full LightGBM","History HGB"):
                if lookback==4 and stride==24:
                    part=reference[reference.model==name].reset_index(drop=True)
                    pd.testing.assert_frame_equal(part[["cutoff","src","dst","label"]],test_meta,check_dtype=False)
                    score=part.score.to_numpy();th=float(reference_metrics.loc[name,"threshold"])
                elif name=="History HGB" and history_cache is not None:
                    score,th=history_cache
                else:
                    if name=="Full LightGBM":
                        model=fit_lgb(x,y,masks["train"],list(range(len(NAMES))),configs[str(fold)]["lightgbm"],seed)
                        columns=list(range(len(NAMES)))
                    else:
                        params=dict(configs[str(fold)]["history"]);params["random_state"]=2026
                        model=HistGradientBoostingClassifier(**params).fit(x[masks["train"]][:,cols],y[masks["train"]])
                        columns=cols
                    scores={key:model.predict_proba(x[masks[key]][:,columns])[:,1] for key in ("cal","test")}
                    scores,calibrator=calibrate(y[masks["cal"]],scores["cal"],scores)
                    th=threshold(y[masks["cal"]],scores["cal"]);score=scores["test"]
                    if name=="History HGB":history_cache=(score,th)
                    if seed==2026 and fold==4:
                        joblib.dump({"model":model,"columns":columns,"calibrator":calibrator,"threshold":th,
                                     "lookback":lookback,"stride":stride,"dataset":dataset},destination/(name.replace(" ","_")+".joblib"))
                    del model
                for cohort,select in (("all",np.ones(len(score),bool)),("common_4h",common)):
                    value=metrics(test_meta.label.to_numpy()[select],score[select],th,test_meta.cutoff.to_numpy()[select])
                    tables.append({"dataset":dataset,"lookback":lookback,"train_stride":stride,"fold":fold,"seed":seed,"model":name,"cohort":cohort,**value})
                part=test_meta.copy();part["score"]=score;part["prediction"]=score>=th
                part["common_4h"]=common;part["model"]=name;part["fold"]=fold;part["seed"]=seed
                predictions.append(part)
            pd.DataFrame(tables).to_csv(destination/"metrics.csv",index=False)
            pd.concat(predictions).to_csv(destination/"predictions.csv.gz",index=False,compression="gzip")
            dump(destination/"complete.json",{"signature":signature,"source_sha256":source,"elapsed_seconds":time.perf_counter()-begin,
                 "reused_v2_reference":lookback==4 and stride==24,"history_seed_policy":"deterministic HGB reused across seeds; seeds are not new data",
                 "completed":datetime.now(timezone.utc).isoformat()})
            print(f"complete {dataset}/{lookback}/{stride}/{fold}/{seed} {time.perf_counter()-begin:.1f}s",flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--dataset",required=True);p.add_argument("--lookback",type=int,required=True,choices=[4,12,24]);p.add_argument("--stride",type=int,required=True,choices=[6,24])
    a=p.parse_args();run(a.dataset,a.lookback,a.stride)
