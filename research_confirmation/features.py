"""Content-addressed 24h/6h Super User features and frozen temporal splits."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from mathoverflow_lab.exploration import make_cutoffs
from research.data import ROOT
from research.features import NAMES
from research_extensions.features import build
from .data import load

DAY=86400


def source_hashes():
    paths=[Path(__file__),ROOT/"research_confirmation"/"data.py",ROOT/"research_extensions"/"features.py",
           ROOT/"research"/"features.py",ROOT/"mathoverflow_lab"/"features.py",ROOT/"mathoverflow_lab"/"exploration.py"]
    return {str(path.relative_to(ROOT)):hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}


def boundaries(meta):
    times=np.sort(meta.loc[meta.cutoff%DAY==0,"cutoff"].unique())
    if len(times)<100:raise ValueError("too few daily evaluation cutoffs")
    result=[]
    for fold,start in enumerate((.5,.6,.7,.8,.9)):
        test0=int(times[round(len(times)*start)])
        stop_i=round(len(times)*(start+.1));test1=int(times[stop_i]) if stop_i<len(times) else int(times[-1]+DAY)
        val0=int(times[round(len(times)*(start-.1))]);cal0=int(times[round(len(times)*(start-.05))])
        result.append({"fold":fold,"validation_start":val0,"calibration_start":cal0,"test_start":test0,"test_stop":test1})
    return result


def masks(meta,boundary):
    t=meta.cutoff.to_numpy(np.int64);midnight=t%DAY==0
    output={
        "train":(t%(6*3600)==0)&(t+DAY<=boundary["validation_start"]),
        "tune":midnight&(t>=boundary["validation_start"])&(t+DAY<=boundary["calibration_start"]),
        "cal":midnight&(t>=boundary["calibration_start"])&(t+DAY<=boundary["test_start"]),
        "test":midnight&(t>=boundary["test_start"])&(t<boundary["test_stop"]),
    }
    for left,right in (("train","tune"),("tune","cal"),("cal","test")):
        assert meta.loc[output[left],"cutoff"].max()+DAY<=meta.loc[output[right],"cutoff"].min()
    y=meta.label.to_numpy()
    for name,mask in output.items():
        if not mask.any() or len(np.unique(y[mask]))!=2:raise ValueError(f"fold partition {name} lacks two classes")
    if sum(output.values()).max()>1:raise AssertionError("overlapping partitions")
    return output


def prepare():
    events,info=load();hashes=source_hashes()
    protocol=ROOT/"research_confirmation"/"superuser_protocol.json"
    fingerprint=hashlib.sha256((hashlib.sha256(protocol.read_bytes()).hexdigest()+json.dumps(info["files"],sort_keys=True)+json.dumps(hashes,sort_keys=True)).encode()).hexdigest()
    folder=ROOT/"data"/"cache"/"confirmation_superuser";folder.mkdir(parents=True,exist_ok=True)
    path=folder/f"features_{fingerprint[:12]}.npz"
    if path.exists():
        z=np.load(path,allow_pickle=False)
        if z["names"].tolist()!=NAMES:raise ValueError("feature schema mismatch")
        meta=pd.DataFrame({key:z[key] for key in ("cutoff","src","dst","pair_code","label")})
        x=z["x"]
    else:
        cutoffs=make_cutoffs(events.time.to_numpy(np.int64),30,24,6)
        meta,x=build(events,cutoffs)
        np.savez_compressed(path,x=x,names=np.array(NAMES),**{key:meta[key].to_numpy() for key in meta})
    if not np.isfinite(x).all() or len(x)!=len(meta):raise AssertionError("invalid features")
    since=x[:,NAMES.index("hours_since_pair_event")]
    if since.max()>24+1e-5:raise AssertionError("candidate outside 24h")
    splits=boundaries(meta)
    split_public=[]
    for boundary in splits:
        partition=masks(meta,boundary)
        split_public.append({**boundary,"partitions":{name:{"rows":int(mask.sum()),"cutoffs":int(meta.loc[mask,"cutoff"].nunique()),
                           "first":int(meta.loc[mask,"cutoff"].min()),"last":int(meta.loc[mask,"cutoff"].max()),
                           "last_label_end":int(meta.loc[mask,"cutoff"].max()+DAY)} for name,mask in partition.items()}})
    output=ROOT/"results"/"confirmation_superuser_v1";output.mkdir(parents=True,exist_ok=True)
    (output/"splits.json").write_text(json.dumps(split_public,indent=2),encoding="utf-8")
    manifest={"dataset":"superuser","rows":len(meta),"features":len(NAMES),"cutoffs":int(meta.cutoff.nunique()),
              "daily_cutoffs":int(meta.loc[meta.cutoff%DAY==0,"cutoff"].nunique()),"cache":path.name,
              "cache_sha256":hashlib.sha256(path.read_bytes()).hexdigest(),"source_sha256":hashes,
              "protocol_sha256":hashlib.sha256(protocol.read_bytes()).hexdigest(),"test_metrics_exposed":False}
    (output/"feature_manifest.json").write_text(json.dumps(manifest,indent=2),encoding="utf-8")
    return meta,x,events,info,splits,manifest


if __name__=="__main__":
    _,_,_,_,_,manifest=prepare();print(json.dumps(manifest,indent=2))
