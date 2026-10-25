"""Official, typed SNAP Super User data with frozen expected counts."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import urllib.request

import numpy as np
import pandas as pd

from research.data import ROOT, KINDS

PAGE="https://snap.stanford.edu/data/sx-superuser.html"
BASE="https://snap.stanford.edu/data"
EXPECTED={"a2q":430033,"c2a":534239,"c2q":479067}


def digest(path:Path):
    h=hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda:stream.read(1024*1024),b""):h.update(chunk)
    return h.hexdigest()


def fetch(kind):
    folder=ROOT/"data"/"confirmation_superuser"/"raw";folder.mkdir(parents=True,exist_ok=True)
    path=folder/f"sx-superuser-{kind}.txt.gz";url=f"{BASE}/{path.name}"
    if not path.exists():
        temporary=path.with_suffix(path.suffix+".part")
        request=urllib.request.Request(url,headers={"User-Agent":"temporal-link-confirmation/1.0"})
        with urllib.request.urlopen(request,timeout=120) as response,temporary.open("wb") as target:
            for chunk in iter(lambda:response.read(1024*1024),b""):target.write(chunk)
        temporary.replace(path)
    frame=pd.read_csv(path,sep=r"\s+",header=None,names=["src","dst","time"],dtype=np.int64,compression="gzip")
    if len(frame)!=EXPECTED[kind] or frame.shape[1]!=3:
        raise ValueError(f"{kind}: {frame.shape}, expected {EXPECTED[kind]}x3")
    return kind,path,frame,{"kind":kind,"url":url,"rows":len(frame),"bytes":path.stat().st_size,"sha256":digest(path)}


def load():
    with ThreadPoolExecutor(max_workers=3) as executor:
        downloaded=list(executor.map(fetch,KINDS))
    parts=[];files=[]
    for kind,path,frame,manifest in downloaded:
        frame=frame.copy();frame["kind"]=kind;parts.append(frame);files.append(manifest)
    raw=pd.concat(parts,ignore_index=True)
    if len(raw)!=sum(EXPECTED.values()):raise AssertionError("raw union count")
    self_events=int((raw.src==raw.dst).sum())
    events=raw.loc[raw.src!=raw.dst].copy()
    users=np.unique(events[["src","dst"]].to_numpy())
    events["src"]=np.searchsorted(users,events.src);events["dst"]=np.searchsorted(users,events.dst)
    events=events.sort_values(["time","src","dst","kind"],kind="stable").reset_index(drop=True)
    info={"dataset":"superuser","official_page":PAGE,"files":files,"raw_events":len(raw),
          "expected_raw_events":sum(EXPECTED.values()),"self_events_removed":self_events,
          "used_events":len(events),"used_users":len(users),"raw_unique_users":int(np.unique(raw[["src","dst"]]).size),
          "first_time":int(events.time.min()),"last_time":int(events.time.max()),
          "type_counts_used":{k:int(v) for k,v in events.kind.value_counts().items()},
          "duplicate_event_rows_retained":int(raw.duplicated(["src","dst","time","kind"]).sum()),
          "node_ids":"remapped anonymous identifiers; no future-derived node features"}
    folder=ROOT/"results"/"confirmation_superuser_v1";folder.mkdir(parents=True,exist_ok=True)
    (folder/"data_manifest.json").write_text(json.dumps(info,indent=2),encoding="utf-8")
    return events,info


if __name__=="__main__":
    _,info=load();print(json.dumps(info,indent=2))
