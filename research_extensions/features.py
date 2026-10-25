"""Fast equivalent 78-feature builder for a 24h candidate / 6h cutoff superset.

Only past events enter features. Validate against every original v2 midnight
4h row before permitting the expanded matrix to be used for training.
"""
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
from numba import njit, prange, set_num_threads
from scipy.sparse import csr_matrix
from research.data import ROOT, KINDS
from research.features import prepare as original_prepare, NAMES
from mathoverflow_lab.exploration import pair_codes, make_cutoffs


def pack(keys, times, kinds):
    order=np.argsort(keys,kind="stable")
    unique, starts=np.unique(keys[order],return_index=True)
    return (unique.astype(np.int64),np.r_[starts,len(order)].astype(np.int64),
            times[order].astype(np.int64),kinds[order].astype(np.int64))


@njit
def bounds(index,key):
    pos=np.searchsorted(index[0],key)
    if pos==len(index[0]) or index[0][pos]!=key:
        return 0,0
    return index[1][pos],index[1][pos+1]


@njit
def count(index,key,start,end):
    a,b=bounds(index,key);v=index[2][a:b]
    return np.searchsorted(v,end)-np.searchsorted(v,start)


@njit
def decay(index,key,t,hours):
    a,b=bounds(index,key);v=index[2][a:b]
    lo=np.searchsorted(v,t-30*86400);hi=np.searchsorted(v,t)
    value=0.
    for j in range(lo,hi):
        value+=np.exp(-(t-v[j])/(hours*3600.))
    return value


@njit(parallel=True)
def temporal_features(cutoff,src,dst,codes,pair,outgoing,incoming,typed,tout,tin):
    result=np.zeros((len(cutoff),78),np.float32)
    for row in prange(len(cutoff)):
        t=cutoff[row];s=src[row];d=dst[row];key=codes[row];reverse=(d<<32)|s
        a,b=bounds(pair,key);v=pair[2][a:b];kind=pair[3][a:b]
        n=np.searchsorted(v,t);lo=np.searchsorted(v,t-30*86400)
        col=0
        for h in (1,4,12,24,72,168,720):
            result[row,col]=count(pair,key,t-h*3600,t);col+=1
        result[row,7]=n
        for k,h in enumerate((4,24,168,720)):
            result[row,8+k]=count(pair,reverse,t-h*3600,t)
        ra,rb=bounds(pair,reverse)
        result[row,12]=np.searchsorted(pair[2][ra:rb],t)
        result[row,13]=(t-v[n-1])/3600.
        result[row,14]=(t-v[0])/86400.
        result[row,15]=count(outgoing,s,t-86400,t)
        result[row,16]=count(outgoing,s,t-7*86400,t)
        result[row,17]=count(outgoing,s,t-30*86400,t)
        result[row,18]=count(incoming,s,t-7*86400,t)
        result[row,19]=count(incoming,d,t-86400,t)
        result[row,20]=count(incoming,d,t-7*86400,t)
        result[row,21]=count(incoming,d,t-30*86400,t)
        result[row,22]=count(outgoing,d,t-7*86400,t)
        for k,h in enumerate((1,6,24,72,168)):
            result[row,29+k]=decay(pair,key,t,h)
        for k,h in enumerate((6,24,168)):
            result[row,34+k]=decay(pair,reverse,t,h)
        gap_n=n-lo-1;mean=0.;std=0.;last=0.
        if gap_n>0:
            for j in range(lo+1,n):mean+=(v[j]-v[j-1])/3600.
            mean/=gap_n
            for j in range(lo+1,n):std+=((v[j]-v[j-1])/3600.-mean)**2
            std=np.sqrt(std/gap_n);last=(v[n-1]-v[n-2])/3600.
        active_days=0;last_day=-1;switches=0;kc=np.zeros(3,np.int64)
        for j in range(lo,n):
            day=v[j]//86400
            if day!=last_day:active_days+=1;last_day=day
            kc[kind[j]]+=1
            if j>lo and kind[j]!=kind[j-1]:switches+=1
        f4=count(pair,key,t-14400,t);f24=count(pair,key,t-86400,t)
        result[row,37]=mean;result[row,38]=std/(mean+1e-6);result[row,39]=last
        result[row,40]=(std-mean)/(std+mean+1e-6);result[row,41]=active_days
        result[row,42]=f4/(1+(f24-f4)/5.)
        result[row,43]=count(pair,reverse,t-86400,t)/(1.+f24)
        result[row,44]=count(outgoing,s,t-86400,t)/(1+count(outgoing,s,t-7*86400,t-86400)/6.)
        result[row,45]=count(incoming,d,t-86400,t)/(1+count(incoming,d,t-7*86400,t-86400)/6.)
        weekday=(t//86400+3)%7
        result[row,46]=np.sin(2*np.pi*weekday/7);result[row,47]=np.cos(2*np.pi*weekday/7)
        entropy=0.
        for k in range(3):
            c=48+9*k;tk=key*3+k;rk=reverse*3+k
            for j,h in enumerate((4,24,168)):result[row,c+j]=count(typed,tk,t-h*3600,t)
            result[row,c+3]=count(typed,rk,t-86400,t)
            result[row,c+4]=decay(typed,tk,t,6);result[row,c+5]=decay(typed,tk,t,24)
            result[row,c+6]=1. if kind[n-1]==k else 0.
            result[row,c+7]=count(tout,s*3+k,t-7*86400,t)
            result[row,c+8]=count(tin,d*3+k,t-7*86400,t)
            p=kc[k]/max(n-lo,1);entropy-=p*np.log(p+1e-9)
        result[row,75]=entropy;result[row,76]=switches
    return result


@njit
def structural_features(src,dst,ui,uj,oi,oj,ot,ii,ij,it):
    result=np.zeros((len(src),7),np.float32)
    for row in range(len(src)):
        s=src[row];d=dst[row];da=ui[s+1]-ui[s];db=ui[d+1]-ui[d]
        p=ui[s];q=ui[d];common=0;aa=0.
        while p<ui[s+1] and q<ui[d+1]:
            if uj[p]==uj[q]:
                node=uj[p];common+=1;aa+=1/np.log(max(ui[node+1]-ui[node],2));p+=1;q+=1
            elif uj[p]<uj[q]:p+=1
            else:q+=1
        result[row,0]=da;result[row,1]=db;result[row,2]=common
        result[row,3]=common/max(da+db-common,1);result[row,4]=aa;result[row,5]=da*db
        p=oi[s];q=ii[d];paths=0
        while p<oi[s+1] and q<ii[d+1]:
            if oj[p]==ij[q]:
                if ot[p]<it[q] and it[q]-ot[p]<=86400:paths+=1
                p+=1;q+=1
            elif oj[p]<ij[q]:p+=1
            else:q+=1
        result[row,6]=paths
    return result


def build(events, cutoffs):
    times=events.time.to_numpy(np.int64);codes=pair_codes(events).astype(np.int64)
    src=events.src.to_numpy(np.int64);dst=events.dst.to_numpy(np.int64)
    kinds=pd.Categorical(events.kind,categories=KINDS).codes.astype(np.int64)
    parts=[]
    for t in cutoffs:
        left,split,right=np.searchsorted(times,[t-86400,t,t+86400])
        candidates=np.unique(codes[left:split]);future=np.unique(codes[split:right])
        if len(candidates):
            parts.append(pd.DataFrame({"cutoff":int(t),"src":candidates>>32,"dst":candidates&0xFFFFFFFF,
                         "pair_code":candidates,"label":np.isin(candidates,future,assume_unique=True).astype(np.uint8)}))
    meta=pd.concat(parts,ignore_index=True)
    indexes=[pack(key,times,kinds) for key in (codes,src,dst,codes*3+kinds,src*3+kinds,dst*3+kinds)]
    print(f"Temporal features: {len(meta):,} candidates",flush=True)
    set_num_threads(4)
    x=temporal_features(*[meta[k].to_numpy(np.int64) for k in ("cutoff","src","dst","pair_code")],*indexes)
    nodes=int(max(src.max(),dst.max()))+1
    for number,(t,rows) in enumerate(meta.groupby("cutoff",sort=True)):
        left,right=np.searchsorted(times,[t-30*86400,t])
        _,reverse_index=np.unique(codes[left:right][::-1],return_index=True)
        ix=right-1-reverse_index
        directed=csr_matrix((times[ix],(src[ix],dst[ix])),shape=(nodes,nodes))
        reverse=directed.T.tocsr();undirected=(directed+reverse).tocsr()
        values=structural_features(rows.src.to_numpy(),rows.dst.to_numpy(),undirected.indptr,undirected.indices,
                 directed.indptr,directed.indices,directed.data,reverse.indptr,reverse.indices,reverse.data)
        x[rows.index,23:29]=values[:,:6];x[rows.index,77]=values[:,6]
        if (number+1)%500==0:print(f"Structure: {number+1}/{meta.cutoff.nunique()} cutoffs",flush=True)
    assert np.isfinite(x).all()
    return meta,x


def prepare(dataset):
    old_meta,old_x,events,info=original_prepare(dataset)
    output=ROOT/"results"/"lookback_density_v1"/dataset;output.mkdir(parents=True,exist_ok=True)
    boundary=json.loads((ROOT/"results"/"research_v2"/dataset/"fold_4"/"split.json").read_text())["validation_start"]
    source=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    key=hashlib.sha256(json.dumps({"files":info["files"],"source":source,"boundary":boundary},sort_keys=True).encode()).hexdigest()[:12]
    path=ROOT/"data"/"cache"/f"extension_{dataset}_{key}.npz"
    if path.exists():
        data=np.load(path,allow_pickle=False)
        meta=pd.DataFrame({k:data[k] for k in ("cutoff","src","dst","pair_code","label")})
        return meta,data["x"]
    cutoffs=make_cutoffs(events.time.to_numpy(),30,24,6)
    cutoffs=cutoffs[(cutoffs%86400==0)|(cutoffs+86400<=boundary)]
    meta,x=build(events,cutoffs)
    subset=(meta.cutoff.to_numpy()%86400==0)&(x[:,NAMES.index("hours_since_pair_event")]<=4)
    original=meta.loc[subset].reset_index(drop=True)
    pd.testing.assert_frame_equal(original,old_meta.reset_index(drop=True),check_dtype=False)
    np.testing.assert_allclose(x[subset],old_x,rtol=1e-5,atol=1e-6)
    max_error=float(np.abs(x[subset]-old_x).max())
    # Preserve the exact original floating-point values for the reference rows.
    x[subset]=old_x
    np.savez_compressed(path,x=x,**{k:meta[k].to_numpy() for k in meta})
    manifest={"dataset":dataset,"rows":len(meta),"positive":int(meta.label.sum()),"features":len(NAMES),
              "cutoffs":meta.cutoff.nunique(),"reference_rows_verified":int(subset.sum()),"max_absolute_feature_difference":max_error,
              "cache":path.name,"source_sha256":source,"original_data":info}
    (output/"feature_validation.json").write_text(json.dumps(manifest,indent=2),encoding="utf-8")
    print(json.dumps({k:v for k,v in manifest.items() if k!="original_data"},indent=2),flush=True)
    return meta,x


if __name__=="__main__":
    import argparse
    p=argparse.ArgumentParser();p.add_argument("--dataset",default="mathoverflow")
    prepare(p.parse_args().dataset)
