"""Exact weighted AP bootstrap with sorting reused across resamples."""
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score


def ranking(y,score,block):
    order=np.argsort(-np.asarray(score),kind="stable")
    values=np.asarray(score)[order]
    ends=np.r_[np.flatnonzero(values[:-1]!=values[1:]),len(values)-1]
    return np.asarray(y,dtype=float)[order],np.asarray(block)[order],ends


def weighted_ap(prepared,multiplicity):
    y,block,ends=prepared
    weights=multiplicity[block]
    positives=np.cumsum(weights*y)[ends]
    total=np.cumsum(weights)[ends]
    if positives[-1]<=0:return np.nan
    increments=np.diff(np.r_[0.,positives])
    precision=np.divide(positives,total,out=np.zeros_like(positives),where=total>0)
    return float(np.sum(increments*precision)/positives[-1])


def compare(a,b,repeats=1000,block_days=7):
    keys=["fold","cutoff","src","dst","label"]
    paired=a[keys+["score"]].merge(b[keys+["score"]],on=keys,suffixes=("_a","_b"),validate="one_to_one")
    if len(paired)!=len(a) or len(paired)!=len(b):raise ValueError("Unmatched comparison cohorts")
    block,_=pd.factorize(pd.MultiIndex.from_arrays([paired.fold,paired.cutoff//(86400*block_days)]),sort=True)
    number=block.max()+1
    pa=ranking(paired.label,paired.score_a,block);pb=ranking(paired.label,paired.score_b,block)
    ones=np.ones(number,np.int64)
    aa=weighted_ap(pa,ones);bb=weighted_ap(pb,ones)
    assert np.isclose(aa,average_precision_score(paired.label,paired.score_a),atol=1e-12)
    assert np.isclose(bb,average_precision_score(paired.label,paired.score_b),atol=1e-12)
    rng=np.random.default_rng(82);delta=[]
    for _ in range(repeats):
        multiplicity=np.bincount(rng.integers(0,number,number),minlength=number)
        value=weighted_ap(pa,multiplicity)-weighted_ap(pb,multiplicity)
        if np.isfinite(value):delta.append(value)
    return {"delta_ap":aa-bb,"ci_low":float(np.quantile(delta,.025)),"ci_high":float(np.quantile(delta,.975)),
            "n":len(paired),"block_days":block_days,"blocks":int(number),"repeats":len(delta)}
