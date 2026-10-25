import numpy as np
import pandas as pd
from research_extensions.features import build
from research_extensions.run import masks_for
from research.features import NAMES,extra_features
from mathoverflow_lab.features import build_examples
from research_extensions.uncertainty import ranking,weighted_ap
from sklearn.metrics import average_precision_score


def test_fast_features_match_reference_and_exclude_future():
    t=31*86400
    events=pd.DataFrame([(0,1,0,"a2q"),(0,1,t-3600,"c2q"),(1,2,t-1800,"a2q"),
                         (0,1,t,"c2a"),(2,1,t+1800,"c2q"),(3,4,t+86400+100,"c2a")],columns=["src","dst","time","kind"])
    m,x=build(events,np.array([t],np.int64))
    original,base=build_examples(events)
    np.testing.assert_allclose(x,np.column_stack([base,extra_features(events,original)]),rtol=1e-5,atol=1e-6)
    changed=events.copy();changed.loc[changed.time>=t,"src"]=4
    altered,xx=build(changed,np.array([t],np.int64))
    np.testing.assert_array_equal(x,xx)
    assert m.label.tolist()==[1,0] and altered.label.tolist()==[0,0]


def test_dense_training_keeps_identical_daily_tests_and_mature_labels():
    cutoffs=np.repeat(np.arange(0,100*86400,6*3600),3)
    meta=pd.DataFrame({"cutoff":cutoffs})
    x=np.zeros((len(meta),len(NAMES)),np.float32)
    x[:,NAMES.index("hours_since_pair_event")]=np.tile([2.,8.,20.],len(meta)//3)
    boundaries={"validation_start":40*86400,"calibration_start":45*86400,"test_start":50*86400,"test_stop":60*86400}
    sparse=masks_for(meta,x,12,24,boundaries);dense=masks_for(meta,x,12,6,boundaries)
    np.testing.assert_array_equal(sparse["test"],dense["test"])
    assert dense["train"].sum()>sparse["train"].sum()
    assert (meta.loc[dense["test"],"cutoff"]%86400==0).all()
    assert meta.loc[dense["train"],"cutoff"].max()+86400<=boundaries["validation_start"]
    assert not (sum(dense.values())>1).any()
    assert (x[dense["test"],NAMES.index("hours_since_pair_event")]<=12).all()


def test_fast_bootstrap_ap_matches_weighted_sklearn_including_ties():
    y=np.array([0,1,1,0,1,0,0,1]);score=np.array([.1,.2,.2,.4,.8,.8,.9,.9])
    block=np.array([0,0,1,1,2,2,3,3])
    prepared=ranking(y,score,block)
    for weights in (np.array([1,1,1,1]),np.array([0,3,1,2]),np.array([4,0,0,0])):
        expected=average_precision_score(y,score,sample_weight=weights[block])
        assert np.isclose(weighted_ap(prepared,weights),expected,atol=1e-12)
