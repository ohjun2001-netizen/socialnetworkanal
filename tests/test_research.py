import numpy as np
import pandas as pd
import torch

from mathoverflow_lab.features import build_examples
from research.features import extra_features
from research.evaluation import folds, metrics, threshold, calibrate, block_comparison
from research.neural import RecurrenceNetwork


def sample_events():
    day = 86400; t = 31*day
    return pd.DataFrame([
        (0, 1, 0, "a2q"), (0, 1, t-3600, "c2q"),
        (1, 2, t-1800, "a2q"), (0, 1, t, "c2a"),
        (2, 1, t+1800, "c2q"), (3, 4, t+day+100, "c2a"),
    ], columns=["src","dst","time","kind"])


def test_future_events_change_labels_but_not_features():
    data = sample_events()
    modified = data.copy()
    modified.loc[modified.time >= 31*86400, "src"] = 4
    m1, x1 = build_examples(data); m2, x2 = build_examples(modified)
    np.testing.assert_array_equal(m1.pair_code, m2.pair_code)
    np.testing.assert_array_equal(x1, x2)
    np.testing.assert_array_equal(extra_features(data, m1), extra_features(modified, m2))
    assert m1.label.tolist() == [1, 0]  # Reverse 2->1 is not 1->2 recurrence.
    assert m2.label.tolist() == [0, 0]
    assert m1.cutoff.max()+86400 <= data.time.max()


def test_label_windows_are_mature_at_every_partition_boundary():
    meta = pd.DataFrame({"cutoff":np.repeat(np.arange(100)*86400, 2), "label":np.tile([0,1],100)})
    tests = []
    for _, masks, bounds in folds(meta):
        for left, right in (("train","tune"),("tune","cal"),("cal","test")):
            assert meta.loc[masks[left],"cutoff"].max()+86400 <= meta.loc[masks[right],"cutoff"].min()
        assert not (sum(masks.values()) > 1).any()
        tests.extend(meta.index[masks["test"]])
    assert len(tests) == len(set(tests))


def test_constant_predictions_and_ties_do_not_create_ranking_skill():
    y = np.array([1,0]*50); s = np.ones(100)
    result = metrics(y,s,threshold(y,s),np.repeat(np.arange(10),10))
    assert result["ap"] == .5
    assert result["auc"] == .5
    assert abs(result["f1"]-2/3) < 1e-12
    assert result["precision_at_5"] == .5


def test_tgn_memory_learns_from_past_and_reset_removes_state():
    torch.manual_seed(42)
    model = RecurrenceNetwork(6)
    model.reset()
    src, dst = torch.tensor([0,1,2]), torch.tensor([1,2,3])
    t, kind = torch.tensor([10.,20.,30.]), torch.tensor([0,1,2])
    model.ingest(src,dst,t,kind,gradients=True)
    logits, _ = model.predict(torch.tensor([0,1]),torch.tensor([1,2]),40.,src,t,kind)
    torch.nn.functional.binary_cross_entropy_with_logits(logits, torch.tensor([1.,0.])).backward()
    assert model.gru.weight_ih.grad is not None
    assert model.gru.weight_ih.grad.abs().sum() > 0
    assert model.last_update.max() < 40
    model.reset()
    assert torch.count_nonzero(model.memory) == 0
    assert model.neighbors.cur_e_id == 0


def test_calibration_does_not_depend_on_test_cohort():
    y = np.array([0,0,0,1,0,1,1,1])
    c = np.array([.1,.2,.3,.4,.5,.6,.7,.8])
    first, fitted = calibrate(y, c, {"cal":c,"test":np.array([.2,.6])})
    second, _ = calibrate(y, c, {"cal":c,"test":np.array([.2,.6,.01,.99])})
    np.testing.assert_array_equal(first["test"],second["test"][:2])
    assert fitted is not None
    assert first["test"][0] < first["test"][1]
    assert np.isfinite(first["test"]).all()


def test_negative_calibration_slope_keeps_original_ranking():
    y = np.array([1,1,1,0,0,0])
    c = np.array([.1,.2,.3,.7,.8,.9])
    result, fitted = calibrate(y,c,{"cal":c,"test":c})
    assert fitted is None
    np.testing.assert_array_equal(result["test"],c)


def test_paired_bootstrap_identical_models_have_zero_difference():
    rows=[]
    for day in range(28):
        for label in (0,1):
            for name in ("a","b"):
                for seed in (1,2):
                    rows.append((0,day*86400,label,2,label,name,seed,.2+.6*label))
    frame=pd.DataFrame(rows,columns=["fold","cutoff","src","dst","label","model","seed","score"])
    result=block_comparison(frame,"a","b",repeats=20)
    assert result["delta_ap"] == result["ci_low"] == result["ci_high"] == 0
