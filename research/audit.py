"""Independently verify saved predictions, split boundaries and metric arithmetic."""
import json
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

from .data import ROOT
from .features import prepare
from .evaluation import folds, metrics


def audit(dataset):
    meta, x, events, info = prepare(dataset)
    assert np.isfinite(x).all(), "Non-finite feature"
    assert not meta.duplicated(["cutoff", "src", "dst"]).any()
    assert (meta.cutoff + 86400 <= events.time.max()).all()
    # Reconstruct a deterministic sample of labels directly from raw events,
    # independent of the feature generation helper.
    rng = np.random.default_rng(390)
    checks = meta.iloc[rng.choice(len(meta), min(1000, len(meta)), replace=False)]
    for row in checks.itertuples():
        actual = ((events.src == row.src) & (events.dst == row.dst) &
                  (events.time >= row.cutoff) & (events.time < row.cutoff + 86400)).any()
        past = ((events.src == row.src) & (events.dst == row.dst) &
                (events.time >= row.cutoff - 14400) & (events.time < row.cutoff)).any()
        assert past and int(actual) == row.label
    folder = ROOT / "results" / "research_v2" / dataset
    unique_test_indices = set()
    evaluated = 0
    device_counts = {}
    protocol_hash=hashlib.sha256((ROOT/"research"/"protocol.json").read_bytes()).hexdigest()
    core_hashes={name:hashlib.sha256((ROOT/"research"/name).read_bytes()).hexdigest()
                 for name in ("data.py","features.py","models.py","neural.py","evaluation.py")}
    for fold, masks, boundaries in folds(meta):
        assert len(pd.read_csv(folder/f"fold_{fold}"/"optuna_trials.csv")) == 16
        for left, right in (("train","tune"),("tune","cal"),("cal","test")):
            assert meta.loc[masks[left], "cutoff"].max() + 86400 <= meta.loc[masks[right], "cutoff"].min()
        test_indices = set(meta.index[masks["test"]])
        assert not test_indices.intersection(unique_test_indices)
        unique_test_indices.update(test_indices)
        truth = meta.loc[masks["test"], ["cutoff","src","dst","label"]].reset_index(drop=True)
        for seed in (2026,2027,2028):
            run = folder / f"fold_{fold}" / f"seed_{seed}"
            assert (run / "complete.json").exists()
            completion=json.loads((run/"complete.json").read_text())
            assert completion["protocol_sha256"] == protocol_hash
            assert completion["neural"] and completion.get("epochs",4) == 4
            assert all(completion["code_sha256"][name] == value for name,value in core_hashes.items())
            saved = pd.read_csv(run / "predictions.csv.gz", float_precision="round_trip")
            table = pd.read_csv(run / "metrics.csv", float_precision="round_trip").set_index("model")
            assert saved.model.nunique() == 17
            for name, p in saved.groupby("model", sort=False):
                p = p.reset_index(drop=True)
                pd.testing.assert_frame_equal(p[["cutoff","src","dst","label"]], truth, check_dtype=False)
                assert np.isfinite(p.score).all()
                row = table.loc[name]
                np.testing.assert_array_equal(p.prediction, p.score >= row.threshold)
                reconstructed = metrics(p.label.to_numpy(), p.score.to_numpy(), row.threshold,
                                        p.cutoff.to_numpy(), probability=pd.notna(row.brier))
                for metric in ("ap","auc","f1","precision","recall","mcc","precision_at_5","recall_at_5"):
                    assert np.isclose(reconstructed[metric], row[metric], atol=1e-10), (dataset,fold,seed,name,metric)
                evaluated += 1
            assert np.isclose(table.loc["Always positive","ap"], truth.label.mean())
            assert np.isclose(table.loc["Always positive","auc"], .5)
            for mode in ("tgn","sage"):
                detail = json.loads((run / f"{mode}_details.json").read_text())
                device_counts[detail["device"]] = device_counts.get(detail["device"],0)+1
    result = {"dataset":dataset,"passed":True,"verified_model_runs":evaluated,
              "label_spotchecks":len(checks),"test_candidates":len(unique_test_indices),
              "neural_device_counts":device_counts,"checks":["past-only candidate existence",
              "protocol and core-code fingerprints", "declared tuning and epoch budgets",
              "24h raw label reconstruction", "full future horizon", "mature labels at split boundaries",
              "disjoint test periods", "same candidates for every model", "finite scores",
              "metric recomputation from saved predictions", "constant-score baseline ties"]}
    (folder / "audit.json").write_text(json.dumps(result,indent=2), encoding="utf-8")
    print(json.dumps(result,indent=2),flush=True)
    return result


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser();p.add_argument("--dataset",default="mathoverflow")
    audit(p.parse_args().dataset)
