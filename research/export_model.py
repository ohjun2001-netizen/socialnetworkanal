"""Export the pre-specified full tree model, not a test-selected winner."""
import json
import joblib
import numpy as np
import pandas as pd
from .data import ROOT
from .features import prepare, NAMES
from .evaluation import folds, calibrate, threshold
from .models import fit_lgb


def export(dataset):
    meta, x, _, _ = prepare(dataset)
    y = meta.label.to_numpy()
    fold, masks, boundaries = list(folds(meta))[-1]
    result = ROOT / "results" / "research_v2" / dataset
    config = json.loads((result / f"fold_{fold}" / "lightgbm_config.json").read_text())
    model = fit_lgb(x,y,masks["train"],list(range(len(NAMES))),config,2026)
    scores = {key:model.predict_proba(x[mask])[:,1] for key, mask in masks.items() if key != "train"}
    calibrated, calibrator = calibrate(y[masks["cal"]],scores["cal"],scores)
    th = threshold(y[masks["cal"]], calibrated["cal"])
    # Check exported artifact reproduces the already frozen experiment.
    saved = pd.read_csv(result / f"fold_{fold}" / "seed_2026" / "predictions.csv.gz", float_precision="round_trip")
    saved = saved[saved.model == "Full LightGBM"]
    np.testing.assert_allclose(saved.score, calibrated["test"], atol=1e-12, rtol=1e-10)
    folder = ROOT / "models" / "research_v2" / dataset
    folder.mkdir(parents=True,exist_ok=True)
    joblib.dump({"model":model,"calibrator":calibrator,"threshold":th,"feature_names":NAMES,
                 "fold":fold,"seed":2026,"boundaries":boundaries,"dataset":dataset,
                 "purpose":"reproduce last-fold Full LightGBM; not refitted on test or selected by test"},
                folder / "full_lightgbm.joblib")
    # Native C++ file opening can fail on Korean Windows paths. Python's
    # Unicode-aware writer preserves the exact native model text.
    (folder / "lightgbm_native.txt").write_text(model.booster_.model_to_string(),encoding="utf-8")
    note = {"dataset":dataset,"model":"Full LightGBM","seed":2026,"fold":fold,
            "threshold":th,"feature_names":NAMES,"reproduces_test_predictions":True,
            "warning":"Load joblib only from trusted sources. Native txt outputs uncalibrated probabilities; apply stored logit calibrator for experiment probabilities."}
    (folder / "metadata.json").write_text(json.dumps(note,indent=2),encoding="utf-8")
    print(f"Exported and verified: {folder}",flush=True)


if __name__ == "__main__":
    import argparse
    p=argparse.ArgumentParser();p.add_argument("--dataset",default="mathoverflow")
    export(p.parse_args().dataset)
