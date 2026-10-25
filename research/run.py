"""Run versioned, resumable, natural-prevalence experiments."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import platform
import time

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

from mathoverflow_lab.features import HISTORY_FEATURES, HAWKES_FEATURES, SNA_FEATURES
from .data import ROOT
from .features import NAMES, prepare
from .evaluation import folds, metrics, threshold, calibrate
from .models import fit_lgb, lightgbm_config, fit_history
from .neural import run_neural


def dump(path, obj):
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False, default=lambda x: x.item() if hasattr(x, "item") else str(x)), encoding="utf-8")


def experiment(dataset="mathoverflow", trials=16, epochs=4, seeds=(2026,2027,2028), only_fold=None, neural=True):
    meta, x, events, info = prepare(dataset)
    y = meta.label.to_numpy(np.uint8)
    output = ROOT / "results" / "research_v2" / dataset
    protocol_bytes = (ROOT / "research" / "protocol.json").read_bytes()
    protocol_hash = hashlib.sha256(protocol_bytes).hexdigest()
    # Code fingerprints identify what was actually run, independent of git cleanliness.
    code_hashes = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in (ROOT / "research").glob("*.py")}
    core_hashes = {key:code_hashes[key] for key in ("features.py","data.py","models.py","neural.py","evaluation.py")}
    dump(output / "run_manifest.json", {
        "started_at": datetime.now(timezone.utc).isoformat(), "dataset": info,
        "protocol_sha256": protocol_hash, "code_sha256": code_hashes,
        "trials": trials, "epochs": epochs, "seeds": list(seeds), "neural_enabled": neural,
        "versions": {p: importlib.metadata.version(p) for p in ("numpy","pandas","scikit-learn","lightgbm","optuna","torch","torch-geometric")},
        "python": platform.python_version(), "feature_names": NAMES,
        "scope": "retrospective rolling evaluation" if dataset == "mathoverflow" else "external-dataset replication with within-domain training; not zero-shot transfer",
    })
    groups = {
        "Full LightGBM": list(range(len(NAMES))),
        "No types LightGBM": [i for i,n in enumerate(NAMES) if not n.startswith("type_")],
        "No decay LightGBM": [i for i,n in enumerate(NAMES) if "decay" not in n],
        "No structure LightGBM": [i for i,n in enumerate(NAMES) if n not in SNA_FEATURES and n != "time_ordered_twohop_30d"],
    }
    for fold, masks, boundaries in folds(meta):
        if only_fold is not None and fold != only_fold:
            continue
        folder = output / f"fold_{fold}"
        folder.mkdir(parents=True, exist_ok=True)
        config_path = folder / "lightgbm_config.json"
        split_info = {key: {"n": int(mask.sum()), "positive": int(y[mask].sum()),
                           "first": int(meta.loc[mask,"cutoff"].min()), "last": int(meta.loc[mask,"cutoff"].max()),
                           "last_label_end": int(meta.loc[mask,"cutoff"].max()+86400)} for key, mask in masks.items()}
        dump(folder / "split.json", {"fold":fold, **boundaries, "partitions":split_info})
        if config_path.exists():
            config = json.loads(config_path.read_text(encoding="utf-8"))
        else:
            print(f"{dataset} fold={fold}: Optuna on tuning interval", flush=True)
            config, search = lightgbm_config(x, y, masks, trials)
            search.to_csv(folder / "optuna_trials.csv", index=False)
            dump(config_path, config)
        for seed in seeds:
            destination = folder / f"seed_{seed}"
            destination.mkdir(exist_ok=True)
            success = destination / "complete.json"
            if success.exists():
                completed = json.loads(success.read_text())
                old_core = {key:completed.get("code_sha256", {}).get(key) for key in core_hashes}
                if (completed.get("neural") == neural and completed.get("protocol_sha256") == protocol_hash
                        and old_core == core_hashes and completed.get("epochs", 4) == epochs):
                    print(f"resume: {dataset} fold={fold} seed={seed} complete", flush=True)
                    continue
            begin = time.perf_counter()
            print(f"{dataset} fold={fold} seed={seed}: training", flush=True)
            scores, fitted, details = {}, {}, {}
            for name, columns in groups.items():
                model = fit_lgb(x, y, masks["train"], columns, config, seed)
                scores[name] = {key: model.predict_proba(x[mask][:,columns])[:,1] for key,mask in masks.items() if key != "train"}
                fitted[name] = {"model":model, "columns":columns}
            cols = [NAMES.index(n) for n in HISTORY_FEATURES]
            model = fit_history(x, y, masks, cols, seed)
            scores["History HGB"] = {key:model.predict_proba(x[mask][:,cols])[:,1] for key,mask in masks.items() if key != "train"}
            fitted["History HGB"] = {"model":model, "columns":cols}

            # Raw scores retain ties and do not depend on evaluation cohort ranks.
            for name, col, negate in (("Pair frequency", "pair_count_lifetime", False),
                                      ("Recency", "hours_since_pair_event", True),
                                      ("Common neighbors", "common_neighbors_30d", False),
                                      ("Adamic Adar", "adamic_adar_30d", False),
                                      ("Jaccard", "jaccard_30d", False),
                                      ("Decay 6h", "pair_decay_6h", False)):
                v = x[:,NAMES.index(col)] * (-1 if negate else 1)
                scores[name] = {key:v[mask] for key,mask in masks.items() if key != "train"}
            for name, value in (("Always positive", 1.), ("Train prevalence", float(y[masks["train"]].mean())), ("EdgeBank", 1.)):
                scores[name] = {key:np.full(mask.sum(), value) for key,mask in masks.items() if key != "train"}

            if neural:
                for mode,name in (("tgn","TGN recurrence"),("sage","Rolling GraphSAGE")):
                    cached = destination / f"{mode}_scores.npz"
                    details_path = destination / f"{mode}_details.json"
                    cached_details = json.loads(details_path.read_text()) if details_path.exists() else {}
                    cached_hash = cached_details.get("neural_code_sha256", code_hashes["neural.py"])
                    if cached.exists() and details_path.exists() and cached_hash == code_hashes["neural.py"] and cached_details.get("max_epochs",4) == epochs:
                        neural_score = np.load(cached)["score"]
                        details[name] = json.loads(details_path.read_text())
                    else:
                        neural_score, details[name] = run_neural(events, meta, masks, mode, seed, destination, epochs)
                        details[name].update(neural_code_sha256=code_hashes["neural.py"], max_epochs=epochs)
                        np.savez_compressed(cached, score=neural_score)
                        dump(details_path, details[name])
                    scores[name] = {key:neural_score[mask] for key,mask in masks.items() if key != "train"}
                # A validation-only score ensemble avoids training a second-stage
                # model on in-sample supervised neural embeddings.
                best_weight = max((0., .25, .5, .75, 1.), key=lambda w: average_precision_score(y[masks["tune"]],
                    w*scores["TGN recurrence"]["tune"]+(1-w)*scores["Full LightGBM"]["tune"]))
                scores["Hybrid ensemble"] = {key:best_weight*scores["TGN recurrence"][key]+(1-best_weight)*scores["Full LightGBM"][key] for key in ("tune","cal","test")}
                details["Hybrid ensemble"] = {"neural_weight":best_weight, "selected_on":"tune only"}

            # Select named candidate by tuning AP before consulting test labels.
            eligible = list(groups) + ["History HGB"] + (["TGN recurrence", "Rolling GraphSAGE", "Hybrid ensemble"] if neural else [])
            selected = max(eligible, key=lambda name: average_precision_score(y[masks["tune"]], scores[name]["tune"]))
            details["selected_by_tuning_ap"] = selected
            rows, predictions = [], []
            for name, s in scores.items():
                is_probability = name in eligible or name in ("Always positive","Train prevalence","EdgeBank")
                if name in eligible:
                    s, calibration = calibrate(y[masks["cal"]], s["cal"], s)
                    if name in fitted:
                        fitted[name]["calibrator"] = calibration
                th = threshold(y[masks["cal"]], s["cal"])
                if name in fitted:
                    fitted[name]["threshold"] = th
                result = metrics(y[masks["test"]], s["test"], th, meta.loc[masks["test"],"cutoff"].to_numpy(), is_probability)
                rows.append({"dataset":dataset,"fold":fold,"seed":seed,"model":name, "tune_ap":float(average_precision_score(y[masks["tune"]], scores[name]["tune"])), **result})
                part = meta.loc[masks["test"], ["cutoff","src","dst","label"]].copy()
                part["score"] = s["test"]; part["prediction"] = s["test"] >= th
                part["model"] = name; part["fold"] = fold; part["seed"] = seed
                predictions.append(part)
            if selected in fitted:
                joblib.dump(fitted[selected], destination / "validation_selected_model.joblib")
            full = fitted["Full LightGBM"]["model"]
            pd.DataFrame({"feature":NAMES, "gain":full.booster_.feature_importance(importance_type="gain")}).to_csv(destination / "feature_gain.csv", index=False)
            pd.DataFrame(rows).to_csv(destination / "metrics.csv", index=False)
            pd.concat(predictions).to_csv(destination / "predictions.csv.gz", index=False, compression="gzip")
            dump(destination / "details.json", details)
            dump(success, {"neural":neural,"protocol_sha256":protocol_hash,"code_sha256":code_hashes,"epochs":epochs,
                           "elapsed_seconds":time.perf_counter()-begin,"completed_at":datetime.now(timezone.utc).isoformat()})
            print(f"{dataset} fold={fold} seed={seed}: complete in {time.perf_counter()-begin:.1f}s", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=("mathoverflow","askubuntu"), default="mathoverflow")
    parser.add_argument("--trials", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--seeds", type=int, nargs="+", default=[2026,2027,2028])
    parser.add_argument("--fold", type=int)
    parser.add_argument("--no-neural", action="store_true")
    a=parser.parse_args()
    experiment(a.dataset,a.trials,a.epochs,a.seeds,a.fold,not a.no_neural)
