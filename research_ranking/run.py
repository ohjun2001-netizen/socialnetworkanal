"""Train CUDA ranking models without computing test metrics."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import time

import joblib
import numpy as np
import torch

from research.data import ROOT
from research.evaluation import calibrate, threshold
from research_confirmation.features import prepare, masks
from .model import predict_logits, train_ranknet

MODELS = ["Frozen Baseline Full LightGBM", "CUDA RankNet", "Fixed 50/50 ensemble"]
OUT = ROOT / "results" / "ranking_v1"


def dump(path, value):
    path.write_text(json.dumps(value, indent=2, default=lambda v: v.item() if hasattr(v, "item") else str(v)), encoding="utf-8")


def sigmoid(value):
    return 1 / (1 + np.exp(-np.clip(value, -30, 30)))


def run(fold):
    meta, x, _, _, splits, feature_manifest = prepare(); y = meta.label.to_numpy(np.uint8)
    partition = masks(meta, splits[fold])
    folder = OUT / f"fold_{fold}"; folder.mkdir(parents=True, exist_ok=True)
    source_paths = [ROOT / "research_ranking" / name for name in ("protocol.json", "model.py", "run.py", "report.py")]
    source_paths += [ROOT / "research_confirmation" / "features.py", ROOT / "research" / "evaluation.py"]
    hashes = {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest() for path in source_paths}
    signature = hashlib.sha256((json.dumps(hashes, sort_keys=True) + feature_manifest["cache_sha256"]).encode()).hexdigest()
    for seed in (2026, 2027, 2028):
        destination = folder / f"seed_{seed}"; destination.mkdir(exist_ok=True)
        complete = destination / "complete.json"
        if complete.exists() and json.loads(complete.read_text()).get("signature") == signature:
            print(f"resume fold={fold} seed={seed}", flush=True); continue
        begin = time.perf_counter()
        print(f"fold={fold} seed={seed}: CUDA RankNet pointwise + pairwise training", flush=True)
        model, mean, scale, history, device, pair_count = train_ranknet(
            x, y, meta, partition["train"], partition["tune"], seed)
        cal_indexes = np.flatnonzero(partition["cal"]); test_indexes = np.flatnonzero(partition["test"])
        raw_cal = sigmoid(predict_logits(model, x, cal_indexes, mean, scale, torch.device("cuda:0")))
        raw_test = sigmoid(predict_logits(model, x, test_indexes, mean, scale, torch.device("cuda:0")))
        calibrated, calibrator = calibrate(y[partition["cal"]], raw_cal, {"cal": raw_cal, "test": raw_test})
        ranker_cal, ranker_test = calibrated["cal"], calibrated["test"]
        ranker_threshold = threshold(y[partition["cal"]], ranker_cal)

        baseline_folder = ROOT / "results" / "confirmation_superuser_v1" / f"fold_{fold}" / f"seed_{seed}"
        baseline_path = baseline_folder / "locked_test_scores.npz"
        baseline = np.load(baseline_path, allow_pickle=False)
        for key in ("cutoff", "src", "dst", "label"):
            expected = meta.loc[partition["test"], key].to_numpy()
            if not np.array_equal(baseline[key], expected):
                raise AssertionError(f"frozen baseline mismatch: {key}")
        baseline_test = baseline["score_0"].astype(np.float64)
        baseline_threshold = float(json.loads((baseline_folder / "thresholds.json").read_text())["Baseline Full LightGBM"])
        ensemble_test = 0.5 * baseline_test + 0.5 * ranker_test
        ensemble_threshold = 0.5 * baseline_threshold + 0.5 * ranker_threshold
        test = meta.loc[partition["test"], ["cutoff", "src", "dst", "label"]].reset_index(drop=True)
        np.savez_compressed(destination / "locked_test_scores.npz",
                            **{key: test[key].to_numpy() for key in test},
                            score_baseline=baseline_test, score_ranker=ranker_test, score_ensemble=ensemble_test,
                            threshold_baseline=np.array([baseline_threshold]),
                            threshold_ranker=np.array([ranker_threshold]),
                            threshold_ensemble=np.array([ensemble_threshold]))
        dump(destination / "training_record.json", {
            "fold": fold, "seed": seed, "signature": signature, "history": history,
            "train_rows": int(partition["train"].sum()), "tune_rows": int(partition["tune"].sum()),
            "cal_rows": int(partition["cal"].sum()), "test_rows": int(partition["test"].sum()),
            "pairwise_training_pairs": int(pair_count), "device": device,
            "calibration_and_threshold": "calibration interval only", "test_metrics_computed": False,
            "baseline_locked_sha256": hashlib.sha256(baseline_path.read_bytes()).hexdigest(),
        })
        if fold == 4 and seed == 2026:
            torch.save({"state_dict": {key: value.detach().cpu() for key, value in model.state_dict().items()},
                        "mean": mean, "scale": scale, "feature_count": x.shape[1]}, destination / "ranknet.pt")
            joblib.dump(calibrator, destination / "calibrator.joblib")
        elapsed = time.perf_counter() - begin
        dump(complete, {"signature": signature, "source_sha256": hashes,
                        "feature_cache_sha256": feature_manifest["cache_sha256"],
                        "gpu_required_and_used": True, "test_metrics_computed": False,
                        "elapsed_seconds": elapsed, "completed": datetime.now(timezone.utc).isoformat()})
        print(f"fold={fold} seed={seed}: locked CUDA scores in {elapsed:.1f}s; test metric untouched", flush=True)
        del model; torch.cuda.empty_cache()
    dump(folder / "complete.json", {"fold": fold, "signature": signature, "seeds": [2026, 2027, 2028],
                                     "gpu_required_and_used": True, "test_metrics_computed": False,
                                     "completed": datetime.now(timezone.utc).isoformat()})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(); parser.add_argument("--fold", type=int, choices=range(5), required=True)
    run(parser.parse_args().fold)
