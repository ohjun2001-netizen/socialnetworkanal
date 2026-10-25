"""No test-driven thresholding; natural prevalence and block uncertainty."""
from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.metrics import (average_precision_score, roc_auc_score, f1_score, precision_score,
                             recall_score, matthews_corrcoef, brier_score_loss, precision_recall_curve)
from sklearn.linear_model import LogisticRegression


def folds(meta: pd.DataFrame):
    times = np.sort(meta.cutoff.unique())
    t = meta.cutoff.to_numpy()
    for f, start in enumerate((.5, .6, .7, .8, .9)):
        test0 = int(times[round(len(times)*start)])
        stop_i = round(len(times)*(start+.1))
        test1 = int(times[stop_i]) if stop_i < len(times) else int(times[-1]+86400)
        val0 = int(times[round(len(times)*(start-.1))])
        cal0 = int(times[round(len(times)*(start-.05))])
        masks = {
            "train": t+86400 <= val0,
            "tune": (t >= val0) & (t+86400 <= cal0),
            "cal": (t >= cal0) & (t+86400 <= test0),
            "test": (t >= test0) & (t < test1),
        }
        for key, mask in masks.items():
            if not mask.any() or meta.loc[mask, "label"].nunique() != 2:
                raise ValueError(f"Fold {f}/{key} lacks two classes")
        yield f, masks, {"validation_start": val0, "calibration_start": cal0, "test_start": test0, "test_stop": test1}


def threshold(y, s):
    precision, recall, ts = precision_recall_curve(y, s)
    f = 2*precision[:-1]*recall[:-1]/np.maximum(precision[:-1]+recall[:-1], 1e-12)
    return float(ts[np.argmax(f)])


def calibrate(cal_y, cal_s, scores):
    # Calibration is fitted only on the held-out calibration interval.
    transform = lambda x: np.log(np.clip(x, 1e-6, 1-1e-6)/(1-np.clip(x, 1e-6, 1-1e-6))).reshape(-1, 1)
    model = LogisticRegression(C=1., max_iter=300).fit(transform(cal_s), cal_y)
    # A negative slope could invert ranking on a short interval. Keep identity
    # in that case; this decision also depends exclusively on calibration data.
    if model.coef_[0, 0] <= 0:
        return {key: np.clip(value, 0, 1) for key, value in scores.items()}, None
    return {key: model.predict_proba(transform(value))[:, 1] for key, value in scores.items()}, model


def metrics(y, s, th, cutoffs=None, probability=True):
    y = np.asarray(y); s = np.asarray(s); pred = s >= th
    both = len(np.unique(y)) == 2
    ap = float(average_precision_score(y, s)) if y.sum() else None
    output = {"ap": ap, "auc": float(roc_auc_score(y, s)) if both else None,
              "f1": float(f1_score(y, pred, zero_division=0)),
              "precision": float(precision_score(y, pred, zero_division=0)),
              "recall": float(recall_score(y, pred, zero_division=0)),
              "mcc": float(matthews_corrcoef(y, pred)),
              "brier": float(brier_score_loss(y, s)) if probability else None,
              "prevalence": float(y.mean()), "ap_lift": ap/y.mean() if ap is not None else None,
              "n": len(y), "positive": int(y.sum()), "threshold": th}
    if cutoffs is not None:
        hits = selected = 0
        # Ties get expected fractional credit, avoiding row-order bias.
        for t in np.unique(cutoffs):
            loc = np.flatnonzero(cutoffs == t); count = min(5, len(loc))
            boundary = np.partition(s[loc], -count)[-count]
            above = s[loc] > boundary; tied = s[loc] == boundary
            hits += y[loc][above].sum() + (count-above.sum()) * y[loc][tied].mean()
            selected += count
        output.update(precision_at_5=float(hits/selected), recall_at_5=float(hits/max(y.sum(), 1)))
    return output


def block_comparison(frame, model_a, model_b, repeats=1000, block_days=7, seed=82):
    # Seed ensemble first: a seed is not an independent data replicate.
    keys = ["fold", "cutoff", "src", "dst", "label"]
    table = frame[frame.model.isin([model_a, model_b])].groupby(keys+["model"], observed=True).score.mean().unstack("model").reset_index()
    if model_a not in table or model_b not in table or table[[model_a, model_b]].isna().any().any():
        raise ValueError("Unmatched predictions for paired comparison")
    table["block"] = table.cutoff // (86400*block_days)
    blocks = [part.index.to_numpy() for _, part in table.groupby(["fold", "block"])]
    y = table.label.to_numpy(); a = table[model_a].to_numpy(); b = table[model_b].to_numpy()
    rng = np.random.default_rng(seed); deltas = []
    for _ in range(repeats):
        ix = np.concatenate([blocks[i] for i in rng.integers(0, len(blocks), len(blocks))])
        if y[ix].sum() and (1-y[ix]).sum():
            deltas.append(average_precision_score(y[ix], a[ix])-average_precision_score(y[ix], b[ix]))
    delta = average_precision_score(y, a)-average_precision_score(y, b)
    return {"model_a": model_a, "model_b": model_b, "delta_ap": float(delta),
            "ci_low": float(np.quantile(deltas, .025)), "ci_high": float(np.quantile(deltas, .975)),
            "block_days": block_days, "blocks": len(blocks), "repeats": len(deltas),
            "interpretation": "paired time-block bootstrap; temporal dependence beyond block length may remain"}
