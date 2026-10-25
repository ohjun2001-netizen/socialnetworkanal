"""Leakage-safe activity/choice decomposition and recency-type experts."""
from __future__ import annotations

import numpy as np
import pandas as pd
from numba import njit, prange

from research.data import KINDS
from research.features import NAMES
from research.evaluation import calibrate
from research.models import fit_lgb

ACTIVITY_FEATURES = [
    "src_out_24h", "src_out_168h", "src_out_720h", "src_in_168h",
    "src_acceleration", "weekday_sin", "weekday_cos",
    *[f"type_{kind}_src_out_168h" for kind in KINDS],
]


@njit
def lower_bound(values, left, right, target):
    while left < right:
        middle = (left + right) // 2
        if values[middle] < target:
            left = middle + 1
        else:
            right = middle
    return left


@njit(parallel=True)
def future_source_labels(cutoff, source, offsets, times):
    result = np.zeros(len(cutoff), np.uint8)
    for row in prange(len(cutoff)):
        left, right = offsets[source[row]], offsets[source[row] + 1]
        first = lower_bound(times, left, right, cutoff[row])
        result[row] = first < right and times[first] < cutoff[row] + 86400
    return result


def activity_view(meta: pd.DataFrame, x: np.ndarray, events: pd.DataFrame):
    """One row per source-cutoff, past-only features plus candidate count."""
    cutoff = meta.cutoff.to_numpy(np.int64); source = meta.src.to_numpy(np.int64)
    first = np.r_[True, (cutoff[1:] != cutoff[:-1]) | (source[1:] != source[:-1])]
    first_index = np.flatnonzero(first)
    group = np.cumsum(first).astype(np.int64) - 1
    unique = meta.iloc[first_index][["cutoff", "src"]].reset_index(drop=True)
    if unique.duplicated().any():
        raise ValueError("source-cutoff rows are not contiguous")
    counts = np.bincount(group).astype(np.float32)
    cols = [NAMES.index(name) for name in ACTIVITY_FEATURES]
    ax = np.column_stack([x[first_index][:, cols], np.log1p(counts)]).astype(np.float32)
    order = np.lexsort((events.time.to_numpy(), events.src.to_numpy()))
    src = events.src.to_numpy(np.int64)[order]; times = events.time.to_numpy(np.int64)[order]
    nodes = int(max(events.src.max(), events.dst.max())) + 1
    offsets = np.r_[0, np.cumsum(np.bincount(src, minlength=nodes))].astype(np.int64)
    ay = future_source_labels(unique.cutoff.to_numpy(np.int64), unique.src.to_numpy(np.int64), offsets, times)
    pair_activity = ay[group]
    # Every pair reconnection requires an outgoing event by its source.
    if not np.all(pair_activity[meta.label.to_numpy(bool)]):
        raise AssertionError("pair label without source activity")
    return unique, ax, ay, group, pair_activity, first_index


def expert_group(x: np.ndarray) -> np.ndarray:
    recent = (x[:, NAMES.index("hours_since_pair_event")] > 4).astype(np.int8)
    last = np.argmax(x[:, [NAMES.index(f"type_{kind}_last") for kind in KINDS]], axis=1).astype(np.int8)
    return recent * len(KINDS) + last


def calibrated_predictions(model, x, y, cal_mask, test_mask, columns):
    raw_cal = model.predict_proba(x[cal_mask][:, columns])[:, 1]
    raw_test = model.predict_proba(x[test_mask][:, columns])[:, 1]
    scores, fitted = calibrate(y[cal_mask], raw_cal, {"cal": raw_cal, "test": raw_test})
    return scores["cal"], scores["test"], fitted


def fit_activity(ax, ay, masks, config, seed):
    model = fit_lgb(ax, ay, masks["train"], list(range(ax.shape[1])), config, seed)
    cal, test, calibration = calibrated_predictions(
        model, ax, ay, masks["cal"], masks["test"], list(range(ax.shape[1])))
    return model, cal, test, calibration


def fit_choice(x, y, train_mask, cal_mask, test_mask, pair_activity, config, seed):
    active_train = train_mask & pair_activity.astype(bool)
    active_cal = cal_mask & pair_activity.astype(bool)
    if y[active_train].min() == y[active_train].max() or y[active_cal].min() == y[active_cal].max():
        raise ValueError("choice partition lacks two classes")
    columns = list(range(x.shape[1]))
    model = fit_lgb(x, y, active_train, columns, config, seed)
    raw_cal = model.predict_proba(x[cal_mask])[:, 1]
    raw_test = model.predict_proba(x[test_mask])[:, 1]
    cal_active_local = pair_activity[cal_mask].astype(bool)
    scores, calibration = calibrate(y[cal_mask][cal_active_local], raw_cal[cal_active_local],
                                    {"cal": raw_cal, "test": raw_test})
    return model, scores["cal"], scores["test"], calibration


def fit_experts(x, y, groups, train_mask, cal_mask, test_mask, config, seed,
                pair_activity=None):
    """Six fixed groups. If pair_activity is supplied, learn choice | activity."""
    cal_score = np.empty(cal_mask.sum(), np.float64)
    test_score = np.empty(test_mask.sum(), np.float64)
    cal_groups = groups[cal_mask]; test_groups = groups[test_mask]
    models = {}; calibrators = {}
    for group in range(6):
        train = train_mask & (groups == group)
        cal = cal_mask & (groups == group)
        test = test_mask & (groups == group)
        if pair_activity is not None:
            train &= pair_activity.astype(bool)
            calibration_rows = pair_activity[cal].astype(bool)
        else:
            calibration_rows = np.ones(cal.sum(), bool)
        if train.sum() < 200 or len(np.unique(y[train])) != 2 or len(np.unique(y[cal][calibration_rows])) != 2:
            raise ValueError(f"expert group {group} lacks usable train/cal classes")
        model = fit_lgb(x, y, train, list(range(x.shape[1])), config, seed)
        raw_cal = model.predict_proba(x[cal])[:, 1]
        raw_test = model.predict_proba(x[test])[:, 1]
        scores, fitted = calibrate(y[cal][calibration_rows], raw_cal[calibration_rows],
                                   {"cal": raw_cal, "test": raw_test})
        cal_score[cal_groups == group] = scores["cal"]
        test_score[test_groups == group] = scores["test"]
        models[group] = model; calibrators[group] = fitted
    if not np.isfinite(cal_score).all() or not np.isfinite(test_score).all():
        raise AssertionError("expert score missing")
    return models, cal_score, test_score, calibrators


def expand_activity_scores(group, mask, activity_mask, values):
    full = np.full(activity_mask.shape, np.nan)
    full[activity_mask] = values
    result = full[group[mask]]
    if not np.isfinite(result).all():
        raise AssertionError("unmapped activity scores")
    return result
