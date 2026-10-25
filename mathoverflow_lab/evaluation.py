"""Temporal-holdout metrics and validation-only operating-point selection."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    average_precision_score, balanced_accuracy_score, brier_score_loss,
    confusion_matrix, f1_score, matthews_corrcoef, precision_recall_curve,
    precision_score, recall_score, roc_auc_score,
)


def choose_threshold(labels: np.ndarray, scores: np.ndarray) -> float:
    precision, recall, thresholds = precision_recall_curve(labels, scores)
    if not len(thresholds):
        return 0.5
    f1 = 2 * precision[:-1] * recall[:-1] / np.maximum(precision[:-1] + recall[:-1], 1e-12)
    return float(thresholds[int(np.nanargmax(f1))])


def binary_metrics(labels: np.ndarray, scores: np.ndarray, threshold: float) -> dict:
    predicted = scores >= threshold
    tn, fp, fn, tp = confusion_matrix(labels, predicted, labels=[0, 1]).ravel()
    prevalence = float(np.mean(labels))
    ap = float(average_precision_score(labels, scores))
    return {
        "roc_auc": float(roc_auc_score(labels, scores)),
        "average_precision": ap,
        "ap_lift": ap / prevalence if prevalence else None,
        "f1": float(f1_score(labels, predicted, zero_division=0)),
        "precision": float(precision_score(labels, predicted, zero_division=0)),
        "recall": float(recall_score(labels, predicted, zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(labels, predicted)),
        "mcc": float(matthews_corrcoef(labels, predicted)),
        "brier": float(brier_score_loss(labels, np.clip(scores, 0, 1))),
        "threshold": float(threshold),
        "candidates": int(len(labels)), "positives": int(labels.sum()),
        "prevalence": prevalence, "tp": int(tp), "fp": int(fp), "tn": int(tn), "fn": int(fn),
    }


def balanced_repeats(
    validation_labels: np.ndarray,
    validation_scores: np.ndarray,
    test_labels: np.ndarray,
    test_scores: np.ndarray,
    *, repeats: int = 100,
    seed: int = 2026,
) -> list[dict]:
    results = []
    for repeat in range(repeats):
        rng = np.random.default_rng(seed + repeat)
        val_pos = np.flatnonzero(validation_labels == 1)
        val_neg = rng.choice(np.flatnonzero(validation_labels == 0), len(val_pos), replace=False)
        test_pos = np.flatnonzero(test_labels == 1)
        test_neg = rng.choice(np.flatnonzero(test_labels == 0), len(test_pos), replace=False)
        val_index = np.r_[val_pos, val_neg]
        test_index = np.r_[test_pos, test_neg]
        threshold = choose_threshold(validation_labels[val_index], validation_scores[val_index])
        results.append({"repeat": repeat, **binary_metrics(test_labels[test_index], test_scores[test_index], threshold)})
    return results


def percentile_rank(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="stable")
    ranks = np.empty(len(values), dtype=np.float64)
    ranks[order] = np.arange(len(values), dtype=np.float64)
    return ranks / max(len(values) - 1, 1)

