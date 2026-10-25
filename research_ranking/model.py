"""CUDA RankNet with pointwise BCE and within-source pairwise ranking loss."""
from __future__ import annotations

import copy
import math

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score
from torch import nn
from torch.nn import functional as F

from research.features import NAMES


class RankNet(nn.Module):
    def __init__(self, width: int):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(width, 128), nn.LayerNorm(128), nn.SiLU(), nn.Dropout(0.10),
            nn.Linear(128, 64), nn.LayerNorm(64), nn.SiLU(), nn.Dropout(0.10),
            nn.Linear(64, 1),
        )

    def forward(self, value):
        return self.network(value).squeeze(-1)


def require_cuda():
    if not torch.cuda.is_available():
        raise RuntimeError("ranking_v1 requires CUDA; refusing silent CPU fallback")
    return torch.device("cuda:0")


def feature_moments(x: np.ndarray, indexes: np.ndarray, chunk: int = 262_144):
    total = np.zeros(x.shape[1], np.float64)
    square = np.zeros(x.shape[1], np.float64)
    count = 0
    for start in range(0, len(indexes), chunk):
        part = x[indexes[start:start + chunk]].astype(np.float64, copy=False)
        total += part.sum(axis=0); square += np.square(part).sum(axis=0); count += len(part)
    mean = total / count
    variance = np.maximum(square / count - mean * mean, 1e-8)
    scale = np.sqrt(variance)
    scale[scale < 1e-4] = 1.0
    return mean.astype(np.float32), scale.astype(np.float32)


def _group_keys(meta: pd.DataFrame, indexes: np.ndarray):
    base = int(meta.src.max()) + 1
    return meta.cutoff.to_numpy(np.int64)[indexes] * np.int64(base) + meta.src.to_numpy(np.int64)[indexes]


def build_pair_indexes(meta: pd.DataFrame, x: np.ndarray, y: np.ndarray, mask: np.ndarray, seed: int):
    """Positive/hard-negative and positive/random-negative pairs within source-cutoff groups."""
    indexes = np.flatnonzero(mask)
    keys = _group_keys(meta, indexes)
    if len(keys) > 1 and not np.all(keys[:-1] <= keys[1:]):
        order = np.argsort(keys, kind="stable"); indexes = indexes[order]; keys = keys[order]
    positive = indexes[y[indexes] == 1]
    if not len(positive):
        raise ValueError("no positive training candidates")
    positive_keys = _group_keys(meta, positive)
    groups = np.unique(positive_keys)
    left = np.searchsorted(keys, groups, side="left"); right = np.searchsorted(keys, groups, side="right")
    pair_count = NAMES.index("pair_count_24h")
    recency = NAMES.index("hours_since_pair_event")
    heuristic = np.log1p(np.maximum(x[:, pair_count], 0)) - 0.05 * x[:, recency]
    rng = np.random.default_rng(seed)
    positive_parts, negative_parts = [], []
    for group, begin, stop in zip(groups, left, right):
        members = indexes[begin:stop]
        pos = members[y[members] == 1]; neg = members[y[members] == 0]
        if not len(pos) or not len(neg):
            continue
        hard = neg[int(np.argmax(heuristic[neg]))]
        positive_parts.append(pos); negative_parts.append(np.full(len(pos), hard, np.int64))
        if len(neg) > 1:
            choices = neg[neg != hard]
            positive_parts.append(pos); negative_parts.append(rng.choice(choices, len(pos), replace=True))
    if not positive_parts:
        raise ValueError("no source-cutoff group contains both classes")
    return np.concatenate(positive_parts), np.concatenate(negative_parts)


def _tensor(x, indexes, mean, scale, device):
    value = (x[indexes].astype(np.float32, copy=False) - mean) / scale
    value = np.clip(value, -12, 12)
    return torch.from_numpy(value).to(device, non_blocking=True)


@torch.inference_mode()
def predict_logits(model, x, indexes, mean, scale, device, batch_size=131_072):
    model.eval(); pieces = []
    for start in range(0, len(indexes), batch_size):
        xb = _tensor(x, indexes[start:start + batch_size], mean, scale, device)
        with torch.amp.autocast("cuda", dtype=torch.float16):
            pieces.append(model(xb).float().cpu().numpy())
    return np.concatenate(pieces) if pieces else np.empty(0, np.float32)


def train_ranknet(x, y, meta, train_mask, tune_mask, seed, maximum_epochs=4, batch_size=65_536):
    device = require_cuda()
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    rng = np.random.default_rng(seed)
    train_indexes = np.flatnonzero(train_mask); tune_indexes = np.flatnonzero(tune_mask)
    mean, scale = feature_moments(x, train_indexes)
    pair_positive, pair_negative = build_pair_indexes(meta, x, y, train_mask, seed)
    positive_weight = min(float((y[train_indexes] == 0).sum() / max((y[train_indexes] == 1).sum(), 1)), 20.0)
    model = RankNet(x.shape[1]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scaler = torch.amp.GradScaler("cuda")
    best_ap, best_state, stale, history = -math.inf, None, 0, []
    torch.cuda.reset_peak_memory_stats(device)
    for epoch in range(maximum_epochs):
        model.train(); point_order = rng.permutation(train_indexes); point_loss = 0.0; point_batches = 0
        for start in range(0, len(point_order), batch_size):
            batch = point_order[start:start + batch_size]
            xb = _tensor(x, batch, mean, scale, device)
            yb = torch.from_numpy(y[batch].astype(np.float32, copy=False)).to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", dtype=torch.float16):
                logits = model(xb)
                loss = F.binary_cross_entropy_with_logits(logits, yb, pos_weight=torch.tensor(positive_weight, device=device))
            scaler.scale(loss).backward(); scaler.step(optimizer); scaler.update()
            point_loss += float(loss.detach()); point_batches += 1
        pair_order = rng.permutation(len(pair_positive)); pair_loss = 0.0; pair_batches = 0
        for start in range(0, len(pair_order), batch_size):
            take = pair_order[start:start + batch_size]
            xp = _tensor(x, pair_positive[take], mean, scale, device)
            xn = _tensor(x, pair_negative[take], mean, scale, device)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", dtype=torch.float16):
                loss = 0.5 * F.softplus(-(model(xp) - model(xn))).mean()
            scaler.scale(loss).backward(); scaler.step(optimizer); scaler.update()
            pair_loss += float(loss.detach()); pair_batches += 1
        tune_logits = predict_logits(model, x, tune_indexes, mean, scale, device)
        tune_score = 1 / (1 + np.exp(-np.clip(tune_logits, -30, 30)))
        tune_ap = float(average_precision_score(y[tune_indexes], tune_score))
        history.append({"epoch": epoch + 1, "tune_ap": tune_ap,
                        "point_loss": point_loss / max(point_batches, 1),
                        "pair_loss": pair_loss / max(pair_batches, 1)})
        if tune_ap > best_ap + 1e-5:
            best_ap = tune_ap; best_state = copy.deepcopy(model.state_dict()); stale = 0
        else:
            stale += 1
            if stale >= 2:
                break
    model.load_state_dict(best_state)
    device_record = {"device": str(device), "gpu": torch.cuda.get_device_name(device),
                     "cuda": torch.version.cuda, "torch": torch.__version__,
                     "peak_gpu_memory_bytes": int(torch.cuda.max_memory_allocated(device))}
    return model, mean, scale, history, device_record, len(pair_positive)


def source_ranking_records(cutoff, source, label, score, k=5):
    frame = pd.DataFrame({"cutoff": cutoff, "src": source, "label": label, "score": score})
    rows = []
    for (time, src), group in frame.groupby(["cutoff", "src"], sort=False):
        relevance = group.sort_values("score", ascending=False, kind="stable").label.to_numpy(np.uint8)
        positives = int(relevance.sum())
        if positives == 0 or len(relevance) < 2:
            continue
        top = relevance[:k]; ranks = np.arange(1, len(top) + 1)
        discounts = 1 / np.log2(ranks + 1)
        dcg = float((top * discounts).sum()); ideal = float(discounts[:min(positives, k)].sum())
        precision = np.cumsum(top) / ranks
        average_precision = float((precision * top).sum() / min(positives, k))
        rows.append({"cutoff": int(time), "src": int(src), "recall_at_1": float(top[:1].sum() / positives),
                     "recall_at_5": float(top.sum() / positives), "ndcg_at_5": dcg / ideal,
                     "map_at_5": average_precision, "candidates": len(relevance), "positives": positives})
    return pd.DataFrame(rows)


def source_ranking_metrics(cutoff, source, label, score, k=5):
    records = source_ranking_records(cutoff, source, label, score, k)
    if records.empty:
        raise ValueError("no source-cutoff group eligible for conditional ranking")
    return {name: float(records[name].mean()) for name in ("recall_at_1", "recall_at_5", "ndcg_at_5", "map_at_5")} | {
        "eligible_groups": len(records), "median_candidates": float(records.candidates.median())}
