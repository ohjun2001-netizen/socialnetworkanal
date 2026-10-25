"""Leakage-safe GPU GAE baseline at validation and test boundaries."""

from __future__ import annotations

from dataclasses import dataclass
import random

import numpy as np
import pandas as pd
import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.utils import negative_sampling

from .exploration import pair_codes
from .features import decode_codes


@dataclass
class GAEOutput:
    scores: np.ndarray
    losses: list[float]
    device: str
    positive_edges: int


class DirectedGAE(nn.Module):
    def __init__(self, input_dim: int = 4, hidden_dim: int = 64, output_dim: int = 32):
        super().__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, output_dim)
        self.decoder = nn.Bilinear(output_dim, output_dim, 1)

    def encode(self, features: Tensor, edges: Tensor) -> Tensor:
        return self.conv2(F.relu(self.conv1(features, edges)), edges)

    def decode(self, embedding: Tensor, pairs: Tensor) -> Tensor:
        return self.decoder(embedding[pairs[0]], embedding[pairs[1]]).flatten()


def _seed(seed: int) -> None:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _snapshot(events: pd.DataFrame, cutoff: int, node_count: int) -> tuple[np.ndarray, np.ndarray]:
    past = events.loc[events.time < cutoff, ["src", "dst"]]
    codes = np.unique(pair_codes(past))
    source, target = decode_codes(codes)
    sent = np.bincount(past.src.to_numpy(np.int64), minlength=node_count)
    received = np.bincount(past.dst.to_numpy(np.int64), minlength=node_count)
    out_degree = np.bincount(source, minlength=node_count)
    in_degree = np.bincount(target, minlength=node_count)
    features = np.column_stack([
        np.log1p(out_degree), np.log1p(in_degree), np.log1p(sent), np.log1p(received),
    ]).astype(np.float32)
    active = (sent + received) > 0
    features[active] = (features[active] - features[active].mean(0)) / np.maximum(features[active].std(0), 1e-6)
    return features, np.vstack([source, target]).astype(np.int64)


def train_and_score(
    events: pd.DataFrame,
    cutoff: int,
    candidates: np.ndarray,
    node_count: int,
    *, epochs: int = 25,
    max_train_edges: int = 150_000,
    seed: int = 2026,
) -> GAEOutput:
    _seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    feature_array, all_edges_array = _snapshot(events, cutoff, node_count)
    rng = np.random.default_rng(seed)
    train_edges_array = all_edges_array
    if train_edges_array.shape[1] > max_train_edges:
        chosen = rng.choice(train_edges_array.shape[1], max_train_edges, replace=False)
        train_edges_array = train_edges_array[:, chosen]
    features = torch.from_numpy(feature_array).to(device)
    all_edges = torch.from_numpy(all_edges_array).long().to(device)
    positive = torch.from_numpy(train_edges_array).long().to(device)
    message_edges = torch.cat([all_edges, all_edges.flip(0)], dim=1)
    model = DirectedGAE().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=1e-5)
    losses = []
    model.train()
    for _ in range(epochs):
        optimizer.zero_grad(set_to_none=True)
        embedding = model.encode(features, message_edges)
        negative = negative_sampling(
            all_edges, num_nodes=node_count, num_neg_samples=positive.shape[1], method="sparse"
        )
        positive_logits = model.decode(embedding, positive)
        negative_logits = model.decode(embedding, negative)
        logits = torch.cat([positive_logits, negative_logits])
        labels = torch.cat([torch.ones_like(positive_logits), torch.zeros_like(negative_logits)])
        loss = F.binary_cross_entropy_with_logits(logits, labels)
        loss.backward(); optimizer.step()
        losses.append(float(loss.detach().cpu()))
    source, target = decode_codes(candidates)
    pairs = torch.from_numpy(np.vstack([source, target])).long().to(device)
    model.eval()
    with torch.no_grad():
        embedding = model.encode(features, message_edges)
        scores = torch.sigmoid(model.decode(embedding, pairs)).cpu().numpy()
    return GAEOutput(scores.astype(np.float64), losses, str(device), int(all_edges_array.shape[1]))

