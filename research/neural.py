"""Task-supervised TGN-style memory and rolling GraphSAGE baselines.

Adapted architecture, not a reproduction of a published benchmark: directed
last-message GRU memory, temporal neighbor attention, and a 24-hour recurrence
head. Batches of past events update simultaneously (no within-batch ordering).
All events in each update are strictly before the prediction cutoff.
"""
from __future__ import annotations
import copy
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F
from torch_geometric.nn import TransformerConv, SAGEConv
from torch_geometric.nn.models.tgn import LastNeighborLoader
from sklearn.metrics import average_precision_score

from .data import KINDS


class RecurrenceNetwork(nn.Module):
    def __init__(self, nodes, mode="tgn", dim=32):
        super().__init__()
        self.mode, self.dim = mode, dim
        self.time_mlp = nn.Sequential(nn.Linear(1, 8), nn.Tanh())
        self.message = nn.Sequential(nn.Linear(2*dim+3+1+8, dim), nn.ReLU())
        self.gru = nn.GRUCell(dim, dim)
        self.attention = TransformerConv(dim, dim//2, heads=2, edge_dim=3+1+8, dropout=.1)
        self.sage1 = SAGEConv(4, dim)
        self.sage2 = SAGEConv(dim, dim)
        self.head = nn.Sequential(nn.Linear(4*dim+2, 64), nn.ReLU(), nn.Dropout(.1), nn.Linear(64, 1))
        self.register_buffer("memory", torch.zeros(nodes, dim), persistent=False)
        self.register_buffer("last_update", torch.zeros(nodes), persistent=False)
        self.register_buffer("counts", torch.zeros(nodes, 4), persistent=False)
        self.register_buffer("assoc", torch.empty(nodes, dtype=torch.long), persistent=False)
        self.neighbors = None

    def reset(self):
        self.memory = torch.zeros_like(self.memory)
        self.last_update.zero_(); self.counts.zero_()
        self.neighbors = LastNeighborLoader(len(self.memory), size=10, device=self.memory.device)

    def ingest(self, src, dst, times, types, gradients=False):
        if not len(src):
            return
        targets = torch.cat([src, dst]); other = torch.cat([dst, src])
        et = torch.cat([times, times]); ek = torch.cat([types, types])
        direction = torch.cat([torch.ones_like(times), -torch.ones_like(times)])
        # Latest event per node in this batch. Stable and deterministic tie rule.
        nodes, inverse = targets.unique(return_inverse=True)
        positions = torch.arange(len(src), device=src.device).repeat(2)
        latest = torch.full((len(nodes),), -1, dtype=torch.long, device=src.device)
        latest.scatter_reduce_(0, inverse, positions, reduce="amax", include_self=True)
        chosen = positions == latest[inverse]
        ti, oi, tt, kk, sign = targets[chosen], other[chosen], et[chosen], ek[chosen], direction[chosen]
        if self.mode == "tgn":
            with torch.set_grad_enabled(gradients):
                elapsed = torch.log1p((tt-self.last_update[ti]).clamp(min=0)/3600).unsqueeze(1)
                message = torch.cat([self.memory[ti], self.memory[oi], F.one_hot(kk, 3).float(),
                                     sign.unsqueeze(1), self.time_mlp(elapsed)], 1)
                updated = self.gru(self.message(message), self.memory[ti])
                if gradients:
                    self.memory = self.memory.detach().index_copy(0, ti, updated)
                else:
                    self.memory[ti] = updated
        self.last_update[ti] = tt
        with torch.no_grad():
            self.counts[:, 0].index_add_(0, src, torch.ones_like(times))
            self.counts[:, 1].index_add_(0, dst, torch.ones_like(times))
            # Per-type comment vs answer totals are past-only node descriptors.
            self.counts[:, 2].index_add_(0, targets, (ek == 0).float())
            self.counts[:, 3].index_add_(0, targets, (ek != 0).float())
        self.neighbors.insert(src, dst)

    def predict(self, src, dst, cutoff, event_src, event_t, event_kind):
        nodes, edge_index, event_ids = self.neighbors(torch.cat([src, dst]).unique())
        self.assoc[nodes] = torch.arange(len(nodes), device=src.device)
        if self.mode == "tgn":
            age = torch.log1p((cutoff-event_t[event_ids]).clamp(min=0)/3600).unsqueeze(1)
            # Edge neighbor -> queried node; preserve original action direction in attrs.
            sign = torch.where(event_src[event_ids] == nodes[edge_index[0]], 1., -1.).unsqueeze(1)
            attrs = torch.cat([F.one_hot(event_kind[event_ids], 3).float(), sign, self.time_mlp(age)], 1)
            z = self.attention(self.memory[nodes], edge_index, attrs)
        else:
            z = F.relu(self.sage1(torch.log1p(self.counts[nodes]), edge_index))
            z = self.sage2(z, edge_index)
        a, b = z[self.assoc[src]], z[self.assoc[dst]]
        age_a = torch.log1p((cutoff-self.last_update[src]).clamp(min=0)/3600).unsqueeze(1)
        age_b = torch.log1p((cutoff-self.last_update[dst]).clamp(min=0)/3600).unsqueeze(1)
        representation = torch.cat([a, b, a*b, torch.abs(a-b), age_a, age_b], 1)
        return self.head(representation).flatten(), representation


def run_neural(events, meta, masks, mode, seed, folder: Path, epochs=4):
    torch.manual_seed(seed); np.random.seed(seed); torch.set_num_threads(4)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    origin = int(events.time.min())
    times = events.time.to_numpy(np.int64)
    event_src = torch.tensor(events.src.to_numpy(), dtype=torch.long, device=device)
    event_dst = torch.tensor(events.dst.to_numpy(), dtype=torch.long, device=device)
    event_t = torch.tensor(times-origin, dtype=torch.float32, device=device)
    event_kind = torch.tensor(pd.Categorical(events.kind, categories=KINDS).codes.astype(np.int64), device=device)
    src = torch.tensor(meta.src.to_numpy(), dtype=torch.long, device=device)
    dst = torch.tensor(meta.dst.to_numpy(), dtype=torch.long, device=device)
    labels = torch.tensor(meta.label.to_numpy(), dtype=torch.float32, device=device)
    groups = [(int(t), group.index.to_numpy()) for t, group in meta.groupby("cutoff", sort=True)]
    model = RecurrenceNetwork(int(max(events.src.max(), events.dst.max()))+1, mode).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=1e-4)
    history = []; best = None; start_time = time.perf_counter()

    def stream(selected_mask, train=False, get_embeddings=False):
        model.train(train); model.reset()
        last_cutoff = int(meta.loc[selected_mask, "cutoff"].max())
        scores = np.full(len(meta), np.nan)
        embeddings = np.zeros((len(meta), 4*model.dim+2), np.float32) if get_embeddings else None
        pointer = 0; losses = []; train_rows = 0
        for cutoff, indices in groups:
            if cutoff > last_cutoff:
                break
            end = int(np.searchsorted(times, cutoff, side="left"))
            is_selected = bool(selected_mask[indices[0]])
            # All event batches stop before cutoff. The final past batch carries
            # a truncated gradient to memory; earlier batches are detached.
            while pointer < end:
                stop = min(pointer+128, end)
                model.ingest(event_src[pointer:stop], event_dst[pointer:stop], event_t[pointer:stop],
                             event_kind[pointer:stop], gradients=train and is_selected and stop == end)
                pointer = stop
            if not is_selected:
                model.memory = model.memory.detach()
                continue
            ids = torch.tensor(indices, device=device)
            with torch.set_grad_enabled(train):
                logits, embedding = model.predict(src[ids], dst[ids], float(cutoff-origin), event_src, event_t, event_kind)
                if train:
                    loss = F.binary_cross_entropy_with_logits(logits, labels[ids])
                    optimizer.zero_grad(set_to_none=True); loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
                    optimizer.step()
                    losses.append(float(loss.detach())*len(ids)); train_rows += len(ids)
                else:
                    scores[indices] = torch.sigmoid(logits).cpu().numpy()
                    if embeddings is not None:
                        embeddings[indices] = embedding.cpu().numpy()
            model.memory = model.memory.detach()
        return scores, sum(losses)/max(train_rows, 1), embeddings

    for epoch in range(1, epochs+1):
        _, loss, _ = stream(masks["train"], train=True)
        score, _, _ = stream(masks["tune"])
        ap = average_precision_score(meta.label.to_numpy()[masks["tune"]], score[masks["tune"]])
        history.append({"epoch": epoch, "loss": loss, "tune_ap": ap})
        print(f"{mode} seed={seed} epoch={epoch} loss={loss:.4f} tune_AP={ap:.4f}", flush=True)
        if best is None or ap > best[0]:
            best = (ap, copy.deepcopy(model.state_dict()), epoch)
    model.load_state_dict(best[1])
    all_mask = masks["tune"] | masks["cal"] | masks["test"]
    score, _, _ = stream(all_mask)
    folder.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": best[1], "mode": mode, "seed": seed, "best_epoch": best[2],
                "architecture": "task-supervised last-message memory + temporal neighbor attention" if mode == "tgn" else "rolling last-10-neighbor GraphSAGE"},
               folder / f"{mode}_{seed}.pt")
    pd.DataFrame(history).to_csv(folder / f"{mode}_{seed}_training.csv", index=False)
    return score, {"device": str(device), "epoch": best[2], "tune_ap": best[0], "elapsed_seconds": time.perf_counter()-start_time}
