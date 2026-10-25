"""Past-only multirelational features with a content-addressed cache."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from mathoverflow_lab.features import FEATURE_NAMES as BASE_NAMES, TemporalIndex, build_examples, _count, _past, _decay, _group_times
from mathoverflow_lab.exploration import pair_codes
from .data import ROOT, KINDS, load

EXTRA_NAMES = [
    "gap_mean_hours", "gap_cv", "gap_last_hours", "burstiness", "active_days_30d",
    "count_acceleration", "reciprocity_24h", "src_acceleration", "dst_acceleration",
    "weekday_sin", "weekday_cos",
    *[f"type_{kind}_{name}" for kind in KINDS for name in (
        "count_4h", "count_24h", "count_168h", "reverse_24h", "decay_6h", "decay_24h",
        "last", "src_out_168h", "dst_in_168h")],
    "type_entropy_30d", "type_switches_30d", "time_ordered_twohop_30d",
]
NAMES = list(BASE_NAMES) + EXTRA_NAMES
SCHEMA = "typed_features_v2.1"


def extra_features(events: pd.DataFrame, meta: pd.DataFrame) -> np.ndarray:
    idx = TemporalIndex.from_events(events)
    typed = {kind: TemporalIndex.from_events(events[events.kind == kind]) for kind in KINDS}
    codes = pair_codes(events)
    order = np.argsort(codes, kind="stable")
    uniq, starts = np.unique(codes[order], return_index=True)
    kind_values = pd.Categorical(events.kind, categories=KINDS).codes
    pair_kinds = {int(c): kind_values[order[s:e]] for c, s, e in zip(uniq, starts, np.r_[starts[1:], len(order)])}
    times = events.time.to_numpy()
    result = np.zeros((len(meta), len(EXTRA_NAMES)), np.float32)
    for number, (cutoff, rows) in enumerate(meta.groupby("cutoff", sort=True)):
        left, right = np.searchsorted(times, [cutoff - 30 * 86400, cutoff])
        # Last observed timestamp on each directed edge; a causal two-hop proxy,
        # not an exhaustive temporal motif count and not a propagation claim.
        outgoing, incoming = {}, {}
        for s, d, t in events.iloc[left:right][["src", "dst", "time"]].itertuples(index=False, name=None):
            outgoing.setdefault(s, {})[d] = t
            incoming.setdefault(d, {})[s] = t
        for row in rows.itertuples():
            c, src, dst, t = int(row.pair_code), int(row.src), int(row.dst), int(cutoff)
            rev = (dst << 32) | src
            history = _past(idx.pair[c], t)
            recent = history[history >= t - 30 * 86400]
            gaps = np.diff(recent) / 3600
            mean, std = (float(gaps.mean()), float(gaps.std())) if len(gaps) else (0., 0.)
            freq4 = _count(history, t - 4 * 3600, t)
            freq24 = _count(history, t - 24 * 3600, t)
            reverse24 = _count(idx.pair.get(rev), t - 24 * 3600, t)
            source_out, target_in = idx.outgoing.get(src), idx.incoming.get(dst)
            weekday = (t // 86400 + 3) % 7
            values = [mean, std / (mean + 1e-6), float(gaps[-1]) if len(gaps) else 0.,
                      (std - mean) / (std + mean + 1e-6), len(np.unique(recent // 86400)),
                      freq4 / (1 + (freq24 - freq4) / 5), reverse24 / (1 + freq24),
                      _count(source_out, t-86400, t) / (1 + _count(source_out, t-7*86400, t-86400)/6),
                      _count(target_in, t-86400, t) / (1 + _count(target_in, t-7*86400, t-86400)/6),
                      np.sin(2*np.pi*weekday/7), np.cos(2*np.pi*weekday/7)]
            last_kind = pair_kinds[c][len(history)-1]
            for k, kind in enumerate(KINDS):
                ti = typed[kind]; ph = ti.pair.get(c)
                values.extend([_count(ph, t-4*3600, t), _count(ph, t-86400, t),
                               _count(ph, t-7*86400, t), _count(ti.pair.get(rev), t-86400, t),
                               _decay(ph, t, 6), _decay(ph, t, 24), float(k == last_kind),
                               _count(ti.outgoing.get(src), t-7*86400, t),
                               _count(ti.incoming.get(dst), t-7*86400, t)])
            ks = pair_kinds[c][len(history)-len(recent):len(history)]
            proportions = np.bincount(ks, minlength=3) / max(len(ks), 1)
            entropy = -float(np.sum(proportions * np.log(proportions + 1e-9)))
            a, b = outgoing.get(src, {}), incoming.get(dst, {})
            paths = sum(a[n] < b[n] and b[n] - a[n] <= 86400 for n in a.keys() & b.keys())
            values.extend([entropy, int(np.count_nonzero(np.diff(ks))), paths])
            result[row.Index] = values
        if (number + 1) % 500 == 0:
            print(f"typed features: {number+1} days", flush=True)
    return result


def prepare(dataset: str, phase: int = 0) -> tuple[pd.DataFrame, np.ndarray, pd.DataFrame, dict]:
    events, info = load(dataset)
    folder = ROOT / "data" / "cache"
    folder.mkdir(parents=True, exist_ok=True)
    fingerprint = hashlib.sha256(json.dumps({"files": info["files"], "schema": SCHEMA, "phase": phase}, sort_keys=True).encode()).hexdigest()
    path = folder / f"{dataset}_phase{phase}_{fingerprint[:12]}.npz"
    if path.exists():
        data = np.load(path, allow_pickle=False)
        if data["names"].tolist() != NAMES:
            raise ValueError("Feature cache schema mismatch")
        meta = pd.DataFrame({key: data[key] for key in ("cutoff", "src", "dst", "pair_code", "label")})
        return meta, data["x"], events, info
    if phase:
        shifted = events.copy()
        shifted["time"] -= phase * 3600
        meta, base = build_examples(shifted)
        meta["cutoff"] += phase * 3600
    else:
        meta, base = build_examples(events)
    extra = extra_features(events, meta)
    x = np.column_stack([base, extra]).astype(np.float32)
    if not np.isfinite(x).all():
        raise ValueError("Nonfinite features")
    np.savez_compressed(path, x=x, names=np.array(NAMES), **{k: meta[k].to_numpy() for k in meta})
    return meta, x, events, info


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(); parser.add_argument("--dataset", default="mathoverflow")
    args = parser.parse_args()
    m, x, _, _ = prepare(args.dataset)
    print({"dataset": args.dataset, "rows": len(m), "positives": int(m.label.sum()), "features": x.shape[1]})
