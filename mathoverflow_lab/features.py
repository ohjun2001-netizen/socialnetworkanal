"""Leakage-safe temporal, point-process, and SNA feature construction."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .exploration import make_cutoffs, pair_codes

HOUR = 3600
DAY = 86400

HISTORY_FEATURES = [
    *[f"pair_count_{hours}h" for hours in (1, 4, 12, 24, 72, 168, 720)],
    "pair_count_lifetime",
    *[f"reverse_count_{hours}h" for hours in (4, 24, 168, 720)],
    "reverse_count_lifetime",
    "hours_since_pair_event",
    "pair_age_days",
    "src_out_24h", "src_out_168h", "src_out_720h", "src_in_168h",
    "dst_in_24h", "dst_in_168h", "dst_in_720h", "dst_out_168h",
]
SNA_FEATURES = [
    "src_degree_30d", "dst_degree_30d", "common_neighbors_30d",
    "jaccard_30d", "adamic_adar_30d", "preferential_attachment_30d",
]
HAWKES_FEATURES = [
    *[f"pair_decay_{hours}h" for hours in (1, 6, 24, 72, 168)],
    *[f"reverse_decay_{hours}h" for hours in (6, 24, 168)],
]
FEATURE_NAMES = HISTORY_FEATURES + SNA_FEATURES + HAWKES_FEATURES


def decode_codes(codes: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(codes, dtype=np.uint64)
    return (
        (values >> np.uint64(32)).astype(np.int64),
        (values & np.uint64(0xFFFFFFFF)).astype(np.int64),
    )


def _group_times(keys: np.ndarray, times: np.ndarray) -> dict[int, np.ndarray]:
    order = np.argsort(keys, kind="stable")
    sorted_keys = keys[order]
    sorted_times = times[order]
    unique, starts = np.unique(sorted_keys, return_index=True)
    ends = np.r_[starts[1:], len(sorted_keys)]
    return {int(key): sorted_times[start:end] for key, start, end in zip(unique, starts, ends)}


def _count(values: np.ndarray | None, start: int, end: int) -> int:
    if values is None:
        return 0
    return int(np.searchsorted(values, end, side="left") - np.searchsorted(values, start, side="left"))


def _past(values: np.ndarray | None, cutoff: int) -> np.ndarray:
    if values is None:
        return np.empty(0, dtype=np.int64)
    return values[:np.searchsorted(values, cutoff, side="left")]


def _decay(values: np.ndarray | None, cutoff: int, tau_hours: int) -> float:
    past = _past(values, cutoff)
    if not len(past):
        return 0.0
    lower = cutoff - 30 * DAY
    recent = past[np.searchsorted(past, lower, side="left"):]
    return float(np.exp(-(cutoff - recent) / (tau_hours * HOUR)).sum())


@dataclass
class TemporalIndex:
    pair: dict[int, np.ndarray]
    outgoing: dict[int, np.ndarray]
    incoming: dict[int, np.ndarray]

    @classmethod
    def from_events(cls, events: pd.DataFrame) -> "TemporalIndex":
        times = events.time.to_numpy(dtype=np.int64)
        codes = pair_codes(events)
        return cls(
            pair=_group_times(codes, times),
            outgoing=_group_times(events.src.to_numpy(dtype=np.int64), times),
            incoming=_group_times(events.dst.to_numpy(dtype=np.int64), times),
        )


def _rolling_neighbors(events: pd.DataFrame, times: np.ndarray, cutoff: int) -> dict[int, set[int]]:
    left = np.searchsorted(times, cutoff - 30 * DAY, side="left")
    right = np.searchsorted(times, cutoff, side="left")
    neighbors: dict[int, set[int]] = defaultdict(set)
    for src, dst in events.iloc[left:right][["src", "dst"]].itertuples(index=False, name=None):
        neighbors[int(src)].add(int(dst))
        neighbors[int(dst)].add(int(src))
    return neighbors


def build_examples(
    events: pd.DataFrame,
    *,
    lookback_hours: int = 4,
    horizon_hours: int = 24,
    warmup_days: int = 30,
    stride_hours: int = 24,
) -> tuple[pd.DataFrame, np.ndarray]:
    """Build one row per candidate pair/cutoff using information strictly before cutoff."""
    times = events.time.to_numpy(dtype=np.int64)
    codes = pair_codes(events)
    cutoffs = make_cutoffs(times, warmup_days, horizon_hours, stride_hours)
    index = TemporalIndex.from_events(events)
    rows: list[list[float | int]] = []
    matrix: list[list[float]] = []
    count_windows = (1, 4, 12, 24, 72, 168, 720)
    reverse_windows = (4, 24, 168, 720)
    for window_number, cutoff in enumerate(cutoffs):
        left = np.searchsorted(times, cutoff - lookback_hours * HOUR, side="left")
        split = np.searchsorted(times, cutoff, side="left")
        right = np.searchsorted(times, cutoff + horizon_hours * HOUR, side="left")
        candidates = np.unique(codes[left:split])
        future = np.unique(codes[split:right])
        labels = np.isin(candidates, future, assume_unique=True).astype(np.uint8)
        source, target = decode_codes(candidates)
        neighbors = _rolling_neighbors(events, times, int(cutoff))
        for code, src, dst, label in zip(candidates, source, target, labels):
            key = int(code)
            reverse_key = (int(dst) << 32) | int(src)
            pair_history = index.pair.get(key)
            reverse_history = index.pair.get(reverse_key)
            past_pair = _past(pair_history, int(cutoff))
            history = [
                float(_count(pair_history, int(cutoff - hours * HOUR), int(cutoff)))
                for hours in count_windows
            ]
            history.append(float(len(past_pair)))
            history.extend(
                float(_count(reverse_history, int(cutoff - hours * HOUR), int(cutoff)))
                for hours in reverse_windows
            )
            history.append(float(len(_past(reverse_history, int(cutoff)))))
            history.append(float((cutoff - past_pair[-1]) / HOUR) if len(past_pair) else 1e6)
            history.append(float((cutoff - past_pair[0]) / DAY) if len(past_pair) else 0.0)
            history.extend([
                float(_count(index.outgoing.get(int(src)), int(cutoff - 24 * HOUR), int(cutoff))),
                float(_count(index.outgoing.get(int(src)), int(cutoff - 168 * HOUR), int(cutoff))),
                float(_count(index.outgoing.get(int(src)), int(cutoff - 720 * HOUR), int(cutoff))),
                float(_count(index.incoming.get(int(src)), int(cutoff - 168 * HOUR), int(cutoff))),
                float(_count(index.incoming.get(int(dst)), int(cutoff - 24 * HOUR), int(cutoff))),
                float(_count(index.incoming.get(int(dst)), int(cutoff - 168 * HOUR), int(cutoff))),
                float(_count(index.incoming.get(int(dst)), int(cutoff - 720 * HOUR), int(cutoff))),
                float(_count(index.outgoing.get(int(dst)), int(cutoff - 168 * HOUR), int(cutoff))),
            ])
            src_neighbors = neighbors.get(int(src), set())
            dst_neighbors = neighbors.get(int(dst), set())
            common = src_neighbors.intersection(dst_neighbors)
            union = src_neighbors.union(dst_neighbors)
            adamic = sum(1.0 / np.log(max(len(neighbors.get(node, ())), 2)) for node in common)
            sna = [
                float(len(src_neighbors)), float(len(dst_neighbors)), float(len(common)),
                float(len(common) / len(union)) if union else 0.0,
                float(adamic), float(len(src_neighbors) * len(dst_neighbors)),
            ]
            hawkes = [
                *[_decay(pair_history, int(cutoff), hours) for hours in (1, 6, 24, 72, 168)],
                *[_decay(reverse_history, int(cutoff), hours) for hours in (6, 24, 168)],
            ]
            matrix.append(history + sna + hawkes)
            rows.append([int(cutoff), int(src), int(dst), key, int(label)])
        if (window_number + 1) % 250 == 0 or window_number + 1 == len(cutoffs):
            print(f"features: {window_number + 1}/{len(cutoffs)} windows, {len(rows):,} examples")
    metadata = pd.DataFrame(rows, columns=["cutoff", "src", "dst", "pair_code", "label"])
    return metadata, np.asarray(matrix, dtype=np.float32)


def save_examples(path: Path, metadata: pd.DataFrame, features: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        features=features,
        cutoff=metadata.cutoff.to_numpy(np.int64),
        src=metadata.src.to_numpy(np.int32),
        dst=metadata.dst.to_numpy(np.int32),
        pair_code=metadata.pair_code.to_numpy(np.uint64),
        label=metadata.label.to_numpy(np.uint8),
        feature_names=np.asarray(FEATURE_NAMES),
    )


def load_examples(path: Path) -> tuple[pd.DataFrame, np.ndarray]:
    payload = np.load(path, allow_pickle=False)
    metadata = pd.DataFrame({key: payload[key] for key in ("cutoff", "src", "dst", "pair_code", "label")})
    names = payload["feature_names"].astype(str).tolist()
    if names != FEATURE_NAMES:
        raise ValueError("Cached feature schema does not match this code version.")
    return metadata, payload["features"]

