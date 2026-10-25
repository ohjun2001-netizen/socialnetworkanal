"""Pre-register a practical lookback using counts, before fitting any model."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .data import dataset_summary, load_events

DAY = 86400
ROOT = Path(__file__).resolve().parents[1]


def pair_codes(frame: pd.DataFrame) -> np.ndarray:
    src = frame.src.to_numpy(dtype=np.uint64)
    dst = frame.dst.to_numpy(dtype=np.uint64)
    return (src << np.uint64(32)) | dst


def make_cutoffs(times: np.ndarray, warmup_days: int, horizon_hours: int, stride_hours: int) -> np.ndarray:
    stride = stride_hours * 3600
    first_midnight = (int(times.min()) // DAY + 1) * DAY
    start = first_midnight + warmup_days * DAY
    stop = int(times.max()) - horizon_hours * 3600
    return np.arange(start, stop + 1, stride, dtype=np.int64)


def inspect_lookbacks(
    events: pd.DataFrame,
    lookbacks: tuple[int, ...] = (4, 12, 24),
    *,
    horizon_hours: int = 24,
    warmup_days: int = 30,
    stride_hours: int = 24,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    times = events.time.to_numpy(dtype=np.int64)
    codes = pair_codes(events)
    cutoffs = make_cutoffs(times, warmup_days, horizon_hours, stride_hours)
    summaries, windows = [], []
    for lookback in lookbacks:
        candidate_counts, positive_counts = [], []
        seen_candidates: set[int] = set()
        seen_positives: set[int] = set()
        for cutoff in cutoffs:
            left = np.searchsorted(times, cutoff - lookback * 3600, side="left")
            split = np.searchsorted(times, cutoff, side="left")
            right = np.searchsorted(times, cutoff + horizon_hours * 3600, side="left")
            candidates = np.unique(codes[left:split])
            future = np.unique(codes[split:right])
            positive = np.intersect1d(candidates, future, assume_unique=True)
            candidate_counts.append(len(candidates))
            positive_counts.append(len(positive))
            seen_candidates.update(map(int, candidates))
            seen_positives.update(map(int, positive))
            windows.append({
                "lookback_hours": lookback,
                "cutoff": int(cutoff),
                "candidates": int(len(candidates)),
                "positives": int(len(positive)),
                "positive_rate": float(len(positive) / len(candidates)) if len(candidates) else 0.0,
            })
        total_candidates = int(sum(candidate_counts))
        total_positives = int(sum(positive_counts))
        summaries.append({
            "lookback_hours": lookback,
            "horizon_hours": horizon_hours,
            "windows": int(len(cutoffs)),
            "candidate_outcomes": total_candidates,
            "positive_outcomes": total_positives,
            "positive_rate": total_positives / total_candidates if total_candidates else 0.0,
            "unique_candidate_pairs": len(seen_candidates),
            "unique_positive_pairs": len(seen_positives),
            "median_candidates_per_window": float(np.median(candidate_counts)),
            "median_positives_per_window": float(np.median(positive_counts)),
            "empty_positive_windows": int(np.sum(np.asarray(positive_counts) == 0)),
        })
    return pd.DataFrame(summaries), pd.DataFrame(windows)


def choose_lookback(summary: pd.DataFrame, minimum_positives: int = 1000) -> int:
    """Choose the shortest window with enough positives; otherwise the largest sample."""
    eligible = summary.loc[summary.positive_outcomes >= minimum_positives]
    if len(eligible):
        return int(eligible.sort_values("lookback_hours").iloc[0].lookback_hours)
    return int(summary.sort_values(["positive_outcomes", "lookback_hours"], ascending=[False, True]).iloc[0].lookback_hours)


def run() -> Path:
    result_dir = ROOT / "results"
    result_dir.mkdir(parents=True, exist_ok=True)
    raw = ROOT / "data" / "raw"
    events, users = load_events(raw)
    summary, windows = inspect_lookbacks(events)
    selected = choose_lookback(summary)
    summary.to_csv(result_dir / "lookback_comparison.csv", index=False)
    windows.to_csv(result_dir / "lookback_windows.csv", index=False)
    payload = {
        "dataset": dataset_summary(events, users, raw),
        "target": "same directed pair reconnects within the next 24 hours",
        "selection_rule": "shortest lookback with at least 1,000 positive outcomes; otherwise most positives",
        "selected_lookback_hours": selected,
        "comparison": summary.to_dict(orient="records"),
    }
    (result_dir / "exploration_summary.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(summary.to_string(index=False))
    print(f"\nSelected lookback: {selected}h")
    return result_dir / "exploration_summary.json"


def main() -> None:
    argparse.ArgumentParser(description=__doc__).parse_args()
    run()


if __name__ == "__main__":
    main()

