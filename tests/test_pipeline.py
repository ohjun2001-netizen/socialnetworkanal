from __future__ import annotations

import pandas as pd

from mathoverflow_lab.exploration import inspect_lookbacks, choose_lookback


def test_temporal_labels_do_not_look_ahead() -> None:
    events = pd.DataFrame({
        "src": [0, 0, 1, 0, 2, 3],
        "dst": [1, 1, 2, 1, 3, 4],
        "time": [0, 31 * 86400 - 3600, 31 * 86400 - 1800, 31 * 86400 + 3600, 31 * 86400 + 7200, 32 * 86400 + 3600],
        "kind": ["a2q"] * 6,
    })
    summary, windows = inspect_lookbacks(
        events, (4,), horizon_hours=24, warmup_days=30, stride_hours=24
    )
    first = windows.iloc[0]
    assert first.candidates == 2
    assert first.positives == 1
    assert summary.iloc[0].positive_outcomes == 1


def test_lookback_selection_uses_counts_not_model_metrics() -> None:
    frame = pd.DataFrame({
        "lookback_hours": [4, 12, 24],
        "positive_outcomes": [999, 1000, 5000],
    })
    assert choose_lookback(frame) == 12
