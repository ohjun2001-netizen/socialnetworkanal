import json

import numpy as np
import pandas as pd

from research.data import ROOT
from research.features import NAMES
from research_ranking.model import build_pair_indexes, source_ranking_metrics


def test_ranking_protocol_is_explicitly_posthoc_and_cuda_required():
    protocol = json.loads((ROOT / "research_ranking" / "protocol.json").read_text())
    assert protocol["created_after_confirmation"] is True
    assert protocol["gpu_requirement"].startswith("CUDA is mandatory")
    assert "NDCG@5" in protocol["success_rule"]


def test_pair_builder_never_crosses_source_cutoff_group():
    meta = pd.DataFrame({"cutoff": [100, 100, 100, 100, 100, 100],
                         "src": [1, 1, 1, 2, 2, 2], "dst": [4, 5, 6, 4, 5, 6]})
    x = np.zeros((6, len(NAMES)), np.float32)
    x[:, NAMES.index("pair_count_24h")] = [2, 1, 3, 1, 4, 2]
    x[:, NAMES.index("hours_since_pair_event")] = [1, 2, 3, 2, 1, 4]
    y = np.array([1, 0, 0, 0, 1, 0], np.uint8)
    positive, negative = build_pair_indexes(meta, x, y, np.ones(6, bool), 2026)
    assert (y[positive] == 1).all() and (y[negative] == 0).all()
    assert np.array_equal(meta.src.to_numpy()[positive], meta.src.to_numpy()[negative])
    assert np.array_equal(meta.cutoff.to_numpy()[positive], meta.cutoff.to_numpy()[negative])


def test_perfect_source_ranking_has_unit_metrics():
    value = source_ranking_metrics(
        np.array([100, 100, 100, 100, 100, 100]),
        np.array([1, 1, 1, 2, 2, 2]),
        np.array([1, 0, 0, 0, 1, 0]),
        np.array([.9, .2, .1, .2, .8, .1]))
    for key in ("recall_at_1", "recall_at_5", "ndcg_at_5", "map_at_5"):
        assert value[key] == 1.0
