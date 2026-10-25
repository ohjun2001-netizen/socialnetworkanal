"""Pinned, typed SNAP mirrors with validated sizes, counts, and provenance."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import urllib.request

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
COMMIT = "a1023fc6c54c70cd92d2152c8dc091829f7e2451"
BASE = f"https://raw.githubusercontent.com/4AlexMin/dynamic-networks/{COMMIT}/human/Interactive%20Digital%20Communities"
KINDS = ("a2q", "c2a", "c2q")
EXPECTED = {"mathoverflow": (107581, 195330, 203639), "askubuntu": (280102, 356822, 327513)}


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def fetch_one(dataset: str, kind: str) -> dict:
    folder = ROOT / "data" / "typed" / dataset
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"sx-{dataset}-{kind}.txt"
    url = f"{BASE}/{path.name}"
    if not path.exists():
        temporary = path.with_suffix(".part")
        with urllib.request.urlopen(url, timeout=45) as response, temporary.open("wb") as stream:
            for chunk in iter(lambda: response.read(1024 * 1024), b""):
                stream.write(chunk)
        frame = pd.read_csv(temporary, sep=r"\s+", header=None, dtype=np.int64)
        expected = EXPECTED[dataset][KINDS.index(kind)]
        if frame.shape != (expected, 3):
            raise ValueError(f"Bad download {dataset}/{kind}: {frame.shape}, expected {expected}x3")
        temporary.replace(path)
    frame = pd.read_csv(path, sep=r"\s+", header=None, dtype=np.int64)
    if frame.shape != (EXPECTED[dataset][KINDS.index(kind)], 3):
        raise ValueError(f"Unexpected local data shape: {path}")
    return {"kind": kind, "rows": len(frame), "url": url, "sha256": digest(path), "bytes": path.stat().st_size}


def load(dataset: str) -> tuple[pd.DataFrame, dict]:
    if dataset not in EXPECTED:
        raise ValueError(dataset)
    with ThreadPoolExecutor(max_workers=3) as executor:
        manifest = list(executor.map(lambda kind: fetch_one(dataset, kind), KINDS))
    parts = []
    for kind in KINDS:
        frame = pd.read_csv(ROOT / "data" / "typed" / dataset / f"sx-{dataset}-{kind}.txt",
                            sep=r"\s+", names=["src", "dst", "time"], dtype=np.int64)
        frame["kind"] = kind
        parts.append(frame)
    raw = pd.concat(parts, ignore_index=True)
    original = ROOT / "data" / "raw" / "sx-mathoverflow.txt"
    multiset_match = None
    if dataset == "mathoverflow" and original.exists():
        previous = pd.read_csv(original, sep=r"\s+", names=["src", "dst", "time"], dtype=np.int64)
        a = raw[["src", "dst", "time"]].sort_values(["time", "src", "dst"]).to_numpy()
        b = previous.sort_values(["time", "src", "dst"]).to_numpy()
        multiset_match = bool(np.array_equal(a, b))
        if not multiset_match:
            raise ValueError("Typed source union differs from previously used MathOverflow events")
    self_events = int((raw.src == raw.dst).sum())
    events = raw.loc[raw.src != raw.dst].copy()
    users = np.unique(events[["src", "dst"]].to_numpy())
    events["src"] = np.searchsorted(users, events.src)
    events["dst"] = np.searchsorted(users, events.dst)
    events = events.sort_values(["time", "src", "dst", "kind"], kind="stable").reset_index(drop=True)
    info = {
        "dataset": dataset, "source": f"https://snap.stanford.edu/data/sx-{dataset}.html",
        "download_source": "pinned public mirror; original SNAP host timed out",
        "mirror_commit": COMMIT, "files": manifest, "raw_events": len(raw),
        "self_events_removed": self_events, "used_events": len(events), "users": len(users),
        "first_time": int(events.time.min()), "last_time": int(events.time.max()),
        "type_counts_used": {k: int(v) for k, v in events.kind.value_counts().items()},
        "identical_multiset_to_v1": multiset_match,
        "duplicate_event_rows_retained": int(raw.duplicated(["src", "dst", "time", "kind"]).sum()),
        "node_ids": "remapped identifiers only; no future-derived node features",
    }
    output = ROOT / "results" / "research_v2" / dataset
    output.mkdir(parents=True, exist_ok=True)
    (output / "data_manifest.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
    return events, info


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=list(EXPECTED), default="mathoverflow")
    args = parser.parse_args()
    _, info = load(args.dataset)
    print(json.dumps(info, indent=2))
