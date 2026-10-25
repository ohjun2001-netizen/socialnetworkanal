"""Download and load the official SNAP MathOverflow temporal network."""

from __future__ import annotations

import hashlib
import gzip
import json
import os
from pathlib import Path
import urllib.request

import numpy as np
import pandas as pd

SOURCE_PAGE = "https://snap.stanford.edu/data/sx-mathoverflow.html"
COMBINED_MIRROR = "https://raw.githubusercontent.com/xuhuifan/CTEM/main/sx-mathoverflow.txt"
FILES = {
    "a2q": "https://snap.stanford.edu/data/sx-mathoverflow-a2q.txt.gz",
    "c2q": "https://snap.stanford.edu/data/sx-mathoverflow-c2q.txt.gz",
    "c2a": "https://snap.stanford.edu/data/sx-mathoverflow-c2a.txt.gz",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download(root: Path) -> dict[str, Path]:
    """Download the official files; accept the byte-equivalent combined mirror."""
    root.mkdir(parents=True, exist_ok=True)
    official_paths = {
        kind: root / f"sx-mathoverflow-{kind}.txt.gz" for kind in FILES
    }
    if all(path.exists() for path in official_paths.values()):
        return official_paths
    combined = root / "sx-mathoverflow.txt"
    if combined.exists():
        return {"all": combined}
    temporary = combined.with_suffix(".txt.part")
    request = urllib.request.Request(COMBINED_MIRROR, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(request, timeout=120) as response, temporary.open("wb") as stream:
        for chunk in iter(lambda: response.read(1024 * 1024), b""):
            stream.write(chunk)
    sample = pd.read_csv(temporary, sep=r"\s+", header=None, nrows=10)
    if sample.shape[1] != 3:
        raise ValueError("Downloaded MathOverflow data is not SRC DST UNIXTS format.")
    os.replace(temporary, combined)
    return {"all": combined}


def load_events(root: Path, *, remap_users: bool = True) -> tuple[pd.DataFrame, np.ndarray]:
    """Return sorted directed events and the raw-ID lookup for mapped user IDs."""
    paths = download(root)
    parts = []
    for kind, path in paths.items():
        part = pd.read_csv(
            path,
            compression="infer",
            sep=r"\s+",
            header=None,
            names=["src", "dst", "time"],
            dtype=np.int64,
        )
        part["kind"] = kind
        parts.append(part)
    frame = pd.concat(parts, ignore_index=True)
    frame = frame.loc[frame.src != frame.dst].copy()
    raw_users = np.unique(frame[["src", "dst"]].to_numpy().ravel())
    if remap_users:
        frame["src"] = np.searchsorted(raw_users, frame.src.to_numpy())
        frame["dst"] = np.searchsorted(raw_users, frame.dst.to_numpy())
    frame = frame.sort_values("time", kind="stable").reset_index(drop=True)
    return frame, raw_users


def utc(seconds: int) -> str:
    return pd.Timestamp(int(seconds), unit="s", tz="UTC").isoformat()


def dataset_summary(frame: pd.DataFrame, raw_users: np.ndarray, root: Path) -> dict:
    paths = download(root)
    raw_rows = 0
    for path in paths.values():
        opener = gzip.open if path.suffix == ".gz" else open
        with opener(path, "rt", encoding="utf-8") as stream:
            raw_rows += sum(1 for _ in stream)
    return {
        "source_page": SOURCE_PAGE,
        "local_source": "combined SNAP data mirrored by the CTEM research repository"
        if (root / "sx-mathoverflow.txt").exists() else "official SNAP interaction files",
        "raw_events": int(raw_rows),
        "events": int(len(frame)),
        "removed_self_events": int(raw_rows - len(frame)),
        "users": int(len(raw_users)),
        "directed_pairs": int(frame[["src", "dst"]].drop_duplicates().shape[0]),
        "first_utc": utc(int(frame.time.min())),
        "last_utc": utc(int(frame.time.max())),
        "days": float((frame.time.max() - frame.time.min()) / 86400),
        "event_types": {key: int(value) for key, value in frame.kind.value_counts().items()},
        "sha256": {kind: sha256(path) for kind, path in download(root).items()},
    }


if __name__ == "__main__":
    project = Path(__file__).resolve().parents[1]
    events, users = load_events(project / "data" / "raw")
    print(json.dumps(dataset_summary(events, users, project / "data" / "raw"), indent=2))

