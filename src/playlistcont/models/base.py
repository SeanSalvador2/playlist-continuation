"""Common recommender interface + a shared playlist-track index.

Every model implements::

    fit(dataset)                          -> self
    recommend(seed_tracks, title, k=500)  -> list[track_uri]

``recommend`` must never return any seed track and must return at most ``k``
track_uris ordered best-first.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import numpy as np
import scipy.sparse as sp

from ..data.schema import Dataset


class TrackIndex:
    """Maps track_uri <-> integer id and builds the playlist x track matrix."""

    def __init__(self, dataset: Dataset):
        self.dataset = dataset
        self.track_uris: List[str] = sorted(dataset.tracks.keys())
        self.uri_to_id: Dict[str, int] = {u: i for i, u in enumerate(self.track_uris)}
        self.n_tracks = len(self.track_uris)
        self.n_playlists = dataset.n_playlists
        # binary playlist (row) x track (col) CSR matrix
        rows, cols = [], []
        for r, pl in enumerate(dataset.playlists):
            for u in pl.tracks:
                cid = self.uri_to_id.get(u)
                if cid is not None:
                    rows.append(r)
                    cols.append(cid)
        data = np.ones(len(rows), dtype=np.float32)
        self.pt = sp.csr_matrix(
            (data, (rows, cols)), shape=(self.n_playlists, self.n_tracks)
        )
        self.track_pop = np.asarray(self.pt.sum(axis=0)).ravel()

    def ids(self, uris: Sequence[str]) -> List[int]:
        return [self.uri_to_id[u] for u in uris if u in self.uri_to_id]

    def uris(self, ids: Sequence[int]) -> List[str]:
        return [self.track_uris[i] for i in ids]


def top_k_excluding(
    scores: np.ndarray, exclude_ids: Sequence[int], k: int
) -> List[int]:
    """Return ids of the k highest scores, skipping ``exclude_ids``."""
    if len(scores) == 0:
        return []
    ex = set(exclude_ids)
    # take a few extra then filter, to cheaply skip excluded ids
    n_take = min(len(scores), k + len(ex) + 1)
    cand = np.argpartition(-scores, n_take - 1)[:n_take]
    cand = cand[np.argsort(-scores[cand])]
    out = [int(i) for i in cand if i not in ex and scores[i] > -np.inf]
    return out[:k]


class Recommender:
    name = "base"

    def fit(self, dataset: Dataset) -> "Recommender":  # pragma: no cover - abstract
        raise NotImplementedError

    def recommend(
        self, seed_tracks: Sequence[str], title: Optional[str] = None, k: int = 500
    ) -> List[str]:  # pragma: no cover - abstract
        raise NotImplementedError
