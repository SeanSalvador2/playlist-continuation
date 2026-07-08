"""Item-item collaborative filtering over track co-occurrence.

Builds a sparse track-track similarity from the playlist x track matrix.  Two
normalizations are offered:

* ``cosine`` — cosine similarity of the track columns (co-occurrence divided by
  the geometric mean of the two tracks' popularities);
* ``pmi``   — positive pointwise mutual information, which down-weights hits
  that co-occur with everything.

To recommend, we sum the similarity columns of the seed tracks and take the
top-k.  The similarity is truncated to the ``topk_sim`` nearest neighbours per
track to keep it sparse and fast.  Empty-seed calls fall back to popularity.
"""
from __future__ import annotations

from typing import List, Optional, Sequence

import numpy as np
import scipy.sparse as sp

from ..data.schema import Dataset
from .base import Recommender, TrackIndex, top_k_excluding


class ItemCFRecommender(Recommender):
    name = "item_cf"

    def __init__(self, normalization: str = "cosine", topk_sim: int = 200,
                 shrink: float = 10.0):
        assert normalization in ("cosine", "pmi", "raw")
        self.normalization = normalization
        self.topk_sim = topk_sim
        self.shrink = shrink

    def fit(self, dataset: Dataset) -> "ItemCFRecommender":
        self.index = TrackIndex(dataset)
        pt = self.index.pt.tocsc()
        co = (pt.T @ pt).tocsr()  # track x track co-occurrence (incl diagonal)
        pop = self.index.track_pop.astype(np.float32)

        co = co.tocoo()
        r, c, v = co.row, co.col, co.data.astype(np.float32)
        mask = r != c
        r, c, v = r[mask], c[mask], v[mask]

        if self.normalization == "raw":
            # unnormalized co-occurrence counts (baseline: popularity-biased)
            s = v
        elif self.normalization == "cosine":
            denom = np.sqrt(pop[r] * pop[c]) + self.shrink
            s = v / denom
        else:  # pmi
            N = float(self.index.n_playlists)
            # p(i,j)/(p(i)p(j)) ; positive PMI
            pmi = np.log((v * N) / (pop[r] * pop[c] + 1e-9) + 1e-9)
            s = np.maximum(pmi, 0.0)

        sim = sp.csr_matrix((s, (r, c)), shape=co.shape)
        self.sim = self._truncate_rows(sim, self.topk_sim)
        self._pop_rank = np.argsort(-pop)
        return self

    @staticmethod
    def _truncate_rows(mat: sp.csr_matrix, k: int) -> sp.csr_matrix:
        mat = mat.tocsr()
        data, indices, indptr = [], [], [0]
        for i in range(mat.shape[0]):
            s, e = mat.indptr[i], mat.indptr[i + 1]
            row_data = mat.data[s:e]
            row_idx = mat.indices[s:e]
            if len(row_data) > k:
                keep = np.argpartition(-row_data, k - 1)[:k]
                row_data = row_data[keep]
                row_idx = row_idx[keep]
            data.append(row_data)
            indices.append(row_idx)
            indptr.append(indptr[-1] + len(row_data))
        return sp.csr_matrix(
            (np.concatenate(data) if data else np.zeros(0),
             np.concatenate(indices) if indices else np.zeros(0, int),
             np.array(indptr)),
            shape=mat.shape,
        )

    def score(self, seed_ids: Sequence[int]) -> np.ndarray:
        scores = np.zeros(self.index.n_tracks, dtype=np.float32)
        for sid in seed_ids:
            row = self.sim.getrow(sid)
            scores[row.indices] += row.data
        return scores

    def recommend(
        self, seed_tracks: Sequence[str], title: Optional[str] = None, k: int = 500
    ) -> List[str]:
        seed_ids = self.index.ids(seed_tracks)
        if not seed_ids:
            out = [int(i) for i in self._pop_rank[:k]]
            return self.index.uris(out)
        scores = self.score(seed_ids)
        ids = top_k_excluding(scores, seed_ids, k)
        # backfill with popularity if the neighbourhood is too small
        if len(ids) < k:
            have = set(ids) | set(seed_ids)
            for i in self._pop_rank:
                if int(i) not in have:
                    ids.append(int(i))
                    if len(ids) >= k:
                        break
        return self.index.uris(ids[:k])
