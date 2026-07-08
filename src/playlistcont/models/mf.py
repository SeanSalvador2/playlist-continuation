"""Matrix factorization via implicit-feedback ALS.

Uses the ``implicit`` library's ``AlternatingLeastSquares`` when available and
falls back to a small self-contained ALS on scipy sparse otherwise (so the
project runs even without the optional dependency).

New (test) playlists are handled by *fold-in*: we form a pseudo user vector as
the mean of the seed tracks' item factors and score all items by dot product.
Empty seeds fall back to popularity.
"""
from __future__ import annotations

from typing import List, Optional, Sequence

import numpy as np
import scipy.sparse as sp

from ..data.schema import Dataset
from .base import Recommender, TrackIndex, top_k_excluding

try:  # optional accelerator
    from implicit.als import AlternatingLeastSquares  # type: ignore
    _HAVE_IMPLICIT = True
except Exception:  # pragma: no cover
    _HAVE_IMPLICIT = False


class _SimpleALS:
    """Minimal implicit-feedback ALS (Hu, Koren & Volinsky) on scipy sparse."""

    def __init__(self, factors=64, reg=0.05, iterations=15, alpha=40.0, seed=0):
        self.factors = factors
        self.reg = reg
        self.iterations = iterations
        self.alpha = alpha
        self.rng = np.random.default_rng(seed)

    def fit(self, pt: sp.csr_matrix):
        n_users, n_items = pt.shape
        C = (pt * self.alpha).tocsr()
        Ct = C.T.tocsr()
        f = self.factors
        self.U = 0.1 * self.rng.standard_normal((n_users, f)).astype(np.float32)
        self.V = 0.1 * self.rng.standard_normal((n_items, f)).astype(np.float32)
        eye = self.reg * np.eye(f, dtype=np.float32)
        for _ in range(self.iterations):
            self._als_step(C, self.V, self.U, eye)
            self._als_step(Ct, self.U, self.V, eye)
        return self

    @staticmethod
    def _als_step(C, Y, X, eye):
        YtY = Y.T @ Y
        for u in range(C.shape[0]):
            s, e = C.indptr[u], C.indptr[u + 1]
            idx = C.indices[s:e]
            conf = C.data[s:e]
            if len(idx) == 0:
                X[u] = 0
                continue
            Yi = Y[idx]
            # A = YtY + Yi^T (C-1) Yi + reg I ; b = Yi^T C p  (p=1)
            A = YtY + (Yi.T * conf) @ Yi + eye
            b = (Yi * (conf + 1.0)[:, None]).sum(axis=0)
            X[u] = np.linalg.solve(A, b)


class ALSRecommender(Recommender):
    name = "als"

    def __init__(self, factors=64, iterations=15, reg=0.05, seed=0, force_fallback=False):
        self.factors = factors
        self.iterations = iterations
        self.reg = reg
        self.seed = seed
        self.force_fallback = force_fallback
        self.backend = None

    def fit(self, dataset: Dataset) -> "ALSRecommender":
        self.index = TrackIndex(dataset)
        pt = self.index.pt
        if _HAVE_IMPLICIT and not self.force_fallback:
            model = AlternatingLeastSquares(
                factors=self.factors, regularization=self.reg,
                iterations=self.iterations, random_state=self.seed,
            )
            # implicit expects a user x item confidence matrix
            model.fit(pt.astype(np.float32), show_progress=False)
            self.item_factors = np.asarray(model.item_factors)[: self.index.n_tracks]
            self.backend = "implicit"
        else:
            als = _SimpleALS(
                factors=self.factors, reg=self.reg,
                iterations=self.iterations, seed=self.seed,
            ).fit(pt)
            self.item_factors = als.V
            self.backend = "numpy_als"
        self._pop_rank = np.argsort(-self.index.track_pop)
        return self

    def user_vector(self, seed_ids: Sequence[int]) -> np.ndarray:
        if not seed_ids:
            return np.zeros(self.item_factors.shape[1], dtype=np.float32)
        return self.item_factors[list(seed_ids)].mean(axis=0)

    def recommend(
        self, seed_tracks: Sequence[str], title: Optional[str] = None, k: int = 500
    ) -> List[str]:
        seed_ids = self.index.ids(seed_tracks)
        if not seed_ids:
            return self.index.uris([int(i) for i in self._pop_rank[:k]])
        u = self.user_vector(seed_ids)
        scores = self.item_factors @ u
        ids = top_k_excluding(scores, seed_ids, k)
        if len(ids) < k:
            have = set(ids) | set(seed_ids)
            for i in self._pop_rank:
                if int(i) not in have:
                    ids.append(int(i))
                    if len(ids) >= k:
                        break
        return self.index.uris(ids[:k])
