"""Scenario-aware routing on top of the hybrid.

The hybrid reranker is trained across all scenarios, so it slightly *underfits*
the two extremes:

* **title-only** (0 seed tracks) -- the title specialist is far stronger, but
  the hybrid dilutes it with popularity/CF candidates that have no seed signal;
* very long seeds -- pure co-occurrence is already excellent.

Rather than retrain, we *route*: blend the hybrid's ranking with the title
model's ranking using a weight that depends on how much seed vs. title signal
the query actually has.  With zero seeds and a real title we lean fully on the
title model; as the seed count grows we hand control back to the hybrid.  The
blend is a weighted reciprocal-rank fusion of the two ordered lists.

``route_weight`` (the title weight as a function of the query shape) is a pure
function so the policy is easy to unit-test and tune.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence

from ..data.schema import Dataset
from .base import Recommender

# title weight alpha by seed-count bucket (only applied when a title is present)
DEFAULT_ROUTING: Dict[int, float] = {0: 1.0, 1: 0.6, 5: 0.35, 10: 0.0}


def route_weight(
    n_seeds: int,
    has_title: bool,
    table: Optional[Dict[int, float]] = None,
) -> float:
    """Title-model blend weight (alpha in [0, 1]) for a query.

    Zero when there is no usable title (the title model would fall back to
    popularity, so it can only hurt).  Otherwise a step function of the seed
    count: the largest table threshold <= ``n_seeds`` wins.
    """
    if not has_title:
        return 0.0
    tbl = table or DEFAULT_ROUTING
    alpha = tbl[min(tbl)]
    for thresh in sorted(tbl):
        if n_seeds >= thresh:
            alpha = tbl[thresh]
    return float(alpha)


def _rrf(order: Sequence[str], kconst: float = 60.0) -> Dict[str, float]:
    """Reciprocal-rank score for an ordered list: 1 / (k + rank)."""
    return {uri: 1.0 / (kconst + rank) for rank, uri in enumerate(order)}


class RoutedHybrid(Recommender):
    """Blend a hybrid recommender with a title recommender, query-adaptively."""

    name = "routed_hybrid"

    def __init__(
        self,
        hybrid: Recommender,
        title: Recommender,
        routing: Optional[Dict[int, float]] = None,
        rrf_k: float = 60.0,
    ):
        self.hybrid = hybrid
        self.title = title
        self.routing = routing or dict(DEFAULT_ROUTING)
        self.rrf_k = rrf_k

    def fit(self, dataset: Dataset) -> "RoutedHybrid":
        # both sub-models are expected to be already fit (shared instances);
        # fit is idempotent-friendly and only wires them if not.
        if not hasattr(self.hybrid, "index"):
            self.hybrid.fit(dataset)
        if not hasattr(self.title, "index"):
            self.title.fit(dataset)
        return self

    def recommend(
        self, seed_tracks: Sequence[str], title: Optional[str] = None, k: int = 500
    ) -> List[str]:
        has_title = bool(title and title.strip())
        alpha = route_weight(len(seed_tracks), has_title, self.routing)
        hy = self.hybrid.recommend(seed_tracks, title, k=k)
        if alpha <= 0.0:
            return hy
        ti = self.title.recommend(seed_tracks, title, k=k)
        if alpha >= 1.0:
            return ti
        hs = _rrf(hy, self.rrf_k)
        ts = _rrf(ti, self.rrf_k)
        keys = set(hs) | set(ts)
        fused = {
            u: (1.0 - alpha) * hs.get(u, 0.0) + alpha * ts.get(u, 0.0)
            for u in keys
        }
        order = sorted(fused, key=lambda u: -fused[u])
        return order[:k]
