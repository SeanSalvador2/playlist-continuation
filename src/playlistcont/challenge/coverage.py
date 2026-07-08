"""Beyond-accuracy metrics: catalog coverage, diversity, and popularity bias.

Accuracy metrics (R-precision / NDCG / clicks) reward hitting the held-out
tracks but say nothing about *which* tracks a model likes to recommend.  A
model can score well while only ever surfacing the same few hits.  These
functions quantify that:

* **catalog_coverage** -- fraction of the whole track catalog that a model ever
  recommends across a batch of queries (higher = explores more of the tail).
* **intra_list_similarity** -- mean pairwise cosine similarity of the top-k
  recommended tracks in interpretable-feature space (lower = more diverse).
* **popularity_bucket** / **bucketize_catalog** -- split the catalog into
  head / torso / tail by play count so accuracy can be reported per bucket.

All are pure functions so they are cheap to unit-test.
"""
from __future__ import annotations

from typing import Dict, List, Mapping, Sequence

import numpy as np


def catalog_coverage(
    recommendations: Sequence[Sequence[str]], n_catalog: int, k: int = 500
) -> float:
    """Fraction of the catalog appearing in the top-k of any recommendation."""
    if n_catalog <= 0:
        return 0.0
    seen: set[str] = set()
    for rec in recommendations:
        seen.update(rec[:k])
    return len(seen) / n_catalog


def intra_list_similarity(
    rec: Sequence[str],
    features: Mapping[str, np.ndarray],
    k: int = 100,
) -> float:
    """Mean pairwise cosine similarity of the top-k recommended tracks.

    Higher means the list is more homogeneous (less diverse).  Tracks without a
    feature vector are skipped.  Returns 0.0 for lists shorter than 2.
    """
    vecs = [features[u] for u in rec[:k] if u in features]
    if len(vecs) < 2:
        return 0.0
    M = np.asarray(vecs, dtype=np.float64)
    norms = np.linalg.norm(M, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    M = M / norms
    sims = M @ M.T
    n = M.shape[0]
    # mean of the strict upper triangle (exclude the diagonal of ones)
    total = (sims.sum() - np.trace(sims)) / 2.0
    n_pairs = n * (n - 1) / 2.0
    return float(total / n_pairs) if n_pairs > 0 else 0.0


def bucketize_catalog(
    track_pop: Mapping[str, float] | np.ndarray,
    track_uris: Sequence[str] | None = None,
    head_frac: float = 0.02,
    tail_frac: float = 0.8,
) -> Dict[str, str]:
    """Label every track head / torso / tail by descending play count.

    ``head_frac`` most-played tracks are the head; the least-played
    ``tail_frac`` are the tail; the middle is the torso.  Accepts either a
    mapping ``uri -> pop`` or a numpy array aligned to ``track_uris``.
    """
    if isinstance(track_pop, np.ndarray):
        assert track_uris is not None
        pairs = list(zip(track_uris, track_pop.tolist()))
    else:
        pairs = list(track_pop.items())
    pairs.sort(key=lambda kv: -kv[1])
    n = len(pairs)
    n_head = max(1, int(round(n * head_frac)))
    n_tail = max(1, int(round(n * tail_frac)))
    labels: Dict[str, str] = {}
    for rank, (uri, _) in enumerate(pairs):
        if rank < n_head:
            labels[uri] = "head"
        elif rank >= n - n_tail:
            labels[uri] = "tail"
        else:
            labels[uri] = "torso"
    return labels


def bucket_hit_rate(
    predicted: Sequence[str],
    ground_truth: Sequence[str],
    buckets: Mapping[str, str],
    k: int = 500,
) -> Dict[str, Dict[str, int]]:
    """Per-bucket recall support: {bucket: {hits, demand}} for one query.

    ``demand`` counts ground-truth tracks in each popularity bucket; ``hits``
    counts how many of those the top-k predictions recovered.  Summing these
    across queries yields per-bucket recall = hits / demand.
    """
    gt = set(ground_truth)
    top = set(predicted[:k])
    out: Dict[str, Dict[str, int]] = {
        b: {"hits": 0, "demand": 0} for b in ("head", "torso", "tail")
    }
    for t in gt:
        b = buckets.get(t)
        if b is None:
            continue
        out[b]["demand"] += 1
        if t in top:
            out[b]["hits"] += 1
    return out
