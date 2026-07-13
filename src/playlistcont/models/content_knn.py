"""Pure content-based recommendation by feature-vector similarity.

This is the honest, textbook *content-based* baseline: it ignores who listened
to what entirely (no popularity, no co-occurrence) and recommends tracks whose
**interpretable feature vector** points in the same direction as the seed.

Concretely, each track ``t`` is the axis vector ``x_t`` (5 mood axes + 10 soft
genre dimensions; see :mod:`playlistcont.data.schema`).  Given a seed set ``S``
we form the **seed centroid** ``c = mean_{t in S} x_t`` and score every candidate
by the **cosine similarity** between its vector and that centroid::

    score(t) = cos(x_t, c) = (x_t . c) / (||x_t|| ||c||)

The top-k highest-cosine tracks (excluding the seeds) are returned.  Because the
score never looks at play counts or which tracks appear together, this model is
**popularity-blind and cold-start immune**: it will happily surface an obscure
tail track if its features match, and it works the instant a track has features,
before it has ever co-occurred with anything.  In the repo's comparison it lands
just above the axis-only Taste-Engine variant (ABLATIONS.md §2e, R-prec 0.122)
and below the popularity floor on accuracy, while covering far more of the
catalogue and recovering tail tracks that popularity never touches
(REPORT.md §4.9) -- the content-vs-CF trade-off in one model.

Two edge cases are handled explicitly and deterministically:

* **Seedless playlists** (the ``title_only`` scenario has zero seed tracks).
  A pure content model has nothing to point at.  With ``seedless="global_mean"``
  (the default) we fall back to the **global feature centroid** -- i.e. rank
  tracks by how prototypical they are of the whole catalogue.  This carries no
  title signal (content-based reads features, not words), so it is a weak,
  popularity-agnostic guess; it is documented as such.  With ``seedless="empty"``
  the model instead returns an empty list, ceding the cold start to models that
  actually have title signal.
* **Featureless tracks** (an all-zero feature vector -- e.g. a real-MPD track
  with no audio features attached).  Their cosine is undefined; we assign them
  ``-inf`` so they are **never recommended**.  A track must have a feature vector
  to be a content-based candidate.

The model is fully deterministic: identical inputs always yield identical output.
"""
from __future__ import annotations

from typing import List, Optional, Sequence

import numpy as np

from ..data.schema import N_AXES, Dataset
from .base import Recommender, TrackIndex, top_k_excluding


class ContentKNNRecommender(Recommender):
    """Content-based recommender: cosine of each track to the seed centroid.

    Parameters
    ----------
    seedless : {"global_mean", "empty"}
        What to do when there are no usable seed tracks.  ``"global_mean"``
        ranks by similarity to the global feature centroid; ``"empty"`` returns
        an empty list (see the module docstring).
    """

    name = "content_knn"

    def __init__(self, seedless: str = "global_mean"):
        assert seedless in ("global_mean", "empty")
        self.seedless = seedless

    def fit(self, dataset: Dataset) -> "ContentKNNRecommender":
        if dataset.features is None:
            raise ValueError(
                "ContentKNNRecommender needs interpretable features; use "
                "synthetic data or loader.attach_features() for the real MPD."
            )
        self.index = TrackIndex(dataset)
        # raw axis matrix aligned to track ids
        self.X = dataset.feature_matrix(self.index.track_uris)  # (n_tracks, N_AXES)
        norms = np.linalg.norm(self.X, axis=1)
        self._has_feat = norms > 1e-9
        # unit-normalized rows; featureless rows stay all-zero
        self.Xn = np.zeros_like(self.X)
        self.Xn[self._has_feat] = self.X[self._has_feat] / norms[self._has_feat, None]
        # global (unit) direction for the seedless fallback
        if self._has_feat.any():
            gm = self.X[self._has_feat].mean(axis=0)
            gn = float(np.linalg.norm(gm))
            self.global_dir = (gm / gn if gn > 1e-9 else gm).astype(np.float32)
        else:
            self.global_dir = np.zeros(N_AXES, dtype=np.float32)
        return self

    # ------------------------------------------------------------------
    def _seed_dir(self, seed_ids: Sequence[int]) -> Optional[np.ndarray]:
        """Unit direction of the seed centroid, or None if it is degenerate."""
        if seed_ids:
            c = self.X[list(seed_ids)].mean(axis=0)
            n = float(np.linalg.norm(c))
            if n > 1e-9:
                return c / n
        return None

    def score(self, seed_ids: Sequence[int]) -> Optional[np.ndarray]:
        """Cosine of every track to the seed centroid (None => defer / empty).

        Featureless tracks get ``-inf`` so they are never recommended.
        """
        d = self._seed_dir(seed_ids)
        if d is None:
            if self.seedless == "empty":
                return None
            d = self.global_dir
            if float(np.linalg.norm(d)) < 1e-9:
                return None
        scores = self.Xn @ d  # cosine for featured tracks (Xn rows are unit or 0)
        return np.where(self._has_feat, scores, -np.inf).astype(np.float32)

    def recommend(
        self, seed_tracks: Sequence[str], title: Optional[str] = None, k: int = 500
    ) -> List[str]:
        seed_ids = self.index.ids(seed_tracks)
        scores = self.score(seed_ids)
        if scores is None:
            return []  # seedless="empty": no content signal, cede the query
        ids = top_k_excluding(scores, seed_ids, k)
        return self.index.uris(ids)
