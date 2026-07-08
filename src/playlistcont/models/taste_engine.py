"""The signature model: an interpretable Taste Engine.

Unlike the black-box recommenders, every score here decomposes into human
axes.  A track is a point in interpretable space (tempo, energy, valence,
acousticness, lyrical-depth, and a soft genre vector).  A listener is modelled
by a *preference vector* over those same axes, formed from two sources:

* **STATED** preferences — the user says, in words or a dict, what they care
  about ("melody and meaning matter most, I love slow sad country").  Parsed
  into axis emphases via a small lexicon.
* **LEARNED** preferences — a logistic regression of "is this track in your
  playlists?" onto the standardized axes.  The coefficients ARE the preference
  vector: a positive coefficient on valence means you lean happy, negative
  means you lean sad; the magnitude is how much that axis discriminates your
  taste.

The two are blended with a ``trust`` parameter (how much to believe what the
user *said* over what they *did*).

We then distill the user's tracks into a few named **flavor clusters**
(k-means centroids turned into English like *"slow, sad, acoustic country with
deep lyrics"*), and score candidates as ``axis-match + co-occurrence``.  Every
recommendation comes with an **explanation** naming the flavor it matched, the
load-bearing axis values, and how many seed tracks it co-occurs with.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Union

import numpy as np
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression

from ..data.schema import AXES, GENRES, N_AXES, SCALAR_AXES, Dataset
from .base import Recommender, TrackIndex
from .itemcf import ItemCFRecommender

# Lexicon: phrase -> {axis: signed emphasis}.  Emphases are in "standardized
# preference" units and get L2-normalized before blending.
_LEXICON: Dict[str, Dict[str, float]] = {
    "slow": {"tempo": -1.0},
    "fast": {"tempo": +1.0},
    "upbeat": {"tempo": +0.8, "energy": +0.8, "valence": +0.6},
    "chill": {"energy": -0.8, "tempo": -0.4},
    "calm": {"energy": -0.9},
    "energetic": {"energy": +1.0},
    "hype": {"energy": +1.0, "tempo": +0.7},
    "sad": {"valence": -1.0},
    "melancholy": {"valence": -0.9, "lyrical_depth": +0.4},
    "happy": {"valence": +1.0},
    "acoustic": {"acousticness": +1.0},
    "electronic": {"acousticness": -0.9, "genre:electronic": +0.8},
    "meaning": {"lyrical_depth": +1.0},
    "lyrical": {"lyrical_depth": +1.0},
    "meaningful": {"lyrical_depth": +1.0},
    "deep": {"lyrical_depth": +0.9},
    "melody": {"acousticness": +0.4, "lyrical_depth": +0.4},
}
# genres get a direct positive emphasis when named
for _g in GENRES:
    _LEXICON[_g] = {f"genre:{_g}": +1.0}


def parse_stated(spec: Union[str, Dict[str, float], None]) -> np.ndarray:
    """Turn a phrase or an axis dict into a preference vector over ``AXES``."""
    w = np.zeros(N_AXES, dtype=np.float32)
    if spec is None:
        return w
    axis_idx = {a: i for i, a in enumerate(AXES)}
    if isinstance(spec, dict):
        for axis, val in spec.items():
            if axis in axis_idx:
                w[axis_idx[axis]] += float(val)
        return w
    text = spec.lower()
    for phrase, effects in _LEXICON.items():
        if phrase in text:
            for axis, val in effects.items():
                if axis in axis_idx:
                    w[axis_idx[axis]] += val
    return w


def _l2(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    return v / n if n > 1e-9 else v


def _minmax(v: np.ndarray) -> np.ndarray:
    lo, hi = float(v.min()), float(v.max())
    return (v - lo) / (hi - lo) if hi > lo else np.zeros_like(v)


@dataclass
class Flavor:
    name: str
    centroid: np.ndarray  # raw axis space
    share: float          # fraction of the user's tracks in this cluster


@dataclass
class Explanation:
    track_uri: str
    flavor: str
    reasons: List[str]
    axis_scores: Dict[str, float] = field(default_factory=dict)

    def text(self) -> str:
        return f"recommended because: " + "; ".join(self.reasons)


def name_flavor(centroid: np.ndarray) -> str:
    """Template-based English name for a centroid in raw axis space."""
    tempo, energy, valence, acoustic, lyric = centroid[:5]
    parts: List[str] = []
    parts.append("slow" if tempo < 0.4 else "fast" if tempo > 0.66 else "mid-tempo")
    parts.append("sad" if valence < 0.4 else "upbeat" if valence > 0.66 else "even-keeled")
    if acoustic > 0.6:
        parts.append("acoustic")
    elif acoustic < 0.3:
        parts.append("electronic")
    if energy > 0.75:
        parts.append("high-energy")
    genre_block = centroid[len(SCALAR_AXES):]
    genre = GENRES[int(np.argmax(genre_block))] if genre_block.size else ""
    tail = f" {genre}" if genre else ""
    if lyric > 0.65:
        tail += " with deep lyrics"
    return ", ".join(parts) + tail


class TasteEngine(Recommender):
    name = "taste_engine"

    def __init__(
        self,
        trust: float = 0.4,
        n_flavors: int = 3,
        axis_weight: float = 1.0,
        cf_weight: float = 0.6,
        seed: int = 0,
    ):
        self.trust = trust
        self.n_flavors = n_flavors
        self.axis_weight = axis_weight
        self.cf_weight = cf_weight
        self.seed = seed

    # ------------------------------------------------------------------
    def fit(self, dataset: Dataset) -> "TasteEngine":
        if dataset.features is None:
            raise ValueError(
                "TasteEngine needs interpretable features; use synthetic data "
                "or loader.attach_features() for the real MPD."
            )
        self.index = TrackIndex(dataset)
        # raw axis matrix aligned to track ids
        self.X = dataset.feature_matrix(self.index.track_uris)  # (n_tracks, N_AXES)
        self.mu = self.X.mean(axis=0)
        self.sigma = self.X.std(axis=0) + 1e-6
        self.Z = (self.X - self.mu) / self.sigma  # standardized
        # internal co-occurrence model for the CF term + explanations
        self.cf = ItemCFRecommender(normalization="cosine", topk_sim=100).fit(dataset)
        self._pop_rank = np.argsort(-self.index.track_pop)
        self.artist_of = dataset.track_artist_map()
        self._rng = np.random.default_rng(self.seed)
        return self

    # ------------------------------------------------------------------
    def learn_weights(self, track_uris: Sequence[str]) -> np.ndarray:
        """Preference vector fit from a user's tracks (LEARNED).

        We combine two signals, both learned from the user's own tracks:

        * the **centroid direction** — how far, and in which direction, the
          user's tracks sit from the global average on each standardized axis
          (a robust marginal signal); and
        * **logistic-regression coefficients** — a discriminative "is this
          track yours?" model that sharpens the axes that actually separate
          the user from everyone else.

        Averaging them keeps the vector stable when axes are correlated (e.g.
        tempo and energy) while still being discriminative.
        """
        ids = self.index.ids(track_uris)
        if not ids:
            return np.zeros(N_AXES, dtype=np.float32)
        centroid = _l2(self.Z[ids].mean(axis=0))
        if len(ids) < 3:
            return centroid
        pos = np.array(ids)
        n_neg = min(self.index.n_tracks - len(pos), max(50, 5 * len(pos)))
        pool = np.setdiff1d(np.arange(self.index.n_tracks), pos, assume_unique=False)
        neg = self._rng.choice(pool, size=min(n_neg, len(pool)), replace=False)
        Xtr = np.vstack([self.Z[pos], self.Z[neg]])
        ytr = np.concatenate([np.ones(len(pos)), np.zeros(len(neg))])
        clf = LogisticRegression(max_iter=500, C=1.0)
        clf.fit(Xtr, ytr)
        coef = _l2(clf.coef_[0].astype(np.float32))
        return _l2(0.5 * centroid + 0.5 * coef)

    def blend(self, stated: np.ndarray, learned: np.ndarray) -> np.ndarray:
        return self.trust * _l2(stated) + (1 - self.trust) * _l2(learned)

    # ------------------------------------------------------------------
    def flavor_clusters(self, track_uris: Sequence[str]) -> List[Flavor]:
        ids = self.index.ids(track_uris)
        if not ids:
            return []
        Xu = self.X[ids]
        k = int(min(self.n_flavors, len(ids)))
        if k < 1:
            return []
        km = KMeans(n_clusters=k, n_init=4, random_state=self.seed).fit(Xu)
        flavors: List[Flavor] = []
        for c in range(k):
            mask = km.labels_ == c
            share = float(mask.mean())
            centroid = km.cluster_centers_[c]
            flavors.append(Flavor(name=name_flavor(centroid), centroid=centroid, share=share))
        flavors.sort(key=lambda f: -f.share)
        return flavors

    # ------------------------------------------------------------------
    def _score_all(self, weights: np.ndarray, seed_ids: Sequence[int]) -> np.ndarray:
        axis = self.Z @ weights  # (n_tracks,)
        axis = _minmax(axis)
        if seed_ids:
            cf = self.cf.score(seed_ids)
            cf = _minmax(cf)
        else:
            cf = np.zeros(self.index.n_tracks, dtype=np.float32)
        return self.axis_weight * axis + self.cf_weight * cf

    def recommend(
        self,
        seed_tracks: Sequence[str],
        title: Optional[str] = None,
        k: int = 500,
        stated: Union[str, Dict[str, float], None] = None,
    ) -> List[str]:
        seed_ids = self.index.ids(seed_tracks)
        learned = self.learn_weights(seed_tracks)
        stated_vec = parse_stated(stated)
        if stated is not None and np.linalg.norm(stated_vec) > 0:
            weights = self.blend(stated_vec, learned)
        else:
            weights = learned
        if not seed_ids and np.linalg.norm(weights) < 1e-9:
            # nothing to go on -> popularity
            return self.index.uris([int(i) for i in self._pop_rank[:k]])
        scores = self._score_all(weights, seed_ids)
        ex = set(seed_ids)
        order = np.argsort(-scores)
        out = [int(i) for i in order if int(i) not in ex][:k]
        return self.index.uris(out)

    # ------------------------------------------------------------------
    def explain(
        self,
        candidate_uri: str,
        seed_tracks: Sequence[str],
        weights: Optional[np.ndarray] = None,
        flavors: Optional[List[Flavor]] = None,
    ) -> Explanation:
        """Human-readable justification for one recommendation."""
        if weights is None:
            weights = self.learn_weights(seed_tracks)
        if flavors is None:
            flavors = self.flavor_clusters(seed_tracks)
        cid = self.index.uri_to_id.get(candidate_uri)
        reasons: List[str] = []
        best_flavor = "your taste"
        if cid is not None:
            x = self.X[cid]
            # nearest flavor cluster
            if flavors:
                d = [np.linalg.norm(x - f.centroid) for f in flavors]
                fl = flavors[int(np.argmin(d))]
                best_flavor = fl.name
                reasons.append(
                    f"matches your '{fl.name}' flavor "
                    f"(valence {x[2]:.2f}, tempo {x[0]:.2f})"
                )
            # co-occurrence support with seeds
            seed_ids = self.index.ids(seed_tracks)
            support = 0
            for sid in seed_ids:
                row = self.cf.sim.getrow(sid)
                if cid in set(row.indices.tolist()):
                    support += 1
            if support:
                reasons.append(f"co-occurs with {support} of your seed tracks")
            # artist overlap
            seed_artists = {self.artist_of.get(t) for t in seed_tracks}
            if self.artist_of.get(candidate_uri) in seed_artists:
                reasons.append("same artist as one of your seeds")
        if not reasons:
            reasons.append("popular pick that broadly fits your taste")
        axis_contrib = {}
        if cid is not None and np.linalg.norm(weights) > 0:
            contrib = self.Z[cid] * weights
            top = np.argsort(-np.abs(contrib))[:3]
            axis_contrib = {AXES[i]: float(contrib[i]) for i in top}
        return Explanation(
            track_uri=candidate_uri, flavor=best_flavor,
            reasons=reasons, axis_scores=axis_contrib,
        )
