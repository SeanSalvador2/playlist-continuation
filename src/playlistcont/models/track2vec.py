"""Track2Vec: word2vec over playlists-as-sentences.

Each playlist is a "sentence" whose "words" are track_uris.  gensim's Word2Vec
learns dense track embeddings from co-occurrence within a sliding window.  To
recommend we take the centroid of the seed embeddings and return the nearest
tracks by cosine similarity.  Empty seeds fall back to popularity.
"""
from __future__ import annotations

from typing import List, Optional, Sequence

import numpy as np

from ..data.schema import Dataset
from .base import Recommender, TrackIndex


class Track2VecRecommender(Recommender):
    name = "track2vec"

    def __init__(self, dim=64, window=8, epochs=5, min_count=1, negative=10,
                 workers=4, seed=0, sg=1):
        self.dim = dim
        self.window = window
        self.epochs = epochs
        self.min_count = min_count
        self.negative = negative
        self.workers = workers
        self.seed = seed
        self.sg = sg

    def fit(self, dataset: Dataset) -> "Track2VecRecommender":
        from gensim.models import Word2Vec

        self.index = TrackIndex(dataset)
        sentences = [pl.tracks for pl in dataset.playlists if pl.tracks]
        self.model = Word2Vec(
            sentences=sentences,
            vector_size=self.dim,
            window=self.window,
            min_count=self.min_count,
            sg=self.sg,
            negative=self.negative,
            workers=self.workers,
            seed=self.seed,
            epochs=self.epochs,
        )
        self._pop_rank = np.argsort(-self.index.track_pop)
        # normalized vectors for fast cosine
        self.model.init_sims = getattr(self.model, "init_sims", None)
        return self

    def recommend(
        self, seed_tracks: Sequence[str], title: Optional[str] = None, k: int = 500
    ) -> List[str]:
        wv = self.model.wv
        seeds = [t for t in seed_tracks if t in wv.key_to_index]
        if not seeds:
            return self.index.uris([int(i) for i in self._pop_rank[:k]])
        centroid = np.mean([wv[t] for t in seeds], axis=0)
        # most_similar returns (uri, score); ask for extra to filter seeds
        try:
            sims = wv.most_similar(positive=[centroid], topn=k + len(seeds))
        except KeyError:
            return self.index.uris([int(i) for i in self._pop_rank[:k]])
        seed_set = set(seed_tracks)
        out = [u for u, _ in sims if u not in seed_set][:k]
        if len(out) < k:
            have = set(out) | seed_set
            for i in self._pop_rank:
                u = self.index.track_uris[int(i)]
                if u not in have:
                    out.append(u)
                    if len(out) >= k:
                        break
        return out[:k]
