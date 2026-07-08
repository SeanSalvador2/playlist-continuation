"""Title model: recommend from playlists with similar names.

Char n-gram TF-IDF over playlist titles gives a robust similarity that handles
the messy, emoji-laden, misspelled titles of real playlists.  For a query
title we find the nearest training playlists and pool their tracks, weighted by
title similarity and inverse track frequency.  This is the only model with real
signal in the **title-only** scenario, where there are no seed tracks at all.

If a query has no title (the ``no_title_*`` scenarios) it falls back to
popularity.
"""
from __future__ import annotations

from typing import List, Optional, Sequence

import numpy as np
import scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfVectorizer

from ..data.schema import Dataset
from .base import Recommender, TrackIndex


class TitleModelRecommender(Recommender):
    name = "title"

    def __init__(self, neighbours=50, ngram=(2, 5), min_title_len=1):
        self.neighbours = neighbours
        self.ngram = ngram
        self.min_title_len = min_title_len

    def fit(self, dataset: Dataset) -> "TitleModelRecommender":
        self.index = TrackIndex(dataset)
        self.playlists = dataset.playlists
        titles = [(pl.name or "").lower().strip() for pl in dataset.playlists]
        self._has_title = np.array([len(t) >= self.min_title_len for t in titles])
        # Fit only on titled playlists but keep row alignment via a safe token.
        corpus = [t if len(t) >= self.min_title_len else "\x00" for t in titles]
        self.vectorizer = TfidfVectorizer(
            analyzer="char_wb", ngram_range=self.ngram, min_df=2
        )
        self.title_mat = self.vectorizer.fit_transform(corpus)  # playlists x features
        # precompute per-playlist track id lists
        self._pl_track_ids = [
            self.index.ids(pl.tracks) for pl in dataset.playlists
        ]
        # idf-style track weighting: rarer tracks more informative
        pop = self.index.track_pop
        self._track_idf = np.log((self.index.n_playlists + 1) / (pop + 1)) + 1.0
        self._pop_rank = np.argsort(-pop)
        return self

    def recommend(
        self, seed_tracks: Sequence[str], title: Optional[str] = None, k: int = 500
    ) -> List[str]:
        seed_ids = set(self.index.ids(seed_tracks))
        if not title or len(title.strip()) < self.min_title_len:
            out = [int(i) for i in self._pop_rank if int(i) not in seed_ids]
            return self.index.uris(out[:k])
        q = self.vectorizer.transform([title.lower().strip()])
        sims = (self.title_mat @ q.T).toarray().ravel()  # cosine (tfidf is L2-normed)
        n_take = min(len(sims), self.neighbours + 1)
        nbr = np.argpartition(-sims, n_take - 1)[:n_take]
        nbr = nbr[np.argsort(-sims[nbr])]

        scores = np.zeros(self.index.n_tracks, dtype=np.float32)
        for pidx in nbr:
            w = sims[pidx]
            if w <= 0:
                continue
            for tid in self._pl_track_ids[pidx]:
                scores[tid] += w * self._track_idf[tid]
        # rank
        order = np.argsort(-scores)
        out = [int(i) for i in order if scores[i] > 0 and int(i) not in seed_ids][:k]
        if len(out) < k:
            have = set(out) | seed_ids
            for i in self._pop_rank:
                if int(i) not in have:
                    out.append(int(i))
                    if len(out) >= k:
                        break
        return self.index.uris(out[:k])
