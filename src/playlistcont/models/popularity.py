"""Popularity baseline: always recommend the globally most-played tracks."""
from __future__ import annotations

from typing import List, Optional, Sequence

import numpy as np

from ..data.schema import Dataset
from .base import Recommender, TrackIndex


class PopularityRecommender(Recommender):
    name = "popularity"

    def fit(self, dataset: Dataset) -> "PopularityRecommender":
        self.index = TrackIndex(dataset)
        self._ranked = np.argsort(-self.index.track_pop)
        return self

    def recommend(
        self, seed_tracks: Sequence[str], title: Optional[str] = None, k: int = 500
    ) -> List[str]:
        exclude = set(self.index.ids(seed_tracks))
        out = [int(i) for i in self._ranked if i not in exclude]
        return self.index.uris(out[:k])
