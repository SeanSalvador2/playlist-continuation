"""Two-stage hybrid: candidate generation + a learned blend (reranker).

Stage 1 (candidate generation): union the top candidates from item-CF, ALS,
Track2Vec, the title model, and popularity.

Stage 2 (rerank): score every candidate with per-candidate features
(co-occurrence score, ALS score, w2v similarity, title score, popularity,
artist overlap with the seed) and a learned blend.  The blend is a LightGBM
ranker if the library is available, else scikit-learn logistic regression.  The
reranker is trained on held-out tracks from the *training* playlists, so it
learns which signal to trust per context — mirroring the winning 2018
"candidate generation + gradient-boosted rerank" recipe.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import numpy as np

from ..data.schema import Dataset
from .base import Recommender, TrackIndex
from .itemcf import ItemCFRecommender
from .mf import ALSRecommender
from .popularity import PopularityRecommender
from .title_model import TitleModelRecommender
from .track2vec import Track2VecRecommender

try:
    import lightgbm as lgb  # type: ignore
    _HAVE_LGB = True
except Exception:  # pragma: no cover
    _HAVE_LGB = False

from sklearn.linear_model import LogisticRegression

FEATURES = ["cf", "als", "w2v", "title", "pop", "artist_overlap"]


class HybridRecommender(Recommender):
    name = "hybrid"

    def __init__(self, cand_per_model=400, n_train_playlists=800, seed=0,
                 submodels: Optional[Dict[str, Recommender]] = None):
        self.cand_per_model = cand_per_model
        self.n_train_playlists = n_train_playlists
        self.seed = seed
        self._external = submodels  # allow sharing already-fit submodels

    # ------------------------------------------------------------------
    def fit(self, dataset: Dataset) -> "HybridRecommender":
        self.index = TrackIndex(dataset)
        self.artist_of = dataset.track_artist_map()
        if self._external:
            self.cf = self._external["item_cf"]
            self.als = self._external["als"]
            self.w2v = self._external["track2vec"]
            self.title = self._external["title"]
            self.pop = self._external["popularity"]
        else:
            self.cf = ItemCFRecommender().fit(dataset)
            self.als = ALSRecommender().fit(dataset)
            self.w2v = Track2VecRecommender().fit(dataset)
            self.title = TitleModelRecommender().fit(dataset)
            self.pop = PopularityRecommender().fit(dataset)
        self._pop_rank = np.argsort(-self.index.track_pop)
        self._pop_norm = self.index.track_pop / (self.index.track_pop.max() + 1e-9)
        self._w2v_wv = self.w2v.model.wv
        self._train_blender(dataset)
        return self

    # ------------------------------------------------------------------
    def _candidate_features(self, seed_tracks, title):
        """Return (candidate_ids, feature_matrix) for one query."""
        seed_ids = self.index.ids(seed_tracks)
        seed_set = set(seed_ids)
        seed_artists = {self.artist_of.get(t) for t in seed_tracks}

        cand: set[int] = set()
        # candidate generation from each source
        for rec in (self.cf, self.als, self.w2v, self.title, self.pop):
            uris = rec.recommend(seed_tracks, title, k=self.cand_per_model)
            cand.update(self.index.ids(uris))
        cand.difference_update(seed_set)
        cand_ids = np.array(sorted(cand), dtype=int)
        if len(cand_ids) == 0:
            return cand_ids, np.zeros((0, len(FEATURES)), np.float32)

        # dense scores for the candidate set
        cf_scores = self.cf.score(seed_ids) if seed_ids else np.zeros(self.index.n_tracks, np.float32)
        if seed_ids:
            u = self.als.user_vector(seed_ids)
            als_scores = self.als.item_factors @ u
        else:
            als_scores = np.zeros(self.index.n_tracks, np.float32)
        # w2v centroid similarity
        w2v_scores = np.zeros(self.index.n_tracks, np.float32)
        seeds_in = [t for t in seed_tracks if t in self._w2v_wv.key_to_index]
        if seeds_in:
            centroid = np.mean([self._w2v_wv[t] for t in seeds_in], axis=0)
            cnorm = centroid / (np.linalg.norm(centroid) + 1e-9)
            for cid in cand_ids:
                uri = self.index.track_uris[cid]
                if uri in self._w2v_wv.key_to_index:
                    v = self._w2v_wv[uri]
                    w2v_scores[cid] = float(cnorm @ (v / (np.linalg.norm(v) + 1e-9)))
        # title scores (rank-based, only if title present)
        title_scores = np.zeros(self.index.n_tracks, np.float32)
        if title:
            t_uris = self.title.recommend(seed_tracks, title, k=self.cand_per_model)
            for rank, uri in enumerate(t_uris):
                tid = self.index.uri_to_id.get(uri)
                if tid is not None:
                    title_scores[tid] = 1.0 - rank / max(1, len(t_uris))

        def norm(a):
            m = a[cand_ids].max()
            return a / (m + 1e-9)

        cf_n, als_n, w2v_n = norm(cf_scores), norm(als_scores), norm(w2v_scores)
        feats = np.zeros((len(cand_ids), len(FEATURES)), np.float32)
        for j, cid in enumerate(cand_ids):
            artist_ov = 1.0 if self.artist_of.get(self.index.track_uris[cid]) in seed_artists else 0.0
            feats[j] = [
                cf_n[cid], als_n[cid], w2v_n[cid],
                title_scores[cid], self._pop_norm[cid], artist_ov,
            ]
        return cand_ids, feats

    # ------------------------------------------------------------------
    def _train_blender(self, dataset: Dataset):
        rng = np.random.default_rng(self.seed)
        train_pls = [pl for pl in dataset.playlists if len(pl.tracks) >= 12]
        rng.shuffle(train_pls)
        train_pls = train_pls[: self.n_train_playlists]

        X_rows, y_rows = [], []
        for pl in train_pls:
            n_seed = rng.choice([5, 10, 25])
            if len(pl.tracks) <= n_seed + 3:
                continue
            seed = pl.tracks[:n_seed]
            held = set(pl.tracks[n_seed:])
            cand_ids, feats = self._candidate_features(seed, pl.name)
            if len(cand_ids) == 0:
                continue
            labels = np.array(
                [1 if self.index.track_uris[c] in held else 0 for c in cand_ids]
            )
            if labels.sum() == 0:
                continue
            # subsample negatives to keep it balanced-ish
            pos_idx = np.where(labels == 1)[0]
            neg_idx = np.where(labels == 0)[0]
            keep_neg = rng.choice(neg_idx, size=min(len(neg_idx), 10 * len(pos_idx) + 20), replace=False)
            sel = np.concatenate([pos_idx, keep_neg])
            X_rows.append(feats[sel])
            y_rows.append(labels[sel])

        X = np.vstack(X_rows) if X_rows else np.zeros((0, len(FEATURES)), np.float32)
        y = np.concatenate(y_rows) if y_rows else np.zeros(0)
        self.blender_backend = "none"
        self._weights = np.ones(len(FEATURES), np.float32)  # fallback equal blend
        if len(y) > 20 and y.sum() > 0 and y.sum() < len(y):
            if _HAVE_LGB:
                self.blender = lgb.LGBMClassifier(
                    n_estimators=120, num_leaves=15, learning_rate=0.1,
                    min_child_samples=10, random_state=self.seed, verbose=-1,
                )
                self.blender.fit(X, y)
                self.blender_backend = "lightgbm"
            else:
                self.blender = LogisticRegression(max_iter=500, class_weight="balanced")
                self.blender.fit(X, y)
                self.blender_backend = "logreg"
        else:
            self.blender = None

    def _score_candidates(self, feats: np.ndarray) -> np.ndarray:
        if self.blender is None:
            return feats @ self._weights
        if self.blender_backend == "lightgbm":
            return self.blender.predict_proba(feats)[:, 1]
        return self.blender.predict_proba(feats)[:, 1]

    # ------------------------------------------------------------------
    def recommend(
        self, seed_tracks: Sequence[str], title: Optional[str] = None, k: int = 500
    ) -> List[str]:
        cand_ids, feats = self._candidate_features(seed_tracks, title)
        if len(cand_ids) == 0:
            return self.index.uris([int(i) for i in self._pop_rank[:k]])
        scores = self._score_candidates(feats)
        order = np.argsort(-scores)
        out = [int(cand_ids[i]) for i in order][:k]
        if len(out) < k:
            have = set(out) | set(self.index.ids(seed_tracks))
            for i in self._pop_rank:
                if int(i) not in have:
                    out.append(int(i))
                    if len(out) >= k:
                        break
        return self.index.uris(out[:k])
