"""Tests for the reusable code added for the ablation suite:
coverage/diversity metrics, scenario-aware routing, and the new model knobs.
"""
import numpy as np
import pytest

from playlistcont.challenge.coverage import (bucket_hit_rate, bucketize_catalog,
                                             catalog_coverage,
                                             intra_list_similarity)
from playlistcont.data.synthetic import make_synthetic
from playlistcont.models.hybrid import FEATURES, HybridRecommender
from playlistcont.models.itemcf import ItemCFRecommender
from playlistcont.models.mf import ALSRecommender
from playlistcont.models.popularity import PopularityRecommender
from playlistcont.models.routed import RoutedHybrid, route_weight
from playlistcont.models.title_model import TitleModelRecommender
from playlistcont.models.track2vec import Track2VecRecommender


# --------------------------- coverage metrics ---------------------------
def test_catalog_coverage():
    recs = [["a", "b", "c"], ["b", "c", "d"]]
    # union {a,b,c,d} over catalog of 8 -> 0.5
    assert catalog_coverage(recs, n_catalog=8, k=3) == pytest.approx(0.5)
    # k truncation
    assert catalog_coverage(recs, n_catalog=8, k=1) == pytest.approx(2 / 8)
    assert catalog_coverage([], 8) == 0.0
    assert catalog_coverage(recs, 0) == 0.0


def test_intra_list_similarity():
    feats = {
        "x": np.array([1.0, 0.0]),
        "y": np.array([1.0, 0.0]),   # identical -> sim 1
        "z": np.array([0.0, 1.0]),   # orthogonal -> sim 0
    }
    assert intra_list_similarity(["x", "y"], feats) == pytest.approx(1.0)
    assert intra_list_similarity(["x", "z"], feats) == pytest.approx(0.0)
    # 3-item mean of pairwise sims: (1 + 0 + 0)/3
    assert intra_list_similarity(["x", "y", "z"], feats) == pytest.approx(1 / 3)
    assert intra_list_similarity(["x"], feats) == 0.0  # too short


def test_bucketize_and_hit_rate():
    pop = np.array([100.0, 50.0, 10.0, 5.0, 1.0])
    uris = ["h", "b", "c", "d", "t"]
    buckets = bucketize_catalog(pop, uris, head_frac=0.2, tail_frac=0.2)
    assert buckets["h"] == "head"
    assert buckets["t"] == "tail"
    assert buckets["c"] == "torso"
    # hit-rate accounting
    bh = bucket_hit_rate(["h", "c"], ground_truth=["h", "c", "t"],
                         buckets=buckets, k=10)
    assert bh["head"] == {"hits": 1, "demand": 1}
    assert bh["torso"] == {"hits": 1, "demand": 1}
    assert bh["tail"] == {"hits": 0, "demand": 1}


# --------------------------- routing logic ---------------------------
def test_route_weight_policy():
    # no title -> always 0
    assert route_weight(0, has_title=False) == 0.0
    assert route_weight(5, has_title=False) == 0.0
    # with title: decreasing step function of seed count
    assert route_weight(0, True) == 1.0
    assert route_weight(1, True) == pytest.approx(0.6)
    assert route_weight(5, True) == pytest.approx(0.35)
    assert route_weight(10, True) == 0.0
    assert route_weight(50, True) == 0.0
    # monotone non-increasing
    ws = [route_weight(n, True) for n in range(0, 12)]
    assert all(ws[i] >= ws[i + 1] for i in range(len(ws) - 1))


@pytest.fixture(scope="module")
def small_data():
    return make_synthetic(n_playlists=400, n_tracks=300, seed=3)


def test_routed_hybrid_extremes(small_data):
    subs = {
        "popularity": PopularityRecommender().fit(small_data),
        "item_cf": ItemCFRecommender().fit(small_data),
        "als": ALSRecommender(iterations=3).fit(small_data),
        "track2vec": Track2VecRecommender(epochs=2).fit(small_data),
        "title": TitleModelRecommender().fit(small_data),
    }
    hybrid = HybridRecommender(submodels=subs, n_train_playlists=100).fit(small_data)
    routed = RoutedHybrid(hybrid, subs["title"]).fit(small_data)
    seed = small_data.playlists[0].tracks[:5]
    title = small_data.playlists[0].name or "party pop"

    # title-only (alpha=1) -> exactly the title model's output
    ti = subs["title"].recommend([], title, k=20)
    ro = routed.recommend([], title, k=20)
    assert ro == ti
    # many seeds (alpha=0) -> exactly the hybrid's output
    many = small_data.playlists[0].tracks[:15]
    hy = hybrid.recommend(many, title, k=20)
    ro2 = routed.recommend(many, title, k=20)
    assert ro2 == hy
    # contract: at most k, and no seed leakage on the seeded call
    assert len(ro) <= 20
    assert not (set(ro2) & set(many))


# --------------------------- new model knobs ---------------------------
def test_itemcf_raw_normalization(small_data):
    m = ItemCFRecommender(normalization="raw").fit(small_data)
    seed = small_data.playlists[1].tracks[:4]
    recs = m.recommend(seed, None, k=30)
    assert 0 < len(recs) <= 30
    assert not (set(recs) & set(seed))
    with pytest.raises(AssertionError):
        ItemCFRecommender(normalization="bogus")


def test_title_word_analyzer(small_data):
    m = TitleModelRecommender(analyzer="word", ngram=(1, 2)).fit(small_data)
    recs = m.recommend([], "party pop dance", k=25)
    assert 0 < len(recs) <= 25


def test_hybrid_drop_features_and_sources(small_data):
    subs = {
        "popularity": PopularityRecommender().fit(small_data),
        "item_cf": ItemCFRecommender().fit(small_data),
        "als": ALSRecommender(iterations=3).fit(small_data),
        "track2vec": Track2VecRecommender(epochs=2).fit(small_data),
        "title": TitleModelRecommender().fit(small_data),
    }
    # drop a feature: the masked column must be zero in emitted features
    m = HybridRecommender(submodels=subs, n_train_playlists=100,
                          drop_features=["title"]).fit(small_data)
    seed = small_data.playlists[2].tracks[:5]
    cand_ids, feats = m._candidate_features(seed, small_data.playlists[2].name)
    if len(cand_ids):
        assert np.allclose(feats[:, FEATURES.index("title")], 0.0)
    # leave-one-source-out still returns a valid list
    m2 = HybridRecommender(submodels=subs, n_train_playlists=100,
                           sources=["item_cf", "als", "popularity"]).fit(small_data)
    recs = m2.recommend(seed, None, k=20)
    assert 0 < len(recs) <= 20
    # forced logistic reranker
    m3 = HybridRecommender(submodels=subs, n_train_playlists=100,
                           reranker="logreg").fit(small_data)
    assert m3.blender_backend in ("logreg", "none")
