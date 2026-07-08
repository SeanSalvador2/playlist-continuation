"""Fit/recommend contract for every recommender on tiny synthetic data."""
import pytest

from playlistcont.data.synthetic import make_synthetic
from playlistcont.data.schema import Dataset
from playlistcont.models.popularity import PopularityRecommender
from playlistcont.models.itemcf import ItemCFRecommender
from playlistcont.models.mf import ALSRecommender
from playlistcont.models.track2vec import Track2VecRecommender
from playlistcont.models.title_model import TitleModelRecommender
from playlistcont.models.taste_engine import TasteEngine
from playlistcont.models.hybrid import HybridRecommender


@pytest.fixture(scope="module")
def data():
    return make_synthetic(n_playlists=400, n_tracks=300, seed=3)


def _all_models(data):
    subs = {
        "popularity": PopularityRecommender().fit(data),
        "item_cf": ItemCFRecommender().fit(data),
        "als": ALSRecommender(iterations=5).fit(data),
        "track2vec": Track2VecRecommender(epochs=2).fit(data),
        "title": TitleModelRecommender().fit(data),
    }
    subs["taste_engine"] = TasteEngine().fit(data)
    subs["hybrid"] = HybridRecommender(
        submodels={k: subs[k] for k in
                   ["item_cf", "als", "track2vec", "title", "popularity"]},
        n_train_playlists=100,
    ).fit(data)
    return subs


def test_contract_all_models(data):
    seed_pl = next(p for p in data.playlists if len(p.tracks) >= 8)
    seed = seed_pl.tracks[:5]
    for name, model in _all_models(data).items():
        recs = model.recommend(seed, seed_pl.name, k=50)
        assert isinstance(recs, list), name
        assert len(recs) <= 50, name
        assert len(recs) == len(set(recs)), f"{name} returned duplicates"
        assert not (set(recs) & set(seed)), f"{name} leaked seed tracks"
        assert all(r.startswith("spotify:track:") for r in recs), name


def test_empty_seed_title_only(data):
    # title-only scenario: no seeds, only a title
    models = _all_models(data)
    for name, model in models.items():
        recs = model.recommend([], "chill indie mellow", k=30)
        assert len(recs) <= 30
        assert len(recs) == len(set(recs)), name


def test_k_respected(data):
    model = ItemCFRecommender().fit(data)
    seed_pl = next(p for p in data.playlists if len(p.tracks) >= 8)
    recs = model.recommend(seed_pl.tracks[:3], seed_pl.name, k=10)
    assert len(recs) == 10


def test_als_fallback_backend(data):
    m = ALSRecommender(iterations=3, force_fallback=True).fit(data)
    assert m.backend == "numpy_als"
    recs = m.recommend(data.playlists[0].tracks[:3], None, k=20)
    assert len(recs) <= 20
