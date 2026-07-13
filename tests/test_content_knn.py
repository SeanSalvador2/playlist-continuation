"""Content-based recommender: hand-computed cosine + edge cases + contract.

The ContentKNN model scores a candidate track by the cosine similarity between
its interpretable feature vector and the mean (centroid) of the seed tracks'
feature vectors.  These tests pin the exact arithmetic on tiny fixtures a reader
can verify on paper, then check the documented edge cases (seedless, featureless,
popularity-blindness), determinism, and the shared recommender contract.
"""
import numpy as np
import pytest

from playlistcont.data.schema import N_AXES, Dataset, Playlist, TrackMeta
from playlistcont.data.synthetic import make_synthetic
from playlistcont.models.content_knn import ContentKNNRecommender


# ---------------------------------------------------------------------------
# tiny hand-built dataset: features live on two axes (tempo=0, energy=1); the
# remaining 13 axes are zero, so cosine is a plain 2-D cosine we can hand-check.
# ---------------------------------------------------------------------------
def _vec(tempo: float, energy: float) -> np.ndarray:
    v = np.zeros(N_AXES, dtype=np.float32)
    v[0] = tempo
    v[1] = energy
    return v


def _meta(uri: str) -> TrackMeta:
    return TrackMeta(
        track_uri=uri, track_name=uri, artist_uri="spotify:artist:a0",
        artist_name="A", album_uri="spotify:album:al0", album_name="Al",
    )


def _make_ds(feats: dict, playlists) -> Dataset:
    tracks = {u: _meta(u) for u in feats}
    pls = [Playlist(pid=i, name="", tracks=list(t)) for i, t in enumerate(playlists)]
    return Dataset(playlists=pls, tracks=tracks, features=feats)


# URIs chosen so the model's cosine, not the id order, drives ranking.
S = "spotify:track:S"   # seed, direction (1, 0)
A = "spotify:track:A"   # (1, 0)  -> cos 1.0 with seed centroid
B = "spotify:track:B"   # (1, 1)  -> cos 1/sqrt(2)
C = "spotify:track:C"   # (0, 1)  -> cos 0.0
D = "spotify:track:D"   # (0, 0)  -> featureless, never recommended


@pytest.fixture()
def toy():
    feats = {S: _vec(1, 0), A: _vec(1, 0), B: _vec(1, 1), C: _vec(0, 1), D: _vec(0, 0)}
    # one playlist just so every track is "known"; content model ignores it
    return _make_ds(feats, [[S, A, B, C, D]])


def test_cosine_ranking_hand_computed(toy):
    m = ContentKNNRecommender().fit(toy)
    # seed centroid = (1, 0); cosines: A=1.0, B=1/sqrt(2)=0.7071, C=0.0
    recs = m.recommend([S], k=10)
    assert recs == [A, B, C]          # exact ranking; D excluded (featureless)
    scores = m.score([m.index.uri_to_id[S]])
    assert scores[m.index.uri_to_id[A]] == pytest.approx(1.0, abs=1e-6)
    assert scores[m.index.uri_to_id[B]] == pytest.approx(1 / np.sqrt(2), abs=1e-6)
    assert scores[m.index.uri_to_id[C]] == pytest.approx(0.0, abs=1e-6)


def test_featureless_track_never_recommended(toy):
    m = ContentKNNRecommender().fit(toy)
    recs = m.recommend([S], k=500)
    assert D not in recs                                   # zero-vector track
    assert m.score([m.index.uri_to_id[S]])[m.index.uri_to_id[D]] == -np.inf


def test_seed_centroid_is_the_mean_direction():
    # two seeds averaging to direction (1, 1): the (1,1) track must rank first.
    P, Q = "spotify:track:P", "spotify:track:Q"
    feats = {P: _vec(2, 0), Q: _vec(0, 2), A: _vec(1, 0), B: _vec(1, 1)}
    ds = _make_ds(feats, [[P, Q, A, B]])
    m = ContentKNNRecommender().fit(ds)
    recs = m.recommend([P, Q], k=10)          # centroid (1,1) -> B is closest
    assert recs[0] == B
    sc = m.score([m.index.uri_to_id[P], m.index.uri_to_id[Q]])
    assert sc[m.index.uri_to_id[B]] == pytest.approx(1.0, abs=1e-6)
    assert sc[m.index.uri_to_id[A]] == pytest.approx(1 / np.sqrt(2), abs=1e-6)


def test_ignores_popularity():
    # X and Y share an identical feature vector but wildly different play counts;
    # a content model must score them identically (no popularity signal).
    X, Y = "spotify:track:X", "spotify:track:Y"
    feats = {S: _vec(1, 0), X: _vec(0, 1), Y: _vec(0, 1)}
    playlists = [[S, X]] * 20 + [[Y]]        # X in 20 playlists, Y in 1
    ds = _make_ds(feats, playlists)
    m = ContentKNNRecommender().fit(ds)
    sc = m.score([m.index.uri_to_id[S]])
    assert sc[m.index.uri_to_id[X]] == pytest.approx(sc[m.index.uri_to_id[Y]])


def test_seedless_global_mean_fallback(toy):
    m = ContentKNNRecommender(seedless="global_mean").fit(toy)
    recs = m.recommend([], title="whatever", k=10)
    assert recs and D not in recs            # non-empty, still skips featureless
    assert recs == m.recommend([], title="different title", k=10)  # title-blind


def test_seedless_empty_mode(toy):
    m = ContentKNNRecommender(seedless="empty").fit(toy)
    assert m.recommend([], title="whatever", k=10) == []


def test_all_zero_seed_falls_back(toy):
    # seeding only the featureless track => degenerate centroid => fallback path.
    m = ContentKNNRecommender(seedless="empty").fit(toy)
    assert m.recommend([D], k=10) == []


def test_needs_features():
    ds = make_synthetic(n_playlists=50, n_tracks=40, seed=0)
    ds.features = None
    with pytest.raises(ValueError):
        ContentKNNRecommender().fit(ds)


# ---------------------------------------------------------------------------
# determinism + shared recommender contract on real synthetic data
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def data():
    return make_synthetic(n_playlists=400, n_tracks=300, seed=3)


def test_determinism(data):
    m1 = ContentKNNRecommender().fit(data)
    m2 = ContentKNNRecommender().fit(data)
    seed_pl = next(p for p in data.playlists if len(p.tracks) >= 8)
    seed = seed_pl.tracks[:5]
    assert m1.recommend(seed, seed_pl.name, k=50) == m2.recommend(seed, seed_pl.name, k=50)
    # repeated calls on the same instance are identical too
    assert m1.recommend(seed, None, k=50) == m1.recommend(seed, None, k=50)


def test_contract(data):
    m = ContentKNNRecommender().fit(data)
    seed_pl = next(p for p in data.playlists if len(p.tracks) >= 8)
    seed = seed_pl.tracks[:5]
    recs = m.recommend(seed, seed_pl.name, k=50)
    assert isinstance(recs, list)
    assert len(recs) <= 50
    assert len(recs) == len(set(recs))                    # no duplicates
    assert not (set(recs) & set(seed))                    # never leaks seeds
    assert all(r.startswith("spotify:track:") for r in recs)


def test_contract_empty_seed(data):
    m = ContentKNNRecommender().fit(data)
    recs = m.recommend([], "chill indie mellow", k=30)
    assert len(recs) <= 30
    assert len(recs) == len(set(recs))


def test_k_respected(data):
    m = ContentKNNRecommender().fit(data)
    seed_pl = next(p for p in data.playlists if len(p.tracks) >= 8)
    recs = m.recommend(seed_pl.tracks[:3], seed_pl.name, k=10)
    assert len(recs) == 10
