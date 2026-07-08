"""Taste-engine sanity: learned weights recover the taste a user is built from."""
import numpy as np
import pytest

from playlistcont.data.schema import AXES, Dataset
from playlistcont.data.synthetic import make_synthetic, ARCHETYPES
from playlistcont.models.taste_engine import (
    TasteEngine, parse_stated, name_flavor,
)

AX = {a: i for i, a in enumerate(AXES)}


@pytest.fixture(scope="module")
def engine_and_data():
    ds = make_synthetic(n_playlists=600, n_tracks=500, seed=5)
    eng = TasteEngine().fit(ds)
    return eng, ds


def _tracks_of_archetype(ds, arch_name, n=25):
    """Pick tracks whose metadata name marks them as the given archetype."""
    uris = [u for u, m in ds.tracks.items()
            if m.track_name.lower().startswith(arch_name.lower())]
    return uris[:n]


def test_learned_weights_sad_slow(engine_and_data):
    eng, ds = engine_and_data
    sad = _tracks_of_archetype(ds, "sad slow country", n=30)
    assert len(sad) >= 10
    w = eng.learn_weights(sad)
    # A user built from sad/slow tracks should have NEGATIVE valence & tempo
    # preference and POSITIVE lyrical_depth / country preference.
    assert w[AX["valence"]] < 0, "expected to lean sad (low valence)"
    assert w[AX["tempo"]] < 0, "expected to lean slow (low tempo)"
    assert w[AX["lyrical_depth"]] > 0, "expected to value deep lyrics"
    assert w[AX["genre:country"]] > 0, "expected country affinity"


def test_learned_weights_gym_rap_contrasts(engine_and_data):
    eng, ds = engine_and_data
    gym = _tracks_of_archetype(ds, "gym rap", n=30)
    w = eng.learn_weights(gym)
    # gym rap is fast + high energy
    assert w[AX["tempo"]] > 0
    assert w[AX["energy"]] > 0
    assert w[AX["genre:rap"]] > 0


def test_parse_stated_directions():
    w = parse_stated("melody and meaning matter most, I love slow sad country")
    assert w[AX["lyrical_depth"]] > 0   # meaning / melody
    assert w[AX["tempo"]] < 0           # slow
    assert w[AX["valence"]] < 0         # sad
    assert w[AX["genre:country"]] > 0   # country


def test_parse_stated_dict():
    w = parse_stated({"valence": -0.8, "tempo": -0.5})
    assert w[AX["valence"]] == pytest.approx(-0.8)
    assert w[AX["tempo"]] == pytest.approx(-0.5)


def test_flavor_clusters_named(engine_and_data):
    eng, ds = engine_and_data
    sad = _tracks_of_archetype(ds, "sad slow country", n=30)
    flavors = eng.flavor_clusters(sad)
    assert len(flavors) >= 1
    joined = " ".join(f.name for f in flavors).lower()
    # the dominant flavour of sad-slow-country tracks should read as slow/sad
    assert "slow" in joined or "sad" in joined
    assert abs(sum(f.share for f in flavors) - 1.0) < 1e-5


def test_explanation_structure(engine_and_data):
    eng, ds = engine_and_data
    sad = _tracks_of_archetype(ds, "sad slow country", n=20)
    recs = eng.recommend(sad[:5], None, k=10)
    ex = eng.explain(recs[0], sad[:5])
    assert ex.track_uri == recs[0]
    assert ex.reasons
    assert isinstance(ex.text(), str) and ex.text().startswith("recommended because")


def test_blend_moves_toward_stated(engine_and_data):
    eng, ds = engine_and_data
    gym = _tracks_of_archetype(ds, "gym rap", n=25)  # learned = fast/energetic
    learned = eng.learn_weights(gym)
    stated = parse_stated("slow sad acoustic")       # opposite direction
    eng.trust = 0.9
    blended = eng.blend(stated, learned)
    # with high trust in a "slow" statement, tempo preference should drop
    assert blended[AX["tempo"]] < learned[AX["tempo"]]


def test_name_flavor_template():
    import numpy as np
    from playlistcont.data.schema import N_AXES, SCALAR_AXES, GENRES
    c = np.zeros(N_AXES)
    c[0] = 0.2   # slow tempo
    c[2] = 0.15  # sad valence
    c[3] = 0.8   # acoustic
    c[4] = 0.8   # deep lyrics
    c[len(SCALAR_AXES) + GENRES.index("country")] = 0.9
    name = name_flavor(c)
    assert "slow" in name and "sad" in name and "country" in name
    assert "deep lyrics" in name
