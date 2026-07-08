"""MPD loader tested on the small fixture slice."""
import os

from playlistcont.data.loader import load_mpd

FIXTURE_DIR = os.path.join(os.path.dirname(__file__), "fixtures")


def test_load_fixture():
    ds = load_mpd(FIXTURE_DIR)
    assert ds.n_playlists == 2
    # 4 distinct track_uris across the two playlists (t1 shared)
    assert ds.n_tracks == 4
    assert set(ds.tracks.keys()) == {
        "spotify:track:t0", "spotify:track:t1",
        "spotify:track:t2", "spotify:track:t3",
    }


def test_playlist_fields():
    ds = load_mpd(FIXTURE_DIR)
    pl0 = ds.playlists[0]
    assert pl0.pid == 0
    assert pl0.name == "Throwback"
    assert pl0.tracks == ["spotify:track:t0", "spotify:track:t1", "spotify:track:t2"]


def test_track_metadata():
    ds = load_mpd(FIXTURE_DIR)
    m = ds.tracks["spotify:track:t0"]
    assert m.artist_uri == "spotify:artist:a0"
    assert m.artist_name == "Artist A"
    assert m.album_name == "Alb0"


def test_track_artist_map_shared_artist():
    ds = load_mpd(FIXTURE_DIR)
    amap = ds.track_artist_map()
    # t0 and t2 are both Artist A (a0)
    assert amap["spotify:track:t0"] == amap["spotify:track:t2"] == "spotify:artist:a0"


def test_max_playlists():
    ds = load_mpd(FIXTURE_DIR, max_playlists=1)
    assert ds.n_playlists == 1
