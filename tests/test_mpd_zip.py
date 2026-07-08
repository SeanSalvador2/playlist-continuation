"""Streaming zip loader tested on a tiny in-memory MPD-style zip.

No network access: the fixture slice json is zipped on the fly into a temp file
that mimics the real ``spotify_million_playlist_dataset.zip`` layout
(``data/mpd.slice.*.json`` members).
"""
import json
import os
import zipfile

from playlistcont.data.mpd_zip import (
    load_challenge_set,
    load_mpd_zip,
    slice_members,
    stream_track_stats,
)

FIXTURE = os.path.join(os.path.dirname(__file__), "fixtures", "mpd.slice.0-1.json")


def _make_zip(tmp_path):
    zpath = os.path.join(tmp_path, "mpd.zip")
    with open(FIXTURE) as fh:
        blob = fh.read()
    with zipfile.ZipFile(zpath, "w", zipfile.ZIP_DEFLATED) as zf:
        # two members with the official naming so ordering is exercised
        zf.writestr("data/mpd.slice.0-1.json", blob)
    return zpath


def test_slice_members_sorted(tmp_path):
    zpath = os.path.join(tmp_path, "m.zip")
    with zipfile.ZipFile(zpath, "w") as zf:
        zf.writestr("data/mpd.slice.1000-1999.json", "{}")
        zf.writestr("data/mpd.slice.0-999.json", "{}")
        zf.writestr("other.txt", "x")
    with zipfile.ZipFile(zpath) as zf:
        assert slice_members(zf) == [
            "data/mpd.slice.0-999.json",
            "data/mpd.slice.1000-1999.json",
        ]


def test_stream_track_stats(tmp_path):
    zpath = _make_zip(tmp_path)
    pop, meta, n_pl, n_int = stream_track_stats(zpath)
    assert n_pl == 2
    # t1 appears in both playlists
    assert pop["spotify:track:t1"] == 2
    assert pop["spotify:track:t0"] == 1
    assert n_int == 5  # 3 + 2 tracks
    assert meta["spotify:track:t0"].artist_name == "Artist A"


def test_load_mpd_zip_matches_schema(tmp_path):
    zpath = _make_zip(tmp_path)
    ds = load_mpd_zip(zpath)
    assert ds.n_playlists == 2
    assert ds.n_tracks == 4
    pl0 = ds.playlists[0]
    assert pl0.name == "Throwback"
    assert pl0.tracks == [
        "spotify:track:t0",
        "spotify:track:t1",
        "spotify:track:t2",
    ]


def test_load_mpd_zip_max_playlists(tmp_path):
    zpath = _make_zip(tmp_path)
    ds = load_mpd_zip(zpath, max_playlists=1)
    assert ds.n_playlists == 1


def test_load_mpd_zip_min_track_count_prunes(tmp_path):
    zpath = _make_zip(tmp_path)
    # only t1 occurs in >=2 playlists; pruning at 2 drops all others.
    ds = load_mpd_zip(zpath, min_track_count=2)
    assert set(ds.tracks.keys()) == {"spotify:track:t1"}
    # both playlists still exist (each contains t1) with only t1 kept
    for pl in ds.playlists:
        assert pl.tracks == ["spotify:track:t1"]


def test_load_mpd_zip_interns_uris(tmp_path):
    zpath = _make_zip(tmp_path)
    ds = load_mpd_zip(zpath)
    # the shared track t1 is the *same* interned object across playlists
    t1_a = [u for u in ds.playlists[0].tracks if u.endswith("t1")][0]
    t1_b = [u for u in ds.playlists[1].tracks if u.endswith("t1")][0]
    assert t1_a is t1_b


def test_load_challenge_set(tmp_path):
    # a minimal challenge_set.json inside a zip
    payload = {
        "date": "2018-01-16",
        "version": "v1",
        "playlists": [
            {"pid": 5, "name": "x", "num_samples": 0, "num_holdouts": 3, "tracks": []}
        ],
    }
    zpath = os.path.join(tmp_path, "challenge.zip")
    with zipfile.ZipFile(zpath, "w") as zf:
        zf.writestr("challenge_set.json", json.dumps(payload))
    pls = load_challenge_set(zpath)
    assert len(pls) == 1 and pls[0]["pid"] == 5
