"""Spotify export loaders tested on fabricated (not real) fixture exports.

Fixtures under ``tests/fixtures/spotify_export/`` are hand-fabricated: a two-file
extended export (~200 music plays over ~3 months, plus podcast rows with null track
metadata, and a mix of skipped flags) and a small basic-format file.  The extended
export is also provided zipped to exercise the zip path.
"""
import os

from playlistcont.history.spotify_export import (
    load_basic_history,
    load_extended_history,
    track_id_from_uri,
)

FIX = os.path.join(os.path.dirname(__file__), "fixtures", "spotify_export")
ZIP = os.path.join(os.path.dirname(__file__), "fixtures", "spotify_export_extended.zip")


def test_extended_dir_parses_and_drops_podcasts():
    h = load_extended_history(FIX)
    assert h.provenance == "spotify_export"
    assert h.ground_truth is None
    # 203 music plays across the two files; 27 podcast rows dropped
    assert h.n_events == 203
    assert h.is_sorted()
    # every kept row has music metadata and a URI
    for e in h.events:
        assert e.track_name and e.artist_name
        assert e.track_uri and e.track_uri.startswith("spotify:track:")


def test_extended_zip_matches_directory():
    hd = load_extended_history(FIX)
    hz = load_extended_history(ZIP)
    assert hz.n_events == hd.n_events
    assert [e.ts for e in hz.events] == [e.ts for e in hd.events]
    assert hz.is_sorted()


def test_extended_multi_file_merge():
    # the fixture is split across two Streaming_History_Audio_*.json files;
    # merging + sorting spans the whole 3-month range in order.
    h = load_extended_history(FIX)
    assert h.events[0].ts < h.events[-1].ts
    files = [f for f in os.listdir(FIX) if f.startswith("Streaming_History_Audio_")]
    assert len(files) >= 2


def test_extended_fields_and_skip_variety():
    h = load_extended_history(FIX)
    e0 = h.events[0]
    assert e0.ts.tzinfo is not None                    # timezone-aware
    assert e0.ts.utcoffset().total_seconds() == 0      # UTC
    assert isinstance(e0.ms_played, int)
    # the export has skipped True, False and null rows
    assert {e.skipped for e in h.events} == {True, False, None}


def test_basic_loader():
    h = load_basic_history(FIX)
    assert h.n_events == 12
    assert h.is_sorted()
    for e in h.events:
        assert e.track_uri is None            # basic export has no URIs
        assert e.album_name is None
        assert e.skipped is None
        assert e.ts.tzinfo is not None
    assert h.n_tracks == 0                     # no URIs -> no keyed tracks


def test_track_id_from_uri():
    assert track_id_from_uri("spotify:track:0123456789ABCDEFGHIJKL") == "0123456789ABCDEFGHIJKL"
    assert len(track_id_from_uri("spotify:track:0123456789ABCDEFGHIJKL")) == 22
    assert track_id_from_uri(None) is None
    assert track_id_from_uri("") is None
