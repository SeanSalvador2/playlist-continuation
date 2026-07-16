"""Offline tests for the Phase 1.5 enrichment pillar.

Everything here runs on tiny fabricated in-memory tables — no network, no real
downloads (``enrichment.sources`` is deliberately untested beyond its import
because it is the only network-touching module).
"""
import os
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import pytest

from playlistcont.data.real_features import (
    EXTENDED_COLUMNS,
    build_extended_feature_frame,
)
from playlistcont.data.schema import AXES, GENRES, SCALAR_AXES
from playlistcont.enrichment import (
    PRIMARY_GENRE_WEIGHT,
    SECONDARY_GENRE_WEIGHT,
    TAG_TO_BUCKETS,
    build_artist_genre_index,
    build_track_genre_index,
    enrich_history,
    map_tags_to_buckets,
    normalize_artist_name,
)
from playlistcont.history.features import attach_extended_features
from playlistcont.history.schema import HistoryTrack, ListenEvent, ListeningHistory
from playlistcont.history.store import HistoryStore
from playlistcont.history.synthetic import make_synthetic_history

UTC = timezone.utc
_AX = {a: i for i, a in enumerate(AXES)}


def _uri(i: int) -> str:
    return f"spotify:track:{'a' * 21}{i}"


def _tid(i: int) -> str:
    return f"{'a' * 21}{i}"


def _history(artists, plays_per_track=None):
    """Tiny real-provenance history: track i by artists[i-1] (1-based uris)."""
    tracks = {
        _uri(i + 1): HistoryTrack(_uri(i + 1), f"Song {i + 1}", a)
        for i, a in enumerate(artists)
    }
    plays_per_track = plays_per_track or {u: 1 for u in tracks}
    events, minute = [], 0
    for u, n in plays_per_track.items():
        for _ in range(n):
            events.append(ListenEvent(
                ts=datetime(2024, 1, 1, 12, tzinfo=UTC).replace(minute=minute % 60,
                                                                hour=12 + minute // 60),
                track_uri=u, track_name="t", artist_name=tracks[u].artist_name,
                ms_played=30_000))
            minute += 1
    return ListeningHistory(events=events, tracks=tracks, provenance="spotify_export")


# ---------------------------------------------------------------------------
# curated mapping
# ---------------------------------------------------------------------------
def test_mapping_only_targets_schema_genres():
    for buckets in TAG_TO_BUCKETS.values():
        assert buckets, "empty tuples are not allowed; omit the tag instead"
        for b in buckets:
            assert b in GENRES


def test_map_tags_spelling_variants_and_unmapped_passthrough():
    # hyphen/space/& variants all hit the same curated keys
    buckets, raw = map_tags_to_buckets(["hip-hop", "R&B", "zorbo-core"])
    assert buckets[0] == "rap"
    assert "rnb" in buckets
    # unmapped tag survives as a raw tag, never force-mapped
    assert "zorbo-core" in raw
    # administrative spam is dropped from raw tags entirely
    _, raw2 = map_tags_to_buckets(["fixme", "bogus artist", "merge into [unknown]", "folk"])
    assert raw2 == ["folk"]


def test_map_tags_bucket_ordering_by_votes():
    # indie gets 3 votes, folk 2, rock 1 -> primary is indie
    buckets, _ = map_tags_to_buckets(
        ["indie rock", "indie", "indie folk", "folk"])
    assert buckets[0] == "indie"
    assert set(buckets) == {"indie", "folk", "rock"}


# ---------------------------------------------------------------------------
# artist-name normalization + collision policy
# ---------------------------------------------------------------------------
def test_normalize_artist_name():
    assert normalize_artist_name("Beyoncé") == normalize_artist_name("beyonce")
    assert normalize_artist_name("AC/DC") == "acdc"
    assert normalize_artist_name("  The   Weeknd ") == "the weeknd"


def test_collision_policy_prefers_most_tagged_row_and_flags_ambiguous():
    table = pd.DataFrame({
        "name": ["Duster", "Duster", "Duster", "Clairo"],
        "tags": [["rock"], ["indie rock", "slowcore", "indie"], None, ["pop"]],
    })
    idx = build_artist_genre_index(table)
    dust = idx[normalize_artist_name("Duster")]
    assert dust.ambiguous is True
    assert dust.n_candidates == 3
    # the 3-tag row won over the 1-tag and untagged rows
    assert "slowcore" in dust.raw_tags and dust.buckets[0] == "indie"
    clairo = idx[normalize_artist_name("Clairo")]
    assert clairo.ambiguous is False and clairo.n_candidates == 1


# ---------------------------------------------------------------------------
# track-genre aggregation (seed-genre caveat -> multi-genre sets)
# ---------------------------------------------------------------------------
def test_track_genre_multi_seed_aggregation():
    table = pd.DataFrame({
        "track_id": [_tid(1), _tid(1), _tid(1), _tid(2)],
        "track_genre": ["indie", "indie-pop", "sad", "metal"],
    })
    idx = build_track_genre_index(table)
    rec = idx[_tid(1)]
    # seed genres aggregate into a set; unmapped 'sad' kept raw
    assert rec.raw_genres == ["indie", "indie-pop", "sad"]
    assert rec.buckets[0] == "indie" and "pop" in rec.buckets
    assert idx[_tid(2)].buckets == ["metal"]


# ---------------------------------------------------------------------------
# enrich_history: weights, precedence, neutral scalars, report
# ---------------------------------------------------------------------------
def test_soft_weight_fill_primary_and_secondary():
    h = _history(["Phoebe Bridgers"])
    artist_table = pd.DataFrame({
        "name": ["Phoebe Bridgers"],
        "tags": [["indie rock", "indie", "folk"]],
    })
    enrich_history(h, artist_table=artist_table)
    v = h.tracks[_uri(1)].features
    assert v is not None
    assert v[_AX["genre:indie"]] == pytest.approx(PRIMARY_GENRE_WEIGHT)
    assert v[_AX["genre:rock"]] == pytest.approx(SECONDARY_GENRE_WEIGHT)
    assert v[_AX["genre:folk"]] == pytest.approx(SECONDARY_GENRE_WEIGHT)
    assert v[_AX["genre:metal"]] == 0.0


def test_neutral_scalars_created_for_featureless_track():
    h = _history(["Phoebe Bridgers"])
    assert h.tracks[_uri(1)].features is None
    enrich_history(h, artist_table=pd.DataFrame(
        {"name": ["Phoebe Bridgers"], "tags": [["folk"]]}))
    v = h.tracks[_uri(1)].features
    for a in SCALAR_AXES:  # mirrors real_features.py's 0.5-neutral convention
        assert v[_AX[a]] == pytest.approx(0.5)


def test_existing_scalars_left_alone_only_genres_written():
    h = _history(["Phoebe Bridgers"])
    vec = np.full(len(AXES), 0.9, dtype=np.float32)
    h.tracks[_uri(1)].features = vec
    enrich_history(h, artist_table=pd.DataFrame(
        {"name": ["Phoebe Bridgers"], "tags": [["folk"]]}))
    for a in SCALAR_AXES:
        assert vec[_AX[a]] == pytest.approx(0.9)     # untouched
    assert vec[_AX["genre:folk"]] == pytest.approx(PRIMARY_GENRE_WEIGHT)
    assert vec[_AX["genre:pop"]] == 0.0              # stale genre reset


def test_track_level_wins_over_artist_level():
    h = _history(["Taylor Swift"])
    artist_table = pd.DataFrame(
        {"name": ["Taylor Swift"], "tags": [["country"]]})
    track_table = pd.DataFrame(
        {"track_id": [_tid(1)], "track_genre": ["pop"]})
    enrich_history(h, artist_table=artist_table, track_table=track_table)
    v = h.tracks[_uri(1)].features
    assert v[_AX["genre:pop"]] == pytest.approx(PRIMARY_GENRE_WEIGHT)
    assert v[_AX["genre:country"]] == 0.0


def test_track_match_without_mappable_bucket_falls_back_to_artist():
    h = _history(["Taylor Swift"])
    artist_table = pd.DataFrame(
        {"name": ["Taylor Swift"], "tags": [["country"]]})
    track_table = pd.DataFrame(   # 'anime' is deliberately unmapped
        {"track_id": [_tid(1)], "track_genre": ["anime"]})
    rep = enrich_history(h, artist_table=artist_table, track_table=track_table)
    v = h.tracks[_uri(1)].features
    assert v[_AX["genre:country"]] == pytest.approx(PRIMARY_GENRE_WEIGHT)
    assert rep.track_matched == 1  # the id DID match; it just didn't map


def test_enrichment_report_numbers():
    # 3 tracks: t1 artist+track matched, t2 unmatched, t3 artist matched (ambiguous)
    h = _history(["Phoebe Bridgers", "Unknown Nobody", "Taylor Swift"],
                 plays_per_track={_uri(1): 3, _uri(2): 2, _uri(3): 1})
    artist_table = pd.DataFrame({
        "name": ["Phoebe Bridgers", "Taylor Swift", "Taylor Swift"],
        "tags": [["folk"], ["pop", "country"], ["bogus artist"]],
    })
    track_table = pd.DataFrame(
        {"track_id": [_tid(1)], "track_genre": ["indie"]})
    rep = enrich_history(h, artist_table=artist_table, track_table=track_table)
    assert rep.n_tracks == 3
    assert rep.artist_matched == 2
    assert rep.track_matched == 1
    assert rep.enriched_tracks == 2
    assert rep.ambiguous_artists == 1
    # coverage is over PLAYS: t1 (3) + t3 (1) of 6 plays
    assert rep.genre_coverage == pytest.approx(4 / 6)
    payload = rep.to_payload()
    assert "artist_tag_rows" not in payload
    # tag rows: unmapped tags carry a NULL bucket, multi-bucket tags fan out
    rows = rep.artist_tag_rows
    assert {"artist_name": "Phoebe Bridgers", "tag": "folk",
            "mapped_bucket": "folk"} in rows
    assert all(r["tag"] != "bogus artist" for r in rows)


def test_no_tables_is_a_noop_report():
    h = _history(["Phoebe Bridgers"])
    rep = enrich_history(h)
    assert rep.enriched_tracks == 0 and rep.genre_coverage == 0.0
    assert h.tracks[_uri(1)].features is None


# ---------------------------------------------------------------------------
# synthetic histories are untouched by the enrichment path
# ---------------------------------------------------------------------------
def test_synthetic_history_untouched(monkeypatch):
    from app.backend import history_engine

    monkeypatch.delenv("PLAYLISTCONT_HISTORY_EXPORT", raising=False)
    monkeypatch.setenv("PLAYLISTCONT_ENRICH", "1")
    # if the engine ever tried to download for a synthetic history, explode
    import playlistcont.enrichment.sources as sources

    def _boom(*a, **k):
        raise AssertionError("network download attempted on synthetic path")

    monkeypatch.setattr(sources, "_download", _boom)

    h = make_synthetic_history(seed=3, n_days=30)
    before = {u: ht.features.copy() for u, ht in h.tracks.items()}
    payload, tag_rows = history_engine._maybe_enrich(h)
    assert payload is None and tag_rows == []
    for u, ht in h.tracks.items():
        assert np.array_equal(ht.features, before[u])
        assert ht.extended is None
    # and the store still has exactly the three core tables
    st = HistoryStore.from_history(h)
    assert set(st.table_counts()) == {"events", "tracks", "track_features"}
    st.close()


# ---------------------------------------------------------------------------
# prebuilt-pickle load path (serve a large enriched history without recompute)
# ---------------------------------------------------------------------------
def test_prebuilt_pickle_history_load(monkeypatch, tmp_path):
    import pickle

    from app.backend import history_engine

    # a tiny already-enriched real history: give one track a full axis vector so
    # the pickle stands in for a genuinely feature+genre-baked export.
    h = _history(["Phoebe Bridgers", "Zach Bryan"])
    vec = np.zeros(len(AXES), dtype=np.float32)
    vec[_AX["energy"]] = 0.4
    vec[_AX["genre:indie"]] = PRIMARY_GENRE_WEIGHT
    h.tracks[_uri(1)].features = vec

    path = tmp_path / "history_enriched.pkl"
    with open(path, "wb") as fh:
        pickle.dump(h, fh)

    monkeypatch.setenv("PLAYLISTCONT_HISTORY_PICKLE", str(path))
    # even with these set, the pickle path must win and must NOT re-stream/re-enrich
    monkeypatch.setenv("PLAYLISTCONT_HISTORY_EXPORT", "/nonexistent/export.zip")
    monkeypatch.setenv("PLAYLISTCONT_ENRICH", "1")
    import playlistcont.enrichment.sources as sources

    def _boom(*a, **k):
        raise AssertionError("download attempted on the prebuilt-pickle path")

    monkeypatch.setattr(sources, "_download", _boom)

    loaded = history_engine._load_history()
    assert loaded.provenance == "spotify_export"
    assert loaded.n_tracks == 2
    assert np.array_equal(loaded.tracks[_uri(1)].features, vec)
    # enrichment is a no-op on the pickle path (genres already baked in)
    payload, tag_rows = history_engine._maybe_enrich(loaded)
    assert payload is None and tag_rows == []


def test_prebuilt_pickle_rejects_non_history(monkeypatch, tmp_path):
    import pickle

    from app.backend import history_engine

    path = tmp_path / "bad.pkl"
    with open(path, "wb") as fh:
        pickle.dump({"not": "a history"}, fh)
    monkeypatch.setenv("PLAYLISTCONT_HISTORY_PICKLE", str(path))
    with pytest.raises(TypeError):
        history_engine._load_history()


# ---------------------------------------------------------------------------
# extended features: frame attach, table shape / nullability
# ---------------------------------------------------------------------------
def test_attach_extended_features_and_store_table(tmp_path):
    h = _history(["A", "B"])
    frame = pd.DataFrame({
        "id": [_tid(1)],
        "popularity": [61],
        "danceability": [0.71],
        "speechiness": [0.05],
        "loudness": [float("nan")],   # nullable in the source
        "liveness": [0.11],
        "key": [7],
        "mode": [1],
        "duration_ms": [201_000],
    })
    matched = attach_extended_features(h, frame=frame)
    assert matched == 1
    assert h.tracks[_uri(1)].extended["popularity"] == 61
    assert h.tracks[_uri(1)].extended["loudness"] is None   # NaN -> NULL
    assert h.tracks[_uri(2)].extended is None
    # features vector is NOT touched by the extended join
    assert h.tracks[_uri(1)].features is None

    st = HistoryStore.from_history(h)
    counts = st.table_counts()
    assert counts["extended_features"] == 1
    df = st.query("SELECT * FROM extended_features")
    assert list(df.columns) == ["uri", "popularity", "danceability", "speechiness",
                                "loudness", "liveness", "key", "mode", "duration_ms"]
    row = df.iloc[0]
    assert row["uri"] == _uri(1)
    assert int(row["popularity"]) == 61
    assert pd.isna(row["loudness"])                         # NULL survived
    assert int(row["duration_ms"]) == 201_000
    st.close()


def test_extended_table_absent_without_data():
    h = _history(["A"])
    st = HistoryStore.from_history(h)
    assert "extended_features" not in st.table_counts()
    assert not st.has_table("extended_features")
    st.close()


def test_artist_tags_table_only_when_supplied():
    h = _history(["A"])
    st = HistoryStore.from_history(h)
    assert not st.has_table("artist_tags")
    assert st.attach_artist_tags([]) == 0
    assert not st.has_table("artist_tags")                  # no data -> no table
    n = st.attach_artist_tags([
        {"artist_name": "A", "tag": "folk", "mapped_bucket": "folk"},
        {"artist_name": "A", "tag": "weirdcore", "mapped_bucket": None},
    ])
    assert n == 2
    df = st.query("SELECT * FROM artist_tags ORDER BY tag")
    assert list(df.columns) == ["artist_name", "tag", "mapped_bucket"]
    assert df["mapped_bucket"].isna().sum() == 1            # NULL bucket kept
    st.close()


# ---------------------------------------------------------------------------
# widened parquet read (offline: tiny fixture parquet)
# ---------------------------------------------------------------------------
def test_build_extended_feature_frame(tmp_path):
    import pyarrow as pa
    import pyarrow.parquet as pq

    ids = [_tid(1), _tid(2), _tid(1), _tid(3)]   # duplicate id 1 (first wins)
    table = pa.table({
        "id": ids,
        "name": ["N1", "N2", "N1-dup", "N3"],
        "popularity": [10, 20, 99, 30],
        "duration_ms": [1000, 2000, 3000, 4000],
        "time_signature": [4, 4, 4, 3],
        "key": [0, 5, 1, 11],
        "mode": [1, 0, 1, 1],
        "tempo": [120.0, 90.0, 100.0, None],
        "danceability": [0.5, 0.6, 0.7, 0.8],
        "energy": [0.1, 0.2, 0.3, 0.4],
        "loudness": [-8.0, -12.0, -6.0, -20.0],
        "speechiness": [0.03, 0.4, 0.05, 0.1],
        "acousticness": [0.9, 0.1, 0.5, 0.2],
        "instrumentalness": [0.0, 0.8, 0.1, 0.05],
        "liveness": [0.1, 0.3, 0.2, 0.9],
        "valence": [0.6, 0.2, 0.5, 0.9],
    })
    path = os.path.join(tmp_path, "feat.parquet")
    pq.write_table(table, path, row_group_size=2)   # multiple row groups

    calls = []
    df = build_extended_feature_frame(
        path, needed_ids={_tid(1), _tid(3)},
        progress=lambda seen, matched: calls.append((seen, matched)))
    assert list(df.columns) == EXTENDED_COLUMNS
    assert sorted(df["id"]) == [_tid(1), _tid(3)]
    # first occurrence of id 1 won (popularity 10, not the dup's 99)
    assert int(df.loc[df["id"] == _tid(1), "popularity"].iloc[0]) == 10
    assert calls and calls[-1][0] == 4 and calls[-1][1] == 2


def test_build_extended_feature_frame_tolerates_missing_columns(tmp_path):
    import pyarrow as pa
    import pyarrow.parquet as pq

    table = pa.table({"id": [_tid(1)], "popularity": [42]})
    path = os.path.join(tmp_path, "thin.parquet")
    pq.write_table(table, path)
    df = build_extended_feature_frame(path)
    assert list(df.columns) == EXTENDED_COLUMNS
    assert df["tempo"].isna().all()


def test_frozen_feature_table_unchanged_signature():
    """The frozen recsys entry points still exist with their exact signatures."""
    import inspect

    from playlistcont.data.real_features import attach_real_features, build_feature_table

    assert list(inspect.signature(build_feature_table).parameters) == [
        "source", "needed_ids", "progress"]
    assert list(inspect.signature(attach_real_features).parameters) == [
        "ds", "source", "progress"]
