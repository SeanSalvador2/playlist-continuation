"""DuckDB history store: row counts + SQL aggregations match raw-event pandas.

The store is the query surface the dashboard (Phase 1) and text-to-SQL (later) build
on, so we check that SQL over it agrees with the same computation done directly on the
raw event stream.
"""
import os
from collections import Counter

import pandas as pd

from playlistcont.history.features import attach_real_features
from playlistcont.history.spotify_export import load_extended_history, track_id_from_uri
from playlistcont.history.store import HistoryStore
from playlistcont.history.synthetic import make_synthetic_history

FIX = os.path.join(os.path.dirname(__file__), "fixtures", "spotify_export")


def test_row_counts_match():
    h = make_synthetic_history(seed=1, n_days=90)
    st = HistoryStore.from_history(h)
    counts = st.table_counts()
    assert counts["events"] == h.n_events
    assert counts["tracks"] == h.n_tracks
    # synthetic tracks all carry features
    assert counts["track_features"] == h.n_tracks
    st.close()


def test_events_per_day_sql_equals_pandas():
    h = make_synthetic_history(seed=2, n_days=90)
    st = HistoryStore.from_history(h)
    sql = st.query("SELECT date, COUNT(*) AS c FROM events GROUP BY date ORDER BY date")
    raw = Counter(e.ts.date() for e in h.events)
    # DuckDB DATE comes back as pandas Timestamps; compare on the calendar date
    assert list(sql["c"]) == [raw[pd.Timestamp(d).date()] for d in sql["date"]]
    assert int(sql["c"].sum()) == h.n_events
    st.close()


def test_top_artists_sql_equals_pandas():
    h = make_synthetic_history(seed=2, n_days=90)
    st = HistoryStore.from_history(h)
    sql = st.query(
        "SELECT t.artist_name AS artist, COUNT(*) AS c "
        "FROM events e JOIN tracks t ON e.track_uri = t.uri "
        "GROUP BY t.artist_name ORDER BY c DESC, artist LIMIT 5"
    )
    raw = Counter(h.tracks[e.track_uri].artist_name for e in h.events)
    expected = sorted(raw.items(), key=lambda kv: (-kv[1], kv[0]))[:5]
    assert list(zip(sql["artist"], sql["c"])) == [(a, c) for a, c in expected]
    st.close()


def test_file_backed_reopen_read_only(tmp_path):
    h = make_synthetic_history(seed=1, n_days=60)
    path = os.path.join(tmp_path, "hist.duckdb")
    st = HistoryStore.from_history(h, path=path)
    counts = st.table_counts()
    st.close()

    ro = HistoryStore.open(path, read_only=True)
    assert ro.table_counts() == counts
    st.close()


def test_real_history_features_absent_until_attached():
    h = load_extended_history(FIX)
    st = HistoryStore.from_history(h)
    # a real export has no axis features yet
    assert st.table_counts()["track_features"] == 0
    st.close()

    # inject a feature table (offline, no network) for a subset and re-store
    import numpy as np

    uris = list(h.tracks)[:3]
    ft = {track_id_from_uri(u): np.arange(15, dtype=np.float32) for u in uris}
    matched = attach_real_features(h, feature_table=ft)
    assert matched == 3
    st2 = HistoryStore.from_history(h)
    assert st2.table_counts()["track_features"] == 3
    st2.close()
