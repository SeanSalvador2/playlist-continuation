"""Analytics queries cross-checked against independent pandas on the raw event stream.

Each :mod:`playlistcont.analytics.queries` function is verified against the same
computation done directly on ``history.events`` (NOT via SQL over the store), the
fixture-equality style of ``tests/test_history_store.py``.  Covers window filtering,
pagination, the rolling mean, the (full-history) discovery definition and CSV round-trip.
"""
import csv
import io
from collections import Counter
from datetime import date

import numpy as np
import pandas as pd

from playlistcont.analytics import queries as q
from playlistcont.data.schema import GENRES, SCALAR_AXES
from playlistcont.history.store import HistoryStore, feature_column
from playlistcont.history.synthetic import make_synthetic_history


def _store(seed=11, n_days=120):
    h = make_synthetic_history(seed=seed, n_days=n_days)
    return h, HistoryStore.from_history(h)


def _events_df(h, start=None, end=None):
    """Independent pandas frame straight from the ListenEvent objects (no SQL)."""
    rows = [{
        "date": e.ts.date(),
        "hour": e.ts.hour,
        "weekday": e.ts.weekday(),
        "track_uri": e.track_uri,
        "ms_played": e.ms_played,
        "skipped": e.skipped,
        "artist_name": h.tracks[e.track_uri].artist_name,
        "album_name": h.tracks[e.track_uri].album_name,
        "track_name": h.tracks[e.track_uri].track_name,
    } for e in h.events]
    df = pd.DataFrame(rows)
    if start is not None:
        df = df[df["date"] >= start]
    if end is not None:
        df = df[df["date"] <= end]
    return df.reset_index(drop=True)


# ---------------------------------------------------------------------------
# summary
# ---------------------------------------------------------------------------
def test_summary_matches_pandas():
    h, st = _store()
    got = q.summary(st)
    df = _events_df(h)

    assert got["total_plays"] == len(df)
    assert got["distinct_tracks"] == df["track_uri"].nunique()
    assert got["distinct_artists"] == df["artist_name"].nunique()
    assert got["total_minutes"] == round(df["ms_played"].sum() / 60000.0, 2)
    skips = int(df["skipped"].sum())
    assert got["skip_rate"] == round(skips / len(df), 4)
    first, last = df["date"].min(), df["date"].max()
    assert got["span"] == {"first": first.isoformat(), "last": last.isoformat(),
                           "days": (last - first).days + 1}
    assert got["plays_per_day"] == round(len(df) / ((last - first).days + 1), 3)
    st.close()


def test_summary_window_filters():
    h, st = _store()
    start, end = date(2022, 2, 1), date(2022, 2, 28)
    got = q.summary(st, start, end)
    df = _events_df(h, start, end)
    assert got["total_plays"] == len(df)
    assert got["start"] == "2022-02-01" and got["end"] == "2022-02-28"
    # a window strictly inside the history is smaller than the whole thing
    assert got["total_plays"] < q.summary(st)["total_plays"]
    st.close()


def test_summary_empty_window():
    _h, st = _store()
    got = q.summary(st, date(2019, 1, 1), date(2019, 1, 2))
    assert got["total_plays"] == 0
    assert got["span"] is None
    assert got["skip_rate"] == 0.0 and got["plays_per_day"] == 0.0
    st.close()


# ---------------------------------------------------------------------------
# top_items
# ---------------------------------------------------------------------------
def test_top_artists_matches_pandas():
    h, st = _store()
    got = q.top_items(st, "artists", by="plays")
    df = _events_df(h)
    counts = Counter(df["artist_name"])
    expected = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
    assert got["total"] == len(expected)
    assert [(r["name"], r["plays"]) for r in got["rows"]] == [(a, c) for a, c in expected]
    # rank is 1-based and dense
    assert [r["rank"] for r in got["rows"]] == list(range(1, len(expected) + 1))
    # share sums to ~1 over the full list
    assert abs(sum(r["share"] for r in got["rows"]) - 1.0) < 1e-2
    st.close()


def test_top_tracks_by_minutes_and_artist_field():
    h, st = _store()
    got = q.top_items(st, "tracks", by="minutes", limit=5)
    df = _events_df(h)
    mins = df.groupby("track_uri")["ms_played"].sum() / 60000.0
    top_uri = mins.idxmax()
    assert got["rows"][0]["name"] == h.tracks[top_uri].track_name
    assert got["rows"][0]["artist"] == h.tracks[top_uri].artist_name
    # ordering is by minutes descending
    vals = [r["minutes"] for r in got["rows"]]
    assert vals == sorted(vals, reverse=True)
    st.close()


def test_top_items_pagination():
    _h, st = _store()
    full = q.top_items(st, "tracks", by="plays")           # everything
    p0 = q.top_items(st, "tracks", by="plays", limit=10, offset=0)
    p1 = q.top_items(st, "tracks", by="plays", limit=10, offset=10)
    assert len(p0["rows"]) == 10 and len(p1["rows"]) == 10
    assert [r["rank"] for r in p1["rows"]] == list(range(11, 21))
    # pages are the corresponding slices of the full ranked list
    assert [r["name"] for r in p0["rows"]] == [r["name"] for r in full["rows"][:10]]
    assert [r["name"] for r in p1["rows"]] == [r["name"] for r in full["rows"][10:20]]
    # limit=0 means everything
    assert q.top_items(st, "tracks", limit=0)["total"] == full["total"]
    assert len(q.top_items(st, "tracks", limit=0)["rows"]) == full["total"]
    st.close()


# ---------------------------------------------------------------------------
# trends
# ---------------------------------------------------------------------------
def test_trends_plays_by_month_matches_pandas():
    h, st = _store()
    got = q.trends(st, "plays", "month")
    df = _events_df(h)
    by_month = Counter((d.year, d.month) for d in df["date"])
    expected = [by_month[k] for k in sorted(by_month)]
    assert [b["value"] for b in got["buckets"]] == [float(v) for v in expected]
    # bucket labels are the first of each month
    assert got["buckets"][0]["bucket"].endswith("-01")
    st.close()


def test_trends_discovery_uses_full_history_first_play():
    h, st = _store()
    # A play is a "first-ever play" iff it is the track's earliest event across the
    # FULL history (events are ts-sorted, so the first occurrence per track).  Compute
    # that index set independently, then measure it inside a window.
    first_seen = set()
    first_ever_ids = set()
    for i, e in enumerate(h.events):
        if e.track_uri not in first_seen:
            first_seen.add(e.track_uri)
            first_ever_ids.add(i)

    # window: the second month only — many tracks were already discovered earlier
    start, end = date(2022, 2, 1), date(2022, 2, 28)
    got = q.trends(st, "discovery", "month", start=start, end=end)
    win_ids = [i for i, e in enumerate(h.events) if start <= e.ts.date() <= end]
    firsts = sum(1 for i in win_ids if i in first_ever_ids)
    expected_share = firsts / len(win_ids)
    assert len(got["buckets"]) == 1
    assert got["buckets"][0]["value"] == round(float(expected_share), 4)

    # honesty of the full-history definition: a naive within-window "first play" would
    # over-count (tracks already known before the window look falsely new at its edge).
    win = _events_df(h, start, end)
    naive_first = win.groupby("track_uri")["date"].min()
    naive_share = (win["track_uri"].map(lambda u: naive_first[u]) == win["date"]).mean()
    assert naive_share > expected_share
    st.close()


def test_trends_skip_rate_matches_pandas():
    h, st = _store()
    got = q.trends(st, "skip_rate", "month")
    df = _events_df(h)
    df["ym"] = df["date"].map(lambda d: (d.year, d.month))
    for b in got["buckets"]:
        y, m = int(b["bucket"][:4]), int(b["bucket"][5:7])
        sub = df[df["ym"] == (y, m)]
        assert b["value"] == round(sub["skipped"].sum() / len(sub), 4)
    st.close()


def test_trends_rolling_is_centered_mean():
    _h, st = _store()
    got = q.trends(st, "plays", "week", rolling=3)
    values = [b["value"] for b in got["buckets"]]
    expected = pd.Series(values).rolling(3, center=True, min_periods=1).mean()
    assert [b["rolling"] for b in got["buckets"]] == [round(float(v), 4) for v in expected]
    assert got["rolling"] == 3
    # without rolling there is no rolling key
    plain = q.trends(st, "plays", "week")
    assert "rolling" not in plain["buckets"][0]
    assert plain["rolling"] is None
    st.close()


# ---------------------------------------------------------------------------
# listening clock
# ---------------------------------------------------------------------------
def test_listening_clock_matches_pandas():
    h, st = _store()
    got = q.listening_clock(st)
    df = _events_df(h)
    cell = Counter(zip(df["weekday"], df["hour"]))
    for w in range(7):
        for hr in range(24):
            assert got["matrix"][w][hr] == cell[(w, hr)]
    assert got["total"] == len(df)
    assert got["max"] == max(cell.values())
    st.close()


# ---------------------------------------------------------------------------
# axes over time + coverage
# ---------------------------------------------------------------------------
def test_axes_over_time_means_and_coverage():
    h, st = _store()
    got = q.axes_over_time(st, "month")
    df = _events_df(h)
    # per-event feature vector from the synthetic tracks
    feats = {u: ht.features for u, ht in h.tracks.items()}
    df["ym"] = df["date"].map(lambda d: (d.year, d.month))
    for b in got["buckets"]:
        y, m = int(b["bucket"][:4]), int(b["bucket"][5:7])
        sub = df[df["ym"] == (y, m)]
        # synthetic: every play has features -> coverage 1.0
        assert b["coverage"] == 1.0
        assert b["plays"] == len(sub)
        stacked = np.vstack([feats[u] for u in sub["track_uri"]])
        for j, axis in enumerate(SCALAR_AXES):
            assert b["means"][axis] == round(float(stacked[:, j].mean()), 4)
    assert got["axes"] == list(SCALAR_AXES)
    st.close()


def test_axes_coverage_partial_without_features():
    """A history with features on only some tracks reports honest <1 coverage."""
    h = make_synthetic_history(seed=5, n_days=60)
    # strip features from half the tracks
    uris = list(h.tracks)
    for u in uris[::2]:
        h.tracks[u].features = None
    st = HistoryStore.from_history(h)
    got = q.axes_over_time(st, "month")
    df = _events_df(h)
    with_feat = {u for u in h.tracks if h.tracks[u].features is not None}
    df["ym"] = df["date"].map(lambda d: (d.year, d.month))
    for b in got["buckets"]:
        y, m = int(b["bucket"][:4]), int(b["bucket"][5:7])
        sub = df[df["ym"] == (y, m)]
        exp_cov = sub["track_uri"].isin(with_feat).mean()
        assert b["coverage"] == round(float(exp_cov), 4)
    assert any(b["coverage"] < 1.0 for b in got["buckets"])
    st.close()


# ---------------------------------------------------------------------------
# genre mix
# ---------------------------------------------------------------------------
def test_genre_mix_matches_pandas():
    h, st = _store()
    got = q.top_flavors_or_genres(st)
    df = _events_df(h)
    feats = {u: ht.features for u, ht in h.tracks.items()}
    stacked = np.vstack([feats[u] for u in df["track_uri"]])
    genre_start = len(SCALAR_AXES)
    means = stacked[:, genre_start:genre_start + len(GENRES)].mean(axis=0)
    expected = {g: round(float(means[i]), 4) for i, g in enumerate(GENRES)}
    got_mix = {m["genre"]: m["mean"] for m in got["mix"]}
    assert got_mix == expected
    assert got["coverage"] == 1.0
    assert got["plays_with_features"] == len(df) == got["total_plays"]
    # sorted descending by mean
    vals = [m["mean"] for m in got["mix"]]
    assert vals == sorted(vals, reverse=True)
    # column names line up with feature_column mapping
    assert feature_column("genre:rap") == "genre_rap"
    st.close()


# ---------------------------------------------------------------------------
# CSV round-trip
# ---------------------------------------------------------------------------
def test_top_items_csv_roundtrip():
    _h, st = _store()
    payload = q.top_items(st, "tracks", by="plays", limit=8)
    text = q.top_items_to_csv(payload)
    parsed = list(csv.DictReader(io.StringIO(text)))
    assert len(parsed) == len(payload["rows"])
    assert list(parsed[0].keys()) == ["rank", "name", "artist", "plays", "minutes", "share"]
    for got_row, src in zip(parsed, payload["rows"]):
        assert int(got_row["rank"]) == src["rank"]
        assert got_row["name"] == src["name"]
        assert got_row["artist"] == src["artist"]
        assert int(got_row["plays"]) == src["plays"]
        assert float(got_row["share"]) == src["share"]
    st.close()


def test_top_artists_csv_has_no_artist_column():
    _h, st = _store()
    payload = q.top_items(st, "artists", limit=4)
    parsed = list(csv.DictReader(io.StringIO(q.top_items_to_csv(payload))))
    assert "artist" not in parsed[0]
    assert list(parsed[0].keys()) == ["rank", "name", "plays", "minutes", "share"]
    st.close()


def test_trends_csv_roundtrip():
    _h, st = _store()
    payload = q.trends(st, "minutes", "week", rolling=3)
    text = q.trends_to_csv(payload)
    parsed = list(csv.DictReader(io.StringIO(text)))
    assert list(parsed[0].keys()) == ["bucket", "value", "rolling"]
    assert len(parsed) == len(payload["buckets"])
    for got_row, src in zip(parsed, payload["buckets"]):
        assert got_row["bucket"] == src["bucket"]
        assert float(got_row["value"]) == src["value"]
    # no rolling column when rolling is off
    plain = q.trends(st, "minutes", "week")
    parsed_plain = list(csv.DictReader(io.StringIO(q.trends_to_csv(plain))))
    assert list(parsed_plain[0].keys()) == ["bucket", "value"]
    st.close()
