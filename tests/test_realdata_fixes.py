"""Regression tests for real-data-only breakage surfaced by a 6-year Spotify history.

Three independent fixes, each checked against a hand-computed expectation on a small
fabricated history (never the owner's real data):

* **Skip-reliability window** — Spotify back-fills ``skipped=false`` before it began
  logging skips, faking a 0% rate that then jumps.  ``skip_reliable_from`` must find the
  first month where real skips appear; ``summary``/``trends``/``stats`` must report the
  pre-logging period as *unmeasured* (null / insufficient), never as a real 0%.
* **Display timezone** — the stored ``hour``/``weekday`` are UTC; the listening clock and
  hour-band habits must convert to a configurable IANA zone (DST-correct, per-timestamp).
* **Year granularity** — a multi-year history must bucket by calendar year on trends/axes.
"""
from datetime import date, datetime, timezone

import numpy as np

from playlistcont.analytics import queries as q
from playlistcont.analytics import stats
from playlistcont.data.schema import N_AXES, SCALAR_AXES
from playlistcont.history.schema import HistoryTrack, ListenEvent, ListeningHistory
from playlistcont.history.store import HistoryStore

UTC = timezone.utc


def _vec(scalars, genre_idx=0):
    v = np.zeros(N_AXES, dtype=np.float32)
    v[: len(SCALAR_AXES)] = scalars
    v[len(SCALAR_AXES) + genre_idx] = 1.0
    return v


def _store(events, with_features=True):
    """Build a HistoryStore from (datetime, skipped) rows; every play shares one track."""
    uri = "spotify:track:fixture0000000000000000"
    track = HistoryTrack(
        track_uri=uri, track_name="Fix", artist_name="Artist",
        album_name="Album", features=_vec([0.5] * 5) if with_features else None,
    )
    evs = []
    for ts, skipped in events:
        evs.append(ListenEvent(
            ts=ts, track_uri=uri, track_name="Fix", artist_name="Artist",
            ms_played=200_000, album_name="Album", skipped=skipped, platform="web"))
    evs.sort(key=lambda e: e.ts)
    h = ListeningHistory(events=evs, tracks={uri: track}, provenance="synthetic")
    return h, HistoryStore.from_history(h)


def _month_rows(year, month, n, skip_fraction):
    """n plays on the 15th at noon UTC; the first ``skip_fraction`` share are skips."""
    n_skip = round(n * skip_fraction)
    rows = []
    for i in range(n):
        ts = datetime(year, month, 15, 12, 0, 0, tzinfo=UTC)
        rows.append((ts, i < n_skip))
    return rows


# ===========================================================================
# Bug 1 — skip-reliable window detection
# ===========================================================================
def _skip_history():
    """Three pre-logging months (skipped=false, 0% real skips) then three at ~30%."""
    rows = []
    for m in (1, 2, 3):
        rows.extend(_month_rows(2021, m, 100, 0.0))     # pre-logging: all false
    for m in (4, 5, 6):
        rows.extend(_month_rows(2021, m, 100, 0.30))    # logging on: ~30% skips
    return _store(rows)


def test_skip_reliable_from_detects_first_logging_month():
    _h, st = _skip_history()
    assert q.skip_reliable_from(st) == date(2021, 4, 1)
    st.close()


def test_skip_reliable_from_none_when_never_logged():
    # A history with a non-null flag that is uniformly false -> no reliable window.
    rows = _month_rows(2021, 1, 50, 0.0) + _month_rows(2021, 2, 50, 0.0)
    _h, st = _store(rows)
    assert q.skip_reliable_from(st) is None
    st.close()


def test_trends_skip_rate_null_before_logging_real_after():
    _h, st = _skip_history()
    got = q.trends(st, "skip_rate", "month")
    assert got["skip_reliable_from"] == "2021-04-01"
    by_bucket = {b["bucket"]: b["value"] for b in got["buckets"]}
    # pre-logging buckets: null, never a fake 0.0
    for m in ("2021-01-01", "2021-02-01", "2021-03-01"):
        assert by_bucket[m] is None
    # post-logging buckets: the real ~30% rate
    for m in ("2021-04-01", "2021-05-01", "2021-06-01"):
        assert by_bucket[m] == 0.3
    st.close()


def test_summary_skip_rate_only_over_reliable_window():
    _h, st = _skip_history()
    got = q.summary(st)
    # 600 plays total, 300 in the reliable window (Apr-Jun), 90 of them skips
    assert got["skip_reliable_from"] == "2021-04-01"
    assert got["skip_rate"] == 0.3                      # 90 / 300, NOT 90 / 600
    assert got["skip_coverage"] == round(300 / 600, 4)  # half the plays are reliable
    assert got["skip_reason"] is None
    st.close()


def test_summary_skip_rate_null_when_never_logged():
    rows = _month_rows(2021, 1, 50, 0.0)
    _h, st = _store(rows)
    got = q.summary(st)
    assert got["skip_reliable_from"] is None
    assert got["skip_rate"] is None
    assert got["skip_coverage"] == 0.0
    assert got["skip_reason"] and "no skips" in got["skip_reason"].lower()
    st.close()


def test_stats_skip_metric_not_recorded_before_logging():
    """A window entirely before skip logging reports skip_rate insufficient, not 0%."""
    _h, st = _skip_history()
    # window A = Jan-Feb (pre-logging), window B = May-Jun (logging on)
    comp = stats.compare_windows(
        st, "2021-01-01", "2021-02-28", "2021-05-01", "2021-06-30")
    skip = next(r for r in comp["metrics"] if r["metric"] == "skip_rate")
    # pre-logging window A has zero skip-flagged plays -> insufficient, means null
    assert skip["insufficient"] is True
    assert skip["n_a"] == 0 and skip["mean_a"] is None
    # window B did carry skips (its reliable count is non-zero even if < MIN_N here)
    assert skip["n_b"] == 200
    st.close()


# ===========================================================================
# Bug 2 — display-timezone conversion for the listening clock / habits
# ===========================================================================
def test_listening_clock_converts_to_western_tz():
    # 2021-01-04 02:30 UTC is a Monday; in America/New_York (EST, -5) it is
    # 2021-01-03 21:30 -> Sunday, hour 21.  Hand-checked.
    ts = datetime(2021, 1, 4, 2, 30, 0, tzinfo=UTC)
    _h, st = _store([(ts, False)])

    utc = q.listening_clock(st, tz="UTC")
    assert utc["tz"] == "UTC"
    assert utc["matrix"][0][2] == 1                 # Monday (0), hour 2 UTC

    ny = q.listening_clock(st, tz="America/New_York")
    assert ny["tz"] == "America/New_York"
    assert ny["matrix"][6][21] == 1                 # Sunday (6), hour 21 local
    assert ny["matrix"][0][2] == 0                  # no longer the UTC cell
    assert ny["total"] == 1
    st.close()


def test_listening_clock_dst_offset_differs_summer_vs_winter():
    # Same 02:30 UTC wall time in winter (EST -5 -> 21:30 prev day) and summer
    # (EDT -4 -> 22:30 prev day): the per-timestamp conversion must differ by an hour.
    winter = datetime(2021, 1, 4, 2, 30, tzinfo=UTC)   # -> 21:00 hour bucket
    summer = datetime(2021, 7, 5, 2, 30, tzinfo=UTC)   # -> 22:00 hour bucket
    _h, st = _store([(winter, False), (summer, False)])
    ny = q.listening_clock(st, tz="America/New_York")
    hours = [h for w in range(7) for h in range(24) if ny["matrix"][w][h]]
    assert sorted(hours) == [21, 22]
    st.close()


def test_resolve_tz_falls_back_to_utc_on_bad_zone():
    assert q.resolve_tz("Not/AZone") == "UTC"
    assert q.resolve_tz("America/New_York") == "America/New_York"
    assert q.resolve_tz(None) in ("UTC",) or isinstance(q.resolve_tz(None), str)


def test_habits_hour_band_uses_local_time():
    # 30 plays at 02:30 UTC in winter -> local 21:30 (evening) in New York, but
    # "night" (>=22 or <5) in UTC.  The habit grouping must bucket them by local hour.
    ts = datetime(2021, 1, 4, 2, 30, tzinfo=UTC)
    rows = [(ts, False)] * 30
    _h, st = _store(rows)
    # We can observe the grouping indirectly: the local-hour band puts all plays in one
    # group.  Group labels are internal, so check via the tz-aware feature frame.
    feat_utc = stats._feature_frame(st, tz="UTC")
    feat_ny = stats._feature_frame(st, tz="America/New_York")
    assert set(int(h) for h in feat_utc["hour"]) == {2}
    assert set(int(h) for h in feat_ny["hour"]) == {21}
    st.close()


# ===========================================================================
# Bug 3 — year granularity on trends / axes
# ===========================================================================
def _multiyear_store():
    rows = []
    counts = {2021: 10, 2022: 25, 2023: 7}
    for year, n in counts.items():
        rows.extend(_month_rows(year, 6, n, 0.0))
    return counts, _store(rows)


def test_trends_year_bucket_counts():
    counts, (_h, st) = _multiyear_store()
    got = q.trends(st, "plays", "year")
    assert got["granularity"] == "year"
    by_bucket = {b["bucket"]: b["value"] for b in got["buckets"]}
    assert by_bucket == {
        "2021-01-01": float(counts[2021]),
        "2022-01-01": float(counts[2022]),
        "2023-01-01": float(counts[2023]),
    }
    st.close()


def test_axes_over_time_year_buckets():
    counts, (_h, st) = _multiyear_store()
    got = q.axes_over_time(st, "year")
    assert got["granularity"] == "year"
    assert [b["bucket"] for b in got["buckets"]] == [
        "2021-01-01", "2022-01-01", "2023-01-01"]
    # every play has features -> full coverage, and one calendar-year bucket each
    for b, year in zip(got["buckets"], (2021, 2022, 2023)):
        assert b["plays"] == counts[year]
        assert b["coverage"] == 1.0
    st.close()
