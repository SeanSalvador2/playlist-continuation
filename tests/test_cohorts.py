"""Tests for cohort dossiers — adaptive lift casts, tiered names, boundary diffs.

Deterministic and offline.  We check the adaptive cast bounds, the tiered-naming
rules (including the >50% single-artist monolith rule and graceful degradation with
no cluster map), and that boundary diffs surface the arrivals/departures of a planted
genre swap.
"""
from datetime import date, datetime, timedelta, timezone

import numpy as np
import pytest

from playlistcont.dynamics.cohorts import (
    CohortModel, adaptive_cast_size, boundary_diff, build_cohort, cohorts_for_spans,
)
from playlistcont.history.schema import HistoryTrack, ListenEvent, ListeningHistory, RegimeSpec
from playlistcont.history.synthetic import make_synthetic_history

UTC = timezone.utc


# ---------------------------------------------------------------------------
# fixtures
# ---------------------------------------------------------------------------
def _monolith_history() -> ListeningHistory:
    """One artist owns ~70% of plays across a 100-day span (plus a few others)."""
    events = []
    tracks = {}
    base = date(2022, 1, 1)
    # dominant artist: 700 plays of one track
    tracks["spotify:track:solo"] = HistoryTrack(
        track_uri="spotify:track:solo", track_name="Solo Anthem", artist_name="Solo Star")
    for i in range(700):
        ts = datetime(base.year, base.month, base.day, 12, tzinfo=UTC) + timedelta(hours=i)
        events.append(ListenEvent(ts=ts, track_uri="spotify:track:solo",
                                  track_name="Solo Anthem", artist_name="Solo Star",
                                  ms_played=180000))
    # a handful of others so there is a cast to compare against
    for a in range(6):
        uri = f"spotify:track:other{a}"
        tracks[uri] = HistoryTrack(track_uri=uri, track_name=f"Other {a}",
                                   artist_name=f"Backup Band {a}")
        for i in range(50):
            ts = datetime(base.year, base.month, base.day, 6, tzinfo=UTC) + timedelta(hours=a * 100 + i)
            events.append(ListenEvent(ts=ts, track_uri=uri, track_name=f"Other {a}",
                                      artist_name=f"Backup Band {a}", ms_played=180000))
    events.sort(key=lambda e: e.ts)
    return ListeningHistory(events=events, tracks=tracks, provenance="test", ground_truth=None)


def _two_regime_history(seed: int = 1) -> ListeningHistory:
    regimes = [
        RegimeSpec(date(2022, 1, 1), date(2022, 4, 1), {"classic rock": 1.0}, 40, "r0"),
        RegimeSpec(date(2022, 4, 1), date(2022, 7, 1), {"gym rap": 1.0}, 40, "r1"),
    ]
    return make_synthetic_history(seed=seed, regimes=regimes, n_days=181,
                                  start_date=date(2022, 1, 1), include_traps=False,
                                  seasonal=False)


# ---------------------------------------------------------------------------
# adaptive cast size
# ---------------------------------------------------------------------------
def test_adaptive_cast_size_edges():
    assert adaptive_cast_size([], 4, 20) == 0
    assert adaptive_cast_size([5, 4], 4, 20) == 2         # shorter than lo -> all
    # a clear cliff beyond the upper bound is cut inside [lo, hi]
    lifts = [10.0] * 5 + [1.01, 1.0, 1.0, 1.0]
    n = adaptive_cast_size(lifts, 4, 8)
    assert 4 <= n <= 8


def test_cast_sizes_stay_within_bounds():
    model = CohortModel.from_store(make_synthetic_history(seed=3, n_days=365))
    cohort = build_cohort(model, date(2022, 1, 1), date(2022, 12, 31))
    # the lift cast is bounded [4, 20] unless fewer artists qualify at all
    assert len(cohort.cast) <= 20
    if cohort.n_cast_qualifying >= 4:
        assert len(cohort.cast) >= 4
    else:
        assert len(cohort.cast) == cohort.n_cast_qualifying
    # every lift-cast entry genuinely clears the materiality floor
    for c in cohort.cast:
        assert c["lift"] >= 1.5 and c["plays"] >= 15


# ---------------------------------------------------------------------------
# naming rules
# ---------------------------------------------------------------------------
def test_monolith_rule_names_the_dominant_artist():
    model = CohortModel.from_store(_monolith_history())
    cohort = build_cohort(model, date(2022, 1, 1), date(2022, 4, 30))
    assert cohort.cast_raw[0]["share"] > 0.5
    assert cohort.name_short == "The Solo Star monolith"
    assert "monolith" in cohort.name_expanded
    assert "over half" in cohort.name_expanded


def test_names_degrade_without_clusters():
    model = CohortModel.from_store(make_synthetic_history(seed=3, n_days=365))
    assert model.cluster is None
    cohort = build_cohort(model, date(2022, 1, 1), date(2022, 6, 30))
    # cluster mix is omitted, but every name tier is still produced
    assert cohort.cluster_mix is None
    assert cohort.name_short and cohort.name_expanded and cohort.explanation
    # the tiered length contracts hold
    assert len(cohort.name_short.split()) <= 4
    assert len(cohort.name_expanded.split()) <= 14
    assert 2 <= cohort.explanation.count(".") <= 4


def test_cluster_mix_present_when_clusters_supplied():
    hist = make_synthetic_history(seed=3, n_days=365)
    model_plain = CohortModel.from_store(hist)
    # assign every play's track to a toy cluster deterministically (no PYTHONHASHSEED)
    clusters = {u: (i % 3) for i, u in enumerate(sorted(set(model_plain.uri.tolist())))}
    model = CohortModel.from_store(
        hist, clusters=clusters,
        cluster_names={0: "cluster A", 1: "cluster B", 2: "cluster C"},
        cluster_genres={0: "rock", 1: "rap", 2: "pop"})
    cohort = build_cohort(model, date(2022, 1, 1), date(2022, 6, 30))
    assert cohort.cluster_mix is not None
    assert "by_cluster" in cohort.cluster_mix and "unmapped" in cohort.cluster_mix
    total = sum(cohort.cluster_mix["by_cluster"].values()) + cohort.cluster_mix["unmapped"]
    assert abs(total - 1.0) < 5e-3          # per-cluster shares are rounded to 4 dp


# ---------------------------------------------------------------------------
# boundary diff
# ---------------------------------------------------------------------------
def test_boundary_diff_surfaces_the_genre_swap():
    model = CohortModel.from_store(_two_regime_history())
    prev_span = (date(2022, 1, 1), date(2022, 3, 31))
    next_span = (date(2022, 4, 1), date(2022, 6, 30))
    diff = boundary_diff(model, prev_span, next_span, date(2022, 4, 1))
    # rock artists leave, rap artists arrive across the planted rock -> rap turn
    arriving = " ".join(a["artist"] for a in diff["arrivals"]).lower()
    departing = " ".join(d["artist"] for d in diff["departures"]).lower()
    assert diff["arrivals"] and diff["departures"]
    assert "rap" in arriving
    assert "rock" in departing


def test_cohorts_for_spans_chains_shift_context():
    model = CohortModel.from_store(_two_regime_history())
    spans = [(date(2022, 1, 1), date(2022, 3, 31)), (date(2022, 4, 1), date(2022, 6, 30))]
    cohorts = cohorts_for_spans(model, spans)
    assert len(cohorts) == 2
    # the first span is the opening one; the second carries a shift-vs-prior phrase
    assert "Opening segment" in cohorts[0].explanation
    assert cohorts[1].shift_swing or cohorts[1].shift_axes


def test_determinism():
    model = CohortModel.from_store(make_synthetic_history(seed=3, n_days=365))
    a = build_cohort(model, date(2022, 1, 1), date(2022, 6, 30))
    b = build_cohort(model, date(2022, 1, 1), date(2022, 6, 30))
    assert a.to_payload() == b.to_payload()
