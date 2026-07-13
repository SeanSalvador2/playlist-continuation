"""Synthetic listening-history generator: determinism + planted-truth self-checks.

The point of these tests is not just "it runs" but "the planted ground truth is really
there and really detectable" — so a Phase-3 change-point detector can be scored on it.
"""
import hashlib
import os
import subprocess
import sys
from collections import Counter

import numpy as np
import pytest

from playlistcont.data.schema import SCALAR_AXES
from playlistcont.data.synthetic import ARCHETYPES
from playlistcont.history.synthetic import make_listener_population, make_synthetic_history

_PROFILE = {a.name: np.array([a.scalars[ax] for ax in SCALAR_AXES]) for a in ARCHETYPES}


@pytest.fixture(scope="module")
def hist():
    return make_synthetic_history(seed=0)


def _event_key(e):
    return (e.ts, e.track_uri, e.ms_played, e.skipped)


def test_same_seed_identical_stream():
    a = make_synthetic_history(seed=3)
    b = make_synthetic_history(seed=3)
    assert a.n_events == b.n_events
    assert [_event_key(e) for e in a.events] == [_event_key(e) for e in b.events]


def test_different_seed_differs():
    a = make_synthetic_history(seed=3)
    b = make_synthetic_history(seed=4)
    assert [_event_key(e) for e in a.events] != [_event_key(e) for e in b.events]


_DIGEST_SNIPPET = """
import hashlib
from playlistcont.history.synthetic import make_synthetic_history
h = make_synthetic_history(seed=7, n_days=120)
blob = "|".join(
    f"{e.ts.isoformat()},{e.track_uri},{e.ms_played},{e.skipped}" for e in h.events
)
print(hashlib.sha256(blob.encode()).hexdigest())
"""


def test_cross_process_determinism_under_hash_randomisation():
    """Same seed must give an identical event stream in *separate processes*.

    Set iteration order depends on PYTHONHASHSEED, which is fixed per interpreter —
    so an in-process same-seed comparison (test_same_seed_identical_stream above)
    can NEVER catch hash-order nondeterminism: both generations see the same hash
    seed and agree with each other while differing across processes.  This test
    therefore spawns subprocesses with different PYTHONHASHSEED values and asserts
    the sha256 digest of the (ts, uri, ms_played, skipped) stream is identical.

    Regression test for a real bug: ``set(a) | set(b)`` in ``_mixture_on`` built the
    drift-blend mixture dict in hash order, reordering the probabilities consumed by
    ``rng.choice`` in the event loop.  Hash seeds 1 and 3 demonstrably produced
    different streams pre-fix (1 and 2 happened to collide, so a single pair is not
    enough — we compare three).
    """
    digests = []
    for hash_seed in ("1", "2", "3"):
        env = dict(os.environ, PYTHONHASHSEED=hash_seed)
        out = subprocess.run(
            [sys.executable, "-c", _DIGEST_SNIPPET],
            env=env, capture_output=True, text=True, timeout=120,
        )
        assert out.returncode == 0, out.stderr
        digests.append(out.stdout.strip())
    assert digests[0] == digests[1] == digests[2], (
        f"event stream is not a pure function of seed: digests {digests}"
    )


def test_events_sorted_and_have_features(hist):
    assert hist.is_sorted()
    # every played track carries an interpretable axis vector
    assert all(hist.tracks[e.track_uri].features is not None for e in hist.events)


def test_regimes_partition_timeline(hist):
    gt = hist.ground_truth
    regimes = gt.regimes
    assert 3 <= len(regimes) <= 5
    start = regimes[0].start
    end = regimes[-1].end
    # contiguous, gapless partition
    for i in range(len(regimes) - 1):
        assert regimes[i].end == regimes[i + 1].start
    # every planted change and drift landmark lies inside the timeline
    for c in gt.change_points:
        assert start <= c.date <= end


def test_pure_regime_signal_is_detectable(hist):
    """During a pure single-archetype regime the mean axis vector is closest to
    that archetype — proof the planted taste is actually present in the plays."""
    gt = hist.ground_truth
    pure = next(r for r in gt.regimes if len(r.mixture) == 1)
    name = next(iter(pure.mixture))
    trap_spans = [(t.start, t.end) for t in gt.traps]
    vecs = []
    for e in hist.events:
        d = e.ts.date()
        if pure.start <= d < pure.end and not any(s <= d < en for s, en in trap_spans):
            vecs.append(hist.tracks[e.track_uri].features[: len(SCALAR_AXES)])
    assert len(vecs) > 100
    mean = np.mean(vecs, axis=0)
    dists = {n: np.linalg.norm(mean - p) for n, p in _PROFILE.items()}
    assert min(dists, key=dists.get) == name


def _rate_per_day(events, start, end):
    days = max(1, (end - start).days)
    n = sum(1 for e in events if start <= e.ts.date() < end)
    return n / days


def test_volume_only_trap_is_volume_not_taste(hist):
    gt = hist.ground_truth
    trap = next(t for t in gt.traps if t.kind == "volume_only")
    regime = next(r for r in gt.regimes if r.start <= trap.start < r.end)

    def mean_axis(start, end):
        v = [hist.tracks[e.track_uri].features[: len(SCALAR_AXES)]
             for e in hist.events if start <= e.ts.date() < end]
        return np.mean(v, axis=0)

    inside = _rate_per_day(hist.events, trap.start, trap.end)
    before = _rate_per_day(hist.events, regime.start, trap.start)
    # volume shifts markedly...
    assert inside > 2.0 * before
    # ...while the taste (mean axis) is essentially unchanged
    drift = np.linalg.norm(mean_axis(trap.start, trap.end) - mean_axis(regime.start, trap.start))
    assert drift < 0.05


def test_binge_trap_has_one_dominant_album(hist):
    gt = hist.ground_truth
    trap = next(t for t in gt.traps if t.kind == "binge")
    albums = Counter(
        hist.tracks[e.track_uri].album_name
        for e in hist.events if trap.start <= e.ts.date() < trap.end
    )
    assert sum(albums.values()) > 0
    _, top = albums.most_common(1)[0]
    assert top / sum(albums.values()) > 0.5


def test_seasonal_annotated_but_not_a_change_point(hist):
    gt = hist.ground_truth
    assert gt.seasonal_spans, "expected an annotated December bump"
    change_dates = {c.date for c in gt.change_points}
    for s in gt.seasonal_spans:
        assert s.start not in change_dates
        assert s.end not in change_dates


def test_population_is_deterministic_and_varied():
    pop = make_listener_population(3, seed=10, n_days=120)
    again = make_listener_population(3, seed=10, n_days=120)
    assert len(pop) == 3
    # reproducible
    for h1, h2 in zip(pop, again):
        assert [_event_key(e) for e in h1.events] == [_event_key(e) for e in h2.events]
    # and distinct listeners
    assert [_event_key(e) for e in pop[0].events] != [_event_key(e) for e in pop[1].events]
