"""Tests for adaptive chapter extraction with floor guarantees and honest labels.

Same philosophy as the rest of the dynamics suite: deterministic, offline, every
structural claim checked.  We exercise three listener shapes — a normal multi-regime
history (real turns -> validated labels), a genuinely stationary single-regime history
(forced time-split + weak verdict), and a drift-heavy history — and assert the two
hard invariants the product contract depends on: ``min_chapters`` is ALWAYS satisfied,
and every boundary carries an honest confidence label.
"""
from datetime import date

import pytest

from playlistcont.dynamics import build_windows
from playlistcont.dynamics.chapters import ChapterPlan, adaptive_chapters
from playlistcont.dynamics.chapters import _confidence_for_scale, _time_split_indices
from playlistcont.dynamics.cohorts import adaptive_cast_size
from playlistcont.history.schema import RegimeSpec
from playlistcont.history.synthetic import make_synthetic_history

_LABELS = {"validated", "exploratory", "forced"}


@pytest.fixture(scope="module")
def default_series():
    hist = make_synthetic_history(seed=3, n_days=730, base_events_per_day=40)
    return build_windows(hist, granularity="week", min_events=30, weighting="plays")


@pytest.fixture(scope="module")
def stationary_series():
    # ONE regime for the whole span, no traps, no seasonality: nothing to detect.
    regimes = [RegimeSpec(date(2022, 1, 1), date(2024, 1, 1), {"classic rock": 1.0}, 40, "r0")]
    hist = make_synthetic_history(seed=5, regimes=regimes, n_days=730,
                                  start_date=date(2022, 1, 1), include_traps=False,
                                  seasonal=False)
    return build_windows(hist, granularity="week", min_events=30, weighting="plays")


# ---------------------------------------------------------------------------
# the floor guarantee + honest labels
# ---------------------------------------------------------------------------
def test_default_history_finds_validated_chapters(default_series):
    plan = adaptive_chapters(default_series, min_chapters=2)
    assert isinstance(plan, ChapterPlan)
    assert len(plan.chapters) >= 2                       # floor satisfied
    assert len(plan.boundaries) == len(plan.chapters) - 1
    # a normal history's turns are found at the recommended penalty, no relaxation
    assert plan.fallback is None
    assert not plan.relaxation_used
    assert all(b.confidence == "validated" for b in plan.boundaries)
    assert plan.structure_strength in ("strong", "moderate")
    # every boundary carries a positive shift and a real most-moved column
    cols = default_series.representation_columns("combined")
    for b in plan.boundaries:
        assert b.confidence in _LABELS
        assert b.shift > 0.0
        assert b.shift_col in cols


def test_stationary_history_forced_split_and_weak(stationary_series):
    plan = adaptive_chapters(stationary_series, min_chapters=2)
    # the floor is STILL satisfied on a listener with no detectable change ...
    assert len(plan.chapters) >= 2
    # ... but honesty is preserved: every seam is forced and the verdict is weak
    assert all(b.confidence == "forced" for b in plan.boundaries)
    assert plan.structure_strength.startswith("weak")
    assert "unusually stable" in plan.structure_strength
    # a truly stationary listener trips the documented time-split fallback
    assert plan.fallback is not None
    assert "split by time" in plan.fallback


def test_min_chapters_floor_is_respected_for_larger_floor(default_series):
    for mc in (2, 3, 4):
        plan = adaptive_chapters(default_series, min_chapters=mc)
        assert len(plan.chapters) >= mc


def test_drift_heavy_history_labels_are_honest():
    # several regimes with modest genre moves -> a churny, drift-heavy timeline
    regimes = [
        RegimeSpec(date(2022, 1, 1), date(2022, 5, 1),
                   {"indie chill": 0.7, "coffeehouse folk": 0.3}, 40, "a"),
        RegimeSpec(date(2022, 5, 1), date(2022, 9, 1),
                   {"indie chill": 0.4, "coffeehouse folk": 0.6}, 40, "b"),
        RegimeSpec(date(2022, 9, 1), date(2023, 1, 1),
                   {"coffeehouse folk": 0.5, "sad slow country": 0.5}, 40, "c"),
        RegimeSpec(date(2023, 1, 1), date(2023, 6, 1),
                   {"sad slow country": 0.8, "party pop": 0.2}, 40, "d"),
    ]
    hist = make_synthetic_history(seed=11, regimes=regimes, n_days=516,
                                  start_date=date(2022, 1, 1), include_traps=False,
                                  seasonal=False)
    ws = build_windows(hist, granularity="week", min_events=30, weighting="plays")
    plan = adaptive_chapters(ws, min_chapters=2)
    assert len(plan.chapters) >= 2
    assert all(b.confidence in _LABELS for b in plan.boundaries)
    # the ladder trace is coarse -> fine with non-decreasing boundary counts
    scales = [lvl["scale"] for lvl in plan.levels]
    assert scales == sorted(scales, reverse=True)
    counts = [lvl["n_boundaries"] for lvl in plan.levels]
    assert all(counts[i] <= counts[i + 1] for i in range(len(counts) - 1))


# ---------------------------------------------------------------------------
# sub-sections
# ---------------------------------------------------------------------------
def test_every_chapter_has_at_least_one_subsection(default_series):
    plan = adaptive_chapters(default_series)
    for ch in plan.chapters:
        assert ch.subsections                            # always >= 1 (the chapter itself)
        # sub-sections tile the chapter contiguously
        assert ch.subsections[0].i0 == ch.i0
        assert ch.subsections[-1].i1 == ch.i1
        for a, b in zip(ch.subsections, ch.subsections[1:]):
            assert a.i1 == b.i0
        assert ch.n_turns == len(ch.subsections) - 1


def test_subsection_min_size_respected(default_series):
    plan = adaptive_chapters(default_series, min_subsection_size=8)
    for ch in plan.chapters:
        if len(ch.subsections) > 1:                      # only when a real split happened
            for s in ch.subsections:
                assert (s.i1 - s.i0) >= 8


def test_headline_is_computed_from_structure(default_series):
    plan = adaptive_chapters(default_series)
    for ch in plan.chapters:
        h = ch.headline()
        if ch.n_turns == 0:
            assert h == "a slow drift — no sharp turn"
        elif ch.n_turns == 1:
            assert h == "one clear turn"
        else:
            assert h.endswith("turns")


# ---------------------------------------------------------------------------
# determinism + unit helpers
# ---------------------------------------------------------------------------
def test_determinism(default_series):
    a = adaptive_chapters(default_series)
    b = adaptive_chapters(default_series)
    assert a.to_payload() == b.to_payload()


def test_confidence_label_ladder():
    assert _confidence_for_scale(1.5) == "validated"
    assert _confidence_for_scale(2.0) == "validated"
    assert _confidence_for_scale(1.0) == "exploratory"
    assert _confidence_for_scale(0.7) == "exploratory"
    assert _confidence_for_scale(0.5) == "forced"
    assert _confidence_for_scale(0.35) == "forced"
    assert _confidence_for_scale(None) == "forced"


def test_time_split_indices_are_equal_and_interior():
    assert _time_split_indices(100, 2) == [50]
    assert _time_split_indices(90, 3) == [30, 60]
    # never emits 0 or n, and never more than parts-1 cuts
    cuts = _time_split_indices(10, 4)
    assert all(0 < c < 10 for c in cuts)
    assert len(cuts) <= 3


def test_adaptive_cast_size_bounds_and_elbow():
    # fewer than lo -> keep them all
    assert adaptive_cast_size([9, 8, 7], 4, 20) == 3
    # count within [lo, hi] -> keep all (a cut at the very end always wins)
    assert adaptive_cast_size([100, 90, 80, 70, 10, 9], 4, 20) == 6
    # count exceeds hi -> cut at the largest relative gap, within [lo, hi]
    assert adaptive_cast_size([100, 90, 80, 70, 10, 9, 8, 7, 6], 4, 6) == 4
    # the big 100->5 drop is below lo=4 so it is ignored; within [4, 6] the best
    # relative gap here is at n=5 (2/1), so the lower bound is honoured
    assert adaptive_cast_size([100, 5, 4, 3, 2, 1, 1, 1], 4, 6) == 5
