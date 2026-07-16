"""Tests for hierarchical (recursive) PELT segmentation.

Philosophy matches ``tests/test_dynamics.py``: deterministic, offline, every
structural claim checked. We use a synthetic history with planted regime changes
so the coarse level of the tree must recover the big turns, and we assert the
tree/ladder invariants that the UI and the Wave-1 analysis depend on.
"""
from datetime import date

import numpy as np
import pytest

from playlistcont.dynamics import (
    build_windows,
    detail_ladder,
    hierarchical_segments,
)
from playlistcont.dynamics.hierarchy import SegmentNode
from playlistcont.history.synthetic import make_synthetic_history


@pytest.fixture(scope="module")
def series():
    # a 2-year synthetic history has planted regime changes; weekly windows are
    # what the recommended detector (and the Wave-1 hierarchy) run on.
    hist = make_synthetic_history(seed=3, n_days=730, base_events_per_day=40)
    return build_windows(hist, granularity="week", min_events=30, weighting="plays")


def test_root_spans_whole_valid_history(series):
    root = hierarchical_segments(series)
    assert root.penalty_level is None
    assert root.depth == 0
    assert root.i0 == 0
    assert root.i1 == int(series.mask.sum())      # all valid windows
    assert root.opening_shift == 0.0


def test_children_partition_parent_contiguously(series):
    root = hierarchical_segments(series)
    for node in root.walk():
        if not node.children:
            continue
        # children tile [i0, i1) with no gaps or overlaps, in order
        assert node.children[0].i0 == node.i0
        assert node.children[-1].i1 == node.i1
        for a, b in zip(node.children, node.children[1:]):
            assert a.i1 == b.i0
            assert a.i1 > a.i0                     # non-empty


def test_leaves_tile_the_whole_series(series):
    root = hierarchical_segments(series)
    leaves = root.leaves()
    assert leaves[0].i0 == 0
    assert leaves[-1].i1 == root.i1
    for a, b in zip(leaves, leaves[1:]):
        assert a.i1 == b.i0


def test_min_size_respected(series):
    # no leaf shorter than min_size except possibly the trailing remainder is fine,
    # but every *split* must leave both sides >= min_size (ruptures guarantee).
    min_size = 8
    root = hierarchical_segments(series, min_size=min_size)
    for node in root.walk():
        if node.children:
            for c in node.children:
                assert c.n_windows >= min_size


def test_penalty_level_decreases_with_depth(series):
    # a child is carved at a scale <= the scale that carved its parent (ladder is
    # applied coarse -> fine, and finer splits only happen deeper).
    root = hierarchical_segments(series, scales=(1.5, 1.0, 0.7, 0.5))
    for node in root.walk():
        for c in node.children:
            if node.penalty_level is not None and c.penalty_level is not None:
                assert c.penalty_level <= node.penalty_level


def test_coarse_level_finds_planted_change(series):
    # the top split(s) should land near a planted regime boundary. We don't know
    # the exact planted dates here, but the coarse tree must be non-trivial on a
    # multi-regime history and every opening boundary must carry a positive shift.
    root = hierarchical_segments(series, scales=(1.5,))
    assert len(root.children) >= 2                 # at least one coarse split
    for c in root.children[1:]:
        assert c.opening_shift > 0.0
        assert c.opening_shift_col in series.representation_columns("combined")


def test_detail_ladder_monotone_and_dated(series):
    scales = [2.0, 1.5, 1.0, 0.7, 0.5, 0.35]
    ladder = detail_ladder(series, scales)
    assert [r["scale"] for r in ladder] == scales
    # penalty is proportional to scale, so strictly decreasing as scale drops
    pens = [r["penalty"] for r in ladder]
    assert pens == sorted(pens, reverse=True)
    # lower penalty -> at least as many boundaries (PELT monotonicity)
    counts = [r["n_boundaries"] for r in ladder]
    assert all(counts[i] <= counts[i + 1] for i in range(len(counts) - 1))
    for r in ladder:
        for b in r["boundaries"]:
            date.fromisoformat(b)                  # parseable ISO dates


def test_determinism(series):
    a = hierarchical_segments(series)
    b = hierarchical_segments(series)
    assert [(n.i0, n.i1, n.penalty_level) for n in a.walk()] == \
           [(n.i0, n.i1, n.penalty_level) for n in b.walk()]


def test_short_series_is_a_bare_root():
    # fewer than 2*min_size valid windows -> no splits, just the root leaf.
    hist = make_synthetic_history(seed=1, n_days=60, base_events_per_day=40)
    ws = build_windows(hist, granularity="week", min_events=30, weighting="plays")
    root = hierarchical_segments(ws, min_size=8)
    assert isinstance(root, SegmentNode)
    assert root.children == []
    assert root.leaves() == [root]
