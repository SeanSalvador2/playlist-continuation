"""Recursive (hierarchical) PELT segmentation — sub-eras inside eras.

Phase 4's :mod:`~playlistcont.dynamics.eras` cuts the timeline **once**, at the
recommended detector's change points, into a flat list of eras.  This module
answers the natural follow-up — *what happened inside an era?* — by applying the
same PELT machinery **recursively** at a decreasing penalty ladder: split the
whole history coarsely, then re-run PELT inside each piece at a lower penalty to
find sub-structure, and so on.  The result is a **tree** of nested segments plus
a flat **detail ladder** (boundary dates as a function of a single global
penalty) that a UI detail-slider can scrub.

CONTRACT (reuses the detector standardisation exactly)
------------------------------------------------------
* The representation matrix is produced **once** by
  :func:`~playlistcont.dynamics.detectors.prepare_matrix` (same z-scoring /
  valid-window compaction / optional deseasonalisation as every matrix detector),
  so a penalty here means the same thing it means in :func:`detectors.pelt`.
* Each recursion level runs ``ruptures`` PELT (``rbf`` cost by default, matching
  :func:`~playlistcont.dynamics.detectors.recommended_detector`) on the **slice**
  ``Z[i0:i1]`` of that segment, with the penalty computed from the *segment's own*
  length: ``penalty = scale * default_penalty(n_segment, d, cost)``.  Standardising
  globally but penalising locally keeps a penalty scale comparable across levels
  while letting a short segment still be split.
* ``min_size`` (in windows) is PELT's minimum segment length AND the guard that
  stops the recursion: a segment shorter than ``2 * min_size`` cannot be split.
* When a level finds no split in a segment, the recursion passes the *same* segment
  down to the next (finer) scale rather than emitting a redundant node — so a tree
  node's ``children`` are always genuine sub-divisions, and ``penalty_level`` on a
  node is the scale that actually carved it out.

HONESTY
-------
Fine-penalty boundaries are **exploratory**, not confirmed changes: the benchmark
only validated the 1.5x setting (see DYNAMICS.md).  Every boundary is returned with
a ``shift`` magnitude (L2 of the standardised mean jump) and the single column that
moved most, so a consumer can tell a big genuine turn from a marginal wobble; the
caller is responsible for flagging seasonal/volume-driven boundaries (this module
stays taste-blind and just reports the geometry).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from typing import List, Optional, Sequence, Tuple

import numpy as np

from .detectors import _segment_shift_score, default_penalty, prepare_matrix
from .windows import WindowSeries


@dataclass
class SegmentNode:
    """One node of the hierarchical segmentation tree.

    * ``i0`` / ``i1`` — half-open index range into the *valid* (unmasked) windows.
    * ``start`` / ``end`` — the window start dates at ``i0`` and ``i1 - 1``.
    * ``penalty_level`` — the penalty *scale* that carved this node out of its
      parent (``None`` for the root, which is the whole history).
    * ``depth`` — 0 for the root, incremented per genuine split.
    * ``opening_shift`` — L2 magnitude of the standardised mean jump at this node's
      *opening* boundary (``0.0`` for the first child / root), with
      ``opening_shift_col`` the column that moved most there.
    * ``children`` — finer sub-segments (empty for a leaf).
    """

    i0: int
    i1: int
    start: date
    end: date
    penalty_level: Optional[float]
    depth: int
    opening_shift: float = 0.0
    opening_shift_col: Optional[str] = None
    children: List["SegmentNode"] = field(default_factory=list)

    @property
    def n_windows(self) -> int:
        return self.i1 - self.i0

    def leaves(self) -> List["SegmentNode"]:
        """The leaf nodes under this node (itself if it has no children)."""
        if not self.children:
            return [self]
        out: List["SegmentNode"] = []
        for c in self.children:
            out.extend(c.leaves())
        return out

    def walk(self) -> List["SegmentNode"]:
        """This node followed by all descendants, pre-order."""
        out = [self]
        for c in self.children:
            out.extend(c.walk())
        return out


def _pelt_split(Z: np.ndarray, cost: str, scale: float, min_size: int) -> List[int]:
    """Interior breakpoints of ``Z`` from PELT at ``scale * default_penalty``.

    Returns breakpoint indices **relative to ``Z``** (each ``0 < b < len(Z)``), in
    order.  An empty list means the segment was not split at this penalty.
    """
    import ruptures as rpt

    n, d = Z.shape
    if n < 2 * min_size:
        return []
    pen = float(scale) * default_penalty(n, d, cost)
    algo = rpt.Pelt(model=cost, min_size=min_size, jump=1).fit(Z)
    return [b for b in algo.predict(pen=pen) if 0 < b < n]


def _boundary_shift(Z: np.ndarray, bkp: int, columns: Sequence[str]) -> Tuple[float, Optional[str]]:
    """(L2 mean-jump magnitude, most-moved column name) at absolute index ``bkp``."""
    if bkp <= 0 or bkp >= len(Z):
        return 0.0, None
    left = Z[max(0, bkp - 12):bkp].mean(axis=0)
    right = Z[bkp:bkp + 12].mean(axis=0)
    diff = right - left
    mag = float(np.linalg.norm(diff))
    col = columns[int(np.argmax(np.abs(diff)))] if len(columns) else None
    return mag, col


def hierarchical_segments(
    series: WindowSeries,
    scales: Sequence[float] = (1.5, 1.0, 0.7, 0.5),
    representation: str = "combined",
    cost: str = "rbf",
    deseasonalize: bool = False,
    min_size: int = 8,
) -> SegmentNode:
    """Recursively PELT-segment ``series`` down a decreasing penalty ladder.

    Returns the **root** :class:`SegmentNode` spanning the whole (valid) history;
    walk ``.children`` for the nested sub-eras.  ``scales`` is applied outermost
    first (coarse -> fine); ``min_size`` is both PELT's minimum segment length and
    the recursion floor (a segment shorter than ``2 * min_size`` windows is a leaf).

    The taste representation, cost, and deseasonalise flag mirror
    :func:`~playlistcont.dynamics.detectors.pelt`; defaults match
    :func:`~playlistcont.dynamics.detectors.recommended_detector` (``rbf`` /
    ``combined`` / not deseasonalised).
    """
    Z, dates = prepare_matrix(series, representation, deseasonalize)
    columns = series.representation_columns(representation)
    n = len(Z)
    # window end for the last valid window in a range (for display 'end')
    valid_idx = series.valid_indices()

    def _end_date(i1: int) -> date:
        # end of the last window covered by [i0, i1): the WindowSeries 'ends' of
        # the original (calendar) index of valid window i1-1.
        return series.ends[valid_idx[i1 - 1]]

    root = SegmentNode(
        i0=0, i1=n, start=dates[0] if n else series.starts[0],
        end=_end_date(n) if n else series.ends[-1],
        penalty_level=None, depth=0)

    def _recurse(node: SegmentNode, level: int) -> None:
        if level >= len(scales):
            return
        scale = scales[level]
        sub = Z[node.i0:node.i1]
        rel = _pelt_split(sub, cost, scale, min_size)
        if not rel:
            _recurse(node, level + 1)          # same segment, finer penalty
            return
        abs_bkps = [node.i0 + b for b in rel]
        edges = [node.i0] + abs_bkps + [node.i1]
        for k in range(len(edges) - 1):
            a, b = edges[k], edges[k + 1]
            shift, col = (_boundary_shift(Z, a, columns) if k > 0 else (0.0, None))
            child = SegmentNode(
                i0=a, i1=b, start=dates[a], end=_end_date(b),
                penalty_level=float(scale), depth=node.depth + 1,
                opening_shift=shift, opening_shift_col=col)
            node.children.append(child)
            _recurse(child, level + 1)

    if n >= 2:
        _recurse(root, 0)
    return root


def detail_ladder(
    series: WindowSeries,
    scales: Sequence[float],
    representation: str = "combined",
    cost: str = "rbf",
    deseasonalize: bool = False,
    min_size: int = 8,
) -> List[dict]:
    """Global PELT boundary dates as a function of a single penalty scale.

    For each scale (coarse -> fine expected) runs PELT **once on the whole series**
    and returns ``{scale, penalty, n_boundaries, boundaries}`` where ``boundaries``
    is the list of ISO window-start dates.  This is the flat detail-slider companion
    to :func:`hierarchical_segments` — the same detector, no recursion.
    """
    Z, dates = prepare_matrix(series, representation, deseasonalize)
    n, d = Z.shape
    out: List[dict] = []
    for scale in scales:
        rel = _pelt_split(Z, cost, float(scale), min_size) if n >= 2 else []
        out.append({
            "scale": float(scale),
            "penalty": float(scale) * default_penalty(n, d, cost),
            "n_boundaries": len(rel),
            "boundaries": [dates[b].isoformat() for b in rel],
        })
    return out
