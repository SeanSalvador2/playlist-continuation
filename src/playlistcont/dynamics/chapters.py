"""Adaptive chapter extraction with a guaranteed floor and honest labels.

Where :mod:`~playlistcont.dynamics.eras` cuts the timeline at whatever the
recommended detector happens to find — which is **nothing** for an unusually
stable listener — this module guarantees a narrative spine for *any* history
while never lying about how solid that spine is.

THE CONTRACT
------------
``adaptive_chapters`` returns a :class:`ChapterPlan` that ALWAYS contains at
least ``min_chapters`` chapters, and every boundary in it carries an honest
confidence label:

* ``"validated"``  — found at the **recommended** penalty (the 1.5x rung that the
  benchmark blessed in :func:`~playlistcont.dynamics.detectors.recommended_detector`);
* ``"exploratory"`` — only appeared after we **relaxed** the penalty one or two
  rungs (1.0x / 0.7x); a real but weaker turn;
* ``"forced"`` — only appeared at the **loosest** rungs (<= 0.5x), or is a pure
  time-split we imposed because nothing was detectable at all.

The plan also carries an overall :attr:`ChapterPlan.structure_strength` verdict
(``"strong"`` / ``"moderate"`` / ``"weak — this listener is unusually stable"``)
derived from how much relaxation was needed and the magnitude of the boundary
shifts.

THE RELAXATION LADDER
---------------------
Detection reuses the exact PELT machinery of
:mod:`~playlistcont.dynamics.hierarchy` / :mod:`~playlistcont.dynamics.detectors`
(``rbf`` cost, ``combined`` representation, the standardised valid-window
matrix), so a penalty *scale* here means precisely what it means everywhere else.
We walk :data:`RELAX_LADDER` = ``(1.5, 1.0, 0.7, 0.5, 0.35, 0.25)`` from the
recommended rung down.  At each rung we run PELT once on the whole series and stop
at the first rung that yields at least ``min_chapters - 1`` boundaries.  Because a
lower penalty only ever *adds* breakpoints, each boundary is labelled by the
**first** rung at which it appeared, so a chapter set found after relaxation
honestly marks which of its turns were confirmed and which were coaxed out.

Inside each chapter we recurse **one level**: the same laddered split is run on
the chapter's own slice (a local penalty computed from the chapter's length, min
``min_subsection_size`` windows) to find sub-sections, again floored at
``min_subsections`` where the chapter is long enough to support them.

THE STATIONARY FALLBACK
-----------------------
If even the loosest rung finds nothing (a genuinely stationary listener), we do
**not** invent a turn.  We split the valid span into ``min_chapters`` equal-time
pieces, mark every seam ``"forced"`` with the note *"no detectable taste change —
split by time for narrative only"*, and set the strength verdict to ``weak``.
min_chapters is satisfied; honesty is preserved.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import List, Optional, Sequence, Tuple, Union

import numpy as np

from ..history.schema import ListeningHistory
from ..history.store import HistoryStore
from .detectors import default_penalty, prepare_matrix
from .hierarchy import _boundary_shift, _pelt_split
from .windows import WindowSeries, build_windows

# the documented relaxation ladder, coarse (recommended) -> fine (forced)
RELAX_LADDER: Tuple[float, ...] = (1.5, 1.0, 0.7, 0.5, 0.35, 0.25)
RECOMMENDED_SCALE = 1.5
# scale thresholds for the confidence label of a boundary's first-appearance rung
_VALIDATED_MIN_SCALE = 1.5      # >= this -> validated (the recommended rung)
_EXPLORATORY_MIN_SCALE = 0.7    # >= this (but below validated) -> exploratory; else forced

# structure-strength shift thresholds are expressed as multiples of the noise
# floor of a pure-noise mean-difference over the flanking windows (see _noise_floor).
_STRONG_SHIFT_MULT = 1.75
_FLANK = 12                     # windows either side used by _boundary_shift


def _confidence_for_scale(scale: Optional[float]) -> str:
    """Confidence label for the penalty *scale* a boundary first appeared at."""
    if scale is None:
        return "forced"
    if scale >= _VALIDATED_MIN_SCALE:
        return "validated"
    if scale >= _EXPLORATORY_MIN_SCALE:
        return "exploratory"
    return "forced"


def _noise_floor(d: int, flank: int = _FLANK) -> float:
    """Expected L2 magnitude of a pure-noise mean-difference over ``flank`` windows.

    Each of ``d`` z-scored columns has per-window variance ~1, so a mean over
    ``flank`` windows has variance ~1/flank; the difference of two such means has
    variance ~2/flank per column, and its L2 over ``d`` columns is ~sqrt(2 d /
    flank).  This is the "nothing happened" baseline a real turn must clear.
    """
    return math.sqrt(2.0 * max(1, d) / max(1, flank))


@dataclass
class ChapterBoundary:
    """One seam between chapters (or sub-sections), with its honesty label.

    * ``i_window`` — index into the *valid* (unmasked) windows where the seam sits.
    * ``date`` — the window-start date at ``i_window``.
    * ``confidence`` — ``"validated"`` / ``"exploratory"`` / ``"forced"``.
    * ``scale`` — the penalty scale that first surfaced this seam (``None`` for a
      pure time-split seam).
    * ``shift`` / ``shift_col`` — L2 magnitude of the standardised mean jump here
      and the single column that moved most.
    """

    i_window: int
    date: date
    confidence: str
    scale: Optional[float]
    shift: float
    shift_col: Optional[str]

    def to_payload(self) -> dict:
        return {
            "date": self.date.isoformat(),
            "confidence": self.confidence,
            "scale": self.scale,
            "shift": round(self.shift, 4),
            "shift_col": self.shift_col,
        }


@dataclass
class SubSection:
    """A sub-span inside a chapter.  ``opening`` is the seam that starts it (``None``
    for the chapter's first sub-section)."""

    index: int
    i0: int
    i1: int
    start: date
    end: date                              # inclusive display end
    opening: Optional[ChapterBoundary] = None

    def to_payload(self) -> dict:
        return {
            "index": self.index,
            "start": self.start.isoformat(),
            "end": self.end.isoformat(),
            "opening": self.opening.to_payload() if self.opening else None,
        }


@dataclass
class Chapter:
    """One narrative chapter: its span, the seam that opens it, and its sub-sections."""

    index: int
    i0: int
    i1: int
    start: date
    end: date                              # inclusive display end
    opening: Optional[ChapterBoundary]     # None for the first chapter
    subsections: List[SubSection] = field(default_factory=list)

    @property
    def n_turns(self) -> int:
        """Number of *internal* sub-section seams (sub-sections minus one)."""
        return max(0, len(self.subsections) - 1)

    def headline(self) -> str:
        """Chapter-shape copy COMPUTED from the sub-section structure.

        ``"a slow drift — no sharp turn"`` when the chapter holds no internal seam,
        ``"one clear turn"`` for a single seam, else ``"<word> turns"``.
        """
        n = self.n_turns
        if n <= 0:
            return "a slow drift — no sharp turn"
        if n == 1:
            return "one clear turn"
        return f"{_num_word(n)} turns"

    def to_payload(self) -> dict:
        return {
            "index": self.index,
            "start": self.start.isoformat(),
            "end": self.end.isoformat(),
            "opening": self.opening.to_payload() if self.opening else None,
            "n_turns": self.n_turns,
            "headline": self.headline(),
            "subsections": [s.to_payload() for s in self.subsections],
        }


@dataclass
class ChapterPlan:
    """The adaptive chapter segmentation with floor guarantees and honest labels."""

    chapters: List[Chapter]
    boundaries: List[ChapterBoundary]      # chapter-level seams (flat, in order)
    structure_strength: str
    relaxation_used: bool
    fallback: Optional[str]                # the time-split note, or None
    levels: List[dict]                     # the ladder trace that was walked
    min_chapters: int
    min_subsections: int

    def to_payload(self) -> dict:
        return {
            "structure_strength": self.structure_strength,
            "relaxation_used": self.relaxation_used,
            "fallback": self.fallback,
            "min_chapters": self.min_chapters,
            "min_subsections": self.min_subsections,
            "n_chapters": len(self.chapters),
            "levels": self.levels,
            "boundaries": [b.to_payload() for b in self.boundaries],
            "chapters": [c.to_payload() for c in self.chapters],
        }

    def as_detections(self):
        """The chapter seams as :class:`~playlistcont.dynamics.detectors.DetectedChange`.

        Handy for feeding :func:`~playlistcont.dynamics.eras.build_eras` so the named
        eras line up one-to-one with the plan's chapters.
        """
        from .detectors import DetectedChange
        return [DetectedChange(date=b.date, score=float(b.shift),
                               method=f"adaptive_chapters:{b.confidence}")
                for b in self.boundaries]


_NUM_WORDS = ["zero", "one", "two", "three", "four", "five", "six", "seven",
              "eight", "nine"]


def _num_word(n: int) -> str:
    return _NUM_WORDS[n] if 0 <= n < len(_NUM_WORDS) else "many"


def _laddered_split(
    Z: np.ndarray, cost: str, ladder: Sequence[float], min_needed: int,
    min_size: int,
) -> Tuple[List[int], dict, List[dict], bool]:
    """Walk the relaxation ladder on ``Z`` until ``>= min_needed`` breakpoints.

    Returns ``(chosen_bkps, first_scale, levels, relaxation_used)`` where
    ``chosen_bkps`` are the interior breakpoint indices at the rung we stopped on
    (sorted), ``first_scale`` maps each breakpoint to the coarsest rung it first
    appeared at, ``levels`` is the per-rung trace, and ``relaxation_used`` is True
    when we had to descend below the first rung.
    """
    n, d = Z.shape
    first_scale: dict = {}
    levels: List[dict] = []
    chosen: List[int] = []
    relaxation_used = False
    rel: List[int] = []
    for step, scale in enumerate(ladder):
        rel = _pelt_split(Z, cost, float(scale), min_size) if n >= 2 * min_size else []
        for b in rel:
            first_scale.setdefault(b, float(scale))
        levels.append({
            "scale": float(scale),
            "penalty": float(scale) * default_penalty(n, d, cost),
            "n_boundaries": len(rel),
            "boundaries": [int(b) for b in rel],
        })
        if len(rel) >= min_needed:
            chosen = list(rel)
            relaxation_used = step > 0
            break
    else:
        # exhausted the ladder without reaching the floor: keep the loosest rung's
        # find (possibly empty -> caller applies its fallback).
        chosen = list(rel)
        relaxation_used = True
    return sorted(chosen), first_scale, levels, relaxation_used


def adaptive_chapters(
    source: Union[WindowSeries, ListeningHistory, HistoryStore],
    min_chapters: int = 2,
    min_subsections: int = 2,
    representation: str = "combined",
    cost: str = "rbf",
    ladder: Sequence[float] = RELAX_LADDER,
    min_size: int = 8,
    min_subsection_size: int = 8,
    deseasonalize: bool = False,
) -> ChapterPlan:
    """Extract at least ``min_chapters`` chapters from ``source``, honestly labelled.

    ``source`` may be a prebuilt :class:`~playlistcont.dynamics.windows.WindowSeries`
    or a history/store (in which case weekly windows are built with the recommended
    ``min_events=30`` / ``weighting="plays"`` config).  See the module docstring for
    the relaxation ladder, the per-boundary confidence labels, the structure-strength
    verdict, and the stationary time-split fallback.
    """
    series = (source if isinstance(source, WindowSeries)
              else build_windows(source, granularity="week", min_events=30,
                                  weighting="plays"))
    Z, dates = prepare_matrix(series, representation, deseasonalize)
    columns = series.representation_columns(representation)
    n = len(Z)
    valid_idx = series.valid_indices()

    def _end_incl(i1: int) -> date:
        """Inclusive last calendar day covered by valid windows [.., i1)."""
        if i1 <= 0 or not len(valid_idx):
            return series.ends[-1] - timedelta(days=1) if len(series.ends) else dates[0]
        return series.ends[valid_idx[i1 - 1]] - timedelta(days=1)

    min_chapters = max(1, int(min_chapters))
    # cannot make more non-empty chapters than there are valid windows
    min_chapters_eff = min(min_chapters, max(1, n))
    min_needed = max(0, min_chapters_eff - 1)

    chosen, first_scale, levels, relaxation_used = _laddered_split(
        Z, cost, ladder, min_needed, min_size)

    fallback: Optional[str] = None
    if len(chosen) < min_needed:
        # stationary: impose an equal-time split so the floor is still met, honestly.
        chosen = _time_split_indices(n, min_chapters_eff)
        first_scale = {b: None for b in chosen}
        relaxation_used = True
        fallback = ("forced: no detectable taste change — split by time for "
                    "narrative only")

    # ---- build chapter-level boundary objects ---------------------------- #
    boundaries: List[ChapterBoundary] = []
    for b in chosen:
        scale = first_scale.get(b)
        shift, col = _boundary_shift(Z, b, columns)
        boundaries.append(ChapterBoundary(
            i_window=int(b), date=dates[b],
            confidence=("forced" if fallback else _confidence_for_scale(scale)),
            scale=scale, shift=float(shift), shift_col=col))

    # ---- assemble chapters and recurse one level for sub-sections -------- #
    edges = [0] + [b.i_window for b in boundaries] + [n]
    chapters: List[Chapter] = []
    for ci in range(len(edges) - 1):
        a, bnd = edges[ci], edges[ci + 1]
        opening = boundaries[ci - 1] if ci > 0 else None
        chapter = Chapter(
            index=ci, i0=a, i1=bnd,
            start=dates[a] if a < n else dates[-1],
            end=_end_incl(bnd), opening=opening)
        chapter.subsections = _subsections_for(
            Z, columns, dates, _end_incl, a, bnd, cost, ladder,
            min_subsections, min_subsection_size)
        chapters.append(chapter)

    strength = _structure_strength(boundaries, Z.shape[1], fallback, relaxation_used)
    return ChapterPlan(
        chapters=chapters, boundaries=boundaries, structure_strength=strength,
        relaxation_used=relaxation_used, fallback=fallback, levels=levels,
        min_chapters=min_chapters, min_subsections=min_subsections)


def _subsections_for(
    Z: np.ndarray, columns, dates, end_incl, i0: int, i1: int, cost: str,
    ladder: Sequence[float], min_subsections: int, min_size: int,
) -> List[SubSection]:
    """One level of laddered sub-segmentation inside chapter ``[i0, i1)``."""
    sub = Z[i0:i1]
    seams: List[int] = []
    first_scale: dict = {}
    if len(sub) >= 2 * min_size:
        min_needed = max(0, int(min_subsections) - 1)
        rel, first_scale, _levels, _relaxed = _laddered_split(
            sub, cost, ladder, min_needed, min_size)
        seams = [i0 + b for b in rel]
        first_scale = {i0 + b: s for b, s in first_scale.items() if (i0 + b) in seams}

    edges = [i0] + seams + [i1]
    out: List[SubSection] = []
    for si in range(len(edges) - 1):
        a, b = edges[si], edges[si + 1]
        opening = None
        if si > 0:
            scale = first_scale.get(a)
            shift, col = _boundary_shift(Z, a, columns)
            opening = ChapterBoundary(
                i_window=int(a), date=dates[a],
                confidence=_confidence_for_scale(scale), scale=scale,
                shift=float(shift), shift_col=col)
        out.append(SubSection(index=si, i0=a, i1=b, start=dates[a],
                              end=end_incl(b), opening=opening))
    return out


def _time_split_indices(n: int, parts: int) -> List[int]:
    """Interior breakpoints splitting ``n`` windows into ``parts`` equal-time pieces."""
    if parts <= 1 or n <= 1:
        return []
    cuts = []
    for k in range(1, parts):
        idx = int(round(k * n / parts))
        idx = max(1, min(n - 1, idx))
        if idx not in cuts:
            cuts.append(idx)
    return sorted(cuts)


def _structure_strength(
    boundaries: List[ChapterBoundary], d: int, fallback: Optional[str],
    relaxation_used: bool,
) -> str:
    """Overall verdict from the boundary labels and shift magnitudes.

    ``weak`` when we fell back to a time-split, or when no boundary reached at least
    the exploratory tier.  ``strong`` when at least one boundary was validated
    without any relaxation and the largest shift clears ``_STRONG_SHIFT_MULT`` times
    the noise floor.  ``moderate`` otherwise.
    """
    stable_tail = " — this listener is unusually stable"
    if fallback or not boundaries:
        return "weak" + stable_tail
    if all(b.confidence == "forced" for b in boundaries):
        return "weak" + stable_tail
    max_shift = max(b.shift for b in boundaries)
    floor = _noise_floor(d)
    any_validated = any(b.confidence == "validated" for b in boundaries)
    if any_validated and not relaxation_used and max_shift >= _STRONG_SHIFT_MULT * floor:
        return "strong"
    return "moderate"
