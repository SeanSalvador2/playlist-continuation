"""Taste-over-time change-point detection (Phase 3).

The scientific centrepiece: *when* did a listener's taste change, scored honestly
against the planted ground truth of the synthetic history generator.  The package
is a small pipeline —

* :mod:`~playlistcont.dynamics.windows` — turn a play stream into a per-window
  feature time series (scalar-axis means, genre shares, flavor shares, plus
  diagnostics), with masking and two weighting modes;
* :mod:`~playlistcont.dynamics.flavors` — fit-once / assign-many flavor clusters
  so flavor identities are stable across windows (unlike the engine's per-call
  refit);
* :mod:`~playlistcont.dynamics.detectors` — the method ladder (``baseline_scan``,
  ``cusum``, ``pelt``, ``bocpd``) behind one :class:`DetectedChange` interface,
  with an optional seasonal-preprocessing toggle;
* :mod:`~playlistcont.dynamics.eval` — precision / recall / F1 / localisation and
  trap- & seasonal-immunity scoring against the planted truth.

:func:`recommended_detector` exposes the benchmark-winning configuration for
Phase 4 to consume.  See ``DYNAMICS.md`` for the full write-up.
"""
from .detectors import (
    DetectedChange,
    RecommendedDetector,
    baseline_scan,
    bocpd,
    cusum,
    default_penalty,
    pelt,
    penalty_sweep,
    recommended_detector,
)
from .eval import score_detections, segmentation_ari
from .flavors import FlavorModel, assign, fit_flavors
from .windows import REPRESENTATIONS, WindowSeries, build_windows

__all__ = [
    "WindowSeries", "build_windows", "REPRESENTATIONS",
    "FlavorModel", "fit_flavors", "assign",
    "DetectedChange", "RecommendedDetector",
    "baseline_scan", "cusum", "pelt", "penalty_sweep", "bocpd",
    "default_penalty", "recommended_detector",
    "score_detections", "segmentation_ari",
]
