"""Score detected change points against a synthetic history's PLANTED truth.

Phase 3's whole claim is *honesty*: a detector is only as good as its numbers
against ground truth it never saw.  This module is the referee.  It takes a list
of :class:`~playlistcont.dynamics.detectors.DetectedChange` and a
:class:`~playlistcont.history.schema.HistoryGroundTruth` and returns
precision / recall / F1, a localisation error, and — the headline honesty
metrics — how many false positives landed inside a **trap** or a **seasonal**
span versus genuinely elsewhere.

WHAT COUNTS AS A HIT (the matching rule, stated once)
-----------------------------------------------------
The planted change_points are turned into **targets**:

* an ``"abrupt"`` change is a point target at its date; a detection *can* hit it
  if it falls within ``+/- tolerance_days``.
* a drift transition is an *interval* target ``[drift_start, drift_end]`` (the
  recorded linear-interpolation window); a detection *can* hit it if it falls in
  ``[drift_start - tolerance_days, drift_end + tolerance_days]``.

Matching is **one-to-one greedy by distance**: form every (detection, target)
pair that *can* hit, sort by distance ascending, and accept pairs greedily,
skipping any detection or target already used.  No detection matches two targets
and no target is hit twice.  Distance (also the per-hit localisation error) is
``|detection - date|`` for an abrupt target and the distance from the detection
to the interval (0 inside, else to the nearer edge) for a drift target.

* **precision** = TP / (TP + FP), **recall** = TP / (TP + FN), **F1** the
  harmonic mean (all defined to 0 on an empty denominator).
* **localisation MAE** = mean hit distance over matched pairs (``nan`` if no TP).

FALSE-POSITIVE ATTRIBUTION (trap & seasonal immunity)
-----------------------------------------------------
Every unmatched detection (a false positive) is attributed to exactly one bucket,
in priority order: it is ``fp_trap`` if it lands within ``tolerance_days`` of any
trap span, else ``fp_seasonal`` if within ``tolerance_days`` of any seasonal span,
else ``fp_other``.  Reported both as counts and as rates:

* ``trap_fp_rate`` = (# trap spans that caught >= 1 false positive) / (# traps)
* ``seasonal_fp_rate`` = (# seasonal spans that caught >= 1 FP) / (# seasonal)

These two rates are the trap-immunity and seasonal-immunity headlines.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta
from typing import Dict, List, Optional, Sequence, Tuple

from ..history.schema import HistoryGroundTruth
from .detectors import DetectedChange


# ===========================================================================
# targets from planted change points
# ===========================================================================
@dataclass
class _Target:
    kind: str            # "abrupt" | "drift"
    start: date          # abrupt: the date; drift: drift_start
    end: date            # abrupt: the date; drift: drift_end

    def distance(self, d: date) -> float:
        if d < self.start:
            return (self.start - d).days
        if d > self.end:
            return (d - self.end).days
        return 0.0


def build_targets(gt: HistoryGroundTruth) -> List[_Target]:
    """Fold planted change_points into abrupt-point and drift-interval targets."""
    targets: List[_Target] = []
    pending_start: Optional[date] = None
    for c in gt.change_points:
        if c.kind == "abrupt":
            targets.append(_Target("abrupt", c.date, c.date))
        elif c.kind == "drift_start":
            pending_start = c.date
        elif c.kind == "drift_end":
            start = pending_start if pending_start is not None else c.date
            targets.append(_Target("drift", start, c.date))
            pending_start = None
    return targets


# ===========================================================================
# matching + scoring
# ===========================================================================
def _greedy_match(
    detections: Sequence[DetectedChange],
    targets: Sequence[_Target],
    tolerance_days: int,
) -> Tuple[List[Tuple[int, int, float]], List[int], List[int]]:
    """One-to-one greedy-by-distance matching.

    Returns ``(matches, unmatched_det_idx, unmatched_tgt_idx)`` where each match is
    ``(det_index, tgt_index, distance)``.  A (det, tgt) pair is a candidate only when
    the detection is within ``tolerance_days`` of the target (point or interval).
    """
    candidates: List[Tuple[float, int, int]] = []
    for di, det in enumerate(detections):
        for ti, tgt in enumerate(targets):
            dist = tgt.distance(det.date)
            if dist <= tolerance_days:
                candidates.append((dist, di, ti))
    candidates.sort(key=lambda c: (c[0], c[1], c[2]))
    used_det: set = set()
    used_tgt: set = set()
    matches: List[Tuple[int, int, float]] = []
    for dist, di, ti in candidates:
        if di in used_det or ti in used_tgt:
            continue
        used_det.add(di)
        used_tgt.add(ti)
        matches.append((di, ti, dist))
    unmatched_det = [i for i in range(len(detections)) if i not in used_det]
    unmatched_tgt = [i for i in range(len(targets)) if i not in used_tgt]
    return matches, unmatched_det, unmatched_tgt


def _within(d: date, start: date, end: date, tol: int) -> bool:
    """True when ``d`` is within ``tol`` days of the span ``[start, end)``."""
    return (start - timedelta(days=tol)) <= d <= (end + timedelta(days=tol))


def score_detections(
    detections: Sequence[DetectedChange],
    ground_truth: HistoryGroundTruth,
    tolerance_days: int = 14,
) -> Dict[str, object]:
    """Precision / recall / F1 / localisation MAE + FP attribution vs planted truth.

    See the module docstring for the exact matching and attribution rules.  Returns a
    flat dict of metrics and counts (``tp, fp, fn, precision, recall, f1,
    localization_mae, fp_trap, fp_seasonal, fp_other, trap_fp_rate,
    seasonal_fp_rate``) ready to drop into a results row.
    """
    targets = build_targets(ground_truth)
    matches, unmatched_det, unmatched_tgt = _greedy_match(
        detections, targets, tolerance_days)

    tp = len(matches)
    fp = len(unmatched_det)
    fn = len(unmatched_tgt)
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * precision * recall / (precision + recall)
          if (precision + recall) else 0.0)
    localization_mae = (sum(m[2] for m in matches) / tp) if tp else float("nan")

    # ---- FP attribution ------------------------------------------------
    traps = ground_truth.traps
    seasonal = ground_truth.seasonal_spans
    fp_trap = fp_seasonal = fp_other = 0
    trap_hit = [False] * len(traps)
    seasonal_hit = [False] * len(seasonal)
    for di in unmatched_det:
        d = detections[di].date
        matched_trap = next(
            (i for i, t in enumerate(traps)
             if _within(d, t.start, t.end, tolerance_days)), None)
        if matched_trap is not None:
            fp_trap += 1
            trap_hit[matched_trap] = True
            continue
        matched_season = next(
            (i for i, s in enumerate(seasonal)
             if _within(d, s.start, s.end, tolerance_days)), None)
        if matched_season is not None:
            fp_seasonal += 1
            seasonal_hit[matched_season] = True
            continue
        fp_other += 1

    trap_fp_rate = (sum(trap_hit) / len(traps)) if traps else 0.0
    seasonal_fp_rate = (sum(seasonal_hit) / len(seasonal)) if seasonal else 0.0

    return {
        "n_detections": len(detections),
        "n_targets": len(targets),
        "tp": tp, "fp": fp, "fn": fn,
        "precision": precision, "recall": recall, "f1": f1,
        "localization_mae": localization_mae,
        "fp_trap": fp_trap, "fp_seasonal": fp_seasonal, "fp_other": fp_other,
        "trap_fp_rate": trap_fp_rate, "seasonal_fp_rate": seasonal_fp_rate,
        "n_traps": len(traps), "n_seasonal": len(seasonal),
    }


# ===========================================================================
# segmentation ARI
# ===========================================================================
def _regime_label(d: date, ground_truth: HistoryGroundTruth) -> int:
    """Index of the planted regime containing date ``d`` (clamped to the ends)."""
    regimes = ground_truth.regimes
    for i, r in enumerate(regimes):
        if r.start <= d < r.end:
            return i
    return len(regimes) - 1 if d >= regimes[-1].start else 0


def segmentation_ari(
    detections: Sequence[DetectedChange],
    ground_truth: HistoryGroundTruth,
    dates: Sequence[date],
) -> float:
    """Adjusted Rand Index between induced and true per-window segmentations.

    Each window in ``dates`` gets a **true** label — the planted regime that
    contains it — and an **induced** label — the number of detected changes dated
    on/before it (so detections carve ``dates`` into contiguous segments).  Returns
    the ``sklearn`` adjusted Rand index of the two labellings (1.0 = the induced
    segmentation partitions the windows exactly like the regimes, 0.0 = chance).
    An empty ``dates`` returns ``nan``.
    """
    from sklearn.metrics import adjusted_rand_score

    dates = list(dates)
    if not dates:
        return float("nan")
    true_labels = [_regime_label(d, ground_truth) for d in dates]
    change_dates = sorted(c.date for c in detections)
    induced: List[int] = []
    for d in dates:
        induced.append(sum(1 for cd in change_dates if cd <= d))
    return float(adjusted_rand_score(true_labels, induced))
