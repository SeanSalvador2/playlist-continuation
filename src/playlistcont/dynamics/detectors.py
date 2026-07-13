"""The change-point method ladder — one interface, four (+1) detectors.

Every detector answers the same question — *on which window dates did this
listener's taste change?* — and returns the same thing: a list of
:class:`DetectedChange` ``(date, score, method)``.  That uniformity is what lets
``experiments/exp_dynamics.py`` line them up in a single grid and score them all
against the planted ground truth on equal footing.

THE STANDARDISATION CONTRACT (read once, applies to cusum/pelt/bocpd)
--------------------------------------------------------------------
All of the matrix-based detectors run on a **z-scored** view of a
representation's columns:

* Take the chosen representation's columns from the :class:`WindowSeries`.
* Keep only **unmasked** windows, in calendar order, and treat that compressed
  sequence as contiguous.  (Masked, low-data windows are dropped rather than
  imputed; documented limitation — a long data gap is stitched over, not
  modelled.)
* Standardise each column to mean 0 / unit variance using the mean and standard
  deviation **over those unmasked windows** (population std, ``ddof=0``; a
  zero-variance column becomes all-zeros via a std floor of 1e-9).

This puts every column on a comparable scale so a shift in, say, a genre share
(range ~0.3) is weighed against a shift in valence (range ~0.1) by its own
variability, not its raw units.  ``baseline_scan`` is the exception: it delegates
to the Phase-2 :func:`~playlistcont.analytics.stats.adjacent_scan`, which does
its own within-test standardisation (effect sizes), so it takes the *store*, not
the matrix.

DESEASONALISATION (a preprocessing toggle, off by default)
----------------------------------------------------------
``deseasonalize=True`` subtracts, per column, a recurring **month-of-year**
offset before standardising, so the generator's December "party pop" bump does
not read as a change every winter.  We estimate the offset as the plain
**overall month-of-year mean** (all years pooled): ``corrected = value -
(mean_of_this_calendar_month - global_mean)``.  It is only applied when the
history spans **>= 18 months** (enough to see a month more than once); on shorter
histories it is a documented no-op.  A more principled *leave-one-year-out*
estimate (correct each year from the *other* years, so a genuine one-off change
that happens to fall in December is not partially erased) is the natural
refinement; with only two years of synthetic data the pooled estimate and LOO
are nearly identical, so we ship the simpler pooled version and note the
limitation.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import date
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from ..analytics import stats as stats_mod
from ..history.store import HistoryStore
from .windows import WindowSeries, build_windows

STD_FLOOR = 1e-9


@dataclass
class DetectedChange:
    """One detected change point: the window ``date``, a ``score``, its ``method``."""

    date: date
    score: float
    method: str


# ===========================================================================
# shared preprocessing: representation -> standardised valid-window matrix
# ===========================================================================
def _month_of_year_deseasonalize(Z: np.ndarray, months: np.ndarray) -> np.ndarray:
    """Subtract the pooled month-of-year offset from each column (see module docstring)."""
    out = Z.copy()
    global_mean = Z.mean(axis=0)
    for m in np.unique(months):
        sel = months == m
        offset = Z[sel].mean(axis=0) - global_mean
        out[sel] -= offset
    return out


def prepare_matrix(
    series: WindowSeries,
    representation: str = "combined",
    deseasonalize: bool = False,
) -> Tuple[np.ndarray, List[date]]:
    """Return ``(Z, dates)``: the standardised valid-window matrix and their dates.

    ``Z`` is ``(n_valid, n_cols)`` z-scored per column; ``dates`` are the start
    dates of the unmasked windows in order.  When ``deseasonalize`` is set and the
    history spans >= 18 months, a pooled month-of-year offset is removed *before*
    standardising.
    """
    idx = series.valid_indices()
    raw = series.submatrix(representation)[idx]
    dates = [series.starts[i] for i in idx]

    if deseasonalize and series.span_days >= int(18 * 30.4):
        months = np.array([d.month for d in dates])
        raw = _month_of_year_deseasonalize(raw, months)

    mu = raw.mean(axis=0)
    sd = raw.std(axis=0)
    sd = np.where(sd < STD_FLOOR, 1.0, sd)
    Z = (raw - mu) / sd
    return Z, dates


# ===========================================================================
# 0. baseline_scan — the Phase-2 detector, wrapped into the common interface
# ===========================================================================
def baseline_scan(
    history_or_store,
    granularity: str = "month",
    q_threshold: float = stats_mod.DEFAULT_Q,
) -> List[DetectedChange]:
    """The BASELINE: delegate to :func:`stats.adjacent_scan` and map its flags.

    This is the Phase-2 adjacent-month scan (FDR + effect-floor gated) that every
    fancier method must beat.  It runs on the store directly (its own effect-size
    standardisation), scans scalar-axis tests month-over-month, and each *flagged*
    boundary becomes a :class:`DetectedChange` dated at the boundary, scored by the
    boundary's max surviving effect size.  It is deliberately representation- and
    deseasonalise-agnostic — it is the fixed reference, including its known habit of
    firing on the December seasonal bump.
    """
    store = (history_or_store if isinstance(history_or_store, HistoryStore)
             else HistoryStore.from_history(history_or_store))
    scan = stats_mod.adjacent_scan(
        store, granularity=granularity, metrics="axes", q_threshold=q_threshold)
    out: List[DetectedChange] = []
    for b in scan["boundaries"]:
        if b["flagged"]:
            out.append(DetectedChange(
                date=date.fromisoformat(b["boundary"]),
                score=float(b["max_effect"]), method="baseline_scan"))
    return out


# ===========================================================================
# 1. cusum — self-implemented multivariate CUSUM
# ===========================================================================
def cusum(
    series: WindowSeries,
    representation: str = "combined",
    deseasonalize: bool = False,
    k: float = 1.0,
    threshold: float = 5.0,
    combine: str = "max",
) -> List[DetectedChange]:
    """Multivariate two-sided CUSUM on the standardised representation.

    For each column we run the classic tabular CUSUM with slack ``k`` (in std
    units, since the input is z-scored)::

        S+_t = max(0, S+_{t-1} + z_t - k)
        S-_t = max(0, S-_{t-1} - z_t - k)

    and take that column's statistic as ``max(S+_t, S-_t)``.  The columns are
    **combined** into one alarm statistic per window either by ``"max"`` (default —
    most sensitive to a jump in *any single* column; a taste change often moves one
    axis first) or ``"l2"`` (the Euclidean norm across columns — pools a diffuse
    change spread over many columns).  When the combined statistic exceeds
    ``threshold`` we raise an alarm, **back-date** it to where the current run
    began (the last window at which the combined statistic was 0 — CUSUM alarms lag
    the true change, and back-dating recovers most of that delay), record the peak
    as the score, and **reset** all accumulators so later changes can be found.

    ``k`` and ``threshold`` are the standard CUSUM knobs (slack in std units, alarm
    level).  **Adaptive baseline (important for offline use):** a persistent regime
    shift keeps the z-scores high forever, so a textbook CUSUM referenced to the
    global mean would re-fire every few windows for the *rest of the regime*.  To
    detect each change **once**, after every alarm we re-anchor the per-column
    reference to the new post-change level (the mean of the windows since the alarm)
    and reset the accumulators — so CUSUM tracks *departures from the current
    regime*, not from the global mean.  Defaults (k=1.0, threshold=5.0, max-combine)
    are what the benchmark uses.
    """
    if combine not in ("max", "l2"):
        raise ValueError(f"unknown combine {combine!r}; expected max|l2")
    Z, dates = prepare_matrix(series, representation, deseasonalize)
    n, d = Z.shape
    out: List[DetectedChange] = []
    if n == 0:
        return out
    # current-regime reference (adapts after each alarm); seeded from the first few
    # windows so a single noisy first window cannot bias the whole run.
    baseline = Z[:min(4, n)].mean(axis=0)
    s_pos = np.zeros(d)
    s_neg = np.zeros(d)
    run_start = 0                # index where the current run began (stat == 0)
    peak = 0.0
    for t in range(n):
        dev = Z[t] - baseline
        s_pos = np.maximum(0.0, s_pos + dev - k)
        s_neg = np.maximum(0.0, s_neg - dev - k)
        per_col = np.maximum(s_pos, s_neg)
        stat = float(per_col.max()) if combine == "max" else float(
            math.sqrt((per_col ** 2).sum()))
        if stat == 0.0:
            run_start = t
        peak = max(peak, stat)
        if stat > threshold:
            onset = min(run_start + 1, n - 1)
            out.append(DetectedChange(date=dates[onset], score=peak, method="cusum"))
            baseline = Z[onset:t + 1].mean(axis=0)   # re-anchor to the new level
            s_pos[:] = 0.0
            s_neg[:] = 0.0
            run_start = t
            peak = 0.0
    return _dedupe_adjacent(out, min_gap=2)


# ===========================================================================
# 2. pelt — via the ruptures library (l2 / rbf costs)
# ===========================================================================
def _segment_shift_score(Z: np.ndarray, bkp: int) -> float:
    """Magnitude of the mean jump at breakpoint ``bkp`` (L2 of the mean difference)."""
    if bkp <= 0 or bkp >= len(Z):
        return 0.0
    left = Z[max(0, bkp - 12):bkp].mean(axis=0)
    right = Z[bkp:bkp + 12].mean(axis=0)
    return float(np.linalg.norm(right - left))


def default_penalty(n: int, d: int, cost: str = "l2") -> float:
    """A cost-appropriate default PELT penalty.

    * ``l2`` -> ``d * log(n)``.  On z-scored data (per-column variance ~1) the
      ``l2`` segment cost is a sum of squared residuals, for which
      ``dim * log(n_samples)`` is the standard Schwarz/BIC penalty (``ruptures``'
      own recommended default).
    * ``rbf`` -> ``log(n)``.  The ``rbf`` kernel cost is bounded ~O(1) per segment
      *regardless of dimension* (ruptures normalises the Gram matrix), so scaling by
      ``d`` over-penalises massively — empirically ``d*log n`` finds **zero** changes
      on a 19-column series that plainly has four.  A dimension-independent
      ``log(n)`` restores sensitivity.

    Exposed so the experiment and :func:`recommended_detector` share one convention.
    """
    if cost == "rbf":
        return math.log(max(2, n))
    return float(d) * math.log(max(2, n))


def pelt(
    series: WindowSeries,
    representation: str = "combined",
    deseasonalize: bool = False,
    cost: str = "l2",
    penalty: Optional[float] = None,
    penalty_scale: float = 1.0,
) -> List[DetectedChange]:
    """PELT change points (``ruptures``) on the standardised representation.

    ``cost`` selects the segment model: ``"l2"`` (piecewise-constant mean — cheap,
    matches how the regimes are actually planted) or ``"rbf"`` (a kernel cost that
    also catches distributional/variance changes).  ``penalty`` is PELT's complexity
    penalty; ``None`` uses ``penalty_scale *`` :func:`default_penalty`.  The scale
    knob exists because a modest boost (1.5x — the benchmark-recommended setting for
    ``rbf``) suppresses *transient* excursions like the December seasonal bump —
    which cost TWO breakpoints for a short segment that reverts — while keeping
    *persistent* regime changes, which cost one breakpoint each and amortise over a
    long segment.  Measured across four generator settings in DYNAMICS.md.  An
    explicit ``penalty`` overrides both.  Each interior breakpoint becomes a
    :class:`DetectedChange` dated at that window, scored by the local mean-shift
    magnitude.
    """
    import ruptures as rpt

    Z, dates = prepare_matrix(series, representation, deseasonalize)
    n, d = Z.shape
    if n < 2:
        return []
    pen = (float(penalty_scale) * default_penalty(n, d, cost)
           if penalty is None else float(penalty))
    algo = rpt.Pelt(model=cost, min_size=2, jump=1).fit(Z)
    bkps = algo.predict(pen=pen)
    out: List[DetectedChange] = []
    for b in bkps:
        if 0 < b < n:                    # ruptures returns n as the final "breakpoint"
            out.append(DetectedChange(
                date=dates[b], score=_segment_shift_score(Z, b),
                method=f"pelt_{cost}"))
    return out


def penalty_sweep(
    series: WindowSeries,
    representation: str = "combined",
    deseasonalize: bool = False,
    cost: str = "l2",
    penalties: Optional[Sequence[float]] = None,
) -> Dict[str, object]:
    """Change-count-vs-penalty curve for PELT, with an elbow-picked default.

    Sweeps ``penalties`` (default: 20 log-spaced multiples of :func:`default_penalty`
    from 0.1x to 10x), records the number of detected changes at each, and picks an
    **elbow** via the "maximum distance to the chord" (Kneedle) rule: min-max
    normalise both axes, draw the line from the first to the last point of the
    (penalty, n_changes) curve, and take the penalty whose point is farthest *below*
    that line — the knee where adding penalty stops buying you a big drop in change
    count.  Returns ``{penalties, n_changes, elbow_penalty, elbow_index}``.
    """
    Z, _ = prepare_matrix(series, representation, deseasonalize)
    n, d = Z.shape
    if penalties is None:
        base = default_penalty(n, d, cost)
        penalties = [base * m for m in np.geomspace(0.1, 10.0, 20)]
    penalties = list(penalties)
    counts: List[int] = []
    if n >= 2:
        import ruptures as rpt
        algo = rpt.Pelt(model=cost, min_size=2, jump=1).fit(Z)
        for pen in penalties:
            bkps = algo.predict(pen=float(pen))
            counts.append(sum(1 for b in bkps if 0 < b < n))
    else:
        counts = [0] * len(penalties)

    elbow_i = _kneedle_elbow(np.array(penalties, dtype=float),
                             np.array(counts, dtype=float))
    return {
        "penalties": penalties,
        "n_changes": counts,
        "elbow_penalty": float(penalties[elbow_i]),
        "elbow_index": int(elbow_i),
    }


def _kneedle_elbow(x: np.ndarray, y: np.ndarray) -> int:
    """Index of the knee of a decreasing curve (max distance below the endpoint chord)."""
    if len(x) < 3:
        return 0
    order = np.argsort(x)
    xs, ys = x[order], y[order]
    xn = (xs - xs.min()) / (np.ptp(xs) or 1.0)
    yn = (ys - ys.min()) / (np.ptp(ys) or 1.0)
    # chord from first to last; distance of each point below it (decreasing curve)
    chord = yn[0] + (yn[-1] - yn[0]) * (xn - xn[0]) / ((xn[-1] - xn[0]) or 1.0)
    dist = chord - yn                     # positive where the curve dips below chord
    return int(order[int(np.argmax(dist))])


# ===========================================================================
# 3. bocpd — Bayesian Online Change-Point Detection (Adams & MacKay 2007)
# ===========================================================================
def bocpd(
    series: WindowSeries,
    representation: str = "combined",
    deseasonalize: bool = False,
    hazard_run_length: float = 20.0,
    prior_var: float = 1.0,
    obs_var: float = 1.0,
    min_run: int = 4,
) -> List[DetectedChange]:
    """Bayesian Online CPD, self-implemented Adams & MacKay (2007) recursion.

    **Model.** Each standardised column is an independent Gaussian stream with
    *known* observation variance ``obs_var`` and a Normal prior on the (unknown)
    segment mean (mean 0, variance ``prior_var``).  Known variance is the honest
    choice here precisely because the input is z-scored to **unit variance by
    construction**, and it makes the filter far more sensitive to a *mean* shift
    than the unknown-variance (Normal-Gamma) alternative, which tends to explain a
    new regime away as inflated variance and never resets (measured — see
    DYNAMICS.md).  The conjugate posterior-predictive for run length ``r`` with
    sufficient stats ``(n_r, sum_r)`` is Gaussian with
    ``mean = (sum_r/obs_var) / (1/prior_var + n_r/obs_var)`` and variance
    ``1/(1/prior_var + n_r/obs_var) + obs_var``; the joint log-predictive sums these
    over columns (independence).

    **Detection.** Under a *constant* hazard ``H = 1/hazard_run_length`` the
    posterior ``P(r_t = 0 | data)`` is provably always exactly ``H`` — it carries no
    information, so (unlike naive write-ups) we do **not** threshold it.  Instead we
    track the **MAP run length** ``r*_t = argmax_r P(r_t)``: between changes it
    climbs by 1 each window, and at a change it *collapses*.  We emit a change
    wherever ``r*`` drops by more than one while the previous run was at least
    ``min_run`` windows long (a guard against post-change jitter), locating the
    onset at ``date[t - r*_t]`` (the run length *is* the estimated time since the
    change, which sharpens localisation) and scoring it by the relative collapse
    ``(r*_{t-1} - r*_t) / r*_{t-1}``.  Adjacent alarms are de-duplicated.
    """
    Z, dates = prepare_matrix(series, representation, deseasonalize)
    n, d = Z.shape
    out: List[DetectedChange] = []
    if n == 0:
        return out
    H = 1.0 / float(hazard_run_length)

    # per-run-length sufficient stats: count and per-column sum (row r = run length)
    cnt = np.zeros((1, 1))
    ssum = np.zeros((1, d))
    R = np.array([1.0])
    rmap = np.empty(n, dtype=int)

    for t in range(n):
        x = Z[t]
        post_prec = 1.0 / prior_var + cnt / obs_var
        post_mean = (ssum / obs_var) / post_prec
        pred_var = 1.0 / post_prec + obs_var
        logpred = (-0.5 * np.log(2.0 * math.pi * pred_var)
                   - (x[None, :] - post_mean) ** 2 / (2.0 * pred_var)).sum(axis=1)
        logpred -= logpred.max()            # stability; cancels on normalisation
        pred = np.exp(logpred)

        growth = R * pred * (1.0 - H)
        cp = float((R * pred * H).sum())
        new_R = np.empty(len(R) + 1)
        new_R[0] = cp
        new_R[1:] = growth
        new_R /= new_R.sum()
        R = new_R
        rmap[t] = int(np.argmax(R))

        cnt = np.vstack([np.zeros((1, 1)), cnt + 1.0])
        ssum = np.vstack([np.zeros((1, d)), ssum + x[None, :]])

    for t in range(1, n):
        if rmap[t] < rmap[t - 1] - 1 and rmap[t - 1] >= min_run:
            onset = max(0, t - rmap[t])
            score = (rmap[t - 1] - rmap[t]) / rmap[t - 1]
            out.append(DetectedChange(date=dates[onset], score=float(score),
                                      method="bocpd"))
    return _dedupe_adjacent(out)


def _dedupe_adjacent(changes: List[DetectedChange], min_gap: int = 1) -> List[DetectedChange]:
    """Collapse runs of near-adjacent change alarms to their highest-scoring window."""
    if not changes:
        return changes
    changes = sorted(changes, key=lambda c: c.date)
    kept: List[DetectedChange] = [changes[0]]
    for c in changes[1:]:
        if (c.date - kept[-1].date).days <= min_gap * 7 + 1:
            if c.score > kept[-1].score:
                kept[-1] = c
        else:
            kept.append(c)
    return kept


# ===========================================================================
# recommended_detector — the configured default Phase 4 consumes
# ===========================================================================
@dataclass
class RecommendedDetector:
    """A fully-configured detector: build windows *this* way, run *this* method.

    The chosen ``method`` / ``representation`` / ``granularity`` / ``deseasonalize``
    (and any ``params``) are the benchmark winner (see DYNAMICS.md).  Call
    :meth:`detect` on a history or store to get ``[DetectedChange]``; Phase 4's
    trajectory/story UI is expected to consume exactly this object.
    """

    method: str
    representation: str
    granularity: str
    deseasonalize: bool
    min_events: int
    weighting: str
    params: Dict[str, object] = None  # type: ignore[assignment]

    def build(self, history_or_store) -> WindowSeries:
        return build_windows(
            history_or_store, granularity=self.granularity,
            min_events=self.min_events, weighting=self.weighting)

    def detect(self, history_or_store) -> List[DetectedChange]:
        params = self.params or {}
        if self.method == "baseline_scan":
            return baseline_scan(history_or_store,
                                 granularity=self.granularity)
        series = self.build(history_or_store)
        if self.method == "cusum":
            return cusum(series, self.representation, self.deseasonalize, **params)
        if self.method.startswith("pelt"):
            cost = self.method.split("_", 1)[1] if "_" in self.method else "l2"
            return pelt(series, self.representation, self.deseasonalize,
                        cost=cost, **params)
        if self.method == "bocpd":
            return bocpd(series, self.representation, self.deseasonalize, **params)
        raise ValueError(f"unknown method {self.method!r}")


def recommended_detector() -> RecommendedDetector:
    """The recommended default detector for Phase 4 (derived from the benchmark).

    Rationale lives in DYNAMICS.md; the choice is: **weekly windows, PELT with an
    ``rbf`` cost on the ``combined`` taste representation, NOT deseasonalised, with
    the penalty scaled to 1.5x the BIC default** (``penalty_scale=1.5``).

    Why each piece (all measured, see DYNAMICS.md):

    * ``pelt_rbf/combined`` is the strongest method x representation cell across all
      four generator settings.
    * ``deseasonalize=False`` — the pooled month-of-year correction is *worse than
      the disease* on 2-year histories: it raises the December false-positive rate
      (its month means are contaminated by whichever regime overlapped each
      December) and washes out genuine changes (30-seed F1 0.98 -> 0.76).
    * ``penalty_scale=1.5`` — at the plain BIC default the detector fires on ~3.3%
      of December seasonal spans (measured over 30 seeds; an earlier 8-seed run
      under-sampled this and read 0.0%), and on the *subtle-change* hard mode it
      fires on 50% of them.  Scaling the penalty 1.5x halves the default-mode
      December FP rate (1.7%) and eliminates it in subtle mode (0%), at ~0.01 F1 on
      default histories — because a transient December bump costs PELT *two*
      breakpoints while a persistent change costs one, a modest penalty boost
      selectively prices out the transient.

    Honest residual: ~1.7% of December spans still catch a false positive at this
    setting, and a December bump at the very end of a history is fundamentally
    indistinguishable from a persistent change (no reversion data).  PELT-``l2`` on
    ``combined`` is the more interpretable alternative (piecewise-constant means)
    but is weaker on subtle changes.
    """
    return RecommendedDetector(
        method="pelt_rbf", representation="combined", granularity="week",
        deseasonalize=False, min_events=30, weighting="plays",
        params={"penalty": None, "penalty_scale": 1.5})
