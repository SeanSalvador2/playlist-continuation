"""Classical-statistics layer over a :class:`~playlistcont.history.store.HistoryStore`.

This is Phase 2: rigorous, plain-English answers to *"did my listening actually
change?"* — and, just as importantly, honest *"no, that's noise"* answers.  Every
function here is a pure translation of a statistical question into arrays pulled from
the three-table store, returning plain dicts ready to hand to ``json`` (the same
contract as :mod:`playlistcont.analytics.queries`).

WHY EFFECT SIZES ARE NOT OPTIONAL (read this before trusting any p-value here)
-----------------------------------------------------------------------------
A personal listening history has **thousands of plays**.  At that sample size a
two-sided test will call almost *any* difference "significant": a valence shift of
0.01 on a 0..1 axis, a skip-rate wobble of half a percentage point — all p < 0.001.
**A bare p-value is therefore meaningless as a "did my taste change?" signal.**  So
every tested metric in this module carries an effect size, and the headline entry point
(:func:`significant_shifts`) refuses to report a change unless it clears BOTH a
false-discovery-controlled q-value AND a minimum effect-size floor (|Cohen's d| >= 0.2,
Cramér's V >= 0.1, |Δproportion| >= 0.05, epsilon² >= 0.01).  "Statistically solid AND
big enough to matter" is the product; p alone is not.

CONVENTIONS (documented once, used everywhere)
----------------------------------------------
* **Windows** are inclusive calendar-date bounds (``date`` / ISO string / ``None`` for
  unbounded), matching :mod:`~playlistcont.analytics.queries`.  "A" is the *earlier /
  baseline* window, "B" the *later / recent* window; every direction and every signed
  effect is **B relative to A** (positive = B is higher), so "rose"/"fell" reads
  naturally.
* **Cohen's d** uses the classic **pooled** standard deviation
  ``s_p = sqrt(((n_a-1) s_a^2 + (n_b-1) s_b^2) / (n_a + n_b - 2))`` with sample variance
  (``ddof=1``); ``d = (mean_b - mean_a) / s_p``.
* **Rank-biserial r** (from the Mann-Whitney U of window A) is reported B-relative:
  ``r = 1 - 2 U_a / (n_a n_b)`` (positive = B stochastically larger).
* **Two-proportion z** is the pooled-variance test; its effect is the raw proportion
  difference ``p_b - p_a``.
* **Cramér's V** for the 2×G genre table is ``sqrt(chi2 / N)`` (``min(r,c)-1 = 1``).
* **Genre bucket** of a play = ``argmax`` of that track's genre axis block (its dominant
  genre).  The chi-square guards expected counts: any bucket with expected < 5 in either
  window is collapsed into an ``"other"`` column; if the table is still invalid (< 2
  columns, or ``other`` itself < 5 expected) the test is skipped, not faked.
* **Hour bands**: morning ``05:00-11:59``, afternoon ``12:00-16:59``, evening
  ``17:00-21:59``, night ``22:00-04:59``.
* **Welch's ANOVA** is the real Welch F (unequal-variance), computed from the group
  ``n_i, mean_i, var_i`` — NOT ``scipy.stats.f_oneway`` (which assumes equal variances).
  See :func:`welch_anova`.  It is paired with Kruskal-Wallis and an epsilon-squared
  effect size (``H / (n - 1)``).
* **Minimum n**: a metric needs >= 30 plays-with-features per window (>= 30 skip-flagged
  plays for the skip test; >= 15 plays per group for ANOVA).  Below that the metric is
  reported as ``"insufficient data"`` — **never silently dropped**.

CALIBRATION HEADLINE (measured by ``experiments/exp_stats_calibration.py``; 24 seeds)
------------------------------------------------------------------------------------
On STATIONARY synthetic histories (single planted regime, no traps, no seasonal), an
adjacent-month scan produces, across every individual axis test, a raw-p false-positive
rate at alpha=0.05 of **~6.5%** (near the nominal 5%; the small excess is genuine
track-selection drift as heavy-rotation favourites concentrate within the fixed
mixture) — yet **0.0%** of month boundaries survive BH + the effect-size floor.  On
DEFAULT histories (planted regime changes present) the same corrected scan flags
**100%** of planted regime transitions within +/-1 month (75/75 across seeds).  In one
line: the correction machinery turns a scan that would cry wolf on ~1-in-15 raw tests
into one that is silent on stationary data and still catches every real change.  These
are the numbers Phase 3's fancier detectors have to beat.
"""
from __future__ import annotations

import math
from datetime import date, datetime, timedelta
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
from scipy import stats

from ..data.schema import GENRES, SCALAR_AXES
from ..history.store import HistoryStore, feature_column
from .queries import resolve_tz, skip_reliable_from

DateLike = Union[str, date, datetime, None]

SCALAR_COLUMNS: List[str] = [feature_column(a) for a in SCALAR_AXES]
GENRE_COLUMNS: List[str] = [feature_column(f"genre:{g}") for g in GENRES]

# ---- gating constants (documented in the module docstring) ----------------- #
MIN_N: int = 30          # plays-with-features required per window for an axis test
MIN_N_PROP: int = 30     # denominator required per window for a proportion test
MIN_GROUP_N: int = 15    # plays required per group for the habit ANOVA
MIN_GROUPS: int = 2

COHEN_D_FLOOR: float = 0.2      # Cohen's "small"
RANK_BISERIAL_FLOOR: float = 0.2
CRAMERS_V_FLOOR: float = 0.1
PROP_DIFF_FLOOR: float = 0.05   # 5 percentage points — a change you'd actually notice
EPSILON_SQ_FLOOR: float = 0.01

DEFAULT_Q: float = 0.05

_EFFECT_FLOOR: Dict[str, float] = {
    "cohen_d": COHEN_D_FLOOR,
    "rank_biserial": RANK_BISERIAL_FLOOR,
    "cramers_v": CRAMERS_V_FLOOR,
    "prop_diff": PROP_DIFF_FLOOR,
    "epsilon_sq": EPSILON_SQ_FLOOR,
}

# hour-band bounds (start hour inclusive); night wraps midnight.
_HOUR_BANDS: List[Tuple[str, int, int]] = [
    ("morning", 5, 12),      # 05:00-11:59
    ("afternoon", 12, 17),   # 12:00-16:59
    ("evening", 17, 22),     # 17:00-21:59
]
_WEEKDAY_LABELS = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]


# ===========================================================================
# window helpers (mirrors queries.py so the two layers agree on windowing)
# ===========================================================================
def _as_date(d: DateLike) -> Optional[date]:
    if d is None or d == "":
        return None
    if isinstance(d, datetime):
        return d.date()
    if isinstance(d, date):
        return d
    return date.fromisoformat(str(d)[:10])


# ===========================================================================
# Benjamini-Hochberg FDR (implemented here to keep deps light; no statsmodels)
# ===========================================================================
def bh_adjust(pvals: Sequence[float]) -> List[float]:
    """Benjamini-Hochberg step-up adjusted p-values (q-values).

    Given raw p-values ``p_1..p_m``, sort ascending, form ``q_(i) = p_(i) * m / i``,
    then enforce monotonic non-decreasing q-values from the largest rank down and clip
    to ``[0, 1]``.  Returns q-values in the **original input order**.  An empty input
    returns ``[]``.

    This is the whole of the FDR machinery — ten lines, no ``statsmodels``.  It is
    pinned against hand-computed fixtures (and cross-checked against
    ``scipy.stats.false_discovery_control``) in ``tests/test_analytics_stats.py``.
    """
    m = len(pvals)
    if m == 0:
        return []
    p = np.asarray(pvals, dtype=float)
    order = np.argsort(p, kind="mergesort")   # stable: ties keep input order
    ranks = np.arange(1, m + 1)
    q_sorted = p[order] * m / ranks
    # enforce monotonicity from the largest p downward, then clip
    q_sorted = np.minimum.accumulate(q_sorted[::-1])[::-1]
    q_sorted = np.clip(q_sorted, 0.0, 1.0)
    out = np.empty(m, dtype=float)
    out[order] = q_sorted
    return [float(v) for v in out]


# ===========================================================================
# effect-size primitives (each hand-verified in the tests)
# ===========================================================================
def cohen_d(a: np.ndarray, b: np.ndarray) -> float:
    """Cohen's d (B relative to A) with the pooled-SD convention (see module docstring)."""
    na, nb = len(a), len(b)
    if na < 2 or nb < 2:
        return 0.0
    va, vb = float(np.var(a, ddof=1)), float(np.var(b, ddof=1))
    sp = math.sqrt(((na - 1) * va + (nb - 1) * vb) / (na + nb - 2))
    if sp == 0.0:
        return 0.0
    return (float(np.mean(b)) - float(np.mean(a))) / sp


def rank_biserial(u_a: float, n_a: int, n_b: int) -> float:
    """B-relative rank-biserial correlation from the Mann-Whitney U of window A."""
    if n_a == 0 or n_b == 0:
        return 0.0
    return 1.0 - 2.0 * u_a / (n_a * n_b)


def two_proportion_z(x_a: int, n_a: int, x_b: int, n_b: int) -> Tuple[float, float, float]:
    """Pooled two-proportion z-test.  Returns ``(z, p_two_sided, p_b - p_a)``."""
    if n_a == 0 or n_b == 0:
        return 0.0, 1.0, 0.0
    pa, pb = x_a / n_a, x_b / n_b
    pool = (x_a + x_b) / (n_a + n_b)
    se = math.sqrt(pool * (1.0 - pool) * (1.0 / n_a + 1.0 / n_b))
    if se == 0.0:
        return 0.0, 1.0, pb - pa
    z = (pb - pa) / se
    p = 2.0 * float(stats.norm.sf(abs(z)))
    return z, p, pb - pa


def welch_anova(groups: Sequence[np.ndarray]) -> dict:
    """Welch's one-way ANOVA (unequal variances) computed from group statistics.

    This is the *Welch* F, **not** the equal-variance ``scipy.stats.f_oneway``.  Formula
    (Welch 1951): with ``w_i = n_i / s_i^2`` (sample variance), ``W = sum w_i``, weighted
    grand mean ``x* = sum(w_i m_i) / W``::

        A  = sum(w_i (m_i - x*)^2) / (k - 1)
        T  = sum((1 - w_i/W)^2 / (n_i - 1))
        B  = 2 (k - 2) / (k^2 - 1) * T
        F  = A / (1 + B)
        df1 = k - 1
        df2 = 1 / ( 3 / (k^2 - 1) * T )
        p   = f.sf(F, df1, df2)

    Returns ``{F, df1, df2, p, k}`` (``F``/``p`` are ``nan`` when fewer than two groups
    have >= 2 observations and positive variance).  Pinned against the published
    hair-colour / pain-threshold example in the tests.
    """
    valid = [np.asarray(g, dtype=float) for g in groups
             if len(g) >= 2 and np.var(g, ddof=1) > 0]
    k = len(valid)
    if k < 2:
        return {"F": float("nan"), "df1": float("nan"), "df2": float("nan"),
                "p": float("nan"), "k": k}
    n = np.array([len(g) for g in valid], dtype=float)
    m = np.array([g.mean() for g in valid], dtype=float)
    v = np.array([g.var(ddof=1) for g in valid], dtype=float)
    w = n / v
    W = w.sum()
    xbar = float((w * m).sum() / W)
    A = float((w * (m - xbar) ** 2).sum() / (k - 1))
    T = float((((1.0 - w / W) ** 2) / (n - 1.0)).sum())
    B = (2.0 * (k - 2) / (k ** 2 - 1)) * T
    F = A / (1.0 + B)
    df1 = float(k - 1)
    df2 = 1.0 / ((3.0 / (k ** 2 - 1)) * T) if T > 0 else float("inf")
    p = float(stats.f.sf(F, df1, df2))
    return {"F": float(F), "df1": df1, "df2": float(df2), "p": p, "k": k}


def epsilon_squared_kw(h: float, n: int) -> float:
    """Epsilon-squared effect size for Kruskal-Wallis: ``H / (n - 1)`` (0..1)."""
    if n <= 1:
        return 0.0
    return float(h) / (n - 1)


# ===========================================================================
# data loading — pull per-play frames ONCE, slice by date in pandas
# ===========================================================================
def _feature_frame(store: HistoryStore, tz: Optional[str] = None):
    """Per-play rows that have features: ``d`` (date), scalar cols, genre cols, genre_bucket.

    ``hour``/``weekday`` are derived from the timestamp **in the display timezone**
    (:func:`~playlistcont.analytics.queries.resolve_tz`), so weekday/hour-band habit
    grouping reflects local wall-clock time rather than the stored UTC hour; DuckDB
    ``AT TIME ZONE`` handles DST per-timestamp.  ``tz="UTC"`` reproduces the stored
    columns exactly.  The stored events table is not modified.
    """
    zone = resolve_tz(tz)
    cols = ", ".join(f"f.{c} AS {c}" for c in SCALAR_COLUMNS + GENRE_COLUMNS)
    df = store.query(
        f"""
        SELECT e.date AS d,
               hour(e.ts AT TIME ZONE '{zone}')         AS hour,
               (isodow(e.ts AT TIME ZONE '{zone}') - 1) AS weekday,
               {cols}
        FROM events e JOIN track_features f ON e.track_uri = f.uri
        ORDER BY e.event_id
        """
    )
    if len(df):
        df["d"] = df["d"].map(_to_date)
        g = df[GENRE_COLUMNS].to_numpy(dtype=float)
        df["genre_bucket"] = [GENRES[i] for i in g.argmax(axis=1)] if len(g) else []
    else:
        df["genre_bucket"] = []
    return df


def _event_frame(store: HistoryStore):
    """Per-play rows for behaviour metrics: ``d`` (date), ``skipped``, ``is_first``."""
    df = store.query(
        """
        WITH firsts AS (
          SELECT track_uri, MIN(event_id) AS fe FROM events GROUP BY track_uri
        )
        SELECT e.date AS d, e.skipped AS skipped, (e.event_id = f.fe) AS is_first
        FROM events e JOIN firsts f ON e.track_uri = f.track_uri
        ORDER BY e.event_id
        """
    )
    if len(df):
        df["d"] = df["d"].map(_to_date)
    return df


def _to_date(x) -> date:
    """Coerce a DuckDB DATE/TIMESTAMP (date, datetime or pandas Timestamp) to ``date``.

    ``datetime`` is a subclass of ``date``, so the order of these checks matters.
    """
    if isinstance(x, datetime):
        return x.date()
    if isinstance(x, date):
        return x
    import pandas as pd
    return pd.Timestamp(x).date()


def _slice(df, start: Optional[date], end: Optional[date]):
    if not len(df):
        return df
    mask = np.ones(len(df), dtype=bool)
    d = df["d"].to_numpy()
    if start is not None:
        mask &= d >= start
    if end is not None:
        mask &= d <= end
    return df[mask]


def _window_days(start: Optional[date], end: Optional[date], sub) -> int:
    """Calendar length of the window (requested bounds if given, else observed span)."""
    if start is not None and end is not None:
        return (end - start).days + 1
    if not len(sub):
        return 0
    lo, hi = min(sub["d"]), max(sub["d"])
    return (hi - lo).days + 1


def _behavior(sub, skip_from: Optional[date] = None) -> dict:
    """Behaviour counts for one window slice of the event frame.

    Skip counts are taken only over plays on/after ``skip_from`` (the skip-reliable start
    from :func:`~playlistcont.analytics.queries.skip_reliable_from`); plays before it carry
    a uniform ``skipped=false`` that predates Spotify logging skips and must not be counted.
    When ``skip_from`` is ``None`` (no reliable window at all) no play is skip-flagged, so
    the skip test degrades to "not recorded" (insufficient) rather than a fake 0%.
    """
    n = len(sub)
    if not n:
        return {"plays": 0, "skip_flagged": 0, "skips": 0, "firsts": 0}
    if skip_from is None:
        flagged = skips = 0
    else:
        rel = sub[sub["d"].to_numpy() >= skip_from]
        skipped = rel["skipped"]
        flagged = int(skipped.notna().sum())
        skips = int((skipped == True).sum())  # noqa: E712  (nullable-bool safe)
    firsts = int((sub["is_first"] == True).sum())  # noqa: E712
    return {"plays": n, "skip_flagged": flagged, "skips": skips, "firsts": firsts}


# ===========================================================================
# genre chi-square with the expected-count guard
# ===========================================================================
def _genre_chi2(counts_a: Dict[str, int], counts_b: Dict[str, int]) -> Optional[dict]:
    """Chi-square of the 2×G genre-bucket table with expected-count collapsing.

    Returns ``{chi2, p, cramers_v, dof, n, buckets}`` or ``None`` when the table cannot
    support a valid test (too few buckets / expected counts too small even after
    collapsing rare buckets into ``other``).
    """
    genres = [g for g in GENRES if (counts_a.get(g, 0) + counts_b.get(g, 0)) > 0]
    ra = sum(counts_a.get(g, 0) for g in genres)
    rb = sum(counts_b.get(g, 0) for g in genres)
    n = ra + rb
    if n == 0 or ra == 0 or rb == 0:
        return None

    keep: List[str] = []
    other_a = other_b = 0
    for g in genres:
        ca, cb = counts_a.get(g, 0), counts_b.get(g, 0)
        col = ca + cb
        exp_a = ra * col / n
        exp_b = rb * col / n
        if exp_a < 5 or exp_b < 5:
            other_a += ca
            other_b += cb
        else:
            keep.append(g)

    obs_a = [counts_a.get(g, 0) for g in keep]
    obs_b = [counts_b.get(g, 0) for g in keep]
    labels = list(keep)
    if other_a + other_b > 0:
        col = other_a + other_b
        if ra * col / n >= 5 and rb * col / n >= 5:
            obs_a.append(other_a)
            obs_b.append(other_b)
            labels.append("other")
        # else: the residual "other" bucket is itself too thin — drop it
    if len(labels) < 2:
        return None

    table = np.array([obs_a, obs_b], dtype=float)
    chi2, p, dof, _exp = stats.chi2_contingency(table, correction=False)
    v = math.sqrt(chi2 / n)  # min(r,c)-1 == 1 for a 2-row table
    return {"chi2": float(chi2), "p": float(p), "cramers_v": float(v),
            "dof": int(dof), "n": int(n), "buckets": labels}


# ===========================================================================
# record builders
# ===========================================================================
def _direction(delta: float) -> str:
    if delta > 0:
        return "up"
    if delta < 0:
        return "down"
    return "flat"


def _axis_records(a_feat, b_feat) -> List[dict]:
    """Welch-t and Mann-Whitney records for every scalar axis (or an insufficient stub)."""
    out: List[dict] = []
    for axis, col in zip(SCALAR_AXES, SCALAR_COLUMNS):
        a = a_feat[col].to_numpy(dtype=float) if len(a_feat) else np.empty(0)
        b = b_feat[col].to_numpy(dtype=float) if len(b_feat) else np.empty(0)
        na, nb = len(a), len(b)
        mean_a = float(np.mean(a)) if na else None
        mean_b = float(np.mean(b)) if nb else None
        if na < MIN_N or nb < MIN_N:
            out.append({
                "metric": axis, "kind": "welch_t", "n_a": na, "n_b": nb,
                "stat": None, "p_raw": None, "effect": None, "effect_name": "cohen_d",
                "direction": None, "mean_a": mean_a, "mean_b": mean_b,
                "insufficient": True,
            })
            continue
        delta = mean_b - mean_a
        d = cohen_d(a, b)
        t = stats.ttest_ind(b, a, equal_var=False)
        out.append({
            "metric": axis, "kind": "welch_t", "n_a": na, "n_b": nb,
            "stat": float(t.statistic), "p_raw": float(t.pvalue),
            "effect": float(d), "effect_name": "cohen_d",
            "direction": _direction(delta), "mean_a": mean_a, "mean_b": mean_b,
            "insufficient": False,
        })
        u = stats.mannwhitneyu(a, b, alternative="two-sided")
        rb = rank_biserial(float(u.statistic), na, nb)
        out.append({
            "metric": axis, "kind": "mannwhitney", "n_a": na, "n_b": nb,
            "stat": float(u.statistic), "p_raw": float(u.pvalue),
            "effect": float(rb), "effect_name": "rank_biserial",
            "direction": _direction(delta), "mean_a": mean_a, "mean_b": mean_b,
            "insufficient": False,
        })
    return out


def _behavior_records(a_beh: dict, b_beh: dict, days_a: int, days_b: int) -> List[dict]:
    out: List[dict] = []

    # plays/day — descriptive ratio, NOT a hypothesis test (no p, stays out of the FDR family)
    ppd_a = a_beh["plays"] / days_a if days_a else 0.0
    ppd_b = b_beh["plays"] / days_b if days_b else 0.0
    ratio = (ppd_b / ppd_a) if ppd_a else None
    out.append({
        "metric": "plays_per_day", "kind": "ratio",
        "n_a": a_beh["plays"], "n_b": b_beh["plays"],
        "stat": ratio, "p_raw": None,
        "effect": (ratio - 1.0) if ratio is not None else None, "effect_name": "rate_ratio",
        "direction": _direction((ppd_b - ppd_a)), "mean_a": ppd_a, "mean_b": ppd_b,
        "insufficient": days_a == 0 or days_b == 0,
    })

    # skip rate — two-proportion z over skip-flagged plays
    fa, fb = a_beh["skip_flagged"], b_beh["skip_flagged"]
    if fa >= MIN_N_PROP and fb >= MIN_N_PROP:
        z, p, diff = two_proportion_z(a_beh["skips"], fa, b_beh["skips"], fb)
        out.append({
            "metric": "skip_rate", "kind": "two_prop_z", "n_a": fa, "n_b": fb,
            "stat": float(z), "p_raw": float(p), "effect": float(diff),
            "effect_name": "prop_diff", "direction": _direction(diff),
            "mean_a": a_beh["skips"] / fa, "mean_b": b_beh["skips"] / fb,
            "insufficient": False,
        })
    else:
        out.append({
            "metric": "skip_rate", "kind": "two_prop_z", "n_a": fa, "n_b": fb,
            "stat": None, "p_raw": None, "effect": None, "effect_name": "prop_diff",
            "direction": None,
            "mean_a": (a_beh["skips"] / fa) if fa else None,
            "mean_b": (b_beh["skips"] / fb) if fb else None,
            "insufficient": True,
        })

    # discovery rate — two-proportion z over all plays (firsts / plays)
    pa_, pb_ = a_beh["plays"], b_beh["plays"]
    if pa_ >= MIN_N_PROP and pb_ >= MIN_N_PROP:
        z, p, diff = two_proportion_z(a_beh["firsts"], pa_, b_beh["firsts"], pb_)
        out.append({
            "metric": "discovery_rate", "kind": "two_prop_z", "n_a": pa_, "n_b": pb_,
            "stat": float(z), "p_raw": float(p), "effect": float(diff),
            "effect_name": "prop_diff", "direction": _direction(diff),
            "mean_a": a_beh["firsts"] / pa_, "mean_b": b_beh["firsts"] / pb_,
            "insufficient": False,
        })
    else:
        out.append({
            "metric": "discovery_rate", "kind": "two_prop_z", "n_a": pa_, "n_b": pb_,
            "stat": None, "p_raw": None, "effect": None, "effect_name": "prop_diff",
            "direction": None,
            "mean_a": (a_beh["firsts"] / pa_) if pa_ else None,
            "mean_b": (b_beh["firsts"] / pb_) if pb_ else None,
            "insufficient": True,
        })
    return out


def _genre_record(a_feat, b_feat) -> dict:
    ca = a_feat["genre_bucket"].value_counts().to_dict() if len(a_feat) else {}
    cb = b_feat["genre_bucket"].value_counts().to_dict() if len(b_feat) else {}
    ca = {str(k): int(v) for k, v in ca.items()}
    cb = {str(k): int(v) for k, v in cb.items()}
    na, nb = len(a_feat), len(b_feat)
    if na < MIN_N or nb < MIN_N:
        return {"metric": "genre_mix", "kind": "chi2", "n_a": na, "n_b": nb,
                "stat": None, "p_raw": None, "effect": None, "effect_name": "cramers_v",
                "direction": None, "mean_a": None, "mean_b": None, "insufficient": True}
    res = _genre_chi2(ca, cb)
    if res is None:
        return {"metric": "genre_mix", "kind": "chi2", "n_a": na, "n_b": nb,
                "stat": None, "p_raw": None, "effect": None, "effect_name": "cramers_v",
                "direction": None, "mean_a": None, "mean_b": None, "insufficient": True}
    return {"metric": "genre_mix", "kind": "chi2", "n_a": na, "n_b": nb,
            "stat": res["chi2"], "p_raw": res["p"], "effect": res["cramers_v"],
            "effect_name": "cramers_v", "direction": None,
            "mean_a": None, "mean_b": None, "insufficient": False,
            "dof": res["dof"], "buckets": res["buckets"]}


# ===========================================================================
# core comparison (works on pre-sliced frames; SQL layer feeds it)
# ===========================================================================
def _compare_core(a_feat, b_feat, a_beh, b_beh, days_a, days_b,
                  metrics: str = "all") -> List[dict]:
    records = _axis_records(a_feat, b_feat)
    if metrics == "all":
        records += _behavior_records(a_beh, b_beh, days_a, days_b)
        records.append(_genre_record(a_feat, b_feat))
    return records


def compare_windows(
    store: HistoryStore,
    a_start: DateLike, a_end: DateLike,
    b_start: DateLike, b_end: DateLike,
    metrics: str = "all",
) -> dict:
    """Compare listening in window A (baseline) vs window B (recent), metric by metric.

    For every scalar taste axis: a Welch t-test **and** a Mann-Whitney U (Cohen's d and
    B-relative rank-biserial r respectively).  For listening behaviour: a descriptive
    plays/day ratio, a skip-rate two-proportion z-test, and a discovery-rate
    two-proportion z-test.  For genre: a chi-square over dominant-genre play counts with
    Cramér's V (expected-count guarded; see :func:`_genre_chi2`).  ``metrics="axes"``
    restricts the output to the scalar-axis tests (used by :func:`adjacent_scan`).

    Returns ``{"window_a", "window_b", "metrics": [record, ...]}`` where each record is
    ``{metric, kind, n_a, n_b, stat, p_raw, effect, effect_name, direction, mean_a,
    mean_b, insufficient}``.  All directions/effects are B relative to A.  No FDR
    correction is applied here — that is :func:`significant_shifts`'s job.
    """
    feat = _feature_frame(store)
    ev = _event_frame(store)
    skip_from = skip_reliable_from(store)
    a_s, a_e = _as_date(a_start), _as_date(a_end)
    b_s, b_e = _as_date(b_start), _as_date(b_end)

    a_feat, b_feat = _slice(feat, a_s, a_e), _slice(feat, b_s, b_e)
    a_ev, b_ev = _slice(ev, a_s, a_e), _slice(ev, b_s, b_e)
    records = _compare_core(
        a_feat, b_feat, _behavior(a_ev, skip_from), _behavior(b_ev, skip_from),
        _window_days(a_s, a_e, a_ev), _window_days(b_s, b_e, b_ev), metrics=metrics,
    )
    return {
        "window_a": {"start": a_s.isoformat() if a_s else None,
                     "end": a_e.isoformat() if a_e else None},
        "window_b": {"start": b_s.isoformat() if b_s else None,
                     "end": b_e.isoformat() if b_e else None},
        "metrics": records,
    }


# ===========================================================================
# plain-English rendering (name_flavor template style)
# ===========================================================================
_AXIS_CLAUSE: Dict[str, Tuple[str, str]] = {
    "tempo": ("you're leaning into faster tracks", "you're leaning into slower tracks"),
    "energy": ("your listening got more intense", "your listening got calmer"),
    "valence": ("your recent listening is noticeably happier",
                "your recent listening is noticeably moodier"),
    "acousticness": ("you're reaching for more acoustic material",
                     "you're reaching for more electronic material"),
    "lyrical_depth": ("the lyrics you play got denser and more meaningful",
                      "the lyrics you play got lighter"),
}
_AXIS_LABEL = {
    "tempo": "Tempo", "energy": "Energy", "valence": "Valence",
    "acousticness": "Acousticness", "lyrical_depth": "Lyrical depth",
}


def _sentence(rec: dict, q: float) -> str:
    """A plain-English one-liner for a surviving shift (name_flavor template style)."""
    metric, eff = rec["metric"], rec["effect"]
    if metric in _AXIS_CLAUSE:
        up, down = _AXIS_CLAUSE[metric]
        delta = (rec["mean_b"] or 0.0) - (rec["mean_a"] or 0.0)
        verb = "rose" if delta > 0 else "fell"
        clause = up if delta > 0 else down
        d = rec.get("cohen_d_effect", eff)
        return (f"{_AXIS_LABEL[metric]} {verb} {abs(delta):.2f} "
                f"(d={d:+.2f}, q={q:.3f}): {clause}.")
    if metric == "genre_mix":
        return (f"Your genre balance shifted (Cramér's V={eff:.2f}, q={q:.3f}): "
                f"the mix of genres you played changed between the two windows.")
    if metric == "skip_rate":
        pp = abs(eff) * 100
        verb = "rose" if eff > 0 else "fell"
        clause = "you bailed on more tracks recently" if eff > 0 else \
                 "you're sticking with tracks more"
        return (f"Skip rate {verb} {pp:.0f} points (Δ={eff:+.2f}, q={q:.3f}): {clause}.")
    if metric == "discovery_rate":
        pp = abs(eff) * 100
        verb = "rose" if eff > 0 else "fell"
        clause = "more of your plays were brand-new tracks" if eff > 0 else \
                 "you leaned on familiar tracks"
        return (f"Discovery rate {verb} {pp:.0f} points (Δ={eff:+.2f}, q={q:.3f}): {clause}.")
    return f"{metric} changed (q={q:.3f})."


def _passes_floor(rec: dict) -> bool:
    floor = _EFFECT_FLOOR.get(rec["effect_name"])
    if floor is None or rec["effect"] is None:
        return False
    return abs(rec["effect"]) >= floor


# ===========================================================================
# significant_shifts — the headline entry point
# ===========================================================================
def significant_shifts(
    store: HistoryStore,
    a_start: DateLike, a_end: DateLike,
    b_start: DateLike, b_end: DateLike,
    q_threshold: float = DEFAULT_Q,
    metrics: str = "all",
) -> dict:
    """FDR-corrected, effect-floored, plain-English answer to "did my taste change?".

    Wraps :func:`compare_windows`, Benjamini-Hochberg-corrects across the **whole tested
    family** at once (:func:`bh_adjust`), and reports only metrics whose q-value clears
    ``q_threshold`` (default 0.05) **and** whose effect size clears the documented floor
    for its kind.  Surviving shifts are rendered as sentences and sorted by effect
    magnitude.  Metrics that lacked the minimum sample size are returned under
    ``"insufficient"`` — never silently dropped.  Descriptive-only metrics (plays/day)
    are excluded from the FDR family (they carry no p-value) but surfaced under
    ``"descriptive"``.

    For a scalar axis the Welch-t and Mann-Whitney records both live in the FDR family
    (so the family size is honest), but a surviving axis yields a **single** sentence:
    the t-test framing (with Cohen's d) is preferred and ``corroborated`` records whether
    the rank test also survived.
    """
    comp = compare_windows(store, a_start, a_end, b_start, b_end, metrics=metrics)
    records = comp["metrics"]

    tested = [r for r in records if r["p_raw"] is not None]
    insufficient = [r for r in records if r.get("insufficient")]
    descriptive = [r for r in records
                   if r["p_raw"] is None and not r.get("insufficient")]

    qvals = bh_adjust([r["p_raw"] for r in tested])
    for r, qv in zip(tested, qvals):
        r["q"] = qv

    survivors = [r for r in tested if r["q"] < q_threshold and _passes_floor(r)]

    # collapse the two scalar-axis tests into one shift per axis
    by_metric: Dict[str, dict] = {}
    for r in survivors:
        m = r["metric"]
        if m in _AXIS_LABEL:
            prev = by_metric.get(m)
            # prefer the Welch-t record (carries Cohen's d) for the sentence
            if prev is None:
                by_metric[m] = dict(r)
            if r["kind"] == "welch_t":
                keep = dict(r)
                keep["corroborated"] = any(
                    s["metric"] == m and s["kind"] == "mannwhitney" for s in survivors)
                keep["cohen_d_effect"] = r["effect"]
                by_metric[m] = keep
            else:
                by_metric.setdefault(m, dict(r))
                by_metric[m]["corroborated"] = any(
                    s["metric"] == m and s["kind"] == "welch_t" for s in survivors)
        else:
            by_metric[m] = dict(r)

    shifts: List[dict] = []
    for m, r in by_metric.items():
        shifts.append({
            "metric": m, "kind": r["kind"],
            "sentence": _sentence(r, r["q"]),
            "effect": r["effect"], "effect_name": r["effect_name"],
            "q": r["q"], "p_raw": r["p_raw"], "direction": r["direction"],
            "mean_a": r["mean_a"], "mean_b": r["mean_b"],
            "n_a": r["n_a"], "n_b": r["n_b"],
            "corroborated": bool(r.get("corroborated", False)),
        })
    shifts.sort(key=lambda s: abs(s["effect"]), reverse=True)

    # de-dup insufficient by metric (the two axis tests share one stub-per-axis anyway)
    seen = set()
    insuff_out: List[dict] = []
    for r in insufficient:
        if r["metric"] in seen:
            continue
        seen.add(r["metric"])
        insuff_out.append({
            "metric": r["metric"], "kind": r["kind"],
            "n_a": r["n_a"], "n_b": r["n_b"],
            "reason": f"need >= {MIN_N} plays per window (had {r['n_a']} / {r['n_b']})",
        })

    return {
        "window_a": comp["window_a"], "window_b": comp["window_b"],
        "q_threshold": q_threshold,
        "family_size": len(tested),
        "n_survivors": len(shifts),
        "shifts": shifts,
        "insufficient": insuff_out,
        "descriptive": descriptive,
    }


# ===========================================================================
# habit_anova — do taste axes differ by weekday / hour-band / month?
# ===========================================================================
def _group_labels(feat, group_by: str) -> np.ndarray:
    if group_by == "weekday":
        return np.array([_WEEKDAY_LABELS[int(w)] for w in feat["weekday"]])
    if group_by == "hour_band":
        def band(h: int) -> str:
            for name, lo, hi in _HOUR_BANDS:
                if lo <= h < hi:
                    return name
            return "night"
        return np.array([band(int(h)) for h in feat["hour"]])
    if group_by == "month":
        return np.array([d.strftime("%Y-%m") for d in feat["d"]])
    raise ValueError(f"unknown group_by {group_by!r}; expected weekday|hour_band|month")


def habit_anova(
    store: HistoryStore,
    start: DateLike = None,
    end: DateLike = None,
    group_by: str = "weekday",
    tz: Optional[str] = None,
) -> dict:
    """Do the taste axes differ across ``group_by`` groups within the window?

    ``group_by`` is ``"weekday"`` (Mon..Sun), ``"hour_band"`` (morning/afternoon/
    evening/night — bounds in the module docstring) or ``"month"`` (calendar month).
    Weekday and hour-band groups are computed in the display timezone ``tz``
    (:func:`~playlistcont.analytics.queries.resolve_tz`; env ``PLAYLISTCONT_TZ`` / UTC by
    default), so "evening" means local evening, not UTC.  Per scalar axis: **Welch's
    ANOVA** (real unequal-variance F, :func:`welch_anova`) **and** Kruskal-Wallis, with an
    epsilon-squared effect size (``H/(n-1)``).  Welch p-values are BH-corrected across the
    five axes; a survivor needs ``q < 0.05`` and ``epsilon² >= 0.01``.  Groups with fewer
    than 15 plays are dropped; axes with fewer than two usable groups are reported as
    insufficient.  Survivors get plain-English summaries.
    """
    feat = _feature_frame(store, tz=tz)
    feat = _slice(feat, _as_date(start), _as_date(end))
    labels = _group_labels(feat, group_by) if len(feat) else np.array([])

    records: List[dict] = []
    welch_ps: List[float] = []
    tested_idx: List[int] = []
    for axis, col in zip(SCALAR_AXES, SCALAR_COLUMNS):
        rec: dict = {"axis": axis, "group_by": group_by}
        if not len(feat):
            rec.update({"insufficient": True, "reason": "no plays with features"})
            records.append(rec)
            continue
        vals = feat[col].to_numpy(dtype=float)
        groups: List[np.ndarray] = []
        group_means: Dict[str, float] = {}
        for g in sorted(set(labels)):
            gv = vals[labels == g]
            if len(gv) >= MIN_GROUP_N:
                groups.append(gv)
                group_means[str(g)] = float(gv.mean())
        if len(groups) < MIN_GROUPS:
            rec.update({"insufficient": True,
                        "reason": f"need >= {MIN_GROUPS} groups with >= {MIN_GROUP_N} plays"})
            records.append(rec)
            continue
        n_total = int(sum(len(g) for g in groups))
        wa = welch_anova(groups)
        h, kp = stats.kruskal(*groups)
        eps = epsilon_squared_kw(float(h), n_total)
        rec.update({
            "insufficient": False, "n": n_total, "k_groups": len(groups),
            "welch_F": wa["F"], "welch_df1": wa["df1"], "welch_df2": wa["df2"],
            "welch_p": wa["p"], "kruskal_H": float(h), "kruskal_p": float(kp),
            "epsilon_sq": eps, "group_means": group_means,
        })
        records.append(rec)
        tested_idx.append(len(records) - 1)
        welch_ps.append(wa["p"])

    qvals = bh_adjust(welch_ps)
    for idx, qv in zip(tested_idx, qvals):
        records[idx]["welch_q"] = qv

    survivors: List[dict] = []
    for idx in tested_idx:
        r = records[idx]
        if r["welch_q"] < DEFAULT_Q and r["epsilon_sq"] >= EPSILON_SQ_FLOOR:
            gm = r["group_means"]
            hi = max(gm, key=gm.get)
            lo = min(gm, key=gm.get)
            label = _AXIS_LABEL.get(r["axis"], r["axis"])
            r["summary"] = (
                f"{label} varies by {group_by.replace('_', ' ')} "
                f"(Welch F={r['welch_F']:.2f}, q={r['welch_q']:.3f}, ε²={r['epsilon_sq']:.2f}): "
                f"highest in {hi} ({gm[hi]:.2f}), lowest in {lo} ({gm[lo]:.2f}).")
            survivors.append({
                "axis": r["axis"], "summary": r["summary"],
                "welch_F": r["welch_F"], "welch_q": r["welch_q"],
                "epsilon_sq": r["epsilon_sq"], "group_means": gm,
            })
    survivors.sort(key=lambda s: s["epsilon_sq"], reverse=True)

    return {
        "group_by": group_by,
        "start": _as_date(start).isoformat() if _as_date(start) else None,
        "end": _as_date(end).isoformat() if _as_date(end) else None,
        "q_threshold": DEFAULT_Q,
        "axes": records,
        "survivors": survivors,
    }


# ===========================================================================
# adjacent_scan — the Phase-3 baseline change detector
# ===========================================================================
def _month_windows(first: date, last: date) -> List[Tuple[date, date, str]]:
    windows: List[Tuple[date, date, str]] = []
    y, m = first.year, first.month
    while (y < last.year) or (y == last.year and m <= last.month):
        start = date(y, m, 1)
        nxt = date(y + 1, 1, 1) if m == 12 else date(y, m + 1, 1)
        windows.append((start, nxt - timedelta(days=1), f"{y:04d}-{m:02d}"))
        y, m = (y + 1, 1) if m == 12 else (y, m + 1)
    return windows


def _week_windows(first: date, last: date) -> List[Tuple[date, date, str]]:
    windows: List[Tuple[date, date, str]] = []
    start = first - timedelta(days=first.weekday())  # Monday on/before first
    while start <= last:
        end = start + timedelta(days=6)
        windows.append((start, end, start.isoformat()))
        start = start + timedelta(days=7)
    return windows


def adjacent_scan(
    store: HistoryStore,
    granularity: str = "month",
    metrics: str = "axes",
    q_threshold: float = DEFAULT_Q,
) -> dict:
    """Baseline change detector: compare every pair of consecutive periods.

    THIS IS THE PHASE-3 BASELINE.  Phase 3's fancier change-point methods must beat the
    detector defined right here: slide a window over consecutive calendar periods
    (``granularity="month"`` or ``"week"``), run :func:`compare_windows` on each adjacent
    pair, and Benjamini-Hochberg-correct across the **entire scan at once** — because a
    scan is one big multiple-comparisons family (a 24-month history at 5 axes × 2 tests
    is ~230 tests; without a scan-wide correction ~5% would fire on pure noise).  A
    boundary is ``flagged`` as a candidate change point when any of its metrics clears
    ``q_threshold`` and its effect-size floor.

    ``metrics="axes"`` (default) scans only the scalar-taste-axis tests — the taste-
    change signal; ``metrics="all"`` also includes behaviour and genre.  Returns the
    timeline of boundaries with per-metric q-values and effects, plus the scan-wide
    ``family_size``.  Everything is loaded once and sliced in pandas, so a full scan is a
    handful of milliseconds.
    """
    feat = _feature_frame(store)
    ev = _event_frame(store)
    skip_from = skip_reliable_from(store)
    if not len(feat):
        return {"granularity": granularity, "metrics": metrics,
                "q_threshold": q_threshold, "family_size": 0, "boundaries": []}

    first = min(min(feat["d"]), min(ev["d"]))
    last = max(max(feat["d"]), max(ev["d"]))
    periods = (_month_windows(first, last) if granularity == "month"
               else _week_windows(first, last) if granularity == "week"
               else None)
    if periods is None:
        raise ValueError(f"unknown granularity {granularity!r}; expected month|week")

    # one compare per adjacent pair; collect every p-value into one family
    boundaries: List[dict] = []
    family_records: List[dict] = []
    for i in range(len(periods) - 1):
        a_s, a_e, a_lab = periods[i]
        b_s, b_e, b_lab = periods[i + 1]
        a_feat, b_feat = _slice(feat, a_s, a_e), _slice(feat, b_s, b_e)
        a_ev, b_ev = _slice(ev, a_s, a_e), _slice(ev, b_s, b_e)
        recs = _compare_core(
            a_feat, b_feat, _behavior(a_ev, skip_from), _behavior(b_ev, skip_from),
            _window_days(a_s, a_e, a_ev), _window_days(b_s, b_e, b_ev), metrics=metrics,
        )
        tested = [r for r in recs if r["p_raw"] is not None]
        family_records.extend(tested)
        boundaries.append({
            "index": i, "boundary": b_s.isoformat(),
            "a": {"start": a_s.isoformat(), "end": a_e.isoformat(), "label": a_lab},
            "b": {"start": b_s.isoformat(), "end": b_e.isoformat(), "label": b_lab},
            "metrics": tested,
        })

    qvals = bh_adjust([r["p_raw"] for r in family_records])
    for r, qv in zip(family_records, qvals):
        r["q"] = qv

    for bnd in boundaries:
        surv = [r for r in bnd["metrics"]
                if r["q"] < q_threshold and _passes_floor(r)]
        bnd["flagged"] = bool(surv)
        bnd["min_q"] = min((r["q"] for r in bnd["metrics"]), default=None)
        bnd["max_effect"] = max((abs(r["effect"]) for r in surv), default=0.0)
        bnd["survivors"] = [
            {"metric": r["metric"], "kind": r["kind"], "q": r["q"],
             "effect": r["effect"], "effect_name": r["effect_name"],
             "direction": r["direction"]}
            for r in surv
        ]

    return {
        "granularity": granularity, "metrics": metrics,
        "q_threshold": q_threshold,
        "family_size": len(family_records),
        "n_flagged": sum(1 for b in boundaries if b["flagged"]),
        "boundaries": boundaries,
    }
