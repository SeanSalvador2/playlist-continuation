"""Hand-verified tests for the Phase-2 classical-statistics layer.

Philosophy (matching ``tests/test_analytics_queries.py``): every statistic is checked
against a number computed *independently* — either pinned from a one-off ``scipy``
computation (documented inline) or a published worked example, or recomputed from the
raw event stream without going through the module under test.  All offline, all
deterministic.
"""
import math
from datetime import date, datetime, timezone

import numpy as np
import pytest
from scipy import stats as sp

from playlistcont.analytics import stats
from playlistcont.data.schema import N_AXES, SCALAR_AXES
from playlistcont.history.schema import HistoryTrack, ListenEvent, ListeningHistory
from playlistcont.history.store import HistoryStore
from playlistcont.history.synthetic import make_synthetic_history

UTC = timezone.utc


# ===========================================================================
# Benjamini-Hochberg
# ===========================================================================
def test_bh_equal_spacing_worked_example():
    # p = [.01,.02,.03,.04,.05], m=5:  q_(i) = p_(i) * 5 / i
    #   i=1: .01*5/1 = .05    i=2: .02*5/2 = .05  ...  i=5: .05*5/5 = .05
    # every adjusted value is exactly 0.05.
    q = stats.bh_adjust([0.01, 0.02, 0.03, 0.04, 0.05])
    assert all(abs(v - 0.05) < 1e-12 for v in q)


def test_bh_monotonicity_enforced():
    # p = [.005,.009,.05,.1], m=4.  Raw q_(i)=p_(i)*4/i = [.02,.018,.0667,.1].
    # The step-up rule enforces non-decreasing q from the top down, pulling the
    # smallest rank's .02 DOWN to its neighbour .018.  Expected: [.018,.018,.0667,.1].
    q = stats.bh_adjust([0.005, 0.009, 0.05, 0.1])
    assert q == pytest.approx([0.018, 0.018, 0.0666667, 0.1], abs=1e-6)


def test_bh_classic_benjamini_hochberg_1995():
    # The 15 p-values from Benjamini & Hochberg (1995), "Controlling the False
    # Discovery Rate", J. R. Statist. Soc. B, 57(1), Table 1 (Neuhaus et al. data).
    p = [0.0001, 0.0004, 0.0019, 0.0095, 0.0201, 0.0278, 0.0298, 0.0344,
         0.0459, 0.3240, 0.4262, 0.5719, 0.6528, 0.7590, 1.0000]
    # Pinned from a one-off computation (and cross-checked against scipy below).
    expected = [0.0015, 0.003, 0.0095, 0.035625, 0.0603, 0.063857, 0.063857,
                0.0645, 0.0765, 0.486, 0.581182, 0.714875, 0.753231, 0.813214, 1.0]
    q = stats.bh_adjust(p)
    assert q == pytest.approx(expected, abs=1e-5)
    # independent oracle: scipy's own BH implementation
    assert q == pytest.approx(
        list(sp.false_discovery_control(p, method="bh")), abs=1e-9)


def test_bh_matches_scipy_on_random_vector():
    rng = np.random.default_rng(0)
    p = rng.uniform(0, 1, size=40)
    assert stats.bh_adjust(p) == pytest.approx(
        list(sp.false_discovery_control(p, method="bh")), abs=1e-9)


def test_bh_empty_and_singleton():
    assert stats.bh_adjust([]) == []
    assert stats.bh_adjust([0.3]) == pytest.approx([0.3])


# ===========================================================================
# effect-size primitives (pinned from a documented scipy one-off)
# ===========================================================================
# a = [.2,.3,.25,.4,.35,.3], b = [.5,.55,.6,.45,.7,.65]
_A = np.array([0.2, 0.3, 0.25, 0.4, 0.35, 0.3])
_B = np.array([0.5, 0.55, 0.6, 0.45, 0.7, 0.65])


def test_cohen_d_pooled_matches_hand_value():
    # pooled SD convention; pinned = 3.316625 (mean_b - mean_a over pooled sd)
    assert stats.cohen_d(_A, _B) == pytest.approx(3.316625, abs=1e-5)
    # sign is B-relative: swapping windows flips the sign
    assert stats.cohen_d(_B, _A) == pytest.approx(-3.316625, abs=1e-5)


def test_welch_t_pinned():
    # scipy.stats.ttest_ind(b, a, equal_var=False): t=5.744563, p=0.000245
    t = sp.ttest_ind(_B, _A, equal_var=False)
    assert float(t.statistic) == pytest.approx(5.744563, abs=1e-5)
    assert float(t.pvalue) == pytest.approx(0.000245, abs=1e-5)


def test_rank_biserial_from_mwu():
    # mannwhitneyu(a,b) U_a = 0 (every a < every b); B-relative r = 1 - 2*0/(36) = 1.
    u = sp.mannwhitneyu(_A, _B, alternative="two-sided")
    assert float(u.statistic) == 0.0
    assert stats.rank_biserial(float(u.statistic), len(_A), len(_B)) == pytest.approx(1.0)
    # a fully above b -> U = n_a*n_b -> r = -1
    u2 = sp.mannwhitneyu(_B, _A, alternative="two-sided")
    assert stats.rank_biserial(float(u2.statistic), len(_B), len(_A)) == pytest.approx(-1.0)


def test_two_proportion_z_pinned():
    # xa=20/100, xb=35/100 -> z=2.375423, p=0.017529, diff=0.15
    z, p, diff = stats.two_proportion_z(20, 100, 35, 100)
    assert z == pytest.approx(2.375423, abs=1e-5)
    assert p == pytest.approx(0.017529, abs=1e-5)
    assert diff == pytest.approx(0.15, abs=1e-9)


def test_genre_chi2_and_cramers_v_pinned():
    # 2x3 table [[30,20,10],[10,25,25]]: chi2=16.984127, dof=2, V=sqrt(chi2/120)=0.376211
    counts_a = {"pop": 30, "rock": 20, "jazz": 10}
    counts_b = {"pop": 10, "rock": 25, "jazz": 25}
    res = stats._genre_chi2(counts_a, counts_b)
    assert res["chi2"] == pytest.approx(16.984127, abs=1e-5)
    assert res["dof"] == 2
    assert res["cramers_v"] == pytest.approx(0.376211, abs=1e-5)
    assert res["n"] == 120


def test_genre_chi2_collapses_rare_buckets_into_other():
    # 'jazz' and 'folk' each have expected < 5 in both rows individually, but together
    # exceed 5 -> collapsed into a kept 'other' column.
    counts_a = {"pop": 60, "rock": 50, "jazz": 3, "folk": 3}
    counts_b = {"pop": 55, "rock": 48, "jazz": 3, "folk": 3}
    res = stats._genre_chi2(counts_a, counts_b)
    assert "jazz" not in res["buckets"] and "folk" not in res["buckets"]
    assert "other" in res["buckets"]


def test_genre_chi2_invalid_returns_none():
    # only one bucket with any data -> no valid test
    assert stats._genre_chi2({"pop": 100}, {"pop": 80}) is None


# ===========================================================================
# Welch's ANOVA — pinned against a PUBLISHED worked example
# ===========================================================================
def test_welch_anova_hair_color_pain_threshold():
    # The classic hair-colour / pain-threshold dataset used in the pingouin
    # `welch_anova` documentation (McClave & Sincich).  Published result:
    #   F = 5.8901, ddof1 = 3, ddof2 = 8.3298, p-unc = 0.018813.
    groups = [
        np.array([62, 60, 71, 55, 48], float),   # light blond
        np.array([63, 57, 52, 41, 43], float),   # dark blond
        np.array([42, 50, 41, 37], float),       # light brunette
        np.array([32, 39, 51, 30, 35], float),   # dark brunette
    ]
    r = stats.welch_anova(groups)
    assert r["F"] == pytest.approx(5.8901, abs=1e-3)
    assert r["df1"] == 3
    assert r["df2"] == pytest.approx(8.3298, abs=1e-3)
    assert r["p"] == pytest.approx(0.018813, abs=1e-5)


def test_welch_anova_is_not_plain_anova():
    # With markedly unequal variances Welch's F must differ from the equal-variance
    # scipy.stats.f_oneway; this guards against silently shipping plain ANOVA.
    g1 = np.array([1.0, 1.1, 0.9, 1.05, 0.95] * 6)      # tiny variance
    g2 = np.array([0.0, 4.0, -3.0, 5.0, -2.0, 3.5] * 5)  # large variance
    g3 = np.array([2.0, 2.2, 1.8, 2.1, 1.9] * 6)
    welch = stats.welch_anova([g1, g2, g3])
    plain = sp.f_oneway(g1, g2, g3)
    assert abs(welch["F"] - float(plain.statistic)) > 1.0
    # Welch df2 is fractional, plain df2 = N - k is an integer
    assert welch["df2"] != round(welch["df2"])


def test_epsilon_squared_kw_pinned():
    # kruskal([1..5],[3..7],[6..10]): H=9.641081, n=15 -> eps2 = H/14 = 0.688649
    g = [[1, 2, 3, 4, 5], [3, 4, 5, 6, 7], [6, 7, 8, 9, 10]]
    H, _ = sp.kruskal(*g)
    assert stats.epsilon_squared_kw(float(H), 15) == pytest.approx(0.688649, abs=1e-5)


# ===========================================================================
# controlled-history builder for gating / wiring tests
# ===========================================================================
def _vec(scalars, genre_idx=0):
    """A length-N_AXES feature vector: given scalar block + a one-hot-ish genre."""
    v = np.zeros(N_AXES, dtype=np.float32)
    v[: len(SCALAR_AXES)] = scalars
    v[len(SCALAR_AXES) + genre_idx] = 1.0
    return v


def _build_store(rows):
    """rows: list of (date, scalar-vector, genre_idx, skipped).  One track per row."""
    tracks = {}
    events = []
    for i, (d, scalars, gi, skipped) in enumerate(rows):
        uri = f"spotify:track:ctl{i:05d}xx"
        tracks[uri] = HistoryTrack(
            track_uri=uri, track_name=f"T{i}", artist_name=f"Artist{i % 5}",
            album_name=f"Album{i % 7}", album_uri=f"spotify:album:a{i % 7}",
            features=_vec(scalars, gi),
        )
        ts = datetime(d.year, d.month, d.day, 12, 0, 0, tzinfo=UTC)
        events.append(ListenEvent(
            ts=ts, track_uri=uri, track_name=f"T{i}", artist_name=f"Artist{i % 5}",
            ms_played=200000, album_name=f"Album{i % 7}", skipped=skipped,
            platform="web_player"))
    events.sort(key=lambda e: e.ts)
    h = ListeningHistory(events=events, tracks=tracks, provenance="synthetic")
    return HistoryStore.from_history(h)


# ===========================================================================
# compare_windows — wiring cross-checked against independent recomputation
# ===========================================================================
def _independent_axis_arrays(h, start, end):
    cols = {ax: [] for ax in SCALAR_AXES}
    for e in h.events:
        d = e.ts.date()
        if start <= d <= end:
            fv = h.tracks[e.track_uri].features
            for j, ax in enumerate(SCALAR_AXES):
                cols[ax].append(float(fv[j]))
    return {ax: np.array(v) for ax, v in cols.items()}


def test_compare_windows_axis_records_match_independent_computation():
    h = make_synthetic_history(seed=3, n_days=240)
    st = HistoryStore.from_history(h)
    a = (date(2022, 1, 1), date(2022, 2, 15))
    b = (date(2022, 3, 1), date(2022, 4, 15))
    comp = stats.compare_windows(st, a[0], a[1], b[0], b[1])
    aa = _independent_axis_arrays(h, *a)
    bb = _independent_axis_arrays(h, *b)
    recs = {(r["metric"], r["kind"]): r for r in comp["metrics"]}
    for ax in SCALAR_AXES:
        t = sp.ttest_ind(bb[ax], aa[ax], equal_var=False)
        d = stats.cohen_d(aa[ax], bb[ax])
        wt = recs[(ax, "welch_t")]
        assert wt["stat"] == pytest.approx(float(t.statistic), rel=1e-9)
        assert wt["p_raw"] == pytest.approx(float(t.pvalue), rel=1e-9)
        assert wt["effect"] == pytest.approx(d, rel=1e-9)
        assert wt["mean_a"] == pytest.approx(float(aa[ax].mean()), rel=1e-9)
        assert wt["mean_b"] == pytest.approx(float(bb[ax].mean()), rel=1e-9)
        u = sp.mannwhitneyu(aa[ax], bb[ax], alternative="two-sided")
        mw = recs[(ax, "mannwhitney")]
        assert mw["stat"] == pytest.approx(float(u.statistic), rel=1e-9)
        assert mw["p_raw"] == pytest.approx(float(u.pvalue), rel=1e-9)
        assert mw["effect"] == pytest.approx(
            stats.rank_biserial(float(u.statistic), len(aa[ax]), len(bb[ax])), rel=1e-9)
    st.close()


def test_compare_windows_direction_is_b_relative():
    # window A low valence, window B high valence -> valence direction "up", d > 0
    lo = [0.2, 0.2, 0.2, 0.2, 0.2]
    hi = [0.8, 0.8, 0.8, 0.8, 0.8]
    rows = []
    for k in range(50):
        rows.append((date(2022, 1, 1 + k % 20), np.array(lo) + 0.01 * (k % 5), 0, False))
    for k in range(50):
        rows.append((date(2022, 3, 1 + k % 20), np.array(hi) + 0.01 * (k % 5), 0, False))
    st = _build_store(rows)
    comp = stats.compare_windows(st, "2022-01-01", "2022-01-31", "2022-03-01", "2022-03-31")
    val = next(r for r in comp["metrics"] if r["metric"] == "valence" and r["kind"] == "welch_t")
    assert val["direction"] == "up"
    assert val["effect"] > 0
    st.close()


# ===========================================================================
# min-n gating / insufficient reporting (never silently dropped)
# ===========================================================================
def test_below_min_n_reported_as_insufficient():
    # 10 plays per window (< MIN_N=30) -> every axis is "insufficient", none dropped
    rows = []
    for k in range(10):
        rows.append((date(2022, 1, 1 + k), np.array([0.3] * 5), 0, False))
    for k in range(10):
        rows.append((date(2022, 3, 1 + k), np.array([0.6] * 5), 1, False))
    st = _build_store(rows)
    comp = stats.compare_windows(st, "2022-01-01", "2022-01-31", "2022-03-01", "2022-03-31")
    axis_recs = [r for r in comp["metrics"] if r["metric"] in SCALAR_AXES]
    assert axis_recs and all(r["insufficient"] and r["p_raw"] is None for r in axis_recs)
    sh = stats.significant_shifts(st, "2022-01-01", "2022-01-31", "2022-03-01", "2022-03-31")
    assert sh["shifts"] == []
    insuff_metrics = {r["metric"] for r in sh["insufficient"]}
    assert set(SCALAR_AXES).issubset(insuff_metrics)  # every axis surfaced, not dropped
    st.close()


# ===========================================================================
# significant_shifts — FDR family, effect floor, sentences
# ===========================================================================
def test_significant_shifts_identical_windows_find_nothing():
    h = make_synthetic_history(seed=5, n_days=200)
    st = HistoryStore.from_history(h)
    # same window compared to itself: no real difference to find
    sh = stats.significant_shifts(st, "2022-02-01", "2022-03-15", "2022-02-01", "2022-03-15")
    assert sh["n_survivors"] == 0
    st.close()


def test_significant_shifts_family_size_and_floor():
    h = make_synthetic_history(seed=7, n_days=730)
    st = HistoryStore.from_history(h)
    # two windows in different regimes -> real, large shifts
    sh = stats.significant_shifts(st, "2022-02-01", "2022-04-30", "2023-05-01", "2023-07-29")
    comp = stats.compare_windows(st, "2022-02-01", "2022-04-30", "2023-05-01", "2023-07-29")
    tested = [r for r in comp["metrics"] if r["p_raw"] is not None]
    # FDR family size == number of tested (p-valued) records, incl. both axis tests
    assert sh["family_size"] == len(tested)
    assert sh["n_survivors"] >= 1
    # every surviving shift clears BOTH q<0.05 and its effect-size floor
    for s in sh["shifts"]:
        assert s["q"] < 0.05
        floor = stats._EFFECT_FLOOR[s["effect_name"]]
        assert abs(s["effect"]) >= floor
        assert isinstance(s["sentence"], str) and s["sentence"].endswith(".")
    # a scalar-axis survivor collapses the t/U pair into ONE shift per axis
    axis_shift_metrics = [s["metric"] for s in sh["shifts"] if s["metric"] in SCALAR_AXES]
    assert len(axis_shift_metrics) == len(set(axis_shift_metrics))
    st.close()


def test_significant_shifts_sorted_by_effect_magnitude():
    h = make_synthetic_history(seed=7, n_days=730)
    st = HistoryStore.from_history(h)
    sh = stats.significant_shifts(st, "2022-02-01", "2022-04-30", "2023-05-01", "2023-07-29")
    mags = [abs(s["effect"]) for s in sh["shifts"]]
    assert mags == sorted(mags, reverse=True)
    st.close()


# ===========================================================================
# habit_anova
# ===========================================================================
def test_habit_anova_weekday_finds_nothing_on_synthetic():
    # The generator makes taste depend on the DATE (regime), never on weekday, so a
    # weekday ANOVA must find no axis differences -> an honest, correct null result.
    h = make_synthetic_history(seed=7, n_days=730)
    st = HistoryStore.from_history(h)
    res = stats.habit_anova(st, group_by="weekday")
    assert res["survivors"] == []
    # but the axes were actually tested (not skipped for lack of data)
    tested = [a for a in res["axes"] if not a.get("insufficient")]
    assert len(tested) == len(SCALAR_AXES)
    for a in tested:
        # Welch (fractional df2), not plain ANOVA's integer N-k
        assert a["welch_df2"] != round(a["welch_df2"]) or a["k_groups"] == 2
    st.close()


def test_habit_anova_month_finds_regime_structure():
    # Months map onto regimes, so grouping taste axes by month WILL differ.
    h = make_synthetic_history(seed=7, n_days=730)
    st = HistoryStore.from_history(h)
    res = stats.habit_anova(st, group_by="month")
    assert len(res["survivors"]) >= 1
    for s in res["survivors"]:
        assert s["welch_q"] < 0.05 and s["epsilon_sq"] >= stats.EPSILON_SQ_FLOOR
        assert s["summary"].startswith(stats._AXIS_LABEL[s["axis"]])
    st.close()


def test_habit_anova_insufficient_when_window_tiny():
    h = make_synthetic_history(seed=5, n_days=200)
    st = HistoryStore.from_history(h)
    # a single calendar day is one weekday -> only one group -> < MIN_GROUPS
    res = stats.habit_anova(st, start="2022-01-01", end="2022-01-01", group_by="weekday")
    assert all(a.get("insufficient") for a in res["axes"])
    assert res["survivors"] == []
    st.close()


def test_habit_anova_rejects_unknown_group_by():
    h = make_synthetic_history(seed=1, n_days=120)
    st = HistoryStore.from_history(h)
    with pytest.raises(ValueError):
        stats.habit_anova(st, group_by="lunar_phase")
    st.close()


# ===========================================================================
# adjacent_scan — the Phase-3 baseline
# ===========================================================================
def test_adjacent_scan_shape_and_family_size():
    h = make_synthetic_history(seed=7, n_days=365)
    st = HistoryStore.from_history(h)
    scan = stats.adjacent_scan(st, granularity="month", metrics="axes")
    # one boundary per adjacent month pair
    assert len(scan["boundaries"]) >= 10
    # scan-wide FDR family == total tested records across every boundary
    fam = sum(len(b["metrics"]) for b in scan["boundaries"])
    assert scan["family_size"] == fam
    # axes-only -> every record is a scalar-axis test
    for b in scan["boundaries"]:
        for r in b["metrics"]:
            assert r["metric"] in SCALAR_AXES
    # flagged boundaries are exactly those with >=1 survivor clearing q and floor
    for b in scan["boundaries"]:
        surv = [r for r in b["metrics"]
                if r["q"] < scan["q_threshold"] and abs(r["effect"]) >= stats._EFFECT_FLOOR[r["effect_name"]]]
        assert b["flagged"] == bool(surv)
    assert scan["n_flagged"] == sum(1 for b in scan["boundaries"] if b["flagged"])
    st.close()


def test_adjacent_scan_flags_near_planted_transitions():
    h = make_synthetic_history(seed=7, n_days=730)
    st = HistoryStore.from_history(h)
    scan = stats.adjacent_scan(st, granularity="month", metrics="axes")

    def midx(d):
        return d.year * 12 + (d.month - 1)

    flagged = [midx(date.fromisoformat(b["b"]["start"]))
               for b in scan["boundaries"] if b["flagged"]]
    transitions = [r.start for r in h.ground_truth.regimes[1:]]
    for t in transitions:  # every planted transition within +/-1 month of a flag
        assert any(abs(f - midx(t)) <= 1 for f in flagged)
    st.close()


def test_adjacent_scan_all_metrics_includes_genre_and_behavior():
    h = make_synthetic_history(seed=7, n_days=365)
    st = HistoryStore.from_history(h)
    scan = stats.adjacent_scan(st, granularity="month", metrics="all")
    kinds = {r["kind"] for b in scan["boundaries"] for r in b["metrics"]}
    assert "chi2" in kinds and "two_prop_z" in kinds
    st.close()


def test_adjacent_scan_rejects_unknown_granularity():
    h = make_synthetic_history(seed=1, n_days=120)
    st = HistoryStore.from_history(h)
    with pytest.raises(ValueError):
        stats.adjacent_scan(st, granularity="fortnight")
    st.close()
