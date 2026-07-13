"""Hand-verified tests for the Phase-3 taste-dynamics package.

Philosophy (matching ``tests/test_analytics_stats.py``): every number is checked
against something computed *independently* — by hand on paper for the tiny
fixtures, or by pandas recomputation off the raw stream.  All offline, all
deterministic.
"""
from datetime import date, datetime, timedelta, timezone

import numpy as np
import pytest

from playlistcont.data.schema import GENRES, N_AXES, SCALAR_AXES
from playlistcont.dynamics import (
    build_windows, cusum, pelt, penalty_sweep, fit_flavors, assign,
    score_detections, segmentation_ari, recommended_detector, DetectedChange,
)
from playlistcont.dynamics.detectors import (
    prepare_matrix, _month_of_year_deseasonalize, default_penalty,
)
from playlistcont.dynamics.windows import WindowSeries
from playlistcont.history.schema import (
    HistoryGroundTruth, HistoryTrack, ListenEvent, ListeningHistory,
    PlantedChange, RegimeSpec, SeasonalSpan, Trap,
)
from playlistcont.history.store import HistoryStore
from playlistcont.history.synthetic import make_synthetic_history

UTC = timezone.utc
N_SCALAR = len(SCALAR_AXES)


# ===========================================================================
# helpers to build tiny hand-controlled data
# ===========================================================================
def _feat(scalar_val: float, genre_idx: int) -> np.ndarray:
    v = np.zeros(N_AXES, dtype=np.float32)
    v[:N_SCALAR] = scalar_val
    v[N_SCALAR + genre_idx] = 0.9
    return v


def _tiny_history() -> ListeningHistory:
    """Two tracks; week1 = A,A,A,B ; week2 = A,B.  Features & order fully known."""
    A = HistoryTrack("spotify:track:A", "A", "ArtA", "AlbA", "spotify:album:A",
                     _feat(0.2, 0), "country")   # genre country (idx 0)
    B = HistoryTrack("spotify:track:B", "B", "ArtB", "AlbB", "spotify:album:B",
                     _feat(0.8, 1), "rap")       # genre rap (idx 1)
    tracks = {A.track_uri: A, B.track_uri: B}
    # week1 Monday = 2022-01-03 .. Sun 2022-01-09 ; week2 = 2022-01-10 ..
    plays = [
        ("A", date(2022, 1, 3)), ("A", date(2022, 1, 4)), ("A", date(2022, 1, 5)),
        ("B", date(2022, 1, 6)),
        ("A", date(2022, 1, 10)), ("B", date(2022, 1, 11)),
    ]
    events = []
    for uri, d in plays:  # already in chronological order -> event_id = index
        events.append(ListenEvent(
            ts=datetime(d.year, d.month, d.day, 12, 0, 0, tzinfo=UTC),
            track_uri=f"spotify:track:{uri}", track_name=uri, artist_name=f"Art{uri}",
            ms_played=200000, album_name=f"Alb{uri}", skipped=False, platform="web"))
    return ListeningHistory(events=events, tracks=tracks, provenance="synthetic")


def _series_from_matrix(matrix: np.ndarray, start: date, step_days: int = 7,
                        span_days: int = 0) -> WindowSeries:
    """Wrap a raw matrix as a WindowSeries whose only group is 'scalar_axes'."""
    n, d = matrix.shape
    starts = [start + timedelta(days=step_days * i) for i in range(n)]
    ends = [s + timedelta(days=step_days) for s in starts]
    cols = [f"c{j}" for j in range(d)]
    return WindowSeries(
        starts=starts, ends=ends, matrix=matrix.astype(float), columns=cols,
        column_groups={"scalar_axes": cols}, mask=np.ones(n, dtype=bool),
        event_counts=np.full(n, 999), granularity="week", weighting="plays",
        min_events=1, flavor_model=None,
        span_days=span_days or (ends[-1] - starts[0]).days)


# ===========================================================================
# 1. window builder vs hand computation
# ===========================================================================
def test_window_builder_plays_weighting_hand_values():
    ws = build_windows(_tiny_history(), granularity="week", min_events=2)
    assert ws.n_windows == 2
    assert ws.mask.tolist() == [True, True]
    assert ws.event_counts.tolist() == [4, 2]

    ci = ws.columns.index
    # week1 scalar mean over 4 plays (3*0.2 + 1*0.8)/4 = 0.35 on every scalar axis
    for ax in SCALAR_AXES:
        assert ws.matrix[0, ci(ax)] == pytest.approx(0.35, abs=1e-6)
    # week1 genre shares: country 3/4, rap 1/4
    assert ws.matrix[0, ci("genre_country")] == pytest.approx(0.75)
    assert ws.matrix[0, ci("genre_rap")] == pytest.approx(0.25)
    # discovery week1: A's first + B's first = 2 of 4 plays
    assert ws.matrix[0, ci("discovery_rate")] == pytest.approx(0.5)
    # intensity week1 = 4 plays / 7 days ; coverage = 1
    assert ws.matrix[0, ci("intensity_plays_per_day")] == pytest.approx(4 / 7)
    assert ws.matrix[0, ci("coverage")] == pytest.approx(1.0)
    # week2 discovery: both tracks already seen -> 0
    assert ws.matrix[1, ci("discovery_rate")] == pytest.approx(0.0)
    # flavor shares always sum to 1 on a valid window
    fcols = [c for c in ws.columns if c.startswith("flavor_")]
    assert ws.matrix[0, [ci(c) for c in fcols]].sum() == pytest.approx(1.0)


def test_window_builder_masking():
    ws = build_windows(_tiny_history(), granularity="week", min_events=3)
    # week2 has only 2 plays -> masked; its row is all NaN
    assert ws.mask.tolist() == [True, False]
    assert np.isnan(ws.matrix[1]).all()


def test_window_builder_unique_weighting_hand_values():
    ws = build_windows(_tiny_history(), granularity="week", min_events=2,
                       weighting="unique")
    ci = ws.columns.index
    # unique week1: distinct {A,B} weighted once -> scalar mean (0.2+0.8)/2 = 0.5
    for ax in SCALAR_AXES:
        assert ws.matrix[0, ci(ax)] == pytest.approx(0.5, abs=1e-6)
    # genre shares country 1/2, rap 1/2
    assert ws.matrix[0, ci("genre_country")] == pytest.approx(0.5)
    # discovery (unique): both distinct tracks' first occurrence is first-ever -> 1.0
    assert ws.matrix[0, ci("discovery_rate")] == pytest.approx(1.0)


def test_window_builder_deterministic_column_order():
    ws = build_windows(_tiny_history(), granularity="week", min_events=2)
    expected_head = list(SCALAR_AXES) + [f"genre_{g}" for g in GENRES]
    assert ws.columns[:len(expected_head)] == expected_head
    assert ws.columns[-3:] == ["discovery_rate", "intensity_plays_per_day", "coverage"]


# ===========================================================================
# 2. flavor identity stability
# ===========================================================================
def test_flavor_fit_is_deterministic():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(60, N_SCALAR))
    m1 = fit_flavors(X, k=3, seed=42)
    m2 = fit_flavors(X, k=3, seed=42)
    assert np.allclose(m1.centroids, m2.centroids)
    assert np.array_equal(assign(m1, X), assign(m2, X))


def test_flavor_identity_stable_across_windows():
    # three well-separated blobs; a track's label must not depend on the batch
    rng = np.random.default_rng(1)
    blobs = np.vstack([
        rng.normal(0.1, 0.02, size=(30, N_SCALAR)),
        rng.normal(0.5, 0.02, size=(30, N_SCALAR)),
        rng.normal(0.9, 0.02, size=(30, N_SCALAR)),
    ])
    model = fit_flavors(blobs, k=3, seed=0)
    # window A = first 45 rows, window B = last 45 rows (overlap in the middle blob)
    labels_all = assign(model, blobs)
    labels_A = assign(model, blobs[:45])
    labels_B = assign(model, blobs[45:])
    assert np.array_equal(labels_A, labels_all[:45])
    assert np.array_equal(labels_B, labels_all[45:])
    assert model.k == 3 and len(model.names) == 3


def test_flavor_k_clamped_to_n_tracks():
    X = np.array([[0.2] * N_SCALAR, [0.8] * N_SCALAR])
    model = fit_flavors(X, k=5, seed=0)
    assert model.k == 2


# ===========================================================================
# 3. CUSUM / PELT on constructed series
# ===========================================================================
def _break_series(seed=0):
    # asymmetric step (30 low + 10 high): after z-scoring the minority high segment
    # sits ~1.7 sigma above baseline, a clean multi-window jump at index 30.
    rng = np.random.default_rng(seed)
    flat = rng.normal(0.0, 0.05, size=(40, 2))
    flat[30:] += 5.0
    return flat


def test_cusum_catches_obvious_break():
    ws = _series_from_matrix(_break_series(), date(2022, 1, 3))
    dets = cusum(ws, "scalar_axes")
    assert len(dets) >= 1
    idxs = [(c.date - date(2022, 1, 3)).days // 7 for c in dets]
    assert any(28 <= i <= 32 for i in idxs)   # near the true break at window 30


def test_cusum_silent_on_flat_noise():
    rng = np.random.default_rng(3)
    flat = _series_from_matrix(rng.normal(0.0, 0.05, size=(40, 2)), date(2022, 1, 3))
    assert cusum(flat, "scalar_axes") == []


def test_pelt_smoke_finds_break():
    ws = _series_from_matrix(_break_series(seed=1), date(2022, 1, 3))
    dets = pelt(ws, "scalar_axes", cost="l2")
    assert len(dets) >= 1
    idxs = [(c.date - date(2022, 1, 3)).days // 7 for c in dets]
    assert any(28 <= i <= 32 for i in idxs)


def test_default_penalty_cost_aware():
    # l2 scales with dimension; rbf does not (documented)
    assert default_penalty(100, 19, "l2") == pytest.approx(19 * np.log(100))
    assert default_penalty(100, 19, "rbf") == pytest.approx(np.log(100))


def test_penalty_sweep_monotone_and_elbow():
    ws = _series_from_matrix(_break_series(seed=2), date(2022, 1, 3))
    sw = penalty_sweep(ws, "scalar_axes", cost="l2")
    counts = sw["n_changes"]
    # more penalty -> never more changes (monotone non-increasing in penalty)
    assert all(counts[i] >= counts[i + 1] for i in range(len(counts) - 1))
    assert sw["elbow_penalty"] > 0


# ===========================================================================
# 4. deseasonalize
# ===========================================================================
def test_month_of_year_deseasonalize_removes_december_bump():
    # 24 monthly rows, one column = base + 5 every December, else base
    base = 2.0
    months = np.array([((i % 12) + 1) for i in range(24)])
    col = np.full(24, base)
    col[months == 12] += 5.0
    Z = col.reshape(-1, 1)
    out = _month_of_year_deseasonalize(Z, months)
    # every corrected value collapses to the global mean (hand-checked):
    # global_mean = base + 2*5/24 = base + 0.41667
    assert np.allclose(out, base + 10.0 / 24.0, atol=1e-9)


def test_prepare_matrix_deseasonalize_gated_on_short_history():
    ws = _series_from_matrix(np.random.default_rng(0).normal(size=(10, 2)),
                             date(2022, 1, 3), step_days=7, span_days=70)
    Zoff, _ = prepare_matrix(ws, "scalar_axes", deseasonalize=False)
    Zon, _ = prepare_matrix(ws, "scalar_axes", deseasonalize=True)
    # < 18 months -> deseasonalize is a documented no-op, identical output
    assert np.allclose(Zoff, Zon)


# ===========================================================================
# 5. eval.py — all scoring rules on paper-checkable fixtures
# ===========================================================================
def _eval_gt() -> HistoryGroundTruth:
    regimes = [
        RegimeSpec(date(2022, 1, 1), date(2022, 6, 1), {"a": 1.0}, 40.0, "r0"),
        RegimeSpec(date(2022, 6, 1), date(2022, 12, 1), {"b": 1.0}, 40.0, "r1"),
    ]
    change_points = [
        PlantedChange(date(2022, 3, 1), "abrupt", "abrupt A"),
        PlantedChange(date(2022, 6, 1), "drift_start", "drift begins"),
        PlantedChange(date(2022, 6, 30), "drift_end", "drift ends"),
        PlantedChange(date(2022, 9, 1), "abrupt", "abrupt B"),
    ]
    traps = [Trap("binge", date(2022, 4, 1), date(2022, 4, 10), "binge")]
    seasonal = [SeasonalSpan(date(2022, 12, 1), date(2022, 12, 31), "party pop", "dec")]
    return HistoryGroundTruth(regimes, change_points, traps, seasonal, seed=0, params={})


def test_score_detections_hand_computed():
    gt = _eval_gt()
    dets = [
        DetectedChange(date(2022, 3, 5), 1.0, "m"),    # TP -> abrupt 03-01, dist 4
        DetectedChange(date(2022, 6, 15), 1.0, "m"),   # TP -> inside drift, dist 0
        DetectedChange(date(2022, 9, 20), 1.0, "m"),   # FP (19d from 09-01) -> other
        DetectedChange(date(2022, 4, 5), 1.0, "m"),    # FP inside binge trap
        DetectedChange(date(2022, 12, 15), 1.0, "m"),  # FP inside seasonal
    ]
    sc = score_detections(dets, gt, tolerance_days=14)
    assert (sc["tp"], sc["fp"], sc["fn"]) == (2, 3, 1)
    assert sc["precision"] == pytest.approx(2 / 5)
    assert sc["recall"] == pytest.approx(2 / 3)
    assert sc["f1"] == pytest.approx(0.5, abs=1e-6)
    assert sc["localization_mae"] == pytest.approx((4 + 0) / 2)
    assert (sc["fp_trap"], sc["fp_seasonal"], sc["fp_other"]) == (1, 1, 1)
    assert sc["trap_fp_rate"] == pytest.approx(1.0)
    assert sc["seasonal_fp_rate"] == pytest.approx(1.0)


def test_score_detections_one_to_one_greedy():
    gt = _eval_gt()
    # two detections crowd the same abrupt target; only the closest matches
    dets = [DetectedChange(date(2022, 3, 2), 1.0, "m"),   # dist 1 -> matches
            DetectedChange(date(2022, 3, 6), 1.0, "m")]   # dist 5 -> becomes FP
    sc = score_detections(dets, gt, tolerance_days=14)
    assert sc["tp"] == 1 and sc["fp"] == 1
    assert sc["localization_mae"] == pytest.approx(1.0)


def test_score_detections_perfect_and_empty():
    gt = _eval_gt()
    perfect = [DetectedChange(date(2022, 3, 1), 1, "m"),
               DetectedChange(date(2022, 6, 15), 1, "m"),
               DetectedChange(date(2022, 9, 1), 1, "m")]
    sc = score_detections(perfect, gt, 14)
    assert sc["f1"] == pytest.approx(1.0)
    assert sc["fp_trap"] == sc["fp_seasonal"] == sc["fp_other"] == 0
    empty = score_detections([], gt, 14)
    assert empty["recall"] == 0.0 and empty["f1"] == 0.0
    assert np.isnan(empty["localization_mae"])


def test_segmentation_ari_hand_cases():
    gt = _eval_gt()   # regime boundary at 2022-06-01
    dates = [date(2022, 1, 1), date(2022, 3, 1), date(2022, 5, 1),
             date(2022, 7, 1), date(2022, 9, 1), date(2022, 11, 1)]
    # a single detection exactly on the boundary reproduces the true partition
    perfect = [DetectedChange(date(2022, 6, 1), 1.0, "m")]
    assert segmentation_ari(perfect, gt, dates) == pytest.approx(1.0)
    # no detections -> one induced cluster vs two true regimes -> ARI 0
    assert segmentation_ari([], gt, dates) == pytest.approx(0.0)
    assert np.isnan(segmentation_ari(perfect, gt, []))


# ===========================================================================
# 6. recommended_detector end-to-end
# ===========================================================================
def test_recommended_detector_config_and_runs():
    rd = recommended_detector()
    assert rd.method == "pelt_rbf" and rd.representation == "combined"
    assert rd.granularity == "week" and rd.deseasonalize is False
    h = make_synthetic_history(seed=5, n_days=730)
    store = HistoryStore.from_history(h)
    dets = rd.detect(store)
    assert isinstance(dets, list) and all(isinstance(c, DetectedChange) for c in dets)
    # a real 2-year history has planted changes; the recommended detector finds some
    assert len(dets) >= 1
    sc = score_detections(dets, h.ground_truth, tolerance_days=14)
    assert sc["f1"] > 0.5     # comfortably better than chance on a planted history
