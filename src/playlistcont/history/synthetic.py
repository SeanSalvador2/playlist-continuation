"""Planted-ground-truth generator for personal listening histories.

This is the backbone of the history modality: it fabricates a person's day-by-day
play stream while recording *exactly* what taste structure was planted, so a later
change-point detector can be scored honestly (did it find the real changes? did it
avoid the traps?).

What is modelled, and why it is detectable:

* **Regimes** — the timeline is partitioned into 3-5 piecewise-stationary spans, each
  an archetype *mixture* drawn from the taste palette
  (:data:`playlistcont.data.synthetic.ARCHETYPES` — reused, not reinvented).  The
  mean interpretable-axis vector of the tracks played in a regime sits near that
  mixture's archetypes, so the planted taste is genuinely present in the data.
* **Transitions** — between regimes taste either flips **abruptly** on a date or
  **drifts** linearly over a ``[drift_start, drift_end]`` window.  Both are recorded
  as :class:`~playlistcont.history.schema.PlantedChange` landmarks.
* **Repetition realism** — each listener keeps a personal pool per archetype; plays
  follow rich-get-richer preferential attachment (heavy rotation of favourites,
  Zipf-like) with a regime-dependent *discovery* rate that admits new tracks.
* **Temporal texture** — daily counts are Poisson with per-listener weekday and
  hour-of-day profiles (evening/weekend heavy); most plays run full length, a
  fraction are skips (short ``ms_played``, ``skipped=True``).
* **Seasonal overlay** — an optional December bump of a designated holiday-ish
  archetype, annotated in the ground truth but **not** a change point (it recurs).
* **Traps** — a ``volume_only`` span (listening volume jumps, mixture unchanged) and a
  ``binge`` span (one album dominates, mixture unchanged).  These exist so a detector
  can be scored on *not* calling them taste changes.

Everything is deterministic from ``seed`` via a single ``numpy`` ``Generator``,
matching the repo's seeding style (:mod:`playlistcont.data.synthetic`).

Honesty note: this is a *deliberately simple* model of taste change (piecewise
stationary + linear drift).  It measures whether a detector recovers planted
structure; it is not a claim about how real listening actually evolves.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from typing import Dict, List, Optional, Tuple

import numpy as np

from ..data.schema import AXES, GENRES, N_AXES, SCALAR_AXES
from ..data.synthetic import ARCHETYPES, Archetype
from .schema import (
    HistoryGroundTruth,
    HistoryTrack,
    ListenEvent,
    ListeningHistory,
    PlantedChange,
    RegimeSpec,
    SeasonalSpan,
    Trap,
)

UTC = timezone.utc

_ARCH_BY_NAME: Dict[str, Archetype] = {a.name: a for a in ARCHETYPES}
_GENRE_IDX = {g: i for i, g in enumerate(GENRES)}

# The archetype whose December listening spikes (holiday-ish party music).
_SEASONAL_ARCHETYPE = "party pop"


# ---------------------------------------------------------------------------
# Archetype -> axis vector.  Mirrors data/synthetic.py's per-track feature
# formula (archetype scalar mean + gaussian noise; a strong home-genre block
# with a little bleed) as a small pure helper, so the two modalities agree on
# what an archetype "sounds like" without importing generator internals.
# ---------------------------------------------------------------------------


def archetype_axis_vector(arch: Archetype, rng: np.random.Generator) -> np.ndarray:
    """One noisy interpretable-axis vector for a track of ``arch`` (length N_AXES)."""
    v = np.zeros(N_AXES, dtype=np.float32)
    for j, ax in enumerate(SCALAR_AXES):
        v[j] = np.clip(arch.scalars[ax] + rng.normal(0, 0.09), 0.0, 1.0)
    g0 = _GENRE_IDX[arch.genre]
    v[len(SCALAR_AXES) + g0] = np.clip(0.85 + rng.normal(0, 0.08), 0.0, 1.0)
    gb = int(rng.integers(0, len(GENRES)))
    v[len(SCALAR_AXES) + gb] += 0.15 * float(rng.random())
    return v


# ---------------------------------------------------------------------------
# Track universe: a fixed pool of tracks per archetype, ordered by intrinsic
# popularity so heavy hitters get discovered first (Zipf-like rotation).
# ---------------------------------------------------------------------------


@dataclass
class _UTrack:
    uri: str
    name: str
    artist_name: str
    album_uri: str
    album_name: str
    features: np.ndarray
    archetype: str
    duration_ms: int


class _Universe:
    """Per-archetype pools of candidate tracks for one listener."""

    def __init__(self, rng: np.random.Generator, pool_size: int = 220):
        self.pools: List[List[_UTrack]] = []
        self.pop_weights: List[np.ndarray] = []
        # albums[a] -> {album_uri: [track indices]} for binge selection
        self.albums: List[Dict[str, List[int]]] = []
        gid = 0
        for a_idx, arch in enumerate(ARCHETYPES):
            n_albums = max(4, pool_size // 5)
            pool: List[_UTrack] = []
            albums: Dict[str, List[int]] = {}
            for i in range(pool_size):
                alb = i % n_albums
                artist = alb % max(1, n_albums // 2)
                t = _UTrack(
                    uri=f"spotify:track:syn{gid:07d}xx",
                    name=f"{arch.name.title()} Track {i}",
                    artist_name=f"{arch.genre.title()} Artist {a_idx}-{artist}",
                    album_uri=f"spotify:album:syn{a_idx:02d}{alb:04d}",
                    album_name=f"{arch.name.title()} Album {alb}",
                    features=archetype_axis_vector(arch, rng),
                    archetype=arch.name,
                    duration_ms=int(np.clip(rng.normal(210_000, 45_000), 60_000, 480_000)),
                )
                pool.append(t)
                albums.setdefault(t.album_uri, []).append(i)
                gid += 1
            # Zipf-like intrinsic popularity; index 0 (rank 1) is the biggest hit,
            # so discovery from the top yields heavy rotation of a few favourites.
            ranks = np.arange(1, pool_size + 1)
            w = 1.0 / np.power(ranks, 1.1)
            w /= w.sum()
            self.pools.append(pool)
            self.pop_weights.append(w)
            self.albums.append(albums)


# ---------------------------------------------------------------------------
# Regime construction
# ---------------------------------------------------------------------------


def _auto_regimes(
    rng: np.random.Generator,
    start_date: date,
    n_days: int,
    base_events_per_day: int,
) -> Tuple[List[RegimeSpec], List[str], List[float]]:
    """Sample 3-5 regimes partitioning the timeline.

    Returns (regimes, transition_kinds, discovery_rates).  ``transition_kinds[i]`` is
    the transition *out of* regime ``i`` ("abrupt" or "drift"); the last is unused.
    Regime 0 is always a **pure** single-archetype regime with an abrupt exit, giving
    tests (and detectors) a clean planted signal to anchor on.
    """
    n_regimes = int(rng.integers(3, 6))
    # segment lengths (days), each >= a floor, summing exactly to n_days
    raw = rng.uniform(0.8, 1.2, size=n_regimes)
    lengths = np.maximum((raw / raw.sum() * n_days).astype(int), 40)
    lengths[-1] += n_days - int(lengths.sum())  # absorb rounding into the last
    if lengths[-1] < 40:  # pathological tiny timeline: fall back to equal split
        lengths = np.full(n_regimes, n_days // n_regimes)
        lengths[-1] += n_days - int(lengths.sum())

    perm = rng.permutation(len(ARCHETYPES))
    regimes: List[RegimeSpec] = []
    transitions: List[str] = []
    discovery: List[float] = []
    day = 0
    for i in range(n_regimes):
        seg = int(lengths[i])
        r_start = start_date + timedelta(days=day)
        r_end = start_date + timedelta(days=day + seg)
        primary = ARCHETYPES[perm[i % len(perm)]].name
        if i == 0:
            mixture = {primary: 1.0}
            label = f"pure {primary}"
        else:
            w = float(rng.uniform(0.65, 0.9))
            secondary = ARCHETYPES[perm[(i + 1) % len(perm)]].name
            if secondary == primary:
                secondary = ARCHETYPES[perm[(i + 2) % len(perm)]].name
            mixture = {primary: w, secondary: round(1.0 - w, 4)}
            label = f"{primary} + {secondary}"
        epd = float(base_events_per_day) * float(rng.uniform(0.8, 1.3))
        regimes.append(RegimeSpec(r_start, r_end, mixture, epd, label))
        # first transition is always abrupt (keeps regime 0 clean); rest random
        transitions.append("abrupt" if i == 0 else ("abrupt" if rng.random() < 0.5 else "drift"))
        discovery.append(float(rng.uniform(0.05, 0.25)))
        day += seg
    return regimes, transitions, discovery


def _build_change_points(
    regimes: List[RegimeSpec], transitions: List[str], drift_half: int
) -> Tuple[List[PlantedChange], List[Tuple[int, date, date]]]:
    """Turn transitions into planted change landmarks + drift windows.

    Returns (change_points, drift_windows) where each drift window is
    ``(regime_index, drift_start, drift_end)`` for the transition out of that regime.
    """
    changes: List[PlantedChange] = []
    windows: List[Tuple[int, date, date]] = []
    for i in range(len(regimes) - 1):
        boundary = regimes[i].end  # == regimes[i + 1].start
        if transitions[i] == "abrupt":
            changes.append(PlantedChange(
                boundary, "abrupt",
                f"abrupt switch: {regimes[i].label} -> {regimes[i + 1].label}",
            ))
        else:
            # keep the window strictly inside both neighbouring regimes
            span_prev = (regimes[i].end - regimes[i].start).days
            span_next = (regimes[i + 1].end - regimes[i + 1].start).days
            w = min(drift_half, span_prev // 2, span_next // 2)
            d_start = boundary - timedelta(days=w)
            d_end = boundary + timedelta(days=w)
            changes.append(PlantedChange(
                d_start, "drift_start",
                f"drift begins: {regimes[i].label} -> {regimes[i + 1].label}",
            ))
            changes.append(PlantedChange(
                d_end, "drift_end",
                f"drift ends: {regimes[i].label} -> {regimes[i + 1].label}",
            ))
            windows.append((i, d_start, d_end))
    return changes, windows


# ---------------------------------------------------------------------------
# Per-listener temporal profiles
# ---------------------------------------------------------------------------


def _weekday_profile(rng: np.random.Generator) -> np.ndarray:
    base = np.array([0.9, 0.9, 0.95, 1.0, 1.15, 1.35, 1.25])  # Mon..Sun, weekend heavy
    jitter = rng.uniform(0.9, 1.1, size=7)
    return base * jitter


def _hour_profile(rng: np.random.Generator) -> np.ndarray:
    hours = np.arange(24)
    # bimodal: a morning commute bump and a dominant evening peak
    morning = np.exp(-0.5 * ((hours - 8) / 2.0) ** 2)
    evening = 1.6 * np.exp(-0.5 * ((hours - 20) / 2.6) ** 2)
    pmf = morning + evening + 0.05
    pmf *= rng.uniform(0.85, 1.15, size=24)
    return pmf / pmf.sum()


# ---------------------------------------------------------------------------
# The generator
# ---------------------------------------------------------------------------


def _mixture_on(
    d: date,
    regimes: List[RegimeSpec],
    drift_windows: List[Tuple[int, date, date]],
    seasonal: bool,
) -> Dict[str, float]:
    """Effective archetype mixture on day ``d`` (drift-blended + seasonal-boosted)."""
    # locate containing regime
    r_idx = 0
    for i, r in enumerate(regimes):
        if r.start <= d < r.end:
            r_idx = i
            break
    else:
        r_idx = len(regimes) - 1
    mix: Dict[str, float] = dict(regimes[r_idx].mixture)

    # drift: linear interpolation across a window centred on a boundary
    for (i, d_start, d_end) in drift_windows:
        if d_start <= d < d_end:
            total = (d_end - d_start).days
            frac = (d - d_start).days / max(1, total)
            a = regimes[i].mixture
            b = regimes[i + 1].mixture
            names = set(a) | set(b)
            mix = {n: (1 - frac) * a.get(n, 0.0) + frac * b.get(n, 0.0) for n in names}
            break

    # December holiday bump of the designated archetype (annotated, not a change)
    if seasonal and d.month == 12:
        mix = dict(mix)
        mix[_SEASONAL_ARCHETYPE] = mix.get(_SEASONAL_ARCHETYPE, 0.0) + 0.6

    tot = sum(mix.values())
    if tot <= 0:
        return {regimes[r_idx].label: 1.0}
    return {n: w / tot for n, w in mix.items() if w > 0}


def make_synthetic_history(
    seed: int = 0,
    start_date: date = date(2022, 1, 1),
    n_days: int = 730,
    base_events_per_day: int = 40,
    regimes: Optional[List[RegimeSpec]] = None,
    include_traps: bool = True,
    seasonal: bool = True,
) -> ListeningHistory:
    """Generate one person's listening history with fully-planted ground truth.

    All randomness derives from ``seed`` (a single ``numpy`` ``Generator``), so the
    same ``seed`` reproduces an identical event stream and a different ``seed`` gives a
    different one.  When ``regimes`` is ``None`` the taste timeline is auto-sampled
    (3-5 regimes); pass an explicit list to pin it.
    """
    rng = np.random.default_rng(seed)

    if regimes is None:
        regimes, transitions, discovery = _auto_regimes(
            rng, start_date, n_days, base_events_per_day
        )
    else:
        regimes = sorted(regimes, key=lambda r: r.start)
        transitions = ["abrupt"] * len(regimes)
        discovery = [0.12] * len(regimes)

    end_date = start_date + timedelta(days=n_days)
    change_points, drift_windows = _build_change_points(regimes, transitions, drift_half=15)

    # ---- traps -----------------------------------------------------------
    traps: List[Trap] = []
    vol_span: Optional[Tuple[date, date]] = None
    binge_span: Optional[Tuple[date, date]] = None
    binge_album: Optional[str] = None
    binge_arch: Optional[str] = None
    if include_traps:
        # volume_only: inside the pure regime 0, leaving clear days on both sides
        r0 = regimes[0]
        r0_len = (r0.end - r0.start).days
        vlen = min(14, max(4, r0_len // 3))
        vstart = r0.start + timedelta(days=(r0_len - vlen) // 2)
        vend = vstart + timedelta(days=vlen)
        vol_span = (vstart, vend)
        traps.append(Trap(
            "volume_only", vstart, vend,
            "listening volume spikes ~3.5x while the archetype mixture is unchanged",
        ))
        # binge: a short span in a later regime dominated by one album
        rb = regimes[1] if len(regimes) > 1 else regimes[0]
        binge_arch = max(rb.mixture, key=rb.mixture.get)
        rb_len = (rb.end - rb.start).days
        blen = min(7, max(3, rb_len // 4))
        bstart = rb.start + timedelta(days=min(10, rb_len // 3))
        bend = bstart + timedelta(days=blen)
        binge_span = (bstart, bend)

    # ---- track universe --------------------------------------------------
    universe = _Universe(rng)
    arch_index = {a.name: i for i, a in enumerate(ARCHETYPES)}
    if binge_arch is not None:
        # pick a binge album (an album with several tracks) from the arch's pool
        albums = universe.albums[arch_index[binge_arch]]
        candidates = [alb for alb, idxs in albums.items() if len(idxs) >= 3]
        binge_album = candidates[int(rng.integers(0, len(candidates)))]

    # per-listener temporal profiles
    weekday_w = _weekday_profile(rng)
    hour_pmf = _hour_profile(rng)

    # personal discovery state: per archetype, discovered track indices + play counts
    discovered: Dict[int, List[int]] = {i: [] for i in range(len(ARCHETYPES))}
    play_counts: Dict[int, List[float]] = {i: [] for i in range(len(ARCHETYPES))}

    tracks: Dict[str, HistoryTrack] = {}
    events: List[ListenEvent] = []
    platforms = ["android", "ios", "web_player", "desktop"]
    platform_w = np.array([0.4, 0.3, 0.2, 0.1])

    def _register(t: _UTrack) -> None:
        if t.uri not in tracks:
            tracks[t.uri] = HistoryTrack(
                track_uri=t.uri, track_name=t.name, artist_name=t.artist_name,
                album_name=t.album_name, album_uri=t.album_uri,
                features=t.features, archetype=t.archetype,
            )

    def _pick_track(a_idx: int, disc_rate: float) -> _UTrack:
        pool = universe.pools[a_idx]
        disc = discovered[a_idx]
        if not disc or rng.random() < disc_rate:
            # admit the next-most-popular undiscovered track
            if len(disc) < len(pool):
                new_i = len(disc)  # pools are pre-sorted by popularity
                disc.append(new_i)
                play_counts[a_idx].append(1.0)
                return pool[new_i]
        # replay: rich-get-richer preferential attachment over discovered tracks
        counts = np.asarray(play_counts[a_idx], dtype=float)
        p = counts / counts.sum()
        choice = int(rng.choice(len(disc), p=p))
        play_counts[a_idx][choice] += 1.0
        return pool[disc[choice]]

    def _regime_of(d: date) -> int:
        for i, r in enumerate(regimes):
            if r.start <= d < r.end:
                return i
        return len(regimes) - 1

    for day_off in range(n_days):
        d = start_date + timedelta(days=day_off)
        r_idx = _regime_of(d)
        mix = _mixture_on(d, regimes, drift_windows, seasonal)
        names = list(mix.keys())
        probs = np.array([mix[n] for n in names])

        lam = regimes[r_idx].events_per_day * weekday_w[d.weekday()]
        if vol_span and vol_span[0] <= d < vol_span[1]:
            lam *= 3.5
        n_ev = int(rng.poisson(max(0.1, lam)))
        if n_ev <= 0:
            continue

        binge_active = bool(binge_span and binge_span[0] <= d < binge_span[1])
        for _ in range(n_ev):
            hour = int(rng.choice(24, p=hour_pmf))
            minute = int(rng.integers(0, 60))
            second = int(rng.integers(0, 60))
            ts = datetime(d.year, d.month, d.day, hour, minute, second, tzinfo=UTC)

            if binge_active and binge_album is not None and rng.random() < 0.75:
                # binge: force a track from the one dominant album (mixture unchanged)
                a_idx = arch_index[binge_arch]
                idxs = universe.albums[a_idx][binge_album]
                t = universe.pools[a_idx][idxs[int(rng.integers(0, len(idxs)))]]
            else:
                a_name = names[int(rng.choice(len(names), p=probs))]
                a_idx = arch_index[a_name]
                t = _pick_track(a_idx, discovery[r_idx])
            _register(t)

            if rng.random() < 0.15:  # a skip
                ms = int(rng.integers(1000, 30000))
                skipped = True
            else:
                ms = int(np.clip(rng.normal(t.duration_ms, 8000), 30000, t.duration_ms + 20000))
                skipped = False
            platform = platforms[int(rng.choice(len(platforms), p=platform_w))]
            events.append(ListenEvent(
                ts=ts, track_uri=t.uri, track_name=t.name, artist_name=t.artist_name,
                ms_played=ms, album_name=t.album_name, skipped=skipped, platform=platform,
            ))

    events.sort(key=lambda e: e.ts)

    if include_traps and binge_span is not None and binge_album is not None:
        alb_name = tracks[next(u for u, ht in tracks.items() if ht.album_uri == binge_album)].album_name \
            if any(ht.album_uri == binge_album for ht in tracks.values()) else binge_album
        traps.append(Trap(
            "binge", binge_span[0], binge_span[1],
            f"one album ('{alb_name}') dominates while the mixture is unchanged",
        ))

    # ---- seasonal spans (annotated, NOT change points) -------------------
    seasonal_spans: List[SeasonalSpan] = []
    if seasonal:
        for year in range(start_date.year, end_date.year + 1):
            s = date(year, 12, 1)
            e = date(year, 12, 31)
            if s < end_date and e >= start_date:
                seasonal_spans.append(SeasonalSpan(
                    max(s, start_date), min(e, end_date - timedelta(days=1)) + timedelta(days=1),
                    _SEASONAL_ARCHETYPE,
                    f"December bump of '{_SEASONAL_ARCHETYPE}' (seasonal, recurring — not a taste change)",
                ))

    gt = HistoryGroundTruth(
        regimes=regimes,
        change_points=change_points,
        traps=traps,
        seasonal_spans=seasonal_spans,
        seed=seed,
        params={
            "start_date": start_date.isoformat(),
            "n_days": n_days,
            "base_events_per_day": base_events_per_day,
            "include_traps": include_traps,
            "seasonal": seasonal,
        },
    )
    return ListeningHistory(
        events=events, tracks=tracks, provenance="synthetic", ground_truth=gt,
    )


def make_listener_population(n_listeners: int, seed: int = 0, **kw) -> List[ListeningHistory]:
    """Generate ``n_listeners`` independent, deterministic listening histories.

    Listener ``i`` uses ``seed + i`` so the population is reproducible from ``seed`` and
    every listener differs.  Extra keyword args pass through to
    :func:`make_synthetic_history`.
    """
    return [make_synthetic_history(seed=seed + i, **kw) for i in range(n_listeners)]
