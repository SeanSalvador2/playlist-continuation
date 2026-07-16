"""Adaptive cohort *dossiers* for a span — who and what defined it, and its name.

This is the repo home of the lift-cast / tiered-naming logic prototyped in the
Wave-5 dossier scripts, generalised so it runs on any listener's store with no
external inputs, and degrades gracefully when the optional co-listening cluster
map is absent.

WHAT A COHORT CARRIES
---------------------
For a span ``[start, end)`` of the play stream:

* an **adaptive lift cast** — artists whose *share within the span* is materially
  higher than their *whole-history share* (``lift = seg_share / all_share``).  We
  keep artists with ``lift >= 1.5`` **and** ``>= 15`` plays, then cut the sorted
  lift list at its **largest relative gap** (an elbow), bounded to ``[4, 20]``.
  These are the span's *distinctive* voices, not merely its loudest;
* a **raw-share cast** — the top-8 artists by plain play share, i.e. the span's
  *prominent* voices (what a listener would call the era after).  Naming anchors
  here, because lift saturates on long spans (every span-exclusive niche artist
  ties at the exclusivity ceiling ``total / span_plays``);
* **signature tracks** — the same adaptive-lift treatment on tracks (``lift >= 2``,
  ``>= 8`` plays, elbow bounded ``[6, 30]``);
* a **feature profile** — mean scalar axes, genre-bucket shares, plays/day,
  discovery rate, coverage;
* an optional **cluster mix** — share of the span's plays per co-listening cluster
  (plus an honest ``unmapped`` bucket) when a cluster assignment is supplied;
* **tiered names** — ``name_short`` (<= 4 words, a timeline label), ``name_expanded``
  (<= 14 words, a tooltip), and ``explanation`` (2-3 sentences drawn only from
  computed values).

THE >50% MONOLITH RULE and CLUSTER-MIX NAMING are ported verbatim: when one artist
owns over half the span the name says so; the genre word in a name comes from the
dominant co-listening cluster when clusters are supplied, else from the span's own
dominant genre bucket.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

from ..analytics.stats import significant_shifts
from ..data.schema import GENRES, SCALAR_AXES
from ..history.schema import ListeningHistory
from ..history.store import HistoryStore
from .eras import _era_play_frame

# adaptive-cast tuning (ported from Wave-5; see module docstring)
CAST_MIN_LIFT = 1.5
CAST_MIN_PLAYS = 15
CAST_BOUNDS = (4, 20)
TRACK_MIN_LIFT = 2.0
TRACK_MIN_PLAYS = 8
TRACK_BOUNDS = (6, 30)
SMALL_SAMPLE_WEEKS = 10.0
MONOLITH_SHARE = 0.50


def adaptive_cast_size(lifts: Sequence[float], lo: int, hi: int) -> int:
    """Cut a lift-ranked (descending) list at its largest relative gap, bounded ``[lo, hi]``.

    Walks candidate cut points ``n`` in ``[lo, hi]`` and keeps the ``n`` that maximises
    ``lifts[n-1] / lifts[n]`` (the biggest fall-off in the curve); a cut at the end of
    the list scores infinitely (nothing beyond to keep).  Returns ``k`` clamped to the
    list length when the list is shorter than ``lo``.
    """
    k = len(lifts)
    if k <= lo:
        return k
    hi = min(hi, k)
    if hi <= lo:
        return min(lo, k)
    best_n, best_gap = lo, -1.0
    for n in range(lo, hi + 1):
        gap = (lifts[n - 1] / max(lifts[n], 1e-9)) if n < k else 9e9
        if gap > best_gap:
            best_gap, best_n = gap, n
    return best_n


def _r(x, n: int = 4) -> float:
    return round(float(x), n)


@dataclass
class Cohort:
    """A named dossier for one span (see the module docstring)."""

    start: date
    end: date                              # inclusive display end
    n_plays: int
    n_weeks: float
    small_sample: bool
    cast: List[dict]                       # adaptive lift cast
    cast_raw: List[dict]                   # top raw-share cast
    signature_tracks: List[dict]
    feature_profile: dict
    cluster_mix: Optional[dict]
    name_short: str
    name_expanded: str
    explanation: str
    n_cast_qualifying: int = 0
    shift_swing: Optional[str] = None      # genre this span swung into vs the prior one
    shift_axes: List[str] = field(default_factory=list)  # <=2 lowercase mood-move phrases

    @property
    def top_artists(self) -> List[str]:
        """The span's prominent (raw-share) artist names, most-played first."""
        return [r["artist"] for r in self.cast_raw]

    def shift_phrase(self) -> str:
        """A lowercase, number-free 'moved like this vs the prior span' phrase (or '')."""
        moves = ([f"swung into {self.shift_swing}"] if self.shift_swing else []) \
            + list(self.shift_axes)
        return ", ".join(moves)

    def to_payload(self) -> dict:
        return {
            "start": self.start.isoformat(),
            "end": self.end.isoformat(),
            "n_plays": self.n_plays,
            "n_weeks": self.n_weeks,
            "small_sample": self.small_sample,
            "name_short": self.name_short,
            "name_expanded": self.name_expanded,
            "explanation": self.explanation,
            "shift_swing": self.shift_swing,
            "shift_axes": self.shift_axes,
            "cast": self.cast,
            "cast_raw": self.cast_raw,
            "n_cast_qualifying": self.n_cast_qualifying,
            "signature_tracks": self.signature_tracks,
            "feature_profile": self.feature_profile,
            "cluster_mix": self.cluster_mix,
        }


@dataclass
class CohortModel:
    """Per-play arrays + whole-history shares, built once, sliced per span.

    Build via :meth:`from_store`.  ``clusters`` (optional) maps a play's ``track_uri``
    to an integer co-listening cluster id; when absent the cluster mix and cluster-based
    naming degrade gracefully to the span's own dominant genre bucket.
    """

    dates_ord: np.ndarray
    artist: np.ndarray
    track: np.ndarray
    uri: np.ndarray
    has_feat: np.ndarray
    is_first: np.ndarray
    scal: np.ndarray
    gbucket: np.ndarray
    art_share_all: Dict[str, float]
    uri_share_all: Dict[str, float]
    uri_label: Dict[str, str]
    total: int
    cluster: Optional[np.ndarray] = None
    n_clusters: int = 0
    cluster_names: Dict[int, str] = field(default_factory=dict)
    cluster_genres: Dict[int, Optional[str]] = field(default_factory=dict)

    @classmethod
    def from_store(
        cls,
        history_or_store: Union[ListeningHistory, HistoryStore],
        clusters: Optional[Dict[str, int]] = None,
        cluster_names: Optional[Dict[int, str]] = None,
        cluster_genres: Optional[Dict[int, Optional[str]]] = None,
    ) -> "CohortModel":
        store = (history_or_store if isinstance(history_or_store, HistoryStore)
                 else HistoryStore.from_history(history_or_store))
        from .windows import SCALAR_COLUMNS
        df = _era_play_frame(store)
        n = len(df)
        dates_ord = np.array([d.toordinal() for d in df["d"]], dtype=np.int64) if n \
            else np.empty(0, dtype=np.int64)
        artist = df["artist_name"].astype(str).to_numpy() if n else np.empty(0, dtype=object)
        track = df["track_name"].astype(str).to_numpy() if n else np.empty(0, dtype=object)
        uri = df["uri"].astype(str).to_numpy() if n else np.empty(0, dtype=object)
        has_feat = df["has_features"].to_numpy(dtype=bool) if n else np.empty(0, dtype=bool)
        is_first = df["is_first"].to_numpy(dtype=bool) if n else np.empty(0, dtype=bool)
        scal = df[SCALAR_COLUMNS].to_numpy(dtype=float) if n else np.empty((0, len(SCALAR_COLUMNS)))
        gbucket = df["genre_bucket"].to_numpy(dtype=int) if n else np.empty(0, dtype=int)

        total = int(n)
        art_share_all: Dict[str, float] = {}
        uri_share_all: Dict[str, float] = {}
        uri_label: Dict[str, str] = {}
        if n:
            ua, ca = np.unique(artist, return_counts=True)
            art_share_all = {a: c / total for a, c in zip(ua, ca)}
            uu, cu = np.unique(uri, return_counts=True)
            uri_share_all = {u: c / total for u, c in zip(uu, cu)}
            for i in range(n):
                if uri[i] not in uri_label:
                    uri_label[uri[i]] = f"{track[i]} — {artist[i]}"

        cluster = None
        n_clusters = 0
        if clusters is not None and n:
            cluster = np.array([int(clusters.get(u, -1)) for u in uri], dtype=int)
            mapped = cluster[cluster >= 0]
            n_clusters = int(mapped.max()) + 1 if len(mapped) else 0
            if cluster_names:
                n_clusters = max(n_clusters, max(cluster_names) + 1)

        return cls(
            dates_ord=dates_ord, artist=artist, track=track, uri=uri,
            has_feat=has_feat, is_first=is_first, scal=scal, gbucket=gbucket,
            art_share_all=art_share_all, uri_share_all=uri_share_all,
            uri_label=uri_label, total=total, cluster=cluster, n_clusters=n_clusters,
            cluster_names=dict(cluster_names or {}),
            cluster_genres=dict(cluster_genres or {}))

    # ------------------------------------------------------------------ #
    def _mask(self, start: date, end_excl: date) -> np.ndarray:
        return (self.dates_ord >= start.toordinal()) & (self.dates_ord < end_excl.toordinal())


# =========================================================================== #
# per-span computation
# =========================================================================== #
def _cast_and_tracks(model: CohortModel, m: np.ndarray) -> Tuple[dict, int]:
    sp = int(m.sum())
    out = {"cast": [], "cast_raw": [], "tracks": [], "cast_size": 0, "signature_size": 0}
    if sp == 0:
        return out, 0
    a_seg, a_c = np.unique(model.artist[m], return_counts=True)
    a_share = a_c / sp
    rows = []
    for a, c, s in zip(a_seg, a_c, a_share):
        if c >= CAST_MIN_PLAYS:
            lift = s / model.art_share_all[a]
            if lift >= CAST_MIN_LIFT:
                rows.append((str(a), lift, int(c), float(s)))
    rows.sort(key=lambda r: (-r[1], -r[2], r[0]))
    n_cast_total = len(rows)
    n_keep = adaptive_cast_size([r[1] for r in rows], *CAST_BOUNDS)
    out["cast"] = [{"artist": r[0], "lift": _r(r[1]), "plays": r[2], "share": _r(r[3])}
                   for r in rows[:n_keep]]
    out["cast_size"] = len(out["cast"])
    order = np.argsort(-a_c)[:8]
    out["cast_raw"] = [{"artist": str(a_seg[i]), "plays": int(a_c[i]), "share": _r(a_share[i])}
                       for i in order]

    u_seg, u_c = np.unique(model.uri[m], return_counts=True)
    u_share = u_c / sp
    trows = []
    for u, c, s in zip(u_seg, u_c, u_share):
        if c >= TRACK_MIN_PLAYS:
            lift = s / model.uri_share_all[u]
            if lift >= TRACK_MIN_LIFT:
                trows.append((str(u), lift, int(c), float(s)))
    trows.sort(key=lambda r: (-r[1], -r[2], r[0]))
    tkeep = adaptive_cast_size([r[1] for r in trows], *TRACK_BOUNDS)
    out["tracks"] = [{"track": model.uri_label.get(r[0], r[0]), "lift": _r(r[1]),
                      "plays": r[2], "share": _r(r[3])} for r in trows[:tkeep]]
    out["signature_size"] = len(out["tracks"])
    return out, n_cast_total


def _cluster_mix(model: CohortModel, m: np.ndarray) -> Optional[dict]:
    if model.cluster is None:
        return None
    sp = int(m.sum())
    if sp == 0:
        return {"by_cluster": {}, "unmapped": 0.0}
    cl = model.cluster[m]
    mix = {}
    for cid in range(model.n_clusters):
        cnt = int((cl == cid).sum())
        if cnt:
            mix[str(cid)] = _r(cnt / sp)
    return {"by_cluster": mix, "unmapped": _r(int((cl == -1).sum()) / sp)}


def _feature_profile(model: CohortModel, m: np.ndarray, start: date, end_excl: date) -> dict:
    sp = int(m.sum())
    days = max(1, (end_excl - start).days)
    fm = m & model.has_feat
    nf = int(fm.sum())
    prof = {"plays": sp, "plays_per_day": _r(sp / days, 2),
            "discovery_rate": _r(model.is_first[m].mean() if sp else 0.0),
            "coverage": _r(nf / sp if sp else 0.0), "mean_axes": {}, "genre_shares": {}}
    if nf:
        cen = model.scal[fm].mean(axis=0)
        prof["mean_axes"] = {a: _r(cen[j]) for j, a in enumerate(SCALAR_AXES)}
        gb = model.gbucket[fm]
        gb = gb[gb >= 0]
        shares = np.bincount(gb, minlength=len(GENRES)) / max(1, nf)
        prof["genre_shares"] = {GENRES[j]: _r(shares[j]) for j in np.argsort(-shares)
                                if shares[j] > 0.02}
    return prof


# ---------------------------------------------------------------- naming ---- #
def _handle(name: str) -> str:
    """Short one-word handle for a timeline label (surname-ish token)."""
    n = str(name)
    if n.lower().startswith("the "):
        n = n[4:]
    toks = n.replace(".", "").split()
    if not toks:
        return str(name)
    if len(toks) == 2:
        return toks[1]
    return toks[0]


def _dom_cluster(cmix: Optional[dict]) -> Tuple[Optional[int], float]:
    if not cmix or not cmix.get("by_cluster"):
        return None, 0.0
    cid, sh = max(cmix["by_cluster"].items(), key=lambda kv: kv[1])
    return int(cid), float(sh)


def _profile_top_genre(prof: dict) -> Optional[str]:
    gs = prof.get("genre_shares") or {}
    return next(iter(gs), None) if gs else None


def _genre_word(model: CohortModel, cmix: Optional[dict], prof: dict) -> Optional[str]:
    """Genre word: the dominant co-listening cluster's genre when available, else the
    span's own dominant genre bucket."""
    cid, _ = _dom_cluster(cmix)
    if cid is not None and model.cluster_genres.get(cid):
        return model.cluster_genres[cid]
    return _profile_top_genre(prof)


_UP = {"acousticness": "more acoustic", "valence": "brighter", "lyrical_depth": "wordier",
       "energy": "more intense", "tempo": "faster"}
_DN = {"acousticness": "more electronic", "valence": "moodier", "lyrical_depth": "lighter",
       "energy": "calmer", "tempo": "slower"}


def _shift_phrases(prev_prof, cur_prof, prev_gen, cur_gen, thr: float = 0.03):
    """(genre-swing-or-None, [<=2 axis phrases]) describing ``cur`` vs ``prev``."""
    if not prev_prof or not prev_prof.get("mean_axes") or not cur_prof.get("mean_axes"):
        return None, []
    pa, ca = prev_prof["mean_axes"], cur_prof["mean_axes"]
    deltas = []
    for ax in ca:
        d = ca[ax] - pa.get(ax, ca[ax])
        if abs(d) >= thr:
            deltas.append((abs(d), _UP[ax] if d > 0 else _DN[ax]))
    deltas.sort(key=lambda x: -x[0])
    swing = cur_gen if (cur_gen and cur_gen != prev_gen) else None
    return swing, [p for _, p in deltas[:2]]


def _article(n: int) -> str:
    return "An" if n in (8, 11, 18) else "A"


def _names(model, cast, prof, cmix, prev_prof, prev_cmix, is_first) -> dict:
    raw = cast["cast_raw"]
    axes = prof["mean_axes"]
    gen = _genre_word(model, cmix, prof)
    prev_gen = _genre_word(model, prev_cmix, prev_prof) if prev_prof else None
    monolith = bool(raw and raw[0]["share"] > MONOLITH_SHARE)
    n = cast["cast_size"]
    a1 = raw[0]["artist"] if raw else None
    a2 = raw[1]["artist"] if len(raw) > 1 else None

    # ---- name_short (<= 4 words)
    if monolith:
        short = f"The {a1} monolith"
    elif a1 and a2:
        short = f"The {_handle(a1)}–{_handle(a2)} axis"
    elif a1:
        short = f"The {_handle(a1)} run"
    else:
        short = f"Quiet {gen or 'mixed'} stretch"

    # ---- name_expanded (<= 14 words)
    swing, ax_shifts = _shift_phrases(prev_prof, prof, prev_gen, gen)
    art = _article(n)
    if monolith:
        exp = f"The {a1} monolith — over half of every play in this span"
    elif not raw:
        exp = f"A quiet {gen or 'mixed'} stretch with no dominant cast"
    elif swing:
        exp = f"{art} {n}-artist cast swinging into {swing}"
        if ax_shifts:
            exp += ", " + ", ".join(ax_shifts)
    elif ax_shifts:
        exp = f"{art} {n}-artist {gen or 'mixed'} cast turning " + ", ".join(ax_shifts)
    else:
        mood = ("acoustic" if axes.get("acousticness", 0) >= 0.55 else
                "high-energy" if axes.get("energy", 0) >= 0.6 else
                "moody" if axes.get("valence", 1) <= 0.4 else "mid-tempo")
        exp = f"{art} {n}-artist {gen or 'mixed'} cast, {mood}"
    exp = " ".join(exp.split()[:14])

    # ---- explanation (2-3 sentences)
    anchor_txt = ", ".join(f"{r['artist']} ({int(r['share'] * 100)}%)" for r in raw[:3]) \
        if raw else "no dominant cast"
    s1 = f"Anchored by {anchor_txt}."
    if is_first:
        s2 = "Opening segment of this view."
    elif swing or ax_shifts:
        moves = ([f"swings into {swing}"] if swing else []) + ax_shifts
        s2 = f"Versus the prior segment it {', '.join(moves)}."
    else:
        s2 = "Little movement from the prior segment."
    if cast["tracks"]:
        t = cast["tracks"][0]
        fact = f"Signature (high-lift) track: {t['track']} (lift {t['lift']}, {t['plays']} plays)."
    else:
        cid, sh = _dom_cluster(cmix)
        fact = (f"Dominant co-listening cluster: {model.cluster_names.get(cid, cid)} "
                f"({int(sh * 100)}%).") if cid is not None else ""
    explanation = " ".join(x for x in [s1, s2, fact] if x)
    return {"name_short": short, "name_expanded": exp, "explanation": explanation,
            "shift_swing": swing, "shift_axes": ax_shifts}


def build_cohort(
    model: CohortModel, start: date, end: date, prev: Optional[Cohort] = None,
    is_first: bool = False, end_inclusive: bool = True,
) -> Cohort:
    """Build a :class:`Cohort` dossier for the span ``[start, end]`` (end inclusive).

    ``prev`` (the preceding cohort in a sequence) drives the *shift-versus-prior*
    phrasing in the names; ``is_first`` marks the opening span of a view.  Set
    ``end_inclusive=False`` to treat ``end`` as an exclusive bound.
    """
    end_excl = (end + timedelta(days=1)) if end_inclusive else end
    end_incl = end if end_inclusive else (end - timedelta(days=1))
    m = model._mask(start, end_excl)
    cast, n_cast_total = _cast_and_tracks(model, m)
    prof = _feature_profile(model, m, start, end_excl)
    cmix = _cluster_mix(model, m)
    prev_prof = prev.feature_profile if prev else None
    prev_cmix = prev.cluster_mix if prev else None
    nm = _names(model, cast, prof, cmix, prev_prof, prev_cmix, is_first)
    n_weeks = round((end_excl - start).days / 7, 1)
    return Cohort(
        start=start, end=end_incl, n_plays=int(m.sum()), n_weeks=n_weeks,
        small_sample=n_weeks < SMALL_SAMPLE_WEEKS, cast=cast["cast"],
        cast_raw=cast["cast_raw"], signature_tracks=cast["tracks"],
        feature_profile=prof, cluster_mix=cmix, name_short=nm["name_short"],
        name_expanded=nm["name_expanded"], explanation=nm["explanation"],
        n_cast_qualifying=n_cast_total, shift_swing=nm["shift_swing"],
        shift_axes=list(nm["shift_axes"]))


def cohorts_for_spans(
    model: CohortModel, spans: Sequence[Tuple[date, date]], end_inclusive: bool = True,
) -> List[Cohort]:
    """Build cohorts for an ordered list of ``(start, end)`` spans, chaining the
    shift-versus-prior context down the sequence."""
    out: List[Cohort] = []
    prev: Optional[Cohort] = None
    for i, (s, e) in enumerate(spans):
        c = build_cohort(model, s, e, prev=prev, is_first=(i == 0),
                         end_inclusive=end_inclusive)
        out.append(c)
        prev = c
    return out


# ------------------------------------------------------------ boundary diff -- #
def _artist_shares(model: CohortModel, m: np.ndarray) -> Tuple[Dict[str, Tuple[int, float]], int]:
    sp = int(m.sum())
    if sp == 0:
        return {}, 0
    a, c = np.unique(model.artist[m], return_counts=True)
    return {str(a[i]): (int(c[i]), c[i] / sp) for i in range(len(a))}, sp


def boundary_diff(
    model: CohortModel, prev_span: Tuple[date, date], next_span: Tuple[date, date],
    bdate: date, end_inclusive: bool = True,
) -> dict:
    """Arrivals / departures / constants across a boundary, plus an FDR-gated shift sentence.

    ARRIVALS: artist under 1% before, >= 3% and >= 30 plays after; DEPARTURES mirror it;
    CONSTANTS: >= 2% on both sides.  The feature-shift sentence comes from
    :func:`~playlistcont.analytics.stats.significant_shifts` over equal flanking windows
    inside the two spans (capped at 182 days for boundary-locality); ``None`` when nothing
    survives.  Requires a store for the shift test — pass it via :func:`boundary_diffs`.
    """
    (ps, pe), (ns, ne) = prev_span, next_span
    pe_excl = (pe + timedelta(days=1)) if end_inclusive else pe
    ne_excl = (ne + timedelta(days=1)) if end_inclusive else ne
    sh_p, _ = _artist_shares(model, model._mask(ps, pe_excl))
    sh_n, _ = _artist_shares(model, model._mask(ns, ne_excl))
    arrivals, departures, constants = [], [], []
    for a in set(sh_p) | set(sh_n):
        pc, psh = sh_p.get(a, (0, 0.0))
        nc, nsh = sh_n.get(a, (0, 0.0))
        if psh < 0.01 and nsh >= 0.03 and nc >= 30:
            arrivals.append((a, _r(nsh), nc))
        if psh >= 0.03 and nsh < 0.01 and pc >= 30:
            departures.append((a, _r(psh), pc))
        if psh >= 0.02 and nsh >= 0.02:
            constants.append((a, _r(psh), _r(nsh)))
    arrivals.sort(key=lambda r: (-r[1], r[0]))
    departures.sort(key=lambda r: (-r[1], r[0]))
    constants.sort(key=lambda r: (-(r[1] + r[2]), r[0]))
    flank = min((pe_excl - ps).days, (ne_excl - ns).days, 182)
    return {
        "boundary": bdate.isoformat(), "flank_days": flank,
        "arrivals": [{"artist": r[0], "share_after": r[1], "plays_after": r[2]}
                     for r in arrivals[:8]],
        "departures": [{"artist": r[0], "share_before": r[1], "plays_before": r[2]}
                       for r in departures[:8]],
        "constants": [{"artist": r[0], "share_before": r[1], "share_after": r[2]}
                      for r in constants[:8]],
    }


def feature_shift_sentence(
    store: HistoryStore, bdate: date, flank_days: int,
) -> Optional[str]:
    """The top FDR-surviving shift sentence across ``bdate`` over ``+/- flank_days``.

    ``None`` when nothing survives or the test cannot be run.
    """
    try:
        ss = significant_shifts(store, bdate - timedelta(days=flank_days), bdate,
                                bdate, bdate + timedelta(days=flank_days))
        if ss["shifts"]:
            return ss["shifts"][0]["sentence"]
    except Exception:
        return None
    return None
