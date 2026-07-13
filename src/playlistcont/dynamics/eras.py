"""Named *eras* — the timeline segmented at detected taste changes (Phase 4).

Phase 3 gives change dates; this module turns the spans *between* them into the
product's headline object: a sequence of :class:`Era`, each with a human name, its
signature sound, who and what defined it, and the honest diagnostics.  Everything
here is **ground-truth-blind** — eras are cut only from detector output (or an
explicit list of dates) and the play stream itself; the planted truth is never
consulted (the tests pass planted dates *in* as if they were detections, they do
not let the builder peek).

WHAT ONE ERA CARRIES
--------------------
* ``start`` / ``end`` (inclusive display end) and ``duration_days``;
* ``name`` — the **dominant fit-once flavor** over the era's feature-plays, named
  with the taste engine's :func:`~playlistcont.models.taste_engine.name_flavor`.
  Flavors are frozen once on the whole history (so identities are comparable
  across eras — the same discipline as :mod:`~playlistcont.dynamics.flavors`);
  the era's name is the most-played cluster's name.  **Tie-break:** the lowest
  cluster id wins (documented, deterministic);
* ``top_artists`` / ``top_tracks`` — the five most-played, by play count;
* ``exemplar_tracks`` — the three tracks (with features) nearest the era's
  scalar-axis centroid: the tracks that most *typify* the era's sound;
* ``mean_axes`` — mean of each scalar mood axis over the era's feature-plays;
* ``dominant_genres`` — the era's genre-bucket shares, top first;
* ``discovery_rate`` — share of the era's plays that were first-ever plays of
  their track (first-ever computed over the whole history);
* ``plays_per_day`` and ``coverage`` (share of plays carrying features).

PROVISIONAL ERAS
----------------
An era whose *opening* change falls in the **final December** of the history is
marked ``provisional=True``: per DYNAMICS.md a December bump with no following
January is fundamentally indistinguishable from a persistent change, so we flag
it rather than assert it ("this may be a holiday spike — check back in January").
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from typing import List, Optional, Sequence, Union

import numpy as np
import pandas as pd

from ..data.schema import GENRES, SCALAR_AXES
from ..history.schema import ListeningHistory
from ..history.store import HistoryStore, feature_column
from .detectors import DetectedChange, recommended_detector
from .flavors import FlavorModel, fit_flavors

SCALAR_COLUMNS = [feature_column(a) for a in SCALAR_AXES]
GENRE_COLUMNS = [feature_column(f"genre:{g}") for g in GENRES]


@dataclass
class Era:
    """One named span of taste between two detected changes (see module docstring)."""

    index: int
    start: date
    end: date                        # inclusive display end
    duration_days: int
    name: str
    flavor_id: int
    top_artists: List[dict]          # [{name, plays}]
    top_tracks: List[dict]           # [{name, artist, plays}]
    exemplar_tracks: List[dict]      # [{name, artist, distance}]
    mean_axes: dict                  # axis -> float
    dominant_genres: List[dict]      # [{genre, share}]
    discovery_rate: float
    plays_per_day: float
    coverage: float
    n_plays: int
    n_feature_plays: int
    provisional: bool = False
    opening_change: Optional[str] = None   # ISO date of the change that opens the era

    def to_payload(self) -> dict:
        return {
            "index": self.index,
            "start": self.start.isoformat(),
            "end": self.end.isoformat(),
            "duration_days": self.duration_days,
            "name": self.name,
            "flavor_id": self.flavor_id,
            "top_artists": self.top_artists,
            "top_tracks": self.top_tracks,
            "exemplar_tracks": self.exemplar_tracks,
            "mean_axes": self.mean_axes,
            "dominant_genres": self.dominant_genres,
            "discovery_rate": self.discovery_rate,
            "plays_per_day": self.plays_per_day,
            "coverage": self.coverage,
            "n_plays": self.n_plays,
            "n_feature_plays": self.n_feature_plays,
            "provisional": self.provisional,
            "opening_change": self.opening_change,
        }


def _to_date(x) -> date:
    if isinstance(x, datetime):
        return x.date()
    if isinstance(x, date):
        return x
    return pd.Timestamp(x).date()


def _era_play_frame(store: HistoryStore) -> pd.DataFrame:
    """One row per play: date, uri, names, is_first, has_features, scalars, genre_bucket."""
    scal = ", ".join(f"f.{c} AS {c}" for c in SCALAR_COLUMNS)
    gen = ", ".join(f"f.{c} AS {c}" for c in GENRE_COLUMNS)
    df = store.query(
        f"""
        WITH firsts AS (
          SELECT track_uri, MIN(event_id) AS fe FROM events GROUP BY track_uri
        )
        SELECT e.event_id AS event_id, e.date AS d, e.track_uri AS uri,
               t.track_name AS track_name, t.artist_name AS artist_name,
               (e.event_id = fr.fe) AS is_first, {scal}, {gen}
        FROM events e
        LEFT JOIN firsts fr ON e.track_uri = fr.track_uri
        LEFT JOIN tracks t ON e.track_uri = t.uri
        LEFT JOIN track_features f ON e.track_uri = f.uri
        ORDER BY e.event_id
        """
    )
    if not len(df):
        df["has_features"] = []
        df["genre_bucket"] = []
        return df
    df["d"] = df["d"].map(_to_date)
    scal_m = df[SCALAR_COLUMNS].to_numpy(dtype=float)
    has_feat = ~np.isnan(scal_m).any(axis=1)
    df["has_features"] = has_feat
    g = df[GENRE_COLUMNS].to_numpy(dtype=float)
    bucket = np.full(len(df), -1, dtype=int)
    if has_feat.any():
        bucket[has_feat] = g[has_feat].argmax(axis=1)
    df["genre_bucket"] = bucket
    df["is_first"] = df["is_first"].astype(bool)
    return df


def _in_final_december(change: date, last: date) -> bool:
    """Is ``change`` in the history's final December (the end-of-history ambiguity)?

    True when the change lands in a December that the history does not survive past
    the following January — i.e. there is no full subsequent year to prove the bump
    reverted (DYNAMICS.md's end-of-history identifiability limit).
    """
    if change.month != 12:
        return False
    if last.year == change.year and last.month == 12:
        return True
    if last.year == change.year + 1 and last.month == 1:
        return True
    return False


def build_eras(
    history_or_store: Union[ListeningHistory, HistoryStore],
    detections: Optional[Sequence[DetectedChange]] = None,
    flavor_model: Optional[FlavorModel] = None,
    flavor_k: int = 4,
    flavor_seed: int = 0,
) -> List[Era]:
    """Segment the timeline at ``detections`` into named :class:`Era` objects.

    ``detections`` defaults to :func:`recommended_detector`'s output on the history.
    Pass an explicit list (e.g. planted change dates, in tests) to pin the cuts.
    Flavors are frozen once on the whole history unless a ``flavor_model`` is given
    (the Journey view passes the window builder's model so names agree everywhere).
    """
    store = (history_or_store if isinstance(history_or_store, HistoryStore)
             else HistoryStore.from_history(history_or_store))
    if detections is None:
        detections = recommended_detector().detect(store)

    df = _era_play_frame(store)
    if not len(df):
        return []

    first, last = min(df["d"]), max(df["d"])
    cut_dates = sorted({_to_date(dc.date) for dc in detections
                        if first < _to_date(dc.date) <= last})

    if flavor_model is None:
        feats = store.query(f"SELECT {', '.join(SCALAR_COLUMNS)} FROM track_features")
        full = store.query(
            f"SELECT {', '.join(SCALAR_COLUMNS + GENRE_COLUMNS)} FROM track_features")
        flavor_model = fit_flavors(
            feats.to_numpy(dtype=float) if len(feats) else np.empty((0, len(SCALAR_COLUMNS))),
            k=flavor_k, seed=flavor_seed,
            full_features=full.to_numpy(dtype=float) if len(full) else None)

    # era boundaries: [first, cut0), [cut0, cut1), ..., [cutN, last]
    bounds: List[tuple] = []
    starts = [first] + list(cut_dates)
    for i, s in enumerate(starts):
        end_excl = cut_dates[i] if i < len(cut_dates) else last + timedelta(days=1)
        opening = starts[i] if i > 0 else None
        bounds.append((s, end_excl, opening))

    day = df["d"].to_numpy()
    eras: List[Era] = []
    for i, (s, end_excl, opening) in enumerate(bounds):
        sub = df[(day >= s) & (day < end_excl)]
        end_incl = end_excl - timedelta(days=1)
        eras.append(_build_one(i, s, end_incl, sub, flavor_model, opening, last))
    return eras


def _build_one(index: int, start: date, end: date, sub: pd.DataFrame,
               flavor_model: FlavorModel, opening: Optional[date],
               last: date) -> Era:
    duration = (end - start).days + 1
    n_plays = int(len(sub))
    feat = sub[sub["has_features"].to_numpy(dtype=bool)] if n_plays else sub
    n_feat = int(len(feat))

    # top artists / tracks by plays
    top_artists = [
        {"name": str(name), "plays": int(cnt)}
        for name, cnt in sub["artist_name"].value_counts().head(5).items()
    ] if n_plays else []
    top_tracks = []
    if n_plays:
        by_uri = sub.groupby("uri")
        counts = by_uri.size().sort_values(ascending=False).head(5)
        for uri, cnt in counts.items():
            row = sub[sub["uri"] == uri].iloc[0]
            top_tracks.append({"name": str(row["track_name"]),
                               "artist": str(row["artist_name"]), "plays": int(cnt)})

    # mean scalar axes + centroid + genre shares over feature-plays
    mean_axes = {a: 0.0 for a in SCALAR_AXES}
    dominant_genres: List[dict] = []
    exemplars: List[dict] = []
    flavor_id = 0
    if n_feat:
        scal = feat[SCALAR_COLUMNS].to_numpy(dtype=float)
        centroid = scal.mean(axis=0)
        for j, a in enumerate(SCALAR_AXES):
            mean_axes[a] = float(centroid[j])

        gb = feat["genre_bucket"].to_numpy(dtype=int)
        shares = np.bincount(gb[gb >= 0], minlength=len(GENRES)) / max(1, len(feat))
        order = np.argsort(-shares)
        dominant_genres = [{"genre": GENRES[j], "share": float(shares[j])}
                           for j in order if shares[j] > 0][:5]

        labels = flavor_model.assign(scal)
        if len(labels):
            counts = np.bincount(labels, minlength=flavor_model.k)
            flavor_id = int(counts.argmax())   # ties -> lowest id (argmax convention)

        # exemplars: distinct feature-tracks nearest the centroid
        exu = {}
        for pos in range(len(feat)):
            uri = feat.iloc[pos]["uri"]
            if uri not in exu:
                exu[uri] = (scal[pos], str(feat.iloc[pos]["track_name"]),
                            str(feat.iloc[pos]["artist_name"]))
        scored = sorted(
            ((float(np.linalg.norm(v[0] - centroid)), v[1], v[2]) for v in exu.values()),
            key=lambda t: (t[0], t[1]))
        exemplars = [{"name": n, "artist": ar, "distance": round(dist, 4)}
                     for dist, n, ar in scored[:3]]

    name = flavor_model.names[flavor_id] if flavor_model.names else "your listening"
    discovery = float(sub["is_first"].mean()) if n_plays else 0.0
    coverage = (n_feat / n_plays) if n_plays else 0.0
    provisional = bool(opening is not None and _in_final_december(opening, last))

    return Era(
        index=index, start=start, end=end, duration_days=duration,
        name=name, flavor_id=flavor_id,
        top_artists=top_artists, top_tracks=top_tracks, exemplar_tracks=exemplars,
        mean_axes=mean_axes, dominant_genres=dominant_genres,
        discovery_rate=round(discovery, 4), plays_per_day=round(n_plays / duration, 3) if duration else 0.0,
        coverage=round(coverage, 4), n_plays=n_plays, n_feature_plays=n_feat,
        provisional=provisional,
        opening_change=opening.isoformat() if opening else None)
