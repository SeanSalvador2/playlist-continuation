"""Turn a listening stream into a per-window feature time series.

A change-point detector does not see individual plays; it sees a **sequence of
window vectors** — "what did this week/month sound like?" — and looks for the
moments the vector jumps.  This module builds that sequence, once, in a
labelled matrix with a fixed, documented column order so every downstream
detector and representation-ablation reads the same thing.

WHAT A WINDOW VECTOR CONTAINS (fixed column order)
--------------------------------------------------
For each calendar window (``week`` or ``month``) the row concatenates, in order:

1. **scalar-axis means** (5 cols: tempo, energy, valence, acousticness,
   lyrical-depth) — averaged over the plays *that have features*.
2. **genre-bucket shares** (10 cols) — share of feature-plays whose *dominant*
   genre (argmax of the genre block) is each genre.
3. **flavor shares** (``k`` cols) — share of feature-plays assigned to each
   frozen flavor cluster (:mod:`playlistcont.dynamics.flavors`).
4. **discovery rate** (1 col) — share of plays that are the first-ever play of
   their track (first-ever computed over the *whole* history, matching
   :mod:`playlistcont.analytics.queries`).
5. **intensity** (1 col) — plays per day over the window's calendar length.
6. **coverage** (1 col) — share of the window's plays that carry features.

Columns 1-3 form the **taste representation** the detectors run on
(``scalar_axes`` / ``genre_shares`` / ``flavor_shares`` / ``combined``).
Columns 4-6 are diagnostics deliberately kept *out* of the detection
representations: intensity is exactly the signal a ``volume_only`` trap moves,
so excluding it makes volume traps *structurally* invisible to the detector
(see DYNAMICS.md).  They are still stored for analysis.

WEIGHTING
---------
``weighting="plays"`` (default) weights every play equally: a track played 40
times contributes 40 rows to the window's means and shares.  ``weighting=
"unique"`` first de-duplicates to the *distinct tracks* played in the window and
weights each once — which flattens heavy rotation, so a ``binge`` (one album
played to death, mixture unchanged) barely moves the unique-weighted vector.
Both are offered so the trap behaviour can be studied; the benchmark uses
``plays``.

MASKING (never silently dropped)
--------------------------------
A window with fewer than ``min_events`` total plays is **masked** (``mask[i] =
False``) — its row is kept in place (filled with NaN) so window indices and
dates stay aligned with the calendar, but detectors skip it.  Masking is on the
*total* play count, not the feature-play count.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timedelta
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from ..data.schema import GENRES, SCALAR_AXES
from ..history.schema import ListeningHistory
from ..history.store import HistoryStore, feature_column
from .flavors import FlavorModel, fit_flavors

SCALAR_COLUMNS: List[str] = [feature_column(a) for a in SCALAR_AXES]
GENRE_COLUMNS: List[str] = [feature_column(f"genre:{g}") for g in GENRES]

# ---- the four named detection representations (column-group names) --------- #
GROUP_SCALAR = "scalar_axes"
GROUP_GENRE = "genre_shares"
GROUP_FLAVOR = "flavor_shares"
GROUP_DISCOVERY = "discovery"
GROUP_INTENSITY = "intensity"
GROUP_COVERAGE = "coverage"

REPRESENTATIONS = ("scalar_axes", "genre_shares", "flavor_shares", "combined")


@dataclass
class WindowSeries:
    """A per-window feature matrix with a fixed, group-labelled column order.

    * ``starts`` / ``ends`` — window bounds (``ends`` exclusive), one per row.
    * ``matrix`` — ``(n_windows, n_cols)`` float array; masked rows are NaN.
    * ``columns`` — column names, aligned with ``matrix`` columns.
    * ``column_groups`` — group name -> list of its column names (fixed order).
    * ``mask`` — boolean, ``True`` where the window has >= ``min_events`` plays.
    * ``event_counts`` — total plays per window (feature and non-feature).
    * ``flavor_model`` — the frozen flavors used for the flavor-share columns.
    """

    starts: List[date]
    ends: List[date]
    matrix: np.ndarray
    columns: List[str]
    column_groups: Dict[str, List[str]]
    mask: np.ndarray
    event_counts: np.ndarray
    granularity: str
    weighting: str
    min_events: int
    flavor_model: FlavorModel
    span_days: int = 0

    # ---- convenience selectors ---------------------------------------- #
    @property
    def n_windows(self) -> int:
        return self.matrix.shape[0]

    def _col_index(self, name: str) -> int:
        return self.columns.index(name)

    def representation_columns(self, representation: str) -> List[str]:
        """Column names for a named representation.

        ``"combined"`` is scalar + genre + flavor (the full taste vector); the
        three singletons are their own groups.  Unknown names raise.
        """
        if representation == "combined":
            return (self.column_groups[GROUP_SCALAR]
                    + self.column_groups[GROUP_GENRE]
                    + self.column_groups[GROUP_FLAVOR])
        if representation in self.column_groups:
            return list(self.column_groups[representation])
        raise ValueError(
            f"unknown representation {representation!r}; expected one of "
            f"{REPRESENTATIONS} or a column-group name")

    def submatrix(self, representation: str) -> np.ndarray:
        """The ``(n_windows, n_selected)`` block for a representation (NaN rows kept)."""
        idx = [self._col_index(c) for c in self.representation_columns(representation)]
        return self.matrix[:, idx]

    def valid_indices(self) -> np.ndarray:
        """Row indices of unmasked windows, in order."""
        return np.nonzero(self.mask)[0]


# ===========================================================================
# calendar window construction
# ===========================================================================
def _to_date(x) -> date:
    if isinstance(x, datetime):
        return x.date()
    if isinstance(x, date):
        return x
    return pd.Timestamp(x).date()


def _week_periods(first: date, last: date) -> List[Tuple[date, date]]:
    """Monday-anchored 7-day windows covering ``[first, last]`` (ends exclusive)."""
    start = first - timedelta(days=first.weekday())  # Monday on/before first
    out: List[Tuple[date, date]] = []
    while start <= last:
        out.append((start, start + timedelta(days=7)))
        start = start + timedelta(days=7)
    return out


def _month_periods(first: date, last: date) -> List[Tuple[date, date]]:
    """Calendar-month windows covering ``[first, last]`` (ends exclusive)."""
    out: List[Tuple[date, date]] = []
    y, m = first.year, first.month
    while (y < last.year) or (y == last.year and m <= last.month):
        start = date(y, m, 1)
        nxt = date(y + 1, 1, 1) if m == 12 else date(y, m + 1, 1)
        out.append((start, nxt))
        y, m = (y + 1, 1) if m == 12 else (y, m + 1)
    return out


# ===========================================================================
# the builder
# ===========================================================================
def _play_frame(store: HistoryStore) -> pd.DataFrame:
    """Per-play rows: date, uri, is_first, has_features, scalar cols, genre_bucket."""
    cols = ", ".join(f"f.{c} AS {c}" for c in SCALAR_COLUMNS + GENRE_COLUMNS)
    df = store.query(
        f"""
        WITH firsts AS (
          SELECT track_uri, MIN(event_id) AS fe FROM events GROUP BY track_uri
        )
        SELECT e.event_id AS event_id, e.date AS d, e.track_uri AS uri,
               (e.event_id = fr.fe) AS is_first, {cols}
        FROM events e
        LEFT JOIN firsts fr ON e.track_uri = fr.track_uri
        LEFT JOIN track_features f ON e.track_uri = f.uri
        ORDER BY e.event_id
        """
    )
    if not len(df):
        df["genre_bucket"] = []
        df["has_features"] = []
        return df
    df["d"] = df["d"].map(_to_date)
    scal = df[SCALAR_COLUMNS].to_numpy(dtype=float)
    has_feat = ~np.isnan(scal).any(axis=1)
    df["has_features"] = has_feat
    g = df[GENRE_COLUMNS].to_numpy(dtype=float)
    bucket = np.full(len(df), -1, dtype=int)
    if has_feat.any():
        bucket[has_feat] = g[has_feat].argmax(axis=1)
    df["genre_bucket"] = bucket
    df["is_first"] = df["is_first"].astype(bool)
    return df


def _track_feature_matrix(store: HistoryStore) -> np.ndarray:
    """The union of the listener's per-track scalar-axis vectors (for fitting flavors)."""
    cols = ", ".join(SCALAR_COLUMNS)
    df = store.query(f"SELECT {cols} FROM track_features")
    return df.to_numpy(dtype=float) if len(df) else np.empty((0, len(SCALAR_COLUMNS)))


def _full_track_matrix(store: HistoryStore) -> np.ndarray:
    cols = ", ".join(SCALAR_COLUMNS + GENRE_COLUMNS)
    df = store.query(f"SELECT {cols} FROM track_features")
    return df.to_numpy(dtype=float) if len(df) else np.empty(
        (0, len(SCALAR_COLUMNS) + len(GENRE_COLUMNS)))


def build_windows(
    history_or_store: Union[ListeningHistory, HistoryStore],
    granularity: str = "week",
    min_events: int = 30,
    weighting: str = "plays",
    flavors: Optional[FlavorModel] = None,
    flavor_k: int = 4,
    flavor_seed: int = 0,
) -> WindowSeries:
    """Build a :class:`WindowSeries` from a listening history (or its store).

    ``granularity`` is ``"week"`` (Monday-anchored) or ``"month"`` (calendar).
    ``weighting`` is ``"plays"`` or ``"unique"`` (see module docstring).  Windows
    with fewer than ``min_events`` total plays are masked, not dropped.

    Flavors: if ``flavors`` is ``None`` a :class:`FlavorModel` is fit **once** on
    the union of this listener's track features (``flavor_k`` clusters,
    ``flavor_seed``), guaranteeing stable flavor identities across windows.
    """
    if weighting not in ("plays", "unique"):
        raise ValueError(f"unknown weighting {weighting!r}; expected plays|unique")
    store = (history_or_store if isinstance(history_or_store, HistoryStore)
             else HistoryStore.from_history(history_or_store))

    if flavors is None:
        flavors = fit_flavors(
            _track_feature_matrix(store), k=flavor_k, seed=flavor_seed,
            full_features=_full_track_matrix(store))
    k = flavors.k
    flavor_cols = [f"flavor_{c}" for c in range(k)]

    columns = (list(SCALAR_COLUMNS) + list(GENRE_COLUMNS) + flavor_cols
               + ["discovery_rate", "intensity_plays_per_day", "coverage"])
    column_groups: Dict[str, List[str]] = {
        GROUP_SCALAR: list(SCALAR_COLUMNS),
        GROUP_GENRE: list(GENRE_COLUMNS),
        GROUP_FLAVOR: list(flavor_cols),
        GROUP_DISCOVERY: ["discovery_rate"],
        GROUP_INTENSITY: ["intensity_plays_per_day"],
        GROUP_COVERAGE: ["coverage"],
    }

    df = _play_frame(store)
    if not len(df):
        return WindowSeries(
            starts=[], ends=[], matrix=np.empty((0, len(columns))), columns=columns,
            column_groups=column_groups, mask=np.empty(0, dtype=bool),
            event_counts=np.empty(0, dtype=int), granularity=granularity,
            weighting=weighting, min_events=min_events, flavor_model=flavors,
            span_days=0)

    # assign flavor labels to every feature-play, once, from frozen centroids
    feat_mask = df["has_features"].to_numpy(dtype=bool)
    flavor_label = np.full(len(df), -1, dtype=int)
    if feat_mask.any():
        scal = df.loc[feat_mask, SCALAR_COLUMNS].to_numpy(dtype=float)
        flavor_label[feat_mask] = flavors.assign(scal)
    df["flavor_label"] = flavor_label

    first, last = min(df["d"]), max(df["d"])
    periods = (_week_periods(first, last) if granularity == "week"
               else _month_periods(first, last) if granularity == "month" else None)
    if periods is None:
        raise ValueError(f"unknown granularity {granularity!r}; expected week|month")

    starts = [p[0] for p in periods]
    ends = [p[1] for p in periods]
    # map each play to a window index by date (searchsorted on window starts)
    start_ord = np.array([s.toordinal() for s in starts])
    day_ord = np.array([d.toordinal() for d in df["d"]])
    widx = np.searchsorted(start_ord, day_ord, side="right") - 1

    n_win = len(periods)
    matrix = np.full((n_win, len(columns)), np.nan, dtype=float)
    event_counts = np.zeros(n_win, dtype=int)

    genre_idx = df["genre_bucket"].to_numpy(dtype=int)
    flav_idx = df["flavor_label"].to_numpy(dtype=int)
    is_first = df["is_first"].to_numpy(dtype=bool)
    uris = df["uri"].to_numpy(dtype=object)
    scal_all = df[SCALAR_COLUMNS].to_numpy(dtype=float)

    n_scalar = len(SCALAR_COLUMNS)
    n_genre = len(GENRE_COLUMNS)
    off_flavor = n_scalar + n_genre
    off_disc = off_flavor + k
    off_int = off_disc + 1
    off_cov = off_int + 1

    for wi in range(n_win):
        sel = widx == wi
        n_all = int(sel.sum())
        event_counts[wi] = n_all
        if n_all == 0:
            continue
        days = (ends[wi] - starts[wi]).days or 1
        matrix[wi, off_int] = n_all / days                       # intensity
        f_sel = sel & feat_mask
        n_feat = int(f_sel.sum())
        matrix[wi, off_cov] = n_feat / n_all                     # coverage

        if weighting == "unique":
            # collapse to distinct tracks in-window; each track weighted once
            # (first occurrence per uri kept, so is_first stays meaningful)
            seen: Dict[object, int] = {}
            for j in np.nonzero(sel)[0]:
                seen.setdefault(uris[j], j)
            keep = np.array(sorted(seen.values()), dtype=int)
            row_feat = keep[feat_mask[keep]]
            disc_rows = keep
        else:
            disc_rows = np.nonzero(sel)[0]
            row_feat = np.nonzero(f_sel)[0]

        # discovery over the (weighting-consistent) row set
        if len(disc_rows):
            matrix[wi, off_disc] = float(is_first[disc_rows].mean())

        if len(row_feat):
            matrix[wi, :n_scalar] = scal_all[row_feat].mean(axis=0)
            gb = genre_idx[row_feat]
            gshare = np.bincount(gb, minlength=n_genre)[:n_genre] / len(row_feat)
            matrix[wi, n_scalar:n_scalar + n_genre] = gshare
            fb = flav_idx[row_feat]
            fshare = np.bincount(fb, minlength=k)[:k] / len(row_feat)
            matrix[wi, off_flavor:off_flavor + k] = fshare

    mask = event_counts >= min_events
    # masked rows carry no usable vector -> NaN their whole row for honesty
    matrix[~mask, :] = np.nan

    return WindowSeries(
        starts=starts, ends=ends, matrix=matrix, columns=columns,
        column_groups=column_groups, mask=mask, event_counts=event_counts,
        granularity=granularity, weighting=weighting, min_events=min_events,
        flavor_model=flavors, span_days=(last - first).days + 1)
