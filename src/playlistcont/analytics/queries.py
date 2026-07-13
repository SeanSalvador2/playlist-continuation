"""Pure analytics queries over a :class:`~playlistcont.history.store.HistoryStore`.

Every function here is a *pure translation* of a listening question into SQL over the
three-table store, returning plain dicts/lists ready to hand straight to ``json``.
None of them touch the web layer or the engine; the FastAPI endpoints in
``app/backend`` are a 1:1 thin wrapper.

Windowing
---------
Each query takes optional ``start`` / ``end`` bounds (``date``, ``datetime``, ISO
string, or ``None``).  Both bounds are **inclusive calendar dates**; ``None`` means
"unbounded on that side", so ``start=None, end=None`` is the full history.  The window
filters on the pre-computed ``events.date`` column.

A note on "discovery"
---------------------
:func:`trends` with ``metric="discovery"`` reports the share of plays in each bucket
that are the *first-ever* play of their track.  "First-ever" is computed over the
**full** history (the earliest event of each track across all time), **not** just the
windowed slice — otherwise every track visible at the left edge of a window would look
falsely "new".  So a discovery bucket answers "how much of what I listened to here was
genuinely new to me", regardless of where the window starts.
"""
from __future__ import annotations

import csv
import io
from datetime import date, datetime
from typing import Dict, List, Optional, Sequence, Union

import pandas as pd

from ..data.schema import GENRES, SCALAR_AXES
from ..history.store import HistoryStore, feature_column

DateLike = Union[str, date, datetime, None]

# scalar-axis columns are named identically in track_features (no ':' to sanitise);
# genre columns are 'genre_<g>'.
SCALAR_COLUMNS: List[str] = [feature_column(a) for a in SCALAR_AXES]
GENRE_COLUMNS: List[str] = [feature_column(f"genre:{g}") for g in GENRES]

_TRUNC = {"day": "day", "week": "week", "month": "month"}
_WEEKDAY_LABELS = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]

MS_PER_MIN = 60_000.0


# ---------------------------------------------------------------------------
# window helpers
# ---------------------------------------------------------------------------
def _as_date(d: DateLike) -> Optional[date]:
    if d is None or d == "":
        return None
    if isinstance(d, datetime):
        return d.date()
    if isinstance(d, date):
        return d
    return date.fromisoformat(str(d)[:10])


def _predicates(start: DateLike, end: DateLike, col: str = "date") -> List[str]:
    s, e = _as_date(start), _as_date(end)
    preds: List[str] = []
    if s is not None:
        preds.append(f"{col} >= DATE '{s.isoformat()}'")
    if e is not None:
        preds.append(f"{col} <= DATE '{e.isoformat()}'")
    return preds


def _where(preds: Sequence[str]) -> str:
    return ("WHERE " + " AND ".join(preds)) if preds else ""


def _iso(ts) -> Optional[str]:
    """Render a DuckDB DATE/TIMESTAMP (a pandas Timestamp) as an ISO date string."""
    if ts is None or (isinstance(ts, float) and pd.isna(ts)):
        return None
    return pd.Timestamp(ts).date().isoformat()


def _span(store: HistoryStore, start: DateLike, end: DateLike) -> Optional[dict]:
    where = _where(_predicates(start, end))
    df = store.query(
        f"SELECT MIN(date) AS first, MAX(date) AS last, COUNT(*) AS n FROM events {where}"
    )
    if int(df["n"].iloc[0]) == 0:
        return None
    first = pd.Timestamp(df["first"].iloc[0]).date()
    last = pd.Timestamp(df["last"].iloc[0]).date()
    return {"first": first.isoformat(), "last": last.isoformat(), "days": (last - first).days + 1}


# ---------------------------------------------------------------------------
# summary
# ---------------------------------------------------------------------------
def summary(store: HistoryStore, start: DateLike = None, end: DateLike = None) -> dict:
    """Headline stats for the window: span, plays, minutes, distinct counts, skip rate.

    ``skip_rate`` is over events that carry a non-null skip flag (a real *basic* export
    has none, so it degrades to ``0.0`` with ``skip_flagged == 0``).  ``plays_per_day``
    divides total plays by the number of **calendar days** in the observed span
    (inclusive), not by the number of days with any listening.
    """
    where = _where(_predicates(start, end))
    row = store.query(
        f"""
        SELECT
          COUNT(*)                                   AS plays,
          COALESCE(SUM(ms_played), 0)                AS ms,
          COUNT(DISTINCT track_uri)                  AS tracks,
          COUNT(skipped)                             AS skip_flagged,
          COALESCE(SUM(CASE WHEN skipped THEN 1 ELSE 0 END), 0) AS skips
        FROM events {where}
        """
    ).iloc[0]
    plays = int(row["plays"])
    minutes = float(row["ms"]) / MS_PER_MIN
    skip_flagged = int(row["skip_flagged"])
    skips = int(row["skips"])
    where_e = _where(_predicates(start, end, col="e.date"))
    artists = store.query(
        f"""
        SELECT COUNT(DISTINCT t.artist_name) AS c
        FROM events e JOIN tracks t ON e.track_uri = t.uri {where_e}
        """
    )["c"].iloc[0]
    span = _span(store, start, end)
    per_day = plays / span["days"] if span else 0.0
    return {
        "start": _as_date(start).isoformat() if _as_date(start) else None,
        "end": _as_date(end).isoformat() if _as_date(end) else None,
        "span": span,
        "total_plays": plays,
        "total_minutes": round(minutes, 2),
        "distinct_tracks": int(row["tracks"]),
        "distinct_artists": int(artists),
        "skip_flagged": skip_flagged,
        "skip_rate": round(skips / skip_flagged, 4) if skip_flagged else 0.0,
        "plays_per_day": round(per_day, 3),
    }


# ---------------------------------------------------------------------------
# top items
# ---------------------------------------------------------------------------
_ENTITY_SQL = {
    "tracks": ("e.track_uri", "t.track_name", "t.artist_name"),
    "artists": ("t.artist_name", "t.artist_name", None),
    "albums": ("t.album_name", "t.album_name", None),
}


def top_items(
    store: HistoryStore,
    entity: str = "tracks",
    start: DateLike = None,
    end: DateLike = None,
    limit: Optional[int] = None,
    offset: int = 0,
    by: str = "plays",
) -> dict:
    """Full ranked list of tracks / artists / albums, ordered by ``by`` (plays|minutes).

    Ranking is over the whole windowed list; ``limit``/``offset`` paginate it (``limit``
    of ``0`` or ``None`` returns everything).  ``share`` is each row's fraction of the
    window total *of the ranking metric* (plays-share when ``by='plays'``,
    minutes-share when ``by='minutes'``).  ``rank`` is the absolute 1-based position in
    the full list, so it stays correct across pages.
    """
    if entity not in _ENTITY_SQL:
        raise ValueError(f"unknown entity {entity!r}; expected one of {list(_ENTITY_SQL)}")
    if by not in ("plays", "minutes"):
        raise ValueError(f"unknown ordering {by!r}; expected 'plays' or 'minutes'")
    key, name_col, artist_col = _ENTITY_SQL[entity]

    preds = _predicates(start, end, col="e.date")
    where = _where(preds)
    order_metric = "plays" if by == "plays" else "minutes"

    agg = store.query(
        f"""
        SELECT
          {key}                                AS item_key,
          ANY_VALUE({name_col})                AS name,
          {("ANY_VALUE(" + artist_col + ")") if artist_col else "NULL"} AS artist,
          COUNT(*)                             AS plays,
          SUM(e.ms_played) / {MS_PER_MIN}      AS minutes
        FROM events e JOIN tracks t ON e.track_uri = t.uri
        {where}
        GROUP BY {key}
        ORDER BY {order_metric} DESC, name ASC
        """
    )
    total_items = int(len(agg))
    total_plays = int(agg["plays"].sum()) if total_items else 0
    total_minutes = float(agg["minutes"].sum()) if total_items else 0.0
    denom = total_plays if by == "plays" else total_minutes

    lo = max(0, int(offset))
    page = agg.iloc[lo:] if not limit else agg.iloc[lo: lo + int(limit)]

    rows: List[dict] = []
    for pos, (_, r) in enumerate(page.iterrows()):
        metric_val = float(r["plays"]) if by == "plays" else float(r["minutes"])
        row = {
            "rank": lo + pos + 1,
            "name": None if r["name"] is None else str(r["name"]),
            "plays": int(r["plays"]),
            "minutes": round(float(r["minutes"]), 2),
            "share": round(metric_val / denom, 4) if denom else 0.0,
        }
        if entity == "tracks":
            row["artist"] = None if r["artist"] is None else str(r["artist"])
        rows.append(row)

    return {
        "entity": entity,
        "by": by,
        "total": total_items,
        "total_plays": total_plays,
        "total_minutes": round(total_minutes, 2),
        "offset": lo,
        "limit": int(limit) if limit else 0,
        "rows": rows,
    }


# ---------------------------------------------------------------------------
# trends
# ---------------------------------------------------------------------------
def trends(
    store: HistoryStore,
    metric: str = "plays",
    granularity: str = "week",
    start: DateLike = None,
    end: DateLike = None,
    rolling: Optional[int] = None,
) -> dict:
    """Per-bucket time series of ``metric`` at ``granularity`` (day|week|month).

    ``metric`` is one of ``plays`` (count), ``minutes`` (listening minutes),
    ``discovery`` (share of plays that are first-ever plays of their track — first play
    computed over the *full* history, see the module docstring) or ``skip_rate`` (share
    of skip-flagged plays that were skips).  ``rolling`` (odd ``k`` recommended) adds a
    **centred** rolling mean of the value over ``k`` buckets (partial windows at the
    edges use whatever buckets exist), returned per bucket as ``rolling``.
    """
    if granularity not in _TRUNC:
        raise ValueError(f"unknown granularity {granularity!r}; expected {list(_TRUNC)}")
    if metric not in ("plays", "minutes", "discovery", "skip_rate"):
        raise ValueError(f"unknown metric {metric!r}")
    trunc = _TRUNC[granularity]
    where = _where(_predicates(start, end, col="e.date"))

    if metric == "discovery":
        df = store.query(
            f"""
            WITH firsts AS (
              SELECT track_uri, MIN(event_id) AS first_ev FROM events GROUP BY track_uri
            )
            SELECT date_trunc('{trunc}', e.date) AS bucket,
                   COUNT(*) AS plays,
                   SUM(CASE WHEN e.event_id = f.first_ev THEN 1 ELSE 0 END) AS firsts
            FROM events e JOIN firsts f ON e.track_uri = f.track_uri
            {where}
            GROUP BY bucket ORDER BY bucket
            """
        )
        values = [
            (float(r["firsts"]) / float(r["plays"])) if r["plays"] else 0.0
            for _, r in df.iterrows()
        ]
    elif metric == "skip_rate":
        df = store.query(
            f"""
            SELECT date_trunc('{trunc}', e.date) AS bucket,
                   COUNT(skipped) AS flagged,
                   SUM(CASE WHEN skipped THEN 1 ELSE 0 END) AS skips
            FROM events e {where}
            GROUP BY bucket ORDER BY bucket
            """
        )
        values = [
            (float(r["skips"]) / float(r["flagged"])) if r["flagged"] else 0.0
            for _, r in df.iterrows()
        ]
    else:
        expr = "COUNT(*)" if metric == "plays" else f"SUM(e.ms_played) / {MS_PER_MIN}"
        df = store.query(
            f"""
            SELECT date_trunc('{trunc}', e.date) AS bucket, {expr} AS value
            FROM events e {where}
            GROUP BY bucket ORDER BY bucket
            """
        )
        values = [float(v) for v in df["value"]]

    buckets = [_iso(b) for b in df["bucket"]]
    roll: List[Optional[float]] = [None] * len(values)
    k = int(rolling) if rolling else 0
    if k and k > 1 and values:
        s = pd.Series(values).rolling(k, center=True, min_periods=1).mean()
        roll = [round(float(v), 4) for v in s]

    out_rows = []
    for i, (b, v) in enumerate(zip(buckets, values)):
        row = {"bucket": b, "value": round(v, 4)}
        if k and k > 1:
            row["rolling"] = roll[i]
        out_rows.append(row)

    return {
        "metric": metric,
        "granularity": granularity,
        "rolling": k if k and k > 1 else None,
        "buckets": out_rows,
    }


# ---------------------------------------------------------------------------
# listening clock
# ---------------------------------------------------------------------------
def listening_clock(store: HistoryStore, start: DateLike = None, end: DateLike = None) -> dict:
    """Hour-of-day (0..23) × weekday (Mon..Sun) matrix of play counts.

    ``matrix[w][h]`` is the number of plays on weekday ``w`` at hour ``h``.
    """
    where = _where(_predicates(start, end))
    df = store.query(
        f"SELECT weekday, hour, COUNT(*) AS c FROM events {where} GROUP BY weekday, hour"
    )
    matrix = [[0 for _ in range(24)] for _ in range(7)]
    for _, r in df.iterrows():
        matrix[int(r["weekday"])][int(r["hour"])] = int(r["c"])
    mx = max((c for row in matrix for c in row), default=0)
    total = int(df["c"].sum()) if len(df) else 0
    return {
        "weekdays": list(_WEEKDAY_LABELS),
        "hours": list(range(24)),
        "matrix": matrix,
        "max": mx,
        "total": total,
    }


# ---------------------------------------------------------------------------
# axes over time
# ---------------------------------------------------------------------------
def axes_over_time(
    store: HistoryStore,
    granularity: str = "week",
    start: DateLike = None,
    end: DateLike = None,
) -> dict:
    """Per-bucket mean of each scalar taste axis, plus honest feature coverage.

    Means are taken **only over plays whose track has features**; ``coverage`` is the
    share of plays in the bucket that had features, so the dashboard can say "axes
    computed on X% of plays".  For a real export with no features attached every bucket
    has ``coverage == 0`` and empty ``means``.
    """
    if granularity not in _TRUNC:
        raise ValueError(f"unknown granularity {granularity!r}; expected {list(_TRUNC)}")
    trunc = _TRUNC[granularity]
    where = _where(_predicates(start, end, col="e.date"))
    means_sql = ", ".join(f"AVG(f.{c}) AS {c}" for c in SCALAR_COLUMNS)
    df = store.query(
        f"""
        SELECT date_trunc('{trunc}', e.date) AS bucket,
               COUNT(*) AS plays,
               COUNT(f.uri) AS with_features,
               {means_sql}
        FROM events e LEFT JOIN track_features f ON e.track_uri = f.uri
        {where}
        GROUP BY bucket ORDER BY bucket
        """
    )
    buckets = []
    for _, r in df.iterrows():
        plays = int(r["plays"])
        wf = int(r["with_features"])
        means = {}
        if wf:
            for axis, col in zip(SCALAR_AXES, SCALAR_COLUMNS):
                val = r[col]
                means[axis] = None if pd.isna(val) else round(float(val), 4)
        buckets.append({
            "bucket": _iso(r["bucket"]),
            "plays": plays,
            "coverage": round(wf / plays, 4) if plays else 0.0,
            "means": means,
        })
    return {"granularity": granularity, "axes": list(SCALAR_AXES), "buckets": buckets}


# ---------------------------------------------------------------------------
# genre / flavour mix
# ---------------------------------------------------------------------------
def top_flavors_or_genres(
    store: HistoryStore, start: DateLike = None, end: DateLike = None
) -> dict:
    """Mean of each ``genre_*`` axis over plays-with-features in the window (a genre mix).

    Synthetic tracks carry genre axes; a raw real export does not until enrichment, so
    ``coverage`` (share of window plays that had features) is returned alongside and the
    ``mix`` is empty when nothing has features.  Mix rows are sorted by mean descending.
    """
    where = _where(_predicates(start, end, col="e.date"))
    means_sql = ", ".join(f"AVG(f.{c}) AS {c}" for c in GENRE_COLUMNS)
    df = store.query(
        f"""
        SELECT COUNT(*) AS plays,
               COUNT(f.uri) AS with_features,
               {means_sql}
        FROM events e LEFT JOIN track_features f ON e.track_uri = f.uri
        {where}
        """
    ).iloc[0]
    plays = int(df["plays"])
    wf = int(df["with_features"])
    mix: List[dict] = []
    if wf:
        for g, col in zip(GENRES, GENRE_COLUMNS):
            val = df[col]
            if not pd.isna(val):
                mix.append({"genre": g, "mean": round(float(val), 4)})
        mix.sort(key=lambda m: (-m["mean"], m["genre"]))
    return {
        "genres": list(GENRES),
        "mix": mix,
        "coverage": round(wf / plays, 4) if plays else 0.0,
        "plays_with_features": wf,
        "total_plays": plays,
    }


# ---------------------------------------------------------------------------
# CSV serialisation (for the download links on top_items / trends)
# ---------------------------------------------------------------------------
def top_items_to_csv(payload: dict) -> str:
    """Serialise a :func:`top_items` payload to CSV text (one row per ranked item)."""
    entity = payload.get("entity", "tracks")
    fields = ["rank", "name"]
    if entity == "tracks":
        fields.append("artist")
    fields += ["plays", "minutes", "share"]
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=fields, extrasaction="ignore")
    w.writeheader()
    for row in payload["rows"]:
        w.writerow({k: ("" if row.get(k) is None else row.get(k)) for k in fields})
    return buf.getvalue()


def trends_to_csv(payload: dict) -> str:
    """Serialise a :func:`trends` payload to CSV text (one row per bucket)."""
    has_roll = payload.get("rolling") is not None
    fields = ["bucket", "value"] + (["rolling"] if has_roll else [])
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=fields, extrasaction="ignore")
    w.writeheader()
    for row in payload["buckets"]:
        w.writerow({k: ("" if row.get(k) is None else row.get(k)) for k in fields})
    return buf.getvalue()
