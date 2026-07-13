"""A DuckDB analytics store over a :class:`ListeningHistory`.

The dashboard (Phase 1) and the change-point work (Phase 3) both want to slice a
listening stream by day/hour/weekday, join tracks to their interpretable axes, and run
ad-hoc aggregations.  Holding the stream in DuckDB gives all of that in SQL for free,
in-memory by default and file-backed when you want to persist or reopen read-only (a
later text-to-SQL phase queries a file store with ``read_only=True``).

Three tables are always built:

* ``events`` — one row per play: ``event_id, ts, date, hour, weekday, track_uri,
  ms_played, skipped, platform``.  ``weekday`` is 0=Monday .. 6=Sunday.
* ``tracks`` — ``uri, track_name, artist_name, album_name``.
* ``track_features`` — ``uri`` plus one column per taste-engine axis, with a row **only**
  for tracks whose axis vector is known (all of them for synthetic histories; for real
  histories only after an audio-features join).  Axis names contain ``:`` (e.g.
  ``genre:country``); columns sanitise that to ``_`` (``genre_country``) — see
  :func:`feature_column` for the mapping.

Two further tables are **optional and additive** (Phase 1.5 enrichment) — created only
when the corresponding data was actually supplied, so pre-enrichment stores and all
their consumers are unchanged:

* ``extended_features`` — ``uri`` plus the widened per-track record
  (:data:`EXTENDED_STORE_COLUMNS`; every value column nullable).  Built during
  :meth:`HistoryStore.from_history` only when at least one track carries an
  ``extended`` dict (see :func:`playlistcont.history.features.attach_extended_features`).
* ``artist_tags`` — raw enrichment folksonomy: ``artist_name, tag, mapped_bucket``
  (``mapped_bucket`` is NULL for tags outside the curated 10-bucket mapping).  Created
  only via an explicit :meth:`HistoryStore.attach_artist_tags` call with data.
"""
from __future__ import annotations

from typing import Dict, List, Optional

import pandas as pd

from ..data.schema import AXES
from .schema import ListeningHistory


def feature_column(axis: str) -> str:
    """DuckDB-safe column name for an axis (``genre:country`` -> ``genre_country``)."""
    return axis.replace(":", "_")


FEATURE_COLUMNS: List[str] = [feature_column(a) for a in AXES]

# widened per-track record stored by attach_extended_features (all nullable);
# pandas nullable dtypes keep NULLs intact through the DuckDB materialisation.
EXTENDED_STORE_COLUMNS: List[str] = [
    "popularity", "danceability", "speechiness", "loudness",
    "liveness", "key", "mode", "duration_ms",
]
_EXTENDED_DTYPES: Dict[str, str] = {
    "popularity": "Int64", "danceability": "Float64", "speechiness": "Float64",
    "loudness": "Float64", "liveness": "Float64", "key": "Int64",
    "mode": "Int64", "duration_ms": "Int64",
}


class HistoryStore:
    """A thin DuckDB wrapper around a listening history.

    Construct via :meth:`from_history` (builds the tables) or :meth:`open` (reopens an
    existing file-backed store, read-only by default).
    """

    def __init__(self, conn):
        self.conn = conn

    # ------------------------------------------------------------------
    @classmethod
    def from_history(
        cls,
        history: ListeningHistory,
        path: Optional[str] = None,
    ) -> "HistoryStore":
        """Build the store from a history (in-memory unless ``path`` is given)."""
        import duckdb

        conn = duckdb.connect(path if path is not None else ":memory:")
        store = cls(conn)
        store._build(history)
        return store

    @classmethod
    def open(cls, path: str, read_only: bool = True) -> "HistoryStore":
        """Reopen an existing file-backed store (read-only by default)."""
        import duckdb

        return cls(duckdb.connect(path, read_only=read_only))

    # ------------------------------------------------------------------
    def _build(self, history: ListeningHistory) -> None:
        events_df = self._events_frame(history)
        tracks_df = self._tracks_frame(history)
        feats_df = self._features_frame(history)

        # register the pandas frames and materialise into real tables
        self.conn.register("_events_df", events_df)
        self.conn.register("_tracks_df", tracks_df)
        self.conn.register("_feats_df", feats_df)
        self.conn.execute("CREATE TABLE events AS SELECT * FROM _events_df")
        self.conn.execute("CREATE TABLE tracks AS SELECT * FROM _tracks_df")
        self.conn.execute("CREATE TABLE track_features AS SELECT * FROM _feats_df")
        self.conn.unregister("_events_df")
        self.conn.unregister("_tracks_df")
        self.conn.unregister("_feats_df")

        # optional, additive: only when at least one track has extended data
        ext_df = self._extended_frame(history)
        if ext_df is not None:
            self.conn.register("_ext_df", ext_df)
            self.conn.execute("CREATE TABLE extended_features AS SELECT * FROM _ext_df")
            self.conn.unregister("_ext_df")

    @staticmethod
    def _events_frame(history: ListeningHistory) -> pd.DataFrame:
        rows = []
        for i, e in enumerate(history.events):
            rows.append({
                "event_id": i,
                "ts": e.ts,
                "date": e.ts.date(),
                "hour": e.ts.hour,
                "weekday": e.ts.weekday(),  # 0=Mon .. 6=Sun
                "track_uri": e.track_uri,
                "ms_played": int(e.ms_played),
                "skipped": e.skipped,
                "platform": e.platform,
            })
        df = pd.DataFrame(rows, columns=[
            "event_id", "ts", "date", "hour", "weekday",
            "track_uri", "ms_played", "skipped", "platform",
        ])
        if not rows:
            return df
        df["ts"] = pd.to_datetime(df["ts"], utc=True)
        df["skipped"] = df["skipped"].astype("boolean")  # nullable bool
        return df

    @staticmethod
    def _tracks_frame(history: ListeningHistory) -> pd.DataFrame:
        rows = [{
            "uri": ht.track_uri,
            "track_name": ht.track_name,
            "artist_name": ht.artist_name,
            "album_name": ht.album_name,
        } for ht in history.tracks.values()]
        return pd.DataFrame(rows, columns=["uri", "track_name", "artist_name", "album_name"])

    @staticmethod
    def _features_frame(history: ListeningHistory) -> pd.DataFrame:
        cols = ["uri"] + FEATURE_COLUMNS
        rows = []
        for ht in history.tracks.values():
            if ht.features is None:
                continue
            row: Dict[str, object] = {"uri": ht.track_uri}
            for j, col in enumerate(FEATURE_COLUMNS):
                row[col] = float(ht.features[j])
            rows.append(row)
        return pd.DataFrame(rows, columns=cols)

    @staticmethod
    def _extended_frame(history: ListeningHistory) -> Optional[pd.DataFrame]:
        """Widened per-track frame, or ``None`` when no track has extended data.

        Every value column is nullable (pandas ``Int64``/``Float64``): a track may
        carry only a subset of the extended record, and the source table itself has
        null feature rows.
        """
        rows = []
        for ht in history.tracks.values():
            if ht.extended is None:
                continue
            row: Dict[str, object] = {"uri": ht.track_uri}
            for col in EXTENDED_STORE_COLUMNS:
                row[col] = ht.extended.get(col)
            rows.append(row)
        if not rows:
            return None
        df = pd.DataFrame(rows, columns=["uri"] + EXTENDED_STORE_COLUMNS)
        for col, dtype in _EXTENDED_DTYPES.items():
            df[col] = pd.array(
                [None if pd.isna(v) else v for v in df[col]], dtype=dtype)
        return df

    # ------------------------------------------------------------------
    def attach_artist_tags(self, rows) -> int:
        """Create the optional ``artist_tags`` table from enrichment tag rows.

        ``rows`` is a list of ``{"artist_name", "tag", "mapped_bucket"}`` dicts (as
        produced by :func:`playlistcont.enrichment.apply.enrich_history`) or an
        equivalent DataFrame.  A no-op returning 0 when there is no data — the
        table exists only when tags were actually supplied.
        """
        df = rows if isinstance(rows, pd.DataFrame) else pd.DataFrame(
            list(rows), columns=["artist_name", "tag", "mapped_bucket"])
        if not len(df):
            return 0
        df = df[["artist_name", "tag", "mapped_bucket"]]
        self.conn.register("_artist_tags_df", df)
        self.conn.execute(
            "CREATE OR REPLACE TABLE artist_tags AS SELECT * FROM _artist_tags_df")
        self.conn.unregister("_artist_tags_df")
        return int(len(df))

    # ------------------------------------------------------------------
    def query(self, sql: str) -> pd.DataFrame:
        """Run SQL and return a pandas DataFrame."""
        return self.conn.execute(sql).df()

    def has_table(self, name: str) -> bool:
        """True when an (optional) table exists in the store."""
        n = self.conn.execute(
            "SELECT COUNT(*) FROM information_schema.tables WHERE table_name = ?",
            [name],
        ).fetchone()[0]
        return bool(n)

    def table_counts(self) -> Dict[str, int]:
        """Row counts for the core tables plus any optional ones that exist."""
        tables = ["events", "tracks", "track_features"]
        tables += [t for t in ("extended_features", "artist_tags") if self.has_table(t)]
        return {
            t: int(self.conn.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0])
            for t in tables
        }

    def close(self) -> None:
        self.conn.close()
