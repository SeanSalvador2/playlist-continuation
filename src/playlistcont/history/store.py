"""A DuckDB analytics store over a :class:`ListeningHistory`.

The dashboard (Phase 1) and the change-point work (Phase 3) both want to slice a
listening stream by day/hour/weekday, join tracks to their interpretable axes, and run
ad-hoc aggregations.  Holding the stream in DuckDB gives all of that in SQL for free,
in-memory by default and file-backed when you want to persist or reopen read-only (a
later text-to-SQL phase queries a file store with ``read_only=True``).

Three tables are built:

* ``events`` — one row per play: ``event_id, ts, date, hour, weekday, track_uri,
  ms_played, skipped, platform``.  ``weekday`` is 0=Monday .. 6=Sunday.
* ``tracks`` — ``uri, track_name, artist_name, album_name``.
* ``track_features`` — ``uri`` plus one column per taste-engine axis, with a row **only**
  for tracks whose axis vector is known (all of them for synthetic histories; for real
  histories only after an audio-features join).  Axis names contain ``:`` (e.g.
  ``genre:country``); columns sanitise that to ``_`` (``genre_country``) — see
  :func:`feature_column` for the mapping.
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

    # ------------------------------------------------------------------
    def query(self, sql: str) -> pd.DataFrame:
        """Run SQL and return a pandas DataFrame."""
        return self.conn.execute(sql).df()

    def table_counts(self) -> Dict[str, int]:
        """Row counts for the three tables (handy for tests / sanity checks)."""
        return {
            t: int(self.conn.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0])
            for t in ("events", "tracks", "track_features")
        }

    def close(self) -> None:
        self.conn.close()
