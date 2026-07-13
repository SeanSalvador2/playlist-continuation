"""History Atlas — the personal listening-analytics engine behind the Library view.

Mirrors :mod:`app.backend.engine`'s pattern: build a data source + store once at
startup, cache it as a process-wide singleton, and expose thin methods the server
translates 1:1 into JSON.  Where ``engine.py`` fits recommendation models over a
synthetic MPD, this fits a personal *listening history* into a
:class:`~playlistcont.history.store.HistoryStore` and answers the canned analytics in
:mod:`playlistcont.analytics.queries`.

Data source (honest by construction):

* By default a **synthetic demo listener** — ``make_synthetic_history(seed=7,
  n_days=730)`` — whose planted ground truth (taste-change dates, regimes) is surfaced
  in the summary payload so the UI can label the demo data as a demo.
* If ``PLAYLISTCONT_HISTORY_EXPORT`` points at a real Spotify GDPR export (a folder or
  ``.zip``) it is loaded instead: the *extended* streaming history if present, else the
  *basic* one.  Real data has **no** ground truth and **no** axis features — features
  are attached only when ``PLAYLISTCONT_FEATURES`` names an audio-features parquet
  path, via :func:`attach_real_features`.
* If additionally ``PLAYLISTCONT_ENRICH=1`` **and** the history is a real export, the
  Phase 1.5 enrichment runs at startup: the two enrichment tables are downloaded (once,
  cached under ``.data/enrichment/``), genre axes are filled via
  :func:`playlistcont.enrichment.enrich_history`, the widened per-track record is
  attached from ``PLAYLISTCONT_FEATURES`` when set, and the resulting
  :class:`~playlistcont.enrichment.EnrichmentReport` is included in the summary payload
  (``enrichment`` key) so the Library provenance banner can state real genre coverage.
  Failures are recorded in the payload (``{"error": ...}``) rather than killing startup.
  The synthetic path is completely unchanged (``enrichment`` is ``None``).
"""
from __future__ import annotations

import os
import threading
from datetime import date, timedelta
from typing import Optional

from playlistcont.analytics import queries, stats
from playlistcont.history.features import attach_extended_features, attach_real_features
from playlistcont.history.schema import ListeningHistory
from playlistcont.history.spotify_export import load_basic_history, load_extended_history
from playlistcont.history.store import HistoryStore
from playlistcont.history.synthetic import make_synthetic_history

SEED = 7
N_DAYS = 730

_PROVENANCE_LABEL = {
    "synthetic": "synthetic demo listener",
    "spotify_export": "your Spotify export",
}


def _load_history() -> ListeningHistory:
    """Load the configured history: a real export if pointed at one, else synthetic."""
    export = os.environ.get("PLAYLISTCONT_HISTORY_EXPORT")
    if not export:
        return make_synthetic_history(seed=SEED, n_days=N_DAYS)

    # Prefer the rich extended export; fall back to the thin basic one if it has no
    # extended files.  Either raises FileNotFoundError only for a truly bad path.
    history = load_extended_history(export)
    if history.n_events == 0:
        history = load_basic_history(export)

    feats = os.environ.get("PLAYLISTCONT_FEATURES")
    if feats:
        attach_real_features(history, source=feats)
    return history


def _maybe_enrich(history: ListeningHistory):
    """Run Phase 1.5 enrichment when configured; never on synthetic data.

    Returns ``(payload_or_None, artist_tag_rows)``.  ``payload`` is the
    :class:`EnrichmentReport` as a dict (plus ``extended_matched``), or
    ``{"error": ...}`` if the download/join failed — startup survives either way.
    """
    if os.environ.get("PLAYLISTCONT_ENRICH") != "1":
        return None, []
    if history.provenance != "spotify_export":
        return None, []  # synthetic path: enrichment never touches it
    try:
        from playlistcont.enrichment import enrich_history, sources

        artist_table = sources.load_artist_tag_table()
        track_table = sources.load_track_genre_table()
        report = enrich_history(
            history, artist_table=artist_table, track_table=track_table)

        payload = report.to_payload()
        feats = os.environ.get("PLAYLISTCONT_FEATURES")
        if feats:
            payload["extended_matched"] = attach_extended_features(
                history, source=feats)
        return payload, report.artist_tag_rows
    except Exception as exc:  # honest failure surface, not a dead dashboard
        return {"error": str(exc)}, []


class HistoryAtlas:
    """Everything the Library view needs, built once and cached."""

    def __init__(self) -> None:
        self.history = _load_history()
        self.enrichment, _tag_rows = _maybe_enrich(self.history)
        self.store = HistoryStore.from_history(self.history)
        if _tag_rows:
            self.store.attach_artist_tags(_tag_rows)
        self.provenance = self.history.provenance
        # A single DuckDB connection is not safe for concurrent use, and FastAPI runs
        # sync endpoints in a threadpool — the Library view fires several panel requests
        # at once.  Serialise every query through this lock so parallel panels can't race
        # on the shared connection.  (Kept here rather than in the Phase 0 store so that
        # package stays a pure single-threaded query surface.)
        self._lock = threading.Lock()
        # full span (window-independent) drives the date-picker bounds in the UI.
        self.full_span = queries.summary(self.store)["span"]

    # ------------------------------------------------------------------ #
    def _ground_truth(self) -> Optional[dict]:
        gt = self.history.ground_truth
        if gt is None:
            return None
        return {
            "seed": gt.seed,
            "changes": [
                {"date": c.date.isoformat(), "kind": c.kind, "description": c.description}
                for c in gt.change_points
            ],
            "regimes": [
                {"start": r.start.isoformat(), "end": r.end.isoformat(), "label": r.label}
                for r in gt.regimes
            ],
        }

    # ------------------------------------------------------------------ #
    #  public API used by the server (thin passthroughs to queries.*)
    # ------------------------------------------------------------------ #
    def summary(self, start=None, end=None) -> dict:
        with self._lock:
            base = queries.summary(self.store, start, end)
        base["provenance"] = self.provenance
        base["provenance_label"] = _PROVENANCE_LABEL.get(self.provenance, self.provenance)
        base["is_synthetic"] = self.provenance == "synthetic"
        base["full_span"] = self.full_span
        base["ground_truth"] = self._ground_truth()
        base["enrichment"] = self.enrichment
        return base

    def top_items(self, entity="tracks", start=None, end=None,
                  limit=None, offset=0, by="plays") -> dict:
        with self._lock:
            return queries.top_items(self.store, entity=entity, start=start, end=end,
                                     limit=limit, offset=offset, by=by)

    def trends(self, metric="plays", granularity="week", start=None, end=None,
               rolling=None) -> dict:
        with self._lock:
            return queries.trends(self.store, metric=metric, granularity=granularity,
                                  start=start, end=end, rolling=rolling)

    def listening_clock(self, start=None, end=None) -> dict:
        with self._lock:
            return queries.listening_clock(self.store, start=start, end=end)

    def axes_over_time(self, granularity="week", start=None, end=None) -> dict:
        with self._lock:
            return queries.axes_over_time(self.store, granularity=granularity,
                                          start=start, end=end)

    def genres(self, start=None, end=None) -> dict:
        with self._lock:
            return queries.top_flavors_or_genres(self.store, start=start, end=end)

    # ------------------------------------------------------------------ #
    #  Phase 2: classical statistics (FDR-corrected, effect-floored)
    # ------------------------------------------------------------------ #
    def _preceding_window(self, b_start: date, b_end: date):
        """The same-length window ending the day before ``b_start`` (clamped to span)."""
        length = (b_end - b_start).days + 1
        a_end = b_start - timedelta(days=1)
        a_start = a_end - timedelta(days=length - 1)
        if self.full_span:
            first = date.fromisoformat(self.full_span["first"])
            if a_start < first:
                a_start = first
        return a_start, a_end

    def shifts(self, mode="auto", start=None, end=None,
               a_start=None, a_end=None, b_start=None, b_end=None) -> dict:
        """Significant shifts between two windows (auto = selected vs preceding same-length)."""
        if mode == "auto":
            b_s = stats._as_date(start)
            b_e = stats._as_date(end)
            if self.full_span:
                b_s = b_s or date.fromisoformat(self.full_span["first"])
                b_e = b_e or date.fromisoformat(self.full_span["last"])
            a_s, a_e = self._preceding_window(b_s, b_e)
        else:
            a_s, a_e = stats._as_date(a_start), stats._as_date(a_end)
            b_s, b_e = stats._as_date(b_start), stats._as_date(b_end)
        with self._lock:
            payload = stats.significant_shifts(self.store, a_s, a_e, b_s, b_e)
        payload["mode"] = mode
        return payload

    def habits(self, group_by="weekday", start=None, end=None) -> dict:
        with self._lock:
            return stats.habit_anova(self.store, start=start, end=end, group_by=group_by)


# ---- process-wide singleton (built lazily, cached) -------------------------- #
_HISTORY_ATLAS: Optional[HistoryAtlas] = None


def get_history_atlas() -> HistoryAtlas:
    global _HISTORY_ATLAS
    if _HISTORY_ATLAS is None:
        _HISTORY_ATLAS = HistoryAtlas()
    return _HISTORY_ATLAS
