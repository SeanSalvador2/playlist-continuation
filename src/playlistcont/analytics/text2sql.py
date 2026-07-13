"""Ask your library — free-form querying of a personal listening history.

This is the Phase 5 "ask" surface over the three-table
:class:`~playlistcont.history.store.HistoryStore` (``events`` / ``tracks`` /
``track_features`` plus the optional ``extended_features`` / ``artist_tags``).  It has
three layers, in increasing order of trust required:

1. **A schema card** (:data:`SCHEMA_CARD`) — one frozen, human/LLM-readable description
   of the tables, their columns and the load-bearing conventions (weekday ``0=Mon``,
   ``ms_played`` is milliseconds, "plays" count events while "tracks" count distinct
   ``track_uri``).  Everything else in this module — templates, the LLM prompt, the
   gold eval — refers back to it, so there is a single source of truth.

2. **A keyless template library** (:data:`TEMPLATES`) — >= 15 parameterised questions
   covering the long tail the canned Library panels don't (most-skipped artists,
   abandoned artists, one-hit wonders, night-owl tracks, comeback artists, ...).  Each
   template has *typed slots* (``int`` / ``date`` / ``enum``) that are validated and
   substituted **by value, never by raw string interpolation of user text** — integers
   go through :func:`int`, dates through :func:`datetime.date.fromisoformat`, enums
   through an allowlist.  This path needs no API keys and no network.

3. **An optional LLM NL->SQL path** (:func:`nl_to_sql`) — a pure prompt builder +
   few-shot set + response extractor around an *injected* ``llm_fn`` callable.  There is
   no API client, no key handling and no network here: the app wires ``llm_fn`` only
   when ``PLAYLISTCONT_LLM_KEY`` is present (future work), and the UI *always* shows the
   generated SQL to the user before it runs.

Both the template path and the LLM path funnel through the **same guardrail**,
:func:`safe_execute`, which is the security boundary and is tested hard:

* It uses DuckDB's *own* parser (``conn.extract_statements``) to split the input into
  statements — so it rejects multi-statement payloads ("``SELECT 1; DROP TABLE
  events``") and comment-smuggled second statements structurally, not by regex.
* It requires exactly **one** statement whose parsed :class:`duckdb.StatementType` is
  ``SELECT`` (``WITH ... SELECT`` also parses as ``SELECT``).
* It then screens the text for a denylist of dangerous leading keywords
  (:data:`_DENY_KEYWORDS`).  This is *belt-and-suspenders*: DuckDB classifies ``PRAGMA``
  as a ``SELECT`` statement, so the type check alone is not sufficient — the keyword
  screen is what stops ``PRAGMA`` / ``ATTACH`` / ``COPY`` / ``INSTALL`` / ``LOAD`` /
  ``CALL`` / ``SET`` and file-reading table functions.  We are honest that a keyword
  screen cannot enumerate every hostile construct; the *real* write protection is the
  read-only transaction below, which does not depend on recognising keywords.
* It ``EXPLAIN``s the (wrapped) query as a dry run so a query that references a missing
  column/table fails at bind time before execution.
* It **enforces a row cap** by wrapping the query as ``SELECT * FROM (<sql>) LIMIT n`` —
  so a ``LIMIT`` is always present in the executed SQL and ``max_rows`` is guaranteed
  regardless of any inner ``LIMIT``.
* It executes inside a ``BEGIN TRANSACTION READ ONLY`` ... ``ROLLBACK`` block.  DuckDB
  rejects *any* write inside a read-only transaction **even on a writable, in-memory
  connection** — this is the runtime guarantee that does not rely on parsing.  When the
  store is file-backed the caller should additionally pass a connection reopened with
  ``read_only=True`` (see :func:`readonly_store`); when it is in-memory (the app's case)
  the read-only transaction plus statement validation is the documented fallback.
* It applies a best-effort **timeout** via ``conn.interrupt()`` fired from a timer.

What this does and does not guarantee (stated honestly): the parser + read-only
transaction reliably prevent multi-statement injection and *writes* (INSERT/UPDATE/
DELETE/DDL/COPY/ATTACH).  The keyword screen additionally blocks the PRAGMA/metadata
surface.  It does **not** claim to prevent a determined reader from reading data they
are already allowed to read, nor is the interrupt timeout a hard real-time bound.
"""
from __future__ import annotations

import hashlib
import re
import threading
from dataclasses import dataclass, field
from datetime import date
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

# ---------------------------------------------------------------------------
# 1. Schema card — the single source of truth
# ---------------------------------------------------------------------------
SCHEMA_CARD: str = """\
You are querying a personal music-listening history stored in DuckDB.  There are three
core tables (always present) and two optional ones.  Use only these tables and columns.

TABLE events  -- one row per play (a "play" == one row here)
  event_id   BIGINT   monotonic id, 0-based, ascending by time (earliest play = 0)
  ts         TIMESTAMP WITH TIME ZONE   exact play time (UTC)
  date       DATE     calendar day of the play
  hour       INTEGER  hour of day 0..23
  weekday    INTEGER  day of week, 0=Monday .. 6=Sunday  (5,6 == weekend)
  track_uri  VARCHAR  foreign key -> tracks.uri
  ms_played  BIGINT   milliseconds this play lasted (divide by 60000 for minutes)
  skipped    BOOLEAN  true if the play was skipped (may be NULL if source lacked it)
  platform   VARCHAR  e.g. 'android','ios','web_player','desktop'

TABLE tracks  -- one row per distinct track
  uri          VARCHAR  primary key (join events.track_uri = tracks.uri)
  track_name   VARCHAR
  artist_name  VARCHAR
  album_name   VARCHAR

TABLE track_features  -- interpretable taste axes; one row per track WITH known features
  uri            VARCHAR  join to tracks.uri
  tempo, energy, valence, acousticness, lyrical_depth   DOUBLE  scalar axes in [0,1]
  genre_country, genre_rap, genre_indie, genre_pop, genre_rock,
  genre_electronic, genre_folk, genre_metal, genre_rnb, genre_jazz   DOUBLE  soft one-hot

OPTIONAL TABLE extended_features (present only after enrichment): uri, popularity,
  danceability, speechiness, loudness, liveness, key, mode, duration_ms.
OPTIONAL TABLE artist_tags (present only after enrichment): artist_name, tag,
  mapped_bucket.

Conventions that matter:
  * "plays" = COUNT(*) over events; "tracks"/"songs" = COUNT(DISTINCT track_uri).
  * "minutes" = SUM(ms_played)/60000.0.
  * weekday is 0=Monday..6=Sunday; weekend = weekday IN (5,6).
  * "first-ever play" of a track = its row with the minimum event_id.
  * a track's artist/name lives in tracks; join events -> tracks by track_uri = uri.
Only single read-only SELECT statements are allowed (no INSERT/UPDATE/DDL/PRAGMA).
"""


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------
class GuardrailError(ValueError):
    """Raised by :func:`safe_execute` when a statement is rejected by the guardrails."""


class SlotError(ValueError):
    """Raised when a template slot value fails type/allowlist validation."""


# ---------------------------------------------------------------------------
# 2b. Guardrails
# ---------------------------------------------------------------------------
# Leading-keyword denylist.  This is a defense-in-depth screen (the read-only
# transaction is the real write barrier); its specific job is the surface DuckDB's
# statement classifier does NOT flag as non-SELECT — most importantly PRAGMA, which
# extract_statements reports as StatementType.SELECT.  We match a keyword only when it
# appears as the first token of a statement or immediately after an opening keyword, and
# also anywhere as a whole word for the always-hostile ones, on a comment-stripped copy.
_DENY_KEYWORDS: Tuple[str, ...] = (
    "insert", "update", "delete", "merge", "upsert",
    "create", "drop", "alter", "truncate", "replace",
    "attach", "detach", "copy", "export", "import",
    "install", "load", "pragma", "call", "set", "reset",
    "vacuum", "analyze", "checkpoint", "use",
    "prepare", "execute", "deallocate",
    # file / system-reading table functions and macros
    "read_csv", "read_parquet", "read_json", "read_text", "read_blob",
    "glob", "sniff_csv", "parquet_scan", "csv_scan",
    "getenv", "system", "shell",
)
_DENY_RE = re.compile(
    r"(?<![A-Za-z0-9_])(?:" + "|".join(re.escape(k) for k in _DENY_KEYWORDS) + r")(?![A-Za-z0-9_])",
    re.IGNORECASE,
)
# strip -- line comments and /* */ block comments before screening
_LINE_COMMENT_RE = re.compile(r"--[^\n]*")
_BLOCK_COMMENT_RE = re.compile(r"/\*.*?\*/", re.DOTALL)


def _strip_comments(sql: str) -> str:
    return _BLOCK_COMMENT_RE.sub(" ", _LINE_COMMENT_RE.sub(" ", sql))


def _screen_keywords(sql: str) -> None:
    """Reject any denylisted keyword found in the (comment-stripped) SQL text."""
    stripped = _strip_comments(sql)
    m = _DENY_RE.search(stripped)
    if m:
        raise GuardrailError(
            f"disallowed keyword {m.group(0)!r}: only read-only SELECT queries are permitted"
        )


def _statement_is_single_select(conn, sql: str) -> None:
    """Use DuckDB's own parser to require exactly one SELECT statement."""
    import duckdb

    try:
        statements = conn.extract_statements(sql)
    except Exception as exc:  # a genuine parser error
        raise GuardrailError(f"could not parse SQL: {exc}") from exc
    if len(statements) == 0:
        raise GuardrailError("empty query")
    if len(statements) > 1:
        raise GuardrailError(
            f"only a single statement is allowed (found {len(statements)}); "
            "multi-statement / stacked queries are rejected"
        )
    st_type = statements[0].type
    if st_type != duckdb.StatementType.SELECT:
        raise GuardrailError(
            f"only SELECT queries are allowed (got {str(st_type).split('.')[-1]})"
        )


def _wrap_with_limit(sql: str, max_rows: int) -> str:
    """Cap rows by wrapping as a subquery, guaranteeing a LIMIT in the executed SQL."""
    inner = sql.strip().rstrip(";").strip()
    return f"SELECT * FROM (\n{inner}\n) AS _guarded LIMIT {int(max_rows)}"


def readonly_store(store):
    """Return a store suitable for free-form SQL: a read-only reopen when file-backed.

    For a file-backed :class:`HistoryStore` opened read-write this reopens the same
    database with ``read_only=True`` (writes are refused at the connection level).  If
    the store is already read-only, or is in-memory (which cannot be reopened), the same
    store is returned and :func:`safe_execute` relies on its read-only transaction +
    statement validation instead — documented in the module docstring.

    (Note: DuckDB refuses a read-only reopen while a read-write connection still holds
    the file.  If that happens the writable store is returned unchanged and
    :func:`safe_execute`'s read-only transaction is the write barrier — which blocks
    writes even on a writable connection.  The app's store is in-memory, so this is a
    no-op there.)
    """
    from ..history.store import HistoryStore

    path, readonly = _database_info(store.conn)
    if path is None or readonly:
        return store  # in-memory or already read-only: safe_execute guards it
    try:
        return HistoryStore.open(path, read_only=True)
    except Exception:
        # a read-write handle still holds the file: fall back to the RO-transaction guard
        return store


def _database_info(conn) -> Tuple[Optional[str], bool]:
    """(on-disk path, is-read-only) of the connection's default database.

    Path is ``None`` for an in-memory database.
    """
    try:
        rows = conn.execute(
            "SELECT path, readonly FROM duckdb_databases() "
            "WHERE database_name NOT IN ('system','temp') AND path IS NOT NULL"
        ).fetchall()
    except Exception:
        return None, False
    if not rows:
        return None, False
    return rows[0][0], bool(rows[0][1])


def safe_execute(
    store,
    sql: str,
    max_rows: int = 500,
    timeout_s: float = 5.0,
) -> dict:
    """Validate and execute a read-only ``SELECT`` against ``store``, safely.

    Returns ``{"sql": <normalized/wrapped SQL executed>, "columns": [...],
    "rows": [ {col: val, ...}, ... ], "row_count": n, "truncated": bool}``.

    Raises :class:`GuardrailError` if the statement is not a single ``SELECT`` or trips
    the keyword screen, and re-raises the underlying DuckDB error (as ``GuardrailError``)
    if the query fails to bind or is interrupted by the timeout.  See the module
    docstring for exactly what the guardrail does and does not guarantee.
    """
    if not isinstance(sql, str) or not sql.strip():
        raise GuardrailError("empty query")
    conn = store.conn

    # (1) structural: DuckDB's own parser — single statement, type SELECT.
    _statement_is_single_select(conn, sql)
    # (2) belt-and-suspenders keyword screen (covers PRAGMA & friends the type misses).
    _screen_keywords(sql)
    # (3) enforce the row cap by wrapping; the executed SQL always carries a LIMIT.
    wrapped = _wrap_with_limit(sql, max_rows)

    timer: Optional[threading.Timer] = None
    started_txn = False
    try:
        # (4) dry-run EXPLAIN so binding errors surface before we execute.
        conn.execute("EXPLAIN " + wrapped)
        # (5) read-only transaction: refuses any write even on a writable connection.
        conn.execute("BEGIN TRANSACTION READ ONLY")
        started_txn = True
        # (6) best-effort timeout via interrupt fired from a timer thread.
        if timeout_s and timeout_s > 0:
            timer = threading.Timer(float(timeout_s), conn.interrupt)
            timer.daemon = True
            timer.start()
        df = conn.execute(wrapped).df()
    except GuardrailError:
        raise
    except Exception as exc:
        raise GuardrailError(f"query failed: {exc}") from exc
    finally:
        if timer is not None:
            timer.cancel()
        if started_txn:
            try:
                conn.execute("ROLLBACK")
            except Exception:
                pass

    columns = list(df.columns)
    records = df.to_dict("records")
    rows = [{c: _jsonable(r[c]) for c in columns} for r in records]
    return {
        "sql": wrapped,
        "columns": columns,
        "rows": rows,
        "row_count": len(rows),
        "truncated": len(rows) >= max_rows,
    }


def _jsonable(v: Any) -> Any:
    """Render a DuckDB/pandas scalar as a JSON-friendly Python value."""
    import math

    import pandas as pd

    if v is None:
        return None
    try:
        if pd.isna(v):
            return None
    except (TypeError, ValueError):
        pass
    if isinstance(v, (bool, int, str)):
        return v
    if isinstance(v, float):
        return None if math.isnan(v) else v
    # pandas Timestamp is a subclass of datetime/date — handle it before date so a
    # DATE column renders as a bare ISO day, not a full datetime.
    if isinstance(v, pd.Timestamp):
        return v.date().isoformat()
    import datetime as _dt

    if isinstance(v, _dt.datetime):
        return v.isoformat()
    if isinstance(v, date):
        return v.isoformat()
    # numpy scalars
    if hasattr(v, "item"):
        try:
            return v.item()
        except Exception:
            pass
    return str(v)


# ---------------------------------------------------------------------------
# 2a. Template library
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Slot:
    """A typed, defaulted parameter of a template.

    ``type`` is one of ``"int"``, ``"date"`` or ``"enum"``.  ``enum`` slots carry
    ``options`` (an allowlist).  ``int`` slots may carry ``min``/``max`` bounds.
    Validation is strict: bad input raises :class:`SlotError`, never silently coerces
    an arbitrary string into SQL.
    """

    name: str
    type: str
    default: Any
    label: str = ""
    min: Optional[int] = None
    max: Optional[int] = None
    options: Optional[Tuple[Any, ...]] = None

    def validate(self, value: Any) -> Any:
        if self.type == "int":
            try:
                iv = int(value)
            except (TypeError, ValueError):
                raise SlotError(f"slot {self.name!r} expects an integer, got {value!r}")
            if self.min is not None and iv < self.min:
                raise SlotError(f"slot {self.name!r} must be >= {self.min}")
            if self.max is not None and iv > self.max:
                raise SlotError(f"slot {self.name!r} must be <= {self.max}")
            return iv
        if self.type == "date":
            if isinstance(value, date):
                return value
            try:
                return date.fromisoformat(str(value)[:10])
            except (TypeError, ValueError):
                raise SlotError(
                    f"slot {self.name!r} expects an ISO date (YYYY-MM-DD), got {value!r}"
                )
        if self.type == "enum":
            opts = self.options or ()
            if value not in opts:
                raise SlotError(
                    f"slot {self.name!r} must be one of {list(opts)}, got {value!r}"
                )
            return value
        raise SlotError(f"unknown slot type {self.type!r}")

    def to_payload(self) -> dict:
        d = {"name": self.name, "type": self.type, "default": _slot_default_json(self.default),
             "label": self.label or self.name}
        if self.min is not None:
            d["min"] = self.min
        if self.max is not None:
            d["max"] = self.max
        if self.options is not None:
            d["options"] = list(self.options)
        return d


def _slot_default_json(v: Any) -> Any:
    return v.isoformat() if isinstance(v, date) else v


@dataclass(frozen=True)
class Template:
    """A parameterised natural-language question over the history."""

    id: str
    question: str            # NL, with {slot} placeholders
    description: str
    slots: Tuple[Slot, ...]
    builder: Callable[[Dict[str, Any]], str]

    def default_slots(self) -> Dict[str, Any]:
        return {s.name: s.default for s in self.slots}

    def validate_slots(self, provided: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        provided = provided or {}
        unknown = set(provided) - {s.name for s in self.slots}
        if unknown:
            raise SlotError(f"unknown slot(s) {sorted(unknown)} for template {self.id!r}")
        out: Dict[str, Any] = {}
        for s in self.slots:
            raw = provided.get(s.name, s.default)
            out[s.name] = s.validate(raw)
        return out

    def build(self, provided: Optional[Dict[str, Any]] = None) -> str:
        vals = self.validate_slots(provided)
        return self.builder(vals).strip()

    def render_question(self, vals: Dict[str, Any]) -> str:
        show = {k: (v.isoformat() if isinstance(v, date) else v) for k, v in vals.items()}
        try:
            return self.question.format(**show)
        except Exception:
            return self.question

    def to_payload(self) -> dict:
        return {
            "id": self.id,
            "question": self.question,
            "description": self.description,
            "slots": [s.to_payload() for s in self.slots],
            "default_sql": self.build(),
        }


# ---- SQL builders --------------------------------------------------------- #
# Every builder receives already-validated slot values (ints / dates / enum members),
# so substitution by value is safe.  Enum values are drawn from a fixed allowlist; there
# is no raw user string anywhere in these strings.
def _b_most_skipped_artists(s):
    return f"""
SELECT t.artist_name AS artist,
       COUNT(*) AS plays,
       SUM(CASE WHEN e.skipped THEN 1 ELSE 0 END) AS skips,
       ROUND(SUM(CASE WHEN e.skipped THEN 1 ELSE 0 END) * 1.0 / COUNT(*), 4) AS skip_rate
FROM events e JOIN tracks t ON e.track_uri = t.uri
GROUP BY t.artist_name
HAVING COUNT(*) >= {s['min_plays']}
ORDER BY skip_rate DESC, plays DESC, artist
LIMIT {s['limit']}
"""


def _b_abandoned_artists(s):
    return f"""
WITH span AS (SELECT MIN(date) AS mn, MAX(date) AS mx FROM events),
     mid AS (SELECT mn + CAST((mx - mn) / 2 AS INTEGER) AS mid_date FROM span),
     per AS (
       SELECT t.artist_name AS artist,
              SUM(CASE WHEN e.date <= (SELECT mid_date FROM mid) THEN 1 ELSE 0 END) AS first_half,
              SUM(CASE WHEN e.date >  (SELECT mid_date FROM mid) THEN 1 ELSE 0 END) AS second_half
       FROM events e JOIN tracks t ON e.track_uri = t.uri
       GROUP BY t.artist_name)
SELECT artist, first_half, second_half
FROM per
WHERE first_half >= {s['min_plays']} AND second_half = 0
ORDER BY first_half DESC, artist
LIMIT {s['limit']}
"""


def _b_one_hit_wonders(s):
    return f"""
WITH track_plays AS (
       SELECT t.artist_name AS artist, e.track_uri AS uri, COUNT(*) AS plays
       FROM events e JOIN tracks t ON e.track_uri = t.uri
       GROUP BY t.artist_name, e.track_uri),
     agg AS (
       SELECT artist,
              MAX(plays) AS top_plays,
              COUNT(*) AS n_tracks,
              SUM(CASE WHEN plays >= {s['other_max']} THEN 1 ELSE 0 END) AS n_big
       FROM track_plays GROUP BY artist)
SELECT artist, top_plays, n_tracks
FROM agg
WHERE top_plays >= {s['hit_plays']} AND n_big = 1
ORDER BY top_plays DESC, artist
LIMIT {s['limit']}
"""


def _b_night_owl_tracks(s):
    return f"""
SELECT ANY_VALUE(t.track_name) AS track,
       ANY_VALUE(t.artist_name) AS artist,
       SUM(CASE WHEN e.hour >= {s['start_hour']} AND e.hour < {s['end_hour']} THEN 1 ELSE 0 END) AS night_plays,
       COUNT(*) AS total_plays
FROM events e JOIN tracks t ON e.track_uri = t.uri
GROUP BY e.track_uri
HAVING SUM(CASE WHEN e.hour >= {s['start_hour']} AND e.hour < {s['end_hour']} THEN 1 ELSE 0 END) >= {s['min_plays']}
ORDER BY night_plays DESC, total_plays DESC, track
LIMIT {s['limit']}
"""


def _b_longest_binge_day_per_artist(s):
    return f"""
WITH per_day AS (
       SELECT t.artist_name AS artist, e.date AS day, COUNT(*) AS plays
       FROM events e JOIN tracks t ON e.track_uri = t.uri
       GROUP BY t.artist_name, e.date),
     ranked AS (
       SELECT artist, day, plays,
              ROW_NUMBER() OVER (PARTITION BY artist ORDER BY plays DESC, day) AS rn
       FROM per_day)
SELECT artist, day, plays
FROM ranked WHERE rn = 1
ORDER BY plays DESC, artist
LIMIT {s['limit']}
"""


def _b_returning_tracks(s):
    return f"""
WITH ordered AS (
       SELECT e.track_uri AS uri, e.date AS day,
              LAG(e.date) OVER (PARTITION BY e.track_uri ORDER BY e.ts) AS prev_day
       FROM events e),
     gaps AS (
       SELECT uri, MAX(day - prev_day) AS max_gap_days
       FROM ordered WHERE prev_day IS NOT NULL
       GROUP BY uri)
SELECT ANY_VALUE(t.track_name) AS track, ANY_VALUE(t.artist_name) AS artist, g.max_gap_days
FROM gaps g JOIN tracks t ON g.uri = t.uri
WHERE g.max_gap_days >= {s['months']} * 30
GROUP BY g.uri, g.max_gap_days
ORDER BY g.max_gap_days DESC, track
LIMIT {s['limit']}
"""


def _b_weekend_vs_weekday_artists(s):
    return f"""
SELECT t.artist_name AS artist,
       SUM(CASE WHEN e.weekday >= 5 THEN 1 ELSE 0 END) AS weekend_plays,
       SUM(CASE WHEN e.weekday <  5 THEN 1 ELSE 0 END) AS weekday_plays,
       COUNT(*) AS plays,
       ROUND(SUM(CASE WHEN e.weekday >= 5 THEN 1 ELSE 0 END) * 1.0 / COUNT(*), 4) AS weekend_share
FROM events e JOIN tracks t ON e.track_uri = t.uri
GROUP BY t.artist_name
HAVING COUNT(*) >= {s['min_plays']}
ORDER BY weekend_share DESC, plays DESC, artist
LIMIT {s['limit']}
"""


def _b_average_session_gap(s):
    return f"""
WITH ordered AS (
       SELECT ts, LAG(ts) OVER (ORDER BY ts) AS prev_ts FROM events),
     gaps AS (
       SELECT epoch(ts) - epoch(prev_ts) AS gap_s
       FROM ordered WHERE prev_ts IS NOT NULL)
SELECT ROUND(AVG(gap_s) / 60.0, 2) AS avg_gap_minutes,
       ROUND(MEDIAN(gap_s) / 60.0, 2) AS median_gap_minutes,
       SUM(CASE WHEN gap_s > {s['gap_minutes']} * 60 THEN 1 ELSE 0 END) + 1 AS n_sessions,
       COUNT(*) + 1 AS n_plays
FROM gaps
"""


def _b_most_replayed_track_per_month(s):
    return f"""
WITH per AS (
       SELECT date_trunc('month', e.date) AS month, e.track_uri AS uri, COUNT(*) AS plays
       FROM events e GROUP BY month, e.track_uri),
     ranked AS (
       SELECT month, uri, plays,
              ROW_NUMBER() OVER (PARTITION BY month ORDER BY plays DESC, uri) AS rn
       FROM per)
SELECT r.month, ANY_VALUE(t.track_name) AS track, ANY_VALUE(t.artist_name) AS artist, r.plays
FROM ranked r JOIN tracks t ON r.uri = t.uri
WHERE r.rn = 1
GROUP BY r.month, r.plays
ORDER BY r.month
LIMIT {s['limit']}
"""


def _b_discovery_by_month(s):
    return """
WITH firsts AS (SELECT track_uri, MIN(event_id) AS first_ev FROM events GROUP BY track_uri)
SELECT date_trunc('month', e.date) AS month,
       COUNT(*) AS plays,
       SUM(CASE WHEN e.event_id = f.first_ev THEN 1 ELSE 0 END) AS new_tracks
FROM events e JOIN firsts f ON e.track_uri = f.track_uri
GROUP BY month
ORDER BY month
"""


def _b_skip_rate_by_hour(s):
    return """
SELECT e.hour AS hour,
       COUNT(*) AS plays,
       ROUND(SUM(CASE WHEN e.skipped THEN 1 ELSE 0 END) * 1.0 / COUNT(*), 4) AS skip_rate
FROM events e
GROUP BY e.hour
ORDER BY e.hour
"""


def _b_one_and_done_tracks(s):
    return f"""
SELECT ANY_VALUE(t.track_name) AS track, ANY_VALUE(t.artist_name) AS artist, COUNT(*) AS plays
FROM events e JOIN tracks t ON e.track_uri = t.uri
GROUP BY e.track_uri
HAVING COUNT(*) = 1
ORDER BY track
LIMIT {s['limit']}
"""


def _b_busiest_days(s):
    return f"""
SELECT e.date AS day,
       COUNT(*) AS plays,
       ROUND(SUM(e.ms_played) / 60000.0, 1) AS minutes
FROM events e
GROUP BY e.date
ORDER BY plays DESC, day
LIMIT {s['limit']}
"""


def _b_longest_listening_streak(s):
    return f"""
WITH days AS (SELECT DISTINCT date AS day FROM events),
     grp AS (
       SELECT day, day - CAST(ROW_NUMBER() OVER (ORDER BY day) AS INTEGER) AS g
       FROM days),
     runs AS (
       SELECT MIN(day) AS start_day, MAX(day) AS end_day, COUNT(*) AS length
       FROM grp GROUP BY g)
SELECT start_day, end_day, length
FROM runs
ORDER BY length DESC, start_day
LIMIT {s['limit']}
"""


def _b_artist_lifespan(s):
    return f"""
SELECT t.artist_name AS artist,
       MIN(e.date) AS first_play,
       MAX(e.date) AS last_play,
       (MAX(e.date) - MIN(e.date)) AS span_days,
       COUNT(*) AS plays
FROM events e JOIN tracks t ON e.track_uri = t.uri
GROUP BY t.artist_name
HAVING COUNT(*) >= {s['min_plays']}
ORDER BY span_days DESC, plays DESC, artist
LIMIT {s['limit']}
"""


def _b_most_skipped_tracks(s):
    return f"""
SELECT ANY_VALUE(t.track_name) AS track, ANY_VALUE(t.artist_name) AS artist,
       COUNT(*) AS plays,
       ROUND(SUM(CASE WHEN e.skipped THEN 1 ELSE 0 END) * 1.0 / COUNT(*), 4) AS skip_rate
FROM events e JOIN tracks t ON e.track_uri = t.uri
GROUP BY e.track_uri
HAVING COUNT(*) >= {s['min_plays']}
ORDER BY skip_rate DESC, plays DESC, track
LIMIT {s['limit']}
"""


def _b_new_artists_per_month(s):
    return """
WITH firsts AS (
       SELECT t.artist_name AS artist, MIN(e.date) AS first_day
       FROM events e JOIN tracks t ON e.track_uri = t.uri
       GROUP BY t.artist_name)
SELECT date_trunc('month', first_day) AS month, COUNT(*) AS new_artists
FROM firsts
GROUP BY month
ORDER BY month
"""


def _b_plays_by_weekday(s):
    return """
SELECT e.weekday AS weekday,
       COUNT(*) AS plays,
       ROUND(SUM(e.ms_played) / 60000.0, 1) AS minutes
FROM events e
GROUP BY e.weekday
ORDER BY e.weekday
"""


def _b_comeback_artists(s):
    return f"""
WITH ordered AS (
       SELECT t.artist_name AS artist, e.date AS day,
              LAG(e.date) OVER (PARTITION BY t.artist_name ORDER BY e.ts) AS prev_day
       FROM events e JOIN tracks t ON e.track_uri = t.uri),
     g AS (
       SELECT artist, MAX(day - prev_day) AS max_gap_days
       FROM ordered WHERE prev_day IS NOT NULL
       GROUP BY artist),
     tot AS (
       SELECT t.artist_name AS artist, COUNT(*) AS plays
       FROM events e JOIN tracks t ON e.track_uri = t.uri
       GROUP BY t.artist_name)
SELECT g.artist, g.max_gap_days, tot.plays
FROM g JOIN tot ON g.artist = tot.artist
WHERE g.max_gap_days >= {s['months']} * 30 AND tot.plays >= {s['min_plays']}
ORDER BY g.max_gap_days DESC, g.artist
LIMIT {s['limit']}
"""


def _b_heaviest_rotation_week(s):
    return f"""
WITH per AS (
       SELECT e.track_uri AS uri, date_trunc('week', e.date) AS week, COUNT(*) AS plays
       FROM events e GROUP BY e.track_uri, week)
SELECT ANY_VALUE(t.track_name) AS track, ANY_VALUE(t.artist_name) AS artist, p.week, p.plays
FROM per p JOIN tracks t ON p.uri = t.uri
GROUP BY p.uri, p.week, p.plays
ORDER BY p.plays DESC, track
LIMIT {s['limit']}
"""


_LIMIT = lambda default=20: Slot("limit", "int", default, label="How many rows", min=1, max=500)

TEMPLATES: Tuple[Template, ...] = (
    Template(
        "most_skipped_artists",
        "Which artists do I skip the most (with at least {min_plays} plays)?",
        "Artists ranked by skip rate, restricted to those with enough plays to be meaningful.",
        (Slot("min_plays", "int", 20, label="Minimum plays", min=1), _LIMIT()),
        _b_most_skipped_artists,
    ),
    Template(
        "abandoned_artists",
        "Which artists did I play at least {min_plays} times in the first half of my "
        "history and then never again?",
        "Artists with >= N plays before the midpoint date and zero plays after it.",
        (Slot("min_plays", "int", 5, label="Minimum early plays", min=1), _LIMIT()),
        _b_abandoned_artists,
    ),
    Template(
        "one_hit_wonders",
        "Which artists are one-hit wonders for me (one track with >= {hit_plays} plays, "
        "everything else under {other_max})?",
        "Artists where exactly one track cleared the hit threshold and no other track did.",
        (Slot("hit_plays", "int", 15, label="Hit threshold (plays)", min=1),
         Slot("other_max", "int", 3, label="Other tracks under", min=1), _LIMIT()),
        _b_one_hit_wonders,
    ),
    Template(
        "night_owl_tracks",
        "Which tracks do I play most between {start_hour}:00 and {end_hour}:00?",
        "Tracks ranked by plays that fall inside the given hour band (night-owl listening).",
        (Slot("start_hour", "int", 0, label="From hour", min=0, max=23),
         Slot("end_hour", "int", 5, label="To hour (exclusive)", min=1, max=24),
         Slot("min_plays", "int", 3, label="Minimum night plays", min=1), _LIMIT()),
        _b_night_owl_tracks,
    ),
    Template(
        "longest_binge_day_per_artist",
        "For each artist, what was my single biggest listening day?",
        "The top artists by their single most-played calendar day.",
        (_LIMIT(),),
        _b_longest_binge_day_per_artist,
    ),
    Template(
        "returning_tracks",
        "Which tracks came back after I ignored them for at least {months} months?",
        "Tracks with a gap of >= K*30 days between two consecutive plays (approx months).",
        (Slot("months", "int", 3, label="Away for at least (months)", min=1), _LIMIT()),
        _b_returning_tracks,
    ),
    Template(
        "weekend_vs_weekday_artists",
        "Which artists are most a weekend thing for me (min {min_plays} plays)?",
        "Artists ranked by the share of their plays that fall on Saturday/Sunday.",
        (Slot("min_plays", "int", 20, label="Minimum plays", min=1), _LIMIT()),
        _b_weekend_vs_weekday_artists,
    ),
    Template(
        "average_session_gap",
        "What's the typical gap between my plays, and how many sessions do I have "
        "(a gap over {gap_minutes} minutes starts a new session)?",
        "Average/median gap between consecutive plays, plus a session count at the threshold.",
        (Slot("gap_minutes", "int", 30, label="Session gap (minutes)", min=1),),
        _b_average_session_gap,
    ),
    Template(
        "most_replayed_track_per_month",
        "What was my most replayed track in each month?",
        "The single most-played track per calendar month.",
        (Slot("limit", "int", 24, label="How many months", min=1, max=500),),
        _b_most_replayed_track_per_month,
    ),
    Template(
        "discovery_by_month",
        "How many brand-new tracks did I discover each month?",
        "Per month: total plays and how many were the first-ever play of their track.",
        (),
        _b_discovery_by_month,
    ),
    Template(
        "skip_rate_by_hour",
        "How does my skip rate change over the hours of the day?",
        "Skip rate and play count for each hour 0..23.",
        (),
        _b_skip_rate_by_hour,
    ),
    Template(
        "one_and_done_tracks",
        "Which tracks did I play exactly once and never again?",
        "Tracks with exactly one play in the whole history.",
        (_LIMIT(50),),
        _b_one_and_done_tracks,
    ),
    Template(
        "busiest_days",
        "What were my busiest listening days?",
        "Calendar days ranked by number of plays, with minutes listened.",
        (_LIMIT(20),),
        _b_busiest_days,
    ),
    Template(
        "longest_listening_streak",
        "What's my longest streak of consecutive days with any listening?",
        "Runs of back-to-back days that each had at least one play, longest first.",
        (_LIMIT(5),),
        _b_longest_listening_streak,
    ),
    Template(
        "artist_lifespan",
        "Which artists have I followed the longest (min {min_plays} plays)?",
        "Per artist: first play, last play and the span of days between them.",
        (Slot("min_plays", "int", 10, label="Minimum plays", min=1), _LIMIT()),
        _b_artist_lifespan,
    ),
    Template(
        "most_skipped_tracks",
        "Which individual tracks do I skip the most (min {min_plays} plays)?",
        "Tracks ranked by skip rate among those with enough plays.",
        (Slot("min_plays", "int", 10, label="Minimum plays", min=1), _LIMIT()),
        _b_most_skipped_tracks,
    ),
    Template(
        "new_artists_per_month",
        "How many new artists did I start listening to each month?",
        "Count of artists whose first-ever play falls in each month.",
        (),
        _b_new_artists_per_month,
    ),
    Template(
        "plays_by_weekday",
        "How are my plays spread across the days of the week?",
        "Plays and minutes per weekday (0=Monday .. 6=Sunday).",
        (),
        _b_plays_by_weekday,
    ),
    Template(
        "comeback_artists",
        "Which artists did I abandon for over {months} months and then come back to "
        "(min {min_plays} total plays)?",
        "Artists with a large consecutive-play gap who nonetheless have real total volume.",
        (Slot("months", "int", 2, label="Gap of at least (months)", min=1),
         Slot("min_plays", "int", 15, label="Minimum total plays", min=1), _LIMIT()),
        _b_comeback_artists,
    ),
    Template(
        "heaviest_rotation_week",
        "What track was in my heaviest single-week rotation ever?",
        "The (track, week) pairs with the most plays of one track inside one calendar week.",
        (_LIMIT(20),),
        _b_heaviest_rotation_week,
    ),
)

TEMPLATES_BY_ID: Dict[str, Template] = {t.id: t for t in TEMPLATES}


def list_templates() -> List[dict]:
    """JSON-ready descriptions of every template (id, question, slots, default SQL)."""
    return [t.to_payload() for t in TEMPLATES]


def get_template(template_id: str) -> Template:
    try:
        return TEMPLATES_BY_ID[template_id]
    except KeyError:
        raise SlotError(f"unknown template {template_id!r}") from None


def run_template(
    store,
    template_id: str,
    slots: Optional[Dict[str, Any]] = None,
    max_rows: int = 500,
    timeout_s: float = 5.0,
    **kw_slots: Any,
) -> dict:
    """Validate slots, build the template's SQL and run it through :func:`safe_execute`.

    Slots may be passed as a dict (``slots=``) or as keyword arguments; both are
    validated against the template's typed slot specs.  Returns the ``safe_execute``
    result augmented with ``template_id`` and the filled-in ``question``.
    """
    tmpl = get_template(template_id)
    provided = dict(slots or {})
    provided.update(kw_slots)
    vals = tmpl.validate_slots(provided)
    sql = tmpl.builder(vals).strip()
    result = safe_execute(store, sql, max_rows=max_rows, timeout_s=timeout_s)
    result["template_id"] = template_id
    result["question"] = tmpl.render_question(vals)
    result["slots"] = {k: _slot_default_json(v) for k, v in vals.items()}
    return result


# ---------------------------------------------------------------------------
# 3. Optional LLM NL->SQL path (no client, no keys, no network)
# ---------------------------------------------------------------------------
# A small gold few-shot set: (question, sql).  These teach the shape of the schema and
# the conventions; they are also part of the eval's LLM-only coverage.
FEW_SHOT: Tuple[Tuple[str, str], ...] = (
    ("How many total plays are in my history?",
     "SELECT COUNT(*) AS plays FROM events"),
    ("What are my top 10 artists by play count?",
     "SELECT t.artist_name AS artist, COUNT(*) AS plays\n"
     "FROM events e JOIN tracks t ON e.track_uri = t.uri\n"
     "GROUP BY t.artist_name ORDER BY plays DESC, artist LIMIT 10"),
    ("How many minutes have I listened in total?",
     "SELECT ROUND(SUM(ms_played) / 60000.0, 1) AS minutes FROM events"),
    ("What's my overall skip rate?",
     "SELECT ROUND(SUM(CASE WHEN skipped THEN 1 ELSE 0 END) * 1.0 / COUNT(*), 4) AS skip_rate\n"
     "FROM events"),
    ("How many plays happened on weekends?",
     "SELECT COUNT(*) AS plays FROM events WHERE weekday >= 5"),
    ("Which artist have I spent the most minutes on?",
     "SELECT t.artist_name AS artist, ROUND(SUM(e.ms_played) / 60000.0, 1) AS minutes\n"
     "FROM events e JOIN tracks t ON e.track_uri = t.uri\n"
     "GROUP BY t.artist_name ORDER BY minutes DESC, artist LIMIT 5"),
    ("What was my most active month?",
     "SELECT date_trunc('month', date) AS month, COUNT(*) AS plays\n"
     "FROM events GROUP BY month ORDER BY plays DESC LIMIT 1"),
)

_PROMPT_INSTRUCTIONS = """\
Translate the user's question into a single DuckDB SQL query over the schema above.
Rules:
  * Output ONLY the SQL, optionally in a ```sql fenced block. No prose.
  * A single read-only SELECT (or WITH ... SELECT). Never write, never use PRAGMA/ATTACH.
  * Use the exact table/column names and the conventions from the schema card.
  * Prefer explicit column aliases; add a LIMIT for "top"/"most" questions.
"""


def build_prompt(question: str) -> str:
    """Assemble the NL->SQL prompt: schema card + instructions + few-shot + question."""
    lines = [SCHEMA_CARD, "", _PROMPT_INSTRUCTIONS, "", "Examples:"]
    for q, sql in FEW_SHOT:
        lines.append(f"Q: {q}\nSQL:\n{sql}\n")
    lines.append(f"Q: {question}\nSQL:")
    return "\n".join(lines)


_FENCE_RE = re.compile(r"```(?:sql)?\s*(.*?)```", re.DOTALL | re.IGNORECASE)


def extract_sql(response: str) -> str:
    """Pull the SQL out of an LLM response: strip ``` fences and leading labels."""
    if not response:
        raise GuardrailError("empty LLM response")
    text = response.strip()
    m = _FENCE_RE.search(text)
    if m:
        text = m.group(1).strip()
    # drop a leading "SQL:" label if present
    text = re.sub(r"^\s*sql\s*:\s*", "", text, flags=re.IGNORECASE)
    return text.strip().rstrip(";").strip()


def nl_to_sql(question: str, llm_fn: Optional[Callable[[str], str]] = None) -> str:
    """Turn a natural-language question into SQL using an injected ``llm_fn``.

    ``llm_fn`` is a callable ``prompt -> completion``.  This function builds the prompt
    (:func:`build_prompt`), calls ``llm_fn`` and extracts the SQL (:func:`extract_sql`).
    It performs **no** validation or execution — pass the result to
    :func:`safe_execute`, or use :func:`ask` for the whole pipeline.  Raises if no
    ``llm_fn`` is provided (the keyless path is :func:`run_template`).
    """
    if llm_fn is None:
        raise GuardrailError(
            "no llm_fn provided: the natural-language path needs an injected LLM "
            "callable; the keyless path is run_template()/list_templates()"
        )
    prompt = build_prompt(question)
    return extract_sql(llm_fn(prompt))


def ask(
    store,
    question: str,
    llm_fn: Optional[Callable[[str], str]] = None,
    max_rows: int = 500,
    timeout_s: float = 5.0,
) -> dict:
    """Full NL pipeline: question -> SQL (via ``llm_fn``) -> guarded execution.

    Returns the :func:`safe_execute` result with the original ``question`` attached.
    The generated SQL is always in the result (``result["sql"]``) so the UI can show it.
    """
    sql = nl_to_sql(question, llm_fn)
    result = safe_execute(store, sql, max_rows=max_rows, timeout_s=timeout_s)
    result["question"] = question
    return result


# ---------------------------------------------------------------------------
# Fingerprint helpers (shared with the gold eval module)
# ---------------------------------------------------------------------------
def result_fingerprint(df) -> Tuple[int, str]:
    """A stable fingerprint of a result frame: (row_count, hash of sorted 1st-col values).

    The hash is over the **first column's values**, rendered to stable strings (floats
    rounded to 4 dp, dates/timestamps as ISO) and sorted, so it is invariant to row
    order but sensitive to the actual returned set.  Row count is kept separately so a
    result that differs only in duplicate multiplicity is still distinguishable.
    """
    n = int(len(df))
    if df.shape[1] == 0:
        payload = ""
    else:
        vals = [_fp_norm(v) for v in df.iloc[:, 0].tolist()]
        payload = "\n".join(sorted(vals))
    h = hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]
    return n, h


def _fp_norm(v: Any) -> str:
    import pandas as pd

    if v is None:
        return "∅"
    try:
        if pd.isna(v):
            return "∅"
    except (TypeError, ValueError):
        pass
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, float):
        return f"{v:.4f}"
    if isinstance(v, pd.Timestamp):
        return v.date().isoformat()
    if hasattr(v, "item") and not isinstance(v, (str, bytes)):
        try:
            v = v.item()
        except Exception:
            pass
        if isinstance(v, float):
            return f"{v:.4f}"
    if isinstance(v, date):
        return v.isoformat()
    return str(v)
