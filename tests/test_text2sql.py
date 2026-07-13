"""Tests for the Phase 5 "ask your library" text-to-SQL surface.

Three concerns, tested independently:

* **Guardrails** (:func:`playlistcont.analytics.text2sql.safe_execute`) — every write /
  DDL / metadata / multi-statement attack class is rejected; plain SELECTs pass; a LIMIT
  is always injected; ``max_rows`` is enforced; and a *file-backed* store reopened
  read-only refuses writes at the connection level.
* **Templates** — every template runs green on a small synthetic store, a handful are
  pinned against independently hand-computed pandas fixtures, and slot-type validation
  rejects bad input instead of coercing it into SQL.
* **Eval** — the templates score 100% on their covered subset of the gold set, and the
  fingerprint mechanics behave as documented.
"""
from __future__ import annotations

from collections import Counter
from datetime import date

import duckdb
import pandas as pd
import pytest

from playlistcont.analytics import text2sql as T
from playlistcont.analytics import text2sql_eval as E
from playlistcont.history.store import HistoryStore
from playlistcont.history.synthetic import make_synthetic_history


# ---------------------------------------------------------------------------
# fixtures
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def small():
    """A small in-memory synthetic store shared across the module."""
    h = make_synthetic_history(seed=5, n_days=45)
    st = HistoryStore.from_history(h)
    yield h, st
    st.close()


def _events_df(h):
    """Independent pandas frame straight from the ListenEvent objects (no SQL)."""
    return pd.DataFrame([{
        "date": e.ts.date(), "hour": e.ts.hour, "weekday": e.ts.weekday(),
        "track_uri": e.track_uri, "ms_played": e.ms_played, "skipped": bool(e.skipped),
        "artist_name": h.tracks[e.track_uri].artist_name,
        "album_name": h.tracks[e.track_uri].album_name,
        "track_name": h.tracks[e.track_uri].track_name,
    } for e in h.events])


# ---------------------------------------------------------------------------
# Guardrails — attack classes
# ---------------------------------------------------------------------------
WRITE_DDL_ATTACKS = [
    "INSERT INTO events VALUES (1)",
    "UPDATE events SET ms_played = 0",
    "DELETE FROM events",
    "DROP TABLE events",
    "CREATE TABLE x (a INT)",
    "CREATE OR REPLACE TABLE tracks AS SELECT 1",
    "ALTER TABLE events ADD COLUMN z INT",
    "TRUNCATE events",
]
METADATA_ATTACKS = [
    "PRAGMA database_list",          # DuckDB classifies PRAGMA as a SELECT statement!
    "PRAGMA table_info('events')",
    "ATTACH '/tmp/evil.db' AS e",
    "COPY events TO '/tmp/leak.csv'",
    "EXPORT DATABASE '/tmp/dump'",
    "INSTALL httpfs",
    "LOAD httpfs",
    "SET memory_limit = '1GB'",
    "CALL pragma_version()",
]
MULTI_STATEMENT_ATTACKS = [
    "SELECT 1; DROP TABLE events",
    "SELECT 1; INSERT INTO events VALUES (1)",
    "SELECT 1 -- comment\n; DROP TABLE events",       # comment-smuggled second stmt
    "SELECT 1;/* block */ DROP TABLE events",
    "SELECT 1 ; ; DELETE FROM events",
]
CTE_WRITE_ATTACKS = [
    "WITH t AS (DELETE FROM events RETURNING *) SELECT * FROM t",
    "WITH t AS (INSERT INTO events VALUES (1) RETURNING *) SELECT * FROM t",
    "WITH t AS (UPDATE events SET ms_played = 0 RETURNING *) SELECT * FROM t",
]
FILE_FUNC_ATTACKS = [
    "SELECT * FROM read_csv('/etc/passwd')",
    "SELECT * FROM read_parquet('/tmp/x.parquet')",
    "SELECT * FROM glob('/**')",
]


@pytest.mark.parametrize("sql", WRITE_DDL_ATTACKS)
def test_write_and_ddl_rejected(small, sql):
    _h, st = small
    with pytest.raises(T.GuardrailError):
        T.safe_execute(st, sql)


@pytest.mark.parametrize("sql", METADATA_ATTACKS)
def test_metadata_and_pragma_rejected(small, sql):
    _h, st = small
    with pytest.raises(T.GuardrailError):
        T.safe_execute(st, sql)


@pytest.mark.parametrize("sql", MULTI_STATEMENT_ATTACKS)
def test_multi_statement_rejected(small, sql):
    _h, st = small
    with pytest.raises(T.GuardrailError):
        T.safe_execute(st, sql)


@pytest.mark.parametrize("sql", CTE_WRITE_ATTACKS)
def test_cte_write_rejected(small, sql):
    _h, st = small
    with pytest.raises(T.GuardrailError):
        T.safe_execute(st, sql)


@pytest.mark.parametrize("sql", FILE_FUNC_ATTACKS)
def test_file_reading_functions_rejected(small, sql):
    _h, st = small
    with pytest.raises(T.GuardrailError):
        T.safe_execute(st, sql)


@pytest.mark.parametrize("sql", ["", "   ", "\n\t"])
def test_empty_rejected(small, sql):
    _h, st = small
    with pytest.raises(T.GuardrailError):
        T.safe_execute(st, sql)


def test_write_attacks_do_not_mutate_the_store(small):
    """Even attempting the attacks must leave the tables untouched."""
    _h, st = small
    before = st.query("SELECT COUNT(*) AS c FROM events")["c"][0]
    for sql in WRITE_DDL_ATTACKS + CTE_WRITE_ATTACKS + MULTI_STATEMENT_ATTACKS:
        with pytest.raises(T.GuardrailError):
            T.safe_execute(st, sql)
    after = st.query("SELECT COUNT(*) AS c FROM events")["c"][0]
    assert before == after


# ---------------------------------------------------------------------------
# Guardrails — SELECT passes, LIMIT injection, row cap
# ---------------------------------------------------------------------------
def test_plain_select_passes(small):
    _h, st = small
    res = T.safe_execute(st, "SELECT COUNT(*) AS plays FROM events")
    assert res["row_count"] == 1
    assert res["columns"] == ["plays"]
    assert res["rows"][0]["plays"] > 0


def test_with_select_passes(small):
    _h, st = small
    res = T.safe_execute(st, "WITH x AS (SELECT 1 AS a) SELECT a FROM x")
    assert res["rows"] == [{"a": 1}]


def test_limit_injected_when_absent(small):
    _h, st = small
    res = T.safe_execute(st, "SELECT event_id FROM events", max_rows=7)
    assert "LIMIT 7" in res["sql"]
    assert res["row_count"] <= 7


def test_max_rows_enforced_even_with_larger_inner_limit(small):
    _h, st = small
    # inner asks for 100 rows; the guard caps at 3
    res = T.safe_execute(st, "SELECT event_id FROM events LIMIT 100", max_rows=3)
    assert res["row_count"] == 3
    assert res["truncated"] is True


def test_result_is_jsonable(small):
    _h, st = small
    res = T.safe_execute(st, "SELECT date, ms_played, skipped FROM events LIMIT 5")
    row = res["rows"][0]
    # DATE renders as a bare ISO day (no time component)
    assert isinstance(row["date"], str) and "T" not in row["date"] and len(row["date"]) == 10


# ---------------------------------------------------------------------------
# Guardrails — read-only reopen on a file-backed store blocks writes
# ---------------------------------------------------------------------------
def test_readonly_reopen_blocks_writes_on_file_store(tmp_path):
    path = str(tmp_path / "hist.duckdb")
    h = make_synthetic_history(seed=8, n_days=20)
    st = HistoryStore.from_history(h, path=path)
    st.close()

    ro = HistoryStore.open(path, read_only=True)
    # reads work
    assert ro.query("SELECT COUNT(*) AS c FROM events")["c"][0] > 0
    # a direct write is refused by the connection (read-only)
    with pytest.raises(duckdb.Error):
        ro.conn.execute("INSERT INTO events (event_id) VALUES (999999)")
    # and safe_execute still rejects the write before it ever reaches the engine
    with pytest.raises(T.GuardrailError):
        T.safe_execute(ro, "INSERT INTO events (event_id) VALUES (999999)")
    ro.close()


def test_readonly_store_helper_returns_readonly_for_file_backed(tmp_path):
    path = str(tmp_path / "hist2.duckdb")
    h = make_synthetic_history(seed=8, n_days=20)
    HistoryStore.from_history(h, path=path).close()   # persist, release the RW handle

    # an already-read-only store is returned unchanged (no needless reopen)
    ro = HistoryStore.open(path, read_only=True)
    try:
        assert T.readonly_store(ro) is ro
        # whatever the helper returns, safe_execute refuses writes through it
        with pytest.raises(T.GuardrailError):
            T.safe_execute(T.readonly_store(ro), "INSERT INTO events (event_id) VALUES (1)")
    finally:
        ro.close()


def test_readonly_store_writable_file_falls_back_to_txn_guard(tmp_path):
    """A writable file store can't be reopened read-only while open; the RO-transaction
    guard in safe_execute still blocks writes."""
    path = str(tmp_path / "hist3.duckdb")
    h = make_synthetic_history(seed=8, n_days=20)
    HistoryStore.from_history(h, path=path).close()
    rw = HistoryStore.open(path, read_only=False)
    try:
        guarded = T.readonly_store(rw)          # falls back to rw (can't reopen)
        with pytest.raises(T.GuardrailError):
            T.safe_execute(guarded, "INSERT INTO events (event_id) VALUES (999999)")
        # and the store is genuinely unmodified
        assert rw.query("SELECT COUNT(*) AS c FROM events WHERE event_id = 999999")["c"][0] == 0
    finally:
        rw.close()


def test_readonly_store_helper_is_noop_for_memory(small):
    _h, st = small
    assert T.readonly_store(st) is st  # in-memory: cannot reopen, same object back


# ---------------------------------------------------------------------------
# Templates — coverage, run-green, and slot validation
# ---------------------------------------------------------------------------
def test_at_least_fifteen_templates():
    assert len(T.TEMPLATES) >= 15
    assert len({t.id for t in T.TEMPLATES}) == len(T.TEMPLATES)  # unique ids


def test_list_templates_shape():
    payloads = T.list_templates()
    assert len(payloads) == len(T.TEMPLATES)
    for p in payloads:
        assert {"id", "question", "description", "slots", "default_sql"} <= set(p)
        assert p["default_sql"].strip()


@pytest.mark.parametrize("tmpl", T.TEMPLATES, ids=[t.id for t in T.TEMPLATES])
def test_every_template_runs_green(small, tmpl):
    _h, st = small
    res = T.run_template(st, tmpl.id)
    assert "sql" in res and "rows" in res
    assert res["template_id"] == tmpl.id
    assert isinstance(res["question"], str) and res["question"]


def test_slot_validation_rejects_bad_types(small):
    _h, st = small
    with pytest.raises(T.SlotError):
        T.run_template(st, "most_skipped_artists", min_plays="not-an-int")
    with pytest.raises(T.SlotError):
        T.run_template(st, "most_skipped_artists", limit=0)          # below min
    with pytest.raises(T.SlotError):
        T.run_template(st, "night_owl_tracks", start_hour=99)        # above max
    with pytest.raises(T.SlotError):
        T.run_template(st, "most_skipped_artists", bogus_slot=3)     # unknown slot
    with pytest.raises(T.SlotError):
        T.get_template("no_such_template")


def test_slot_int_coercion_is_safe(small):
    """A stringy integer is accepted (coerced via int()); a SQL-injection string is not."""
    _h, st = small
    ok = T.run_template(st, "most_skipped_artists", min_plays="5")   # coerced to 5
    assert ok["slots"]["min_plays"] == 5
    with pytest.raises(T.SlotError):
        T.run_template(st, "most_skipped_artists", min_plays="5; DROP TABLE events")


# ---- pinned pandas fixtures (independent of the SQL) ---------------------- #
def test_pinned_plays_by_weekday(small):
    h, st = small
    df = _events_df(h)
    res = T.run_template(st, "plays_by_weekday")
    got = {r["weekday"]: r["plays"] for r in res["rows"]}
    expected = Counter(df["weekday"])
    assert got == {int(k): int(v) for k, v in expected.items()}
    # weekdays returned in 0..6 order
    assert [r["weekday"] for r in res["rows"]] == sorted(got)


def test_pinned_one_and_done_tracks(small):
    h, st = small
    df = _events_df(h)
    counts = Counter(df["track_uri"])
    expected_once = {u for u, c in counts.items() if c == 1}
    res = T.run_template(st, "one_and_done_tracks", limit=500)
    # every returned track was played exactly once, and the count matches
    assert res["row_count"] == len(expected_once)
    for r in res["rows"]:
        assert r["plays"] == 1


def test_pinned_most_skipped_artists(small):
    h, st = small
    df = _events_df(h)
    min_plays = 10
    grp = df.groupby("artist_name")
    stats = grp.agg(plays=("skipped", "size"), skips=("skipped", "sum"))
    stats = stats[stats["plays"] >= min_plays]
    stats["skip_rate"] = (stats["skips"] / stats["plays"]).round(4)
    stats = stats.sort_values(["skip_rate", "plays", "artist_name"],
                              ascending=[False, False, True])
    expected = list(stats.index[:20])
    res = T.run_template(st, "most_skipped_artists", min_plays=min_plays, limit=20)
    assert [r["artist"] for r in res["rows"]] == expected
    # and the skip rate matches the hand computation for the top row
    if expected:
        assert res["rows"][0]["skip_rate"] == float(stats.iloc[0]["skip_rate"])


def test_pinned_discovery_by_month(small):
    h, st = small
    # first-ever play == earliest event index per track (events are ts-sorted)
    seen = set()
    first_ids = set()
    for i, e in enumerate(h.events):
        if e.track_uri not in seen:
            seen.add(e.track_uri)
            first_ids.add(i)
    df = _events_df(h)
    df["is_first"] = [i in first_ids for i in range(len(df))]
    df["month"] = df["date"].map(lambda d: (d.year, d.month))
    expected = df.groupby("month")["is_first"].sum().to_dict()
    res = T.run_template(st, "discovery_by_month")
    got = {}
    for r in res["rows"]:
        y, m = int(r["month"][:4]), int(r["month"][5:7])
        got[(y, m)] = r["new_tracks"]
    assert got == {k: int(v) for k, v in expected.items()}


# ---------------------------------------------------------------------------
# LLM path (pure, injected callable — no client, no keys, no network)
# ---------------------------------------------------------------------------
def test_build_prompt_contains_schema_and_question():
    prompt = T.build_prompt("How many plays on Mondays?")
    assert "TABLE events" in prompt            # schema card embedded
    assert "How many plays on Mondays?" in prompt
    assert "weekday" in prompt                 # convention documented


@pytest.mark.parametrize("raw,expected", [
    ("SELECT 1", "SELECT 1"),
    ("```sql\nSELECT 1\n```", "SELECT 1"),
    ("```\nSELECT 1;\n```", "SELECT 1"),
    ("SQL: SELECT 1", "SELECT 1"),
])
def test_extract_sql_strips_fences_and_labels(raw, expected):
    assert T.extract_sql(raw) == expected


def test_nl_to_sql_uses_injected_llm(small):
    _h, st = small
    calls = {}

    def fake_llm(prompt):
        calls["prompt"] = prompt
        return "```sql\nSELECT COUNT(*) AS plays FROM events\n```"

    res = T.ask(st, "how many plays?", llm_fn=fake_llm)
    assert "TABLE events" in calls["prompt"]           # prompt was built with the schema
    assert res["rows"][0]["plays"] > 0
    assert res["question"] == "how many plays?"
    assert "LIMIT" in res["sql"]                         # still guarded


def test_nl_to_sql_requires_llm():
    with pytest.raises(T.GuardrailError):
        T.nl_to_sql("anything", llm_fn=None)


def test_llm_generated_write_is_still_blocked(small):
    _h, st = small
    with pytest.raises(T.GuardrailError):
        T.ask(st, "drop everything", llm_fn=lambda p: "DROP TABLE events")


# ---------------------------------------------------------------------------
# Eval — templates score 100% on their subset; fingerprint mechanics
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def demo():
    st = E.demo_store()
    yield st
    st.close()


def test_eval_gold_is_perfect(demo):
    out = E.run_eval(demo, "gold")
    assert out["n"] == 40
    assert out["accuracy"] == 1.0, [d for d in out["details"] if not d["ok"]]


def test_eval_templates_cover_and_score_100(demo):
    out = E.run_eval(demo, "templates")
    # exactly the template-answerable questions are considered
    assert out["n"] == len(E.TEMPLATE_QUESTION_IDS) == 20
    assert out["accuracy"] == 1.0, [d for d in out["details"] if not d["ok"]]


def test_eval_has_40_questions_with_llm_only_marked():
    assert len(E.EVAL_QUESTIONS) == 40
    assert len(E.LLM_ONLY_QUESTION_IDS) == 20
    assert len(E.TEMPLATE_QUESTION_IDS) == 20
    # every question has an expected fingerprint
    assert set(E.EXPECTED_FINGERPRINTS) == {q.id for q in E.EVAL_QUESTIONS}


def test_eval_via_llm_source(demo):
    """A perfect 'LLM' that returns each question's gold SQL scores 100% over all 40."""
    gold_by_q = {q.question: q.gold_sql for q in E.EVAL_QUESTIONS}
    out = E.run_eval(demo, lambda prompt: gold_by_q[prompt.rsplit("Q: ", 1)[1].split("\nSQL:")[0]])
    assert out["n"] == 40 and out["accuracy"] == 1.0


def test_fingerprint_order_invariant_but_value_sensitive(demo):
    a = demo.query("SELECT event_id FROM events WHERE event_id < 50 ORDER BY event_id")
    b = demo.query("SELECT event_id FROM events WHERE event_id < 50 ORDER BY event_id DESC")
    # same set of 50 ids in a different order -> identical fingerprint
    assert T.result_fingerprint(a) == T.result_fingerprint(b)
    # a different set -> different fingerprint
    c = demo.query("SELECT event_id FROM events WHERE event_id < 49 ORDER BY event_id")
    assert T.result_fingerprint(c) != T.result_fingerprint(a)


def test_fingerprint_recompute_matches_constants(demo):
    """The stored constants reproduce exactly against the seed-7 demo store."""
    recomputed = E.compute_fingerprints(demo)
    assert recomputed == E.EXPECTED_FINGERPRINTS
