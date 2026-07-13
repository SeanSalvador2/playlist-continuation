"""Canned personal-listening analytics over a :class:`HistoryStore`.

The :mod:`playlistcont.history` package turns a person's play stream into a DuckDB
store (``events`` / ``tracks`` / ``track_features``).  This package is the pure query
layer the "Library" dashboard (Phase 1) sits on: each function in
:mod:`playlistcont.analytics.queries` takes a store plus an optional date window and
returns plain JSON-ready dicts/lists — no web framework, no engine, no side effects.

Later phases build on the *same* functions: Phase 2 (a statistics layer) wraps these
aggregates, and Phase 5 (text-to-SQL) generates queries against the identical schema,
so keeping this layer thin and pure is deliberate.
"""
from __future__ import annotations

from . import queries, stats

__all__ = ["queries", "stats"]
