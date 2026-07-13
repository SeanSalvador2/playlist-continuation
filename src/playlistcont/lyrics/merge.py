"""Fold a directory of per-track lyric-interview JSON files into one tidy
:class:`pandas.DataFrame` (and optionally a CSV) -- one row per track.

Mirrors the merge shape implied by ``LYRICS_RESEARCH.md`` Sec 6: scalar axes,
a one-hot ``theme_*`` block (same soft-one-hot convention as ``genre:*`` in
``playlistcont.data.schema``), the emotional-arc fields, and provenance/meta
columns. Files that fail :func:`playlistcont.lyrics.validate.validate_interview`
(or are not even parseable JSON) are skipped and reported rather than raising,
so a batch run with a handful of bad outputs can still be merged and the
failures triaged separately.
"""
from __future__ import annotations

import json
import os
from typing import List, Optional, Tuple

import pandas as pd

from .schema import THEME_VOCAB
from .validate import validate_interview

META_COLUMNS: Tuple[str, ...] = (
    "track_id",
    "lyric_sentiment_intended",
    "lyric_narrative",
    "lyric_complexity",
    "confidence",
    "language",
    "arc_start",
    "arc_middle",
    "arc_end",
)

TAIL_COLUMNS: Tuple[str, ...] = (
    "message",
    "schema_version",
    "interviewer",
    "lyrics_sha256",
)

THEME_COLUMNS: Tuple[str, ...] = tuple(f"theme_{t}" for t in THEME_VOCAB)

COLUMNS: Tuple[str, ...] = META_COLUMNS + THEME_COLUMNS + TAIL_COLUMNS


def _one_hot_themes(themes: List[str]) -> dict:
    # Plain 0/1 ints (not bool) so the column plays nicely with numeric
    # feature matrices and round-trips cleanly through CSV, matching the
    # soft-one-hot ``genre:*`` convention in ``playlistcont.data.schema``.
    active = set(themes)
    return {f"theme_{t}": int(t in active) for t in THEME_VOCAB}


def _row_from_interview(track_id: str, obj: dict) -> dict:
    row = {
        "track_id": track_id,
        "lyric_sentiment_intended": float(obj["lyric_sentiment_intended"]),
        "lyric_narrative": float(obj["lyric_narrative"]),
        "lyric_complexity": float(obj["lyric_complexity"]),
        "confidence": float(obj["confidence"]),
        "language": obj["language"],
        "arc_start": obj["emotional_arc"]["start"],
        "arc_middle": obj["emotional_arc"]["middle"],
        "arc_end": obj["emotional_arc"]["end"],
        "message": obj["message"],
        "schema_version": obj["schema_version"],
        "interviewer": obj["interviewer"],
        "lyrics_sha256": obj["lyrics_sha256"],
    }
    row.update(_one_hot_themes(obj["themes"]))
    return row


def merge_interviews(
    interview_dir: str,
    out_csv: Optional[str] = None,
) -> Tuple[pd.DataFrame, List[Tuple[str, List[str]]]]:
    """Merge every ``*.json`` file in ``interview_dir`` into one DataFrame.

    Returns ``(df, skipped)`` where ``df`` has one row per valid interview
    (columns: :data:`COLUMNS`) and ``skipped`` is a list of
    ``(filename, violations)`` for files that failed to parse or validate.
    If ``out_csv`` is given, ``df`` is also written there (index=False).
    """
    rows = []
    skipped: List[Tuple[str, List[str]]] = []

    filenames = sorted(f for f in os.listdir(interview_dir) if f.endswith(".json"))
    for fname in filenames:
        path = os.path.join(interview_dir, fname)
        track_id = fname[: -len(".json")]
        try:
            with open(path, "r", encoding="utf-8") as fh:
                obj = json.load(fh)
        except (OSError, json.JSONDecodeError) as exc:
            skipped.append((fname, [f"unreadable/invalid JSON: {exc}"]))
            continue

        violations = validate_interview(obj)
        if violations:
            skipped.append((fname, violations))
            continue

        rows.append(_row_from_interview(track_id, obj))

    df = pd.DataFrame(rows, columns=list(COLUMNS))
    if out_csv:
        df.to_csv(out_csv, index=False)
    return df, skipped
