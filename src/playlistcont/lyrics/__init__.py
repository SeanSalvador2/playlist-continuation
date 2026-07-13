"""LLM lyric-interview tier: durable schema, validator, and DataFrame merge.

See ``LYRICS_RESEARCH.md`` (research scratchpad, not part of this repo) for the
full design rationale. This package only ships the *machinery* around the
interview contract -- it never contains raw lyric text (copyrighted) and never
calls an LLM itself; callers are responsible for producing the per-track JSON
files (e.g. via a Batches API run) and pointing :mod:`playlistcont.lyrics.merge`
at the directory that holds them.
"""
from .schema import (
    EMOTION_VOCAB,
    QUESTION_BATTERY,
    SCHEMA_VERSION,
    THEME_VOCAB,
    EmotionalArc,
    LyricInterview,
    lyrics_sha256,
)
from .validate import validate_interview

__all__ = [
    "EMOTION_VOCAB",
    "QUESTION_BATTERY",
    "SCHEMA_VERSION",
    "THEME_VOCAB",
    "EmotionalArc",
    "LyricInterview",
    "lyrics_sha256",
    "validate_interview",
]
