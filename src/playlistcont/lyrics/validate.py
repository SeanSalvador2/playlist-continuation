"""Validator for the lyric-interview JSON contract (see :mod:`schema`).

``validate_interview`` never raises on malformed input -- malformed input is
exactly the thing it reports. It always returns a list of human-readable
violation strings; an empty list means the object satisfies the contract.
"""
from __future__ import annotations

import re
from typing import Any, List, Optional, Set, Tuple

from .schema import (
    EMOTION_VOCAB,
    ISO_639_1_CODES,
    MAX_MESSAGE_WORDS,
    MAX_THEMES,
    MIN_THEMES,
    REQUIRED_FIELDS,
    THEME_VOCAB,
)

# 8+-word contiguous shingle overlap between the interview's paraphrase and
# the raw lyrics is treated as verbatim quoting (Sec 7 of the research doc:
# "the interview prompt should explicitly forbid quoting lines").
SHINGLE_SIZE = 8

_WORD_RE = re.compile(r"[a-z0-9']+")


def _word_tokens(text: str) -> List[str]:
    """Lowercase word tokens, punctuation/whitespace-normalized."""
    return _WORD_RE.findall(text.lower())


def _shingles(words: List[str], n: int) -> Set[Tuple[str, ...]]:
    if len(words) < n:
        return set()
    return {tuple(words[i:i + n]) for i in range(len(words) - n + 1)}


def _has_verbatim_overlap(message: str, lyrics_text: str, n: int = SHINGLE_SIZE) -> bool:
    """True if any n-word contiguous run of the lyrics also appears in message."""
    msg_words = _word_tokens(message)
    lyr_words = _word_tokens(lyrics_text)
    if len(msg_words) < n or len(lyr_words) < n:
        return False
    return not _shingles(lyr_words, n).isdisjoint(_shingles(msg_words, n))


def _is_unit_number(x: Any) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool) and 0.0 <= float(x) <= 1.0


def validate_interview(obj: Any, lyrics_text: Optional[str] = None) -> List[str]:
    """Return all contract violations for ``obj`` (empty list == valid).

    ``obj`` should be the parsed-JSON dict for one track's interview. When
    ``lyrics_text`` (the raw cached lyric text for that track) is supplied,
    an additional NO-VERBATIM check runs: reject if any 8+-word contiguous
    shingle of the lyrics appears in ``message`` after normalizing
    whitespace/case/punctuation.
    """
    violations: List[str] = []

    if not isinstance(obj, dict):
        return [f"top-level object must be a dict, got {type(obj).__name__}"]

    for f in REQUIRED_FIELDS:
        if f not in obj:
            violations.append(f"missing required field: {f}")
    if violations:
        # Every check below assumes the field is present; nothing more to
        # check safely.
        return violations

    # --- themes ------------------------------------------------------
    themes = obj["themes"]
    if not isinstance(themes, list) or not all(isinstance(t, str) for t in themes):
        violations.append("themes must be a list of strings")
    else:
        if not (MIN_THEMES <= len(themes) <= MAX_THEMES):
            violations.append(
                f"themes must have between {MIN_THEMES} and {MAX_THEMES} "
                f"entries, got {len(themes)}"
            )
        if len(set(themes)) != len(themes):
            violations.append("themes must not contain duplicates")
        bad_themes = [t for t in themes if t not in THEME_VOCAB]
        if bad_themes:
            violations.append(
                f"themes not in controlled vocabulary: {bad_themes}"
            )

    # --- scalar [0, 1] fields -----------------------------------------
    for f in ("lyric_narrative", "lyric_complexity", "lyric_sentiment_intended",
              "confidence"):
        if not _is_unit_number(obj[f]):
            violations.append(f"{f} must be a number in [0, 1], got {obj[f]!r}")

    # --- message --------------------------------------------------------
    message = obj["message"]
    if not isinstance(message, str) or not message.strip():
        violations.append("message must be a non-empty string")
    else:
        n_words = len(message.split())
        if n_words > MAX_MESSAGE_WORDS:
            violations.append(
                f"message must be <= {MAX_MESSAGE_WORDS} words, got {n_words}"
            )
        if lyrics_text and _has_verbatim_overlap(message, lyrics_text):
            violations.append(
                f"message appears to quote lyrics verbatim "
                f"(>= {SHINGLE_SIZE}-word contiguous overlap)"
            )

    # --- emotional_arc ----------------------------------------------
    arc = obj["emotional_arc"]
    if not isinstance(arc, dict):
        violations.append("emotional_arc must be an object")
    else:
        for key in ("start", "middle", "end"):
            if key not in arc:
                violations.append(f"emotional_arc missing '{key}'")
            elif arc[key] not in EMOTION_VOCAB:
                violations.append(
                    f"emotional_arc.{key} not in emotion vocabulary: {arc[key]!r}"
                )
        extra = set(arc) - {"start", "middle", "end"}
        if extra:
            violations.append(f"emotional_arc has unexpected keys: {sorted(extra)}")

    # --- language ------------------------------------------------------
    language = obj["language"]
    if not isinstance(language, str) or not re.fullmatch(r"[a-z]{2}", language):
        violations.append(
            f"language must be a 2-letter lowercase ISO 639-1 code, got {language!r}"
        )
    elif language not in ISO_639_1_CODES:
        violations.append(f"language {language!r} is not a recognized ISO 639-1 code")

    # --- versioning / provenance -----------------------------------
    if not isinstance(obj["schema_version"], str) or not obj["schema_version"]:
        violations.append("schema_version must be a non-empty string")
    if not isinstance(obj["interviewer"], str) or not obj["interviewer"]:
        violations.append("interviewer must be a non-empty string")
    sha = obj["lyrics_sha256"]
    if not isinstance(sha, str) or not re.fullmatch(r"[0-9a-f]{64}", sha):
        violations.append("lyrics_sha256 must be a 64-char lowercase hex sha256 digest")

    return violations
