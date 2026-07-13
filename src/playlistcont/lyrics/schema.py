"""The LLM lyric-interview output contract, as code.

This mirrors ``LYRICS_RESEARCH.md`` Sec 5.2 (question battery + JSON schema)
and Sec 6 (recommended lyric-axis schema), with a few deliberate deviations
documented below. Nothing here calls an LLM or touches raw lyric text; it
just fixes the *shape* every interview answer must have so that
:mod:`playlistcont.lyrics.validate` and :mod:`playlistcont.lyrics.merge` have
a stable contract to check against.

Deviations from research-doc Sec 5.2 (documented per task instructions):

1. **Field names carry a ``lyric_`` prefix** (``lyric_narrative``,
   ``lyric_complexity``, ``lyric_sentiment_intended``) instead of Sec 5.2's
   bare names (``narrative_score``, ``complexity``, ``intended_sentiment``).
   This matches the final axis names chosen in Sec 6 and the ``genre:`` /
   scalar-axis naming convention already used in
   ``playlistcont.data.schema``.
2. **``lyric_sentiment_intended`` lives in [0, 1]**, not Sec 5.2's [-1, 1].
   Sec 6 is explicit that "All scalar axes deliberately live in [0, 1] to
   slot into the existing ``SCALAR_AXES`` convention" -- so the interview
   schema follows Sec 6, not the earlier draft in 5.2. 0.0 = bleak/negative,
   1.0 = joyful/positive, 0.5 = neutral/mixed.
3. **Added ``confidence`` in [0, 1]** (not present in Sec 5.2). The research
   doc flags non-English and ambiguous-irony songs as lower-confidence cases
   (Sec 3 "honest failures", Sec 5.1 Q7); this field makes that judgment call
   part of the record instead of an unrecorded vibe.
4. **Added versioning/meta fields** ``schema_version``, ``interviewer``, and
   ``lyrics_sha256`` per the caching design in Sec 5.3 (keyed cache entries
   that must be invalidated on schema or text changes) -- Sec 5.2's schema
   snippet only covered the *judgment* fields, not the provenance envelope.
5. **Question 7 asks directly for the ISO 639-1 code** rather than Sec 5.1's
   "Is the language English? If not, name it" -- simpler to validate and
   maps 1:1 onto the ``language`` field.
6. **``themes`` and ``emotional_arc`` vocabularies are copied verbatim**
   from Sec 5.2 (20 themes, 11 emotions) -- no deviation there.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Any, Dict, List, Tuple

SCHEMA_VERSION = "1.0"

# ---------------------------------------------------------------------------
# Fixed question battery (Sec 5.1). Identical for every song: this is what
# makes per-track caching and later comparability possible.
# ---------------------------------------------------------------------------
QUESTION_BATTERY: Tuple[str, ...] = (
    "What are the main themes? Choose 1-3 from the controlled vocabulary only.",
    "Is this song a story or a vibe? Score 0.0 (pure vibe/hook -- no events, "
    "no arc) to 1.0 (full narrative -- characters, events, resolution).",
    "What is the song's message -- what is the narrator trying to say? "
    "Answer in <= 25 words, as a paraphrase. Never quote lyric lines "
    "verbatim.",
    "Describe the emotional arc: the dominant emotion of the opening, the "
    "middle, and the ending, each from the controlled emotion list.",
    "Rate lyrical complexity 0.0-1.0 (vocabulary, imagery, wordplay, "
    "figurative density -- not repetition alone).",
    "Rate sentiment as intended by the narrator (not surface word choice), "
    "0.0 (bleak/negative) to 1.0 (joyful/positive) -- e.g. a swagger song "
    "using 'bad' to mean good scores positive, not negative.",
    "What language are the lyrics written in? Give the ISO 639-1 code.",
)

# ---------------------------------------------------------------------------
# Controlled vocabularies (frozen -- changing these is a schema_version bump)
# ---------------------------------------------------------------------------
THEME_VOCAB: Tuple[str, ...] = (
    "love_romance", "heartbreak_loss", "party_celebration",
    "struggle_perseverance", "wealth_status", "violence_conflict",
    "faith_spirituality", "nostalgia_memory", "rebellion_defiance",
    "identity_selfworth", "social_commentary", "family_friendship",
    "lust_desire", "grief_death", "escape_freedom", "place_hometown",
    "nature_seasons", "fame_music_itself", "humor_absurdity", "other",
)

EMOTION_VOCAB: Tuple[str, ...] = (
    "joy", "sadness", "anger", "fear", "longing", "pride", "calm",
    "excitement", "despair", "hope", "neutral",
)

# Full ISO 639-1 two-letter code set (used by the validator to catch typos
# like "eng" or "En" while still accepting any real language, not just the
# handful this project's own corpus happens to use).
ISO_639_1_CODES = frozenset({
    "aa", "ab", "ae", "af", "ak", "am", "an", "ar", "as", "av", "ay", "az",
    "ba", "be", "bg", "bh", "bi", "bm", "bn", "bo", "br", "bs",
    "ca", "ce", "ch", "co", "cr", "cs", "cu", "cv", "cy",
    "da", "de", "dv", "dz",
    "ee", "el", "en", "eo", "es", "et", "eu",
    "fa", "ff", "fi", "fj", "fo", "fr", "fy",
    "ga", "gd", "gl", "gn", "gu", "gv",
    "ha", "he", "hi", "ho", "hr", "ht", "hu", "hy", "hz",
    "ia", "id", "ie", "ig", "ii", "ik", "io", "is", "it", "iu",
    "ja", "jv",
    "ka", "kg", "ki", "kj", "kk", "kl", "km", "kn", "ko", "kr", "ks", "ku",
    "kv", "kw", "ky",
    "la", "lb", "lg", "li", "ln", "lo", "lt", "lu", "lv",
    "mg", "mh", "mi", "mk", "ml", "mn", "mr", "ms", "mt", "my",
    "na", "nb", "nd", "ne", "ng", "nl", "nn", "no", "nr", "nv", "ny",
    "oc", "oj", "om", "or", "os",
    "pa", "pi", "pl", "ps", "pt",
    "qu",
    "rm", "rn", "ro", "ru", "rw",
    "sa", "sc", "sd", "se", "sg", "si", "sk", "sl", "sm", "sn", "so", "sq",
    "sr", "ss", "st", "su", "sv", "sw",
    "ta", "te", "tg", "th", "ti", "tk", "tl", "tn", "to", "tr", "ts", "tt",
    "tw", "ty",
    "ug", "uk", "ur", "uz",
    "ve", "vi", "vo",
    "wa", "wo",
    "xh",
    "yi", "yo",
    "za", "zh", "zu",
})

MIN_THEMES = 1
MAX_THEMES = 3
MAX_MESSAGE_WORDS = 25

# Scalar fields that must be numbers in [0, 1].
SCALAR_FIELDS: Tuple[str, ...] = (
    "lyric_narrative",
    "lyric_complexity",
    "lyric_sentiment_intended",
    "confidence",
)

REQUIRED_FIELDS: Tuple[str, ...] = (
    "themes",
    "lyric_narrative",
    "message",
    "emotional_arc",
    "lyric_complexity",
    "lyric_sentiment_intended",
    "language",
    "confidence",
    "schema_version",
    "interviewer",
    "lyrics_sha256",
)


def lyrics_sha256(lyrics_text: str) -> str:
    """SHA-256 hex digest of raw lyric text, used as the cache-invalidation key."""
    return hashlib.sha256(lyrics_text.encode("utf-8")).hexdigest()


@dataclass
class EmotionalArc:
    start: str
    middle: str
    end: str

    def to_dict(self) -> Dict[str, str]:
        return {"start": self.start, "middle": self.middle, "end": self.end}

    @classmethod
    def from_dict(cls, obj: Dict[str, Any]) -> "EmotionalArc":
        return cls(start=obj["start"], middle=obj["middle"], end=obj["end"])


@dataclass
class LyricInterview:
    """One song's answers to :data:`QUESTION_BATTERY`, plus provenance.

    This is a plain data container -- construction does not validate; use
    :func:`playlistcont.lyrics.validate.validate_interview` on the result of
    :meth:`to_dict` (or on a loaded JSON file) to check the contract.
    """

    themes: List[str]
    lyric_narrative: float
    message: str
    emotional_arc: EmotionalArc
    lyric_complexity: float
    lyric_sentiment_intended: float
    language: str
    confidence: float
    schema_version: str = SCHEMA_VERSION
    interviewer: str = ""
    lyrics_sha256: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "themes": list(self.themes),
            "lyric_narrative": self.lyric_narrative,
            "message": self.message,
            "emotional_arc": self.emotional_arc.to_dict(),
            "lyric_complexity": self.lyric_complexity,
            "lyric_sentiment_intended": self.lyric_sentiment_intended,
            "language": self.language,
            "confidence": self.confidence,
            "schema_version": self.schema_version,
            "interviewer": self.interviewer,
            "lyrics_sha256": self.lyrics_sha256,
        }

    @classmethod
    def from_dict(cls, obj: Dict[str, Any]) -> "LyricInterview":
        return cls(
            themes=list(obj["themes"]),
            lyric_narrative=obj["lyric_narrative"],
            message=obj["message"],
            emotional_arc=EmotionalArc.from_dict(obj["emotional_arc"]),
            lyric_complexity=obj["lyric_complexity"],
            lyric_sentiment_intended=obj["lyric_sentiment_intended"],
            language=obj["language"],
            confidence=obj["confidence"],
            schema_version=obj.get("schema_version", SCHEMA_VERSION),
            interviewer=obj.get("interviewer", ""),
            lyrics_sha256=obj.get("lyrics_sha256", ""),
        )
