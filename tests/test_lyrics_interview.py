"""Offline unit tests for the lyric-interview schema/validator/merge machinery.

All lyric text here is FABRICATED placeholder text written for this test file
-- never real lyrics -- per the project's copyright discipline (see
``playlistcont.lyrics`` docstring and ``LYRICS_RESEARCH.md`` Sec 7).
"""
import json
import os

import pandas as pd
import pytest

from playlistcont.lyrics.merge import COLUMNS, THEME_COLUMNS, merge_interviews
from playlistcont.lyrics.schema import (
    EMOTION_VOCAB,
    QUESTION_BATTERY,
    SCHEMA_VERSION,
    THEME_VOCAB,
    EmotionalArc,
    LyricInterview,
    lyrics_sha256,
)
from playlistcont.lyrics.validate import validate_interview

# A fabricated "song" -- placeholder text only, no real lyrics.
FAKE_LYRICS = """[Verse 1]
we are dancing under fabricated placeholder neon lights tonight
counting fake stars above the imaginary parking lot again
[Chorus]
this made up hook repeats about a pretend summer road trip
this made up hook repeats about a pretend summer road trip
""".strip()

FAKE_SHA = lyrics_sha256(FAKE_LYRICS)


def _valid_dict(**overrides):
    obj = {
        "themes": ["party_celebration", "nostalgia_memory"],
        "lyric_narrative": 0.4,
        "message": "A narrator remembers a carefree summer night out with friends.",
        "emotional_arc": {"start": "excitement", "middle": "joy", "end": "calm"},
        "lyric_complexity": 0.3,
        "lyric_sentiment_intended": 0.8,
        "language": "en",
        "confidence": 0.9,
        "schema_version": SCHEMA_VERSION,
        "interviewer": "claude-sonnet-subagent",
        "lyrics_sha256": FAKE_SHA,
    }
    obj.update(overrides)
    return obj


# ---------------------------------------------------------------------------
# Constants sanity
# ---------------------------------------------------------------------------

def test_question_battery_has_seven_questions():
    assert len(QUESTION_BATTERY) == 7
    assert all(isinstance(q, str) and q for q in QUESTION_BATTERY)


def test_theme_vocab_is_frozen_tuple_of_twenty():
    assert isinstance(THEME_VOCAB, tuple)
    assert len(THEME_VOCAB) == 20
    assert len(set(THEME_VOCAB)) == 20  # no dupes


def test_emotion_vocab_is_frozen_tuple():
    assert isinstance(EMOTION_VOCAB, tuple)
    assert len(set(EMOTION_VOCAB)) == len(EMOTION_VOCAB)


# ---------------------------------------------------------------------------
# validate_interview: happy path
# ---------------------------------------------------------------------------

def test_valid_interview_has_no_violations():
    assert validate_interview(_valid_dict()) == []


def test_valid_interview_with_lyrics_text_still_passes():
    # the message is a genuine paraphrase, not a quote -- should not trip
    # the verbatim check even when lyrics_text is supplied
    assert validate_interview(_valid_dict(), lyrics_text=FAKE_LYRICS) == []


# ---------------------------------------------------------------------------
# validate_interview: each violation class
# ---------------------------------------------------------------------------

def test_not_a_dict():
    v = validate_interview(["not", "a", "dict"])
    assert len(v) == 1
    assert "must be a dict" in v[0]


def test_missing_required_field():
    obj = _valid_dict()
    del obj["confidence"]
    v = validate_interview(obj)
    assert any("missing required field: confidence" in x for x in v)


@pytest.mark.parametrize("themes", [[], ["a", "b", "c", "d"]])
def test_themes_count_out_of_bounds(themes):
    v = validate_interview(_valid_dict(themes=themes))
    assert any("between 1 and 3" in x for x in v)


def test_themes_not_in_vocab():
    v = validate_interview(_valid_dict(themes=["not_a_real_theme"]))
    assert any("not in controlled vocabulary" in x for x in v)


def test_themes_duplicates_rejected():
    v = validate_interview(_valid_dict(themes=["love_romance", "love_romance"]))
    assert any("duplicates" in x for x in v)


def test_themes_wrong_type():
    v = validate_interview(_valid_dict(themes="love_romance"))
    assert any("themes must be a list of strings" in x for x in v)


@pytest.mark.parametrize(
    "field", ["lyric_narrative", "lyric_complexity", "lyric_sentiment_intended", "confidence"]
)
@pytest.mark.parametrize("bad_value", [-0.1, 1.5, "0.5"])
def test_scalar_fields_out_of_range_or_wrong_type(field, bad_value):
    v = validate_interview(_valid_dict(**{field: bad_value}))
    assert any(field in x and "[0, 1]" in x for x in v)


def test_emotional_arc_invalid_emotion():
    obj = _valid_dict(emotional_arc={"start": "furious", "middle": "joy", "end": "calm"})
    v = validate_interview(obj)
    assert any("emotional_arc.start" in x for x in v)


def test_emotional_arc_missing_key():
    obj = _valid_dict(emotional_arc={"start": "joy", "middle": "joy"})
    v = validate_interview(obj)
    assert any("missing 'end'" in x for x in v)


def test_emotional_arc_wrong_type():
    v = validate_interview(_valid_dict(emotional_arc="joy"))
    assert any("emotional_arc must be an object" in x for x in v)


def test_message_too_long():
    long_message = " ".join(["word"] * 26)
    v = validate_interview(_valid_dict(message=long_message))
    assert any("<= 25 words" in x for x in v)


def test_message_empty():
    v = validate_interview(_valid_dict(message="   "))
    assert any("non-empty string" in x for x in v)


def test_language_bad_format():
    v = validate_interview(_valid_dict(language="ENG"))
    assert any("ISO 639-1" in x for x in v)


def test_language_unrecognized_code():
    v = validate_interview(_valid_dict(language="zz"))
    assert any("not a recognized ISO 639-1 code" in x for x in v)


def test_schema_version_empty():
    v = validate_interview(_valid_dict(schema_version=""))
    assert any("schema_version" in x for x in v)


def test_interviewer_empty():
    v = validate_interview(_valid_dict(interviewer=""))
    assert any("interviewer" in x for x in v)


def test_lyrics_sha256_bad_format():
    v = validate_interview(_valid_dict(lyrics_sha256="not-a-hash"))
    assert any("lyrics_sha256" in x for x in v)


# ---------------------------------------------------------------------------
# NO-VERBATIM check
# ---------------------------------------------------------------------------

def test_verbatim_quote_is_rejected():
    # 8+ contiguous words lifted straight from FAKE_LYRICS
    quoting_message = "This made up hook repeats about a pretend summer road trip, apparently."
    v = validate_interview(_valid_dict(message=quoting_message), lyrics_text=FAKE_LYRICS)
    assert any("verbatim" in x for x in v)


def test_verbatim_check_is_case_and_punctuation_insensitive():
    quoting_message = "THIS MADE UP HOOK REPEATS ABOUT A PRETEND SUMMER, road-trip!"
    v = validate_interview(_valid_dict(message=quoting_message), lyrics_text=FAKE_LYRICS)
    assert any("verbatim" in x for x in v)


def test_verbatim_check_skipped_without_lyrics_text():
    quoting_message = "This made up hook repeats about a pretend summer road trip, apparently."
    v = validate_interview(_valid_dict(message=quoting_message))
    assert not any("verbatim" in x for x in v)


def test_short_overlap_is_not_flagged():
    # shares a few words with the lyrics but well under the 8-word shingle
    message = "A pretend summer night sparks nostalgic feelings for the narrator overall."
    v = validate_interview(_valid_dict(message=message), lyrics_text=FAKE_LYRICS)
    assert not any("verbatim" in x for x in v)


# ---------------------------------------------------------------------------
# LyricInterview dataclass round-trip
# ---------------------------------------------------------------------------

def test_round_trip_dataclass_to_dict_from_dict():
    original = LyricInterview(
        themes=["party_celebration", "nostalgia_memory"],
        lyric_narrative=0.4,
        message="A narrator remembers a carefree summer night out with friends.",
        emotional_arc=EmotionalArc(start="excitement", middle="joy", end="calm"),
        lyric_complexity=0.3,
        lyric_sentiment_intended=0.8,
        language="en",
        confidence=0.9,
        interviewer="claude-sonnet-subagent",
        lyrics_sha256=FAKE_SHA,
    )
    as_dict = original.to_dict()
    assert validate_interview(as_dict) == []

    round_tripped = LyricInterview.from_dict(as_dict)
    assert round_tripped.to_dict() == as_dict

    # JSON round trip too (this is the actual on-disk cache format)
    reloaded = json.loads(json.dumps(as_dict))
    assert validate_interview(reloaded) == []
    assert reloaded == as_dict


def test_lyrics_sha256_is_stable_and_content_addressed():
    assert lyrics_sha256(FAKE_LYRICS) == FAKE_SHA
    assert lyrics_sha256(FAKE_LYRICS + " ") != FAKE_SHA
    assert len(FAKE_SHA) == 64


# ---------------------------------------------------------------------------
# merge_interviews
# ---------------------------------------------------------------------------

def _write_json(path, obj):
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(obj, fh)


def test_merge_produces_expected_shape_and_values(tmp_path):
    d = tmp_path / "interviews"
    d.mkdir()

    track_a = _valid_dict(
        themes=["party_celebration", "nostalgia_memory"],
        lyric_sentiment_intended=0.9,
    )
    track_b = _valid_dict(
        themes=["grief_death"],
        lyric_sentiment_intended=0.1,
        emotional_arc={"start": "sadness", "middle": "despair", "end": "hope"},
    )

    _write_json(d / "track_a.json", track_a)
    _write_json(d / "track_b.json", track_b)
    # a deliberately invalid file: bad scalar range
    _write_json(d / "track_bad.json", _valid_dict(confidence=5.0))
    # a deliberately unparseable file
    (d / "track_broken.json").write_text("{not valid json", encoding="utf-8")

    out_csv = tmp_path / "features.csv"
    df, skipped = merge_interviews(str(d), out_csv=str(out_csv))

    assert list(df.columns) == list(COLUMNS)
    assert len(df) == 2
    assert set(df["track_id"]) == {"track_a", "track_b"}

    row_a = df[df["track_id"] == "track_a"].iloc[0]
    assert row_a["lyric_sentiment_intended"] == pytest.approx(0.9)
    assert row_a["theme_party_celebration"] == 1
    assert row_a["theme_nostalgia_memory"] == 1
    assert row_a["theme_grief_death"] == 0

    row_b = df[df["track_id"] == "track_b"].iloc[0]
    assert row_b["arc_start"] == "sadness"
    assert row_b["arc_end"] == "hope"
    assert row_b["theme_grief_death"] == 1

    # every theme has its own one-hot column, and only the active ones are True
    assert len(THEME_COLUMNS) == len(THEME_VOCAB)
    assert row_a[list(THEME_COLUMNS)].sum() == 2
    assert row_b[list(THEME_COLUMNS)].sum() == 1

    skipped_names = {name for name, _ in skipped}
    assert skipped_names == {"track_bad.json", "track_broken.json"}
    bad_violations = dict(skipped)["track_bad.json"]
    assert any("confidence" in v for v in bad_violations)

    assert out_csv.exists()
    reloaded = pd.read_csv(out_csv)
    assert len(reloaded) == 2
    assert set(reloaded["track_id"]) == {"track_a", "track_b"}


def test_merge_empty_directory_returns_empty_frame_with_columns(tmp_path):
    d = tmp_path / "empty"
    d.mkdir()
    df, skipped = merge_interviews(str(d))
    assert len(df) == 0
    assert list(df.columns) == list(COLUMNS)
    assert skipped == []


def test_merge_ignores_non_json_files(tmp_path):
    d = tmp_path / "mixed"
    d.mkdir()
    _write_json(d / "track_a.json", _valid_dict())
    (d / "README.txt").write_text("not an interview", encoding="utf-8")
    df, skipped = merge_interviews(str(d))
    assert len(df) == 1
    assert skipped == []
