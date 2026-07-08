"""Submission writer + validator."""
import os

from playlistcont.challenge.submission import (
    validate_submission, write_submission,
)


def _preds():
    return {
        0: ["spotify:track:x0", "spotify:track:x1"],
        1: ["spotify:track:x2"],
    }


def test_write_and_validate_ok(tmp_path):
    p = str(tmp_path / "sub.csv")
    write_submission(p, _preds(), "my team", "me@example.com")
    res = validate_submission(p)
    assert res.ok, res.errors


def test_first_line_is_team_info(tmp_path):
    p = str(tmp_path / "sub.csv")
    write_submission(p, _preds(), "my team", "me@example.com")
    with open(p) as fh:
        first = fh.readline().strip()
    assert first.startswith("team_info,my team,me@example.com")


def test_detects_seed_leak(tmp_path):
    p = str(tmp_path / "sub.csv")
    write_submission(p, {0: ["spotify:track:x0", "spotify:track:seed"]},
                     "t", "t@e.com")
    res = validate_submission(p, seeds={0: ["spotify:track:seed"]})
    assert not res.ok
    assert any("leaks seed" in e for e in res.errors)


def test_detects_too_many(tmp_path):
    p = str(tmp_path / "sub.csv")
    write_submission(p, {0: [f"spotify:track:x{i}" for i in range(600)]},
                     "t", "t@e.com")
    # writer truncates to 500, so this should be valid
    res = validate_submission(p)
    assert res.ok, res.errors


def test_detects_bad_uri(tmp_path):
    p = str(tmp_path / "sub.csv")
    with open(p, "w") as fh:
        fh.write("team_info,t,t@e.com\n\n0,not_a_uri\n")
    res = validate_submission(p)
    assert not res.ok
    assert any("bad track_uri" in e for e in res.errors)


def test_missing_team_info(tmp_path):
    p = str(tmp_path / "sub.csv")
    with open(p, "w") as fh:
        fh.write("0,spotify:track:x0\n")
    res = validate_submission(p)
    assert not res.ok
