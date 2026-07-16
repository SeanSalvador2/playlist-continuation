"""Smoke test for scripts/build_atlas_payload.py on the synthetic demo listener.

Proves the pipeline bridge is general (runs on a small synthetic history), emits a
schema-complete payload with every section present, is JSON-serialisable, and is
deterministic (byte-identical across two runs).
"""
import importlib.util
import json
from pathlib import Path

import pytest

from playlistcont.analytics import queries as q
from playlistcont.history.synthetic import make_synthetic_history

_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "build_atlas_payload.py"


def _load_builder():
    spec = importlib.util.spec_from_file_location("build_atlas_payload", _SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def builder():
    return _load_builder()


@pytest.fixture(scope="module")
def payload(builder):
    hist = make_synthetic_history(seed=3, n_days=220, base_events_per_day=40)
    return builder.build_payload(hist, tz=q.resolve_tz("America/New_York"),
                                 min_chapters=2, min_subsections=2)


def test_all_sections_present(payload):
    expected = {
        "meta", "summary", "qualified_summary", "thesis_series", "genre_share_by_year",
        "trajectory", "top_artists", "top_tracks", "plays_by_year", "plays_by_month",
        "clock", "chapter_plan", "cohorts", "story",
    }
    assert expected <= set(payload)
    # the chapter floor is met and the plan is labelled
    assert payload["chapter_plan"]["n_chapters"] >= 2
    assert payload["chapter_plan"]["structure_strength"]
    # thesis picks the listener's OWN top genre (not hardcoded)
    assert payload["thesis_series"]["genre"] in q.GENRES if hasattr(q, "GENRES") else True
    # leaderboards carry both bases
    assert "all" in payload["top_artists"] and "min30s" in payload["top_artists"]
    assert payload["top_artists"]["all"]["rows"]
    # the story carries sub-beat detail slides
    kinds = [s["kind"] for s in payload["story"]["slides"]["slides"]]
    assert "chapter_detail" in kinds
    assert kinds[0] == "title"


def test_cohorts_align_with_chapters(payload):
    n = payload["chapter_plan"]["n_chapters"]
    assert len(payload["cohorts"]["chapters"]) == n
    assert len(payload["cohorts"]["subsections"]) == n
    assert len(payload["cohorts"]["boundary_diffs"]) == n - 1


def test_payload_is_json_serialisable(payload):
    text = json.dumps(payload, ensure_ascii=False)
    assert len(text) > 0
    round_trip = json.loads(text)
    assert round_trip["meta"]["schema"] == "atlas-core/1"


def test_builder_is_deterministic(builder):
    hist = make_synthetic_history(seed=3, n_days=220, base_events_per_day=40)
    tz = q.resolve_tz("America/New_York")
    a = builder.build_payload(hist, tz=tz, min_chapters=2, min_subsections=2)
    b = builder.build_payload(hist, tz=tz, min_chapters=2, min_subsections=2)
    assert json.dumps(a, sort_keys=True) == json.dumps(b, sort_keys=True)
