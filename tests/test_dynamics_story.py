"""Hand-verified tests for the Phase-4 taste-journey package.

Same philosophy as ``tests/test_dynamics.py``: every number is checked against
something computed independently — by hand for the tiny fixtures, or against a
committed golden file for the narrative.  All offline, all deterministic.

Coverage map:

* trajectory — determinism, the documented sign convention, human-readable
  loadings, and a hand-built two-cluster fixture that must separate along PC1;
* eras — planted boundaries/durations when fed the planted dates directly, name
  stability across two builds, exemplars carry features, and the provisional flag
  firing exactly for a final-December opening;
* story / fact-check — the harness catching each violation class (tampered number,
  renamed proper noun, orphan numeral, bad fact_path), a golden-file render, the
  real seed-7 story surviving its own audit end-to-end, and the polish hook
  (reword passes, number-change falls back to template).
"""
import copy
import json
from datetime import date, datetime, timezone
from pathlib import Path

import numpy as np
import pytest

from playlistcont.data.schema import GENRES, SCALAR_AXES
from playlistcont.dynamics import (
    DetectedChange, build_eras, build_windows, compute_trajectory,
    gather_facts, human_label, recommended_detector, render_story, verify_story,
)
from playlistcont.dynamics.chapters import adaptive_chapters
from playlistcont.dynamics.cohorts import CohortModel, cohorts_for_spans
from playlistcont.dynamics.story import StoryVerificationError
from playlistcont.dynamics.windows import WindowSeries
from playlistcont.history.schema import RegimeSpec
from playlistcont.history.store import HistoryStore, feature_column
from playlistcont.history.synthetic import make_synthetic_history

UTC = timezone.utc
SCALAR_COLUMNS = [feature_column(a) for a in SCALAR_AXES]
FIXTURES = Path(__file__).parent / "fixtures"


# ===========================================================================
# fixtures
# ===========================================================================
def _two_regime_history(seed: int = 1):
    """A pinned 2-regime history: pure rock -> pure rap, abrupt on 2022-04-01."""
    regimes = [
        RegimeSpec(date(2022, 1, 1), date(2022, 4, 1), {"classic rock": 1.0}, 40, "r0"),
        RegimeSpec(date(2022, 4, 1), date(2022, 7, 1), {"gym rap": 1.0}, 40, "r1"),
    ]
    return make_synthetic_history(
        seed=seed, regimes=regimes, n_days=181, start_date=date(2022, 1, 1),
        include_traps=False, seasonal=False)


def _combined_series(matrix: np.ndarray, n_genre: int = 2, n_flavor: int = 2) -> WindowSeries:
    """Wrap a raw matrix as a WindowSeries with real scalar/genre/flavor column groups.

    The first 5 columns are the scalar axes, the next ``n_genre`` are genre shares,
    the last ``n_flavor`` are flavor shares — so ``representation_columns('combined')``
    resolves and the human labels map back to real names.
    """
    n, d = matrix.shape
    assert d == 5 + n_genre + n_flavor
    scal = list(SCALAR_COLUMNS)
    gen = [feature_column(f"genre:{g}") for g in GENRES[:n_genre]]
    flav = [f"flavor_{i}" for i in range(n_flavor)]
    cols = scal + gen + flav
    from datetime import timedelta
    starts = [date(2022, 1, 3) + timedelta(days=7 * i) for i in range(n)]
    ends = [s + timedelta(days=7) for s in starts]
    return WindowSeries(
        starts=starts, ends=ends, matrix=matrix.astype(float), columns=cols,
        column_groups={"scalar_axes": scal, "genre_shares": gen, "flavor_shares": flav},
        mask=np.ones(n, dtype=bool), event_counts=np.full(n, 300), granularity="week",
        weighting="plays", min_events=1, flavor_model=None,
        span_days=(ends[-1] - starts[0]).days)


# ===========================================================================
# 1. trajectory
# ===========================================================================
def test_trajectory_deterministic_across_two_calls():
    ws = build_windows(_two_regime_history(), granularity="week", min_events=30)
    a = compute_trajectory(ws)
    b = compute_trajectory(ws)
    assert a.to_payload() == b.to_payload()
    assert [c.caption for c in a.components] == [c.caption for c in b.components]


def _two_cluster_matrix() -> np.ndarray:
    """12 windows: first 6 low, last 6 high on the FIRST scalar column (tempo) only."""
    m = np.zeros((12, 9))
    m[:6, 0] = 0.1
    m[6:, 0] = 0.9
    return m


def test_trajectory_sign_convention_leading_loading_positive():
    tr = compute_trajectory(_combined_series(_two_cluster_matrix()), n_components=2)
    pc1 = tr.components[0]
    lead = pc1.loadings[0]                              # loadings are sorted by |loading| desc
    # sign convention: the largest-|loading| column is oriented positive
    assert lead["column"] == "tempo"
    assert lead["loading"] > 0
    assert abs(lead["loading"]) == max(abs(d["loading"]) for d in pc1.loadings)


def test_trajectory_two_clusters_separate_along_pc1():
    tr = compute_trajectory(_combined_series(_two_cluster_matrix()))
    xs = np.array([p["coords"][0] for p in tr.points])
    # the two hand-built clusters must land on opposite sides of PC1, well apart
    assert xs[:6].mean() * xs[6:].mean() < 0
    assert abs(xs[:6].mean() - xs[6:].mean()) > 1.5
    # a pure one-axis separation is entirely explained by PC1
    assert tr.components[0].explained_variance_ratio > 0.99


def test_trajectory_loadings_map_to_human_column_names():
    m = np.zeros((8, 9)); m[:, 0] = np.linspace(0, 1, 8)
    tr = compute_trajectory(_combined_series(m))
    labels = {d["label"] for c in tr.components for d in c.loadings}
    assert "tempo" in labels and "energy" in labels
    assert any(lbl.endswith("share") for lbl in labels)     # genre share label
    # every loading column is a real column of the series, every label is non-empty
    real_cols = set(tr.columns)
    assert all(d["label"] for c in tr.components for d in c.loadings)
    assert human_label("genre_metal") == "metal share"
    assert human_label("lyrical_depth") == "lyrical depth"


# ===========================================================================
# 2. eras
# ===========================================================================
def test_eras_boundaries_and_durations_from_planted_dates():
    h = _two_regime_history()
    det = [DetectedChange(date(2022, 4, 1), 1.0, "planted")]
    eras = build_eras(h, det)
    assert len(eras) == 2
    assert eras[0].start == date(2022, 1, 1) and eras[0].end == date(2022, 3, 31)
    assert eras[1].start == date(2022, 4, 1)
    assert eras[0].duration_days == 90        # Jan 1 .. Mar 31 inclusive
    assert eras[1].duration_days == 91        # Apr 1 .. Jun 30 inclusive
    # ground-truth-blind cut lands exactly where we fed the change
    assert eras[1].opening_change == "2022-04-01"


def test_era_names_stable_across_two_builds():
    h = _two_regime_history()
    det = [DetectedChange(date(2022, 4, 1), 1.0, "planted")]
    a = build_eras(h, det)
    b = build_eras(h, det)
    assert [e.name for e in a] == [e.name for e in b]
    assert [e.flavor_id for e in a] == [e.flavor_id for e in b]


def test_era_exemplars_have_features():
    h = _two_regime_history()
    det = [DetectedChange(date(2022, 4, 1), 1.0, "planted")]
    store = HistoryStore.from_history(h)
    feat_uris = set(store.query("SELECT uri FROM track_features")["uri"])
    name_to_uri = {}
    for _, r in store.query("SELECT uri, track_name FROM tracks").iterrows():
        name_to_uri[str(r["track_name"])] = str(r["uri"])
    for e in build_eras(store, det):
        assert e.exemplar_tracks                      # non-empty
        for ex in e.exemplar_tracks:
            assert name_to_uri[ex["name"]] in feat_uris


def test_provisional_fires_only_for_final_december_opening():
    # history that ENDS in December, with the opening change inside that final December
    regimes = [
        RegimeSpec(date(2023, 1, 1), date(2023, 12, 1), {"classic rock": 1.0}, 40, "a"),
        RegimeSpec(date(2023, 12, 1), date(2024, 1, 1), {"party pop": 1.0}, 40, "b"),
    ]
    h = make_synthetic_history(seed=2, regimes=regimes, n_days=365,
                               start_date=date(2023, 1, 1),
                               include_traps=False, seasonal=False)
    eras = build_eras(h, [DetectedChange(date(2023, 12, 1), 1.0, "planted")])
    assert eras[0].provisional is False               # opens the history, not a December bump
    assert eras[1].provisional is True                # opens on the final December

    # a mid-history (non-December) change is never provisional
    eras2 = build_eras(_two_regime_history(),
                       [DetectedChange(date(2022, 4, 1), 1.0, "planted")])
    assert all(e.provisional is False for e in eras2)


# ===========================================================================
# 3. story + fact-check harness
# ===========================================================================
def _seed7_story():
    h = make_synthetic_history(seed=7, n_days=730)
    det = recommended_detector().detect(h)
    ws = build_windows(h, granularity="week", min_events=30)
    eras = build_eras(h, det, flavor_model=ws.flavor_model)
    facts = gather_facts(h, eras, det)
    return h, facts, render_story(facts)


def _small_story():
    h = _two_regime_history()
    det = [DetectedChange(date(2022, 4, 1), 1.0, "planted")]
    eras = build_eras(h, det)
    facts = gather_facts(h, eras, det)
    return facts, render_story(facts)


def test_verify_passes_on_real_seed7_story_end_to_end():
    _h, facts, story = _seed7_story()
    assert len(story.slides) == 11
    verify_story(story, facts)                         # must not raise
    # every slide carries at least one claim
    assert all(s.claims for s in story.slides)


def test_harness_catches_tampered_number():
    facts, story = _small_story()
    s = copy.deepcopy(story)
    era = next(sl for sl in s.slides if sl.kind == "era")
    era.body[0] = era.body[0].replace("90", "999")     # falsify the duration in prose
    with pytest.raises(StoryVerificationError):
        verify_story(s, facts)


def test_harness_catches_renamed_proper_noun():
    facts, story = _small_story()
    s = copy.deepcopy(story)
    era = next(sl for sl in s.slides if sl.kind == "era")
    era.body[1] = era.body[1].replace("Rock Artist 4-0", "Fake Band 9-9")
    with pytest.raises(StoryVerificationError):
        verify_story(s, facts)


def test_harness_catches_orphan_numeral():
    facts, story = _small_story()
    s = copy.deepcopy(story)
    s.slides[0].body.append("An extra 42 sneaks in.")   # numeral with no backing claim
    with pytest.raises(StoryVerificationError):
        verify_story(s, facts)


def test_harness_catches_bad_fact_path():
    facts, story = _small_story()
    s = copy.deepcopy(story)
    s.slides[0].claims[0].fact_path = "meta.not_a_real_fact"
    with pytest.raises(StoryVerificationError):
        verify_story(s, facts)


def test_harness_catches_claim_value_not_matching_fact():
    facts, story = _small_story()
    s = copy.deepcopy(story)
    # point a claim at a real but wrong fact: value no longer equals the fact
    s.slides[0].claims[0].value = 123456789
    with pytest.raises(StoryVerificationError):
        verify_story(s, facts)


def test_story_golden_render():
    facts, story = _small_story()
    got = {"slides": [{"kind": s.kind, "title": s.title, "body": s.body}
                      for s in story.slides]}
    golden = json.loads((FIXTURES / "story_golden.json").read_text())
    assert got == golden


def test_polish_reword_passes_reverification():
    facts, _ = _small_story()

    def reword(facts_, slides):
        # change only wording, never numbers/names
        out = []
        for sl in slides:
            out.append({"title": sl.title,
                        "body": [b.replace("Around", "In").replace("Defined by", "Led by")
                                 for b in sl.body]})
        return out

    story = render_story(facts, mode="polished", polish_fn=reword)
    assert story.mode == "polished"
    assert any("In April 2022" in " ".join(s.body) for s in story.slides)


def test_polish_changing_a_number_falls_back_to_template():
    facts, template = _small_story()

    def tamper(facts_, slides):
        return [{"title": sl.title, "body": [b.replace("90 days", "88 days") for b in sl.body]}
                for sl in slides]

    story = render_story(facts, mode="polished", polish_fn=tamper)
    assert story.mode == "template"                    # audit failed -> template shipped
    got = [{"kind": s.kind, "title": s.title, "body": s.body} for s in story.slides]
    want = [{"kind": s.kind, "title": s.title, "body": s.body} for s in template.slides]
    assert got == want


def test_gather_facts_arc_and_boundaries_are_consistent():
    facts, _ = _small_story()
    data = facts.data
    assert data["meta"]["n_eras"] == 2
    assert len(data["boundaries"]) == 1
    # arc's biggest shift really is the max |effect| across boundary shifts
    all_effects = [abs(s["effect"]) for b in data["boundaries"] for s in b["shifts"]]
    assert abs(data["arc"]["biggest_shift"]["effect"]) == max(all_effects)


# ===========================================================================
# 4. sub-chapter narrative (ChapterPlan + cohorts -> sub-beats), additive
# ===========================================================================
def _subbeat_story():
    h = make_synthetic_history(seed=7, n_days=730)
    ws = build_windows(h, granularity="week", min_events=30)
    plan = adaptive_chapters(ws)
    det = plan.as_detections()
    eras = build_eras(h, det, flavor_model=ws.flavor_model)
    model = CohortModel.from_store(h)
    sub_cohorts = [cohorts_for_spans(model, [(s.start, s.end) for s in ch.subsections])
                   for ch in plan.chapters]
    facts = gather_facts(h, eras, det, chapter_plan=plan, sub_cohorts=sub_cohorts)
    return plan, facts, render_story(facts)


def test_sub_beats_render_and_survive_the_fact_check():
    plan, facts, story = _subbeat_story()
    details = [s for s in story.slides if s.kind == "chapter_detail"]
    # one chapter-detail slide per chapter, each backed by claims
    assert len(details) == len(plan.chapters)
    for s in details:
        assert s.claims                                   # ensemble names / cast are audited
    verify_story(story, facts)                            # the whole thing survives its audit


def test_sub_beat_headlines_are_computed_from_structure():
    plan, facts, _ = _subbeat_story()
    for i, ch in enumerate(plan.chapters):
        headline = facts.data["eras"][i]["headline"]
        assert headline == ch.headline()
        n = ch.n_turns
        if n == 0:
            assert headline == "a slow drift — no sharp turn"
        elif n == 1:
            assert headline == "one clear turn"
        else:
            assert headline.endswith("turns")


def test_sub_beats_are_opt_in_and_backward_compatible():
    # without a ChapterPlan, gather_facts adds no sub-beat keys and the slide set
    # is exactly the classic title/era/transition/arc sequence
    facts, story = _small_story()
    assert all("sub_beats" not in e and "headline" not in e for e in facts.data["eras"])
    assert not any(s.kind == "chapter_detail" for s in story.slides)
