"""Backend tests for the Taste Atlas dashboard API.

The synthetic corpus + models are seeded, so every assertion here is
deterministic. One app instance is built for the whole module (fitting takes a
few seconds).
"""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from app.backend.server import create_app


@pytest.fixture(scope="module")
def client():
    app = create_app(warm=True)
    with TestClient(app) as c:
        yield c


# ---- meta / config -------------------------------------------------------- #
def test_health(client):
    assert client.get("/api/health").json()["status"] == "ok"


def test_config_shape(client):
    cfg = client.get("/api/config").json()
    assert [a["key"] for a in cfg["axes"]["scalar"]] == [
        "tempo", "energy", "valence", "acousticness", "lyrical_depth"]
    assert len(cfg["axes"]["genres"]) == 10
    assert len(cfg["personas"]) >= 3
    assert len(cfg["sample_playlists"]) >= 1
    assert len(cfg["catalog"]) > 100
    # sample playlists carry real seed tracks
    assert all(p["seed_uris"] for p in cfg["sample_playlists"])


# ---- profile -------------------------------------------------------------- #
def test_profile_weights_and_blend(client):
    cfg = client.get("/api/config").json()
    seed = cfg["sample_playlists"][0]["seed_uris"]
    body = {"scalars": {"valence": -0.8}, "genres": ["country"],
            "seed_uris": seed, "trust": 0.5}
    prof = client.post("/api/profile", json=body).json()
    assert prof["has_stated"] is True
    assert prof["n_seed_tracks"] == len(seed)
    # three weight vectors over all 15 axes
    for key in ("stated", "learned", "blended"):
        assert len(prof["weights"][key]) == 15
    # a stated country/valence preference shows up in the stated vector
    stated = {w["axis"]: w["value"] for w in prof["weights"]["stated"]}
    assert stated["valence"] < 0
    assert stated["genre:country"] > 0
    # clusters are named and sum shares to ~1
    assert prof["clusters"]
    assert abs(sum(c["share"] for c in prof["clusters"]) - 1.0) < 1e-6
    assert all(isinstance(c["name"], str) and c["name"] for c in prof["clusters"])


def test_trust_blends_between_stated_and_learned(client):
    cfg = client.get("/api/config").json()
    seed = cfg["sample_playlists"][0]["seed_uris"]
    base = {"scalars": {"valence": -0.9}, "genres": ["country"], "seed_uris": seed}
    lo = client.post("/api/profile", json={**base, "trust": 0.0}).json()
    hi = client.post("/api/profile", json={**base, "trust": 1.0}).json()
    # at trust 0 the blended vector equals the learned vector; at trust 1 the stated
    lo_blend = {w["axis"]: w["value"] for w in lo["weights"]["blended"]}
    lo_learn = {w["axis"]: w["value"] for w in lo["weights"]["learned"]}
    hi_blend = {w["axis"]: w["value"] for w in hi["weights"]["blended"]}
    hi_stated = {w["axis"]: w["value"] for w in hi["weights"]["stated"]}
    assert all(abs(lo_blend[a] - lo_learn[a]) < 1e-3 for a in lo_blend)
    assert all(abs(hi_blend[a] - hi_stated[a]) < 1e-3 for a in hi_blend)


# ---- recommend + explanations --------------------------------------------- #
def test_recommend_items_and_explanations(client):
    cfg = client.get("/api/config").json()
    seed = cfg["sample_playlists"][0]["seed_uris"]
    body = {"scalars": {}, "genres": [], "seed_uris": seed, "trust": 0.4, "k": 20}
    rec = client.post("/api/recommend", json=body).json()
    items = rec["items"]
    assert len(items) == 20
    # ranks are 1..k, scores non-increasing, seeds excluded
    assert [it["rank"] for it in items] == list(range(1, 21))
    scores = [it["score"] for it in items]
    assert scores == sorted(scores, reverse=True)
    assert not (set(it["uri"] for it in items) & set(seed))
    # every item carries an explanation
    for it in items:
        assert it["flavor"]
        assert it["reasons"]
        assert isinstance(it["axis_bars"], list)
        assert it["cooccurrence"] >= 0
    # at least one item has co-occurrence evidence with the seeds
    assert any(it["cooccurrence"] > 0 for it in items)


def test_recommend_is_deterministic(client):
    body = {"scalars": {"energy": 0.7}, "genres": ["rap"], "seed_uris": [],
            "trust": 0.6, "k": 15}
    a = client.post("/api/recommend", json=body).json()
    b = client.post("/api/recommend", json=body).json()
    assert [x["uri"] for x in a["items"]] == [x["uri"] for x in b["items"]]


# ---- personas: the adversarial rescue ------------------------------------- #
def test_persona_detail(client):
    d = client.get("/api/personas/gym_rap").json()
    assert d["archetype"] == "gym rap"
    assert d["recommendations"]["items"]
    assert 0.0 <= d["fit"] <= 1.0


def test_adversarial_rescue(client):
    """Wrong stated prefs at high trust collapse the fit; low trust rescues it."""
    hi = client.get("/api/personas/sad_country?trust=0.9&adversarial=true").json()
    lo = client.get("/api/personas/sad_country?trust=0.1&adversarial=true").json()
    assert lo["fit"] > hi["fit"]
    assert lo["fit"] >= 0.8   # trusting the tracks keeps recs on-taste
    assert hi["fit"] <= 0.2   # trusting the wrong prefs derails them


def test_trust_curve_monotone_collapse(client):
    curve = client.get("/api/personas/sad_country/curve?adversarial=true").json()["curve"]
    fits = [c["fit"] for c in curve]
    # first point (all learned) beats last point (all wrong-stated)
    assert fits[0] > fits[-1]


def test_unknown_persona_404(client):
    assert client.get("/api/personas/nope").status_code == 404


# ---- results explorer ----------------------------------------------------- #
def test_results_overview_real_winner(client):
    ov = client.get("/api/results/overview").json()
    real = ov["real"]
    # item-CF wins the real MPD overall
    top = max(real, key=lambda r: r["r_precision"])
    assert top["model"] == "item_cf"
    assert abs(top["r_precision"] - 0.148) < 0.002


def test_results_scenarios_grid(client):
    grid = client.get("/api/results/scenarios?dataset=real&metric=r_precision").json()
    assert len(grid["scenarios"]) == 10
    assert len(grid["matrix"]) == len(grid["models"])
    assert all(len(row) == 10 for row in grid["matrix"])


def test_results_held_table(client):
    rows = client.get("/api/results/held").json()["rows"]
    assert len(rows) == 9
    verdicts = {r["verdict"] for r in rows}
    assert verdicts <= {"held", "didn't hold", "partial"}


def test_results_trust_ablation(client):
    tc = client.get("/api/results/trust").json()
    assert len(tc["honest"]) == 5 and len(tc["adversarial"]) == 5
    # honest is roughly flat; adversarial collapses as trust rises
    assert tc["adversarial"][0]["r_precision"] > tc["adversarial"][-1]["r_precision"]


# ---- journey (Phase 4: trajectory + named eras + fact-checked story) ------- #
def test_journey_shape_and_consistency(client):
    j = client.get("/api/history/journey").json()
    assert "error" not in j
    assert j["is_synthetic"] is True
    # detector is the benchmark winner
    assert j["detector"]["method"] == "pelt_rbf"
    # eras = detections + 1; each named
    assert len(j["eras"]) == len(j["detections"]) + 1
    assert all(e["name"] for e in j["eras"])
    # trajectory: one point per measurable weekly window, 2 captioned components
    tr = j["trajectory"]
    assert tr["n_windows"] == len(tr["points"]) > 10
    assert len(tr["components"]) == 2 and all(c["caption"] for c in tr["components"])
    # story: ordered slides, each carrying at least one claim
    slides = j["story"]["slides"]
    assert slides[0]["kind"] == "title" and slides[-1]["kind"] == "arc"
    assert all(s["claims"] for s in slides)
    # planted landmarks are exposed as demo diagnostics
    assert j["planted"] and all("date" in p for p in j["planted"])


def test_journey_is_cached_singleton(client):
    a = client.get("/api/history/journey").json()
    b = client.get("/api/history/journey").json()
    assert a["detections"] == b["detections"]
    assert [s["title"] for s in a["story"]["slides"]] == [s["title"] for s in b["story"]["slides"]]


# ---- ask your library (Phase 5: template library + guarded free-form SQL) -- #
def test_ask_templates_lists_typed_slots(client):
    payload = client.get("/api/history/ask/templates").json()
    templates = payload["templates"]
    assert len(templates) >= 15
    ids = {t["id"] for t in templates}
    assert "most_skipped_artists" in ids and "one_hit_wonders" in ids
    # the schema card is shipped so the UI can show the "what can I ask" reference
    assert "TABLE events" in payload["schema_card"]
    # every template exposes its default SQL and typed slots
    for t in templates:
        assert t["default_sql"].strip()
        for s in t["slots"]:
            assert s["type"] in ("int", "date", "enum")


def test_ask_run_template_returns_rows_and_sql(client):
    body = {"template_id": "most_skipped_artists", "slots": {"min_plays": 20, "limit": 5}}
    r = client.post("/api/history/ask/run", json=body).json()
    assert r["template_id"] == "most_skipped_artists"
    assert r["row_count"] <= 5
    assert r["columns"][0] == "artist"
    assert "LIMIT" in r["sql"]                     # the executed SQL is always shown
    assert r["slots"]["min_plays"] == 20


def test_ask_run_template_rejects_bad_slot(client):
    r = client.post("/api/history/ask/run",
                    json={"template_id": "most_skipped_artists", "slots": {"min_plays": "oops"}})
    assert r.status_code == 400


def test_ask_sql_runs_guarded_select(client):
    r = client.post("/api/history/ask/sql",
                    json={"sql": "SELECT COUNT(*) AS n FROM events"}).json()
    assert r["ok"] is True
    assert r["rows"][0]["n"] > 0
    assert "LIMIT" in r["sql"]                     # normalized SQL carries the row cap


def test_ask_sql_blocks_writes_with_structured_error(client):
    for attack in ["DROP TABLE events", "INSERT INTO events VALUES (1)",
                   "SELECT 1; DROP TABLE events", "PRAGMA database_list"]:
        r = client.post("/api/history/ask/sql", json={"sql": attack}).json()
        assert r["ok"] is False
        assert r["error"]
    # the store is untouched after the attacks
    ok = client.post("/api/history/ask/sql",
                     json={"sql": "SELECT COUNT(*) AS n FROM events"}).json()
    assert ok["ok"] is True and ok["rows"][0]["n"] > 0
