"""Taste Atlas — FastAPI application.

Wraps the interpretable Taste Engine (plus item-CF and popularity, fitted on a
seeded synthetic MPD at startup) as a local web service, and serves the built
React dashboard from a single port (``playlistcont dashboard``).

Endpoints (all under ``/api``):

    GET  /api/health
    GET  /api/config                 axes, genres, sample seed playlists, catalog
    POST /api/profile                stated + seed + trust -> learned/blended weights, clusters
    POST /api/recommend              stated + seed + trust -> ranked recs with explanations
    GET  /api/personas               pre-built persona list
    GET  /api/personas/{id}          persona detail (weights, recs, fit meter)
    GET  /api/personas/{id}/curve    fit-vs-trust curve (the adversarial rescue)
    GET  /api/results/overview       synthetic vs real OVERALL comparison
    GET  /api/results/scenarios      per-scenario metric heatmap
    GET  /api/results/held           what-held / what-didn't table
    GET  /api/results/trust          taste-engine trust ablation curves
    GET  /api/results/taste-real     taste engine on the real matched subset
    GET  /api/history/journey        Phase 4: trajectory + named eras + fact-checked story
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import PlainTextResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from playlistcont.analytics import queries as Q
from playlistcont.analytics import text2sql as T2S

from . import results as R
from .engine import get_atlas
from .history_engine import get_history_atlas

FRONTEND_DIST = Path(__file__).resolve().parents[1] / "frontend" / "dist"


class ProfileRequest(BaseModel):
    scalars: Dict[str, float] = Field(default_factory=dict)
    genres: List[str] = Field(default_factory=list)
    seed_uris: List[str] = Field(default_factory=list)
    trust: float = 0.4


class RecommendRequest(ProfileRequest):
    k: int = 24


class AskTemplateRequest(BaseModel):
    template_id: str
    slots: Dict[str, object] = Field(default_factory=dict)


class AskSqlRequest(BaseModel):
    sql: str


def create_app(warm: bool = True) -> FastAPI:
    app = FastAPI(title="Taste Atlas", version="1.0.0")
    app.add_middleware(
        CORSMiddleware, allow_origins=["*"],
        allow_methods=["*"], allow_headers=["*"],
    )

    if warm:
        get_atlas()          # build the synthetic corpus + fit models up front
        get_history_atlas()  # build the listening history + DuckDB store up front

    # ---- meta ---------------------------------------------------------- #
    @app.get("/api/health")
    def health() -> dict:
        return {"status": "ok"}

    @app.get("/api/config")
    def config() -> dict:
        a = get_atlas()
        return {
            "axes": a.axes_config(),
            "sample_playlists": a.sample_playlists(),
            "personas": a.personas(),
            "catalog": a.catalog_scatter,
        }

    # ---- taste lab ----------------------------------------------------- #
    @app.post("/api/profile")
    def profile(req: ProfileRequest) -> dict:
        return get_atlas().profile(
            scalars=req.scalars, genres=req.genres,
            seed_uris=req.seed_uris, trust=req.trust,
        )

    @app.post("/api/recommend")
    def recommend(req: RecommendRequest) -> dict:
        k = max(1, min(60, int(req.k)))
        return get_atlas().recommend(
            scalars=req.scalars, genres=req.genres,
            seed_uris=req.seed_uris, trust=req.trust, k=k,
        )

    # ---- personas ------------------------------------------------------ #
    @app.get("/api/personas")
    def personas() -> dict:
        return {"personas": get_atlas().personas()}

    @app.get("/api/personas/{persona_id}")
    def persona(persona_id: str,
                trust: Optional[float] = None,
                adversarial: bool = False) -> dict:
        try:
            return get_atlas().persona_detail(
                persona_id, trust=trust, adversarial=adversarial)
        except KeyError:
            raise HTTPException(status_code=404, detail="unknown persona")

    @app.get("/api/personas/{persona_id}/curve")
    def persona_curve(persona_id: str, adversarial: bool = True) -> dict:
        try:
            return get_atlas().persona_trust_curve(persona_id, adversarial=adversarial)
        except KeyError:
            raise HTTPException(status_code=404, detail="unknown persona")

    # ---- results explorer ---------------------------------------------- #
    @app.get("/api/results/overview")
    def results_overview() -> dict:
        return R.overview()

    @app.get("/api/results/scenarios")
    def results_scenarios(
        dataset: str = Query("real", pattern="^(real|synthetic)$"),
        metric: str = Query("r_precision", pattern="^(r_precision|ndcg|clicks)$"),
    ) -> dict:
        return R.scenarios(dataset=dataset, metric=metric)

    @app.get("/api/results/held")
    def results_held() -> dict:
        return {"rows": R.held()}

    @app.get("/api/results/trust")
    def results_trust() -> dict:
        return R.trust_curves()

    @app.get("/api/results/taste-real")
    def results_taste_real() -> dict:
        return {"rows": R.taste_real_subset()}

    # ---- library (personal listening analytics) ------------------------ #
    @app.get("/api/history/summary")
    def history_summary(start: Optional[str] = None, end: Optional[str] = None) -> dict:
        return get_history_atlas().summary(start=start, end=end)

    @app.get("/api/history/top")
    def history_top(
        entity: str = Query("tracks", pattern="^(tracks|artists|albums)$"),
        by: str = Query("plays", pattern="^(plays|minutes)$"),
        start: Optional[str] = None,
        end: Optional[str] = None,
        limit: int = 0,
        offset: int = 0,
        format: str = Query("json", pattern="^(json|csv)$"),
    ):
        payload = get_history_atlas().top_items(
            entity=entity, by=by, start=start, end=end,
            limit=limit or None, offset=offset,
        )
        if format == "csv":
            return PlainTextResponse(Q.top_items_to_csv(payload), media_type="text/csv")
        return payload

    @app.get("/api/history/trends")
    def history_trends(
        metric: str = Query("plays", pattern="^(plays|minutes|discovery|skip_rate)$"),
        granularity: str = Query("week", pattern="^(day|week|month)$"),
        start: Optional[str] = None,
        end: Optional[str] = None,
        rolling: Optional[int] = None,
        format: str = Query("json", pattern="^(json|csv)$"),
    ):
        payload = get_history_atlas().trends(
            metric=metric, granularity=granularity, start=start, end=end, rolling=rolling,
        )
        if format == "csv":
            return PlainTextResponse(Q.trends_to_csv(payload), media_type="text/csv")
        return payload

    @app.get("/api/history/clock")
    def history_clock(start: Optional[str] = None, end: Optional[str] = None) -> dict:
        return get_history_atlas().listening_clock(start=start, end=end)

    @app.get("/api/history/axes")
    def history_axes(
        granularity: str = Query("week", pattern="^(day|week|month)$"),
        start: Optional[str] = None,
        end: Optional[str] = None,
    ) -> dict:
        return get_history_atlas().axes_over_time(granularity=granularity, start=start, end=end)

    @app.get("/api/history/genres")
    def history_genres(start: Optional[str] = None, end: Optional[str] = None) -> dict:
        return get_history_atlas().genres(start=start, end=end)

    @app.get("/api/history/shifts")
    def history_shifts(
        mode: str = Query("auto", pattern="^(auto|explicit)$"),
        start: Optional[str] = None,
        end: Optional[str] = None,
        a_start: Optional[str] = None,
        a_end: Optional[str] = None,
        b_start: Optional[str] = None,
        b_end: Optional[str] = None,
    ) -> dict:
        return get_history_atlas().shifts(
            mode=mode, start=start, end=end,
            a_start=a_start, a_end=a_end, b_start=b_start, b_end=b_end,
        )

    @app.get("/api/history/habits")
    def history_habits(
        group_by: str = Query("weekday", pattern="^(weekday|hour_band|month)$"),
        start: Optional[str] = None,
        end: Optional[str] = None,
    ) -> dict:
        return get_history_atlas().habits(group_by=group_by, start=start, end=end)

    # ---- journey (Phase 4: trajectory + named eras + fact-checked story) ---- #
    @app.get("/api/history/journey")
    def history_journey() -> dict:
        return get_history_atlas().journey()

    # ---- ask your library (Phase 5: template library + guarded free-form SQL) ---- #
    @app.get("/api/history/ask/templates")
    def history_ask_templates() -> dict:
        return get_history_atlas().ask_templates()

    @app.post("/api/history/ask/run")
    def history_ask_run(req: AskTemplateRequest) -> dict:
        try:
            return get_history_atlas().ask_run_template(req.template_id, req.slots)
        except T2S.SlotError as exc:
            raise HTTPException(status_code=400, detail=str(exc))
        except T2S.GuardrailError as exc:  # a generated template SQL should never trip this
            raise HTTPException(status_code=400, detail=str(exc))

    @app.post("/api/history/ask/sql")
    def history_ask_sql(req: AskSqlRequest) -> dict:
        """Guarded raw-SQL execution for power users.

        Returns a structured result: ``{"ok": true, ...}`` with the normalized SQL and
        rows, or ``{"ok": false, "error": ...}`` when the guardrail rejects or the query
        fails — so the UI can show the message inline rather than treating it as a crash.
        """
        try:
            result = get_history_atlas().ask_sql(req.sql)
            result["ok"] = True
            return result
        except T2S.GuardrailError as exc:
            return {"ok": False, "error": str(exc), "sql": req.sql}

    # ---- static frontend (mounted last so /api wins) ------------------- #
    if FRONTEND_DIST.exists():
        app.mount("/", StaticFiles(directory=str(FRONTEND_DIST), html=True), name="app")

    return app


app = create_app()
