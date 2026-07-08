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
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from . import results as R
from .engine import get_atlas

FRONTEND_DIST = Path(__file__).resolve().parents[1] / "frontend" / "dist"


class ProfileRequest(BaseModel):
    scalars: Dict[str, float] = Field(default_factory=dict)
    genres: List[str] = Field(default_factory=list)
    seed_uris: List[str] = Field(default_factory=list)
    trust: float = 0.4


class RecommendRequest(ProfileRequest):
    k: int = 24


def create_app(warm: bool = True) -> FastAPI:
    app = FastAPI(title="Taste Atlas", version="1.0.0")
    app.add_middleware(
        CORSMiddleware, allow_origins=["*"],
        allow_methods=["*"], allow_headers=["*"],
    )

    if warm:
        get_atlas()  # build the synthetic corpus + fit models up front

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

    # ---- static frontend (mounted last so /api wins) ------------------- #
    if FRONTEND_DIST.exists():
        app.mount("/", StaticFiles(directory=str(FRONTEND_DIST), html=True), name="app")

    return app


app = create_app()
