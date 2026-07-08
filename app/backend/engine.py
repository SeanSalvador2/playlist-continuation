"""Taste Atlas — the in-memory recommendation engine behind the dashboard.

On first use this builds a small, seeded synthetic MPD (a few seconds), fits the
interpretable :class:`TasteEngine`, item-CF and popularity models, and caches
everything.  The web layer (``server.py``) is a thin translation of the methods
here into JSON — so the whole app is self-contained with zero data downloads and
stays deterministic for the tests.

The vocabulary is cartographic on purpose: interpretable axes are *coordinates*,
flavor clusters are named *territories*, and the trust dial moves between the map
the listener *drew* (stated) and the map *surveyed* from their tracks (learned).
"""
from __future__ import annotations

import functools
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import numpy as np

from playlistcont.data.schema import AXES, GENRES, SCALAR_AXES, N_AXES
from playlistcont.data.synthetic import ARCHETYPES, make_synthetic
from playlistcont.models.itemcf import ItemCFRecommender
from playlistcont.models.popularity import PopularityRecommender
from playlistcont.models.taste_engine import TasteEngine, parse_stated, name_flavor, _l2

# ---- build parameters (kept small so startup is seconds, not minutes) ------- #
N_PLAYLISTS = 4000
N_TRACKS = 2500
SEED = 7

# genre -> archetype name (each synthetic archetype owns one genre, so the
# dominant-genre axis recovers a track's home archetype exactly).
_GENRE_TO_ARCH = {a.genre: a.name for a in ARCHETYPES}

# Human labels + poles for the five scalar mood axes (used by the sliders).
AXIS_META = {
    "tempo": {"label": "Tempo", "low": "slow", "high": "fast"},
    "energy": {"label": "Energy", "low": "calm", "high": "intense"},
    "valence": {"label": "Valence", "low": "sad", "high": "happy"},
    "acousticness": {"label": "Acousticness", "low": "electronic", "high": "acoustic"},
    "lyrical_depth": {"label": "Lyrical depth", "low": "filler", "high": "deep"},
}


@dataclass
class Persona:
    id: str
    name: str
    blurb: str
    archetype: str          # true home archetype (ground truth)
    seed_uris: List[str]
    stated: Dict[str, float]           # the honest stated preference (axis dict)
    stated_wrong: Dict[str, float]     # deliberately wrong stated (adversarial)
    trust: float = 0.4
    adversarial: bool = False


def _round(v: float, n: int = 4) -> float:
    return float(np.round(float(v), n))


class TasteAtlas:
    """Everything the dashboard needs, fitted once and cached."""

    def __init__(self, n_playlists: int = N_PLAYLISTS, n_tracks: int = N_TRACKS,
                 seed: int = SEED):
        self.seed = seed
        self.dataset = make_synthetic(n_playlists=n_playlists, n_tracks=n_tracks, seed=seed)
        self.taste = TasteEngine(seed=seed).fit(self.dataset)
        self.itemcf = ItemCFRecommender(normalization="cosine", topk_sim=200).fit(self.dataset)
        self.pop = PopularityRecommender().fit(self.dataset)

        idx = self.taste.index
        self.track_uris = idx.track_uris
        self.X = self.taste.X                     # raw axes (n_tracks, N_AXES)
        self.pop_norm = idx.track_pop / (idx.track_pop.max() + 1e-9)

        # per-track dominant genre / home archetype (from the genre axis block)
        genre_block = self.X[:, len(SCALAR_AXES):]
        self._dom_genre_id = np.argmax(genre_block, axis=1)
        self._home_arch = np.array(
            [_GENRE_TO_ARCH[GENRES[g]] for g in self._dom_genre_id]
        )
        self.meta = self.dataset.tracks
        self._rng = np.random.default_rng(seed)

        self._personas = self._build_personas()
        self._sample_playlists = self._build_sample_playlists()

    # ------------------------------------------------------------------ #
    #  static reference payloads
    # ------------------------------------------------------------------ #
    def axes_config(self) -> dict:
        return {
            "scalar": [
                {"key": a, **AXIS_META[a]} for a in SCALAR_AXES
            ],
            "genres": list(GENRES),
        }

    @functools.cached_property
    def catalog_scatter(self) -> List[dict]:
        """A downsampled catalogue for the Taste Map background terrain."""
        n = min(600, len(self.track_uris))
        # popularity-biased sample so the terrain reflects real density
        p = self.pop_norm + 1e-3
        p = p / p.sum()
        pick = self._rng.choice(len(self.track_uris), size=n, replace=False, p=p)
        out = []
        for i in pick:
            x = self.X[int(i)]
            out.append({
                "tempo": _round(x[0], 3), "energy": _round(x[1], 3),
                "valence": _round(x[2], 3), "acousticness": _round(x[3], 3),
                "lyrical_depth": _round(x[4], 3),
                "genre": GENRES[int(self._dom_genre_id[int(i)])],
                "pop": _round(self.pop_norm[int(i)], 3),
            })
        return out

    # ------------------------------------------------------------------ #
    #  helpers
    # ------------------------------------------------------------------ #
    def _track_card(self, uri: str) -> dict:
        cid = self.taste.index.uri_to_id[uri]
        x = self.X[cid]
        m = self.meta[uri]
        return {
            "uri": uri,
            "track_name": m.track_name,
            "artist_name": m.artist_name,
            "genre": GENRES[int(self._dom_genre_id[cid])],
            "archetype": self._home_arch[cid],
            "axes": {a: _round(x[j], 3) for j, a in enumerate(SCALAR_AXES)},
            "pop": _round(self.pop_norm[cid], 3),
        }

    def _stated_to_axis_dict(self, scalars: Optional[Dict[str, float]],
                             genres: Optional[Sequence[str]]) -> Dict[str, float]:
        """Turn slider payload into the axis dict ``parse_stated`` understands."""
        spec: Dict[str, float] = {}
        for a in SCALAR_AXES:
            v = float((scalars or {}).get(a, 0.0))
            if abs(v) > 1e-9:
                spec[a] = v
        for g in (genres or []):
            if g in GENRES:
                spec[f"genre:{g}"] = 1.0
        return spec

    def _weights(self, stated_spec: Dict[str, float], seed_uris: Sequence[str],
                 trust: float) -> Dict[str, np.ndarray]:
        learned = self.taste.learn_weights(list(seed_uris))
        stated = parse_stated(stated_spec)
        has_stated = np.linalg.norm(stated) > 1e-9
        if has_stated:
            blended = trust * _l2(stated) + (1 - trust) * _l2(learned)
        else:
            blended = _l2(learned)
        return {"stated": stated, "learned": learned, "blended": blended,
                "has_stated": has_stated}

    @staticmethod
    def _weights_payload(vec: np.ndarray) -> List[dict]:
        out = []
        for j, a in enumerate(AXES):
            out.append({"axis": a, "value": _round(vec[j], 4)})
        return out

    # ------------------------------------------------------------------ #
    #  public API used by the server
    # ------------------------------------------------------------------ #
    def profile(self, scalars=None, genres=None, seed_uris=None, trust=0.4) -> dict:
        seed_uris = list(seed_uris or [])
        spec = self._stated_to_axis_dict(scalars, genres)
        w = self._weights(spec, seed_uris, trust)
        flavors = self.taste.flavor_clusters(seed_uris) if seed_uris else []
        clusters = []
        for f in flavors:
            centroid = f.centroid
            clusters.append({
                "name": f.name,
                "share": _round(f.share, 3),
                "axes": {a: _round(centroid[j], 3) for j, a in enumerate(SCALAR_AXES)},
                "genre": GENRES[int(np.argmax(centroid[len(SCALAR_AXES):]))],
            })
        return {
            "trust": _round(trust, 3),
            "has_stated": bool(w["has_stated"]),
            "n_seed_tracks": len(seed_uris),
            "weights": {
                "stated": self._weights_payload(_l2(w["stated"])),
                "learned": self._weights_payload(w["learned"]),
                "blended": self._weights_payload(w["blended"]),
            },
            "clusters": clusters,
        }

    def recommend(self, scalars=None, genres=None, seed_uris=None, trust=0.4,
                  k: int = 24) -> dict:
        seed_uris = list(seed_uris or [])
        spec = self._stated_to_axis_dict(scalars, genres)
        w = self._weights(spec, seed_uris, trust)
        blended = w["blended"]
        seed_ids = self.taste.index.ids(seed_uris)

        if not seed_ids and np.linalg.norm(blended) < 1e-9:
            ranked = [int(i) for i in self.pop._ranked[:k]]
            scores = self.pop_norm[ranked]
        else:
            all_scores = self.taste._score_all(blended, seed_ids)
            ex = set(seed_ids)
            order = np.argsort(-all_scores)
            ranked = [int(i) for i in order if int(i) not in ex][:k]
            scores = all_scores[ranked]

        flavors = self.taste.flavor_clusters(seed_uris) if seed_uris else []
        smax = float(scores.max()) if len(scores) else 1.0
        smax = smax if smax > 1e-9 else 1.0
        seed_neighbours = [
            set(self.taste.cf.sim.getrow(sid).indices.tolist()) for sid in seed_ids
        ]
        seed_artists = {self.meta[t].artist_uri for t in seed_uris if t in self.meta}

        items = []
        for rank, (cid, sc) in enumerate(zip(ranked, scores)):
            uri = self.track_uris[cid]
            card = self._track_card(uri)
            expl = self.taste.explain(uri, seed_uris, weights=blended, flavors=flavors)
            axis_bars = [
                {"axis": a, "contribution": _round(v, 4),
                 "label": AXIS_META.get(a, {}).get("label", a.replace("genre:", "").title())}
                for a, v in expl.axis_scores.items()
            ]
            # co-occurrence support: how many seed tracks list this candidate
            # among their nearest co-occurring neighbours
            support = sum(1 for nb in seed_neighbours if cid in nb)
            same_artist = self.meta[uri].artist_uri in seed_artists
            items.append({
                "rank": rank + 1,
                **card,
                "score": _round(sc / smax, 4),
                "flavor": expl.flavor,
                "reasons": expl.reasons,
                "axis_bars": axis_bars,
                "cooccurrence": support,
                "same_artist": bool(same_artist),
            })
        return {"trust": _round(trust, 3), "n_seed_tracks": len(seed_uris),
                "items": items}

    # ------------------------------------------------------------------ #
    #  personas + sample playlists
    # ------------------------------------------------------------------ #
    def _archetype_tracks(self, arch_name: str, n: int, seed: int) -> List[str]:
        genre = next(a.genre for a in ARCHETYPES if a.name == arch_name)
        gid = GENRES.index(genre)
        mask = np.where(self._dom_genre_id == gid)[0]
        # favour popular, canonical tracks of the archetype
        w = self.pop_norm[mask] + 1e-3
        w = w / w.sum()
        rng = np.random.default_rng(seed)
        take = min(n, len(mask))
        pick = rng.choice(mask, size=take, replace=False, p=w)
        return [self.track_uris[int(i)] for i in pick]

    def _build_personas(self) -> Dict[str, Persona]:
        specs = [
            dict(id="sad_country", name="Slow, sad country",
                 blurb="Heartbreak ballads, deep lyrics, acoustic and unhurried.",
                 archetype="sad slow country",
                 stated={"tempo": -0.9, "valence": -0.9, "acousticness": 0.8,
                         "lyrical_depth": 0.9, "genre:country": 1.0},
                 stated_wrong={"tempo": 0.9, "energy": 0.9, "valence": 0.9,
                               "acousticness": -0.8, "genre:electronic": 1.0},
                 trust=0.4, adversarial=True),
            dict(id="gym_rap", name="Gym rap",
                 blurb="High-tempo, high-energy rap for beast mode.",
                 archetype="gym rap",
                 stated={"tempo": 0.9, "energy": 1.0, "genre:rap": 1.0},
                 stated_wrong={"energy": -0.9, "acousticness": 0.9,
                               "lyrical_depth": 0.8, "genre:folk": 1.0},
                 trust=0.4, adversarial=False),
            dict(id="indie_chill", name="Indie chill",
                 blurb="Mellow, dreamy indie for a slow Sunday.",
                 archetype="indie chill",
                 stated={"energy": -0.7, "tempo": -0.4, "genre:indie": 1.0},
                 stated_wrong={"energy": 1.0, "tempo": 0.9, "genre:metal": 1.0},
                 trust=0.4, adversarial=False),
        ]
        out: Dict[str, Persona] = {}
        for i, s in enumerate(specs):
            seed_uris = self._archetype_tracks(s["archetype"], n=16, seed=1000 + i)
            out[s["id"]] = Persona(
                id=s["id"], name=s["name"], blurb=s["blurb"],
                archetype=s["archetype"], seed_uris=seed_uris,
                stated=s["stated"], stated_wrong=s["stated_wrong"],
                trust=s["trust"], adversarial=s["adversarial"],
            )
        return out

    def personas(self) -> List[dict]:
        return [{"id": p.id, "name": p.name, "blurb": p.blurb,
                 "archetype": p.archetype, "adversarial": p.adversarial,
                 "n_seed_tracks": len(p.seed_uris)} for p in self._personas.values()]

    def persona_detail(self, persona_id: str, trust: Optional[float] = None,
                       adversarial: bool = False) -> dict:
        p = self._personas[persona_id]
        trust = p.trust if trust is None else float(trust)
        stated = p.stated_wrong if adversarial else p.stated
        # split stated dict back into scalars + genres for the shared code paths
        scalars = {a: v for a, v in stated.items() if a in SCALAR_AXES}
        genres = [a.split(":", 1)[1] for a in stated if a.startswith("genre:")]
        prof = self.profile(scalars=scalars, genres=genres,
                            seed_uris=p.seed_uris, trust=trust)
        recs = self.recommend(scalars=scalars, genres=genres,
                              seed_uris=p.seed_uris, trust=trust, k=24)
        # honesty meter: share of the top-20 recs that are truly this archetype
        top = recs["items"][:20]
        on_target = sum(1 for it in top if it["archetype"] == p.archetype)
        fit = on_target / len(top) if top else 0.0
        return {
            "id": p.id, "name": p.name, "blurb": p.blurb,
            "archetype": p.archetype, "adversarial": adversarial,
            "trust": _round(trust, 3),
            "stated_axes": self._stated_to_axis_dict(scalars, genres),
            "profile": prof,
            "recommendations": recs,
            "fit": _round(fit, 3),
            "seed_sample": [self._track_card(u) for u in p.seed_uris[:6]],
        }

    def persona_trust_curve(self, persona_id: str, adversarial: bool = True) -> dict:
        """Fit vs trust — the adversarial rescue as a curve."""
        p = self._personas[persona_id]
        stated = p.stated_wrong if adversarial else p.stated
        scalars = {a: v for a, v in stated.items() if a in SCALAR_AXES}
        genres = [a.split(":", 1)[1] for a in stated if a.startswith("genre:")]
        curve = []
        for t in [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]:
            recs = self.recommend(scalars=scalars, genres=genres,
                                  seed_uris=p.seed_uris, trust=t, k=20)
            top = recs["items"][:20]
            on = sum(1 for it in top if it["archetype"] == p.archetype)
            curve.append({"trust": t, "fit": _round(on / len(top) if top else 0.0, 3)})
        return {"id": p.id, "adversarial": adversarial, "archetype": p.archetype,
                "curve": curve}

    def _build_sample_playlists(self) -> List[dict]:
        """A handful of real synthetic playlists a user can load as a seed."""
        out = []
        # pick medium-length, clearly-titled playlists across archetypes
        seen_arch = set()
        rng = np.random.default_rng(self.seed + 5)
        order = rng.permutation(len(self.dataset.playlists))
        for i in order:
            pl = self.dataset.playlists[int(i)]
            if not pl.name or len(pl.tracks) < 12 or len(pl.tracks) > 45:
                continue
            # dominant archetype of the playlist
            ids = self.taste.index.ids(pl.tracks)
            if not ids:
                continue
            arch = self._home_arch[np.array(ids)]
            vals, counts = np.unique(arch, return_counts=True)
            dom = vals[int(np.argmax(counts))]
            if dom in seen_arch:
                continue
            seen_arch.add(dom)
            out.append({
                "pid": pl.pid, "name": pl.name, "archetype": dom,
                "n_tracks": len(pl.tracks),
                "seed_uris": pl.tracks[:12],
                "sample": [self._track_card(u) for u in pl.tracks[:5]],
            })
            if len(out) >= 6:
                break
        return out

    def sample_playlists(self) -> List[dict]:
        return self._sample_playlists


# ---- process-wide singleton (built lazily, cached) -------------------------- #
_ATLAS: Optional[TasteAtlas] = None


def get_atlas() -> TasteAtlas:
    global _ATLAS
    if _ATLAS is None:
        _ATLAS = TasteAtlas()
    return _ATLAS
