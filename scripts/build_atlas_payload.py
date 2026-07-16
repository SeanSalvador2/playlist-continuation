#!/usr/bin/env python3
"""Build the CORE *taste-atlas* payload for any listener, using repo code only.

    python scripts/build_atlas_payload.py --history <pickle-or-export-path> \
        --out payload.json [--features <parquet>] [--tz America/New_York]

``--history`` accepts, transparently:

* a **pickled** :class:`~playlistcont.history.schema.ListeningHistory` (``*.pkl``);
* a Spotify **extended** streaming-history directory or ``.zip``
  (``Streaming_History_Audio_*.json`` — loaded via
  :func:`~playlistcont.history.spotify_export.load_extended_history`);
* a Spotify **basic** streaming-history directory or ``.zip``
  (``StreamingHistory*.json`` — :func:`load_basic_history`).

``--features`` (optional) is a parquet of per-track interpretable axes to join onto a
real export that carries none; without it a real export still produces every section,
but the feature-dependent ones (trajectory, cohort sound profiles, story flavor names)
degrade honestly to empty/generic rather than fabricating axes.

WHAT THIS POWERS (and what it does NOT)
---------------------------------------
The emitted JSON is the deterministic core that drives the *narrative* atlas pages:
the summary header, the thesis chart, the chapter timeline + cohort dossiers, the
fact-checked story (with sub-chapter beats), the weekly taste trajectory, the
top-50 leaderboards, the plays-over-time and genre-share charts, and the listening
clock.  It deliberately does **not** compute the ML add-on surfaces — the co-listening
**starmap**, the recommender **lenses**, or the **skip-model** — which need trained
models out of this script's scope (their pages consume separate artifacts).

Determinism: no wall-clock or randomness leaks into the payload; the same inputs
always yield byte-identical JSON.
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
from datetime import date
from typing import Optional

import numpy as np

from playlistcont.analytics import queries as q
from playlistcont.data.schema import GENRES
from playlistcont.dynamics.chapters import adaptive_chapters
from playlistcont.dynamics.cohorts import (
    CohortModel, boundary_diff, cohorts_for_spans, feature_shift_sentence,
)
from playlistcont.dynamics.eras import build_eras
from playlistcont.dynamics.story import gather_facts, render_story
from playlistcont.dynamics.trajectory import compute_trajectory
from playlistcont.dynamics.windows import build_windows
from playlistcont.history.schema import ListeningHistory
from playlistcont.history.store import HistoryStore

MS_MIN_PLAY = 30_000  # a "real" play: >= 30 seconds


# --------------------------------------------------------------------------- #
# loading
# --------------------------------------------------------------------------- #
def load_history(path: str) -> ListeningHistory:
    """Load a history from a pickle, or a Spotify extended/basic export dir/zip."""
    from playlistcont.history import spotify_export as se

    if path.endswith(".pkl") or path.endswith(".pickle"):
        with open(path, "rb") as fh:
            obj = pickle.load(fh)
        if not isinstance(obj, ListeningHistory):
            raise SystemExit(f"pickle at {path} is a {type(obj).__name__}, not a ListeningHistory")
        return obj
    # export dir or zip: try extended first, then basic
    try:
        hist = se.load_extended_history(path)
        if hist.events:
            return hist
    except Exception:
        pass
    return se.load_basic_history(path)


def maybe_attach_features(history: ListeningHistory, features_path: Optional[str]) -> ListeningHistory:
    """Best-effort join of an interpretable-axes source onto a real export.

    Delegates to :func:`playlistcont.history.features.attach_real_features` (which owns
    the audio-features join contract).  A failure is non-fatal — the payload just keeps
    its honest zero feature coverage rather than fabricating axes.
    """
    if not features_path:
        return history
    try:
        from playlistcont.history.features import attach_real_features
        matched = attach_real_features(history, source=features_path)
        print(f"[info] attached features to {matched} tracks from {features_path}",
              file=sys.stderr)
    except Exception as exc:  # pragma: no cover - optional path
        print(f"[warn] could not attach features from {features_path}: {exc}", file=sys.stderr)
    return history


# --------------------------------------------------------------------------- #
# feature-derived sections (thesis / genre-by-year) off the cohort arrays
# --------------------------------------------------------------------------- #
def _year_array(model: CohortModel) -> np.ndarray:
    if not len(model.dates_ord):
        return np.empty(0, dtype=int)
    return np.array([date.fromordinal(int(o)).year for o in model.dates_ord], dtype=int)


def genre_share_by_year(model: CohortModel) -> dict:
    """Per-calendar-year share of each genre over feature-plays (dominant bucket)."""
    years = _year_array(model)
    fm = model.has_feat & (model.gbucket >= 0)
    out = []
    for yr in sorted(set(int(y) for y in years[fm])) if fm.any() else []:
        sel = fm & (years == yr)
        nf = int(sel.sum())
        if not nf:
            continue
        shares = np.bincount(model.gbucket[sel], minlength=len(GENRES)) / nf
        row = {"year": yr, "feature_plays": nf,
               "shares": {GENRES[j]: round(float(shares[j]), 4)
                          for j in np.argsort(-shares) if shares[j] > 0}}
        out.append(row)
    return {"genres": list(GENRES), "by_year": out}


def thesis_series(model: CohortModel) -> dict:
    """Dominant-genre share by year for the listener's OWN top genre (not hardcoded).

    The top genre is the one with the most feature-plays over the whole history; the
    series is that genre's share of feature-plays in each calendar year.
    """
    fm = model.has_feat & (model.gbucket >= 0)
    if not fm.any():
        return {"genre": None, "by_year": []}
    totals = np.bincount(model.gbucket[fm], minlength=len(GENRES))
    top_idx = int(totals.argmax())
    genre = GENRES[top_idx]
    years = _year_array(model)
    series = []
    for yr in sorted(set(int(y) for y in years[fm])):
        sel = fm & (years == yr)
        nf = int(sel.sum())
        share = float((model.gbucket[sel] == top_idx).sum()) / nf if nf else 0.0
        series.append({"year": yr, "share": round(share, 4), "feature_plays": nf})
    return {"genre": genre, "by_year": series}


def top_leaderboards(store: HistoryStore, entity: str, limit: int = 50) -> dict:
    """Top-``limit`` for an entity on two bases: all plays, and >= 30s plays.

    The 'all' base reuses :func:`playlistcont.analytics.queries.top_items`; the
    '>= 30s' base applies the same ranking to plays of at least 30 seconds via a
    direct query (top_items has no ms filter), so a leaderboard can honestly separate
    genuine listens from skips/previews.
    """
    all_base = q.top_items(store, entity=entity, limit=limit, by="plays")
    key, name_col, artist_col = q._ENTITY_SQL[entity]
    art_sel = f"ANY_VALUE({artist_col})" if artist_col else "NULL"
    agg = store.query(
        f"""
        SELECT {key} AS item_key, ANY_VALUE({name_col}) AS name, {art_sel} AS artist,
               COUNT(*) AS plays, SUM(e.ms_played) / {q.MS_PER_MIN} AS minutes
        FROM events e JOIN tracks t ON e.track_uri = t.uri
        WHERE e.ms_played >= {MS_MIN_PLAY}
        GROUP BY {key}
        ORDER BY plays DESC, name ASC
        LIMIT {int(limit)}
        """
    )
    total = int(all_base["total_plays"]) or 1
    rows = []
    for pos, (_, r) in enumerate(agg.iterrows()):
        row = {"rank": pos + 1,
               "name": None if r["name"] is None else str(r["name"]),
               "plays": int(r["plays"]),
               "minutes": round(float(r["minutes"]), 2),
               "share": round(int(r["plays"]) / total, 4)}
        if entity == "tracks":
            row["artist"] = None if r["artist"] is None else str(r["artist"])
        rows.append(row)
    return {"all": all_base, "min30s": {"entity": entity, "min_ms": MS_MIN_PLAY, "rows": rows}}


def qualified_summary(store: HistoryStore, summary: dict) -> dict:
    """Honest coverage caveats around the headline numbers.

    Reports feature coverage (how much of the stream carries interpretable axes — the
    ceiling on every sound-based claim), skip-flag reliability, and the display tz.
    """
    fc = store.query(
        """
        SELECT COUNT(*) AS plays, COUNT(f.uri) AS with_features
        FROM events e LEFT JOIN track_features f ON e.track_uri = f.uri
        """
    ).iloc[0]
    plays = int(fc["plays"])
    wf = int(fc["with_features"])
    return {
        "feature_coverage": round(wf / plays, 4) if plays else 0.0,
        "feature_plays": wf,
        "skip_rate": summary.get("skip_rate"),
        "skip_reliable_from": summary.get("skip_reliable_from"),
        "skip_reason": summary.get("skip_reason"),
        "caveats": [
            "Sound-based claims (axes, cohorts, trajectory) cover only "
            f"{round(100 * wf / plays, 1) if plays else 0.0}% of plays with features.",
            ("Skip rate is computed only over the reliable window; "
             f"{summary.get('skip_reason')}" if summary.get("skip_rate") is None
             else "Skip rate is computed only over the reliable logging window."),
        ],
    }


# --------------------------------------------------------------------------- #
# chapter plan + cohorts + story
# --------------------------------------------------------------------------- #
def build_narrative(store, history, ws, model, min_chapters, min_subsections):
    """The chapter plan, its cohort dossiers, and the fact-checked story."""
    plan = adaptive_chapters(ws, min_chapters=min_chapters, min_subsections=min_subsections)

    # chapter-level cohort dossiers
    chapter_spans = [(c.start, c.end) for c in plan.chapters]
    chapter_cohorts = cohorts_for_spans(model, chapter_spans)

    # sub-section cohort dossiers, one list per chapter (feeds the story sub-beats)
    sub_cohorts = []
    for ch in plan.chapters:
        spans = [(s.start, s.end) for s in ch.subsections]
        sub_cohorts.append(cohorts_for_spans(model, spans))

    # boundary diffs between consecutive chapters
    b_diffs = []
    for i in range(len(plan.chapters) - 1):
        prev_span, next_span = chapter_spans[i], chapter_spans[i + 1]
        bnd = plan.boundaries[i]
        diff = boundary_diff(model, prev_span, next_span, bnd.date)
        diff["feature_shift"] = feature_shift_sentence(store, bnd.date, diff["flank_days"])
        diff["confidence"] = bnd.confidence
        b_diffs.append(diff)

    # named eras aligned to the chapters, then the verified story with sub-beats
    detections = plan.as_detections()
    eras = build_eras(store, detections, flavor_model=ws.flavor_model)
    facts = gather_facts(store, eras, detections,
                         chapter_plan=plan, sub_cohorts=sub_cohorts)
    story = render_story(facts)

    return {
        "chapter_plan": plan.to_payload(),
        "cohorts": {
            "chapters": [c.to_payload() for c in chapter_cohorts],
            "subsections": [[c.to_payload() for c in chs] for chs in sub_cohorts],
            "boundary_diffs": b_diffs,
        },
        "story": {"facts": facts.to_payload(), "slides": story.to_payload()},
    }


# --------------------------------------------------------------------------- #
# top-level payload
# --------------------------------------------------------------------------- #
def build_payload(history: ListeningHistory, tz: str, min_chapters: int,
                  min_subsections: int) -> dict:
    store = HistoryStore.from_history(history)
    ws = build_windows(history, granularity="week", min_events=30, weighting="plays")
    model = CohortModel.from_store(store)

    summary = q.summary(store)
    payload = {
        "meta": {
            "schema": "atlas-core/1",
            "provenance": history.provenance,
            "n_events": history.n_events,
            "n_tracks": history.n_tracks,
            "tz": tz,
            "powers": ["summary", "thesis", "chapters", "cohorts", "story",
                       "trajectory", "leaderboards", "plays_over_time",
                       "genre_share", "clock"],
            "requires_ml_addons": ["starmap", "lenses", "skip_model"],
        },
        "summary": summary,
        "qualified_summary": qualified_summary(store, summary),
        "thesis_series": thesis_series(model),
        "genre_share_by_year": genre_share_by_year(model),
        "trajectory": compute_trajectory(ws).to_payload(),
        "top_artists": top_leaderboards(store, "artists", 50),
        "top_tracks": top_leaderboards(store, "tracks", 50),
        "plays_by_year": q.trends(store, metric="plays", granularity="year"),
        "plays_by_month": q.trends(store, metric="plays", granularity="month"),
        "clock": q.listening_clock(store, tz=tz),
    }
    payload.update(build_narrative(store, history, ws, model,
                                   min_chapters, min_subsections))
    store.close()
    return payload


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--history", required=True,
                    help="pickled ListeningHistory (*.pkl) OR a Spotify export dir/zip")
    ap.add_argument("--out", required=True, help="output JSON path")
    ap.add_argument("--features", default=None,
                    help="optional per-track interpretable-axes parquet to join onto a "
                         "real export (out-of-the-box synthetic/enriched histories carry "
                         "their own axes)")
    ap.add_argument("--tz", default=os.environ.get("PLAYLISTCONT_TZ", "America/New_York"),
                    help="IANA display timezone for the clock (default: ET / $PLAYLISTCONT_TZ)")
    ap.add_argument("--min-chapters", type=int, default=2)
    ap.add_argument("--min-subsections", type=int, default=2)
    args = ap.parse_args(argv)

    history = load_history(args.history)
    history = maybe_attach_features(history, args.features)
    tz = q.resolve_tz(args.tz)
    payload = build_payload(history, tz, args.min_chapters, args.min_subsections)

    with open(args.out, "w") as fh:
        json.dump(payload, fh, ensure_ascii=False, separators=(",", ":"))
    size = os.path.getsize(args.out)
    print(f"wrote {args.out} ({size:,} bytes) — {len(payload)} top-level sections, "
          f"{payload['chapter_plan']['n_chapters']} chapters, "
          f"strength={payload['chapter_plan']['structure_strength']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
