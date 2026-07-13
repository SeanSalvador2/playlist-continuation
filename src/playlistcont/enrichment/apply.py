"""Fill the genre block of a real listening history's axis vectors.

Real exports have no genre signal at all (the genre axes are hard-zero), so this
module joins the two enrichment tables onto a :class:`ListeningHistory` and
writes **soft genre weights** into each track's axis vector:

* **Artist-level first** (MusicBrainz tags via normalized name match), then
  **track-level where available** (maharshipandya seed genres via exact id) —
  track-level wins on conflict because it is track-specific evidence, while an
  artist tag paints every track by that artist with the same brush.  If the
  track-level join matched but produced no mappable bucket, the artist-level
  assignment is kept (evidence that maps beats evidence that doesn't).
* **Soft weights, not hard one-hots**: primary bucket gets 0.85 (matching the
  synthetic generator's convention in ``data/synthetic.py``), any further
  buckets get 0.4.  Genre axes not assigned stay 0.
* **Scalar axes are never touched.**  A track that has no feature vector yet
  gets one created with the neutral-0.5 scalar convention of
  ``data/real_features._row_to_axis`` before its genre block is filled.

Everything is injectable: pass in-memory DataFrames (or prebuilt indexes) so
tests run offline on tiny fabricated fixtures; production callers pass the
frames loaded by :mod:`playlistcont.enrichment.sources`.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

from ..data.schema import AXES, GENRES, N_AXES, SCALAR_AXES
from ..history.schema import ListeningHistory
from ..history.spotify_export import track_id_from_uri
from .artist_genres import (
    ArtistGenreRecord,
    build_artist_genre_index,
    map_tag,
    normalize_artist_name,
)
from .track_genres import TrackGenreRecord, build_track_genre_index

# soft-weight convention (primary matches data/synthetic.py's 0.85 home-genre
# weight; secondary is a deliberately weaker "also plausibly this" signal)
PRIMARY_GENRE_WEIGHT = 0.85
SECONDARY_GENRE_WEIGHT = 0.4

_AX = {a: i for i, a in enumerate(AXES)}
_GENRE_IDX = {g: _AX[f"genre:{g}"] for g in GENRES}


@dataclass
class EnrichmentReport:
    """What the enrichment run actually achieved — honest numbers, no vibes.

    ``genre_coverage`` is the share of **plays** (not tracks) whose track ended
    up with at least one genre bucket, so heavy-rotation tracks count for what
    they are.  ``artist_tag_rows`` carries the raw folksonomy rows
    (``artist_name, tag, mapped_bucket-or-None``) for
    :meth:`HistoryStore.attach_artist_tags`.
    """

    n_tracks: int = 0
    artist_matched: int = 0            # tracks whose artist name resolved
    track_matched: int = 0             # tracks whose id resolved in the track table
    genre_coverage: float = 0.0        # share of PLAYS whose track got any genre
    ambiguous_artists: int = 0         # distinct matched artist names with collisions
    enriched_tracks: int = 0           # tracks that received >=1 genre bucket
    artist_tag_rows: List[dict] = field(default_factory=list, repr=False)

    def to_payload(self) -> dict:
        """JSON-ready summary (drops the raw tag rows)."""
        return {
            "n_tracks": self.n_tracks,
            "artist_matched": self.artist_matched,
            "track_matched": self.track_matched,
            "genre_coverage": round(self.genre_coverage, 4),
            "ambiguous_artists": self.ambiguous_artists,
            "enriched_tracks": self.enriched_tracks,
        }


def _neutral_vector() -> np.ndarray:
    """Fresh axis vector with the 0.5-neutral scalar convention, genres zero."""
    v = np.zeros(N_AXES, dtype=np.float32)
    for a in SCALAR_AXES:
        v[_AX[a]] = 0.5
    return v


def fill_genre_block(vec: np.ndarray, buckets: List[str]) -> np.ndarray:
    """Overwrite ONLY the genre block of ``vec`` with soft weights, in place.

    ``buckets[0]`` -> :data:`PRIMARY_GENRE_WEIGHT`, the rest ->
    :data:`SECONDARY_GENRE_WEIGHT`; unassigned genre axes are reset to 0.
    Scalar axes are untouched.
    """
    for g, idx in _GENRE_IDX.items():
        vec[idx] = 0.0
    for i, b in enumerate(buckets):
        vec[_GENRE_IDX[b]] = PRIMARY_GENRE_WEIGHT if i == 0 else SECONDARY_GENRE_WEIGHT
    return vec


def _as_artist_index(artist_table, names) -> Dict[str, ArtistGenreRecord]:
    if artist_table is None:
        return {}
    if isinstance(artist_table, dict):
        return artist_table
    return build_artist_genre_index(artist_table, names=names)


def _as_track_index(track_table, ids) -> Dict[str, TrackGenreRecord]:
    if track_table is None:
        return {}
    if isinstance(track_table, dict):
        return track_table
    return build_track_genre_index(track_table, track_ids=ids)


def enrich_history(
    history: ListeningHistory,
    artist_table=None,
    track_table=None,
) -> EnrichmentReport:
    """Fill genre axes of ``history``'s tracks in place; return an honest report.

    ``artist_table`` — LeData-shaped DataFrame (``name``, ``tags``) or a prebuilt
    ``{normalized_name -> ArtistGenreRecord}`` index.  ``track_table`` —
    maharshipandya-shaped DataFrame (``track_id``, ``track_genre``) or a prebuilt
    ``{track_id -> TrackGenreRecord}`` index.  Either may be ``None`` (that
    source is simply skipped).  Precedence and weight conventions are in the
    module docstring.
    """
    artist_names = sorted({ht.artist_name for ht in history.tracks.values()})
    ids = sorted({t for t in (track_id_from_uri(u) for u in history.tracks)
                  if t is not None})
    a_index = _as_artist_index(artist_table, artist_names)
    t_index = _as_track_index(track_table, ids)

    report = EnrichmentReport(n_tracks=len(history.tracks))
    enriched_uris = set()
    matched_artist_names: Dict[str, ArtistGenreRecord] = {}

    for uri, ht in history.tracks.items():
        a_rec = a_index.get(normalize_artist_name(ht.artist_name))
        tid = track_id_from_uri(uri)
        t_rec = t_index.get(tid) if tid is not None else None

        if a_rec is not None:
            report.artist_matched += 1
            matched_artist_names[ht.artist_name] = a_rec
        if t_rec is not None:
            report.track_matched += 1

        # artist-level first, track-level wins when it actually maps
        buckets: List[str] = a_rec.buckets if a_rec is not None else []
        if t_rec is not None and t_rec.buckets:
            buckets = t_rec.buckets
        if not buckets:
            continue

        if ht.features is None:
            ht.features = _neutral_vector()
        fill_genre_block(ht.features, buckets)
        report.enriched_tracks += 1
        enriched_uris.add(uri)

    plays = len(history.events)
    hits = sum(1 for e in history.events if e.track_uri in enriched_uris)
    report.genre_coverage = hits / plays if plays else 0.0
    report.ambiguous_artists = sum(
        1 for rec in matched_artist_names.values() if rec.ambiguous)

    # raw folksonomy rows for HistoryStore.attach_artist_tags: one row per
    # (artist, tag, mapped bucket); unmapped tags get a single row with None.
    for name in sorted(matched_artist_names):
        rec = matched_artist_names[name]
        for tag in rec.raw_tags:
            mapped = map_tag(tag)
            if mapped:
                for b in mapped:
                    report.artist_tag_rows.append(
                        {"artist_name": name, "tag": tag, "mapped_bucket": b})
            else:
                report.artist_tag_rows.append(
                    {"artist_name": name, "tag": tag, "mapped_bucket": None})
    return report
