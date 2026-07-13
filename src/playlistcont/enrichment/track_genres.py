"""Track-id -> genre buckets via the maharshipandya track table (BSD).

Join key: the exact 22-char Spotify ``track_id`` — the same key
``data/real_features.py`` and the history exports already share, so this join
has none of the name-ambiguity of the artist path.

IMPORTANT documented caveat (from review of the source): the table's
``track_genre`` column is the **SEED GENRE the track was fetched under**, not a
curated per-track label.  The dataset was built by querying Spotify's
recommendation endpoint once per genre seed (1,000 tracks each), so a track that
surfaced under several seeds appears on several rows with different
``track_genre`` values (~21% of tracks; 114,000 rows vs 89,741 unique ids).
Consequences we build in rather than hide:

* A track's "genre" here is really *"the set of seed genres that surfaced it"* —
  useful soft evidence, not ground truth.
* We therefore aggregate all rows per id into a genre **set** and map it to the
  repo's 10 buckets via the same curated mapping as the artist path
  (:data:`playlistcont.enrichment.artist_genres.TAG_TO_BUCKETS`), producing a
  ranked multi-bucket assignment instead of pretending there is one true label.
* Seed genres with no honest bucket (``anime``, ``brazil``, ``study``, ...) are
  kept as raw genres, never force-mapped.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional

from .artist_genres import map_tags_to_buckets


@dataclass
class TrackGenreRecord:
    """One track id's aggregated seed-genre evidence."""

    track_id: str
    raw_genres: List[str] = field(default_factory=list)  # seed genres, dataset order
    buckets: List[str] = field(default_factory=list)     # mapped, primary first


def build_track_genre_index(
    table,
    track_ids: Optional[Iterable[str]] = None,
) -> Dict[str, TrackGenreRecord]:
    """Index a maharshipandya-shaped table (``track_id``, ``track_genre``) by id.

    ``track_ids`` (optional) restricts the index to those ids (pass the
    history's ids to keep the index tiny).  Multi-row tracks are aggregated into
    a genre set; buckets are ordered by vote count then first appearance, via
    the same :func:`~playlistcont.enrichment.artist_genres.map_tags_to_buckets`
    the artist path uses.
    """
    wanted = set(track_ids) if track_ids is not None else None
    genres_by_id: Dict[str, List[str]] = {}
    for tid, genre in zip(table["track_id"], table["track_genre"]):
        tid = str(tid)
        if wanted is not None and tid not in wanted:
            continue
        lst = genres_by_id.setdefault(tid, [])
        g = str(genre)
        if g not in lst:
            lst.append(g)

    out: Dict[str, TrackGenreRecord] = {}
    for tid, genres in genres_by_id.items():
        buckets, raw = map_tags_to_buckets(genres)
        out[tid] = TrackGenreRecord(track_id=tid, raw_genres=raw, buckets=buckets)
    return out


def lookup_track_genres(
    track_ids: Iterable[str],
    table=None,
    index: Optional[Dict[str, TrackGenreRecord]] = None,
) -> Dict[str, TrackGenreRecord]:
    """Resolve track ids -> :class:`TrackGenreRecord` (unmatched ids omitted)."""
    ids = [str(t) for t in track_ids]
    if index is None:
        if table is None:
            raise ValueError("provide either table or index")
        index = build_track_genre_index(table, track_ids=ids)
    return {t: index[t] for t in ids if t in index}
