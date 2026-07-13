"""Genre/metadata enrichment for real listening histories (Phase 1.5).

Real Spotify exports carry no genre signal — the taste engine's 10 genre axes
are hard-zero for real tracks until this package joins two license-clean tables
onto them (see ``ENRICHMENT.md`` at the repo root for sources, measured hit
rates, licenses and caveats):

* :mod:`~playlistcont.enrichment.track_genres` — exact Spotify-id join against
  ``maharshipandya/spotify-tracks-dataset`` (BSD; seed-genre caveat documented
  in the module).
* :mod:`~playlistcont.enrichment.artist_genres` — normalized artist-NAME join
  against ``LeData/media-metadata-musicbrainz-artists`` (CC0; collision policy
  documented in the module), plus the curated tag -> 10-bucket mapping both
  paths share.
* :mod:`~playlistcont.enrichment.apply` — writes soft genre weights into the
  history's axis vectors and returns an :class:`EnrichmentReport`.
* :mod:`~playlistcont.enrichment.sources` — the only module that touches the
  network, and only when explicitly called (never at import, never in tests).
"""
from __future__ import annotations

from .apply import (
    PRIMARY_GENRE_WEIGHT,
    SECONDARY_GENRE_WEIGHT,
    EnrichmentReport,
    enrich_history,
    fill_genre_block,
)
from .artist_genres import (
    TAG_TO_BUCKETS,
    ArtistGenreRecord,
    build_artist_genre_index,
    lookup_artist_genres,
    map_tags_to_buckets,
    normalize_artist_name,
)
from .track_genres import (
    TrackGenreRecord,
    build_track_genre_index,
    lookup_track_genres,
)

__all__ = [
    "EnrichmentReport",
    "enrich_history",
    "fill_genre_block",
    "PRIMARY_GENRE_WEIGHT",
    "SECONDARY_GENRE_WEIGHT",
    "TAG_TO_BUCKETS",
    "ArtistGenreRecord",
    "build_artist_genre_index",
    "lookup_artist_genres",
    "map_tags_to_buckets",
    "normalize_artist_name",
    "TrackGenreRecord",
    "build_track_genre_index",
    "lookup_track_genres",
]
