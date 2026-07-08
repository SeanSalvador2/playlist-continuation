"""Core in-memory representation of an MPD-style corpus.

The real Spotify Million Playlist Dataset ships as 1,000 JSON "slice" files,
each holding 1,000 playlists.  A single playlist looks like::

    {
      "name": "musical",
      "pid": 5,
      "modified_at": 1493424000,
      "num_tracks": 66,
      "num_albums": 47,
      "num_followers": 1,
      "tracks": [
        {
          "pos": 0,
          "artist_name": "Degiheugi",
          "track_uri": "spotify:track:7vqa3sDmtEaVJ2gcvxtRID",
          "artist_uri": "spotify:artist:3TVXtAsR1Inumwj472S9r4",
          "track_name": "Finalement",
          "album_uri": "spotify:album:2KrxsD86ARO5beq7Q0Drfqa",
          "album_name": "Dancing Chords and Fireflies",
          "duration_ms": 166264
        }, ...
      ]
    }

We collapse that into three flat structures that every model consumes:

* ``playlists`` — list of :class:`Playlist` (pid, name, ordered track_uris).
* ``tracks``    — dict track_uri -> :class:`TrackMeta` (artist/album/name).
* ``features``  — optional dict track_uri -> interpretable axis vector, only
                  populated for synthetic data (or a user-supplied audio
                  features file for the real MPD; see ``loader.attach_features``).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

# Interpretable feature axes used by the taste engine.  The first block is a set
# of scalar "mood" axes in [0, 1]; genres are appended as a soft one-hot block.
SCALAR_AXES: List[str] = [
    "tempo",          # slow (0) -> fast (1)
    "energy",         # calm (0) -> intense (1)
    "valence",        # sad (0) -> happy (1)
    "acousticness",   # electric (0) -> acoustic (1)
    "lyrical_depth",  # filler lyrics (0) -> deep/meaningful (1)
]

GENRES: List[str] = [
    "country", "rap", "indie", "pop", "rock",
    "electronic", "folk", "metal", "rnb", "jazz",
]

AXES: List[str] = SCALAR_AXES + [f"genre:{g}" for g in GENRES]
N_AXES = len(AXES)


@dataclass
class TrackMeta:
    track_uri: str
    track_name: str
    artist_uri: str
    artist_name: str
    album_uri: str
    album_name: str


@dataclass
class Playlist:
    pid: int
    name: str
    tracks: List[str]  # ordered list of track_uris


@dataclass
class Dataset:
    """A whole MPD-style corpus plus optional interpretable features."""

    playlists: List[Playlist]
    tracks: Dict[str, TrackMeta]
    features: Optional[Dict[str, np.ndarray]] = field(default=None)

    # ---- convenience ---------------------------------------------------
    def artist_of(self, track_uri: str) -> Optional[str]:
        m = self.tracks.get(track_uri)
        return m.artist_uri if m else None

    @property
    def n_playlists(self) -> int:
        return len(self.playlists)

    @property
    def n_tracks(self) -> int:
        return len(self.tracks)

    def track_artist_map(self) -> Dict[str, str]:
        return {uri: m.artist_uri for uri, m in self.tracks.items()}

    def feature_matrix(self, track_order: List[str]) -> np.ndarray:
        """Stack axis vectors for the given track order (zeros if unknown)."""
        if self.features is None:
            raise ValueError("Dataset has no interpretable features attached.")
        out = np.zeros((len(track_order), N_AXES), dtype=np.float32)
        for i, uri in enumerate(track_order):
            v = self.features.get(uri)
            if v is not None:
                out[i] = v
        return out
