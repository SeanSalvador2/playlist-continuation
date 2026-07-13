"""Personal listening-history modality: timestamped play-event streams.

See :mod:`playlistcont.history.schema` for the data model and its honest limitations.
Synthetic histories with planted ground truth come from
:mod:`playlistcont.history.synthetic`; real Spotify exports from
:mod:`playlistcont.history.spotify_export`; a DuckDB analytics store from
:mod:`playlistcont.history.store`; and the interpretable-axis adapter from
:mod:`playlistcont.history.features`.
"""
from __future__ import annotations

from .schema import (
    HistoryGroundTruth,
    HistoryTrack,
    ListenEvent,
    ListeningHistory,
    PlantedChange,
    RegimeSpec,
    SeasonalSpan,
    Trap,
)
from .synthetic import make_listener_population, make_synthetic_history

__all__ = [
    "ListenEvent",
    "HistoryTrack",
    "ListeningHistory",
    "RegimeSpec",
    "PlantedChange",
    "Trap",
    "SeasonalSpan",
    "HistoryGroundTruth",
    "make_synthetic_history",
    "make_listener_population",
]
