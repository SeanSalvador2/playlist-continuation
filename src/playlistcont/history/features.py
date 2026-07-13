"""Thin feature adapter: interpretable axis vectors for history tracks.

Two provenances, two paths:

* **Synthetic** — tracks already carry their axis vector (the generator planted it).
  :func:`synthetic_feature_matrix` just stacks what is already in ``history.tracks``.
* **Real** — a Spotify export has no audio features.  :func:`attach_real_features`
  reuses the *existing* :mod:`playlistcont.data.real_features` machinery
  (``build_feature_table``, which is already keyed by the bare 22-char track id and is
  independent of the MPD ``Dataset``) to look features up, then writes them onto the
  history's tracks.  No change to ``real_features.py`` is needed.

Testability: pass ``feature_table`` (a ``{track_id -> axis vector}`` dict) to
:func:`attach_real_features` to inject features directly — no parquet, no network.  Omit
it (and pass ``source=...``) to stream from a real audio-features table in production.
"""
from __future__ import annotations

from typing import Dict, Iterable, List, Optional

import numpy as np

from ..data.schema import AXES, N_AXES
from .schema import ListeningHistory
from .spotify_export import track_id_from_uri


def synthetic_feature_matrix(
    history: ListeningHistory, uris: Iterable[str]
) -> np.ndarray:
    """Stack the known axis vectors for ``uris`` (zeros for any without features)."""
    uris = list(uris)
    out = np.zeros((len(uris), N_AXES), dtype=np.float32)
    for i, u in enumerate(uris):
        ht = history.tracks.get(u)
        if ht is not None and ht.features is not None:
            out[i] = ht.features
    return out


def real_axis_lookup(
    uris: Iterable[str],
    feature_table: Dict[str, np.ndarray],
) -> Dict[str, np.ndarray]:
    """Map ``uris`` -> axis vector using a ``{track_id -> vector}`` table.

    Pure and offline: ``feature_table`` is keyed by the bare 22-char id (the key
    ``real_features.build_feature_table`` returns), matched via
    :func:`track_id_from_uri`.  Unmatched URIs are simply omitted.
    """
    out: Dict[str, np.ndarray] = {}
    for u in uris:
        tid = track_id_from_uri(u)
        if tid is not None and tid in feature_table:
            out[u] = feature_table[tid]
    return out


def attach_real_features(
    history: ListeningHistory,
    source: Optional[str] = None,
    feature_table: Optional[Dict[str, np.ndarray]] = None,
) -> int:
    """Attach real audio-feature axis vectors to a history's tracks, in place.

    Provide either ``feature_table`` (a ``{track_id -> axis vector}`` dict, e.g. for
    tests) or ``source`` (a parquet path/dir streamed via
    ``real_features.build_feature_table``).  Returns the number of tracks matched.
    """
    if feature_table is None:
        if source is None:
            raise ValueError("provide either feature_table or source")
        from ..data.real_features import build_feature_table

        needed = {track_id_from_uri(u) for u in history.tracks}
        needed.discard(None)
        feature_table = build_feature_table(source, needed_ids=needed)

    matched = 0
    for uri, ht in history.tracks.items():
        tid = track_id_from_uri(uri)
        vec = feature_table.get(tid) if tid is not None else None
        if vec is not None:
            ht.features = np.asarray(vec, dtype=np.float32)
            matched += 1
    return matched
