"""Faithful loader for the real Spotify Million Playlist Dataset.

The MPD is distributed as ``data/mpd.slice.*.json`` files (1,000 playlists
each) after you register + download from AIcrowd:
https://www.aicrowd.com/challenges/spotify-million-playlist-dataset-challenge

This module reads any subset of those slice files into a :class:`Dataset`.
It is schema-faithful, so switching the whole project from synthetic to real
data is a one-line change::

    from playlistcont.data.loader import load_mpd
    ds = load_mpd("/path/to/spotify_mpd/data", max_slices=5)

The real MPD carries no audio features, so ``ds.features`` is ``None`` unless
you call :func:`attach_features` with a CSV of per-track audio features (e.g.
exported from the Spotify audio-features API or an offline genre model).  See
the README for the axis mapping used there.
"""
from __future__ import annotations

import glob
import json
import os
from typing import Dict, List, Optional

import numpy as np

from .schema import AXES, Dataset, Playlist, SCALAR_AXES, TrackMeta


def _playlist_from_json(obj: dict) -> tuple[Playlist, List[TrackMeta]]:
    metas: List[TrackMeta] = []
    uris: List[str] = []
    for t in obj.get("tracks", []):
        uri = t["track_uri"]
        uris.append(uri)
        metas.append(
            TrackMeta(
                track_uri=uri,
                track_name=t.get("track_name", ""),
                artist_uri=t.get("artist_uri", ""),
                artist_name=t.get("artist_name", ""),
                album_uri=t.get("album_uri", ""),
                album_name=t.get("album_name", ""),
            )
        )
    pl = Playlist(pid=int(obj["pid"]), name=obj.get("name", "") or "", tracks=uris)
    return pl, metas


def load_mpd_files(paths: List[str], max_playlists: Optional[int] = None) -> Dataset:
    """Load an explicit list of MPD slice json files into a :class:`Dataset`."""
    playlists: List[Playlist] = []
    tracks: Dict[str, TrackMeta] = {}
    for p in paths:
        with open(p, "r", encoding="utf-8") as fh:
            blob = json.load(fh)
        for obj in blob.get("playlists", []):
            pl, metas = _playlist_from_json(obj)
            playlists.append(pl)
            for m in metas:
                tracks.setdefault(m.track_uri, m)
            if max_playlists is not None and len(playlists) >= max_playlists:
                return Dataset(playlists=playlists, tracks=tracks)
    return Dataset(playlists=playlists, tracks=tracks)


def load_mpd(
    data_dir: str,
    max_slices: Optional[int] = None,
    max_playlists: Optional[int] = None,
) -> Dataset:
    """Load MPD slice files from a directory (sorted for determinism)."""
    paths = sorted(glob.glob(os.path.join(data_dir, "mpd.slice.*.json")))
    if not paths:
        # Also accept a directory of arbitrary *.json slices (e.g. fixtures).
        paths = sorted(glob.glob(os.path.join(data_dir, "*.json")))
    if max_slices is not None:
        paths = paths[:max_slices]
    if not paths:
        raise FileNotFoundError(f"No MPD slice json files found in {data_dir!r}")
    return load_mpd_files(paths, max_playlists=max_playlists)


def attach_features(ds: Dataset, features_csv: str) -> Dataset:
    """Attach interpretable axis features to a real-MPD dataset from a CSV.

    Expected columns: ``track_uri`` plus any of the axis names in
    ``schema.AXES``.  Missing axes default to 0.5 (neutral) for scalar mood
    axes and 0 for genre axes.  This is the documented bridge that lets the
    taste engine run on the real MPD once you supply audio features / genres.
    """
    import csv

    features: Dict[str, np.ndarray] = {}
    with open(features_csv, newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            uri = row["track_uri"]
            v = np.full(len(AXES), 0.0, dtype=np.float32)
            for i, axis in enumerate(AXES):
                if axis in SCALAR_AXES:
                    v[i] = 0.5
                if axis in row and row[axis] != "":
                    v[i] = float(row[axis])
            features[uri] = v
    ds.features = features
    return ds
