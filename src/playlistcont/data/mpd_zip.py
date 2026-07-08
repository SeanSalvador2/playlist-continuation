"""Streaming reader for the real MPD distributed as a single ``.zip``.

The official Million Playlist Dataset ships as ``spotify_million_playlist_dataset.zip``
containing 1,000 ``data/mpd.slice.*.json`` members (1,000 playlists each,
~33 GB uncompressed).  Extracting it is impractical, so this module reads each
slice member *directly from the zip* with the stdlib :mod:`zipfile` and parses
it on the fly.  Peak memory is one slice (1,000 playlists) at a time, so the
whole 1M-playlist corpus can be swept, counted, or subsampled without ever
materialising the full JSON.

Two entry points:

* :func:`stream_track_stats` — one streaming pass over (a subset of) the zip
  that accumulates per-track play counts and lightweight metadata.  Cheap
  enough (a Counter + a dict) to run at the full 1M-playlist scale; used for
  the popularity headline and for choosing a catalogue frequency threshold.
* :func:`load_mpd_zip` — build an in-memory :class:`Dataset` for a subsample of
  playlists (optionally pruning tracks below a frequency threshold), which the
  existing models consume unchanged.

Everything here is pure/streaming and unit-tested on a tiny in-memory zip; no
network access is involved.
"""
from __future__ import annotations

import json
import zipfile
from collections import Counter
from typing import Callable, Dict, Iterator, List, Optional, Tuple

from .loader import _playlist_from_json
from .schema import Dataset, Playlist, TrackMeta


def slice_members(zf: zipfile.ZipFile) -> List[str]:
    """Sorted list of MPD slice member names inside an open zip.

    Accepts either the official ``data/mpd.slice.*.json`` layout or a flat set
    of ``*.json`` members (used by the test fixture), so the loader is robust
    to how the archive was built.
    """
    names = [n for n in zf.namelist() if n.endswith(".json")]
    slices = [n for n in names if "mpd.slice" in n]
    chosen = slices if slices else names

    def _key(n: str) -> Tuple[int, str]:
        # sort by the starting pid encoded in mpd.slice.<start>-<end>.json
        base = n.rsplit("/", 1)[-1]
        try:
            start = int(base.split(".")[2].split("-")[0])
            return (start, n)
        except (IndexError, ValueError):
            return (0, n)

    return sorted(chosen, key=_key)


def iter_raw_playlists(
    zip_path: str,
    max_slices: Optional[int] = None,
    max_playlists: Optional[int] = None,
) -> Iterator[dict]:
    """Yield raw playlist dicts streamed from the zip, one slice at a time."""
    with zipfile.ZipFile(zip_path) as zf:
        members = slice_members(zf)
        if max_slices is not None:
            members = members[:max_slices]
        seen = 0
        for name in members:
            blob = json.loads(zf.read(name))
            for obj in blob.get("playlists", []):
                yield obj
                seen += 1
                if max_playlists is not None and seen >= max_playlists:
                    return


def stream_track_stats(
    zip_path: str,
    max_slices: Optional[int] = None,
    max_playlists: Optional[int] = None,
    progress: Optional[Callable[[int], None]] = None,
) -> Tuple[Counter, Dict[str, TrackMeta], int, int]:
    """One streaming pass: track play counts + metadata over the (sub)corpus.

    Returns ``(pop_counter, track_meta, n_playlists, n_interactions)`` where
    ``pop_counter[uri]`` is the number of playlists a track appears in.  Runs at
    the full 1M scale in bounded memory (a Counter and a metadata dict keyed by
    the ~2.26M unique track_uris).
    """
    pop: Counter = Counter()
    meta: Dict[str, TrackMeta] = {}
    n_pl = 0
    n_int = 0
    for obj in iter_raw_playlists(zip_path, max_slices, max_playlists):
        n_pl += 1
        for t in obj.get("tracks", []):
            uri = t["track_uri"]
            pop[uri] += 1
            n_int += 1
            if uri not in meta:
                meta[uri] = TrackMeta(
                    track_uri=uri,
                    track_name=t.get("track_name", ""),
                    artist_uri=t.get("artist_uri", ""),
                    artist_name=t.get("artist_name", ""),
                    album_uri=t.get("album_uri", ""),
                    album_name=t.get("album_name", ""),
                )
        if progress is not None and n_pl % 100000 == 0:
            progress(n_pl)
    return pop, meta, n_pl, n_int


def load_mpd_zip(
    zip_path: str,
    max_playlists: Optional[int] = None,
    max_slices: Optional[int] = None,
    min_track_count: int = 1,
    keep_uris: Optional[set] = None,
    progress: Optional[Callable[[int], None]] = None,
) -> Dataset:
    """Build an in-memory :class:`Dataset` from a subsample of the zip.

    ``min_track_count`` prunes tracks appearing in fewer than that many of the
    loaded playlists (a two-pass filter): this shrinks the catalogue and the
    item-CF co-occurrence matrix dramatically at almost no recall cost, since
    ultra-rare tracks are essentially unrecoverable anyway.  ``keep_uris``, if
    given, restricts the catalogue to that whitelist (e.g. tracks that also
    occur in the challenge set), applied *in addition* to ``min_track_count``.

    Track_uri strings are interned so the 66M-scale interaction lists share one
    copy of each of the ~2.26M unique URIs.
    """
    # ---- pass 1: which tracks clear the frequency threshold ---------------
    if min_track_count > 1 or keep_uris is not None:
        pop: Counter = Counter()
        for obj in iter_raw_playlists(zip_path, max_slices, max_playlists):
            for t in obj.get("tracks", []):
                pop[t["track_uri"]] += 1
        allowed = {u for u, c in pop.items() if c >= min_track_count}
        if keep_uris is not None:
            allowed |= set(keep_uris)
        del pop
    else:
        allowed = None  # keep everything

    # ---- pass 2: materialise the Dataset ----------------------------------
    playlists: List[Playlist] = []
    tracks: Dict[str, TrackMeta] = {}
    intern = {}  # track_uri -> canonical interned string
    n = 0
    for obj in iter_raw_playlists(zip_path, max_slices, max_playlists):
        pl, metas = _playlist_from_json(obj)
        if allowed is not None:
            kept_uris = []
            kept_metas = []
            for uri, m in zip(pl.tracks, metas):
                if uri in allowed:
                    kept_uris.append(uri)
                    kept_metas.append(m)
            pl = Playlist(pid=pl.pid, name=pl.name, tracks=kept_uris)
            metas = kept_metas
        if not pl.tracks:
            continue
        pl.tracks = [intern.setdefault(u, u) for u in pl.tracks]
        playlists.append(pl)
        for m in metas:
            tracks.setdefault(m.track_uri, m)
        n += 1
        if progress is not None and n % 100000 == 0:
            progress(n)
    return Dataset(playlists=playlists, tracks=tracks)


def load_challenge_set(json_path_or_zip: str) -> List[dict]:
    """Load the official ``challenge_set.json`` (raw playlist dicts).

    Accepts either a path to the extracted json or to the challenge zip
    (``spotify_million_playlist_dataset_challenge.zip``), reading the member
    directly in the latter case.
    """
    if json_path_or_zip.endswith(".zip"):
        with zipfile.ZipFile(json_path_or_zip) as zf:
            name = next(n for n in zf.namelist() if n.endswith("challenge_set.json"))
            data = json.loads(zf.read(name))
    else:
        with open(json_path_or_zip, "r", encoding="utf-8") as fh:
            data = json.load(fh)
    return data["playlists"]
