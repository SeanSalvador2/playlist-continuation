"""Load real personal listening histories from a Spotify GDPR data export.

Spotify ships two very different export formats when a user requests their data:

* **Extended streaming history** ("Account data → extended") — one record per play in
  ``Streaming_History_Audio_*.json`` files, with timestamps, URIs, ``ms_played`` and a
  ``skipped`` flag.  This is the rich source :func:`load_extended_history` parses.
* **Basic streaming history** ("Account data") — a thin ``StreamingHistory*.json`` with
  only ``endTime``, ``artistName``, ``trackName`` and ``msPlayed``; no URIs, no skip
  flag.  :func:`load_basic_history` handles it (``track_uri`` becomes ``None``).

Both return the same :class:`~playlistcont.history.schema.ListeningHistory`
container with ``provenance="spotify_export"`` and ``ground_truth=None`` — real data
has no planted truth.

Honesty notes:

* Podcast and video plays appear in the extended export with **null track metadata**;
  we drop them (this modality is about *music* listening).
* The basic export's ``endTime`` has no timezone and Spotify documents it as UTC, so we
  attach UTC; treat sub-day timing as approximate.
* No audio features are present in either export — tracks carry no axis vector until
  you join one (see :mod:`playlistcont.history.features`).
"""
from __future__ import annotations

import glob
import io
import json
import os
import zipfile
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple

from .schema import HistoryTrack, ListenEvent, ListeningHistory

UTC = timezone.utc

_EXTENDED_GLOB = "Streaming_History_Audio_*.json"
_BASIC_GLOB = "StreamingHistory*.json"


def track_id_from_uri(uri: Optional[str]) -> Optional[str]:
    """Extract the bare 22-char id from ``spotify:track:<id>`` (``None`` -> ``None``).

    Matches the join key used by :mod:`playlistcont.data.real_features` (the id after
    the final ``:``), so real audio features can be attached on the shared id.
    """
    if not uri:
        return None
    return uri.rsplit(":", 1)[-1]


def _parse_ts_iso(s: str) -> datetime:
    """Parse an ISO-8601 UTC timestamp (``...Z`` or offset) into aware UTC."""
    dt = datetime.fromisoformat(s.replace("Z", "+00:00"))
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=UTC)
    return dt.astimezone(UTC)


def _read_json_members(path: str, pattern: str) -> List[Tuple[str, object]]:
    """Read every JSON member matching ``pattern`` from a directory or a zip.

    Returns a list of ``(member_name, parsed_json)`` sorted by member name so
    multi-file exports merge deterministically.
    """
    out: List[Tuple[str, object]] = []
    if zipfile.is_zipfile(path):
        with zipfile.ZipFile(path) as zf:
            names = sorted(
                n for n in zf.namelist()
                if _matches(os.path.basename(n), pattern)
            )
            for n in names:
                with zf.open(n) as fh:
                    out.append((n, json.load(io.TextIOWrapper(fh, encoding="utf-8"))))
    elif os.path.isdir(path):
        paths = sorted(glob.glob(os.path.join(path, "**", pattern), recursive=True))
        for p in paths:
            with open(p, encoding="utf-8") as fh:
                out.append((p, json.load(fh)))
    else:
        raise FileNotFoundError(f"{path!r} is neither a zip nor a directory")
    return out


def _matches(name: str, pattern: str) -> bool:
    import fnmatch

    return fnmatch.fnmatch(name, pattern)


def _finalize(events: List[ListenEvent]) -> ListeningHistory:
    events.sort(key=lambda e: e.ts)
    tracks: Dict[str, HistoryTrack] = {}
    for e in events:
        if e.track_uri and e.track_uri not in tracks:
            tracks[e.track_uri] = HistoryTrack(
                track_uri=e.track_uri,
                track_name=e.track_name,
                artist_name=e.artist_name,
                album_name=e.album_name,
            )
    return ListeningHistory(
        events=events, tracks=tracks, provenance="spotify_export", ground_truth=None,
    )


def load_extended_history(path: str) -> ListeningHistory:
    """Load a Spotify *extended* streaming history from a directory or zip.

    Accepts the folder (or ``.zip``) containing ``Streaming_History_Audio_*.json``.
    Records with null ``master_metadata_track_name`` (podcast/video rows) are dropped.
    """
    members = _read_json_members(path, _EXTENDED_GLOB)
    events: List[ListenEvent] = []
    for _name, blob in members:
        if not isinstance(blob, list):
            continue
        for rec in blob:
            track_name = rec.get("master_metadata_track_name")
            if not track_name:
                continue  # podcast / video / unavailable — drop
            events.append(ListenEvent(
                ts=_parse_ts_iso(rec["ts"]),
                track_uri=rec.get("spotify_track_uri"),
                track_name=track_name,
                artist_name=rec.get("master_metadata_album_artist_name") or "",
                ms_played=int(rec.get("ms_played") or 0),
                album_name=rec.get("master_metadata_album_album_name"),
                skipped=rec.get("skipped"),
                platform=rec.get("platform"),
            ))
    return _finalize(events)


def load_basic_history(path: str) -> ListeningHistory:
    """Load a Spotify *basic* streaming history (``StreamingHistory*.json``).

    The basic export has no URIs, so every event's ``track_uri`` is ``None``; ``ts`` is
    parsed from ``endTime`` (``"%Y-%m-%d %H:%M"``) and attached as UTC.
    """
    members = _read_json_members(path, _BASIC_GLOB)
    events: List[ListenEvent] = []
    for _name, blob in members:
        if not isinstance(blob, list):
            continue
        for rec in blob:
            track_name = rec.get("trackName")
            if not track_name:
                continue
            ts = datetime.strptime(rec["endTime"], "%Y-%m-%d %H:%M").replace(tzinfo=UTC)
            events.append(ListenEvent(
                ts=ts,
                track_uri=None,
                track_name=track_name,
                artist_name=rec.get("artistName") or "",
                ms_played=int(rec.get("msPlayed") or 0),
                album_name=None,
                skipped=None,
                platform=None,
            ))
    return _finalize(events)
