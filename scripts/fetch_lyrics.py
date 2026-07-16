#!/usr/bin/env python3
"""Fetch lyrics for a personal music library from LRCLIB (lrclib.net).

**Standalone script** — Python 3.9+ standard library only, zero third-party
dependencies, single file. Copy it anywhere and run it; it does not import
anything from the ``playlistcont`` package.

Usage
-----

From a Spotify GDPR data export (the folder or ``.zip`` Spotify emails you;
either the "basic" ``StreamingHistory*.json`` or "extended"
``Streaming_History_Audio_*.json`` files — both are handled, and most-played
tracks are fetched first)::

    python scripts/fetch_lyrics.py --export ~/Downloads/my_spotify_data --out .data/lyrics_cache

From a plain track list — a CSV with an ``artist,track[,album]`` header, or a
text file of ``Artist - Track`` lines, one per line::

    python scripts/fetch_lyrics.py --tracks my_tracks.csv --out .data/lyrics_cache

Useful flags: ``--limit 25`` for a trial run before committing to the whole
library, ``--force`` to refetch tracks that already have a cache file,
``--delay 2.0`` to be more polite than the 1.5s default, ``--self-test`` to
run the built-in offline test suite (mocked HTTP, no network) instead of
fetching anything.

Output
------

For every track this writes ``<out>/<hash-of-artist+track>.json`` — one
record: ``artist, track, album, spotify_track_id, matched, source,
lrclib_id, instrumental, plain_lyrics, has_synced, fetched_at``. It also
appends one line per attempt to ``<out>/manifest.jsonl`` (an audit trail —
what was requested, what came back, when) and one row per non-match or error
to ``<out>/failures.csv``.

**The cache directory contains copyrighted lyric text and must never be
committed.** The default output location, ``.data/lyrics_cache``, sits under
``.data/`` — this repo's ``.gitignore`` already excludes the whole ``.data/``
tree (see the ``.data/`` entry in ``.gitignore``), so the default is safe as-is.
If you point ``--out`` somewhere else, make sure *that* path is gitignored
too before running. Only derived, non-reconstructive features computed from
this cache (sentiment scores, embeddings, theme labels — never raw lyric
text) belong in version control.

Runtime math: the default 1.5s politeness delay between requests means
roughly 40 tracks/minute, so a personal library of ~5,000 tracks takes on
the order of **~2 hours**. The script is fully resumable (already-cached
tracks are skipped on the next run, ``--force`` overrides), so it is safe to
Ctrl-C and restart, or to run it across several sessions.
"""
from __future__ import annotations

import argparse
import csv
import fnmatch
import glob
import hashlib
import io
import json
import os
import re
import sys
import time
import unittest
import urllib.error
import urllib.parse
import urllib.request
import zipfile
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from unittest import mock

UTC = timezone.utc

LRCLIB_BASE = "https://lrclib.net"
USER_AGENT = (
    "playlist-continuation-taste-atlas/0.1 "
    "(personal research; +https://github.com/seansalvador2/playlist-continuation)"
)

DEFAULT_OUT = ".data/lyrics_cache"
DEFAULT_DELAY = 1.5
DEFAULT_MAX_RETRIES = 3
PROGRESS_EVERY = 25

_EXTENDED_GLOB = "Streaming_History_Audio_*.json"
_BASIC_GLOB = "StreamingHistory*.json"


class LyricsFetchError(Exception):
    """Raised for a network/HTTP failure that survived all retries.

    Distinct from a *definitive* miss (LRCLIB has no match for this track):
    that case is a normal, cacheable result (``matched: False``), not an
    exception. This exception means we genuinely don't know the answer yet —
    the caller does not write a cache file, so the track is retried on the
    next run.
    """


# ---------------------------------------------------------------------------
# Track requests: the unit of work
# ---------------------------------------------------------------------------


@dataclass
class TrackRequest:
    """One track to look up on LRCLIB.

    ``play_count`` drives fetch ordering for export mode (most-played
    first); it is left at 1 for track-list mode, which has no play data.
    """

    artist: str
    track: str
    album: Optional[str] = None
    spotify_track_id: Optional[str] = None
    play_count: int = 1


# ---------------------------------------------------------------------------
# Spotify GDPR export parsing (field names mirror
# src/playlistcont/history/spotify_export.py — copied, not imported, so this
# script stays standalone)
# ---------------------------------------------------------------------------


def _track_id_from_uri(uri: Optional[str]) -> Optional[str]:
    """Extract the bare 22-char id from ``spotify:track:<id>`` (``None`` -> ``None``)."""
    if not uri:
        return None
    return uri.rsplit(":", 1)[-1]


def _iter_export_json(path: str, pattern: str):
    """Yield parsed JSON blobs for every export member matching ``pattern``.

    ``path`` may be a directory (searched recursively) or a ``.zip``, matching
    the two shapes Spotify's data-export download can arrive in.
    """
    if zipfile.is_zipfile(path):
        with zipfile.ZipFile(path) as zf:
            names = sorted(
                n for n in zf.namelist() if fnmatch.fnmatch(os.path.basename(n), pattern)
            )
            for n in names:
                with zf.open(n) as fh:
                    yield json.load(io.TextIOWrapper(fh, encoding="utf-8"))
    elif os.path.isdir(path):
        for p in sorted(glob.glob(os.path.join(path, "**", pattern), recursive=True)):
            with open(p, encoding="utf-8") as fh:
                yield json.load(fh)
    else:
        raise FileNotFoundError(f"{path!r} is neither a zip nor a directory")


def build_track_list_from_export(path: str) -> List[TrackRequest]:
    """Build the unique, play-count-aggregated track list from a Spotify export.

    Handles both the extended (``Streaming_History_Audio_*.json``) and basic
    (``StreamingHistory*.json``) formats, merging both if present. Podcast /
    video rows (null track name) are dropped. Returned list is sorted by
    descending play count so the most-played tracks are fetched first.
    """
    agg: Dict[Tuple[str, str], TrackRequest] = {}

    def add(artist: str, track: str, album: Optional[str], spotify_id: Optional[str]) -> None:
        artist = (artist or "").strip()
        track = (track or "").strip()
        if not artist or not track:
            return
        key = (artist.lower(), track.lower())
        existing = agg.get(key)
        if existing is None:
            agg[key] = TrackRequest(
                artist=artist,
                track=track,
                album=(album.strip() if album else None),
                spotify_track_id=spotify_id,
                play_count=1,
            )
        else:
            existing.play_count += 1
            if album and not existing.album:
                existing.album = album.strip()
            if spotify_id and not existing.spotify_track_id:
                existing.spotify_track_id = spotify_id

    found_any = False
    for blob in _iter_export_json(path, _EXTENDED_GLOB):
        found_any = True
        if not isinstance(blob, list):
            continue
        for rec in blob:
            track_name = rec.get("master_metadata_track_name")
            if not track_name:
                continue  # podcast / video / unavailable — drop, same as spotify_export.py
            add(
                rec.get("master_metadata_album_artist_name") or "",
                track_name,
                rec.get("master_metadata_album_album_name"),
                _track_id_from_uri(rec.get("spotify_track_uri")),
            )
    for blob in _iter_export_json(path, _BASIC_GLOB):
        found_any = True
        if not isinstance(blob, list):
            continue
        for rec in blob:
            track_name = rec.get("trackName")
            if not track_name:
                continue
            add(rec.get("artistName") or "", track_name, None, None)

    if not found_any:
        raise FileNotFoundError(
            f"No {_EXTENDED_GLOB} or {_BASIC_GLOB} files found under {path!r}"
        )

    tracks = list(agg.values())
    tracks.sort(key=lambda t: (-t.play_count, t.artist.lower(), t.track.lower()))
    return tracks


# ---------------------------------------------------------------------------
# Plain track-list parsing: CSV (artist,track[,album]) or "Artist - Track" text
# ---------------------------------------------------------------------------


def load_track_list(path: Path) -> List[TrackRequest]:
    """Parse ``--tracks``: a CSV with an ``artist,track[,album]`` header, or a
    plain-text file of ``Artist - Track`` lines (``#``-prefixed and blank
    lines ignored)."""
    text = path.read_text(encoding="utf-8-sig")
    lines = text.splitlines()
    first_nonblank = next((ln for ln in lines if ln.strip()), "")
    header_fields = {c.strip().lower() for c in first_nonblank.split(",")}
    is_csv = path.suffix.lower() == ".csv" or (
        "," in first_nonblank and {"artist", "track"} <= header_fields
    )

    tracks: List[TrackRequest] = []
    if is_csv:
        reader = csv.DictReader(io.StringIO(text))
        fieldmap = {(k or "").strip().lower(): k for k in (reader.fieldnames or [])}
        if "artist" not in fieldmap or "track" not in fieldmap:
            raise ValueError(
                f"{path}: CSV must have 'artist' and 'track' columns (optional 'album')"
            )
        for row in reader:
            artist = (row.get(fieldmap["artist"]) or "").strip()
            track = (row.get(fieldmap["track"]) or "").strip()
            album_key = fieldmap.get("album")
            album = (row.get(album_key) or "").strip() if album_key else ""
            if not artist or not track:
                continue
            tracks.append(TrackRequest(artist=artist, track=track, album=album or None))
    else:
        for raw_line in lines:
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            if " - " not in line:
                print(f"  (skipping unparseable line: {line!r})", file=sys.stderr)
                continue
            artist, track = line.split(" - ", 1)
            artist, track = artist.strip(), track.strip()
            if artist and track:
                tracks.append(TrackRequest(artist=artist, track=track))
    return tracks


# ---------------------------------------------------------------------------
# Deterministic, filesystem-safe cache filenames
# ---------------------------------------------------------------------------

_SLUG_RE = re.compile(r"[^A-Za-z0-9]+")


def cache_key(artist: str, track: str) -> str:
    """Deterministic, filesystem-safe cache key derived from artist + track.

    Album is deliberately excluded so re-fetching with slightly different
    album metadata still resolves to the same cache entry. A readable slug
    prefix is kept for humans browsing the cache dir; the hash suffix
    guarantees uniqueness and safety regardless of unicode/punctuation.
    """
    norm = f"{artist.strip().lower()}\x1f{track.strip().lower()}"
    digest = hashlib.sha256(norm.encode("utf-8")).hexdigest()[:24]
    slug = _SLUG_RE.sub("_", f"{artist}_{track}").strip("_").lower()[:60]
    return f"{slug}_{digest}" if slug else digest


# ---------------------------------------------------------------------------
# LRCLIB HTTP layer
# ---------------------------------------------------------------------------


class _HttpResult:
    __slots__ = ("status", "body")

    def __init__(self, status: int, body: bytes):
        self.status = status
        self.body = body


def _http_get(url: str, timeout: float = 15.0) -> _HttpResult:
    """One raw HTTP GET. HTTP-level errors (4xx/5xx) come back as a normal
    ``_HttpResult`` with that status code; only genuine network failures
    (DNS, connection refused, timeout) raise :class:`LyricsFetchError`.

    This is the single seam the self-test patches (``urllib.request.urlopen``).
    """
    req = urllib.request.Request(
        url, headers={"User-Agent": USER_AGENT, "Accept": "application/json"}
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            status = getattr(resp, "status", None) or resp.getcode()
            return _HttpResult(status, resp.read())
    except urllib.error.HTTPError as e:
        body = e.read() if hasattr(e, "read") else b""
        return _HttpResult(e.code, body)
    except urllib.error.URLError as e:
        raise LyricsFetchError(f"network error contacting LRCLIB: {e.reason}") from e
    except (TimeoutError, OSError) as e:
        # A read/connect timeout or reset during getresponse()/read() is raised
        # as a bare TimeoutError (or other OSError) on some platforms — NOT wrapped
        # in URLError — so catch it here and let the retry/backoff layer handle it
        # instead of crashing the whole run.
        raise LyricsFetchError(f"network error contacting LRCLIB: {e}") from e


def _backoff_sleep(attempt: int) -> None:
    """Exponential backoff: 2s, 4s, 8s, capped at 60s."""
    time.sleep(min(60.0, 2.0 * (2**attempt)))


def _lrclib_request(
    path: str, params: Dict[str, str], max_retries: int, timeout: float = 15.0
) -> Tuple[int, bytes]:
    """GET ``LRCLIB_BASE + path`` with retries on 429/5xx and on transient
    network errors. Returns ``(status, body)`` for any response the caller
    should interpret itself (200, 404, or a still-bad status after retries
    are exhausted); raises :class:`LyricsFetchError` only if every retry hit
    a network-level (not HTTP-level) failure.
    """
    url = f"{LRCLIB_BASE}{path}?{urllib.parse.urlencode(params)}"
    attempt = 0
    while True:
        try:
            result = _http_get(url, timeout=timeout)
        except LyricsFetchError:
            if attempt >= max_retries:
                raise
            _backoff_sleep(attempt)
            attempt += 1
            continue

        if result.status in (200, 404):
            return result.status, result.body
        if result.status == 429 or 500 <= result.status < 600:
            if attempt >= max_retries:
                return result.status, result.body
            _backoff_sleep(attempt)
            attempt += 1
            continue
        return result.status, result.body  # some other 4xx — no point retrying


_WORD_RE = re.compile(r"[^a-z0-9]+")


def _norm_name(s: Optional[str]) -> str:
    return _WORD_RE.sub("", (s or "").lower())


def _pick_best_candidate(candidates: List[dict], t: TrackRequest) -> Optional[dict]:
    """From ``/api/search`` results, prefer an exact-ish artist+title match;
    otherwise fall back to the first (LRCLIB's own relevance ordering)."""
    if not candidates:
        return None
    target_artist, target_track = _norm_name(t.artist), _norm_name(t.track)
    for c in candidates:
        if _norm_name(c.get("artistName")) == target_artist and _norm_name(c.get("trackName")) == target_track:
            return c
    return candidates[0]


def _build_result(t: TrackRequest, data: dict, source_endpoint: str, fetched_at: str) -> dict:
    return {
        "artist": t.artist,
        "track": t.track,
        "album": t.album,
        "spotify_track_id": t.spotify_track_id,
        "matched": True,
        "source": "lrclib",
        "source_endpoint": source_endpoint,
        "lrclib_id": data.get("id"),
        "instrumental": bool(data.get("instrumental", False)),
        "plain_lyrics": data.get("plainLyrics"),
        "has_synced": bool(data.get("syncedLyrics")),
        "fetched_at": fetched_at,
    }


def _no_match_result(t: TrackRequest, fetched_at: str) -> dict:
    return {
        "artist": t.artist,
        "track": t.track,
        "album": t.album,
        "spotify_track_id": t.spotify_track_id,
        "matched": False,
        "source": "lrclib",
        "source_endpoint": None,
        "lrclib_id": None,
        "instrumental": False,
        "plain_lyrics": None,
        "has_synced": False,
        "fetched_at": fetched_at,
    }


def fetch_lyrics(t: TrackRequest, delay: float, max_retries: int) -> dict:
    """Fetch one track's lyrics from LRCLIB: ``/api/get`` first, ``/api/search``
    fallback on 404. Raises :class:`LyricsFetchError` if LRCLIB is
    unreachable/erroring after retries — that is NOT a definitive miss and
    should not be cached as one.
    """
    fetched_at = datetime.now(UTC).isoformat()

    get_params = {"artist_name": t.artist, "track_name": t.track}
    if t.album:
        get_params["album_name"] = t.album
    status, body = _lrclib_request("/api/get", get_params, max_retries=max_retries)

    if status == 200:
        return _build_result(t, json.loads(body), "lrclib_get", fetched_at)

    if status == 404:
        search_params = {"track_name": t.track, "artist_name": t.artist}
        s_status, s_body = _lrclib_request("/api/search", search_params, max_retries=max_retries)
        if s_status == 200:
            candidates = json.loads(s_body)
            best = _pick_best_candidate(candidates, t)
            if best is not None:
                return _build_result(t, best, "lrclib_search", fetched_at)
            return _no_match_result(t, fetched_at)
        if s_status == 404:
            return _no_match_result(t, fetched_at)
        raise LyricsFetchError(
            f"LRCLIB /api/search returned unexpected status {s_status} for "
            f"{t.artist!r} - {t.track!r}"
        )

    raise LyricsFetchError(
        f"LRCLIB /api/get returned unexpected status {status} for {t.artist!r} - {t.track!r}"
    )


# ---------------------------------------------------------------------------
# Cache I/O
# ---------------------------------------------------------------------------


def _atomic_write_json(path: Path, obj: dict) -> None:
    """Write via a temp file + rename so a Ctrl-C mid-write can never leave a
    half-written (corrupt) cache file behind."""
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")
    tmp.replace(path)


def _append_manifest(fh, entry: dict) -> None:
    fh.write(json.dumps(entry, ensure_ascii=False) + "\n")
    fh.flush()


def _append_failure(path: Path, artist: str, track: str, album: str, reason: str) -> None:
    is_new = not path.exists()
    with path.open("a", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        if is_new:
            writer.writerow(["artist", "track", "album", "reason"])
        writer.writerow([artist, track, album, reason])


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


def run(args: argparse.Namespace) -> Dict[str, int]:
    """Run the full fetch loop. Returns a stats dict (also printed as a
    summary) so callers — including the self-test — can assert on outcomes.
    """
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = out_dir / "manifest.jsonl"
    failures_path = out_dir / "failures.csv"

    if args.export:
        tracks = build_track_list_from_export(args.export)
    else:
        tracks = load_track_list(Path(args.tracks))

    if args.limit is not None:
        tracks = tracks[: args.limit]

    total = len(tracks)
    print(f"Loaded {total} unique tracks.")

    stats: Counter = Counter()
    manifest_fh = manifest_path.open("a", encoding="utf-8")
    try:
        for i, t in enumerate(tracks, start=1):
            key = cache_key(t.artist, t.track)
            cache_file = out_dir / f"{key}.json"

            if cache_file.exists() and not args.force:
                stats["skipped"] += 1
                continue

            try:
                result = fetch_lyrics(t, delay=args.delay, max_retries=args.max_retries)
            except Exception as e:
                # Never let one track (a network blip, an odd response, anything)
                # kill a multi-hour run — log it as a failure and keep going.
                # No cache file is written, so a rerun retries it automatically.
                stats["attempted"] += 1
                stats["errors"] += 1
                stats["failed"] += 1
                _append_manifest(manifest_fh, {
                    "artist": t.artist, "track": t.track, "album": t.album,
                    "cache_key": key, "matched": False, "error": str(e),
                    "fetched_at": datetime.now(UTC).isoformat(),
                })
                _append_failure(failures_path, t.artist, t.track, t.album or "", f"error: {e}")
            else:
                _atomic_write_json(cache_file, result)
                _append_manifest(manifest_fh, {**result, "cache_key": key})
                stats["attempted"] += 1
                if result["instrumental"]:
                    stats["instrumental"] += 1
                elif result["matched"]:
                    stats["matched"] += 1
                else:
                    stats["failed"] += 1
                    _append_failure(failures_path, t.artist, t.track, t.album or "", "no_match")

            if i % PROGRESS_EVERY == 0 or i == total:
                print(
                    f"[{i}/{total}] matched={stats['matched']} instrumental={stats['instrumental']} "
                    f"failed={stats['failed']} skipped={stats['skipped']}"
                )

            if i < total:
                time.sleep(args.delay)
    except KeyboardInterrupt:
        print(
            "\nInterrupted — the current track's write already completed atomically. "
            "Re-run the same command to resume; already-cached tracks are skipped.",
            file=sys.stderr,
        )
    finally:
        manifest_fh.close()

    attempted = stats["attempted"]
    hit = stats["matched"] + stats["instrumental"]
    hit_rate = (hit / attempted * 100.0) if attempted else 0.0
    print(
        "\n--- summary ---\n"
        f"attempted:    {attempted}\n"
        f"matched:      {stats['matched']}\n"
        f"instrumental: {stats['instrumental']}\n"
        f"failed:       {stats['failed']}\n"
        f"skipped (already cached): {stats['skipped']}\n"
        f"hit rate:     {hit_rate:.1f}%"
    )
    return dict(stats)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="fetch_lyrics.py",
        description="Fetch lyrics for a personal music library from LRCLIB (lrclib.net).",
    )
    src = parser.add_mutually_exclusive_group()
    src.add_argument("--export", help="Spotify GDPR data export folder or .zip")
    src.add_argument("--tracks", help="CSV (artist,track[,album]) or 'Artist - Track' text file")
    parser.add_argument("--out", default=DEFAULT_OUT, help=f"cache output dir (default: {DEFAULT_OUT})")
    parser.add_argument("--delay", type=float, default=DEFAULT_DELAY,
                        help=f"seconds between requests (default: {DEFAULT_DELAY})")
    parser.add_argument("--max-retries", type=int, default=DEFAULT_MAX_RETRIES,
                        help=f"max retries on 429/5xx (default: {DEFAULT_MAX_RETRIES})")
    parser.add_argument("--limit", type=int, default=None, help="only process the first N tracks (trial runs)")
    parser.add_argument("--force", action="store_true", help="refetch even if a cache file already exists")
    parser.add_argument("--self-test", action="store_true", help="run the offline self-test suite and exit")
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.self_test:
        return run_self_test()

    if not args.export and not args.tracks:
        parser.error("one of --export or --tracks is required (or pass --self-test)")

    run(args)
    return 0


# ---------------------------------------------------------------------------
# Self-test: offline, mocks urllib.request.urlopen — no network required
# ---------------------------------------------------------------------------


class _FakeHTTPResponse:
    """Minimal stand-in for the object ``urlopen()`` returns on success."""

    def __init__(self, body: bytes, status: int = 200):
        self._body = body
        self.status = status

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def read(self) -> bytes:
        return self._body

    def getcode(self) -> int:
        return self.status


def _http_error(url: str, code: int, payload: bytes = b"{}") -> urllib.error.HTTPError:
    return urllib.error.HTTPError(url, code, "status", None, io.BytesIO(payload))


class FetchLogicTests(unittest.TestCase):
    def test_get_success(self):
        body = json.dumps({
            "id": 42, "trackName": "Someone Like You", "artistName": "Adele",
            "albumName": "21", "instrumental": False,
            "plainLyrics": "I heard that you're settled down...",
            "syncedLyrics": "[00:01.00] I heard that you're settled down",
        }).encode()
        with mock.patch("urllib.request.urlopen", return_value=_FakeHTTPResponse(body)):
            t = TrackRequest(artist="Adele", track="Someone Like You", album="21")
            result = fetch_lyrics(t, delay=0, max_retries=3)
        self.assertTrue(result["matched"])
        self.assertFalse(result["instrumental"])
        self.assertEqual(result["lrclib_id"], 42)
        self.assertTrue(result["has_synced"])
        self.assertEqual(result["source_endpoint"], "lrclib_get")

    def test_404_falls_back_to_search(self):
        search_body = json.dumps([{
            "id": 7, "trackName": "Hello", "artistName": "Adele",
            "instrumental": False, "plainLyrics": "hello it's me",
            "syncedLyrics": None,
        }]).encode()
        responses = [
            _http_error("https://lrclib.net/api/get", 404),
            _FakeHTTPResponse(search_body),
        ]
        with mock.patch("urllib.request.urlopen", side_effect=responses):
            t = TrackRequest(artist="Adele", track="Hello")
            result = fetch_lyrics(t, delay=0, max_retries=3)
        self.assertTrue(result["matched"])
        self.assertEqual(result["source_endpoint"], "lrclib_search")
        self.assertEqual(result["lrclib_id"], 7)

    def test_404_then_empty_search_is_a_clean_no_match(self):
        responses = [
            _http_error("https://lrclib.net/api/get", 404),
            _FakeHTTPResponse(b"[]"),
        ]
        with mock.patch("urllib.request.urlopen", side_effect=responses):
            t = TrackRequest(artist="Nobody", track="Obscure B-Side")
            result = fetch_lyrics(t, delay=0, max_retries=3)
        self.assertFalse(result["matched"])
        self.assertIsNone(result["plain_lyrics"])

    def test_instrumental_is_matched_not_failed(self):
        body = json.dumps({
            "id": 99, "trackName": "Interlude", "artistName": "Some Band",
            "instrumental": True, "plainLyrics": None, "syncedLyrics": None,
        }).encode()
        with mock.patch("urllib.request.urlopen", return_value=_FakeHTTPResponse(body)):
            t = TrackRequest(artist="Some Band", track="Interlude")
            result = fetch_lyrics(t, delay=0, max_retries=3)
        self.assertTrue(result["matched"])
        self.assertTrue(result["instrumental"])
        self.assertIsNone(result["plain_lyrics"])

    def test_429_retries_then_succeeds(self):
        success_body = json.dumps({
            "id": 1, "trackName": "T", "artistName": "A",
            "instrumental": False, "plainLyrics": "x", "syncedLyrics": None,
        }).encode()
        responses = [_http_error("u", 429), _FakeHTTPResponse(success_body)]
        with mock.patch("urllib.request.urlopen", side_effect=responses), \
             mock.patch("time.sleep") as sleep_mock:
            t = TrackRequest(artist="A", track="T")
            result = fetch_lyrics(t, delay=0, max_retries=3)
        self.assertTrue(result["matched"])
        self.assertEqual(sleep_mock.call_count, 1)

    def test_retries_exhausted_raises(self):
        responses = [_http_error("u", 500)] * 4  # initial + 3 retries
        with mock.patch("urllib.request.urlopen", side_effect=responses), \
             mock.patch("time.sleep") as sleep_mock:
            t = TrackRequest(artist="A", track="T")
            with self.assertRaises(LyricsFetchError):
                fetch_lyrics(t, delay=0, max_retries=3)
        self.assertEqual(sleep_mock.call_count, 3)


class ExportParsingTests(unittest.TestCase):
    def test_extended_export_aggregates_counts_and_sorts(self):
        import tempfile
        with tempfile.TemporaryDirectory() as d:
            records = [
                {"ts": "2024-01-01T00:00:00Z", "master_metadata_track_name": "Hello",
                 "master_metadata_album_artist_name": "Adele",
                 "master_metadata_album_album_name": "25",
                 "spotify_track_uri": "spotify:track:abc123", "ms_played": 200000},
                {"ts": "2024-01-02T00:00:00Z", "master_metadata_track_name": "Hello",
                 "master_metadata_album_artist_name": "Adele",
                 "master_metadata_album_album_name": "25",
                 "spotify_track_uri": "spotify:track:abc123", "ms_played": 200000},
                {"ts": "2024-01-03T00:00:00Z", "master_metadata_track_name": "Alright",
                 "master_metadata_album_artist_name": "Janet Jackson",
                 "master_metadata_album_album_name": None,
                 "spotify_track_uri": "spotify:track:xyz789", "ms_played": 150000},
                {"ts": "2024-01-04T00:00:00Z", "master_metadata_track_name": None,
                 "master_metadata_album_artist_name": None,
                 "episode_name": "Some Podcast", "ms_played": 900000},
            ]
            fp = Path(d) / "Streaming_History_Audio_0.json"
            fp.write_text(json.dumps(records), encoding="utf-8")
            tracks = build_track_list_from_export(d)
        self.assertEqual(len(tracks), 2)
        self.assertEqual(tracks[0].track, "Hello")
        self.assertEqual(tracks[0].play_count, 2)
        self.assertEqual(tracks[0].spotify_track_id, "abc123")
        self.assertEqual(tracks[0].album, "25")
        self.assertEqual(tracks[1].track, "Alright")
        self.assertEqual(tracks[1].play_count, 1)

    def test_basic_export_parses_without_uris(self):
        import tempfile
        with tempfile.TemporaryDirectory() as d:
            records = [
                {"endTime": "2024-01-01 00:00", "artistName": "Sublime",
                 "trackName": "What I Got", "msPlayed": 180000},
            ]
            fp = Path(d) / "StreamingHistory0.json"
            fp.write_text(json.dumps(records), encoding="utf-8")
            tracks = build_track_list_from_export(d)
        self.assertEqual(len(tracks), 1)
        self.assertEqual(tracks[0].track, "What I Got")
        self.assertIsNone(tracks[0].spotify_track_id)
        self.assertIsNone(tracks[0].album)

    def test_missing_export_files_raises(self):
        import tempfile
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(FileNotFoundError):
                build_track_list_from_export(d)


class TrackListParsingTests(unittest.TestCase):
    def test_csv_with_header(self):
        import tempfile
        with tempfile.TemporaryDirectory() as d:
            fp = Path(d) / "tracks.csv"
            fp.write_text("Artist,Track,Album\nAdele,Hello,25\nSublime,What I Got,\n", encoding="utf-8")
            tracks = load_track_list(fp)
        self.assertEqual(len(tracks), 2)
        self.assertEqual(tracks[0].artist, "Adele")
        self.assertEqual(tracks[0].album, "25")
        self.assertIsNone(tracks[1].album)

    def test_plain_text_lines(self):
        import tempfile
        with tempfile.TemporaryDirectory() as d:
            fp = Path(d) / "tracks.txt"
            fp.write_text("# my favorites\nAdele - Hello\n\nSublime - What I Got\n", encoding="utf-8")
            tracks = load_track_list(fp)
        self.assertEqual(len(tracks), 2)
        self.assertEqual(tracks[0].artist, "Adele")
        self.assertEqual(tracks[0].track, "Hello")


class CacheKeyTests(unittest.TestCase):
    def test_deterministic_and_safe(self):
        k1 = cache_key("Adele", "Hello")
        k2 = cache_key("Adele", "Hello")
        k3 = cache_key("Adele", "Someone Like You")
        self.assertEqual(k1, k2)
        self.assertNotEqual(k1, k3)
        self.assertRegex(k1, r"^[A-Za-z0-9_.-]+$")

    def test_case_and_whitespace_insensitive(self):
        self.assertEqual(cache_key(" Adele ", "hello"), cache_key("adele", "Hello "))


class EndToEndRunTests(unittest.TestCase):
    def test_fetch_then_resume_skips_without_network(self):
        import tempfile
        success_body = json.dumps({
            "id": 1, "trackName": "T", "artistName": "A",
            "instrumental": False, "plainLyrics": "x", "syncedLyrics": None,
        }).encode()
        with tempfile.TemporaryDirectory() as d:
            tracks_fp = Path(d) / "tracks.txt"
            tracks_fp.write_text("Artist A - Track One\nArtist B - Track Two\n", encoding="utf-8")
            out_dir = Path(d) / "cache"

            args = argparse.Namespace(
                export=None, tracks=str(tracks_fp), out=str(out_dir),
                delay=0, max_retries=3, limit=None, force=False,
            )
            with mock.patch("urllib.request.urlopen", return_value=_FakeHTTPResponse(success_body)):
                stats = run(args)
            self.assertEqual(stats["attempted"], 2)
            self.assertEqual(stats["matched"], 2)
            self.assertEqual(stats.get("skipped", 0), 0)
            cache_files = list(out_dir.glob("*.json"))
            self.assertEqual(len(cache_files), 2)
            self.assertTrue((out_dir / "manifest.jsonl").exists())

            def _boom(*a, **kw):
                raise AssertionError("network should not be hit on a resumed run")

            with mock.patch("urllib.request.urlopen", side_effect=_boom):
                stats2 = run(args)
            self.assertEqual(stats2.get("skipped", 0), 2)
            self.assertEqual(stats2.get("attempted", 0), 0)

    def test_force_refetches_cached_track(self):
        import tempfile
        success_body = json.dumps({
            "id": 1, "trackName": "T", "artistName": "A",
            "instrumental": False, "plainLyrics": "x", "syncedLyrics": None,
        }).encode()
        with tempfile.TemporaryDirectory() as d:
            tracks_fp = Path(d) / "tracks.txt"
            tracks_fp.write_text("Artist A - Track One\n", encoding="utf-8")
            out_dir = Path(d) / "cache"
            args = argparse.Namespace(
                export=None, tracks=str(tracks_fp), out=str(out_dir),
                delay=0, max_retries=3, limit=None, force=False,
            )
            with mock.patch("urllib.request.urlopen", return_value=_FakeHTTPResponse(success_body)):
                run(args)
            args.force = True
            with mock.patch("urllib.request.urlopen", return_value=_FakeHTTPResponse(success_body)) as m:
                stats = run(args)
            self.assertEqual(stats["attempted"], 1)
            self.assertTrue(m.called)


def run_self_test() -> int:
    """Run the whole in-file test suite (no network) and return a process exit code."""
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    for obj in list(globals().values()):
        if isinstance(obj, type) and issubclass(obj, unittest.TestCase):
            suite.addTests(loader.loadTestsFromTestCase(obj))
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    sys.exit(main())
