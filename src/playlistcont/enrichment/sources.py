"""Download-and-cache helpers for the two license-clean enrichment tables.

Two sources, both identified and measured in the Phase 1.5 enrichment research
(see ``ENRICHMENT.md`` at the repo root):

* **``maharshipandya/spotify-tracks-dataset``** (BSD) — 114k rows / ~89.7k unique
  Spotify ``track_id``s with a ``track_genre`` seed-genre label plus the full
  standard audio-feature set.  Joins on the exact 22-char Spotify track id.
* **``LeData/media-metadata-musicbrainz-artists``** (CC0-1.0) — 1.52M MusicBrainz
  artist entities with a user-editable ``tags`` folksonomy field.  Joins on
  normalized artist *name* (no Spotify id on either side).

Network discipline: nothing here touches the network at import time, and no test
should ever call the ``download_*``/``load_*`` functions without an injected
frame.  Downloads happen only when these functions are explicitly called, and the
files are cached under ``.data/enrichment/`` (gitignored via the existing
``.data/`` rule) so each table is fetched at most once.
"""
from __future__ import annotations

import os
from typing import Optional

TRACK_GENRE_REPO = "maharshipandya/spotify-tracks-dataset"
TRACK_GENRE_FILE = "dataset.csv"

ARTIST_TAG_REPO = "LeData/media-metadata-musicbrainz-artists"
ARTIST_TAG_FILE = "data/train-00000-of-00001.parquet"

DEFAULT_CACHE_DIR = os.path.join(".data", "enrichment")
CACHE_ENV = "PLAYLISTCONT_ENRICHMENT_CACHE"


def cache_dir(override: Optional[str] = None) -> str:
    """Resolve the cache directory: explicit arg > ``$PLAYLISTCONT_ENRICHMENT_CACHE``
    > ``.data/enrichment`` relative to the current working directory."""
    return override or os.environ.get(CACHE_ENV) or DEFAULT_CACHE_DIR


def _download(repo_id: str, filename: str, cache: Optional[str]) -> str:
    """Fetch ``filename`` from a HF dataset repo into the cache dir (once).

    Returns the local path.  Skips the network entirely when the file is already
    cached — so repeated startups with ``PLAYLISTCONT_ENRICH=1`` download nothing.
    """
    root = cache_dir(cache)
    local = os.path.join(root, repo_id.replace("/", "__"), os.path.basename(filename))
    if os.path.exists(local) and os.path.getsize(local) > 0:
        return local
    os.makedirs(os.path.dirname(local), exist_ok=True)
    from huggingface_hub import hf_hub_download

    got = hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        repo_type="dataset",
        local_dir=os.path.join(root, repo_id.replace("/", "__"), "_hf"),
    )
    # move to the flat, stable location we check above
    os.replace(got, local)
    return local


def download_track_genre_table(cache: Optional[str] = None) -> str:
    """Download (or reuse) the maharshipandya track->seed-genre CSV; return its path."""
    return _download(TRACK_GENRE_REPO, TRACK_GENRE_FILE, cache)


def download_artist_tag_table(cache: Optional[str] = None) -> str:
    """Download (or reuse) the LeData MusicBrainz artist-tags parquet; return its path."""
    return _download(ARTIST_TAG_REPO, ARTIST_TAG_FILE, cache)


def load_track_genre_table(cache: Optional[str] = None):
    """Load the track->seed-genre table as a DataFrame (downloading if needed).

    Columns of interest: ``track_id`` (22-char Spotify id), ``track_genre`` (the
    SEED genre the track was fetched under — see ``track_genres.py`` for the
    caveat), ``artists``, ``track_name``.
    """
    import pandas as pd

    return pd.read_csv(download_track_genre_table(cache))


def load_artist_tag_table(cache: Optional[str] = None):
    """Load the MusicBrainz artist table as a DataFrame (downloading if needed).

    Columns of interest: ``name``, ``tags`` (array of folksonomy strings,
    empty/None for ~79% of rows), ``mb_id``.
    """
    import pandas as pd

    return pd.read_parquet(download_artist_tag_table(cache))
