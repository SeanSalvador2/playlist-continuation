"""Bridge real Spotify audio features onto the interpretable taste-engine axes.

The real MPD carries no audio features, so the taste engine cannot run on it out
of the box.  We join a public audio-features table (Spotify track_id -> tempo,
energy, valence, acousticness, instrumentalness, ...) onto the MPD track_uris on
the shared 22-character track id.

Axis mapping (documented, honest about the proxies):

======================  ==========================================================
Taste-engine axis       Real-MPD source
======================  ==========================================================
tempo                   audio ``tempo`` BPM, min-max squashed to [0, 1] (30..250)
energy                  audio ``energy`` (already in [0, 1])
valence                 audio ``valence`` (already in [0, 1])
acousticness            audio ``acousticness`` (already in [0, 1])
lyrical_depth           PROXY: ``1 - instrumentalness`` (presence of vocals; the
                        dataset has no lyric-content signal, so this is a rough
                        stand-in and is reported as such)
genre:*                 UNAVAILABLE in the audio table -> left at 0 (the MPD has
                        no per-track genre; the genre axes simply never fire)
======================  ==========================================================

The parquet/CSV is streamed by row-group and filtered to the tracks we actually
need, so peak memory stays bounded even though the source table has tens of
millions of rows.
"""
from __future__ import annotations

import glob
import os
from typing import Dict, Iterable, List, Optional, Set

import numpy as np

from .schema import AXES, N_AXES, SCALAR_AXES, Dataset

# indices of the scalar axes inside AXES (genres follow, left at 0)
_AX = {a: i for i, a in enumerate(AXES)}


def _uri_to_id(uri: str) -> str:
    return uri.rsplit(":", 1)[-1]


def _row_to_axis(tempo, energy, valence, acousticness, instrumentalness) -> np.ndarray:
    v = np.zeros(N_AXES, dtype=np.float32)
    # neutral 0.5 for scalar axes by default, then fill what we have
    for a in SCALAR_AXES:
        v[_AX[a]] = 0.5
    if tempo is not None and np.isfinite(tempo):
        v[_AX["tempo"]] = float(np.clip((tempo - 30.0) / (250.0 - 30.0), 0.0, 1.0))
    if energy is not None and np.isfinite(energy):
        v[_AX["energy"]] = float(np.clip(energy, 0.0, 1.0))
    if valence is not None and np.isfinite(valence):
        v[_AX["valence"]] = float(np.clip(valence, 0.0, 1.0))
    if acousticness is not None and np.isfinite(acousticness):
        v[_AX["acousticness"]] = float(np.clip(acousticness, 0.0, 1.0))
    if instrumentalness is not None and np.isfinite(instrumentalness):
        v[_AX["lyrical_depth"]] = float(np.clip(1.0 - instrumentalness, 0.0, 1.0))
    return v


def build_feature_table(
    source: str,
    needed_ids: Optional[Set[str]] = None,
    progress=None,
) -> Dict[str, np.ndarray]:
    """Stream an audio-features parquet (or dir of parquets) into id->axis vec.

    Only rows whose id is in ``needed_ids`` are kept (pass ``None`` to keep all,
    which is memory-heavy for large tables).  Returns a dict keyed by the bare
    22-char track id (not the full ``spotify:track:`` uri).
    """
    import pyarrow.parquet as pq

    if os.path.isdir(source):
        paths = sorted(glob.glob(os.path.join(source, "**", "*.parquet"),
                                 recursive=True))
    else:
        paths = [source]
    cols = ["id", "tempo", "energy", "valence", "acousticness", "instrumentalness"]
    out: Dict[str, np.ndarray] = {}
    seen_rows = 0
    for path in paths:
        pf = pq.ParquetFile(path)
        for rg in range(pf.num_row_groups):
            tbl = pf.read_row_group(rg, columns=cols)
            ids = tbl.column("id").to_pylist()
            tempo = tbl.column("tempo").to_pylist()
            energy = tbl.column("energy").to_pylist()
            valence = tbl.column("valence").to_pylist()
            acou = tbl.column("acousticness").to_pylist()
            instr = tbl.column("instrumentalness").to_pylist()
            for i, tid in enumerate(ids):
                if needed_ids is not None and tid not in needed_ids:
                    continue
                if tid in out:
                    continue
                out[tid] = _row_to_axis(tempo[i], energy[i], valence[i],
                                        acou[i], instr[i])
            seen_rows += len(ids)
            if progress is not None:
                progress(seen_rows, len(out))
    return out


# ---------------------------------------------------------------------------
# Widened read (Phase 1.5 enrichment) — ADDITIVE.  ``build_feature_table`` /
# ``attach_real_features`` are frozen (the recsys pillar depends on their exact
# behaviour); this function is a separate, wider streaming read used by the
# personal-history modality.
# ---------------------------------------------------------------------------

# All useful columns in the audio-features table (measured schema: 17 columns;
# we skip only ``null_response``, a data-quality flag that nulls the whole
# feature row when set).
EXTENDED_COLUMNS: List[str] = [
    "id", "name", "popularity", "duration_ms", "time_signature", "key",
    "mode", "tempo", "danceability", "energy", "loudness", "speechiness",
    "acousticness", "instrumentalness", "liveness", "valence",
]


def build_extended_feature_frame(
    source: str,
    needed_ids: Optional[Set[str]] = None,
    progress=None,
):
    """Stream the audio-features parquet(s) into a wide per-track DataFrame.

    Same row-group streaming and id-filtering discipline as
    :func:`build_feature_table`, but reads **all** useful columns
    (:data:`EXTENDED_COLUMNS`) instead of the frozen 6-column subset.  Returns a
    ``pandas.DataFrame`` with one row per matched id (first occurrence wins,
    matching ``build_feature_table``'s dedup rule).  ``progress(seen_rows,
    n_matched)`` is called after each row group when given.

    Missing columns in a source file are tolerated (filled with nulls) so tiny
    test fixtures do not need the full 16-column schema.
    """
    import pandas as pd
    import pyarrow.parquet as pq

    if os.path.isdir(source):
        paths = sorted(glob.glob(os.path.join(source, "**", "*.parquet"),
                                 recursive=True))
    else:
        paths = [source]

    frames: List["pd.DataFrame"] = []
    seen: Set[str] = set()
    seen_rows = 0
    for path in paths:
        pf = pq.ParquetFile(path)
        avail = set(pf.schema_arrow.names)
        cols = [c for c in EXTENDED_COLUMNS if c in avail]
        if "id" not in cols:
            raise ValueError(f"{path} has no 'id' column")
        for rg in range(pf.num_row_groups):
            df = pf.read_row_group(rg, columns=cols).to_pandas()
            seen_rows += len(df)
            if needed_ids is not None:
                df = df[df["id"].isin(needed_ids)]
            if len(df):
                df = df[~df["id"].isin(seen)].drop_duplicates("id", keep="first")
            if len(df):
                seen.update(df["id"].tolist())
                frames.append(df)
            if progress is not None:
                progress(seen_rows, len(seen))

    if frames:
        out = pd.concat(frames, ignore_index=True)
    else:
        out = pd.DataFrame(columns=EXTENDED_COLUMNS)
    # tolerate schema-poor sources: guarantee every extended column exists
    for c in EXTENDED_COLUMNS:
        if c not in out.columns:
            out[c] = None
    return out[EXTENDED_COLUMNS].reset_index(drop=True)


def attach_real_features(ds: Dataset, source: str, progress=None) -> int:
    """Attach audio-feature axis vectors to a real-MPD ``Dataset`` in place.

    Returns the number of catalogue tracks that were matched.  Sets
    ``ds.features`` (uri -> axis vector) only for matched tracks; the taste
    engine's ``feature_matrix`` fills unmatched tracks with zeros.
    """
    needed_ids = {_uri_to_id(u) for u in ds.tracks}
    id_feats = build_feature_table(source, needed_ids=needed_ids, progress=progress)
    features: Dict[str, np.ndarray] = {}
    for uri in ds.tracks:
        vec = id_feats.get(_uri_to_id(uri))
        if vec is not None:
            features[uri] = vec
    ds.features = features
    return len(features)
