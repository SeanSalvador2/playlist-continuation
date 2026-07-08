"""Taste Engine on real MPD, using real Spotify audio features.

The MPD has no audio features, so we join a public audio-features table (see
``playlistcont.data.real_features``) onto the track_uris.  Match rate is
reported honestly.  We then restrict the corpus to the *feature-covered*
tracks (the "matched subset") so the taste engine operates on a fully-featured
catalogue, and compare it against reference models (popularity, item-CF, title)
on the standard 10-scenario protocol.

    python experiments/run_taste_real.py --zip .data/....zip \
        --features .data/af/data --n-corpus 150000

Writes ``results/real/taste_real.csv`` + a small figure.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import warnings
from collections import Counter, defaultdict

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "src"))
REAL_DIR = os.path.join(ROOT, "results", "real")
FIG_DIR = os.path.join(REAL_DIR, "figures")
VIZ = os.path.join(ROOT, "visualizations")

from playlistcont.data.schema import Dataset, Playlist  # noqa: E402
from playlistcont.data.mpd_zip import load_mpd_zip  # noqa: E402
from playlistcont.data.real_features import attach_real_features  # noqa: E402
from playlistcont.challenge.scenarios import (  # noqa: E402
    build_scenarios, train_test_split, SCENARIO_NAMES,
)
from playlistcont.challenge.metrics import evaluate_one, aggregate  # noqa: E402
from playlistcont.models.popularity import PopularityRecommender  # noqa: E402
from playlistcont.models.itemcf import ItemCFRecommender  # noqa: E402
from playlistcont.models.title_model import TitleModelRecommender  # noqa: E402
from playlistcont.models.taste_engine import TasteEngine  # noqa: E402


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def filter_to_features(ds: Dataset) -> Dataset:
    """Keep only tracks that have an audio-feature vector; drop empties."""
    feats = ds.features or {}
    keep = set(feats.keys())
    pls = []
    for pl in ds.playlists:
        t = [u for u in pl.tracks if u in keep]
        if t:
            pls.append(Playlist(pid=pl.pid, name=pl.name, tracks=t))
    tracks = {u: m for u, m in ds.tracks.items() if u in keep}
    return Dataset(playlists=pls, tracks=tracks, features={u: feats[u] for u in tracks})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--zip", default=os.path.join(ROOT, ".data",
                    "spotify_million_playlist_dataset.zip"))
    ap.add_argument("--features", default=os.path.join(ROOT, ".data", "af", "data"))
    ap.add_argument("--n-corpus", type=int, default=150000)
    ap.add_argument("--min-count", type=int, default=3)
    ap.add_argument("--n-test", type=int, default=6000)
    ap.add_argument("--per-scenario", type=int, default=300)
    ap.add_argument("--k", type=int, default=500)
    ap.add_argument("--seed", type=int, default=1)
    args = ap.parse_args()
    os.makedirs(FIG_DIR, exist_ok=True)
    t0 = time.time()

    log(f"Loading corpus n={args.n_corpus} min_count={args.min_count}")
    ds = load_mpd_zip(args.zip, max_playlists=args.n_corpus,
                      min_track_count=args.min_count,
                      progress=lambda n: log(f"  loaded {n}"))
    n_before = ds.n_tracks
    log(f"  corpus {ds.n_playlists} playlists, {n_before} tracks")

    log("Attaching real audio features (streaming shards)...")
    n_match = attach_real_features(
        ds, args.features,
        progress=lambda rows, hits: (log(f"  scanned {rows} rows, {hits} matched")
                                     if rows % 12800000 == 0 else None))
    match_rate = n_match / max(1, n_before)
    log(f"  matched {n_match}/{n_before} tracks = {100*match_rate:.1f}%")

    log("Restricting to feature-covered subset...")
    ds = filter_to_features(ds)
    log(f"  matched subset: {ds.n_playlists} playlists, {ds.n_tracks} tracks")

    train_pls, test_pls = train_test_split(ds, n_test=args.n_test, seed=args.seed)
    train_ds = Dataset(playlists=train_pls, tracks=ds.tracks, features=ds.features)
    cases = build_scenarios(test_pls, per_scenario=args.per_scenario, seed=args.seed)
    log(f"  train {len(train_pls)} / test {len(test_pls)}; cases: "
        f"{dict(Counter(c.scenario for c in cases))}")
    track_artist = ds.track_artist_map()

    models = {}
    for name, m in [("popularity", PopularityRecommender()),
                    ("item_cf", ItemCFRecommender()),
                    ("title", TitleModelRecommender()),
                    ("taste_engine", TasteEngine())]:
        t = time.time()
        m.fit(train_ds)
        models[name] = m
        log(f"  {name:14s} fit {time.time()-t:.1f}s")

    log("Evaluating...")
    results = defaultdict(lambda: defaultdict(list))
    for name, m in models.items():
        for c in cases:
            recs = m.recommend(c.seed_tracks, c.title, k=args.k)
            results[name][c.scenario].append(evaluate_one(recs, c.holdout, track_artist))

    rows = []
    for name in models:
        for scen in SCENARIO_NAMES:
            per = results[name].get(scen)
            if per:
                rows.append(dict(model=name, scenario=scen, n=len(per), **aggregate(per)))
        allc = [x for scen in results[name] for x in results[name][scen]]
        rows.append(dict(model=name, scenario="OVERALL", n=len(allc), **aggregate(allc)))
    df = pd.DataFrame(rows)
    os.makedirs(REAL_DIR, exist_ok=True)
    df.to_csv(os.path.join(REAL_DIR, "taste_real.csv"), index=False)
    meta = dict(n_corpus=args.n_corpus, min_count=args.min_count,
                match_rate=round(match_rate, 4), n_matched_tracks=n_match,
                n_catalog_before=n_before, subset_playlists=ds.n_playlists,
                subset_tracks=ds.n_tracks, per_scenario=args.per_scenario,
                runtime_s=round(time.time() - t0, 1))
    with open(os.path.join(REAL_DIR, "taste_real_meta.json"), "w") as fh:
        json.dump(meta, fh, indent=2)

    # figure: overall R-precision bars on the matched subset
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ov = df[df.scenario == "OVERALL"].set_index("model")["r_precision"].sort_values()
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.barh(ov.index, ov.values, color="#72B7B2")
    ax.set_title(f"Taste Engine on real MPD (feature-matched subset, "
                 f"{100*match_rate:.0f}% match)")
    ax.set_xlabel("Overall R-precision")
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    p = os.path.join(FIG_DIR, "real_taste_subset.png")
    fig.savefig(p, dpi=110)
    import shutil
    shutil.copy(p, os.path.join(VIZ, "real_taste_subset.png"))
    plt.close(fig)

    log(f"Done in {time.time()-t0:.0f}s. match={100*match_rate:.1f}%")
    print(df[df.scenario == "OVERALL"].set_index("model")[
        ["r_precision", "ndcg", "clicks"]].round(4).to_string())


if __name__ == "__main__":
    main()
