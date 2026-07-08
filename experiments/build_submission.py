"""Generate an official AIcrowd submission for the real challenge set.

Trains the best model (RoutedHybrid = two-stage hybrid + scenario-aware title
routing) on a real-MPD subsample, predicts 500 tracks for each of the 10,000
challenge playlists, and writes a gzipped ``submission.csv.gz`` in the official
format.  The catalogue is forced to include every track that occurs in the
challenge set (``keep_uris``) so seed signal and candidate coverage are not lost
to frequency pruning.

    python experiments/build_submission.py \
        --zip .data/spotify_million_playlist_dataset.zip \
        --challenge .data/spotify_million_playlist_dataset_challenge.zip \
        --n-corpus 400000 --min-count 5 --out .data/submission.csv

Validates with the project validator and (if present) the official
``verify_submission.py`` bundled in the challenge zip.
"""
from __future__ import annotations

import argparse
import csv
import gzip
import os
import sys
import time
import warnings

warnings.filterwarnings("ignore")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "src"))

from playlistcont.data.mpd_zip import load_mpd_zip, load_challenge_set  # noqa: E402
from playlistcont.models.itemcf import ItemCFRecommender  # noqa: E402
from playlistcont.models.mf import ALSRecommender  # noqa: E402
from playlistcont.models.track2vec import Track2VecRecommender  # noqa: E402
from playlistcont.models.title_model import TitleModelRecommender  # noqa: E402
from playlistcont.models.popularity import PopularityRecommender  # noqa: E402
from playlistcont.models.hybrid import HybridRecommender  # noqa: E402
from playlistcont.models.routed import RoutedHybrid  # noqa: E402


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def generate_submission(model, popularity, challenge_playlists, out_path,
                        team, email, logfn=log):
    """Predict 500 tracks per challenge playlist with ``model`` and write the
    official submission (exactly 500 unique non-seed tracks per pid).

    Returns ``(gz_path, seeds, n_playlists)``.  ``popularity`` is used only to
    backfill when the model returns fewer than 500 usable tracks.
    """
    t0 = time.time()
    preds, seeds = {}, {}
    for i, p in enumerate(challenge_playlists):
        pid = int(p["pid"])
        title = p.get("name")
        seed = [t["track_uri"] for t in p.get("tracks", [])]
        seeds[pid] = seed
        recs = model.recommend(seed, title, k=500)
        seen = set(seed)
        out = []
        for u in recs:
            if u not in seen:
                out.append(u); seen.add(u)
            if len(out) == 500:
                break
        if len(out) < 500:
            for u in popularity.recommend(seed, title, k=500 + len(seed) + 1000):
                if u not in seen:
                    out.append(u); seen.add(u)
                if len(out) == 500:
                    break
        preds[pid] = out
        if (i + 1) % 2000 == 0:
            logfn(f"  predicted {i+1}/{len(challenge_playlists)} "
                  f"({time.time()-t0:.0f}s)")

    with open(out_path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["team_info", team, email])
        fh.write("\n")
        for pid, uris in preds.items():
            assert len(uris) == 500 and len(set(uris)) == 500, (pid, len(uris))
            w.writerow([pid] + uris)
    gz = out_path + ".gz"
    with open(out_path, "rb") as fin, gzip.open(gz, "wb", compresslevel=6) as fout:
        fout.writelines(fin)
    logfn(f"  wrote {out_path} + {gz} "
          f"({os.path.getsize(gz)/1e6:.1f} MB gz, "
          f"{os.path.getsize(out_path)/1e6:.1f} MB raw)")
    return gz, seeds, len(preds)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--zip", default=os.path.join(ROOT, ".data",
                    "spotify_million_playlist_dataset.zip"))
    ap.add_argument("--challenge", default=os.path.join(ROOT, ".data",
                    "spotify_million_playlist_dataset_challenge.zip"))
    ap.add_argument("--n-corpus", type=int, default=400000)
    ap.add_argument("--min-count", type=int, default=5)
    ap.add_argument("--out", default=os.path.join(ROOT, ".data", "submission.csv"))
    ap.add_argument("--team", default="playlistcont")
    ap.add_argument("--email", default="ssalvador2204@gmail.com")
    ap.add_argument("--als-factors", type=int, default=128)
    args = ap.parse_args()
    t0 = time.time()

    log("Loading challenge set...")
    ch = load_challenge_set(args.challenge)
    ch_track_uris = {t["track_uri"] for p in ch for t in p.get("tracks", [])}
    log(f"  {len(ch)} challenge playlists, {len(ch_track_uris)} distinct seed tracks")

    log(f"Loading corpus n={args.n_corpus} min_count={args.min_count} "
        f"(+challenge tracks kept)...")
    ds = load_mpd_zip(args.zip, max_playlists=args.n_corpus,
                      min_track_count=args.min_count, keep_uris=ch_track_uris,
                      progress=lambda n: log(f"  loaded {n}"))
    log(f"  corpus {ds.n_playlists} playlists, {ds.n_tracks} tracks")
    seed_cov = len(ch_track_uris & set(ds.tracks.keys())) / max(1, len(ch_track_uris))
    log(f"  challenge seed-track coverage: {100*seed_cov:.1f}%")

    log("Fitting submodels + hybrid + routed...")
    subs = {
        "popularity": PopularityRecommender().fit(ds),
        "item_cf": ItemCFRecommender().fit(ds),
        "als": ALSRecommender(factors=args.als_factors, iterations=15).fit(ds),
        "track2vec": Track2VecRecommender(workers=4).fit(ds),
        "title": TitleModelRecommender().fit(ds),
    }
    log(f"  submodels fit ({time.time()-t0:.0f}s)")
    hy = HybridRecommender(submodels=subs, n_train_playlists=2000).fit(ds)
    log(f"  hybrid fit, reranker={hy.blender_backend} ({time.time()-t0:.0f}s)")
    routed = RoutedHybrid(hybrid=hy, title=subs["title"]).fit(ds)

    log("Predicting challenge playlists...")
    gz, seeds, n = generate_submission(routed, subs["popularity"], ch, args.out,
                                       args.team, args.email)

    # ---- validate ---------------------------------------------------
    from playlistcont.challenge.submission import validate_submission
    res = validate_submission(args.out, seeds=seeds)
    log(f"project validator: ok={res.ok} errors={len(res.errors)}")
    for e in res.errors[:5]:
        log(f"    {e}")
    log(f"Done in {time.time()-t0:.0f}s.")


if __name__ == "__main__":
    main()
