"""Fast official submission for the real 10k challenge set.

The full RoutedHybrid is the strongest model but its per-query cost (a title
TF-IDF matmul over ~143k playlists + candidate generation from five submodels)
makes 10,000 challenge predictions take hours.  For a tractable submission we
use a lightweight **query-shape router** built from the two models that the
internal real-data evaluation shows carry the signal:

* **item-CF** for every playlist that has at least one seed track (co-occurrence
  is the dominant signal on the real MPD, and item-CF is ~50x faster per query
  than the hybrid);
* the **title model** for the pure cold-start `title_only` bucket (0 seeds),
  where item-CF has nothing to work with.

This mirrors the RoutedHybrid *policy* (hand the cold start to the title
specialist, let co-occurrence own the seeded queries) at a fraction of the cost.
Predictions are guaranteed to be exactly 500 unique non-seed tracks per pid.

    python experiments/build_submission_fast.py --n-corpus 200000 --min-count 8
"""
from __future__ import annotations

import argparse
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
from playlistcont.models.title_model import TitleModelRecommender  # noqa: E402
from playlistcont.models.popularity import PopularityRecommender  # noqa: E402
from playlistcont.models.base import Recommender  # noqa: E402


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


class FastRouter(Recommender):
    """title model when there are no seed tracks, else item-CF."""
    name = "fast_router"

    def __init__(self, item_cf, title):
        self.item_cf = item_cf
        self.title = title

    def fit(self, ds):
        return self

    def recommend(self, seed_tracks, title=None, k=500):
        if not seed_tracks and title and title.strip():
            return self.title.recommend(seed_tracks, title, k=k)
        return self.item_cf.recommend(seed_tracks, title, k=k)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--zip", default=os.path.join(ROOT, ".data",
                    "spotify_million_playlist_dataset.zip"))
    ap.add_argument("--challenge", default=os.path.join(ROOT, ".data",
                    "spotify_million_playlist_dataset_challenge.zip"))
    ap.add_argument("--n-corpus", type=int, default=200000)
    ap.add_argument("--min-count", type=int, default=8)
    ap.add_argument("--out", default=os.path.join(ROOT, ".data", "submission.csv"))
    ap.add_argument("--team", default="playlistcont")
    ap.add_argument("--email", default="ssalvador2204@gmail.com")
    args = ap.parse_args()
    t0 = time.time()

    ch = load_challenge_set(args.challenge)
    ch_uris = {t["track_uri"] for p in ch for t in p.get("tracks", [])}
    log(f"challenge {len(ch)} playlists, {len(ch_uris)} distinct seed tracks")

    log(f"loading corpus n={args.n_corpus} min_count={args.min_count}")
    ds = load_mpd_zip(args.zip, max_playlists=args.n_corpus,
                      min_track_count=args.min_count,
                      progress=lambda n: log(f"  loaded {n}"))
    cov = len(ch_uris & set(ds.tracks)) / max(1, len(ch_uris))
    log(f"corpus {ds.n_playlists} pls, {ds.n_tracks} tracks; "
        f"challenge seed coverage {100*cov:.1f}%")

    log("fitting item_cf + title + popularity...")
    icf = ItemCFRecommender().fit(ds)
    ttl = TitleModelRecommender().fit(ds)
    pop = PopularityRecommender().fit(ds)
    router = FastRouter(icf, ttl)
    log(f"fit done ({time.time()-t0:.0f}s)")

    from build_submission import generate_submission
    from playlistcont.challenge.submission import validate_submission
    gz, seeds, n = generate_submission(router, pop, ch, args.out,
                                       args.team, args.email, logfn=log)
    res = validate_submission(args.out, seeds=seeds)
    log(f"validator ok={res.ok} errors={len(res.errors)} pids={n} "
        f"coverage={100*cov:.1f}%")
    for e in res.errors[:5]:
        log("   " + e)

    # official verifier (bundled in the challenge zip)
    try:
        import json, zipfile, importlib.util, tempfile
        with zipfile.ZipFile(args.challenge) as zf:
            cs_name = next(x for x in zf.namelist() if x.endswith("challenge_set.json"))
            vs_name = next(x for x in zf.namelist() if x.endswith("verify_submission.py"))
            tmp = tempfile.mkdtemp()
            cs_path = os.path.join(tmp, "challenge_set.json")
            open(cs_path, "wb").write(zf.read(cs_name))
            vs_path = os.path.join(tmp, "verify_submission.py")
            open(vs_path, "wb").write(zf.read(vs_name))
        spec = importlib.util.spec_from_file_location("verify_submission", vs_path)
        vs = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(vs)
        errs = vs.verify_submission(cs_path, args.out)
        log(f"OFFICIAL verify_submission.py: {errs} errors "
            f"({'PASS' if errs == 0 else 'FAIL'})")
    except Exception as e:
        log(f"official verifier skipped: {e}")

    import json
    meta = dict(model="FastRouter(item_cf | title for cold-start)",
                n_corpus=args.n_corpus, min_count=args.min_count,
                n_tracks=ds.n_tracks, challenge_seed_coverage=round(cov, 4),
                gz_path=gz, gz_mb=round(os.path.getsize(gz) / 1e6, 2),
                raw_mb=round(os.path.getsize(args.out) / 1e6, 2),
                validator_ok=res.ok, n_playlists=n,
                runtime_s=round(time.time() - t0, 1))
    with open(os.path.join(ROOT, "results", "real", "submission_meta.json"), "w") as fh:
        json.dump(meta, fh, indent=2)
    log(f"done in {time.time()-t0:.0f}s -> {gz} ({meta['gz_mb']} MB)")


if __name__ == "__main__":
    main()
