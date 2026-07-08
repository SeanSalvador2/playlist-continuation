"""Real-data run: train every model on a real-MPD subsample, evaluate on a
held-out set of real playlists split into the 10 official scenarios, and write
the headline tables + figures under ``results/real/``.

This is the real-data analogue of ``run_comparison.py``.  Data is streamed
directly from the official ``spotify_million_playlist_dataset.zip`` (never
extracted); see ``playlistcont.data.mpd_zip``.

    python experiments/run_real.py --zip .data/spotify_million_playlist_dataset.zip \
        --n-corpus 250000 --min-count 5 --n-test 10000 --per-scenario 1000

Everything is deterministic given the seed.  Results CSVs are checkpointed so
the write-ups / notebook can reload them without retraining.
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

from playlistcont.data.schema import Dataset  # noqa: E402
from playlistcont.data.mpd_zip import load_mpd_zip  # noqa: E402
from playlistcont.data.real_features import attach_real_features  # noqa: E402
from playlistcont.challenge.scenarios import (  # noqa: E402
    build_scenarios, train_test_split, SCENARIO_NAMES,
)
from playlistcont.challenge.metrics import evaluate_one, aggregate  # noqa: E402
from playlistcont.models.popularity import PopularityRecommender  # noqa: E402
from playlistcont.models.itemcf import ItemCFRecommender  # noqa: E402
from playlistcont.models.mf import ALSRecommender  # noqa: E402
from playlistcont.models.track2vec import Track2VecRecommender  # noqa: E402
from playlistcont.models.title_model import TitleModelRecommender  # noqa: E402
from playlistcont.models.taste_engine import TasteEngine  # noqa: E402
from playlistcont.models.hybrid import HybridRecommender  # noqa: E402
from playlistcont.models.routed import RoutedHybrid  # noqa: E402


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--zip", default=os.path.join(ROOT, ".data",
                    "spotify_million_playlist_dataset.zip"))
    ap.add_argument("--n-corpus", type=int, default=250000,
                    help="playlists loaded as the real corpus (train+test)")
    ap.add_argument("--min-count", type=int, default=5,
                    help="prune tracks appearing in < this many corpus playlists")
    ap.add_argument("--n-test", type=int, default=10000)
    ap.add_argument("--per-scenario", type=int, default=1000)
    ap.add_argument("--k", type=int, default=500)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--features", default=None,
                    help="audio-features parquet/csv to attach (enables taste engine)")
    ap.add_argument("--als-factors", type=int, default=128)
    ap.add_argument("--als-iters", type=int, default=15)
    ap.add_argument("--challenge", default=None,
                    help="challenge zip/json; if given, also writes a submission")
    ap.add_argument("--submission-out", default=os.path.join(ROOT, ".data",
                    "submission.csv"))
    ap.add_argument("--team", default="playlistcont")
    ap.add_argument("--email", default="ssalvador2204@gmail.com")
    args = ap.parse_args()

    os.makedirs(FIG_DIR, exist_ok=True)
    t0 = time.time()

    challenge_uris = None
    ch = None
    if args.challenge:
        from playlistcont.data.mpd_zip import load_challenge_set
        ch = load_challenge_set(args.challenge)
        challenge_uris = {t["track_uri"] for p in ch for t in p.get("tracks", [])}
        log(f"Challenge set: {len(ch)} playlists, {len(challenge_uris)} distinct "
            f"seed tracks")

    # Catalogue = tracks with freq >= min_count in the corpus subsample.  We do
    # NOT force-keep challenge tracks (it would bloat the co-occurrence matrix);
    # coverage of the challenge seeds is measured and reported instead.
    log(f"Loading real corpus: n_corpus={args.n_corpus} min_count={args.min_count}")
    ds = load_mpd_zip(args.zip, max_playlists=args.n_corpus,
                      min_track_count=args.min_count,
                      progress=lambda n: log(f"  loaded {n} playlists"))
    log(f"Corpus: {ds.n_playlists} playlists, {ds.n_tracks} tracks")

    if args.features:
        n_matched = attach_real_features(ds, args.features)
        log(f"Attached audio features: {n_matched}/{ds.n_tracks} tracks matched "
            f"({100*n_matched/max(1,ds.n_tracks):.1f}%)")

    train_pls, test_pls = train_test_split(ds, n_test=args.n_test, seed=args.seed)
    train_ds = Dataset(playlists=train_pls, tracks=ds.tracks, features=ds.features)
    cases = build_scenarios(test_pls, per_scenario=args.per_scenario, seed=args.seed)
    log(f"Train {len(train_pls)} / test {len(test_pls)}; "
        f"scenario cases: {dict(Counter(c.scenario for c in cases))}")
    track_artist = ds.track_artist_map()

    # ---- fit models -------------------------------------------------
    log("Fitting models...")
    submodels = {}
    fit_times = {}

    def fit(name, model):
        t = time.time()
        model.fit(train_ds)
        fit_times[name] = time.time() - t
        submodels[name] = model
        log(f"  {name:16s} fit in {fit_times[name]:6.1f}s")
        return model

    fit("popularity", PopularityRecommender())
    fit("item_cf", ItemCFRecommender())
    fit("als", ALSRecommender(factors=args.als_factors, iterations=args.als_iters))
    fit("track2vec", Track2VecRecommender(workers=4))
    fit("title", TitleModelRecommender())
    models = dict(submodels)
    if ds.features is not None:
        fit("taste_engine", TasteEngine())
        models["taste_engine"] = submodels["taste_engine"]
    hy = HybridRecommender(submodels={k: submodels[k] for k in
                           ["item_cf", "als", "track2vec", "title", "popularity"]},
                           n_train_playlists=2000)
    fit("hybrid", hy)
    models["hybrid"] = hy
    routed = RoutedHybrid(hybrid=hy, title=submodels["title"])
    routed.fit(train_ds)
    models["routed_hybrid"] = routed
    log(f"  routed_hybrid wired (reuses hybrid+title)")

    # ---- evaluate ---------------------------------------------------
    log("Evaluating...")
    results = defaultdict(lambda: defaultdict(list))
    for mname, model in models.items():
        t = time.time()
        for c in cases:
            recs = model.recommend(c.seed_tracks, c.title, k=args.k)
            results[mname][c.scenario].append(
                evaluate_one(recs, c.holdout, track_artist)
            )
        log(f"  {mname:16s} evaluated in {time.time()-t:6.1f}s")

    # ---- tidy frame -------------------------------------------------
    rows = []
    for mname in models:
        for scen in SCENARIO_NAMES:
            per = results[mname].get(scen)
            if not per:
                continue
            rows.append(dict(model=mname, scenario=scen, n=len(per),
                             **aggregate(per)))
        allc = [m for scen in results[mname] for m in results[mname][scen]]
        rows.append(dict(model=mname, scenario="OVERALL", n=len(allc),
                         **aggregate(allc)))
    df = pd.DataFrame(rows)
    os.makedirs(REAL_DIR, exist_ok=True)
    df.to_csv(os.path.join(REAL_DIR, "real_results.csv"), index=False)

    meta = dict(n_corpus=args.n_corpus, min_count=args.min_count,
                n_train=len(train_pls), n_test=len(test_pls),
                n_tracks_catalog=ds.n_tracks, per_scenario=args.per_scenario,
                als_factors=args.als_factors, als_iters=args.als_iters,
                seed=args.seed, k=args.k,
                fit_times={k: round(v, 1) for k, v in fit_times.items()},
                als_backend=getattr(submodels["als"], "backend", "?"),
                hybrid_reranker=getattr(hy, "blender_backend", "?"),
                features_attached=ds.features is not None,
                runtime_s=round(time.time() - t0, 1))
    with open(os.path.join(REAL_DIR, "real_meta.json"), "w") as fh:
        json.dump(meta, fh, indent=2)

    write_markdown(df, meta)
    make_charts(df)

    # ---- optional: official submission from the best model ----------
    if ch is not None:
        from build_submission import generate_submission
        from playlistcont.challenge.submission import validate_submission
        log("Generating official submission with routed_hybrid...")
        seed_cov = len(challenge_uris & set(ds.tracks.keys())) / max(1, len(challenge_uris))
        log(f"  challenge seed-track catalogue coverage: {100*seed_cov:.1f}%")
        gz, seeds, n = generate_submission(
            routed, submodels["popularity"], ch, args.submission_out,
            args.team, args.email, logfn=log)
        res = validate_submission(args.submission_out, seeds=seeds)
        log(f"  submission validated: ok={res.ok} errors={len(res.errors)} "
            f"pids={n}")
        meta["submission"] = dict(path=gz, n_playlists=n,
                                  seed_coverage=round(seed_cov, 4),
                                  gz_mb=round(os.path.getsize(gz) / 1e6, 1),
                                  validator_ok=res.ok,
                                  validator_errors=len(res.errors))
        with open(os.path.join(REAL_DIR, "real_meta.json"), "w") as fh:
            json.dump(meta, fh, indent=2)

    log(f"Done in {time.time()-t0:.1f}s. Wrote results/real/.")
    overall = df[df.scenario == "OVERALL"].set_index("model")
    print("\nOVERALL (micro-averaged):")
    print(overall[["r_precision", "ndcg", "clicks"]]
          .sort_values("r_precision", ascending=False).round(4).to_string())


def _pivot(df, metric):
    d = df[df.scenario != "OVERALL"]
    return d.pivot_table(index="model", columns="scenario", values=metric)


def write_markdown(df, meta):
    path = os.path.join(REAL_DIR, "real_results.md")
    overall = df[df.scenario == "OVERALL"].set_index("model")[
        ["r_precision", "ndcg", "clicks"]].sort_values(
        "r_precision", ascending=False).round(4)
    with open(path, "w") as fh:
        fh.write("# Real-MPD model comparison\n\n")
        fh.write(f"Corpus: {meta['n_corpus']} playlists (min_count "
                 f"{meta['min_count']}), catalogue {meta['n_tracks_catalog']} "
                 f"tracks; train {meta['n_train']} / test {meta['n_test']}; "
                 f"{meta['per_scenario']} cases/scenario.\n\n")
        fh.write("## Overall (micro-averaged over all scenario cases)\n\n")
        fh.write(overall.to_markdown())
        fh.write("\n\n")
        for metric, better in [("r_precision", "higher"), ("ndcg", "higher"),
                               ("clicks", "lower")]:
            fh.write(f"## {metric} by scenario ({better} is better)\n\n")
            piv = _pivot(df, metric).round(4)
            piv = piv.reindex(columns=[s for s in SCENARIO_NAMES if s in piv.columns])
            fh.write(piv.to_markdown())
            fh.write("\n\n")
    log(f"  wrote {path}")


def make_charts(df):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    for metric in ["r_precision", "ndcg", "clicks"]:
        piv = _pivot(df, metric)
        piv = piv.reindex(columns=[s for s in SCENARIO_NAMES if s in piv.columns])
        fig, ax = plt.subplots(figsize=(13, 6))
        x = np.arange(len(piv.columns))
        w = 0.8 / max(1, len(piv.index))
        for i, model in enumerate(piv.index):
            ax.bar(x + i * w, piv.loc[model].values, w, label=model)
        ax.set_xticks(x + 0.4 - w / 2)
        ax.set_xticklabels(piv.columns, rotation=35, ha="right")
        ax.set_ylabel(metric)
        ax.set_title(f"Real MPD: {metric} by model and scenario "
                     f"({'lower is better' if metric=='clicks' else 'higher is better'})")
        ax.legend(ncol=4, fontsize=8)
        ax.grid(axis="y", alpha=0.3)
        fig.tight_layout()
        fig.savefig(os.path.join(FIG_DIR, f"real_{metric}_by_scenario.png"), dpi=110)
        plt.close(fig)

    overall = df[df.scenario == "OVERALL"].set_index("model")
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for ax, metric in zip(axes, ["r_precision", "ndcg", "clicks"]):
        s = overall[metric].sort_values(ascending=(metric != "clicks"))
        ax.barh(s.index, s.values, color="#E45756" if metric == "clicks" else "#4C78A8")
        ax.set_title(f"Real MPD overall {metric}")
        ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "real_overall_summary.png"), dpi=110)
    plt.close(fig)
    log("  wrote real figures")


if __name__ == "__main__":
    main()
