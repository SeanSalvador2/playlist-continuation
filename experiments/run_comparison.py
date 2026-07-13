"""Train every model on synthetic MPD, evaluate on all 10 scenarios, and write
a results table (CSV + Markdown) plus charts.

Run:  python experiments/run_comparison.py            # default fast scale
      python experiments/run_comparison.py --big       # closer to the spec scale
      python experiments/run_comparison.py --real DIR   # real MPD slice dir

Everything is deterministic given the seed.  Designed to finish in a few
minutes at the default scale.
"""
from __future__ import annotations

import argparse
import os
import sys
import time
import warnings
from collections import defaultdict

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "src"))
RESULTS_DIR = os.path.join(ROOT, "results")
FIG_DIR = os.path.join(RESULTS_DIR, "figures")

from playlistcont.data.schema import Dataset  # noqa: E402
from playlistcont.data.synthetic import make_synthetic  # noqa: E402
from playlistcont.data.loader import load_mpd  # noqa: E402
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
from playlistcont.models.content_knn import ContentKNNRecommender  # noqa: E402
from playlistcont.models.hybrid import HybridRecommender  # noqa: E402


def build_dataset(args):
    if args.real:
        ds = load_mpd(args.real, max_playlists=args.n_playlists)
        print(f"Loaded real MPD: {ds.n_playlists} playlists, {ds.n_tracks} tracks")
        if ds.features is None:
            print("  (no audio features attached -> TasteEngine will be skipped)")
        return ds
    ds = make_synthetic(
        n_playlists=args.n_playlists, n_tracks=args.n_tracks, seed=args.seed
    )
    print(f"Synthetic MPD: {ds.n_playlists} playlists, {ds.n_tracks} tracks")
    return ds


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-playlists", type=int, default=10000)
    ap.add_argument("--n-tracks", type=int, default=5000)
    ap.add_argument("--n-test", type=int, default=3000)
    ap.add_argument("--per-scenario", type=int, default=100)
    ap.add_argument("--k", type=int, default=500)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--big", action="store_true", help="20k playlists / 15k tracks")
    ap.add_argument("--real", type=str, default=None, help="real MPD slice dir")
    ap.add_argument("--tag", type=str, default="",
                    help="suffix for output filenames, e.g. --tag _content_knn "
                         "writes results_content_knn.csv and leaves results.csv "
                         "untouched (default: overwrite results.csv)")
    args = ap.parse_args()
    if args.big:
        args.n_playlists, args.n_tracks, args.n_test = 20000, 15000, 4000

    os.makedirs(FIG_DIR, exist_ok=True)
    t0 = time.time()
    ds = build_dataset(args)
    train_pls, test_pls = train_test_split(ds, n_test=args.n_test, seed=args.seed)
    train_ds = Dataset(playlists=train_pls, tracks=ds.tracks, features=ds.features)
    cases = build_scenarios(test_pls, per_scenario=args.per_scenario, seed=args.seed)
    from collections import Counter
    print("Scenario case counts:", dict(Counter(c.scenario for c in cases)))
    track_artist = ds.track_artist_map()

    # ---- fit models -------------------------------------------------
    print("\nFitting models...")
    submodels = {}
    fit_times = {}
    def fit(name, model):
        t = time.time()
        model.fit(train_ds)
        fit_times[name] = time.time() - t
        submodels[name] = model
        print(f"  {name:14s} fit in {fit_times[name]:5.1f}s")
        return model

    fit("popularity", PopularityRecommender())
    fit("item_cf", ItemCFRecommender())
    fit("als", ALSRecommender())
    fit("track2vec", Track2VecRecommender())
    fit("title", TitleModelRecommender())
    models = dict(submodels)
    if ds.features is not None:
        fit("taste_engine", TasteEngine())
        models["taste_engine"] = submodels["taste_engine"]
        # pure content-based baseline (feature cosine to the seed centroid)
        fit("content_knn", ContentKNNRecommender())
        models["content_knn"] = submodels["content_knn"]
    # hybrid reuses the already-fit submodels
    hy = HybridRecommender(submodels={k: submodels[k] for k in
                                       ["item_cf", "als", "track2vec", "title", "popularity"]})
    fit("hybrid", hy)
    models["hybrid"] = hy

    # ---- evaluate ---------------------------------------------------
    print("\nEvaluating...")
    # results[model][scenario] = list of per-case metric dicts
    results = defaultdict(lambda: defaultdict(list))
    for mi, (mname, model) in enumerate(models.items()):
        t = time.time()
        for c in cases:
            recs = model.recommend(c.seed_tracks, c.title, k=args.k)
            results[mname][c.scenario].append(
                evaluate_one(recs, c.holdout, track_artist)
            )
        print(f"  {mname:14s} evaluated in {time.time()-t:5.1f}s")

    # ---- aggregate into tidy frame ----------------------------------
    rows = []
    for mname in models:
        for scen in SCENARIO_NAMES:
            per = results[mname].get(scen)
            if not per:
                continue
            agg = aggregate(per)
            rows.append(dict(model=mname, scenario=scen, n=len(per), **agg))
        # overall (micro over all cases)
        allcases = [m for scen in results[mname] for m in results[mname][scen]]
        agg = aggregate(allcases)
        rows.append(dict(model=mname, scenario="OVERALL", n=len(allcases), **agg))
    df = pd.DataFrame(rows)
    tag = args.tag
    df.to_csv(os.path.join(RESULTS_DIR, f"results{tag}.csv"), index=False)

    # ---- markdown tables --------------------------------------------
    write_markdown(df, tag)
    make_charts(df, tag)

    print(f"\nDone in {time.time()-t0:.1f}s. Wrote results/ and figures/.")
    overall = df[df.scenario == "OVERALL"].set_index("model")
    print("\nOVERALL (micro-averaged over all cases):")
    print(overall[["r_precision", "ndcg", "clicks"]].round(4).to_string())


def _pivot(df, metric):
    d = df[df.scenario != "OVERALL"]
    return d.pivot_table(index="model", columns="scenario", values=metric)


def write_markdown(df, tag=""):
    path = os.path.join(RESULTS_DIR, f"results{tag}.md")
    overall = df[df.scenario == "OVERALL"].set_index("model")[
        ["r_precision", "ndcg", "clicks"]
    ].round(4)
    with open(path, "w") as fh:
        fh.write("# Model comparison results\n\n")
        fh.write("Generated by `experiments/run_comparison.py` on synthetic MPD.\n\n")
        fh.write("## Overall (micro-averaged over all scenario cases)\n\n")
        fh.write(overall.sort_values("r_precision", ascending=False).to_markdown())
        fh.write("\n\n")
        for metric, better in [("r_precision", "higher"), ("ndcg", "higher"),
                               ("clicks", "lower")]:
            fh.write(f"## {metric} by scenario ({better} is better)\n\n")
            piv = _pivot(df, metric).round(4)
            piv = piv.reindex(columns=[s for s in SCENARIO_NAMES if s in piv.columns])
            fh.write(piv.to_markdown())
            fh.write("\n\n")
    print("  wrote", path)


def make_charts(df, tag=""):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig_dir = FIG_DIR if not tag else os.path.join(RESULTS_DIR, f"figures{tag}")
    os.makedirs(fig_dir, exist_ok=True)

    for metric in ["r_precision", "ndcg", "clicks"]:
        piv = _pivot(df, metric)
        piv = piv.reindex(columns=[s for s in SCENARIO_NAMES if s in piv.columns])
        fig, ax = plt.subplots(figsize=(13, 6))
        n_models = len(piv.index)
        x = np.arange(len(piv.columns))
        w = 0.8 / max(1, n_models)
        for i, model in enumerate(piv.index):
            ax.bar(x + i * w, piv.loc[model].values, w, label=model)
        ax.set_xticks(x + 0.4 - w / 2)
        ax.set_xticklabels(piv.columns, rotation=35, ha="right")
        ax.set_ylabel(metric)
        ax.set_title(f"{metric} by model and scenario "
                     f"({'lower is better' if metric=='clicks' else 'higher is better'})")
        ax.legend(ncol=4, fontsize=8)
        ax.grid(axis="y", alpha=0.3)
        fig.tight_layout()
        p = os.path.join(fig_dir, f"{metric}_by_scenario.png")
        fig.savefig(p, dpi=110)
        plt.close(fig)
        print("  wrote", p)

    # overall summary chart
    overall = df[df.scenario == "OVERALL"].set_index("model")
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for ax, metric in zip(axes, ["r_precision", "ndcg", "clicks"]):
        s = overall[metric].sort_values(ascending=(metric != "clicks"))
        ax.barh(s.index, s.values, color="#4C78A8")
        ax.set_title(f"Overall {metric}")
        ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    p = os.path.join(fig_dir, "overall_summary.png")
    fig.savefig(p, dpi=110)
    plt.close(fig)
    print("  wrote", p)


if __name__ == "__main__":
    main()
