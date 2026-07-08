"""Experiment 4: robustness & error analysis.

  A. Seed sensitivity: rebuild the whole pipeline under several data seeds and
     report mean +/- std of the OVERALL metrics for every model (error bars).
  B. Data-scale study: 2k / 5k / 10k / 20k playlists -- which models benefit
     most from more data?
  C. Popularity-bias / coverage: head/torso/tail recall, catalog coverage and
     intra-list similarity (diversity) per model.

Run:  python experiments/exp_robustness.py
      python experiments/exp_robustness.py --quick   # skip 20k scale point
"""
from __future__ import annotations

import argparse
from collections import defaultdict

import numpy as np
import pandas as pd

from _common import (FIG_DIR as FIGP, STD, Scale, Timer, bar_plot, evaluate,
                     grouped_bar, line_plot, overall_row, prepare, save_csv)
from playlistcont.challenge.coverage import (bucket_hit_rate, bucketize_catalog,
                                             catalog_coverage,
                                             intra_list_similarity)
from playlistcont.data.schema import Dataset
from playlistcont.models.hybrid import HybridRecommender
from playlistcont.models.itemcf import ItemCFRecommender
from playlistcont.models.mf import ALSRecommender
from playlistcont.models.popularity import PopularityRecommender
from playlistcont.models.taste_engine import TasteEngine
from playlistcont.models.title_model import TitleModelRecommender
from playlistcont.models.track2vec import Track2VecRecommender

MODELS_ORDER = ["popularity", "item_cf", "als", "track2vec", "title",
                "taste_engine", "hybrid"]


def build_all(prep):
    """Fit all seven models on prep.train_ds (submodels shared into hybrid)."""
    subs = {
        "popularity": PopularityRecommender().fit(prep.train_ds),
        "item_cf": ItemCFRecommender().fit(prep.train_ds),
        "als": ALSRecommender().fit(prep.train_ds),
        "track2vec": Track2VecRecommender().fit(prep.train_ds),
        "title": TitleModelRecommender().fit(prep.train_ds),
    }
    models = dict(subs)
    models["taste_engine"] = TasteEngine().fit(prep.train_ds)
    models["hybrid"] = HybridRecommender(
        submodels=subs, n_train_playlists=600).fit(prep.train_ds)
    return models


def overall_all(prep, models):
    rows = []
    for name in MODELS_ORDER:
        o = overall_row(evaluate(models[name], prep)); o["model"] = name
        rows.append(o)
    return pd.DataFrame(rows)


def exp_seed_sensitivity(seeds):
    with Timer("A seed sensitivity"):
        per_seed = []
        for sd in seeds:
            prep = prepare(STD, data_seed=sd)
            models = build_all(prep)
            df = overall_all(prep, models); df["seed"] = sd
            per_seed.append(df)
            print(f"    seed={sd} done")
        alldf = pd.concat(per_seed, ignore_index=True)
        agg = alldf.groupby("model").agg(
            r_precision_mean=("r_precision", "mean"),
            r_precision_std=("r_precision", "std"),
            ndcg_mean=("ndcg", "mean"),
            ndcg_std=("ndcg", "std"),
            clicks_mean=("clicks", "mean"),
            clicks_std=("clicks", "std"),
        ).reindex(MODELS_ORDER).reset_index()
        for _, r in agg.iterrows():
            print(f"    {r.model:>13}  R-prec={r.r_precision_mean:.4f} "
                  f"+/- {r.r_precision_std:.4f}")
        bar_plot(agg.model, agg.r_precision_mean, "OVERALL R-precision",
                 f"Headline comparison, mean +/- std over {len(seeds)} seeds",
                 f"{FIGP}/robust_seed_rprec.png", err=agg.r_precision_std)
        return alldf, agg


def exp_data_scale(scales):
    with Timer("B data scale"):
        rows = []
        for npl in scales:
            ntr = int(npl * 0.6)
            ntest = min(2000, npl // 4)
            sc = Scale(n_playlists=npl, n_tracks=ntr, n_test=ntest,
                       per_scenario=STD.per_scenario, k=STD.k, seed=1)
            prep = prepare(sc)
            models = build_all(prep)
            df = overall_all(prep, models); df["n_playlists"] = npl
            df["n_cases"] = len(prep.cases)
            rows.append(df)
            print(f"    scale={npl}: {len(prep.cases)} cases done")
        alldf = pd.concat(rows, ignore_index=True)
        piv = alldf.pivot_table(index="n_playlists", columns="model",
                                values="r_precision")
        piv = piv.reindex(columns=MODELS_ORDER)
        line_plot(list(piv.index),
                  {m: piv[m].values for m in piv.columns},
                  "training playlists", "OVERALL R-precision",
                  "Data-scale study: R-precision vs corpus size",
                  f"{FIGP}/robust_data_scale.png", logx=True)
        return alldf


def exp_data_scale_fixed(scales, n_tracks_fixed=3500):
    """Fixed-catalog scale study: same track catalog, more playlists = denser
    co-occurrence.  This isolates 'benefit from data' from 'harder task'."""
    with Timer("B2 data scale (fixed catalog)"):
        rows = []
        for npl in scales:
            ntest = min(2000, npl // 4)
            sc = Scale(n_playlists=npl, n_tracks=n_tracks_fixed, n_test=ntest,
                       per_scenario=STD.per_scenario, k=STD.k, seed=1)
            prep = prepare(sc)
            models = build_all(prep)
            df = overall_all(prep, models); df["n_playlists"] = npl
            rows.append(df)
            print(f"    scale={npl} (catalog={n_tracks_fixed}): "
                  f"{len(prep.cases)} cases done")
        alldf = pd.concat(rows, ignore_index=True)
        piv = alldf.pivot_table(index="n_playlists", columns="model",
                                values="r_precision").reindex(columns=MODELS_ORDER)
        line_plot(list(piv.index), {m: piv[m].values for m in piv.columns},
                  "training playlists (fixed 3.5k-track catalog)",
                  "OVERALL R-precision",
                  "Data-scale study (fixed catalog): denser data helps",
                  f"{FIGP}/robust_data_scale_fixed.png", logx=True)
        return alldf


def exp_popularity_bias():
    with Timer("C popularity bias / coverage"):
        prep = prepare(STD)
        models = build_all(prep)
        # popularity buckets from the fitted index of any model
        idx = models["popularity"].index
        pop_arr = idx.track_pop
        buckets = bucketize_catalog(pop_arr, idx.track_uris)
        n_catalog = len(idx.track_uris)
        feats = prep.ds.features

        rows = []
        for name in MODELS_ORDER:
            model = models[name]
            bstat = defaultdict(lambda: {"hits": 0, "demand": 0})
            recs_all = []
            ils = []
            for c in prep.cases:
                recs = model.recommend(c.seed_tracks, c.title, k=prep.scale.k)
                recs_all.append(recs)
                bh = bucket_hit_rate(recs, c.holdout, buckets, k=prep.scale.k)
                for b, d in bh.items():
                    bstat[b]["hits"] += d["hits"]
                    bstat[b]["demand"] += d["demand"]
                ils.append(intra_list_similarity(recs, feats, k=100))
            cov = catalog_coverage(recs_all, n_catalog, k=prep.scale.k)
            row = dict(model=name, coverage=cov,
                       intra_list_sim=float(np.mean(ils)))
            for b in ("head", "torso", "tail"):
                d = bstat[b]
                row[f"{b}_recall"] = d["hits"] / d["demand"] if d["demand"] else 0.0
            rows.append(row)
            print(f"    {name:>13}  head={row['head_recall']:.3f} "
                  f"torso={row['torso_recall']:.3f} tail={row['tail_recall']:.3f} "
                  f"cov={cov:.3f} ILS={row['intra_list_sim']:.3f}")
        df = pd.DataFrame(rows)
        grouped_bar(df.model,
                    {"head": df.head_recall, "torso": df.torso_recall,
                     "tail": df.tail_recall},
                    "recall@500",
                    "Popularity bias: recall by track popularity bucket",
                    f"{FIGP}/robust_pop_bias.png")
        bar_plot(df.model, df.coverage, "catalog coverage (fraction of catalog)",
                 "Catalog coverage per model", f"{FIGP}/robust_coverage.png",
                 color="#54A24B")
        bar_plot(df.model, df.intra_list_sim,
                 "intra-list similarity (lower = more diverse)",
                 "Recommendation diversity per model",
                 f"{FIGP}/robust_diversity.png", color="#B279A2")
        return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3])
    args = ap.parse_args()

    seed_df, seed_agg = exp_seed_sensitivity(args.seeds)
    save_csv(seed_df, "robust_seed_raw.csv")
    save_csv(seed_agg, "robust_seed_summary.csv")

    scales = [2000, 5000, 10000] if args.quick else [2000, 5000, 10000, 20000]
    scale_df = exp_data_scale(scales)
    save_csv(scale_df, "robust_data_scale.csv")

    scale_fixed_df = exp_data_scale_fixed(scales)
    save_csv(scale_fixed_df, "robust_data_scale_fixed.csv")

    pop_df = exp_popularity_bias()
    save_csv(pop_df, "robust_pop_bias.csv")


if __name__ == "__main__":
    main()
