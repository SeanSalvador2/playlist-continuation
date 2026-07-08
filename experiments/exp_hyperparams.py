"""Experiment 1: per-model hyperparameter studies.

For every knob we sweep 3-6 values, hold the others at their default, refit the
model, evaluate on the full 10-scenario suite, and record the OVERALL metrics.
Item-CF, ALS, Track2Vec and the Title model are covered here; the hybrid has
its own script (exp_hybrid.py) because it is more expensive.

Run:  python experiments/exp_hyperparams.py
"""
from __future__ import annotations

import pandas as pd

from _common import (FIG_DIR as FIGP, STD, Timer, evaluate, line_plot,
                     overall_row, prepare, save_csv)
from playlistcont.models.itemcf import ItemCFRecommender
from playlistcont.models.mf import ALSRecommender
from playlistcont.models.title_model import TitleModelRecommender
from playlistcont.models.track2vec import Track2VecRecommender


def sweep(prep, make_model, knob, values, fixed_desc):
    rows = []
    for v in values:
        model = make_model(v).fit(prep.train_ds)
        o = overall_row(evaluate(model, prep))
        o.update({knob: v, "fixed": fixed_desc})
        rows.append(o)
        print(f"    {knob}={v!s:>10}  r_prec={o['r_precision']:.4f}  "
              f"ndcg={o['ndcg']:.4f}  clicks={o['clicks']:.3f}")
    return pd.DataFrame(rows)


def main():
    prep = prepare(STD)
    print(f"Data: {prep.ds.n_playlists} playlists, {prep.ds.n_tracks} tracks, "
          f"{len(prep.cases)} scenario cases")
    all_rows = []

    # ---------------- Item-CF ----------------
    with Timer("item_cf"):
        df = sweep(prep, lambda v: ItemCFRecommender(normalization=v),
                   "normalization", ["raw", "cosine", "pmi"], "topk_sim=200")
        df["model"] = "item_cf"; df["knob"] = "normalization"; all_rows.append(df)
        line_plot(range(len(df)), {"R-precision": df.r_precision, "NDCG": df.ndcg},
                  "normalization", "score", "Item-CF: similarity normalization",
                  f"{FIGP}/itemcf_normalization.png",
                  xticklabels=list(df.normalization))

        df = sweep(prep, lambda v: ItemCFRecommender(topk_sim=v),
                   "topk_sim", [25, 50, 100, 200, 400], "normalization=cosine")
        df["model"] = "item_cf"; df["knob"] = "topk_sim"; all_rows.append(df)
        line_plot(df.topk_sim, {"R-precision": df.r_precision, "NDCG": df.ndcg},
                  "top-K neighbours", "score", "Item-CF: neighbour truncation",
                  f"{FIGP}/itemcf_topk.png")

    # ---------------- ALS ----------------
    with Timer("als"):
        df = sweep(prep, lambda v: ALSRecommender(factors=v),
                   "factors", [16, 32, 64, 128], "reg=0.05,iters=15")
        df["model"] = "als"; df["knob"] = "factors"; all_rows.append(df)
        line_plot(df.factors, {"R-precision": df.r_precision, "NDCG": df.ndcg},
                  "latent factors", "score", "ALS: number of factors",
                  f"{FIGP}/als_factors.png")

        df = sweep(prep, lambda v: ALSRecommender(reg=v),
                   "reg", [0.001, 0.01, 0.05, 0.1, 0.5], "factors=64,iters=15")
        df["model"] = "als"; df["knob"] = "reg"; all_rows.append(df)
        line_plot(df.reg, {"R-precision": df.r_precision, "NDCG": df.ndcg},
                  "regularization", "score", "ALS: regularization",
                  f"{FIGP}/als_reg.png", logx=True)

        df = sweep(prep, lambda v: ALSRecommender(iterations=v),
                   "iterations", [3, 5, 10, 15, 25], "factors=64,reg=0.05")
        df["model"] = "als"; df["knob"] = "iterations"; all_rows.append(df)
        line_plot(df.iterations, {"R-precision": df.r_precision, "NDCG": df.ndcg},
                  "ALS iterations", "score", "ALS: iterations",
                  f"{FIGP}/als_iters.png")

    # ---------------- Track2Vec (undertraining focus) ----------------
    with Timer("track2vec"):
        df = sweep(prep, lambda v: Track2VecRecommender(epochs=v),
                   "epochs", [5, 10, 20, 40], "dim=64,window=8,min_count=1")
        df["model"] = "track2vec"; df["knob"] = "epochs"; all_rows.append(df)
        line_plot(df.epochs, {"R-precision": df.r_precision, "NDCG": df.ndcg},
                  "epochs", "score", "Track2Vec: epochs (undertraining test)",
                  f"{FIGP}/track2vec_epochs.png")

        df = sweep(prep, lambda v: Track2VecRecommender(dim=v, epochs=20),
                   "dim", [32, 64, 128], "epochs=20,window=8")
        df["model"] = "track2vec"; df["knob"] = "dim"; all_rows.append(df)
        line_plot(df.dim, {"R-precision": df.r_precision, "NDCG": df.ndcg},
                  "embedding dim", "score", "Track2Vec: embedding dimension",
                  f"{FIGP}/track2vec_dim.png")

        df = sweep(prep, lambda v: Track2VecRecommender(window=v, epochs=20),
                   "window", [3, 8, 15, 30], "epochs=20,dim=64")
        df["model"] = "track2vec"; df["knob"] = "window"; all_rows.append(df)
        line_plot(df.window, {"R-precision": df.r_precision, "NDCG": df.ndcg},
                  "window size", "score", "Track2Vec: context window",
                  f"{FIGP}/track2vec_window.png")

        df = sweep(prep, lambda v: Track2VecRecommender(min_count=v, epochs=20),
                   "min_count", [1, 2, 5], "epochs=20,dim=64")
        df["model"] = "track2vec"; df["knob"] = "min_count"; all_rows.append(df)
        line_plot(df.min_count, {"R-precision": df.r_precision, "NDCG": df.ndcg},
                  "min_count", "score", "Track2Vec: min_count",
                  f"{FIGP}/track2vec_mincount.png")

    # ---------------- Title model ----------------
    with Timer("title"):
        # char vs word ngrams (representative ranges for each analyzer)
        cfgs = [("char (2,5)", dict(analyzer="char_wb", ngram=(2, 5))),
                ("char (3,6)", dict(analyzer="char_wb", ngram=(3, 6))),
                ("word (1,1)", dict(analyzer="word", ngram=(1, 1))),
                ("word (1,2)", dict(analyzer="word", ngram=(1, 2)))]
        rows = []
        for label, kw in cfgs:
            m = TitleModelRecommender(**kw).fit(prep.train_ds)
            ev = evaluate(m, prep)
            o = overall_row(ev); o["config"] = label
            o["title_only_rprec"] = float(
                ev[ev.scenario == "title_only"].iloc[0].r_precision)
            rows.append(o)
            print(f"    {label:>12}  r_prec={o['r_precision']:.4f}  "
                  f"title_only={o['title_only_rprec']:.4f}")
        df = pd.DataFrame(rows)
        df["model"] = "title"; df["knob"] = "analyzer_ngram"; all_rows.append(df)
        line_plot(range(len(df)),
                  {"OVERALL R-prec": df.r_precision,
                   "title_only R-prec": df.title_only_rprec},
                  "analyzer / ngram", "R-precision", "Title: word vs char ngrams",
                  f"{FIGP}/title_analyzer.png", xticklabels=list(df.config))

        df = sweep(prep, lambda v: TitleModelRecommender(neighbours=v),
                   "neighbours", [10, 25, 50, 100, 200], "char_wb (2,5)")
        df["model"] = "title"; df["knob"] = "neighbours"; all_rows.append(df)
        line_plot(df.neighbours, {"R-precision": df.r_precision, "NDCG": df.ndcg},
                  "neighbour playlists", "score", "Title: neighbour playlists",
                  f"{FIGP}/title_neighbours.png")

    out = pd.concat(all_rows, ignore_index=True)
    save_csv(out, "hyperparams.csv")


if __name__ == "__main__":
    main()
