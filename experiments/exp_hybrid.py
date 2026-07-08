"""Experiment 1e + 3a: hybrid hyperparameters, reranker choice, feature and
candidate-source ablations.

Submodels are fit once and shared across every hybrid config, so only the
candidate pooling + reranker are re-done per config.

Run:  python experiments/exp_hybrid.py
"""
from __future__ import annotations

import pandas as pd

from _common import (FIG_DIR as FIGP, STD, Timer, bar_plot, evaluate,
                     line_plot, overall_row, prepare, save_csv)
from playlistcont.models.hybrid import FEATURES, HybridRecommender
from playlistcont.models.itemcf import ItemCFRecommender
from playlistcont.models.mf import ALSRecommender
from playlistcont.models.popularity import PopularityRecommender
from playlistcont.models.title_model import TitleModelRecommender
from playlistcont.models.track2vec import Track2VecRecommender


def fit_submodels(train_ds):
    subs = {
        "popularity": PopularityRecommender().fit(train_ds),
        "item_cf": ItemCFRecommender().fit(train_ds),
        "als": ALSRecommender().fit(train_ds),
        "track2vec": Track2VecRecommender(epochs=10).fit(train_ds),
        "title": TitleModelRecommender().fit(train_ds),
    }
    return subs


def hy(subs, **kw):
    return HybridRecommender(submodels=subs, n_train_playlists=600, **kw)


def main():
    prep = prepare(STD)
    print(f"Data: {prep.ds.n_playlists} playlists, {len(prep.cases)} cases")
    with Timer("fit submodels"):
        subs = fit_submodels(prep.train_ds)
    all_rows = []

    # ---------------- candidate-pool size ----------------
    with Timer("cand_per_model"):
        rows = []
        for v in [100, 200, 400, 800]:
            m = hy(subs, cand_per_model=v).fit(prep.train_ds)
            o = overall_row(evaluate(m, prep)); o["cand_per_model"] = v
            rows.append(o)
            print(f"    cand={v:>4}  r_prec={o['r_precision']:.4f} "
                  f"ndcg={o['ndcg']:.4f}")
        df = pd.DataFrame(rows); df["knob"] = "cand_per_model"; all_rows.append(df)
        line_plot(df.cand_per_model,
                  {"R-precision": df.r_precision, "NDCG": df.ndcg},
                  "candidates per source", "score",
                  "Hybrid: candidate-pool size", f"{FIGP}/hybrid_candpool.png")

    # ---------------- reranker choice ----------------
    with Timer("reranker"):
        rows = []
        for rr in ["equal", "logreg", "lightgbm"]:
            m = hy(subs, reranker=rr).fit(prep.train_ds)
            o = overall_row(evaluate(m, prep)); o["reranker"] = rr
            o["backend"] = m.blender_backend
            rows.append(o)
            print(f"    reranker={rr:>9} ({m.blender_backend})  "
                  f"r_prec={o['r_precision']:.4f} ndcg={o['ndcg']:.4f}")
        df = pd.DataFrame(rows); df["knob"] = "reranker"; all_rows.append(df)
        bar_plot(df.reranker, df.r_precision, "OVERALL R-precision",
                 "Hybrid: reranker (equal blend vs logistic vs LightGBM)",
                 f"{FIGP}/hybrid_reranker.png")

    # ---------------- feature ablation (drop each) ----------------
    with Timer("feature ablation"):
        base = hy(subs).fit(prep.train_ds)
        base_o = overall_row(evaluate(base, prep))
        rows = [dict(dropped="(none)", **base_o,
                     delta_rprec=0.0)]
        print(f"    baseline           r_prec={base_o['r_precision']:.4f}")
        for f in FEATURES:
            m = hy(subs, drop_features=[f]).fit(prep.train_ds)
            o = overall_row(evaluate(m, prep))
            o["dropped"] = f
            o["delta_rprec"] = o["r_precision"] - base_o["r_precision"]
            rows.append(o)
            print(f"    drop {f:>14}  r_prec={o['r_precision']:.4f} "
                  f"(delta {o['delta_rprec']:+.4f})")
        df = pd.DataFrame(rows); df["knob"] = "feature_ablation"; all_rows.append(df)
        abl = df[df.dropped != "(none)"]
        bar_plot(abl.dropped, abl.delta_rprec,
                 "delta R-precision when dropped",
                 "Hybrid: feature ablation (more negative = more important)",
                 f"{FIGP}/hybrid_feature_ablation.png", color="#E45756")

    out = pd.concat(all_rows, ignore_index=True)
    save_csv(out, "hybrid.csv")


if __name__ == "__main__":
    main()
