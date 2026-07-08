"""Experiment 2: taste-engine ablations (the signature model).

Covers:
  A. stated vs learned vs blended across the trust knob (0..1) -- honest user.
  B. adversarial test: user states the OPPOSITE of their behaviour; show that
     leaning on learned weights (low trust) rescues performance.
  C. axis ablation: zero each interpretable axis, measure the R-precision hit.
  D. flavor-cluster count k: recommendation metric (flat -> flavors are for
     explanation) plus cluster quality (ARI/purity) vs the true archetypes.
  E. cf-score vs axis-match weighting sweep.

Stated preferences are injected per case as an *oracle honest* direction: the
standardized centroid of the user's full playlist (seed + holdout).  The
adversarial variant simply negates it.

Run:  python experiments/exp_taste.py
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score

from _common import (FIG_DIR as FIGP, STD, Timer, evaluate, line_plot,
                     overall_row, prepare, save_csv)
from playlistcont.data.schema import AXES, N_AXES
from playlistcont.models.taste_engine import TasteEngine, _l2


def stated_dict_for(engine, case, adversarial=False):
    """Oracle 'honest' stated preference from the full playlist direction."""
    full = list(case.seed_tracks) + list(case.holdout)
    ids = engine.index.ids(full)
    if not ids:
        return None
    vec = _l2(engine.Z[ids].mean(axis=0))
    if adversarial:
        vec = -vec
    return {AXES[i]: float(vec[i]) for i in range(N_AXES)}


def make_recommend_fn(engine, adversarial=False):
    def fn(model, case, k):
        stated = stated_dict_for(engine, case, adversarial)
        return model.recommend(case.seed_tracks, case.title, k=k, stated=stated)
    return fn


def archetype_of(name: str) -> str:
    return name.split(" Song ")[0].strip().lower()


def main():
    prep = prepare(STD)
    print(f"Data: {prep.ds.n_playlists} playlists, {len(prep.cases)} cases")
    with Timer("fit taste engine"):
        eng = TasteEngine().fit(prep.train_ds)
    all_rows = []
    trusts = [0.0, 0.25, 0.5, 0.75, 1.0]

    # ---------- A: honest stated, trust sweep ----------
    with Timer("A trust (honest)"):
        rows = []
        for t in trusts:
            eng.trust = t
            o = overall_row(evaluate(eng, prep, make_recommend_fn(eng, False)))
            o["trust"] = t; rows.append(o)
            print(f"    trust={t:.2f} (honest)  r_prec={o['r_precision']:.4f}")
        dfh = pd.DataFrame(rows); dfh["mode"] = "honest"

    # ---------- B: adversarial stated, trust sweep ----------
    with Timer("B trust (adversarial)"):
        rows = []
        for t in trusts:
            eng.trust = t
            o = overall_row(evaluate(eng, prep, make_recommend_fn(eng, True)))
            o["trust"] = t; rows.append(o)
            print(f"    trust={t:.2f} (adversarial)  r_prec={o['r_precision']:.4f}")
        dfa = pd.DataFrame(rows); dfa["mode"] = "adversarial"
    df_trust = pd.concat([dfh, dfa], ignore_index=True)
    df_trust["knob"] = "trust"; all_rows.append(df_trust)
    line_plot(trusts,
              {"honest stated": dfh.r_precision,
               "adversarial stated": dfa.r_precision},
              "trust in stated preferences (0=learned only, 1=stated only)",
              "OVERALL R-precision",
              "Taste engine: stated vs learned across trust",
              f"{FIGP}/taste_trust.png")
    eng.trust = 0.4  # restore default

    # ---------- C: axis ablation (learned only) ----------
    with Timer("C axis ablation"):
        Z0 = eng.Z.copy()
        base = overall_row(evaluate(eng, prep))
        rows = [dict(axis="(none)", **base, delta=0.0)]
        print(f"    baseline r_prec={base['r_precision']:.4f}")
        for j, ax in enumerate(AXES):
            eng.Z = Z0.copy(); eng.Z[:, j] = 0.0
            o = overall_row(evaluate(eng, prep))
            o["axis"] = ax; o["delta"] = o["r_precision"] - base["r_precision"]
            rows.append(o)
            print(f"    drop {ax:>16}  r_prec={o['r_precision']:.4f} "
                  f"(delta {o['delta']:+.4f})")
        eng.Z = Z0
        dfax = pd.DataFrame(rows); dfax["knob"] = "axis_ablation"
        all_rows.append(dfax)
        abl = dfax[dfax.axis != "(none)"].sort_values("delta")
        from _common import bar_plot
        bar_plot(abl.axis, abl.delta, "delta R-precision when axis dropped",
                 "Taste engine: axis ablation (more negative = carries signal)",
                 f"{FIGP}/taste_axis_ablation.png", color="#E45756",
                 horizontal=True)

    # ---------- D: flavor count k (metric flat + cluster quality) ----------
    with Timer("D flavor k"):
        ks = [1, 2, 3, 5, 8]
        # recommendation metric vs k (expected ~flat: flavors are explanatory)
        rec_rows = []
        for kk in ks:
            eng.n_flavors = kk
            o = overall_row(evaluate(eng, prep)); o["k"] = kk
            rec_rows.append(o)
        eng.n_flavors = 3
        # cluster quality vs k on a sample of test playlists
        sample = [c for c in prep.cases if len(c.holdout) >= 15][:120]
        qual_rows = []
        for kk in ks:
            aris, purs = [], []
            for c in sample:
                uris = (list(c.seed_tracks) + list(c.holdout))
                ids = eng.index.ids(uris)
                if len(ids) < kk + 1:
                    continue
                Xu = eng.X[ids]
                labels_true = [archetype_of(prep.ds.tracks[u].track_name)
                               for u in uris if u in eng.index.uri_to_id]
                labels_true = labels_true[:len(ids)]
                km = KMeans(n_clusters=kk, n_init=4, random_state=0).fit(Xu)
                aris.append(adjusted_rand_score(labels_true, km.labels_))
                # purity
                pur = 0
                for c_id in set(km.labels_):
                    mask = km.labels_ == c_id
                    lt = [labels_true[i] for i in range(len(labels_true)) if mask[i]]
                    if lt:
                        pur += max([lt.count(x) for x in set(lt)])
                purs.append(pur / len(labels_true))
            qual_rows.append(dict(k=kk, ari=float(np.mean(aris)),
                                  purity=float(np.mean(purs)),
                                  rec_rprec=rec_rows[ks.index(kk)]["r_precision"]))
            print(f"    k={kk}  ARI={np.mean(aris):.3f} purity={np.mean(purs):.3f} "
                  f"rec_rprec={rec_rows[ks.index(kk)]['r_precision']:.4f}")
        dfk = pd.DataFrame(qual_rows); dfk["knob"] = "flavor_k"; all_rows.append(dfk)
        line_plot(ks, {"cluster ARI vs archetypes": dfk.ari,
                       "cluster purity": dfk.purity,
                       "recommendation R-precision": dfk.rec_rprec},
                  "number of flavor clusters k", "score",
                  "Taste engine: flavor count (quality vs ranking impact)",
                  f"{FIGP}/taste_flavor_k.png")

    # ---------- E: cf-weight vs axis-weight ----------
    with Timer("E cf vs axis weight"):
        rows = []
        for cfw in [0.0, 0.15, 0.3, 0.6, 1.0, 2.0]:
            eng.cf_weight = cfw; eng.axis_weight = 1.0
            o = overall_row(evaluate(eng, prep)); o["cf_weight"] = cfw
            rows.append(o)
            print(f"    cf_weight={cfw:.2f} (axis_weight=1)  "
                  f"r_prec={o['r_precision']:.4f}")
        eng.cf_weight = 0.6
        dfc = pd.DataFrame(rows); dfc["knob"] = "cf_weight"; all_rows.append(dfc)
        line_plot(dfc.cf_weight, {"R-precision": dfc.r_precision, "NDCG": dfc.ndcg},
                  "cf_weight (axis_weight fixed at 1.0)", "score",
                  "Taste engine: CF term vs axis-match term",
                  f"{FIGP}/taste_cf_weight.png")

    out = pd.concat(all_rows, ignore_index=True)
    save_csv(out, "taste.csv")


if __name__ == "__main__":
    main()
