"""Experiment 3: ensemble analysis.

  A. Leave-one-model-out from the hybrid's candidate union -- how much does each
     candidate source contribute?
  B. Scenario-aware routing: the plain hybrid loses the title-only scenario to
     the title specialist.  Route/blend by seed-count and measure the honest
     gain (title_only up, OVERALL not hurt).

Run:  python experiments/exp_ensemble.py
"""
from __future__ import annotations

import pandas as pd

from _common import (FIG_DIR as FIGP, STD, Timer, bar_plot, evaluate,
                     grouped_bar, overall_row, prepare, save_csv,
                     scenario_value)
from exp_hybrid import fit_submodels, hy
from playlistcont.models.routed import RoutedHybrid

SOURCES = ["item_cf", "als", "track2vec", "title", "popularity"]


def main():
    prep = prepare(STD)
    print(f"Data: {prep.ds.n_playlists} playlists, {len(prep.cases)} cases")
    with Timer("fit submodels"):
        subs = fit_submodels(prep.train_ds)
    all_rows = []

    # ---------- A: leave-one-model-out ----------
    with Timer("leave-one-out"):
        base = hy(subs, sources=SOURCES).fit(prep.train_ds)
        base_ev = evaluate(base, prep)
        base_o = overall_row(base_ev)
        rows = [dict(removed="(full union)", **base_o, delta_rprec=0.0,
                     title_only=scenario_value(base_ev, "title_only", "r_precision"))]
        print(f"    full union       r_prec={base_o['r_precision']:.4f}")
        for s in SOURCES:
            keep = [x for x in SOURCES if x != s]
            m = hy(subs, sources=keep).fit(prep.train_ds)
            ev = evaluate(m, prep); o = overall_row(ev)
            o["removed"] = s
            o["delta_rprec"] = o["r_precision"] - base_o["r_precision"]
            o["title_only"] = scenario_value(ev, "title_only", "r_precision")
            rows.append(o)
            print(f"    -{s:>11}     r_prec={o['r_precision']:.4f} "
                  f"(delta {o['delta_rprec']:+.4f})  "
                  f"title_only={o['title_only']:.4f}")
        df = pd.DataFrame(rows); df["experiment"] = "leave_one_out"
        all_rows.append(df)
        lo = df[df.removed != "(full union)"]
        bar_plot(lo.removed, lo.delta_rprec,
                 "delta OVERALL R-precision when source removed",
                 "Hybrid: leave-one-source-out (more negative = more valuable)",
                 f"{FIGP}/ensemble_leave_one_out.png", color="#E45756")

    # ---------- B: scenario-aware routing ----------
    with Timer("routing"):
        hybrid = hy(subs).fit(prep.train_ds)
        title = subs["title"]
        routed = RoutedHybrid(hybrid, title).fit(prep.train_ds)

        ev_h = evaluate(hybrid, prep)
        ev_t = evaluate(title, prep)
        ev_r = evaluate(routed, prep)

        scen_list = ["title_only", "title_1", "title_5", "title_10",
                     "title_25", "OVERALL"]
        rows = []
        for scen in scen_list:
            rows.append(dict(
                scenario=scen,
                hybrid=scenario_value(ev_h, scen, "r_precision"),
                title=scenario_value(ev_t, scen, "r_precision"),
                routed=scenario_value(ev_r, scen, "r_precision"),
            ))
        df = pd.DataFrame(rows); df["experiment"] = "routing"
        all_rows.append(df)
        for _, r in df.iterrows():
            print(f"    {r.scenario:>12}  hybrid={r.hybrid:.4f} "
                  f"title={r.title:.4f} routed={r.routed:.4f}")
        grouped_bar(
            df.scenario,
            {"hybrid": df.hybrid, "title specialist": df.title, "routed": df.routed},
            "R-precision",
            "Scenario-aware routing recovers title-only without hurting OVERALL",
            f"{FIGP}/ensemble_routing.png")

    out = pd.concat(all_rows, ignore_index=True)
    save_csv(out, "ensemble.csv")


if __name__ == "__main__":
    main()
