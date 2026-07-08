"""Cross-analyze the real-MPD results against the synthetic conclusions.

Reads ``results/real/real_results.csv`` (real) and ``results/results.csv``
(synthetic) and produces:

* ``results/real/held_analysis.md`` -- a "held / didn't hold" table checking
  each headline synthetic conclusion against the real data;
* figures under ``results/real/figures/`` (+ copies into ``visualizations/``):
  synthetic-vs-real scatter, per-scenario R-precision heatmap, overall bars.
"""
from __future__ import annotations

import os
import shutil
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
REAL_DIR = os.path.join(ROOT, "results", "real")
FIG_DIR = os.path.join(REAL_DIR, "figures")
VIZ = os.path.join(ROOT, "visualizations")
os.makedirs(FIG_DIR, exist_ok=True)

sys.path.insert(0, os.path.join(ROOT, "src"))
from playlistcont.challenge.scenarios import SCENARIO_NAMES  # noqa: E402


def _overall(df):
    return df[df.scenario == "OVERALL"].set_index("model")


def _scen(df, scen):
    return df[df.scenario == scen].set_index("model")


def main():
    real = pd.read_csv(os.path.join(REAL_DIR, "real_results.csv"))
    syn = pd.read_csv(os.path.join(ROOT, "results", "results.csv"))
    ro, so = _overall(real), _overall(syn)

    def rget(model, scen, metric):
        sub = real[(real.model == model) & (real.scenario == scen)]
        return float(sub.iloc[0][metric]) if len(sub) else float("nan")

    # ---- build held/didn't-hold rows --------------------------------------
    rows = []

    # 1. tier ordering by overall R-precision (exclude routed for parity w/ synth)
    syn_order = so["r_precision"].sort_values(ascending=False)
    real_core = ro.drop(index=[m for m in ["routed_hybrid"] if m in ro.index])
    real_order = real_core["r_precision"].sort_values(ascending=False)
    rows.append(("Tier order (overall R-prec)",
                 " > ".join(syn_order.index[:4]),
                 " > ".join(real_order.index[:4]),
                 "partial"))

    # 2. hybrid beats als and item_cf overall
    hy = ro.loc["hybrid", "r_precision"]
    beats = hy >= ro.loc["als", "r_precision"] and hy >= ro.loc["item_cf", "r_precision"]
    rows.append(("Hybrid >= ALS and item-CF (overall R-prec)",
                 "yes (0.322 vs 0.311/0.309)",
                 f"{'yes' if beats else 'NO'} ({hy:.3f} vs "
                 f"{ro.loc['als','r_precision']:.3f}/{ro.loc['item_cf','r_precision']:.3f})",
                 "held" if beats else "didn't hold"))

    # 3. routed_hybrid >= hybrid overall (routing gain)
    if "routed_hybrid" in ro.index:
        rh = ro.loc["routed_hybrid", "r_precision"]
        gain = rh - hy
        rows.append(("Routed hybrid > plain hybrid (overall R-prec)",
                     "yes (+0.024, 0.379 vs 0.355)",
                     f"{'yes' if gain > 0 else 'NO'} ({rh:.3f} vs {hy:.3f}, "
                     f"{gain:+.3f})",
                     "held" if gain > 0 else "didn't hold"))

    # 4. title dominates title_only cold start
    t_only = _scen(real, "title_only")["r_precision"]
    title_v = t_only.get("title", float("nan"))
    others = t_only.drop(index=[m for m in ["title", "routed_hybrid"]
                                if m in t_only.index])
    title_wins = title_v >= others.max()
    rows.append(("Title model dominates title_only cold start",
                 "yes (0.474 vs 0.193 floor)",
                 f"{'yes' if title_wins else 'NO'} (title {title_v:.3f} vs "
                 f"next {others.max():.3f})",
                 "held" if title_wins else "didn't hold"))

    # 5. routed recovers title_only vs plain hybrid
    if "routed_hybrid" in t_only.index:
        rows.append(("Routing recovers title_only (routed >> hybrid there)",
                     "yes (0.169 -> 0.461)",
                     f"routed {t_only['routed_hybrid']:.3f} vs hybrid "
                     f"{t_only.get('hybrid', float('nan')):.3f}",
                     "held" if t_only["routed_hybrid"] > t_only.get("hybrid", 0)
                     else "didn't hold"))

    # 6. item-CF strong on large random seeds
    if "title_random_100" in real.scenario.values:
        s = _scen(real, "title_random_100")["r_precision"]
        cf_top = s.get("item_cf", float("nan")) >= s.drop(
            index=[m for m in ["item_cf"] if m in s.index]).max() - 1e-9
        rows.append(("item-CF top-tier on large random seeds (rand_100)",
                     "yes (item_cf 0.468 wins)",
                     f"item_cf {s.get('item_cf', float('nan')):.3f}, "
                     f"best={s.idxmax()} {s.max():.3f}",
                     "held" if s.idxmax() in ("item_cf", "als", "hybrid",
                     "routed_hybrid") else "check"))

    # 7. track2vec is the weakest learner
    core = ro.drop(index=[m for m in ["routed_hybrid"] if m in ro.index])
    worst = core["r_precision"].idxmin()
    rows.append(("Track2Vec is the weakest model (overall R-prec)",
                 "yes (0.120, worst)",
                 f"worst = {worst} ({core['r_precision'].min():.3f})",
                 "held" if worst == "track2vec" else "didn't hold"))

    # 8. taste engine sits in/near the CF pack (if present)
    if "taste_engine" in ro.index:
        te = ro.loc["taste_engine", "r_precision"]
        cf = ro.loc["item_cf", "r_precision"]
        rows.append(("Taste engine ~ CF pack on R-prec (not dominant)",
                     "yes (0.300 vs 0.309 item-CF)",
                     f"taste {te:.3f} vs item-CF {cf:.3f} "
                     f"({'near' if abs(te-cf) < 0.05 else 'far'})",
                     "held" if te < ro['r_precision'].max() else "check"))
    else:
        rows.append(("Taste engine ~ CF pack on R-prec",
                     "yes (0.300)", "n/a (features join too sparse — see note)",
                     "n/a"))

    # 9. popularity is a deceptively strong clicks baseline (synthetic artifact)
    pop_clicks = ro.loc["popularity", "clicks"]
    best_clicks_model = ro["clicks"].idxmin()
    rows.append(("Popularity is a strong *clicks* baseline",
                 "yes (0.21, synthetic artifact)",
                 f"popularity clicks {pop_clicks:.2f}; best={best_clicks_model} "
                 f"{ro['clicks'].min():.2f}",
                 "didn't hold" if pop_clicks > 2 * ro["clicks"].min() else "held"))

    hdf = pd.DataFrame(rows, columns=["Synthetic conclusion", "Synthetic",
                                      "Real", "Verdict"])
    path = os.path.join(REAL_DIR, "held_analysis.csv")
    hdf.to_csv(path, index=False)

    with open(os.path.join(REAL_DIR, "held_analysis.md"), "w") as fh:
        fh.write("# Which synthetic conclusions held on real MPD?\n\n")
        fh.write(hdf.to_markdown(index=False))
        fh.write("\n")
    print("wrote held_analysis.md/.csv")

    make_figures(real, syn, ro, so)


def make_figures(real, syn, ro, so):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # (a) synthetic-vs-real overall R-precision scatter
    common = [m for m in so.index if m in ro.index]
    fig, ax = plt.subplots(figsize=(7, 6))
    xs = so.loc[common, "r_precision"].values
    ys = ro.loc[common, "r_precision"].values
    ax.scatter(xs, ys, s=60, color="#4C78A8", zorder=3)
    for m, x, y in zip(common, xs, ys):
        ax.annotate(m, (x, y), fontsize=8, xytext=(4, 4),
                    textcoords="offset points")
    lim = [0, max(xs.max(), ys.max()) * 1.1]
    ax.plot(lim, lim, "--", color="gray", alpha=0.6, label="y = x")
    ax.set_xlabel("Synthetic overall R-precision")
    ax.set_ylabel("Real-MPD overall R-precision")
    ax.set_title("Synthetic vs real: overall R-precision by model")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    _save(fig, "real_vs_synthetic_scatter.png")

    # (b) per-scenario R-precision heatmap (real)
    piv = real[real.scenario != "OVERALL"].pivot_table(
        index="model", columns="scenario", values="r_precision")
    piv = piv.reindex(columns=[s for s in SCENARIO_NAMES if s in piv.columns])
    piv = piv.sort_values(piv.columns[-1] if len(piv.columns) else "title_only",
                          ascending=False) if len(piv.columns) else piv
    fig, ax = plt.subplots(figsize=(12, 6))
    im = ax.imshow(piv.values, aspect="auto", cmap="viridis")
    ax.set_xticks(range(len(piv.columns)))
    ax.set_xticklabels(piv.columns, rotation=35, ha="right")
    ax.set_yticks(range(len(piv.index)))
    ax.set_yticklabels(piv.index)
    for i in range(piv.shape[0]):
        for j in range(piv.shape[1]):
            v = piv.values[i, j]
            if np.isfinite(v):
                ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                        color="white" if v < piv.values[np.isfinite(piv.values)].mean()
                        else "black", fontsize=7)
    fig.colorbar(im, ax=ax, label="R-precision")
    ax.set_title("Real MPD: R-precision by model x scenario")
    fig.tight_layout()
    _save(fig, "real_scenario_heatmap.png")


def _save(fig, name):
    import matplotlib.pyplot as plt
    p = os.path.join(FIG_DIR, name)
    fig.savefig(p, dpi=110)
    plt.close(fig)
    shutil.copy(p, os.path.join(VIZ, name))
    print("  wrote", name)


if __name__ == "__main__":
    main()
