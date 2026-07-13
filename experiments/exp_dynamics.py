"""Phase 3 benchmark: which change-point method + representation actually wins?

The scientific centrepiece.  We run the whole method ladder against the *planted*
ground truth of the synthetic history generator, over a grid of

    methods         {baseline_scan, cusum, pelt_l2, pelt_rbf, bocpd}
    representations {scalar_axes, genre_shares, flavor_shares, combined}
    deseasonalize   {off, on}
    seeds           >= 8
    generator       {auto (mixed abrupt+drift), controlled (clean abrupt)}

and score every cell with precision / recall / F1, localisation MAE, segmentation
ARI, and — the honesty headline — trap-immunity and seasonal-immunity false-
positive rates.  Weekly windows, +/-14-day matching tolerance.

WINNER RULE (documented, honest).  We rank configurations by a **composite** that
rewards accuracy and punishes the two immunity failures:

    composite = mean_F1  -  LAMBDA * (trap_fp_rate + seasonal_fp_rate) / 2

with ``LAMBDA = 0.5`` (a false alarm on a trap/seasonal span is worth half an F1
point of accuracy — deliberately steep, because "don't cry wolf on a December
bump" is a product requirement here, not a nicety).  If nothing clears the
baseline by a meaningful composite margin we say so.

GENERATOR NOTE.  ``make_synthetic_history`` forces **abrupt** transitions whenever
explicit ``regimes`` are supplied, so the "controlled" setting is clean abrupt
changes; genuine *drift* transitions are exercised by the "auto" setting, whose
sampler plants drift on ~half of interior transitions.  Both traps and the
December seasonal bump are on in both settings.

Runs in a couple of minutes.  ``python experiments/exp_dynamics.py``.
"""
from __future__ import annotations

import os
import sys
import time
from collections import defaultdict
from datetime import date, timedelta

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "src"))

from playlistcont.dynamics import (  # noqa: E402
    baseline_scan, bocpd, build_windows, cusum, pelt, penalty_sweep,
    score_detections, segmentation_ari,
)
from playlistcont.history.schema import RegimeSpec  # noqa: E402
from playlistcont.history.store import HistoryStore  # noqa: E402
from playlistcont.history.synthetic import make_synthetic_history  # noqa: E402

RESULTS_DIR = os.path.join(ROOT, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

N_SEEDS = 8
START = date(2022, 1, 1)
N_DAYS = 730
TOL = 14
LAMBDA = 0.5
REPRESENTATIONS = ("scalar_axes", "genre_shares", "flavor_shares", "combined")


# ---------------------------------------------------------------------------
# the two generator settings
# ---------------------------------------------------------------------------
def make_history(setting: str, seed: int):
    if setting == "auto":
        return make_synthetic_history(seed=seed, start_date=START, n_days=N_DAYS)
    if setting == "controlled":
        # four well-separated pure/near-pure regimes; generator makes them abrupt.
        names = ["sad slow country", "gym rap", "smooth jazz", "edm rave"]
        bounds = [0, 190, 370, 550, N_DAYS]
        regimes = []
        for i in range(4):
            s = START + timedelta(days=bounds[i])
            e = START + timedelta(days=bounds[i + 1])
            regimes.append(RegimeSpec(s, e, {names[i]: 1.0}, 40.0, f"pure {names[i]}"))
        return make_synthetic_history(
            seed=seed, start_date=START, n_days=N_DAYS, regimes=regimes,
            include_traps=True, seasonal=True)
    raise ValueError(setting)


# ---------------------------------------------------------------------------
# run one detector cell
# ---------------------------------------------------------------------------
def run_method(method, series, store, representation, deseason):
    if method == "baseline_scan":
        return baseline_scan(store)                     # monthly; rep/deseason n/a
    if method == "cusum":
        return cusum(series, representation, deseason)
    if method == "pelt_l2":
        return pelt(series, representation, deseason, cost="l2")
    if method == "pelt_rbf":
        return pelt(series, representation, deseason, cost="rbf")
    if method == "bocpd":
        return bocpd(series, representation, deseason)
    raise ValueError(method)


FANCY = ["cusum", "pelt_l2", "pelt_rbf", "bocpd"]


def main() -> None:
    t0 = time.time()
    rows = []
    drift_transitions = abrupt_transitions = 0

    for setting in ("auto", "controlled"):
        for si in range(N_SEEDS):
            seed = 4000 + si if setting == "auto" else 7000 + si
            h = make_history(setting, seed)
            gt = h.ground_truth
            for c in gt.change_points:
                if c.kind == "abrupt":
                    abrupt_transitions += 1
                elif c.kind == "drift_start":
                    drift_transitions += 1
            store = HistoryStore.from_history(h)
            series = build_windows(store, granularity="week", min_events=30)
            dates = [series.starts[i] for i in series.valid_indices()]

            # baseline once per history (representation/deseason agnostic)
            det = baseline_scan(store)
            sc = score_detections(det, gt, TOL)
            rows.append(_row("baseline_scan", "(native axes)", False,
                             setting, seed, sc, segmentation_ari(det, gt, dates)))

            for rep in REPRESENTATIONS:
                for deseason in (False, True):
                    for method in FANCY:
                        det = run_method(method, series, store, rep, deseason)
                        sc = score_detections(det, gt, TOL)
                        ari = segmentation_ari(det, gt, dates)
                        rows.append(_row(method, rep, deseason, setting, seed, sc, ari))
            store.close()

    df = pd.DataFrame(rows)
    agg = _aggregate(df)
    agg.to_csv(os.path.join(RESULTS_DIR, "dynamics_benchmark.csv"), index=False)
    print("  wrote", os.path.relpath(
        os.path.join(RESULTS_DIR, "dynamics_benchmark.csv"), ROOT))

    _print_summary(agg, drift_transitions, abrupt_transitions)
    _penalty_analysis()
    print(f"\n[done in {time.time() - t0:.1f}s]")


def _row(method, rep, deseason, setting, seed, sc, ari):
    return {
        "method": method, "representation": rep, "deseasonalize": deseason,
        "setting": setting, "seed": seed,
        "f1": sc["f1"], "precision": sc["precision"], "recall": sc["recall"],
        "localization_mae": sc["localization_mae"],
        "trap_fp_rate": sc["trap_fp_rate"], "seasonal_fp_rate": sc["seasonal_fp_rate"],
        "fp_other": sc["fp_other"], "n_det": sc["n_detections"], "ari": ari,
    }


def _aggregate(df: pd.DataFrame) -> pd.DataFrame:
    out = []
    keys = ["method", "representation", "deseasonalize"]
    for key, g in df.groupby(keys, sort=False):
        mean_f1 = g["f1"].mean()
        trap = g["trap_fp_rate"].mean()
        seas = g["seasonal_fp_rate"].mean()
        composite = mean_f1 - LAMBDA * (trap + seas) / 2.0
        out.append({
            "method": key[0], "representation": key[1], "deseasonalize": key[2],
            "n_runs": len(g),
            "mean_f1": mean_f1, "mean_precision": g["precision"].mean(),
            "mean_recall": g["recall"].mean(),
            "mean_loc_mae": np.nanmean(g["localization_mae"].to_numpy(dtype=float)),
            "trap_fp_rate": trap, "seasonal_fp_rate": seas,
            "fp_other_per_run": g["fp_other"].mean(), "mean_ari": g["ari"].mean(),
            "composite": composite,
        })
    return pd.DataFrame(out).sort_values("composite", ascending=False).reset_index(drop=True)


def _print_summary(agg: pd.DataFrame, drift, abrupt):
    print(f"\nGenerator planted: {abrupt} abrupt + {drift} drift transitions "
          f"across {2 * N_SEEDS} histories.")
    print("\n=== Top 12 configurations by composite (F1 - "
          f"{LAMBDA}*(trap+seasonal FP rate)/2) ===")
    cols = ["method", "representation", "deseasonalize", "mean_f1", "mean_precision",
            "mean_recall", "mean_loc_mae", "trap_fp_rate", "seasonal_fp_rate",
            "mean_ari", "composite"]
    with pd.option_context("display.width", 200, "display.max_columns", 20):
        print(agg[cols].head(12).to_string(index=False,
              float_format=lambda x: f"{x:.3f}"))

    base = agg[agg.method == "baseline_scan"].iloc[0]
    win = agg.iloc[0]
    print("\n=== BASELINE (Phase-2 adjacent-month scan) ===")
    print(f"  F1={base.mean_f1:.3f}  trap_fp={base.trap_fp_rate:.3f}  "
          f"seasonal_fp={base.seasonal_fp_rate:.3f}  composite={base.composite:.3f}")
    print("\n=== WINNER ===")
    print(f"  {win.method} / {win.representation} / deseason={win.deseasonalize}")
    print(f"  F1={win.mean_f1:.3f}  P={win.mean_precision:.3f}  R={win.mean_recall:.3f}  "
          f"loc_MAE={win.mean_loc_mae:.1f}d  trap_fp={win.trap_fp_rate:.3f}  "
          f"seasonal_fp={win.seasonal_fp_rate:.3f}  ARI={win.mean_ari:.3f}")
    margin = win.composite - base.composite
    print(f"  composite {win.composite:.3f} vs baseline {base.composite:.3f} "
          f"(margin {margin:+.3f})")
    if margin < 0.03:
        print("  VERDICT: no method beats the baseline by a meaningful margin.")
    else:
        print(f"  VERDICT: winner beats the baseline composite by {margin:.3f}.")

    # representation ablation: best composite per representation (fancy methods only)
    print("\n=== Representation ablation (best composite per representation, fancy methods) ===")
    fancy = agg[agg.method != "baseline_scan"]
    for rep in REPRESENTATIONS:
        sub = fancy[fancy.representation == rep]
        if len(sub):
            b = sub.iloc[0]
            print(f"  {rep:14s} best: {b.method:9s} deseason={str(b.deseasonalize):5s} "
                  f"F1={b.mean_f1:.3f} composite={b.composite:.3f}")

    # seasonal on/off effect for the baseline-prone methods
    print("\n=== Deseasonalize on/off effect (mean over reps, per method) ===")
    for method in FANCY:
        m = agg[agg.method == method]
        off = m[m.deseasonalize == False]  # noqa: E712
        on = m[m.deseasonalize == True]    # noqa: E712
        if len(off) and len(on):
            print(f"  {method:9s} seasonal_fp off={off.seasonal_fp_rate.mean():.3f} "
                  f"-> on={on.seasonal_fp_rate.mean():.3f}   "
                  f"F1 off={off.mean_f1.mean():.3f} -> on={on.mean_f1.mean():.3f}")


def _penalty_analysis():
    print("\n=== PELT-l2 penalty sweep (combined rep, auto setting) ===")
    counts_by_pen = defaultdict(list)
    elbows = []
    for si in range(N_SEEDS):
        h = make_history("auto", 4000 + si)
        store = HistoryStore.from_history(h)
        series = build_windows(store, granularity="week", min_events=30)
        sw = penalty_sweep(series, "combined", cost="l2")
        for p, c in zip(sw["penalties"], sw["n_changes"]):
            counts_by_pen[round(p, 1)].append(c)
        elbows.append(sw["elbow_penalty"])
        store.close()
    pens = sorted(counts_by_pen)
    print("  penalty -> mean #changes (8 seeds):")
    for p in pens[::3]:
        print(f"    pen={p:7.1f}  mean#={np.mean(counts_by_pen[p]):.2f}")
    print(f"  elbow-picked penalty: mean={np.mean(elbows):.1f} std={np.std(elbows):.1f}")


if __name__ == "__main__":
    main()
