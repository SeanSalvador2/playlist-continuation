"""Calibration of the Phase-2 statistics layer — the honesty centrepiece.

The claim the whole module rests on: at listening-history sample sizes a bare p-value
is worthless, and only Benjamini-Hochberg + an effect-size floor turns a change scan
into an honest detector.  This script *measures* that claim two ways and writes the
numbers into ``results/stats_calibration.csv`` (and the ``stats.py`` module docstring).

1. FALSE-POSITIVE control on STATIONARY data.  We generate single-regime histories
   (``regimes=[one RegimeSpec]``, ``include_traps=False``, ``seasonal=False`` — a
   genuinely piecewise-constant taste with no planted change anywhere), run
   ``adjacent_scan`` at month granularity across many seeds, and compare:
     * the RAW-p false-positive rate at alpha=0.05 across every individual axis test
       (theory says ~5%), versus
     * the fraction of month boundaries that survive BH + the effect-size floor
       (should be ~0 — there is no real change to find).

2. SENSITIVITY on DEFAULT data.  We generate the normal planted-change histories and
   ask how many of the planted regime transitions the corrected scan flags, with a
   +/-1-month tolerance.

Runs in well under a minute.  ``python experiments/exp_stats_calibration.py``.
"""
from __future__ import annotations

import os
import sys
from datetime import date, timedelta

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "src"))

from playlistcont.analytics import stats  # noqa: E402
from playlistcont.data.synthetic import ARCHETYPES  # noqa: E402
from playlistcont.history.schema import RegimeSpec  # noqa: E402
from playlistcont.history.store import HistoryStore  # noqa: E402
from playlistcont.history.synthetic import make_synthetic_history  # noqa: E402

RESULTS_DIR = os.path.join(ROOT, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

N_SEEDS = 24
START = date(2022, 1, 1)
N_DAYS = 730
ALPHA = 0.05
# A stable two-archetype mixture: realistic within-window variance, zero change.
STATIONARY_MIX = {"indie chill": 0.7, "classic rock": 0.3}


def _month_index(d: date) -> int:
    return d.year * 12 + (d.month - 1)


def _stationary_store(seed: int) -> HistoryStore:
    regime = RegimeSpec(
        start=START, end=START + timedelta(days=N_DAYS),
        mixture=dict(STATIONARY_MIX), events_per_day=40.0, label="stationary",
    )
    h = make_synthetic_history(
        seed=seed, start_date=START, n_days=N_DAYS,
        regimes=[regime], include_traps=False, seasonal=False,
    )
    return HistoryStore.from_history(h)


def run_stationary() -> pd.DataFrame:
    rows = []
    tot_tests = tot_raw_hits = tot_bounds = tot_flagged = 0
    for i in range(N_SEEDS):
        st = _stationary_store(seed=1000 + i)
        scan = stats.adjacent_scan(st, granularity="month", metrics="axes")
        n_tests = scan["family_size"]
        raw_hits = sum(
            1 for b in scan["boundaries"] for r in b["metrics"] if r["p_raw"] < ALPHA)
        n_bounds = len(scan["boundaries"])
        n_flagged = scan["n_flagged"]
        rows.append({
            "mode": "stationary", "seed": 1000 + i,
            "n_axis_tests": n_tests, "raw_p_hits": raw_hits,
            "raw_fp_rate": raw_hits / n_tests if n_tests else 0.0,
            "n_boundaries": n_bounds, "n_flagged_corrected": n_flagged,
            "corrected_flag_rate": n_flagged / n_bounds if n_bounds else 0.0,
        })
        tot_tests += n_tests
        tot_raw_hits += raw_hits
        tot_bounds += n_bounds
        tot_flagged += n_flagged
        st.close()
    df = pd.DataFrame(rows)
    df.attrs["raw_fp_rate"] = tot_raw_hits / tot_tests if tot_tests else 0.0
    df.attrs["corrected_flag_rate"] = tot_flagged / tot_bounds if tot_bounds else 0.0
    df.attrs["tot_tests"] = tot_tests
    df.attrs["tot_bounds"] = tot_bounds
    return df


def run_default() -> pd.DataFrame:
    rows = []
    tot_trans = tot_detected = 0
    for i in range(N_SEEDS):
        h = make_synthetic_history(seed=2000 + i, start_date=START, n_days=N_DAYS)
        st = HistoryStore.from_history(h)
        scan = stats.adjacent_scan(st, granularity="month", metrics="axes")
        flagged_months = [_month_index(date.fromisoformat(b["b"]["start"]))
                          for b in scan["boundaries"] if b["flagged"]]
        # ground-truth transitions = interior regime boundaries
        transitions = [r.start for r in h.ground_truth.regimes[1:]]
        detected = 0
        for t in transitions:
            ti = _month_index(t)
            if any(abs(fi - ti) <= 1 for fi in flagged_months):
                detected += 1
        rows.append({
            "mode": "default", "seed": 2000 + i,
            "n_transitions": len(transitions), "n_detected": detected,
            "sensitivity": detected / len(transitions) if transitions else 0.0,
            "n_flagged_boundaries": len(flagged_months),
        })
        tot_trans += len(transitions)
        tot_detected += detected
        st.close()
    df = pd.DataFrame(rows)
    df.attrs["sensitivity"] = tot_detected / tot_trans if tot_trans else 0.0
    df.attrs["tot_transitions"] = tot_trans
    df.attrs["tot_detected"] = tot_detected
    return df


def main() -> None:
    print(f"Stationary false-positive control ({N_SEEDS} seeds, {N_DAYS}-day single regime)...")
    stat_df = run_stationary()
    print(f"  raw-p FP rate @ alpha={ALPHA}: {stat_df.attrs['raw_fp_rate']*100:.2f}% "
          f"({stat_df.attrs['tot_tests']} axis tests)")
    print(f"  BH + effect-floor flag rate:  {stat_df.attrs['corrected_flag_rate']*100:.2f}% "
          f"({stat_df.attrs['tot_bounds']} month boundaries)")

    print(f"\nDefault planted-change sensitivity ({N_SEEDS} seeds, +/-1 month tolerance)...")
    def_df = run_default()
    print(f"  planted transitions flagged: {def_df.attrs['sensitivity']*100:.1f}% "
          f"({def_df.attrs['tot_detected']}/{def_df.attrs['tot_transitions']})")

    out = pd.concat([stat_df, def_df], ignore_index=True)
    path = os.path.join(RESULTS_DIR, "stats_calibration.csv")
    out.to_csv(path, index=False)
    print("\n  wrote", os.path.relpath(path, ROOT))

    print("\nHEADLINE for the stats.py docstring:")
    print(f"  stationary raw-p FP  ~ {stat_df.attrs['raw_fp_rate']*100:.1f}%  "
          f"-> corrected {stat_df.attrs['corrected_flag_rate']*100:.1f}%")
    print(f"  default sensitivity  = {def_df.attrs['sensitivity']*100:.0f}% "
          f"of planted transitions (+/-1 month)")


if __name__ == "__main__":
    main()
