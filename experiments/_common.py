"""Shared harness for the ablation experiments.

Every experiment script imports these helpers so the data pipeline, scenario
construction, evaluation loop and plotting are identical and seeded.  Nothing
here downloads anything: all data is the seeded synthetic MPD.

Standard scale (fast, ~35s for a full 7-model comparison) is defined by
``STD`` below; individual scripts can override it.
"""
from __future__ import annotations

import os
import sys
import time
import warnings
from collections import defaultdict
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "src"))

RESULTS_DIR = os.path.join(ROOT, "results")
ABLATION_DIR = os.path.join(RESULTS_DIR, "ablations")
FIG_DIR = os.path.join(ABLATION_DIR, "figures")
os.makedirs(ABLATION_DIR, exist_ok=True)
os.makedirs(FIG_DIR, exist_ok=True)

from playlistcont.data.schema import Dataset  # noqa: E402
from playlistcont.data.synthetic import make_synthetic  # noqa: E402
from playlistcont.challenge.scenarios import (  # noqa: E402
    build_scenarios, train_test_split, SCENARIO_NAMES,
)
from playlistcont.challenge.metrics import evaluate_one, aggregate  # noqa: E402


@dataclass
class Scale:
    n_playlists: int = 6000
    n_tracks: int = 3500
    n_test: int = 2000
    per_scenario: int = 50
    k: int = 500
    seed: int = 1


STD = Scale()


@dataclass
class Prepared:
    ds: Dataset
    train_ds: Dataset
    cases: list
    track_artist: Dict[str, str]
    scale: Scale


def prepare(scale: Scale = STD, data_seed: Optional[int] = None) -> Prepared:
    """Build synthetic data, split, and construct the 10 scenario cases."""
    seed = scale.seed if data_seed is None else data_seed
    ds = make_synthetic(n_playlists=scale.n_playlists, n_tracks=scale.n_tracks,
                        seed=seed)
    train_pls, test_pls = train_test_split(ds, n_test=scale.n_test, seed=seed)
    train_ds = Dataset(playlists=train_pls, tracks=ds.tracks, features=ds.features)
    cases = build_scenarios(test_pls, per_scenario=scale.per_scenario, seed=seed)
    return Prepared(ds=ds, train_ds=train_ds, cases=cases,
                    track_artist=ds.track_artist_map(), scale=scale)


def evaluate(model, prep: Prepared,
             recommend_fn: Optional[Callable] = None) -> pd.DataFrame:
    """Evaluate one fitted model across all scenarios.

    ``recommend_fn(model, case, k)`` overrides how predictions are produced
    (used to inject ``stated`` preferences for the taste engine).  Returns a
    tidy frame with one row per scenario plus an OVERALL row.
    """
    k = prep.scale.k
    per_scen: Dict[str, list] = defaultdict(list)
    for c in prep.cases:
        if recommend_fn is None:
            recs = model.recommend(c.seed_tracks, c.title, k=k)
        else:
            recs = recommend_fn(model, c, k)
        per_scen[c.scenario].append(evaluate_one(recs, c.holdout, prep.track_artist))
    rows = []
    allcases = []
    for scen in SCENARIO_NAMES:
        per = per_scen.get(scen)
        if not per:
            continue
        allcases.extend(per)
        agg = aggregate(per)
        rows.append(dict(scenario=scen, n=len(per), **agg))
    rows.append(dict(scenario="OVERALL", n=len(allcases), **aggregate(allcases)))
    return pd.DataFrame(rows)


def overall_row(df: pd.DataFrame) -> dict:
    r = df[df.scenario == "OVERALL"].iloc[0]
    return {"r_precision": float(r.r_precision), "ndcg": float(r.ndcg),
            "clicks": float(r.clicks)}


def scenario_value(df: pd.DataFrame, scenario: str, metric: str) -> float:
    sub = df[df.scenario == scenario]
    return float(sub.iloc[0][metric]) if len(sub) else float("nan")


# ---------------------------------------------------------------------------
# plotting helpers
# ---------------------------------------------------------------------------
def _mpl():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def line_plot(x, series: Dict[str, Sequence[float]], xlabel, ylabel, title,
              path, xticklabels=None, logx=False, marker="o"):
    plt = _mpl()
    fig, ax = plt.subplots(figsize=(8, 5))
    for label, ys in series.items():
        ax.plot(x, ys, marker=marker, label=label)
    if logx:
        ax.set_xscale("log")
    if xticklabels is not None:
        ax.set_xticks(x)
        ax.set_xticklabels(xticklabels)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    if len(series) > 1:
        ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)
    print("  wrote", os.path.relpath(path, ROOT))


def bar_plot(labels, values, ylabel, title, path, color="#4C78A8",
             horizontal=False, err=None):
    plt = _mpl()
    fig, ax = plt.subplots(figsize=(9, 5))
    if horizontal:
        ax.barh(labels, values, color=color, xerr=err)
        ax.set_xlabel(ylabel)
    else:
        ax.bar(labels, values, color=color, yerr=err, capsize=3)
        ax.set_ylabel(ylabel)
        plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
    ax.set_title(title)
    ax.grid(axis="x" if horizontal else "y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)
    print("  wrote", os.path.relpath(path, ROOT))


def grouped_bar(labels, group_series: Dict[str, Sequence[float]], ylabel, title,
                path, err_series: Optional[Dict[str, Sequence[float]]] = None):
    plt = _mpl()
    fig, ax = plt.subplots(figsize=(11, 5.5))
    x = np.arange(len(labels))
    n = len(group_series)
    w = 0.8 / max(1, n)
    for i, (name, vals) in enumerate(group_series.items()):
        err = err_series.get(name) if err_series else None
        ax.bar(x + i * w, vals, w, label=name, yerr=err, capsize=2)
    ax.set_xticks(x + 0.4 - w / 2)
    ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=8)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(fontsize=8, ncol=min(4, n))
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)
    print("  wrote", os.path.relpath(path, ROOT))


def save_csv(df: pd.DataFrame, name: str) -> str:
    path = os.path.join(ABLATION_DIR, name)
    df.to_csv(path, index=False)
    print("  wrote", os.path.relpath(path, ROOT))
    return path


class Timer:
    def __init__(self, label):
        self.label = label

    def __enter__(self):
        self.t = time.time()
        print(f"[{self.label}] start")
        return self

    def __exit__(self, *a):
        print(f"[{self.label}] done in {time.time() - self.t:.1f}s")
