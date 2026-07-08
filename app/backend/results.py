"""Load the committed result CSVs and shape them for the Results Explorer.

The dashboard doubles as the project's results viewer, so these readers turn the
files under ``results/`` and ``results/real/`` into compact JSON.  Everything is
read once and cached; the CSVs are small.
"""
from __future__ import annotations

import functools
from pathlib import Path
from typing import List

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULTS = REPO_ROOT / "results"

# canonical model order + display labels (best-on-real first)
MODEL_ORDER = ["item_cf", "hybrid", "routed_hybrid", "als", "taste_engine",
               "track2vec", "title", "popularity"]
MODEL_LABEL = {
    "item_cf": "Item-CF", "hybrid": "Hybrid", "routed_hybrid": "Routed hybrid",
    "als": "ALS", "taste_engine": "Taste Engine", "track2vec": "Track2Vec",
    "title": "Title", "popularity": "Popularity",
}
SCENARIO_ORDER = ["title_only", "title_1", "title_5", "no_title_5", "title_10",
                  "no_title_10", "title_25", "title_random_25", "title_100",
                  "title_random_100"]
SCENARIO_LABEL = {
    "title_only": "Title only", "title_1": "Title + 1", "title_5": "Title + 5",
    "no_title_5": "5, no title", "title_10": "Title + 10",
    "no_title_10": "10, no title", "title_25": "Title + 25",
    "title_random_25": "Title + rand 25", "title_100": "Title + 100",
    "title_random_100": "Title + rand 100",
}


def _order(models: List[str]) -> List[str]:
    known = [m for m in MODEL_ORDER if m in models]
    extra = [m for m in models if m not in MODEL_ORDER]
    return known + extra


@functools.lru_cache(maxsize=None)
def _read(path: str) -> pd.DataFrame:
    return pd.read_csv(RESULTS / path)


def overview() -> dict:
    """Headline OVERALL metrics for every model, synthetic and real."""
    out = {}
    for key, path in (("synthetic", "results.csv"), ("real", "real/real_results.csv")):
        df = _read(path)
        ov = df[df["scenario"] == "OVERALL"].copy()
        rows = []
        for m in _order(list(ov["model"])):
            r = ov[ov["model"] == m].iloc[0]
            rows.append({
                "model": m, "label": MODEL_LABEL.get(m, m),
                "r_precision": round(float(r["r_precision"]), 4),
                "ndcg": round(float(r["ndcg"]), 4),
                "clicks": round(float(r["clicks"]), 3),
            })
        out[key] = rows
    return out


def scenarios(dataset: str = "real", metric: str = "r_precision") -> dict:
    path = "real/real_results.csv" if dataset == "real" else "results.csv"
    df = _read(path)
    df = df[df["scenario"] != "OVERALL"]
    models = _order(list(df["model"].unique()))
    scen = [s for s in SCENARIO_ORDER if s in set(df["scenario"])]
    matrix = []
    for m in models:
        row = []
        for s in scen:
            sub = df[(df["model"] == m) & (df["scenario"] == s)]
            row.append(round(float(sub.iloc[0][metric]), 4) if len(sub) else None)
        matrix.append(row)
    return {
        "dataset": dataset, "metric": metric,
        "models": [{"key": m, "label": MODEL_LABEL.get(m, m)} for m in models],
        "scenarios": [{"key": s, "label": SCENARIO_LABEL.get(s, s)} for s in scen],
        "matrix": matrix,
    }


def held() -> List[dict]:
    df = _read("real/held_analysis.csv")
    out = []
    for _, r in df.iterrows():
        out.append({
            "conclusion": str(r["Synthetic conclusion"]),
            "synthetic": str(r["Synthetic"]),
            "real": str(r["Real"]),
            "verdict": str(r["Verdict"]).strip().lower(),
        })
    return out


def trust_curves() -> dict:
    """Taste-Engine R-precision vs trust, honest vs adversarial (ablation)."""
    df = _read("ablations/taste.csv")
    tr = df[df["knob"] == "trust"].copy() if "knob" in df.columns else pd.DataFrame()
    honest, adversarial = [], []
    for _, r in tr.iterrows():
        pt = {"trust": round(float(r["trust"]), 3),
              "r_precision": round(float(r["r_precision"]), 4),
              "clicks": round(float(r["clicks"]), 3)}
        (honest if r["mode"] == "honest" else adversarial).append(pt)
    honest.sort(key=lambda p: p["trust"])
    adversarial.sort(key=lambda p: p["trust"])
    return {"honest": honest, "adversarial": adversarial}


def taste_real_subset() -> List[dict]:
    """Taste Engine on the real MPD's matched audio-feature subset (OVERALL)."""
    df = _read("real/taste_real.csv")
    ov = df[df["scenario"] == "OVERALL"]
    rows = []
    for m in _order(list(ov["model"])):
        r = ov[ov["model"] == m].iloc[0]
        rows.append({"model": m, "label": MODEL_LABEL.get(m, m),
                     "r_precision": round(float(r["r_precision"]), 4),
                     "ndcg": round(float(r["ndcg"]), 4),
                     "clicks": round(float(r["clicks"]), 3)})
    return rows
