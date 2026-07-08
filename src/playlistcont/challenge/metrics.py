"""The three official RecSys-2018 metrics.

References for exact definitions:
  https://www.aicrowd.com/challenges/spotify-million-playlist-dataset-challenge
  "Recsys Challenge 2018: Automatic Music Playlist Continuation" (Chen et al.)

All three operate on an ordered list of predicted ``track_uri`` (the seed
tracks already removed) versus the held-out ground-truth set.

* **R-precision** — track matches in the first R predictions, plus 0.25 credit
  for *artist* matches (an artist the ground truth contains but whose exact
  track we missed), divided by R = |ground truth|.  Capped at 1.0.
* **NDCG** — normalized discounted cumulative gain over the full prediction
  list, binary relevance.
* **Recommended Songs Clicks** — floor(rank_of_first_hit / 10); the UI reveals
  10 songs per "refresh", so this is how many refreshes until the first hit.
  If no hit in the (up to 500) predictions, the penalty is 51.
"""
from __future__ import annotations

import math
from collections import Counter
from typing import Dict, List, Mapping, Sequence

MAX_CLICKS = 51


def r_precision(
    predicted: Sequence[str],
    ground_truth: Sequence[str],
    track_artist: Mapping[str, str] | None = None,
) -> float:
    """R-precision with 0.25 artist-match credit (official variant)."""
    gt = list(ground_truth)
    R = len(gt)
    if R == 0:
        return 0.0
    gt_tracks = set(gt)
    top = list(predicted[:R])

    track_hits = len(gt_tracks.intersection(top))

    artist_credit = 0.0
    if track_artist is not None:
        # Remaining artist demand after exact-track matches are removed.
        matched_tracks = gt_tracks.intersection(top)
        remaining = Counter(
            track_artist.get(t) for t in gt if t not in matched_tracks
        )
        remaining.pop(None, None)
        for t in top:
            if t in gt_tracks:
                continue
            a = track_artist.get(t)
            if a is not None and remaining.get(a, 0) > 0:
                remaining[a] -= 1
                artist_credit += 0.25
    score = (track_hits + artist_credit) / R
    return min(score, 1.0)


def ndcg(predicted: Sequence[str], ground_truth: Sequence[str]) -> float:
    """Normalized DCG with binary relevance."""
    gt = set(ground_truth)
    if not gt:
        return 0.0
    dcg = 0.0
    for i, t in enumerate(predicted):
        if t in gt:
            dcg += 1.0 / math.log2(i + 2)
    idcg = sum(1.0 / math.log2(i + 2) for i in range(len(gt)))
    return dcg / idcg if idcg > 0 else 0.0


def clicks(predicted: Sequence[str], ground_truth: Sequence[str]) -> float:
    """Recommended Songs Clicks (lower is better); penalty 51 if never hit."""
    gt = set(ground_truth)
    for i, t in enumerate(predicted):
        if t in gt:
            return i // 10
    return MAX_CLICKS


def evaluate_one(
    predicted: Sequence[str],
    ground_truth: Sequence[str],
    track_artist: Mapping[str, str] | None = None,
) -> Dict[str, float]:
    return {
        "r_precision": r_precision(predicted, ground_truth, track_artist),
        "ndcg": ndcg(predicted, ground_truth),
        "clicks": clicks(predicted, ground_truth),
    }


def aggregate(rows: List[Dict[str, float]]) -> Dict[str, float]:
    if not rows:
        return {"r_precision": 0.0, "ndcg": 0.0, "clicks": float(MAX_CLICKS)}
    n = len(rows)
    return {
        "r_precision": sum(r["r_precision"] for r in rows) / n,
        "ndcg": sum(r["ndcg"] for r in rows) / n,
        "clicks": sum(r["clicks"] for r in rows) / n,
    }
