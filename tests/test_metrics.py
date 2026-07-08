"""Metrics tested against hand-computed values."""
import math

from playlistcont.challenge.metrics import (
    aggregate, clicks, evaluate_one, ndcg, r_precision,
)


def test_r_precision_perfect():
    gt = ["a", "b", "c", "d"]
    pred = ["a", "b", "c", "d", "x", "y"]
    assert r_precision(pred, gt) == 1.0


def test_r_precision_half_tracks():
    # R = 4, first 4 preds contain 2 ground-truth tracks -> 2/4
    gt = ["a", "b", "c", "d"]
    pred = ["a", "z", "b", "w", "c", "d"]
    assert r_precision(pred, gt) == 0.5


def test_r_precision_artist_credit():
    # R=2. First 2 preds: "a" (exact hit) and "z". "z" shares artist A2
    # with ground-truth track "b" (which we missed) -> 0.25 credit.
    gt = ["a", "b"]
    pred = ["a", "z"]
    track_artist = {"a": "A1", "b": "A2", "z": "A2"}
    # (1 track + 0.25 artist) / 2 = 0.625
    assert abs(r_precision(pred, gt, track_artist) - 0.625) < 1e-9


def test_r_precision_empty_gt():
    assert r_precision(["a"], []) == 0.0


def test_ndcg_hand():
    # gt has 2 items; predicted hits at positions 0 and 2 (0-indexed).
    gt = ["a", "b"]
    pred = ["a", "x", "b"]
    dcg = 1 / math.log2(2) + 1 / math.log2(4)  # pos0 + pos2
    idcg = 1 / math.log2(2) + 1 / math.log2(3)  # ideal: both at top
    assert abs(ndcg(pred, gt) - dcg / idcg) < 1e-9


def test_ndcg_perfect_order():
    gt = ["a", "b", "c"]
    pred = ["a", "b", "c", "d"]
    assert abs(ndcg(pred, gt) - 1.0) < 1e-9


def test_clicks_first_bucket():
    gt = ["z"]
    # first hit at index 5 -> floor(5/10) = 0
    assert clicks(["a", "b", "c", "d", "e", "z"], gt) == 0


def test_clicks_second_bucket():
    gt = ["z"]
    pred = [f"x{i}" for i in range(11)] + ["z"]  # first hit at index 11 -> 1
    assert clicks(pred, gt) == 1


def test_clicks_miss_penalty():
    assert clicks(["a", "b"], ["z"]) == 51


def test_aggregate():
    rows = [
        {"r_precision": 1.0, "ndcg": 1.0, "clicks": 0},
        {"r_precision": 0.0, "ndcg": 0.0, "clicks": 2},
    ]
    agg = aggregate(rows)
    assert agg["r_precision"] == 0.5
    assert agg["clicks"] == 1.0


def test_evaluate_one_keys():
    out = evaluate_one(["a"], ["a"], {"a": "x"})
    assert set(out) == {"r_precision", "ndcg", "clicks"}
