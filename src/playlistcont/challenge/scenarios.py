"""Build the 10 official challenge scenarios from a held-out set of playlists.

The RecSys-2018 test set has 10,000 incomplete playlists, 1,000 per scenario:

  1.  title only (0 seed tracks)
  2.  title + first 1 track
  3.  title + first 5 tracks
  4.  first 5 tracks, NO title
  5.  title + first 10 tracks
  6.  first 10 tracks, NO title
  7.  title + first 25 tracks
  8.  title + 25 random tracks
  9.  title + first 100 tracks
  10. title + 100 random tracks

For each we take a full playlist, expose the seed (title and/or the seed
tracks) and hold out the remaining tracks as ground truth.  "random" scenarios
sample the seed positions at random (order is then hidden); "first" scenarios
keep the leading tracks.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import numpy as np

from ..data.schema import Dataset, Playlist

# (name, n_seed, use_title, random_seed_positions)
SCENARIO_SPECS = [
    ("title_only", 0, True, False),
    ("title_1", 1, True, False),
    ("title_5", 5, True, False),
    ("no_title_5", 5, False, False),
    ("title_10", 10, True, False),
    ("no_title_10", 10, False, False),
    ("title_25", 25, True, False),
    ("title_random_25", 25, True, True),
    ("title_100", 100, True, False),
    ("title_random_100", 100, True, True),
]

SCENARIO_NAMES = [s[0] for s in SCENARIO_SPECS]


@dataclass
class TestCase:
    pid: int
    scenario: str
    title: Optional[str]        # None if the scenario hides the title
    seed_tracks: List[str]      # visible seed (may be empty)
    holdout: List[str]          # ground truth to predict


def _min_len_for(n_seed: int) -> int:
    # need at least n_seed + a few held-out tracks to be a useful case
    return n_seed + max(3, n_seed // 5 + 1)


def build_scenarios(
    playlists: List[Playlist],
    per_scenario: int = 1000,
    seed: int = 0,
) -> List[TestCase]:
    """Turn a list of (test) playlists into scenario test cases.

    Playlists are assigned to scenarios greedily by whether they are long
    enough (the 100-track scenarios need long playlists).  Each playlist is
    used for at most one scenario so seeds/holdouts never leak across cases.
    """
    rng = np.random.default_rng(seed)
    pool = list(playlists)
    rng.shuffle(pool)
    used = set()
    cases: List[TestCase] = []

    def make_case(pl, name, n_seed, use_title, randomize) -> TestCase:
        tracks = pl.tracks
        if n_seed == 0:
            seed_tracks, holdout = [], list(tracks)
        elif randomize:
            perm = rng.permutation(len(tracks))
            seed_pos = set(perm[:n_seed].tolist())
            seed_tracks = [tracks[i] for i in sorted(seed_pos)]
            holdout = [tracks[i] for i in range(len(tracks)) if i not in seed_pos]
        else:
            seed_tracks = list(tracks[:n_seed])
            holdout = list(tracks[n_seed:])
        return TestCase(
            pid=pl.pid, scenario=name,
            title=(pl.name if use_title else None),
            seed_tracks=seed_tracks, holdout=holdout,
        )

    # Group scenarios by seed size and process the demanding (long) groups
    # first, so the scarce long playlists are shared round-robin between
    # scenarios that need the same length (e.g. title_100 vs title_random_100).
    from itertools import groupby

    groups = groupby(
        sorted(SCENARIO_SPECS, key=lambda s: -s[1]), key=lambda s: s[1]
    )
    for n_seed, specs in groups:
        specs = list(specs)
        need = _min_len_for(n_seed)
        counts = {spec[0]: 0 for spec in specs}
        turn = 0
        for pl in pool:
            if all(c >= per_scenario for c in counts.values()):
                break
            if pl.pid in used or len(pl.tracks) < need:
                continue
            # pick the next scenario in this group that still needs cases
            for _ in range(len(specs)):
                spec = specs[turn % len(specs)]
                turn += 1
                if counts[spec[0]] < per_scenario:
                    used.add(pl.pid)
                    cases.append(make_case(pl, spec[0], spec[1], spec[2], spec[3]))
                    counts[spec[0]] += 1
                    break
    return cases


def train_test_split(
    ds: Dataset, n_test: int = 3000, seed: int = 0
) -> tuple[List[Playlist], List[Playlist]]:
    """Split playlists into (train, test) by pid; test feeds build_scenarios."""
    rng = np.random.default_rng(seed)
    idx = rng.permutation(ds.n_playlists)
    test_idx = set(idx[:n_test].tolist())
    train, test = [], []
    for i, pl in enumerate(ds.playlists):
        (test if i in test_idx else train).append(pl)
    return train, test
