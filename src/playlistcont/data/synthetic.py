"""SyntheticMPD: a statistically realistic stand-in for the real MPD.

Design goals (so experiments are meaningful without the gated download):

* **Power-law track popularity** — a few hits, a long tail (like real playlists).
* **Genre-clustered co-occurrence** — tracks of the same taste archetype
  co-occur far more often than chance, so item-CF / MF / w2v have signal.
* **Titles correlated with content** — a "gym rap" playlist gets a title drawn
  from a rap/workout vocabulary, so the title model has signal.
* **Latent taste archetypes** — every track and playlist is generated from one
  of ten named archetypes (e.g. *sad slow country*, *gym rap*, *indie chill*),
  each with a profile over the interpretable axes.  This is exactly what the
  taste engine is meant to recover, so we can validate it.

Everything is seeded and fast: the default ~20k playlists / ~15k tracks build
in a few seconds and train end-to-end in minutes.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import numpy as np

from .schema import (
    AXES,
    GENRES,
    N_AXES,
    SCALAR_AXES,
    Dataset,
    Playlist,
    TrackMeta,
)


@dataclass
class Archetype:
    name: str
    # mean value for each scalar axis (tempo, energy, valence, acousticness, lyrical_depth)
    scalars: Dict[str, float]
    genre: str
    # words that show up in playlist titles of this archetype
    title_words: List[str]


ARCHETYPES: List[Archetype] = [
    Archetype("sad slow country",
              dict(tempo=.25, energy=.30, valence=.20, acousticness=.70, lyrical_depth=.82),
              "country", ["heartbreak", "whiskey", "lonely", "back roads", "cry", "country", "tears"]),
    Archetype("gym rap",
              dict(tempo=.82, energy=.92, valence=.55, acousticness=.08, lyrical_depth=.38),
              "rap", ["gym", "workout", "beast mode", "hype", "grind", "trap", "pump"]),
    Archetype("indie chill",
              dict(tempo=.45, energy=.40, valence=.50, acousticness=.62, lyrical_depth=.62),
              "indie", ["chill", "indie", "mellow", "dreamy", "vibes", "sunday", "bedroom"]),
    Archetype("party pop",
              dict(tempo=.72, energy=.86, valence=.82, acousticness=.18, lyrical_depth=.30),
              "pop", ["party", "pop", "dance", "friday", "bangers", "hits", "throwback"]),
    Archetype("classic rock",
              dict(tempo=.60, energy=.72, valence=.55, acousticness=.35, lyrical_depth=.52),
              "rock", ["rock", "classic", "roadtrip", "dad rock", "guitar", "80s", "anthem"]),
    Archetype("late night rnb",
              dict(tempo=.50, energy=.48, valence=.44, acousticness=.42, lyrical_depth=.58),
              "rnb", ["late night", "rnb", "smooth", "slow jams", "bedroom", "mood", "sensual"]),
    Archetype("edm rave",
              dict(tempo=.88, energy=.95, valence=.70, acousticness=.05, lyrical_depth=.20),
              "electronic", ["edm", "rave", "festival", "drop", "house", "electro", "bass"]),
    Archetype("coffeehouse folk",
              dict(tempo=.40, energy=.35, valence=.52, acousticness=.80, lyrical_depth=.76),
              "folk", ["folk", "coffeehouse", "acoustic", "morning", "cozy", "singer songwriter"]),
    Archetype("metal rage",
              dict(tempo=.80, energy=.95, valence=.35, acousticness=.08, lyrical_depth=.46),
              "metal", ["metal", "rage", "mosh", "heavy", "brutal", "headbang", "thrash"]),
    Archetype("smooth jazz",
              dict(tempo=.45, energy=.40, valence=.56, acousticness=.72, lyrical_depth=.50),
              "jazz", ["jazz", "smooth", "dinner", "lounge", "cafe", "sax", "bossa"]),
]


class SyntheticMPD:
    """Generate a :class:`Dataset` that mimics MPD statistics.

    Parameters
    ----------
    n_playlists, n_tracks : corpus scale.
    seed : RNG seed (fully deterministic output).
    archetype_purity : fraction of a playlist's tracks drawn from its own
        archetype pool (the rest come from global popularity / a secondary
        archetype), controlling how genre-clustered co-occurrence is.
    """

    def __init__(
        self,
        n_playlists: int = 20000,
        n_tracks: int = 15000,
        seed: int = 0,
        archetype_purity: float = 0.8,
        pop_exponent: float = 1.3,
    ):
        self.n_playlists = n_playlists
        self.n_tracks = n_tracks
        self.seed = seed
        self.archetype_purity = archetype_purity
        self.pop_exponent = pop_exponent
        self.rng = np.random.default_rng(seed)

    # ------------------------------------------------------------------
    def _build_tracks(self):
        rng = self.rng
        n = self.n_tracks
        # Assign each track a home archetype (roughly balanced).
        home = rng.integers(0, len(ARCHETYPES), size=n)
        # Power-law base popularity via a Zipf-like weight over tracks.
        ranks = np.arange(1, n + 1)
        rng.shuffle(ranks)
        base_pop = 1.0 / np.power(ranks, self.pop_exponent)
        base_pop /= base_pop.sum()

        genre_idx = {g: i for i, g in enumerate(GENRES)}
        features = np.zeros((n, N_AXES), dtype=np.float32)
        metas: List[TrackMeta] = []
        # a modest set of artists per archetype so artist-overlap has meaning
        artists_per_arch = max(20, n // (len(ARCHETYPES) * 12))
        for i in range(n):
            a = ARCHETYPES[home[i]]
            # scalar axes: archetype mean + noise, clipped to [0,1]
            for j, ax in enumerate(SCALAR_AXES):
                features[i, j] = np.clip(a.scalars[ax] + rng.normal(0, 0.09), 0, 1)
            # genre block: mostly the home genre with a little bleed
            g0 = genre_idx[a.genre]
            features[i, len(SCALAR_AXES) + g0] = np.clip(0.85 + rng.normal(0, 0.08), 0, 1)
            # small bleed into a random neighbour genre
            gb = rng.integers(0, len(GENRES))
            features[i, len(SCALAR_AXES) + gb] += 0.15 * rng.random()

            artist_n = int(home[i]) * artists_per_arch + rng.integers(0, artists_per_arch)
            metas.append(
                TrackMeta(
                    track_uri=f"spotify:track:s{i:06d}",
                    track_name=f"{a.name.title()} Song {i}",
                    artist_uri=f"spotify:artist:a{artist_n:05d}",
                    artist_name=f"{a.genre.title()} Artist {artist_n}",
                    album_uri=f"spotify:album:al{i % (n // 3 + 1):05d}",
                    album_name=f"Album {i % (n // 3 + 1)}",
                )
            )
        self._home = home
        self._base_pop = base_pop
        self._features = features
        self._metas = metas
        # Precompute per-archetype pools + normalized popularity within pool.
        self._pools = []
        for k in range(len(ARCHETYPES)):
            idx = np.where(home == k)[0]
            w = base_pop[idx]
            w = w / w.sum() if w.sum() > 0 else np.ones(len(idx)) / len(idx)
            self._pools.append((idx, w))

    def _playlist_length(self) -> int:
        # lognormal-ish lengths, median ~30, tail long enough for the
        # 100-track scenarios; clipped to [8, 250].
        L = int(np.clip(self.rng.lognormal(mean=3.35, sigma=0.7), 8, 250))
        return L

    def _make_title(self, arch: Archetype) -> str:
        rng = self.rng
        words = rng.choice(arch.title_words, size=rng.integers(1, 3), replace=False)
        title = " ".join(words)
        if rng.random() < 0.3:
            title += " " + rng.choice(["mix", "playlist", "songs", "2018", "vol 2"])
        # small chance of an empty/uninformative title (like real MPD)
        if rng.random() < 0.05:
            title = ""
        return title.strip()

    def _build_playlists(self):
        rng = self.rng
        playlists: List[Playlist] = []
        all_idx = np.arange(self.n_tracks)
        for pid in range(self.n_playlists):
            primary = rng.integers(0, len(ARCHETYPES))
            arch = ARCHETYPES[primary]
            secondary = rng.integers(0, len(ARCHETYPES))
            L = self._playlist_length()
            n_primary = int(round(L * self.archetype_purity))
            n_secondary = int(round(L * (1 - self.archetype_purity) * 0.6))
            n_global = L - n_primary - n_secondary

            picks: List[int] = []
            for pool_k, count in ((primary, n_primary), (secondary, n_secondary)):
                if count <= 0:
                    continue
                idx, w = self._pools[pool_k]
                take = min(count, len(idx))
                chosen = rng.choice(idx, size=take, replace=False, p=w)
                picks.extend(chosen.tolist())
            if n_global > 0:
                chosen = rng.choice(all_idx, size=n_global, replace=False, p=self._base_pop)
                picks.extend(chosen.tolist())

            # dedupe, preserve order (playlists are ordered in MPD)
            seen = set()
            ordered = []
            for t in picks:
                if t not in seen:
                    seen.add(t)
                    ordered.append(t)
            uris = [self._metas[t].track_uri for t in ordered]
            playlists.append(Playlist(pid=pid, name=self._make_title(arch), tracks=uris))
        self._playlists = playlists

    # ------------------------------------------------------------------
    def generate(self) -> Dataset:
        self._build_tracks()
        self._build_playlists()
        tracks = {m.track_uri: m for m in self._metas}
        features = {
            self._metas[i].track_uri: self._features[i] for i in range(self.n_tracks)
        }
        return Dataset(playlists=self._playlists, tracks=tracks, features=features)


def make_synthetic(n_playlists=20000, n_tracks=15000, seed=0, **kw) -> Dataset:
    """Convenience wrapper."""
    return SyntheticMPD(n_playlists=n_playlists, n_tracks=n_tracks, seed=seed, **kw).generate()
