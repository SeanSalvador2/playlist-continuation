# Playlist Continuation — Spotify Million Playlist Dataset (RecSys 2018)

A from-scratch playlist-continuation project built around the **Spotify Million
Playlist Dataset Challenge** (ACM RecSys Challenge 2018). It implements a full,
tested challenge harness and a catalogue of seven recommenders — from a
popularity baseline up to a two-stage learned ensemble — and compares them
across all ten official test scenarios. The signature piece is an
**interpretable Taste Engine** that recommends by explicit, human-readable taste
axes and explains every recommendation.

Because the real MPD is gated behind AIcrowd registration, everything runs
end-to-end on a built-in **synthetic MPD generator** that reproduces the
statistical structure of real playlists; a single flag switches to the real
data once you've downloaded it.

```
python experiments/run_comparison.py     # trains all 7 models, evaluates 10 scenarios, ~40s
pytest                                    # 34 tests
```

---

## The challenge

Given an incomplete playlist (its title and/or some of its tracks), recommend up
to **500** tracks that fit. The test set has **10 scenarios × 1,000 playlists**:

| # | Scenario | Seed given |
|---|----------|------------|
| 1 | title only | title, 0 tracks |
| 2 | title + first 1 | title + 1 track |
| 3 | title + first 5 | title + 5 tracks |
| 4 | first 5, no title | 5 tracks |
| 5 | title + first 10 | title + 10 tracks |
| 6 | first 10, no title | 10 tracks |
| 7 | title + first 25 | title + 25 tracks |
| 8 | title + random 25 | title + 25 random tracks |
| 9 | title + first 100 | title + 100 tracks |
| 10 | title + random 100 | title + 100 random tracks |

**Metrics** (all implemented in `challenge/metrics.py`, unit-tested against
hand-computed values):

- **R-precision** — track matches in the top-R predictions plus **0.25 credit**
  for matching an artist the ground truth contains (R = number of held-out
  tracks). Higher is better.
- **NDCG** — normalized discounted cumulative gain over the full list. Higher is
  better.
- **Recommended Songs Clicks** — `floor(rank_of_first_hit / 10)`; the number of
  "refresh" clicks (10 songs each) before the first correct track. Penalty 51 if
  no hit in 500. Lower is better.

**Submission format** (`challenge/submission.py` writes and validates it):

```
team_info,my cool team name,contact@email.com

pid, track_uri_1, track_uri_2, ... (≤500, no dupes, no seed tracks)
...
```

---

## Data strategy: synthetic ↔ real MPD

The real MPD ships as 1,000 JSON slice files of 1,000 playlists each. We support
both real and synthetic data behind one `Dataset` type, so a single flag swaps
them everywhere.

**Real MPD** (`data/loader.py`): a schema-faithful loader for the official
`mpd.slice.*.json` files. Download the dataset from AIcrowd, then:

```python
from playlistcont.data.loader import load_mpd
ds = load_mpd("/path/to/spotify_mpd/data", max_slices=20)
```

**Synthetic MPD** (`data/synthetic.py`): a generator that mimics real MPD
statistics so experiments are meaningful without the download:

- **power-law track popularity** (a few hits, a long tail);
- **genre-clustered co-occurrence** — tracks from the same latent *taste
  archetype* co-occur far above chance, giving CF/MF/w2v real signal;
- **titles correlated with content** — a "gym rap" playlist gets a title drawn
  from a rap/workout vocabulary, giving the title model real signal;
- **ten latent taste archetypes** (e.g. *sad slow country*, *gym rap*, *indie
  chill*, *edm rave*, *coffeehouse folk*), each with a profile over the
  interpretable feature axes — exactly what the Taste Engine is meant to recover.

```python
from playlistcont.data.synthetic import make_synthetic
ds = make_synthetic(n_playlists=20000, n_tracks=15000, seed=0)  # default scale
```

Only the synthetic generator produces the interpretable per-track feature axes
(tempo, energy, valence, acousticness, lyrical-depth, genre vector). The real
MPD carries no audio features, so for real data you attach them via
`loader.attach_features(ds, "audio_features.csv")` — a CSV of per-track audio
features / genres exported from the Spotify audio-features API or an offline
genre model. See the [Taste Engine on real data](#taste-engine-on-real-data)
note below.

---

## Model catalogue

All models share one interface — `fit(dataset)` and
`recommend(seed_tracks, title, k=500)` — so they are directly swappable and
comparable (`models/base.py`).

| Model | File | Intuition |
|-------|------|-----------|
| **Popularity** | `popularity.py` | Always recommend the globally most-played tracks. The floor every other model must beat. |
| **Item-item CF** | `itemcf.py` | Tracks that co-occur in many playlists are similar. Builds a sparse track–track similarity (cosine or PMI) and sums the seed tracks' neighbour columns. Fast, strong, and *unbeatable on large random seeds*. |
| **ALS (matrix factorization)** | `mf.py` | Learn low-rank playlist/track factors from implicit feedback; new playlists are folded in as the mean of their seed item-vectors. Uses the `implicit` library, with a self-contained numpy ALS fallback. The most consistent single model. |
| **Track2Vec** | `track2vec.py` | word2vec over "playlists as sentences, tracks as words". Recommend the nearest tracks to the seed centroid. Captures sequential/embedding structure but needs a lot of data to shine. |
| **Title model** | `title_model.py` | Char n-gram TF-IDF over playlist *titles* → nearest playlists → pool their tracks. The **only** model with signal in the title-only scenario. |
| **Hybrid ensemble** | `hybrid.py` | Two-stage: union candidates from all models above, then rerank with a **LightGBM** classifier (logistic-regression fallback) on per-candidate features — CF score, ALS score, w2v similarity, title score, popularity, artist overlap. Trained on held-out tracks from the training playlists. Mirrors the winning 2018 recipe and wins overall. |
| **Taste Engine** ⭐ | `taste_engine.py` | Interpretable, explainable recommender — see below. |

---

## ⭐ The Taste Engine (signature model)

Every other model is a black box: it returns tracks but can't tell you *why*.
The Taste Engine reasons in **interpretable axes** and explains itself.

**1. Interpretable track features.** Each track is a point in axis space:
`tempo` (slow↔fast), `energy` (calm↔intense), `valence` (sad↔happy),
`acousticness` (electric↔acoustic), `lyrical_depth` (filler↔meaningful), plus a
soft **genre vector**. (Synthetic data generates these; real data supplies them
via an audio-features CSV.)

**2. Two sources of preference.**

- **STATED** — the user says what they care about, in words or a dict:
  `"melody and meaning matter most, I love slow sad country"`. A lexicon
  (`parse_stated`) turns that into signed axis emphases (lyrical_depth ↑, valence
  ↓, tempo ↓, country ↑).
- **LEARNED** — a logistic regression of *"is this track in your playlists?"*
  onto the standardized axes, blended with the user's centroid direction for
  stability. The coefficients **are** the preference vector: negative valence =
  you lean sad, positive lyrical_depth = you value meaning.

They are combined with a **`trust`** parameter (how much to believe what the user
*said* over what they *did*).

**3. Named flavor clusters.** k-means on the user's tracks in raw axis space,
each centroid turned into English by a template — e.g.
`"slow, sad, acoustic country with deep lyrics"`.

**4. Scoring + explanations.** Candidates are scored as *weighted axis-match +
co-occurrence*, and every recommendation carries a human explanation.

### Example (real output from `make_synthetic`, a "sad slow country" listener)

```
FLAVOR CLUSTERS:
  - slow, sad, acoustic country with deep lyrics  (share 38%)
  - slow, sad country with deep lyrics            (share 25%)

LEARNED preference weights (top axes):
  genre:country      +0.60
  lyrical_depth      +0.52
  valence            -0.42     ← leans sad
  tempo              -0.29     ← leans slow

TOP RECOMMENDATIONS WITH EXPLANATIONS (stated+learned blend):
  spotify:track:s000093
     recommended because: matches your 'slow, sad, acoustic country with deep
     lyrics' flavor (valence 0.11, tempo 0.19); co-occurs with 8 of your seed
     tracks; same artist as one of your seeds
  spotify:track:s001513
     recommended because: matches your 'slow, sad country with deep lyrics'
     flavor (valence 0.13, tempo 0.32); co-occurs with 8 of your seed tracks
```

The Taste Engine is a **real, trained model in the comparison**, not a mock. It
optimizes for *explainable taste match* rather than raw recall, so it sits in the
CF pack on R-precision (0.30) while being the only model that can justify itself
and accept explicit user control. Its unit tests verify that a user built from
sad/slow tracks actually gets negative valence/tempo and positive
lyrical-depth/country learned weights.

### Taste Engine on real data

The MPD has no audio features, so the axis mapping for real data is:

| Axis | Real-MPD source |
|------|-----------------|
| tempo, energy, valence, acousticness | Spotify audio-features API (`tempo`→normalized, `energy`, `valence`, `acousticness`) |
| lyrical_depth | proxy from a lyrics model / genre prior (documented, user-supplied) |
| genre vector | artist genres from the Spotify artist endpoint, one-hot/soft-encoded |

Provide these as a CSV and call `loader.attach_features(ds, csv)`; the engine
then runs identically to the synthetic case. Without features it is skipped and
the other six models still run.

---

## Results summary

Full analysis in [`RESULTS.md`](RESULTS.md); tables in
[`results/`](results/); charts in [`results/figures/`](results/figures/). Overall
(micro-averaged over all scenario cases, synthetic MPD):

| model | R-precision | NDCG | Clicks (↓) |
|-------|-------------|------|------------|
| **hybrid** | **0.322** | **0.608** | **0.126** |
| als | 0.311 | 0.568 | 0.433 |
| item_cf | 0.309 | 0.562 | 0.451 |
| title | 0.308 | 0.577 | 0.222 |
| taste_engine | 0.300 | 0.514 | 1.348 |
| popularity | 0.168 | 0.435 | 0.206 |
| track2vec | 0.120 | 0.313 | 6.246 |

Headline findings: the **hybrid ensemble wins overall**; the **title model is
indispensable for the title-only cold start** (0.474 vs a 0.193 floor); simple
**item-CF is unbeatable on large random seeds**; and the **Taste Engine trades a
little recall for full interpretability**. (These are synthetic-data numbers —
good for relative comparison, not comparable to the real leaderboard.)

### Deep-dive experiments and ablations

For the full research study — per-model hyperparameter sweeps, the taste-engine
ablations (stated-vs-learned trust curves, an adversarial "wrong preferences"
rescue, axis and flavor-count ablations), ensemble leave-one-out, a
scenario-aware **routed hybrid** that fixes the title-only failure (+0.024
overall), and robustness (3-seed error bars, data-scale, popularity-bias /
coverage / diversity) — see **[`ABLATIONS.md`](ABLATIONS.md)**. All of it is
reproducible via `python experiments/run_all.py` (~30 min); scripts in
[`experiments/`](experiments/), outputs in
[`results/ablations/`](results/ablations/).

Selected results: a **routed hybrid** tops the suite at **0.379** overall
R-precision (vs 0.357 for the plain hybrid); **track2vec gets worse with more
epochs** (undertraining was the wrong diagnosis); the taste engine's accuracy is
**collaborative filtering, not its axes** (pure axis-match scores only 0.122);
and **ALS benefits most from denser data**.

---

## Running on the real MPD

1. Register and download from
   [AIcrowd](https://www.aicrowd.com/challenges/spotify-million-playlist-dataset-challenge).
2. Point the loader / experiment at the slice directory:
   ```bash
   python experiments/run_comparison.py --real /path/to/spotify_mpd/data
   ```
3. (Optional) attach audio features to enable the Taste Engine, then write a
   submission:
   ```python
   from playlistcont.challenge.submission import write_submission
   preds = {case.pid: model.recommend(case.seed_tracks, case.title, 500) for case in cases}
   write_submission("submission.csv", preds, "my team", "me@example.com")
   ```

---

## Project layout

```
playlist-continuation/
├── pyproject.toml
├── README.md · RESULTS.md
├── src/playlistcont/
│   ├── data/        schema.py · loader.py (real MPD) · synthetic.py
│   ├── challenge/   metrics.py · scenarios.py · submission.py
│   └── models/      base.py · popularity · itemcf · mf · track2vec · title_model · hybrid · taste_engine
├── experiments/run_comparison.py
├── results/         results.csv · results.md · figures/*.png
└── tests/           metrics · loader · submission · recommenders · taste_engine (+ fixture)
```

## Install

```bash
pip install -e .            # core
pip install -e ".[fast]"    # + implicit (ALS) and lightgbm (reranker); both have fallbacks
pip install -e ".[dev]"     # + pytest
```

---

## References

- Chen et al., *"Recsys Challenge 2018: Automatic Music Playlist Continuation"*, RecSys 2018. https://dl.acm.org/doi/10.1145/3240323.3240342
- Zamani, Schedl, Lamere, Chen, *"An Analysis of Approaches Taken in the ACM RecSys Challenge 2018 for Automatic Music Playlist Continuation"*, 2019. https://arxiv.org/abs/1810.01520
- AIcrowd challenge page. https://www.aicrowd.com/challenges/spotify-million-playlist-dataset-challenge
- Volkovs et al. (1st place, "Two-stage Model for Automatic Playlist Continuation") and hojinYang (2nd place) — candidate generation + gradient-boosted rerank ensembles that inspired the hybrid model.
- Hu, Koren & Volinsky, *"Collaborative Filtering for Implicit Feedback Datasets"*, 2008 (the ALS used in `mf.py`).
```
