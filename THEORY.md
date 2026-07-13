# THEORY — recommender systems, from scratch, on this repo's data

*A math-for-humans companion to the code.* The goal is intuition first: every
idea is introduced in plain words and a tiny worked example you can check on
paper, with the heavier algebra pushed to the appendices. Everything is tied to
the actual models in [`src/playlistcont/models/`](src/playlistcont/models/) and
to numbers we actually measured — where a number appears, it cites the committed
file it comes from. For the full formal treatment (all the update equations in
one place) see **[`REPORT.md`](REPORT.md) §3.4**; this document is the gentle
on-ramp to it.

**Contents**

1. [The recommendation problem, on our data](#1-the-recommendation-problem-on-our-data)
2. [Neighbourhood collaborative filtering](#2-neighbourhood-collaborative-filtering)
3. [Matrix factorization (ALS)](#3-matrix-factorization-als)
4. [Embeddings (track2vec / item2vec)](#4-embeddings-track2vec--item2vec)
5. [Content-based recommendation](#5-content-based-recommendation)
6. [Hybrids](#6-hybrids)
7. [Evaluation — and why offline metrics mislead](#7-evaluation--and-why-offline-metrics-mislead)
8. [What actually mattered here](#8-what-actually-mattered-here)
- [Appendix A: item-CF normalizations, derived](#appendix-a-item-cf-normalizations-derived)
- [Appendix B: the ALS update, derived](#appendix-b-the-als-update-derived)
- [Appendix C: skip-gram in one page](#appendix-c-skip-gram-in-one-page)

---

## 1. The recommendation problem, on our data

Forget stars and thumbs for a moment. Our data is **playlists**. A playlist is an
ordered bag of tracks with a title. The whole corpus is a big table: rows are
playlists, columns are tracks, and a cell is 1 if that playlist contains that
track and 0 otherwise. Call that matrix $A$. It is **binary** and **implicit** —
there are no ratings, only "this track was put in this playlist." Absence is not
dislike; it usually just means the curator never saw the track.

```
            t1  t2  t3  t4
   P1        1   1   0   0
   P2        1   1   1   0
   P3        1   0   1   0
   P4        0   1   0   1
```

The task ("automatic playlist continuation") is: given an **incomplete** playlist
— its title and/or a few of its tracks, called the **seed** — predict the tracks
that were held out. In this repo the seed is exactly what
[`challenge/scenarios.py`](src/playlistcont/challenge/scenarios.py) exposes: the
ten official RecSys-2018 scenarios range from *title only, zero seed tracks* to
*title + 100 seed tracks*. Everything a model gets to reason from is the seed; the
held-out tracks are the answer key.

Two structural facts about this matrix drive every design decision below:

- **It is enormous and almost entirely zero (sparse).** The real MPD is
  1,000,000 playlists × 2,262,292 tracks (REPORT.md §4.10). You never form $A$
  densely; you exploit its sparsity.
- **Popularity is wildly skewed (a power law).** A few hits are in huge numbers
  of playlists; the vast majority of tracks are in a handful. Our synthetic
  generator reproduces this on purpose
  ([`data/synthetic.py`](src/playlistcont/data/synthetic.py)). This skew is the
  villain of half the stories in this document: the easy way to look accurate is
  to keep recommending the hits, and that is exactly what a good system must
  learn *not* to do.

Every model in the catalogue is one answer to the same question — *given the seed,
score every track, return the top 500* — and they differ only in **where the
score comes from**: who-listens-with-whom (collaborative filtering, §2–4), or
what-the-track-is-like (content, §5), or a blend (§6).

---

## 2. Neighbourhood collaborative filtering

**The whole idea in one sentence:** recommend tracks that keep company with the
tracks you already have. "Keeps company with" = co-occurs in many playlists. This
is *collaborative* filtering because the signal comes entirely from the crowd's
behaviour, never from any property of the track itself.

There are two symmetric ways to slice it.

### 2.1 User-based (here: playlist-based)

Find playlists similar to the seed, then recommend what *they* contain. Similarity
between two playlists is how much their track sets overlap, normalized so long
playlists don't dominate — **cosine similarity** on the binary rows:

$$
\text{sim}(P, Q) \;=\; \frac{|P \cap Q|}{\sqrt{|P|\,|Q|}}.
$$

**Worked example (by hand, on the matrix above).** Our seed is the one-track
playlist $Q=\{t_1\}$. Cosines to each training playlist:

| neighbour | overlap with $Q$ | $\sqrt{|Q||P|}$ | cosine |
|---|---|---|---|
| $P_1=\{t_1,t_2\}$ | 1 | $\sqrt{1\cdot2}$ | $1/\sqrt2 = 0.707$ |
| $P_2=\{t_1,t_2,t_3\}$ | 1 | $\sqrt{1\cdot3}$ | $1/\sqrt3 = 0.577$ |
| $P_3=\{t_1,t_3\}$ | 1 | $\sqrt{1\cdot2}$ | $1/\sqrt2 = 0.707$ |
| $P_4=\{t_2,t_4\}$ | 0 | — | $0$ |

Now pool the neighbours' tracks (excluding the seed $t_1$), each vote weighted by
the neighbour's similarity:

$$
s(t_2) = \underbrace{0.707}_{P_1} + \underbrace{0.577}_{P_2} = 1.284, \qquad
s(t_3) = \underbrace{0.577}_{P_2} + \underbrace{0.707}_{P_3} = 1.284.
$$

They tie. That is honest and instructive: with this little data the playlist
neighbourhood cannot separate $t_2$ from $t_3$. Item-based CF, next, *can* — which
is one reason this project uses it.

### 2.2 Item-based (the repo's `item_cf`)

Instead of playlist-to-playlist, precompute **track-to-track** similarity once, up
front, and at query time just add up the seed tracks' columns. This is
[`models/itemcf.py`](src/playlistcont/models/itemcf.py), and it is the workhorse
of the whole project.

Start from the **co-occurrence matrix** $C = A^\top A$: the entry $c_{ij}$ counts
playlists containing *both* $t_i$ and $t_j$, and the diagonal $c_i = c_{ii}$ is
the track's popularity. From the matrix in §1:

$$
c_1=3,\; c_2=3,\; c_3=2,\; c_4=1;\qquad
c_{12}=2,\; c_{13}=2,\; c_{14}=0,\; c_{23}=1,\; c_{24}=1,\; c_{34}=0.
$$

Raw co-occurrence is a trap: popular tracks co-occur with everything, so $c_{ij}$
mostly measures popularity, not affinity. We fix that by normalizing. Two
normalizations, both in the code:

**Cosine** divides out the geometric mean of the two popularities:
$\text{sim}^{\cos}_{ij} = c_{ij}/\sqrt{c_i c_j}$. By hand, the neighbours of $t_1$:

$$
\text{sim}^{\cos}_{12} = \frac{2}{\sqrt{3\cdot3}} = \frac{2}{3} = 0.667, \qquad
\text{sim}^{\cos}_{13} = \frac{2}{\sqrt{3\cdot2}} = \frac{2}{\sqrt6} = 0.816, \qquad
\text{sim}^{\cos}_{14} = 0.
$$

Seed $\{t_1\}$ therefore ranks **$t_3 \;(0.816) > t_2 \;(0.667)$** — even though
$t_1$ co-occurs *equally* (twice) with each! Cosine breaks the tie in $t_3$'s
favour because $t_3$ is rarer ($c_3=2 < c_2=3$): co-occurring with a rare track is
more surprising, hence stronger evidence. That is the tie the playlist
neighbourhood couldn't break.

**PMI** (positive pointwise mutual information) is even more aggressive about
discounting popularity — it asks how much *more* than chance two tracks co-occur:

$$
\text{sim}^{\text{pmi}}_{ij} = \max\!\Big(\log \frac{c_{ij}\,N}{c_i\,c_j},\,0\Big),
\qquad N=\text{number of playlists}=4.
$$

$$
\text{pmi}_{13} = \log\frac{2\cdot4}{3\cdot2} = \log\frac{8}{6} = 0.288, \qquad
\text{pmi}_{12} = \log\frac{2\cdot4}{3\cdot3} = \log\frac{8}{9} = -0.118 \to 0.
$$

PMI **zeroes out** the $t_1$–$t_2$ pairing entirely: two popular tracks co-occurring
twice is no more than you'd expect by chance, so it carries no information. Only
the surprising $t_1$–$t_3$ link survives. (Derivations for both in
[Appendix A](#appendix-a-item-cf-normalizations-derived).)

**Scoring a real seed.** For a seed set $S$, item-CF scores every candidate track
$t$ by $\;s(t) = \sum_{s\in S}\text{sim}_{st}\;$ — sum the seed columns, take the
top-$k$. The similarity matrix is truncated to each track's top few hundred
neighbours so it stays sparse and fast. That is the entire model.

**Why it is the workhorse.** On both synthetic and real data, item-CF is the
strongest *single* signal, and on large random seeds it is essentially unbeatable
(item-CF wins `title_random_100` on real data at 0.246, REPORT.md §4.10). With a
big random sample of a playlist's tracks, "what co-occurs with these" is nearly a
sufficient statistic for "what else belongs" — and no amount of extra machinery
improves on it (a theme we return to in §6 and §8).

---

## 3. Matrix factorization (ALS)

Neighbourhood CF never *compresses* anything — it stores the whole co-occurrence
structure. Matrix factorization instead **learns a small number of hidden
dimensions** ("latent factors") that explain the data, and represents every
playlist and every track as a short vector in that shared space. This is
[`models/mf.py`](src/playlistcont/models/mf.py).

### 3.1 The picture

Approximate the giant sparse matrix $A$ ($P \times T$) as the product of two thin
matrices:

$$
A \;\approx\; U\,V^\top, \qquad U \in \mathbb{R}^{P\times f}, \; V\in\mathbb{R}^{T\times f},
$$

with $f$ small (say 64). Row $u_p$ is playlist $p$'s taste as a point in
$f$-dimensional space; row $v_t$ is track $t$'s position in the *same* space. The
predicted affinity of playlist $p$ for track $t$ is their dot product
$u_p^\top v_t$: large when they point the same way.

**What is a latent factor?** Nobody labels them, but they line up with real
structure. Imagine $f=2$ where factor 1 turned out to mean "country-ness" and
factor 2 "rap-ness". Then a country track has $v \approx (0.9, 0.1)$, a rap track
$v \approx (0.1, 0.9)$, and a country-leaning playlist has $u \approx (0.8, 0.2)$.

**Toy 2-factor example (checkable by hand).** Take four tracks and the factor
matrix

$$
V = \begin{pmatrix} 0.9 & 0.1 \\ 0.8 & 0.2 \\ 0.1 & 0.9 \\ 0.2 & 0.8 \end{pmatrix}
\;\;(t_1,t_2\text{ country};\; t_3,t_4\text{ rap}), \qquad
u = (0.8,\,0.2)\;\;(\text{a country playlist}).
$$

Scores $V u$:

$$
s(t_1)=0.9\cdot0.8+0.1\cdot0.2=0.74,\quad
s(t_2)=0.66,\quad
s(t_3)=0.1\cdot0.8+0.9\cdot0.2=0.26,\quad
s(t_4)=0.32.
$$

The country tracks score $\sim0.7$, the rap tracks $\sim0.3$: the factorization
*generalizes* — it will rank a country track this playlist has never seen above a
rap track, because it placed them by the company they keep across the whole
corpus. That generalization is why ALS is the most consistent single model in the
study, and why it benefits most from more data (+0.014 R-precision going from 2k
to 20k playlists, REPORT.md §4.8): more rows sharpen the factors.

### 3.2 Why implicit feedback needs *confidence weighting*

Here is the subtlety that trips people up. Our $A$ is all 1s and 0s, and the 0s
are not "dislike" — they are "don't know." If you fit a plain least-squares
factorization to $A$, you spend all your effort forcing the millions of 0s to
zero, and the model learns "recommend nothing."

Hu, Koren & Volinsky (2008) fix this by splitting each entry into a **preference**
$r_{pt}\in\{0,1\}$ (did the track appear?) and a **confidence** $c_{pt}$ (how sure
are we?). We are very sure about the 1s and barely sure about the 0s:

$$
c_{pt} = 1 + \alpha\, A_{pt} \quad(\alpha=40 \text{ in the code}).
$$

The objective weights every squared error by that confidence:

$$
\min_{U,V}\;\sum_{p,t} c_{pt}\big(r_{pt} - u_p^\top v_t\big)^2
\;+\; \lambda\Big(\textstyle\sum_p\|u_p\|^2 + \sum_t\|v_t\|^2\Big).
$$

A missing track ($c=1$) contributes 40× less than an observed one ($c=41$), so the
model chases the 1s and treats the 0s as soft background. This one idea is the
whole reason MF works for playlists.

### 3.3 Solving it: alternating least squares

The objective is not convex in $U$ and $V$ jointly, but **fix one and it's a
convex least-squares problem in the other** — with a closed-form solution. So we
alternate: solve all track factors holding playlists fixed, then all playlist
factors holding tracks fixed, repeat. Each half-step has the exact update

$$
u_p = \big(V^\top C^p V + \lambda I\big)^{-1} V^\top C^p r_p,
$$

where $C^p$ is the diagonal confidence matrix for playlist $p$. (Full derivation
in [Appendix B](#appendix-b-the-als-update-derived); the exact loop is
`_SimpleALS._als_step` in [`mf.py`](src/playlistcont/models/mf.py).)

**Cold start / fold-in.** A brand-new seed playlist has no learned $u_p$. We
approximate it as the mean of its seed tracks' factors, $u = \frac1{|S|}\sum_{t\in
S} v_t$, and score by $Vu$. Zero seeds ⇒ nothing to fold in ⇒ fall back to
popularity.

---

## 4. Embeddings (track2vec / item2vec)

Word2vec learns word meanings from the company words keep in sentences. Swap
"sentence → playlist" and "word → track" and you get **track2vec** /
**item2vec** (Barkan & Koenigstein 2016): learn a dense vector per track from the
company it keeps in playlists. This is
[`models/track2vec.py`](src/playlistcont/models/track2vec.py).

### 4.1 The skip-gram objective, in words

Slide a window along each playlist. For every track $t$, its *context* is the
tracks near it in the same playlist. Train each track's vector so that it
**predicts its context** and does *not* predict random tracks ("negative
samples"). Formally we maximize

$$
\sum_{t}\;\sum_{t'\in\text{ctx}(t)} \Big[\log\sigma(v_{t'}^\top v_t)
\;+\; \sum_{m=1}^{M}\mathbb{E}_{t_m\sim P_n}\,\log\sigma(-v_{t_m}^\top v_t)\Big],
$$

where $\sigma$ is the logistic function. The first term pulls co-occurring tracks
together; the second pushes random pairs apart. (One-page unpacking in
[Appendix C](#appendix-c-skip-gram-in-one-page).)

### 4.2 Relation to matrix factorization

This is not a different universe from §3. Levy & Goldberg (2014) showed skip-gram
with negative sampling is *implicitly factorizing a shifted PMI matrix* of the
co-occurrence counts. So track2vec, ALS, and even the PMI item-CF of §2 are three
lenses on the same object: the co-occurrence structure of $A$. They differ in how
they compress and retrieve it.

### 4.3 The honest negative result

To recommend, track2vec averages the seed vectors into a centroid and returns the
nearest tracks by cosine. In this project **that retrieval step is the weak
link.** The natural fix — "train longer" — makes it *worse*: R-precision falls
monotonically from 0.182 at 5 epochs to 0.099 at 40 (REPORT.md §4.5). More
training over-specializes each vector to exact *local* co-occurrence, and the
mean-pooled centroid then blurs across a playlist's several sub-tastes. The one
knob that helps is a *wider context window* — i.e. pushing the model toward
whole-playlist co-occurrence, which is exactly what item-CF and ALS already
capture directly. A clean lesson that a fancier model class is not automatically
better.

---

## 5. Content-based recommendation

Everything so far is collaborative: it knows *nothing* about the tracks
themselves, only who listened together. A **content-based** recommender is the
opposite — it reads a description of each track and recommends tracks that are
*like* the ones you have, using no behavioural data at all. This is the new
[`models/content_knn.py`](src/playlistcont/models/content_knn.py).

### 5.1 Feature vectors and cosine scoring

In this repo every track carries an interpretable **feature vector** $x_t \in
\mathbb{R}^{15}$: five mood axes (tempo, energy, valence, acousticness,
lyrical-depth) plus a ten-dimensional soft genre one-hot (see
[`data/schema.py`](src/playlistcont/data/schema.py)). Content-based scoring is
then almost embarrassingly simple. Form the **seed centroid** — the mean feature
vector of the seed tracks — and score every candidate by cosine similarity to it:

$$
c = \frac1{|S|}\sum_{t\in S} x_t, \qquad
\text{score}(t) = \cos(x_t, c) = \frac{x_t^\top c}{\|x_t\|\,\|c\|}.
$$

No popularity, no co-occurrence — only "does this track point the same way as the
average of your seeds." That purity forces two design decisions the code makes
explicit and documents:

- **Seedless playlists** (the `title_only` scenario, zero seed tracks): a pure
  content model has nothing to point at. `content_knn` falls back to the **global
  feature centroid** (rank by how prototypical a track is of the whole
  catalogue). This carries *no title signal* — content reads features, not words —
  so it is a deliberately weak guess, and it shows up as such in the numbers below.
- **Featureless tracks** (an all-zero vector): their cosine is undefined, so they
  get score $-\infty$ and are **never recommended**. A track must have features to
  be a content candidate — which is also content-based recommendation's superpower
  (next).

### 5.2 This repo's honest finding

Content-based recommendation is where the project is most candid about a
recommender that *loses on the headline metric and is worth keeping anyway.*

**On accuracy, pure content is weak — below the popularity floor.** `content_knn`
scores **0.078** overall R-precision
([`results/results_content_knn.csv`](results/results_content_knn.csv), OVERALL
row), beneath both the popularity floor of **0.162** (REPORT.md §4.1) and the
axis-only Taste-Engine variant at **0.122** (ABLATIONS.md §2e, `cf_weight=0`). It
is at its worst exactly where it has no seed to read — **0.020** on `title_only`
(same file) versus the 0.193 popularity floor there — and at its best on a large
random seed, **0.183** on `title_random_100`, where the centroid finally has
enough tracks to pin down a direction. The lesson: **feature similarity alone is a
weak ranking signal**, because it is blind to what people actually put together
and blind to popularity.

That same axis-only humbling is why the *Taste Engine's* accuracy turned out to be
its co-occurrence term, not its interpretable axes: dropping any single axis moves
R-precision by at most ±0.003 (REPORT.md §4.4). Content features are for
*explaining*, not for *ranking*.

**But look at the other half of the ledger.** Because content is popularity-blind,
it recommends across the *whole* catalogue and recovers the long tail that
popularity never touches (measured in [`notebooks/04_cf_vs_content.ipynb`](notebooks/04_cf_vs_content.ipynb)):

| model | head recall | tail recall | catalog coverage |
|---|---|---|---|
| popularity | 1.000 | **0.000** | **0.112** |
| item_cf | 0.881 | 0.545 | 0.997 |
| **content_knn** | 0.371 | **0.781** | **1.000** |

Popularity is a perfect echo chamber: flawless on hits, **zero** tail recall, and
it ever surfaces only ~11% of the catalogue. `content_knn` is its mirror image —
mediocre on hits, but the **strongest tail recall in the project** and **full
catalogue coverage**. The committed beyond-accuracy table tells the same story for
the two content-leaning models (taste_engine and track2vec both reach 0.772 tail
recall and 1.000 coverage versus popularity's 0.000 / 0.160; REPORT.md §4.9).

**Why does this happen — and why is it worth 0.078?**

1. **Popularity blindness cuts both ways.** The head-chasing that makes popularity
   look accurate is precisely what pins its tail recall at zero. Content never
   chases the head, so it is the only model that reliably digs into the tail.
2. **Cold-start immunity.** A brand-new track with no plays has an empty column in
   $A$ — invisible to *every* collaborative model in §2–4 until people put it in
   playlists. But it has a feature vector the moment it exists, so content can
   recommend it on day one. That is the classic reason to keep a content model in
   a real system even when it loses the offline metric.
3. **Coverage is a product goal, not a rounding error.** A system that only ever
   plays the hits is a bad music product regardless of its R-precision. Content is
   how you serve the catalogue.

So content-based earns its place not by winning the accuracy race but by covering
the ground the accurate models refuse to — which is the whole reason §6 blends
them.

---

## 6. Hybrids

If item-CF owns big seeds, the title model owns the cold start (§7), and content
owns the tail, the obvious move is to **combine** them. This repo has two hybrids,
and they are case studies in *when blending helps and when it hurts.*

### 6.1 Trust-blend (a per-user hybrid)

Inside the Taste Engine, the two *sources of preference* — what a listener
**stated** in words and what the model **learned** from their tracks — are blended
by a single knob:

$$
w = \tau\,\hat w_{\text{stated}} + (1-\tau)\,\hat w_{\text{learned}}.
$$

$\tau$ ("trust") is a safety valve. When the stated preference is honest, saying
your taste out loud helps a little (R-precision 0.323 → 0.328 as $\tau$ goes 0→1).
When the stated preference is *adversarial* (the user states the opposite of their
true taste), high trust is a cliff — but **leaning on the learned side ($\tau=0$)
fully rescues it**: 0.326 at $\tau=0$ collapsing to 0.019 at $\tau=1$ (REPORT.md
§4.3). The design lesson: expose the blend weight, and default it to the side you
trust.

### 6.2 Two-stage hybrid (candidate generation + rerank)

The heavyweight, [`models/hybrid.py`](src/playlistcont/models/hybrid.py), mirrors
the recipe that won the 2018 challenge:

- **Stage 1 — candidate generation.** Union the top candidates from item-CF, ALS,
  track2vec, the title model, and popularity. Cheap models nominate; you only need
  *recall* here.
- **Stage 2 — learned rerank.** Score each candidate with a gradient-boosted
  classifier (LightGBM, with a logistic-regression fallback) over per-candidate
  features — each model's score, popularity, artist overlap — trained on held-out
  tracks from the *training* playlists. This stage buys *precision*.

**When it helps, and when it doesn't.** On synthetic data the hybrid wins overall
(0.322, README results table). But the honest and more interesting result is where
it *loses*: on `title_random_100`, plain item-CF beats the full ensemble (0.468 vs
0.355, REPORT.md §4.2), and on the **real** MPD plain item-CF wins outright
(0.148 vs the hybrid's 0.131, REPORT.md §4.10). Once co-occurrence is near-
sufficient, the reranker only adds noise. **More model is not always better** —
blend when your signals are genuinely complementary, not out of habit.

### 6.3 Routed hybrid (blend by query shape)

The routed hybrid ([`models/routed.py`](src/playlistcont/models/routed.py)) takes
that lesson to heart: instead of always blending, it *routes* by how much seed it
was given, leaning entirely on the title model at zero seeds and entirely on
co-occurrence at large seeds. On synthetic data it tops the suite (0.379 vs the
plain hybrid's 0.357) by fixing the cold start without regressing anything else
(README key findings §2). On real data routing is net-neutral overall but **still
rescues the `title_only` bucket** (0.044 → 0.076, REPORT.md §4.10) — the part of
the division of labour that transferred.

---

## 7. Evaluation — and why offline metrics mislead

You cannot improve what you measure badly. The challenge uses three metrics, all
in [`challenge/metrics.py`](src/playlistcont/challenge/metrics.py) and unit-tested
against hand values. Let $R$ be the number of held-out ("ground-truth") tracks.

### 7.1 R-precision (the headline)

Fraction of the top-$R$ predictions that are correct, *plus* 0.25 credit for
getting an **artist** right even when you missed the exact track:

$$
\text{R-prec} = \frac{\#\{\text{track hits in top }R\} + 0.25\cdot\#\{\text{extra artist hits}\}}{R}.
$$

**Worked example.** Ground truth is $R=4$ tracks $\{a,b,c,d\}$. Your top-4 is
$[a, x, b, y]$, and $y$ happens to be by the same artist as the missed track $c$.
Then: 2 exact track hits ($a,b$), plus one artist hit ($y$→$c$'s artist) worth
0.25:

$$
\text{R-prec} = \frac{2 + 0.25}{4} = 0.5625.
$$

The artist credit is a deliberate softening — half-credit for being in the right
neighbourhood.

### 7.2 NDCG (rank-sensitive)

R-precision doesn't care *where* in the top-$R$ your hits land; NDCG does, by
discounting hits deeper in the list logarithmically. A hit at position $i$
(0-indexed) is worth $1/\log_2(i+2)$; normalize by the best possible arrangement.

**Worked example.** Ground truth $\{a,b\}$, prediction $[a, x, b]$. Hits at
positions 0 and 2:

$$
\text{DCG} = \frac{1}{\log_2 2} + \frac{1}{\log_2 4} = 1 + 0.5 = 1.5, \qquad
\text{IDCG} = \frac{1}{\log_2 2} + \frac{1}{\log_2 3} = 1 + 0.631 = 1.631,
$$

$$
\text{NDCG} = 1.5 / 1.631 = 0.920.
$$

The perfect ordering $[a,b,\dots]$ would score 1.0; slipping $b$ to position 2 costs
about 8%.

### 7.3 Recommended-Songs Clicks (a UX proxy)

How many "give me 10 more" refreshes until the first hit:
$\lfloor \text{rank of first hit} / 10 \rfloor$, with a penalty of 51 if you never
hit. First hit at position 3 → $\lfloor 3/10\rfloor = 0$ clicks (great). First hit
at position 12 → $\lfloor 12/10\rfloor = 1$ click. This rewards getting *something*
right near the very top.

### 7.4 Why offline metrics mislead

Three cautions this project ran into directly:

- **They reward popularity.** Held-out tracks are themselves popularity-skewed, so
  a model that recommends hits scores respectably while serving a terrible product.
  Popularity's synthetic *clicks* looked strong (0.21) and then **collapsed to
  21.7 on real data** (REPORT.md §4.10) — the flattering number was an artifact.
- **They are blind to coverage and tail.** R-precision cannot see that popularity
  has **zero** tail recall and ~16% catalogue coverage (REPORT.md §4.9). That is
  why we report beyond-accuracy metrics from
  [`challenge/coverage.py`](src/playlistcont/challenge/coverage.py) alongside the
  headline — the content story in §5 is invisible otherwise.
- **A single overall number hides the division of labour.** The most important
  qualitative result in the whole study is that **different models win different
  scenarios** (REPORT.md §4.2): the title model owns `title_only` (0.474 vs the
  0.193 floor) because it is the only model with any signal at zero seeds; item-CF
  owns large random seeds; the ensemble owns the middle. Averaging them into one
  leaderboard number erases exactly the structure a designer needs to see. Report
  the breakdown.

---

## 8. What actually mattered here

Pulling the threads together — and this is the section
[`ABLATIONS.md`](ABLATIONS.md) exists to support:

1. **Co-occurrence is almost everything.** Item-CF is the strongest single signal
   on both synthetic and real data, and it *wins outright on real data* (0.148,
   REPORT.md §4.10). The Taste Engine's accuracy is its co-occurrence term, not its
   axes (axis-only 0.122 < floor 0.162; ABLATIONS.md §2e). If you build one thing,
   build good item-CF.

2. **More model is not always better.** The two-stage hybrid wins on synthetic data
   but *loses to plain item-CF on real data*, and the reranker actively hurts on
   large random seeds (REPORT.md §4.2, §4.10). Blend only genuinely complementary
   signals.

3. **The interpretable axes earn their keep as explanations, not ranking.** Dropping
   any single axis barely moves accuracy (±0.003, REPORT.md §4.4), yet the flavor
   clusters recover the true archetypes almost perfectly (ARI 0.98 at $k=3$, §4.4).
   Know which job a component is actually doing.

4. **Accuracy and coverage are a genuine trade-off, and the loser is worth keeping.**
   Popularity wins the head and owns *zero* of the tail; pure content
   (`content_knn`, 0.078) loses the headline but delivers the **best tail recall and
   full catalogue coverage** in the project (§5.2, notebook 04) and is immune to
   cold start. A real system keeps the content model on the bench for exactly the
   queries the accurate models fail.

5. **Offline numbers are a starting point, not a verdict.** The synthetic study got
   the *architecture* lessons right (division of labour, the title model's cold-start
   role, co-occurrence's dominance on big seeds) but the *ranking-order* lessons
   partly wrong (hybrid supremacy, popularity's clicks) — caught only by re-running
   on real data (REPORT.md §4.10). Always ask what your evaluation cannot see.

For the executable version of this story — building item-CF from scratch, watching
it approximate the library model, and plotting the accuracy-vs-coverage trade-off
— see [`notebooks/04_cf_vs_content.ipynb`](notebooks/04_cf_vs_content.ipynb).

---

## Appendix A: item-CF normalizations, derived

Treat each track as its **column** of $A$ — a binary vector over playlists. Two
tracks' raw co-occurrence $c_{ij}$ is the dot product of their columns, and a
track's popularity $c_i$ is its column's squared norm (binary ⇒ $\|a_i\|^2 =
\sum_p a_{pi}^2 = \sum_p a_{pi} = c_i$).

**Cosine** is then literally the cosine of the angle between the two column
vectors:

$$
\cos(a_i,a_j) = \frac{a_i^\top a_j}{\|a_i\|\,\|a_j\|} = \frac{c_{ij}}{\sqrt{c_i}\sqrt{c_j}} = \frac{c_{ij}}{\sqrt{c_i c_j}}.
$$

The code adds a shrinkage $\lambda$ in the denominator
($c_{ij}/(\sqrt{c_ic_j}+\lambda)$, $\lambda=10$) so a pair that co-occurs once by
luck between two rare tracks isn't crowned a perfect match — it pulls low-support
similarities toward zero, a poor man's Bayesian prior.

**PMI** compares the observed joint rate to the independence assumption. Estimate
probabilities by frequencies: $\hat p(i)=c_i/N$, $\hat p(i,j)=c_{ij}/N$. Then

$$
\text{PMI} = \log\frac{\hat p(i,j)}{\hat p(i)\hat p(j)} = \log\frac{c_{ij}/N}{(c_i/N)(c_j/N)} = \log\frac{c_{ij}\,N}{c_i c_j}.
$$

Positive PMI clips the negatives to 0 (pairs that co-occur *less* than chance carry
no positive recommendation signal). Because it divides by $c_i c_j$ (not the
gentler $\sqrt{c_i c_j}$), PMI discounts popularity harder than cosine — which is
why, in §2, it zeroed the popular $t_1$–$t_2$ pair that cosine merely down-weighted.

## Appendix B: the ALS update, derived

Hold $V$ fixed and minimize the objective over a single playlist's factor $u_p$.
Writing $C^p = \text{diag}(c_{p1},\dots,c_{pT})$ and $r_p$ for playlist $p$'s
preference row, the per-playlist objective is

$$
J(u_p) = (r_p - Vu_p)^\top C^p (r_p - Vu_p) + \lambda\|u_p\|^2.
$$

Differentiate and set to zero:

$$
\frac{\partial J}{\partial u_p} = -2V^\top C^p(r_p - Vu_p) + 2\lambda u_p = 0
\;\;\Longrightarrow\;\;
(V^\top C^p V + \lambda I)\,u_p = V^\top C^p r_p,
$$

$$
\boxed{\,u_p = (V^\top C^p V + \lambda I)^{-1} V^\top C^p r_p\,}.
$$

It is an $f\times f$ solve ($f\approx 64$), cheap because $f$ is tiny. Hu et al.'s
trick is to write $V^\top C^p V = V^\top V + V^\top(C^p - I)V$: the first term is
shared across all playlists (compute once per sweep), and $C^p - I$ is nonzero only
on the handful of tracks playlist $p$ actually contains, so each solve touches only
those rows. The track factors follow by the symmetric update with $U$ fixed.
Alternating the two half-steps monotonically decreases $J$; ~15 sweeps suffice.

## Appendix C: skip-gram in one page

For a centre track $t$ and one context track $t'$, skip-gram with negative sampling
turns "do these two belong together?" into logistic regression on the dot product
of their vectors. The probability they are a true co-occurring pair is
$\sigma(v_{t'}^\top v_t)$ with $\sigma(z)=1/(1+e^{-z})$. We want that near 1 for
real pairs and near 0 for $M$ random "negative" tracks $t_m$ drawn from a noise
distribution $P_n$ (popularity$^{3/4}$ in the usual recipe). The per-pair loss:

$$
\ell = -\log\sigma(v_{t'}^\top v_t) - \sum_{m=1}^{M}\log\sigma(-v_{t_m}^\top v_t).
$$

Gradient descent on $\ell$ nudges $v_t$ toward its true context vectors and away
from the negatives. Summed over every centre–context pair in every playlist, this
is the objective in §4.1. Levy & Goldberg (2014) proved the optimum satisfies
$v_{t'}^\top v_t = \text{PMI}(t,t') - \log M$ — i.e. skip-gram is factorizing a
shifted PMI matrix, tying it straight back to Appendix A and closing the loop
between the embedding (§4), the factorization (§3), and the neighbourhood (§2)
views of the very same co-occurrence structure.

---

*Numbers in this document are cited to the committed file they come from
(`results/…`, `REPORT.md`, `ABLATIONS.md`). If you re-run the harness and see a
figure drift, trust the committed file over your memory — and read
[`REPORT.md`](REPORT.md) §3.4 for the formal statements this document paraphrases.*
