# Results & analysis

> **Looking for the real-data numbers?** These are the **synthetic-MPD** results.
> The real 1M-playlist run (and which of these conclusions transferred) is in
> **[`REAL_RESULTS.md`](REAL_RESULTS.md)**. Short version: on real data plain
> **item-CF wins**, not the hybrid.

All numbers below were produced by actually running
`python experiments/run_comparison.py` in this repo on **synthetic MPD**
(10,000 playlists, 5,000 tracks, 3,000 held-out playlists split into the 10
official scenarios, 100 cases per scenario except the 100-seed ones which are
limited by how many playlists are long enough → 25 each). The whole run,
including fitting all seven models, finishes in ~40 seconds.

The raw tables are in [`results/results.csv`](results/results.csv) and
[`results/results.md`](results/results.md); charts are in
[`results/figures/`](results/figures/).

> These are **synthetic-data** results. They are useful for *relative* model
> comparison and for validating that each model behaves as designed, but the
> absolute numbers are not comparable to the real MPD leaderboard (the 2018
> winner scored R-precision ≈ 0.224 on real data). Synthetic playlists are more
> internally consistent than real ones, so every model looks stronger and
> "popularity" in particular is unrealistically competitive on *clicks*.

## Overall (micro-averaged over all 690 scenario cases)

| model         | R-precision | NDCG   | Clicks (↓) |
| ------------- | ----------- | ------ | ---------- |
| **hybrid**    | **0.3224**  | **0.6084** | **0.1259** |
| als           | 0.3113      | 0.5681 | 0.4329     |
| item_cf       | 0.3090      | 0.5620 | 0.4506     |
| title         | 0.3077      | 0.5765 | 0.2224     |
| taste_engine  | 0.2996      | 0.5137 | 1.3482     |
| popularity    | 0.1677      | 0.4347 | 0.2059     |
| track2vec     | 0.1197      | 0.3132 | 6.2459     |

**The hybrid ensemble wins on all three metrics overall** — exactly the
"candidate generation + learned rerank" story that dominated the real 2018
challenge. It is not the best in *every* scenario, though, which is where the
analysis gets interesting.

## Who wins which scenario (R-precision)

| scenario          | winner        | note |
| ----------------- | ------------- | ---- |
| title_only        | **title** (0.474 vs 0.193 floor) | Only the title model beats popularity; everyone else has no seed and falls back to popularity. |
| title_1           | hybrid / title / taste (~0.45) | With one seed track, title similarity still carries most of the signal. |
| title_5, no_title_5 | hybrid (0.34/0.35) | Enough co-occurrence signal for CF/ensemble to take over. |
| title_10, no_title_10 | hybrid (0.27) | Ensemble lead widens as the seed grows. |
| title_25          | hybrid (0.30)  | |
| title_random_25   | item_cf / als (0.44/0.45) | Random seeds sample the whole playlist's taste, so neighbourhood models shine. |
| title_100         | als (0.27)     | Deep MF generalizes best from a long seed. |
| title_random_100  | **item_cf (0.47)** | Co-occurrence with a large random seed is almost unbeatable. |

### Key findings

1. **The title model is indispensable for the cold-start (title-only)
   scenario.** It scores R-precision 0.474 / NDCG 0.753 there; every other
   standalone model is pinned at the popularity floor (0.193 / 0.492) because
   they have no seed tracks to work with. This is the single most important
   qualitative result and it justifies including a dedicated title model.

2. **The ensemble wins overall but *loses* title-only to its own specialist**
   (hybrid 0.169 vs title 0.474). The blender is trained on seeded contexts, so
   it learns to trust co-occurrence/MF features and dilutes the pure title
   signal exactly where it should defer to it. An honest limitation: a
   production system would route title-only queries to the title model (or add
   a "has-seed" gate feature). We deliberately left this un-patched so the
   comparison shows the failure mode rather than hiding it.

3. **Neighbourhood models (item-CF) beat everything on the large *random*
   seeds.** With 25–100 randomly-sampled seed tracks, co-occurrence is a nearly
   sufficient statistic, and the simple item-CF model edges out even the
   ensemble (title_random_100: item_cf 0.468 vs hybrid 0.355). More model is not
   always better.

4. **ALS is the most consistent single model** — never the worst, strong across
   the seed-size range, and best on the longest ordered seeds (title_100). It is
   the backbone the ensemble leans on most.

5. **The taste engine is competitive, not dominant — by design.** Its
   R-precision (0.300 overall) sits in the CF pack, and its *clicks* are worse
   (1.35) because axis-matching surfaces taste-consistent tracks that don't
   always co-occur with the exact seed, pushing the first exact hit down the
   list. Its value is the interpretability layer (named flavor clusters +
   per-recommendation explanations + stated-preference control), which none of
   the other models offer. It is a real, trained model in the comparison, not a
   mock — it just optimizes for *explainable taste match* rather than raw recall.

6. **Track2Vec underperforms at this scale.** With only ~7k training playlists
   and 5k tracks, the word2vec embeddings are undertrained and centroid
   similarity is noisy (clicks 6.2). It improves with `--big` scale and more
   epochs, but the honest takeaway is that shallow embeddings need much more
   data than matrix factorization to become competitive on this task.

7. **Popularity is a deceptively strong *clicks* baseline on synthetic data.**
   Because synthetic playlists over-sample a shared popular core, the first
   popular recommendation often hits early (clicks 0.21). On the real MPD this
   gap would be far larger; treat the synthetic clicks numbers with suspicion.

## Ensemble gains

Averaged over all cases, the hybrid improves NDCG from 0.568 (best single
model, ALS) to 0.608 (+7%) and cuts clicks from 0.22 (title) to 0.126 (−43%).
The gains concentrate in the mid-seed scenarios (title_5 through title_25),
where no single signal dominates and the learned blend can combine
co-occurrence, MF, title, and artist-overlap features. Where one signal is
already near-optimal (title-only → title; large random seed → item-CF), the
ensemble matches or slightly trails the specialist.

## Reproducing

```bash
python experiments/run_comparison.py            # ~40s, the numbers above
python experiments/run_comparison.py --big      # 20k playlists / 15k tracks
python experiments/run_comparison.py --real /path/to/mpd/data   # real MPD
```
