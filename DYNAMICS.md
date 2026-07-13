# Taste over time — change-point detection (Phase 3)

This is the scientific centrepiece: *when* did a listener's taste change, and can
a detector recover those moments **honestly** — scored against planted ground
truth, without crying wolf on the traps and seasonal bumps that are designed to
fool it? Phase 2 (`analytics/stats.py`) shipped a baseline change scanner and
documented its one weakness: it flags the December seasonal bump as a change.
Phase 3 builds a proper method ladder, ablates the feature representation, adds a
seasonal-handling toggle, and measures all of it against the generator's truth.

Every number below comes from a seeded, offline, runnable script:

```bash
python experiments/exp_dynamics.py     # ~100 s, writes results/dynamics_benchmark.csv
```

The package lives in [`src/playlistcont/dynamics/`](src/playlistcont/dynamics/);
the recommended default detector that Phase 4 consumes is
`dynamics.recommended_detector()`.

## Setup

* **Data & ground truth.** The `history/synthetic.py` generator plants, per
  listener: 3–5 piecewise-stationary **regimes** (archetype mixtures), the
  **transitions** between them (either *abrupt* on a date, or *drift* over a
  recorded `[drift_start, drift_end]` window), a `volume_only` **trap** (listening
  volume jumps ~3.5×, mixture unchanged) and a `binge` **trap** (one album
  dominates, mixture unchanged), and a recurring **December seasonal** bump of
  "party pop". Traps and the seasonal bump are annotated as *not* change points —
  flagging them is a false positive.
* **Two generator settings.** `auto` (the default sampler: mixed abrupt+drift,
  traps + seasonal on) and `controlled` (four well-separated pure regimes). Note a
  generator constraint: supplying explicit `regimes` forces **abrupt** transitions,
  so genuine *drift* is exercised only by `auto` — whose sampler plants drift on
  ~half of interior transitions. Across the 16 histories (8 seeds × 2 settings)
  the generator planted **37 abrupt + 13 drift** transitions.
* **Windows.** `windows.build_windows` turns the play stream into a per-window
  feature matrix (weekly, ~104 windows over the 2-year histories). Fixed column
  order: 5 scalar-axis means, 10 genre-bucket shares, *k* flavor shares, then three
  diagnostics (discovery rate, intensity = plays/day, coverage). Windows below
  `min_events` (=30) plays are **masked**, not dropped. The four detection
  **representations** are `scalar_axes`, `genre_shares`, `flavor_shares`, and
  `combined` (all three concatenated); the diagnostics are deliberately kept out of
  the detection space — see *Trap immunity*.
* **Stable flavors.** The taste engine's `flavor_clusters` refits k-means on every
  call, so cluster identities would flap window-to-window. `flavors.fit_flavors`
  fixes this by fitting k-means (explicit `random_state`, `n_init`) **once** on the
  union of the listener's track scalar features, then *assigning* every window's
  plays to those frozen centroids (`flavors.assign`). Names reuse the engine's
  `name_flavor`. Stable identity is what makes "flavor share" a comparable time
  series; it is unit-tested.
* **Standardisation (applies to cusum/pelt/bocpd).** Detectors run on a **z-scored**
  view: take a representation's columns, keep the unmasked windows in order, and
  standardise each column to mean 0 / unit variance over those windows (population
  std, 1e-9 floor). This weighs a genre-share move against a valence move by its
  own variability, not its raw units. `baseline_scan` is the exception — it delegates
  to Phase-2 `adjacent_scan`, which does its own within-test effect-size scaling.
* **Metrics.** Precision / recall / F1 with **±14-day** matching tolerance,
  localisation MAE (mean hit distance), segmentation ARI, and false-positive
  attribution split into `fp_trap` / `fp_seasonal` / `fp_other`.

## The method ladder

All detectors return the same `[DetectedChange(date, score, method)]`, so the
benchmark lines them up on equal footing.

**`baseline_scan`** — the Phase-2 detector wrapped into the common interface. It
compares consecutive calendar months, BH-corrects the whole scan as one family, and
flags a boundary whose scalar-axis tests clear q<0.05 *and* the effect-size floor.
It is representation- and season-agnostic (the fixed reference), and it inherits
Phase-2's known habit of firing on December.

**`cusum`** — self-implemented multivariate CUSUM. Per column, the two-sided tabular
recursion with slack `k` (in σ units, since the input is z-scored):
`S⁺ₜ = max(0, S⁺ₜ₋₁ + zₜ − k)`, `S⁻ₜ = max(0, S⁻ₜ₋₁ − zₜ − k)`, column statistic
`max(S⁺, S⁻)`. Columns are **combined** by `max` (default — most sensitive to a jump
in any *single* axis; a taste change often moves one axis first) or `l2` (pools a
diffuse change). On crossing `threshold` we back-date the alarm to where the run
began (CUSUM lags the change; back-dating recovers most of the delay) and reset.
**Adaptive baseline (crucial for offline use):** a persistent regime shift keeps the
z-scores high forever, so a textbook CUSUM referenced to the *global* mean re-fires
every few windows for the rest of the regime. We instead re-anchor the reference to
the new post-change level after each alarm, so CUSUM measures departures from the
*current* regime, not the global mean — this turned mean F1 from ~0.3 to ~0.89.
Defaults: `k=1.0`, `threshold=5.0`, `combine="max"`.

**`pelt`** — exact PELT from the `ruptures` library (pinned `ruptures>=1.1.9`;
installs and runs cleanly here — no fallback needed). Two costs: `l2`
(piecewise-constant mean — cheap, and it matches how regimes are actually planted)
and `rbf` (a kernel cost that also catches distributional changes). The penalty is a
parameter; the default is cost-aware (`default_penalty`): `d·log n` for `l2` (the
Schwarz/BIC penalty for a Gaussian mean-shift on z-scored data) but `log n` for `rbf`
— the rbf kernel cost is bounded ~O(1) per segment *regardless of dimension*, so
scaling by `d` over-penalises so hard it finds **zero** changes on a 19-column series
that plainly has four. `penalty_sweep` returns the change-count-vs-penalty curve and
an **elbow** default via the Kneedle rule (min-max normalise both axes, take the point
farthest below the first→last chord).

**`bocpd`** — self-implemented Bayesian Online CPD (Adams & MacKay 2007). Each z-scored
column is an independent Gaussian with **known** variance (=1, honest because the input
is standardised to unit variance by construction) and a Normal prior on the segment
mean; the joint log-predictive sums per-column Gaussians. Two design points worth
stating, because both bit us:

1. Under a *constant* hazard, `P(rₜ=0 | data)` is provably always exactly the hazard
   `H` — it carries no information. So we do **not** threshold it (a common tutorial
   mistake). Instead we track the **MAP run length** `r*ₜ = argmax P(rₜ)`: it climbs by
   1 each window and *collapses* at a change. We emit a change where `r*` drops by more
   than one (guarded by a minimum prior run length), locate the onset at `date[t − r*ₜ]`
   (the run length *is* the time since the change), and score by the relative collapse.
2. The unknown-variance (Normal-Gamma) version *absorbs* a new regime as inflated
   variance and never resets (measured: it silently missed the first change on every
   seed). The known-variance model, which cannot explain a mean shift away as noise, is
   both more principled here and far more sensitive. **BOCPD ships** (not experimental):
   with these two fixes it reaches mean F1 ≈ 0.90–0.95, competitive with PELT — its only
   real weakness is localisation (MAE ~6 days vs PELT's ~1.5), an inherent detection
   delay of an online filter.

### Seasonal handling (`deseasonalize`, a preprocessing toggle)

With the toggle on, and only when the history spans **≥ 18 months**, we subtract a
recurring month-of-year offset per column before standardising: `corrected = value −
(mean_of_this_calendar_month − global_mean)`, estimated as the pooled overall
month-of-year mean. Below 18 months it is a documented no-op. A leave-one-year-out
estimate (correct each year from the *others*, so a genuine one-off December change is
not partially erased) is the principled refinement; with only two years of synthetic
data pooled and LOO are nearly identical, so we ship the simpler pooled version and
note the limitation. The toggle is evaluated **both ways** below.

### Evaluation rules (`eval.py`)

Planted change_points become **targets**: an abrupt change is a point target (hit if a
detection is within tolerance); a drift transition is an *interval* target `[drift_start,
drift_end]` (hit if a detection falls in `[start−tol, end+tol]`). Matching is
**one-to-one greedy by distance** — form every within-tolerance (detection, target)
pair, sort by distance, accept greedily skipping anything already used. Distance (=
per-hit localisation error) is `|det − date|` for abrupt and distance-to-interval for
drift. Every unmatched detection is attributed, in priority order, to `fp_trap` (within
tolerance of a trap span), else `fp_seasonal`, else `fp_other`; `trap_fp_rate` and
`seasonal_fp_rate` are the fraction of trap / seasonal spans that caught a false alarm.
All of this is pinned to hand-computed fixtures in `tests/test_dynamics.py`.

## Headline benchmark

Top configurations by the composite `F1 − 0.5·(trap_fp_rate + seasonal_fp_rate)/2`
(λ=0.5: a false alarm on a trap or December bump costs half an F1 point — steep on
purpose, because "don't cry wolf" is a product requirement here). 16 runs per row
(8 seeds × 2 settings), weekly windows, ±14-day tolerance.

| method | representation | deseason | F1 | P | R | loc MAE (d) | trap FP | seasonal FP | ARI | composite |
|---|---|---|---|---|---|---|---|---|---|---|
| **pelt_rbf** | **combined** | **off** | **1.000** | 1.000 | 1.000 | 1.5 | **0.000** | **0.000** | 0.969 | **1.000** |
| pelt_rbf | flavor_shares | off | 0.970 | 0.979 | 0.969 | 1.4 | 0.000 | 0.031 | 0.938 | 0.962 |
| pelt_l2 | scalar_axes | off | 0.970 | 0.979 | 0.969 | 1.5 | 0.000 | 0.062 | 0.934 | 0.954 |
| pelt_l2 | combined | off | 0.975 | 0.958 | 1.000 | 1.5 | 0.000 | 0.094 | 0.960 | 0.952 |
| cusum | scalar_axes | off | 0.939 | 0.972 | 0.922 | 1.8 | 0.000 | 0.031 | 0.901 | 0.931 |
| bocpd | flavor_shares | off | 0.949 | 0.947 | 0.969 | 6.0 | 0.000 | 0.094 | 0.850 | 0.925 |
| **baseline_scan** | (native axes) | — | **0.538** | — | — | — | **0.375** | **0.938** | — | **0.210** |

Full grid: [`results/dynamics_benchmark.csv`](results/dynamics_benchmark.csv).

**Winner: PELT with an `rbf` cost on the `combined` representation, no
deseasonalisation** — perfect F1, zero trap and zero seasonal false positives, ARI
0.969, ~1.5-day localisation. It beats the Phase-2 baseline on composite by **+0.790**
(1.000 vs 0.210). The baseline isn't *bad* at finding changes (recall is high) — it is
sunk by false positives: it flags **93.8%** of December seasonal spans and **37.5%** of
trap spans. PELT-`l2` on `combined` is statistically tied (F1 0.975, composite 0.952)
and more interpretable (piecewise-constant means, exactly the planted structure); it is
the robust alternative if a kernel cost is unwanted.

## Representation ablation — which feature space carries the change signal?

This is the taste-over-*time* analogue of the repo's axis-vs-CF finding (`ABLATIONS.md`
§2c/2e, where interpretable axes carried most of the taste-engine signal). Best composite
per representation (fancy methods):

| representation | best method | F1 | composite |
|---|---|---|---|
| **combined** | pelt_rbf | **1.000** | **1.000** |
| flavor_shares | pelt_rbf | 0.970 | 0.962 |
| scalar_axes | pelt_l2 | 0.970 | 0.954 |
| genre_shares | pelt_l2 (deseason) | 0.946 | 0.931 |

**The change signal is broadly distributed, and `combined` wins by pooling it.** Each
single space is already strong (F1 ≈ 0.95–0.97), because a regime change here *is* a
change in archetype mixture, which moves mood axes, genre shares and flavor shares all
at once. The mood axes (`scalar_axes`) and the fit-once **flavors** each carry the signal
slightly better than raw `genre_shares` — the same lesson as Phase 2's ablations: the
continuous interpretable-axis space is where personal-taste structure lives most cleanly.
Concatenating everything (`combined`) gives PELT the most columns to localise a shift and
pushes it to a clean sweep. In short: **taste-over-time change is carried best by the
interpretable mood + flavor space, and best of all by pooling all three views.**

## Seasonal on/off effect

The toggle does what it should for the seasonally-fragile detectors and is *unnecessary*
(even mildly harmful) for the seasonally-robust one:

| method | seasonal FP (off → on) | F1 (off → on) |
|---|---|---|
| cusum | 0.391 → **0.156** | 0.873 → 0.543 |
| bocpd | 0.328 → **0.219** | 0.905 → 0.812 |
| pelt_l2 | 0.172 → 0.203 | 0.944 → 0.756 |
| pelt_rbf | 0.109 → 0.180 | 0.959 → 0.783 |

Deseasonalising roughly **halves** CUSUM's December false-positive rate — it is the right
fix for the adjacent/online detectors that see a December bump as a local step. But it
*hurts* PELT: PELT's **global segmentation already ignores** the transient December blip
(a 4–5-week excursion that reverts is not worth a segment under the BIC penalty), so
subtracting a pooled month-of-year mean — which, on a 2-year history, is contaminated by
whatever regime happened to overlap each December — only adds noise and drops F1. This is
the honest nuance behind the winner's `deseason=off`: **the best detector doesn't need the
seasonal fix, and the seasonal fix helps only the detectors that aren't the best.**

## Trap immunity

Trap false-positive rates are **0.000** across the entire top of the table. This is
*structural*, not lucky: the `volume_only` trap moves listening volume, and we deliberately
excluded the `intensity` (plays/day) diagnostic from every detection representation — so a
volume spike is literally invisible to the detector. The `binge` trap plays one album to
death but leaves the archetype *mixture* unchanged, so under plays-weighting it barely
moves the mood/genre/flavor vector, and PELT's penalty won't buy a segment for it. (A
`unique`-weighting mode in `build_windows` flattens binge even further by counting each
track once — offered and documented, though the benchmark uses `plays`.) The only detector
that trips traps at all is the Phase-2 baseline (37.5%), which is not surprising: adjacent
month-pair testing is exactly the regime a short intense burst can fool.

## Penalty-sweep sensitivity

For PELT-`l2` on `combined` (auto setting, 8 seeds), the change count vs penalty:

| penalty | mean #changes |
|---|---|
| 8.8 | 5.75 |
| 37.9 | 4.00 |
| 78.3 | 3.62 |
| 162.1 | 3.00 |
| 335.4 | 2.00 |
| 693.9 | 0.50 |

The true number of targets averages **3.25** per auto history. The BIC default penalty
(`d·log n` ≈ 88 for a 19-column, ~104-window series) lands at ~3.6 changes — a slight
over-detection that the ±14-day matching and the composite tolerate, and it is what the
headline uses. The **Kneedle elbow over-penalises** here (mean 350 ± 156 → ~2 changes,
under-detecting): the count-vs-penalty curve is smooth and convex, so the "max distance to
chord" knee sits too far right. The practical takeaway: the change count is *stable across a
broad penalty band* (roughly 40–160 all give 3–4 changes), so the exact penalty is not
delicate; we prefer the analytic BIC default over the automatic elbow, and expose both via
`penalty_sweep` for inspection.

## Limitations

* **Synthetic only, and the planted structure is simple by design.** Regimes are
  piecewise-stationary archetype mixtures with linear drift — a structure a good detector
  *should* recover, which is exactly why F1 reaches 1.0. This measures whether the methods
  work on the planted model, **not** that real taste evolves this cleanly. Real listening is
  messier (gradual, multi-scale, seasonal in more than one way); expect every headline
  number to be optimistic on real data.
* **Deseasonalisation is under-powered at 2 years.** Pooled month-of-year means are
  contaminated by regime overlap; the honest fix (leave-one-year-out) needs ≥ 3 years to bite.
* **Masked-window stitching.** Detectors treat the sequence of unmasked windows as
  contiguous — a long data gap is stitched over, not modelled.
* **CUSUM / BOCPD localisation.** As online filters they lag; BOCPD's ~6-day MAE reflects an
  inherent detection delay that back-dating only partly removes.

## What Phase 4 should consume

`dynamics.recommended_detector()` returns the benchmark winner as a configured object:

```python
RecommendedDetector(
    method="pelt_rbf", representation="combined", granularity="week",
    deseasonalize=False, min_events=30, weighting="plays", params={"penalty": None},
)
# rd.detect(history_or_store) -> [DetectedChange(date, score, method)]
```

Weekly windows over the `combined` taste representation, PELT with the `rbf` cost and the
BIC-style default penalty, **no** deseasonalisation. Phase 4's trajectory + story UI should
call `.detect(...)` to get the change dates, then narrate each segment between them (the
regimes) using the Phase-2 `significant_shifts` layer for the *what-changed* copy. If a
kernel cost is undesirable, swap `method="pelt_l2"` — it is tied on accuracy and more
directly interpretable.
