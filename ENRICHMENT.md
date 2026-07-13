# Track-metadata enrichment (Phase 1.5)

Real tracks in this repo used to get only 5 audio-feature axes (via
`data/real_features.py`) and **no genre signal at all** — the 10 `genre:*` axes
were hard-zero for every real track. This phase adds two license-clean joins
plus a widened read of the audio-features table we already stream. Code lives
in `src/playlistcont/enrichment/`; the evidence base is the Phase 1.5
enrichment research (measured schemas, hit rates, licenses — summarized below
with the caveats intact).

## Sources

| Source | Adds | Join key | License |
|---|---|---|---|
| `maharshipandya/spotify-tracks-dataset` (HF) | seed-genre labels for ~89.7k unique tracks (114k rows) | exact 22-char Spotify `track_id` | **BSD** — clean |
| `LeData/media-metadata-musicbrainz-artists` (HF) | MusicBrainz folksonomy tags for 1.52M artists | normalized artist **name** (no shared id) | **CC0-1.0** — public domain |
| `ozefe/spotify_audio_features` (already in use) | 10 previously-ignored columns (`popularity, danceability, speechiness, loudness, liveness, key, mode, duration_ms, time_signature, name`) | exact Spotify `id` | see caveat below |

Downloads happen only at explicit call time (`enrichment/sources.py`), never at
import and never in tests; files cache under `.data/enrichment/` (gitignored).

### License caveat on the audio-features table itself

`ozefe/spotify_audio_features` is tagged `license: other` /
"spotify-developer-terms": its raw data comes from an Anna's Archive scrape of
Spotify's internal catalogue, and the dataset card itself says *"License:
Unspecified / Proprietary (Research Use Only)"*. **This repo already depended
on that table before this change** — the widened read adds no new exposure,
but the pre-existing murkiness is real and worth knowing about. The two *new*
sources were chosen specifically because their licenses are clean (BSD / CC0).

## What each join does

### Track-level: seed genres (exact id join)

**Caveat first**: the `track_genre` column is the **seed genre the track was
fetched under** when the dataset was built (1,000 tracks pulled per genre
seed), *not* a curated per-track label. ~21% of tracks appear under 2+ seeds
(114,000 rows vs 89,741 unique ids). We therefore aggregate each track's rows
into a genre **set** and treat it as soft evidence, never a single true label.

### Artist-level: MusicBrainz tags (normalized name join)

`tags` is user-editable folksonomy — genuinely useful genre/style labels for
well-tagged artists, mixed with administrative spam (`fixme`, `bogus artist`,
…) that we drop via an explicit denylist. Only ~21% of the 1.52M rows carry
any tag, but coverage is much better for artists people actually listen to
(measurements below).

**Collision policy** (names are not unique — cover bands, homonyms): when
multiple rows share a normalized name we keep the row with the most usable
tags and set `ambiguous=True` on the result. This is a heuristic and it can
lose: in our own validation, "Alex G" resolved to a same-named electronic
artist rather than the indie songwriter. The flag is surfaced (and counted in
the `EnrichmentReport`) precisely so downstream consumers don't mistake these
for confident matches.

### The curated tag → bucket mapping

Both joins map raw tags/seed genres onto the repo's 10 genre buckets
(`schema.GENRES`) through one explicit reviewed dict
(`artist_genres.TAG_TO_BUCKETS`, **187 canonical tags**, covering 68 of the
114 maharshipandya seed genres and the most frequent MusicBrainz tags).
Tags with no honest bucket (`classical`, `reggae`, `anime`, `latin`,
nationality and mood tags, …) are **kept as raw tags, never force-mapped** —
they land in the optional `artist_tags` store table
(`artist_name, tag, mapped_bucket-or-NULL`) instead of polluting the axes.

### How the axis vectors are written

Artist-level fills first; track-level **wins on conflict** (it is
track-specific; an artist tag paints every track the same). If the track join
matched but none of its seed genres map, the artist assignment is kept.
Weights are soft, mirroring the synthetic generator's convention: primary
bucket 0.85, further buckets 0.4, everything else 0. Scalar axes are never
touched; feature-less tracks get a fresh vector with the 0.5-neutral scalar
convention of `real_features.py`.

## Measured hit rates (small samples — read as indicative, not exhaustive)

Live-validated in this environment against the full downloaded tables
(1,520,644 artist rows; 114,000 track rows):

* **Research battery, 15 real artists across popularity tiers**
  (mega → niche/indie): name match **15/15**, ≥1 usable mapped genre bucket
  **14/15** (only Hovvdy had a tagless row), 4/15 flagged ambiguous — one of
  which ("Alex G") is a confirmed wrong-homonym resolution.
* **50 random artist names drawn from the maharshipandya table** (skews
  international/obscure): name match **29/50**, ≥1 mapped bucket **15/50**.
  Random-catalogue artists are much worse than artists-people-listen-to.
* **Fixture history of 200 real popular tracks (1,130 plays, 125 artists)
  built from maharshipandya rows**: combined genre coverage **91.4% of
  plays** (185/200 tracks); artist-only ablation 83.8%, track-only 78.7%;
  183/200 tracks artist-matched; 24 ambiguous artists. *Note*: the track-id
  join matched 200/200 here **by construction** (ids were sampled from the
  same table) — a real personal export will match far fewer by id (the
  research's partial overlap test against a 255M-row catalogue found id
  overlap concentrated in mainstream tracks), which is exactly why the
  artist-name path exists.

The `EnrichmentReport` returned by `enrich_history` (and surfaced in the
Library summary payload when `PLAYLISTCONT_ENRICH=1`) reports these same
quantities for *your* data: `n_tracks, artist_matched, track_matched,
genre_coverage` (share of **plays** whose track got any genre),
`ambiguous_artists`.

## What remains unsolved

* **AcousticBrainz is unverified**: its hosts were unreachable from the
  research environment and no HF mirror exists; mood-model axes remain a
  stretch goal until someone re-tests `data.metabrainz.org` elsewhere.
* **Spotify `/v1/artists` genres** is the right long-term keyed source
  (first-party taxonomy, batchable 50 ids/call) but needs a registered OAuth
  app — untested from here.
* Name-join ambiguity is flagged, not solved; resolving via real MBIDs (or
  Spotify artist ids once OAuth exists) is the actual fix.
* Track-id coverage on real exports is unmeasured at personal scale until a
  real export is run through this pipeline; the report makes that number
  visible per-run rather than promising one here.
