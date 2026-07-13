"""Artist-name -> genre tags via the MusicBrainz artist-tag table (CC0).

Join key honesty: the LeData table has **no Spotify id** — the only shared key
with a listening history is the artist *name* string.  We use a normalized
exact match (lowercase, punctuation and diacritics stripped, whitespace
collapsed).  That resolves 15/15 of the research battery's real artist names,
but names are not unique in MusicBrainz (cover bands, homonyms), so a
**collision policy** is required and documented here:

    When multiple artist rows share a normalized name, prefer the row with the
    most usable (non-administrative) tags — homonym/cover-band rows are almost
    always sparsely tagged — and record ``ambiguous=True`` on the result so
    downstream consumers can surface the uncertainty instead of hiding it.

Tag honesty: ``tags`` is MusicBrainz's user-editable folksonomy, not a curated
genre field.  Administrative spam (``fixme``, ``bogus artist``, ...) is dropped
via :data:`TAG_DENYLIST`; everything else is kept as a raw tag.  Mapping raw
tags onto the repo's 10 genre buckets (:data:`playlistcont.data.schema.GENRES`)
goes through :data:`TAG_TO_BUCKETS`, an explicit reviewed dict — tags with no
entry are **kept as raw tags, never force-mapped**.
"""
from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from ..data.schema import GENRES

# ---------------------------------------------------------------------------
# normalization
# ---------------------------------------------------------------------------

_PUNCT_RE = re.compile(r"[^\w\s]", re.UNICODE)
_WS_RE = re.compile(r"\s+")


def normalize_artist_name(name: str) -> str:
    """Lowercase, strip diacritics and punctuation, collapse whitespace.

    ``"Beyoncé"`` and ``"beyonce"`` normalize identically; ``"AC/DC"`` -> ``"acdc"``.
    """
    s = unicodedata.normalize("NFKD", str(name))
    s = "".join(c for c in s if not unicodedata.combining(c))
    s = _PUNCT_RE.sub("", s.lower())
    return _WS_RE.sub(" ", s).strip()


def canonical_tag(tag: str) -> str:
    """Canonical spelling for tag lookup: lowercase, ``&``->``and``, separators
    (``-``, ``/``, ``_``) -> spaces, whitespace collapsed.

    This folds the two tables' spelling conventions together: maharshipandya's
    hyphenated seed genres (``hip-hop``, ``alt-rock``) and MusicBrainz's spaced
    folksonomy (``hip hop``, ``pop/rock``) hit the same mapping keys.
    """
    s = str(tag).lower().strip().replace("&", " and ")
    s = re.sub(r"[-/_]", " ", s)
    return _WS_RE.sub(" ", s).strip()


# ---------------------------------------------------------------------------
# curated tag -> genre-bucket mapping (explicit, reviewed; keys are canonical
# per ``canonical_tag``).  First bucket in each tuple is the primary.  Built by
# inspecting (a) all 114 maharshipandya seed genres and (b) the most frequent
# MusicBrainz folksonomy tags in the research sample + full table.  Tags that
# do not honestly fit one of the 10 buckets (classical, blues, reggae, latin,
# regional/nationality tags, mood tags, ...) are deliberately ABSENT — they
# pass through as raw tags.
# ---------------------------------------------------------------------------

TAG_TO_BUCKETS: Dict[str, Tuple[str, ...]] = {
    # ---- country --------------------------------------------------------
    "country": ("country",),
    "country rock": ("country", "rock"),
    "country pop": ("country", "pop"),
    "alt country": ("country", "indie"),
    "alternative country": ("country", "indie"),
    "outlaw country": ("country",),
    "contemporary country": ("country",),
    "country folk": ("country", "folk"),
    "nashville sound": ("country",),
    "honky tonk": ("country",),
    "bluegrass": ("country", "folk"),
    "americana": ("country", "folk"),
    "rockabilly": ("rock", "country"),
    "sertanejo": ("country",),          # Brazilian country
    # ---- rap ------------------------------------------------------------
    "rap": ("rap",),
    "hip hop": ("rap",),
    "hiphop": ("rap",),
    "trap": ("rap",),
    "gangsta rap": ("rap",),
    "conscious hip hop": ("rap",),
    "east coast hip hop": ("rap",),
    "west coast hip hop": ("rap",),
    "southern hip hop": ("rap",),
    "uk hip hop": ("rap",),
    "grime": ("rap",),
    "jazz rap": ("rap", "jazz"),
    "pop rap": ("rap", "pop"),
    # ---- indie ----------------------------------------------------------
    "indie": ("indie",),
    "indie rock": ("indie", "rock"),
    "indie pop": ("indie", "pop"),
    "indie folk": ("indie", "folk"),
    "indietronica": ("indie", "electronic"),
    "rock and indie": ("rock", "indie"),
    "noise pop": ("indie",),
    "dream pop": ("indie", "pop"),
    "shoegaze": ("indie", "rock"),
    "chamber pop": ("indie", "pop"),
    "baroque pop": ("indie", "pop"),
    "art pop": ("pop", "indie"),
    "lo fi indie": ("indie",),
    "bedroom pop": ("indie", "pop"),
    "slacker rock": ("indie", "rock"),
    "britpop": ("rock", "indie"),
    "post rock": ("rock", "indie"),
    "math rock": ("rock", "indie"),
    "garage rock": ("rock", "indie"),
    "neo psychedelia": ("rock", "indie"),
    # ---- pop ------------------------------------------------------------
    "pop": ("pop",),
    "pop rock": ("pop", "rock"),
    "classic pop and rock": ("pop", "rock"),
    "power pop": ("pop", "rock"),
    "dance pop": ("pop", "electronic"),
    "synth pop": ("pop", "electronic"),
    "synthpop": ("pop", "electronic"),
    "electropop": ("pop", "electronic"),
    "k pop": ("pop",),
    "j pop": ("pop",),
    "c pop": ("pop",),
    "cantopop": ("pop",),
    "mandopop": ("pop",),
    "j idol": ("pop",),
    "pop film": ("pop",),
    "teen pop": ("pop",),
    "bubblegum pop": ("pop",),
    "adult contemporary": ("pop",),
    "pop and chart": ("pop",),
    "europop": ("pop",),
    "pop soul": ("pop", "rnb"),
    "new wave": ("rock", "pop"),
    "soft rock": ("rock", "pop"),
    "pop punk": ("rock", "pop"),
    "folk pop": ("folk", "pop"),
    "disco": ("pop", "electronic"),
    "eurodance": ("electronic", "pop"),
    # ---- rock -----------------------------------------------------------
    "rock": ("rock",),
    "classic rock": ("rock",),
    "hard rock": ("rock",),
    "alt rock": ("rock", "indie"),
    "alternative rock": ("rock", "indie"),
    "alternative": ("rock", "indie"),
    "progressive rock": ("rock",),
    "psych rock": ("rock",),
    "psychedelic rock": ("rock",),
    "art rock": ("rock",),
    "blues rock": ("rock",),
    "acoustic rock": ("rock", "folk"),
    "folk rock": ("folk", "rock"),
    "southern rock": ("rock", "country"),
    "surf rock": ("rock",),
    "glam rock": ("rock",),
    "grunge": ("rock",),
    "punk": ("rock",),
    "punk rock": ("rock",),
    "post punk": ("rock", "indie"),
    "ska punk": ("rock",),
    "emo": ("rock",),
    "goth": ("rock",),
    "gothic rock": ("rock",),
    "rock and roll": ("rock",),
    "rock n roll": ("rock",),
    "j rock": ("rock",),
    "experimental rock": ("rock",),
    "stoner rock": ("rock", "metal"),
    "aor": ("rock",),
    "yacht rock": ("rock", "pop"),
    "hardcore": ("rock", "metal"),
    "post hardcore": ("rock", "metal"),
    # ---- electronic -----------------------------------------------------
    "electronic": ("electronic",),
    "electronica": ("electronic",),
    "edm": ("electronic",),
    "electro": ("electronic",),
    "house": ("electronic",),
    "deep house": ("electronic",),
    "chicago house": ("electronic",),
    "progressive house": ("electronic",),
    "techno": ("electronic",),
    "detroit techno": ("electronic",),
    "minimal techno": ("electronic",),
    "trance": ("electronic",),
    "ambient": ("electronic",),
    "idm": ("electronic",),
    "breakbeat": ("electronic",),
    "drum and bass": ("electronic",),
    "dubstep": ("electronic",),
    "hardstyle": ("electronic",),
    "garage": ("electronic",),          # spotify's 'garage' seed = UK garage
    "uk garage": ("electronic",),
    "trip hop": ("electronic",),
    "downtempo": ("electronic",),
    "chillout": ("electronic",),
    "club": ("electronic",),
    "dance": ("electronic", "pop"),
    "j dance": ("electronic",),
    "future bass": ("electronic",),
    "synthwave": ("electronic",),
    "industrial": ("electronic", "metal"),
    "jungle": ("electronic",),
    "acid jazz": ("jazz", "electronic"),
    # ---- folk -----------------------------------------------------------
    "folk": ("folk",),
    "contemporary folk": ("folk",),
    "traditional folk": ("folk",),
    "freak folk": ("folk", "indie"),
    "celtic": ("folk",),
    "singer songwriter": ("folk",),
    "songwriter": ("folk",),
    # ---- metal ----------------------------------------------------------
    "metal": ("metal",),
    "heavy metal": ("metal",),
    "black metal": ("metal",),
    "death metal": ("metal",),
    "doom metal": ("metal",),
    "thrash metal": ("metal",),
    "power metal": ("metal",),
    "nu metal": ("metal",),
    "progressive metal": ("metal",),
    "sludge metal": ("metal",),
    "speed metal": ("metal",),
    "symphonic metal": ("metal",),
    "alternative metal": ("metal",),
    "industrial metal": ("metal", "electronic"),
    "glam metal": ("metal", "rock"),
    "folk metal": ("metal", "folk"),
    "metalcore": ("metal",),
    "deathcore": ("metal",),
    "grindcore": ("metal",),
    # ---- rnb ------------------------------------------------------------
    "rnb": ("rnb",),
    "r n b": ("rnb",),
    "r and b": ("rnb",),
    "rhythm and blues": ("rnb",),
    "contemporary r and b": ("rnb",),
    "soul": ("rnb",),
    "neo soul": ("rnb",),
    "motown": ("rnb",),
    "funk": ("rnb",),
    "quiet storm": ("rnb",),
    "doo wop": ("rnb",),
    "gospel": ("rnb",),
    "new jack swing": ("rnb", "pop"),
    # ---- jazz -----------------------------------------------------------
    "jazz": ("jazz",),
    "vocal jazz": ("jazz",),
    "smooth jazz": ("jazz",),
    "jazz fusion": ("jazz",),
    "swing": ("jazz",),
    "big band": ("jazz",),
    "bebop": ("jazz",),
    "cool jazz": ("jazz",),
    "free jazz": ("jazz",),
    "bossa nova": ("jazz",),
}

# every mapped bucket must be one of the schema's 10 genres — checked at import
assert all(b in GENRES for buckets in TAG_TO_BUCKETS.values() for b in buckets)

# MusicBrainz administrative/meta tags that are never genre signal.  Exact
# canonical matches, plus the prefix rules in ``_is_administrative``.
TAG_DENYLIST = frozenset({
    "special purpose artist", "special purpose", "meta artist",
    "tag spam", "spam", "spamess", "cotm", "cotm candidate",
    "2008 universal fire victim", "non music", "nature sounds",
    "production music", "has german audiobooks", "german audiobook reader",
    "audio drama", "audiobook", "spoken word", "label as artist",
    "series title as artist", "fuzzy artist series", "fictitious artist",
    "clean up", "use actual artists", "change to the actual performers",
    "needs merging", "unknown",
})

_DENY_PREFIXES = ("fixme", "merge", "bogus")


def _is_administrative(canon: str) -> bool:
    return canon in TAG_DENYLIST or canon.startswith(_DENY_PREFIXES)


def map_tag(tag: str) -> Tuple[str, ...]:
    """Map one raw tag to genre buckets; ``()`` when the tag is unmapped."""
    return TAG_TO_BUCKETS.get(canonical_tag(tag), ())


def map_tags_to_buckets(tags: Iterable[str]) -> Tuple[List[str], List[str]]:
    """Map raw tags -> (ordered genre buckets, kept raw tags).

    * Administrative tags (:data:`TAG_DENYLIST` + ``fixme``/``merge``/``bogus``
      prefixes) are dropped entirely.
    * Every surviving tag is kept in ``raw_tags`` (deduped, original spelling,
      input order) whether or not it maps — unmapped tags are NOT discarded.
    * Buckets are ordered by how many tags voted for them (descending), ties
      broken by first appearance, so ``buckets[0]`` is the primary genre.
    """
    raw: List[str] = []
    seen_raw = set()
    votes: Dict[str, int] = {}
    first_pos: Dict[str, int] = {}
    pos = 0
    for tag in tags:
        canon = canonical_tag(tag)
        if not canon or _is_administrative(canon):
            continue
        if canon not in seen_raw:
            seen_raw.add(canon)
            raw.append(str(tag).strip())
        for b in TAG_TO_BUCKETS.get(canon, ()):
            votes[b] = votes.get(b, 0) + 1
            if b not in first_pos:
                first_pos[b] = pos
            pos += 1
    buckets = sorted(votes, key=lambda b: (-votes[b], first_pos[b]))
    return buckets, raw


# ---------------------------------------------------------------------------
# name-keyed index over the artist table
# ---------------------------------------------------------------------------


@dataclass
class ArtistGenreRecord:
    """One resolved artist: kept raw tags, ordered genre buckets, ambiguity flag."""

    name: str                       # name as it appears in the chosen table row
    raw_tags: List[str] = field(default_factory=list)
    buckets: List[str] = field(default_factory=list)
    ambiguous: bool = False         # >1 table row shared this normalized name
    n_candidates: int = 1


def build_artist_genre_index(
    table,
    names: Optional[Iterable[str]] = None,
) -> Dict[str, ArtistGenreRecord]:
    """Index a LeData-shaped artist table (``name``, ``tags`` columns) by
    normalized name.

    ``names`` (optional) restricts the index to those artist names' normalized
    forms — pass the history's artist names to avoid indexing all 1.52M rows.

    Collision policy (documented in the module docstring): among rows sharing a
    normalized name, keep the row with the most usable tags; mark the record
    ``ambiguous=True`` and record ``n_candidates``.
    """
    wanted = None
    if names is not None:
        wanted = {normalize_artist_name(n) for n in names}

    # (best_tag_count, record) per normalized name
    best: Dict[str, Tuple[int, ArtistGenreRecord]] = {}
    for name, tags in zip(table["name"], table["tags"]):
        norm = normalize_artist_name(name)
        if not norm or (wanted is not None and norm not in wanted):
            continue
        # tags arrive as list/ndarray, or None/NaN when the row has none
        if tags is None or isinstance(tags, float):
            tag_list: List[str] = []
        else:
            tag_list = [str(t) for t in tags]
        buckets, raw = map_tags_to_buckets(tag_list)
        prev = best.get(norm)
        if prev is None:
            best[norm] = (len(raw), ArtistGenreRecord(
                name=str(name), raw_tags=raw, buckets=buckets))
        else:
            prev_count, rec = prev
            rec.ambiguous = True
            rec.n_candidates += 1
            if len(raw) > prev_count:
                new = ArtistGenreRecord(
                    name=str(name), raw_tags=raw, buckets=buckets,
                    ambiguous=True, n_candidates=rec.n_candidates)
                best[norm] = (len(raw), new)
    return {norm: rec for norm, (_, rec) in best.items()}


def lookup_artist_genres(
    names: Sequence[str],
    table=None,
    index: Optional[Dict[str, ArtistGenreRecord]] = None,
) -> Dict[str, ArtistGenreRecord]:
    """Resolve artist names -> :class:`ArtistGenreRecord`, keyed by the *input*
    name.  Provide either a prebuilt ``index`` or the raw ``table``.  Unmatched
    names are omitted."""
    if index is None:
        if table is None:
            raise ValueError("provide either table or index")
        index = build_artist_genre_index(table, names=names)
    out: Dict[str, ArtistGenreRecord] = {}
    for n in names:
        rec = index.get(normalize_artist_name(n))
        if rec is not None:
            out[n] = rec
    return out
