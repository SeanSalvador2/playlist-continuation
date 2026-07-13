"""The fact-checked narrative — "the story of your taste" (Phase 4).

This module is the point of the phase.  A taste story is exactly the kind of
copy an LLM will happily *fabricate* — a plausible-sounding "you moved from indie
to metal in the spring" that no number backs.  So the story here is built the
opposite way round: **facts first, prose second, and every rendered claim audited
against the facts before the story is allowed to leave the module.**

THREE LAYERS
------------
1. :func:`gather_facts` computes a typed :class:`StoryFacts` tree holding *every*
   number and name the story may use — era facts (names, dates, top artists/tracks,
   mean axes, discovery), boundary facts (the FDR-surviving shifts across each
   change, via the Phase-2 :func:`~playlistcont.analytics.stats.significant_shifts`
   layer over equal-length flanking windows), and overall arc facts (first-vs-last
   era, total span, the biggest single shift, the most-persistent artist, the
   discovery peak).  All display numbers are stored at display precision so a claim
   can equal a fact exactly.
2. :func:`render_story` turns those facts into ordered :class:`Slide` objects in
   the ``name_flavor`` house style — a title slide, one slide per era, one
   transition slide per boundary (with a provisional caveat when flagged), and a
   closing arc slide.  Deterministic given the facts.
3. :func:`verify_story` is the **fact-check harness**.  Every slide carries
   ``claims: list[Claim]`` where a :class:`Claim` is ``(text_fragment, fact_path,
   value)``.  Verification checks, for every slide: (a) each claim's ``value``
   equals the fact at ``fact_path``; (b) every number in the rendered text is
   traceable to a claim (each numeric span must sit inside a claim fragment that
   *displays* that value under the documented formatting rules); (c) every proper
   noun (artist / track / era name — detected as a mid-sentence Title-case run)
   sits inside a claim fragment.  Any violation raises
   :class:`StoryVerificationError`.  ``render_story`` runs the harness **before
   returning** — a story that fails its own audit never ships.

LLM POLISH HOOK (design only)
-----------------------------
``render_story(facts, mode="polished", polish_fn=None)`` lets a caller pass a
``polish_fn(facts, template_slides) -> [{"title","body"}]`` that *rewords* the
prose.  The rewritten slides are **re-verified by the same harness**: the claims
must survive the rewording — numbers and names must still be present and
traceable.  Polish changes prose, never facts.  If the polished draft fails
verification (e.g. it invented or altered a number), we **fall back to the
template** story rather than ship an unverified narrative.  The shipped default is
a no-op; no API client is implemented here.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from typing import Callable, List, Optional, Sequence, Union

from ..analytics import stats as stats_mod
from ..history.schema import ListeningHistory
from ..history.store import HistoryStore
from .detectors import DetectedChange, recommended_detector
from .eras import Era, build_eras

_MONTHS = ["", "January", "February", "March", "April", "May", "June",
           "July", "August", "September", "October", "November", "December"]

# axis / metric display labels and effect symbols (kept parallel to stats.py)
_LABEL = {
    "tempo": "Tempo", "energy": "Energy", "valence": "Valence",
    "acousticness": "Acousticness", "lyrical_depth": "Lyrical depth",
    "genre_mix": "Genre mix", "skip_rate": "Skip rate",
    "discovery_rate": "Discovery rate",
}
_SYM = {"cohen_d": "d", "rank_biserial": "r", "cramers_v": "V", "prop_diff": "Δ"}
_AXIS_METRICS = {"tempo", "energy", "valence", "acousticness", "lyrical_depth"}
_BEHAVIORAL = {"skip_rate", "discovery_rate"}


def _month_year(d: date) -> str:
    return f"{_MONTHS[d.month]} {d.year}"


# ===========================================================================
# Fact layer
# ===========================================================================
@dataclass
class Claim:
    """A single audited assertion: a text fragment, the fact it comes from, its value."""

    text_fragment: str
    fact_path: str
    value: object

    def to_payload(self) -> dict:
        return {"text": self.text_fragment, "fact_path": self.fact_path, "value": self.value}


@dataclass
class StoryFacts:
    """The typed fact tree every claim in the story must resolve against.

    ``data`` is a plain nested ``dict`` / ``list`` structure addressable by a dotted
    ``fact_path`` (list indices are integers), so it is directly JSON-ready and its
    leaves are exactly the numbers/names the templates are allowed to print.
    """

    data: dict
    eras: List[Era] = field(default_factory=list)

    _MISSING = object()

    def resolve(self, path: str):
        """Return the fact at a dotted ``path`` (``"eras.0.name"``), or raise ``KeyError``."""
        node = self.data
        for part in path.split("."):
            if isinstance(node, list):
                idx = int(part)
                if idx < 0 or idx >= len(node):
                    raise KeyError(path)
                node = node[idx]
            elif isinstance(node, dict):
                if part not in node:
                    raise KeyError(path)
                node = node[part]
            else:
                raise KeyError(path)
        return node

    def to_payload(self) -> dict:
        return self.data


def _flank_windows(eras: List[Era], i: int, cap: int = 56):
    """Equal-length windows on each side of the boundary opening era ``i+1``."""
    d = eras[i + 1].start
    before = (d - eras[i].start).days
    after = (eras[i + 1].end - d).days + 1
    flank = max(7, min(before, after, cap))
    a_end = d - timedelta(days=1)
    a_start = a_end - timedelta(days=flank - 1)
    b_start = d
    b_end = b_start + timedelta(days=flank - 1)
    return a_start, a_end, b_start, b_end


def gather_facts(
    history_or_store: Union[ListeningHistory, HistoryStore],
    eras: List[Era],
    detections: Optional[Sequence[DetectedChange]] = None,
) -> StoryFacts:
    """Compute every number and name the story may use (see the module docstring)."""
    store = (history_or_store if isinstance(history_or_store, HistoryStore)
             else HistoryStore.from_history(history_or_store))

    n_eras = len(eras)
    total_plays = sum(e.n_plays for e in eras)
    start_date = eras[0].start if eras else None
    end_date = eras[-1].end if eras else None
    span_days = ((end_date - start_date).days + 1) if eras else 0

    era_facts: List[dict] = []
    for e in eras:
        era_facts.append({
            "number": e.index + 1,
            "name": e.name,
            "start": e.start.isoformat(),
            "end": e.end.isoformat(),
            "duration_days": e.duration_days,
            "plays": e.n_plays,
            "plays_per_day": round(e.plays_per_day, 1),
            "discovery_rate": round(e.discovery_rate, 2),
            "provisional": e.provisional,
            "top_artists": [a["name"] for a in e.top_artists],
            "top_tracks": [t["name"] for t in e.top_tracks],
            "mean_axes": {k: round(v, 2) for k, v in e.mean_axes.items()},
        })

    # ---- boundary facts: FDR-surviving shifts across each change ---------- #
    boundary_facts: List[dict] = []
    for i in range(n_eras - 1):
        a_s, a_e, b_s, b_e = _flank_windows(eras, i)
        payload = stats_mod.significant_shifts(store, a_s, a_e, b_s, b_e)
        shifts = []
        for s in payload["shifts"]:
            metric = s["metric"]
            if metric not in _AXIS_METRICS and metric not in _BEHAVIORAL \
                    and metric != "genre_mix":
                continue
            shifts.append({
                "metric": metric,
                "label": _LABEL.get(metric, metric),
                "sym": _SYM.get(s["effect_name"], "e"),
                "effect": round(float(s["effect"]), 2),
                "q": round(float(s["q"]), 3),
                "direction": s["direction"],
                "behavioral": metric in _BEHAVIORAL,
                "sentence": s["sentence"],
            })
        boundary_facts.append({
            "index": i,
            "date": eras[i + 1].start.isoformat(),
            "from_era": eras[i].name,
            "to_era": eras[i + 1].name,
            "provisional": eras[i + 1].provisional,
            "shifts": shifts,
        })

    # ---- arc facts ------------------------------------------------------- #
    arc: dict = {}
    if n_eras:
        arc["first_era"] = eras[0].name
        arc["last_era"] = eras[-1].name
        arc["span_days"] = span_days
        arc["n_eras"] = n_eras

        # biggest single shift across all boundaries
        best = None
        for b in boundary_facts:
            for s in b["shifts"]:
                if best is None or abs(s["effect"]) > abs(best["effect"]):
                    best = {"label": s["label"], "sym": s["sym"], "effect": s["effect"],
                            "q": s["q"], "to_era": b["to_era"]}
        arc["biggest_shift"] = best

        # most-persistent artist: present in the most eras (tie -> more total plays, then name)
        presence: dict = {}
        plays_tot: dict = {}
        for e in eras:
            for a in e.top_artists:
                presence[a["name"]] = presence.get(a["name"], 0) + 1
                plays_tot[a["name"]] = plays_tot.get(a["name"], 0) + a["plays"]
        if presence:
            best_artist = sorted(
                presence, key=lambda nm: (-presence[nm], -plays_tot[nm], nm))[0]
            arc["most_persistent_artist"] = {
                "name": best_artist, "era_count": presence[best_artist]}

        # discovery peak
        peak = max(eras, key=lambda e: e.discovery_rate)
        arc["discovery_peak"] = {"era": peak.name, "rate": round(peak.discovery_rate, 2)}

    data = {
        "meta": {
            "total_plays": total_plays,
            "span_days": span_days,
            "n_eras": n_eras,
            "start": start_date.isoformat() if start_date else None,
            "end": end_date.isoformat() if end_date else None,
        },
        "eras": era_facts,
        "boundaries": boundary_facts,
        "arc": arc,
    }
    return StoryFacts(data=data, eras=eras)


# ===========================================================================
# Slides & rendering
# ===========================================================================
@dataclass
class Slide:
    """One narrated slide: kind, title, body lines, its claims, and a UI payload."""

    kind: str
    title: str
    body: List[str]
    claims: List[Claim] = field(default_factory=list)
    payload: dict = field(default_factory=dict)

    @property
    def text(self) -> str:
        return self.title + "\n" + "\n".join(self.body)

    def to_payload(self) -> dict:
        return {
            "kind": self.kind, "title": self.title, "body": self.body,
            "claims": [c.to_payload() for c in self.claims], "payload": self.payload,
        }


@dataclass
class Story:
    slides: List[Slide]
    mode: str = "template"

    def to_payload(self) -> dict:
        return {"mode": self.mode, "slides": [s.to_payload() for s in self.slides]}


def _c(fragment: str, path: str, facts: StoryFacts) -> Claim:
    """Build a claim by reading its value straight from the facts (so a==fact holds)."""
    return Claim(text_fragment=fragment, fact_path=path, value=facts.resolve(path))


def _iso_my(iso: str) -> str:
    return _month_year(date.fromisoformat(iso))


def _title_slide(facts: StoryFacts) -> Slide:
    m = facts.data["meta"]
    plays, span, n = m["total_plays"], m["span_days"], m["n_eras"]
    start_my, end_my = _iso_my(m["start"]), _iso_my(m["end"])
    body = [
        f"{plays} plays across {span} days, in {n} chapters.",
        f"From {start_my} to {end_my}.",
    ]
    claims = [
        _c(str(plays), "meta.total_plays", facts),
        _c(str(span), "meta.span_days", facts),
        _c(str(n), "meta.n_eras", facts),
        _c(start_my, "meta.start", facts),
        _c(end_my, "meta.end", facts),
    ]
    return Slide("title", "The story of your taste", body, claims,
                 payload={"n_eras": n})


def _era_slide(facts: StoryFacts, i: int) -> Slide:
    e = facts.data["eras"][i]
    p = f"eras.{i}"
    num, name = e["number"], e["name"]
    start_my, end_my = _iso_my(e["start"]), _iso_my(e["end"])
    title = f"Chapter {num}: {name}"
    claims = [_c(str(num), f"{p}.number", facts), _c(name, f"{p}.name", facts)]

    body = [f"{start_my} to {end_my} · {e['duration_days']} days."]
    claims += [_c(start_my, f"{p}.start", facts), _c(end_my, f"{p}.end", facts),
               _c(str(e["duration_days"]), f"{p}.duration_days", facts)]

    artists = e["top_artists"][:3]
    if artists:
        joined = (", ".join(artists[:-1]) + f" and {artists[-1]}"
                  if len(artists) > 1 else artists[0])
        body.append(f"Defined by {joined}.")
        for j, a in enumerate(artists):
            claims.append(_c(a, f"{p}.top_artists.{j}", facts))

    ax = e["mean_axes"]
    body.append(
        f"Its sound sat at energy {ax['energy']:.2f}, valence {ax['valence']:.2f} "
        f"and acousticness {ax['acousticness']:.2f}.")
    claims += [
        _c(f"{ax['energy']:.2f}", f"{p}.mean_axes.energy", facts),
        _c(f"{ax['valence']:.2f}", f"{p}.mean_axes.valence", facts),
        _c(f"{ax['acousticness']:.2f}", f"{p}.mean_axes.acousticness", facts),
    ]

    if e["provisional"]:
        body.append(
            "This chapter is provisional: it opens on a December bump we cannot yet "
            "tell apart from a lasting change. Check back in January.")

    return Slide("era", title, body, claims,
                 payload={"index": i, "provisional": e["provisional"],
                          "top_artists": e["top_artists"], "mean_axes": ax})


def _shift_verb(direction) -> str:
    if direction == "up":
        return "rose"
    if direction == "down":
        return "eased"
    return "shifted"


def _transition_slide(facts: StoryFacts, i: int) -> Slide:
    b = facts.data["boundaries"][i]
    p = f"boundaries.{i}"
    to_name = b["to_era"]
    date_my = _iso_my(b["date"])
    title = f"The turn into {to_name}"
    claims = [_c(to_name, f"{p}.to_era", facts)]

    body = [f"Around {date_my}, your listening changed."]
    claims.append(_c(date_my, f"{p}.date", facts))

    shifts = b["shifts"][:4]
    if shifts:
        for j, s in enumerate(shifts):
            eff = f"{s['effect']:+.2f}"
            q = f"{s['q']:.3f}"
            body.append(f"{s['label']} {_shift_verb(s['direction'])} "
                        f"({s['sym']}={eff}, q={q}).")
            claims.append(_c(eff, f"{p}.shifts.{j}.effect", facts))
            claims.append(_c(q, f"{p}.shifts.{j}.q", facts))
    else:
        body.append("The change was in the overall mix rather than any single axis.")

    if b["provisional"]:
        body.append(
            "This turn is provisional: a December bump with no January after it looks "
            "the same as a lasting change.")

    return Slide("transition", title, body, claims,
                 payload={"index": i, "provisional": b["provisional"], "shifts": shifts})


def _arc_slide(facts: StoryFacts) -> Slide:
    a = facts.data["arc"]
    body = [f"You went from {a['first_era']} to {a['last_era']}."]
    claims = [_c(a["first_era"], "arc.first_era", facts),
              _c(a["last_era"], "arc.last_era", facts)]

    if a.get("biggest_shift"):
        bs = a["biggest_shift"]
        eff = f"{bs['effect']:+.2f}"
        q = f"{bs['q']:.3f}"
        body.append(
            f"The biggest single move was {bs['label'].lower()} at {bs['sym']}={eff} "
            f"(q={q}) on the turn into {bs['to_era']}.")
        claims += [_c(eff, "arc.biggest_shift.effect", facts),
                   _c(q, "arc.biggest_shift.q", facts),
                   _c(bs["to_era"], "arc.biggest_shift.to_era", facts)]

    if a.get("most_persistent_artist"):
        mp = a["most_persistent_artist"]
        body.append(
            f"{mp['name']} stayed with you longest, across {mp['era_count']} of "
            f"{a['n_eras']} chapters.")
        claims += [_c(mp["name"], "arc.most_persistent_artist.name", facts),
                   _c(str(mp["era_count"]), "arc.most_persistent_artist.era_count", facts),
                   _c(str(a["n_eras"]), "arc.n_eras", facts)]

    if a.get("discovery_peak"):
        dp = a["discovery_peak"]
        body.append(f"Discovery peaked in your {dp['era']} era at {dp['rate']:.2f}.")
        claims += [_c(dp["era"], "arc.discovery_peak.era", facts),
                   _c(f"{dp['rate']:.2f}", "arc.discovery_peak.rate", facts)]

    return Slide("arc", "How far you travelled", body, claims, payload={})


def _template_slides(facts: StoryFacts) -> List[Slide]:
    slides = [_title_slide(facts)]
    n = facts.data["meta"]["n_eras"]
    for i in range(n):
        slides.append(_era_slide(facts, i))
        if i < len(facts.data["boundaries"]):
            slides.append(_transition_slide(facts, i))
    if n:
        slides.append(_arc_slide(facts))
    return slides


def render_story(
    facts: StoryFacts,
    mode: str = "template",
    polish_fn: Optional[Callable[[StoryFacts, List[Slide]], List[dict]]] = None,
) -> Story:
    """Render ordered, fact-checked slides from ``facts``.

    ``mode="template"`` renders the deterministic house-style narrative and verifies
    it before returning (a template that fails its own audit raises — that is a bug).
    ``mode="polished"`` additionally applies ``polish_fn`` (a no-op by default), which
    may *reword* the slides; the reworded slides are re-verified by the same harness
    and, on any failure, the function **falls back to the verified template story**.
    """
    slides = _template_slides(facts)
    template = Story(slides=slides, mode="template")
    verify_story(template, facts)          # never ship an unaudited template

    if mode != "polished":
        return template

    fn = polish_fn or _noop_polish
    rewritten = fn(facts, slides)
    polished = [
        Slide(kind=s.kind, title=r.get("title", s.title), body=list(r.get("body", s.body)),
              claims=s.claims, payload=s.payload)
        for s, r in zip(slides, rewritten)
    ]
    polished_story = Story(slides=polished, mode="polished")
    try:
        verify_story(polished_story, facts)
    except StoryVerificationError:
        return template                    # polish never overrides the facts
    return polished_story


def _noop_polish(facts: StoryFacts, slides: List[Slide]) -> List[dict]:
    return [{"title": s.title, "body": list(s.body)} for s in slides]


# ===========================================================================
# The fact-check harness
# ===========================================================================
class StoryVerificationError(Exception):
    """Raised when a rendered story does not survive its own fact-check."""

    def __init__(self, violations: List[str]):
        self.violations = violations
        super().__init__("; ".join(violations))


_NUM_RE = re.compile(r"[-+]?\d+(?:\.\d+)?%?")
# a Title-case run whose continuation words also start Title-case or with a digit
# (artist / track names like "Rock Artist 4-0", or a "Month YYYY" date).
_NAME_RE = re.compile(r"[A-Z][A-Za-z]*(?:[ -][A-Z0-9][A-Za-z0-9-]*)*")

# capitalised words the templates legitimately use that are NOT proper nouns
_ALLOWED_CAPS = {
    "Energy", "Tempo", "Valence", "Acousticness", "Lyrical", "Genre", "Skip",
    "Discovery", "Chapter", "V",
    "January", "February", "March", "April", "May", "June", "July", "August",
    "September", "October", "November", "December",
}


def _surface_forms(value) -> set:
    """Documented string renderings of a fact value (see verify_story)."""
    forms: set = set()
    if isinstance(value, bool):
        return forms
    if isinstance(value, int):
        forms.add(str(value))
    elif isinstance(value, float):
        for nd in (0, 1, 2, 3):
            forms.add(f"{value:.{nd}f}")
            forms.add(f"{value:+.{nd}f}")
        if float(value).is_integer():
            forms.add(str(int(value)))
    elif isinstance(value, str):
        forms.add(value)
        try:
            forms.add(_month_year(date.fromisoformat(value)))
        except ValueError:
            pass
    return forms


def _fragment_shows_value(fragment: str, value) -> bool:
    forms = _surface_forms(value)
    if not forms:                          # non-numeric / non-string (e.g. bool) values
        return True
    return any(f and f in fragment for f in forms)


def _fragment_spans(text: str, fragment: str) -> List[tuple]:
    spans, start = [], 0
    while True:
        idx = text.find(fragment, start)
        if idx < 0:
            return spans
        spans.append((idx, idx + len(fragment)))
        start = idx + 1


def _covered(span: tuple, frag_spans: List[tuple]) -> bool:
    return any(fs <= span[0] and span[1] <= fe for fs, fe in frag_spans)


def _is_sentence_initial(text: str, start: int) -> bool:
    j = start - 1
    while j >= 0 and text[j] == " ":
        j -= 1
    return j < 0 or text[j] in ".:!?\n"


def verify_story(story: Story, facts: StoryFacts) -> None:
    """Audit every slide against the facts; raise :class:`StoryVerificationError` on any miss.

    For each slide: (a) each claim's value must equal the fact at its ``fact_path``
    and its fragment must appear in the text and *display* the value (documented
    formatting rules in :func:`_surface_forms`); (b) every numeric span in the text
    must sit inside some claim fragment; (c) every mid-sentence Title-case run
    (artist/track/era name) must sit inside some claim fragment.  Returns ``None``
    when the whole story is clean.
    """
    violations: List[str] = []
    for si, slide in enumerate(story.slides):
        text = slide.text
        frag_spans: List[tuple] = []
        for claim in slide.claims:
            tag = f"slide[{si}]:{claim.fact_path}"
            # (a) claim value must equal the fact it points at
            try:
                fv = facts.resolve(claim.fact_path)
            except (KeyError, ValueError, IndexError):
                violations.append(f"{tag}: fact_path does not resolve")
                continue
            if fv != claim.value:
                violations.append(f"{tag}: claim value {claim.value!r} != fact {fv!r}")
            # fragment must appear in the rendered text
            spans = _fragment_spans(text, claim.text_fragment)
            if not spans:
                violations.append(
                    f"{tag}: fragment {claim.text_fragment!r} not in rendered text")
                continue
            # fragment must actually display the value
            if not _fragment_shows_value(claim.text_fragment, claim.value):
                violations.append(
                    f"{tag}: fragment {claim.text_fragment!r} does not show value {claim.value!r}")
            frag_spans.extend(spans)

        # (b) every number in the text is inside a claim fragment
        for mnum in _NUM_RE.finditer(text):
            if not _covered(mnum.span(), frag_spans):
                violations.append(
                    f"slide[{si}]: untraceable number {mnum.group()!r} in text")

        # (c) every proper-noun run is inside a claim fragment.  A run that *starts*
        # a sentence begins with an ordinary capitalised word ("From", "Chapter",
        # "Around", or a name itself); trim that leading word, which is not a proper
        # noun on its own, and audit whatever name/date follows it.
        for mname in _NAME_RE.finditer(text):
            s0, e0 = mname.span()
            token = mname.group()
            if _is_sentence_initial(text, s0):
                head, _, rest = token.partition(" ")
                if not rest:
                    continue                # a lone sentence-initial capitalised word
                s0 += len(head) + 1
                token = rest
            if token in _ALLOWED_CAPS:
                continue
            if not _covered((s0, e0), frag_spans):
                violations.append(
                    f"slide[{si}]: untraceable proper noun {token!r} in text")

    if violations:
        raise StoryVerificationError(violations)
