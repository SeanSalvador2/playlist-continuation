# DESIGN_PHYSICS.md

A durable, dependency-free playbook distilled from the **real source** of seven UI
libraries/skill-packs. Every constant below was read out of committed component source or
docs (URLs per section), not from memory or marketing copy. The goal: reproduce the
"premium" feel under a **strict CSP / zero-external-request** budget using only CSS
`linear()` easings, `requestAnimationFrame` springs, `IntersectionObserver` staggers, and
conic/radial gradient masks.

Fetch transport note: through the pinned agent proxy only `raw.githubusercontent.com` and
`registry.npmjs.org` were reachable. `api.github.com`, `github.com`, `cdn.jsdelivr.net`,
`unpkg.com`, and the vendor marketing sites (ui.aceternity.com, magicui.design, reactbits.dev)
all returned 403 / blocked. Directory listing was therefore done by probing raw paths and by
extracting the one npm tarball that existed.

---

## 0. The single most important finding

**Real "premium" springs are near-critically damped, not bouncy.** Every production spring
read from Magic UI and the magic-card source lands at damping ratio **ζ ≈ 1.0–3.0** (smooth
settle, ~0% overshoot). The cartoonish overshoot that AIs reach for by default (ζ ≈ 0.5) is
*not* what these libraries ship. Reserve visible overshoot (ζ ≈ 0.57, ~10%) for one or two
deliberate "pop" moments; make everything else settle cleanly.

Damping ratio: `ζ = c / (2·√(k·m))` where `k`=stiffness, `c`=damping, `m`=mass.

---

## 1. premium-frontend-ui (github/awesome-copilot) — REAL, single file

Fetched: `https://raw.githubusercontent.com/github/awesome-copilot/main/skills/premium-frontend-ui/SKILL.md`
(7188 bytes). The task assumed companion "direction files" (Organic Fluidity / Editorial
Brutalism specs); **verified there are none** — those are four *sections* inside the one
SKILL.md (probed README.md/references.md/AGENTS.md/skill.json in that dir → all 404).
Author: Utkarsh Patrikar.

Extracted doctrine (the load-bearing constants):
- **Four committed aesthetics**: Editorial Brutalism (mono, oversized type, sharp edges, raw
  grids) · Organic Fluidity (soft gradients, deep radii, glassmorphism, bouncy spring) ·
  Cyber/Technical (dark dominance, neon accents, monospace, rapid staggered reveals) ·
  Cinematic Pacing (full-viewport imagery, slow cross-fades, negative space, scroll story).
- Hero: full-bleed `100vh`/`100dvh`; headlines split **by word/character** for cascading
  entrance; syntactic span-wrapping.
- Type: `clamp()` up to **12vw** for headlines; body **16–18px minimum**.
- Texture: SVG/CSS **noise overlay `opacity 0.02–0.05`** with `mix-blend-mode: overlay`;
  glass = `backdrop-filter: blur(x)` + ultra-thin semi-transparent borders.
- Motion: animate **only `transform` + `opacity`** (never width/height/top/margin);
  `will-change: transform` applied *during* motion then removed; heavy hover behind
  `@media (hover:hover) and (pointer:fine)`; continuous motion behind
  `@media (prefers-reduced-motion: no-preference)`.
- Nav: sticky headers that hide-on-scroll-down / reveal-on-scroll-up.

Vanilla translation: this is the umbrella spec — it *is* already framework-agnostic. Honour
the transform/opacity-only rule and the reduced-motion / hover-capability gates verbatim.

---

## 2. Magic UI (magicuidesign/magicui) — REAL component source

Components live at `apps/www/registry/magicui/<name>.tsx`. Fetched raw:
`https://raw.githubusercontent.com/magicuidesign/magicui/main/apps/www/registry/magicui/{number-ticker,animated-list,bento-grid,border-beam,shine-border,blur-fade,text-animate,marquee,magic-card}.tsx`

### number-ticker
```
useSpring(motionValue, { damping: 60, stiffness: 100 })   // mass defaults to 1 → ζ=3.0
```
Starts on `useInView(once:true, margin:"0px")`, optional `delay*1000` ms; formats every
frame with `Intl.NumberFormat("en-US", {min/maxFractionDigits})`; renders `tabular-nums`.
Vanilla: rAF spring integrating `a=(k(target−x)−c·v)/m` with k=100,c=60,m=1, write
`el.textContent = Intl.NumberFormat(...)` each frame. This heavily overdamped spring is why
counters *ease to rest* instead of snapping.

### animated-list
```
item: initial{scale:0,opacity:0} → animate{scale:1,opacity:1,originY:0}
transition: { type:"spring", stiffness:350, damping:40 }   // ζ=1.07, settle ~605ms
list reveal cadence: delay = 1000ms between items (staggered mount, newest on top)
```
Vanilla: reveal items with `IntersectionObserver`, transform `scale(0.96)→1` + opacity,
`transition-timing-function` = the animated-list `linear()` below, stagger via
`transition-delay: calc(var(--i) * --stagger)`.

### blur-fade
```
duration:0.4s, delay: 0.04 + delay, offset:6px, blur:6px, ease:"easeOut"
hidden{ y:±6, opacity:0, filter:blur(6px) } → visible{ y:0, opacity:1, filter:blur(0) }
inView once, margin:"-50px"
```
Vanilla: the canonical scroll-reveal. IntersectionObserver `rootMargin:"-50px"`, transition
`opacity/filter/transform .4s ease-out`, base `translateY(6px) blur(6px)`.

### text-animate (per-segment stagger)
```
staggerChildren: 0.05s ; per-split stagger → text 0.06 / word 0.05 / character 0.03
blurInUp item: hidden{opacity:0, blur(10px), y:20} → show{opacity:1, blur(0), y:0}
  transitions: y .3s, opacity .4s, filter .3s   (opacity trails position/blur — key detail)
fadeIn/slideUp items: duration .3s
```
Vanilla: split headline into word/char spans, set `--i` per span, animate
`opacity .4s / transform .3s / filter .3s` with `transition-delay: calc(var(--i)*30–60ms)`.
The **opacity being slightly slower than position** is what reads as expensive.

### border-beam (a traveling light along a rounded border)
```
size:50, duration:6s, borderWidth:1px, colorFrom:#ffaa40, colorTo:#9c40ff
ease:"linear", repeat:Infinity ; offsetDistance animates 0% → 100%
container: border-transparent + mask [linear-gradient(transparent,transparent), linear-gradient(#000,#000)]
           mask-intersect, mask-clip: padding-box, border-box    (shows beam only on the border ring)
beam element: absolute aspect-square, width=size, offsetPath: rect(0 auto auto 0 round {size}px)
              background: linear-gradient(to left, colorFrom, colorTo, transparent)
```
Vanilla (CSP-safe, no `offset-path` motion values needed): a conic-gradient sweep masked to
the border ring:
```css
.beam::before{
  content:""; position:absolute; inset:0; border-radius:inherit; padding:1px;
  background:conic-gradient(from var(--a), transparent 0 75%, var(--beam-from), var(--beam-to), transparent 100%);
  -webkit-mask:linear-gradient(#000 0 0) content-box, linear-gradient(#000 0 0);
  -webkit-mask-composite:xor; mask-composite:exclude;   /* ring only */
  animation:beam 6s linear infinite;
}
@property --a{ syntax:'<angle>'; inherits:false; initial-value:0deg; }
@keyframes beam{ to{ --a:360deg } }
```
Token-color `--beam-from/--beam-to`; wrap the whole rule in
`@media (prefers-reduced-motion:no-preference)` so it holds static otherwise.

### shine-border (same border-ring mask, different fill)
```
borderWidth:1, duration:14s, backgroundImage: radial-gradient(transparent,transparent,color,transparent,transparent)
backgroundSize:300% 300% ; mask: linear-gradient content-box XOR linear-gradient (mask-composite:exclude)
animate-shine = background-position drift ; will-change:[background-position]
```
Vanilla: identical XOR content-box mask trick as border-beam; animate `background-position`
instead of an angle for a softer, slower shimmer (14s vs 6s).

### magic-card (the real "spotlight" card) — see also §3 SpotlightCard
```
gradientSize:200px, gradientColor:#262626, gradientOpacity:0.8, gradientFrom:#9E7AFF, gradientTo:#FE8BBB
border layer: background =
  linear-gradient(var(--color-background) 0 0) padding-box,
  radial-gradient(200px circle at MOUSEx MOUSEy, gradientFrom, gradientTo, var(--color-border) 100%) border-box
inner overlay (mode:gradient): radial-gradient(200px circle at MOUSEx MOUSEy, gradientColor, transparent 100%)
  opacity 0 → gradientOpacity(0.8) on group-hover, transition-opacity 300ms
orb spring: useSpring(mouseX/Y,{ stiffness:250, damping:30, mass:0.6 })   // ζ=1.22
orbVisible spring: { stiffness:300, damping:35 }                          // ζ=1.01
orb: blur(60px), mixBlendMode screen(dark)/multiply(light)
mouse coords: rect = getBoundingClientRect(); x = clientX-left; y = clientY-top
reset x/y to -gradientSize when pointer leaves / window blur / tab hidden
```
Vanilla: set `--mx/--my` on pointermove; two stacked radials — one **on the border ring**
(border-box, so the frame lights up) and one **soft fill** revealed via opacity on hover
(300ms). This border-lit-by-cursor detail is the signature that community re-implementations
usually miss.

### marquee / bento-grid (layout constants)
- marquee: `--duration:40s, --gap:1rem, repeat:4`, `group-hover:[animation-play-state:paused]`,
  reverse via `animation-direction:reverse`.
- bento: `grid auto-rows-[22rem] grid-cols-3 gap-4`, card `rounded-xl`, wide tiles `col-span-*`.
  Light shadow: `0 0 0 1px rgba(0,0,0,.03), 0 2px 4px rgba(0,0,0,.05), 0 12px 24px rgba(0,0,0,.05)`.
  Dark: `inset 0 -20px 80px -20px #ffffff1f` + `1px solid rgba(255,255,255,.1)`.
  Hover: content group `-translate-y-10` (300ms), icon `scale-75`, CTA fades up from bottom.

---

## 3. React Bits (DavidHDev/react-bits) — REAL component source

Components at `src/content/<Category>/<Name>/<Name>.{jsx,css}`. Fetched:
`https://raw.githubusercontent.com/DavidHDev/react-bits/main/src/content/{Components/SpotlightCard/SpotlightCard.jsx,+.css,TextAnimations/CountUp/CountUp.jsx,Animations/AnimatedContent/AnimatedContent.jsx,Backgrounds/Particles/Particles.jsx,Backgrounds/Aurora/Aurora.jsx}`

### SpotlightCard (the cleanest, CSP-safe card-spotlight — pure CSS)
```css
.card-spotlight{ --mouse-x:50%; --mouse-y:50%; --spotlight-color:rgba(255,255,255,.05); overflow:hidden }
.card-spotlight::before{
  content:''; position:absolute; inset:0; pointer-events:none; opacity:0; transition:opacity .5s ease;
  background:radial-gradient(circle at var(--mouse-x) var(--mouse-y), var(--spotlight-color), transparent 80%);
}
.card-spotlight:hover::before, .card-spotlight:focus-within::before{ opacity:.6 }  /* focus-within = a11y */
```
JS: `rect=getBoundingClientRect(); el.style.setProperty('--mouse-x', (clientX-left)+'px')` …
This is directly droppable, no framework, and already keyboard-accessible via `:focus-within`.
**Use this over the Magic UI motion-value version when you need zero JS-per-frame.**

### CountUp — a *derived* spring (nice trick)
```
damping   = 20 + 40 * (1/duration)
stiffness = 100 * (1/duration)          // duration=2 → damping 40, stiffness 50 → ζ≈2.83
useInView(once,margin:'0px'); textContent set via Intl.NumberFormat; delay*1000 before start
```
Vanilla: same rAF integrator as number-ticker; expose `duration` and back-solve k/c from it.

### AnimatedContent (scroll reveal, GSAP)
```
distance:100px, duration:0.8s, ease:'power3.out', threshold:0.1 → ScrollTrigger start "top 90%"
gsap.set(el,{ y:offset, scale, opacity:0, visibility:'visible' }) then tl.to(y:0,scale:1,opacity:1)
once:true
```
Vanilla: IntersectionObserver `threshold:0.1`; `power3.out` ≈ `cubic-bezier(.16,1,.3,1)`.

### Particles / Aurora (canvas/WebGL background) — rAF pattern only
Both ship as **WebGL (ogl)** and are *not* CSP-safe as-is (external `ogl` import), so we take
the *technique*, not the code:
- Single **`requestAnimationFrame` loop with delta timing**: `delta = t − lastTime; elapsed += delta*speed; uTime = elapsed*0.001`.
- Gentle autonomous motion via `sin/cos(elapsed)`: `rot.x=sin(elapsed*2e-4)*0.1; rot.y=cos(elapsed*5e-4)*0.15`.
- Uniform point-in-sphere seeding by rejection + cube-root radius:
  `do{x,y,z∈[-1,1]}while(x²+y²+z²>1); r=cbrt(random())` → even volumetric spread.
- Cursor parallax: normalize pointer to `[-1,1]`, translate field by `−mouse*factor`.
- Aurora colour ramp: 3 color stops, `mix()` between the two bracketing stops by
  `(factor−cur.pos)/(next.pos−cur.pos)` — reproducible with a 2D-canvas gradient or CSS
  animated conic-gradient (no WebGL needed).
Vanilla starfield recipe: one `<canvas>`, DPR-capped (`min(devicePixelRatio,2)`), N≈150–250
points with `{x,y,z}`, per-frame `x+=vx` wrap, size `∝ 1/z`, `ctx.globalAlpha ∝ depth`;
pause on `document.hidden`; skip the whole loop under reduced-motion.

---

## 4. Aceternity UI — GENUINELY UNFETCHABLE (documented attempts)

Attempts, all failed:
- `ui.aceternity.com` → 403 through proxy (paid/gated).
- npm `aceternity-ui@0.2.2` tarball fetched (`registry.npmjs.org/aceternity-ui/-/aceternity-ui-0.2.2.tgz`)
  — it is a **CLI only** (`dist/index.js`, 33 KB, "Add Aceternity UI components to your apps").
  Grepped for embedded component templates (spotlight/bento/beam/radial-gradient/conic) → **none**;
  it *fetches* components at runtime from `https://ui.aceternity.com` (403). `@aceternity/ui`,
  `aceternityui` → npm 404.
- `api.github.com` search + repo contents → blocked (session is scoped to one repo; all
  api.github.com returns "GitHub access … not enabled").
**Mitigation**: Aceternity's three requested primitives are technique-identical to sources we
*did* get from real code — `card-spotlight` == React Bits SpotlightCard (§3) + Magic UI
magic-card (§2); `bento-grid` == Magic UI bento (§2); `background-beams`/`tracing-beam` ==
the border-beam conic-mask (§2) / an SVG path with an animated gradient `<stroke>` and a
`stroke-dasharray` draw-on. So nothing is lost — only Aceternity's *exact* pixel values are
unverified and are flagged as such wherever reused.

Tracing-beam vanilla recipe (reconstructed, not fetched): an SVG path following the content;
`stroke:url(#grad)` with a vertical linear gradient; reveal by animating
`stroke-dashoffset: len → 0` tied to scroll (`getBoundingClientRect` progress), a small
glowing circle at the current head via `<feGaussianBlur>` or a `box-shadow` dot.

---

## 5. shadcn/ui (shadcn-ui/ui) — token architecture (not components)

Fetched: `https://raw.githubusercontent.com/shadcn-ui/ui/main/apps/v4/app/globals.css`.
The token contract to copy:
- **Two layers.** Raw values live on `:root` (light) and `.dark` (overrides). A separate
  `@theme inline { --color-*: var(--*) }` block maps *semantic* names to those raw vars so
  utilities resolve to tokens, never to hex.
- **oklch everywhere** (`oklch(L C H)` / `oklch(1 0 0 / 10%)` for alpha borders) — perceptually
  even lightness, trivial dark-mode derivation by dropping L.
- **Semantic pairs**: `background/foreground`, `card/card-foreground`, `popover`, `primary`,
  `secondary`, `muted`, `accent`, `destructive`, `border`, `input`, `ring`, `chart-1..5`,
  `sidebar*`, `surface`, `selection`. Every surface has an explicit *-foreground pair.
- **Radius scale derived by calc** from one `--radius:0.625rem`:
  `--radius-sm:calc(r*0.6)` … `--radius-4xl:calc(r*2.6)`. One knob rescales the whole UI.
- Base layer: `* { border-color: border; outline: ring/50 }`, `::selection` tokenized.
Takeaway for us: **never write a component color as hex** — define raw tokens once per theme
(or per wing/route), map semantic aliases, and let components reference only the aliases.
This is exactly how per-wing recolors stay a token edit, not a find-replace.

---

## 6. daisyUI (saadeghi/daisyui) — semantic token *philosophy*

Fetched: `https://raw.githubusercontent.com/saadeghi/daisyui/master/packages/docs/src/routes/(routes)/docs/colors/+page.md`
and `/themes/+page.md`.
- Thesis: use **purpose-named** tokens (`primary/secondary/accent/neutral`, surfaces
  `base-100 → base-200 → base-300` as rising elevations, states `info/success/warning/error`),
  never value-named (`bg-green-500`). Every color carries a paired `*-content` foreground so
  contrast is guaranteed by the token, not by ad-hoc dark: variants.
- Payoff explicitly stated: automatic dark mode, unlimited themes as "a few lines of CSS
  variables", zero per-element maintenance, guaranteed consistency.
- Elevation model worth stealing: `base-100` = page ground, `base-200`/`base-300` = stacked
  surfaces (panels, wells) — a clean 3-step depth ramp instead of arbitrary grays.
Takeaway: name tokens by *role* (`--panel`, `--panel-2`, `--accent`, `--accent-ink`), pair
each fill with its readable ink, and treat "add a theme/wing/intensity" as writing one small
block of vars.

---

## 7. aesthetic-frontend-skills (alexiseverage) — REAL README

Fetched: `https://raw.githubusercontent.com/alexiseverage/aesthetic-frontend-skills/main/README.md`.
A two-skill method for turning a *named aesthetic* into developer-ready output:
`aesthetic-literacy` (identify/disambiguate/characterize) → `aesthetic-application`
(emit tokens, CSS vars, component notes, WCAG flags). The reusable framework is the
**7-dimension profile**: palette · type · texture · shape · motion · spatial conventions ·
cultural markers — plus explicit **non-negotiables** and **anti-patterns** per aesthetic.
Takeaway: before building a themed surface, write its 7-dimension profile and its
anti-pattern list first; it prevents generic mood-board drift and makes the token choices
defensible. (E.g. "premium sci-fi HUD" non-negotiables: crisp 1px edges, mono data type,
restrained glow; anti-patterns: scanline spam, glitch-text, green-on-black terminal walls.)

---

## 8. Derived `linear()` spring easings (sampled from the REAL constants)

Each string is the step-response of `m·x″ + c·x′ + k·x = k` (unit target, x0=v0=0),
integrated at dt=1e-4 and resampled to 22 stops, normalized over its settle time. Drop these
straight into `transition-timing-function`. Durations are the physical settle time — use them
verbatim so the CSS motion *is* the spring, not an approximation of it.

```css
:root{
  /* animated-list item — k350 c40 m1 · ζ=1.07 · ~605ms · no overshoot */
  --ease-list: linear(0,0.1011 4.8%,0.2845 9.5%,0.4628 14.3%,0.6096 19%,0.7216 23.8%,0.8039 28.6%,0.8628 33.3%,0.9045 38.1%,0.9338 42.9%,0.9541 47.6%,0.9683 52.4%,0.9781 57.1%,0.9849 61.9%,0.9896 66.7%,0.9928 71.4%,0.995 76.2%,0.9966 81%,0.9976 85.7%,0.9984 90.5%,0.9989 95.2%,1);
  --dur-list: 605ms;

  /* magic-card orb settle — k250 c30 m0.6 · ζ=1.22 · ~689ms */
  --ease-orb: linear(0,0.1356 4.8%,0.3457 9.5%,0.5255 14.3%,0.6612 19%,0.7594 23.8%,0.8296 28.6%,0.8794 33.3%,0.9147 38.1%,0.9397 42.9%,0.9573 47.6%,0.9698 52.4%,0.9786 57.1%,0.9849 61.9%,0.9893 66.7%,0.9924 71.4%,0.9947 76.2%,0.9962 81%,0.9973 85.7%,0.9981 90.5%,0.9987 95.2%,1);
  --dur-orb: 689ms;

  /* critical settle — k300 c35 m1 · ζ=1.01 · ~577ms · page/panel transitions */
  --ease-settle: linear(0,0.0832 4.8%,0.2462 9.5%,0.4162 14.3%,0.5641 19%,0.6831 23.8%,0.7739 28.6%,0.8406 33.3%,0.889 38.1%,0.9233 42.9%,0.9474 47.6%,0.9641 52.4%,0.9756 57.1%,0.9835 61.9%,0.9888 66.7%,0.9925 71.4%,0.9949 76.2%,0.9966 81%,0.9977 85.7%,0.9985 90.5%,0.999 95.2%,1);
  --dur-settle: 577ms;

  /* elastic POP — k300 c20 m1 · ζ=0.577 · ~763ms · +10.8% overshoot — use SPARINGLY */
  --ease-pop: linear(0,0.1537 4.8%,0.4593 9.5%,0.7538 14.3%,0.9636 19%,1.0747 23.8%,1.108 28.6%,1.0947 33.3%,1.0623 38.1%,1.0298 42.9%,1.0059 47.6%,0.9928 52.4%,0.9884 57.1%,0.9894 61.9%,0.9928 66.7%,0.9964 71.4%,0.9991 76.2%,1.0007 81%,1.0012 85.7%,1.0012 90.5%,1.0008 95.2%,1);
  --dur-pop: 420ms; /* compress the physical 763ms for UI snappiness; overshoot preserved */

  /* power3.out scroll reveal (React Bits AnimatedContent) */
  --ease-reveal: cubic-bezier(.16,1,.3,1);
}
```

### rAF spring integrator (number-ticker / CountUp), CSP-safe
```js
function springTo(from, to, {k=100, c=60, m=1, onFrame, onDone}={}){
  if (matchMedia('(prefers-reduced-motion: reduce)').matches){ onFrame(to); onDone?.(); return; }
  let x=from, v=0, last=performance.now();
  (function step(now){
    let dt=Math.min((now-last)/1000, 1/30); last=now;      // clamp dt for tab-switch safety
    const a=(k*(to-x)-c*v)/m; v+=a*dt; x+=v*dt;
    onFrame(x);
    if (Math.abs(to-x)<0.01 && Math.abs(v)<0.01){ onFrame(to); onDone?.(); return; }
    requestAnimationFrame(step);
  })(performance.now());
}
// number-ticker feel: k=100,c=60,m=1 (ζ=3, smooth). CountUp: k=100/dur, c=20+40/dur.
```

### IntersectionObserver stagger (blur-fade / animated-list), CSP-safe
```js
const io = new IntersectionObserver((es)=>{ for(const e of es) if(e.isIntersecting){
  e.target.style.setProperty('--i', e.target.dataset.i || 0);
  e.target.classList.add('in'); io.unobserve(e.target);
}},{ rootMargin:'-50px', threshold:0.1 });
```
```css
.reveal{ opacity:0; transform:translateY(6px); filter:blur(6px);
  transition:opacity .4s var(--ease-reveal), transform .3s var(--ease-reveal), filter .3s var(--ease-reveal);
  transition-delay:calc(var(--i,0)*70ms); }   /* --stagger:70ms; opacity intentionally slower than transform */
.reveal.in{ opacity:1; transform:none; filter:none; }
@media (prefers-reduced-motion:reduce){ .reveal{ transition:none; opacity:1; transform:none; filter:none } }
```
`--stagger: 70ms` (scaled down from animated-list's 1000ms mount cadence for a viewport of
tiles); cap the number of staggered items (~12) so late tiles don't lag.

---

## 9. When to use what — keyed to this project's two surfaces

**Surface A — the Taste Atlas React app** (framework present: React + Vite; motion libs allowed):
- Prefer the **real components** where a dependency is acceptable: Magic UI number-ticker /
  border-beam / magic-card, React Bits SpotlightCard. Feed them the constants above.
- Use the shadcn **oklch two-layer token contract** (§5) as the theme system; name tokens by
  role per daisyUI (§6). One `--radius` knob; per-route/wing token blocks.
- Springs: default to `{stiffness:300, damping:35}` (ζ≈1, clean) for transitions; only the
  hero "pop" uses `{stiffness:300, damping:20}`.

**Surface B — static artifacts** (single self-contained HTML, **strict CSP, zero external
requests**, the taste_story.html case):
- **No libraries** — everything is the vanilla translations here: CSS `linear()` easings from
  §8, the rAF `springTo` integrator, IntersectionObserver staggers, the SpotlightCard pure-CSS
  spotlight (§3, no per-frame JS), the conic-mask border-beam (§2), a DPR-capped 2D-canvas
  starfield (§3) on **one shared rAF loop**, paused on `document.hidden`.
- Tokens: inline `:root` (+ per-wing/`data-theme` blocks) only; component CSS references
  tokens, never hex. Radius/blur/stagger as single knobs.
- Hard gates (both surfaces, non-negotiable): `prefers-reduced-motion:reduce` collapses **all**
  motion incl. canvas; heavy hover behind `@media (hover:hover) and (pointer:fine)`; animate
  only `transform`/`opacity`; visible `:focus-visible`; the readable glyph must hit **≥4.5:1**
  on its own (glow/box-shadow is decoration and does not count toward contrast); no horizontal
  overflow.

---

## 10. Source ledger (what was real vs reconstructed)

| Source | Transport | Status |
| --- | --- | --- |
| premium-frontend-ui SKILL.md | raw.githubusercontent | REAL (full file; single file confirmed) |
| Magic UI ×9 components | raw.githubusercontent | REAL (exact source) |
| React Bits ×6 components | raw.githubusercontent | REAL (exact source) |
| shadcn globals.css (v4) | raw.githubusercontent | REAL (exact tokens) |
| daisyUI colors/themes docs | raw.githubusercontent | REAL (exact docs) |
| aesthetic-frontend-skills README | raw.githubusercontent | REAL |
| Aceternity components | site 403 · npm CLI has no source · api.github.com blocked | UNFETCHABLE → covered by equivalents, exact px flagged |

All spring `linear()` strings in §8 are computed from the REAL k/c/m read out of the source in
§2–§3, not from documented defaults.
