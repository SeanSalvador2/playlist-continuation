import { useEffect, useState } from "react";
import {
  api, type EraCard, type Journey as JourneyData, type StorySlide,
} from "../api";
import { useTheme } from "../theme";
import { SERIES } from "../palette";
import { TrajectoryChart, type TrajPt } from "./charts";

// ==========================================================================
//  Journey — the taste trajectory, named eras, and the fact-checked story
// ==========================================================================
export function Journey() {
  const [data, setData] = useState<JourneyData | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    api.historyJourney().then(setData).catch((e) => setError(String(e)));
  }, []);

  if (error) {
    return <div className="page"><div className="card card-pad">Could not reach the API: {error}</div></div>;
  }
  if (!data) {
    return <div className="page"><div className="center-empty">Tracing your taste through time…</div></div>;
  }
  if (data.error) {
    return <div className="page"><div className="card card-pad">Could not build the journey: {data.error}</div></div>;
  }

  return (
    <div className="page reveal">
      <div className="page-head">
        <div className="eyebrow">Journey</div>
        <h1 className="page-title">The story of your taste.</h1>
        <p className="page-lede">
          Your listening projected into two dimensions, segmented into named eras at the
          moments your taste actually changed, and narrated into a story where every
          number and name is computed from your plays and machine-verified.
        </p>
      </div>

      {data.is_synthetic && (
        <div className="tag adv" role="note" style={{ marginBottom: 16 }}>
          ◆ synthetic demo listener · {data.detections.length} detected changes
          {data.planted ? ` · ${data.planted.length} planted landmarks` : ""}
        </div>
      )}

      <div className="stack">
        <TrajectoryPanel data={data} />
        <ErasRow eras={data.eras} />
        <StorySlideshow data={data} />
      </div>
    </div>
  );
}

// ---- trajectory panel ----------------------------------------------------- //
function eraOf(dateISO: string, detections: string[]): number {
  let e = 0;
  for (const d of detections) if (dateISO >= d) e += 1;
  return e;
}

function TrajectoryPanel({ data }: { data: JourneyData }) {
  const { mode } = useTheme();
  const [showPlanted, setShowPlanted] = useState(false);
  const colors = SERIES[mode];
  const tr = data.trajectory;
  const detDates = data.detections.map((d) => d.date);

  const n = tr.points.length;
  const points: TrajPt[] = tr.points.map((p, i) => ({
    x: p.coords[0] ?? 0,
    y: p.coords[1] ?? 0,
    era: eraOf(p.date, detDates),
    date: p.date,
    t: n > 1 ? i / (n - 1) : 1,
  }));

  // place a marker at the trajectory point nearest each change date
  const nearest = (iso: string) => {
    let best = 0, bestDelta = Infinity;
    tr.points.forEach((p, i) => {
      const delta = Math.abs(Date.parse(p.date) - Date.parse(iso));
      if (delta < bestDelta) { bestDelta = delta; best = i; }
    });
    return best;
  };
  const markers = data.detections.map((d) => {
    const i = nearest(d.date);
    const era = data.eras.find((e) => e.opening_change === d.date);
    return {
      x: points[i]?.x ?? 0, y: points[i]?.y ?? 0, date: d.date,
      label: era?.name ?? "", provisional: !!era?.provisional,
    };
  });
  const planted = showPlanted && data.planted
    ? data.planted.filter((p) => p.kind === "abrupt" || p.kind === "drift_start").map((p) => {
        const i = nearest(p.date);
        return { x: points[i]?.x ?? 0, y: points[i]?.y ?? 0, date: p.date };
      })
    : [];

  const pc1 = tr.components[0], pc2 = tr.components[1];

  return (
    <section className="card card-pad" aria-label="Taste trajectory">
      <div className="card-title">
        Taste trajectory
        <span style={{ display: "flex", gap: 8, alignItems: "center", flexWrap: "wrap" }}>
          <span className="hint mono">{tr.n_windows} weekly windows · {(tr.total_explained * 100).toFixed(0)}% of variance in 2D</span>
          {data.is_synthetic && data.planted && (
            <button className="btn ghost" aria-pressed={showPlanted} onClick={() => setShowPlanted(!showPlanted)}>
              {showPlanted ? "✓ planted markers" : "show planted markers"}
            </button>
          )}
        </span>
      </div>

      {n > 1 ? (
        <>
          <TrajectoryChart
            points={points}
            eraColors={colors}
            xCaption={pc1?.caption ?? ""}
            yCaption={pc2?.caption ?? ""}
            xExplained={pc1?.explained_variance_ratio ?? 0}
            yExplained={pc2?.explained_variance_ratio ?? 0}
            markers={markers}
            planted={planted}
          />
          <p className="hint" style={{ marginTop: 8 }}>
            Each point is a week, coloured by era and shaded from faint (early) to solid (recent);
            the line is the order you actually travelled. Diamonds mark detected changes
            {planted.length ? "; the faint crosses are the planted landmarks (demo diagnostics only)" : ""}.
            <b> PC1</b> ≈ {pc1?.caption}; <b>PC2</b> ≈ {pc2?.caption}.
          </p>
        </>
      ) : (
        <p className="hint">Not enough measurable windows to draw a trajectory.</p>
      )}
    </section>
  );
}

// ---- era cards row -------------------------------------------------------- //
function ErasRow({ eras }: { eras: EraCard[] }) {
  const { mode } = useTheme();
  const colors = SERIES[mode];
  return (
    <section className="card card-pad" aria-label="Eras">
      <div className="card-title">Your eras <span className="hint">named by dominant flavour</span></div>
      <div className="eras-row">
        {eras.map((e) => (
          <article className="era-card" key={e.index} style={{ borderTopColor: colors[e.index % colors.length] }}>
            <div className="era-head">
              <span className="era-num mono">Chapter {e.index + 1}</span>
              {e.provisional && (
                <span className="tag adv era-prov" role="note"
                      title="This era opens on a December bump we cannot yet tell apart from a lasting change — check back in January.">
                  provisional
                </span>
              )}
            </div>
            <h3 className="era-name">{e.name}</h3>
            <div className="hint mono era-dates">{mY(e.start)} – {mY(e.end)} · {e.duration_days}d</div>
            <MiniAxes axes={e.mean_axes} color={colors[e.index % colors.length]} />
            <div className="era-artists">
              {e.top_artists.slice(0, 3).map((a) => (
                <span className="chip" key={a.name}>{a.name}</span>
              ))}
            </div>
            <div className="hint era-foot mono">
              {e.plays_per_day}/day · {(e.discovery_rate * 100).toFixed(0)}% discovery
            </div>
          </article>
        ))}
      </div>
    </section>
  );
}

const MINI_AXES = ["energy", "valence", "acousticness"];
function MiniAxes({ axes, color }: { axes: Record<string, number>; color: string }) {
  return (
    <div className="mini-axes">
      {MINI_AXES.map((k) => (
        <div className="mini-axis" key={k}>
          <span className="mini-k">{k.slice(0, 3)}</span>
          <span className="mini-bar"><span style={{ width: `${Math.round((axes[k] ?? 0) * 100)}%`, background: color }} /></span>
          <span className="mini-v mono">{(axes[k] ?? 0).toFixed(2)}</span>
        </div>
      ))}
    </div>
  );
}

// ---- story slideshow ------------------------------------------------------ //
function StorySlideshow({ data }: { data: JourneyData }) {
  const { mode } = useTheme();
  const slides = data.story.slides;
  const [i, setI] = useState(0);
  const slide = slides[i];
  const colors = SERIES[mode];

  const prev = () => setI((v) => Math.max(0, v - 1));
  const next = () => setI((v) => Math.min(slides.length - 1, v + 1));

  return (
    <section className="card card-pad story-card" aria-label="Story">
      <div className="card-title">
        Story of your taste
        <span className="hint">slide {i + 1} of {slides.length} · {data.story.mode}</span>
      </div>

      <div className={`story-slide story-k-${slide.kind}`} role="group" aria-label={`Slide ${i + 1}`}>
        <div className="story-kind mono">{slide.kind}</div>
        <h2 className="story-title">{slide.title}</h2>
        {slide.body.map((line, k) => (
          <p className="story-line" key={k}>{line}</p>
        ))}
        {slide.kind === "transition" && <TransitionBars slide={slide} color={colors[0]} />}
      </div>

      <div className="story-nav">
        <button className="btn" onClick={prev} disabled={i === 0} aria-label="Previous slide">‹ Prev</button>
        <span className="story-dots" role="group" aria-label="Slide navigation">
          {slides.map((_, k) => (
            <button key={k} className={`story-dot ${k === i ? "on" : ""}`} onClick={() => setI(k)}
                    aria-current={k === i ? "true" : undefined}
                    aria-label={`Go to slide ${k + 1}`} />
          ))}
        </span>
        <button className="btn" onClick={next} disabled={i === slides.length - 1} aria-label="Next slide">Next ›</button>
      </div>

      <p className="coverage-banner block" role="note" style={{ marginTop: 14 }}>
        Every claim in this story is <b>computed from your plays and machine-verified</b>: each
        number and name is traced back to a fact before the slide is allowed to render. Prose can be
        polished by a language model, but the facts can never be.
      </p>
    </section>
  );
}

function TransitionBars({ slide, color }: { slide: StorySlide; color: string }) {
  const shifts = (slide.payload?.shifts as { label: string; effect: number; sym: string }[]) ?? [];
  if (!shifts.length) return null;
  const max = Math.max(...shifts.map((s) => Math.abs(s.effect)), 0.5);
  return (
    <div className="transition-bars">
      {shifts.map((s, k) => (
        <div className="tbar-row" key={k}>
          <span className="tbar-k">{s.label}</span>
          <span className="tbar-track">
            <span className="tbar-fill" style={{ width: `${(Math.abs(s.effect) / max) * 100}%`, background: color }} />
          </span>
          <span className="tbar-v mono">{s.sym}={s.effect >= 0 ? "+" : ""}{s.effect.toFixed(2)}</span>
        </div>
      ))}
    </div>
  );
}

function mY(iso: string): string {
  const d = new Date(`${iso}T00:00:00Z`);
  return d.toLocaleString("en-US", { month: "short", year: "numeric", timeZone: "UTC" });
}
