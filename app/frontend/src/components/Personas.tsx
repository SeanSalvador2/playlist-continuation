import { useEffect, useState } from "react";
import {
  api, type Config, type PersonaDetail, type TrustCurve,
} from "../api";
import { useDebounced } from "../hooks";
import { useTheme } from "../theme";
import { ADVERSARIAL, HONEST } from "../palette";
import { WeightBars } from "./WeightBars";
import { LineChart } from "./charts";

export function Personas({ config }: { config: Config }) {
  const { mode } = useTheme();
  const [id, setId] = useState(config.personas[0]?.id ?? "");
  const [adversarial, setAdversarial] = useState(false);
  const [trust, setTrust] = useState(0.4);
  const [detail, setDetail] = useState<PersonaDetail | null>(null);
  const [curve, setCurve] = useState<TrustCurve | null>(null);

  const dTrust = useDebounced(trust, 140);

  useEffect(() => {
    if (!id) return;
    let alive = true;
    api.persona(id, dTrust, adversarial).then((d) => alive && setDetail(d));
    return () => { alive = false; };
  }, [id, dTrust, adversarial]);

  useEffect(() => {
    if (!id) return;
    let alive = true;
    api.personaCurve(id, adversarial).then((c) => alive && setCurve(c));
    return () => { alive = false; };
  }, [id, adversarial]);

  const selected = config.personas.find((p) => p.id === id);
  const fit = detail?.fit ?? 0;
  const fitColor = fit >= 0.66 ? "var(--held)" : fit >= 0.34 ? "var(--partial)" : "var(--didnt)";

  return (
    <div className="page reveal">
      <div className="page-head">
        <div className="eyebrow">Personas</div>
        <h1 className="page-title">Pre-built listeners, and a stress test for trust.</h1>
        <p className="page-lede">
          Load a persona to see the taste its tracks reveal (learned) against what it claims
          to want (stated). Then break it: feed a persona the <em>wrong</em> stated preferences
          and use the trust dial to watch recommendations collapse — and recover.
        </p>
      </div>

      <div className="persona-grid" role="group" aria-label="Choose a persona" style={{ marginBottom: 22 }}>
        {config.personas.map((p) => (
          <button key={p.id} className="persona-card" aria-pressed={p.id === id}
                  onClick={() => { setId(p.id); setAdversarial(false); setTrust(0.4); }}>
            <div className="persona-name">{p.name}</div>
            <div className="persona-blurb">{p.blurb}</div>
            <span className={`tag ${p.adversarial ? "adv" : ""}`}>
              {p.adversarial ? "adversarial demo ready" : p.archetype}
            </span>
          </button>
        ))}
      </div>

      {detail && (
        <div className="stack">
          <section className="card card-pad">
            <div className="card-title">Stated vs learned — {selected?.name}</div>
            <div className="compare">
              <div>
                <h3 style={{ fontSize: 15, marginBottom: 4 }}>What it says (stated)</h3>
                <p className="hint" style={{ marginBottom: 12 }}>
                  {adversarial ? "deliberately wrong preferences" : "the honest self-description"}
                </p>
                <WeightBars weights={detail.profile.weights.stated} color="var(--accent)" />
              </div>
              <div>
                <h3 style={{ fontSize: 15, marginBottom: 4 }}>What its tracks reveal (learned)</h3>
                <p className="hint" style={{ marginBottom: 12 }}>
                  logistic-regression coefficients on the axes
                </p>
                <WeightBars weights={detail.profile.weights.learned}
                            color={mode === "dark" ? "#199e70" : "#1baf7a"} />
              </div>
            </div>
          </section>

          <div className="grid-2">
            <section className="card card-pad rescue">
              <div className="card-title">The trust stress test</div>
              <div className="seg" role="group" aria-label="Stated preference mode" style={{ marginBottom: 6 }}>
                <button aria-pressed={!adversarial} onClick={() => setAdversarial(false)}>Honest</button>
                <button aria-pressed={adversarial} onClick={() => setAdversarial(true)}>Wrong prefs</button>
              </div>

              <div className="fit-meter">
                <div style={{ display: "flex", alignItems: "baseline", gap: 10 }}>
                  <span className="fit-num" style={{ color: fitColor }}>{Math.round(fit * 100)}%</span>
                  <span className="hint">of the top-20 recs are truly “{detail.archetype}”</span>
                </div>
                <div className="bar" style={{ marginTop: 8 }}>
                  <span style={{ width: `${fit * 100}%`, background: fitColor }} />
                </div>
              </div>

              <label style={{ display: "block" }}>
                <div className="slider-head">
                  <span className="slider-label">Trust in stated prefs</span>
                  <span className="slider-val">{Math.round(trust * 100)}%</span>
                </div>
                <input type="range" min={0} max={1} step={0.05} value={trust}
                       onChange={(e) => setTrust(parseFloat(e.target.value))}
                       aria-label={`Trust ${Math.round(trust * 100)} percent`} />
                <div className="slider-poles"><span>learned</span><span>stated</span></div>
              </label>

              {adversarial && (
                <div className="callout">
                  {fit >= 0.66
                    ? "Low trust ignores the bad prefs and leans on the tracks — the recommendations stay on-taste."
                    : "High trust believes the wrong prefs and the recommendations drift off-taste. Drag trust back down to rescue them."}
                </div>
              )}
            </section>

            <section className="card card-pad">
              <div className="card-title">Fit vs trust {adversarial ? "· adversarial" : "· honest"}</div>
              {curve && (
                <LineChart
                  xLabel="trust (stated weight)"
                  yLabel="on-taste share"
                  yMax={1.05}
                  xTicks={[0, 0.2, 0.4, 0.6, 0.8, 1]}
                  lines={[{
                    name: adversarial ? "wrong prefs" : "honest prefs",
                    color: adversarial ? ADVERSARIAL[mode] : HONEST[mode],
                    points: curve.curve.map((c) => ({ x: c.trust, y: c.fit })),
                  }]}
                />
              )}
              <p className="hint" style={{ marginTop: 8 }}>
                Mirrors the taste-engine ablation: honest prefs are flat across trust; wrong
                prefs collapse as trust rises.
              </p>
            </section>
          </div>

          <section className="card card-pad">
            <div className="card-title">Recommendations at {Math.round(trust * 100)}% trust</div>
            <div className="recs" style={{ display: "grid", gridTemplateColumns: "repeat(auto-fill, minmax(220px, 1fr))" }}>
              {detail.recommendations.items.slice(0, 12).map((r) => (
                <div className="rec" key={r.uri} style={{ ["--fl" as string]: "var(--hairline-2)" }}>
                  <div className="rec-top">
                    <span className="rec-rank">{String(r.rank).padStart(2, "0")}</span>
                    <span className="rec-name">{r.track_name}</span>
                  </div>
                  <div className="rec-artist">{r.artist_name}</div>
                  <span className="tag" style={{
                    marginTop: 8,
                    background: r.archetype === detail.archetype ? "var(--held-soft)" : "var(--didnt-soft)",
                    color: r.archetype === detail.archetype ? "var(--held)" : "var(--didnt)",
                  }}>{r.genre}</span>
                </div>
              ))}
            </div>
          </section>
        </div>
      )}
    </div>
  );
}
