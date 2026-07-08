import { useEffect, useState } from "react";
import {
  api, type HeldRow, type Overview, type ScenarioGrid, type TrustAblation,
} from "../api";
import { useTheme } from "../theme";
import { ADVERSARIAL, HONEST } from "../palette";
import { GroupedBars, Heatmap, LineChart } from "./charts";

type Metric = "r_precision" | "ndcg" | "clicks";
const METRIC_LABEL: Record<Metric, string> = {
  r_precision: "R-precision", ndcg: "NDCG", clicks: "Clicks",
};

export function Results() {
  const { mode } = useTheme();
  const [ov, setOv] = useState<Overview | null>(null);
  const [held, setHeld] = useState<HeldRow[]>([]);
  const [trust, setTrust] = useState<TrustAblation | null>(null);
  const [dataset, setDataset] = useState<"real" | "synthetic">("real");
  const [metric, setMetric] = useState<Metric>("r_precision");
  const [grid, setGrid] = useState<ScenarioGrid | null>(null);

  useEffect(() => {
    api.overview().then(setOv);
    api.held().then((d) => setHeld(d.rows));
    api.trust().then(setTrust);
  }, []);
  useEffect(() => {
    api.scenarios(dataset, metric).then(setGrid);
  }, [dataset, metric]);

  // union of models for the synthetic-vs-real headline bars
  const bars = ov ? buildBars(ov) : [];

  return (
    <div className="page reveal">
      <div className="page-head">
        <div className="eyebrow">Results Explorer</div>
        <h1 className="page-title">What held when the map met the territory.</h1>
        <p className="page-lede">
          Every number here is committed under <span className="mono">results/</span>. The headline:
          on the <strong>real 1M-playlist MPD</strong>, plain item-CF wins — the two-stage reranker
          that dominated synthetic data did not transfer.
        </p>
      </div>

      {/* headline stats — the real-MPD story */}
      <div className="stat-row" style={{ marginBottom: 20 }}>
        <div className="stat">
          <div className="label">Real MPD · winner</div>
          <div className="num">Item-CF</div>
          <div className="sub">0.148 R-precision on 6,000 held-out real playlists</div>
        </div>
        <div className="stat">
          <div className="label">Reranker transfer</div>
          <div className="num" style={{ color: "var(--didnt)" }}>Didn’t hold</div>
          <div className="sub">hybrid 0.131 &lt; item-CF 0.148; routing net-neutral</div>
        </div>
        <div className="stat">
          <div className="label">Cold-start rescue</div>
          <div className="num" style={{ color: "var(--held)" }}>Held</div>
          <div className="sub">title model owns <span className="mono">title_only</span> (0.076 vs 0.044)</div>
        </div>
        <div className="stat">
          <div className="label">Scale</div>
          <div className="num">66.3M</div>
          <div className="sub">interactions · 2.26M unique tracks streamed from the zip</div>
        </div>
      </div>

      <div className="stack">
        <section className="card card-pad">
          <div className="card-title">Overall R-precision — synthetic vs real</div>
          {ov && (
            <GroupedBars
              rows={bars}
              series={[
                { name: "Synthetic", hint: "20k playlists" },
                { name: "Real MPD", hint: "150k subsample" },
              ]}
            />
          )}
          <p className="hint" style={{ marginTop: 6 }}>
            The tiers reshuffle: the synthetic hybrid’s lead evaporates and item-CF rises to the top on real data.
          </p>
        </section>

        <section className="card card-pad">
          <div className="card-title">
            Per-scenario {METRIC_LABEL[metric]}
            <span style={{ display: "flex", gap: 8 }}>
              <span className="seg" role="group" aria-label="Dataset">
                <button aria-pressed={dataset === "real"} onClick={() => setDataset("real")}>Real</button>
                <button aria-pressed={dataset === "synthetic"} onClick={() => setDataset("synthetic")}>Synthetic</button>
              </span>
              <span className="seg" role="group" aria-label="Metric">
                {(["r_precision", "ndcg", "clicks"] as Metric[]).map((m) => (
                  <button key={m} aria-pressed={metric === m} onClick={() => setMetric(m)}>
                    {METRIC_LABEL[m]}
                  </button>
                ))}
              </span>
            </span>
          </div>
          {grid && (
            <Heatmap rows={grid.models} cols={grid.scenarios} matrix={grid.matrix}
                     metricLabel={METRIC_LABEL[metric]} />
          )}
          <p className="hint" style={{ marginTop: 6 }}>
            Darker = higher. {metric === "clicks" ? "For clicks, darker means more refresh clicks — lower is better." : "Read across a row to see where a model earns its keep."}
          </p>
        </section>

        <div className="grid-2">
          <section className="card card-pad">
            <div className="card-title">Taste-Engine trust curve (ablation)</div>
            {trust && (
              <LineChart
                xLabel="trust (stated weight)"
                yLabel="R-precision"
                yMax={0.4}
                xTicks={[0, 0.25, 0.5, 0.75, 1]}
                lines={[
                  { name: "honest", color: HONEST[mode], points: trust.honest.map((p) => ({ x: p.trust, y: p.r_precision })) },
                  { name: "wrong prefs", color: ADVERSARIAL[mode], points: trust.adversarial.map((p) => ({ x: p.trust, y: p.r_precision })) },
                ]}
              />
            )}
            <p className="hint" style={{ marginTop: 6 }}>
              With honest stated prefs, trust barely matters; with wrong prefs, believing them
              (high trust) collapses R-precision from 0.33 to 0.02.
            </p>
          </section>

          <section className="card card-pad">
            <div className="card-title">The taste axes are explanation, not ranking</div>
            <p style={{ fontSize: 14, color: "var(--ink-2)", marginBottom: 10 }}>
              Pure axis-matching scores only <b>0.122</b> R-precision — below the popularity floor.
              Dropping any single interpretable axis moves R-precision by at most <b>±0.003</b>.
            </p>
            <p style={{ fontSize: 14, color: "var(--ink-2)" }}>
              The axes earn their keep as <b>flavour clusters</b> (adjusted Rand index peaks at
              <b> 0.98</b> for k=3) and as the per-recommendation <b>WHY</b> — the accuracy comes
              from the co-occurrence term it shares with item-CF.
            </p>
          </section>
        </div>

        <section className="card card-pad">
          <div className="card-title">Synthetic conclusions vs the real run</div>
          <div style={{ overflowX: "auto" }}>
            <table className="held">
              <thead>
                <tr>
                  <th>Conclusion</th><th>Synthetic</th><th>Real</th><th>Verdict</th>
                </tr>
              </thead>
              <tbody>
                {held.map((r, i) => (
                  <tr key={i}>
                    <td>{r.conclusion}</td>
                    <td className="small">{r.synthetic}</td>
                    <td className="small">{r.real}</td>
                    <td><Verdict v={r.verdict} /></td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </section>
      </div>
    </div>
  );
}

function Verdict({ v }: { v: string }) {
  const cls = v === "held" ? "held" : v === "partial" ? "partial" : "didnt";
  const label = v === "held" ? "held" : v === "partial" ? "partial" : "didn’t hold";
  return <span className={`verdict ${cls}`} role="status">{label}</span>;
}

function buildBars(ov: Overview) {
  const bySynth = new Map(ov.synthetic.map((r) => [r.model, r.r_precision]));
  // real has the fullest model set; keep its order (best-on-real first)
  return ov.real.map((r) => ({
    label: r.label,
    values: [bySynth.get(r.model) ?? null, r.r_precision],
  }));
}
