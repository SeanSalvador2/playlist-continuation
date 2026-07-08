import { useMemo, useState } from "react";
import type { AxisMeta, CatalogPoint, Cluster, Rec } from "../api";
import { useTheme } from "../theme";
import { chartInk, SERIES } from "../palette";

const SCALARS = ["tempo", "energy", "valence", "acousticness", "lyrical_depth"] as const;

interface Tip { x: number; y: number; title: string; sub: string }

export function TasteMap({
  catalog, clusters, recs, axesMeta,
}: {
  catalog: CatalogPoint[];
  clusters: Cluster[];
  recs: Rec[];
  axesMeta: AxisMeta[];
}) {
  const { mode } = useTheme();
  const ink = chartInk(mode);
  const palette = SERIES[mode];
  const [xk, setXk] = useState("valence");
  const [yk, setYk] = useState("energy");
  const [tip, setTip] = useState<Tip | null>(null);

  const W = 640, H = 460, pad = 40;
  const px = (v: number) => pad + v * (W - 2 * pad);
  const py = (v: number) => H - pad - v * (H - 2 * pad);

  const meta = (k: string) => axesMeta.find((a) => a.key === k)!;

  // assign each rec (and cluster) a colour by nearest flavour centroid in 5D
  const clusterColor = (i: number) => palette[i % palette.length];
  const nearestCluster = useMemo(() => {
    return (axes: Record<string, number>) => {
      if (!clusters.length) return -1;
      let best = 0, bd = Infinity;
      clusters.forEach((c, i) => {
        let d = 0;
        for (const s of SCALARS) d += (c.axes[s] - (axes[s] ?? 0)) ** 2;
        if (d < bd) { bd = d; best = i; }
      });
      return best;
    };
  }, [clusters]);

  return (
    <div className="map-wrap">
      <div className="map-controls">
        <label>
          <span className="mono" style={{ fontSize: 11, color: "var(--muted)" }}>X</span>
          <select value={xk} onChange={(e) => setXk(e.target.value)} aria-label="Map horizontal axis">
            {SCALARS.map((s) => <option key={s} value={s}>{meta(s).label}</option>)}
          </select>
        </label>
        <label>
          <span className="mono" style={{ fontSize: 11, color: "var(--muted)" }}>Y</span>
          <select value={yk} onChange={(e) => setYk(e.target.value)} aria-label="Map vertical axis">
            {SCALARS.map((s) => <option key={s} value={s}>{meta(s).label}</option>)}
          </select>
        </label>
        <span className="hint">Terrain = the catalogue · rings = your flavour territories · pins = recommendations</span>
      </div>

      <div style={{ position: "relative" }}>
        <svg viewBox={`0 0 ${W} ${H}`} width="100%" className="map-svg" role="img"
             aria-label={`Taste map: catalogue plotted by ${meta(xk).label} and ${meta(yk).label}, with flavour territories and recommendation pins`}>
          {/* grid */}
          {[0.25, 0.5, 0.75].map((t) => (
            <g key={t}>
              <line x1={px(t)} x2={px(t)} y1={pad} y2={H - pad} stroke={ink.grid} strokeWidth={1} />
              <line x1={pad} x2={W - pad} y1={py(t)} y2={py(t)} stroke={ink.grid} strokeWidth={1} />
            </g>
          ))}
          {/* frame */}
          <rect x={pad} y={pad} width={W - 2 * pad} height={H - 2 * pad} fill="none" stroke={ink.axis} strokeWidth={1} />
          {/* axis pole labels */}
          <text x={pad} y={H - pad + 22} fontSize={10.5} fill={ink.muted} fontFamily="var(--font-mono)">{meta(xk).low}</text>
          <text x={W - pad} y={H - pad + 22} textAnchor="end" fontSize={10.5} fill={ink.muted} fontFamily="var(--font-mono)">{meta(xk).high}</text>
          <text x={pad} y={H - pad + 34} fontSize={11.5} fill={ink.secondary} fontWeight={600}>{meta(xk).label} →</text>
          <text x={pad - 8} y={H - pad} fontSize={10.5} fill={ink.muted} fontFamily="var(--font-mono)" transform={`rotate(-90 ${pad - 8} ${H - pad})`}>{meta(yk).low}</text>
          <text x={pad - 8} y={pad} fontSize={10.5} fill={ink.muted} fontFamily="var(--font-mono)" transform={`rotate(-90 ${pad - 8} ${pad})`} textAnchor="end">{meta(yk).high}</text>
          <text x={pad - 24} y={pad + (H - 2 * pad) / 2} fontSize={11.5} fill={ink.secondary} fontWeight={600} transform={`rotate(-90 ${pad - 24} ${pad + (H - 2 * pad) / 2})`} textAnchor="middle">{meta(yk).label} →</text>

          {/* terrain: neutral catalogue points */}
          {catalog.map((c, i) => (
            <circle key={i} cx={px(c[xk as keyof CatalogPoint] as number)}
                    cy={py(c[yk as keyof CatalogPoint] as number)} r={2.2}
                    fill={ink.muted} fillOpacity={0.28} />
          ))}

          {/* flavour territories: concentric contour rings */}
          {clusters.map((c, i) => {
            const cx = px(c.axes[xk] ?? 0), cy = py(c.axes[yk] ?? 0);
            const col = clusterColor(i);
            const base = 26 + c.share * 64;
            return (
              <g key={i} className="fade-in">
                {[1, 0.66, 0.36].map((r, ri) => (
                  <circle key={ri} cx={cx} cy={cy} r={base * r} fill={col}
                          fillOpacity={ri === 2 ? 0.14 : 0.05} stroke={col}
                          strokeOpacity={0.5} strokeWidth={ri === 0 ? 1.6 : 1}
                          strokeDasharray={ri === 0 ? "0" : "3 3"} />
                ))}
                <circle cx={cx} cy={cy} r={3} fill={col} stroke={ink.surface} strokeWidth={1.4} />
                <text x={cx} y={cy - base - 6} textAnchor="middle" fontSize={11}
                      fontWeight={600} fill={ink.primary} fontFamily="var(--font-display)">
                  {shortFlavor(c.name)}
                </text>
              </g>
            );
          })}

          {/* recommendation pins */}
          {recs.slice(0, 24).map((r) => {
            const ci = nearestCluster(r.axes);
            const col = ci < 0 ? palette[0] : clusterColor(ci);
            const cx = px(r.axes[xk] ?? 0), cy = py(r.axes[yk] ?? 0);
            return (
              <g key={r.uri}
                 onMouseMove={(e) => setTip({ x: e.clientX, y: e.clientY, title: r.track_name, sub: `${r.artist_name} · ${r.genre}` })}
                 onMouseLeave={() => setTip(null)}>
                <path d={`M${cx},${cy - 5} L${cx + 4.5},${cy} L${cx},${cy + 5} L${cx - 4.5},${cy} Z`}
                      fill={col} stroke={ink.surface} strokeWidth={1.2} />
              </g>
            );
          })}
        </svg>
        {tip && (
          <div className="tooltip" style={{ left: tip.x + 14, top: tip.y + 14 }} role="status">
            <strong>{tip.title}</strong>
            <div className="mono">{tip.sub}</div>
          </div>
        )}
      </div>

      {clusters.length > 0 && (
        <div className="map-legend">
          {clusters.map((c, i) => (
            <span className="key" key={i}>
              <span className="swatch" style={{ background: clusterColor(i) }} />
              {shortFlavor(c.name)}
            </span>
          ))}
        </div>
      )}
    </div>
  );
}

function shortFlavor(name: string): string {
  // keep the map labels legible: first few descriptive tokens
  const parts = name.split(",").map((s) => s.trim());
  const genre = parts[parts.length - 1].split(" ").slice(-1)[0];
  return `${parts[0]} ${genre}`;
}
