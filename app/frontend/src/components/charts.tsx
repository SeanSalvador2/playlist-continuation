import { useState } from "react";
import { useTheme } from "../theme";
import { chartInk, seqStep, SERIES } from "../palette";

// ---- shared tooltip -------------------------------------------------------- //
interface Tip { x: number; y: number; title: string; lines: string[] }
function Tooltip({ tip }: { tip: Tip | null }) {
  if (!tip) return null;
  return (
    <div className="tooltip" style={{ left: tip.x + 14, top: tip.y + 14 }} role="status">
      <strong>{tip.title}</strong>
      {tip.lines.map((l, i) => <div key={i} className="mono">{l}</div>)}
    </div>
  );
}

// ==========================================================================
//  Grouped horizontal bars — model comparison (synthetic vs real)
// ==========================================================================
export function GroupedBars({
  rows, series, unit = "", higherBetter = true,
}: {
  rows: { label: string; values: (number | null)[] }[];
  series: { name: string; hint?: string }[];
  unit?: string;
  higherBetter?: boolean;
}) {
  const { mode } = useTheme();
  const ink = chartInk(mode);
  const [tip, setTip] = useState<Tip | null>(null);
  const colors = SERIES[mode];

  const rowH = 46;
  const barGap = 4;
  const barH = (rowH - 14 - (series.length - 1) * barGap) / series.length;
  const padL = 128, padR = 54, padT = 8;
  const W = 640;
  const plotW = W - padL - padR;
  const H = padT + rows.length * rowH + 8;
  const maxV = Math.max(
    ...rows.flatMap((r) => r.values.map((v) => (v == null ? 0 : v))), 0.0001,
  );

  return (
    <figure style={{ margin: 0 }}>
      <div style={{ position: "relative" }}>
        <svg viewBox={`0 0 ${W} ${H}`} width="100%" role="img"
             aria-label={`Grouped bar chart comparing ${series.map((s) => s.name).join(" and ")}`}>
          {[0.25, 0.5, 0.75, 1].map((t) => (
            <line key={t} x1={padL + t * plotW} x2={padL + t * plotW} y1={padT} y2={H - 14}
                  stroke={ink.grid} strokeWidth={1} />
          ))}
          {rows.map((r, ri) => {
            const y0 = padT + ri * rowH;
            return (
              <g key={r.label}>
                <text x={padL - 10} y={y0 + rowH / 2} textAnchor="end" dominantBaseline="middle"
                      fontSize={12.5} fill={ink.primary} fontWeight={500}
                      fontFamily="var(--font-body)">{r.label}</text>
                {r.values.map((v, si) => {
                  const by = y0 + 7 + si * (barH + barGap);
                  if (v == null) {
                    return (
                      <text key={si} x={padL + 4} y={by + barH / 2} dominantBaseline="middle"
                            fontSize={10} fill={ink.muted} fontFamily="var(--font-mono)"
                            fontStyle="italic">n/a</text>
                    );
                  }
                  const w = (v / maxV) * plotW;
                  return (
                    <g key={si}
                       onMouseMove={(e) => setTip({
                         x: e.clientX, y: e.clientY, title: r.label,
                         lines: [`${series[si].name}: ${v.toFixed(3)}${unit}`],
                       })}
                       onMouseLeave={() => setTip(null)}>
                      <rect x={padL} y={by} width={Math.max(w, 1.5)} height={barH}
                            rx={Math.min(4, barH / 2)} fill={colors[si]} />
                      <text x={padL + w + 6} y={by + barH / 2} dominantBaseline="middle"
                            fontSize={10.5} fill={ink.secondary}
                            fontFamily="var(--font-mono)">{v.toFixed(3)}</text>
                    </g>
                  );
                })}
              </g>
            );
          })}
          <line x1={padL} x2={padL} y1={padT} y2={H - 14} stroke={ink.axis} strokeWidth={1} />
        </svg>
        <Tooltip tip={tip} />
      </div>
      <div className="legend">
        {series.map((s, i) => (
          <span className="key" key={s.name}>
            <span className="line" style={{ background: colors[i], height: 10, width: 10, borderRadius: 3 }} />
            {s.name}{s.hint ? <span className="hint"> · {s.hint}</span> : null}
          </span>
        ))}
        <span className="hint">{higherBetter ? "longer is better" : "shorter is better"}</span>
      </div>
    </figure>
  );
}

// ==========================================================================
//  Heatmap — model × scenario magnitude (sequential blue)
// ==========================================================================
export function Heatmap({
  rows, cols, matrix, metricLabel,
}: {
  rows: { key: string; label: string }[];
  cols: { key: string; label: string }[];
  matrix: (number | null)[][];
  metricLabel: string;
}) {
  const { mode } = useTheme();
  const ink = chartInk(mode);
  const [tip, setTip] = useState<Tip | null>(null);
  const flat = matrix.flat().filter((v): v is number => v != null);
  const max = Math.max(...flat, 0.0001);
  const min = Math.min(...flat, 0);

  const cellW = 74, cellH = 34, padL = 116, padT = 78;
  const W = padL + cols.length * cellW + 8;
  const H = padT + rows.length * cellH + 8;

  return (
    <figure style={{ margin: 0 }}>
      <div className="heat-scroll" style={{ position: "relative" }}>
        <svg viewBox={`0 0 ${W} ${H}`} width={W} style={{ maxWidth: "none" }} role="img"
             aria-label={`Heatmap of ${metricLabel} by model and scenario`}>
          {cols.map((c, ci) => (
            <text key={c.key} x={padL + ci * cellW + cellW / 2} y={padT - 8}
                  transform={`rotate(-40 ${padL + ci * cellW + cellW / 2} ${padT - 8})`}
                  textAnchor="start" fontSize={10.5} fill={ink.secondary}
                  fontFamily="var(--font-mono)">{c.label}</text>
          ))}
          {rows.map((r, ri) => (
            <text key={r.key} x={padL - 10} y={padT + ri * cellH + cellH / 2}
                  textAnchor="end" dominantBaseline="middle" fontSize={12} fontWeight={500}
                  fill={ink.primary} fontFamily="var(--font-body)">{r.label}</text>
          ))}
          {rows.map((r, ri) =>
            cols.map((c, ci) => {
              const v = matrix[ri][ci];
              const t = v == null ? 0 : (v - min) / (max - min);
              const fill = v == null ? ink.surface : seqStep(t);
              const textFill = t > 0.55 ? "#ffffff" : ink.primary;
              return (
                <g key={`${ri}-${ci}`}
                   onMouseMove={(e) => setTip({
                     x: e.clientX, y: e.clientY, title: `${r.label} · ${c.label}`,
                     lines: [`${metricLabel}: ${v == null ? "—" : v.toFixed(3)}`],
                   })}
                   onMouseLeave={() => setTip(null)}>
                  <rect x={padL + ci * cellW + 1} y={padT + ri * cellH + 1}
                        width={cellW - 2} height={cellH - 2} rx={4} fill={fill}
                        stroke={ink.surface} strokeWidth={1} />
                  <text x={padL + ci * cellW + cellW / 2} y={padT + ri * cellH + cellH / 2}
                        textAnchor="middle" dominantBaseline="middle" fontSize={10.5}
                        fill={v == null ? ink.muted : textFill} fontFamily="var(--font-mono)">
                    {v == null ? "—" : v.toFixed(2)}
                  </text>
                </g>
              );
            }),
          )}
        </svg>
        <Tooltip tip={tip} />
      </div>
    </figure>
  );
}

// ==========================================================================
//  Listening clock — weekday × hour play-count heatmap (sequential blue)
// ==========================================================================
export function ClockHeatmap({
  weekdays, hours, matrix, max,
}: {
  weekdays: string[]; hours: number[]; matrix: number[][]; max: number;
}) {
  const { mode } = useTheme();
  const ink = chartInk(mode);
  const [tip, setTip] = useState<Tip | null>(null);

  const cellW = 24, cellH = 26, padL = 44, padT = 22, gap = 2;
  const W = padL + hours.length * cellW + 8;
  const H = padT + weekdays.length * cellH + 20;

  return (
    <figure style={{ margin: 0 }}>
      <div className="heat-scroll" style={{ position: "relative" }}>
        <svg viewBox={`0 0 ${W} ${H}`} width={W} style={{ maxWidth: "none" }} role="img"
             aria-label="Listening clock: play counts by weekday and hour of day">
          {hours.map((h) => (
            (h % 3 === 0) ? (
              <text key={h} x={padL + h * cellW + cellW / 2} y={padT - 8} textAnchor="middle"
                    fontSize={9.5} fill={ink.muted} fontFamily="var(--font-mono)">{h}</text>
            ) : null
          ))}
          {weekdays.map((wd, wi) => (
            <text key={wd} x={padL - 10} y={padT + wi * cellH + cellH / 2}
                  textAnchor="end" dominantBaseline="middle" fontSize={11} fontWeight={500}
                  fill={ink.primary} fontFamily="var(--font-body)">{wd}</text>
          ))}
          {weekdays.map((wd, wi) =>
            hours.map((h) => {
              const v = matrix[wi]?.[h] ?? 0;
              const t = max > 0 ? v / max : 0;
              const fill = v === 0 ? ink.surface : seqStep(t);
              return (
                <rect key={`${wi}-${h}`} x={padL + h * cellW + gap / 2} y={padT + wi * cellH + gap / 2}
                      width={cellW - gap} height={cellH - gap} rx={3} fill={fill}
                      stroke={ink.grid} strokeWidth={0.75}
                      onMouseMove={(e) => setTip({
                        x: e.clientX, y: e.clientY, title: `${wd} · ${h}:00`,
                        lines: [`${v} play${v === 1 ? "" : "s"}`],
                      })}
                      onMouseLeave={() => setTip(null)} />
              );
            }),
          )}
          <text x={padL} y={H - 4} fontSize={10} fill={ink.secondary}
                fontFamily="var(--font-mono)">hour of day →</text>
        </svg>
        <Tooltip tip={tip} />
      </div>
    </figure>
  );
}

// ==========================================================================
//  Line chart — trust curves (honest vs adversarial)
// ==========================================================================
export function LineChart({
  lines, xLabel, yLabel, yMax, xTicks,
}: {
  lines: { name: string; color: string; points: { x: number; y: number }[] }[];
  xLabel: string; yLabel: string; yMax?: number;
  xTicks?: number[];
}) {
  const { mode } = useTheme();
  const ink = chartInk(mode);
  const [tip, setTip] = useState<Tip | null>(null);

  const padL = 46, padR = 62, padT = 14, padB = 40;
  const W = 560, H = 300;
  const plotW = W - padL - padR, plotH = H - padT - padB;
  const allY = lines.flatMap((l) => l.points.map((p) => p.y));
  const ymax = yMax ?? Math.max(...allY, 0.1) * 1.1;
  const xs = lines[0]?.points.map((p) => p.x) ?? [];
  const xmin = Math.min(...xs, 0), xmax = Math.max(...xs, 1);
  const X = (x: number) => padL + ((x - xmin) / (xmax - xmin || 1)) * plotW;
  const Y = (y: number) => padT + plotH - (y / ymax) * plotH;
  const ticks = xTicks ?? xs;

  return (
    <figure style={{ margin: 0 }}>
      <div style={{ position: "relative" }}>
        <svg viewBox={`0 0 ${W} ${H}`} width="100%" role="img"
             aria-label={`Line chart of ${yLabel} versus ${xLabel}`}>
          {[0, 0.25, 0.5, 0.75, 1].map((t) => {
            const y = padT + plotH - t * plotH;
            return (
              <g key={t}>
                <line x1={padL} x2={padL + plotW} y1={y} y2={y} stroke={ink.grid} strokeWidth={1} />
                <text x={padL - 8} y={y} textAnchor="end" dominantBaseline="middle"
                      fontSize={10} fill={ink.muted} fontFamily="var(--font-mono)">
                  {(t * ymax).toFixed(2)}
                </text>
              </g>
            );
          })}
          {ticks.map((t) => (
            <text key={t} x={X(t)} y={H - padB + 16} textAnchor="middle"
                  fontSize={10} fill={ink.muted} fontFamily="var(--font-mono)">{t.toFixed(1)}</text>
          ))}
          <text x={padL + plotW / 2} y={H - 4} textAnchor="middle" fontSize={11} fill={ink.secondary}>{xLabel}</text>
          <line x1={padL} x2={padL} y1={padT} y2={padT + plotH} stroke={ink.axis} strokeWidth={1} />
          <line x1={padL} x2={padL + plotW} y1={padT + plotH} y2={padT + plotH} stroke={ink.axis} strokeWidth={1} />
          {lines.map((l) => {
            const d = l.points.map((p, i) => `${i ? "L" : "M"}${X(p.x)},${Y(p.y)}`).join(" ");
            const last = l.points[l.points.length - 1];
            return (
              <g key={l.name}>
                <path d={d} fill="none" stroke={l.color} strokeWidth={2.4}
                      strokeLinejoin="round" strokeLinecap="round" />
                {l.points.map((p, i) => (
                  <circle key={i} cx={X(p.x)} cy={Y(p.y)} r={4} fill={l.color}
                          stroke={ink.surface} strokeWidth={1.5}
                          onMouseMove={(e) => setTip({
                            x: e.clientX, y: e.clientY, title: l.name,
                            lines: [`${xLabel} ${p.x.toFixed(2)}`, `${yLabel} ${p.y.toFixed(3)}`],
                          })}
                          onMouseLeave={() => setTip(null)} />
                ))}
                {last && (
                  <text x={X(last.x) + 8} y={Y(last.y)} dominantBaseline="middle"
                        fontSize={11} fontWeight={600} fill={l.color}>{l.name}</text>
                )}
              </g>
            );
          })}
        </svg>
        <Tooltip tip={tip} />
      </div>
    </figure>
  );
}
