import { useEffect, useMemo, useState } from "react";
import {
  api, type AxesOverTime, type Clock, type Config, type GenreMix,
  type HistorySummary, type TopItems, type Trends, type Window,
} from "../api";
import { useTheme } from "../theme";
import { HONEST, SERIES } from "../palette";
import { ClockHeatmap, LineChart } from "./charts";

// ==========================================================================
//  Library — a personal listening-analytics dashboard over a listening history
// ==========================================================================
export function Library({ config }: { config: Config }) {
  const [summary, setSummary] = useState<HistorySummary | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [window, setWindow] = useState<Window>({ start: null, end: null });

  // fetch the (window-scoped) summary whenever the window changes
  useEffect(() => {
    api.historySummary(window).then(setSummary).catch((e) => setError(String(e)));
  }, [window]);

  if (error) {
    return <div className="page"><div className="card card-pad">Could not reach the API: {error}</div></div>;
  }
  if (!summary) {
    return <div className="page"><div className="center-empty">Reading the listening history…</div></div>;
  }

  const fullSpan = summary.full_span;

  return (
    <div className="page reveal">
      <div className="page-head">
        <div className="eyebrow">Library</div>
        <h1 className="page-title">Your listening, mapped honestly.</h1>
        <p className="page-lede">
          Full ranked lists, trends, a listening clock and taste-axis drift over a
          personal play history — not a top-five. Every panel reacts to the window below,
          and the taste panels state exactly how much of your listening they could measure.
        </p>
      </div>

      <Header summary={summary} />

      <div className="stack">
        <WindowPicker window={window} setWindow={setWindow} fullSpan={fullSpan} />
        <TopListsPanel window={window} />
        <TrendsPanel window={window} />
        <div className="grid-2">
          <ClockPanel window={window} />
          <GenrePanel window={window} />
        </div>
        <AxesPanel window={window} config={config} />
      </div>
    </div>
  );
}

// ---- header: provenance badge + span + stat cards ------------------------- //
function Header({ summary }: { summary: HistorySummary }) {
  const s = summary;
  const span = s.span;
  return (
    <div style={{ marginBottom: 20 }}>
      <div style={{ display: "flex", alignItems: "center", gap: 12, flexWrap: "wrap", marginBottom: 14 }}>
        <span className={`tag ${s.is_synthetic ? "adv" : ""}`} role="note">
          {s.is_synthetic ? "◆ " : "● "}{s.provenance_label}
        </span>
        {span ? (
          <span className="hint mono">
            {span.first} → {span.last} · {span.days} days
          </span>
        ) : (
          <span className="hint">no plays in this window</span>
        )}
        {s.is_synthetic && s.ground_truth && (
          <span className="hint">
            demo data · {s.ground_truth.changes.length} planted taste-change landmarks
          </span>
        )}
      </div>
      <div className="stat-row">
        <Stat label="Plays" num={fmt(s.total_plays)} sub={`${s.plays_per_day}/day on average`} />
        <Stat label="Minutes" num={fmt(Math.round(s.total_minutes))}
              sub={`${Math.round(s.total_minutes / 60).toLocaleString()} hours of listening`} />
        <Stat label="Artists" num={fmt(s.distinct_artists)} sub={`${fmt(s.distinct_tracks)} distinct tracks`} />
        <Stat label="Skip rate" num={`${Math.round(s.skip_rate * 100)}%`}
              sub={s.skip_flagged ? `of ${fmt(s.skip_flagged)} flagged plays` : "no skip data"} />
      </div>
    </div>
  );
}

function Stat({ label, num, sub }: { label: string; num: string; sub: string }) {
  return (
    <div className="stat">
      <div className="label">{label}</div>
      <div className="num">{num}</div>
      <div className="sub">{sub}</div>
    </div>
  );
}

// ---- window picker: presets + custom date inputs -------------------------- //
const PRESETS: { id: string; label: string; days: number | null }[] = [
  { id: "30", label: "Last 30 days", days: 30 },
  { id: "90", label: "Last 90 days", days: 90 },
  { id: "365", label: "Last 365 days", days: 365 },
  { id: "all", label: "All time", days: null },
];

function presetWindow(fullLast: string, days: number): Window {
  const last = new Date(`${fullLast}T00:00:00Z`);
  const start = new Date(last);
  start.setUTCDate(start.getUTCDate() - (days - 1));
  return { start: start.toISOString().slice(0, 10), end: fullLast };
}

function WindowPicker({
  window, setWindow, fullSpan,
}: {
  window: Window; setWindow: (w: Window) => void;
  fullSpan: { first: string; last: string; days: number } | null;
}) {
  const [active, setActive] = useState("all");
  const min = fullSpan?.first;
  const max = fullSpan?.last;

  const pick = (p: typeof PRESETS[number]) => {
    setActive(p.id);
    if (p.days == null || !fullSpan) setWindow({ start: null, end: null });
    else setWindow(presetWindow(fullSpan.last, p.days));
  };

  return (
    <section className="card card-pad" aria-label="Window">
      <div className="card-title">Window <span className="hint">all panels react to this</span></div>
      <div style={{ display: "flex", gap: 12, alignItems: "center", flexWrap: "wrap" }}>
        <span className="seg" role="group" aria-label="Window preset">
          {PRESETS.map((p) => (
            <button key={p.id} aria-pressed={active === p.id} onClick={() => pick(p)}>{p.label}</button>
          ))}
        </span>
        <label className="hint" style={{ display: "flex", gap: 6, alignItems: "center" }}>
          from
          <input type="date" className="date-input" min={min} max={max} value={window.start ?? ""}
                 aria-label="Window start date"
                 onChange={(e) => { setActive("custom"); setWindow({ start: e.target.value || null, end: window.end }); }} />
        </label>
        <label className="hint" style={{ display: "flex", gap: 6, alignItems: "center" }}>
          to
          <input type="date" className="date-input" min={min} max={max} value={window.end ?? ""}
                 aria-label="Window end date"
                 onChange={(e) => { setActive("custom"); setWindow({ start: window.start, end: e.target.value || null }); }} />
        </label>
      </div>
    </section>
  );
}

// ---- top lists panel: entity tabs, by toggle, paginated table, CSV -------- //
const PAGE = 25;
type Entity = "tracks" | "artists" | "albums";

function TopListsPanel({ window }: { window: Window }) {
  const [entity, setEntity] = useState<Entity>("tracks");
  const [by, setBy] = useState<"plays" | "minutes">("plays");
  const [offset, setOffset] = useState(0);
  const [data, setData] = useState<TopItems | null>(null);

  // reset to first page whenever the query dimensions change
  useEffect(() => { setOffset(0); }, [entity, by, window]);

  useEffect(() => {
    api.historyTop(window, entity, by, PAGE, offset).then(setData);
  }, [window, entity, by, offset]);

  const total = data?.total ?? 0;
  const csvUrl = api.historyTopCsvUrl(window, entity, by);

  return (
    <section className="card card-pad" aria-label="Top lists">
      <div className="card-title">
        Full ranked list
        <span style={{ display: "flex", gap: 8 }}>
          <span className="seg" role="group" aria-label="Entity">
            {(["tracks", "artists", "albums"] as Entity[]).map((e) => (
              <button key={e} aria-pressed={entity === e} onClick={() => setEntity(e)}>
                {e[0].toUpperCase() + e.slice(1)}
              </button>
            ))}
          </span>
          <span className="seg" role="group" aria-label="Rank by">
            {(["plays", "minutes"] as const).map((b) => (
              <button key={b} aria-pressed={by === b} onClick={() => setBy(b)}>
                {b === "plays" ? "By plays" : "By minutes"}
              </button>
            ))}
          </span>
        </span>
      </div>

      <div style={{ overflowX: "auto" }}>
        <table className="held lib-table">
          <thead>
            <tr>
              <th style={{ width: 44 }}>#</th>
              <th>{entity === "tracks" ? "Track" : entity === "artists" ? "Artist" : "Album"}</th>
              {entity === "tracks" && <th>Artist</th>}
              <th style={{ textAlign: "right" }}>Plays</th>
              <th style={{ textAlign: "right" }}>Minutes</th>
              <th style={{ width: 120 }}>Share</th>
            </tr>
          </thead>
          <tbody>
            {(data?.rows ?? []).map((r) => (
              <tr key={`${r.rank}-${r.name}`}>
                <td className="mono" style={{ color: "var(--muted)" }}>{r.rank}</td>
                <td style={{ fontWeight: 600 }}>{r.name}</td>
                {entity === "tracks" && <td className="small">{r.artist}</td>}
                <td className="mono" style={{ textAlign: "right" }}>{fmt(r.plays)}</td>
                <td className="mono" style={{ textAlign: "right" }}>{fmt(Math.round(r.minutes))}</td>
                <td>
                  <span className="share-bar" aria-hidden="true">
                    <span style={{ width: `${Math.min(100, r.share * 100)}%` }} />
                  </span>
                  <span className="mono share-pct">{(r.share * 100).toFixed(1)}%</span>
                </td>
              </tr>
            ))}
            {!data?.rows.length && (
              <tr><td colSpan={6} className="hint" style={{ padding: 16 }}>No plays in this window.</td></tr>
            )}
          </tbody>
        </table>
      </div>

      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginTop: 12, gap: 12, flexWrap: "wrap" }}>
        <span className="hint">
          {total ? `${offset + 1}–${Math.min(offset + PAGE, total)} of ${fmt(total)} ${entity}` : "—"}
        </span>
        <span style={{ display: "flex", gap: 8, alignItems: "center" }}>
          <a className="btn ghost" href={csvUrl} download={`top_${entity}_by_${by}.csv`}>Download CSV</a>
          <button className="btn" disabled={offset <= 0} onClick={() => setOffset(Math.max(0, offset - PAGE))}>Prev</button>
          <button className="btn" disabled={offset + PAGE >= total} onClick={() => setOffset(offset + PAGE)}>Next</button>
        </span>
      </div>
    </section>
  );
}

// ---- trends panel: metric + granularity + optional rolling mean ----------- //
type Metric = "plays" | "minutes" | "discovery" | "skip_rate";
const METRIC_LABEL: Record<Metric, string> = {
  plays: "Plays", minutes: "Minutes", discovery: "Discovery rate", skip_rate: "Skip rate",
};
type Gran = "day" | "week" | "month";

function TrendsPanel({ window }: { window: Window }) {
  const { mode } = useTheme();
  const [metric, setMetric] = useState<Metric>("plays");
  const [gran, setGran] = useState<Gran>("week");
  const [rolling, setRolling] = useState(false);
  const [data, setData] = useState<Trends | null>(null);

  const k = rolling ? 5 : null;
  useEffect(() => {
    api.historyTrends(window, metric, gran, k).then(setData);
  }, [window, metric, gran, rolling]);

  const isShare = metric === "discovery" || metric === "skip_rate";
  const buckets = data?.buckets ?? [];
  const lines = useMemo(() => {
    const raw = {
      name: METRIC_LABEL[metric],
      color: HONEST[mode],
      points: buckets.map((b, i) => ({ x: i, y: b.value })),
    };
    const out = [raw];
    if (rolling && buckets.some((b) => b.rolling != null)) {
      out.push({
        name: "rolling mean",
        color: SERIES[mode][2],
        points: buckets.map((b, i) => ({ x: i, y: b.rolling ?? 0 })),
      });
    }
    return out;
  }, [buckets, metric, mode, rolling]);

  const csvUrl = api.historyTrendsCsvUrl(window, metric, gran, k);

  return (
    <section className="card card-pad" aria-label="Trends">
      <div className="card-title">
        Trend over time
        <span style={{ display: "flex", gap: 8, flexWrap: "wrap" }}>
          <span className="seg" role="group" aria-label="Metric">
            {(["plays", "minutes", "discovery", "skip_rate"] as Metric[]).map((m) => (
              <button key={m} aria-pressed={metric === m} onClick={() => setMetric(m)}>{METRIC_LABEL[m]}</button>
            ))}
          </span>
          <span className="seg" role="group" aria-label="Granularity">
            {(["day", "week", "month"] as Gran[]).map((g) => (
              <button key={g} aria-pressed={gran === g} onClick={() => setGran(g)}>{g}</button>
            ))}
          </span>
          <button className="btn ghost" aria-pressed={rolling} onClick={() => setRolling(!rolling)}>
            {rolling ? "✓ rolling mean" : "rolling mean"}
          </button>
        </span>
      </div>
      {buckets.length ? (
        <LineChart
          xLabel={`${gran} (index)`}
          yLabel={METRIC_LABEL[metric]}
          yMax={isShare ? 1 : undefined}
          xTicks={tickIndices(buckets.length)}
          lines={lines}
        />
      ) : <p className="hint">No plays in this window.</p>}
      <div style={{ display: "flex", justifyContent: "space-between", marginTop: 6, gap: 10, flexWrap: "wrap" }}>
        <p className="hint" style={{ margin: 0 }}>
          {metric === "discovery"
            ? "Discovery = share of plays that were the first-ever play of that track (measured across the full history)."
            : metric === "skip_rate"
            ? "Skip rate = share of skip-flagged plays that were skips."
            : `${METRIC_LABEL[metric]} per ${gran}.`}
        </p>
        <a className="btn ghost" href={csvUrl} download={`trend_${metric}_${gran}.csv`}>Download CSV</a>
      </div>
    </section>
  );
}

function tickIndices(n: number): number[] {
  if (n <= 1) return [0];
  const step = Math.max(1, Math.round(n / 6));
  const out: number[] = [];
  for (let i = 0; i < n; i += step) out.push(i);
  if (out[out.length - 1] !== n - 1) out.push(n - 1);
  return out;
}

// ---- listening clock ------------------------------------------------------ //
function ClockPanel({ window }: { window: Window }) {
  const [data, setData] = useState<Clock | null>(null);
  useEffect(() => { api.historyClock(window).then(setData); }, [window]);
  return (
    <section className="card card-pad" aria-label="Listening clock">
      <div className="card-title">Listening clock <span className="hint">plays by weekday × hour</span></div>
      {data && data.total ? (
        <ClockHeatmap weekdays={data.weekdays} hours={data.hours} matrix={data.matrix} max={data.max} />
      ) : <p className="hint">No plays in this window.</p>}
    </section>
  );
}

// ---- genre mix ------------------------------------------------------------ //
function GenrePanel({ window }: { window: Window }) {
  const { mode } = useTheme();
  const [data, setData] = useState<GenreMix | null>(null);
  useEffect(() => { api.historyGenres(window).then(setData); }, [window]);
  const palette = SERIES[mode];
  const max = data?.mix.length ? Math.max(...data.mix.map((m) => m.mean)) : 1;
  return (
    <section className="card card-pad" aria-label="Genre mix">
      <div className="card-title">Genre mix <span className="hint">mean genre axis over the window</span></div>
      <CoverageBanner coverage={data?.coverage ?? 0} plays={data?.plays_with_features ?? 0}
                      total={data?.total_plays ?? 0} />
      {data && data.mix.length ? (
        <div className="genre-mix">
          {data.mix.slice(0, 10).map((m, i) => (
            <div className="genre-row" key={m.genre}>
              <span className="k mono">{m.genre}</span>
              <span className="genre-bar">
                <span style={{ width: `${(m.mean / max) * 100}%`, background: palette[i % palette.length] }} />
              </span>
              <span className="v mono">{m.mean.toFixed(3)}</span>
            </div>
          ))}
        </div>
      ) : <p className="hint">No genre features in this window.</p>}
    </section>
  );
}

// ---- taste axes over time (small multiples) ------------------------------- //
function AxesPanel({ window, config }: { window: Window; config: Config }) {
  const { mode } = useTheme();
  const [gran, setGran] = useState<Gran>("month");
  const [data, setData] = useState<AxesOverTime | null>(null);
  useEffect(() => { api.historyAxes(window, gran).then(setData); }, [window, gran]);

  const labelOf = (key: string) => config.axes.scalar.find((a) => a.key === key)?.label ?? key;
  const buckets = data?.buckets ?? [];
  // coverage-weighted overall coverage across buckets
  const covPlays = buckets.reduce((s, b) => s + b.plays * b.coverage, 0);
  const totPlays = buckets.reduce((s, b) => s + b.plays, 0);
  const coverage = totPlays ? covPlays / totPlays : 0;

  return (
    <section className="card card-pad" aria-label="Taste axes over time">
      <div className="card-title">
        Taste axes over time
        <span className="seg" role="group" aria-label="Axes granularity">
          {(["week", "month"] as Gran[]).map((g) => (
            <button key={g} aria-pressed={gran === g} onClick={() => setGran(g)}>{g}</button>
          ))}
        </span>
      </div>
      {buckets.length ? (
        <div className="grid-2 axes-multiples">
          {(data?.axes ?? []).map((axis) => (
            <div key={axis} className="axis-multiple">
              <div className="axis-multiple-head">
                <strong>{labelOf(axis)}</strong>
                <span className="coverage-banner">axes computed on {Math.round(coverage * 100)}% of plays</span>
              </div>
              <LineChart
                xLabel={`${gran} (index)`}
                yLabel={labelOf(axis)}
                yMax={1}
                xTicks={tickIndices(buckets.length)}
                lines={[{
                  name: labelOf(axis),
                  color: SERIES[mode][0],
                  points: buckets.map((b, i) => ({ x: i, y: b.means[axis] ?? 0 })),
                }]}
              />
            </div>
          ))}
        </div>
      ) : <p className="hint">No plays in this window.</p>}
    </section>
  );
}

function CoverageBanner({ coverage, plays, total }: { coverage: number; plays: number; total: number }) {
  return (
    <div className="coverage-banner block" role="note">
      Axes computed on <b>{Math.round(coverage * 100)}%</b> of plays
      {total ? ` (${fmt(plays)} of ${fmt(total)} had features)` : ""}.
    </div>
  );
}

function fmt(n: number): string {
  return n.toLocaleString();
}
