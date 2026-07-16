import { useEffect, useMemo, useState } from "react";
import {
  api, type AskResult, type AskSlot, type AskTemplate, type AxesOverTime,
  type Clock, type Config, type GenreMix,
  type HabitsResult, type HistorySummary, type ShiftsResult, type Shift,
  type TopItems, type Trends, type Window,
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
        <ShiftsPanel window={window} />
        <TopListsPanel window={window} />
        <TrendsPanel window={window} />
        <div className="grid-2">
          <ClockPanel window={window} />
          <GenrePanel window={window} />
        </div>
        <AxesPanel window={window} config={config} />
        <HabitsPanel window={window} config={config} />
        <AskPanel />
      </div>
    </div>
  );
}

// ==========================================================================
//  Ask your library — keyless template library + guarded free-form SQL
// ==========================================================================
function toCsv(result: AskResult): string {
  const cols = result.columns;
  const esc = (v: unknown) => {
    const s = v === null || v === undefined ? "" : String(v);
    return /[",\n]/.test(s) ? `"${s.replace(/"/g, '""')}"` : s;
  };
  const lines = [cols.join(",")];
  for (const row of result.rows) lines.push(cols.map((c) => esc(row[c])).join(","));
  return lines.join("\n");
}

function ResultTable({ result }: { result: AskResult }) {
  if (!result.columns.length) return <p className="hint">No columns returned.</p>;
  return (
    <div style={{ overflowX: "auto" }}>
      <table className="held lib-table" aria-label="Query results">
        <thead>
          <tr>{result.columns.map((c) => <th key={c}>{c}</th>)}</tr>
        </thead>
        <tbody>
          {result.rows.map((row, i) => (
            <tr key={i}>
              {result.columns.map((c) => (
                <td key={c} className="mono small">
                  {row[c] === null || row[c] === undefined ? "—" : String(row[c])}
                </td>
              ))}
            </tr>
          ))}
          {!result.rows.length && (
            <tr><td colSpan={result.columns.length} className="hint" style={{ padding: 16 }}>
              No rows matched.
            </td></tr>
          )}
        </tbody>
      </table>
    </div>
  );
}

// The always-visible "we always show the SQL" disclosure, shared by both paths.
function SqlDisclosure({ sql }: { sql: string }) {
  return (
    <div style={{ marginTop: 12 }}>
      <div className="hint" style={{ marginBottom: 4 }}>
        We always show the exact SQL we ran — nothing runs that you can't read.
      </div>
      <pre className="mono" aria-label="Executed SQL" style={{
        background: "var(--card-2, rgba(127,127,127,0.08))", padding: 12,
        borderRadius: 8, overflowX: "auto", fontSize: 12, margin: 0,
        whiteSpace: "pre-wrap", wordBreak: "break-word",
      }}>{sql}</pre>
    </div>
  );
}

function SlotInput({ slot, value, onChange }: {
  slot: AskSlot; value: string; onChange: (v: string) => void;
}) {
  const common = { "aria-label": slot.label, className: "date-input" as const };
  if (slot.type === "enum") {
    return (
      <label className="hint" style={{ display: "flex", gap: 6, alignItems: "center" }}>
        {slot.label}
        <select {...common} value={value} onChange={(e) => onChange(e.target.value)}>
          {(slot.options ?? []).map((o) => <option key={String(o)} value={String(o)}>{String(o)}</option>)}
        </select>
      </label>
    );
  }
  return (
    <label className="hint" style={{ display: "flex", gap: 6, alignItems: "center" }}>
      {slot.label}
      <input
        {...common}
        type={slot.type === "date" ? "date" : "number"}
        min={slot.min} max={slot.max}
        value={value}
        style={{ width: slot.type === "date" ? undefined : 90 }}
        onChange={(e) => onChange(e.target.value)}
      />
    </label>
  );
}

function AskPanel() {
  const [templates, setTemplates] = useState<AskTemplate[]>([]);
  const [schemaCard, setSchemaCard] = useState("");
  const [activeId, setActiveId] = useState<string>("");
  const [slotVals, setSlotVals] = useState<Record<string, string>>({});
  const [result, setResult] = useState<AskResult | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    api.askTemplates().then((t) => {
      setTemplates(t.templates);
      setSchemaCard(t.schema_card);
      if (t.templates.length) selectTemplate(t.templates[0]);
    }).catch((e) => setError(String(e)));
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const active = useMemo(() => templates.find((t) => t.id === activeId) ?? null, [templates, activeId]);

  function selectTemplate(t: AskTemplate) {
    setActiveId(t.id);
    const init: Record<string, string> = {};
    for (const s of t.slots) init[s.name] = String(s.default);
    setSlotVals(init);
    setResult(null);
    setError(null);
  }

  async function runTemplate() {
    if (!active) return;
    setBusy(true); setError(null);
    try {
      const slots: Record<string, number | string> = {};
      for (const s of active.slots) {
        slots[s.name] = s.type === "int" ? Number(slotVals[s.name]) : slotVals[s.name];
      }
      setResult(await api.askRun(active.id, slots));
    } catch (e) {
      setError(`Could not run this question: ${String(e)}`);
    } finally {
      setBusy(false);
    }
  }

  const csvHref = useMemo(() => {
    if (!result || !result.columns.length) return null;
    return "data:text/csv;charset=utf-8," + encodeURIComponent(toCsv(result));
  }, [result]);

  return (
    <section className="card card-pad" aria-label="Ask your library">
      <div className="card-title">
        Ask your library
        <span className="hint">free-form questions your dashboards don’t cover — no keys needed</span>
      </div>
      <p className="hint" style={{ marginTop: 0 }}>
        Pick a question, tune its parameters, and run it. Everything goes through a
        read-only guardrail (single SELECT only) and we always show you the SQL.
      </p>

      {error && <div className="callout" style={{ marginBottom: 12 }}>{error}</div>}

      <div style={{ display: "flex", gap: 12, alignItems: "center", flexWrap: "wrap", marginBottom: 10 }}>
        <label className="hint" style={{ display: "flex", gap: 6, alignItems: "center" }}>
          Question
          <select
            className="date-input" aria-label="Template question"
            value={activeId}
            onChange={(e) => {
              const t = templates.find((x) => x.id === e.target.value);
              if (t) selectTemplate(t);
            }}
            style={{ minWidth: 320 }}
          >
            {templates.map((t) => <option key={t.id} value={t.id}>{t.question}</option>)}
          </select>
        </label>
      </div>

      {active && (
        <>
          <p className="hint" style={{ marginTop: 0 }}>{active.description}</p>
          {active.slots.length > 0 && (
            <div style={{ display: "flex", gap: 14, alignItems: "center", flexWrap: "wrap", marginBottom: 12 }}>
              {active.slots.map((s) => (
                <SlotInput key={s.name} slot={s} value={slotVals[s.name] ?? ""}
                           onChange={(v) => setSlotVals((prev) => ({ ...prev, [s.name]: v }))} />
              ))}
            </div>
          )}
          <button className="btn" onClick={runTemplate} disabled={busy}>
            {busy ? "Running…" : "Run question"}
          </button>
        </>
      )}

      {result && (
        <div style={{ marginTop: 16 }}>
          <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", gap: 12, flexWrap: "wrap" }}>
            <span className="hint">
              {result.row_count} row{result.row_count === 1 ? "" : "s"}
              {result.truncated ? " (capped)" : ""}
            </span>
            {csvHref && (
              <a className="btn ghost" href={csvHref} download={`${result.template_id ?? "query"}.csv`}>
                Download CSV
              </a>
            )}
          </div>
          <div style={{ marginTop: 10 }}><ResultTable result={result} /></div>
          <SqlDisclosure sql={result.sql} />
        </div>
      )}

      <AdvancedSql schemaCard={schemaCard} />
    </section>
  );
}

// The collapsed power-user raw-SQL box (same guardrail as the templates).
function AdvancedSql({ schemaCard }: { schemaCard: string }) {
  const [open, setOpen] = useState(false);
  const [sql, setSql] = useState("SELECT COUNT(*) AS plays FROM events");
  const [result, setResult] = useState<AskResult | null>(null);
  const [busy, setBusy] = useState(false);

  async function run() {
    setBusy(true);
    try {
      setResult(await api.askSql(sql));
    } catch (e) {
      setResult({ ok: false, error: String(e), sql, columns: [], rows: [], row_count: 0, truncated: false });
    } finally {
      setBusy(false);
    }
  }

  return (
    <details style={{ marginTop: 20 }} onToggle={(e) => setOpen((e.target as HTMLDetailsElement).open)}>
      <summary className="hint" style={{ cursor: "pointer" }}>Advanced: write your own SQL</summary>
      {open && (
        <div style={{ marginTop: 12 }}>
          <div className="coverage-banner block" role="note" style={{ marginBottom: 10 }}>
            Power-user mode. Only a <b>single read-only SELECT</b> runs — writes, DDL,
            <span className="mono"> PRAGMA</span>, <span className="mono">ATTACH</span>,
            <span className="mono"> COPY</span> and multi-statement queries are rejected,
            and every query runs in a read-only transaction with an enforced row cap.
          </div>
          <textarea
            className="mono" aria-label="Raw SQL" value={sql}
            onChange={(e) => setSql(e.target.value)}
            spellCheck={false}
            style={{ width: "100%", minHeight: 90, padding: 10, borderRadius: 8,
                     fontSize: 12, boxSizing: "border-box" }}
          />
          <div style={{ display: "flex", gap: 8, alignItems: "center", marginTop: 8 }}>
            <button className="btn" onClick={run} disabled={busy}>{busy ? "Running…" : "Run SQL"}</button>
            {schemaCard && (
              <details style={{ marginLeft: "auto" }}>
                <summary className="hint" style={{ cursor: "pointer" }}>schema reference</summary>
                <pre className="mono" style={{ fontSize: 11, overflowX: "auto", maxHeight: 260,
                     whiteSpace: "pre-wrap" }}>{schemaCard}</pre>
              </details>
            )}
          </div>
          {result && result.ok === false && (
            <div className="callout" style={{ marginTop: 12 }}>
              Rejected: {result.error}
            </div>
          )}
          {result && result.ok !== false && (
            <div style={{ marginTop: 12 }}>
              <span className="hint">{result.row_count} row{result.row_count === 1 ? "" : "s"}
                {result.truncated ? " (capped)" : ""}</span>
              <div style={{ marginTop: 10 }}><ResultTable result={result} /></div>
              <SqlDisclosure sql={result.sql} />
            </div>
          )}
        </div>
      )}
    </details>
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
        <SkipStat summary={s} />
      </div>
    </div>
  );
}

// Skip rate is honest about Spotify's late-arriving skip flag: null before logging began,
// and captioned with the month it started + how much of the window it covers.
function SkipStat({ summary: s }: { summary: HistorySummary }) {
  const monthLabel = (iso: string | null) =>
    iso ? new Date(`${iso}T00:00:00Z`).toLocaleDateString(undefined,
      { month: "short", year: "numeric", timeZone: "UTC" }) : null;
  if (s.skip_rate == null) {
    return <Stat label="Skip rate" num="—"
                 sub={s.skip_reason ?? "no skip data in this window"} />;
  }
  const from = monthLabel(s.skip_reliable_from);
  const partial = s.skip_coverage > 0 && s.skip_coverage < 0.999;
  const sub = from
    ? `since ${from}${partial ? ` · ${Math.round(s.skip_coverage * 100)}% of plays` : ""}`
    : `of ${fmt(s.skip_flagged)} flagged plays`;
  return <Stat label="Skip rate" num={`${Math.round(s.skip_rate * 100)}%`} sub={sub} />;
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
type Gran = "day" | "week" | "month" | "year";

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
  // Bug 1: for skip_rate, buckets before Spotify began logging skips carry a null value;
  // find where real data starts so we can grey that region and drop the null points.
  const reliableFrom = metric === "skip_rate" ? data?.skip_reliable_from ?? null : null;
  const firstReliableIdx = useMemo(() => {
    if (metric !== "skip_rate") return -1;
    const i = buckets.findIndex((b) => b.value != null);
    return i;
  }, [buckets, metric]);
  const reliableMonth = reliableFrom
    ? new Date(`${reliableFrom}T00:00:00Z`).toLocaleDateString(undefined,
        { month: "long", year: "numeric", timeZone: "UTC" })
    : null;

  const lines = useMemo(() => {
    // Drop null buckets from the plotted points (null y would break the path); the greyed
    // region + caption below explain the gap for the pre-logging skip period.
    const pts = (pick: (b: Trends["buckets"][number]) => number | null | undefined) =>
      buckets.map((b, i) => ({ x: i, y: pick(b) }))
             .filter((p): p is { x: number; y: number } => p.y != null);
    const raw = { name: METRIC_LABEL[metric], color: HONEST[mode], points: pts((b) => b.value) };
    const out = [raw];
    if (rolling && buckets.some((b) => b.rolling != null)) {
      out.push({ name: "rolling mean", color: SERIES[mode][2], points: pts((b) => b.rolling) });
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
            {(["day", "week", "month", "year"] as Gran[]).map((g) => (
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
          shadeUntilX={firstReliableIdx > 0 ? firstReliableIdx - 0.5 : undefined}
          shadeLabel={firstReliableIdx > 0 ? "no skip data" : undefined}
        />
      ) : <p className="hint">No plays in this window.</p>}
      <div style={{ display: "flex", justifyContent: "space-between", marginTop: 6, gap: 10, flexWrap: "wrap" }}>
        <p className="hint" style={{ margin: 0 }}>
          {metric === "discovery"
            ? "Discovery = share of plays that were the first-ever play of that track (measured across the full history)."
            : metric === "skip_rate"
            ? (reliableMonth
                ? `Skip rate = share of skip-flagged plays that were skips. Spotify began recording skips in ${reliableMonth}; earlier plays (greyed) show no skip data.`
                : "Skip rate = share of skip-flagged plays that were skips. Spotify recorded no skips in this history.")
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
      <div className="card-title">
        Listening clock
        <span className="hint">
          plays by weekday × hour{data?.tz ? ` · times shown in ${data.tz}` : ""}
        </span>
      </div>
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
          {(["week", "month", "year"] as Gran[]).map((g) => (
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

// ---- significant shifts: FDR-corrected, effect-floored change detection ---- //
// Effect magnitude that fills the bar, per effect kind (documented floors live in
// the backend; these are display scales, not thresholds).
const EFFECT_FULL: Record<string, number> = {
  cohen_d: 0.8, rank_biserial: 0.8, cramers_v: 0.5, prop_diff: 0.3,
};
const EFFECT_SYMBOL: Record<string, string> = {
  cohen_d: "d", rank_biserial: "r", cramers_v: "V", prop_diff: "Δ",
};

function ShiftsPanel({ window }: { window: Window }) {
  const { mode } = useTheme();
  const [data, setData] = useState<ShiftsResult | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    setData(null); setError(null);
    api.historyShifts(window).then(setData).catch((e) => setError(String(e)));
  }, [window]);

  const a = data?.window_a;
  const b = data?.window_b;

  return (
    <section className="card card-pad" aria-label="Significant shifts">
      <div className="card-title">
        Significant shifts
        <span className="hint">selected window vs the preceding same-length window</span>
      </div>

      {error && <p className="hint">Could not compute shifts: {error}</p>}
      {!data && !error && <p className="hint">Testing for real changes…</p>}

      {data && (
        <>
          <div className="shift-windows hint mono" role="note">
            {a?.start && a?.end
              ? <>baseline <b>{a.start} → {a.end}</b> &nbsp;vs&nbsp; recent <b>{b?.start} → {b?.end}</b></>
              : <>not enough history before the selected window to form a baseline</>}
          </div>

          {data.shifts.length > 0 ? (
            <div className="shift-list">
              {data.shifts.map((s) => <ShiftCard key={`${s.metric}-${s.kind}`} shift={s} mode={mode} />)}
            </div>
          ) : (
            <div className="callout" style={{ marginTop: 12 }}>
              {data.family_size > 0
                ? "No change here is both statistically solid and big enough to matter. Across "
                  + `${data.family_size} tests, nothing cleared the FDR + effect-size bar — your `
                  + "listening in these two windows is statistically indistinguishable."
                : "Not enough plays in these windows to test for a shift."}
            </div>
          )}

          {data.insufficient.length > 0 && (
            <p className="hint" style={{ marginTop: 12 }}>
              <span className="tag">insufficient data</span>{" "}
              couldn’t test {data.insufficient.map((i) => i.metric).join(", ")} — {data.insufficient[0].reason}.
            </p>
          )}

          <p className="coverage-banner block" role="note" style={{ marginTop: 14 }}>
            Results are <b>FDR-corrected</b> (Benjamini-Hochberg, q&nbsp;&lt;&nbsp;{data.q_threshold})
            with effect-size floors: we only show changes that are both statistically solid and
            big enough to matter. A bare p-value means little at this many plays.
          </p>
        </>
      )}
    </section>
  );
}

function ShiftCard({ shift, mode }: { shift: Shift; mode: "light" | "dark" }) {
  const full = EFFECT_FULL[shift.effect_name] ?? 1;
  const pct = Math.min(100, (Math.abs(shift.effect) / full) * 100);
  const sym = EFFECT_SYMBOL[shift.effect_name] ?? "e";
  const color = SERIES[mode][0];
  return (
    <div className="shift-card">
      <p className="shift-sentence">{shift.sentence}</p>
      <div className="shift-meta">
        <span className="shift-bar" aria-hidden="true">
          <span style={{ width: `${pct}%`, background: color }} />
        </span>
        <span className="mono shift-eff">
          {sym}={shift.effect >= 0 ? "+" : ""}{shift.effect.toFixed(2)} · q={shift.q.toFixed(3)}
          {shift.corroborated && <span className="hint"> · both tests agree</span>}
        </span>
      </div>
    </div>
  );
}

// ---- habits: weekday / time-of-day taste ANOVA ---------------------------- //
type GroupBy = "weekday" | "hour_band";
const GROUP_LABEL: Record<GroupBy, string> = {
  weekday: "By weekday", hour_band: "By time of day",
};

function HabitsPanel({ window, config }: { window: Window; config: Config }) {
  const [groupBy, setGroupBy] = useState<GroupBy>("weekday");
  const [data, setData] = useState<HabitsResult | null>(null);

  useEffect(() => {
    setData(null); api.historyHabits(window, groupBy).then(setData);
  }, [window, groupBy]);

  const labelOf = (key: string) => config.axes.scalar.find((a) => a.key === key)?.label ?? key;
  const noun = groupBy === "weekday" ? "day of the week" : "time of day";

  return (
    <section className="card card-pad" aria-label="Habits">
      <div className="card-title">
        Habits
        <span className="seg" role="group" aria-label="Group habits by">
          {(["weekday", "hour_band"] as GroupBy[]).map((g) => (
            <button key={g} aria-pressed={groupBy === g} onClick={() => setGroupBy(g)}>
              {GROUP_LABEL[g]}
            </button>
          ))}
        </span>
      </div>

      {!data && <p className="hint">Testing habits…</p>}
      {data && (
        data.survivors.length > 0 ? (
          <div className="shift-list">
            {data.survivors.map((s) => (
              <div className="shift-card" key={s.axis}>
                <p className="shift-sentence">
                  {s.summary.replace(s.axis, labelOf(s.axis))}
                </p>
                <div className="shift-meta">
                  <span className="mono shift-eff">
                    Welch F={s.welch_F.toFixed(1)} · q={s.welch_q.toFixed(3)} · ε²={s.epsilon_sq.toFixed(2)}
                  </span>
                </div>
              </div>
            ))}
          </div>
        ) : (
          <div className="callout">
            Your taste axes don’t depend on {noun} — no axis differs across {noun} once
            we FDR-correct and require a real effect size. (Volume may vary; the *character*
            of what you play doesn’t.)
          </div>
        )
      )}
      <p className="hint" style={{ marginTop: 10 }}>
        Welch’s ANOVA (unequal-variance) + Kruskal-Wallis per axis, BH-corrected across axes,
        with an ε²&nbsp;≥&nbsp;0.01 effect floor.
      </p>
    </section>
  );
}

function fmt(n: number): string {
  return n.toLocaleString();
}
