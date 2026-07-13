// Typed wrappers over the Taste Atlas backend API.

export interface AxisMeta { key: string; label: string; low: string; high: string }
export interface AxesConfig { scalar: AxisMeta[]; genres: string[] }

export interface TrackCard {
  uri: string;
  track_name: string;
  artist_name: string;
  genre: string;
  archetype: string;
  axes: Record<string, number>;
  pop: number;
}

export interface SamplePlaylist {
  pid: number;
  name: string;
  archetype: string;
  n_tracks: number;
  seed_uris: string[];
  sample: TrackCard[];
}

export interface PersonaSummary {
  id: string;
  name: string;
  blurb: string;
  archetype: string;
  adversarial: boolean;
  n_seed_tracks: number;
}

export interface CatalogPoint {
  tempo: number; energy: number; valence: number;
  acousticness: number; lyrical_depth: number;
  genre: string; pop: number;
}

export interface Config {
  axes: AxesConfig;
  sample_playlists: SamplePlaylist[];
  personas: PersonaSummary[];
  catalog: CatalogPoint[];
}

export interface Weight { axis: string; value: number }
export interface Cluster {
  name: string;
  share: number;
  axes: Record<string, number>;
  genre: string;
}
export interface Profile {
  trust: number;
  has_stated: boolean;
  n_seed_tracks: number;
  weights: { stated: Weight[]; learned: Weight[]; blended: Weight[] };
  clusters: Cluster[];
}

export interface AxisBar { axis: string; contribution: number; label: string }
export interface Rec extends TrackCard {
  rank: number;
  score: number;
  flavor: string;
  reasons: string[];
  axis_bars: AxisBar[];
  cooccurrence: number;
  same_artist: boolean;
}
export interface Recommendations {
  trust: number;
  n_seed_tracks: number;
  items: Rec[];
}

export interface PersonaDetail {
  id: string; name: string; blurb: string; archetype: string;
  adversarial: boolean; trust: number;
  stated_axes: Record<string, number>;
  profile: Profile;
  recommendations: Recommendations;
  fit: number;
  seed_sample: TrackCard[];
}

export interface TrustCurve {
  id: string; adversarial: boolean; archetype: string;
  curve: { trust: number; fit: number }[];
}

export interface Payload {
  scalars: Record<string, number>;
  genres: string[];
  seed_uris: string[];
  trust: number;
  k?: number;
}

// ---- results explorer ----
export interface OverviewRow {
  model: string; label: string;
  r_precision: number; ndcg: number; clicks: number;
}
export interface Overview { synthetic: OverviewRow[]; real: OverviewRow[] }
export interface ScenarioGrid {
  dataset: string; metric: string;
  models: { key: string; label: string }[];
  scenarios: { key: string; label: string }[];
  matrix: (number | null)[][];
}
export interface HeldRow {
  conclusion: string; synthetic: string; real: string; verdict: string;
}
export interface TrustAblation {
  honest: { trust: number; r_precision: number; clicks: number }[];
  adversarial: { trust: number; r_precision: number; clicks: number }[];
}

// ---- library (personal listening analytics) ----
export interface Span { first: string; last: string; days: number }
export interface GroundTruthChange { date: string; kind: string; description: string }
export interface HistorySummary {
  start: string | null; end: string | null;
  span: Span | null;
  total_plays: number; total_minutes: number;
  distinct_tracks: number; distinct_artists: number;
  skip_flagged: number; skip_rate: number; plays_per_day: number;
  provenance: string; provenance_label: string; is_synthetic: boolean;
  full_span: Span | null;
  ground_truth: {
    seed: number;
    changes: GroundTruthChange[];
    regimes: { start: string; end: string; label: string }[];
  } | null;
}
export interface TopRow {
  rank: number; name: string; artist?: string;
  plays: number; minutes: number; share: number;
}
export interface TopItems {
  entity: string; by: string; total: number;
  total_plays: number; total_minutes: number;
  offset: number; limit: number; rows: TopRow[];
}
export interface TrendBucket { bucket: string; value: number; rolling?: number | null }
export interface Trends {
  metric: string; granularity: string; rolling: number | null; buckets: TrendBucket[];
}
export interface Clock {
  weekdays: string[]; hours: number[]; matrix: number[][]; max: number; total: number;
}
export interface AxesOverTime {
  granularity: string; axes: string[];
  buckets: { bucket: string; plays: number; coverage: number; means: Record<string, number | null> }[];
}
export interface GenreMix {
  genres: string[]; mix: { genre: string; mean: number }[];
  coverage: number; plays_with_features: number; total_plays: number;
}

export interface Window { start: string | null; end: string | null }
function win(w: Window): string {
  const p = new URLSearchParams();
  if (w.start) p.set("start", w.start);
  if (w.end) p.set("end", w.end);
  return p.toString();
}

async function get<T>(url: string): Promise<T> {
  const r = await fetch(url);
  if (!r.ok) throw new Error(`${url} -> ${r.status}`);
  return r.json() as Promise<T>;
}
async function post<T>(url: string, body: unknown): Promise<T> {
  const r = await fetch(url, {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!r.ok) throw new Error(`${url} -> ${r.status}`);
  return r.json() as Promise<T>;
}

export const api = {
  config: () => get<Config>("/api/config"),
  profile: (p: Payload) => post<Profile>("/api/profile", p),
  recommend: (p: Payload) => post<Recommendations>("/api/recommend", p),
  persona: (id: string, trust: number, adversarial: boolean) =>
    get<PersonaDetail>(`/api/personas/${id}?trust=${trust}&adversarial=${adversarial}`),
  personaCurve: (id: string, adversarial: boolean) =>
    get<TrustCurve>(`/api/personas/${id}/curve?adversarial=${adversarial}`),
  overview: () => get<Overview>("/api/results/overview"),
  scenarios: (dataset: string, metric: string) =>
    get<ScenarioGrid>(`/api/results/scenarios?dataset=${dataset}&metric=${metric}`),
  held: () => get<{ rows: HeldRow[] }>("/api/results/held"),
  trust: () => get<TrustAblation>("/api/results/trust"),
  tasteReal: () => get<{ rows: OverviewRow[] }>("/api/results/taste-real"),

  // ---- library ----
  historySummary: (w: Window) => {
    const q = win(w);
    return get<HistorySummary>(`/api/history/summary${q ? `?${q}` : ""}`);
  },
  historyTop: (w: Window, entity: string, by: string, limit: number, offset: number) => {
    const p = new URLSearchParams(win(w));
    p.set("entity", entity); p.set("by", by);
    p.set("limit", String(limit)); p.set("offset", String(offset));
    return get<TopItems>(`/api/history/top?${p.toString()}`);
  },
  historyTopCsvUrl: (w: Window, entity: string, by: string) => {
    const p = new URLSearchParams(win(w));
    p.set("entity", entity); p.set("by", by); p.set("format", "csv");
    return `/api/history/top?${p.toString()}`;
  },
  historyTrends: (w: Window, metric: string, granularity: string, rolling: number | null) => {
    const p = new URLSearchParams(win(w));
    p.set("metric", metric); p.set("granularity", granularity);
    if (rolling) p.set("rolling", String(rolling));
    return get<Trends>(`/api/history/trends?${p.toString()}`);
  },
  historyTrendsCsvUrl: (w: Window, metric: string, granularity: string, rolling: number | null) => {
    const p = new URLSearchParams(win(w));
    p.set("metric", metric); p.set("granularity", granularity);
    if (rolling) p.set("rolling", String(rolling));
    p.set("format", "csv");
    return `/api/history/trends?${p.toString()}`;
  },
  historyClock: (w: Window) => {
    const q = win(w);
    return get<Clock>(`/api/history/clock${q ? `?${q}` : ""}`);
  },
  historyAxes: (w: Window, granularity: string) => {
    const p = new URLSearchParams(win(w));
    p.set("granularity", granularity);
    return get<AxesOverTime>(`/api/history/axes?${p.toString()}`);
  },
  historyGenres: (w: Window) => {
    const q = win(w);
    return get<GenreMix>(`/api/history/genres${q ? `?${q}` : ""}`);
  },
};
