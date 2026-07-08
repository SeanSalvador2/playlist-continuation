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
};
