import { useEffect, useMemo, useState } from "react";
import {
  api, type Config, type Profile, type Recommendations, type SamplePlaylist,
} from "../api";
import { useDebounced } from "../hooks";
import { AxisSlider, GenreChips, SeedPicker, TrustDial } from "./controls";
import { TasteMap } from "./TasteMap";
import { FlavorCards } from "./FlavorCards";
import { RecCard } from "./RecCard";

export function TasteLab({ config }: { config: Config }) {
  const scalarAxes = config.axes.scalar;
  const [scalars, setScalars] = useState<Record<string, number>>(
    () => Object.fromEntries(scalarAxes.map((a) => [a.key, 0])),
  );
  const [genres, setGenres] = useState<string[]>([]);
  const [seed, setSeed] = useState<SamplePlaylist | null>(null);
  const [trust, setTrust] = useState(0.4);

  const [profile, setProfile] = useState<Profile | null>(null);
  const [recs, setRecs] = useState<Recommendations | null>(null);
  const [loading, setLoading] = useState(false);

  const payload = useMemo(() => ({
    scalars, genres, seed_uris: seed?.seed_uris ?? [], trust,
  }), [scalars, genres, seed, trust]);
  const debounced = useDebounced(payload, 160);

  useEffect(() => {
    let alive = true;
    setLoading(true);
    Promise.all([
      api.profile(debounced),
      api.recommend({ ...debounced, k: 24 }),
    ]).then(([p, r]) => {
      if (!alive) return;
      setProfile(p); setRecs(r); setLoading(false);
    }).catch(() => alive && setLoading(false));
    return () => { alive = false; };
  }, [debounced]);

  const toggleGenre = (g: string) =>
    setGenres((cur) => cur.includes(g) ? cur.filter((x) => x !== g) : [...cur, g]);

  const clusters = profile?.clusters ?? [];

  return (
    <div className="page reveal">
      <div className="page-head">
        <div className="eyebrow">Taste Lab</div>
        <h1 className="page-title">Chart a listener’s taste, one axis at a time.</h1>
        <p className="page-lede">
          State what you value on human axes, blend it with what a seed playlist reveals,
          and watch every recommendation re-rank live — each one explained by the axes it
          matches and the tracks it travels with.
        </p>
      </div>

      <div className="lab-grid">
        {/* ---- controls rail ---- */}
        <div className="rail">
          <section className="card card-pad" aria-label="Stated preferences">
            <div className="card-title">Stated preferences</div>
            {scalarAxes.map((a) => (
              <AxisSlider key={a.key} meta={a} value={scalars[a.key]}
                          onChange={(v) => setScalars((s) => ({ ...s, [a.key]: v }))} />
            ))}
            <div style={{ height: 12 }} />
            <div className="card-title" style={{ marginBottom: 8 }}>Genre flavours</div>
            <GenreChips genres={config.axes.genres} selected={genres} onToggle={toggleGenre} />
          </section>

          <section className="card card-pad" aria-label="Trust dial">
            <div className="card-title">Trust dial <span className="hint">stated ↔ learned</span></div>
            <TrustDial value={trust} onChange={setTrust} />
          </section>

          <section className="card card-pad" aria-label="Seed playlist">
            <div className="card-title">Seed playlist</div>
            <SeedPicker playlists={config.sample_playlists}
                        selectedPid={seed?.pid ?? null} onPick={setSeed} />
          </section>
        </div>

        {/* ---- centre: map + flavours ---- */}
        <div className="stack">
          <section className="card card-pad">
            <div className="card-title">
              The taste map
              <span className="hint" aria-live="polite">{loading ? "re-surveying…" : " "}</span>
            </div>
            <TasteMap catalog={config.catalog} clusters={clusters}
                      recs={recs?.items ?? []} axesMeta={scalarAxes} />
          </section>

          <section className="card card-pad">
            <div className="card-title">Flavour territories
              <span className="hint">named k-means centroids</span></div>
            <FlavorCards clusters={clusters} />
          </section>
        </div>

        {/* ---- recommendations ---- */}
        <div className="lab-recs">
          <section className="card card-pad" style={{ position: "sticky", top: 84 }}>
            <div className="card-title">
              Recommendations
              <span className="hint">{recs?.items.length ?? 0} · every one explained</span>
            </div>
            <div className="recs">
              {(recs?.items ?? []).map((r) => (
                <RecCard key={r.uri} rec={r} clusters={clusters} />
              ))}
              {!recs?.items.length && <p className="hint">No recommendations yet.</p>}
            </div>
          </section>
        </div>
      </div>
    </div>
  );
}
