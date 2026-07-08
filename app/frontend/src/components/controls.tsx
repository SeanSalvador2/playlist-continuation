import type { AxisMeta, SamplePlaylist } from "../api";

export function AxisSlider({
  meta, value, onChange,
}: {
  meta: AxisMeta; value: number; onChange: (v: number) => void;
}) {
  const pct = Math.round(value * 100);
  return (
    <div className="slider-row">
      <div className="slider-head">
        <span className="slider-label">{meta.label}</span>
        <span className="slider-val">{value > 0 ? "+" : ""}{value.toFixed(2)}</span>
      </div>
      <input
        type="range" min={-1} max={1} step={0.05} value={value}
        onChange={(e) => onChange(parseFloat(e.target.value))}
        aria-label={`${meta.label}: ${meta.low} to ${meta.high}, currently ${pct}%`}
      />
      <div className="slider-poles"><span>{meta.low}</span><span>{meta.high}</span></div>
    </div>
  );
}

export function GenreChips({
  genres, selected, onToggle,
}: {
  genres: string[]; selected: string[]; onToggle: (g: string) => void;
}) {
  return (
    <div className="chips" role="group" aria-label="Genre flavours">
      {genres.map((g) => (
        <button key={g} className="chip" aria-pressed={selected.includes(g)}
                onClick={() => onToggle(g)}>{g}</button>
      ))}
    </div>
  );
}

export function TrustDial({ value, onChange }: { value: number; onChange: (v: number) => void }) {
  const pct = Math.round(value * 100);
  return (
    <div className="trust-card">
      <div className="trust-readout">
        <div className="pct">{pct}%</div>
        <div className="lbl">weight on what you <b>said</b></div>
      </div>
      <input
        type="range" min={0} max={1} step={0.05} value={value}
        onChange={(e) => onChange(parseFloat(e.target.value))}
        aria-label={`Trust dial: ${pct}% stated preference, ${100 - pct}% learned from tracks`}
      />
      <div className="trust-ends">
        <span><b>learned</b><br />from tracks</span>
        <span style={{ textAlign: "right" }}><b>stated</b><br />by you</span>
      </div>
    </div>
  );
}

export function SeedPicker({
  playlists, selectedPid, onPick,
}: {
  playlists: SamplePlaylist[]; selectedPid: number | null; onPick: (p: SamplePlaylist | null) => void;
}) {
  return (
    <div className="seed-list">
      <button className="seed-item" aria-pressed={selectedPid === null}
              onClick={() => onPick(null)}>
        <span className="seed-name">No seed playlist</span>
        <span className="seed-meta">stated preferences only</span>
      </button>
      {playlists.map((p) => (
        <button key={p.pid} className="seed-item" aria-pressed={selectedPid === p.pid}
                onClick={() => onPick(p)}>
          <span className="seed-name">{p.name || "untitled"}</span>
          <span className="seed-meta">{p.archetype} · {p.n_tracks} tracks</span>
        </button>
      ))}
    </div>
  );
}
