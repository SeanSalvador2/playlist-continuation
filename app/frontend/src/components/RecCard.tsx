import { memo } from "react";
import type { Cluster, Rec } from "../api";
import { useTheme } from "../theme";
import { genreHue } from "../palette";

function axisLabel(a: string): string {
  return a.startsWith("genre:") ? a.split(":")[1] : a.replace("_", " ");
}

export const RecCard = memo(function RecCard({ rec }: { rec: Rec; clusters: Cluster[] }) {
  const { mode } = useTheme();
  const hue = genreHue(rec.genre, mode);
  const maxAbs = Math.max(...rec.axis_bars.map((b) => Math.abs(b.contribution)), 1e-6);

  return (
    <article className="rec" style={{ ["--fl" as string]: hue }}>
      <div className="rec-top">
        <span className="rec-rank">{String(rec.rank).padStart(2, "0")}</span>
        <span className="rec-name">{rec.track_name}</span>
        <span className="rec-genre">{rec.genre}</span>
      </div>
      <div className="rec-artist">{rec.artist_name}</div>

      <div className="why">
        <div className="axisbars" role="group" aria-label="Axis match">
          {rec.axis_bars.slice(0, 3).map((b) => {
            const w = (Math.abs(b.contribution) / maxAbs) * 50;
            const pos = b.contribution >= 0;
            return (
              <div className="axisbar" key={b.axis}>
                <span className="k" title={axisLabel(b.axis)}>{axisLabel(b.axis)}</span>
                <span className="axisbar-track">
                  <span className="mid" />
                  <span className={`fill ${pos ? "pos" : "neg"}`}
                        style={pos
                          ? { left: "50%", width: `${w}%` }
                          : { right: "50%", width: `${w}%` }} />
                </span>
                <span className="v">{b.contribution >= 0 ? "+" : ""}{b.contribution.toFixed(2)}</span>
              </div>
            );
          })}
        </div>
        <div className="why-chips">
          {rec.cooccurrence > 0 && (
            <span className="why-chip evidence">
              <span className="dot" />co-occurs with {rec.cooccurrence} seed{rec.cooccurrence > 1 ? "s" : ""}
            </span>
          )}
          {rec.same_artist && <span className="why-chip">same artist as a seed</span>}
          <span className="why-chip">fits “{shorten(rec.flavor)}”</span>
        </div>
      </div>
    </article>
  );
});

function shorten(s: string): string {
  return s.length > 42 ? s.slice(0, 40) + "…" : s;
}
