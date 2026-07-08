import type { Cluster } from "../api";
import { useTheme } from "../theme";
import { SERIES } from "../palette";

const SCALARS: { key: string; label: string }[] = [
  { key: "tempo", label: "tempo" },
  { key: "energy", label: "energy" },
  { key: "valence", label: "valence" },
  { key: "acousticness", label: "acoustic" },
  { key: "lyrical_depth", label: "lyrics" },
];

export function FlavorCards({ clusters }: { clusters: Cluster[] }) {
  const { mode } = useTheme();
  const palette = SERIES[mode];
  if (!clusters.length) {
    return <p className="hint">Pick a seed playlist to distil its flavour territories.</p>;
  }
  return (
    <div className="flavors">
      {clusters.map((c, i) => (
        <div className="flavor" key={i} style={{ ["--fl" as string]: palette[i % palette.length] }}>
          <div className="flavor-share">{Math.round(c.share * 100)}% of tracks</div>
          <div className="flavor-name">{c.name}</div>
          <div className="mini-axes">
            {SCALARS.map((s) => (
              <div className="mini-axis" key={s.key}>
                <span className="k">{s.label}</span>
                <span className="mini-bar">
                  <span style={{ width: `${Math.round((c.axes[s.key] ?? 0) * 100)}%` }} />
                </span>
              </div>
            ))}
          </div>
        </div>
      ))}
    </div>
  );
}
