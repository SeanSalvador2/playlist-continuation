import type { Weight } from "../api";

function label(a: string): string {
  return a.startsWith("genre:") ? a.split(":")[1] : a.replace("_", " ");
}

/** Signed, bipolar bars for the top-N axes of a preference vector. */
export function WeightBars({ weights, color, n = 6 }: {
  weights: Weight[]; color: string; n?: number;
}) {
  const top = [...weights]
    .filter((w) => Math.abs(w.value) > 1e-4)
    .sort((a, b) => Math.abs(b.value) - Math.abs(a.value))
    .slice(0, n);
  if (!top.length) return <p className="hint">No stated axes.</p>;
  const max = Math.max(...top.map((w) => Math.abs(w.value)), 1e-6);
  return (
    <div className="weight-rows">
      {top.map((w) => {
        const width = (Math.abs(w.value) / max) * 50;
        const pos = w.value >= 0;
        return (
          <div className="weight-row" key={w.axis}>
            <span className="k" title={label(w.axis)}>{label(w.axis)}</span>
            <span className="wbar-track">
              <span className="mid" />
              <span className="fill" style={pos
                ? { left: "50%", width: `${width}%`, background: color }
                : { right: "50%", width: `${width}%`, background: color, opacity: 0.7 }} />
            </span>
            <span className="v">{w.value >= 0 ? "+" : ""}{w.value.toFixed(2)}</span>
          </div>
        );
      })}
    </div>
  );
}
