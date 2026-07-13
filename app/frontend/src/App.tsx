import { useEffect, useState } from "react";
import { api, type Config } from "./api";
import { useTheme } from "./theme";
import { BrandMark, MoonIcon, SunIcon } from "./components/icons";
import { TasteLab } from "./components/TasteLab";
import { Personas } from "./components/Personas";
import { Results } from "./components/Results";
import { Library } from "./components/Library";
import { Journey } from "./components/Journey";

type View = "lab" | "personas" | "results" | "library" | "journey";
const VIEWS: { id: View; label: string }[] = [
  { id: "lab", label: "Taste Lab" },
  { id: "personas", label: "Personas" },
  { id: "library", label: "Library" },
  { id: "journey", label: "Journey" },
  { id: "results", label: "Results Explorer" },
];

export default function App() {
  const { mode, toggle } = useTheme();
  const [view, setView] = useState<View>("lab");
  const [config, setConfig] = useState<Config | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    api.config().then(setConfig).catch((e) => setError(String(e)));
  }, []);

  return (
    <>
      <a href="#main" className="skip-link">Skip to content</a>
      <header className="topbar">
        <div className="brand">
          <BrandMark />
          <div>
            <div className="brand-name">Taste Atlas</div>
            <div className="brand-sub">interpretable taste · MPD</div>
          </div>
        </div>
        <nav className="nav" aria-label="Views">
          {VIEWS.map((v) => (
            <button
              key={v.id}
              className="nav-btn"
              aria-current={view === v.id ? "page" : undefined}
              onClick={() => setView(v.id)}
            >
              {v.label}
            </button>
          ))}
        </nav>
        <div className="topbar-spacer" />
        <button
          className="icon-btn"
          onClick={toggle}
          aria-label={`Switch to ${mode === "light" ? "dark" : "light"} theme`}
        >
          {mode === "light" ? <MoonIcon /> : <SunIcon />}
        </button>
      </header>

      <main id="main">
        {error && <div className="page"><div className="card card-pad">Could not reach the API: {error}</div></div>}
        {!error && !config && <div className="page"><div className="center-empty">Surveying the taste terrain…</div></div>}
        {config && view === "lab" && <TasteLab config={config} />}
        {config && view === "personas" && <Personas config={config} />}
        {config && view === "library" && <Library config={config} />}
        {config && view === "journey" && <Journey />}
        {config && view === "results" && <Results />}
      </main>
    </>
  );
}
