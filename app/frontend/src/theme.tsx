import {
  createContext, useCallback, useContext, useEffect, useMemo, useState,
} from "react";

export type Mode = "light" | "dark";

interface ThemeCtx { mode: Mode; toggle: () => void }
const Ctx = createContext<ThemeCtx>({ mode: "light", toggle: () => {} });

function initialMode(): Mode {
  const saved = localStorage.getItem("atlas-theme");
  if (saved === "light" || saved === "dark") return saved;
  return window.matchMedia?.("(prefers-color-scheme: dark)").matches ? "dark" : "light";
}

export function ThemeProvider({ children }: { children: React.ReactNode }) {
  const [mode, setMode] = useState<Mode>(initialMode);
  useEffect(() => {
    document.documentElement.setAttribute("data-theme", mode);
    localStorage.setItem("atlas-theme", mode);
  }, [mode]);
  const toggle = useCallback(() => setMode((m) => (m === "light" ? "dark" : "light")), []);
  const value = useMemo(() => ({ mode, toggle }), [mode, toggle]);
  return <Ctx.Provider value={value}>{children}</Ctx.Provider>;
}

export const useTheme = () => useContext(Ctx);
