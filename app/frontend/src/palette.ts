import type { Mode } from "./theme";

// Chart colours — the validated data-viz reference palette (light / dark steps).
// Chart marks use these; UI chrome uses the CSS-variable tokens in styles.css.
export const SERIES: Record<Mode, string[]> = {
  light: ["#2a78d6", "#1baf7a", "#eda100", "#008300", "#4a3aa7", "#e34948", "#e87ba4", "#eb6834"],
  dark: ["#3987e5", "#199e70", "#c98500", "#008300", "#9085e9", "#e66767", "#d55181", "#d95926"],
};

// Sequential blue ramp (for the scenario heatmap magnitude encoding).
export const SEQ_BLUE = [
  "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b",
];

// Diverging / paired accents for honest (cool) vs adversarial (warm) curves.
export const HONEST: Record<Mode, string> = { light: "#2a78d6", dark: "#3987e5" };
export const ADVERSARIAL: Record<Mode, string> = { light: "#e34948", dark: "#e66767" };

export function chartInk(mode: Mode) {
  return mode === "dark"
    ? { primary: "#f2f5f8", secondary: "#a9b4c2", muted: "#6e7a88", grid: "#26303b", axis: "#3a4552", surface: "#141b23" }
    : { primary: "#14181d", secondary: "#55606e", muted: "#8a94a2", grid: "#e3e6ec", axis: "#c7ccd6", surface: "#ffffff" };
}

// Map any genre to a stable categorical hue (used sparingly, for ≤3 flavour
// regions and their pins — never all 10 genres at once).
const GENRE_ORDER = [
  "country", "rap", "indie", "pop", "rock",
  "electronic", "folk", "metal", "rnb", "jazz",
];
export function genreHue(genre: string, mode: Mode): string {
  const i = GENRE_ORDER.indexOf(genre);
  const s = SERIES[mode];
  return s[(i < 0 ? 0 : i) % s.length];
}

// A short scale [0,1] -> sequential blue step, for heatmap cells.
export function seqStep(t: number): string {
  const clamped = Math.max(0, Math.min(1, t));
  const i = Math.round(clamped * (SEQ_BLUE.length - 1));
  return SEQ_BLUE[i];
}
