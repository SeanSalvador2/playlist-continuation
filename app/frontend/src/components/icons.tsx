// Small inline SVG icons. Decorative marks are aria-hidden; the brand mark is a
// contour/compass glyph echoing the cartographic identity.

export function BrandMark({ size = 34 }: { size?: number }) {
  return (
    <svg width={size} height={size} viewBox="0 0 34 34" fill="none" aria-hidden="true" className="brand-mark">
      <rect x="1" y="1" width="32" height="32" rx="8" fill="var(--accent)" />
      {/* contour rings */}
      <circle cx="17" cy="17" r="11" stroke="var(--accent-ink)" strokeOpacity="0.35" strokeWidth="1.4" fill="none" />
      <circle cx="17" cy="17" r="7" stroke="var(--accent-ink)" strokeOpacity="0.55" strokeWidth="1.4" fill="none" />
      {/* compass needle */}
      <path d="M17 7 L20 17 L17 27 L14 17 Z" fill="var(--accent-ink)" fillOpacity="0.92" />
      <circle cx="17" cy="17" r="1.9" fill="var(--accent)" stroke="var(--accent-ink)" strokeWidth="1.1" />
    </svg>
  );
}

export function SunIcon() {
  return (
    <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" aria-hidden="true">
      <circle cx="12" cy="12" r="4" />
      <path d="M12 2v2M12 20v2M2 12h2M20 12h2M4.9 4.9l1.4 1.4M17.7 17.7l1.4 1.4M19.1 4.9l-1.4 1.4M6.3 17.7l-1.4 1.4" />
    </svg>
  );
}

export function MoonIcon() {
  return (
    <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
      <path d="M21 12.8A9 9 0 1 1 11.2 3a7 7 0 0 0 9.8 9.8z" />
    </svg>
  );
}

export function LinkIcon() {
  return (
    <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" aria-hidden="true">
      <path d="M10 13a5 5 0 0 0 7 0l3-3a5 5 0 0 0-7-7l-1 1M14 11a5 5 0 0 0-7 0l-3 3a5 5 0 0 0 7 7l1-1" />
    </svg>
  );
}
