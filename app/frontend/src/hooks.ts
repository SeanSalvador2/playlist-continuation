import { useEffect, useRef, useState } from "react";

/** Debounce a rapidly-changing value (e.g. a slider) before it triggers fetches. */
export function useDebounced<T>(value: T, delay = 180): T {
  const [v, setV] = useState(value);
  useEffect(() => {
    const t = setTimeout(() => setV(value), delay);
    return () => clearTimeout(t);
  }, [value, delay]);
  return v;
}

/** Stable ref to the latest value (avoids stale closures in async handlers). */
export function useLatest<T>(value: T) {
  const ref = useRef(value);
  ref.current = value;
  return ref;
}
