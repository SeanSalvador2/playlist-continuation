import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// The built app is served statically by the FastAPI backend at "/", so assets
// use relative paths. In dev, /api is proxied to the backend on :8000.
export default defineConfig({
  plugins: [react()],
  base: "./",
  server: {
    port: 5173,
    proxy: {
      "/api": "http://127.0.0.1:8000",
    },
  },
  build: {
    outDir: "dist",
    sourcemap: false,
  },
});
