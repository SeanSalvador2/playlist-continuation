import { defineConfig, devices } from "@playwright/test";

// The pre-installed chromium in the sandbox; do NOT run `playwright install`.
const CHROMIUM = "/opt/pw-browsers/chromium-1194/chrome-linux/chrome";
const PORT = 8098;

export default defineConfig({
  testDir: "./e2e",
  timeout: 120_000,
  expect: { timeout: 30_000 },
  fullyParallel: false,
  workers: 1,
  reporter: [["list"]],
  use: {
    baseURL: `http://127.0.0.1:${PORT}`,
    trace: "retain-on-failure",
    launchOptions: { executablePath: CHROMIUM },
  },
  projects: [
    {
      name: "chromium",
      use: { ...devices["Desktop Chrome"], launchOptions: { executablePath: CHROMIUM } },
    },
  ],
  // Boots the FastAPI backend which also serves the built frontend (one port).
  webServer: {
    command: `python -m uvicorn app.backend.server:app --host 127.0.0.1 --port ${PORT}`,
    cwd: "../..",
    url: `http://127.0.0.1:${PORT}/api/health`,
    timeout: 90_000,
    reuseExistingServer: !process.env.CI,
  },
});
