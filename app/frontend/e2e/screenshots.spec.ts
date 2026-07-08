import { test } from "@playwright/test";
import path from "node:path";

// Captures the polished screenshots embedded in the README / visualizations.
const OUT = path.resolve(process.cwd(), "../screenshots");

test.use({ viewport: { width: 1480, height: 1000 }, deviceScaleFactor: 2 });

test("capture atlas screenshots", async ({ page }) => {
  await page.goto("/");
  await page.getByRole("heading", { name: /Chart a listener/i }).waitFor();
  await page.getByRole("button", { name: /beast mode/i }).click();
  await page.locator(".lab-recs .rec").first().waitFor();
  await page.waitForTimeout(700);
  await page.screenshot({ path: path.join(OUT, "01-taste-lab.png"), fullPage: false });

  // explanation chips close-up: crop to a recommendation card
  const rec = page.locator(".lab-recs .rec").first();
  await rec.screenshot({ path: path.join(OUT, "02-explanation-chips.png") });

  // full taste lab (map + flavours + recs)
  await page.screenshot({ path: path.join(OUT, "03-taste-lab-full.png"), fullPage: true });

  // personas — adversarial rescue
  await page.getByRole("button", { name: "Personas" }).click();
  await page.getByRole("button", { name: /Slow, sad country/i }).click();
  await page.getByText(/Stated vs learned/i).first().waitFor();
  await page.getByRole("button", { name: "Wrong prefs", exact: true }).click();
  const trust = page.getByLabel(/^Trust \d+ percent$/);
  await trust.focus();
  for (let i = 0; i < 16; i++) await trust.press("ArrowRight");
  await page.waitForTimeout(700);
  await page.screenshot({ path: path.join(OUT, "04-personas-adversarial.png"), fullPage: true });

  // results explorer (dark to show theme-awareness)
  await page.getByRole("button", { name: "Results Explorer" }).click();
  await page.locator("table.held tbody tr").first().waitFor();
  await page.waitForTimeout(500);
  await page.screenshot({ path: path.join(OUT, "05-results-explorer.png"), fullPage: true });

  await page.getByRole("button", { name: /Switch to dark theme/i }).click();
  await page.waitForTimeout(400);
  await page.screenshot({ path: path.join(OUT, "06-results-dark.png"), fullPage: false });
});
