import { test, expect } from "@playwright/test";
import fs from "node:fs";
import { createRequire } from "node:module";

const require = createRequire(import.meta.url);
const axeSource: string = fs.readFileSync(require.resolve("axe-core"), "utf8");

async function audit(page: import("@playwright/test").Page, label: string) {
  await page.evaluate(axeSource);
  const results = await page.evaluate(async () => {
    // @ts-expect-error injected global
    return await window.axe.run(document, {
      runOnly: { type: "tag", values: ["wcag2a", "wcag2aa", "wcag21a", "wcag21aa"] },
    });
  });
  const violations = (results as { violations: { id: string; impact: string; nodes: unknown[] }[] }).violations;
  console.log(`\n[a11y:${label}] ${violations.length} violation types`);
  for (const v of violations) console.log(`  - ${v.impact}: ${v.id} (${v.nodes.length})`);
  return violations;
}

test("axe-core audit across all views (light + dark)", async ({ page }) => {
  await page.goto("/");
  await page.getByRole("heading", { name: /Chart a listener/i }).waitFor();
  await page.getByRole("button", { name: /beast mode/i }).click();
  await page.locator(".lab-recs .rec").first().waitFor();
  const lab = await audit(page, "taste-lab");

  await page.getByRole("button", { name: "Personas" }).click();
  await page.getByRole("button", { name: /Slow, sad country/i }).click();
  await page.getByText(/Stated vs learned/i).first().waitFor();
  const personas = await audit(page, "personas");

  await page.getByRole("button", { name: "Results Explorer" }).click();
  await page.locator("table.held tbody tr").first().waitFor();
  const results = await audit(page, "results");

  // dark theme
  await page.getByRole("button", { name: /Switch to dark theme/i }).click();
  await page.waitForTimeout(300);
  const dark = await audit(page, "results-dark");

  const all = [...lab, ...personas, ...results, ...dark];
  const serious = all.filter((v) => v.impact === "serious" || v.impact === "critical");
  expect(serious, JSON.stringify(serious.map((v) => `${v.id}:${v.impact}`))).toHaveLength(0);
});
