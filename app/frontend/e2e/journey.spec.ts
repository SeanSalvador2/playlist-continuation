import { expect, test } from "@playwright/test";

test.describe("Journey", () => {
  test("view loads, era cards render, and the slideshow advances", async ({ page }) => {
    await page.goto("/");
    await page.getByRole("button", { name: "Journey" }).click();
    await page.getByRole("heading", { level: 1, name: /The story of your taste/i }).waitFor();

    // demo banner for the synthetic listener
    await expect(page.getByText(/synthetic demo listener/i)).toBeVisible();

    // the trajectory chart renders with its PC captions
    await expect(page.locator("svg[aria-label*='Taste trajectory']")).toBeVisible({ timeout: 15_000 });
    await expect(page.getByText(/PC1 ≈/i).first()).toBeVisible();

    // era cards render (the seed-7 demo has five)
    const eras = page.locator(".era-card");
    await expect(eras.first()).toBeVisible();
    const count = await eras.count();
    expect(count).toBeGreaterThanOrEqual(2);
    await expect(eras.first().locator(".era-name")).not.toBeEmpty();

    // the story slideshow advances via Next
    const title = page.locator(".story-title");
    const first = (await title.textContent()) ?? "";
    await page.getByRole("button", { name: /Next/ }).click();
    await expect
      .poll(async () => (await title.textContent()) ?? "", { timeout: 10_000 })
      .not.toBe(first);

    // Prev returns to the first slide
    await page.getByRole("button", { name: /Prev/ }).click();
    await expect(title).toHaveText(first);

    // the honest, machine-verified footer is present
    await expect(page.getByText(/traced back to a fact before the slide/i)).toBeVisible();

    // toggling planted markers (demo diagnostics) does not break the chart
    await page.getByRole("button", { name: /planted markers/i }).click();
    await expect(page.locator("svg[aria-label*='Taste trajectory']")).toBeVisible();
  });

  test("axe-core audit of the Journey view (light + dark)", async ({ page }) => {
    const fs = await import("node:fs");
    const { createRequire } = await import("node:module");
    const require = createRequire(import.meta.url);
    const axeSource: string = fs.readFileSync(require.resolve("axe-core"), "utf8");

    async function audit(label: string) {
      await page.evaluate(axeSource);
      const results = await page.evaluate(async () => {
        // @ts-expect-error injected global
        return await window.axe.run(document, {
          runOnly: { type: "tag", values: ["wcag2a", "wcag2aa", "wcag21a", "wcag21aa"] },
        });
      });
      const violations = (results as { violations: { id: string; impact: string }[] }).violations;
      console.log(`\n[a11y:${label}] ${violations.length} violation types`);
      for (const v of violations) console.log(`  - ${v.impact}: ${v.id}`);
      return violations;
    }

    await page.goto("/");
    await page.getByRole("button", { name: "Journey" }).click();
    await page.getByRole("heading", { level: 1, name: /The story of your taste/i }).waitFor();
    await page.locator(".era-card").first().waitFor();
    const light = await audit("journey-light");

    await page.getByRole("button", { name: /Switch to dark theme/i }).click();
    await page.waitForTimeout(300);
    const dark = await audit("journey-dark");

    const serious = [...light, ...dark].filter((v) => v.impact === "serious" || v.impact === "critical");
    expect(serious, JSON.stringify(serious.map((v) => `${v.id}:${v.impact}`))).toHaveLength(0);
  });
});
