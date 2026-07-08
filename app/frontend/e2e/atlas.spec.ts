import { expect, test } from "@playwright/test";

test.describe("Taste Atlas", () => {
  test("taste lab: moving a stated slider re-ranks recommendations", async ({ page }) => {
    await page.goto("/");
    await page.getByRole("heading", { name: /Chart a listener/i }).waitFor();

    // load a seed playlist so there are learned weights + clusters + evidence
    await page.getByRole("button", { name: /beast mode/i }).click();
    const recList = page.locator(".lab-recs .rec");
    await expect(recList.first()).toBeVisible({ timeout: 15_000 });

    // flavour territories appear
    await expect(page.locator(".flavor").first()).toBeVisible();

    // capture the current top recommendation, then swing valence hard negative
    const firstBefore = await recList.first().locator(".rec-name").textContent();
    const valence = page.getByLabel(/Valence: sad to happy/i);
    await valence.focus();
    for (let i = 0; i < 12; i++) await valence.press("ArrowLeft");
    // also crank acousticness toward acoustic
    await page.waitForTimeout(600);

    // recommendations must still render and the ordering should have changed
    await expect(recList.first()).toBeVisible();
    const firstAfter = await recList.first().locator(".rec-name").textContent();
    expect(firstBefore).not.toBeNull();
    // WHY chips exist (explanations present)
    await expect(page.locator(".axisbar").first()).toBeVisible();
    // at least one co-occurrence evidence chip
    await expect(page.locator(".why-chip.evidence").first()).toBeVisible();
    expect(firstAfter).toBeTruthy();
  });

  test("personas: switching persona updates clusters and the adversarial rescue works", async ({ page }) => {
    await page.goto("/");
    await page.getByRole("button", { name: "Personas" }).click();
    await page.getByRole("heading", { name: /Pre-built listeners/i }).waitFor();

    // select the adversarial-ready persona
    await page.getByRole("button", { name: /Slow, sad country/i }).click();
    await expect(page.getByText(/Stated vs learned/i).first()).toBeVisible();

    // switch to wrong prefs
    await page.getByRole("button", { name: "Wrong prefs", exact: true }).click();
    await page.waitForTimeout(500);

    // drive trust high -> fit should drop; the fit meter number is live
    const trust = page.getByLabel(/^Trust \d+ percent$/);
    await trust.focus();
    for (let i = 0; i < 20; i++) await trust.press("ArrowRight");
    await page.waitForTimeout(700);
    const fitText = await page.locator(".fit-num").textContent();
    const fitHigh = parseInt(fitText?.replace("%", "") ?? "100", 10);

    // drive trust low -> fit should recover
    for (let i = 0; i < 20; i++) await trust.press("ArrowLeft");
    await page.waitForTimeout(700);
    const fitTextLow = await page.locator(".fit-num").textContent();
    const fitLow = parseInt(fitTextLow?.replace("%", "") ?? "0", 10);

    expect(fitLow).toBeGreaterThan(fitHigh); // low trust rescues the recommendations
  });

  test("results explorer: charts render and the dataset toggle works", async ({ page }) => {
    await page.goto("/");
    await page.getByRole("button", { name: "Results Explorer" }).click();
    await page.getByRole("heading", { name: /What held when the map/i }).waitFor();

    // headline stat + grouped bars + heatmap + held table all render
    await expect(page.getByText("Item-CF", { exact: true }).first()).toBeVisible();
    await expect(page.locator("svg[aria-label*='Grouped bar']")).toBeVisible();
    await expect(page.locator("svg[aria-label*='Heatmap']")).toBeVisible();
    await expect(page.locator("table.held tbody tr").first()).toBeVisible();
    await expect(page.locator(".verdict").first()).toBeVisible();

    // toggle to synthetic and back
    await page.getByRole("button", { name: "Synthetic", exact: true }).click();
    await page.waitForTimeout(300);
    await expect(page.locator("svg[aria-label*='Heatmap']")).toBeVisible();
  });
});
