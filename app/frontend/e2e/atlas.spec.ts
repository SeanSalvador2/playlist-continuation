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

    // capture the current top-5 ordering, then raise trust and push
    // acousticness hard toward acoustic — the ranking must change
    const orderOf = async () => {
      const names = await recList.locator(".rec-name").allTextContents();
      return names.slice(0, 5).join(" | ");
    };
    const before = await orderOf();
    // explanations present while CF evidence is strong: axis bars + evidence chip
    await expect(page.locator(".axisbar").first()).toBeVisible();
    await expect(page.locator(".why-chip.evidence").first()).toBeVisible();

    const trust = page.getByRole("slider", { name: /^Trust dial:/ });
    await trust.focus();
    for (let i = 0; i < 14; i++) await trust.press("ArrowRight");
    const acoustic = page.getByLabel(/Acousticness: electronic to acoustic/i);
    await acoustic.focus();
    for (let i = 0; i < 20; i++) await acoustic.press("ArrowRight");
    await page.waitForTimeout(800);

    await expect(recList.first()).toBeVisible();
    await expect.poll(orderOf, { timeout: 10_000 }).not.toBe(before);
    // explanations still present after the re-rank
    await expect(page.locator(".axisbar").first()).toBeVisible();
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
