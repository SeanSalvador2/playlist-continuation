import { expect, test } from "@playwright/test";

test.describe("Library", () => {
  test("loads, renders summary cards, tabs switch the table, custom window narrows", async ({ page }) => {
    await page.goto("/");
    await page.getByRole("button", { name: "Library" }).click();
    await page.getByRole("heading", { name: /Your listening, mapped honestly/i }).waitFor();

    // provenance badge for the default synthetic demo listener
    await expect(page.getByText(/synthetic demo listener/i)).toBeVisible();

    // summary stat cards render with numbers
    const stats = page.locator(".stat");
    await expect(stats).toHaveCount(4);
    const playsCard = stats.filter({ has: page.getByText("Plays", { exact: true }) });
    await expect(playsCard).toBeVisible();
    await expect(stats.filter({ has: page.getByText("Artists", { exact: true }) })).toBeVisible();

    // top table renders tracks (names look like track names)
    const firstRowName = page.locator("table.lib-table tbody tr").first().locator("td").nth(1);
    await expect(firstRowName).toBeVisible({ timeout: 15_000 });
    await expect(firstRowName).toContainText(/Track/i);

    // switching entity tab to Artists changes the table content
    await page.getByRole("button", { name: "Artists", exact: true }).click();
    await expect(firstRowName).toContainText(/Artist/i);

    // the trends line chart and listening clock render
    await expect(page.locator("svg[aria-label*='Line chart']").first()).toBeVisible();
    await expect(page.locator("svg[aria-label*='Listening clock']")).toBeVisible();

    // custom window narrows results: read all-time plays, then clamp to one month
    const playsNum = playsCard.locator(".num");
    const allTime = parseInt((await playsNum.textContent())!.replace(/[^0-9]/g, ""), 10);

    await page.getByLabel("Window start date").fill("2023-12-01");
    await page.getByLabel("Window end date").fill("2023-12-31");
    await expect
      .poll(async () => parseInt((await playsNum.textContent())!.replace(/[^0-9]/g, ""), 10), { timeout: 10_000 })
      .toBeLessThan(allTime);
  });
});
