import { expect, test } from "@playwright/test";

test("perf smoke: initial load and interaction budget", async ({ page }) => {
  const start = Date.now();
  await page.goto("/", { waitUntil: "networkidle" });
  const loadMs = Date.now() - start;

  await expect(page.locator(".doc-shell__article")).toBeVisible();
  expect(loadMs).toBeLessThan(8000);

  const searchInput = page.locator(".search-panel__input").first();
  await searchInput.fill("routing");
  await expect(page.locator(".search-panel__result").first()).toBeVisible();
});
