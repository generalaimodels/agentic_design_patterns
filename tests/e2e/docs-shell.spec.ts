import { expect, test } from "@playwright/test";

test("renders three-panel docs experience", async ({ page, isMobile }) => {
  test.skip(isMobile, "Desktop layout validation only");

  await page.goto("/");
  await expect(page.locator(".doc-shell__article")).toBeVisible();
  await expect(page.locator(".left-rail")).toBeVisible();
  await expect(page.locator(".right-rail")).toBeVisible();
  await expect(page.locator(".markdown-body h1").first()).toBeVisible();
});

test("mobile drawers open and close", async ({ page, isMobile }) => {
  test.skip(!isMobile, "Mobile behavior validation only");

  await page.goto("/");
  await page.getByRole("button", { name: "Contents" }).click();
  await expect(page.locator(".left-rail.left-rail--open")).toBeVisible();

  await page.getByRole("button", { name: "Search" }).click();
  await expect(page.locator(".doc-shell__right-wrap--open")).toBeVisible();
});
