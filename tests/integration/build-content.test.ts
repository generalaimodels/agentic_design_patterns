import { buildArtifacts } from "@/content/build-content";

describe("build-content integration", () => {
  it("builds docs, placeholders, search index, and redirects", async () => {
    const result = await buildArtifacts();
    expect(result.ok).toBe(true);

    if (!result.ok) {
      return;
    }

    expect(result.value.docs.length).toBeGreaterThanOrEqual(35);
    expect(result.value.docs.some((doc) => doc.sourceId === "generated-placeholder")).toBe(true);
    expect(result.value.navigation.chapters.length).toBeGreaterThanOrEqual(26);
    expect(Object.keys(result.value.searchIndex.termPostings).length).toBeGreaterThan(0);
    expect(Object.keys(result.value.redirects).length).toBeGreaterThan(0);
  });
});
