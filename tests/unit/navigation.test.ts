import { promises as fs } from "node:fs";
import os from "node:os";
import path from "node:path";
import { buildNavigation } from "@/content/nav/build-nav";
import type { BuildError } from "@/content/types";

describe("navigation builder", () => {
  it("parses chapter index and emits placeholders for missing chapters", async () => {
    const tempDir = await fs.mkdtemp(path.join(os.tmpdir(), "nav-test-"));
    const indexPath = path.join(tempDir, "README.md");

    await fs.writeFile(
      indexPath,
      [
        "### Chapter 1: Routing",
        "### Chapter 2: Parallelization",
        "### Chapter 3: Reflection"
      ].join("\n"),
      "utf8"
    );

    const docs = [
      {
        id: "a",
        sourceId: "local",
        sourcePath: "Routing/routing.md",
        title: "Chapter 1: Routing",
        slug: ["routing", "routing"],
        canonicalPath: "/docs/routing/routing",
        legacyPaths: [],
        chapterNumber: 1,
        headings: [],
        excerpt: "",
        html: "",
        text: "",
        readingMinutes: 1,
        media: []
      }
    ];

    const result = await buildNavigation(docs, indexPath);
    expect(result.ok).toBe(true);
    if (!result.ok) {
      return;
    }

    expect(result.value.navigation.chapters).toHaveLength(3);
    expect(result.value.navigation.chapters[0].canonicalPath).toBe("/docs/routing/routing");
    expect(result.value.missingChapters.map((item) => item.chapterNumber)).toEqual([2, 3]);
  });

  it("enforces exhaustive build error handling", () => {
    const describeError = (error: BuildError): string => {
      switch (error.type) {
        case "SOURCE_FETCH_FAILED":
        case "MARKDOWN_PARSE_FAILED":
        case "INVALID_HTML_REMOVED":
        case "SLUG_COLLISION":
        case "MISSING_REQUIRED_SOURCE":
        case "ASSET_RESOLUTION_FAILED":
          return error.type;
        default: {
          const exhaustive: never = error;
          return exhaustive;
        }
      }
    };

    expect(
      describeError({
        type: "MISSING_REQUIRED_SOURCE",
        sourceId: "x"
      })
    ).toBe("MISSING_REQUIRED_SOURCE");
  });
});
