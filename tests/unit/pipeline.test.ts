import { renderMarkdownDocument } from "@/content/markdown/pipeline";

describe("markdown pipeline", () => {
  it("falls back to h2 title when h1 is missing", async () => {
    const result = await renderMarkdownDocument({
      sourceId: "local-docs",
      sourcePath: "example.md",
      fullPath: "/tmp/example.md",
      content: "## Secondary Title\n\nParagraph body"
    });

    expect(result.ok).toBe(true);
    if (!result.ok) {
      return;
    }
    expect(result.value.title).toBe("Secondary Title");
    expect(result.value.readingMinutes).toBe(1);
    expect(result.value.slug).toEqual(["example"]);
  });

  it("removes unsafe html and reports removal count", async () => {
    const result = await renderMarkdownDocument({
      sourceId: "local-docs",
      sourcePath: "xss.md",
      fullPath: "/tmp/xss.md",
      content: "# Safe\n\n<script>alert('x')</script><div onclick=\"evil()\">ok</div>"
    });

    expect(result.ok).toBe(true);
    if (!result.ok) {
      return;
    }

    expect(result.value.html).not.toContain("<script>");
    expect(result.value.invalidHtmlRemoved).toBeGreaterThanOrEqual(1);
  });
});
