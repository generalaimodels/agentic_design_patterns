import { fetchGitHubSource } from "@/content/sources/github";

describe("github source fetcher", () => {
  const originalFetch = global.fetch;

  afterEach(() => {
    global.fetch = originalFetch;
  });

  it("fetches markdown files from mocked github api", async () => {
    const responses: Record<string, unknown> = {
      "https://api.github.com/repos/octo/demo/contents/docs?ref=main": [
        { type: "file", path: "docs/a.md", name: "a.md", size: 10, content: Buffer.from("# A").toString("base64"), encoding: "base64" }
      ]
    };

    global.fetch = (async (url: string | URL) => {
      const body = responses[String(url)];
      if (!body) {
        return new Response("not found", { status: 404 });
      }
      return new Response(JSON.stringify(body), { status: 200 });
    }) as typeof fetch;

    const result = await fetchGitHubSource({
      kind: "github",
      id: "gh",
      owner: "octo",
      repo: "demo",
      ref: "main",
      paths: ["docs"],
      tokenEnv: "GITHUB_TOKEN",
      required: false
    });

    expect(result.ok).toBe(true);
    if (!result.ok) {
      return;
    }

    expect(result.value).toHaveLength(1);
    expect(result.value[0].content).toContain("# A");
  });
});
