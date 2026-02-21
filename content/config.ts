import type { BuildConfig } from "./types";

export const buildConfig: BuildConfig = {
  sources: [
    {
      kind: "local",
      id: "local-docs",
      root: "docs",
      include: ["**/*.md"],
      exclude: ["**/node_modules/**", "**/.git/**"],
      required: true,
      maxFileBytes: 2 * 1024 * 1024
    },
    {
      kind: "github",
      id: "github-optional",
      owner: "",
      repo: "",
      ref: "main",
      paths: [],
      tokenEnv: "GITHUB_TOKEN",
      required: false,
      maxFileBytes: 2 * 1024 * 1024
    }
  ],
  nav: {
    indexFile: "docs/README.md",
    missingChapterPolicy: "placeholder-pages"
  },
  search: {
    maxTermsPerDoc: 256,
    bm25: {
      k1: 1.2,
      b: 0.75
    }
  },
  render: {
    sanitize: true,
    allowMermaid: true,
    katex: true
  }
};
