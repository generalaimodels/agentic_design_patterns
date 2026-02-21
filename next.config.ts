import type { NextConfig } from "next";
import path from "node:path";
import { fileURLToPath } from "node:url";

const configDirectory = path.dirname(fileURLToPath(import.meta.url));
const repositoryName = process.env.GITHUB_REPOSITORY?.split("/")[1];
const isUserOrOrgPagesRepo = repositoryName?.endsWith(".github.io");
const resolvedBasePath = process.env.GITHUB_ACTIONS
  ? process.env.GITHUB_PAGES_BASE_PATH ?? (repositoryName && !isUserOrOrgPagesRepo ? `/${repositoryName}` : "")
  : "";

const nextConfig: NextConfig = {
  output: "export",
  basePath: resolvedBasePath,
  assetPrefix: resolvedBasePath,
  outputFileTracingRoot: configDirectory,
  images: {
    unoptimized: true
  },
  trailingSlash: true,
  experimental: {
    optimizePackageImports: ["katex", "mermaid"]
  }
};

export default nextConfig;
