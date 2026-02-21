import type { NextConfig } from "next";
import path from "node:path";
import { fileURLToPath } from "node:url";

const configDirectory = path.dirname(fileURLToPath(import.meta.url));

const nextConfig: NextConfig = {
  output: "export",
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
