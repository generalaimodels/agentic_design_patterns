import { TextDecoder } from "node:util";
import { err, ok } from "../result";
import type { BuildError, ContentSource, Result, SourceDocument } from "../types";

const utf8Decoder = new TextDecoder("utf-8", { fatal: true });

type GitHubContentFile = {
  type: "file";
  path: string;
  size: number;
  name: string;
  content?: string;
  encoding?: "base64";
};

type GitHubContentDirectory = {
  type: "dir";
  path: string;
  name: string;
};

type GitHubContentItem = GitHubContentFile | GitHubContentDirectory;

function encodeGitHubPath(pathValue: string): string {
  return pathValue
    .split("/")
    .filter((segment) => segment.length > 0)
    .map((segment) => encodeURIComponent(segment))
    .join("/");
}

async function fetchGitHubJson<T>(
  url: string,
  token: string | undefined
): Promise<Result<T, BuildError>> {
  const response = await fetch(url, {
    headers: {
      Accept: "application/vnd.github+json",
      ...(token ? { Authorization: `Bearer ${token}` } : {})
    }
  });

  if (!response.ok) {
    const detail = await response.text();
    return err({
      type: "SOURCE_FETCH_FAILED",
      sourceId: "github",
      detail: `GitHub request failed (${response.status}) for ${url}: ${detail}`
    });
  }

  const json = (await response.json()) as T;
  return ok(json);
}

function decodeBase64Utf8(value: string): Result<string, BuildError> {
  try {
    const raw = Buffer.from(value, "base64");
    return ok(utf8Decoder.decode(raw));
  } catch (error) {
    const detail = error instanceof Error ? error.message : String(error);
    return err({
      type: "SOURCE_FETCH_FAILED",
      sourceId: "github",
      detail: `Unable to decode GitHub base64 markdown as UTF-8: ${detail}`
    });
  }
}

async function fetchPathRecursively(
  source: Extract<ContentSource, { kind: "github" }>,
  token: string | undefined,
  pathValue: string,
  maxFileBytes: number
): Promise<Result<SourceDocument[], BuildError>> {
  const encoded = encodeGitHubPath(pathValue);
  const apiUrl = `https://api.github.com/repos/${source.owner}/${source.repo}/contents/${encoded}?ref=${encodeURIComponent(source.ref)}`;

  const payloadResult = await fetchGitHubJson<GitHubContentItem | GitHubContentItem[]>(apiUrl, token);
  if (!payloadResult.ok) {
    return err({
      ...payloadResult.error,
      sourceId: source.id
    });
  }

  const payload = payloadResult.value;
  const items = Array.isArray(payload) ? payload : [payload];
  const documents: SourceDocument[] = [];

  for (const item of items) {
    if (item.type === "dir") {
      const nestedResult = await fetchPathRecursively(source, token, item.path, maxFileBytes);
      if (!nestedResult.ok) {
        return nestedResult;
      }
      documents.push(...nestedResult.value);
      continue;
    }

    if (!item.path.endsWith(".md")) {
      continue;
    }

    if (item.size > maxFileBytes) {
      return err({
        type: "SOURCE_FETCH_FAILED",
        sourceId: source.id,
        detail: `GitHub markdown file exceeds size limit (${item.size} > ${maxFileBytes}): ${item.path}`
      });
    }

    let markdown = "";
    if (item.content && item.encoding === "base64") {
      const decodeResult = decodeBase64Utf8(item.content.replace(/\n/g, ""));
      if (!decodeResult.ok) {
        return err({
          ...decodeResult.error,
          sourceId: source.id
        });
      }
      markdown = decodeResult.value;
    } else {
      const blobUrl = `https://api.github.com/repos/${source.owner}/${source.repo}/contents/${encodeGitHubPath(
        item.path
      )}?ref=${encodeURIComponent(source.ref)}`;
      const fileResult = await fetchGitHubJson<GitHubContentFile>(blobUrl, token);
      if (!fileResult.ok) {
        return err({ ...fileResult.error, sourceId: source.id });
      }
      if (!fileResult.value.content || fileResult.value.encoding !== "base64") {
        return err({
          type: "SOURCE_FETCH_FAILED",
          sourceId: source.id,
          detail: `GitHub API did not return base64 content for ${item.path}`
        });
      }
      const decodeResult = decodeBase64Utf8(fileResult.value.content.replace(/\n/g, ""));
      if (!decodeResult.ok) {
        return err({ ...decodeResult.error, sourceId: source.id });
      }
      markdown = decodeResult.value;
    }

    documents.push({
      sourceId: source.id,
      sourcePath: `github/${source.owner}/${source.repo}/${item.path}`,
      fullPath: `github://${source.owner}/${source.repo}/${item.path}@${source.ref}`,
      content: markdown
    });
  }

  return ok(documents);
}

export async function fetchGitHubSource(
  source: Extract<ContentSource, { kind: "github" }>
): Promise<Result<SourceDocument[], BuildError>> {
  if (!source.owner || !source.repo || source.paths.length === 0) {
    return ok([]);
  }

  const token = process.env[source.tokenEnv];
  const maxFileBytes = source.maxFileBytes ?? 2 * 1024 * 1024;
  const results: SourceDocument[] = [];

  for (const sourcePath of source.paths) {
    const fetchResult = await fetchPathRecursively(source, token, sourcePath, maxFileBytes);
    if (!fetchResult.ok) {
      return fetchResult;
    }
    results.push(...fetchResult.value);
  }

  results.sort((left, right) => left.sourcePath.localeCompare(right.sourcePath));
  return ok(results);
}
