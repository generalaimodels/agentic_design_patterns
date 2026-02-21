import { promises as fs } from "node:fs";
import { createRequire } from "node:module";
import path from "node:path";
import { TextDecoder } from "node:util";
import { err, ok } from "../result";
import type { BuildError, ContentSource, Result, SourceDocument } from "../types";

const utf8Decoder = new TextDecoder("utf-8", { fatal: true });
const require = createRequire(import.meta.url);

type Picomatch = (
  pattern: string | readonly string[],
  options?: {
    dot?: boolean;
  }
) => (input: string) => boolean;

type DirectoryEntry = {
  name: string;
  isDirectory(): boolean;
  isSymbolicLink(): boolean;
  isFile(): boolean;
};

type FileStat = {
  isDirectory(): boolean;
};

const picomatch = require("picomatch") as Picomatch;

function toPosixPath(value: string): string {
  return value.split(path.sep).join(path.posix.sep);
}

function matchesAny(relativePath: string, matchers: Array<(input: string) => boolean>): boolean {
  for (const matcher of matchers) {
    if (matcher(relativePath)) {
      return true;
    }
  }
  return false;
}

async function readUtf8WithBounds(
  sourceId: string,
  fullPath: string,
  maxFileBytes: number
): Promise<Result<string, BuildError>> {
  try {
    const stat = await fs.stat(fullPath);
    if (!stat.isFile()) {
      return err({
        type: "SOURCE_FETCH_FAILED",
        sourceId,
        detail: `Non-file markdown path encountered: ${fullPath}`
      });
    }
    if (stat.size > maxFileBytes) {
      return err({
        type: "SOURCE_FETCH_FAILED",
        sourceId,
        detail: `File exceeds max size (${stat.size} > ${maxFileBytes}): ${fullPath}`
      });
    }
    const raw = await fs.readFile(fullPath);
    return ok(utf8Decoder.decode(raw));
  } catch (error) {
    const detail = error instanceof Error ? error.message : String(error);
    return err({
      type: "SOURCE_FETCH_FAILED",
      sourceId,
      detail: `Failed to read UTF-8 markdown at ${fullPath}: ${detail}`
    });
  }
}

export async function fetchLocalSource(
  source: Extract<ContentSource, { kind: "local" }>
): Promise<Result<SourceDocument[], BuildError>> {
  const absoluteRoot = path.resolve(source.root);
  const includeMatchers = source.include.map((pattern) => picomatch(pattern, { dot: true }));
  const excludeMatchers = source.exclude.map((pattern) => picomatch(pattern, { dot: true }));
  const maxFileBytes = source.maxFileBytes ?? 2 * 1024 * 1024;
  const visitedDirectories = new Set<string>();
  const documents: SourceDocument[] = [];

  async function walkDirectory(currentDirectory: string): Promise<Result<void, BuildError>> {
    let resolvedDirectory: string;
    try {
      resolvedDirectory = await fs.realpath(currentDirectory);
    } catch (error) {
      const detail = error instanceof Error ? error.message : String(error);
      return err({
        type: "SOURCE_FETCH_FAILED",
        sourceId: source.id,
        detail: `Unable to resolve directory ${currentDirectory}: ${detail}`
      });
    }

    if (visitedDirectories.has(resolvedDirectory)) {
      return ok(undefined);
    }
    visitedDirectories.add(resolvedDirectory);

    let entries: DirectoryEntry[];
    try {
      entries = await fs.readdir(currentDirectory, { withFileTypes: true, encoding: "utf8" });
    } catch (error) {
      const detail = error instanceof Error ? error.message : String(error);
      return err({
        type: "SOURCE_FETCH_FAILED",
        sourceId: source.id,
        detail: `Unable to read directory ${currentDirectory}: ${detail}`
      });
    }

    for (const entry of entries) {
      const absolutePath = path.join(currentDirectory, entry.name);
      const relativePath = toPosixPath(path.relative(absoluteRoot, absolutePath));

      if (relativePath.startsWith("..")) {
        continue;
      }

      if (matchesAny(relativePath, excludeMatchers)) {
        continue;
      }

      if (entry.isDirectory()) {
        const nestedResult = await walkDirectory(absolutePath);
        if (!nestedResult.ok) {
          return nestedResult;
        }
        continue;
      }

      if (entry.isSymbolicLink()) {
        let linkStat: FileStat;
        try {
          linkStat = await fs.stat(absolutePath);
        } catch {
          continue;
        }

        if (linkStat.isDirectory()) {
          const nestedResult = await walkDirectory(absolutePath);
          if (!nestedResult.ok) {
            return nestedResult;
          }
        }
        continue;
      }

      if (!entry.isFile()) {
        continue;
      }

      if (!relativePath.endsWith(".md")) {
        continue;
      }

      if (!matchesAny(relativePath, includeMatchers)) {
        continue;
      }

      const readResult = await readUtf8WithBounds(source.id, absolutePath, maxFileBytes);
      if (!readResult.ok) {
        return readResult;
      }

      documents.push({
        sourceId: source.id,
        sourcePath: relativePath,
        fullPath: absolutePath,
        content: readResult.value
      });
    }

    return ok(undefined);
  }

  try {
    const stat = await fs.stat(absoluteRoot);
    if (!stat.isDirectory()) {
      return err({
        type: "SOURCE_FETCH_FAILED",
        sourceId: source.id,
        detail: `Local source root is not a directory: ${source.root}`
      });
    }
  } catch (error) {
    const detail = error instanceof Error ? error.message : String(error);
    return err({
      type: "SOURCE_FETCH_FAILED",
      sourceId: source.id,
      detail: `Cannot access source root ${source.root}: ${detail}`
    });
  }

  const walkResult = await walkDirectory(absoluteRoot);
  if (!walkResult.ok) {
    return walkResult;
  }

  documents.sort((left, right) => left.sourcePath.localeCompare(right.sourcePath));
  return ok(documents);
}
