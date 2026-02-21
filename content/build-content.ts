import { promises as fs } from "node:fs";
import path from "node:path";
import { buildConfig } from "./config";
import { renderMarkdownDocument } from "./markdown/pipeline";
import { buildNavigation } from "./nav/build-nav";
import { err, ok } from "./result";
import { buildRelatedIndex } from "./related/build-related";
import { buildSearchIndex } from "./search/build-index";
import { fetchGitHubSource } from "./sources/github";
import { fetchLocalSource } from "./sources/local";
import { buildLegacyPaths, slugifySegment, toCanonicalPath } from "./utils";
import type {
  BuildError,
  Diagnostic,
  DocRecord,
  GeneratedArtifacts,
  Result,
  SourceDocument,
  ContentSource,
  MediaRecord
} from "./types";

const GENERATED_DIR = path.resolve("src/generated");
const PUBLIC_ASSET_DIR = path.resolve("public/content-assets");
const BASE_PATH = process.env.GITHUB_ACTIONS ? "/agentic_design_patterns" : "";

type SourceMap = Map<string, SourceDocument>;

type SourceIndex = {
  byIdAndPath: SourceMap;
  bySourceId: Map<string, ContentSource>;
};

function sourceDocKey(sourceId: string, sourcePath: string): string {
  return `${sourceId}:${sourcePath}`;
}

function buildErrorMessage(error: BuildError): string {
  switch (error.type) {
    case "SOURCE_FETCH_FAILED":
    case "MARKDOWN_PARSE_FAILED":
      return error.detail;
    case "INVALID_HTML_REMOVED":
      return `Potentially unsafe HTML fragments were removed (${error.removedCount}).`;
    case "SLUG_COLLISION":
      return `Slug collision for ${error.slug}: ${error.docs.join(", ")}`;
    case "MISSING_REQUIRED_SOURCE":
      return `Missing required source: ${error.sourceId}`;
    case "ASSET_RESOLUTION_FAILED":
      return `Unable to resolve asset: ${error.asset}`;
  }
}

function toPosixPath(value: string): string {
  return value.split(path.sep).join(path.posix.sep);
}

function escapeRegExp(value: string): string {
  return value.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
}

function uniqueStringVariants(value: string): string[] {
  const variants = new Set<string>([value]);
  try {
    variants.add(decodeURI(value));
  } catch {
    // Keep the original value when decode fails.
  }
  try {
    variants.add(encodeURI(value));
  } catch {
    // Keep the original value when encode fails.
  }
  return [...variants.values()];
}

async function writeJsonAtomically(fileName: string, payload: unknown): Promise<void> {
  const destination = path.join(GENERATED_DIR, fileName);
  const temporary = `${destination}.tmp`;
  const serialized = `${JSON.stringify(payload, null, 2)}\n`;

  await fs.mkdir(path.dirname(destination), { recursive: true });
  await fs.writeFile(temporary, serialized, "utf8");
  await fs.rename(temporary, destination);
}

async function resetPublicAssetDirectory(): Promise<void> {
  await fs.rm(PUBLIC_ASSET_DIR, { recursive: true, force: true });
  await fs.mkdir(PUBLIC_ASSET_DIR, { recursive: true });
}

function isAbsoluteOrExternal(value: string): boolean {
  return /^(https?:)?\/\//i.test(value) || value.startsWith("data:") || value.startsWith("#") || value.startsWith("/");
}

async function resolveMedia(
  doc: DocRecord,
  sourceDocument: SourceDocument,
  source: ContentSource,
  diagnostics: Diagnostic[]
): Promise<MediaRecord[]> {
  const resolvedMedia: MediaRecord[] = [];

  for (const media of doc.media) {
    if (isAbsoluteOrExternal(media.src)) {
      resolvedMedia.push(media);
      continue;
    }

    if (source.kind === "local") {
      const originDirectory = path.dirname(sourceDocument.fullPath);
      const absoluteAssetPath = path.resolve(originDirectory, media.src);

      try {
        const stat = await fs.stat(absoluteAssetPath);
        if (!stat.isFile()) {
          diagnostics.push({
            severity: "error",
            code: "ASSET_RESOLUTION_FAILED",
            message: `Resolved media path is not a file: ${absoluteAssetPath}`,
            docPath: doc.sourcePath
          });
          continue;
        }

        const relativeToWorkspace = toPosixPath(path.relative(process.cwd(), absoluteAssetPath));
        if (relativeToWorkspace.startsWith("..")) {
          diagnostics.push({
            severity: "error",
            code: "ASSET_RESOLUTION_FAILED",
            message: `Asset resolves outside workspace and is blocked: ${absoluteAssetPath}`,
            docPath: doc.sourcePath
          });
          continue;
        }

        const publicRelative = `content-assets/${relativeToWorkspace}`;
        const destinationAssetPath = path.resolve("public", ...publicRelative.split("/"));
        await fs.mkdir(path.dirname(destinationAssetPath), { recursive: true });
        await fs.copyFile(absoluteAssetPath, destinationAssetPath);

        resolvedMedia.push({
          ...media,
          resolved: `${BASE_PATH}/${encodeURI(publicRelative)}`
        });
      } catch {
        diagnostics.push({
          severity: "error",
          code: "ASSET_RESOLUTION_FAILED",
          message: `Unable to resolve media asset ${media.src} from ${doc.sourcePath}`,
          docPath: doc.sourcePath
        });
      }
      continue;
    }

    if (source.kind === "github") {
      const sourcePrefix = `github/${source.owner}/${source.repo}/`;
      if (!sourceDocument.sourcePath.startsWith(sourcePrefix)) {
        resolvedMedia.push(media);
        continue;
      }

      const repoRelativePath = sourceDocument.sourcePath.slice(sourcePrefix.length);
      const repoDirectory = path.posix.dirname(repoRelativePath);
      const assetPath = path.posix.normalize(path.posix.join(repoDirectory, media.src));
      const rawUrl = `https://raw.githubusercontent.com/${source.owner}/${source.repo}/${encodeURIComponent(
        source.ref
      )}/${assetPath}`;

      resolvedMedia.push({
        ...media,
        resolved: rawUrl
      });
      continue;
    }

    resolvedMedia.push(media);
  }

  return resolvedMedia;
}

function rewriteHtmlMediaSources(html: string, media: MediaRecord[]): string {
  let output = html;

  for (const asset of media) {
    for (const variant of uniqueStringVariants(asset.src)) {
      const pattern = new RegExp(`(src=["'])${escapeRegExp(variant)}(["'])`, "g");
      output = output.replace(pattern, `$1${asset.resolved}$2`);
    }
  }

  return output;
}

function validateInternalAnchors(doc: DocRecord, diagnostics: Diagnostic[]): void {
  const headingIds = new Set(doc.headings.map((heading) => heading.id));
  const matches = [...doc.html.matchAll(/href=\"#([^\"]+)\"/g)];
  for (const match of matches) {
    const anchorId = match[1];
    if (!headingIds.has(anchorId)) {
      diagnostics.push({
        severity: "warning",
        code: "ASSET_RESOLUTION_FAILED",
        message: `Internal anchor target not found: #${anchorId}`,
        docPath: doc.sourcePath
      });
    }
  }
}

function validateHeadingHierarchy(doc: DocRecord, diagnostics: Diagnostic[]): void {
  let previousDepth = 0;
  for (const heading of doc.headings) {
    if (previousDepth === 0 && heading.depth !== 1) {
      diagnostics.push({
        severity: "warning",
        code: "MARKDOWN_PARSE_FAILED",
        message: `Document starts with H${heading.depth} instead of H1.`,
        docPath: doc.sourcePath
      });
    }
    if (previousDepth !== 0 && heading.depth > previousDepth + 1) {
      diagnostics.push({
        severity: "warning",
        code: "MARKDOWN_PARSE_FAILED",
        message: `Heading hierarchy jumps from H${previousDepth} to H${heading.depth}.`,
        docPath: doc.sourcePath
      });
    }
    previousDepth = heading.depth;
  }
}

function enforceImageAltText(doc: DocRecord): Result<void, BuildError> {
  const imageTags = [...doc.html.matchAll(/<img\b[^>]*>/gi)];
  for (const tagMatch of imageTags) {
    const tag = tagMatch[0];
    const altMatch = tag.match(/\balt=(?:"([^"]*)"|'([^']*)')/i);
    const alt = (altMatch?.[1] ?? altMatch?.[2] ?? "").trim();
    if (alt.length >= 3) {
      continue;
    }
    return err({
      type: "MARKDOWN_PARSE_FAILED",
      docPath: doc.sourcePath,
      detail: "Image tag is missing a descriptive alt attribute with at least 3 characters."
    });
  }
  return ok(undefined);
}

function createPlaceholderDoc(chapterNumber: number, chapterTitle: string): DocRecord {
  const slug = [`chapter-${chapterNumber}-${slugifySegment(chapterTitle) || "placeholder"}`];
  const title = `Chapter ${chapterNumber}: ${chapterTitle}`;
  const html = [
    "<article class=\"placeholder-doc\">",
    `<h1>${title}</h1>`,
    "<p>This chapter is defined in the index and reserved for upcoming content.</p>",
    "</article>"
  ].join("");

  return {
    id: `placeholder:chapter-${chapterNumber}`,
    sourceId: "generated-placeholder",
    sourcePath: `generated/chapter-${chapterNumber}.md`,
    title,
    slug,
    canonicalPath: toCanonicalPath(slug),
    legacyPaths: [],
    chapterNumber,
    headings: [
      {
        depth: 1,
        text: title,
        id: `chapter-${chapterNumber}`
      }
    ],
    excerpt: "This chapter is not yet available.",
    html,
    text: `${title}. This chapter is not yet available.`,
    readingMinutes: 1,
    media: []
  };
}

function collectSlugCollisions(docs: DocRecord[]): Result<void, BuildError> {
  const byCanonicalPath = new Map<string, string[]>();

  for (const doc of docs) {
    const bucket = byCanonicalPath.get(doc.canonicalPath) ?? [];
    bucket.push(doc.sourcePath);
    byCanonicalPath.set(doc.canonicalPath, bucket);
  }

  for (const [canonicalPath, docPaths] of byCanonicalPath) {
    if (docPaths.length > 1) {
      return err({
        type: "SLUG_COLLISION",
        slug: canonicalPath,
        docs: docPaths
      });
    }
  }

  return ok(undefined);
}

async function fetchSourceDocuments(sources: ContentSource[]): Promise<
  Result<{ documents: SourceDocument[]; diagnostics: Diagnostic[]; sourceIndex: SourceIndex }, BuildError>
> {
  const documents: SourceDocument[] = [];
  const diagnostics: Diagnostic[] = [];
  const byIdAndPath: SourceMap = new Map();
  const bySourceId = new Map<string, ContentSource>();

  for (const source of sources) {
    bySourceId.set(source.id, source);

    const fetchResult = source.kind === "local" ? await fetchLocalSource(source) : await fetchGitHubSource(source);

    if (!fetchResult.ok) {
      if (source.required) {
        return err(fetchResult.error);
      }

      diagnostics.push({
        severity: "warning",
        code: "SOURCE_FETCH_FAILED",
        message: buildErrorMessage(fetchResult.error)
      });
      continue;
    }

    if (fetchResult.value.length === 0 && source.required) {
      return err({ type: "MISSING_REQUIRED_SOURCE", sourceId: source.id });
    }

    for (const document of fetchResult.value) {
      documents.push(document);
      byIdAndPath.set(sourceDocKey(document.sourceId, document.sourcePath), document);
    }
  }

  documents.sort((left, right) => left.sourcePath.localeCompare(right.sourcePath));

  return ok({
    documents,
    diagnostics,
    sourceIndex: {
      byIdAndPath,
      bySourceId
    }
  });
}

export async function buildArtifacts(): Promise<Result<GeneratedArtifacts, BuildError>> {
  await resetPublicAssetDirectory();

  const sourceCollection = await fetchSourceDocuments(buildConfig.sources);
  if (!sourceCollection.ok) {
    return sourceCollection;
  }

  const diagnostics = [...sourceCollection.value.diagnostics];
  const docs: DocRecord[] = [];

  for (const sourceDocument of sourceCollection.value.documents) {
    const renderResult = await renderMarkdownDocument(sourceDocument);
    if (!renderResult.ok) {
      const source = sourceCollection.value.sourceIndex.bySourceId.get(sourceDocument.sourceId);
      if (source?.required) {
        return err(renderResult.error);
      }

      diagnostics.push({
        severity: "error",
        code: renderResult.error.type,
        message: buildErrorMessage(renderResult.error),
        docPath: sourceDocument.sourcePath
      });
      continue;
    }

    const source = sourceCollection.value.sourceIndex.bySourceId.get(sourceDocument.sourceId);
    if (!source) {
      return err({
        type: "SOURCE_FETCH_FAILED",
        sourceId: sourceDocument.sourceId,
        detail: `Unknown source id for rendered document: ${sourceDocument.sourceId}`
      });
    }

    const rendered = renderResult.value;
    const record: DocRecord = {
      id: sourceDocKey(sourceDocument.sourceId, sourceDocument.sourcePath),
      sourceId: sourceDocument.sourceId,
      sourcePath: sourceDocument.sourcePath,
      title: rendered.title,
      slug: rendered.slug,
      canonicalPath: rendered.canonicalPath,
      legacyPaths: buildLegacyPaths(sourceDocument.sourcePath),
      chapterNumber: rendered.chapterNumber,
      headings: rendered.headings,
      excerpt: rendered.excerpt,
      html: rendered.html,
      text: rendered.text,
      readingMinutes: rendered.readingMinutes,
      media: rendered.media
    };

    if (rendered.invalidHtmlRemoved > 0) {
      diagnostics.push({
        severity: "warning",
        code: "INVALID_HTML_REMOVED",
        message: `Potentially unsafe HTML fragments were removed (${rendered.invalidHtmlRemoved}).`,
        docPath: sourceDocument.sourcePath
      });
    }

    record.media = await resolveMedia(record, sourceDocument, source, diagnostics);
    record.html = rewriteHtmlMediaSources(record.html, record.media);
    const altValidation = enforceImageAltText(record);
    if (!altValidation.ok) {
      return altValidation;
    }
    validateHeadingHierarchy(record, diagnostics);
    validateInternalAnchors(record, diagnostics);
    docs.push(record);
  }

  const collisionCheck = collectSlugCollisions(docs);
  if (!collisionCheck.ok) {
    return collisionCheck;
  }

  const navigationResult = await buildNavigation(docs, buildConfig.nav.indexFile);
  if (!navigationResult.ok) {
    return navigationResult;
  }

  for (const missingChapter of navigationResult.value.missingChapters) {
    docs.push(createPlaceholderDoc(missingChapter.chapterNumber, missingChapter.title));
  }

  docs.sort((left, right) => left.canonicalPath.localeCompare(right.canonicalPath));

  const searchIndex = buildSearchIndex(docs, buildConfig.search.maxTermsPerDoc, buildConfig.search.bm25);
  const relatedIndex = buildRelatedIndex(searchIndex);

  const redirects: Record<string, string> = {};
  for (const doc of docs) {
    for (const alias of doc.legacyPaths) {
      redirects[alias] = doc.canonicalPath;
    }
  }

  return ok({
    docs,
    navigation: navigationResult.value.navigation,
    searchIndex,
    relatedIndex,
    redirects,
    diagnostics
  });
}

export async function runBuild(): Promise<void> {
  const result = await buildArtifacts();
  if (!result.ok) {
    console.error(JSON.stringify(result.error, null, 2));
    process.exitCode = 1;
    return;
  }

  await writeJsonAtomically("content-manifest.json", result.value.docs);
  await writeJsonAtomically("navigation.json", result.value.navigation);
  await writeJsonAtomically("search-index.json", result.value.searchIndex);
  await writeJsonAtomically("related-index.json", result.value.relatedIndex);
  await writeJsonAtomically("redirects.json", result.value.redirects);
  await writeJsonAtomically("diagnostics.json", result.value.diagnostics);

  console.log(
    `Generated ${result.value.docs.length} docs, ${Object.keys(result.value.searchIndex.termPostings).length} search terms, ${Object.keys(result.value.redirects).length} redirects.`
  );
}
