import path from "node:path";

export function slugifySegment(segment: string): string {
  return segment
    .toLowerCase()
    .normalize("NFKD")
    .replace(/[\u0300-\u036f]/g, "")
    .replace(/[^a-z0-9]+/g, "-")
    .replace(/^-+|-+$/g, "")
    .replace(/-{2,}/g, "-");
}

export function buildSlugFromSourcePath(sourcePath: string, fallbackTitle: string): string[] {
  const withoutExt = sourcePath.replace(/\.md$/i, "");
  const rawSegments = withoutExt.split("/").filter(Boolean);
  const slugSegments = rawSegments.map(slugifySegment).filter(Boolean);
  if (slugSegments.length > 0) {
    return slugSegments;
  }
  return [slugifySegment(fallbackTitle) || "document"];
}

export function normalizeWhitespace(value: string): string {
  return value.replace(/\s+/g, " ").trim();
}

export function computeReadingMinutes(text: string, wordsPerMinute = 220): number {
  const words = text.length === 0 ? 0 : text.split(/\s+/).length;
  return Math.max(1, Math.ceil(words / wordsPerMinute));
}

export function chapterFromTitle(title: string): number | undefined {
  const match = title.match(/chapter\s+([0-9]+)/i);
  return match ? Number(match[1]) : undefined;
}

export function buildLegacyPaths(sourcePath: string): string[] {
  const withoutExt = sourcePath.replace(/\.md$/i, "");
  const segments = withoutExt.split("/").filter(Boolean);
  const encoded = segments.map((segment) => encodeURIComponent(segment));
  if (encoded.length === 0) {
    return [];
  }
  return [`/docs/${encoded.join("/")}`];
}

export function toCanonicalPath(slug: string[]): string {
  return `/docs/${slug.join("/")}`;
}

export function fileNameBase(fullPath: string): string {
  return path.basename(fullPath, path.extname(fullPath));
}

export function tokenizeText(input: string): string[] {
  return input
    .toLowerCase()
    .replace(/[^a-z0-9\s]+/g, " ")
    .split(/\s+/)
    .filter((token) => token.length >= 2);
}
