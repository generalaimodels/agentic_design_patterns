import { promises as fs } from "node:fs";
import { slugifySegment, tokenizeText } from "../utils";
import type { DocRecord, NavigationData, NavigationDocItem, Result, BuildError } from "../types";
import { err, ok } from "../result";

export type ChapterIndexItem = {
  chapterNumber: number;
  title: string;
};

export type NavigationBuildResult = {
  navigation: NavigationData;
  missingChapters: ChapterIndexItem[];
  chapterIndex: ChapterIndexItem[];
};

function parseChapterIndex(content: string): ChapterIndexItem[] {
  const chapters: ChapterIndexItem[] = [];
  const lines = content.split(/\r?\n/);

  for (const line of lines) {
    const match = line.match(/^###\s+Chapter\s+([0-9]+):\s+(.+)$/i);
    if (!match) {
      continue;
    }
    chapters.push({
      chapterNumber: Number(match[1]),
      title: match[2].trim()
    });
  }

  chapters.sort((left, right) => left.chapterNumber - right.chapterNumber);
  return chapters;
}

function chapterScore(doc: DocRecord, chapter: ChapterIndexItem): number {
  let score = 0;

  if (doc.chapterNumber === chapter.chapterNumber) {
    score += 500;
  }

  const docTitleTerms = new Set(tokenizeText(doc.title));
  const chapterTitleTerms = tokenizeText(chapter.title);

  for (const term of chapterTitleTerms) {
    if (docTitleTerms.has(term)) {
      score += 40;
    }
  }

  const lowerPath = doc.sourcePath.toLowerCase();
  const chapterSlugTerm = slugifySegment(chapter.title);
  if (chapterSlugTerm && lowerPath.includes(chapterSlugTerm.replace(/-/g, "_"))) {
    score += 120;
  }
  if (chapterSlugTerm && lowerPath.includes(chapterSlugTerm)) {
    score += 120;
  }

  const pathDepth = doc.sourcePath.split("/").length;
  score -= pathDepth * 2;

  return score;
}

function docToNavItem(doc: DocRecord): NavigationDocItem {
  return {
    kind: "doc",
    title: doc.title,
    chapterNumber: doc.chapterNumber,
    slug: doc.slug,
    canonicalPath: doc.canonicalPath
  };
}

export async function buildNavigation(
  docs: DocRecord[],
  indexFilePath: string
): Promise<Result<NavigationBuildResult, BuildError>> {
  let indexMarkdown = "";
  try {
    indexMarkdown = await fs.readFile(indexFilePath, "utf8");
  } catch (error) {
    const detail = error instanceof Error ? error.message : String(error);
    return err({
      type: "SOURCE_FETCH_FAILED",
      sourceId: "local-docs",
      detail: `Unable to read navigation index file ${indexFilePath}: ${detail}`
    });
  }

  const chapterIndex = parseChapterIndex(indexMarkdown);
  const docsById = new Map(docs.map((doc) => [doc.id, doc]));
  const usedDocIds = new Set<string>();
  const chapterItems: NavigationDocItem[] = [];
  const missingChapters: ChapterIndexItem[] = [];

  for (const chapter of chapterIndex) {
    let bestDoc: DocRecord | undefined;
    let bestScore = Number.NEGATIVE_INFINITY;

    for (const doc of docs) {
      if (usedDocIds.has(doc.id)) {
        continue;
      }
      const score = chapterScore(doc, chapter);
      if (score > bestScore) {
        bestScore = score;
        bestDoc = doc;
      }
    }

    if (!bestDoc || bestScore < 140) {
      missingChapters.push(chapter);
      chapterItems.push({
        kind: "placeholder",
        title: `Chapter ${chapter.chapterNumber}: ${chapter.title}`,
        chapterNumber: chapter.chapterNumber,
        slug: [
          `chapter-${chapter.chapterNumber}-${slugifySegment(chapter.title) || "placeholder"}`
        ],
        canonicalPath: `/docs/chapter-${chapter.chapterNumber}-${slugifySegment(chapter.title) || "placeholder"}`
      });
      continue;
    }

    usedDocIds.add(bestDoc.id);
    chapterItems.push(docToNavItem(bestDoc));
  }

  const fallbackItems = [...docsById.values()]
    .filter((doc) => !usedDocIds.has(doc.id))
    .sort((left, right) => left.canonicalPath.localeCompare(right.canonicalPath))
    .map(docToNavItem);

  return ok({
    navigation: {
      chapters: chapterItems,
      fallback: fallbackItems
    },
    missingChapters,
    chapterIndex
  });
}
