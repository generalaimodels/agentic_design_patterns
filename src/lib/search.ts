import type { SearchIndex } from "@/content/types";

export type RankedSearchResult = {
  docId: string;
  score: number;
};

function tokenizeText(input: string): string[] {
  return input
    .toLowerCase()
    .replace(/[^a-z0-9\\s]+/g, " ")
    .split(/\\s+/)
    .filter((token) => token.length >= 2);
}

export function rankSearchResults(index: SearchIndex, query: string, limit = 12): RankedSearchResult[] {
  const queryTerms = tokenizeText(query);
  if (queryTerms.length === 0) {
    return [];
  }

  const scores = new Map<string, number>();
  const totalDocs = index.documents.length;

  for (const term of queryTerms) {
    const postings = index.termPostings[term];
    if (!postings || postings.length === 0) {
      continue;
    }

    const df = index.documentFrequencies[term] ?? 0;
    if (df === 0) {
      continue;
    }

    const idf = Math.log(1 + (totalDocs - df + 0.5) / (df + 0.5));

    for (const posting of postings) {
      const docLength = index.docLengths[posting.docId] ?? 0;
      const lengthNorm =
        index.config.avgDocLength === 0
          ? 1
          : 1 - index.config.b + index.config.b * (docLength / index.config.avgDocLength);
      const numerator = posting.tf * (index.config.k1 + 1);
      const denominator = posting.tf + index.config.k1 * lengthNorm;
      const bm25 = idf * (denominator === 0 ? 0 : numerator / denominator);
      scores.set(posting.docId, (scores.get(posting.docId) ?? 0) + bm25);
    }
  }

  return [...scores.entries()]
    .map(([docId, score]) => ({ docId, score }))
    .sort((left, right) => right.score - left.score)
    .slice(0, limit);
}
