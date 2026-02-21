import type { RelatedIndex, SearchIndex } from "../types";

function idf(totalDocs: number, docFreq: number): number {
  return Math.log(1 + (totalDocs - docFreq + 0.5) / (docFreq + 0.5));
}

export function buildRelatedIndex(searchIndex: SearchIndex, maxRelated = 8): RelatedIndex {
  const related = new Map<string, Map<string, number>>();
  const totalDocs = searchIndex.documents.length;

  for (const [term, postings] of Object.entries(searchIndex.termPostings)) {
    const df = searchIndex.documentFrequencies[term] ?? 0;
    if (df <= 1) {
      continue;
    }

    const weightBase = idf(totalDocs, df);

    for (let i = 0; i < postings.length; i += 1) {
      const left = postings[i];
      const leftWeight = left.tf * weightBase;

      for (let j = i + 1; j < postings.length; j += 1) {
        const right = postings[j];
        const rightWeight = right.tf * weightBase;
        const contribution = leftWeight * rightWeight;

        if (!related.has(left.docId)) {
          related.set(left.docId, new Map());
        }
        if (!related.has(right.docId)) {
          related.set(right.docId, new Map());
        }

        const leftMap = related.get(left.docId)!;
        const rightMap = related.get(right.docId)!;

        leftMap.set(right.docId, (leftMap.get(right.docId) ?? 0) + contribution);
        rightMap.set(left.docId, (rightMap.get(left.docId) ?? 0) + contribution);
      }
    }
  }

  const output: RelatedIndex = {};
  for (const doc of searchIndex.documents) {
    const neighbors = related.get(doc.id) ?? new Map<string, number>();
    output[doc.id] = [...neighbors.entries()]
      .sort((left, right) => right[1] - left[1])
      .slice(0, maxRelated)
      .map(([docId, score]) => ({ docId, score }));
  }

  return output;
}
