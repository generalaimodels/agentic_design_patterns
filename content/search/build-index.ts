import type { DocRecord, SearchIndex } from "../types";
import { tokenizeText } from "../utils";

export function buildSearchIndex(
  docs: DocRecord[],
  maxTermsPerDoc: number,
  bm25: { k1: number; b: number }
): SearchIndex {
  const termPostings: SearchIndex["termPostings"] = {};
  const documentFrequencies: SearchIndex["documentFrequencies"] = {};
  const docLengths: SearchIndex["docLengths"] = {};
  const documents: SearchIndex["documents"] = [];

  const perDocumentTerms = new Map<string, Map<string, number>>();

  for (const doc of docs) {
    const tokens = tokenizeText(`${doc.title} ${doc.text}`);
    docLengths[doc.id] = tokens.length;

    const tfMap = new Map<string, number>();
    for (const token of tokens) {
      tfMap.set(token, (tfMap.get(token) ?? 0) + 1);
    }

    const topTerms = [...tfMap.entries()]
      .sort((left, right) => right[1] - left[1])
      .slice(0, maxTermsPerDoc);

    const trimmedMap = new Map(topTerms);
    perDocumentTerms.set(doc.id, trimmedMap);

    for (const [term] of trimmedMap) {
      documentFrequencies[term] = (documentFrequencies[term] ?? 0) + 1;
    }

    documents.push({
      id: doc.id,
      title: doc.title,
      canonicalPath: doc.canonicalPath,
      excerpt: doc.excerpt
    });
  }

  for (const [docId, tfMap] of perDocumentTerms) {
    for (const [term, tf] of tfMap) {
      if (!termPostings[term]) {
        termPostings[term] = [];
      }
      termPostings[term].push({ docId, tf });
    }
  }

  for (const postings of Object.values(termPostings)) {
    postings.sort((left, right) => right.tf - left.tf);
  }

  const totalLength = Object.values(docLengths).reduce((sum, len) => sum + len, 0);
  const avgDocLength = docs.length === 0 ? 0 : totalLength / docs.length;

  return {
    version: 1,
    config: {
      k1: bm25.k1,
      b: bm25.b,
      avgDocLength
    },
    documents,
    docLengths,
    documentFrequencies,
    termPostings
  };
}
