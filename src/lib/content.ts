import manifestData from "@/src/generated/content-manifest.json";
import navigationData from "@/src/generated/navigation.json";
import searchIndexData from "@/src/generated/search-index.json";
import relatedData from "@/src/generated/related-index.json";
import redirectsData from "@/src/generated/redirects.json";
import diagnosticsData from "@/src/generated/diagnostics.json";
import type {
  Diagnostic,
  DocRecord,
  NavigationData,
  RelatedDoc,
  RelatedIndex,
  RedirectMap,
  SearchIndex
} from "@/content/types";

const docs = manifestData as DocRecord[];
const navigation = navigationData as NavigationData;
const searchIndex = searchIndexData as SearchIndex;
const relatedIndex = relatedData as RelatedIndex;
const redirects = redirectsData as RedirectMap;
const diagnostics = diagnosticsData as Diagnostic[];

const docsByCanonicalPath = new Map<string, DocRecord>(docs.map((doc) => [doc.canonicalPath, doc]));
const docsById = new Map<string, DocRecord>(docs.map((doc) => [doc.id, doc]));

function normalizeDocsPath(pathValue: string): string {
  const cleaned = pathValue.replace(/\/+$/, "");
  return cleaned.startsWith("/docs/") ? cleaned : `/docs/${cleaned.replace(/^\/+/, "")}`;
}

export function getDocs(): DocRecord[] {
  return docs;
}

export function getNavigation(): NavigationData {
  return navigation;
}

export function getSearchIndex(): SearchIndex {
  return searchIndex;
}

export function getDiagnostics(): Diagnostic[] {
  return diagnostics;
}

export function resolveRoute(routePath: string): { canonicalPath: string; redirectFrom?: string } | undefined {
  const normalized = normalizeDocsPath(routePath);
  const encoded = encodeURI(normalized);
  let decoded = normalized;
  try {
    decoded = decodeURI(normalized);
  } catch {
    decoded = normalized;
  }

  if (docsByCanonicalPath.has(normalized)) {
    return { canonicalPath: normalized };
  }

  const redirectTarget = redirects[normalized] ?? redirects[encoded] ?? redirects[decoded];
  if (redirectTarget) {
    return {
      canonicalPath: redirectTarget,
      redirectFrom: normalized
    };
  }

  return undefined;
}

export function getDocByCanonicalPath(canonicalPath: string): DocRecord | undefined {
  return docsByCanonicalPath.get(canonicalPath);
}

export function getDocBySlug(slug: string[]): { doc?: DocRecord; redirectTo?: string } {
  const routePath = normalizeDocsPath(slug.join("/"));
  const route = resolveRoute(routePath);
  if (!route) {
    return {};
  }

  if (route.redirectFrom) {
    return { redirectTo: route.canonicalPath };
  }

  return { doc: docsByCanonicalPath.get(route.canonicalPath) };
}

export function getRelatedDocs(docId: string): RelatedDoc[] {
  const neighbors = relatedIndex[docId] ?? [];
  return neighbors
    .map((item) => docsById.get(item.docId))
    .filter((doc): doc is DocRecord => Boolean(doc))
    .map(
      (doc): RelatedDoc => ({
        id: doc.id,
        title: doc.title,
        canonicalPath: doc.canonicalPath
      })
    );
}

export function getAllRouteSlugs(): string[][] {
  const slugs: string[][] = [];

  for (const doc of docs) {
    slugs.push(doc.slug);
  }

  for (const [legacyPath] of Object.entries(redirects)) {
    const rawSegments = legacyPath.replace(/^\/docs\//, "").split("/").filter(Boolean);
    const decodedSegments = rawSegments.map((segment) => {
      try {
        return decodeURIComponent(segment);
      } catch {
        return segment;
      }
    });
    if (decodedSegments.length > 0) {
      slugs.push(decodedSegments);
    }
  }

  const unique = new Map<string, string[]>();
  for (const slug of slugs) {
    unique.set(slug.join("/"), slug);
  }

  return [...unique.values()].sort((left, right) => left.join("/").localeCompare(right.join("/")));
}
