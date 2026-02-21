export type Result<T, E> = { ok: true; value: T } | { ok: false; error: E };

export type ContentSource =
  | {
      kind: "local";
      id: string;
      root: string;
      include: string[];
      exclude: string[];
      required: boolean;
      maxFileBytes?: number;
    }
  | {
      kind: "github";
      id: string;
      owner: string;
      repo: string;
      ref: string;
      paths: string[];
      tokenEnv: "GITHUB_TOKEN";
      required: boolean;
      maxFileBytes?: number;
    };

export type Heading = {
  depth: 1 | 2 | 3 | 4 | 5 | 6;
  text: string;
  id: string;
};

export type MediaRecord = {
  kind: "image" | "video" | "audio";
  src: string;
  resolved: string;
};

export type DocRecord = {
  id: string;
  sourceId: string;
  sourcePath: string;
  title: string;
  slug: string[];
  canonicalPath: string;
  legacyPaths: string[];
  chapterNumber?: number;
  headings: Heading[];
  excerpt: string;
  html: string;
  text: string;
  readingMinutes: number;
  media: MediaRecord[];
};

export type BuildError =
  | { type: "SOURCE_FETCH_FAILED"; sourceId: string; detail: string }
  | { type: "MARKDOWN_PARSE_FAILED"; docPath: string; detail: string }
  | { type: "INVALID_HTML_REMOVED"; docPath: string; removedCount: number }
  | { type: "SLUG_COLLISION"; docs: string[]; slug: string }
  | { type: "MISSING_REQUIRED_SOURCE"; sourceId: string }
  | { type: "ASSET_RESOLUTION_FAILED"; docPath: string; asset: string };

export type BuildConfig = {
  sources: ContentSource[];
  nav: { indexFile: "docs/README.md"; missingChapterPolicy: "placeholder-pages" };
  search: { maxTermsPerDoc: number; bm25: { k1: number; b: number } };
  render: { sanitize: true; allowMermaid: true; katex: true };
};

export type SourceDocument = {
  sourceId: string;
  sourcePath: string;
  fullPath: string;
  content: string;
};

export type Diagnostic = {
  severity: "error" | "warning";
  code: BuildError["type"];
  message: string;
  docPath?: string;
};

export type NavigationDocItem = {
  kind: "doc" | "placeholder";
  title: string;
  chapterNumber?: number;
  slug: string[];
  canonicalPath: string;
};

export type NavigationData = {
  chapters: NavigationDocItem[];
  fallback: NavigationDocItem[];
};

export type SearchDocument = {
  id: string;
  title: string;
  canonicalPath: string;
  excerpt: string;
};

export type RelatedDoc = {
  id: string;
  title: string;
  canonicalPath: string;
};

export type SearchIndex = {
  version: 1;
  config: { k1: number; b: number; avgDocLength: number };
  documents: SearchDocument[];
  docLengths: Record<string, number>;
  documentFrequencies: Record<string, number>;
  termPostings: Record<string, Array<{ docId: string; tf: number }>>;
};

export type RelatedIndex = Record<string, Array<{ docId: string; score: number }>>;

export type RedirectMap = Record<string, string>;

export type GeneratedArtifacts = {
  docs: DocRecord[];
  navigation: NavigationData;
  searchIndex: SearchIndex;
  relatedIndex: RelatedIndex;
  redirects: RedirectMap;
  diagnostics: Diagnostic[];
};
