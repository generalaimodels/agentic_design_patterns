import { toString } from "mdast-util-to-string";
import rehypeAutolinkHeadings from "rehype-autolink-headings";
import rehypeKatex from "rehype-katex";
import rehypePrettyCode from "rehype-pretty-code";
import rehypeRaw from "rehype-raw";
import rehypeSanitize from "rehype-sanitize";
import rehypeSlug from "rehype-slug";
import rehypeStringify from "rehype-stringify";
import remarkGfm from "remark-gfm";
import remarkMath from "remark-math";
import remarkParse from "remark-parse";
import remarkRehype from "remark-rehype";
import { unified } from "unified";
import type { Node } from "unist";
import { visit } from "unist-util-visit";
import { err, ok } from "../result";
import { buildSlugFromSourcePath, chapterFromTitle, computeReadingMinutes, normalizeWhitespace } from "../utils";
import { markdownSanitizeSchema } from "./sanitize-schema";
import type { BuildError, Heading, MediaRecord, Result, SourceDocument } from "../types";

type RenderedMarkdown = {
  title: string;
  chapterNumber?: number;
  slug: string[];
  canonicalPath: string;
  legacyPaths: string[];
  headings: Heading[];
  html: string;
  text: string;
  excerpt: string;
  readingMinutes: number;
  media: MediaRecord[];
  invalidHtmlRemoved: number;
};

function escapeHtml(value: string): string {
  return value
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/\"/g, "&quot;")
    .replace(/'/g, "&#39;");
}

function remarkMermaidToSafeHtml() {
  return (tree: Node): void => {
    visit(tree, "code", (node: any) => {
      if (typeof node.lang === "string" && node.lang.toLowerCase() === "mermaid") {
        const escaped = escapeHtml(String(node.value ?? ""));
        node.type = "html";
        node.value = `<pre class=\"mermaid-block\"><code class=\"language-mermaid\">${escaped}</code></pre>`;
      }
    });
  };
}

function normalizeUnicodeMath(input: string): string {
  const textModeReplacements: Array<[RegExp, string]> = [
    [/\u03b1/g, "alpha"],
    [/\u03b2/g, "beta"],
    [/\u03b3/g, "gamma"],
    [/\u03b4/g, "delta"],
    [/\u03b5/g, "epsilon"],
    [/\u2191/g, "up"],
    [/\u2193/g, "down"]
  ];

  const replacements: Array<[RegExp, string]> = [
    [/\u03b1/g, "\\\\alpha"],
    [/\u03b2/g, "\\\\beta"],
    [/\u03b3/g, "\\\\gamma"],
    [/\u03b4/g, "\\\\delta"],
    [/\u03b5/g, "\\\\epsilon"],
    [/\u2191/g, "\\\\uparrow"],
    [/\u2193/g, "\\\\downarrow"]
  ];

  let output = input.replace(/\\text\{([^{}]*)\}/g, (full, content: string) => {
    let normalizedContent = content;
    for (const [pattern, replacement] of textModeReplacements) {
      normalizedContent = normalizedContent.replace(pattern, replacement);
    }
    return `\\text{${normalizedContent}}`;
  });

  for (const [pattern, replacement] of replacements) {
    output = output.replace(pattern, replacement);
  }
  return output;
}

function remarkNormalizeMathUnicode() {
  return (tree: Node): void => {
    visit(tree, "inlineMath", (node: any) => {
      if (typeof node.value === "string") {
        node.value = normalizeUnicodeMath(node.value);
      }
    });
    visit(tree, "math", (node: any) => {
      if (typeof node.value === "string") {
        node.value = normalizeUnicodeMath(node.value);
      }
    });
  };
}

function detectUnsafeHtmlSignals(markdown: string): number {
  const matches = markdown.match(/<script|javascript:|on[a-z]+\s*=|<iframe|<object|<embed/gi);
  return matches ? matches.length : 0;
}

function hasClassName(node: any, className: string): boolean {
  const classValue = node?.properties?.className;
  if (Array.isArray(classValue)) {
    return classValue.includes(className);
  }
  if (typeof classValue === "string") {
    return classValue.split(/\s+/).includes(className);
  }
  return false;
}

function appendClassName(node: any, className: string): void {
  const current = node?.properties?.className;
  if (Array.isArray(current)) {
    if (!current.includes(className)) {
      current.push(className);
    }
    return;
  }
  if (typeof current === "string") {
    const classes = current.split(/\s+/).filter(Boolean);
    if (!classes.includes(className)) {
      classes.push(className);
    }
    node.properties.className = classes;
    return;
  }
  node.properties.className = [className];
}

function isExternalAbsoluteHttpHref(href: string): boolean {
  return /^https?:\/\//i.test(href);
}

function copyButtonNode(): any {
  return {
    type: "element",
    tagName: "button",
    properties: {
      type: "button",
      className: ["copy-code-btn"],
      dataCopyButton: "true",
      ariaLabel: "Copy code block"
    },
    children: [{ type: "text", value: "Copy" }]
  };
}

function rehypeEnhanceDocumentStructure() {
  return (tree: Node): void => {
    visit(tree, "element", (node: any, index: number | undefined, parent: any) => {
      if (typeof node.tagName !== "string") {
        return;
      }

      const tagName = node.tagName.toLowerCase();
      node.properties = node.properties ?? {};

      if (tagName === "table" && parent && typeof index === "number") {
        if (!(parent.tagName === "div" && hasClassName(parent, "table-scroll"))) {
          parent.children[index] = {
            type: "element",
            tagName: "div",
            properties: {
              className: ["table-scroll"]
            },
            children: [node]
          };
        }
        return;
      }

      if (tagName === "img") {
        appendClassName(node, "media-asset");
        node.properties.loading = "lazy";
        node.properties.decoding = "async";
        node.properties.fetchpriority = "low";
        if (!node.properties.alt) {
          node.properties.alt = "";
        }
      }

      if (tagName === "video" || tagName === "audio") {
        appendClassName(node, "media-asset");
        node.properties.controls = true;
        node.properties.preload = "metadata";
      }

      if (tagName === "a") {
        const href = typeof node.properties.href === "string" ? node.properties.href : "";
        if (isExternalAbsoluteHttpHref(href)) {
          node.properties.target = "_blank";
          node.properties.rel = "noopener noreferrer";
          appendClassName(node, "external-link");
        }
      }

      if (tagName === "pre") {
        appendClassName(node, "code-frame");
        const firstChild = Array.isArray(node.children) ? node.children[0] : undefined;
        const firstTagName = typeof firstChild?.tagName === "string" ? firstChild.tagName.toLowerCase() : "";
        const firstChildClasses = firstChild?.properties?.className;
        if (Array.isArray(firstChildClasses)) {
          const langClass = firstChildClasses.find((value) => typeof value === "string" && value.startsWith("language-"));
          if (typeof langClass === "string") {
            node.properties.dataLanguage = langClass.replace(/^language-/, "");
          }
        } else if (typeof firstChildClasses === "string") {
          const match = firstChildClasses.match(/\blanguage-([a-z0-9_-]+)/i);
          if (match) {
            node.properties.dataLanguage = match[1];
          }
        }
        const hasMermaid =
          (Array.isArray(firstChildClasses) && firstChildClasses.includes("language-mermaid")) ||
          (typeof firstChildClasses === "string" && /\blanguage-mermaid\b/.test(firstChildClasses));
        const alreadyHasButton =
          Array.isArray(node.children) && node.children.some((child: any) => child?.tagName === "button");

        if (firstTagName === "code" && !hasMermaid && !alreadyHasButton) {
          node.children.push(copyButtonNode());
        }
      }
    });
  };
}

function extractHeadings(tree: any): Heading[] {
  const headings: Heading[] = [];
  const usedIds = new Map<string, number>();

  visit(tree, "heading", (node: any) => {
    const depth = Number(node.depth);
    if (!Number.isInteger(depth) || depth < 1 || depth > 6) {
      return;
    }
    const headingText = normalizeWhitespace(toString(node));
    if (!headingText) {
      return;
    }

    const baseId = headingText
      .toLowerCase()
      .replace(/[^a-z0-9\s-]+/g, "")
      .trim()
      .replace(/\s+/g, "-")
      .replace(/-{2,}/g, "-") || "section";

    const count = usedIds.get(baseId) ?? 0;
    usedIds.set(baseId, count + 1);
    const id = count === 0 ? baseId : `${baseId}-${count}`;

    headings.push({
      depth: depth as Heading["depth"],
      text: headingText,
      id
    });
  });

  return headings;
}

function extractMedia(tree: any): MediaRecord[] {
  const media: MediaRecord[] = [];
  const seen = new Set<string>();

  function push(kind: MediaRecord["kind"], src: string): void {
    const key = `${kind}:${src}`;
    if (seen.has(key)) {
      return;
    }
    seen.add(key);
    media.push({ kind, src, resolved: src });
  }

  visit(tree, "image", (node: any) => {
    if (typeof node.url === "string" && node.url.length > 0) {
      push("image", node.url);
    }
  });

  visit(tree, "html", (node: any) => {
    if (typeof node.value !== "string") {
      return;
    }
    const htmlValue = node.value;
    const matches = [...htmlValue.matchAll(/<(img|video|audio|source)[^>]*?src=["']([^"']+)["'][^>]*>/gi)];
    for (const match of matches) {
      const tag = match[1].toLowerCase();
      const src = match[2];
      if (tag === "img") {
        push("image", src);
      } else if (tag === "audio") {
        push("audio", src);
      } else if (tag === "video") {
        push("video", src);
      } else {
        const lower = src.toLowerCase();
        if (lower.endsWith(".mp3") || lower.endsWith(".wav") || lower.endsWith(".ogg")) {
          push("audio", src);
        } else {
          push("video", src);
        }
      }
    }
  });

  return media;
}

function buildExcerpt(text: string): string {
  if (!text) {
    return "No preview available.";
  }
  const truncated = text.slice(0, 220);
  return truncated.length < text.length ? `${truncated.trimEnd()}...` : truncated;
}

export async function renderMarkdownDocument(
  document: SourceDocument
): Promise<Result<RenderedMarkdown, BuildError>> {
  try {
    const mdast = unified().use(remarkParse).use(remarkGfm).use(remarkMath).parse(document.content);
    const headings = extractHeadings(mdast);
    const media = extractMedia(mdast);
    const title = headings.find((heading) => heading.depth === 1)?.text ??
      headings.find((heading) => heading.depth === 2)?.text ??
      document.sourcePath.replace(/\.md$/i, "");

    const unsafeSignalsBefore = detectUnsafeHtmlSignals(document.content);

    const htmlFile = await unified()
      .use(remarkParse)
      .use(remarkGfm)
      .use(remarkMath)
      .use(remarkNormalizeMathUnicode)
      .use(remarkMermaidToSafeHtml)
      .use(remarkRehype, { allowDangerousHtml: true })
      .use(rehypeRaw)
      .use(rehypeSanitize, markdownSanitizeSchema)
      .use(rehypeKatex, {
        strict: "ignore",
        trust: false
      })
      .use(rehypeSlug)
      .use(rehypeAutolinkHeadings, {
        behavior: "append",
        properties: {
          className: ["heading-anchor"],
          ariaHidden: "true"
        },
        content: {
          type: "text",
          value: " #"
        }
      })
      .use(rehypePrettyCode, {
        theme: "github-dark",
        keepBackground: false,
        defaultLang: {
          block: "plaintext",
          inline: "plaintext"
        }
      })
      .use(rehypeEnhanceDocumentStructure)
      .use(rehypeStringify)
      .process(document.content);

    const html = String(htmlFile.value);
    const unsafeSignalsAfter = detectUnsafeHtmlSignals(html);
    const plainText = normalizeWhitespace(toString(mdast));
    const slug = buildSlugFromSourcePath(document.sourcePath, title);

    return ok({
      title,
      chapterNumber: chapterFromTitle(title),
      slug,
      canonicalPath: `/docs/${slug.join("/")}`,
      legacyPaths: [],
      headings,
      html,
      text: plainText,
      excerpt: buildExcerpt(plainText),
      readingMinutes: computeReadingMinutes(plainText),
      media,
      invalidHtmlRemoved: Math.max(0, unsafeSignalsBefore - unsafeSignalsAfter)
    });
  } catch (error) {
    const detail = error instanceof Error ? error.message : String(error);
    return err({
      type: "MARKDOWN_PARSE_FAILED",
      docPath: document.sourcePath,
      detail
    });
  }
}
