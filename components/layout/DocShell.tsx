"use client";

import { useMemo, useState, type ReactElement } from "react";
import clsx from "clsx";
import type { DocRecord, NavigationData, SearchIndex } from "@/content/types";
import { LeftRail } from "@/components/layout/LeftRail";
import { RightRail } from "@/components/layout/RightRail";
import { MarkdownRenderer } from "@/components/markdown/MarkdownRenderer";

type DocShellProps = {
  doc: DocRecord;
  navigation: NavigationData;
  searchIndex: SearchIndex;
  relatedDocs: DocRecord[];
};

function normalizeHeaderTitle(title: string): string {
  const normalized = title.replace(/^chapter\s+\d+(?:\.\d+)*\s*:\s*/i, "").trim();
  return normalized.length > 0 ? normalized : title;
}

function RailToggleIcon({ side, collapsed }: { side: "left" | "right"; collapsed: boolean }): ReactElement {
  const arrowDirection = side === "left" ? (collapsed ? "right" : "left") : collapsed ? "left" : "right";
  const arrowPath = arrowDirection === "left" ? "M12 6 8 10l4 4" : "M8 6l4 4-4 4";
  const dividerPath = side === "left" ? "M7 4.5v11" : "M13 4.5v11";

  return (
    <svg className="doc-shell__toggle-icon" viewBox="0 0 20 20" aria-hidden="true" focusable="false">
      <rect className="doc-shell__toggle-icon-frame" x="2.5" y="3" width="15" height="14" rx="3" />
      <path className="doc-shell__toggle-icon-divider" d={dividerPath} />
      <path className="doc-shell__toggle-icon-arrow" d={arrowPath} />
    </svg>
  );
}

export function DocShell({ doc, navigation, searchIndex, relatedDocs }: DocShellProps): ReactElement {
  const [leftOpen, setLeftOpen] = useState(false);
  const [rightOpen, setRightOpen] = useState(false);
  const [leftDesktopVisible, setLeftDesktopVisible] = useState(true);
  const [rightDesktopVisible, setRightDesktopVisible] = useState(true);

  const activePath = useMemo(() => doc.canonicalPath, [doc.canonicalPath]);
  const headerTitle = useMemo(() => normalizeHeaderTitle(doc.title), [doc.title]);

  return (
    <div className="doc-shell">
      <header className="doc-shell__header" role="banner">
        <div className="doc-shell__header-side doc-shell__header-side--left">
          <button
            type="button"
            className="doc-shell__toggle doc-shell__toggle--mobile"
            aria-expanded={leftOpen}
            aria-controls="docs-left-rail"
            onClick={() => setLeftOpen((open) => !open)}
          >
            Contents
          </button>
          <button
            type="button"
            className="doc-shell__toggle doc-shell__toggle--desktop"
            aria-label={leftDesktopVisible ? "Collapse chapters rail" : "Expand chapters rail"}
            aria-pressed={!leftDesktopVisible}
            onClick={() => setLeftDesktopVisible((visible) => !visible)}
          >
            <RailToggleIcon side="left" collapsed={!leftDesktopVisible} />
          </button>
        </div>
        <div className="doc-shell__title-wrap">
          <p className="doc-shell__eyebrow">Agentic AI Knowledge Platform</p>
          <h1 className="doc-shell__title" id="doc-title">
            {headerTitle}
          </h1>
        </div>
        <div className="doc-shell__header-side doc-shell__header-side--right">
          <button
            type="button"
            className="doc-shell__toggle doc-shell__toggle--mobile"
            aria-expanded={rightOpen}
            aria-controls="docs-right-rail"
            onClick={() => setRightOpen((open) => !open)}
          >
            Search
          </button>
          <button
            type="button"
            className="doc-shell__toggle doc-shell__toggle--desktop"
            aria-label={rightDesktopVisible ? "Collapse tools rail" : "Expand tools rail"}
            aria-pressed={!rightDesktopVisible}
            onClick={() => setRightDesktopVisible((visible) => !visible)}
          >
            <RailToggleIcon side="right" collapsed={!rightDesktopVisible} />
          </button>
        </div>
      </header>

      <div
        className={clsx("doc-shell__layout", {
          "doc-shell__layout--left-collapsed": !leftDesktopVisible,
          "doc-shell__layout--right-collapsed": !rightDesktopVisible
        })}
      >
        <LeftRail
          railId="docs-left-rail"
          navigation={navigation}
          activePath={activePath}
          mobileOpen={leftOpen}
          onClose={() => setLeftOpen(false)}
        />

        <main className="doc-shell__main" role="main" aria-labelledby="doc-title">
          <article className="doc-shell__article" aria-label="Document content">
            <p className="doc-shell__meta">
              <span>{doc.readingMinutes} min read</span>
              <span>{doc.sourcePath}</span>
            </p>
            <MarkdownRenderer html={doc.html} />
          </article>
        </main>

        <div
          id="docs-right-rail"
          className={clsx("doc-shell__right-wrap", {
            "doc-shell__right-wrap--open": rightOpen
          })}
        >
          <RightRail searchIndex={searchIndex} headings={doc.headings} relatedDocs={relatedDocs} />
        </div>
      </div>
    </div>
  );
}
