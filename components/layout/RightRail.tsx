"use client";

import Link from "next/link";
import type { ReactElement } from "react";
import type { DocRecord, RelatedDoc, SearchIndex } from "@/content/types";
import { SearchPanel } from "@/components/search/SearchPanel";

type RightRailProps = {
  searchIndex: SearchIndex;
  headings: DocRecord["headings"];
  relatedDocs: RelatedDoc[];
};

export function RightRail({ searchIndex, headings, relatedDocs }: RightRailProps): ReactElement {
  const tocHeadings = headings.filter((heading) => heading.depth >= 2 && heading.depth <= 3);

  return (
    <aside className="right-rail" role="complementary" aria-label="Document tools and related navigation">
      <section className="right-rail__section">
        <h3 className="right-rail__title">On this page</h3>
        {tocHeadings.length === 0 ? (
          <p className="right-rail__empty">No section headings available.</p>
        ) : (
          <ul className="right-rail__list">
            {tocHeadings.map((heading) => (
              <li key={heading.id} className={`right-rail__toc-depth-${heading.depth}`}>
                <a href={`#${heading.id}`}>{heading.text}</a>
              </li>
            ))}
          </ul>
        )}
      </section>

      <section className="right-rail__section right-rail__search">
        <h3 className="right-rail__title">Search</h3>
        <SearchPanel index={searchIndex} />
      </section>

      <section className="right-rail__section">
        <h3 className="right-rail__title">Related documents</h3>
        {relatedDocs.length === 0 ? (
          <p className="right-rail__empty">No related documents found.</p>
        ) : (
          <ul className="right-rail__list">
            {relatedDocs.map((doc) => (
              <li key={doc.id}>
                <Link href={doc.canonicalPath}>{doc.title}</Link>
              </li>
            ))}
          </ul>
        )}
      </section>
    </aside>
  );
}
