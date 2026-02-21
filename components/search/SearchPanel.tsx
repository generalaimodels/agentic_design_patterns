"use client";

import Link from "next/link";
import { useId, useMemo, useState, type ReactElement } from "react";
import { z } from "zod";
import type { SearchIndex } from "@/content/types";
import { rankSearchResults } from "@/src/lib/search";

type SearchPanelProps = {
  index: SearchIndex;
};

const querySchema = z.string().max(200);

export function SearchPanel({ index }: SearchPanelProps): ReactElement {
  const [query, setQuery] = useState("");
  const inputId = `search-input-${useId().replace(/:/g, "")}`;
  const docsById = useMemo(
    () => new Map(index.documents.map((doc) => [doc.id, doc])),
    [index.documents]
  );

  const results = useMemo(() => {
    const trimmed = query.trim();
    if (trimmed.length < 2) {
      return [];
    }

    const ranked = rankSearchResults(index, trimmed, 12);
    return ranked
      .map((entry) => {
        const doc = docsById.get(entry.docId);
        if (!doc) {
          return undefined;
        }
        return {
          id: doc.id,
          title: doc.title,
          canonicalPath: doc.canonicalPath,
          excerpt: doc.excerpt,
          score: entry.score
        };
      })
      .filter((item): item is NonNullable<typeof item> => Boolean(item));
  }, [docsById, index, query]);

  return (
    <div className="search-panel">
      <label className="search-panel__label" htmlFor={inputId}>
        Search documents
      </label>
      <input
        id={inputId}
        type="search"
        value={query}
        onChange={(event) => {
          const candidate = event.target.value;
          const parsed = querySchema.safeParse(candidate);
          if (parsed.success) {
            setQuery(parsed.data);
            return;
          }
          setQuery(candidate.slice(0, 200));
        }}
        placeholder="Search by title, term, concept"
        className="search-panel__input"
      />
      <ul className="search-panel__results">
        {results.map((result) => (
          <li key={result.id} className="search-panel__result">
            <Link href={result.canonicalPath} className="search-panel__link">
              <span className="search-panel__result-title">{result.title}</span>
              <span className="search-panel__result-excerpt">{result.excerpt}</span>
            </Link>
          </li>
        ))}
      </ul>
      {query.trim().length < 2 ? <p className="search-panel__empty">Type at least 2 characters.</p> : null}
      {query.trim().length >= 2 && results.length === 0 ? <p className="search-panel__empty">No results</p> : null}
    </div>
  );
}
