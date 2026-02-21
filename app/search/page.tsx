import Link from "next/link";
import type { ReactElement } from "react";
import { SearchPanel } from "@/components/search/SearchPanel";
import { getSearchIndex } from "@/src/lib/content";

export default function SearchPage(): ReactElement {
  return (
    <main className="search-page">
      <header className="search-page__header">
        <h1>Search</h1>
        <Link href="/" className="search-page__home-link">
          Back to docs
        </Link>
      </header>
      <SearchPanel index={getSearchIndex()} />
    </main>
  );
}
