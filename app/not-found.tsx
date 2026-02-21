import Link from "next/link";
import type { ReactElement } from "react";

export default function NotFound(): ReactElement {
  return (
    <main className="not-found-page">
      <h1>Document not found</h1>
      <p>The requested documentation route does not exist in the generated manifest.</p>
      <Link href="/" className="not-found-page__link">
        Go to documentation home
      </Link>
    </main>
  );
}
