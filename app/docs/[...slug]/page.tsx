import { notFound, redirect } from "next/navigation";
import type { ReactElement } from "react";
import { DocShell } from "@/components/layout/DocShell";
import { getAllRouteSlugs, getDocBySlug, getNavigation, getRelatedDocs, getSearchIndex } from "@/src/lib/content";

export const dynamicParams = false;

export function generateStaticParams(): Array<{ slug: string[] }> {
  return getAllRouteSlugs().map((slug) => ({ slug }));
}

type RouteProps = {
  params: Promise<{ slug: string[] }>;
};

export default async function DocumentPage({ params }: RouteProps): Promise<ReactElement> {
  const route = await params;
  const resolved = getDocBySlug(route.slug);

  if (resolved.redirectTo) {
    redirect(resolved.redirectTo);
  }

  if (!resolved.doc) {
    notFound();
  }

  return (
    <DocShell
      doc={resolved.doc}
      navigation={getNavigation()}
      searchIndex={getSearchIndex()}
      relatedDocs={getRelatedDocs(resolved.doc.id)}
    />
  );
}
