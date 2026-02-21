"use client";

import { useRouter } from "next/navigation";
import { useEffect, useId, type ReactElement } from "react";
import { MermaidBlock } from "@/components/markdown/MermaidBlock";

type MarkdownRendererProps = {
  html: string;
};

function isModifiedClick(event: MouseEvent): boolean {
  return event.metaKey || event.ctrlKey || event.shiftKey || event.altKey;
}

function shouldUseSpaNavigation(href: string): boolean {
  return href.startsWith("/") || href.startsWith("./") || href.startsWith("../");
}

function resolveHrefToPath(href: string): string {
  const resolved = new URL(href, window.location.href);
  return `${resolved.pathname}${resolved.search}${resolved.hash}`;
}

export function MarkdownRenderer({ html }: MarkdownRendererProps): ReactElement {
  const rawId = useId();
  const containerId = `markdown-${rawId.replace(/:/g, "")}`;
  const router = useRouter();

  useEffect(() => {
    const container = document.getElementById(containerId);
    if (!container) {
      return;
    }

    const clickHandler = async (event: MouseEvent): Promise<void> => {
      const target = event.target as HTMLElement | null;
      if (!target) {
        return;
      }

      const copyButton = target.closest(".copy-code-btn") as HTMLButtonElement | null;
      if (copyButton) {
        const pre = copyButton.closest("pre");
        const code = pre?.querySelector("code");
        const payload = code?.textContent ?? "";
        if (!payload) {
          return;
        }
        try {
          await navigator.clipboard.writeText(payload);
          copyButton.textContent = "Copied";
          window.setTimeout(() => {
            copyButton.textContent = "Copy";
          }, 1200);
        } catch {
          copyButton.textContent = "Error";
          window.setTimeout(() => {
            copyButton.textContent = "Copy";
          }, 1200);
        }
        event.preventDefault();
        return;
      }

      const anchor = target.closest("a[href]") as HTMLAnchorElement | null;
      if (!anchor) {
        return;
      }
      if (event.defaultPrevented || isModifiedClick(event) || event.button !== 0) {
        return;
      }

      const href = anchor.getAttribute("href") ?? "";
      if (!href || href.startsWith("#")) {
        return;
      }
      if (!shouldUseSpaNavigation(href)) {
        return;
      }

      event.preventDefault();
      router.push(resolveHrefToPath(href));
    };

    container.addEventListener("click", clickHandler);

    return () => {
      container.removeEventListener("click", clickHandler);
    };
  }, [containerId, router, html]);

  return (
    <section className="markdown-host">
      <div id={containerId} className="markdown-body" dangerouslySetInnerHTML={{ __html: html }} />
      <MermaidBlock containerId={containerId} />
    </section>
  );
}
