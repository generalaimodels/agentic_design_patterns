"use client";

import { useEffect } from "react";

type MermaidBlockProps = {
  containerId: string;
};

let mermaidInitialized = false;

export function MermaidBlock({ containerId }: MermaidBlockProps): null {
  useEffect(() => {
    const container = document.getElementById(containerId);
    if (!container) {
      return;
    }

    const blocks = Array.from(container.querySelectorAll("pre.mermaid-block > code.language-mermaid"));
    if (blocks.length === 0) {
      return;
    }

    void import("mermaid").then((module) => {
      const mermaid = module.default;

      if (!mermaidInitialized) {
        mermaid.initialize({
          startOnLoad: false,
          securityLevel: "strict",
          theme: "neutral"
        });
        mermaidInitialized = true;
      }

      for (const [index, block] of blocks.entries()) {
        const source = block.textContent?.trim();
        const hostPre = block.closest("pre");
        if (!source || !hostPre || hostPre.dataset.mermaidRendered === "true") {
          continue;
        }

        const renderTarget = document.createElement("div");
        renderTarget.className = "mermaid-render";

        const graphId = `${containerId}-mermaid-${index}`;
        void mermaid
          .render(graphId, source)
          .then((result) => {
            renderTarget.innerHTML = result.svg;
            hostPre.replaceWith(renderTarget);
          })
          .catch(() => {
            hostPre.dataset.mermaidRendered = "true";
          });
      }
    });
  }, [containerId]);

  return null;
}
