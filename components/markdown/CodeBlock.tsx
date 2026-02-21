"use client";

import { useState, type ReactElement } from "react";

type CodeBlockProps = {
  code: string;
  language?: string;
};

export function CodeBlock({ code, language = "text" }: CodeBlockProps): ReactElement {
  const [copied, setCopied] = useState(false);

  const onCopy = async (): Promise<void> => {
    try {
      await navigator.clipboard.writeText(code);
      setCopied(true);
      window.setTimeout(() => setCopied(false), 1200);
    } catch {
      setCopied(false);
    }
  };

  return (
    <pre className="code-block" data-language={language}>
      <code>{code}</code>
      <button type="button" className="copy-code-btn" onClick={onCopy}>
        {copied ? "Copied" : "Copy"}
      </button>
    </pre>
  );
}
