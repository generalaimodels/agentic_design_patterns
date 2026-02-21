import type { Metadata } from "next";
import type { ReactElement, ReactNode } from "react";
import { Source_Serif_4, Space_Grotesk } from "next/font/google";
import "../styles/tokens.css";
import "../styles/globals.css";
import "../styles/docs.css";
import "katex/dist/katex.min.css";

const displayFont = Space_Grotesk({
  subsets: ["latin"],
  variable: "--font-display",
  weight: ["400", "500", "600", "700"]
});

const bodyFont = Source_Serif_4({
  subsets: ["latin"],
  variable: "--font-body",
  weight: ["400", "500", "600", "700"]
});

export const metadata: Metadata = {
  title: "Agentic AI Knowledge Platform",
  description: "High-fidelity markdown knowledge system with deterministic indexing"
};

export default function RootLayout({ children }: { children: ReactNode }): ReactElement {
  return (
    <html lang="en">
      <body className={`${displayFont.variable} ${bodyFont.variable}`}>{children}</body>
    </html>
  );
}
