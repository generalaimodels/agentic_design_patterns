"use client";

import Link from "next/link";
import clsx from "clsx";
import type { ReactElement } from "react";
import type { NavigationData } from "@/content/types";

type LeftRailProps = {
  navigation: NavigationData;
  activePath: string;
  mobileOpen: boolean;
  railId?: string;
  onClose: () => void;
};

function NavList({
  label,
  items,
  activePath,
  onClose
}: {
  label: string;
  items: NavigationData["chapters"];
  activePath: string;
  onClose: () => void;
}): ReactElement {
  return (
    <section className="left-rail__section">
      <h3 className="left-rail__heading">{label}</h3>
      <ul className="left-rail__list">
        {items.map((item) => (
          <li key={item.canonicalPath}>
            <Link
              href={item.canonicalPath}
              onClick={onClose}
              className={clsx("left-rail__item", {
                "left-rail__item--active": item.canonicalPath === activePath,
                "left-rail__item--placeholder": item.kind === "placeholder"
              })}
            >
              <span>{item.title}</span>
            </Link>
          </li>
        ))}
      </ul>
    </section>
  );
}

export function LeftRail({ navigation, activePath, mobileOpen, railId, onClose }: LeftRailProps): ReactElement {
  return (
    <aside
      id={railId}
      className={clsx("left-rail", { "left-rail--open": mobileOpen })}
      role="navigation"
      aria-label="Documentation navigation"
    >
      <div className="left-rail__content">
        <NavList label="Chapters" items={navigation.chapters} activePath={activePath} onClose={onClose} />
        {navigation.fallback.length > 0 ? (
          <NavList label="Additional Docs" items={navigation.fallback} activePath={activePath} onClose={onClose} />
        ) : null}
      </div>
    </aside>
  );
}
