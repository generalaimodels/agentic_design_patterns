import { redirect } from "next/navigation";
import { getNavigation } from "@/src/lib/content";

export default function HomePage(): never {
  const navigation = getNavigation();
  const firstPath = navigation.chapters[0]?.canonicalPath ?? navigation.fallback[0]?.canonicalPath ?? "/docs";
  redirect(firstPath);
}
