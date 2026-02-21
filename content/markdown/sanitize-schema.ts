import type { Schema } from "hast-util-sanitize";
import { defaultSchema } from "rehype-sanitize";

const globalAttributes = [
  ...(defaultSchema.attributes?.["*"] ?? []),
  "className",
  "id",
  "ariaHidden",
  "ariaLabel",
  "dataMermaid"
];

export const markdownSanitizeSchema: Schema = {
  ...defaultSchema,
  tagNames: [
    ...(defaultSchema.tagNames ?? []),
    "video",
    "audio",
    "source",
    "details",
    "summary",
    "figure",
    "figcaption",
    "mark",
    "span",
    "div",
    "form",
    "label",
    "input",
    "textarea",
    "select",
    "option",
    "button"
  ],
  attributes: {
    ...defaultSchema.attributes,
    "*": globalAttributes,
    a: [...(defaultSchema.attributes?.a ?? []), "href", "title", "target", "rel"],
    img: [...(defaultSchema.attributes?.img ?? []), "src", "alt", "title", "width", "height", "loading"],
    video: ["src", "controls", "width", "height", "poster", "preload", "autoplay", "muted", "loop"],
    audio: ["src", "controls", "preload", "autoplay", "muted", "loop"],
    source: ["src", "type"],
    div: ["className", "id", "dataMermaid"],
    code: [...(defaultSchema.attributes?.code ?? []), "className"],
    form: ["action", "method", "id", "className"],
    label: ["for", "id", "className"],
    input: ["type", "name", "value", "placeholder", "required", "checked", "min", "max", "step", "id", "className"],
    textarea: ["name", "placeholder", "required", "rows", "cols", "id", "className"],
    select: ["name", "required", "id", "className"],
    option: ["value", "selected"],
    button: ["type", "disabled", "name", "value", "id", "className"]
  },
  protocols: {
    ...defaultSchema.protocols,
    href: ["http", "https", "mailto", "#", "/"],
    src: ["http", "https", "data", "/", "./", "../"]
  }
};
