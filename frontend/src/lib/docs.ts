/**
 * Read the project's own records at build time, so the site quotes them rather than copies.
 *
 * The docs are imported with Vite's `?raw` from the repo root. Nothing here fetches at runtime.
 */

export interface Defect {
  id: string;
  title: string;
  /** The entry's `Status:` field, verbatim markdown with its line wrapping collapsed. */
  status: string;
}

const HEADING = /^(DEF-\d{4}) — (.+)$/;

export function parseDefects(markdown: string): Defect[] {
  const defects: Defect[] = [];
  for (const section of markdown.split("\n## ").slice(1)) {
    const [heading = "", ...body] = section.split("\n");
    const match = HEADING.exec(heading.trim());
    if (!match) continue;
    defects.push({ id: match[1]!, title: match[2]!.trim(), status: field(body, "Status") });
  }
  return defects;
}

export interface InstrumentFailure {
  n: number;
  title: string;
  /** The rest of the paragraph, verbatim markdown with its line wrapping collapsed. */
  body: string;
}

const ITEM = /^\*\*(\d+)\. (.+?)\*\*\s*/;

/** RETROSPECTIVE.md §4: its heading, and each `**N. Title.**` paragraph beneath it. */
export function parseInstrumentFailures(markdown: string): {
  heading: string;
  items: InstrumentFailure[];
} {
  const start = markdown.search(/^## 4\. /m);
  if (start === -1) throw new Error("RETROSPECTIVE.md has no §4");
  const rest = markdown.slice(start);
  const end = rest.slice(1).search(/^##+ /m);
  const section = end === -1 ? rest : rest.slice(0, end + 1);

  const [headingLine = "", ...body] = section.split("\n");
  const heading = headingLine.replace(/^## 4\.\s*/, "").replace(/⭐\s*/u, "").trim();

  const items: InstrumentFailure[] = [];
  for (const paragraph of body.join("\n").split(/\n\s*\n/)) {
    const text = paragraph.replace(/\s+/g, " ").trim();
    const match = ITEM.exec(text);
    if (!match) continue;
    items.push({
      n: Number(match[1]),
      title: match[2]!.replace(/\.$/, ""),
      body: text.slice(match[0].length),
    });
  }
  return { heading, items };
}

/** A `- **Name:** ...` bullet, including its indented continuation lines. */
function field(lines: string[], name: string): string {
  const start = lines.findIndex((line) => line.startsWith(`- **${name}:**`));
  if (start === -1) return "";
  const collected = [lines[start]!.slice(`- **${name}:**`.length)];
  for (const line of lines.slice(start + 1)) {
    if (!line.startsWith("  ")) break;
    collected.push(line);
  }
  return collected.join(" ").replace(/\s+/g, " ").trim();
}
