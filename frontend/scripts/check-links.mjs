/**
 * `npm run check:links`, after `npm run build`. Exits non-zero on any broken internal link.
 * I/O only: the logic is `brokenLinks`, tested in `tests/links.test.ts`. `base` is read from the
 * Astro config itself, so the check cannot drift from what the site was built with.
 */
import { readdir, readFile } from "node:fs/promises";
import path from "node:path";
import { fileURLToPath } from "node:url";
import config from "../astro.config.mjs";
import { brokenLinks } from "./links.mjs";

const dist = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..", "dist");
const files = {};
for (const entry of await readdir(dist, { recursive: true, withFileTypes: true })) {
  if (!entry.isFile()) continue;
  const full = path.join(entry.parentPath, entry.name);
  const key = path.relative(dist, full).split(path.sep).join("/");
  files[key] = key.endsWith(".html") ? await readFile(full, "utf8") : "";
}

const broken = brokenLinks(files, config.base ?? "/");
const pages = Object.keys(files).filter((file) => file.endsWith(".html")).length;
for (const { file, href, reason } of broken) console.error(`${file}: ${href} (${reason})`);
console.log(`${pages} pages checked under base ${config.base}; ${broken.length} broken links.`);
process.exit(broken.length > 0 ? 1 : 0);
