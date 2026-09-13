/**
 * Link check over the BUILT site, run in CI after `astro build`.
 *
 * The site is served from a sub-path (`base`) on GitHub Pages. A root-absolute link written
 * without it works under `astro dev` and 404s once deployed, and no type check sees that.
 */

const LINK = /\s(?:href|src)="([^"]*)"/g;

/**
 * @param {Record<string, string>} files built files, keyed by path relative to `dist/`
 * @param {string} base the configured base path, with its trailing slash
 * @returns {{ file: string, href: string, reason: string }[]}
 */
export function brokenLinks(files, base) {
  const broken = [];
  for (const [file, text] of Object.entries(files)) {
    if (!file.endsWith(".html")) continue;
    for (const [, href] of text.matchAll(LINK)) {
      if (/^[a-z]+:/i.test(href)) continue;
      const [path = "", fragment] = href.split("#");
      if (path === "") {
        if (fragment && !idsIn(text).has(fragment)) {
          broken.push({ file, href, reason: "no such id" });
        }
        continue;
      }
      if (href.startsWith("/") && !href.startsWith(base)) {
        broken.push({ file, href, reason: "outside the base path" });
        continue;
      }
      const target = resolve(file, path, base);
      const page = target in files ? target : `${target}index.html`;
      if (!(page in files)) {
        broken.push({ file, href, reason: "no such file" });
      } else if (fragment && !idsIn(files[page]).has(fragment)) {
        broken.push({ file, href, reason: "no such id" });
      }
    }
  }
  return broken;
}

/** The `dist/`-relative path a link opens: `/base/runs/x/` or `../x/` -> `runs/x/`. */
function resolve(file, path, base) {
  const parts = path.startsWith("/")
    ? path.slice(base.length).split("/")
    : [...file.split("/").slice(0, -1), ...path.split("/")];
  const out = [];
  for (const part of parts) {
    if (part === "..") out.pop();
    else if (part !== "." && part !== "") out.push(part);
  }
  const joined = out.join("/");
  return path.endsWith("/") && joined ? `${joined}/` : joined;
}

function idsIn(html) {
  return new Set([...html.matchAll(/\sid="([^"]+)"/g)].map((match) => match[1]));
}
