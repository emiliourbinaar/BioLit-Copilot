// @ts-check
import { defineConfig } from "astro/config";

// Static output only: no server, no adapter, no runtime fetch. Everything the site shows is
// read at build time from the committed fixtures and docs.
export default defineConfig({
  output: "static",
  // Hosted on GitHub Pages as a project site, so every page lives under /BioLit-Copilot/.
  // The trailing slash is load-bearing: components build links as `${BASE_URL}runs/...`.
  site: "https://emiliourbinaar.github.io",
  base: "/BioLit-Copilot/",
  vite: {
    // The docs pages quote DEFECTS.md and RETROSPECTIVE.md verbatim, imported with `?raw`
    // from the repo root, which is outside this project's root.
    server: { fs: { allow: [".."] } },
  },
});
