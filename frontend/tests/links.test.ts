import { describe, expect, it } from "vitest";
import { brokenLinks } from "../scripts/links.mjs";

const BASE = "/BioLit-Copilot/";

describe("brokenLinks", () => {
  it("reports a root-absolute link that ignores the base path, which works locally and 404s on Pages", () => {
    const files = {
      "index.html": `<a href="/runs/statins/">statins</a>`,
      "runs/statins/index.html": `<h1>statins</h1>`,
    };

    expect(brokenLinks(files, BASE)).toEqual([
      { file: "index.html", href: "/runs/statins/", reason: "outside the base path" },
    ]);
  });

  it("reports a link to a page or asset the build did not produce, and accepts one it did", () => {
    const files = {
      "index.html": [
        `<a href="/BioLit-Copilot/runs/statins/">ok</a>`,
        `<link rel="icon" href="/BioLit-Copilot/favicon.svg">`,
        `<a href="runs/statins/">relative ok</a>`,
        `<a href="/BioLit-Copilot/runs/warfarin/">missing</a>`,
        `<a href="https://doi.org/10.1/x">external</a>`,
      ].join(""),
      "runs/statins/index.html": `<a href="../../findings/">missing relative</a>`,
      "favicon.svg": "<svg/>",
    };

    expect(brokenLinks(files, BASE)).toEqual([
      { file: "index.html", href: "/BioLit-Copilot/runs/warfarin/", reason: "no such file" },
      { file: "runs/statins/index.html", href: "../../findings/", reason: "no such file" },
    ]);
  });

  it("reports a fragment that names no id on its target page, on the same page or another", () => {
    const files = {
      "findings/index.html": [
        `<li id="DEF-0009"></li>`,
        `<a href="#DEF-0009">ok</a>`,
        `<a href="#DEF-0010">missing here</a>`,
        `<a href="/BioLit-Copilot/runs/statins/#finding-DEF-0007">ok there</a>`,
        `<a href="/BioLit-Copilot/runs/statins/#finding-DEF-0001">missing there</a>`,
      ].join(""),
      "runs/statins/index.html": `<aside id="finding-DEF-0007"></aside>`,
    };

    expect(brokenLinks(files, BASE)).toEqual([
      { file: "findings/index.html", href: "#DEF-0010", reason: "no such id" },
      {
        file: "findings/index.html",
        href: "/BioLit-Copilot/runs/statins/#finding-DEF-0001",
        reason: "no such id",
      },
    ]);
  });
});
