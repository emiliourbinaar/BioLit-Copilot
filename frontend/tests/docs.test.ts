import { describe, expect, it } from "vitest";
import defectsMarkdown from "../../docs/DEFECTS.md?raw";
import retrospectiveMarkdown from "../../docs/RETROSPECTIVE.md?raw";
import { parseDefects, parseInstrumentFailures } from "../src/lib/docs";
import { RUNS } from "../src/lib/fixtures";

describe("parseDefects", () => {
  it("reads every entry's id, title and status from DEFECTS.md itself, not from a copy", () => {
    // The /findings page quotes the defect log. A hand-copied list in TypeScript would drift
    // from the log the first time an entry's status changed -- DEF-0007's already did once.
    const defects = parseDefects(defectsMarkdown);
    const ids = defects.map((d) => d.id);

    expect(ids).toContain("DEF-0001");
    expect(ids).toContain("DEF-0007");
    expect(new Set(ids).size).toBe(ids.length);
    for (const defect of defects) {
      expect(defect.title, defect.id).not.toBe("");
      expect(defect.status, defect.id).not.toBe("");
    }

    // End to end: a callout's headline in a committed fixture IS the log's title for that id.
    for (const run of RUNS) {
      for (const finding of run.findings) {
        expect(defects.find((d) => d.id === finding.defect_id)?.title).toBe(finding.headline);
      }
    }
  });
});

describe("parseInstrumentFailures", () => {
  it("reads the retrospective's instrument-failure section, heading and every numbered item", () => {
    // The home page leads with these. Quoted from RETROSPECTIVE.md §4 so the count in its
    // heading and the items beneath it can never disagree with the document.
    const section = parseInstrumentFailures(retrospectiveMarkdown);

    expect(section.heading).toMatch(/instrument/i);
    expect(section.heading).not.toMatch(/⭐/);
    expect(section.items.length).toBeGreaterThan(0);
    expect(section.items.map((item) => item.n)).toEqual(
      section.items.map((_, index) => index + 1),
    );
    for (const item of section.items) {
      expect(item.title, `item ${item.n}`).not.toBe("");
      expect(item.body, `item ${item.n}`).not.toBe("");
      expect(item.body, `item ${item.n}`).not.toMatch(/^\*\*\d\./);
    }
  });
});
