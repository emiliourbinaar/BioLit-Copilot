import { describe, expect, it } from "vitest";
import {
  assertNotesHold,
  citedPaperIds,
  countOf,
  ledgerArithmetic,
  licenceLabel,
  parseAnchor,
  queryConcepts,
  RUN_NOTES,
  RUNS,
  stageByName,
  tiedWithPrevious,
  validateFixture,
} from "../src/lib/fixtures";

describe("RUN_NOTES", () => {
  it("carries, for every run, a predicate that holds against that run's committed fixture", () => {
    // Each note is a claim about a live retrieval. The predicate is what makes it checkable,
    // so a regeneration that falsifies a note is caught rather than published.
    for (const run of RUNS) {
      const note = RUN_NOTES[run.slug]!;
      expect(typeof note.text).toBe("string");
      expect(note.holds(run), `${run.slug}: note no longer true`).toBe(true);
    }
  });

  it("refuses to publish a note its run does not bear out", () => {
    // Statins' data under cisplatin's note: 17 clusters, not three tied ones. The note would be
    // a false sentence on a public page, so the build must stop.
    const statins = RUNS.find((run) => run.slug === "statins-rhabdomyolysis")!;
    const impostor = { ...statins, slug: "cisplatin-nephrotoxicity" };

    expect(() => assertNotesHold([impostor])).toThrow(/cisplatin-nephrotoxicity/);
    expect(() => assertNotesHold(RUNS)).not.toThrow();
  });
});

describe("ledgerArithmetic", () => {
  it("spells out the subtraction where units match, and refuses to where they change", () => {
    // Derived from the fixture, not hard-coded: this test once asserted statins' gate dropped a
    // `duplicate_paper_id`, a fact about one retrieval that DEF-0008's fix made false. The site
    // shows the sum for whatever the run did; the test holds it to the run's own `dropped`.
    const statins = RUNS.find((run) => run.slug === "statins-rhabdomyolysis")!;
    const gate = stageByName(statins, "licence_gate")!;
    const sum = ledgerArithmetic(gate)!;

    expect(sum.n_in).toBe(gate.n_in);
    expect(sum.terms.map((t) => t.reason).sort()).toEqual(Object.keys(gate.dropped ?? {}).sort());
    expect(sum.n_in - sum.terms.reduce((acc, t) => acc + t.count, 0)).toBe(sum.n_out);
    expect(sum.balances).toBe(true);

    // pmids -> papers, papers -> entities: an undeclared unit change reads as impossible growth,
    // so no sum is offered at all.
    expect(ledgerArithmetic(stageByName(statins, "retrieve")!)).toBeNull();
    expect(ledgerArithmetic(stageByName(statins, "ner_linking")!)).toBeNull();

    const broken = { ...gate, n_out: gate.n_out - 1 };
    expect(ledgerArithmetic(broken)!.balances).toBe(false);
  });
});

// ⚠️ TEST-AFTER, recorded as such: parseAnchor, licenceLabel and citedPaperIds were written in
// the module's first draft, before these tests existed. They pin behaviour; they were not RED.
describe("parseAnchor", () => {
  it("reads both anchor kinds the Python grammar defines, and refuses anything else", () => {
    expect(parseAnchor("stage:licence_gate/dropped/duplicate_paper_id")).toEqual({
      kind: "stage",
      stage: "licence_gate",
      reason: "duplicate_paper_id",
    });
    expect(parseAnchor("cluster:MESH:D002945|MESH:D007674")).toEqual({
      kind: "cluster",
      key: "MESH:D002945|MESH:D007674",
    });
    expect(() => parseAnchor("licence_gate")).toThrow(/anchor/);
    expect(() => parseAnchor("cluster:")).toThrow(/anchor/);
    for (const run of RUNS) {
      for (const finding of run.findings) expect(() => parseAnchor(finding.anchor)).not.toThrow();
    }
  });
});

describe("countOf", () => {
  it("agrees a ledger unit with its count, for every unit the committed fixtures use", () => {
    expect(countOf(1, "papers")).toBe("1 paper");
    expect(countOf(1, "entities")).toBe("1 entity");
    expect(countOf(2, "pmids")).toBe("2 pmids");
    expect(countOf(0, "clusters")).toBe("0 clusters");
    const units = new Set(RUNS.flatMap((r) => r.stages.flatMap((s) => [s.unit_in, s.unit_out])));
    for (const unit of units) expect(countOf(1, unit ?? "")).not.toMatch(/s$/);
  });
});

describe("licenceLabel", () => {
  it("spells the fixture's licence token without changing what it says", () => {
    expect(licenceLabel("cc_by")).toBe("CC BY");
    expect(licenceLabel("cc_by_nc_nd")).toBe("CC BY-NC-ND");
    expect(licenceLabel(null)).toBe("no licence");
    expect(licenceLabel("publisher_specific")).toBe("publisher_specific");
  });
});

describe("citedPaperIds", () => {
  it("lists every paper a cluster cites once, and every one has a stub to attribute", () => {
    for (const run of RUNS) {
      const cited = citedPaperIds(run);
      expect(new Set(cited).size).toBe(cited.length);
      for (const id of cited) {
        expect(run.papers[id]?.license, `${run.slug}: ${id}`).toBeTruthy();
        expect(run.papers[id]?.doi, `${run.slug}: ${id}`).toBeTruthy();
      }
    }
  });
});

describe("validateFixture", () => {
  it("refuses a schema version it does not understand, and a slug that disagrees with its file", () => {
    // A fixture the site cannot fully render must fail the build, not render partially.
    const statins = RUNS.find((run) => run.slug === "statins-rhabdomyolysis")!;
    const path = "../fixtures/statins-rhabdomyolysis.json";

    expect(validateFixture(path, statins)).toBe(statins);
    expect(() => validateFixture(path, { ...statins, schema_version: 2 })).toThrow(/schema_version/);
    expect(() => validateFixture(path, { ...statins, slug: "other" })).toThrow(/filename/);
    expect(() => validateFixture(path, { ...statins, findings: undefined })).toThrow(/findings/);
  });
});

describe("queryConcepts", () => {
  it("reads the concepts a question resolved to from the select stage's own note", () => {
    // Cisplatin's tie is explained by this: one linked concept, the drug, so no disease side
    // has anything to be near. Isotretinoin resolved two, and its tree distances vary. The site
    // must read that from what the stage said, never re-resolve the question itself.
    const bySlug = (slug: string) => RUNS.find((run) => run.slug === slug)!;
    expect(queryConcepts(bySlug("cisplatin-nephrotoxicity"))).toEqual(["cisplatin"]);
    expect(queryConcepts(bySlug("isotretinoin-depression"))).toEqual([
      "depressive disorder",
      "isotretinoin",
    ]);
    const silent = { ...bySlug("cisplatin-nephrotoxicity"), stages: [] };
    expect(queryConcepts(silent)).toEqual([]);
  });
});

describe("tiedWithPrevious", () => {
  it("marks a cluster the ranker could not separate from the one above it, and only those", () => {
    // Where both signals tie, the order is MeSH-id order and means nothing about relevance.
    // A reader must be able to see that, rather than read rank 1 as "most relevant".
    const cisplatin = RUNS.find((run) => run.slug === "cisplatin-nephrotoxicity")!;
    const byRank = [...cisplatin.clusters].sort((a, b) => a.rank - b.rank);
    // Every cisplatin cluster ties (one linked query concept, so no proximity signal), and the
    // first has nothing above it. Shape from the data, not a count from one retrieval.
    expect(byRank.map((c) => tiedWithPrevious(cisplatin, c))).toEqual(
      byRank.map((_, index) => index > 0),
    );

    const isotretinoin = RUNS.find((run) => run.slug === "isotretinoin-depression")!;
    const first = isotretinoin.clusters.find((c) => c.rank === 1)!;
    const second = isotretinoin.clusters.find((c) => c.rank === 2)!;
    expect(tiedWithPrevious(isotretinoin, second)).toBe(first.proximity === second.proximity);
  });
});
