# End-to-end pipeline — wiring Phases 1–5 through `PipelineState`

- **Date:** 2026-08-31
- **Status:** Approved, not yet planned or implemented
- **Kind:** Integration sub-project. **Not a new measurement question** — no new metric, no gold standard, no experiment. Its purpose is to close the gap the project has flagged repeatedly: five phases of individually validated components that have never been connected to each other in a running system.

## Summary

Build a CLI that takes a natural-language query and runs it through the real components: PubMed retrieval → NER → canonicalization → deterministic extraction → clustering, printing a per-stage report and dumping the final `PipelineState` as JSON. The Critic and Synthesis stages are **explicitly stubbed and reported as unimplemented**, never silently skipped.

**No LLM calls. No paid API calls. No credential is required at any point.** Every component on the path is either local (the NER checkpoint, the MeSH dictionary) or a free NCBI endpoint. There is no pricing step and no authorization gate, because there is nothing to authorize.

**One blocking defect must be fixed first**, and §0 establishes it by measurement rather than assertion: PMC licence classification is broken against the live service, so *every* retrieved paper currently classifies as unextractable and a naive wiring would produce an entirely empty pipeline.

---

## §0 — What was measured before anything was designed

Following the project's standing practice: the facts below were measured against the live NCBI services on 2026-08-31, not read off the source or remembered.

### The configured PMC OA endpoint is dead

```
GET https://www.ncbi.nlm.nih.gov/pmc/utils/oa/oa.fcgi          -> HTTP 404
GET https://pmc.ncbi.nlm.nih.gov/utils/oa/oa.fcgi              -> HTTP 404
```

Both return 404 **with no parameters at all**, serving an NCBI "WWW Error 404 Diagnostic" HTML page rather than the service's own XML error document. The path no longer exists. `_PMC_OA` in `biolit/clients/pubmed.py` points at it.

### The consequence: nothing is extractable, and the pipeline would be empty

| query | papers | `extraction_allowed` | PMC OA 404s |
|---|---|---|---|
| `metformin cardiovascular outcomes` | 20 | **0** | 11 |
| the same, restricted to the PMC **open-access subset** | 20 | **0** | 20 |

All 20 papers in each run had abstracts. The second row is the diagnostic one: papers that are *definitionally* open access classify as `unknown`. That is what makes this infrastructure rot rather than an ambiguous policy question.

The causal chain is entirely deterministic: `_classify_pmc` cannot reach the service → `normalize_license(None)` → `LicenseTier.unknown` → `extraction_allowed_for(unknown) is False` → `build_record` returns `None` for every paper. A naive wiring today yields **20 candidate papers → 0 extracted records → 0 clusters → 0 findings**, with every component behaving exactly as written.

### This has never been exercised, which is the gap this sub-project exists to close

Every eval constructs its papers with `extraction_allowed=True` (e.g. `biolit_evals/extract_eval.py:1109`). The licence gate has therefore never run against real retrieved data in the project's history. It is the clearest single instance of the "validated components, never connected" gap.

### The `efetch` 404 defect and the licence defect are the same defect

The known, deliberately-unfixed defect — `PubMedClient.efetch` raising on an HTTP 404 from the PMC OA service — has the dead endpoint as its **cause**, not merely its trigger. It fired on the first realistic query attempted, on 11 of 20 PMIDs. Phase 5 routed around it via `efetch_abstracts`, which performs no PMC lookup; this pipeline cannot, because it needs `Paper` objects carrying licence tiers. Replacing the dead call removes the 404 source. **One fix, not two.**

### A working replacement exists on NCBI's own infrastructure

`efetch db=pmc` on the standard eutils host — already used elsewhere in the client — returns the article's own `<permissions>` block, which is the **publisher's licence statement**, not an inference from PMC membership. This preserves the Phase 1 rule *"never infer rights from PMC presence"* exactly as written.

It discriminates correctly, which is the bar this fix must clear — truthfulness, not mere availability:

| case | response | classification |
|---|---|---|
| `PMC8917620` (open-access subset) | `ali:license_ref` = `https://creativecommons.org/licenses/by/4.0/`, `<body>` present | permitted — correct |
| `PMC1401093` (the Phase 5 404) | **5,968-byte stub, no `<permissions>`, no `<body>`** | refused — correct |

Across 60 papers from three real clinical queries, 41 carried a PMC id:

| count | licence as reported |
|---:|---|
| 17 | `creativecommons.org/licenses/by/4.0/` |
| 6 | `creativecommons.org/licenses/by-nc-nd/4.0/` |
| 5 | `creativecommons.org/licenses/by-nc/4.0/` |
| 2 | `creativecommons.org/licenses/by-nc-sa/4.0/` |
| 1 | `creativecommons.org/licenses/by-nc/3.0/` |
| 1 | `creativecommons.org/licenses/by-nc-nd/3.0/` |
| 4 | `"This file is available for text mining…"` (NIH manuscript statement, no CC token) |
| 1 each | Cochrane editorial policies · diabetesjournals licence · ACS AuthorChoice · Sage reuse guidelines · `None` |

3 of 41 were bodyless stubs. Under the **existing** tier table this permits **32 of 41** — about **53% of retrieved papers** — while the publisher-specific terms correctly fail closed. The mechanism produces real output without blanket-permitting anything.

---

## §1 — The licence classification fix

This is step one and everything else depends on it.

### What changes

- `_PMC_OA` and the per-article OA lookup are removed.
- A new module-level helper reads the licence from a PMC article element: the `ali:license_ref` URL when present, falling back to the `<license>` element's text.
- A **URL → canonical token** normalizer feeds the **existing** `normalize_license` and `extraction_allowed_for`.

**Classification keys on the `<permissions>` block alone.** `<body>` presence is reported in §0 as corroborating evidence that the two cases are genuinely different documents; it is **not** a criterion. A future article could plausibly carry a permissive licence with no body shipped, and refusing it on that basis would be wrong. Absence of a licence is what refuses, and nothing else.

### What deliberately does NOT change

**`LicenseTier` semantics are untouched.** `unknown` continues to mean not extractable. This fix restores a broken lookup; it does not redefine a tier. Any change to what `unknown` permits would weaken a deliberate Phase 4 compliance safeguard in order to work around infrastructure rot, and is rejected outright (see §9).

### The substring trap, called out because it is a live bug and not a hypothetical

The dead service returned tokens like `CC BY`, which `_canonicalize` mapped to `cc_by`. The replacement returns **URLs**. Naive substring matching is wrong in a way that is easy to ship:

- **`by-nc` is a proper substring of `by-nc-nd` and of `by-nc-sa`.** Matching shortest-first mislabels both.
- Version suffixes (`3.0/`, `4.0/`) must be stripped; both versions appear in the live sample.
- That `by-nc` and `by-nc-nd` currently map to the *same* tier is coincidence, not safety. The mapping must be correct independent of the tier table, because the tier table can change.

**Requirement:** matching is by exact path segment after stripping the version, never by `in`. Each observed licence form gets its own fixture, and the ordering gets an explicit mutation per ADR-0014 and ADR-0016 rule 1 — decided by running the weakened case.

### Batching, and the testability win it brings

`efetch db=pmc` accepts comma-separated ids. Permissions for all N papers are fetched in **one** request rather than N sequential ones inside `_parse_article`.

This is not only an efficiency change. It makes `_parse_article` a **pure function over a pre-fetched permissions map**, so it becomes testable with no network at all — where today every test of it must mock a per-article HTTP call. That is a targeted improvement to code this work is already changing, not unrelated refactoring.

### The "available for text mining" decision, recorded as a decision

Four articles in the live sample carry the NIH author-manuscript statement, whose prose asserts text mining is permitted but which carries no CC token. **They are refused** (`unknown` → not extractable). Permitting on the basis of parsed prose would be exactly the safeguard-weakening rejected in §9, and prose is not a licence identifier. Recorded here so a future reader sees a decision rather than an oversight.

---

## §2 — `PipelineState` gains exactly one field

`PipelineState` already carries `question`, `sub_queries`, `candidate_papers`, `extracted_records`, `clusters`, `contradictions`, `citations`, `answer`. Nothing about the happy path requires a new field.

What is missing is any notion of **stage status**, without which `contradictions: []` is indistinguishable from "the Critic ran and found nothing."

```python
class StageStatus(StrEnum):          # ADR-0005: StrEnum for string enums
    completed = "completed"
    not_implemented = "not_implemented"

class StageReport(BaseModel):
    name: str
    status: StageStatus
    n_in: int
    n_out: int
    dropped: dict[str, int] = Field(default_factory=dict)   # reason -> count
    note: str | None = None                                  # e.g. the ADR pointer

class PipelineState(BaseModel):
    ...                                # the existing eight fields, unchanged
    stages: list[StageReport] = Field(default_factory=list)
```

Because this lives in `PipelineState` rather than only in the printed report, it **survives the JSON dump** — so a machine reader also sees `not_implemented` and never reads an empty list as "no contradictions found."

Justified by a demonstrated consumer (the CLI report), per ADR-0013's "no infrastructure without a demonstrated consumer." No `blocked`/`skipped`/`failed` members are defined until something produces them.

---

## §3 — Data flow

```
query
  └─ esearch(query, retmax=N)                      -> pmids
  └─ efetch(pmids)  + ONE batched db=pmc permissions call
                                                   -> candidate_papers: list[Paper]
  └─ per paper: extract_entities(abstract, model, score_threshold=...)
                canonicalize(entities, abstract, linker=DictionaryLinker(MeshDictionary))
                                                   -> entities_by_paper: dict[str, list[Entity]]
  └─ build_record(paper, entities=..., extractor=SameSentenceAsEntitiesExtractor(entities_by_paper))
        ^ THE LICENCE GATE — returns None when extraction_allowed is False
                                                   -> extracted_records: dict[str, ExtractedRecord]
  └─ cluster_papers(records, texts=abstracts, pairing=SameSentencePairing(), min_size=2)
                                                   -> clusters: list[Cluster]
  └─ Critic     -> StageReport(status=not_implemented, note=ADR-0017);  contradictions stays []
  └─ Synthesis  -> StageReport(status=not_implemented);                 citations/answer stay empty
```

Component selections, each with the decision that produced it:

- **Retriever** — `PubMedClient` (`biolit/clients/pubmed.py`), dormant since Phase 1, exercised at real scale by Phase 5's corpus fetch.
- **Extractor** — `SameSentenceAsEntitiesExtractor` (`biolit/extract/deterministic.py`). **ADR-0015 rejected `LlmExtractor`** (deterministic F1 0.6238 vs 0.3054). No model, no network.
- **Pairing** — `SameSentencePairing`, per **ADR-0013** (paper-pair F1 0.6327 vs 0.5484 for cross-product, and roughly halves downstream Critic calls).

Note `SameSentenceAsEntitiesExtractor` takes `entities_by_paper` at construction while `build_record` takes that paper's `entities` separately; both receive the same mapping. `cluster_papers` needs `texts` because `ExtractedRecord` carries no text.

---

## §4 — The drop ledger

**On real data most of the interesting behaviour is drops**, so every stage records what it dropped and why. This is the substance of the report, not decoration.

| stage | drop reasons recorded |
|---|---|
| `retrieve` | `no_abstract`, `no_pmc_id` |
| `licence_gate` | `licence_refused`, broken out by the observed licence string |
| `ner_linking` | `entity_unlinked` (NIL) |
| `extract` | `zero_findings` |
| `cluster` | `no_pairs`, `singleton_key` (dropped by `min_size=2`) |

### The licence line must read as intended behaviour

With roughly half of real papers correctly refused, a bare count reads as a bug on first run. The report states the rule inline:

```
licence gate:  32 permitted, 28 refused
  refused because the publisher's permissions block carries no CC licence
  (LicenseTier.unknown -> not extractable). This is the Phase 1 compliance
  rule working as designed, not a failure.
    5  no CC token (Cochrane, ACS AuthorChoice, Sage, diabetesjournals, none)
    4  NIH "available for text mining" statement - prose, not a licence identifier
   19  no PMC record at all
```

(The illustrative counts above are the §0 sample: 60 retrieved, 41 with a PMC id,
32 permitted, 9 refused despite having a record, 19 with no PMC record.)

---

## §5 — The two unimplemented stages

The Critic was the stage named in the request. **`answer` and `citations` belong to a Synthesis stage that has never been built either** — no ADR retired it; it simply does not exist. Marking only the Critic would move the silent gap one field over rather than closing it, so both are marked, with different notes because they are unimplemented for different reasons:

- **Critic** — `status=not_implemented`, note: *"Contradiction detection is not implemented. The CTD-derived gold standard was retired by Gate 2 (π̂ = 0.067) and the re-scoped alternative was declined; see ADR-0017."* This is a **deliberately retired** stage.
- **Synthesis** — `status=not_implemented`, note: *"Answer synthesis and citation assembly are not yet built."* This is simply **not yet built**, and must not be dressed up as a decision.

Neither stage emits a placeholder value. `contradictions`, `citations` and `answer` keep their empty/None defaults, and the `StageReport` is what carries the meaning.

---

## §6 — CLI

```bash
uv run python -m biolit.pipeline --query "metformin and lactic acidosis" --max-papers 20
```

- `--query` (required), `--max-papers` (default 20), `--json-out` (optional path for the `PipelineState` dump).
- Prints the per-stage report to stdout; writes the full state as JSON when `--json-out` is given.
- Heavy imports (torch/transformers via `NerModel`) stay **local to `main()`**, matching `end_to_end.main` and the E402 exemption the project already grants.
- `main()` receives no direct unit test, per project convention; every function it calls is tested in its own module.

**No run log.** Nothing here is a measured metric or a controlled experiment, so the reproducibility argument that justifies `evals/*_runs.jsonl` does not apply. Revisit only if a real need to compare specific runs appears.

---

## §7 — Error handling

Fail closed, matching existing precedent throughout the codebase:

- An unparseable or absent `<permissions>` block yields **no licence** — never a guess, never a default-permit.
- A paper with no abstract is dropped and counted; text is never fabricated.
- An NCBI HTTP error surfaces as a **named stage failure** with the failing stage identified, rather than a partial result that looks complete.
- `findings_from_sentence_indices` already drops out-of-range indices rather than clamping; that behaviour is inherited unchanged.

---

## §8 — Testing

Unit tests never touch the network (project rule), so every HTTP interaction is a respx cassette. Required cassettes include the two shapes that caused this entire situation:

1. **A 404 cassette** — absent from the existing suite, which covers only the 200-with-`<error>` body. Its absence is why the defect shipped, and it is the same shape as the structured-abstract truncation that survived an extensive suite because every cassette happened to hold an unstructured abstract.
2. **A restricted-stub cassette** — a bodyless, permissions-less `efetch db=pmc` response (the real `PMC1401093` shape), asserting it is refused rather than permitted.

Further requirements:

- One fixture per licence form observed in §0, including both `3.0` and `4.0` versions.
- An **explicit mutation** on the `by-nc` / `by-nc-nd` ordering, run and its output recorded, per ADR-0016 rule 1.
- One end-to-end test over fixture papers with known licences, asserting the **stage ledger** — including that both stubs report `not_implemented`, which is the requirement most likely to regress silently.
- Mutants are derived from **this spec's requirement list**, not from the lines the tests already touch (ADR-0016 rule 2).

---

## §9 — Rejected alternatives

1. **Redefine what `LicenseTier.unknown` permits** — *rejected outright, not deferred.* The measurement in §0 (20/20 refused on the PMC open-access subset specifically) shows a broken mechanism, not an ambiguous policy. Changing `unknown` to work around infrastructure rot would weaken a deliberate compliance safeguard, which is backwards.
2. **A `run_pipeline(...) -> PipelineState` library function with a thin CLI over it** — rejected on **ADR-0013** grounds: no current consumer exists for the library form. Same standard that kept `PipelineState.clusters` unwired and cluster-size capping unbuilt until a real consumer appeared.
3. **A committed `evals/pipeline_runs.jsonl`** — rejected for now; see §6.
4. **Scope to wiring only, deferring the licence fix** — held as the **fallback** had the timeboxed research found no viable replacement. It did (§0), so the fix proceeds. Retained here so the fallback is on record: a small known-licence fixture corpus, with live retrieval honestly marked blocked alongside the stubs.
5. **Implementing a placeholder Critic** — rejected. A placeholder that returns anything at all is worse than an empty list, because it manufactures a result where none exists.

---

## §10 — Open items, recorded as genuinely uncertain

These are unknown until the work runs, and are listed so they are not later mistaken for things this design settled:

1. **Whether every observed licence form maps cleanly.** The §0 sample is 41 articles from three queries. Other publisher-specific forms will appear. The fail-closed default bounds the damage — an unrecognised form is refused, never permitted — so the risk is under-permitting, which is the safe direction.
2. **Whether `min_size=2` leaves enough clusters to be worth showing on a ~20-paper query.** Clustering drops singleton keys, and a small, topically-narrow result set may yield zero clusters legitimately. If that is the common case, the useful demo size may be larger than 20. This is a question the report will answer on its first real run; it is **not** a reason to change `min_size`, which is ADR-0013's validated setting.
3. **The permissions block describes the full text in PMC, while extraction runs on the abstract retrieved from PubMed.** This spec does not change that relationship — the existing code already gates abstract extraction on the PMC licence, and restoring the lookup preserves the semantics exactly. Flagged because it is a real question about the policy, and one this integration work is explicitly not the place to reopen.
