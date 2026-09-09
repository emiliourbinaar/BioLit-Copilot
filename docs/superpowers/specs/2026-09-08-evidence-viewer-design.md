# Evidence viewer — a static frontend that shows what the pipeline refuses to claim

- **Date:** 2026-09-08
- **Status:** Approved, not yet planned or implemented
- **Kind:** Presentation sub-project. **Not a measurement question.** No new metric, no gold
  standard, no experiment, and no new claim about the pipeline. Every number it displays is
  produced by the existing pipeline or already recorded in `EVAL_REPORT.md` / `DEFECTS.md`.
  ⚠️ **The bar here is deliberately lower than the eval work's** — this is showing the work,
  not extending it — with one exception, §4's licence invariant, which is a correctness
  requirement rather than a presentation one.

## Summary

A static site over **committed, licence-sanitised fixtures** of four frozen pipeline runs. It is
neither a live demo nor a rendered retrospective: it is an **evidence viewer**, built around the
stage ledger, showing what each run retrieved, what it dropped and why, what it answered, and —
pinned to the runs that exhibit them — the measured defects.

No backend, no server, no `torch` in the deploy, no NCBI call at view time.

---

## §1 — Why canned, and why not a demo

**Live runs are rejected on measured grounds, not convenience.**

- A 20-paper run **exceeded a 120-second foreground timeout** on the development machine; NER is
  CPU-bound and dominates. Serving it needs `torch` plus the checkpoint resident in the process.
- ⭐ **The decisive reason is DEF-0005.** Cluster yield collapses in proportion to how collective
  the drug term is — `no_cluster` **31%** for single agents, **74%** for drug classes — so a
  live query about "anticoagulants and bleeding" returns **nothing at all**. A demo whose
  documented failure mode is an empty answer will produce one in front of a viewer.
- Canned fixtures guarantee the interesting cases are visible rather than left to chance.

**And not a plain demo.** A query box over a nice answer looks like every other literature-search
demo until someone reads the ledger. The ledger is the differentiator: `dropped 20:
licence_refused`, `noted 211: entity_unlinked`, `critic: NOT IMPLEMENTED`. The site is built
around it.

**Non-goals, explicitly.** No query box. No live retrieval. No LLM. No login, no analytics, no
database. Not a replacement for `RETROSPECTIVE.md` — it links out to the prose rather than
restating it.

---

## §2 — The four featured runs

Chosen for what each demonstrably surfaces, verified against the frozen states and
`evals/gold/acronym_labels.jsonl`.

| Slug | Query | What it is there to show |
|---|---|---|
| `statins-rhabdomyolysis` | statins and rhabdomyolysis | **ADR-0022.** The three `answers` clusters carrying `Atorvastatin` that `select_stage` deleted, and their recovery. The resulting order is perfectly stratified: 9 `answers` then 8 `background`. Also carries **`AT`** — DEF-0004's NER span error, where a type-constrained linker would score a *success* and fix nothing |
| `isotretinoin-depression` | isotretinoin and depression | **Why filtering could not fix the lead.** `depression` NILs against the local table in all 26 papers containing it; NCBI's own translation rescues it. The filter keeps **5 of 5** and the answer still opens on `Isotretinoin \| Acne Vulgaris` (18 papers) — the largest cluster and the least on-query |
| `cisplatin-nephrotoxicity` | cisplatin nephrotoxicity | **The acronym census, at its most legible.** `CP` → *Cleft Palate* where the abstract says cisplatin; `RA` → *Rheumatoid Arthritis* where it says rosmarinic acid; plus `AITC`, `GSH`. Four wrong pairs in a **3-cluster** run, so the page is small and the linking failure is the whole story |
| `metformin-lactic-acidosis` | metformin and lactic acidosis | **The ledger at full size.** The deepest funnel and the largest cluster count, plus the pre-`select_stage` failure that shipped `Creatinine \| Hyperkalemia` into an answer about lactic acidosis |

**`amiodarone-pulmonary-toxicity` is held in reserve, not built.** It carries `APT` at 26
mentions — the single largest wrong pair in the census — but 15 clusters makes a busier page and
cisplatin already tells the acronym story more clearly. Adding it later is a fixture regeneration
and a list entry, nothing more.

**Copy comes from the annotator, not from us.** `acronym_labels.jsonl` carries a written `reason`
per row: *"The text defines 'GSH' as glutathione, not Glucocorticoid-Remediable Aldosteronism."*
Those strings are displayed verbatim. No paraphrasing — a paraphrase of an adjudication is a new
claim.

---

## §3 — Fixtures must be regenerated, and the old ones are wrong

The existing frozen states under `data/synth/` **predate ADR-0022** and cannot be used:

- They were produced before pharmacological-class matching existed, so `statins and
  rhabdomyolysis` shows **12 kept, not 17** — precisely the number the site is built to feature.
- `state.clusters` is **overwritten in place** by `select_stage`, so a saved state carries the
  post-filter set with no record of what was dropped at cluster level.
- `data/` is gitignored in its entirety, so nothing is committed today.

Regeneration requires live NCBI retrieval and local NER. **That is a build-time step, run rarely
and deliberately — never at view time.**

---

## §4 — The fixture schema, and the licence invariant

### The invariant

> **No fixture field may carry text originating from a paper the licence gate refused.**

This is a correctness requirement, not presentation. It exists because of **DEF-0006**:
`PipelineState.candidate_papers` retains every retrieved `Paper` with its abstract, refused or
not, and `--json-out` serialises the lot — measured at **7 of 8 refused papers carrying a full
verbatim abstract**, one of them 2,096 characters at `license_tier='unknown'`.

### Schema

```
FixtureRun
  schema_version : int
  slug           : str
  query          : str
  generated_at   : ISO-8601 UTC
  source_pin     : str          # see §5
  stages         : [StageReport]   # verbatim; counts and notes only, no text
  clusters       : [ { key, concept_names[], paper_ids[], rank,
                       score: {matched, proximity}, label? } ]
  answer         : str             # rendered from ExtractedRecord, already gated
  papers         : { paper_id: { title, journal, year, doi, pmid,
                                 license, license_tier, extraction_allowed } }
  findings       : [ { defect_id, anchor, headline, reason } ]
```

⭐ **`abstract` does not exist in the schema.** Not blank, not optional — **absent**. A field that
must always be empty is a field someone eventually fills. DEF-0006 happened because a
serialisation path outgrew an argument about which contracts see a `Paper`; the response is to
make the unsafe shape unrepresentable rather than to remember not to populate it.

**Paper stubs are emitted for every tier; text for none.** "20 retrieved, 8 refused, tiers
`unknown`/`non_commercial`/`open`" is the interesting part of the licence gate and needs no
abstract to tell.

**`label` on a cluster is present only where a frozen relevance label exists**, and the viewer
must mark those as what they are — see §7's honesty requirements.

### Attribution is a hard requirement, and the pipeline does not supply it

The answer is rendered from `ExtractedRecord`, which is licence-gated, so its sentences come only
from papers the gate allowed. ⚠️ **But every allowed tier is a Creative Commons licence, and
every one of them requires attribution.** The synthesis stage's own ledger note says *"Citation
assembly is not yet built."*

**Therefore the viewer must display per-paper licence and DOI wherever answer text appears.** The
fixture carries `license`, `license_tier` and `doi` for exactly this reason. This is a viewer
obligation; it does not change the pipeline.

---

## §5 — Regeneration as a first-class step

### The generator

`biolit_evals/fixture_export.py`, following the naming and shape of `acronym_export.py` and
`relevance_export.py` — a build tool that writes an artifact, `main()` untested per project
convention.

```bash
uv run python -m biolit_evals.fixture_export            # all four
uv run python -m biolit_evals.fixture_export --slug statins-rhabdomyolysis
```

For each featured run it **imports and calls the stage functions in-process** — it does NOT
shell out to `python -m biolit.pipeline --json-out` — projects the resulting `PipelineState`
into `FixtureRun`, asserts the §4 invariant, and writes `frontend/src/fixtures/<slug>.json`,
which **is** committed, unlike `data/`.

⚠️ **In-process is a correctness requirement, not a style preference.** `--json-out` is the
path DEF-0006 is filed against; driving the generator through it would write an unsanitised
artifact containing refused papers' abstracts to disk *before* any sanitising step could run.
The unsafe file must never exist, not merely never be committed.

It is the only sanctioned way to produce a fixture. Hand-editing one is out of scope; the
staleness pin below would not catch it, and the invariant test would only catch licence errors.

### ⭐ The staleness pin

The requirement: **a stale "17 kept" must be structurally hard to ship, not merely something we
remember to re-check.**

`source_pin` is a hash over the modules that determine what a fixture *claims*:

| Module | Why it is pinned |
|---|---|
| `biolit/query/concepts.py` | decides which clusters match |
| `biolit/query/ranking.py` | decides the order and the score shown |
| `biolit/pipeline/stages.py` | produces every `StageReport` the site renders |
| `biolit/cluster/pairing.py` | decides which clusters exist at all |

⚠️ **Hashed over the parsed AST with docstrings stripped, not over raw file bytes.** This repo's
modules carry very heavy comments and docstrings that are edited constantly; a byte hash would
fire on every prose edit, and a check that cries wolf gets suppressed. Normalising through
`ast.parse` → strip docstring nodes → `ast.dump` makes the pin fire on **behaviour** changes and
stay quiet on comment changes. That is the intended trade: a pure-comment edit does not
invalidate a fixture, and it should not.

A test in the backend suite compares each committed fixture's `source_pin` against the current
tree and **fails with the regeneration command in its message** when they differ.

⚠️ **The consequence, stated rather than discovered:** changing any of those four modules turns
the backend suite red, and the only way to green is a fixture regeneration that needs live
NCBI. That is the intended cost — it is what "structurally hard to ship a stale claim" buys —
but it means a behaviour change and a fixture refresh are now one unit of work. The *test*
remains hermetic; only the *remedy* needs the network, so the project's "unit tests never touch
the network" rule is intact.

**Accepted limitation, stated rather than discovered later:** the pin catches changes to the four
modules above. A change to the MeSH artifacts, the NER checkpoint, or NCBI's query translation
changes the fixtures without changing the pin. Those are recorded in `generated_at` and are the
same drift `ADR-0023` already documents; the pin is not claimed to cover them.

---

## §6 — Tests

Unit tests only. No network, no NER, no torch — every test runs against a constructed
`PipelineState` or a committed fixture.

1. **The licence invariant.** Build a `PipelineState` containing a refused paper whose abstract
   holds a distinctive sentinel string, project it, and assert the sentinel appears **nowhere**
   in the serialised fixture. This is the test that would have caught DEF-0006.
2. **`abstract` is unrepresentable.** The fixture models set `model_config =
   ConfigDict(extra="forbid")`, so an `abstract` key raises rather than being silently
   ignored — pydantic's default would drop it quietly, which is indistinguishable from safety
   until someone reads the schema. The schema is the guard, so the schema is tested.
3. **Attribution completeness.** Every `paper_id` referenced by a cluster resolves to a stub
   carrying `license` and `doi`.
4. **Ledger integrity survives projection.** Where `unit_in == unit_out`, assert
   `n_in - sum(dropped) == n_out`, the check `StageReport` already documents as available.
5. **Staleness pin.** Each committed fixture's `source_pin` matches the current AST hash.
6. **Pin normalisation.** Editing a comment or docstring does not change the hash; changing a
   statement does. Pinned directly, because the pin's entire value is that it does not cry wolf.

---

## §7 — Site structure

```
/                       thesis; the six instrument failures, each linking to the run
                        that exhibits it
/runs/<slug>            THE CORE VIEW
      ├── question + provenance (generated_at, paper budget)
      ├── StageLedger      ← the hero element
      ├── DropPanel        ← per stage: what left, and why
      ├── ClusterList      ← ordered, with rank and score
      ├── AnswerPane       ← answer + REQUIRED per-paper attribution
      └── FindingCallout[] ← defects pinned to this run
/findings               DEF-0001..0006 and the six failures → the runs showing them
/evidence               links out to RETROSPECTIVE / DECISIONS / DEFECTS / EVAL_REPORT
```

### `StageLedger` — the component the site exists for

Renders `StageReport` directly:

- `n_in`/`n_out` with **both unit labels**, because stages legitimately change unit and an
  undeclared change reads as impossible growth
- `dropped` and `noted` **visually distinct** — conflating them once made the first rendered
  ledger misreport three of seven stages
- `critic` rendered as **NOT IMPLEMENTED**, never as an empty result

### Honesty requirements on the presentation

These are load-bearing, not styling:

1. **Cluster labels are marked as frozen annotation labels**, with a note that the pass they came
   from had a compromised control instrument (ADR-0021). The site must not present `answers` /
   `background` as ground truth the pipeline achieved.
2. **Ordering is marked as unvalidated.** ADR-0020 and ADR-0022 both ship on a design argument
   plus a mechanical check, with no attributable label evidence. A site that shows a nicely
   ordered list implies the ordering was validated; a one-line qualifier prevents that.
3. **`critic` is visibly absent, not quietly missing.**

---

## §8 — Stack and build

**Astro**, static output. Content-heavy, ships no JavaScript except where interactivity is asked
for, and reads markdown natively for `/evidence`. `.gitignore` already carries a Node/Next block
that covers `node_modules/`, `.next/`, `out/`, `.turbo/`.

- Fixtures are imported as local JSON at build time — no runtime fetch, no API.
- `frontend/` is a sibling of `backend/`; the Python CI is unchanged.
- Frontend checks (typecheck, build) are a separate command from the backend's four-step CI and
  do not join it in this spec.

---

## §9 — Out of scope

- Live pipeline execution, a query box, any server.
- Fixing DEF-0006 in `--json-out` itself. **This spec routes around it by never using that path**;
  the defect stays open and separately filed.
- Changing the synthesis template to emit citations. The viewer supplies attribution; the
  pipeline is untouched.
- Adding the frontend to the backend CI gate.
- Any new measurement, gold standard, or claim.

---

## §10 — Risks

| Risk | Response |
|---|---|
| Regeneration needs live NCBI and drifts | Accepted and documented; `generated_at` records the moment, and §5 states what the pin does not cover |
| The site implies validation the project does not have | §7's honesty requirements, treated as load-bearing |
| Attribution obligation is missed in a later edit | Test 3 makes it structural, not editorial |
| Fixture regeneration is forgotten after a ranker change | §5's staleness pin, which fails CI with the command to run |
