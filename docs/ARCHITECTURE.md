# BioLit Copilot — Architecture

**Where this is.** Phases 1–5 are built and measured, and the layers are now **wired into a
runnable end-to-end pipeline** (`biolit.pipeline`, `python -m biolit.pipeline --query "..."`).
`PipelineState.clusters` was a declared field with no producer from Phase 1 until then; the
pipeline's `cluster_stage` populates it, closing the gap ADR-0013 left open.

**The Critic is the one layer that is still a stub, and deliberately so.** Phase 5 ran its free
half and stopped: Gate 2 measured its derived gold standard invalid and retired it unspent
(ADR-0017), so there is no validated gold to build a Critic against. The pipeline therefore
reports `critic` as an explicit `StageStatus.not_implemented` rather than returning an empty
result — `contradictions: []` on its own is indistinguishable from "ran and found nothing",
which is the distinction the stage ledger exists to preserve.

**Synthesis is built and deterministic (ADR-0019).** Gate A set out to decide whether an LLM
arm earns that slot and could not: its comparative axes rank a one-word-per-paper index above a
real characterisation, so no arm was ever bought and the design failure was established for $0.
`synthesis_stage` renders every selected cluster from the source sentences and invents nothing.
**Generative narrative synthesis over clusters is out of scope — see `SCOPE.md` SR-0001.**

**The query is consulted exactly twice**: by `esearch`, and by `select_stage`. Until 2026-09-05
it was consulted only once — `PipelineState.question` was set in `__main__` and read by nothing,
so every cluster retrieval happened to produce went into the answer. `select_stage` keeps
clusters sharing a MeSH concept with the question, using NCBI's own query translation because
the local alias table NILs on terms users actually type (`depression`, `gastrointestinal
bleeding`, `thyroid dysfunction`). It prunes an off-topic tail — 83 clusters to **75** on the
frozen corpus.

**It also orders the survivors (ADR-0020), and the two are one call because they consume the
identical signal** — selection thresholds concept overlap, ordering grades it. They stay two
facts in the stage ledger, because they are two decisions. Filtering alone could never have
fixed the lead: clusters arrive in `sorted(by_key)` order, so the lead was whichever survivor
had the alphabetically smallest MeSH id, and a filter changes which clusters survive without
changing the sort. ⚠️ **The within-cluster rule is untouched** — `render_cluster` still orders
papers by year then id and carries no ranking; ADR-0020 extends that rule to clusters rather
than repealing it, and cluster order now means exactly one thing, relevance to the question.

**A cluster side matches the query if it IS a query concept or belongs to a pharmacological
class the query named (ADR-0022, member→class only).** A drug class is not a tree ancestor of
its members — `Atorvastatin` is filed under chemical structure and its class under chemical
actions and uses, sharing no node — so this reads MeSH's `PharmacologicalAction` field, which
needs a second build artifact (`build_mesh_actions`) alongside the tree.

⛔ **Neither the ordering nor the class-matching has attributable label evidence.** The blind
annotation that scored the ordering had a compromised control instrument (ADR-0021), and the
attempt to build a fresh label set for the class matching closed as a negative result about the
instrument (ADR-0023). Both ship on their design argument plus mechanical verification. See
`EVAL_REPORT.md` and `DEFECTS.md` DEF-0003/DEF-0005.

**The search for a replacement gold is closed, and that closes the paper-pair Critic with it
(ADR-0018).** Six corpora across three structural families were measured; the best candidate
annotated at π̂ 0.60 with no separation between its filtered and unfiltered subsets. The
recurring failure is not the corpora but **the unit**: `ContradictionFinding(paper_id_a,
paper_id_b, …)` asks two abstracts to be commensurable, while real literature separates its
findings by population, dose, route and endpoint in ways that make most opposed-looking pairs
genuinely compatible. That held even in a corpus built from systematic reviews *specifically to
hold population and intervention fixed*. **Reviving this stage means changing the unit — a unit
that carries the qualifying context explicitly — which is a new spec, not a next step.**

## Layer map

```
PubMed ──► extract_entities ──► canonicalize ──► licence gate ──► build_record ──► group
           biolit.ner           biolit.canon     build_record      biolit.extract   biolit.cluster
           (Phase 2)            (Phase 3)        (Phase 1 rule)    (Phase 4)        (Phase 3)

  ──► select + order ──────►  [ critic ]  ──────────►  synthesis
      biolit.query                not_implemented          biolit.synth.template
      (concept overlap, exact or   (ADR-0017, ADR-0018)    (ADR-0019, deterministic)
       pharmacological class;
       then relevance order)
      (ADR-0020, ADR-0022)
```

`biolit.pipeline` runs that left-to-right path over the real components and prints a per-stage
ledger accounting for every paper — including what the licence gate refused and why. The
deterministic `SameSentenceAsEntitiesExtractor` is the extractor that ships; the LLM arm was
measured and rejected (ADR-0015) and is not on this path.

**The licence gate has exactly one enforcement point,** `build_record` in `biolit.extract.base`.
It suppresses the whole record rather than just the findings, because `Entity.text` and
`Finding.text` both carry verbatim abstract substrings. No stage decides extraction rights
itself.

Every layer is paired with a module in `biolit_evals` that scores it against gold and appends
one JSON line per run to a committed log. The eval packages are separate from `biolit` on
purpose — production code carries no scoring logic — but both arms of every eval run the
*production* path rather than a parallel gold-only implementation (ADR-0010).

## Phase 1 (Foundations)
Backend-forward monorepo. `Paper` is the single normalization boundary for PubMed and
bioRxiv/medRxiv. Full-text existence (`text_type`) and extraction rights (`license_tier`,
`extraction_allowed`) are independent axes. Hybrid LangGraph state: one `PipelineState`
plus pure per-node `Input`/`Output` contracts joined by `project_*`/`merge_*` adapters.
Storage schema (papers + pgvector embeddings) is defined but dormant until Phase 3.

See `docs/DECISIONS.md` for ADRs and `docs/superpowers/specs/` for the Phase 1 spec.

## NER layer (Phase 2)
`biolit.ner.extract_entities` wraps a local, fine-tuned PubMedBERT/BC5CDR token-classification
checkpoint (chemicals + diseases) behind a lazily-loaded `NerModel`. It runs CPU-first with
GPU auto-detect (`Settings.ner_device = "auto"`) and is the entity-anchoring layer the
Extractor agent will call in Phase 4 to ground extracted claims in named chemical/disease
mentions. The eval harness lives in `biolit_evals` (`biolit_evals.ner_eval`): a pure,
multiset-based strict entity-level P/R/F1 scorer (`biolit_evals/scoring.py`, exact
`(start, end, label)` match — not `seqeval`) run against both the BC5CDR test split and a
blind-annotated in-domain sample (`evals/gold/domain_sample.jsonl`, ADR-0006), with every run
appended as one JSON line to `evals/runs.jsonl`. See `docs/EVAL_REPORT.md` for the first
real numbers, methodology caveats, and limitations.

## Canonicalization layer (Phase 3)
`biolit.canon.canonicalize(entities, text, *, linker)` sits between `extract_entities`
(Phase 2, unchanged) and clustering, populating `Entity.canonical_id` / `canonical_name`
with a MeSH concept. The `linker` is injected (keyword-only) so the layer stays testable
offline and so a different linking strategy is a constructor argument, not an edit.

Each entity is looked up on its own surface first; `merge_fragments` — a structural
re-merge of adjacent same-label spans split by the subword tokenizer (`GLP` + `1RA` →
`GLP-1RA`), the direct fix for the ADR-0008 finding — then contributes a merged candidate
that is applied **only to constituents that did not link individually**. That ordering
matters: the merge rule is deliberately permissive, so a spurious merge can fill a gap but
can never overwrite a correct individual link with one wrong shared concept.

Linking goes through a `Linker` protocol. The only implementation today is
`DictionaryLinker`, an offline lookup over a CTD→MeSH alias table; the protocol is the
seam an embedding-based fallback (e.g. SapBERT) slots into without a redesign. Unlinked
entities stay NIL (`None`) and are never dropped, so clustering can fall back to a
surface-form key.

The alias artifact is built offline and gitignored: `python -m biolit.canon.build_mesh`
downloads the CTD chemical/disease vocabularies and writes `data/canon/mesh_aliases.json.gz`.
CTD columns are resolved **by name from the file header**, not by fixed index — a
positional assumption silently misread ids and synonyms once already (see the eval report).

Evaluation lives in `biolit_evals.canon_eval`: linking precision/recall/F1 scored on gold
mentions (isolating linking from NER), plus a **NIL rate** and an **ambiguous-tiebreak
rate**, against BC5CDR's gold MeSH IDs and a blind-annotated domain sample, appended to
`evals/canon_runs.jsonl`. See `docs/EVAL_REPORT.md` for results and methodology.

`biolit_evals.end_to_end` measures the production path (`extract_entities` → `canonicalize`
→ `canonical_id`) against gold MeSH IDs — the only eval that exercises `merge_fragments`.
Its primary metric is concept-level micro-averaged P/R/F1 (document-level set matching,
because clustering consumes a paper's concept set, not its spans), reported pooled and per
label, alongside a permanent outcome census (exact / mergeable / truncated / missed, with
truncation sub-classified) that accounts categorically for every point of loss.

`NerModel` windows long inputs before inference: this checkpoint's tokenizer declares no
`model_max_length`, so a document over BERT's 512-token limit would otherwise reach the
model and raise a tensor-size error (2.2% of real abstracts). Windows are packed from whole
sentences with a one-sentence overlap, and offsets are shifted back into document
coordinates and de-duplicated, so `extract_entities` is safe on text of any length and
every consumer inherits that.

## Clustering layer (Phase 3)

`biolit.cluster` decides which chemical pairs with which disease inside a paper, then groups
papers by shared `(chemical, disease)` keys. `SameSentencePairing` is the strategy of record:
it beats the full cross product on **every arm and every metric level** (paper-pair F1
0.5484 → 0.6327 on the real pipeline) and roughly **halves the Critic's future workload** —
460 → 279 distinct comparisons on Test-500, which is a direct cost result once every
comparison becomes an LLM call.

**A dedicated CID relation extractor was scoped and rejected** (ADR-0013). Three reasons in
order of weight: 40.3% of gold relations lose an endpoint upstream, where no pairing strategy
can reach them; the free heuristic already captures 77.0% of a precision-perfect ceiling; and
part of same-sentence's recall is accidental co-clustering that a precision-oriented
extractor would give up. The package lives in `biolit/` rather than `biolit_evals/` because
the measurement's outcome was that it **wins** — the consumer rule turns on whether a
module's value was contingent on the result, not on how many callers it currently has.

## Extraction layer (Phase 4)

`biolit.extract` selects the sentences of an abstract that state a finding. Two
implementations behind one `Extractor` protocol:

- `SameSentenceAsEntitiesExtractor` — deterministic, free, **F1 0.6238**. This is what ships.
- `LlmExtractor` — one Claude call per abstract, **F1 0.3054**. **Not shipped, in any role.**

The LLM arm over-selects (4.55 sentences per paper against 2.31) and pays for it in
precision. Cost was not the reason it was rejected: a full run measured **$2.48** with zero
refusals across ~1500 calls. It was rejected because a sweep of seventeen free positional
heuristics found that **every budget-matched one of them beats it** — including `last 1`, the
single closing sentence of each abstract plus random padding. An arm that cannot beat the
opening sentences of an abstract is not doing the task the eval was built to measure
(ADR-0015).

`llm.py` nevertheless stays on `master` while ADR-0011's and ADR-0012's rejected mechanisms
did not, for two reasons recorded in ADR-0015: it is the arm under measurement and the eval
report's reproduction recipe runs it, so deleting it makes the published headline
unreproducible; and it takes its client as `Any` without importing `anthropic`, so `biolit`
gains no hard dependency from it. **What is rejected is the arm's use as an extractor, not
the seam.**

`biolit_evals.baselines` is the comparator harness that produced that verdict. It exists
because `recall_on_endpoint_lost` is scored against a population *defined by the control's
own misses*, so the control scores 0 there by construction — a claim was once shipped on that
guaranteed-zero comparison and had to be retracted. Everything in it is deterministic given a
seed, reads the arms from the committed run log rather than re-running them, and needs no
credential.
