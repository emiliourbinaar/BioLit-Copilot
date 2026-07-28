# Clustering: pricing naive chemical–disease pairing

**Date:** 2026-07-28
**Status:** Approved for planning
**Phase:** 3 (clustering), first measurement

## The question

`Cluster.key` has been `"metformin|PCOS"` — a **chemical|disease pair** — since Phase 1, and
`ClusteringInput.records` feeds it from the Extractor. Every concept-level metric in this
project was justified by "clustering consumes a paper's concept set," but clustering has
never been built, and exploring it surfaced a problem that had been sitting invisibly inside
the word:

**A paper's concept set does not determine its pairs.** An abstract with 3 chemicals and 4
diseases yields 12 candidate pairs; gold says ~2.13 are real. Deciding *which* chemical
relates to *which* disease is relation extraction — it is literally the BC5CDR CID task, a
known-hard benchmark, not a thin grouping step.

This spec defines the work that decides whether that phase is worth scoping: **what naive
pairing actually costs in precision, and whether a free deterministic heuristic closes
enough of the gap to make a relation model unnecessary.**

## Gold data exists (a correction)

An earlier claim in this project's planning — that BC5CDR labels concept sets but not which
papers belong together — is **wrong**, and is corrected here rather than quietly dropped.

BC5CDR is the BioCreative V **Chemical–Disease Relation** corpus. `CDR_Data.zip` ships gold
**CID relations**: document-level `PMID<tab>CID<tab>chemicalMeSH<tab>diseaseMeSH` lines that
the existing PubTator parser currently skips. **Two papers belong to the same gold cluster
iff they share a gold CID pair.**

Verified against the real corpus, not recalled:

| | test 500 | all 1500 (train+dev+test) |
|---|---|---|
| CID relation lines | 1066 | — |
| documents with ≥1 CID | **500 / 500** | 1500 / 1500 |
| mean CID pairs per document | 2.13 (max 21) | — |
| distinct (chemical, disease) pairs | 941 | 2434 |
| **pairs shared by >1 document** | **80** | **325** |
| documents in a multi-document cluster | 172 (34%) | **758 (51%)** |
| clusters of size ≥3 | 21 (max 9) | 122 (max 25) |

This makes the eval a scoring problem against gold rather than an invented intrinsic metric,
and it lines up exactly with the domain model that already existed.

## Provisional figures from exploration

Measured on **gold entities** during design, so these are a **ceiling** and are labeled as
such wherever cited. They must be **reproduced by the harness** before being quoted anywhere
else; they are recorded here as the motivation, not as results.

| | keys/doc | key P | key R | multi-paper clusters | gold-real | **cluster P** | cluster R |
|---|---|---|---|---|---|---|---|
| Cross-product | 10.8 | 0.197 | 1.000 | 426 | 80 | **18.8%** | 100% |
| Same-sentence | 4.8 | 0.310 | 0.699 | 195 | 72 | **36.9%** | 90% |

Gold is 2.13 pairs/doc, so blind cross-product over-generates **5×**.

### The key-level / cluster-level gap must be stated wherever these are cited

Same-sentence co-occurrence looks like it sacrifices 30% of key recall (1.000 → 0.699) but
costs only 10% of gold clusters (80 → 72). **That is not a discrepancy — it is the finding.**
Most keys the heuristic drops were held by a single paper and were never going to form a
cluster, so dropping them is free at the level that matters. The same set-absorption effect
that has understated cost four times in this project (merge 91→11, abbreviations 543→4,
paraphrase 1136→431, fallback slice +0.0254 vs +0.0200) here runs in our favour, and a reader
who sees only the key-level number will misread the heuristic as far more lossy than it is.

**Recall is the easy side.** Cross-product key recall is 1.000 *by construction*. This is
purely a precision problem.

## Non-goals

- Building CID relation extraction. This measurement decides whether it earns its own phase.
- Surface-form / NIL keys. Their cost is **measured**, not built (see below).
- The Critic itself.
- Retrieval / pgvector.
- Any clustering beyond exact key match — no fuzzy or hierarchical grouping.

## Architecture

### `biolit.cluster`

```python
# biolit/cluster/pairing.py
class PairingStrategy(Protocol):
    def pairs(self, entities: Sequence[Entity], text: str) -> set[tuple[str, str]]: ...

class CrossProductPairing:   # every chemical x every disease
class SameSentencePairing:   # ...only if both spans fall within one sentence
```

```python
# biolit/cluster/group.py
def cluster_papers(
    records: Sequence[ExtractedRecord], *, texts: Mapping[str, str],
    pairing: PairingStrategy, min_size: int = 2,
) -> list[Cluster]
```

`pairing` is injected keyword-only, mirroring `canonicalize(..., linker=...)`. A real
relation extractor later becomes a constructor argument, not a rewrite — the same seam that
let the embedding fallback be measured without touching `canonicalize`.

**Key format.** `PairingStrategy.pairs` returns `(chemical_id, disease_id)` tuples;
`cluster_papers` renders each as `f"{chemical_id}|{disease_id}"` into `Cluster.key`. The
chemical side is always first, so a key is unambiguous without carrying labels.

**Multi-id mentions expand.** A gold mention may carry more than one MeSH id. Pairing takes
the cross-product of the two endpoints' id sets, so a mention with 2 ids facing one with 1
contributes 2 pairs. This matches how the provisional figures above were computed, and
anchor 1 (key recall = 1.0000) would fail if it were done any other way.

`SameSentencePairing` reuses `biolit.ner.windowing._sentence_spans`, **promoted to public**:
it now has a second consumer and is no longer a private helper.

### Four decisions, each defensible the other way

**1. `texts` is passed separately.** `ExtractedRecord` carries `paper_id` and `entities` but
**no text**, and same-sentence pairing needs it. Rather than amend the Phase-1 contract,
`cluster_papers` takes a `paper_id → text` mapping — matching `canonicalize(entities, text,
*, linker)`, which already takes text explicitly. `PipelineState.candidate_papers` supplies
it in the graph.

**2. `min_size=2` — singleton keys are not clusters.** The Critic compares papers *within* a
cluster, so a key held by one paper is inert. This is what makes cross-product
over-generation survivable, and it is the mechanism behind the key/cluster gap above.

**3. NIL entities do not form keys in v1 — a stated deviation.** `ARCHITECTURE.md` says
clustering "can fall back to a surface-form key." This spec declines that for now, and
flags it rather than silently dropping it. A `nil:<surface>|MESH:D…` key is unscoreable
against gold CID, and the e2e NIL rate is 0.320, so admitting them would inject a large
unmeasurable population into a measurement whose entire purpose is precision.

**The cost is logged, not assumed**, and **split by which side was NIL — chemical or
disease.** DISEASE has NIL'd at roughly double CHEMICAL's rate throughout canonicalization
(29.2% vs 14.9% on perfect spans, 1.96×). If that holds here, the unmeasurable population
this decision creates is **not uniform**, and that is worth knowing now rather than
discovering later. If the loss is large, surface-form keys become a scoped follow-up with
evidence behind it.

**4. Unplaceable entities fail closed.** `Entity.start`/`end` are `int | None`. An entity
with no offset cannot be assigned to a sentence, so `SameSentencePairing` forms no pairs
from it and **reports the count**. This follows `check_document_context`'s precedent: a
silent zero is indistinguishable from "there was nothing here," and that is the failure mode
to design against.

Output is deterministically ordered (clusters by key, `paper_ids` sorted) so runs are
reproducible and diffable.

## The eval harness

`biolit_evals/cluster_eval.py`, appending one JSON line per run to `evals/cluster_runs.jsonl`.

### Two arms × two pairing strategies, always reported together

| arm | corpus | entities from | isolates |
|---|---|---|---|
| **A — pairing ceiling** | all 1500 | BC5CDR gold mentions + gold MeSH ids | pairing, from NER *and* linking error |
| **B — end-to-end** | test 500 | real `extract_entities` → `canonicalize` | nothing; this is the production path |

**Arm A synthesizes its input.** `cluster_papers` consumes `ExtractedRecord`s, so the harness
builds one per document from the gold mentions — `paper_id` = pmid, `entities` = gold spans
carrying their gold MeSH ids as `canonical_id`, other fields left empty. This keeps a single
production code path under test in both arms rather than a parallel gold-only implementation.
Arm A loads all three PubTator members (`CDR_TrainingSet`, `CDR_DevelopmentSet`,
`CDR_TestSet`); `load_bc5cdr_documents` already takes a `member` argument, so no change is
needed there.

**The corpus split is a correctness requirement, not a sample-size preference.** The NER
checkpoint was fine-tuned on BC5CDR's training split, so any arm running real NER must be
held out to the test 500 or the number is contaminated. Arm A bypasses NER entirely, so the
full corpus is safe there — and worth having, because multi-document clusters are sparse
(122 clusters of size ≥3 across 1500, versus 21 in test-500 alone). A precision number this
consequential should not rest on a thin sample when a safe way to widen it exists.

Arm A is **a ceiling for Arm B**, labeled as such everywhere, on the same rule applied to the
oracle ceiling. The gap between the arms prices what NER and linking cost clustering.

Same-sentence is a **control against cross-product**, not an alternative: without it a
"relation extraction is required" conclusion is unattributable between *needs a model* and
*needs one line of sentence logic*. This is the same discipline that produced the TF-IDF
control in Phase 3C and the MeSH-enrichment null.

### Three metric levels, because they disagree

1. **Key level** — micro-averaged P/R/F1 over `(pmid, key)` against gold CID, set-per-document.
2. **Cluster-key level** — is a predicted ≥2-paper cluster's key a gold ≥2-paper cluster key?
3. **Paper-pair level — the primary metric.** Over unordered paper pairs: is (A, B)
   co-clustered in prediction and in gold?

**Why paper-pair is primary.** `ContradictionFinding` is `(paper_id_a, paper_id_b, label,
rationale)` — the Critic's unit of work is a *pair of papers*, not a cluster and not a key.
Every false paper-pair is a wasted or wrong Critic comparison, and since the Critic will be
LLM-backed, that is direct cost. Levels 1 and 2 are **diagnostic**, not competing candidates
for primary.

### Raw counts, not only aggregates

Following the Phase 3C mention-level lesson, the harness logs the underlying counts:

- clusters formed; gold clusters
- **paper-pairs generated — the Critic's actual workload**; gold paper-pairs
- pairs-per-cluster distribution
- **share of total generated paper-pairs contributed by the top-5 largest clusters**

That last one directly prices the quadratic risk: a single 25-paper cluster is 300
comparisons on its own. An aggregate cannot show how much of the Critic's budget one
oversized cluster burns; this can, and it is free to derive from what is already logged.

Plus the section-1 diagnostics: NIL-blocked pairs **split by chemical vs disease side**, and
unplaceable-entity counts.

### Two anchors — and they check different things

**Anchor 1 (harness correctness): cross-product pairing on gold entities must reproduce key
recall = 1.0000 exactly, on both corpora.** True by construction — a gold CID pair always
connects two annotated entities, so the cross-product of gold entity ids necessarily
contains every gold pair. Any deviation means the harness is wrong (gold parsing, MeSH id
prefixing, or label assignment), not that the number is interesting. It halts the run.

This anchor already did work during design: CID lines carry **bare** ids (`D015738`) while
gold mentions carry `MESH:D015738`, and recall landing on exactly 1.0000 is what proved the
prefixing correct. Enforced rather than remembered.

**Anchor 2 (loader correctness): gold multi-paper cluster counts must reproduce 80 (test-500)
and 325 (all-1500).**

**These two check different things and must not be conflated.** Anchor 2 validates
`load_bc5cdr_cid_relations` against independently established gold statistics — it is a
**loader-correctness check, not a clustering-quality check**. "The loader reproduces known
gold counts" says nothing whatsoever about whether the pairing strategy is accurate. The
docstring and the report both state this explicitly.

### New loader

`load_bc5cdr_cid_relations(zip_url, member) → dict[pmid, set[tuple[str, str]]]`, parsing the
CID lines the existing PubTator parser skips. No invented ids, PMIDs, or abstract text; the
loader test uses real observed corpus lines (e.g. `8701013 CID D015738 D003693`).

## Testing

Follows the existing `biolit.canon` discipline: new code gets tests regardless of size, real
corpus-shaped fixtures, nothing fabricated.

**Two tests exist specifically to kill mutants that can be named in advance.** This project
has twice shipped a test suite that a wrong implementation passed, and both times it was
found by asking "is there a plausible wrong implementation every one of these tests accepts?"

- **`SameSentencePairing` that silently returns the cross-product.** Killed by a fixture
  where the two strategies must *differ*: a chemical and disease in one sentence, another
  disease in a second sentence. A strategy ignoring sentence boundaries fails it.
- **A paper-pair metric that is secretly the cluster-key metric.** Killed by a fixture where
  the two levels *disagree* — identical cluster-key score, different paper-pair score.
  **This test exists to prevent the primary metric from silently aliasing a diagnostic one.**
  Without it, the number the whole recommendation rests on could be measuring something else
  entirely while every test passed. That is a stronger guarantee than a regression test, and
  it is the `mentions_wrong_unaligned` lesson applied before the fact rather than after.

Also pinned: `min_size=2` drops singletons; deterministic ordering; NIL exclusion counted by
chemical vs disease side; `start is None` fails closed; the anchor helper raises on mismatch
(its own test); top-5 concentration on a fixture with one oversized cluster.

**`SameSentencePairing` gets dedicated tests for its use of the now-public `_sentence_spans`,
including the `None`-start fail-closed path** — it is a second consumer, and inheriting
coverage from the NER windowing module's own tests would leave this path unowned.

**No new dependencies.** Sentence splitting is the existing regex; the `nvidia|triton` pin
guard is untouched.

## Risks

- **A negative result is a real outcome, pre-committed.** If neither strategy reaches usable
  precision, the finding is *"naive pairing is insufficient; CID relation extraction needs
  its own phase"* — that is the deliverable, not a failure, and it is documented exactly as
  ADR-0011 and ADR-0012 were.

- **Gold CID is narrower than co-mention. This corrects how to read the number, not how much
  the problem matters.** BC5CDR annotates *chemical-induced-disease* relations, not "this
  abstract discusses both," so cross-product's 0.197 is not pure noise — much of it is
  genuine co-mention that gold correctly declines to call a relation. Stating this prevents
  the number from reading as worse than it is and overstating the case for a relation model.
  **It is not an argument that low precision matters less in practice.** The Critic's own
  task — comparing specific drug–disease claims across papers — *is* the narrow,
  causal-relation-shaped task BC5CDR annotates. A pair that is co-mention rather than
  causal relation is exactly the pair the Critic should not have been handed. The caveat
  adjusts interpretation of the metric; it does not soften the finding.

- **The tail stays thin even at 1500 documents.** 325 gold clusters, 122 of size ≥3. Better
  than 80/21, but precision on large clusters rests on few examples. Per this project's own
  repeatedly-learned lesson (every error-composition pattern in the 90-mention domain sample
  reversed at scale), this licenses the **numbers** and not error-**composition** claims
  about cluster size.

- **Arm A is a ceiling no real system reaches**, for the same reason the oracle ceiling was:
  entities are granted, perfect, and free of NER and linking error.

## Deliverable

The 4-configuration table at all three metric levels; the Critic-workload counts with top-5
concentration; the NIL-side split; both anchors reproduced — and **a recommendation on
whether CID relation extraction is worth scoping as its own phase**, brought back before any
decision to build it.
