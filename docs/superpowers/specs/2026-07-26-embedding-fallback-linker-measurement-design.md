# Embedding fallback linker: measuring the precision cost

**Date:** 2026-07-26
**Status:** Approved for planning
**Phase:** 3C (canonicalization), final measurement

## The question

The oracle ceiling established that granting gold ids to all 1679 EXACT-but-NIL mentions
would move concept-level F1 from **0.7697 → 0.8518 (+0.0821)**, of which the
genuine-paraphrase slice (1136 mentions) accounts for **+0.0816** and the
abbreviation-addressable slice (543 mentions) for **+0.0008**.

That ceiling is **recall-only by construction**: granting a gold id can only convert
`fn`→`tp`, so `fp` stayed pinned at 312 across all four oracle variants. It is therefore
an upper bound that no real linker can reach, and its distance from reality is entirely
unmeasured.

This spec defines the work that produces the missing number: **what a real fallback
linker actually costs in precision, as a curve over its confidence threshold.**

## Scope discipline (carried forward)

This project has twice produced figures that were true of a favorably-selected
subpopulation and silent about system cost (the mention-count overstatement, 91→11 and
543→4). The measurement is therefore defined over the **full NIL population, not the
1136**.

| population | count | source |
|---|---|---|
| predicted mentions | 10038 | `n_predicted` |
| — linked by dictionary | 6829 | `n_predicted_linked` |
| — **NIL (fallback fires here)** | **3209** | difference |
| NIL *and* exact-span gold-aligned | 1679 | `exact_link.NIL` |
| → abbreviation-addressable | 543 | `oracle_abbrev` |
| → paraphrase | 1136 | `oracle_paraphrase` |
| **NIL, not exact-span gold-aligned** | **1530** | remainder |

The 1530 are truncated spans, spurious NER output, and the 53 `GOLD_UNLINKABLE` mentions
where gold carries no MeSH id and **NIL is the correct answer**. A real linker cannot see
the gold alignment, so it fires on all 3209. Every confident wrong link among the 1530 is
a new false positive that a 1136-scoped measurement structurally cannot observe.

The 1136-scoped result is produced as a free by-product of the same pass. It is reported
**only alongside** the full-population result and **always labeled as a slice figure**,
under the same rule applied to the ceiling numbers: a ceiling or slice figure stays
labeled as such everywhere it is cited.

## Non-goals

- Shipping the fallback linker. This measurement decides whether it is worth building.
- Improving NER. The 3209 NIL population is taken exactly as the current pipeline emits it.
- Overriding dictionary hits. See blast radius below.
- Approximate nearest-neighbour search, sharding, or serving concerns.

## Architecture

### Blast radius: NIL-only

The fallback implements the existing `Linker` protocol and fires **only where
`DictionaryLinker` returns NIL**. It never overrides a dictionary hit. Consequences:

- The 5832 `LINKED_CORRECT` mentions are untouched — the fallback cannot regress them.
- The 488 `LINKED_WRONG` mentions are untouched — the fallback cannot fix them either.
- Exactly the 3209 NIL mentions are in play.

`DictionaryLinker` already sits behind the `Linker` protocol with a docstring
anticipating this, so **`canonicalize()` needs no change**.

### Surface-only, no document context

SapBERT encodes a mention string in isolation. The fallback therefore requires nothing
from `check_document_context` and has no document-completeness precondition — a genuine
asymmetry with abbreviation expansion, which does. Record this in the report: it is the
reason this mechanism is cheap to place in the pipeline and that one was not.

### Components

**1. `biolit.canon.mesh.build_concept_labels(chem_text, disease_text)`**

Returns `dict[str, set[EntityLabel]]` mapping concept id → labels. Reuses the existing
`_read_ctd_dump` by-name column resolution; `build_alias_table`'s signature is unchanged.

Deliberately a **`set`**, not a scalar. A concept appearing in both CTD dumps must be
recorded as carrying both labels, not silently resolved last-write-wins — that is the
precise shape of defect this project has repeatedly caught. The count of multi-label
concepts is reported as a build diagnostic.

Written by `build_mesh.py` to `data/canon/concept_labels.json.gz` from the same
downloaded CTD text as the alias table, so the two artifacts cannot drift.

**2. `biolit_evals.embedding_index`**

- Encoder: **SapBERT** (`cambridgeltl/SapBERT-from-PubMedBERT-fulltext`) — the same
  PubMedBERT backbone the NER model already uses, contrastively trained over UMLS
  synonym pairs, which is exactly the "same concept, different surface" relation the
  paraphrase slice represents.
- Indexes **all 551,669 aliases** (not the 192,816 preferred names). A concept's score is
  the **max** similarity over its aliases, per SapBERT's own inference procedure.
- fp16 on disk (~847 MB, under gitignored `data/`), fp32 for the matmul.
- **Exact brute-force cosine, chunked.** No FAISS/HNSW. The query side is only 3209
  mentions, so exact search costs seconds — and ANN would inject approximation error into
  a measurement whose entire purpose is error attribution.

**3. `biolit_evals.tfidf_baseline`**

Character n-gram TF-IDF (`analyzer="char_wb"`, `ngram_range=(3,3)`) over the same 551,669
aliases, sparse cosine, same max-over-aliases rule, same sweep.

This is a **control, not an alternative**. Without it a SapBERT gain is unattributable
between semantic generalization and plain fuzzy string matching — the same distinction
left open when the MeSH-enrichment hypothesis came back null at 0.1%. If char n-grams
close most of the gap, the answer is a better string matcher, not a model, and that is a
materially different build.

Adds `scikit-learn` + `scipy` as eval dependencies. CPU-only; does not affect the
`nvidia|triton` pin guard.

**4. Sweep harness in `biolit_evals.end_to_end`**

Re-scores end-to-end at each threshold, emitting concept-level P/R/F1 with **real `fp`
movement**, in four arms:

| arm | scorer | candidates |
|---|---|---|
| 1 | SapBERT | unconstrained |
| 2 | SapBERT | label-constrained |
| 3 | TF-IDF | unconstrained |
| 4 | TF-IDF | label-constrained |

Label-constrained admits a candidate only if the mention's `EntityLabel` is in the
concept's label set.

## The sweep

A mention links to its top-1 candidate only if similarity ≥ threshold, else stays NIL.

The sweep is computed **by breakpoint, not on a fixed grid**: take the top-1 similarity of
each of the 3209 NIL mentions, sort descending, and treat each distinct value as a
threshold. This yields the *exact* curve — a fixed grid can straddle the region where F1
peaks and misreport the shape. Re-scoring is pure set arithmetic over 500 documents, so
the full breakpoint sweep is cheap.

The full curve is persisted to the run log. The report shows a readable subsample (0.00 to
1.00 in steps of 0.05) **plus the exact argmax-F1 breakpoint**, so the peak is never an
artifact of where the grid happened to land.

Reported per arm per threshold: precision, recall, F1, count fired, count correct, count
wrong — plus the 1136-slice breakdown as a labeled by-product.

**Two anchors are correctness checks on the harness itself, not results:**

- **threshold = 1.0** — nothing fires; the run MUST reproduce baseline F1 **0.7697** and
  `fp` **312** exactly. Any deviation means the harness is wrong, and the sweep is invalid
  until it reproduces.
- **threshold = 0** — every NIL mention takes its top-1; maximum recall, worst precision.

The curve between them is the deliverable. **No single operating point is reported as
"the" number** — the threshold is where the precision/recall tradeoff lives, and a single
point risks hiding whether the real opportunity is larger or smaller than whatever value
happened to be chosen.

## Testing

Follows the existing `biolit.canon` TDD discipline; new code gets tests regardless of size.

- `build_concept_labels`: a known chemical alias and a known disease alias each resolving
  to the correct label from CTD source text; plus a concept present in both dumps
  producing a two-element set.
- Index: max-over-aliases returns the concept whose *best* alias matches, not its
  preferred name, on a hand-built fixture with a stub encoder.
- Threshold gate: at 1.0 nothing fires; at 0 everything fires.
- Label constraint: a candidate of the wrong label is excluded.
- Real model download sits behind the existing `heavy` marker.

Fixtures use real CTD-shaped text. No gold MeSH ids, PMIDs, or abstract text are invented.

## Risks

- **Encode throughput is unmeasured.** 551k aliases through a BERT-base on CPU is
  estimated at tens of minutes but has not been timed. A pilot on ~5k aliases must
  establish the real rate before committing to the full build.
- **A negative result is a real outcome.** If the curve shows no threshold where F1 beats
  0.7697, the fallback is rejected — and per the standing "no infrastructure without a
  demonstrated consumer" rule, that branch stays unmerged with the finding documented,
  exactly as `phase-3c-mesh-alias-enrichment` was handled.
- **Concept-level scoring may mask mention-level cost.** Duplicate concepts within a
  document collapse, so several wrong mention links can cost a single `fp`. Note this
  where the result is cited.

## Deliverable

The threshold curve for all four arms, the labeled 1136-slice by-product, and a
recommendation on whether to build — brought back before any decision to ship.
