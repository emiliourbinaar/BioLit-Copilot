# End-to-End Canonicalization Eval — Design Spec

- **Date:** 2026-07-23
- **Status:** Approved (brainstorm complete)
- **Phase:** 3, sub-project B (measurement; precedes clustering)
- **Related:** ADR-0008 (canonicalization is a clustering prerequisite), ADR-0009 (a merge may only fill a NIL gap)

## 1. Context & goal

Every canonicalization number reported so far (BC5CDR F1 0.7842, domain F1 0.7455) was
produced by feeding **gold surface forms** to the linker. That deliberately isolated
linking from NER — but it means the production path has never been measured:
`extract_entities` → `canonicalize` (merge + link) → `canonical_id`. In particular
`merge_fragments`, the direct ADR-0008 fix, contributes **nothing** to any published
number and is backed by unit tests alone.

A 49-sentence probe over the domain corpus found that of 90 gold entities, only **2** were
fragmented into the multi-adjacent-span shape `merge_fragments` can address, and merging
recovered **0** gold spans (merging `GLP` + `1RA` yields `GLP-1RA`; gold is `GLP-1RAs`).
The dominant failure was **truncation inside a single prediction** (11 cases: `CFD`→`CF`,
`insulin resistance`→`resistance`), which merging structurally cannot repair. This eval
exists to establish whether that holds at full scale and to make the breakdown permanent.

**Goal:** measure the real path end to end, and attach to that number a permanent
categorical breakdown explaining every point of loss.

## 2. Scope

**In scope:** document-level gold loaders; a reusable outcome census; end-to-end scoring
(concept-level micro-averaged P/R/F1, pooled and per label); end-to-end NIL reported beside
the existing gold-surface NIL; a descriptive merge audit; a run log; a report section.

**Explicitly out of scope — do not do these in this task:**
- Any change to `merge_fragments` or its merge heuristic.
- Truncation / boundary recovery of any kind. Recovering dropped characters *within* one
  span is a different mechanism solving a different problem than ADR-0008 targeted; if it
  is worth pursuing it gets its own brainstorm and decision, informed by this census at
  full scale — not designed reactively off an 11-case sample.
- Scoping the SapBERT fallback.
- Clustering.

## 3. Data plumbing

Both gold loaders return flat mention lists and **discard the document text**, so neither
can feed the NER model. Add document-level variants in `biolit_evals/mesh_gold.py`:

- `parse_pubtator_documents(text) -> list[GoldDocument]` — BC5CDR from `CDR_Data.zip`.
- `load_domain_norm_documents(path) -> list[GoldDocument]` — the 49 blind-annotated sentences.

```python
@dataclass(frozen=True)
class GoldDocument:
    pmid: str
    text: str
    mentions: list[GoldMention]
```

For PubTator, `text` is the document's title and abstract joined with **a single space** —
`title + " " + abstract`. This is not an assumption: it was verified against the real
corpus before this spec was written, and **9809 of 9809 mentions across all 500 test
documents** satisfy `text[start:end] == mention_text` under it. A zero-length separator
aligns only 2 of the first document's 11 mentions, so the separator is load-bearing.
(Any single character works, since only its length affects offsets; `" "` is chosen as the
faithful rendering.) The 9809 total also independently matches the Phase 2 BC5CDR mention
count (`tp + fn = 9809`), so this parse agrees with the already-validated NER gold; 9718 of
them carry a MeSH ID and 91 are annotated unlinkable.

**Methodological gain worth recording:** `CDR_Data.zip` carries the **original natural
abstract text**, unlike `tner/bc5cdr`, whose text is space-joined tokens with punctuation
pre-separated. Phase 2 flagged that difference as a confound making its 0.81-vs-0.66
comparison "not strictly comparable". This eval runs on natural text for **both** corpora,
so the confound does not apply to its numbers.

## 4. The outcome census (permanent, not a probe artifact)

Every gold mention is classified by NER outcome. This is the categorical breakdown *of the
end-to-end number* — it accounts for every point of loss, and is reported on every run.

| Category | Definition |
|---|---|
| `EXACT` | some prediction matches the gold `(start, end, label)` exactly |
| `MERGEABLE` | ≥2 same-label predictions overlap the gold span (merging could join them) |
| `TRUNCATED` | exactly 1 same-label prediction overlaps the gold but is not identical |
| `MISSED` | no same-label prediction overlaps the gold |

`TRUNCATED` is sub-classified to answer whether truncation is systematic:

| Sub-category | Definition |
|---|---|
| `PREFIX_OF_GOLD` | the prediction is a leading substring of gold (**suffix dropped**) |
| `SUFFIX_OF_GOLD` | the prediction is a trailing substring of gold (**prefix dropped**) |
| `INTERIOR_OR_OTHER` | overlapping but neither (includes predictions extending past gold) |

Each `TRUNCATED` case also records the character delta (gold length − prediction length).
The **entire census is reported per label (CHEMICAL / DISEASE) as well as pooled.**

## 5. Metrics

### 5.1 Concept-level P/R/F1 — the primary metric

Reported because it is what clustering consumes: clustering keys on the set of concepts a
paper mentions, not on exact character spans.

Per document, over concepts:
- `G` = the set of MeSH IDs across that document's gold mentions. Mentions annotated
  unlinkable contribute nothing. A composite mention (`D1|D2`) contributes **both** IDs —
  such a mention genuinely denotes both concepts, so both are required.
- `P` = the set of non-NIL `canonical_id`s on the document's canonicalized predictions.
- `tp = |G ∩ P|`, `fp = |P − G|`, `fn = |G − P|`.

**Averaging is micro:** `tp`, `fp`, `fn` are **summed across all documents** and
precision/recall/F1 computed once from those totals — consistent with the Phase 2 span
scorer's micro-averaging discipline. It is **not** a macro-average of per-document scores.

**Set semantics within a document** (deliberate, and a considered difference from the
Phase 2 span scorer's multiset counting): at concept level the question is presence or
absence — "does this paper mention concept X" — so a concept mentioned five times in one
abstract counts once. Multiset counting would let a single frequently-repeated entity
dominate the score.

### 5.2 Per-label concept P/R/F1

The same computation partitioned by label: `G_CHEMICAL` / `P_CHEMICAL` from gold mentions
and predictions labelled CHEMICAL, likewise DISEASE. Reported alongside pooled.

Rationale: this project has already produced per-type asymmetries (differing NIL and
tiebreak rates), and the CTD schema defect was **chemical-only** — disease parsing was
correct throughout. A pooled metric is demonstrably capable of masking a per-type failure
in exactly this code path.

### 5.3 End-to-end NIL, reported beside gold-surface NIL

`e2e_nil_rate` = predicted entities with no `canonical_id` ÷ all predicted entities.

Reported next to the existing gold-surface NIL (0.259 BC5CDR / 0.406 domain) with an
explicit statement that **the two have different populations and denominators** — one is
over model predictions, the other over gold mentions — so they are comparable in direction
only, never subtractable. The gap between them is the headroom any prediction-side fix
(merging or otherwise) could address.

### 5.4 Merge audit — descriptive only

Counts: candidates proposed, candidates that linked, candidates whose span exactly equals a
gold span, and constituents that inherited a merged concept. Reported as **raw counts, not
precision/recall**, because the probe indicates n is tiny (2 on the domain corpus) and a
ratio over n=2 would be noise presented as a metric. If full-scale n turns out large enough
to support a rate, that is a follow-up decision, not an assumption baked in now.

## 6. Components

- `backend/src/biolit_evals/outcome_census.py` — `Outcome`/`TruncationKind` `StrEnum`s,
  `classify_outcome(gold, predictions) -> OutcomeRecord`, `census(records) -> Census`.
  Pure, no model, fully unit-tested.
- `backend/src/biolit_evals/end_to_end.py` — `score_end_to_end(...) -> E2EMetrics`
  (concept metrics pooled + per label, e2e NIL, census, merge audit) and
  `run_e2e_eval(...)` appending one JSON line to `evals/e2e_runs.jsonl`, with all impure
  inputs injected, mirroring `run_eval` / `run_canon_eval`.
- `backend/src/biolit_evals/mesh_gold.py` — add `GoldDocument`,
  `parse_pubtator_documents`, `load_domain_norm_documents`. Existing functions unchanged.

## 7. Testing

TDD throughout; tdd-guard active. Unit tests use fake predictors and fixtures — no model,
no downloads. Real runs are `heavy`-marked.

- **Census classification:** one test per category and per truncation sub-category,
  including the real shapes seen in the probe (`GLP`+`1RA` → `MERGEABLE`; `CFD`→`CF` →
  `TRUNCATED`/`PREFIX_OF_GOLD`; `insulin resistance`→`resistance` →
  `TRUNCATED`/`SUFFIX_OF_GOLD`).
- **Concept metrics:** a case with precision ≠ recall so a swapped or arithmetic-mean F1
  fails; a case proving set-semantics (a concept mentioned twice in one document counts
  once); a case proving micro-averaging (two documents whose pooled totals differ from the
  mean of their per-document scores).
- **`parse_pubtator_documents` cross-validation (required):** assert that, for the same
  source text, the documents' mentions **flattened in order are equal to** the output of
  the already-validated flat `parse_pubtator` — same total count and identical
  `(start, end, label)` (and `mesh_ids`) per mention. This is a new code path extracting
  offsets against natural text for the first time; it must be checked against known-good
  parsing rather than trusted because downstream census tests pass. Cover it with both the
  existing PubTator fixture and a second fixture containing two documents, so
  document-boundary handling is exercised.
- **Offset-basis assertion (required, and distinct from the above):** for every mention in
  every parsed document, assert `document.text[m.start:m.end] == m.text`. The
  cross-validation test above compares mentions produced by two parsers that both read
  offsets from the *same mention lines*, so it cannot detect a wrong document-text
  reconstruction — only this assertion pins the title/abstract separator. It is the direct
  analogue of the span/text check that guards the domain gold loader. The `heavy` real-data
  smoke test asserts the same across all 500 BC5CDR documents, where it must hold for
  9809/9809 mentions.
- **Run log:** exact key-set assertion, parent-directory creation, append-only — the
  pattern used by the two existing runners.

## 8. Reporting

A new section in `docs/EVAL_REPORT.md`: end-to-end concept-level P/R/F1 (pooled and per
label) for BC5CDR and the domain sample, the full census with truncation sub-classification
per label, e2e NIL beside gold-surface NIL with the incomparability caveat, and the merge
audit counts. It must state plainly whether the probe's finding — that merging recovers
approximately nothing while truncation dominates — holds at full scale, and whether
truncation looks systematic (consistently suffix-dropping? concentrated in one label?)
**without** proposing a fix, which is a separate future decision.

## 9. Task-ordering sketch

1. `GoldDocument` + `parse_pubtator_documents` + cross-validation test against
   `parse_pubtator` + the offset-basis assertion.
2. `load_domain_norm_documents` + test.
3. `outcome_census.py`: `Outcome`/`TruncationKind`, `classify_outcome` + per-category tests.
4. `census()` aggregation (pooled + per label) + tests.
5. Concept-level metrics (micro, set-per-document, pooled + per label) + tests.
6. `score_end_to_end` wiring census + concept metrics + e2e NIL + merge audit, with an
   injected fake predictor + fake linker.
7. `run_e2e_eval` + run log + CLI; heavy-marked real smoke.
8. Real runs on both corpora; `docs/EVAL_REPORT.md` section; `docs/ARCHITECTURE.md` touch-up.

No real model or download is exercised until tasks 1–7 are committed.
