# Entity Canonicalization (NEN) — Design Spec

- **Date:** 2026-07-22
- **Status:** Approved (brainstorm complete)
- **Phase:** 3, sub-project A (prerequisite for entity-keyed clustering)
- **Motivating decision:** ADR-0008 (entity canonicalization is a Phase 3 prerequisite, not a clustering subtask)
- **Related:** ADR-0005 (StrEnum), ADR-0006 (blind annotation), ADR-0003 (storage deferred to Phase 3)

## 1. Context & goal

Phase 2's domain error analysis found that **60% of domain NER errors are boundary
disagreements** — drug-class abbreviations and hyphenated compounds fragmenting under the
subword tokenizer (`GLP-1RAs` → `GLP` + `1RA`, gold `CFD` predicted as `CF`). The danger
is not the NER score: it is that fragmented spans **cluster inconsistently across papers**,
so entity-keyed clustering silently splits papers that belong together and the failure
surfaces as *missing clusters*, never as a flagged NER error (ADR-0008).

This sub-project builds a **named-entity normalization (NEN)** layer, `canonicalize(entities,
text)`, that maps each detected CHEMICAL/DISEASE span to a stable **MeSH concept ID**, so
clustering keys on a concept rather
than a fragile surface string. Detection (NER, Phase 2) and normalization (NEN, this work)
remain **separate concerns with separate evals**, mirroring how the project has kept every
correctness-critical component independently measurable.

Priorities carried from the project: correctness/faithfulness > eval-harness rigor >
orchestration > frontend; dependency-careful and offline-first; test-first throughout.

## 2. Scope

**In scope:**
- `canonicalize(entities: list[Entity], text: str) -> list[Entity]` populating `canonical_id`
  / `canonical_name` on each entity (NIL-able). It takes the source `text` because
  fragment-merge (§3.1) must inspect the characters *between* adjacent spans to tell a hyphen
  from a space. The **fallback-agnostic** requirement lives one level down, in the `Linker`
  protocol (§3.1) — that is the seam a second linking stage slots into — so the added `text`
  parameter does not compromise it.
- A standalone, test-first fragment-merge sub-component (the direct ADR-0008 fix).
- A dictionary linker over an offline CTD→MeSH alias table.
- A canonicalization eval harness (benchmark + blind domain sample) with its own metrics.

**Out of scope (deliberately deferred):**
- The clustering step itself (next Phase 3 sub-project) — this spec only guarantees
  clustering *can* key on `canonical_id`.
- Retrieval / pgvector wiring.
- The SapBERT (or other embedding) fallback linker — **interface-only** here. It is added
  later *only if* the canonicalization eval shows a coverage gap the dictionary cannot close.
- The Phase 4 Extractor. `extract_entities`'s signature and responsibility are untouched.

## 3. Architecture

### 3.1 New package `biolit.canon`

NEN is a distinct concern from `biolit.ner`, so it gets a sibling package:

- **`mesh.py`** — `MeshConcept` (id, name) and `MeshDictionary`: builds/loads the offline
  alias→concept table from the CTD vocabularies; exposes `lookup(normalized: str) -> MeshConcept | None`.
- **`fragments.py`** — `merge_fragments(entities: list[Entity], text: str) -> list[FragmentCandidate]`:
  re-merges adjacent same-label spans separated only by a hyphen or zero gap in the *source
  text*, emitting merged candidates **without discarding the originals**. Its own tested unit.
- **`linker.py`** — a `Linker` `Protocol` (the fallback-agnostic seam) and `DictionaryLinker`
  (uses `MeshDictionary`). A future `SapBertLinker` implements the same protocol and slots in
  behind `DictionaryLinker` with no redesign.
- **`canonicalize.py`** — orchestrates the flow in §4 and is the public entrypoint.

### 3.2 Entity model change

`biolit.domain.records.Entity` gains two optional fields:

```python
canonical_id: str | None = None      # e.g. "MESH:D000067299" or "OMIM:125853"; None = NIL
canonical_name: str | None = None    # CTD preferred name; None = NIL
```

Both default `None` (NIL). Existing Entity construction and the Phase 2 pipeline are
unaffected (new optional fields). The MeSH/OMIM prefix is preserved on `canonical_id` so the
vocabulary of origin stays visible.

### 3.3 The seam

```
extract_entities(text)  ── unchanged (pure NER) ──►  [Entity(text,label,start,end)]
                                                        │
                              canonicalize(entities, text)  ── this sub-project ──►
                                                        │
                       [Entity(..., canonical_id, canonical_name)]  ──►  clustering (later)
```

Clustering (a later sub-project) is a **thin mechanical grouping** with no linking logic:
key = `canonical_id` when linked, else `nil:<normalized-surface>` so unlinked entities are
still clustered by surface form and never dropped — and NIL entities stay visibly distinct
from linked ones.

## 4. Data flow (deterministic throughout)

For a document's entities plus its source text:

1. `merge_fragments(entities, text)` produces zero or more **merged candidates** (originals
   retained).
2. For each entity, assemble lookup attempts in order: **(a)** any merged candidate it
   participates in, then **(b)** its own surface form — each passed through surface
   normalization (case-fold, whitespace/punctuation normalization).
3. `DictionaryLinker.link(attempt)` returns a `MeshConcept | None`. **First hit wins.**
4. On a hit, set `canonical_id` / `canonical_name`. On no hit across all attempts, leave NIL.

Merged-candidate-first ordering is what fixes the ADR-0008 fragmentation cases — but it is
also the exact risk the adversarial merge tests in §7 guard against (a bad merge that
coincidentally resolves to a real alias).

## 5. Data sources & artifacts (all offline after first build)

- **Alias table:** `CTD_chemicals.tsv.gz` (~10.6 MB) + `CTD_diseases.tsv.gz` (~1.8 MB) from
  ctdbase.org — MeSH-aligned with synonym columns. CTD/MEDIC is the vocabulary BC5CDR
  normalization is built against, so coverage is aligned by construction. Built into a
  compact cached artifact (alias → id + preferred name) via `python -m biolit.canon.build_mesh`,
  cached in a gitignored data dir (same pattern as the NER model checkpoint), with a
  verification count printed on build.
- **Benchmark gold:** `CDR_Data.zip` (~20 MB) from the **ungated** `bigbio/bc5cdr` HF repo,
  parsed directly from PubTator format — title/abstract lines plus mention lines
  `PMID⇥start⇥end⇥text⇥type⇥MeSH_ID`. **No `bigbio` package or `trust_remote_code`** — we
  parse the format ourselves.
- **Prefix reconciliation:** CTD ids are prefixed (`MESH:D…`, `OMIM:…`); BC5CDR gold ids are
  bare (`D…`, with `-1` for unlinkable). Comparison strips/normalizes the prefix; the
  reconciliation rule lives in one place and is unit-tested.

## 6. Eval harness (mirrors the NER harness in `biolit_evals`)

- **Primary metric — linking accuracy on gold mentions:** feed the **gold spans** (not NER
  output) into the linker and measure whether the assigned MeSH ID matches gold. This
  **isolates linking quality from NER quality** — a linking regression cannot hide behind an
  NER change, and vice versa.
- **NIL rate & coverage:** fraction of gold mentions the dictionary leaves NIL — the number
  that decides whether the SapBERT fallback is actually warranted.
- **Ambiguous-alias tiebreak rate (its own metric):** how often the deterministic tiebreak
  ("exact-name over synonym, then lexicographically smallest id") actually *fires*. Logged
  separately from accuracy: a silent, arbitrary resolution on a common surface form would
  otherwise be invisible inside the aggregate number, and this rate is what tells us whether
  the tiebreak rule needs revisiting.
- **Blind domain normalization sample (ADR-0006 style):** an in-domain cross-check. The gold
  MeSH IDs are annotated **cold — not viewing `DictionaryLinker`'s own output.** "Blind" here
  means *not anchored to the system's predictions*; annotators **may and should consult the
  MeSH thesaurus/browser directly** to find the correct concept IDs. This is not zero-tool
  annotation — it is annotation without the anchoring bias of seeing what the linker guessed.
  **This distinction is stated explicitly in the eval harness documentation** so the
  methodology is unambiguous.
- **Reporting:** strict entity-level linking **precision / recall / F1**, plus the NIL and
  tiebreak rates. Each real run appended as one JSON line to a canonicalization run log
  (same append-only pattern as `evals/runs.jsonl`). The real CTD build and BC5CDR gold run
  sit behind the `heavy` pytest marker; unit tests use a tiny fixture alias table.
- **The eval is built before deciding whether the fallback is needed** — the decision is
  data-driven, not assumed.

## 7. Testing strategy

TDD throughout, following the project's established discipline (tdd-guard active; write-then-Edit).

- **`merge_fragments` gets its own dedicated suite** — it is the direct ADR-0008 fix and
  earns the same rigor as the scorer did:
  - **Positive regressions:** `GLP` + `1RA` → merged candidate `GLP-1RA`; the `CFD`/`CF`
    boundary case; `SGLT2is` / `DPP4is` fragmentation.
  - **Adversarial (must-NOT-merge):** at least one case of two genuinely distinct,
    correctly-split same-label entities that happen to be adjacent with a hyphen/zero gap,
    asserting they are **not** merged. This guards the merged-candidate-first lookup order
    (§4) from silently assigning a wrong *shared* canonical id when a bad merge coincidentally
    resolves to a real alias.
- **Prefix reconciliation** (MeSH/OMIM/`-1`) is unit-tested in isolation.
- **`DictionaryLinker`** is tested against a small fixture alias table (exact hit, synonym
  hit, NIL, ambiguous→tiebreak).
- **Fixture-before-download discipline (ADR-0006):** no real CTD/BC5CDR download happens
  until the fixture-based units are committed. Real-data paths are `heavy`-marked.

## 8. Deferred / future work

- **SapBERT (or similar embedding) fallback** behind the `Linker` protocol — added only if
  §6's NIL rate / recall gap justifies it. No implementation now; the interface is kept
  fallback-agnostic so a second linking stage requires no redesign.
- A broader domain eval set is re-derived *after* canonicalization exists (project note),
  spending the annotation effort once against the normalized pipeline.

## 9. Task-ordering sketch (for the plan)

1. `Entity` gains `canonical_id` / `canonical_name` (with the model-validation test).
2. `merge_fragments` + its dedicated suite (positive + adversarial), fixture-only.
3. `MeshDictionary` load/lookup + prefix reconciliation, fixture-only.
4. `Linker` protocol + `DictionaryLinker`, fixture-only.
5. `canonicalize()` orchestration wiring 2–4, fixture-only.
6. Canonicalization eval harness: gold parsers (CTD build + BC5CDR PubTator), linking scorer
   with NIL + tiebreak metrics, run log; `heavy`-marked real run.
7. Blind domain normalization sample + documented methodology.
8. `docs/EVAL_REPORT.md` canonicalization section + `docs/ARCHITECTURE.md` update; ADR
   addendum if any design detail is decided during implementation.

No real model or dataset touches gold until the fixture-based units (1–5) are committed.
