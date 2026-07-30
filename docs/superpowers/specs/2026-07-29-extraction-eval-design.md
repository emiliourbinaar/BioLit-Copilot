# Phase 4 extraction eval — design

**Date:** 2026-07-29
**Status:** Approved (brainstorm complete, ready for implementation planning)
**Corpus:** BC5CDR Test-500
**Model:** `claude-opus-5` ($5 / $25 per MTok)

---

## 1. Why this sub-project exists

Phases 1–3 produced a well-measured deterministic pipeline (NER, canonicalization, clustering) and
**zero agents**. All seven agent contracts are declared in `biolit/state/contracts.py`, but only the
Extractor has adapters (`project_extractor` / `merge_extractor`), there is no `langgraph` dependency,
and there is no LLM dependency or LLM code anywhere in `src/`. For a project whose stated priorities
rank orchestration clarity above frontend polish, that is the largest gap.

The Extractor is the right first agent: the codebase already has its state seam, `extract_entities`
was explicitly designed as "the stable anchor Phase 4's Extractor will call," and ADR-0013 supplies a
ready-made discipline for pricing it before building it.

**The hard part is not the extraction code. It is what to measure against.** BC5CDR annotates
entities and CID relations — not study types, sample sizes, or findings. This spec's substance is
the measurement design.

---

## 2. Scope: `key_findings` only

`ExtractedRecord` declares three LLM-fillable fields. Only one is in scope.

| field | in scope | why |
|---|---|---|
| `key_findings` | **yes** | Has a plausible consumer: the Critic compares claims across papers |
| `study_type` | no | **No contract declares it.** `CriticInput` takes `clusters` + `records`; `SynthesisInput` takes `question` + `records` + `contradictions`. Neither needs it. |
| `sample_size` | no | Same, and the hardest of the three to extract reliably |

`study_type` and `sample_size` stay `None` until a consumer demonstrably needs them — the same
standard ADR-0013 applied to CID relation extraction and to cluster-size capping, and the standard
that kept ADR-0011's ingest and ADR-0012's fallback linker unmerged. This project has already
measured three phases' worth of quantities for a consumer that did not exist; that is not repeated
here.

---

## 3. Interface decisions

### 3.1 `key_findings: list[str]` → `list[Finding]`

```python
class Finding(BaseModel):
    text: str
    start: int
    end: int
    sentence_index: int
```

Mirrors `Entity`'s shape. Without offsets a citation cannot point at a location and a repeated
sentence is ambiguous. `sentence_index` is retained rather than discarded because **it is the
model's actual output** — keeping it makes a run log auditable against the prompt without
re-deriving the sentence split.

This is the interface decision recorded explicitly, in the same manner as the clustering plan's
multi-id representation decision and the label map's serialized-list-vs-in-memory-set choice.

### 3.2 The model returns sentence indices, never character offsets

**The single most important design decision in this spec.**

The abstract is presented to the model as a numbered sentence list; the model returns
`{"finding_sentences": [1, 2]}`. The harness maps indices back to spans through
`biolit.ner.windowing.sentence_spans`.

```
[0] Metformin was administered to 40 patients with PCOS.
[1] Treatment reduced fasting insulin by 23% (p<0.01).
[2] Two patients developed lactic acidosis.
```

Rationale, stated as approved:

> The model's task shrinks to what LLMs are actually reliable at, and the harness owns the one thing
> that needs to be exact — converting a **detected** failure mode into a **structurally impossible**
> one.

LLMs are unreliable at character arithmetic. Asking for offsets invites exactly the hallucinated-span
failure this design exists to eliminate. An index either exists in the sentence list or it does not.

Faithfulness therefore becomes an **invariant the harness enforces**, not a metric that can fail —
which is why the eval's headline is coverage rather than grounding. This follows the project's
established preference for structural guarantees over measured hope (ADR-0009: a merged fragment may
only fill a NIL gap, never overwrite a link; `linked_ids` excludes NIL rather than assuming it
harmless).

**Out-of-range indices are dropped, never clamped or guessed.** A returned index outside
`[0, len(sentence_spans))` yields no `Finding` for that index and increments a diagnostic counter.
Consistent with every other fail-closed decision in this project. Stated here explicitly rather than
left to be decided implicitly in code.

**Schema gap to be aware of:** structured outputs do not support numerical bounds
(`minimum` / `maximum`), so the JSON schema can guarantee a list of integers but **not** that they
are in range. The harness bounds-checks; the out-of-range rate is a reported reliability diagnostic.

### 3.3 The `Extractor` seam

```python
class Extractor(Protocol):
    def findings(self, paper: Paper) -> list[Finding]: ...
```

Mirrors the `Linker` and `PairingStrategy` protocols. Two implementations in `biolit/extract/`:

- **`SameSentenceAsEntitiesExtractor`** — the deterministic control. Selects sentences containing
  both a linked chemical and a linked disease. Free, no API. This is the control that makes an LLM
  score attributable, filling the role the character n-gram TF-IDF control filled in Phase 3C:
  without it, a gain cannot be credited to the LLM rather than to the task being easy.
- **`LlmExtractor`** — `claude-opus-5`, structured outputs via `output_config.format`, prompt-cached
  system prompt (Opus 5's 512-token cache minimum means the instruction block caches),
  `stop_reason == "refusal"` checked **before** reading `content`.

The Protocol is what lets the eval score both through one code path — the same premise that made the
clustering eval's arms comparable, and ADR-0010's argument that the eval must measure the production
path.

---

## 4. Two failure modes that look alike and are not

`Paper.extraction_allowed` already exists (ADR-0004's licence axis) and is wired in both clients.
This is its first consumer. It and a safety refusal both surface as "extraction produced nothing,"
and **they must not be conflated.**

| | `extraction_allowed=False` | `stop_reason == "refusal"` |
|---|---|---|
| **Scope** | **Suppresses the entire `ExtractedRecord`** | **Empties `key_findings` only** |
| **Entities** | Not emitted | **Retained** |
| **Why** | The licence concern applies to *every* text-bearing field | The LLM declined *one call*; it has no bearing on entities |
| **Diagnostic** | `papers_licence_skipped` | `papers_refused` |

**Why the licence gate suppresses everything.** `ExtractedRecord` is **not text-free**: both
`Entity.text` and `Finding.text` carry verbatim abstract substrings. A non-extractable paper that
still emitted entities would leak abstract text downstream even with zero findings. Emptying only
`key_findings` would therefore be insufficient.

**Why the refusal empties only findings.** Entities are produced by the separate, deterministic
NER/linking stage, which never consults the LLM. A refusal on the findings call says nothing about
them, so they stay.

**Why gating inside `LlmExtractor` is sufficient as the single enforcement point.** `Paper` appears
in exactly two contracts — `RetrieverOutput` and `ExtractorInput` (`contracts.py:25,29`). **The
Extractor is the last node that ever sees a `Paper`.** Critic and Synthesis take only
`ExtractedRecord` / `Cluster` / `ContradictionFinding`. So a non-extractable paper producing no
record means nothing downstream can act on its content regardless of how future phases are built.
Recorded here rather than assumed.

BC5CDR is fully extractable, so the licence path is exercised by **unit tests only** — never by the
eval run. Stated as a known coverage limit rather than allowed to look tested.

---

## 5. The harness

`biolit_evals/extract_eval.py`, corpus **Test-500** — chosen for comparability with every other
pipeline-level number in this project (Phase 3B concept F1, Phase 3C's fallback sweep, clustering
Arm B, the 40.3% endpoint loss).

**No split protects against LLM contamination.** BC5CDR is a public 2015 benchmark almost certainly
present in any large pretraining scrape. Unlike the NER checkpoint — fine-tuned on BC5CDR *train*,
which is why clustering Arm B was held out — an LLM's exposure is uniform across splits. This is an
unfixable limitation to document, not a problem solvable by split choice, and it affects the LLM arm
only, not the deterministic controls.

### 5.1 Gold: sentences where a gold CID relation's endpoints co-occur

A paper's gold sentence set is every sentence index containing a gold mention of **both** endpoints
of at least one gold CID relation. Computable entirely from assets already in the repo — gold
mentions (which carry offsets), `load_bc5cdr_cid_relations`, and `sentence_spans`. Zero annotation
cost, full Test-500 scale.

### 5.2 Three arms

| arm | entities | role |
|---|---|---|
| **control-gold** | gold mentions | Recall-only ceiling; carries anchor #1 |
| **control-real** | real NER + linking | The entity-conditioned number the LLM must beat |
| **llm** | **none — reads text directly** | `claude-opus-5` |

**`control-gold` is fully scored, not merely an anchor carrier — and it is a ceiling for RECALL
only.** Applying the ADR-0013 rule prospectively rather than retroactively: ask what it can make
worse. Its recall is 1.0000 by construction, because every gold sentence contains both gold endpoints
and is therefore selected. Its **precision is not 1.0**, because it selects any sentence where a gold
chemical and a gold disease co-occur — including pairs that are not gold CID relations. So it is not
degenerate with gold, and it is not a ceiling for precision or F1.

That precision figure is itself a result: it measures what "co-occurrence ≠ relation" costs on
*perfect* entities, isolating the proxy's own imprecision from any pipeline error. Structurally this
is the direct analogue of `cross_product` in the clustering eval — recall 1.0000, precision poor —
and it should be cited with the same care.

**This inverts the standing finding for the first time in the project.** `control-real` is capped by
the 40.3% endpoint loss. The LLM arm is **not** — it reads the abstract directly and never consults
NER, so its ceiling is 1.0. The headroom between `control-real` and 1.0 is precisely the value an LLM
adds by *escaping* the upstream entity bottleneck rather than inheriting it. ADR-0013 directs pricing
downstream components against an entity-conditioned ceiling; the honest finding here may be that this
particular component is not entity-conditioned at all. That is measured, not assumed.

**Scope limit on that claim:** findings escape the bottleneck; **clustering does not**. The Critic
consumes both, so no pipeline-level claim inherits this.

### 5.3 Primary metric

Micro-averaged sentence-selection **P / R / F1** over `(paper_id, sentence_index)` pairs.

**Precision is load-bearing and recall is not quotable alone.** Selecting every sentence scores
recall 1.0 — the exact over-selection trap the clustering eval hit from the other direction. Mean
sentences-selected-per-paper is logged alongside every arm.

### 5.4 `control-real`'s false negatives decompose three ways

Its ceiling must not be reported as a single conflated number. The third bucket becomes visible only
because gold sentences are defined by *gold*-mention co-occurrence while the control runs on *real*
spans.

| bucket | condition | what it prices |
|---|---|---|
| **(a) endpoint lost** | ≥1 endpoint has no linked mention anywhere in the paper | **Unrecoverable** by any window or pairing mechanism. The 40.3% population. |
| **(b) linked, never co-sentential** | Both endpoints linked somewhere, no sentence holds both | Recoverable by a wider window — prices same-paragraph / N-token variants |
| **(c) co-sentential elsewhere** | Both linked and co-sentential, but in a sentence other than the gold one | Span disagreement between gold mentions and real NER. Costs a false positive **and** a false negative |

Three buckets, three different fixes. Conflating them would make "the ceiling" unactionable — the
same defect as Phase 3C's diluted `mentions_wrong / fp` ratio.

### 5.5 Bucket (a) is where bottleneck escape is proven

**The LLM arm's recall is additionally reported restricted to bucket (a)** — the population
`control-real` cannot reach by construction.

An aggregate score comparison does not rule out the LLM simply being generally better at the shared
part of the task without ever reaching what `control-real` structurally cannot. **Meaningfully
positive recall on bucket (a) specifically is the direct empirical proof.** Same evidentiary standard
as citing `same_sentence` recall 0.7346 > oracle 0.6967 rather than arguing structurally that the
oracle was one-sided.

### 5.6 Diagnostics

`out_of_range_index_rate`, `papers_refused`, `papers_licence_skipped`, mean sentences-per-paper per
arm, and the bucket (a)/(b)/(c) counts.

---

## 6. Anchors, on the real run path

1. **`assert_gold_sentence_recall_anchor`** — `control-gold` recall must round to exactly
   **1.0000**. Gold sentences are *defined* by gold-endpoint co-occurrence, so a co-occurrence
   selector running on gold mentions cannot miss one. A miss means the harness is wrong, not that the
   selector underperformed. Tolerance is `round(r, 4)` — Phase 3C's near-miss lesson showed that is
   what catches a harness dropping 1 gold item in 1000.
2. **`assert_gold_sentence_count_anchor`** — tabulated gold-sentence count for Test-500. A
   **loader-correctness check, explicitly not a quality check** — the same distinction
   `assert_gold_cluster_anchor` documents.
3. **`assert_dataset_size`** — reused unchanged from `cluster_eval`.
4. **Bucket closure** — (a) + (b) + (c) must equal `control-real`'s total false negatives exactly.
   A cheap identity that catches a misclassified bucket, in the spirit of the three-way
   reachable-share / oracle-recall / cross-product-recall agreement at 0.5966.

**Gate discipline:** if an anchor fires, the harness is wrong. Report the mismatch; do **not** adjust
an anchor to match an observation; do **not** commit the run as a result.

---

## 7. Testing

- **Tests never touch the API.** `LlmExtractor` takes an injected client; unit tests use a stub
  returning canned JSON. The full suite stays offline and fast.
- Real-model runs go behind the existing `heavy` pytest marker (Phase 2 precedent).
- `main()` gets no direct unit test — precedent: `end_to_end.main()`, `ner_eval.main()`,
  `cluster_eval.main()`.
- Every impure input injected (`log_path`, `git_sha`, `now`), matching `run_e2e_eval`,
  `run_canon_eval`, `run_cluster_eval`.
- **Determinism fixtures use 7 reverse-inserted elements, not 2.** On the clustering branch a
  2-element fixture let a dropped `sorted()` escape on 4 of 12 `PYTHONHASHSEED` values — a third of
  CI runs. Not re-learned here.
- **Commit code before running.** `git_sha()` records HEAD and ignores a dirty tree, so a run log
  line can otherwise name a sha that does not contain the code that produced it. This cost a repeated
  multi-minute run on the clustering branch.

---

## 8. Risks

| risk | handling |
|---|---|
| **The gold-sentence proxy is invalid** | Leads the write-up as a caveat, not buried in limitations. Plus a small hand spot-check explicitly labelled anecdote — Phase 3B established small samples support *numbers*, not *composition* claims. |
| **LLM run-to-run variance** | `temperature` is not accepted on Opus 5 and adaptive thinking is on by default, so output is not deterministic. **Run the LLM arm twice, log both lines, report the delta as a stability diagnostic** (~$4). Reporting a single LLM number as if reproducible would be the dishonest option. |
| **One prompt prices one prompt** | Explicit limitation, the same shape as ADR-0012's "24% prices *top-1 with one global threshold*, not the best achievable fallback." |
| **Over-selection gaming recall** | Precision primary alongside recall; selection rate logged; no recall quotable without it. |
| **Safety refusals on drug / adverse-event text** | `stop_reason` checked before reading `content`; refusals counted and reported. Opus 5 ships elevated safeguards. **A non-zero rate is a notable operational finding for a biomedical product**, not merely a diagnostic. |
| **Contamination inflating the LLM arm** | Unfixable; documented. Affects the LLM arm only — the controls are unaffected, which is part of why they matter. |

---

## 9. Deliverables

- `biolit/extract/` — `Extractor` protocol, `SameSentenceAsEntitiesExtractor`, `LlmExtractor`
- `Finding` in `biolit/domain/records.py`; `ExtractedRecord.key_findings: list[Finding]`
- `biolit_evals/extract_eval.py` — gold construction, three arms, metrics, buckets, anchors, `main()`
- `evals/extract_runs.jsonl` — **committed**, one line per arm plus the LLM's second variance run
- An `EVAL_REPORT.md` section
- An ADR — the outcome is a build-or-not decision (does the LLM arm earn its place over the free
  control), so it warrants one either way

## 10. Explicitly not in scope

- `study_type`, `sample_size` (§2)
- Alternative window heuristics (same-paragraph, N-token) — bucket (b) prices them; building them is
  a separate decision
- Prompt optimization beyond one reasonable prompt (§8)
- Wiring the Extractor into a LangGraph node — no `langgraph` dependency exists yet; that is
  orchestration work, not eval work
- Any change to clustering, canonicalization, or NER
