# NER Eval Report (Phase 2)

First real run of BioLit Copilot's local biomedical NER layer (`biolit.ner`) against a
real model download and real inference, on both the BC5CDR benchmark and an in-domain
hand-annotated sample. This is a single point-in-time run at commit `3be9b4f`
(`phase-2-ner` branch), produced entirely by the harness in `biolit_evals`.

## Checkpoint

- **Model:** `Francesco-A/BiomedNLP-PubMedBERT-base-uncased-abstract-bc5cdr-ner-v1`
- **Architecture:** `BertForTokenClassification`, base model
  `microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract`
- **Output head (`id2label` from `config.json`):**
  `{0: "O", 1: "B-Chemical", 2: "I-Chemical", 3: "B-Disease", 4: "I-Disease"}`
- **License:** MIT, as stated on the model page (confirmed in Task 1 and re-confirmed
  here — no license change since).
- **Model card's own reported metrics:** F1 0.8775, precision 0.8582, recall 0.8977,
  accuracy 0.9727 (methodology unstated on the card; not assumed to be the same
  scoring method used below — see "Scoring methodology").
- **Model card caveat, verbatim in spirit:** the card describes this checkpoint as an
  **experimental/testing version for educational purposes, not intended for clinical
  or production use.** This applies to every number in this report — see
  "Limitations" #3.

## Headline result: BC5CDR test split

```
bc5cdr: P=0.7703 R=0.8539 F1=0.8099 (tp=8376 fp=2498 fn=1433, n=5865)
```

- **Precision:** 0.7703
- **Recall:** 0.8539
- **F1:** 0.8099
- **n_examples:** 5865 (sentences in the `tner/bc5cdr` `test` split)
- **Counts:** tp=8376, fp=2498, fn=1433

This is the unbiased headline number (see "Limitations" #5). It runs the full,
unmodified BC5CDR test split with no cherry-picking or filtering.

## Supporting cross-check: domain sample

```
domain: P=0.7143 R=0.6111 F1=0.6587 (tp=55 fp=22 fn=35, n=49)
```

- **Precision:** 0.7143
- **Recall:** 0.6111
- **F1:** 0.6587
- **n_examples:** 49 (blind-annotated sentences, see "Domain sample composition" below)
- **Counts:** tp=55, fp=22, fn=35

This number is **not** a second benchmark result of equal standing to BC5CDR — it is a
narrow, single-annotator, single-run sanity check on our own corpus. See "Limitations."

## Interpreting the gap: why 0.66 here vs 0.81 on the benchmark

The obvious reading — "the model is much worse on our corpus" — is not what the errors
actually show. Breaking down all 57 domain errors by whether the predicted and gold
spans overlap in character range (regenerate with
`uv run python -m biolit_evals.error_analysis --dataset domain` from `backend/`; see
`backend/src/biolit_evals/error_analysis.py`):

| | count | of which overlap a span on the other side |
|---|---|---|
| False negatives | 35 | 16 (46%) |
| False positives | 22 | 18 (82%) |
| **Total** | **57** | **34 (60%)** |

**60% of the errors are boundary disagreements on entities the model did detect**, not
failures to find an entity at all. Strict exact-span matching penalises each of these
*twice* — once as a false positive for the wrong span, once as a false negative for the
missed gold span — so a single tokenisation slip costs two errors.

The dominant pattern is **drug-class abbreviations and hyphenated compound names being
split**:

| Gold entity (missed) | What the model predicted instead |
|---|---|
| `CFD` (×10) | `CF` (×4) |
| `GLP-1RAs`, `GLP-1RA` | `GLP` + `1RA` |
| `SGLT2is`, `DPP4is` | partial / no span |
| `sodium-glucose cotransporter-2 inhibitors` | `sodium-glucose` |
| `Cangfudaotan Decoction` | `Cangfudaotan Decoc`, `Cangfudaotan` |

The row below is **not a false negative** — `Lp(a)` was deliberately excluded from
gold (see convention #3), so there is no missed gold span; the model's `Lp` + `a`
prediction is a **pure false positive** with no gold counterpart at all. Listed
separately so the table above stays coherent (every row in it is a genuine
gold-vs-predicted boundary disagreement):

| Predicted (no gold counterpart) | Why it's not a FN |
|---|---|
| `Lp` + `a` | `Lp(a)` is excluded from gold entirely (convention #3) — pure FP |

This is a pattern worth investigating about the model *on this corpus*: BC5CDR is built
largely around individually-named chemicals, while contemporary diabetes and
cardiovascular literature is dense with class-level abbreviations (`GLP-1RAs`,
`SGLT2is`, `DPP4is`) and non-Western therapeutic names. The model's subword tokenizer
fragments exactly these, and `aggregation_strategy="simple"` does not reassemble them.
The pattern is real in the sense that these specific fragmentations are visible in the
predictions — but see the confound below before reading anything quantitative into it.

**Confound: the two corpora are not measured on the same input format.** BC5CDR
examples reach the model as `bio_tags_to_spans` output — pre-tokenized dataset tokens
rejoined with single spaces, so hyphens and punctuation are already whitespace-
separated before the model ever sees the text (e.g. `Famotidine - associated
delirium .`). The domain sample, by contrast, is natural sentence text with hyphens and
punctuation left intact (`GLP-1RAs`, `sodium-glucose cotransporter-2 inhibitors`).
Pre-separating hyphens removes exactly the failure mode this section identifies as the
domain sample's dominant error — a hyphenated compound the tokenizer would otherwise
fragment arrives at the model already split on whitespace, so the model never has to
solve that sub-problem on BC5CDR the way it does on the domain sample. This means an
unknown share of the 0.8099 → 0.6587 drop is a tokenization-format artifact of how each
corpus was fed to the model, not corpus difficulty or annotation-convention mismatch.
**The two F1 numbers are therefore not strictly comparable**, and the boundary-
disagreement breakdown above should be read as evidence the pattern exists, not as a
clean measurement of how much of the gap it explains.

**The honest caveat:** this measurement cannot cleanly separate *model weakness*,
*annotation-convention mismatch*, and the *tokenization-format confound* just described.
Our annotator counted drug-class terms as CHEMICAL; BC5CDR's own guidelines may not, and
the model was fine-tuned to BC5CDR's conventions. With n=49 sentences, 3 abstracts, and
one non-expert annotator, the gap is a signal worth investigating — not a quantified
statement of production accuracy.

**Consequence for later phases (why this matters beyond a score):** Phase 3 clusters
papers by shared entities and Phase 5 detects contradictions within those clusters. If
`GLP-1RAs` fragments into `GLP` and `1RA` inconsistently across papers, entity-keyed
clustering will split papers that belong together, and the failure will surface as
missing clusters rather than as a visible NER error. Entity normalisation — or at
minimum an abbreviation-aware span merge — should be treated as a prerequisite for
clustering, not a later refinement.

## Raw `runs.jsonl` lines

Both runs appended one line each to `backend/evals/runs.jsonl` (append-only log,
never rewritten):

```json
{"timestamp": "2026-07-22T06:48:24.889007+00:00", "model_id": "Francesco-A/BiomedNLP-PubMedBERT-base-uncased-abstract-bc5cdr-ner-v1", "dataset": "domain", "split": "domain", "precision": 0.7142857142857143, "recall": 0.6111111111111112, "f1": 0.6586826347305389, "tp": 55, "fp": 22, "fn": 35, "n_examples": 49, "git_sha": "3be9b4f"}
{"timestamp": "2026-07-22T06:50:15.623867+00:00", "model_id": "Francesco-A/BiomedNLP-PubMedBERT-base-uncased-abstract-bc5cdr-ner-v1", "dataset": "bc5cdr", "split": "test", "precision": 0.7702777266875115, "recall": 0.8539096747884596, "f1": 0.8099405308707633, "tp": 8376, "fp": 2498, "fn": 1433, "n_examples": 5865, "git_sha": "3be9b4f"}
```

## Config used

Ran with the repository's configured defaults, unmodified for this task:
`Settings.ner_score_threshold = 0.5`, `Settings.ner_device = "auto"` (resolved to CPU —
no CUDA device available in this environment), `Settings.ner_batch_size = 16`. No
threshold or setting was retuned to change either result.

## Timings

| Step | Wall clock | Notes |
|---|---|---|
| Heavy smoke test (`test_bc5cdr_smoke.py`, includes first-time model weight download) | 91.8s | `1 passed in 91.80s (0:01:31)` |
| Domain eval (`--dataset domain`, 49 examples, weights already cached) | 5.9s | includes process startup + model load |
| BC5CDR eval (`--dataset bc5cdr`, 5865 examples, weights already cached) | 2m 45.2s | includes dataset download (~5865-row JSON, not the model) + full-split inference on CPU |

All runs were single-process, CPU-only (`torch==2.13.0+cpu`), on the machine used for
this task. No GPU was available or used.

## Dataset tag mapping — confirmed

`biolit_evals/datasets.py::load_bc5cdr_test` hardcodes the **dataset's** tag map
(distinct from the model's `id2label` above — see the in-code comment for why these
must not be conflated):

```python
id2label = {0: "O", 1: "B-Chemical", 2: "B-Disease", 3: "I-Disease", 4: "I-Chemical"}
```

**Confirmed against the real dataset.** `tner/bc5cdr`'s `dataset/label.json` (fetched
directly via `huggingface_hub.hf_hub_download`) is:

```json
{"O": 0, "B-Chemical": 1, "B-Disease": 2, "I-Disease": 3, "I-Chemical": 4}
```

Inverting this gives exactly the hardcoded map above — **it matched, no change to the
map was needed.** Sanity-checked against `ds[0]`: tokens
`['Famotidine', '-', 'associated', 'delirium', '.']`, tags `[1, 0, 0, 2, 0]`, i.e.
"Famotidine" = `B-Chemical`, "delirium" = `B-Disease` — chemically and clinically
correct, consistent with the map.

**A real bug was found and fixed in the process, unrelated to the label values
themselves.** The originally-intended call, `load_dataset("tner/bc5cdr",
split="test")`, fails outright under the pinned `datasets==5.0.0`:

```
RuntimeError: Dataset scripts are no longer supported, but found bc5cdr.py
```

`datasets>=4` dropped support for Hub datasets that ship a Python loading script (as
`tner/bc5cdr` does, via `bc5cdr.py`); it must now be loaded from Hugging Face's
auto-generated script-free parquet mirror instead. The fix pins
`revision="refs/convert/parquet"` on the `load_dataset` call. Verified this mirror is
faithful to the source: `dataset/test.json` (fetched directly) has exactly 5865 lines,
matching `len(ds)` from the parquet-revision load, and the first row is byte-identical
between the two (`{"tags": [1, 0, 0, 2, 0], "tokens": ["Famotidine", "-",
"associated", "delirium", "."]}`). Without this fix, `load_bc5cdr_test()` — and
therefore the entire BC5CDR eval run — could not execute at all; it is not a
cosmetic change.

**The manual-verification comment has been replaced with an enforced, in-code
check.** `datasets.py::_verify_bc5cdr_label_map` fetches only
`dataset/label.json` (a few dozen bytes — no extra dataset download beyond what
`load_bc5cdr_test` already does) via `huggingface_hub.hf_hub_download`, inverts the
hardcoded `id2label`, and raises a `ValueError` naming the expected vs. found mapping
if they ever diverge. This runs automatically on every real `load_bc5cdr_test()` call
(it does not run in the fast/offline test suite, since nothing in that suite calls
`load_bc5cdr_test`). The in-code comment above the map now says the mapping is
verified by this check rather than "NOT automatically verified."

## Scoring methodology

Scoring uses a **custom strict entity-level matcher** in `biolit_evals/scoring.py`: an
entity is a true positive only on exact `(start, end, label)` match, counted as a
**multiset per example** and micro-averaged. It is **not** `seqeval`, and these
numbers have not been cross-validated against a `seqeval` run. Spans are derived from
BC5CDR BIO tags by `bio_tags_to_spans`, which joins tokens with single spaces, so
character offsets are relative to that reconstructed text rather than the original
document.

**This matcher is systematically stricter than seqeval, not just uncross-validated
against it.** seqeval scores at token/tag granularity reconstructed from BIO labels, so
a partial-overlap prediction like `CF` inside gold `CFD` cannot exist as its own scored
unit there — the shared tokens simply contribute to the same tag sequence. Here, spans
are compared as literal character ranges, so `CF` (predicted) vs. `CFD` (gold) is two
distinct spans: a legal false positive for the predicted span *and* a legal false
negative for the missed gold span. Every boundary-disagreement error in "Interpreting
the gap" above is double-counted this way. This means the BC5CDR headline of 0.8099 is
biased **downward** relative to published seqeval-based numbers for comparable
models — the strict character-span matcher is intrinsically harder to score well on
than seqeval's token-tag matching, independent of any real difference in the
predictions.

This is *the same kind of metric* published BC5CDR results report (strict
entity-level micro-F1), so a BC5CDR F1 of 0.8099 landing inside the commonly-cited
~0.80–0.90 range for models of this class is a reasonable sanity check that nothing is
badly broken — it is **not** a seqeval-equivalent or leaderboard-comparable run, and
should not be quoted as directly comparable to any specific published leaderboard
number, including the model card's own self-reported 0.8775 (whose scoring
methodology is undocumented on the card). If anything, given the downward bias just
described, the "same kind of metric" framing is conservative, not generous — this
matcher does not flatter the model relative to seqeval-based comparisons.

**Is 0.8099 plausible?** Yes. It sits inside the ~0.80–0.90 strict entity-level F1
range the task brief cites as expected for models of this class, and close to — a few
points below — the model card's own self-reported 0.8775. Precision (0.7703) trailing
recall (0.8539) is a directionally sensible pattern for the default 0.5 score
threshold: `aggregation_strategy="simple"` in the HF pipeline groups sub-word tokens
into spans somewhat liberally, and a strict `(start, end, label)` matcher penalizes
any boundary token disagreement as a full miss on both sides (one FP on the predicted
span, one FN on the gold span) even when the entity was substantively found — this is
a known property of strict-boundary evaluation, not evidence of a labeling bug. No
diagnosis-worthy anomaly (near-zero or implausibly-high F1) was observed, so per the
task's stop condition, no further investigation was performed.

## Domain sample composition

Source: `backend/evals/gold/domain_sample.jsonl`, rebuilt in Task 7 from **full**
post-truncation-fix abstracts (see "Limitations" #4). Verified directly from the
committed file (not copied from prose in the Task 7 report — see the discrepancy note
below):

- **49 sentences**, drawn from **3 PubMed abstracts**:
  - PMID 42458939 — "Cangfudaotan Decoction Improves Endometrial Receptivity in
    Polycystic Ovary Syndrome Rats..." (PCOS/metformin topic) — 17 sentences
  - PMID 42476899 — "Evaluating elevated lipoprotein(a) clinical care impact in an
    academic health center" (statins/CVD topic) — 15 sentences
  - PMID 42441967 — "Glucagon-Like Peptide-1 Receptor Agonists and Risk for Ischemic
    Optic Neuropathy" (metformin/T2D topic) — 17 sentences
- **90 total entities: CHEMICAL 49, DISEASE 41** (counted directly from the committed
  JSONL). **Note:** the Task 7 report's prose states "CHEMICAL 47, DISEASE 43" for
  the same file/commit (`3be9b4f`) — the total (90) matches but the per-label split
  does not. This report uses the counts computed directly from the committed file,
  which is authoritative; the Task 7 prose total appears to have a transcription
  error in the per-label breakdown. Section distribution (below), independently
  re-verified the same way, matched the Task 7 report exactly.
- **Zero-entity sentences: 19/49 (39%)** — administrative/methodological/demographic
  sentences with no canonical chemical or disease mention.
- **Section distribution** (49 sentences, from structured-abstract `LABEL:` prefixes):

  | Section | Count |
  |---|---|
  | RESULTS | 18 |
  | METHODS | 7 |
  | BACKGROUND | 6 |
  | CONCLUSION | 5 |
  | DISCUSSION | 3 |
  | OBJECTIVE | 2 |
  | MEASUREMENTS | 2 |
  | LIMITATIONS | 2 |
  | DESIGN | 1 |
  | SETTING | 1 |
  | PARTICIPANTS | 1 |
  | PRIMARY FUNDING SOURCE | 1 |

  RESULTS (37%) is the largest single section — the section the pre-fix truncation
  bug made invisible — and CONCLUSION/DISCUSSION are both represented.

### Annotation conventions (applied consistently, from Task 7)

1. Every literal mention is its own span — a spelled-out term and its parenthetical
   abbreviation are two separate entities of the same label (e.g. "Polycystic ovary
   syndrome (PCOS)" → two DISEASE spans).
2. Drug-class terms are CHEMICAL, including bare class-adjective + shared "agents"
   coordination (e.g. "antihypertensive, antiplatelet, or antihyperglycemic agents" →
   three separate CHEMICAL spans, none including the word "agents" itself).
3. Named endogenous hormones/steroids that are simple chemical substances are
   CHEMICAL (testosterone) — but peptide/protein hormones and cytokine/biomarker names
   are excluded as out-of-scope proteins, not chemicals (LH, TNF-α, IL-1β, IL-6, CRP,
   HOXA10, LIF, integrin αvβ3, mTOR, BMI, aOR, HOMA-IR, and — a deliberate, consistent
   call — **lipoprotein(a)/Lp(a)**, treated as a lipoprotein particle/biomarker rather
   than a simple chemical; see "Limitations" #2 for the effect this had on the sample).
4. Dosage/formulation strings are excluded from the entity span — "metformin (50
   mg/kg/d)" tags only "metformin".
5. Chemical/disease names inside trial-arm or drug-combination group labels are
   tagged (e.g. "PCOS-Metformin" yields both a DISEASE and a CHEMICAL span).
6. Named herbal/formula preparations used as a treatment are CHEMICAL (e.g.
   "Cangfudaotan Decoction"/"CFD" and five named docking-study compounds).
7. Adjectival/abbreviated disease forms are tagged even as a modifier (e.g. "IR" for
   insulin resistance, contextually established).
8. Generic, non-named condition/process phrases are excluded even when clearly
   pathological in context (e.g. "hormonal imbalances", "ophthalmic conditions").
9. No overlapping/nested spans — only the maximal phrase is tagged at a given
   occurrence; a shorter phrase is tagged separately only where it recurs as an
   independent, non-nested mention later in the sentence.
10. No coordinated/nested "type 1 and type 2 diabetes"-style mentions occurred in the
    selected sentences.

## Limitations

1. **The domain sample was annotated blind and from scratch per ADR-0006** (no model
   predictions visible during annotation), **by an LLM subagent — not a domain
   expert** — **as a single annotator with no inter-annotator agreement measured.**
   There is no second rater and no adjudication process; every judgment call in the
   "Annotation conventions" above reflects one perspective only.
2. **The domain sample is 49 sentences from only 3 abstracts** — a narrow lexical base
   where one paper's terminology can swing the F1 substantially. Quantifying this: the
   gold file has 90 entities but only **41 distinct surface forms** (counted directly
   from `backend/evals/gold/domain_sample.jsonl`), and `CFD` alone accounts for
   **12 of the 90 entities (13%)**, all from the one abstract that introduces it. Of
   those 12, **10 are false negatives** (the model predicts `CF`, not `CFD` — see
   "Interpreting the gap"), so **10 of the 35 total false negatives (29%) come from a
   single acronym in a single abstract.** Notably,
   `lipoprotein(a)`/`Lp(a)` was **deliberately excluded** from CHEMICAL (judged a
   biomarker/lipoprotein particle rather than a chemical substance — see convention
   #3 above), which zeroed out most of one abstract's entities: 10 of that abstract's
   15 sentences ended up with zero tagged entities. Effectively, **only ~2 of the 3
   abstracts carry most of the entity signal** the domain F1 is computed over.
3. **The checkpoint's own model card describes it as experimental/educational, not
   for clinical or production use.** No claim in this report should be read as
   supporting clinical or production deployment.
4. **The domain sample was rebuilt from full abstracts only after fixing a
   `PubMedClient` bug** (commit `50d59e2`, "fix(pubmed): capture all AbstractText
   sections, not just the first") **that had been truncating structured abstracts to
   their first section.** An earlier domain sample built on the truncated text
   (committed at `0640305`) was discarded and never read or reused during the rebuild
   (Task 7); every number in this report uses the rebuilt, post-fix sample only.
5. **BC5CDR is the unbiased headline number; the domain sample is a supporting
   cross-check only.** BC5CDR is a large (5865-sentence), independently-curated,
   widely-used benchmark with no BioLit-specific selection bias. The domain sample is
   small, single-annotator, narrow in topic, and was built specifically to probe our
   own corpus — it is informative about where the model might struggle on
   BioLit-relevant text, but it is not a substitute benchmark and should never be
   quoted in place of, or averaged with, the BC5CDR number.

## Files changed for this task

- `backend/src/biolit_evals/datasets.py` — dataset-tag-map verification made a real
  in-code check (`_verify_bc5cdr_label_map`); `load_bc5cdr_test` fixed to load from
  `revision="refs/convert/parquet"` (required — the unpinned call fails under
  `datasets>=4`).
- `backend/tests/evals/test_bc5cdr_smoke.py` — new, `@pytest.mark.heavy` real-model
  smoke test.
- `backend/evals/runs.jsonl` — two new lines (domain, bc5cdr), appended.
- `docs/EVAL_REPORT.md` — this file.
- `docs/ARCHITECTURE.md` — "NER layer (Phase 2)" section added.

---

# Entity canonicalization (NEN) — Phase 3, sub-project A

Everything below concerns the **named-entity normalization** layer
(`biolit.canon.canonicalize`), which maps each detected CHEMICAL/DISEASE span to a MeSH
concept ID so clustering can key on a concept rather than a surface string. It is a
separate concern from the NER numbers above, with its own gold, metric, and run log
(`backend/evals/canon_runs.jsonl`).

## Headline: BC5CDR linking (gold MeSH IDs)

| | value |
|---|---|
| Precision | **0.9210** |
| Recall | **0.6828** |
| **F1** | **0.7842** |
| Correct / n | 6635 / 9718 |
| Linked | 7204 |
| **NIL rate** | **0.259** |
| **Ambiguous-tiebreak rate** | **0.003** |

## Supporting cross-check: blind domain sample

| | value |
|---|---|
| Precision | **1.0000** |
| Recall | **0.5942** |
| **F1** | **0.7455** |
| Correct / n | 41 / 69 |
| Linked | 41 |
| **NIL rate** | **0.406** |
| **Ambiguous-tiebreak rate** | **0.043** |

The blind domain gold holds 90 annotated mentions (CHEMICAL 49 / DISEASE 41 — identical
composition to the Phase 2 NER domain sample, confirming offsets were preserved). Of
those, **21 have no correct MeSH concept at all** and are excluded from scoring, leaving
n = 69 linkable mentions.

## What is being measured (and what is not)

Linking is scored on **gold mentions**: the gold surface form is fed directly to the
linker and we check whether the returned concept ID is among the mention's gold MeSH
IDs. This deliberately **isolates linking quality from NER quality** — a linking
regression cannot hide behind an NER change, and vice versa. These numbers therefore say
nothing about end-to-end extraction accuracy; they are a measurement of the normalization
layer alone.

Scoring is restricted to gold mentions that have a real MeSH ID. Mentions annotated as
unlinkable (`-1` in BC5CDR, `-1` in the domain sample) are excluded from precision/recall
rather than counted as "correct NILs", which would inflate the score.

Two further scoring conventions worth stating plainly. **Composite mentions are scored
leniently:** where BC5CDR gold carries several IDs for one mention (`D1|D2`), a link to
*either* counts as correct. **`nil_rate` is a share of linkable gold**, not of all
mentions — its denominator is the same `n` as recall's.

## Interpreting the result: the failure mode is abstention, not error

Both datasets show the same shape, and it is the actionable finding:

- **Precision is high** (0.92 benchmark, 1.00 domain). When the dictionary linker commits
  to a concept it is usually right — though "usually" is not "always": on BC5CDR it linked
  7204 mentions and got 6635 correct, so **569 links (8% of its commitments) are wrong.**
  On the domain sample all 41 links happened to be correct, but that is 41 links from 3
  abstracts and should not be read as "the linker never errs".
- **NIL dominates the recall gap** (25.9% benchmark, 40.6% domain — as a share of
  *linkable* gold mentions): surfaces absent from the CTD alias table after
  case/whitespace normalization. Of BC5CDR's 3083 non-correct mentions, 2514 (~81%) are
  NIL and 569 (~19%) are wrong links.

So the linker's dominant failure mode is abstention rather than error — but not
exclusively. That answers the
question ADR-0008 deferred: **an embedding-based fallback (e.g. SapBERT) should be aimed
at unmatched surfaces, not at correcting wrong links**, and the `Linker` protocol seam
exists precisely so one can be added behind `DictionaryLinker` without a redesign. The
higher domain NIL rate (40.6% vs 25.9%) is consistent with our corpus being dense with
drug-class abbreviations and non-Western therapeutic names that CTD's alias list does not
carry — the same lexical territory the Phase 2 error analysis flagged.

The **ambiguous-alias tiebreak rate is logged as its own metric** rather than folded into
accuracy, because a silent arbitrary resolution on a common surface would otherwise be
invisible inside the aggregate. It fires on 0.3% of benchmark mentions and 4.3% of domain
mentions (3 of 69). That **bounds** how much damage the deterministic rule (exact-name over
synonym, then lexicographically smallest ID) can do — it cannot account for more than 0.3%
of benchmark outcomes — but it is a bound, not a measurement: the scorer counts `tiebroken`
and `correct` independently and never crosses them, so the code cannot currently say
whether tiebroken links were right. (On the domain sample it is implied: precision is
1.0000, so all 3 were correct. On BC5CDR's 31, unknown.) If that rate climbs, the rule
needs revisiting — and the scorer should then also report `tiebroken_correct`.

## Blind annotation methodology (ADR-0006)

The domain normalization gold was annotated **blind**, which here means **not anchored to
`DictionaryLinker`'s own output** — the annotator never ran the linker, never loaded the
CTD artifact, and never saw a system prediction. It does **not** mean annotating with zero
reference tools: the annotator **consulted the official NLM MeSH lookup API directly** to
find and verify concept IDs, and recorded `-1` wherever no MeSH concept could be verified.
Every assigned ID was confirmed against that API rather than recalled from memory. Note
that the gold includes three Supplementary Concept Records (`C541528`, `C047353`,
`C047331`), which are resolved through MeSH's supplementary-concept lookup rather than the
descriptor endpoint — so "the MeSH lookup API" here means the descriptor *and*
supplementary-record paths, not the descriptor endpoint alone. This distinction matters —
the point of blindness is to avoid the anchoring bias of reviewing a pre-populated
candidate list, not to force unaided guessing.

## Reproducing these numbers

The MeSH alias artifact and the BC5CDR gold corpus are **downloaded, not committed**
(they are gitignored). Rebuild and re-run from `backend/`:

```
uv run python -m biolit.canon.build_mesh              # CTD -> data/canon/mesh_aliases.json.gz
uv run python -m biolit_evals.canon_eval --dataset bc5cdr
uv run python -m biolit_evals.canon_eval --dataset domain
```

The artifact built for this report contains **551,669 aliases** from the CTD chemical and
disease vocabularies.

Re-running both evals after the final review fixes reproduced the numbers exactly, and the
run log now records the alias count of the artifact that was scored:

```json
{"timestamp": "2026-07-24T00:43:29.490329+00:00", "git_sha": "f4ffee0", "dataset": "domain", "artifact_source": "data/canon/mesh_aliases.json.gz", "n_aliases": 551669, "n": 69, "correct": 41, "linked": 41, "precision": 1.0, "recall": 0.5942028985507246, "f1": 0.7454545454545455, "nil_rate": 0.4057971014492754, "tiebreak_rate": 0.043478260869565216}
{"timestamp": "2026-07-24T00:43:34.388943+00:00", "git_sha": "f4ffee0", "dataset": "bc5cdr", "artifact_source": "data/canon/mesh_aliases.json.gz", "n_aliases": 551669, "n": 9718, "correct": 6635, "linked": 7204, "precision": 0.9210161021654636, "recall": 0.6827536530150237, "f1": 0.7841862663987708, "nil_rate": 0.25869520477464497, "tiebreak_rate": 0.003189956781230706}
```

`n_aliases` exists because `artifact_source` alone is a constant path string — identical on
the invalid pre-fix line below and the good ones. The alias count is the cheapest available
detector of "the artifact was rebuilt wrong again".

## A defect this eval caught (and why the first log line is wrong)

The **first** `bc5cdr` line in `backend/evals/canon_runs.jsonl` (git_sha `0bfb783`) reads
P=0.4211 / R=0.2695 / F1=0.3287. **That run is invalid** and is retained only because the
run log is append-only history.

The first real CTD build silently produced a broken alias table: chemical IDs came out
**double-prefixed** (`MESH:MESH:D008687`) and chemical synonyms were read from the
`Definition` column. Root cause: the unit-test fixtures had been written against an
outdated, simplified CTD schema, so positional column parsing passed every offline test
while misreading the real files. The real `CTD_chemicals.tsv` carries an
already-prefixed `ChemicalID` and splits synonyms into `MESHSynonyms` / `CTDCuratedSynonyms`.
Every chemical link therefore failed to match gold.

It was caught by probing the real artifact with known terms before trusting the number,
not by the test suite. The fix (`8b49f30`) resolves CTD columns **by name from the file
header** instead of by fixed index, so a future CTD column addition cannot silently
misread ids or synonyms again; the fixtures were rewritten to the real schema. The alias
table grew from 275,451 to 551,669 entries, and F1 went from 0.3287 to 0.7842. This is
recorded rather than quietly amended because it is the clearest example in this project of
a green test suite coexisting with a wrong result.

## Limitations

1. **Linking is scored on gold mentions only.** These numbers do not measure end-to-end
   (NER → canonicalization) performance, and must not be quoted as such.
2. **The fragment-merge is not measured by these numbers at all.** The scorer calls
   `Linker.link()` on gold surface forms directly; it never calls `canonicalize()`, so
   neither `merge_fragments` nor the merge-then-link orchestration contributes anything to
   the reported 0.7842 / 0.7455. That matters because `merge_fragments` — the direct fix
   for the ADR-0008 fragmentation finding, and the most heuristic component here — is
   currently backed by **unit tests only**. Its merge rule is deliberately permissive (a
   connector-joined run merges when *any* token is fragment-shaped), so combination
   notation like `ASA/clopidogrel` will also merge; `canonicalize` therefore applies a
   merged concept only to constituents that did not link individually, so a spurious merge
   can fill a gap but never overwrite a correct link. An eval that exercises the merge path
   end-to-end is outstanding work.
3. **The domain sample is small and single-annotator.** 69 linkable mentions drawn from
   3 abstracts, annotated by an LLM subagent — API-grounded, but **not a domain expert**,
   with no inter-annotator agreement. The domain precision of 1.0000 rests on only 41
   links and should not be read as "the linker is perfect".
4. **BC5CDR is the unbiased headline; the domain sample is a supporting cross-check**
   only, exactly as with the Phase 2 NER numbers.
5. **CTD is a moving target.** The alias table is built from live CTD downloads, so
   re-running later may shift these numbers as CTD adds concepts and synonyms. The run log
   records the git SHA but not a CTD release version.
6. **`CTDCuratedSynonyms` is deliberately excluded** from the alias table (it mixes CAS
   numbers and non-MeSH variants). Including it might lift recall at some cost to
   precision; this has not been measured.
7. **Normalization is light** (casefold + whitespace collapse). No plural stripping,
   punctuation folding, or abbreviation expansion is applied before lookup — a
   deliberate v1 choice, and part of why the NIL rate is what it is.

---

# End-to-end canonicalization (Phase 3B)

Everything above measures linking with **gold** surface forms fed to the linker. This
section measures the **production path** for the first time:
`extract_entities` → `canonicalize` (merge + link) → `canonical_id`, scored against gold
MeSH IDs. It is the only eval that exercises `merge_fragments` at all.

Run log: `backend/evals/e2e_runs.jsonl`. Both corpora run on **natural text** (BC5CDR from
`CDR_Data.zip`, not the space-joined `tner/bc5cdr` tokens), so the Phase 2 tokenization
confound does not apply here.

## Concept-level results

The primary metric is **concept-level P/R/F1**: per document, the set of predicted
`canonical_id`s against the set of gold MeSH IDs. This is what clustering consumes — a
paper's concepts, not its character spans. Averaging is **micro** (tp/fp/fn summed across
all documents, then P/R/F1 computed once from those totals), matching the Phase 2 scorer's
discipline. Within a document the ids are **sets**, so a concept mentioned five times
counts once and cannot dominate the corpus totals.

| | P | R | **F1** | tp / fp / fn | units scored | e2e NIL |
|---|---|---|---|---|---|---|
| **BC5CDR** | 0.8822 | 0.6826 | **0.7697** | 2336 / 312 / 1086 | 500 abstracts | 0.320 |
| Domain sample | 0.7838 | 0.5000 | **0.6105** | 29 / 8 / 29 | 49 **sentences** (3 abstracts) | 0.455 |

**The two rows are not on the same unit of analysis.** BC5CDR scores whole abstracts. The
domain gold is annotated per sentence, so each *sentence* is scored as a document — 49
units drawn from only 3 abstracts. That inflates the concept denominator relative to
abstract-level scoring (58 gold concept slots at sentence granularity versus 26 at abstract
granularity) and partly defeats the set-semantics rationale below: a concept appearing in
five sentences of one abstract counts five times here, not once. Read the domain row as a
sentence-level sanity check, never as an abstract-level result comparable to BC5CDR.

Per label:

| Corpus | Label | P | R | F1 |
|---|---|---|---|---|
| BC5CDR | CHEMICAL | 0.877 | 0.801 | **0.837** |
| BC5CDR | DISEASE | 0.886 | 0.597 | **0.713** |
| Domain | CHEMICAL | 0.588 | 0.333 | **0.426** |
| Domain | DISEASE | 0.950 | 0.679 | **0.792** |

**End-to-end NIL vs gold-surface NIL.** e2e NIL (0.320 BC5CDR / 0.455 domain) sits above
the gold-surface NIL reported earlier (0.259 / 0.406), which is expected: it is computed
over *model predictions*, including imperfect spans, while gold-surface NIL is computed
over *gold mentions*. **The two have different populations and denominators — they are
comparable in direction only and must never be subtracted.**

## The outcome census

Every gold mention classified by how NER covered it. This is the categorical breakdown of
the number above; it accounts for every point of loss.

**BC5CDR (9809 gold mentions):**

| Outcome | Count | Share |
|---|---|---|
| `EXACT` | 8052 | 82.1% |
| `MISSED` | 782 | 8.0% |
| `TRUNCATED` | 715 | 7.3% |
| `MERGEABLE` | 260 | 2.7% |

Truncation kind: `INTERIOR_OR_OTHER` 276 (38.6%), `PREFIX_OF_GOLD` 238 (33.3%),
`SUFFIX_OF_GOLD` 201 (28.1%).

Per label — `EXACT` share: **CHEMICAL 4594/5385 = 85.3%**, **DISEASE 3458/4424 = 78.2%**.
DISEASE carries 2.5× more truncation (509 vs 206); CHEMICAL carries more mergeable
fragmentation (181 vs 79).

**Validation:** the census totals reproduce the canonical BC5CDR counts exactly — 9809
mentions, **5385 chemical, 4424 disease**. The 9809 total *is* asserted by the heavy smoke
test, so it is a guard rather than an independent check; the **per-label split is not
pinned anywhere**, and a parsing, alignment, or double-counting bug would have shifted it.

**Domain sample (90 gold mentions):** `EXACT` 55, `MISSED` 22, `TRUNCATED` 11,
`MERGEABLE` 2; truncation `PREFIX_OF_GOLD` 9, `SUFFIX_OF_GOLD` 2.

## What this changes: the 49-sentence probe was wrong on all three counts

This eval was scoped because a 49-sentence probe suggested `merge_fragments` was inert and
truncation dominated. **At full scale, all three of the probe's conclusions fail to hold.**

1. **Merging is not inert — but its effect is small.** Probe: 2 candidates, 0 useful. At
   scale: 122 candidates proposed, 74 exactly reconstruct a gold span, 52 link, and 91
   entity constituents received a concept they would not otherwise have had. 260 gold
   mentions (2.7%) sit in the mergeable shape.

   Those counts describe activity, not benefit, so they were checked against an **ablation**
   — the same corpus scored with merged-candidate gap-filling disabled:

   | | P | R | F1 | tp | fp |
   |---|---|---|---|---|---|
   | With merging | 0.8822 | 0.6826 | **0.7697** | 2336 | 312 |
   | Without merging | 0.8820 | 0.6794 | **0.7676** | 2325 | 311 |
   | **Delta** | +0.0002 | +0.0032 | **+0.0021** | **+11** | +1 |

   So merging is worth **+0.0021 F1 — 11 additional correct concepts out of 3422 gold
   slots**, at the cost of one extra false positive. Real, positive, and much smaller than
   the raw audit counts suggest: 91 constituents gaining a concept converts to only 11
   concept-level gains, because document-level set semantics collapse constituents whose
   concept was already found elsewhere in the same document. The correct reading is
   "merging is not inert and is not harmful", **not** "merging is important".

   On the domain corpus the probe's result is unchanged in this branch's own run
   (`candidates=2, linked=0, matching_gold=0`); the reversal is a BC5CDR-scale finding only.
2. **Truncation is not systematically suffix-dropping.** The domain sample's 9-of-11
   `PREFIX_OF_GOLD` looked like a clean signal; at scale the three kinds are far more even
   (38.6% / 33.3% / 28.1%), so there is no single dominant boundary-error mode to target.

   **Caveat on that split:** `INTERIOR_OR_OTHER` — the largest bucket — is a catch-all that
   also holds predictions extending *past* gold (an over-extension, recorded with a
   negative `char_delta`), not only interior truncations. So the honest claim is that the
   clean prefix-dropping pattern seen in the domain sample **does not survive at scale**;
   the precise composition of the remaining 38.6% is not separated by the current census.
   `char_delta` is computed per mention but not yet aggregated, which is what would split
   it — see Limitations.
3. **The label asymmetry reverses.** The domain sample showed CHEMICAL as the problem
   (39% exact vs DISEASE 88%). At scale **CHEMICAL is the stronger label** (85.3% vs 78.2%
   exact), and DISEASE truncates 2.5× more often.

The domain sample retains its value as an in-domain cross-check, but with 90 mentions from
3 abstracts it cannot support conclusions about error *composition* — every apparent
pattern in it was reversed or flattened by the full corpus. That is the finding.

The merge audit is reported as **raw counts, not precision/recall**: a rate over 122
candidates invites over-reading. The question those counts cannot answer — whether the
recovered concepts are *correct* — is answered by the ablation above, not by the audit.

**Windowing caveat on the audit.** Long-document windowing (below) was introduced in the
same branch as this measurement, and windowing can in principle create span shapes that
generate extra merge candidates. Attributing candidates to their source documents:
**88 of the 122 (72%) come from documents that were never windowed** and therefore cannot
be windowing artifacts; 34 come from the 32 windowed documents. This was measured: before
nested-span de-duplication was added the audit reported 129 candidates, so 7 of those were
windowing artifacts and are now gone. The finding survives, but the audit counts are not
windowing-independent.

## A second production bug this eval caught

The first full-scale run **crashed**:

```
RuntimeError: The size of tensor a (549) must match the size of tensor b (512)
```

This checkpoint's tokenizer declares no `model_max_length`, so it never truncates, and
BERT-family position embeddings cap input at 512 tokens. **11 of the 500 BC5CDR abstracts
(2.2%) exceed it** (median 267 tokens, max 722). It had never surfaced because every prior
measurement — Phase 2 NER, Phase 3A linking, the domain sample — ran on *sentences*. Phase
4's Extractor feeds whole abstracts, so this would have crashed on roughly 2% of real
papers in production.

Fixed in `fce5fa6` by windowing inside `NerModel`, not in the eval: fixing it only in the
eval would have made this report claim success against a code path production does not
run. Windows are packed from whole sentences (an entity practically never spans a sentence
boundary), consecutive windows overlap by one sentence so a mis-placed boundary cannot cut
an entity in half, and offsets are shifted back into document coordinates and
de-duplicated. The overlap guarantee is pinned as a property — every internal window
boundary is spanned by some other window — with a paired test showing that guarantee
genuinely fails when overlap is disabled. The heavy regression test asserts more than "no
crash": entities must be recovered from text **beyond** the old 512-token cutoff, with
every offset still slicing correctly out of the document.

Like the CTD schema defect above, this is recorded rather than quietly fixed: both are
cases where a green test suite coexisted with broken behavior on real input, and both were
caught only by running real data end to end.

## Limitations

1. **Concept-level scoring is document-level set matching.** It rewards finding *a*
   mention of the right concept and is blind to how many times or how precisely it was
   found. That is the right metric for clustering, and the wrong one for judging span
   quality — the census is there for that.
2. **The domain sample is too small for composition claims** (90 mentions, 3 abstracts,
   single non-expert annotator). See above: every error-composition pattern it showed was
   contradicted at scale. Treat its *numbers* as a cross-check and its *breakdowns* as
   anecdote.
3. **`MERGEABLE` counts opportunity, not success.** It marks gold mentions where ≥2
   predictions overlap, i.e. where merging *could* help; whether the merged surface then
   links is a separate question, answered by the merge audit.
4. **Windowing changes NER behavior on long documents**, so end-to-end numbers here are
   not directly comparable to the Phase 2 sentence-level NER F1.
5. **These are single-run point measurements** against a CTD artifact built on one day
   (551,669 aliases); CTD is a moving target and the run log records the alias count but no
   CTD release version.
6. **The domain row scores sentences, not abstracts** (see the note under the results
   table). Its 49 units come from 3 abstracts, so it is not on the same unit of analysis as
   the BC5CDR row and the two F1 values should not be compared directly.
7. **`INTERIOR_OR_OTHER` is a catch-all.** It holds interior truncations *and*
   over-extensions (predictions longer than gold, recorded with a negative `char_delta`).
   `char_delta` is computed per mention but not aggregated into the census or the run log,
   so the report cannot presently say what share of that 38.7% is over-extension. Splitting
   it is the obvious next refinement of the census.
8. **Pooled concept metrics are label-blind.** The pooled sets are built from concept ids
   without regard to label, so a CHEMICAL prediction carrying an id that gold annotated as
   DISEASE still scores as a true positive (visible in the run log: pooled fp=312 against
   per-label fps summing to 314). Defensible for clustering, which keys on concepts rather
   than types, but it means the headline F1 does not penalize label confusion. The
   per-label rows do.
9. **Windowing statistics are not recorded in the run log.** "11 of 500 documents exceed
   512 tokens; 32 exceed the 450-token window budget" comes from ad-hoc measurement, not
   from a logged field, so it is not self-verifying on a future run.
