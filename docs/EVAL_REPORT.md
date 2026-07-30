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

---

## ⭐ Headline finding of the canonicalization phase

**The dominant recoverable loss is missing dictionary aliases on spans NER already got
exactly right — worth up to +0.0821 F1, and it is overwhelmingly a DISEASE problem.**

Of 9809 gold mentions, 8052 were spanned exactly by NER. **1679 of those still linked to
nothing** because the CTD-derived alias table had no entry for the surface. Granting each
its gold id — a perfect fallback over exactly that population — moves concept F1
**0.7697 → 0.8518**, and **DISEASE takes 84.6% of the gain**.

**For scale, against every other canonicalization mechanism this project measured:**

| Mechanism | Concept F1 delta | tp gained |
|---|---|---|
| **Alias gap on exact spans** (ceiling) | **+0.0821** | **+434** |
| Fragment merging (`merge_fragments`, ADR-0008) | +0.0021 | +11 |

**That is a 39× difference.** The alias gap closes **35.7% of the entire remaining gap to a
perfect score**; merging closes 0.9%. Phase 3A's ADR-0008 work was aimed at the smaller of
the two by a wide margin — the *measurement* is what revealed which mechanism mattered, and
neither the gold-surface linking eval nor the span census could have shown it alone.

**This is a ceiling, not a forecast**, and is flagged as such at every citation. It assumes
perfect precision over the 1679; a real linker pays false positives the oracle never incurs.
The realized gain is unmeasured and will be smaller — possibly much smaller.

Full derivation, per-label breakdown, and caveats: [The oracle ceiling](#the-oracle-ceiling-what-a-perfect-fallback-would-be-worth).

---

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

## Linking on exact spans: sizing what a fallback linker could actually reach

The census stops at span quality. Its largest cell — `EXACT`, 8052 mentions — says NER got
the span exactly right, but says nothing about whether linking then succeeded. That cell is
where the classic *"correct span, missing alias"* failure lives, and it is the only place a
fallback linker could operate: `MISSED` has no span to link, and `TRUNCATED`/`MERGEABLE`
reach the linker damaged.

So the `EXACT` cell is partitioned by what linking did with it. Four outcomes, not two —
"did not produce the right id" conflates three different problems:

**BC5CDR, all 8052 exact-span gold mentions:**

| Outcome | Count | Share | Reachable by a fallback linker? |
|---|---|---|---|
| `LINKED_CORRECT` | 5832 | 72.4% | — already correct |
| `NIL` | **1679** | **20.9%** | **Yes — this is the addressable population** |
| `LINKED_WRONG` | 488 | 6.1% | **No** — the dictionary answered confidently and wrongly; a fallback is never consulted |
| `GOLD_UNLINKABLE` | 53 | 0.7% | No — gold is annotated `-1`; abstaining is the *correct* answer |

Separating `GOLD_UNLINKABLE` and `LINKED_WRONG` out matters: lumping them into "not linked"
would put the addressable figure at 2220 rather than 1679 — a **32% overstatement** of what
a fallback could reach.

**The original abstention finding survives, re-scoped.** On gold surfaces (Phase 3A) NIL was
~81% of the recall gap. On *predicted* exact spans it is 1679 of 2167 non-correct linkable
outcomes = **77.5%**. Abstention still dominates error; that conclusion was narrow, not
wrong. (The rates themselves — 21.0% NIL here on linkable exact spans vs 25.9% on all gold
surfaces — are **not comparable**: different populations. Mentions whose span NER nails are
plausibly the more canonical surfaces, so a lower NIL rate is the expected direction, but
this run does not test that.)

### The asymmetry: this is a DISEASE problem

| Label | Exact spans | `LINKED_CORRECT` | `NIL` | `LINKED_WRONG` | `GOLD_UNLINKABLE` | NIL rate (linkable) |
|---|---|---|---|---|---|---|
| CHEMICAL | 4594 | 3580 | 680 | 314 | 20 | **14.9%** |
| DISEASE | 3458 | 2252 | **999** | 174 | 33 | **29.2%** |

**DISEASE abstains on perfect spans at 1.96× the CHEMICAL rate**, and supplies **59.5% of
the entire addressable population (999 of 1679)** despite contributing fewer exact spans in
the first place. The two stages compound in the same direction: DISEASE is already the
weaker label at NER (78.2% exact vs 85.3%), and is then the weaker label at linking.

This is the finding that bears on *what kind* of work is worth scoping. A general embedding
fallback treats both labels alike; a nearly 2× concentration in one entity type is instead
consistent with **thinner disease-side alias coverage in CTD** — which targeted dictionary
expansion would address more cheaply and more verifiably than a fallback model. **That
hypothesis is not tested here.** The direct check is to count aliases contributed per source
file when building the artifact, which the builder does not currently record; the current
551,669 aliases are pooled and `MeshConcept` carries no source label.

### Sizing caveat: 1679 mentions is not 1679 concepts

Against the full 9809-mention corpus: `EXACT`-but-`NIL` is **17.1%** of all gold mentions —
the **largest single recoverable bucket**, larger than the detection gap (`MISSED`, 8.0%)
and larger than span damage (`TRUNCATED` + `MERGEABLE`, 9.9%). It is, however, slightly
*smaller* than those two combined (1679 vs 1757).

**These are mention counts, and the headline metric is document-level concept sets.** This
report has already been burned by exactly that conversion once: the merge audit's 91
constituents gaining a concept converted to **11** concept-level gains, because set
semantics collapse concepts already found elsewhere in the same document. The same collapse
applies here and its magnitude is unknown, so **no claim is made about what fixing these
1679 mentions would be worth in F1.**

### The oracle ceiling: what a perfect fallback would be worth

Re-scoring the same predictions while granting each of those 1679 mentions its gold id — a
fallback with **perfect recall and perfect precision** over exactly that population, and
nothing else. `LINKED_WRONG` is deliberately **not** granted: a fallback is never consulted
when the dictionary already answered, so granting those would measure perfect *linking*, not
a perfect *fallback*. Computed in the same pass as the baseline, so both come from identical
predictions.

| | P | R | F1 | tp | fp | fn |
|---|---|---|---|---|---|---|
| Baseline | 0.8822 | 0.6826 | **0.7697** | 2336 | 312 | 1086 |
| Oracle (perfect fallback) | 0.8988 | 0.8095 | **0.8518** | 2770 | 312 | 652 |
| **Delta** | +0.0166 | +0.1269 | **+0.0821** | **+434** | **0** | −434 |

`n_granted` = 1679, exactly the audit's `NIL` count — the two reconcile. **`fp` is
unchanged**, as the construction requires: granted ids are gold ids, so the ceiling moves
recall only and cannot manufacture precision.

**+0.0821 F1 — roughly 39× the merge ablation's +0.0021.** It closes **35.7% of the entire
remaining gap to a perfect score**. For scale, this single population is worth more than
every other canonicalization mechanism measured in this project combined.

The mention→concept collapse this section warned about is real but mild: **1679 mentions
convert to 434 concept-level gains (25.8%)**, a 3.9× shrink rather than the merge audit's
8.3×. So the caution was warranted — quoting 1679 as the opportunity would have overstated
it fourfold — but unlike the merge case the population survives the conversion at a size
that still matters.

### The ceiling is overwhelmingly a DISEASE result

| Label | Baseline F1 | Oracle F1 | Delta | tp gained | Baseline R → Oracle R |
|---|---|---|---|---|---|
| CHEMICAL | 0.8375 | 0.8652 | **+0.0277** | +67 | 0.8013 → 0.8480 |
| DISEASE | 0.7133 | 0.8411 | **+0.1278** | **+367** | 0.5971 → 0.7817 |

**DISEASE captures 367 of the 434 concept gains — 84.6% — and its F1 moves 4.6× further
than CHEMICAL's.** Note this is *more* concentrated than the mention-level split predicted
(DISEASE was 59.5% of the addressable mentions but 84.6% of the realized concept gains). The
amplification is consistent with chemical mentions repeating within an abstract, so a
chemical missed on one mention is more often recovered from another, while disease mentions
are more varied — but that mechanism is inferred here, not measured.

DISEASE would go from the clearly weaker label (0.7133) to near parity with CHEMICAL
(0.8411 vs 0.8652), i.e. this population accounts for most of the label gap in the headline
number.

**One fact appears to narrow the design space:** every one of these 1679 mentions has a
MeSH concept *by definition* — the gold assigns one. Nothing is missing from MeSH; what is
missing is the **surface→concept alias** in the CTD-derived table. That looks like something
a richer alias source (MeSH's own entry terms, or UMLS) supplies directly, without a model.

> ⚠️ **That inference was tested and is wrong.** See
> [Testing the richer-dictionary hypothesis](#testing-the-richer-dictionary-hypothesis-a-null-result)
> below: MeSH's own entry terms close **0.1%** of the gap. "The concept exists in MeSH" does
> **not** imply "a fuller alias list reaches it", because the missing surfaces are
> abbreviations and paraphrases rather than absent synonyms. The paragraph above is kept as
> written because it is the reasoning the next measurement overturned.

### What this does and does not license

- It is a **ceiling**, not a forecast. A real fallback resolves *some* fraction of the 1679
  and pays a precision cost on the rest; every wrong link adds an `fp` the oracle never
  incurs. The realized gain will be a fraction of +0.0821 and could be much smaller.
- It does not compare the two candidate interventions. It bounds the *population* both would
  target; it says nothing about which reaches more of it, or at what cost.
- Single corpus, single run. The domain sample was not scored against the oracle.

**Nothing here decides the SapBERT fallback.** It sizes the population, locates it in
DISEASE, and establishes that the population is large enough for the question to be worth
asking — which the merge ablation's +0.0021 established was *not* true of merging.

## Testing the richer-dictionary hypothesis: a null result

The section above reasoned that because every addressable mention *has* a MeSH concept, the
gap must be one of **alias coverage** — fixable by a fuller, still-deterministic dictionary,
with no model. That reasoning was explicit, it was quoted forward into planning, and
**measurement contradicts it.**

MeSH's own descriptor and supplemental-record entry terms were ingested directly from the
2026 dumps (`desc2026` + `supp2026`), filtered to the disease and chemical branches
(descriptor trees `C*`, `F03*`, `D*`; supplemental classes 1 and 3), and unioned with the
CTD table. Measured model-free over all 9718 linkable gold mentions — valid because an
`EXACT` span means the predicted surface *is* the gold surface, so the addressable
population is a subset of gold surfaces the CTD table fails to link:

| | Count | Share of the gap |
|---|---|---|
| CTD-NIL population (the alias gap) | 2514 | 100% |
| **Rescued correctly by MeSH entry terms** | **3** | **0.1%** |
| Rescued but linked wrongly | 1 | 0.0% |
| Still NIL | 2510 | 99.8% |

**A richer MeSH-derived dictionary closes essentially none of the gap.** Previously-correct
CTD links broken by the union: **0** — so the union is harmless, just useless.

The reason is visible in the alias counts: CTD contributes 551,669 aliases, MeSH 784,332,
and their union only 819,422 — an overlap of ~517k. **CTD's `MESHSynonyms` column already
*is* MeSH's entry terms.** Enriching CTD from MeSH re-adds what it already had.

**Validation that this is a real null and not a broken ingest:** the ingest yields 271,668
distinct concept ids, and for **2448 of the 2510** surfaces that remain NIL, the gold
concept **is present** in the MeSH table — only the surface is missing. A parsing failure
would have shown the concepts absent. (62, or 2.5%, are genuinely absent — mostly
supplemental records and a few descriptors outside the kept trees, e.g. `blood urea
nitrogen`; a small known cost of the scope filter.)

### What the missing surfaces actually are

Classifying the 2510, with naive punctuation-folding and depluralization as the
"better normalization" test:

| Category | Count | Share | Tractable by |
|---|---|---|---|
| Deterministic normalization (punctuation, plurals) | 50 | 2.0% | trivial code |
| **Abbreviation-shaped** (`TdP`, `DOX`, `ALF`, `SSc`) | 1024 | 40.8% | **in-document abbreviation expansion** |
| Paraphrase / other | 1436 | 57.2% | embedding similarity |

Real examples of why a dictionary cannot win: `TdP` → *Torsades de Pointes*, `NO` →
*Nitric Oxide*, `cognitive deficits` → *Cognition Disorders*, `hepatic injury` →
*Chemical and Drug Induced Liver Injury*, `idiopathic cardiomyopathy` →
*Cardiomyopathy, Dilated*. These are not missing synonyms; they are **author abbreviations
and semantic paraphrases**. No controlled vocabulary enumerates them, which is why adding
more vocabulary changed nothing.

**The abbreviation share is the actionable surprise.** ~41% (an undercount — lowercase forms
like `bort`, `dex` fall into the "paraphrase" bucket) are abbreviations that these abstracts
typically *define on first use* — "torsades de pointes (TdP)". That is addressable by
in-document abbreviation expansion, a deterministic, well-established technique, **not** by
an embedding model. It is a third option that neither the earlier framing nor the fallback
proposal contained.

**Revised reading:** this is **not** a "richer alias source" problem. It splits into a
sizeable deterministic sub-problem (abbreviations) and a genuine
"does-a-model-generalize" sub-problem (paraphrase). The earlier inference from "the concept
exists in MeSH" was wrong, and is corrected rather than quietly dropped.

**Caveat on population:** these shares are measured over the 2514 CTD-NIL *gold-surface*
mentions, a superset proxy for the 1679 `EXACT`-but-`NIL` mentions. The surfaces are the
same kind and the direction is not in doubt, but the exact percentages are not the
end-to-end population's.

**Not run end to end, deliberately:** with 3 of 2514 surfaces rescued, the enriched
dictionary's concept-level effect is bounded at approximately zero, so a model run was not
spent to confirm a null. That is a stated decision, not an omission.

### Splitting the ceiling: abbreviations are nearly worthless at concept level *for clustering*

> **Read the scope before the number.** Everything in this section is measured against
> **document-level concept-set scoring** — the metric clustering consumes. It establishes
> that abbreviation expansion is not worth building **for clustering**. It does **not**
> establish that abbreviation expansion isn't worth building, and must not be cited that
> way. The entire reason the number is small is a property of the *metric*, not of the
> mechanism: see [why](#why-the-slice-vanishes) below. Against a **mention-level** metric
> the same 543 mentions are 543 real corrections.

The classification above suggested ~41% of the gap was abbreviations, reachable
deterministically. That is a **mention-level** share, so it was measured against the metric
instead — the same oracle-ablation pattern, granting gold ids only to the subset a
deterministic mechanism would reach.

"Abbreviation-addressable" is defined by whether the mechanism *actually works*: the surface
is defined in that document as `long form (SHORT)`, **and** expanding it links to the gold
concept. Not a shape heuristic — that would count abbreviations whose long form the
dictionary cannot resolve either.

| Slice | Mentions | Share | tp gained | F1 delta |
|---|---|---|---|---|
| **All** `EXACT`-but-`NIL` | 1679 | 100% | +434 | **+0.0821** |
| Abbreviation-addressable | 543 | 32.3% | **+4** | **+0.0008** |
| Paraphrase / other | 1136 | 67.7% | +431 | **+0.0816** |

**A third of the population is worth 1% of the value.** In-document abbreviation expansion —
the deterministic mechanism that looked like the cheap win — is worth **+0.0008 F1**.
Per label it is +0 concepts for CHEMICAL and +4 for DISEASE.

#### Why the slice vanishes

**The mechanism is obvious in hindsight and worth stating, because it generalizes.** An
abbreviation that is *defined in the document* has its **long form in that same document**
— that is what "defined" means. The long form is itself a mention, and the dictionary
usually links it. So the concept is *already in the document's predicted set*, and granting
the abbreviation mention adds nothing the document did not already have. Document-level set
semantics absorb the entire slice.

Paraphrase surfaces behave the opposite way: `cognitive deficits` or `hepatic injury` are
often the *only* way that concept appears in the abstract, so recovering them adds genuinely
new concepts — which is why 1136 mentions convert to 431 concepts (38%) while 543 convert to
4 (0.7%).

**This is the third time mention counts have overstated concept-level value in this
project** — merging 91 → 11, abbreviations 543 → 4, paraphrase 1136 → 431. The pattern is
now established well enough to treat any mention-count opportunity claim as unpriced until
ablated. (The two slices sum to +435 against the combined +434: one concept is reachable
from both slices in the same document.)

**Consequence for scoping.** The deterministic route does not compete with the embedding
route for this metric — **paraphrase is ~99% of the ceiling (+0.0816 of +0.0821)**, not the
~57% the mention-level split implied. If an embedding fallback is built, its target is the
paraphrase slice, and abbreviation expansion should *not* be scoped as a cheaper alternative
to it.

#### Scope limitation (restated deliberately — this is half the finding)

**Retired for clustering. Open for the rest of the pipeline.**

The +0.0008 is small *because document-level set semantics hide the correction*, not because
the correction is wrong. All 543 mentions genuinely go from NIL to a correct concept id; the
metric simply cannot see it, because the same concept was already recovered from the long
form elsewhere in the document.

Any consumer that reads canonical ids **per mention** rather than per document therefore
gets the full value:

- **Phase 4 (extraction / evidence attribution)** — a claim anchored to the mention `TdP`
  carries no concept id today. Its long form being linked three sentences away does not help
  a per-mention lookup.
- **Phase 5 (contradiction detection)** — comparing two papers' claims requires resolving
  the entity *in the claim*, not somewhere in the abstract.

Both should **re-measure against their own metric**, not inherit this result. `oracle_abbrev`
and `oracle_paraphrase` are logged on every e2e run precisely so that re-measurement has a
baseline to compare against.

**The one-line version for anyone citing this:** *abbreviation expansion buys nothing for
clustering and may still be necessary for mention-level work.*

### The precondition guard (fail closed, not documented-and-hoped)

Abbreviation expansion needs the **whole document** — the defining `long form (SHORT)` is
usually sentences away from the mentions it licenses. Handed a fragment it finds nothing,
which is **indistinguishable from "this document had no abbreviations"**. Documentation
alone was judged insufficient: undetected precondition violations are a recurring failure
category here (the 512-token truncation crash, the unconditional merge, the CTD column
schema all shipped green test suites).

`biolit.canon.context.check_document_context(entities, text)` reports two kinds of failure:

- **Provable** — a span out of bounds, or not slicing to its own entity text. The caller
  passed text that is definitively not the document these entities came from.
- **Heuristic** — the text is shaped like a fragment. *No provable check can catch this
  one*: a sentence and the entities extracted from it are perfectly self-consistent, so only
  the shape of the text betrays it.

The scorer consults it, **skips** expansion, logs the reason, and reports
`abbrev_context_skipped` in the run log — the skip is recorded rather than inferred from a
zero.

**Calibration, measured rather than assumed.** Requiring at least one sentence break
separates the corpora perfectly on its own: **0 of 500** BC5CDR abstracts flagged, **49 of
49** domain sentences flagged. A character floor contributes nothing at any value up to 200
and begins producing false positives at 250 (2 real abstracts). The shortest real BC5CDR
abstract is **204** characters, so an initial 200-char floor sat 4 characters from firing on
real data; it was lowered to 100 as a pure backstop against degenerate input. **`0` skips on
BC5CDR also confirms the +0.0008 abbreviation figure above is not an artifact of the guard
suppressing the mechanism.**

The domain corpus result is the guard doing its job: those 49 "documents" are single
sentences, so abbreviation expansion is **structurally unmeasurable** there — previously
that would have silently reported zero.

### Reproducing this without the module

The ingest is **deliberately not on `master`** (ADR-0011): no infrastructure without a
demonstrated consumer, and a falsified hypothesis is not one. It is preserved on the
unmerged branch **`phase-3c-mesh-alias-enrichment`** (`a246703`, `5e2d3d5`). Everything
needed to rebuild it:

- **Sources:** `https://nlmpubs.nlm.nih.gov/projects/mesh/MESH_FILES/xmlmesh/desc2026.gz`
  (16 MB) and `.../supp2026.gz` (45 MB). Note the year in the filename — the directory
  listing is the reliable way to find the current release.
- **Parse:** stream with `iterparse` (uncompressed dumps are hundreds of MB). Per record
  take `DescriptorName/String` (or `SupplementalRecordName/String`) as the preferred alias
  and every `ConceptList/Concept/TermList/Term/String` as a synonym; id is `MESH:` + UI.
- **Scope filter (load-bearing):** descriptors whose `TreeNumberList` contains a number
  starting `C<digit>`, `D<digit>`, or `F03`; supplemental records with **`SCRClass`** in
  {`1` chemical, `3` disease}. The attribute is `SCRClass`, **not**
  `SupplementalRecordType`. Without this filter the table gains anatomy, organism and
  technique surfaces that a CHEMICAL/DISEASE span can match wrongly.
- **Expected figures if reproduced correctly:** 153,539 aliases after descriptors, 784,332
  after supplementals, 819,422 in union with CTD, 271,668 distinct concept ids, and 3/2514
  rescued.

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

### Truncation: a null result, and why no truncation-recovery work is scoped

This is stated as a **null result**, not as an open task deferred for lack of time.

The 90-mention probe suggested a clean, designable target: truncation that consistently
drops suffixes (9 of 11), concentrated in CHEMICAL. At 9809 mentions **every part of that
picture is contradicted**:

- The suffix-dropping pattern flattens into a near-even three-way split
  (38.6% / 33.3% / 28.1%).
- The label asymmetry **inverts** — CHEMICAL is the *stronger* label (85.3% exact vs
  DISEASE 78.2%), and DISEASE truncates 2.5× more often, the opposite of what the probe
  showed.
- The largest bucket, `INTERIOR_OR_OTHER` (38.6%), is a **catch-all mixing distinct failure
  types** — interior truncations and predictions extending *past* gold are counted
  together. It is not one phenomenon.

So there is **no single systematic pattern to design a repair mechanism against.** A
truncation-recovery sub-project scoped today would be designed against a pattern that the
full corpus says does not exist; the probe's apparent signal was small-sample noise. That
is the reason none is being scoped — not that it was skipped.

What would change this conclusion is a *finer* census, not a bigger one: splitting
`INTERIOR_OR_OTHER` by the sign of `char_delta` (already computed per mention, not yet
aggregated) would establish whether over-extension and interior truncation are separate,
individually-systematic phenomena. Until that exists, any repair design would be guesswork.

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

## Phase 3C: pricing the embedding fallback — a real but weak positive

> **Read the scope before the number.** Everything here is measured against
> **document-level concept-set scoring** — the metric clustering consumes. It establishes
> that an embedding fallback is not worth **shipping for clustering** at its current
> quality. It does **not** establish that the paraphrase gap is unreachable, and it does
> **not** transfer to mention-level consumers. See
> [Scope limitation](#scope-limitation-this-is-a-clustering-verdict-not-a-pipeline-verdict)
> below — the same discipline applied to abbreviation expansion applies here, in the
> opposite direction.

The oracle ceiling priced the `EXACT`-but-`NIL` population at **+0.0821 F1**, and the
ceiling-splitting analysis showed **paraphrase is ~99% of it (+0.0816)**. But that ceiling
is **recall-only by construction** — granting gold ids converts `fn`→`tp` and can never add
a false positive, which is why `fp` stayed pinned at 312 across all four oracle variants.
Its distance from reality was entirely unmeasured.

This section closes that gap: **what a real fallback linker actually costs in precision**,
as a curve over its confidence threshold, measured over the **full 3209-mention NIL
population** rather than the favorable 1136-mention paraphrase slice.

### The three numbers that decide this

The headline ΔF1 is the *least* informative figure here, and citing it alone oversells the
result. The argument against shipping rests on these three:

| # | Measure | Value | Why it matters |
|---|---|---|---|
| 1 | **Mention-level precision at the peak** | **53.9%** (365 correct / 677 fired) | At its own best operating point the fallback is barely better than a coin flip. Every second link it makes is wrong. |
| 2 | **`mentions_wrong` / Δ`fp`** | **2.89** | 312 wrong mention links cost only 108 *new* false positives. Concept-level scoring absorbs ~2.9 wrong links per error it reports — the apparent value depends on the metric hiding the cost. |
| 3 | **Realized gain / oracle ceiling** | **24%** (+0.0200 of +0.0821) | A naive top-1 embedding linker recovers a quarter of the theoretical maximum. Three quarters of the ceiling stays out of reach. |

**On #2 specifically — the logged field is the weaker of the two ratios.** The sweep emits
`mentions_wrong / fp` using **total** `fp` (312/420 = **0.743**), but that denominator
includes the 312 false positives the dictionary already had. The quantity that answers "how
much error is the concept view absorbing" is `mentions_wrong / Δfp`. It is consistent across
all four arms: **2.89 / 2.76 / 2.32 / 2.72**. Cite the delta ratio; the logged field is
diluted.

### The curve: full 3209 NIL population, argmax-F1 breakpoint

Baseline (dictionary only, no fallback): **P 0.8822 / R 0.6826 / F1 0.7697**, `fp` 312.

| arm | thr | P | R | **F1** | ΔF1 | `fp` | fired | correct | wrong | wrong on non-aligned | mention P | wrong/Δfp |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **SapBERT, label-constrained** | 0.9098 | 0.8565 | 0.7326 | **0.7897** | **+0.0200** | 420 | 677 | 365 | 312 | 219 | **53.9%** | **2.89** |
| SapBERT, unconstrained | 0.9126 | 0.8540 | 0.7314 | 0.7880 | +0.0183 | 428 | 669 | 349 | 320 | 218 | 52.2% | 2.76 |
| TF-IDF, label-constrained | 0.8533 | 0.8563 | 0.7086 | 0.7755 | +0.0058 | 407 | 348 | 128 | 220 | 147 | 36.8% | 2.32 |
| TF-IDF, unconstrained | 0.8852 | 0.8758 | 0.6946 | 0.7748 | +0.0051 | 337 | 132 | 64 | 68 | 47 | 48.5% | 2.72 |

The sweep is computed **by breakpoint, not on a grid**: every distinct top-1 similarity is
a threshold, yielding 1345–1376 exact points per arm, so the peak is never an artifact of
where a grid landed. Full curves are in `backend/evals/fallback_sweep.jsonl`.

**Shape:** a clean single peak in every arm. Firing on everything is *worse than not firing*
— at threshold 0.57 SapBERT unconstrained scores **F1 0.7027**, well below the 0.7697
baseline. F1 climbs to its peak near 0.91 and returns to baseline at 1.0. **An unthresholded
embedding fallback actively harms this pipeline.**

**Label-constraining helps slightly and costs nothing:** +0.0017 F1 (0.7880 → 0.7897). The
concept→label map is built from the same CTD download as the alias table in the same run, so
the two artifacts cannot drift; the build asserts **0 orphan concepts** (verified, not
assumed) and reports **79 multi-label concepts** — the empirical justification for the map
being `dict[str, set[EntityLabel]]` rather than a scalar. A last-write-wins scalar would have
silently mislabeled all 79.

### This is *not* the MeSH-ingest null. The effect is genuine.

**ADR-0011 rejected dictionary enrichment because the effect did not exist** — 3 rescued
surfaces out of 2514, a measured zero. **This is a different verdict on different evidence.**
The effect here is real, statistically unambiguous, and correctly attributed:

**SapBERT beats the character n-gram TF-IDF control by 3.4× (+0.0200 vs +0.0058).**

That control existed precisely so a gain could not be credited to the model until plain
fuzzy string matching was ruled out as the explanation — the same discipline that produced
the enrichment null. It was not ruled in. Character n-grams over the identical 552,756 alias
rows, with identical preprocessing (`normalize_surface`) and the identical sweep, recover
under a third of the gain. The mechanism doing the work is **semantic generalization, not
string similarity**, and the two are materially different builds.

So the finding is: **the paraphrase gap is real, it needs a model rather than a better string
matcher, and the obvious model recovers only a quarter of it at ~54% mention precision.**
The reason not to ship is **cost and fragility relative to value**, not absence of effect.

### Why the full population, and what the slice would have hidden

The measurement covers all **3209** NIL mentions, not the 1679 gold-aligned ones and not the
1136 paraphrase slice. A real linker cannot see gold alignment, so it fires on all 3209;
every confident wrong link among the **1530 non-gold-aligned** NILs (truncated spans,
spurious NER output, and the 53 `GOLD_UNLINKABLE` mentions where **NIL is the correct
answer**) is a false positive a 1136-scoped measurement structurally cannot observe.

That was not a hypothetical risk. It is measured, twice over:

- **219 of 312 wrong links (70.2%) land on the 1530 non-gold-aligned mentions** — invisible
  to a slice-scoped run.
- The **paraphrase-slice curve reports +0.0254 against the full population's +0.0200 — a 27%
  relative overstatement**, at a flattering 69.0% mention precision (462/670) instead of
  53.9%.

**Slice figures, labeled as slice figures wherever cited:**

| arm (paraphrase slice, 1136 mentions — **slice, not full population**) | thr | F1 | ΔF1 |
|---|---|---|---|
| SapBERT, label-constrained | 0.8540 | 0.7951 | +0.0254 |
| SapBERT, unconstrained | 0.8540 | 0.7944 | +0.0247 |
| TF-IDF, label-constrained | 0.7899 | 0.7822 | +0.0125 |
| TF-IDF, unconstrained | 0.7899 | 0.7805 | +0.0108 |

**This is the fourth time in this project that a favorably-selected population reported a
better number than the system would actually pay** (merging 91→11, abbreviations 543→4,
paraphrase 1136→431, and now slice +0.0254 vs full +0.0200). The pattern is established well
enough that a subpopulation figure should be treated as unpriced until measured against the
full population the mechanism would actually run on.

### Scope limitation: this is a clustering verdict, not a pipeline verdict

**Rejected for clustering. Explicitly open — and explicitly *more* concerning — for
mention-level consumers.**

This mirrors the abbreviation-expansion caveat, in the opposite direction. There, a
mechanism looked worthless at concept level (+0.0008) while genuinely correcting 543
mentions, so mention-level consumers were told **not to inherit the rejection**. Here the
asymmetry runs the other way:

- **For clustering** (document-level concept sets), 53.9% mention precision is survivable,
  because set semantics absorb ~2.9 wrong links per counted error. That absorption is the
  only reason the F1 delta is positive at all.
- **For a mention-level consumer, that same absorption does not happen and the error is
  fully exposed.** A Phase 4 evidence-attribution claim anchored to a specific mention, or a
  Phase 5 contradiction comparison resolving the entity *inside* a claim, would be reading a
  concept id that is **wrong roughly half the time**. That is unusable as ground truth for a
  specific claim.

**Phase 4 and Phase 5 must not inherit this conclusion in either direction.** "+0.0200 F1,
documented not shipped" is a statement about clustering's metric. The mention-level statement
is stronger and more negative: *at 53.9% precision this mechanism is not fit to anchor a
claim, and would need a substantially higher threshold (trading away most of the recall) or a
better model before it could be.* Both phases should re-measure against their own metric.

**The one-line version for anyone citing this:** *the embedding fallback is a real effect,
too weak and too costly to ship for clustering, and too imprecise to anchor mention-level
claims at all.*

### Reproducing this without the module

The implementation is **deliberately not on `master`** (ADR-0012), preserved on the unmerged
branch **`phase-3c-fallback-measurement`** (`0f4eac7`..`fbcae08`). The run log
`backend/evals/fallback_sweep.jsonl` **is** on that branch and carries all four full curves.
Everything needed to rebuild:

- **Encoder:** `cambridgeltl/SapBERT-from-PubMedBERT-fulltext`, **CLS pooling**, L2-normalized,
  `max_length=32`, batch 128, CPU. SapBERT is contrastively trained over UMLS synonym pairs —
  the "same concept, different surface" relation the paraphrase slice *is*.
- **Index:** all **552,756 (alias, concept) pairs** from 551,669 distinct aliases over 192,816
  concepts — not the 192,816 preferred names. A concept scores as the **max** over its aliases.
  fp16 on disk (~849 MB), fp32 for the matmul.
- **Search:** **exact** brute-force chunked cosine, no ANN. The query side is only 3209
  mentions, so exact search costs seconds, and approximation would inject error into a
  measurement whose entire purpose is error attribution.
- **Control:** `TfidfVectorizer(analyzer="char_wb", ngram_range=(3,3))` over the identical
  rows with the identical `normalize_surface` preprocessing.
- **Sweep:** breakpoint-exact (every distinct top-1 score is a threshold).
- **Command:** `uv run python -m biolit_evals.fallback_sweep --dataset bc5cdr --rebuild-index`
- **Cost if reproduced:** ~46 min to encode the index (CPU, ~199 aliases/sec after
  length-sorted batching); NER over the 500 BC5CDR abstracts is only ~46s.

### Harness validation and two anomalies that were chased down

**The anchor gate is a correctness check on the harness, not a result.** At the top threshold
nothing fires, so all four arms must reproduce the dictionary-only baseline *exactly*. They
do: **F1 0.7697, `fp` 312, `mentions_fired` 0, `n_nil` 3209**. A mismatch halts the run with
`SystemExit`; the anchors were never adjusted to match observations.

An additional **pre-flight** ran `collect_nils` over real BC5CDR with real NER and **no
scorer at all**, checking three quantities the gate does not: it reproduced the
**1679 / 543 / 1136** gold-aligned / abbreviation-addressable / paraphrase split exactly.
That is what establishes the slice by-product measures the population it claims to, since
`abbrev_addressable` is new code and its agreement with `score_end_to_end`'s oracle partition
was otherwise unverified on real data.

Two anomalies in the output were investigated rather than waved through:

1. **Two mentions scored cosine `1.0000001`.** If the fallback had found an *exact* dictionary
   alias the dictionary itself left NIL, that would be a harness inconsistency, not a result.
   Checked directly: **zero NIL surfaces are exact alias keys.** The dictionary and the
   fallback are consistent. The mechanism is presumed to be a truncation collision but was
   **not verified** — both links were correct and they move F1 by 0.0002.
2. **`max_length=32` truncates 39,937 of 551,669 aliases (7.24%)**, the longest being 292
   tokens (long IUPAC chemical names). This biases **against** SapBERT, so the reported gain
   is a **lower bound** and the SapBERT-beats-TF-IDF conclusion is strengthened by it rather
   than threatened. Not re-run at a longer length.

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
9. **The oracle ceiling assumes perfect precision, which no real linker has.** It grants
   gold ids and therefore adds zero false positives by construction. A fallback that
   resolves surfaces the dictionary abstained on will link some of them wrongly, costing
   precision the ceiling never pays. +0.0821 is an upper bound that a real system cannot
   reach, not an expected gain.
10. **Alias coverage is not attributed per source file.** The DISEASE-concentration finding
    points at thinner disease-side CTD coverage, but the artifact builder pools all aliases
    and `MeshConcept` records no source, so that explanation is untested — an alternative
    (disease surfaces in abstracts are simply more variable than chemical names) is equally
    consistent with the same numbers.
11. **Windowing statistics are not recorded in the run log.** "11 of 500 documents exceed
   512 tokens; 32 exceed the 450-token window budget" comes from ad-hoc measurement, not
   from a logged field, so it is not self-verifying on a future run.
12. **The fallback sweep logs the diluted `mentions_wrong / fp` ratio.** The field in
    `fallback_sweep.jsonl` divides by *total* `fp`, which includes the 312 baseline false
    positives the fallback did not cause. The decision-relevant quantity is
    `mentions_wrong / Δfp` (2.89 rather than 0.743) and it must be derived from the logged
    `fp` at the point and the baseline `fp` at the anchor — the log does not carry it
    directly.
13. **The fallback was measured as top-1 with a single global threshold**, which is the
    simplest possible policy. Per-label thresholds, margin-based abstention (top-1 vs
    top-2 gap), and score calibration were all out of scope. The 24%-of-ceiling figure
    therefore prices *this* policy, not the best achievable embedding fallback, and a
    stronger policy is the obvious place to look if this is ever revisited.
14. **`max_length=32` truncates 7.24% of aliases**, so the SapBERT arms are a lower bound.
    The direction is known and favors the conclusion drawn, but the magnitude of the
    understatement was not measured.

---

## Phase 3: pricing the pairing problem — same-sentence co-occurrence, and why CID relation extraction does not get its own phase

> **Read the scope before the number.** Everything here is measured against
> **paper-pair** scoring — the unit the Critic actually consumes, since
> `ContradictionFinding` is `(paper_id_a, paper_id_b, label, rationale)`. Cluster-key and
> key metrics are reported alongside as **diagnostics** and are materially worse; see
> [the key-vs-cluster gap](#the-key-vs-cluster-gap-the-same-mechanism-looks-worse-the-finer-you-score-it).
> The verdict here is about **whether to build a relation extractor**, not about how good
> clustering is in absolute terms.

ADR-0008 defines a cluster as papers about the same chemical–disease relationship. Nothing in
the pipeline decides *which* chemical goes with *which* disease inside a paper, so the
cheapest possible answer is the full cross product of every linked chemical against every
linked disease. That is obviously lossy on precision. The open question was **how lossy, and
whether closing the gap needs a real CID relation extractor** — a model, a training set, and
its own project phase.

BC5CDR answers this directly, and this is worth stating because an earlier note in this
project got it wrong: **`CDR_Data.zip` ships gold CID relations.** The `PMID<tab>CID<tab>chemMeSH<tab>disMeSH`
lines are 4-field records that the 6-field mention parser already skipped, so they had been
sitting unread in a corpus this project has used since Phase 2. There is a real gold standard
for "which papers belong together"; no synthetic clustering target was needed.

### The two arms, and why they run on different corpora

| arm | entities | corpus | why |
|---|---|---|---|
| **A** | synthesized from **gold** mentions + gold MeSH ids | all **1500** docs | no model runs, so no contamination risk; maximum sample |
| **B** | the **real** NER + linking pipeline | **Test-500** only | the NER checkpoint was fine-tuned on BC5CDR *train*; any arm running real NER must be held out |

The Test-500 restriction on Arm B is a **correctness requirement, not a sample-size
preference.** Arm A is unconstrained because it never invokes the checkpoint.

**The two arms are therefore not directly comparable, and an earlier draft of the spec claimed
they were.** See [the corrected ceiling claim](#the-arm-a-ceiling-claim-was-false-and-the-rule-that-replaces-it)
below — that error is kept and labelled rather than edited away, because the rule it produced
is reusable.

### Headline: same-sentence co-occurrence wins on every arm and every level

Paper-pair (primary), from `evals/cluster_runs.jsonl` at `7430c6e`:

| arm | strategy | P | R | **F1** | tp | fp | fn | clusters | Critic calls | cluster comparisons | largest | top-5 share |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| A — gold / 1500 | `cross_product` | 0.2743 | **1.0000** | 0.4305 | 1938 | 5128 | 0 | 1772 | 7066 | 10064 | 37 | 0.204 |
| A — gold / 1500 | `same_sentence` | 0.4786 | 0.8818 | **0.6204** | 1709 | 1862 | 229 | 747 | 3571 | 4399 | 28 | 0.268 |
| B — real / 500 | `cross_product` | 0.4000 | 0.8720 | 0.5484 | 184 | 276 | 27 | 229 | 460 | 648 | 11 | 0.364 |
| B — real / 500 | `same_sentence` | 0.5556 | 0.7346 | **0.6327** | 155 | 124 | 56 | 96 | 279 | 336 | 11 | 0.503 |

Gold: **325 clusters / 1938 paper-pairs** on all 1500; **80 clusters / 211 paper-pairs** on
Test-500.

**Two workload columns, not one, and the distinction is not cosmetic.** *Critic calls* is the
count of **distinct** paper pairs; *cluster comparisons* counts a pair once per cluster it
appears in. `ContradictionFinding` is `(paper_id_a, paper_id_b, label, rationale)` with **no
key field**, so a pair co-clustered under two different keys is **one** Critic call whose
result cannot be reported twice — the call count is the cost figure, and the comparison count
is the basis for the concentration view (`top5_pair_share` is a share of the latter). An
earlier version of this harness reported only the comparison count, under a field named
`n_paper_pairs`, **overstating the Critic's workload by 20–42%**; see
[limitations](#limitations) and the two superseded log lines at `7430c6e`.

Two things fall out immediately:

1. **A free deterministic heuristic beats the cross product decisively** — +0.1899 F1 on gold
   entities, +0.0843 on the real pipeline. No model, no training data, one sentence-boundary
   lookup per entity.
2. **It roughly halves the Critic's workload** — Critic calls drop 7066 → 3571 on Arm A
   (**1.98×**) and 460 → 279 on Arm B (**1.65×**). Since every call is an LLM invocation in
   Phase 5, this is a direct cost result, not just a quality one. On the comparison basis the
   reduction looks larger (2.29× and 1.93×); the call basis is the honest one to quote.

This reproduces the design-time exploration that motivated the harness (cluster precision
18.8% → 36.9% on Test-500, vs 18.34% → 33.47% measured on all 1500) — same direction, close
magnitude, different corpus.

### The key-vs-cluster gap: the same mechanism looks worse the finer you score it

Cite the paper-pair number, but never without this. The identical run scored at three levels:

| arm | strategy | paper-pair F1 (**primary**) | cluster-key F1 | key F1 |
|---|---|---|---|---|
| A — gold / 1500 | `cross_product` | 0.4305 | 0.3100 | 0.3243 |
| A — gold / 1500 | `same_sentence` | **0.6204** | 0.4664 | 0.4392 |
| B — real / 500 | `cross_product` | 0.5484 | 0.3819 | 0.2905 |
| B — real / 500 | `same_sentence` | **0.6327** | 0.5341 | 0.3317 |

On Arm B, same-sentence pairing scores **0.6327 at paper-pair level and 0.3317 at key level —
a 0.3010 gap.** Paper-pair scoring is more forgiving because two papers count as correctly
co-clustered if they share *any* predicted key; it does not require the key itself to be a
real relation. That forgiveness is legitimate — it is exactly what the Critic needs, since the
Critic compares two papers and does not consume the key — but it means **the key-level number
is the honest measure of how well the system identifies actual chemical–disease relationships,
and it is roughly half.** Any future consumer that reads the cluster key as a claim about a
real relation should use the key-level figure, not the headline.

### The `Arm A is a ceiling` claim was false, and the rule that replaces it

The approved spec asserted that Arm A bounds Arm B, and that the gap between them "prices what
NER and linking cost clustering." **Both claims are wrong, and the data refutes them
directly.** Made apples-to-apples — Arm A re-run restricted to Test-500 (scratch run,
deliberately **not** committed to the log, since it is a diagnostic and not a result):

| entities | strategy | P | R | **F1** |
|---|---|---|---|---|
| **gold** (Arm A construction, Test-500) | `same_sentence` | 0.4279 | 0.9005 | 0.5802 |
| **real pipeline** (Arm B) | `same_sentence` | 0.5556 | 0.7346 | **0.6327** |

**The real pipeline scores higher on precision and F1 than the supposed ceiling.** The
mechanism is structural, not incidental: granting perfect entities can only *add* candidate
pairs, and adding candidates against a fixed gold pair set can only cost precision. Arm A is a
ceiling for **recall only** — 1.0000 by construction, which the run log confirms exactly.

This is the same shape as Phase 3C's oracle ceiling being recall-only, so it is stated here as
a checkable rule rather than a corrected number:

> **The recall-only ceiling rule.** Any construction that improves recall by *granting
> something correct*, without touching what the system emits *wrongly*, is a ceiling for
> recall alone — never for precision or F1. Before calling anything a ceiling, ask what it can
> make **worse**. If the answer is "nothing," it is recall-only, and its F1 is not an upper
> bound on anything.

**The second-order finding, and its limits.** NER and linking act as an unintended **precision
filter**: they lose recall, but the entities they *do* resolve are the more canonical ones,
and canonical entities correlate with the relations BC5CDR annotates. This is a property of
**this corpus and this gold standard**, not a general result, and it is emphatically **not** an
argument that noisier extraction is better or that better NER is unnecessary. The direct
measure of what better NER would buy is in the next section: **40.3% of gold relations lose an
endpoint outright**, and no pairing mechanism recovers those.

### The entity-conditioned ceiling: the number that actually prices relation extraction

Neither arm above answers "what would a *perfect* pairing mechanism score on the entities real
extraction actually delivers." That is the only construction directly comparable to
`same_sentence`, so it is the one that prices whether relation extraction earns a phase. Credit
a pair only where **both endpoints are present in that paper's real linked entity set** and the
pair is gold-true. This is `entity_conditioned_oracle`, logged on every run beside the
strategies — not a scratch script, because the recommendation's second reason rests on it:

```
gold CID relations (Test-500)        1066
  reachable on real entities          636   (59.7%)
  endpoint lost by NER/linking        430   (40.3%)   <- no pairing mechanism recovers these
```

| construction | P | R | **F1** | clusters | Critic calls | comparisons |
|---|---|---|---|---|---|---|
| **ORACLE** — perfect pairing on real entities | **1.0000** | 0.6967 | **0.8212** | 57 | 147 | 157 |
| `same_sentence` | 0.5556 | **0.7346** | 0.6327 | 96 | 279 | 336 |
| `cross_product` | 0.4000 | 0.8720 | 0.5484 | 229 | 460 | 648 |

Headroom from `same_sentence` to the oracle is **+0.1886 F1**, and `same_sentence` already
captures **77.0%** of it.

**Three independent checks validate the construction**, none of them forced by the harness:

1. **On Arm A the oracle is perfect** — 3116 of 3116 gold relations reachable, F1 1.0000 at all
   three levels. It has to be: a gold relation's endpoints are themselves gold-annotated
   mentions, so entities synthesized from that gold cannot fail to reach one.
   `assert_full_reachability_on_gold_entities` now gates exactly this.
2. **A three-way identity, each side computed by a different code path:** the reachable share
   (636/1066), the oracle's key-level recall, and `cross_product`'s key-level recall are all
   **0.5966**. A cross product over real entities recovers exactly the reachable gold keys and
   no more, which is what "conditioned on real entities" must mean if the code is right.
3. **The oracle's distinct-call count equals its true positives exactly** (147 = 147) with
   `fp` 0 — confirming precision 1.0000 is structural, not incidental.

**This oracle is *not* a hard F1 bound, and calling it one would repeat the exact error
corrected above.** Its precision is 1.0000 **by construction** — it emits only gold-true keys —
so it is a **precision-perfect, recall-capped** construction. Applying the rule: what can it
make worse? Nothing. So it bounds recall and nothing else.

**The empirical proof, not just the structural argument:** `same_sentence`'s paper-pair recall
is **0.7346, which exceeds the oracle's 0.6967.** A mechanism already in the table beats the
"ceiling" on recall. The mechanism is that paper-pair recall credits a co-clustered pair
whenever the two papers share *any* predicted key, so a **wrong** key held by both papers can
accidentally co-cluster a genuinely gold-related pair. The oracle, emitting only gold-true
keys, forgoes those accidental wins. A mechanism trading precision for recall can therefore
exceed F1 0.8212.

### Recommendation: do not scope CID relation extraction

Three reasons, in order of weight:

1. **The binding constraint is upstream, not in pairing.** 430 of 1066 gold relations
   (**40.3%**) have an endpoint the NER + linking pipeline never produced for that paper. No
   pairing mechanism — heuristic, learned, or perfect — can pair an entity that was never
   extracted. A relation extractor would compete for the remaining 59.7%.
2. **The free heuristic already captures 77.0% of a precision-perfect ceiling at zero model
   cost.** `same_sentence` reaches F1 0.6327 against the oracle's 0.8212. The entire
   pairing-strategy question is worth at most +0.1886 F1, and three quarters of that is
   already taken by a sentence-boundary lookup.
3. **The realistic win is *smaller* than +0.1886.** Part of `same_sentence`'s recall is
   accidental co-clustering through wrong shared keys — proven by its recall exceeding the
   oracle's. A precision-oriented relation extractor would give those up, so it starts from a
   worse recall position than the raw headroom suggests.

**Interpretation caveat on the precision numbers, and what it does *not* license.** Gold CID is
narrower than co-mention: BC5CDR annotates a chemical–disease pair only where the abstract
asserts a causal induction relationship, so some pairs scored as false positives are genuine
co-mentions that simply are not CID relations. That caveat corrects **how the precision number
should be read** — it is not a clean measure of "wrong pairs" — but it does **not** make the
precision problem matter less practically. The Critic's actual task is judging whether two
papers contradict each other about a drug–disease effect, which is itself the narrow,
causal-relation-shaped task BC5CDR annotates. A pair that is a co-mention but not a CID
relation is still a wasted Critic call.

### Carried forward, not built: cluster-size capping

`top5_pair_share` on Arm B `same_sentence` is **0.503** — half of all cluster comparisons come
from five clusters, and the largest single cluster contributes 55 (11 papers). On Arm A the
largest `same_sentence` cluster is 28 papers, or 378 comparisons from one key. Cluster-size
capping — sampling or truncating oversized clusters before they reach the Critic — is therefore
a real design consideration for whenever the Critic gets scoped.

**It is deliberately not built now.** There is no Critic, so there is no consumer, and this
project's standing rule is no infrastructure without a demonstrated consumer — the same
standard that left the MeSH ingest and the embedding fallback unmerged. The measurement that
would motivate a cap now exists and is committed; the cap itself waits for something to
protect.

### ⭐ Standing project-level finding: upstream entity loss is the dominant bottleneck, twice over

This is bigger than the clustering sub-project, and it is named here rather than left implicit
across two ADRs.

**Twice now, a phase has gone looking for a downstream mechanism to fix a quality problem, and
measurement has shown the loss happens upstream, before that mechanism ever runs.**

| phase | the mechanism that looked like the answer | what measurement actually showed |
|---|---|---|
| **3C — canonicalization** | fragment merging, then an embedding fallback linker | The dominant loss was **1679 gold mentions whose span NER got exactly right but which linked to nothing** — priced at **+0.0821 concept F1**, **39× the fragment-merge ablation** and the largest recoverable loss measured anywhere in this project. The fallback linker then realized only **24%** of that ceiling, at 53.9% mention precision. |
| **3 — clustering** | a CID relation extractor with its own phase | **40.3% of gold relations lose an endpoint to NER/linking outright.** The entire pairing question is worth ≤ +0.1886 F1, of which a free heuristic already takes **77%**. |

In both cases the entities that were never extracted or never linked bounded the result, and
in both cases that loss was **downstream-unrecoverable**: no linker fixes a span NER missed,
and no pairing strategy pairs an entity that does not exist. The pattern is not two
coincidences — it is the same finding, and it says where the next investment belongs:

> **Entity recall — NER coverage and linking coverage — is this pipeline's dominant,
> downstream-unrecoverable constraint. Downstream reasoning components should be priced against
> an entity-conditioned ceiling, not an oracle-entity one, or they will be scoped against
> headroom that does not exist.**

The entity-conditioned oracle in this section is the concrete template for doing that.

### Diagnostics: the NIL populations are different populations

| arm | `nil_chemical_mentions` | `nil_disease_mentions` | papers with no linked chemical | papers with no linked disease | unplaceable entities |
|---|---|---|---|---|---|
| A — gold / 1500 | 117 | 109 | 0 | 0 | 0 |
| B — real / 500 | 1276 | 1933 | 7 | 29 | 0 |

Arm B shows the **1.5× DISEASE skew** expected of real linking, consistent with Phase 3A/3C's
finding that linking failure is overwhelmingly a disease problem. Arm A does not, because there
"NIL" means the *gold* carries no MeSH id at all (BC5CDR's `-1` annotations) — a different
population entirely. **Do not conflate them:** Arm B's NILs are mentions a real linker failed
to resolve; Arm A's 226 are mentions with nothing to resolve *to*.

**An unplanned cross-harness check fell out of this.** Arm B's two counters sum to
**1276 + 1933 = 3209**, which is *exactly* the full NIL population Phase 3C's fallback sweep
measured on the same split. Two independently built harnesses — one scoring concept sets for a
linker sweep, one scoring paper pairs for clustering — partition the same 3209 mentions. That
is not a designed anchor, but it is the strongest available evidence that both harnesses see
the same pipeline output, and it retroactively corroborates the Phase 3C denominator.

Arm A's two counters were **structurally always 0** until a harness defect was found and fixed
during Task 8: `synthesize_records` dropped zero-id gold mentions entirely rather than
materializing them as `canonical_id=None`, and `pairing_diagnostics` counts NIL on entities
that *exist* — so the diagnostic was inert for the entire arm, while the unlinkable gold
mentions were exactly what it existed to expose. Fixed in `7430c6e`. Metric invariance was
verified rather than assumed: both pairing strategies filter `canonical_id is None`, so a NIL
entity is provably indistinguishable from an absent one to every scored metric, and only the
diagnostic moves.

The corrected counters were then **independently confirmed against the corpus** rather than
trusted from the log: counting gold mentions with an empty `mesh_ids` across all three splits
gives **117 CHEMICAL / 109 DISEASE of 28,785 gold mentions (226 total, 0.8%)** — exactly the
committed figures. On Test-500 alone the same count is 30 / 61, consistent with the 53
`GOLD_UNLINKABLE` mentions Phase 3A reported over its narrower exact-span population.

### Reproducing these numbers

```bash
cd backend
# Arm A — gold entities, all three splits (1500 docs), no model loaded
uv run python -m biolit_evals.cluster_eval --arm A
# Arm B — real NER + linking, Test-500 only (loads the checkpoint)
uv run python -m biolit_evals.cluster_eval --arm B
```

The corpus split is fixed by the arm, not by a flag, precisely so the Test-500 holdout on Arm B
cannot be widened by a careless command line.

Both write to `evals/cluster_runs.jsonl`. **Three anchors plus one guard gate every run on the
real path**, not only in tests:

- **`assert_key_recall_anchor`** — `cross_product` key recall must round to exactly **1.0000**
  on gold entities. The cross product emits every chemical × disease combination by
  construction, so any gold key it misses means the harness lost gold, not that the strategy
  underperformed. Tolerance is `round(recall, 4)`, tight enough to catch a harness dropping 1
  gold pair in 1000.
- **`assert_gold_cluster_anchor`** — gold cluster counts must match the independently
  established statistics (**500 → 80**, **1500 → 325**). This is a **loader-correctness check
  on `load_bc5cdr_cid_relations`, not a clustering-quality check**; it validates that the CID
  relations were parsed and reconciled correctly before any strategy is scored.
- **`assert_full_reachability_on_gold_entities`** — on Arm A the oracle must reach **every**
  gold relation, for the structural reason given above. This gates the ceiling the
  build-or-not decision rests on. Arm B is deliberately **not** gated: there the 40.3% loss
  *is* the result, and anchoring a finding is how a harness stops being able to surprise you.
- **`assert_dataset_size`** — a run whose document count contradicts its `dataset` tag halts
  before any scoring. Without it, an Arm B run over anything but exactly 500 documents was
  scored and logged with **no anchor firing at all**, since the gold-cluster anchor returns
  silently for untabulated sizes and the key-recall anchor is Arm-A-only.

If any of them fires, the harness is wrong: report the mismatch, do not adjust an anchor to
match the observation, and do not commit the run as a result.

### Limitations

1. **The two superseded log lines at `7430c6e` carry a wrong field.** They report
   `workload.n_paper_pairs`, which summed n(n-1)/2 per cluster — the *cluster-comparison*
   count — while being named and cited as the paper-pair count, overstating the Critic's
   workload by **20–42%**. They are kept rather than edited, per this project's practice of
   documenting a wrong log line instead of rewriting results; the lines at `ce489c4` and later
   carry `n_distinct_paper_pairs` and `n_cluster_comparisons` separately. No P/R/F1 at any
   level was ever affected — all 24 triples recompute exactly from their own `tp/fp/fn`.
2. **The Arm A @ Test-500 comparison is a recipe, not a logged run.** No flag exposes it,
   deliberately — a split flag on Arm A is exactly the misuse `assert_dataset_size` exists to
   prevent. The entity-conditioned oracle, which carries more decision weight, *is* a committed
   code path. The recipe below was run to produce the figures quoted above, and all four gates
   pass on it:

   ```python
   from datetime import UTC, datetime
   from biolit.config import get_settings
   from biolit_evals.cluster_eval import run_cluster_eval
   from biolit_evals.mesh_gold_download import (
       TEST_MEMBER, load_bc5cdr_cid_relations, load_bc5cdr_documents,
   )
   url = get_settings().bc5cdr_cdr_zip_url
   line = run_cluster_eval(
       documents=load_bc5cdr_documents(url, TEST_MEMBER),
       relations=load_bc5cdr_cid_relations(url, TEST_MEMBER),
       arm="A", dataset="bc5cdr_test500", log_path="/tmp/scratch.jsonl",
       git_sha="scratch", now=datetime.now(UTC).isoformat(),
   )
   ```

   Yields `cross_product` F1 0.4089 and `same_sentence` P 0.4279 / R 0.9005 / **F1 0.5802** —
   below Arm B's 0.6327 on the identical corpus. Write to a scratch path, not
   `evals/cluster_runs.jsonl`: this is a diagnostic, not a result.
3. **`SameSentencePairing` inherits the sentence splitter's known mis-splits** on
   abbreviations ("e.g. metformin"). In windowing that is harmless because consecutive windows
   overlap by a sentence; in pairing a mis-split can drop a real pair or invent one. That cost
   is inside the reported numbers rather than corrected for, which is why both strategies are
   reported side by side.
4. **`top5_pair_share` is a share of cluster comparisons, not of Critic calls.** Concentration
   is a property of cluster sizes, so the comparison basis is the correct one for that
   diagnostic — but it means the figure is not directly comparable to the call counts beside
   it.
5. **Gold CID is narrower than co-mention**, so the precision figures understate how many
   predicted pairs are defensible co-mentions. This corrects how precision should be *read*,
   not how much it matters — see the recommendation section for why the Critic's task is
   itself the narrow, causal-relation-shaped one.
6. **Alternative pairing heuristics were not swept.** Same-paragraph, N-token windows, and
   dependency-path variants are all cheap and unmeasured, so the 77.0%-of-ceiling figure
   prices *same-sentence specifically*, not the best achievable deterministic heuristic.
