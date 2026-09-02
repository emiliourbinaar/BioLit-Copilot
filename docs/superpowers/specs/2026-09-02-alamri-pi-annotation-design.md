# Alamri & Stevenson π̂ annotation pass — validating a candidate paper-pair contradiction gold

- **Date:** 2026-09-02
- **Status:** Design approved 2026-09-02. Builder and exporter implemented under TDD 2026-09-02;
  **no sheet generated and no annotation performed.** §0 figures and §4 corrected against the
  built module — see the marked notes.
- **Kind:** Measurement. A validity probe on a candidate gold source, run *before* any harness is
  built on it — the ordering ADR-0017 identified as the phase's most transferable lesson.
- **Depends on:** ADR-0017 (Gate 1/Gate 2 methodology, the two recorded protocol defects, and the
  standing rule that **π is a ceiling and never a correction factor**), ADR-0016 (verification
  rules), ADR-0013 (no infrastructure without a demonstrated consumer).
- **Does not decide:** corpus size. **Deferred by explicit instruction until π̂ returns**, because
  size only matters if the corpus survives on validity grounds.

## Summary

Four dataset families have now been measured against `ContradictionFinding`'s paper-pair unit.
Three failed structurally (§0.1). The fourth — Alamri & Stevenson's cardiovascular claim-pair
corpus — **clears every structural bar that killed the others**, and is the first candidate whose
remaining risk is not structural but *derived-pair validity*: whether a `YS × NO` claim pair under
a shared clinical question is a genuine disagreement between the two papers, or two compatible
findings separated by population, dose, or endpoint — the exact failure mode that retired the CTD
proxy at π̂ = 0.067.

That question is not answerable from the corpus's metadata. It is answerable by the instrument
Phase 5 already built and calibrated. **This design reuses that instrument unchanged** and adds
two things: π̂ is estimated *separately* for a lexically-flagged and a clean subset, so the reading
also says whether the lexical proxy is an actionable filter rather than merely a number; and the
sample is capped so no paper can appear more than once.

**Cost: zero.** Human annotation plus free NCBI `efetch`. No LLM call, no credential, no
authorization gate — there is nothing to authorize.

---

## §0 — What was measured before anything was designed

All figures below come from throwaway probes against the real artifacts on 2026-09-01/02, not from
dataset descriptions. Corpus: `data/alamri/corpus.xml`, 99,375 bytes, CC BY-NC-SA 2.0 UK, no DUA.

### §0.1 The prior families, and why this one is not another of them

| family | fatal finding | measured |
|---|---|---|
| **SciFact** | 0 derivable contradiction pairs; CONTRADICT claims are annotator *negations* of a SUPPORT twin | 45.99–75.53% of CONTRADICT claims have a near-twin sharing **identical evidence documents** |
| **HealthVer** | no source document ids **anywhere in the release** — the paper-pair unit is unrecoverable in principle | 477 claims (25.77%) with opposed passages, 0 with ids |
| **NLI4CT** | constructed balance; contradiction instances are metadata comparability, not scientific disagreement | 950/950 by design; Comparison+Results+Contradiction = 47 |
| **MultiCite** | wrong domain, and cited paper text is absent from the corpus entirely | 94.3% NLP; 1.8% of `@DIF@` spans carry explicit disagreement language |
| **SciCite** | right domain, both paper ids present — but the disagreement signal is not in the cited abstract | `label2` non-discriminative (81.1% vs 86.5%); 73/11,020 (0.66%) unambiguous; median **20%** term overlap with cited abstract; 2/21 numbers recoverable; 62% cite ≥2 references |

The common mechanism, diagnosed and then tested outside the family it was diagnosed in:
**claim-verification gold is authored by negating a proposition and re-checking it against the same
evidence, which destroys cross-paper signal.** The citation-context probe confirmed that the
failure survives outside that family, for a different reason — ADR-0017's A2 hazard, gold living in
text the Critic structurally cannot read.

**Alamri fails on none of these terms**, which is why the retirement conditional did not fire.

### §0.2 Alamri's structural properties — all clear

| property | measured | why it matters |
|---|---:|---|
| derivable contradiction pairs (`YS × NO`, shared question) | **727** | the unit is natively paper-pair |
| — of which excluded as self-pairs | **1** | PMID 20228403 carries both a `YS` and a `NO` claim under one question (olmesartan vs valsartan), which would derive a paper contradicting itself |
| derivable agreement pairs (`YS × YS`, `NO × NO`) | **1,047** | a real control class exists |
| class balance | **40.98%** contradiction | no 2.52%-prevalence problem (CTD's) |
| productive questions | **24 / 24** | no dead reviews |
| near-duplicate claim pairs in the whole corpus | **1** | not a negation-twin corpus — SciFact's killer is absent |
| claims appearing **verbatim** in their own abstract | **257 / 259 (99.2%)** | the judgment is recoverable from the abstract; the A2 hazard does not fire |
| abstracts retrievable via `efetch` | **254 / 254** | no missing-text attrition |
| overlap with BC5CDR | **0** | no contamination of any prior phase's corpus |
| original annotator IAA on the `YS`/`NO` assertion | **97%** | the source label itself is reliable |

**Disagreement was discovered, not constructed.** Claims were extracted from papers already
included in published systematic reviews and assigned `YS`/`NO` against the review's own clinical
question; no annotator wrote a negation. This is precisely the property SciFact lacks.

### §0.3 The remaining risk, quantified — why this pass exists

The corpus is structurally sound. The **derivation** from it may not be. Two lexical signals were
measured over the 727 contradiction pairs:

| signal | rate |
|---|---:|
| the two claims carry **differing population qualifiers** | 34.25% (249) |
| a claim shares **no key term** with its own review question | 37.96% (276) |
| **either** signal fires (**flagged**) | **58.46%** (425 pairs) |
| **neither** fires (**clean**) | **41.54%** (302 pairs) |

Plus two acknowledged properties that sampling cannot fix: pairs are **non-independent** (254
papers, median 4 appearances, max 22), and the corpus is **selection-biased** — papers were
enriched from systematic-review forest plots, so they are not a random sample of the literature.

A `YS × NO` pair whose two claims describe different populations is exactly the dual-pharmacology
shape that killed CTD. **Whether that shape dominates is an empirical question, and this design
answers it separately per subset rather than in aggregate.**

### §0.4 Sampling feasibility — verified before the caps were proposed

Pool sizes and diversity:

```
FLAGGED     pairs=425    distinct papers=187  distinct questions=20
CLEAN       pairs=302    distinct papers=167  distinct questions=22
AGREEMENT   pairs=1047   distinct papers=250  distinct questions=24
DISTRACTOR  pairs=21134  distinct papers=254  (188/276 key-term-disjoint question pairs)
```

⚠️ **A measured constraint that changed the design.** The initially proposed question cap of ≤ 2 is
**infeasible once the 5 cross-question distractors are added**, and no draw order rescues it:

```
45 pairs, cap_paper<=1, cap_question<=2, order F,C,A,D  -> drew [15,15,10,3]  INFEASIBLE
45 pairs, cap_paper<=1, cap_question<=2, order D,F,C,A  -> INFEASIBLE on 12/12 seeds
45 pairs, cap_paper<=1, cap_question<=3, order F,C,A,D  -> feasible on 12/12 seeds
```

The cause is arithmetic, not a defect: a distractor spans **two** questions and so consumes a slot
in each, and 45 pairs over 24 questions already needs ~1.9 slots per question before distractors
are counted. **The cap moves 2 → 3 because the batch composition changed, not because a reading was
unwelcome** — recorded here rather than silently adjusted, per the standing gate rule.

At `cap_paper ≤ 1`, `cap_question ≤ 3`, seed 20260902 — re-verified through the built module
against the real corpus, feasible on 12/12 seeds:

```
distinct papers used = 90/254   max appearances of any paper = 1
questions touched    = 23/24    max question-slots on any one question = 3
```

**Every paper in the batch appears exactly once**, down from a corpus maximum of 22.

---

## §1 — What is being measured

**π** = the fraction of derived `YS × NO` pairs that a blind human annotator judges a **genuine
disagreement between the two papers**, read from the two abstracts alone.

Estimated **separately for two strata**, so the result answers two questions at once: is the
derived gold valid, and is the lexical proxy an actionable filter for future gold construction.

**π is a ceiling. It is never a correction factor.** (ADR-0017, carried forward unchanged.)

---

## §2 — Why exactly 15 per stratum

`evaluate_gate2` **raises on any `n != 15`**. Its bands are literal binomial-tail counts computed at
n = 15, not a proportional rule:

| g (genuine) | verdict | power |
|---|---|---|
| g ≤ 7 | `STOP` | under true π=0.8, P(g≤7)=0.0042; under π=0.7, P=0.0500 |
| 8 ≤ g ≤ 10 | `CONTINUE_FLAGGED` | deliberately weak against π=0.6 (P=0.2131) |
| g ≥ 11 | `CONTINUE` | |

So **the sample is shaped to fit the calibrated instrument, rather than the instrument recalibrated
to fit a convenient sample.** That is the project's standing gate rule applied in advance instead of
in response to a reading. `evaluate_gate1`, `evaluate_gate2`, `wilson_interval` and
`parse_annotations` are reused **unmodified**; a proportional split of a 45-pair batch would have
required touching them, and is rejected for that reason alone.

---

## §3 — The batch: 45 pairs

| stratum | n | drawn from | role |
|---|---:|---|---|
| **C** — clean (neither §0.3 signal) | 15 | 302 pairs | Gate 2, n=15 → π̂_C |
| **F** — flagged (either §0.3 signal) | 15 | 425 pairs | Gate 2, n=15 → π̂_F |
| **A** — agreement control (`YS×YS` / `NO×NO`, shared question) | 10 | 1,047 pairs | strictness read |
| **D** — cross-question distractors (papers from key-term-disjoint questions) | 5 | 21,134 pairs | known-unrelated anchor |

**Independence caps:** each paper appears in **at most one pair in the entire batch**; each question
contributes **at most 3 pairs**. Draw order `F → C → A → D`, fixed and recorded. The seed is a
**required** argument, as in Phase 5 — never defaulted.

All 45 are **shuffled together** before presentation. The annotator is not told strata exist.

**Leakage check:** unlike Phase 5's null-endpoint tell, which identified `insufficient_overlap` with
certainty, agreement and contradiction pairs here are structurally identical — same question, same
presentation, same two-abstract format. Class is not inferable from layout.

---

## §4 — Blind protocol: two changes from Phase 5, each fixing a recorded defect

**(a) Show the review's clinical question — exactly ONE per row, distractors included.** ADR-0017 records the endpoint confound: showing one
randomly-chosen MeSH endpoint closed a label leak but stopped telling the annotator *which
chemical* the pair turned on (chemical-shown 7/14 exact match vs disease-shown 3/16, Fisher
p = 0.12). Alamri's question names population, intervention, comparator and outcome — precisely the
context that was missing. **It leaks nothing**: every pair under a review shares its question
regardless of class.

⚠️ **Corrected during implementation.** This section originally said distractor rows show *both*
questions, which contradicts §3's invariant that class is not inferable from layout — a
two-question row identifies a distractor with certainty, and this batch's sole annotator is also
its designer and knows five are present. Distractor rows therefore show **one** of the two
questions, chosen by seed. That makes a distractor an honest instance of the same task every other
row poses ("do these two abstracts disagree about this question?"), whose correct answer happens to
be `insufficient_overlap` because the second paper does not address the question at all.

**Two further layout tells, found by the same invariant and closed the same way.** `pair_id` is an
opaque digest rather than a readable composite, because a readable one would be visibly longer for
a distractor (two question ids, not one). And the two abstracts are presented in seed-randomised
order, because in the manifest `paper_id_a` of a contradiction pair is always the `YS` paper — left
unshuffled, "abstract A answers yes" would hold across every contradiction row in the batch.

**(b) Show full abstracts, never the extracted claim sentence.** Two independent reasons. The Critic
only ever sees abstracts, so anything else scores a judgment the arm never has to make (ADR-0017's
A2 hazard). And Alamri's own annotators keyed on those claim sentences — displaying one would hand
over the original judgment rather than eliciting an independent one.

**Withheld:** `ASSERTION` (`YS`/`NO`), `TYPE`, review PMID, and **review title** — titles frequently
state the answer outright (e.g. *"Arginine supplementation for improving maternal and neonatal
outcomes"*).

**Labels unchanged**, reusing `ANNOTATOR_LABELS`: `contradiction`, `agreement`,
`insufficient_overlap`, `cant_tell`. Each row also takes a free-text `reason`.

---

## §5 — Defect (b): the single-annotator confound, which this design does *not* fully fix

ADR-0017 records that one annotator cannot separate proxy failure from annotator strictness:
`insufficient_overlap` was applied to 18/30 rows where gold had 7, and to **4 of 8** gold
`agreement` pairs. Note also that **κ = 0.116 there was gold-vs-annotator agreement, not IRR**; with
one annotator no IRR statistic exists, and none will exist here either.

What this design adds is a **discriminator**, pre-registered before any label exists. The 10
agreement pairs are within-question and genuinely related; the 5 distractors are cross-question and
genuinely unrelated. Their *contrast* separates the two explanations:

| observed | reading |
|---|---|
| D mostly `insufficient_overlap`, A mostly `agreement` | label used **discriminatingly** — π̂ readings stand on their own |
| D and A **both** mostly `insufficient_overlap` | **general strictness** — π̂ readings are confounded in the same direction and must be reported with that caveat, not as clean validity estimates |
| D **not** mostly `insufficient_overlap` (< 4/5) | the label is not being used for its intended meaning at all; the strictness read is uninformative and must be reported as such |

**Numeric pre-commitment:** if `insufficient_overlap` on the 10 agreement pairs runs at **≥ 5/10**
(Phase 5's 4/8 rate or worse), both π̂ readings are reported as strictness-confounded.

**I reject myself as second annotator.** I would be validating gold that I would then use to grade a
model — circular, and the same shape as a score that cannot be lost. The preserved Phase 5 labels at
`backend/evals/gold/annotation_batch_1_labels.jsonl` provide a same-annotator cross-batch strictness
comparison at zero additional cost, and that is the only second reading available.

---

## §6 — Pre-registered decision rule

Written before any number exists. **Exhaustive over all nine verdict combinations** — a
pre-registration with an unruled cell is not a pre-registration.

Gate 1 runs first, and it is a **precondition, not a tiebreak**: over all 45 rows,
`3 × cant_tell > 45` → `REVISE_PROTOCOL`, i.e. `cant_tell > 15`.

> ⚠️ **If Gate 1 returns `REVISE_PROTOCOL`, the Gate 2 readings are not interpreted at all.** A high
> `cant_tell` rate means the protocol was unreadable, and nothing can then be concluded about the
> proxy in either direction. This is the reading that made ADR-0017's 1/30 interpretable.

Then Gate 2, run twice at n=15 with unmodified bands:

| π̂_C (clean) | π̂_F (flagged) | reading | action |
|---|---|---|---|
| `STOP` | `STOP` | invalid regardless of filtering | **Retire the paper-pair gold search as a four-family measured negative. Write the ADR; bring the draft before committing.** |
| `STOP` | `CONTINUE_FLAGGED` | proxy **inverted** | Do not proceed. Investigate the proxy before any use. |
| `STOP` | `CONTINUE` | proxy **strongly inverted** | Do not proceed. Investigate the proxy before any use. |
| `CONTINUE_FLAGGED` | `STOP` | filter directionally right, clean subset only marginal | Clean subset is a **provisional** candidate. Size question opens carrying an explicit marginal-validity caveat. No paid run on one marginal reading. |
| `CONTINUE_FLAGGED` | `CONTINUE_FLAGGED` | filter buys nothing; whole corpus marginal | Do not proceed. The corpus is not clearly usable and the filter does not rescue it. |
| `CONTINUE_FLAGGED` | `CONTINUE` | inverted, both marginal | Do not proceed. Investigate. |
| `CONTINUE` | `STOP` | ⭐ **the lexical proxy is an actionable filter** | Clean 302-pair subset is candidate gold. **Size question opens.** |
| `CONTINUE` | `CONTINUE_FLAGGED` | filter partially actionable | Clean subset is candidate gold; flagged subset excluded as marginal. Size question opens on the clean subset. |
| `CONTINUE` | `CONTINUE` | gold valid, filter unnecessary | Full 727 pairs candidate. **Size question opens.** |

**Three standing prohibitions, restated so they cannot be rationalised later:**

1. **No band moves to match an observation.** If a gate fires, the harness is right and the design
   is wrong; the mismatch gets reported.
2. **No second batch is drawn on an unwanted reading.** The bands were calibrated so that one batch
   suffices; re-drawing after a bad result is exactly what a pre-committed stop rule forbids.
3. **π̂ is not used to correct any downstream score.** It bounds one.

---

## §7 — Artifacts

| artifact | path | committed? |
|---|---|---|
| frozen derived-pair manifest + hash | `evals/gold/alamri_contradiction_pairs.jsonl` | **yes** — ids and flags only |
| blind sheet | `data/alamri_annotation_batch_1.{json,md}` | **no** — gitignored, carries abstract text |
| annotator labels | `evals/gold/alamri_annotation_batch_1_labels.jsonl` | **yes** — no abstract text |
| run log | `evals/annotation_runs.jsonl`, `"step": "alamri_annotation_gates"` | **yes** |
| cached abstracts | `data/alamri/abstracts.json` | **no** — gitignored |

The manifest is **frozen and hashed before the sheet is drawn**, so the sample is reproducible from
the seed and the hash. It carries `pair_id`, `question_id`, `paper_id_a`, `paper_id_b`, the derived
label, the stratum, and which §0.3 signal fired. The label file carries `pair_id`, `derived_label`,
`annotator_label`, `stratum` and `reason` — mirroring Phase 5's schema so the two batches are
directly comparable.

---

## §8 — Implementation notes

**Reused unmodified:** `evaluate_gate1`, `evaluate_gate2`, `wilson_interval`, `parse_annotations`
and `ANNOTATOR_LABELS`, all from `biolit_evals/annotation_export.py`. This is the point of the
design; the instrument is already tested and calibrated, and touching it would forfeit that.

**New:** a pair-manifest builder and a stratified sheet exporter. Phase 5's `export_blind_sheet`
cannot be reused — it is CTD-specific, taking `GoldPair` with `chemical_id`/`disease_id`. The
markdown block format the parser expects is preserved exactly (a `## {i}.` heading carrying the
backticked `pair_id`, then a fenced block with `label:` and `reason:` lines), so `parse_annotations`
works against the new sheet without modification.

**Conventions:** TDD with tdd-guard on. Ruff `E,F,I,UP,B`, line length 100. `StrEnum` per ADR-0005.
`datetime.now(UTC)`. Imports at top of file. `main()` gets no direct unit test. Unit tests touch no
network — abstract fetching is exercised against fixtures only. All commands run from `backend/` via
`uv run`.

**Cost: zero.** Abstracts come from free NCBI `efetch` (all 254 already cached from the §0 probes).
There is no LLM call, no credential requirement, no pricing step and no authorization gate.

---

## §9 — Explicitly out of scope

- **Corpus size.** Deferred by instruction until π̂ returns.
- Any Critic arm, baseline, or scoring harness. ADR-0013: no infrastructure without a demonstrated
  consumer, and the consumer here is not demonstrated until §6 returns a `CONTINUE` on π̂_C.
- Any paid run.
