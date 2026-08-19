# Phase 5 — Contradiction detection (the Critic): eval design

- **Date:** 2026-08-19
- **Status:** Design approved; no implementation started
- **Supersedes:** nothing. **Depends on:** ADR-0010 (both arms run the production path),
  ADR-0013 (entity-conditioned ceilings; cluster-size capping deferred), ADR-0014 (operand
  coverage), ADR-0015 (a restricted-population result must report its baselines).

## Summary

Phase 5 asks whether an LLM Critic can judge that two papers disagree about a chemical's
effect on a disease. Unlike every prior phase, **BC5CDR supplies no gold labels for this
task**, so the design's centre of gravity is the gold standard rather than the agent.

Gold is derived from CTD's `DirectEvidence` field, which records for each curated
chemical–disease relation whether a paper asserts `marker/mechanism` (the chemical causes or
aggravates the disease) or `therapeutic` (it treats it), together with the PMIDs asserting
it. Two papers taking opposite directions on the same `(chemical, disease)` key are a
contradiction candidate.

The measured facts that shaped every decision below are in §0. The two that mattered most:
**BC5CDR's own gold CID relations can never contain a contradiction** (they are all
`marker/mechanism` by definition), and **only 6 opposed pairs exist inside Test-500** — so
Phase 5 cannot run on the corpus every prior phase used, and its corpus is purpose-built
from CTD with abstracts fetched from PubMed.

---

## §0 — What was measured before anything was designed

All figures below come from throwaway probes against the real artifacts, not from
assumption. CTD snapshot stamped `Report created: Thu Jul 30 13:59:07 EDT 2026`.

### CTD direct evidence

| Quantity | Value |
|---|---:|
| Direct-evidence rows carrying PMIDs | 109,591 |
| Distinct PMIDs with direct evidence | 84,328 |
| BC5CDR PMIDs present in CTD direct evidence | 1,499 / 1,500 |
| Papers assigned **both** directions on one key | **0** |

The last row is why the label function is unambiguous and needs no tie-break.

### BC5CDR cannot supply contradictions — a structural zero

Joining CTD directions onto BC5CDR's own gold CID keys:

| Split | co-keyed pairs | opposed | agreeing | direction unknown |
|---|---:|---:|---:|---:|
| Test-500 | 225 | **0** | 215 | 10 |
| Train | 192 | **0** | 189 | 3 |
| Dev | 254 | **0** | 246 | 8 |

This is not a data gap. **BC5CDR annotates chemical-*induced*-disease relations only**, so
every gold CID key is `marker/mechanism` by construction and `therapeutic` cannot appear.
A contradiction is impossible by definition of what the corpus annotates.

This is the same shape ADR-0015 was forced to retract a claim over — a comparison whose
outcome is fixed before any measurement — caught here before anything was built on it.

### Where contradictions actually live

| Population | Opposed pairs |
|---|---:|
| Both papers in Test-500 | **6** |
| Both papers in BC5CDR-1500 | 40 |
| Exactly one paper in BC5CDR-1500 | 3,184 |
| **All of CTD** | **72,045** (3,827 keys, 24,945 papers) |

Test-500 is not viable. The corpus must be built from CTD.

### Prevalence and skew

| Quantity | Value |
|---|---:|
| Opposed co-keyed pairs | 72,045 |
| Agreeing co-keyed pairs | 2,784,307 |
| **Contradiction prevalence among co-keyed pairs** | **0.0252** |
| Top-5 keys' share of opposed pairs | 0.119 |
| Keys contributing ≤ 20 pairs each | 3,321 keys / 13,617 pairs |

Skew is mild — 0.119 against the 0.503 `top5_pair_share` ADR-0013 recorded for
same-sentence clusters — so a per-key cap is cheap to impose.

### After excluding all 1,500 BC5CDR PMIDs

| Quantity | Before | After |
|---|---:|---:|
| Opposed keys | 3,827 | 3,732 |
| Opposed pairs | 72,045 | 68,821 (−4.5%) |
| Prevalence | 0.0252 | 0.0252 |

Opposed pairs available under a per-key cap: **1 → 3,732**, 2 → 6,364, 3 → 8,371,
5 → 11,477, 10 → 16,755.

Pairs obtainable with **one pair per key and no paper reused anywhere**: **2,759**
(2,759 distinct keys, 5,518 distinct papers) — 9× the 300 required.

Hard negatives available (two papers sharing a chemical, no curated relation between
them): **> 200,000**.

---

## §1 — Corpus construction and exclusion

### Source and parsing

`CTD_chemicals_diseases.tsv.gz`, **`DirectEvidence` rows only**. The remaining ~13M rows are
gene-inferred associations with no paper asserting the relation and are discarded. A new
setting `ctd_chemicals_diseases_url` joins the two CTD URLs already in `config.py`.

Two parsing rules, each of which produced a wrong answer during probing and each of which
therefore gets a test:

1. **Header selection must be by content, not position.** CTD's column header is a `#`
   comment line *followed by further* `#` lines, so a "last comment wins" rule yields an
   empty header and zero parsed rows. The header is the comment line containing both
   `ChemicalName` and `DirectEvidence`.
2. **ID namespaces differ across sources.** CTD writes `ChemicalID` bare (`C046983`) and
   `DiseaseID` prefixed (`MESH:D054198`); BC5CDR prefixes both. Joining without
   normalisation yields zero matches.

Rule 2 is the dangerous one: **an unnormalised join reports zero contradictions, which is
indistinguishable from a true negative result.** During probing this produced a plausible
"0 contradictions" reading twice, for two different reasons. The builder therefore carries a
**positive-control anchor** — a fixture known to join must join — so a normalisation
regression fails loudly instead of publishing an empty corpus as a finding.

### Exclusion policy

**All 1,500 BC5CDR PMIDs are excluded**, not only the 500 training documents. The NER
checkpoint was fine-tuned on the training split and the dev split was plausibly used for
selection; the measured cost of removing all three is 4.5% of opposed pairs with prevalence
unchanged. Paying 4.5% to retire the contamination question entirely is the correct trade,
and it makes Phase 5 the first phase not confined to Test-500.

### Sampling

Three constraints, all verified affordable in §0:

1. **One pair per `(chemical, disease)` key** — 300 contradiction pairs drawn from 300
   distinct keys, so no single drug dominates the eval.
2. **No paper appears in more than one pair.** This makes the pairs independent trials, so
   binomial confidence intervals are valid rather than understated. This constraint is
   load-bearing for §3's statistics, not cosmetic.
3. **Balanced across the three classes**, 300 each — forced by the 2.52% prevalence.

Class definitions:

- **`contradiction`** — opposed directions on a shared key.
- **`agreement`** — the same single direction on a shared key.
- **`insufficient_overlap`** — **hard negatives**: two papers sharing **at least one endpoint**
  (the same chemical, or the same disease) with no CTD-curated relation joining them, drawn
  from both sub-populations. Randomly drawn unrelated pairs are rejected as trivially
  separable; a shared endpoint tests whether the Critic evaluates the *relationship* rather
  than pattern-matching shared entities. Availability was measured on the shared-chemical
  sub-population alone (> 200,000), which is already three orders of magnitude above the 300
  required.

### Reproducibility — CTD is a living database

Every prior eval in this project downloaded a frozen artifact. **CTD is republished
continuously**, so a later rebuild would silently resample.

Therefore **the committed manifest is the frozen artifact, not CTD**:
`evals/gold/contradiction_pairs.jsonl`, one line per pair carrying both PMIDs, the key, the
gold label, and the CTD directions it was derived from, written in **seeded-shuffled order**.
The CTD release stamp is recorded in the run log. Rebuilding from a newer CTD is an explicit
operation producing a new manifest, never an implicit re-draw.

Abstracts are **fetched, never committed**, through the `biolit/clients/pubmed.py` client
unused since Phase 1, cached under gitignored `data/`. Same posture as BC5CDR, and it avoids
committing copyrighted text.

### Abstract availability and the pre-committed drop-rate rule

PubMed abstract availability across CTD's decades-spanning PMIDs is **unmeasured**. The
design absorbs it by drawing a seeded, pre-ordered candidate pool of **900 per class** (3×
target) *before any fetching*, fetching the pool, dropping pairs where either abstract is
missing, and taking the first 300 per class in the recorded order. Topping up means consuming
more of a pre-ordered list; it never means re-drawing.

The rule is stated in terms of **resulting N**, because N is what threatens the eval and
defining materiality on the drop rate would leave the judgement call exactly where it must
not be:

| Outcome | Action |
|---|---|
| All classes reach 300 | Proceed as designed. |
| A class lands at **200–299** | **Accept the smaller N, document it, leave other classes at 300.** Do not downsample the others. |
| A class falls **below 200** | Top up from the pre-committed pool extension; record the differential filtering as a limitation. |

**Why 200 is the floor:** at p = 0.5 the 95% interval is ±0.069 at N = 200 against ±0.057 at
N = 300, which still separates arms differing by ~0.10 — the size of gap Phase 4 produced. At
N = 100 the interval is ±0.098 and the eval can no longer distinguish anything worth
distinguishing.

Metrics are **macro-averaged** so unequal class sizes cannot silently reweight the headline.

**The confound topping-up creates, recorded in advance:** if abstract availability correlates
with publication era or indexing quality, filtering one class harder makes its papers
systematically unlike the others' in a way unrelated to the label. **Per-class abstract-length
and publication-year distributions are therefore reported regardless of outcome**, and
divergence is flagged as a confound rather than discovered afterwards.

### Anchors (halt the run)

- No sampled PMID appears in BC5CDR's 1,500.
- No paper appears in two pairs.
- No key contributes more than one pair.
- Class counts match the manifest exactly.
- The fetched corpus matches the committed manifest exactly.
- The label function re-derives the manifest's labels exactly from the recorded CTD
  directions (the positive control, in its final form).

---

## §2 — Gold definition and hand-validation

### The label function

For papers *a*, *b* and key *k = (chemical, disease)*:

- **`contradiction`** — CTD assigns *a* `{marker/mechanism}` and *b* `{therapeutic}` on *k*,
  or the reverse.
- **`agreement`** — both carry the same single direction on *k*.
- **`insufficient_overlap`** — they share one endpoint and CTD curates no key joining them.

Unambiguous because no paper is ever assigned both directions on one key (§0, 0 cases).

### The proxy's threat model

The proxy assumes "*a* says C causes D" plus "*b* says C treats D" constitutes a
disagreement. Four ways that fails, in descending order of estimated likelihood:

1. **CTD curates the relationship studied, not the conclusion reached.** `therapeutic`
   records that the paper concerns C as a treatment for D. A paper reporting that C *failed*
   to treat D may carry the same tag as one reporting success, in which case the pair is not a
   contradiction and may even be an agreement. Partially mitigated by CTD curating positive
   assertions — but that is a belief about curation practice, not a measurement. **This is
   the primary reason the hand-validation exists.**
2. **Both true in different contexts** — dose, population, duration. A chemical causing D in
   overdose and treating D at therapeutic dose does not contradict itself.
3. **Broad MeSH disease terms** spanning clinically distinct conditions.
4. **Temporal supersession** — a 1985 finding overturned in 2005 is a real disagreement, but
   arguably about the state of evidence rather than between the papers.

### Two-tier gold

**CTD gold gives scale** — 900 pairs, enough to compare arms and beat baselines with tight
intervals. **It does not give validity**, for a reason that must be stated before any number
is read:

> **CTD's contradiction label *is* a direction flip.** A Critic that merely detects direction
> flips therefore scores ≈ 1.0 on this class, while a Critic reasoning correctly about
> contradiction scores only **π**, the fraction of those pairs that are genuine
> disagreements.

Two consequences follow, and both invert the naive reading:

- **Recall near 1.0 on the CTD contradiction class is evidence of proxy-mimicry, not of
  quality.**
- The direction-lexicon free baseline (§3) should score *very high* here. If it matches the
  LLM arm, the correct conclusion is that CTD gold measures cue-matching rather than
  reasoning — a finding, not a failure.

**π is a ceiling on honest performance, not a correction factor.** No measured score is
divided by it, rescaled by it, or otherwise adjusted. It is reported beside the number as the
level a correct Critic is expected to reach.

The human-labelled subsample is therefore **where quality is actually measured**; CTD gold
supplies scale, baselines, and arm-vs-arm comparison.

### Hand-validation protocol

Blind, following the ADR-0006 `domain_sample.jsonl` precedent. The annotator sees both
abstracts and the key, **not** the CTD label, and picks one of the three labels or
`cant_tell`, with a free-text reason. Annotation happens **before** any LLM output exists, so
nothing anchors on an arm's answers.

**Target: 100 contradiction pairs plus 50 spread across the other two classes.** The 50 are
not padding — without them there is no way to distinguish a genuinely high π from a protocol
that rubber-stamps whatever it is shown.

**Staged, with a tractability gate:** a first batch of ~30 mixed across classes is annotated
to confirm the blind protocol is workable before the remaining ~7 hours are committed.
Annotation may stop early if the time budget does not stretch; **whatever N is reached is
documented exactly as the drop-rate rule documents a smaller-than-target class.** The
achieved N and its interval are reported, never a target N.

Interval on π at various sizes (at π ≈ 0.8):

| Contradiction pairs annotated | 95% CI on π |
|---:|---:|
| 30 | ±0.143 |
| 60 | ±0.101 |
| 100 | ±0.078 |
| 150 | ±0.064 |

---

## §3 — Harness and arms

### Placement

`biolit/critic/` in the production package, following ADR-0015's precedent for
`extract/llm.py` rather than ADR-0011/0012's unmerged-branch handling: these are the arms
under measurement, the reproduction recipe runs them, and the client is taken as `Any` with
no `anthropic` import, so `biolit` gains no dependency.

**Pre-committed removal condition:** if every arm is rejected, the standing consumer rule
applies and `biolit/critic/` is the first thing removed. Recorded now so it is not
re-litigated later.

- `critic/base.py` — the `Critic` protocol, injected keyword-only like `Linker`,
  `PairingStrategy` and `Extractor`.
- `critic/llm.py` — pair judgment.
- `critic/direction.py` — per-paper direction labelling, composed into pair labels.

Baselines are comparators rather than shipping candidates and live in
`biolit_evals/critic_baselines.py`, alongside `baselines.py`.

### The two input arms are a data difference, not a code difference

Both arms run the **same** `LlmCritic` with the **same** prompt; only the text differs:

```
CriticInput(paper_id, text)   # text = full abstract | joined finding sentences
```

If the arms used different prompts, any gap would confound extraction quality with prompt
wording — and pricing `biolit.extract` at a real consumer is the entire point of the
comparison. The findings arm runs the **production path** (real NER → canonicalisation →
`SameSentenceAsEntitiesExtractor`) per ADR-0010, not a gold-only shortcut.

**Zero-finding policy, decided in advance:** when the extractor returns nothing, the arm
receives empty text and must answer from nothing. It does **not** fall back to the abstract.
A fallback would silently convert the findings arm into the abstract arm on exactly the
papers where extraction failed — the failure mode most likely to flatter it.

**The zero-finding rate is reported per gold class as well as in aggregate**, for the same
reason the drop rate is: it is a nuisance variable that may correlate with class. Concretely,
if therapeutic-direction abstracts name drugs and diseases more explicitly, the entity-gated
extractor fires more often on them and the findings arm looks better on `agreement` for
reasons unrelated to judging agreement.

### Free baselines

Per ADR-0015's standing rule the LLM arm must clear the **best** of these, not the majority
class:

1. **Majority class** — 1/3 on balanced gold by construction; required for the
   natural-prevalence projection.
2. **Direction lexicon** — cue matching (`induced by`, `caused`, `-induced` against
   `treatment of`, `therapy`, `efficacy`) per paper, composed into a pair label. **This is
   Phase 5's `first 4`** — the free heuristic that decides whether the LLM earns its cost.
3. **Concept overlap** — shared canonical IDs, separating `insufficient_overlap` from the
   rest.

**Pre-registered expectation:** baseline 2 should score high on CTD gold, because CTD's
contradiction label *is* a direction flip. Recorded before running so the outcome cannot be
rationalised afterwards.

### Metrics

- **Macro-F1 over the three classes** as primary — macro because the drop-rate rule permits
  unequal N and micro would let a larger class dominate.
- Per-class P/R/F1 and the full confusion matrix; **contradiction-class precision and
  recall** called out as the headline pair.
- **Natural-prevalence projection**: measured sensitivity and specificity carried to 2.52%
  prevalence, reported beside the balanced figure. Worked illustration: at 80% sensitivity
  and 95% specificity, precision is **0.29** — seven in ten flagged contradictions would be
  false. Balanced accuracy must never be read as deployment performance.
- Binomial CIs on every rate, valid **because pairs share no papers** (§1).
- Agreement with human gold on the validated subsample, reported separately with π as the
  stated ceiling.

### Paired comparison — scope stated explicitly

**McNemar's test applies to every arm-vs-arm comparison on the CTD gold**: the two input-mode
arms, the task-decomposition arm, and the three free baselines alike. The licensing property
is that every arm emits a label for every one of the same pairs — the direction arm's
per-paper outputs are composed into pair labels *before* scoring, so once composed it is the
same 900 units. The test runs on the binary correct/incorrect indicator per pair, which is
standard for a multi-class task, and applies equally when restricted to the contradiction
class and to the human-labelled subsample at their smaller N.

**The one exception:** the direction arm's *per-paper* diagnostics — its error-localisation
advantage — are **not** paired against the pair arms, because the unit is a paper rather than
a pair. They are reported on their own terms and never as a paired comparison.

### Refusals and parse failures — pre-committed scoring

Phase 4 measured zero refusals, but that task was "which sentences state findings". This one
asks a model to **adjudicate claims about drug harm**, so the refusal rate is genuinely
uncertain.

Scoring is fixed in advance and **both numbers are always reported**: macro-F1 **excluding**
refusals, and macro-F1 **counting refusals as wrong**. Reporting only the first inflates the
arm; only the second conflates capability with policy. Refusals and unparseable outputs are
recorded with their rates, never silently dropped.

### Run log

`evals/critic_runs.jsonl`, one JSON line per run via the existing `_meta.py` conventions,
carrying: arm, model, prompt version, seed, N per class, macro-F1, per-class P/R/F1,
confusion matrix, refusal and parse-failure rates, zero-finding rate (aggregate and per
class), **input/output token counts, measured cost**, the **CTD release stamp**, and a
**manifest hash**.

### CLI

```
python -m biolit_evals.critic_eval --arm {abstract,findings,direction,lexicon,majority,overlap} [--limit N]
```

The three free arms need no credential and run free.

---

## §4 — Pricing

### What is exact and what is not

The **call count is exact**, because §1's disjointness constraint fixes it: 900 pairs sharing
no papers means exactly 1,800 distinct papers.

| Arm | Unit | Calls |
|---|---|---:|
| Pair judgment, full abstract | pair | 900 |
| Pair judgment, findings only | pair | 900 |
| Direction decomposition | paper | 1,800 |
| **Total** | | **3,600** |

**Token counts are not known and no dollar figure is committed in this document.** 3,600
calls is more than double Phase 4's ~1,500, and pair calls carry two abstracts rather than
one, so any intuition anchored on Phase 4's measured $2.48 is anchored on a smaller and
cheaper workload. A previously circulated "$1–3" figure is explicitly **not** a planning
number.

### The pilot

`--limit 30` on each of the three LLM arms. **`--limit` counts *pairs* for every arm**, so the
direction arm makes two calls per pair: 30 + 30 + 60 = **120 calls**. It records per-call
input and output tokens with their spread, refusal rate, parse-failure rate, and latency.

`--limit` is a valid random subsample because the manifest is written in **seeded-shuffled
order**, so head-N is representative rather than key-ordered. Pilot runs are tagged
`pilot: true` and anchored to their limited size, so a pilot can never be mistaken for a full
run — the `assert_dataset_size` precedent from ADR-0013, applied here.

### The authorization figure — named method

Two distinct sources of error, handled separately, because they demand different remedies.

**(a) Statistical uncertainty — a one-sided 95% prediction bound on the realized total.**

A per-call percentile multiplied by N is the wrong estimator: the bill is a **sum** of ~3,600
draws, which concentrates, so assuming every call is a 95th-percentile call bounds a scenario
that cannot occur. The correct object is a prediction bound on the sum, which carries both
the uncertainty in the estimated mean and the sampling variation of the realized total.

Per arm *a*, with pilot sample mean cost `m_a`, pilot sample SD `s_a`, pilot size `n_a`, and
full-run call count `N_a`:

```
point estimate     T̂  = Σ_a  N_a · m_a
variance           V   = Σ_a  s_a² · ( N_a² / n_a  +  N_a )
AUTHORIZATION BOUND    = T̂ + t(0.95, min_a n_a − 1) · sqrt(V)
```

The `N_a²/n_a` term is the uncertainty in the estimated mean; the `+N_a` term is the
realized sum's own variation. Arms are computed separately and combined through the variance
because their cost distributions differ materially (two abstracts against one, full text
against extracted sentences); pooling them would misstate the spread.

**95%, one-sided, with Student's *t*** at the smallest arm's pilot degrees of freedom
(`t(0.95, 29) = 1.699` at the planned pilot size). 95% rather than 99% because it is paired
with the hard kill-switch below, which is a stronger safeguard than a wider interval. The
normal approximation is sound here: per-call cost is bounded above by the context limit, so
the distribution has no heavy tail and the CLT applies well at n = 30–60 per arm.

At the planned pilot size this bound lands **+7.9% above the point estimate at CV = 0.5 and
+15.8% at CV = 1.0** — computed, not asserted. A larger pilot tightens it slowly
(`--limit 40` gives +6.8% / +13.6% for 40 more calls), which is why 30 is the chosen size:
the bound is dominated by the realized sum's own variation, not by pilot precision.

**(b) Systematic error — which the bound above does NOT cover, and this is the point.**

Phase 4's committed estimate was **24% low**. A 95% prediction bound of +8–16% would not have
caught it, because that miss was **bias, not variance** — and no choice of percentile fixes a
measurement bug. Widening the interval until it covers a bias would be the wrong repair: it
would mask the defect while inflating every future estimate. Bias is therefore eliminated at
the source rather than padded for:

- Token counts are read from the **API's own reported usage fields**, never estimated from
  text length.
- **Every** call is counted, including retries, refusals, and parse failures.
- Prices are pinned to the model's published rates at run time and recorded in the run log
  alongside the token counts, so a later price change cannot silently invalidate the figure.
- **Post-run reconciliation:** actual spend is compared against the authorization bound and
  logged. A breach is recorded as a finding about the estimator, not quietly absorbed.

**Kill-switch:** the run aborts if cumulative measured spend exceeds **1.25 ×** the
authorization bound, so a systematic error cannot run away across 3,600 calls.

### Build order

| # | Step | Cost |
|---:|---|---|
| 1 | Corpus builder, manifest, anchors | free |
| 2 | Abstract fetch; drop rate by class | free |
| 3 | Free baselines, metrics, run log | free |
| 4 | Human annotation, first batch (~30, mixed) | annotator time |
| 5 | **Pilot** (120 calls) | smallest paid step — **requires authorization** |
| 6 | **Authorization gate** — measured tokens, bound, refusal rate, baseline scores | — |
| 7 | Full run | **requires authorization** |
| 8 | Human annotation to target N | annotator time |
| 9 | Report and ADR | free |

**Step 3 may retire step 7's premise.** If the direction-lexicon baseline scores near-ceiling
— which §3 pre-registers as likely — the paid run's purpose changes *before* it is bought,
from "can the LLM do this" to "can the LLM beat cue-matching on the human-labelled
subsample", which is a different and much cheaper question, possibly answerable at N ≈ 150
rather than 900. Staging exists so the free half of the work can retire the expensive half.

### The gate

**No paid call runs without explicit authorization, including the pilot.** At step 6 the
decision is made on measured tokens, the extrapolated bound, the refusal rate, and the free
baselines' scores — never on a figure written in advance.

---

## §5 — Limitations, recorded before any result exists

1. **The gold population is shaped toward direction-flip disagreements.** It is constructed
   entirely from CTD's direction field, so the eval contains **no true positives** of the
   dose, population, or effect-size kind. Performance on those is **untested, not merely
   weaker** — nothing in this design can confirm or deny it. This is distinct from, and
   additional to, limitation 2.
2. **The proxy's own error rate is unknown until measured.** How often an opposed direction
   pair is not a genuine disagreement is estimated by the §2 hand-validation, at the interval
   its achieved N supports.
3. **Balanced sampling overstates deployment performance.** Natural prevalence among co-keyed
   pairs is 2.52%; the projection in §3 is the honest reading. This project has recorded four
   prior instances of a favourably-selected population reporting a better number than the
   system would pay.
4. **CTD is a living database.** Results are reproducible only against the committed manifest
   and the recorded release stamp, not against a fresh CTD download.
5. **Abstract-availability filtering may differ by class.** Mitigated by the §1 rule and the
   per-class distribution reporting, not eliminated.
6. **The prompt is fixed and unswept.** As in ADR-0015, this measures one prompt. Prompt
   iteration is a separate experiment requiring its own baseline comparison.
7. **`insufficient_overlap` gold is negative evidence.** It asserts CTD curates no relation
   between two papers, which is not the same as no relation existing. Curation gaps
   contribute an unmeasured share.

---

## §6 — Open items for the implementation plan

- Per-class abstract availability (§1) — measured in step 2, feeds the drop-rate rule.
- Achieved annotation N (§2) — determined by the staged protocol.
- Per-call token distributions (§4) — measured by the pilot, feeds the authorization bound.
