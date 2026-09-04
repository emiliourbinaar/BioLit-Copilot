# Synthesis Gate A — does an LLM cluster characterization earn its place over a deterministic template?

- **Date:** 2026-09-03
- **Status:** Design approved 2026-09-03. **No implementation started. No API call authorized** —
  the pilot and the full run are each their own approval step (§7).
- **Kind:** Measurement. A baseline gate run *before* the stage it would justify is designed, per
  ADR-0015's precedent that a paid arm is priced against a free control first.
- **Depends on:** ADR-0015 (an arm scored on a population it cannot lose on proves nothing; every
  paid arm is priced against a budget-matched free control), ADR-0017 (an unmeasurable but
  quotable metric is the failure the harness exists to prevent), ADR-0018 (the Critic is closed;
  clusters carry no epistemic status), ADR-0013 (no infrastructure without a demonstrated
  consumer).
- **Does not decide:** whether Synthesis ships at all, what the `question`/`answer` assembly layer
  looks like, or whether population/endpoint extraction is worth building. All three are
  downstream of this result.

## Summary

ADR-0018 closed the Critic permanently, so clusters reaching Synthesis carry **no
agreement/contradiction signal, and never will under this pipeline's design**. Synthesis is
therefore scoped to **characterizing a cluster** — saying what a group of papers reports about
one entity pair — and explicitly *not* to answering the user's question as a relevance-scored
judgment, because relevance is the unmeasurable-but-quotable metric ADR-0017 exists to forbid.

Before any LLM Synthesis stage is designed, Gate A asks the ADR-0015 question: **does an LLM arm
beat a deterministic template on the content shape the pipeline can actually support?** The
expected answer is no, and that outcome is pre-committed as legitimate rather than as a failure.

---

## §0 — What was measured before anything was designed

Read off the real modules on 2026-09-03, not assumed from the data model's field names.

### §0.1 What Synthesis can actually reach

| field | state |
|---|---|
| `Cluster` | `key` + `paper_ids`. No scores, no per-paper structure. |
| `ExtractedRecord.key_findings` | **populated** — `SameSentenceAsEntitiesExtractor` emits `Finding(text, start, end, sentence_index)` via `biolit/extract/base.py:57` |
| `ExtractedRecord.entities` | **populated**, carrying `canonical_id` from Phase 3 |
| `Paper` | **populated** — `year`, `journal`, `title`, `mesh_terms`, `authors`, `pmid` |
| `ExtractedRecord.study_type` | ⚠️ **declared, no producer anywhere in `src/`** |
| `ExtractedRecord.sample_size` | ⚠️ **declared, no producer anywhere in `src/`** |
| `Citation` | ⚠️ **declared, never constructed** |
| `PipelineState.contradictions` | permanently empty (ADR-0018) |

The three warned rows are the same shape as `PipelineState.clusters` before the pipeline was
wired — fields that read as available signal and are not. **Any design leaning on them is
proposing a new extraction phase without saying so**, which is why option 3 (population/endpoint
extraction) is held as a conditional next phase with its own Phase-4-style measurement rather
than folded in here.

### §0.2 Cluster sizes are extremely skewed, so the sample must be stratified

From the clustering eval (`evals/cluster_runs.jsonl` at `7430c6e`): `top5_pair_share` on Arm B
`same_sentence` is **0.503** — half of all cluster comparisons come from five clusters — the
largest Arm B cluster holds **11 papers**, and the largest Arm A cluster **28**.

An unstratified sample of clusters would be almost entirely small ones. A 2-paper cluster and a
12-paper cluster are different tasks, and a metric averaged across them without regard to size
hides that. This is also why the cluster-size cap measured in Phase 3 was never built: it had no
consumer. **Synthesis is the first candidate consumer**, but the cap stays unbuilt until Gate A
says there is a stage to protect.

---

## §1 — What Gate A asks, and a tautology warning stated first

⚠️ **On faithfulness and coverage the template scores 1.0 by construction.** A template that
emits source sentences verbatim is perfectly faithful and perfectly covering *definitionally*,
not because it is good. Scoring the two arms against each other on those axes and declaring the
template the winner would be **ADR-0015's tautology arriving from the opposite direction**: a
score one arm cannot lose. ADR-0015's retraction and ADR-0017's A2 hazard are both this shape,
and recognising it a third time, in advance, is the whole reason it is written here.

So the comparison is deliberately asymmetric:

- **Faithfulness and coverage are DISQUALIFIERS for the LLM arm, not scores.** The template
  clearing them carries no information.
- **The only comparative axis is compression against retention** — saying the same thing shorter
  while preserving what distinguishes each paper. That is the one thing an LLM can do that a
  template structurally cannot.

**No second comparative axis is added.** Readability, coherence, helpfulness and "would a person
prefer this" are ADR-0017's forbidden metric under new names, and adding one would reintroduce
exactly the failure this gate is placed to prevent.

---

## §2 — Metric suite: deterministic, no annotation

A **content unit** is a numeral, a canonical entity mention (`canonical_id` from Phase 3), or a
paper reference (PMID / year+journal).

| metric | definition | role |
|---|---|---|
| **Support rate** | fraction of output content units appearing in the cluster's source (`key_findings` + `Paper` metadata), **or** equal to the cluster's paper count | disqualifier |
| **Entity hallucination** | count of output entity mentions absent from source — reported separately because it is the *dangerous* failure rather than merely a sloppy one | hard disqualifier |
| **Coverage** | fraction of cluster papers carrying at least one identifying reference in the output | disqualifier |
| **DCR** — distinguishing-content retention | per paper, the content tokens unique to it *within its own cluster*; the paper is retained if at least one survives into the output | comparative |
| **Compression** | output length ÷ concatenated source-findings length | comparative |

### ⚠️ A known bias in this suite, pre-registered rather than discovered afterwards

Support is strict, and strictness penalises **exactly what the LLM is for**: legitimate
aggregation ("five of six measured HbA1c") that the metric cannot verify. Building a test that
is biased against one arm is the same family of error as scoring an arm on a population it
cannot lose on — the bias simply points the other way.

The mitigation is reporting, not a looser threshold: **unsupported units are reported by
category, never only as a rate.** A support rate of 0.90 composed of hallucinated entities and
one composed of unverifiable-but-true aggregates are different results, and collapsing them
would let a limitation of the metric masquerade as a failure of the arm.

### ⚠️ A second known bias, on the deciding axis — added 2026-09-04, before any arm was run

Found while implementing the harness, and recorded here rather than in a footnote to the
results, because it lands on DCR: one of only two comparative axes, and the number this gate
actually decides on. **No arm has been run and nothing has been spent at the time of writing**,
which is what keeps this a pre-registration rather than a post-hoc excuse.

DCR counts a paper retained when one of its distinguishing tokens survives *as a token*. That
rewards copying. The deterministic template reproduces finding sentences nearly verbatim and so
cannot lose; an LLM that paraphrases faithfully can preserve every fact and still score lower.
Measured on two findings — `Metformin reduced hirsutism scores.` and `Metformin reduced
ovulation latency.` — the template scores `retained=2, scorable=2` while a paraphrase carrying
both facts (`decreased androgenic symptom severity ... shortened the time to ovulation`) scores
`retained=1, lost=('p1',)`.

**The bias favours the arm this gate already defaults to shipping**, so it cannot be waved
through as conservative. It is the mirror image of the substring-matching error the harness
rejected during implementation, which ran the other way, toward the LLM.

The mitigation is again reporting, not a looser threshold: **`lost` is reported per paper, never
only as a rate**, and a DCR gap composed of genuine dropped content is a different result from
one composed of faithful paraphrase. A reading that treats a DCR difference as decisive without
inspecting `lost` is not supported by this design. Semantic retention is not measurable without
annotation, which is the whole reason Gate A exists in this form (§1) — so the honest move is to
name the limit, not to engineer around it.

---

## §3 — Pre-registered decision rule, written before any number exists

### Disqualifiers — LLM arm only

| check | threshold |
|---|---|
| support rate | ≥ 0.95 |
| entity hallucinations | **= 0** |
| coverage | ≥ 0.90 |

### The comparative test

The LLM must sit strictly outside the template on the compression/DCR frontier: **higher
compression, at a DCR loss no greater than the tolerance in §3.1.**

### §3.1 The DCR tolerance, derived — and why a flat fraction was the wrong shape

The draft of this design carried a flat tolerance of 0.10 pooled DCR. Deriving what that
actually permits changed the rule, and the derivation is recorded here rather than presented as
self-evident:

> Pooled over the sampled papers, a flat 0.10 constrains only the *total* loss and says nothing
> about **where** it lands. An arm could erase both papers of five separate 2-paper clusters —
> ten papers, roughly 0.06 pooled — and pass comfortably, having produced five characterizations
> that characterize nothing. A pooled fraction cannot see this, because a 2-paper cluster and a
> 12-paper cluster contribute to it identically.

The tolerance is therefore stated **in papers, per cluster**:

> **The LLM may erase the distinguishing content of at most ONE paper per cluster, and NONE in
> clusters of ≤ 3 papers.**

The reasoning for the small-cluster exemption is that losing one of two papers is not a degraded
characterization but an absent one. The ceiling is **20 papers** — one from each of the 20
non-small clusters, zero from the 10 small ones. That works out near 0.11 pooled, so the
original 0.10 was about the right magnitude and simply could not express the constraint that
mattered. **The exact pooled equivalent is computed from the frozen sample and recorded then,
not asserted now.**

### Outcomes

| result | action |
|---|---|
| LLM fails any disqualifier | **Ship the template.** LLM arm rejected on faithfulness. |
| Clears disqualifiers, misses the frontier | **Ship the template.** No demonstrated advantage — ADR-0013. |
| Clears both | LLM arm has a measured advantage. Design the real stage in its own spec. |

⭐ **"Ship the template" is pre-committed as the expected outcome and a legitimate result.** No
threshold moves to admit an observed number; if a threshold fires, the harness is right and the
design is wrong.

---

## §4 — The deterministic control

```
## metformin — polycystic ovary syndrome
6 papers, 2003–2019.

- 2007 · N Engl J Med · PMID 17517762
  "Clomiphene is superior to metformin in achieving live birth in
   infertile women with PCOS."
- 2019 · Hum Reprod · PMID 30649412
  "Metformin reduced ovarian hyperstimulation syndrome incidence."
```

Sorted by year then PMID, so ordering is deterministic and carries no implicit ranking. Papers
with multiple findings list all of them. **Papers with zero extracted findings are printed with
an explicit `(no finding sentence extracted)` marker rather than dropped** — `zero_findings` is
already a tracked ledger key, and silently omitting those papers would inflate the control's
coverage against the very metric coverage is meant to measure.

---

## §5 — The LLM arm

Same inputs, nothing more. Follows the established shape of `biolit/extract/llm.py`: a `_SYSTEM`
constant, a JSON `_SCHEMA`, `_MAX_TOKENS = 16000` (the adaptive-thinking cap applies here too —
a small budget is spent on thinking and returns `stop_reason: max_tokens` with partial content),
and `USAGE_FIELDS` for token accounting. **No prices in code**, per the same rule: a rate
committed to the repo rots silently into a wrong cost estimate.

```
You characterize what a group of biomedical papers reports about one
entity pair. You will receive the pair and, for each paper, its year,
journal, PMID, and the sentences stating its findings.

Write a short characterization of the group. State what the papers
report, preserving what distinguishes each one from the others.

Do NOT judge whether the papers agree or disagree with each other. Do
not say findings are consistent, conflicting, contradictory, or mixed.
That judgment is out of scope and is not supported by this pipeline.

Use only what you are given. Every number and entity you write must
appear in the input, except a count of the papers themselves. Do not
add background, mechanism, or clinical recommendation.
```

### §5.1 Anti-judgment compliance — logged, not scored

**An LLM asked to characterize a cluster will volunteer agreement language unprompted**, and any
such claim is exactly what ADR-0018 established this pipeline cannot support. The rate at which
the output uses agreement/disagreement vocabulary is therefore recorded as a diagnostic on every
run — **including the pilot report (§7), where it must appear explicitly** — and is deliberately
**not** part of any threshold. It measures instruction compliance, not characterization quality,
and folding it into a score would give the gate a second comparative axis by the back door.

---

## §6 — Corpus and sampling

Run the free pipeline over a fixed query set; freeze the resulting clusters with a manifest hash
and a **required** `--seed` (never defaulted, per the Phase 5 and Alamri convention). Then sample
**30 clusters, stratified by size**:

| band | papers per cluster | clusters sampled |
|---|---|---|
| small | 2–3 | 10 |
| medium | 4–7 | 10 |
| large | 8+ | 10 |

Stratification is required by §0.2's skew, not a preference. All metrics are reported **per band
as well as pooled**, because a result that holds on large clusters and fails on small ones is a
different finding from one that holds everywhere, and a pooled number alone cannot distinguish
them.

---

## §7 — Cost, and the gated process

⚠️ **Gate A cannot be run at zero spend.** Nothing here needs a human annotator, but one of the
two arms is an LLM, so real API calls are required. Phase 4's LLM extractor cost **$2.48** across
~1500 calls; this is 30 calls and should cost far less — **and "should cost far less" is exactly
the kind of rough figure this project has a standing rule against planning from.**

**No number is pre-authorized by this spec.** The process is three separate approvals:

1. **Pilot** — 3 clusters, one per size band. Its own authorization step. Produces real measured
   token counts, plus every §2 metric and the §5.1 compliance diagnostic on those three.
2. **Measured extrapolation** — the pilot's actual token counts extrapolated to 30 clusters,
   brought back as a measurement rather than an estimate. Its own approval.
3. **Full run** — 30 clusters, only on explicit approval of (2).

This mirrors Phase 4's structure. Each step is refusable, and a pilot that reads badly is a
legitimate place to stop without ever running the full 30.

---

## §8 — What Gate A does not measure

Not readability, coherence, helpfulness, or whether a person would prefer the output. **Clearing
Gate A is not evidence the output is good** — only that an LLM arm earns its place over a free
control on a checkable axis. A design that later reaches for any of those as a headline is the
ADR-0017 failure returning in a new costume.

Gate A also does not decide whether Synthesis ships. If the template wins, the open question
becomes whether a template-only Synthesis is substantive enough to be worth shipping under that
name — and if it is not, option 3 (population/endpoint extraction as the differentiator) becomes
the conditional next phase, requiring its own Phase-4-style measurement against a deterministic
regex/MeSH baseline before any scope commitment.
