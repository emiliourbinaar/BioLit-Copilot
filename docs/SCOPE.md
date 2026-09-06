# Scope Record

Things deliberately **not** built, and why. Lighter than an ADR: an ADR records a decision
about how the system works, a scope record records a decision about what the system is not
going to attempt. Newest at top. Citable as `SCOPE.md SR-000N`.

A scope record is not a backlog entry. "Not now" here means the reasoning below has to be
answered before the work starts, not merely that nobody has got to it.

---

## SR-0001 — Narrative synthesis across clusters is out of scope

- **Date:** 2026-09-05
- **Status:** Accepted — **not pursued.** Reopening requires a structurally different framing,
  named below, and a separate justification.
- **Applies to:** an LLM (or any generative) stage that reads several characterised clusters
  and writes a connected prose answer over them. It does **not** apply to the shipped
  deterministic `synthesis_stage` (ADR-0019), and it does not apply to query-conditioned
  selection or ordering, which are ordinary deterministic work.

### The reasoning

**Narrative synthesis sits in exactly the regime Gate A proved undecidable, and it sits there
worse.** ADR-0019's finding was structural, not a tuning gap: with no gold, every metric must
be computed from the source text, so every axis necessarily measures *overlap with the
source* — and overlap-based axes are all maximised by the same degenerate strategy, naming
every paper while saying almost nothing about any of them. That was demonstrated, not
argued: an `index` arm emitting `- PMID x: <one word>` scored coverage 1.0, hallucinated 0,
DCR 1.0 and compression 0.0516 against the template's 1.1417 — dominating both comparative
axes while characterising nothing.

Nothing about that argument depends on the unit being a single cluster. Moving up a level to
"a connected answer across clusters" inherits the whole problem and adds to it:

1. **The output space is larger**, so the space of degenerate outputs that score well is
   larger too. Every axis that failed on one cluster fails on twenty.
2. **The quality question moves closer to the forbidden metric.** ADR-0017 explicitly ruled
   out scoring readability or "would a person prefer this". "Is this a faithful
   characterisation of what these papers found" at least *points* at something checkable
   against source text, and it still was not checkable. **"Is this a coherent narrative" does
   not even point that way** — coherence is a property of the prose, not a relation between
   the prose and the sources. It is nearer to the forbidden metric than the question Gate A
   already failed to answer.
3. **The cross-paper claims a narrative wants to make are the ones ADR-0018 established the
   pipeline cannot support.** A narrative earns its keep by saying how findings relate — these
   agree, that one dissents, the effect is larger in this population. ADR-0018 closed the
   search for gold on exactly that relation after six corpora across three structural
   families. A narrative stage would therefore be generating precisely the claims the project
   has no way to check, which is the worst possible place to put an unverifiable component.

**And the prior from two completed rounds is not neutral.** Phase 4 (ADR-0015) was a
*decidable* comparison — there was gold, the LLM extractor was measured against it, and it
lost to the first four sentences of the abstract at an identical selection budget. Gate A
(ADR-0019) was *undecidable* — no gold, and the axes ranked outputs in the opposite order to
their quality. The pattern across both is not "LLMs are bad here"; it is **that this project
gets a usable answer exactly when it has gold, and gets nothing when it does not.** Problem B
has no gold and no cheap path to one.

### The one framing that would be genuinely different

**Constrain the output space until checking becomes decidable again.** The index arm won
because the output was free text and the metrics saw only overlap and brevity. If the
generated output were instead a **fixed schema** — a set number of slots, each a short claim
carrying a mandatory pointer to the source sentence it came from — then two things change at
once. Brevity gaming becomes impossible, because the schema fixes the length. And each slot
becomes a small faithfulness judgment of exactly the kind this project already checks
deterministically, against the source sentence the slot itself names.

That converts a narrative-quality question into a set of extraction-shaped questions, and
extraction is the regime where this project's comparisons have actually been decidable.

⚠️ **Scoped honestly, that is an enriched deterministic template with LLM-filled slots — it is
not narrative synthesis.** The structure, the ordering, and the claim types would all still be
chosen deterministically; the model would only fill in constrained cells. Anyone reopening
this should be clear that it does not deliver the thing "narrative synthesis" names, and
should not be sold as though it does. If what is actually wanted is connected prose, this
framing does not provide it and the objections above still stand in full.

**Not pursued now.** It is available as a future thread, and it needs its own justification:
a pre-registered decision criterion written *before* anything is built, and a clear statement
of what the slots are and why those slots. It does not inherit authorisation from this record.

### What was rejected alongside it

- **Buy gold for answer quality.** Real, and closed on cost and precedent: ADR-0017 and
  ADR-0018 each closed an annotation programme, and "is this a good answer" needs far more
  labels than "is this pair a contradiction" while being more preference-shaped. Note the
  contrast with the *cluster-relevance* annotation task, which is live precisely because it
  is small (83 rows), bounded by an already-frozen artifact, and asks something much closer
  to objective.
- **Ask a capability question instead of a quality one** ("does the output assert a
  cross-paper relation the template structurally cannot?"). Decidable as posed, but it
  relocates the problem rather than solving it — an unchecked cross-paper assertion is the
  hallucination ADR-0018 says cannot be verified here.
- **Ship it unevaluated and label it as such.** Intellectually honest and common in real
  systems. Rejected because not doing this is the project's entire differentiator.
- **Run a third gate to find out.** Rejected on expected value. ADR-0019 is a strong result;
  a second negative result reached by the same route reads as a pattern rather than a lesson,
  and the reasoning above already predicts the outcome. If the prediction is wrong, the
  fixed-schema framing is where to demonstrate it — not another free-text gate.
