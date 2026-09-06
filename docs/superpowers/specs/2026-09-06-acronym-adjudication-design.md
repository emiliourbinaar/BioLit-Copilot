# Acronym link adjudication — design

**Status:** pre-registration. Written and committed **before any label exists**, on the same
terms as `2026-09-05-cluster-relevance-annotation-design.md`. Every band in §5 is fixed here;
none of them may move to match an observation.

**Question:** DEF-0001 records that short all-caps surfaces link confidently to whichever
concept owns that acronym in CTD, with no context check. It states **no error rate**, because
none was measured. This pass measures one.

---

## 1. Population

Every mention across the eight frozen states in `data/synth/states/` whose surface is
alphabetic, all-caps, at most four characters, and **linked** (`canonical_id` is not null),
reduced to distinct `(surface, concept_id)` pairs.

| | |
|---|---|
| entity mentions in the eight states | 5,969 |
| linked mentions matching the surface rule | 204 |
| distinct `(surface, concept)` pairs | **44** |

These are the numbers DEF-0001 published, reproduced exactly from the same states.

⚠️ **THIS IS A CENSUS, NOT A SAMPLE.** The 44 pairs are the complete enumeration of the class
in this corpus — not a draw from a population of acronym links. **No confidence interval is
reported and none would mean anything**, because there is no sampling distribution to have one
over. The rate describes these eight queries. Whether it transfers to other queries is a
question this design cannot answer, and §7 says so rather than letting a plausible-looking
interval imply otherwise.

## 2. What the annotator judges

For one pair: **does the linked concept mean what this surface means in these documents?**

Not whether the link is useful, not whether the concept is a reasonable guess, not whether the
paper is relevant. Only whether the concept is the thing the abstract's author wrote.

## 3. What a row shows, and what it hides

**Shows:**

- the surface (`GSH`)
- the linked concept: id, canonical name, **and its aliases** — without them a concept named
  `APT` or `NAD` is unjudgeable, since the name repeats the acronym and says nothing
- one or two **verbatim context windows** from the abstracts the surface appears in

**Hides:**

- ⛔ **the type-violation flag (§6).** Non-negotiable. If the sheet marked which pairs the
  linker's own type table already contradicts, those rows would be labelled `wrong` on the
  flag rather than on the text, and §6's reading would be validated by the labels it caused.
- which rows are controls (§4)
- the mention count of the pair. Same argument as the relevance pass's paper count: frequency
  is not evidence about whether a link is right, and `APT` at 26 mentions is the *most*
  frequent pair and among the clearest errors.
- the query the documents came from, and whether the pair reached an answer.

⚠️ **The context window is shown unhighlighted and unannotated.** 35 of the 44 pairs carry an
in-document parenthetical gloss (`Allyl isothiocyanate (AITC)`), and a regex found them — but
the sheet does not mark them. Marking them would make the label a judgment about the regex's
output rather than about the document, and would silently split the row set into "rows where
the tool worked" and "rows where it did not".

## 4. Controls

**Eight control rows.** Each pairs a real surface with a **real concept drawn from a different
pair**. The ids are real and the concepts are real; the pairing is deliberately wrong, and is
recorded as such in the manifest and nowhere else. Same precedent as the relevance pass's
cross-query distractors: a control has to be indistinguishable in the export or it measures
nothing.

⛔ **NO CONTROL MAY TOUCH THE §7.1 DISCLOSED SET, ON EITHER SIDE.** Not its surface, and not
its concept. Gate 2 is what makes every other reading attributable, so a control the annotator
can reject *from memory* rather than from the text corrupts the one gate this pass cannot
afford to lose — and it corrupts it in the flattering direction, reading as discrimination.

Both sides matter and the second is the less obvious one. A control on a disclosed **surface**
is rejectable because the annotator was told what it means. A control wearing a disclosed
**concept** is rejectable because they were told that concept is a known bogus linking target;
recognising `Aphakia, congenital primary` from the `CPA` discussion primes the same reflex
without a word of the passage being read.

**This was caught on the built artifact, not in design.** The first draw put **3 of 8 controls
on disclosed surfaces** (`ICH`, `DIC`, `APT`) and a fourth on a disclosed concept — nearly half
of Gate 2's instrument answerable without reading. The exclusion is now enforced in
`choose_controls` and asserted by a test.

⚠️ **The exclusion has its own cost, accepted rather than hidden.** An annotator who knows this
rule can infer that any row on a disclosed surface is a real row. That is a much weaker leak
than the one it replaces: knowing a row is not a control says nothing about which of `correct`,
`wrong` or `granularity` applies, which is the entire judgment. Gate 2's integrity is worth
more than that inference.

Without controls a permissive annotator and a broken linker produce the same numbers — the
finding ADR-0018 contributed to this project's method, and the reason its π̂ readings were
attributable where Phase 5's were not.

## 5. Gates — fixed before any label exists

### Gate 1 — is the task tractable? `cant_tell` on real rows.

**Fires `REVISE_CONTEXT` at ≥ 0.25** (11 of 44).

⚠️ **This is NOT the 0.15 the relevance pass used, and the difference is deliberate.** ADR-0018
recorded the specific mistake of carrying a constant calibrated for one question into a second
one: *a calibrated constant is calibrated for one question*. 0.15 was calibrated for "can you
judge topical relevance". This asks "can you tell what this acronym means here", which has a
different base rate — nine pairs carry no in-document gloss at all, and a gloss-free pair is
not automatically unjudgeable (`COPD`, `ATP`) but is more likely to be.

0.25 is set here as the point at which the deliverable stops being computable: below it the
rate still rests on ≥ 33 pairs; above it the denominator is doing more work than the labels.

**Firing does not stop the pass.** It means *show more context per row and re-label* — a
revision of the instrument, not of the finding. `cant_tell` rows are excluded from Gate 3's
denominator and the exclusion is reported with the rate, never absorbed into it.

### Gate 2 — is the reading attributable? Controls.

**`DISCRIMINATING` at ≥ 7 of 8 controls labelled `wrong`**, otherwise `CONFOUNDED`.

Carried from the relevance pass **and the carry is justified rather than assumed**: unlike
Gate 1's threshold, this is the same instrument asking the same question — is the annotator
using the rejection label for its meaning — over controls built the same way to be unambiguous.

A `CONFOUNDED` reading does not stop the pass; it strips Gate 3 of any causal reading and
leaves it as description.

### Gate 3 — the deliverable. The error rate.

Reported **two ways, both pre-registered**, because they answer different questions:

- **per pair** (n = 44): how often does this linking mechanism produce a wrong concept?
- **mention-weighted** (n = 204): how much wrong text does a reader actually see? These differ
  a lot — `APT` alone is 26 of the 204 mentions.

Reported as counts across all three real labels, never collapsed to a single "error" number:
`correct`, `wrong`, `granularity`. **No threshold and no pass/fail.** This gate is the
measurement DEF-0001 asks for; there is no band it could fail.

### Gate 4 — the type-violation flag, DESCRIPTIVE ONLY.

Contingency of the §6 flag against the labels. **Reported as a table of counts with no test
and no causal claim.** See §7: the flag was disclosed to the annotator before labelling, so
this cannot be the blind test it was designed to be.

## 6. The type-violation flag

`data/canon/concept_labels.json.gz` types each concept `CHEMICAL` or `DISEASE`. A link whose
concept type contradicts the mention's own NER label is an **internal inconsistency**,
detectable with data the pipeline already holds and no annotation at all.

Measured across the eight states: **87 of 3,696 linked mentions (2.4%)**. Within this pass's
population: **10 of 44 pairs, 74 of 204 mentions** — including the largest pair by mention
count.

⚠️ **A violation proves an inconsistency, not a link error.** For `DIC` labelled `DISEASE` and
linked to `Dacarbazine` (typed `CHEMICAL`), either the text means disseminated intravascular
coagulation and the *link* is wrong, or it means the drug and the *NER label* is wrong. The
flag screens; it does not adjudicate. Resolving which is what the labels are for — and what
§7 explains they can no longer do blind.

This is recorded as **DEF-0004**.

## 7. Limitations

### 7.1 — ⛔ Seventeen of the 44 pairs were disclosed to the annotator before labelling

Larger and more damaging than the concentrated-prior caveat on the relevance pass, and stated
first because it is the limitation that most constrains what these labels can support.

**From `DEFECTS.md` (written 2026-09-05), named with a strong signal about their answer:**
`GSH`, `ATN`, `CP`, `CPA`, `AITC` in the "wrong side of that coin" table; `ICH`, `ATP`, `HCC`,
`FXS` named as correct; and `IP` as the entry's opening worked example.

**From the session that produced this design:** the ten type-violating pairs were shown to the
annotator as a table — `APT`, `GSH`, `CP`, `CPA`, `RA`, `AT`, `PCC`, `BLM`, `CD`, `DIC` — and
two glosses were quoted verbatim, `CP` → cisplatin and `CPA` → cyclophosphamide.

**Distinct pairs disclosed: 17 of 44 (39%).**

⚠️ **This list read 16 on its first writing, and `IP` was the omission.** It is DEF-0001's
*opening* example, disclosed with a direction — "in an amiodarone pulmonary-toxicity corpus IP
is interstitial pneumonitis" — and it was missed because the list was compiled from the entry's
summary table rather than from its prose. It surfaced only when a regenerated control draw put
`Incontinentia Pigmenti` in as a donor concept: **found by looking at the built artifact, not
by re-reading the list.** The error is recorded here rather than silently corrected, because a
disclosure list that can be wrong is exactly the thing §7.1 exists to make auditable.

**`NAD` is deliberately excluded.** §3 names it as a concept whose canonical name repeats its
own acronym, which is a fact about the sheet's legibility and carries no signal about whether
the link is right. The test for this list is whether the **answer** was signalled, not whether
the surface was typed — otherwise every surface named in passing inflates the contaminated
denominator and understates what the pass established.

**Why it happened, recorded rather than excused.** The type-violation finding was surfaced
because the annotator had asked for a decision on whether to open DEF-0004, and that decision
could not be made without seeing it. Withholding a material finding to protect a test would
have been the worse error. But the cost is real and lands precisely on Gate 4: **the labels on
the ten flagged pairs cannot test the flag, because the annotator was told which ten they
were.** Gate 4 is therefore descriptive only (§5), and DEF-0004 rests on the mechanical
inconsistency — which needs no labels — rather than on agreement with these ones.

**What survives.** Gate 3's rate is affected differently and less. Ten of the seventeen carry a
disclosed *direction* (`GSH`, `ATN`, `CP`, `CPA`, `AITC`, `IP` as wrong; `ICH`, `ATP`, `HCC`, `FXS`
as correct); the other seven were disclosed only as type-violating, which §6 explicitly says
does not determine the answer. **Gate 3 is reported with a disclosed/undisclosed split**, the
same treatment §8.2 of the relevance design gave contaminated queries — reported separately,
never folded into a denominator that reads clean.

### 7.4 — ⛔ The controls were structurally identifiable, and the design did it

**Added 2026-09-06, after labelling.** A control is built by taking a real pair and swapping in
another pair's concept, keeping the surface and its own passages. The population is a **census**
(§1), so every surface is already in the sheet as a real row — which means **a control always
duplicates a real row's surface, and the 8 duplicated surfaces in the sheet are exactly the 8
controls**: `AH`, `ALD`, `DDAB`, `ES`, `GBS`, `HMG`, `IBS`, `OA`.

An annotator noticing a surface twice with two different concepts knows one of them is planted,
and controls are always wrong. **Gate 2's 8-of-8 reading is therefore not the clean instrument
§4 claims.** This is a defect in the design, found after labelling, and it is recorded rather
than argued away — §4's exclusion work protected the controls from *memory* and left them open
to *structure*.

**The label pattern is inconsistent with the heuristic having been used, and that is evidence
rather than reassurance.** A duplicate-spotter reasons "one of these two is the plant, so the
other is real" and marks the other `correct`. Instead:

| surface | control member | real member |
|---|---|---|
| `AH` | wrong | **wrong** |
| `ALD` | wrong | **wrong** |
| `ES` | wrong | **wrong** |
| `IBS` | wrong | **wrong** |
| `HMG` | wrong | **granularity** |
| `DDAB`, `GBS`, `OA` | wrong | correct |

Four pairs where **both** members read wrong, and one where the real member read `granularity`,
are what a reader who judged each row on its passage produces and what a duplicate-spotter
cannot. Gate 2 is reported as `DISCRIMINATING` **with this caveat attached**, not as a clean
verdict.

**The fix for a census population is structural: controls must come from OUTSIDE the census.**
Re-pairing an existing row cannot work when the population is exhaustive, because there is no
spare surface to hide behind. A future pass draws control surfaces from acronym mentions the
census excludes — unlinked ones, or ones outside the length rule — so no control duplicates
anything.

### 7.2 — Single annotator, no second reading

Unchanged from every previous pass in this project. No inter-annotator agreement is available
and none is claimed. Gate 2's controls are what make the reading attributable at all.

### 7.3 — The census bound

§1. The rate describes eight queries' worth of abstracts in two therapeutic areas. It is not
an estimate of the linker's error rate in general, and the write-up must not be phrased as if
it were.

## 8. What this pass cannot establish

- **Whether refusing these links would improve the pipeline.** It trades a wrong link for a
  NIL, and this project's measured dominant failure mode is already *abstention, not error*
  (Phase 3: e2e NIL rate 0.45). That trade needs its own measurement against BC5CDR gold,
  where both sides are scoreable. DEF-0004 records the tradeoff and explicitly does not build
  the fix.
- **Whether the wrong links reached a reader.** Some pairs are in clusters that `select_stage`
  dropped. Not measured here; a separate question.
- **Any rate for surfaces outside the class.** Long acronyms, mixed-case abbreviations and
  ordinary words are untouched by this design and no number here extends to them.
