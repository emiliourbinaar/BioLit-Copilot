# Defect Log

Measured defects in shipped components that are **recorded and not yet fixed**. Newest at
top. Citable as `DEFECTS.md DEF-000N`.

This log exists because these were being recorded inside whichever ADR happened to notice
them, which buries a standing defect in the prose of a decision about something else. An
entry here is a claim about behaviour that has been *verified against real data*, with the
verification shown. A suspicion is not a defect entry.

**Every entry so far is in entity linking.** That is the project's known dominant bottleneck
(Phase 3: e2e NIL rate 0.45 on the domain sample, 0.32 on BC5CDR), and it is where a defect
does the most damage, because everything downstream — pairing, clustering, selection,
synthesis — consumes `canonical_id` and has no way to second-guess it.

---

## DEF-0004 — A link whose concept type contradicts the mention's own NER label is never refused, because the linker is never told the label

- **Date:** 2026-09-06
- **Component:** `biolit.canon` — the `Linker` protocol and `DictionaryLinker`
- **Status:** Recorded, **not fixed**. Opened as a stub before the DEF-0001 adjudication ran,
  because the tradeoff below is true regardless of how those labels came out; **decomposed
  2026-09-06** once they existed — 8 of the 10 flagged pairs are link errors, 1 is an NER span
  error and 1 is both.
- **Severity:** Wrong entity, silently, at full confidence — the same reader-visible damage as
  DEF-0001, but with a **mechanically detectable** signature that DEF-0001's class does not have.

### What was observed

`data/canon/concept_labels.json.gz` types each concept `CHEMICAL` or `DISEASE`. Comparing that
type against the NER label the mention already carries, across all eight frozen states:

| | |
|---|---|
| linked mentions | 3,696 |
| whose concept's own type contradicts the mention's NER label | **87 (2.4%)** |
| — mention labelled `CHEMICAL` → concept typed `DISEASE` | 59 |
| — mention labelled `DISEASE` → concept typed `CHEMICAL` | 28 |

Within DEF-0001's population of short all-caps pairs: **10 of 44 pairs, 74 of 204 mentions**,
including the largest pair by mention count.

```
26 mentions  APT  DISEASE  -> MESH:C071989 'APT'                              CHEMICAL
 9 mentions  GSH  CHEMICAL -> MESH:C563177 'Glucocorticoid-Remediable Aldo…'  DISEASE
 9 mentions  CP   CHEMICAL -> MESH:C566991 'Cleft Palate, Isolated, And Me…'  DISEASE
 8 mentions  CPA  CHEMICAL -> MESH:C537786 'Aphakia, congenital primary'      DISEASE
 8 mentions  RA   CHEMICAL -> MESH:D001172 'Arthritis, Rheumatoid'            DISEASE
 6 mentions  AT   CHEMICAL -> MESH:D001260 'Ataxia Telangiectasia'            DISEASE
 4 mentions  PCC  CHEMICAL -> OMIM:115700  'CATARACT 4, MULTIPLE TYPES'       DISEASE
 2 mentions  BLM  CHEMICAL -> MESH:D001816 'Bloom Syndrome'                   DISEASE
 1 mention   CD, 1 mention DIC  (pairs whose NER label varies across documents)
```

### Why it is possible

**`concept_labels.json.gz` has no consumer anywhere in `src/`.** It was built for ADR-0012's
embedding-fallback measurement, which was declined, and the artifact outlived the mechanism.

The deeper reason is structural rather than an unset flag: the `Linker` protocol is

```python
def link(self, surface: str) -> LinkResult: ...
```

It receives the surface and nothing else. **The mention's label is never passed to the linker**,
so this check is not switched off — it is unavailable on the shipped path. `EVAL_REPORT.md`
records that "label-constraining helps slightly and costs nothing: +0.0017 F1 (0.7880 →
0.7897)"; that was measured on a harness path that is not the one the pipeline runs.

### ⚠️ What a violation proves, and what it does not

**It proves an internal inconsistency, not a link error.** For `DIC` labelled `DISEASE` and
linked to `Dacarbazine` (typed `CHEMICAL`), either the text means disseminated intravascular
coagulation and the *link* is wrong, or it means the drug and the *NER label* is wrong. One of
the two is wrong; which one requires reading the abstract.

That distinction is why this entry claims 87 inconsistencies and **not** 87 wrong links.

### Why not fixed here

⛔ **The obvious fix is not free, and its cost lands on this project's dominant failure mode.**
Refusing a type-violating link converts a wrong entity into a NIL — and Phase 3 measured the
e2e NIL rate at **0.45 on the domain sample, 0.32 on BC5CDR**, recording the failure mode as
*abstention, not error*. Trading precision for recall in the direction the bottleneck already
runs is exactly the trade ADR-0011's rejection of dictionary enrichment turned on.

It also cannot be measured here: these eight states have no gold, so both sides of the trade
are unscoreable on them. The measurement belongs on BC5CDR, where a refused link that should
have been kept shows up as a loss instead of disappearing.

Per ADR-0013, the fix does not get built before that measurement exists. What this entry
establishes is that the signal is present, free, and currently discarded.

### ⭐ UPDATED 2026-09-06 — the decomposition, from the DEF-0001 labels

**All ten flagged pairs were labelled `wrong`.** None was `correct`, `granularity` or
`cant_tell`. Over the census the flag's precision is **10 / 10** and its recall over wrong pairs
is **10 / 23** — it catches under half the errors, and misses nothing it fires on.

⚠️ **DESCRIPTIVE ONLY, and the reason is not a formality.** The ten flagged pairs were shown to
the annotator as a table before labelling, so these labels cannot be the blind test of the flag
this was designed as (spec §7.1). This entry still rests on the mechanical inconsistency, which
needs no labels. What *is* new evidence is the decomposition below, because the annotator was
never told what any of these surfaces actually meant.

**Which side was wrong — the link, or the NER label?** §"What a violation proves" said a
violation identifies an inconsistency without saying which half caused it. The passages settle
it for 8 of 10:

| pair | what the passage says | at fault |
|---|---|---|
| `APT` | amiodarone-induced pulmonary toxicity | **link** |
| `GSH` | glutathione | **link** |
| `CP` | cisplatin | **link** |
| `CPA` | cyclophosphamide | **link** |
| `RA` | rosmarinic acid | **link** |
| `PCC` | prothrombin complex concentrate | **link** |
| `BLM` | bleomycin | **link** |
| `DIC` | disseminated intravascular coagulation | **link** |
| `AT` | *the letters `AT` taken out of `ATO` (atorvastatin)* | **NER span** |
| `CD` | a chemical probe name in one passage; conduct disorders in the other | **both** |

⛔ **`AT` is the case that stops the obvious fix from being obvious.** The mention is a
*fragment of a longer token* — NER cut `AT` out of `ATO` — so no linking decision could have
been right, and refusing the link would suppress a symptom while leaving a span defect
upstream. A type-constrained linker would score this as a success and fix nothing.

**What this does and does not change about the fix.** It strengthens the case that the signal is
real: 8 of 10 are squarely link errors that a type check would have caught for free. It does
**not** touch the reason the fix is not built here — refusing these links converts wrong
entities into NILs, and abstention is already this project's dominant failure mode. That trade
still needs BC5CDR, where both sides of it score.

**One number in this entry made more precise.** The "74 of 204 mentions" above counts
individually type-violating *mentions*; the ten flagged *pairs* carry **77** mentions in total.
The two differ because `CD` and `DIC` are labelled inconsistently across documents, so some of
their mentions violate and some do not. Both figures are correct about different things and the
distinction is now stated rather than left for a reader to trip over.

---

## DEF-0001 — Short acronym surfaces link confidently to whichever concept owns that acronym in CTD, with no context check

- **Date:** 2026-09-05
- **Component:** `biolit.canon` — `MeshDictionary.lookup` / `DictionaryLinker`
- **Status:** ⭐ **MEASURED 2026-09-06** — 23 of 44 pairs (52.3%) and 132 of 204 mentions (64.7%) carry a wrong concept. Still **not fixed**; no consumer-side workaround, see "Why not fixed here".
- **Severity:** Wrong entity, silently, at full confidence. `LinkResult.tiebroken` is `False`
  for these, so nothing downstream can tell them from a clean link.

### What was observed

`Amiodarone | Incontinentia Pigmenti` appears as a real 2-paper cluster in the frozen
`amiodarone pulmonary toxicity` run, and therefore in that query's answer. Verified in the
state dump:

```
surface 'IP'  label=DISEASE  ->  MESH:D007184 (Incontinentia Pigmenti)
```

In an amiodarone pulmonary-toxicity corpus `IP` is interstitial pneumonitis. It linked to a
dermatological genetic disorder because that concept owns the alias `IP` in the CTD table,
the alias is unambiguous *in the table*, and nothing consults the surrounding text.

The same run's largest cluster has the same shape: `APT` → `MESH:C071989`, where in context
APT is the query's own subject, amiodarone pulmonary toxicity.

### The class, measured

Across all eight frozen states (5,969 entity mentions):

| | |
|---|---|
| linked mentions whose surface is all-caps and ≤ 4 characters | **204 (3.4%)** |
| distinct (surface, concept) pairs among them | **44** |

⚠️ **Not all 204 are errors, and the entry does not claim they are.** Several are correct —
`ICH` → Cerebral Hemorrhage, `ATP` → Adenosine Triphosphate, `HCC` → Carcinoma Hepatocellular,
`FXS` → Fragile X Syndrome. That is precisely the defect: correctness here is decided by which
concept happens to own the acronym in CTD, not by anything about the document. Verified
examples from the wrong side of that coin:

| surface | linked to | what the abstract itself says |
|---|---|---|
| `GSH` | Glucocorticoid-Remediable Aldosteronism | `…Malondialdehyde (MDA) and glutathione (GSH)` |
| `ATN` | Oculocutaneous albinism type 1 | `…prior platinum-associated acute tubular necrosis (ATN)` |
| `CP` | Cleft Palate, Isolated, And Mental Retardation | `…Apoptosis In Vivo and In Vitro. Cisplatin (CP)` |
| `CPA` | Aphakia, congenital primary | `…methotrexate (MTX), cyclophosphamide (CPA)` |
| `AITC` | 2,3,4-tri-O-acetylarabinopyranosyl isothiocyanate | `…(HEK)-293 cells. Allyl isothiocyanate (AITC)` |

⭐ **AMENDED 2026-09-06 — the third column was originally this entry's own guesses, headed "in
context almost certainly", and two of the five were wrong.** `CP` was guessed as creatine
phosphokinase / cardiopulmonary and is **cisplatin**; `CPA` was guessed as cyproterone acetate
and is **cyclophosphamide**. The column now quotes the source abstracts verbatim instead.

**This is the entry's own caution being vindicated in a specific way, and it is kept rather
than tidied:** the entry refused to state a rate because the pairs had not been adjudicated,
and the informal reading it did offer — on the five rows it was most confident about — was
wrong on two. A rate derived from that reading would have been wrong in a way nothing would
have caught.

### ⭐ MEASURED 2026-09-06 — the rate, at last

All 44 pairs were adjudicated blind against their source passages, plus 8 controls, under gates
fixed before any label existed (`docs/superpowers/specs/2026-09-06-acronym-adjudication-design.md`;
`rows_hash ad22f212d9a39b55`, `labels_hash 538e8b522f04b370`).

| | per pair | per mention |
|---|---|---|
| **`wrong`** — the concept is not what the author meant | **23 / 44 (52.3%)** | **132 / 204 (64.7%)** |
| `granularity` — right subject, wrong level (DEF-0002's shape) | 2 / 44 | 18 / 204 |
| `correct` | 19 / 44 | 54 / 204 |
| `cant_tell` | **0** | — |

**A majority of these links are wrong, and the reader-facing figure is worse than the
mechanism-facing one.** Per pair asks how often the mechanism errs; per mention asks how much
wrong text a reader sees. The gap is not noise — the wrong pairs are the frequent ones, led by
`APT` at 26 mentions.

⚠️ **This is a CENSUS of these eight queries, not an estimate.** The 44 pairs are the complete
enumeration of the class in this corpus, so there is no sampling distribution and **no interval
is reported**. Whether 52% transfers to other queries is a question this design cannot answer.

**Reported split, per ADR-0016 rule 6**, because 17 of the 44 were named to the annotator before
labelling:

| | pairs `wrong` | mentions `wrong` |
|---|---|---|
| disclosed (17) | 13 / 17 | 93 / 131 |
| **undisclosed (27)** | **10 / 27** | **39 / 73** |

⛔ **The gap between the two rows must NOT be read as disclosure bias.** The disclosed set was
*selected for looking wrong* — `DEFECTS.md` picked known-bad examples and DEF-0004's flag picked
inconsistencies — so a higher rate there is expected by construction, disclosure or not.
Selection and disclosure are confounded and this design cannot separate them. What the split
does establish is that **on 27 pairs carrying no prior disclosure at all, 10 are still wrong**.

**Two signs the labels are independent judgment rather than an echo.** The annotator marked
`ICH` as `granularity` where this entry had named it *correct* — reading intracranial
haemorrhage as broader than `Cerebral Hemorrhage`, which is DEF-0002's exact shape. And for the
six pairs disclosed only as type-violating, with no direction attached (`RA`, `AT`, `PCC`,
`BLM`, `CD`, `DIC`), the recorded reasons name meanings that were never disclosed to them —
rosmarinic acid, prothrombin complex concentrate, bleomycin, conduct disorders.

**A caveat on Gate 2 belongs with this reading rather than buried in the spec:** the controls
were structurally identifiable by surface duplication — every control kept its surface, and over
a census that surface is already in the sheet as a real row. The label pattern refutes the
shortcut having been used (four duplicated surfaces had *both* members marked `wrong`, which a
duplicate-spotter cannot produce), but the instrument was not clean. Generalised as
**ADR-0021**, which is a design-methodology finding rather than an entity-linking one and so
lives in `DECISIONS.md` rather than here.

See also **DEF-0004**, which the adjudication decomposes.

### Why it is not a selection or synthesis defect

It reaches the reader through the answer, which is where it was noticed, but nothing in
`select_stage` or `synthesis_stage` could have prevented it. By the time a cluster exists,
`IP` *is* Incontinentia Pigmenti as far as every downstream stage can tell: the concept id is
well-formed, the cluster key is legitimate, two papers really do share it, and `select_stage`
correctly keeps it because Amiodarone is a query concept. Filtering it out downstream would
mean suppressing a correctly-formed cluster on a guess, which is worse than showing it.

### Why not fixed here

A fix belongs in linking and needs a design, not a patch. The obvious candidates each carry a
cost that has to be measured rather than assumed:

- **Refuse short all-caps surfaces outright.** Cheap, and it would discard `ICH`, `ATP`,
  `HCC`, `FXS` along with the errors — trading a precision problem for a recall problem in a
  pipeline whose measured bottleneck is already abstention (Phase 3 recorded the failure mode
  as *abstention, not error*).
- **Require the acronym's expansion to appear in the same document.** The standard approach,
  and the right shape, since abstracts usually define an acronym on first use. Real work: a
  definition-detection pass plus a decision about what to do when no definition is found.
- **Score candidates against document context.** ADR-0012 already measured and declined the
  embedding fallback linker as too weak and too costly for its benefit; that reading was about
  a different problem, but it is the nearest prior and should be consulted before anyone
  proposes embeddings here.

Per ADR-0013, none of that gets built before someone decides which failure this project would
rather have.

---

## DEF-0002 — A bare parent-concept mention is indistinguishable from its specific child

- **Date:** 2026-09-04 (recorded in ADR-0019; moved here 2026-09-05 so it is findable)
- **Component:** `biolit.canon` — same context-free lookup as DEF-0001
- **Status:** Recorded, not fixed.

A bare `TNF` resolves to the parent MeSH concept of a source's `TNF-α`, so concept
**hierarchy** produces a false entity-hallucination positive by a different mechanism than
the Unicode-normalisation class fixed on the Gate A branch. It was left unfixed there because
it was moot for a retired gate — see ADR-0019's consequences section, which carries the full
context.

It is logged here because it is not actually specific to that gate: it is the same root cause
as DEF-0001 — a surface form is linked without reference to what the document is about — and
any fix for one should be checked against the other.

**⭐ UPDATE 2026-09-07 — the SELECTION half of this shape is closed on the chemical side; the
LINKING half is untouched.** Gate 3 of the relevance pass found the same granularity mismatch
arriving at a different stage: `select_stage` deleted three `answers` clusters carrying
`Atorvastatin` because the query resolved to the drug **class**, class-to-member on the
chemical side. **ADR-0022 fixes that** — a cluster side now matches if it belongs to a
pharmacological class the query named, and all three are recovered. ⚠️ **This is not a fix for
the defect above, and the distinction is the point.** ADR-0022 changes what the *filter* does
with two correctly-linked concepts whose granularity differs; DEF-0002 is about the *linker*
producing the wrong concept from an ambiguous surface, and nothing here gives the linker any
context it did not have. The disease side is also still open: no equivalent relation was built
for diseases, and MeSH's `PharmacologicalAction` field only covers chemicals. **What ADR-0022
does establish for any future attempt here: the MeSH tree cannot express a drug class's
membership at all** — `Atorvastatin` is filed under chemical structure (D03/D10) and its class
under actions and uses (D27), sharing no node — so a hierarchy walk is the wrong instrument for
this half of the problem, whichever stage it is attempted at.

---

## DEF-0003 — ADR-0020's hierarchy term is unreachable: every cluster the ranker sees scores proximity 0

- **Date:** 2026-09-05
- **Component:** `biolit.query.ranking._relevance_key`
- **Status:** ⭐ **FIXED 2026-09-05 as a structural revision — and UNVALIDATED.** Proximity is
  now the maximum over sides that are not already exact matches. Verified mechanically only:
  queries whose clusters all tie fell from 4 of 8 to 2 of 8. **These labels are spent as a
  blind test of the fix** and were not re-read; see ADR-0020's addendum. ⚠️ **Added 2026-09-06:**
  they also come from a pass whose control instrument was later found compromised (ADR-0021), so
  they could not have validated the fix even if spending them were permitted. **This fix rests
  on the mechanical check and on nothing else** — not on a null result from the labels, which is
  a different and weaker claim.
- **Found by:** Gate 4 of the cluster-relevance annotation pass, which is what it was for.

### What was observed

ADR-0020 scores a cluster as `(-exact_matches, min_tree_distance, key)`. The second term
never varies. Measured over the frozen corpus, **four of eight queries produce a single
distinct score across every cluster** — amiodarone, cisplatin, isotretinoin, lithium — so the
ranker falls back entirely to the MeSH-id order it was built to replace.

The mechanism, verified on `isotretinoin and depression`:

```
query concepts: [MESH:D003866 Depressive Disorder, MESH:D015474 Isotretinoin]
  Acne Vulgaris              score=(-1, 0)   disease-side distance to D003866: None
  Anxiety Disorders          score=(-1, 0)   disease-side distance to D003866: 3
  Mental Disorders           score=(-1, 0)   disease-side distance to D003866: 2
  Psychotic Disorders        score=(-1, 0)   disease-side distance to D003866: 4
```

The distances ADR-0020 relies on are computed correctly and then discarded. `_relevance_key`
minimises over **every** (cluster side × query concept) pair, and the cluster's own chemical
side *is* a query concept — `distance(Isotretinoin, Isotretinoin) == 0` — so the minimum is 0
for every cluster and the disease-side signal never reaches the sort.

⛔ **This is total, not partial.** `select_stage` keeps only clusters with at least one exact
concept match, and any exact match contributes distance 0. **Every cluster the ranker is ever
handed therefore scores proximity 0**, so the term is unreachable in production by
construction. The score is operationally `(-exact_matches, key)`.

### What that means for the reading it produced

Gate 4a improved leads from 3/8 to 5/8, and **that improvement came entirely from the
exact-match count**. Nothing in it is evidence about hierarchy proximity, which is the part of
ADR-0020 the MeSH tree artifact exists to serve. The distances quoted in ADR-0020's decision
section — Mental Disorders 2, Anxiety 3, Psychotic 4, Acne no shared tree — are correct
readings of the tree and were never reachable by the code.

### Why this is not simply fixed

The fix is structural rather than a constant: proximity should describe the **residual** match
quality of sides that are not already exact — a cluster is as good as its *worst*-matched
side, so the term wants a max (or a sum) over sides rather than a min over all pairs. Under
that reading, `Isotretinoin | Anxiety Disorders` scores 3 and `Isotretinoin | Acne Vulgaris`
scores unmatched, which is the ordering ADR-0020 intended.

⚠️ **The labels this defect was found with are spent as a blind test of the fix.** §5 permits a
revision changing the score's structure and forbids one tuning a constant, and max-over-sides is
squarely the permitted kind — but it would be measured against labels chosen *before* the defect
was known and read *after*. **No Gate 4 re-run was performed**, and none should be quoted as
validating this. Evidence about hierarchy proximity needs a fresh label set or queries outside
these eight.

**What the fix does not fix.** Two queries still tie across every cluster: `amiodarone pulmonary
toxicity` and `cisplatin nephrotoxicity`. Their disease terms resolve to no MeSH concept even
through NCBI, so the query is a lone chemical and every kept cluster carries it, while the
disease sides share no tree with any chemical. That is a query-linking limit, not a scoring one,
and it is recorded in ADR-0020's addendum rather than here.
