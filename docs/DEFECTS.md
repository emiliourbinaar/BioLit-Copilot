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

## DEF-0001 — Short acronym surfaces link confidently to whichever concept owns that acronym in CTD, with no context check

- **Date:** 2026-09-05
- **Component:** `biolit.canon` — `MeshDictionary.lookup` / `DictionaryLinker`
- **Status:** Recorded, not fixed. No consumer-side workaround; see "Why not fixed here".
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

| surface | linked to | in context almost certainly |
|---|---|---|
| `GSH` | Glucocorticoid-Remediable Aldosteronism | glutathione |
| `ATN` | Oculocutaneous albinism type 1 | acute tubular necrosis |
| `CP` | Cleft Palate, Isolated, And Mental Retardation | creatine phosphokinase / cardiopulmonary |
| `CPA` | Aphakia, congenital primary | cyproterone acetate |
| `AITC` | 2,3,4-tri-O-acetylarabinopyranosyl isothiocyanate | allyl isothiocyanate |

**How many of the 44 are wrong has not been measured**, and stating a defect rate would
require adjudicating each pair against its source abstracts. That is a small, bounded
labelling task; it is not done, and no rate is claimed here.

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

---

## DEF-0003 — ADR-0020's hierarchy term is unreachable: every cluster the ranker sees scores proximity 0

- **Date:** 2026-09-05
- **Component:** `biolit.query.ranking._relevance_key`
- **Status:** Recorded, not fixed. A structural revision is permitted by the annotation
  design's §5, but see "Why this is not simply fixed" below.
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

⚠️ **But the labels this defect was found with are now spent as a blind test of any fix.** The
design's §5 permits a revision that changes the score's structure and forbids one that tunes a
constant, and a max-over-sides revision is squarely the permitted kind. It would still be
measured against labels chosen *before* the defect was known but read *after* — so a re-run of
Gate 4 is a weaker instrument than the one that produced this reading, and must be reported as
such rather than quoted alongside it. That call belongs to the project owner, not to this log.
