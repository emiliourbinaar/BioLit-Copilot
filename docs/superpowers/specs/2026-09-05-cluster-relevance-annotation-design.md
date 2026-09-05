# Cluster-relevance annotation — validating query-conditioned selection on the frozen corpus

- **Date:** 2026-09-05
- **Status:** **Design. Not yet run.** No labels exist. Nothing in the pipeline depends on this
  reading, and `select_stage` ships without it — this validates a filter already in production
  rather than gating its release.
- **Kind:** Measurement. A validity probe on a shipped deterministic component, run against a
  corpus that was frozen for an unrelated purpose and therefore cannot have been selected to
  flatter it.
- **Depends on:** ADR-0017 (π is a **ceiling**, never a correction factor; readability/preference
  is a forbidden metric), ADR-0018 (a single annotator needs a discriminating control before any
  reading is attributable), ADR-0016 (verification rules; decide by running the weakened case),
  ADR-0013 (no infrastructure without a demonstrated consumer), ADR-0015 (an arm scored on a
  population it cannot lose on proves nothing).
- **Validates two things, from one round of labelling:** cluster **membership** (does the
  filter keep the right clusters) and cluster **ordering** (does the ranker proposed in
  ADR-0020 lead with the right one). §9 records why the label schema needed no change to
  cover the second, and what it deliberately cannot certify.
- **Does not decide:** which of two equally on-topic clusters should come first. §9.2 argues
  that is a preference judgment this project has forbidden itself, and that the ranker is not
  asked to get it right.

---

## 1. What this measures, and why it is worth 83 labels

`select_stage` keeps a cluster when either side of its `chemical|disease` key is a MeSH concept
NCBI translated the query into. That is a deterministic, checkable rule, and on the frozen
corpus it keeps **70 of 83 clusters**. What no measurement yet says is whether those are the
*right* 70.

The question is small enough to answer exactly. The corpus is already frozen — eight PubMed
queries, 470 papers, 83 clusters after Ruling 26's merge-by-key — and every cluster is a short,
inspectable object: a chemical concept, a disease concept, and a list of papers. **83 labels
covers the entire population, not a sample.** There is no sampling error to reason about and no
stratification to design, which is the opposite of every prior annotation pass in this project.

⚠️ **This is a sanity check on a filter, not a gold standard.** One annotator produces it. It
cannot support an inter-annotator agreement figure, it must never be used to tune a threshold
into existence, and it must never be used to score a generative arm. Its only sanctioned uses
are §6's.

---

## 2. Unit and label schema

**Unit:** one `(query, cluster)` pair. 83 real rows.

The annotator sees the query, the cluster's two concept names, the paper count, and the year
range. **Not** the cluster key's raw MeSH ids, and **not** the finding sentences — the judgment
is about topical relevance, and reading the findings invites judging quality instead.

| label | meaning |
|---|---|
| `answers` | This cluster is part of what the question asked for. |
| `background` | Genuinely about the query's subject, but not what was asked — the drug's indication, its main comparator, a co-occurring condition. |
| `off_topic` | Not what the question was about. |
| `cant_tell` | The query, the concepts, or the pairing is too ambiguous to judge. |

**Why `background` is a separate class and not folded into either neighbour.** It is the class
the whole design turns on. `Isotretinoin | Acne Vulgaris` (18 papers) is not off-topic for
"isotretinoin and depression" — it is the indication, and a reader would reasonably want it —
but it is not what was asked, and it is currently what the answer *opens with*. Collapsing it
into `answers` would declare the current behaviour correct by definition; collapsing it into
`off_topic` would demand the filter drop it, which is probably wrong. Keeping it separate is
what lets the same 83 labels speak to both the filter and to ordering (§7).

`cant_tell` exists as a **tractability gate**, exactly as in ADR-0018's Gate 1. It is not a
convenience escape hatch, and §5 makes a high rate fatal to the whole reading rather than
something to work around.

**The first three labels are an ordinal scale, and that is what lets one round of labelling
validate ordering as well as membership.** `answers` > `background` > `off_topic` is a
3-grade relevance judgment, so a ranker's output can be checked against it directly (§5,
Gate 4) with no additional annotation and no change to what the annotator is asked. `cant_tell`
is not a grade and is excluded from every ordering computation.

---

## 3. Protocol

1. **Write the label definitions first.** §2 is fixed before any row is seen. It is not revised
   mid-pass; if it proves inadequate, the pass stops and restarts with a new definition and no
   labels carried over.
2. **Blind to the filter.** The export **must not** carry `select_stage`'s keep/drop decision,
   and must not be ordered by it. Rows are shuffled under a recorded seed across all queries, so
   consecutive rows come from different questions. Without this the labels measure agreement
   with a decision already seen, which is not a measurement.
3. **Blind to cluster size ordering.** Rows are not sorted by paper count. Size correlates with
   relevance and would leak an ordering cue.
4. **Distractor controls, per ADR-0018.** Inject **8 distractor rows**: a cluster drawn from one
   query, presented under a *different* query, chosen so the pairing is genuinely unrelated
   (e.g. `Warfarin | Thromboembolism` presented under "cisplatin nephrotoxicity"). They are
   shuffled in with the real rows and are indistinguishable in the export. **91 rows total.**
5. **One pass, no revisiting.** Labels are appended as they are made; earlier rows are not
   revised after later ones clarify the schema. Revisiting silently converts a blind pass into a
   calibrated one.
6. **The annotator is the project owner.** This is stated as a limitation, not managed away —
   see §8.

---

## 4. Export format

One JSONL file, one row per line, written to `data/relevance/` (gitignored, like every other
corpus artifact) with a manifest hash committed to the run log alongside the seed.

```json
{"row_id": "r07", "query": "isotretinoin and depression",
 "chemical": "Isotretinoin", "disease": "Acne Vulgaris", "n_papers": 18,
 "year_range": "1990-2024", "label": null}
```

`row_id` is opaque and carries no information about query, source cluster, or distractor status.
The mapping from `row_id` back to `(query, cluster_key, is_distractor, filter_kept)` lives in a
**separate** manifest file that the annotator does not open until every label is written.

Content hashing follows `synth_corpus.sample_hash`: sorted, content-addressed, stable over
order and serialization, so the exact row set is pinned before any label exists.

---

## 5. Gates and pre-registered readings

Written before any label exists. **Numbers are read as they fall; no threshold moves to match
an observation.**

**Gate 1 — tractability.** `cant_tell` over the 83 real rows.
- ≥ 15% → **`REVISE_SCHEMA`**: the label definitions do not fit the data. **Stop. Do not compute
  Gate 2, and do not report the filter's numbers at all** — a filter reading computed over rows
  the annotator could not judge is not interpretable. The precedent is ADR-0018's Gate 1, which
  was a genuine precondition and not a formality.
- < 15% → proceed.

**Gate 2 — control discrimination.** The 8 distractors.
- ≥ 7/8 labelled `off_topic` → **`DISCRIMINATING`**. The annotator is using the label for its
  meaning, and the Gate 3 reading is attributable to the filter.
- < 7/8 → **`CONFOUNDED`**. Report Gate 3 descriptively and draw no conclusion from it. This is
  ADR-0018's contribution: without this control, a permissive annotator and a good filter are
  indistinguishable.

**Gate 3 — the filter's confusion matrix**, over the 83 real rows, with 95% Wilson intervals on
each rate and the raw counts always printed beside them.

|  | labelled `answers` | labelled `background` | labelled `off_topic` |
|---|---|---|---|
| **kept** (70) | correct | *the ordering question* | over-inclusion |
| **dropped** (13) | ⛔ **false drop** | under-inclusion | correct |

**Pre-registered actions, so the reading cannot be rationalised afterwards:**

- ⛔ **Any `answers` cluster the filter dropped is a defect to fix, not a rate to tolerate.**
  The population is 83 and each case is individually inspectable, so "an acceptable false-drop
  rate" is not a meaningful object here. Each one gets diagnosed to its cause — a query term
  that failed to link, a concept-granularity mismatch, an OR that should have matched — and
  fixed or explicitly recorded as unfixable.
- **`background` clusters that were kept are the ordering evidence (§7), not filter errors.**
  The filter is not asked to drop them and is not scored down for keeping them.
- **`off_topic` clusters that were kept measure over-inclusion**, and are reported as a rate
  with its interval. No threshold is pre-committed, because there is no principled one to
  commit to — the number is reported and read, and if it is bad the response is a diagnosed
  change to the rule, never a tuned constant.

**Gate 4 — ordering**, over the same labels, for the ranker proposed in ADR-0020. Computed per
query from the per-row labels; the annotator is never asked an ordering question (§9.1).
Distractors and `cant_tell` rows are excluded throughout.

- **4a — lead correctness (the user-visible property).** For each query, does the ranker's
  first cluster belong to that query's highest non-empty label tier? Reported as a count out
  of the number of **informative** queries, listed by name, and ⛔ **split into the three
  queries whose current leads were pre-judged during design and the rest — see §8.1, which
  requires that split at every point of use**, with the baseline alongside:
  under today's MeSH-id sort the lead is `background`-or-worse on at least three of eight
  (`Isotretinoin | Acne Vulgaris`, `Bicarbonates | Acidosis`, `Heparin | Hemorrhage`).
- **4b — tier inversions.** Count of cluster pairs the ranker places in the opposite order to
  their labels, over pairs whose labels differ. Ties are not inversions and are not errors —
  see §9.2. Reported per query and pooled, as a raw count over the raw number of comparable
  pairs, never as a bare rate.
- **4c — the diagnostic that separates two very different failures.** A query with **no
  `answers` cluster at all** is not a ranking failure; it means the cluster that would answer
  the question does not exist, which is a retrieval or entity-linking failure and belongs in
  `DEFECTS.md`. This is worth having: `isotretinoin and depression` may well be such a query,
  since no depression cluster exists in that run at all (bare `depression` NILs in all 26
  papers that contain it). **Such queries are excluded from 4a and 4b and reported
  separately.** Without this split, a linking defect would be scored as a ranking defect.

**Pre-registered action:** 4a and 4b decide whether ADR-0020's score ships as specified, is
revised, or is withdrawn. ⛔ **A revision may change the score's *structure* — which signals,
in which order — and may never introduce a threshold constant tuned against these labels.**
That would be fitting the ranker to its own validation set, and the labels are the whole
population, so there is no held-out data to catch it.

**Power for Gate 4 is worse than for Gate 3 and is stated first.** The denominator for 4a is
**8 queries minus those excluded by 4c** — plausibly 5 or 6. A single query moves it by 15–20
points, so 4a is evidence about *specific queries* and is not a rate. 4b has a larger
denominator (hundreds of comparable pairs) but the pairs are not independent, since they share
clusters and queries, so no interval is quoted for it. **Neither reading can support a claim
that the ranker is good in general; both can support a claim that it does or does not fix the
cases that are currently broken.** That is the question ADR-0020 actually asks.

**Power is stated up front, not discovered afterwards.** 13 dropped clusters is a small
denominator: one unexpected false drop moves that rate by roughly 8 points. Any statement about
the *drop* side is therefore weak evidence about a rate and strong evidence about a specific
case — which is precisely why the pre-registered action above is case-by-case rather than
rate-based. ADR-0018's underpowering was a defect because the design leaned on a rate it could
not resolve; this design deliberately does not lean on one.

---

## 6. Sanctioned uses of the labels

1. Diagnosing and fixing individual false drops (§5).
2. Reporting the filter's over-inclusion rate with its interval.
3. Deciding whether ADR-0020's ranking score ships, is revised, or is withdrawn (§5, Gate 4).
4. Separating a ranking failure from a retrieval/linking failure (§5, Gate 4c).
5. Validating a derived proxy on its own terms (§8.2) — and *only* by comparison against these
   labels, never by assuming the proxy.

**Explicitly not sanctioned:** tuning any constant in `select_stage` **or in the ranking
score**; scoring any generative output; training anything; reporting an accuracy figure
detached from its Gate 1 and Gate 2 verdicts; claiming the ranker is good *in general* on the
strength of a denominator of 5–8 queries.

---

## 7. What these labels give the ordering decision (ADR-0020)

⚠️ **Corrected 2026-09-05, after this section was first written.** It originally said ordering
was blocked because `render_cluster` states its ordering "carries no implicit ranking — a
reader must not be able to infer importance from position". **That rule was checked against
the source and it governs the order of PAPERS WITHIN a cluster, not the order of clusters
within an answer.** Cluster order has only ever been justified as "reproducible and diffable"
(`cluster_papers`). The original sentence is left visible here rather than quietly deleted,
because it was load-bearing for the claim that ordering needed an overturn, and it was wrong.
ADR-0020 sets out what is actually changing.

The **measured** case for ordering is already strong and does not depend on this pass:
selection keeps 70 of 83 clusters and does not change what an answer leads with. "isotretinoin
and depression" keeps 5 of 5 and still opens on `Isotretinoin | Acne Vulgaris`.

### 7.1 A tighter filter was tried and refuted (2026-09-05)

Before conceding that ordering is required, the stronger form of the filter was measured:
**require the cluster's disease side to match a disease-side concept of the query**, falling
back to today's any-side OR only when the query has no disease concept even through NCBI.
Disease-side membership came from the existing unused `data/canon/concept_labels.json.gz`.

It fails, and it fails in three separate ways:

| | any-side OR | disease-required |
|---|---|---|
| clusters kept (of 83) | 70 | **33** |
| queries returning an **empty answer** | 0 | **2 of 8** |
| queries whose lead cluster changed | — | **0 of 8** |

1. ⛔ **It empties two answers.** `isotretinoin and depression` keeps **0 of 5**: NCBI resolves
   the query to `Depressive Disorder` (D003866) while the clusters carry `Mental Disorders`,
   `Anxiety Disorders` and `Psychotic Disorders` — siblings and parents, never D003866.
   `lithium and thyroid dysfunction` keeps **0 of 7**: the query resolves to `Thyroiditis`, the
   clusters are `Hypothyroidism` and `Hyperthyroidism`. **The fail-open guard does not fire in
   either case, because the disease side did resolve — it just resolved to a different node of
   the same hierarchy.** This is DEF-0002's root cause reaching a second consumer.
2. **It drops clusters that are plainly on-query**: `Metformin | Acute Kidney Injury` (9
   papers, a core complication of metformin-associated lactic acidosis), `HMG-CoA reductase
   inhibitors | Myalgia` (8) and `| Muscular Diseases` (8) for statins/rhabdomyolysis,
   `Warfarin | Stroke` for warfarin/bleeding risk — the risk actually being traded off.
3. ⭐ **It changes the lead on zero queries.** metformin still opens on `Bicarbonates |
   Acidosis`; warfarin still opens on `Heparin | Hemorrhage`.

Point 3 is the structural one, and it is why no filter of any strictness can close this gap.
Clusters are emitted in `sorted(by_key)` order, so the lead is **whichever surviving cluster
has the alphabetically smallest MeSH descriptor id** — `Bicarbonates` D001639 sorts before
`Metformin` D008687, `Heparin` D006493 before `Warfarin` D014859. A filter changes which
clusters survive; it cannot change the sort. Relevance and MeSH-id collation are uncorrelated
by construction, so **the lead is fixable only by ordering.**

What these labels add is the missing quantity: **how many kept clusters are `background` rather
than `answers`, and how often a `background` cluster currently precedes an `answers` one under
MeSH-id ordering.** That is computable from the labels with no extra annotation, it is the exact
input an ordering ADR needs, and collecting it costs nothing beyond keeping `background` as a
distinct class in §2. If the count is small the ordering question closes cheaply; if it is large
the ADR has its evidence.

---

## 8. Limitations, stated rather than managed

### 8.1 Single non-independent annotator

The annotator built the system. Blinding to the filter's decision (§3.2), shuffling (§3.2),
withholding the finding sentences (§2), and the distractor control (§3.4) are real mitigations
and they are not a substitute for independence. **No inter-annotator agreement figure can be
computed from this pass, and none should be quoted.** The honest description is: a structured,
blinded, single-annotator sanity check over a complete population, with a control that says
whether the labels track their definitions.

⛔ **A SHARPER LIMITATION THAN THE GENERAL ONE ABOVE, AND THE MORE DAMAGING OF THE TWO.
Specific clusters have already been named and discussed as known failing examples, in this
design process, before any label exists — and they are disproportionately the rows that will
drive Gate 4a.**

This is not the generic non-independence caveat spread evenly across 83 rows. It is a
**concentrated prior judgment** on a handful of rows that carry unusual leverage:

- **`Isotretinoin | Acne Vulgaris`**, **`Bicarbonates | Acidosis`** and **`Heparin |
  Hemorrhage`** have each been named repeatedly, by name, as *the wrong thing to lead with*.
  They are the current leads of three of the eight queries. **Gate 4a's denominator is 5–8
  queries after §5's 4c exclusions, so a pre-judged lead on three queries is a prior on
  something like half of that gate's evidence.**
- A further set has been named in argument as clusters a filter *should* have kept —
  `Metformin | Acute Kidney Injury`, `HMG-CoA reductase inhibitors | Myalgia` and
  `| Muscular Diseases`, `Warfarin | Stroke`, `Aspirin | Hemorrhage`, `SSRI | Hemorrhage`,
  `Lactic Acid | Acidosis` — and the isotretinoin and lithium cluster sets were enumerated in
  full while arguing that the tightened filter wrongly emptied them.
- Weaker but real: **all 83 clusters have appeared by name in printed listings** during this
  work, so no row is being seen for the first time at labelling.
- The MeSH tree distances from `Depressive Disorder` and `Thyroiditis` to their competing
  clusters were computed and reported *before* labelling, so the ranker's likely behaviour on
  the flagship cases is already known to the annotator.

**Consequence, and it is not fully mitigable.** Blinding to the filter's decision does nothing
here: the contamination is not knowledge of what `select_stage` did, it is a remembered
argument about what these particular clusters *ought* to be. Shuffling does not help either,
since the bias is attached to the cluster's identity, not its position.

**What this means for how the readings may be used.** Gate 3's membership numbers are affected
least — they run over all 83 rows, most of which carry no specific prior. **Gate 4a is affected
most, and its reading must be reported with this limitation attached every time it is
quoted.** ⛔ **A Gate 4a result that merely confirms the three pre-judged leads should be
treated as the weakest possible evidence — it is close to checking whether the ranker agrees
with an opinion already formed and written down.** The informative part of Gate 4a is
therefore the queries whose leads have *not* been discussed: `cisplatin nephrotoxicity`,
`NSAIDs and gastrointestinal bleeding`, `statins and rhabdomyolysis`, `amiodarone pulmonary
toxicity`. **Those should be reported separately from the three pre-judged ones**, and if the
gate is read as a single pooled number the pooling must be called out.

The genuinely clean fix is a second annotator with no exposure to this process, which is not
available. **The available fix is to say so at every point of use rather than to discount it
once here**, which is what the paragraph above requires.

### 8.2 The MeSH-indexing cross-check is a candidate, not a shortcut

PubMed indexes every paper with MeSH descriptors, which suggests a free derived relevance
proxy: score a cluster by the overlap between its concepts and the descriptors on its own
papers. **It is not trusted, and it is not used to produce labels.**

This project has already spent one full phase discovering that a plausible derived proxy meant
something other than what it appeared to mean — the CTD contradiction proxy read π̂ = 0.067 with
an upper bound of 0.298, below anything usable (ADR-0017). The standing rule that came out of
that is: **validate a derived label's meaning before spending against it.**

So the ordering is fixed and non-negotiable: **human labels first, then the proxy is measured
against them.** Agreement is reported as a ceiling, per ADR-0017 — never as a correction factor,
and never used to "extend" the human labels to unlabelled data unless a separate decision
authorises it on the strength of the measured agreement. If the proxy disagrees with the human
labels, the proxy is discarded, not the labels.

### 8.3 Eight queries, all drug–adverse-effect shaped

The frozen corpus is eight queries of one clinical shape. Findings generalise to that shape and
are not evidence about mechanism questions, comparative-effectiveness questions, or anything
outside chemical–disease pairing. Stated because the corpus was frozen for Gate A and inherited
here, which is a genuine strength for freedom-from-selection-bias and a genuine limit on
external validity.

---

## 9. Does one round cover both questions? Yes — and what it refuses to cover

This section exists because the question was asked directly before implementation started:
does the design generalise to validate **ranking**, or only binary keep/drop?

### 9.1 The label schema needed no change; the pre-registration did

`answers` > `background` > `off_topic` was already a 3-grade ordinal scale, chosen in §2 so
that `background` — the indication, the comparator, the co-occurring condition — could not be
collapsed into either neighbour. **That is exactly the grading a ranker is checked against**,
and it is checked from the same per-row labels with nothing added.

There is also **no conflict with the blinding protocol**, which was the one real risk. §3.2
requires rows shuffled across queries so the annotator cannot see the filter's decision; an
ordering judgment would seem to require seeing a query's clusters together. It does not:
**ordering is *derived* from the per-row labels after the fact, never elicited.** The annotator
still sees one `(query, cluster)` row at a time in shuffled order and answers the same
question. Gate 4 is computed later, by machine, from labels and the ranker's output.

⭐ **What genuinely had to change is the pre-registration, and that is not a formality.** §5
and §6 committed only to membership readings. Deciding *after* seeing labels that they also
settle an ordering question would be choosing an analysis with the answers in hand — the
failure this project's gate discipline exists to prevent. Gate 4 and the widened §6 are
therefore added **now, before any label exists**, or they cannot be used at all.

### 9.2 What it deliberately will not certify: within-tier order

If a query has four `answers` clusters, these labels cannot say which of the four should lead,
and **the design refuses to ask.** Two reasons, and the second is the load-bearing one:

1. **The ranker is not asked to get it right.** ADR-0020's extended rule is that cluster order
   carries relevance to the query and nothing else. Clusters the labels grade equally are, by
   that rule, correctly in any order; the remaining order is the deterministic `sorted(by_key)`
   tiebreak and asserts nothing. Ties are therefore not errors in Gate 4b, and scoring them as
   errors would be scoring the ranker against a rule the ADR explicitly declines to adopt.
2. **Finer grades would be the forbidden metric.** Asking a single annotator to rank four
   equally on-topic clusters is asking which they *prefer* — ADR-0017's explicitly excluded
   axis, with arbitrary inter-grade boundaries and no external referent. A 5- or 7-point scale
   would produce more discriminating numbers and would not produce more trustworthy ones.

**The failure being fixed is "the lead is not about what was asked", not "the lead is the
second-best on-topic cluster".** The first is a defect a reader notices immediately and the
labels catch it; the second is a preference this project has no way to adjudicate and no
reason to.

### 9.3 Cost of covering both

Zero additional labels. 83 real rows plus 8 distractors, unchanged. The whole extension is
Gate 4, the widened sanctioned-use list, and this section — all of which must be written
before labelling begins, which is why they are.

The one real cost is **statistical, not clerical, and it is stated rather than absorbed**: Gate
4a's denominator is 5–8 queries after 4c's exclusions, so it can support "this fixes the cases
that are broken" and cannot support "this ranker is good". §5 says so at the point of reading,
and §6 forbids the stronger claim.
