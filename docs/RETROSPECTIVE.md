# BioLit Copilot — Retrospective

**What this document is.** The project's argument, end to end, in one place. Everything here
is recorded somewhere else too — across 23 ADRs, 6 defect entries, a scope record and
`EVAL_REPORT.md` — but only in the order it happened to be discovered. This is the same
material in the order it makes sense.

**It cites; it does not recompute.** Every number below has a home in `EVAL_REPORT.md`, a
committed run log, or the ADR that decided it. No claim appears here that appears nowhere else.

---

## 1. The thesis

BioLit Copilot is a multi-agent biomedical literature assistant, built to a single rule: **price
every layer against a free baseline before allowing it to ship.** The pipeline works — it takes
a clinical question, retrieves, recognises entities, canonicalizes them to MeSH, enforces
licence rights, extracts findings, clusters them, filters and orders against the question, and
answers deterministically. But the interesting output is not the pipeline. It is that the rule
kept returning *no*: four of the five phases ended in a negative result, three separate
mechanisms were measured and rejected, and the most-cited numbers in the project are the ones
that killed something. **Total paid model spend: $2.48.**

The second thing this project turned out to be about was not planned. Six times, the thing that
broke was **the measuring instrument**, not the component under test — and each time the
downgrade was published in place of the number it invalidated. That is §4, and it is the part I
would want read first.

---

## 2. The ledger

| Pass | Question asked | Result | Outcome | Paid |
|---|---|---|---|---|
| **Phase 1** — foundations | What is the storage/state/rights model? | `Paper` as single normalization boundary; full-text existence and extraction rights as independent axes | Shipped | $0 |
| **Phase 2** — NER | Can a local checkpoint find chemicals and diseases? | **F1 0.8099** on BC5CDR test; blind in-domain sample annotated from scratch | Shipped | $0 |
| **Phase 3** — canonicalization | Can surfaces be linked to MeSH? | linking **F1 0.7842**, concept-level **0.7697** | Shipped | $0 |
| ↳ dictionary enrichment | Would more MeSH aliases help? | **3 of 2514** missing surfaces rescued — CTD already contains MeSH's entry terms | **Rejected** | $0 |
| ↳ embedding fallback | Would SapBERT rescue the NILs? | **+0.0200 F1** at **53.9%** mention precision — every second new link wrong | **Rejected** | $0 |
| **Phase 3** — clustering | Does pairing need a relation extractor? | same-sentence **paper-pair F1 0.6327** vs cross-product 0.5484; **40.3%** of gold relations lose an endpoint upstream first | Shipped; extractor **rejected** | $0 |
| **Phase 4** — extraction | Does an LLM beat a positional heuristic? | LLM **sentence-selection F1 0.3054** vs deterministic **0.6238**, identical budget | **Rejected** | **$2.48** |
| **Phase 5** — Critic | Is the derived gold standard valid? | **π̂ 0.067** on blind annotation; stop rule fired | **Retired unspent** | $0 |
| ↳ replacement gold | Does any corpus support the paper-pair unit? | six corpora, three families; best **π̂ 0.60** — annotator agreement, not an F1 — with zero separation | **Closed** | $0 |
| **Gate A** — synthesis | Can a metric suite with no gold certify quality? | a one-word-per-paper `index` arm **dominates the template** on both comparative axes | **Gate retired**, template ships | $0 |
| **Post-phase** — selection | Does consulting the question help? | 83 clusters → 75; a strictly tighter filter returns an **empty answer on 2 of the 8 queries** | Shipped | $0 |
| **Post-phase** — ordering | Does relevance ordering fix the lead? | Gate 4a **leads correct 5 of 8, baseline 3 of 8** — but **2 of 3 vs 2 of 3** on uncontaminated queries; control instrument later compromised | Shipped, **unvalidated** | $0 |
| **Post-phase** — acronym census | How wrong is linking on short acronyms? | **23 of 44 pairs (52.3%)**, **132 of 204 mentions (64.7%)** wrong | Defect recorded | $0 |
| **Post-phase** — class matching | Can class-level queries be fixed? | 70 → **75** kept, three named false drops recovered, zero off-topic admitted | Shipped, ordering **unvalidated** | $0 |
| ↳ its validation | Can a fresh label set score it? | **4 of 25** candidate queries productive; mechanism fires where yield collapses | **Closed as a negative result about the instrument** | $0 |

---

## 3. The arc, in five acts

**Act I — the layers hold up.** NER and canonicalization were built and measured against
BC5CDR plus a blind in-domain sample annotated from scratch (ADR-0006, so the annotator could
not be anchored by the model's output). F1 0.8099 for NER, linking F1 0.7842 for
canonicalization. Nothing dramatic; the foundation is real, and every later negative result
stands on it being real.

**Act II — the mechanisms that looked promising lost.** Three in a row. MeSH dictionary
enrichment rescued 3 surfaces of 2514, because CTD already contained MeSH's entry terms — the
alias gap was never an alias-coverage gap (ADR-0011). A SapBERT embedding fallback bought
+0.0200 F1 at 53.9% mention precision, so every second link it added was wrong (ADR-0012). And
chemical–disease relation extraction was priced *before* being built: same-sentence
co-occurrence already captures 77.0% of a precision-perfect ceiling (a share of available
headroom, not an accuracy), and **40.3% of gold relations lose an endpoint before pairing is
ever consulted** (ADR-0013). ⭐ That last number
became the project's dominant standing finding: **upstream entity loss is the binding
constraint, and it is unrecoverable downstream.** No linker fixes a span NER missed; no pairing
strategy pairs an entity that does not exist.

**Act III — the one paid experiment, and it lost.** Phase 4 bought an LLM sentence extractor:
~1500 calls, $2.48, the only paid arm ever authorised. It scored **0.3054** against a
deterministic control's **0.6238** on the same sentence-selection task at an identical budget,
and a later sweep of seventeen free positional heuristics found **every budget-matched one
of them beats it**
(ADR-0015). Cost was not the reason it was rejected. It simply lost.

**Act IV — Phase 5 stopped before spending anything.** The Critic needed a gold standard, and
the CTD-derived proxy was measured before it was used: a blind annotation read **π̂ 0.067**,
the pre-registered stop rule fired, and the paid arms were retired unspent (ADR-0017). The
search for a replacement closed after six corpora across three structural families; the best
candidate cleared every structural bar and still annotated at **π̂ 0.60** (agreement, not an
F1) **with zero separation** between its filtered and unfiltered subsets (ADR-0018). ⭐ The
recurring failure was not the corpora but **the unit**:
`ContradictionFinding(paper_id_a, paper_id_b, …)` asks two abstracts to be commensurable, and
real literature separates its findings by population, dose, route and endpoint in ways that
make most opposed-looking pairs genuinely compatible. Reviving the Critic
means changing the unit, which is a new spec rather than a next step.

**Act V — the pipeline was assembled, and then the instruments started failing.** Selection and
ordering shipped (ADR-0020), scored by a blind 91-row annotation with four gates fixed in
writing beforehand. That pass found a real defect in the ranker it was scoring (DEF-0003), then
its own control instrument was found compromised (ADR-0021), which retroactively withdrew the
validation. A follow-up fix for class-level drug queries shipped on a structural argument
(ADR-0022), and the attempt to validate *it* closed as a negative result about the instrument
(ADR-0023), surfacing a new clustering defect on the way (DEF-0005). **This act produced no
validated improvement and four durable findings.**

---

## 4. ⭐ Six times the instrument was the thing that broke

This is the part of the project I would defend hardest, and it was not designed. *(The
retrospective was planned around five of these; auditing turned up a sixth — ADR-0013's ceiling
claim — which had been corrected in place and never counted alongside the others.)*

**1. A "ceiling" that the mechanism beat.** ADR-0013's spec claimed Arm A (gold entities, 1500
docs) was a ceiling for Arm B (real pipeline, 500 docs). The arms ran on different corpora and
were never comparable; made apples-to-apples they **invert** — Arm A restricted to Test-500
scores paper-pair F1 **0.5802**, *below* the real pipeline's **0.6327**. **The reusable rule:** any
construction that improves recall by granting something correct, without touching what the
system emits wrongly, bounds recall alone. *Before calling anything a ceiling, ask what it can
make worse.*

**2. A headline retracted after review.** Phase 4 originally reported the LLM arm recovering
**47% of a population the deterministic control missed** — scored against a comparator that was
**structurally incapable of scoring above zero on that population**. Two spend-matched
baselines overturned it. **The rule:** when an arm is scored on a population defined by another
arm's failures, that second arm's zero is a tautology; only a baseline matched on the first
arm's spend tests the claim. The retraction is kept **in place of** the claim.

**3. A gold standard measured invalid before it was used.** Phase 5's CTD-derived contradiction
proxy read **π̂ 0.067** against a blind annotation. The stop rule had been written before the
number existed, so it fired cleanly and the paid arms were never called (ADR-0017). This is the
instrument failing in the good direction: caught by design, for $0.

**4. A metric suite that ranked outputs in the opposite order to their quality.** Gate A had no
gold, so every axis had to be computed from the source — and every source-overlap axis is
maximised by the same degenerate strategy. An `index` arm emitting `PMID x: <one word>`
**dominated the template on both comparative axes** (ADR-0019). **The rule:** a deterministic
metric suite with no gold cannot certify generative quality once brevity and coverage alone can
be gamed. That is a structural limit, not a tuning gap. **No arm was ever bought; the gate was
retired instead.**

**5. A control that could be answered without doing the task.** The relevance pass planted 8
distractors and read **8 of 8** — `DISCRIMINATING`. But the population was *exhaustive*, so
every distractor duplicated a real row's cluster key, and an annotator noticing the same
concept pair twice knows one is planted without judging anything (ADR-0021). ⛔ **These labels
cannot distinguish the shortcut from genuine reading**, so Gate 2's verdict was withdrawn and a
shipped component's validation was **retroactively downgraded**. **The rule:** a control is an
instrument only if it can be answered *only* by doing the task — and memory is just one way
that fails; structure is another, and it needs no disclosure at all to bite. The contrasting
case is kept beside it: ADR-0018's Alamri pass drew from a sample with spare material and
reused nothing.

**6. An instrument that could not reach the mechanism.** ADR-0022's ordering claim needed a
fresh label set — the two existing ones were spent, one to disclosure and one to control
duplication. 25 new queries were resolved and run behind a screen whose blindness is enforced
by the import graph. **4 cleared the productivity floor.** The reason was structural: cluster
yield collapses in proportion to how collective the drug term is (`no_cluster` **31%** for
single agents, **74%** for drug classes), and the mechanism under test fires only at the
collective end (ADR-0023). **No arm disagreement was ever scored.**

### The inverse also happened, once

**DEF-0003 is the eval finding a defect in the component it was built to score.** Gate 4 was
built to grade the new ranker. It found that the ranker's hierarchy term was **unreachable by
construction** — every cluster scored proximity 0 — so the improvement Gate 4a had just
reported came *entirely* from the other term. Four of eight queries produced a single distinct
score across all their clusters. That is the harness doing exactly what a harness is for, and
it is why the harness is the point of this project rather than the pipeline.

---

## 5. What shipped, and on what evidence

⚠️ **The evidence column is the honest one, and it is not flattering everywhere.** Three shipped
behaviours rest on a design argument and a mechanical check, with **no attributable label
evidence at all.** They are marked as such here rather than described in the same voice as the
gold-measured layers.

| Component | Ships | Evidence basis | What that actually means |
|---|---|---|---|
| `biolit.ner` | ✅ | **Gold-measured** | F1 0.8099 on BC5CDR test + blind in-domain sample |
| `biolit.canon` | ✅ | **Gold-measured** | linking F1 0.7842 — ⚠️ but see DEF-0001: 52% wrong on short acronyms |
| `biolit.cluster` | ✅ | **Gold-measured** | same-sentence **paper-pair** F1 0.6327 against gold CID relations |
| `biolit.extract` (deterministic) | ✅ | **Gold-measured** | **sentence-selection** F1 0.6238, beat the paid arm and 17 free heuristics |
| Licence gate | ✅ | **Rule-verified** | one enforcement point; never infers rights from PMC presence |
| `select_stage` filtering | ✅ | **Gold-measured (weakly)** | Gate 3's confusion matrix over 83 labelled clusters; 3 false drops found and later fixed |
| **Cluster ordering (ADR-0020)** | ✅ | ⛔ **Design argument + mechanical check only** | **No attributable evidence of lead improvement.** Not a null result — the instrument that would have made the reading attributable was itself defective (ADR-0021). DEF-0003's fix rests on all-tied queries falling 4/8 → 2/8 and nothing else. |
| **Pharmacological-class matching (ADR-0022)** | ✅ | ⛔ **Design argument + mechanical check only** | Verified: it recovers the three named false drops. Everything else — keep-set composition, improved ordering — re-reads spent labels and is **descriptive**. Its ordering claim is **not validatable by this instrument at reasonable cost** (ADR-0023). |
| **Synthesis template (ADR-0019)** | ✅ | ⚠️ **Ships because its rival's gate was undecidable** | Not "the template won a comparison." No LLM arm was ever bought. The template ships because it is the only option requiring no unjustifiable judgment call, and everything it says traces to a source sentence. |
| `biolit.critic` | ❌ | **Explicit `not_implemented`** | No validated gold exists. Reported as a stub rather than returning `contradictions: []`, which would be indistinguishable from "ran and found nothing." |

---

## 6. The standing rules, and the mistake each one cost

Every rule below was bought with an error. They are stated in the ADRs; the pairing with the
mistake is what makes them transferable.

| Rule | The mistake that produced it |
|---|---|
| **No infrastructure without a demonstrated consumer** (ADR-0013) | Two modules built before anything needed them |
| **An arm scored on a population it cannot lose on proves nothing** (ADR-0015) | Phase 4's 47% headline, retracted |
| **Before calling anything a ceiling, ask what it can make worse** (ADR-0013) | A "ceiling" the real pipeline beat by 0.05 F1 |
| **Every operand of a compound condition needs its own fixture** (ADR-0014) | A test that passed for the wrong reason |
| **Six verification rules** — incl. *disclosed evidence is tracked as a named constant and reported split, never averaged into one headline* (ADR-0016) | A 5/8 headline that became 2/3 vs 2/3 once split |
| **A control must be answerable ONLY by doing the task** (ADR-0021) | 8-of-8 discrimination that a duplicate-spotter could have produced |
| **π is a ceiling, not a score; a single annotator needs a discriminating control** (ADR-0017/0018) | Two annotation passes that could not attribute their readings |
| **Never move a threshold to match an observation** | Refused three times: DCR's flat tolerance, ADR-0022's `k=2`, ADR-0023's cluster floor |
| **Upstream entity loss is the binding constraint** (ADR-0013) | Two rounds of optimising downstream of the real bottleneck |

---

## 7. Known defects

All six are recorded rather than quietly carried. Full entries in `DEFECTS.md`.

| | What | Status |
|---|---|---|
| **DEF-0001** | Short acronyms link to whichever concept owns them in CTD, with no context check | **Measured** — 52.3% of pairs, 64.7% of mentions wrong; 10/27 on undisclosed pairs. Not fixed: refusing these links converts wrong entities into NILs, and abstention is already the dominant failure mode |
| **DEF-0002** | A bare parent-concept mention is indistinguishable from its specific child | Not fixed; **partially routed around** downstream since ADR-0022 (chemical side, selection only) |
| **DEF-0003** | ADR-0020's hierarchy term was unreachable — every cluster scored proximity 0 | **Fixed** structurally; **unvalidated** — mechanical check only |
| **DEF-0004** | A link whose concept type contradicts the mention's own NER label is never refused | Not fixed. 87 of 3,696 links (2.4%); 10/10 precision on the adjudicated set. The check is free and **unreachable** — the `Linker` protocol never receives the label |
| **DEF-0005** | Same-sentence clustering collapses on class-referring prose | Not fixed. `no_cluster` 74% on class queries vs 31% on single agents; **flat in paper count** |
| **DEF-0006** | `--json-out` serialises abstracts of papers the licence gate refused | Not fixed. 7 of 8 refused papers carried a verbatim abstract. The gate is correct; its *sufficiency argument* lapsed when a third consumer of `Paper` appeared |

---

## 8. What is genuinely unfinished

**The Critic needs a different unit, not a better corpus.** Six corpora failed the same way.
A unit carrying qualifying context explicitly — population, dose, route, endpoint — is a new
spec (ADR-0018, SR-0001).

**Entity recall is still the ceiling.** 40.3% of gold relations lose an endpoint before pairing
runs. Every downstream component should be priced against an entity-conditioned ceiling rather
than an oracle-entity one; ADR-0013 gives the template.

**DEF-0005 caps class-level questions.** Ask about "anticoagulants" rather than "warfarin" and
the pipeline mostly returns nothing, at any corpus size. It also gives ADR-0013's unmeasured
alternative (4) — a same-paragraph or N-token window — its first real motivation.

**Two shipped behaviours have no attributable evidence** (§5). Closing that needs a different
instrument, not more queries.

**`frontend/` is empty**, deliberately: the eval harness was the differentiating work and was
never allowed to be crowded out.

---

## 9. Verifying anything here

Every number traces to `EVAL_REPORT.md`, a committed run log under `evals/*.jsonl`, or the ADR
that decided it. Annotation labels and their hashes are committed; the sheets they came from
are gitignored because they carry abstract text.

```bash
cd backend
uv run ruff check . && uv run ruff format --check . && uv run pyright && uv run pytest -q

# One-time: both MeSH artifacts are required; no network, parses a dump already on disk.
uv run python -m biolit.canon.build_mesh_tree
uv run python -m biolit.canon.build_mesh_actions

# The whole pipeline, free: no credential, no LLM, no paid API.
uv run python -m biolit.pipeline --query "metformin and lactic acidosis" --max-papers 20
```

⚠️ **Two reproducibility caveats, stated rather than discovered later.** The relevance and
ADR-0023 measurements depend on NCBI's `TranslationSet`, cached under a **gitignored** `data/`;
NCBI's translation can drift, so regenerate the cache and expect drift rather than assuming a
regression. And the per-phase gold corpora (BC5CDR, CTD) download on first run and are not
committed.

---

*Companion documents: `DECISIONS.md` (23 ADRs, newest first) · `DEFECTS.md` (6 entries) ·
`SCOPE.md` (work deliberately not attempted) · `EVAL_REPORT.md` (every number and its
methodology) · `ARCHITECTURE.md` (how the layers fit together).*
