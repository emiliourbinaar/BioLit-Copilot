# BioLit Copilot

A multi-agent biomedical literature research assistant, built as a **measurement-first**
project: every layer is priced against a free baseline before it is allowed to ship, and
three of the mechanisms that looked most promising were rejected on their own numbers.

**Status — Phases 1–5 closed; the pipeline runs end to end and answers the question it was
asked.** `frontend/` is empty. What exists is the NER → canonicalization → clustering →
extraction stack, each with its own eval harness and wired into a runnable CLI; a
contradiction-detection harness whose 900-pair corpus and free baselines are built and whose
paid arms were **retired by a pre-registered stop rule before they were ever called**; a
deterministic Synthesis stage that ships because the LLM arm's *gate* was shown undecidable
rather than because the arm lost; and query-conditioned selection and ordering, measured
against a blind 91-row annotation whose four gates were fixed before any label existed — and
whose control instrument later proved compromised, leaving **no attributable evidence of lead
improvement** and the defect it found resting on mechanical verification alone, and a follow-up attempt to validate the
later pharmacological-class fix that closed as a negative result about the instrument. 23
architecture decisions record what was measured and what was rejected, alongside a scope
record (`docs/SCOPE.md`) for work deliberately not attempted and a defect log
(`docs/DEFECTS.md`) for measured failures that are recorded rather than quietly carried —
including a blind adjudication finding **52% of short acronym links wrong**, in a layer whose
aggregate F1 is 0.78.

**Total paid model spend across the whole project: $2.48** — Phase 4's extractor arm, ~1500
calls, the one LLM arm ever authorised. Phase 5's Critic arms and Gate A's synthesis arm were
both **retired before a single call was made**, each by a rule written down before the number
existed. Every negative result below was reached by measuring something free first.

## What this demonstrates

Five phases, each priced before the next was built: **NER → canonicalization → clustering →
extraction → Critic**, plus post-phase work on selection and synthesis. At the phase level the
trail contains four negative results and one positive one. (Three individual *mechanisms* were
also measured and rejected — see below.)

- **Negative result — the LLM extractor was measured and rejected** in favour of a free
  deterministic control: **F1 0.3054 against 0.6238**. Cost was not the reason. **ADR-0015.**
- **Negative result — the Critic's gold standard was measured and retired before any paid run
  happened.** A stop rule fixed before the number existed read STOP on the blind annotation
  (π̂ **0.067**), so **the paid Critic arms were retired unspent**. **ADR-0017.**
- **Negative result — the search for a replacement gold standard was closed on measurement, not
  on giving up.** Six corpora across three structural families were checked against real data.
  Five failed structurally; the sixth cleared every structural bar and still annotated at
  π̂ **0.60** with **zero separation** between its filtered and unfiltered subsets. **ADR-0018.**
- **Negative result — Synthesis Gate A closed as a finding about the *gate*, not the arm.** With
  no gold, every metric must be computed from the source, so every axis measures overlap with
  the source — and the overlap axes share one degenerate maximiser: name every paper, say almost
  nothing about any of them. An `index` arm emitting `PMID x: <one word>` **dominated the
  template on both comparative axes**, so the suite could not have certified quality had the
  paid arm been run. It was not run. The deterministic template ships. **ADR-0019.**
- **Positive result — deterministic clustering and extraction hold up, and they are wired into a
  real pipeline that answers the question it was asked.** Same-sentence pairing F1 **0.6327**,
  roughly halving downstream LLM calls; `python -m biolit.pipeline` runs a live query through
  the whole stack and now selects and orders clusters against that query. **ADR-0013, ADR-0020.**

The reasoning behind each lives in the ADR, not here.

**The measurement caught a defect in the component it was validating.** The blind relevance
annotation's Gate 4 was built to score the new ranker. It found that the ranker's hierarchy
term was **unreachable by construction** — it minimised over every (cluster side × query
concept) pair, an exactly-matched side contributes `distance(x, x) == 0`, and the filter only
ever passes clusters with at least one exact match, so every cluster scored proximity 0. The
headline improvement it had just produced came *entirely* from the other term. The defect is
fixed structurally, and the labels are recorded as **spent** for that question rather than
re-read to bless the fix: DEF-0003.

## What is actually here

| Layer | Module | Headline number |
|---|---|---|
| **Entity recognition** (Phase 2) | `biolit.ner` | F1 **0.8099** on the BC5CDR test split |
| **Canonicalization** (Phase 3) | `biolit.canon` | linking F1 **0.7842**; concept-level F1 **0.7697** — but on short all-caps acronyms, adjudicated blind, **52% of links are wrong** (DEF-0001) |
| **Clustering / pairing** (Phase 3) | `biolit.cluster` | same-sentence pairing F1 **0.6327**, ~halving downstream LLM calls |
| **Sentence extraction** (Phase 4) | `biolit.extract` | deterministic control F1 **0.6238** — the LLM arm scored **0.3054** and was **not shipped** |
| **Contradiction detection** (Phase 5) | `biolit.critic` | 900-pair corpus, 3 free baselines at chance — **gold proxy measured invalid (π̂ 0.067) and the paid run cancelled before it was ever called**; the replacement-gold search closed at π̂ 0.60 after six corpora |
| **Query-conditioned selection** | `biolit.query` | clusters filtered and ordered against the asked question, fail-open and ledgered; query concepts come free from esearch's own `TranslationSet` |
| **Synthesis** | `biolit.synth` | deterministic template, **shipped because the gate for its LLM rival was shown undecidable** (ADR-0019); each paper quoted once per answer |
| **End-to-end pipeline** | `biolit.pipeline` | runnable CLI over the real components, with a per-stage drop ledger; the Critic remains an explicit `not_implemented` stub, not an empty result |
| **Eval harness** | `biolit_evals` | 704 tests; every run appended to a committed JSONL log |

## The part worth reading

The eval harness is the point of this project, not the pipeline. These are the findings that
shaped it:

**Upstream entity loss dominates, and it is unrecoverable downstream.** Measured twice: 1679
gold mentions whose spans NER got exactly right but which linked to nothing (worth +0.0821
concept F1 — 39× the next-largest ablation), and 40.3% of gold chemical–disease relations
losing an endpoint before pairing is ever consulted. No linker fixes a span NER missed, and
no pairing strategy pairs an entity that does not exist. See ADR-0013.

**Three mechanisms were measured and rejected.** MeSH dictionary enrichment (rescued 3 of
2514 missing surfaces — CTD already *contains* MeSH's entry terms); a SapBERT embedding
fallback (+0.0200 F1, but 53.9% mention precision — every second link wrong); and an LLM
sentence extractor (loses to four lines of positional heuristic). Each is documented with the
number that killed it. ADR-0011, ADR-0012, ADR-0015.

**A headline claim of this project was retracted after review, and the retraction is kept in
place of the claim.** Phase 4 originally reported the LLM arm recovering 47% of a population
the deterministic control missed — scored against a comparator that was **structurally
incapable of scoring above zero on it**. The generalized lesson is recorded so it is caught
next time: *when an arm is scored on a population defined by another arm's failures, that
second arm's zero is a tautology, and only a baseline matched on the first arm's spend tests
the claim.* Two such baselines overturned it; a later sweep of seventeen free positional
heuristics found **every budget-matched one of them beats the LLM arm.**

**The same trap was caught the second time before anything was built.** Phase 5 needed
contradiction gold, and BC5CDR was the obvious source. It cannot supply it: BC5CDR annotates
chemical-*induced*-disease relations only, so every gold key is `marker/mechanism` and
disagreement is impossible by construction — **0 opposed pairs across all three splits**,
against 650 agreeing ones. Same fixed-outcome shape as the retraction above, found by checking
the corpus first rather than by reporting a number out of it. Gold derives from CTD's
direction field instead, and because that label *is* a direction flip, the design reports the
blind-annotation agreement rate as a **ceiling and never as a correction factor** — a critic
that merely detects direction flips would score near 1.0, so high recall against this proxy is
evidence of mimicry rather than quality.

**Then the replacement gold failed too, and the stop rule caught it.** Gold derived from CTD's
direction field, so the blind annotation measured its validity before any paid arm ran. A human
agreed with **1 of 15** proxy-labelled contradictions (π̂ = 0.067, 95% CI [0.012, 0.298]) — CTD's
`marker/mechanism` vs `therapeutic` mostly encodes *dual pharmacology*, not disagreement. A drug
that treats a condition and can also cause it is not a contradiction. Gate 2, calibrated and
committed before any result existed, read STOP and **the paid run was cancelled rather than
renegotiated.** The corpus and baselines remain valid; the labels do not. This is the first time
in the project the check was in place *before* the claim rather than added after being wrong.

**Then the search for a replacement gold was closed — on measurement, not on giving up.** Six
corpora across three structural families were checked against real data rather than their
descriptions. Five failed *structurally*, each for a specific measured reason: SciFact's
CONTRADICT claims are annotator negations of a SUPPORT twin sharing identical evidence
documents (46–76% of them); HealthVer ships no source document ids at all; NLI4CT's balance is
constructed 950/950; MultiCite is 94.3% NLP papers and omits the cited text entirely; SciCite's
disagreement signal is not recoverable from the cited abstract (median 20% term overlap). The
sixth — a cardiovascular claim-pair corpus derived from systematic reviews — cleared every one
of those bars, and was annotated blind at 45 pairs. It read **π̂ = 0.60 in both strata (9/15
each)**, and the lexical filter meant to rescue it produced **zero separation**.

**That last reading is reported with its limits, in both directions.** The 95% interval is
[0.36, 0.80], so a usable π of 0.80 is *not excluded* — this is a "do not proceed", not a
"proven invalid", and the ADR says so rather than rounding toward the tidier conclusion. The
filter null is weaker still: at n=15 per stratum a test would detect a real 0.8-vs-0.6 filter
effect only **11.2%** of the time. That underpowering traces to a specific mistake worth
recording — the n was inherited from a **one-sample** stop rule and carried unexamined into a
**two-sample** comparison. *A calibrated constant is calibrated for one question; carrying it to
a second is a new assumption.* One thing did work: cross-question distractor pairs, five extra
rows, established that the annotator was not simply being strict — the first batch in the
project where that confound is ruled out by measurement rather than assumed away.

**An aggregate F1 can hide a class where the component is wrong more often than right.**
Canonicalization scores linking F1 **0.7842**, and the number is real. But short all-caps
acronyms — 204 linked mentions across eight queries — were adjudicated blind against their
source abstracts, and **23 of 44 distinct (surface, concept) pairs are wrong: 52% by pair, 65%
by mention.** `GSH` → *Glucocorticoid-Remediable Aldosteronism* where the abstract says
glutathione; `CP` → *Cleft Palate* where it says cisplatin; `RA` → *Rheumatoid Arthritis* where
it says rosmarinic acid. The cause is that a surface is linked to whichever concept owns that
acronym in CTD, with nothing consulting the surrounding text. This is a **census** of those
eight queries, not a sample, so no interval is quoted and none would mean anything.

Two things make it worth reading past the number. **A free check already exists and is
discarded:** the project's own concept-type table contradicts the mention's NER label on
**87 of 3,696 links**, catching 10 of these 44 pairs at 10/10 precision — and it is unreachable
because the `Linker` protocol takes a surface and is never told the label (DEF-0004). **And the
obvious fix is still not obvious:** one of those ten, `AT`, is NER cutting the letters out of
`ATO` (atorvastatin), so a type-constrained linker would score it a success and fix a span
defect not at all. The fix stays unbuilt because refusing these links converts wrong entities
into NILs, and abstention is already this project's measured dominant failure mode — a trade
that has to be scored on BC5CDR, not asserted here.

**That pass also cost the project a rule, and caught a defect in its own controls.** 17 of the
44 pairs had been named — in the defect log, or in the design conversation — before labelling
began, so the reading is reported **split**: 13/17 disclosed against **10/27 undisclosed**. The
gap is explicitly *not* read as disclosure bias, because the disclosed set was selected for
looking wrong in the first place, and selection and disclosure are confounded here in a way the
design cannot separate. Having hit that shape twice, it is now **ADR-0016 rule 6**: evidence
shown to an annotator before labelling gets tracked as a named constant and reported split,
never averaged into one headline. Separately, the controls turned out to be **structurally
identifiable** — each was built by re-pairing a real row and keeping its surface, and the
population is a census, so every control duplicated a real row's surface and could be spotted
without reading. The labels refute the shortcut having been used (four duplicated surfaces had
*both* members marked wrong, which a duplicate-spotter cannot produce), but the gate is
reported with the caveat attached rather than as clean: surviving on evidence found afterwards
is not the same as being designed correctly.

**A gate with no gold cannot certify generative quality, and that is structural.** Synthesis
Gate A asked whether an LLM synthesiser earns the stage over the deterministic template, using
a metric suite built without a gold standard. Every axis therefore had to be computed from the
source text, so every axis measured *overlap with the source* — and every overlap axis is
maximised by the same degenerate strategy. Demonstrated rather than argued: an `index` arm
emitting `- PMID x: <one word>` scored coverage 1.0, hallucinated 0, distinguishing-content
retention 1.0 and compression 0.0516 against the template's 1.1417, **dominating on both
comparative axes**, while a genuine synthesis failed 28 of 30 clusters. The gate was retired,
not the arm — and the sharper process lesson is separate: **a harness self-test that runs the
CONTROL arm is structurally blind to defects only the TREATMENT arm can trigger.** Five such
defects were found on that branch, and every one passed the end-to-end self-test both before
and after its fix. ADR-0019.

**A validation pass was retired against itself, and the exact reason matters more than the
verdict.** Query-conditioned ordering was measured against 91 blind rows — 83 real clusters
plus 8 cross-query distractors — with four gates fixed before a single label existed. The sheet
showed the question and two concept names and nothing else: no MeSH ids, no paper counts, no
year ranges, because size and recency are signals the ranker is *forbidden* to use and a label
nudged by either would let the gate reward them. The headline read **5/8 leads correct against
a 3/8 baseline**; three of the eight queries had their computed ordering disclosed before
labelling, and on the three clean queries it read **2/3 against a 2/3 baseline**.

Then the control instrument itself failed. Every distractor was a real cluster shown under a
foreign question, and the population is *exhaustive* — so **all 8 distractors duplicated a real
row's cluster key**, and an annotator spotting the same concept pair twice knows one is planted
without judging anything. Unlike a later pass with the same defect, these labels **cannot**
distinguish that shortcut from genuine reading: each duplicated key read `off_topic` for the
distractor and something else for its twin, which is what both produce (ADR-0021).

⚠️ **The precise standing status, because the distinction is the whole point:** there is **no
attributable evidence of lead improvement from these labels** — not a null result, but a
compromised instrument — and **DEF-0003's fix stands on independent mechanical verification
only.** "Inconclusive" would lose exactly that difference. What the pass *did* establish is
unaffected, and it is not small: Gate 4 found DEF-0003, a real defect in the component it was
built to score. ADR-0020, ADR-0021, DEF-0003.

Gate 3 found a second one, and closing it overturned the obvious fix. Three `answers` clusters
carrying `Atorvastatin` were deleted under "statins and rhabdomyolysis" because the query
resolves to the drug *class* — so walk the MeSH tree down from class to member and keep them.
**You cannot.** `Atorvastatin` is filed under chemical structure (D03/D10) and
`HMG-CoA Reductase Inhibitors` under chemical actions and uses (D27); they share no node, and
the tree distance between a statin and the statin class is `None`. For drug classes the
relation simply is not in the hierarchy — it is MeSH's `PharmacologicalAction` field, in a dump
already on disk. Matching on it recovers all three, and because filtering more widely while
scoring unchanged would have buried them at positions 15–17 of 17 beneath seven background
clusters, the *same* predicate now drives both the filter and the ranker. ⚠️ Everything beyond
"it recovers the three" re-reads labels already spent on DEF-0003, so it is reported as
descriptive, not as validation. ADR-0022.

Validating that ordering claim then failed, and the failure is the more useful result. Two
prior label sets were spent — one to disclosure, one to a broken control — so a third needed
queries never run or discussed. Twenty-five were resolved and run behind a screen whose
blindness is enforced by the import graph rather than by intent: it imports nothing from
`biolit`, and a test spawns a fresh interpreter and asserts the ranker is absent from
`sys.modules`. **Four cleared the productivity floor.** Not a bad draw — cluster yield collapses
in proportion to how collective the drug term is (`no_cluster` 31% for single agents, 74% for
drug classes), and this mechanism fires only on the collective end. No arm disagreement was
ever scored. ADR-0023, DEF-0005.

## Development

```bash
cd backend
uv sync
uv run pytest -q
uv run ruff check . && uv run pyright
```

Run a query through the whole pipeline — live PubMed retrieval, NER, canonicalization, the
licence gate, deterministic extraction, clustering, query-conditioned selection and ordering,
and synthesis. Free: no credential, no LLM call, no paid API:

```bash
# One-time: build the two MeSH artifacts selection and ordering read. No network — both
# parse the NLM descriptor dump at data/mesh/desc2026.gz, which is gitignored along with
# their outputs. Both are REQUIRED: a missing one fails the run at SELECT rather than
# degrading quietly.
uv run python -m biolit.canon.build_mesh_tree     # 31,108 descriptors, 65,360 tree placements
uv run python -m biolit.canon.build_mesh_actions  # 2,838 descriptors, 5,084 class memberships

uv run python -m biolit.pipeline --query "metformin and lactic acidosis" --max-papers 20
# add --json-out run.json to dump the full PipelineState
```

It prints a per-stage ledger accounting for every paper — including what the licence gate
refused and why, how many clusters selection kept and on what concept evidence, and whether
ordering ran or fell open — then a deterministic answer to the query. The Critic stage is
labelled `NOT IMPLEMENTED` rather than returning an empty result that would read as "ran and
found nothing".

Free evals (no credential, no API call — they download the public BC5CDR corpus on first run):

```bash
uv run python -m biolit_evals.ner_eval
uv run python -m biolit_evals.end_to_end
uv run python -m biolit_evals.extract_eval --arm control
uv run python -m biolit_evals.baselines
```

## Reading order

- `docs/EVAL_REPORT.md` — every number, its methodology, and its limitations
- `docs/DECISIONS.md` — 23 ADRs, newest first; ADR-0013 and ADR-0015 carry the standing
  findings, ADR-0017 closes Phase 5 as a negative result, ADR-0018 closes the replacement-gold
  search and records why a stratified null needs its own power calculation, ADR-0019 retires
  Synthesis Gate A as a finding about the gate, ADR-0020 orders clusters by relevance without
  repealing the within-cluster no-ranking rule, ADR-0021 records why an annotation control over
  an exhaustive population cannot be a re-paired member of it, ADR-0022 records that a drug
  class is not a tree ancestor of its members — so the fix everyone reaches for first is
  impossible — ADR-0023 closes its validation as a negative result about the instrument rather
  than the fix, and ADR-0016 collects six
  verification rules — why a passing test is not evidence the suite would notice a regression,
  and why evidence disclosed to an annotator has to be tracked rather than averaged away
- `docs/SCOPE.md` — work deliberately **not** attempted, with the reasoning that would have to
  be answered to reopen it. SR-0001 puts narrative synthesis out of scope and names the one
  framing that would be genuinely different
- `docs/DEFECTS.md` — measured defects in shipped components, recorded rather than carried
  quietly. Every entry so far is in entity linking, which is the project's dominant and
  downstream-unrecoverable bottleneck. DEF-0001 carries the 52% acronym rate and the blind
  adjudication behind it; DEF-0004 the free type check the linker cannot reach
- `docs/ARCHITECTURE.md` — how the layers fit together
- `docs/superpowers/specs/` and `docs/superpowers/plans/` — per-phase specs and plans
