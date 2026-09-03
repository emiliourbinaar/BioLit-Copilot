# BioLit Copilot

A multi-agent biomedical literature research assistant, built as a **measurement-first**
project: every layer is priced against a free baseline before it is allowed to ship, and
three of the mechanisms that looked most promising were rejected on their own numbers.

**Status — Phases 1–4 complete; Phase 5 ran its free half and stopped there on purpose; the
pipeline runs end to end.** `frontend/` is empty. What exists is the NER → canonicalization →
clustering → extraction stack, each with its own eval harness and wired into a runnable CLI,
plus a contradiction-detection harness whose 900-pair corpus and free baselines are built and
whose paid arms were **retired by a pre-registered stop rule before they were ever called**.
The follow-up search for a replacement gold standard is now **closed after six corpora across
three structural families** — none produced a usable label (ADR-0018). 18 architecture
decisions record what was measured and what was rejected.

## What this demonstrates

Five phases, each priced before the next was built: **NER → canonicalization → clustering →
extraction → Critic**. At the phase level the trail contains three negative results and one
positive one. (Three individual *mechanisms* were also measured and rejected — see below.)

- **Negative result — the LLM extractor was measured and rejected** in favour of a free
  deterministic control: **F1 0.3054 against 0.6238**. Cost was not the reason. **ADR-0015.**
- **Negative result — the Critic's gold standard was measured and retired before any paid run
  happened.** A stop rule fixed before the number existed read STOP on the blind annotation
  (π̂ **0.067**), so **the paid Critic arms were retired unspent**. **ADR-0017.**
- **Negative result — the search for a replacement gold standard was closed on measurement, not
  on giving up.** Six corpora across three structural families were checked against real data.
  Five failed structurally; the sixth cleared every structural bar and still annotated at
  π̂ **0.60** with **zero separation** between its filtered and unfiltered subsets. **ADR-0018.**
- **Positive result — deterministic clustering and extraction hold up, and they are wired into a
  real pipeline.** Same-sentence pairing F1 **0.6327**, roughly halving downstream LLM calls;
  `python -m biolit.pipeline` runs a live query through the whole stack. **ADR-0013.**

The reasoning behind each lives in the ADR, not here.

## What is actually here

| Layer | Module | Headline number |
|---|---|---|
| **Entity recognition** (Phase 2) | `biolit.ner` | F1 **0.8099** on the BC5CDR test split |
| **Canonicalization** (Phase 3) | `biolit.canon` | linking F1 **0.7842**; concept-level F1 **0.7697** |
| **Clustering / pairing** (Phase 3) | `biolit.cluster` | same-sentence pairing F1 **0.6327**, ~halving downstream LLM calls |
| **Sentence extraction** (Phase 4) | `biolit.extract` | deterministic control F1 **0.6238** — the LLM arm scored **0.3054** and was **not shipped** |
| **Contradiction detection** (Phase 5) | `biolit.critic` | 900-pair corpus, 3 free baselines at chance — **gold proxy measured invalid (π̂ 0.067) and the paid run cancelled before it was ever called**; the replacement-gold search closed at π̂ 0.60 after six corpora |
| **End-to-end pipeline** | `biolit.pipeline` | runnable CLI over the real components, with a per-stage drop ledger; Critic and synthesis are explicit `not_implemented` stubs, not empty results |
| **Eval harness** | `biolit_evals` | 557 tests; every run appended to a committed JSONL log |

## The part worth reading

The eval harness is the point of this project, not the pipeline. Six findings shaped it:

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

## Development

```bash
cd backend
uv sync
uv run pytest -q
uv run ruff check . && uv run pyright
```

Run a query through the whole pipeline — live PubMed retrieval, NER, canonicalization, the
licence gate, deterministic extraction, and clustering. Free: no credential, no LLM call, no
paid API:

```bash
uv run python -m biolit.pipeline --query "metformin and lactic acidosis" --max-papers 20
# add --json-out run.json to dump the full PipelineState
```

It prints a per-stage ledger accounting for every paper — including what the licence gate
refused and why — and labels the Critic and synthesis stages `NOT IMPLEMENTED` rather than
returning an empty result that would read as "ran and found nothing".

Free evals (no credential, no API call — they download the public BC5CDR corpus on first run):

```bash
uv run python -m biolit_evals.ner_eval
uv run python -m biolit_evals.end_to_end
uv run python -m biolit_evals.extract_eval --arm control
uv run python -m biolit_evals.baselines
```

## Reading order

- `docs/EVAL_REPORT.md` — every number, its methodology, and its limitations
- `docs/DECISIONS.md` — 18 ADRs, newest first; ADR-0013 and ADR-0015 carry the standing
  findings, ADR-0017 closes Phase 5 as a negative result, ADR-0018 closes the replacement-gold
  search and records why a stratified null needs its own power calculation, and ADR-0016 records
  why a passing test is not evidence the suite would notice a regression
- `docs/ARCHITECTURE.md` — how the layers fit together
- `docs/superpowers/specs/` and `docs/superpowers/plans/` — per-phase specs and plans
