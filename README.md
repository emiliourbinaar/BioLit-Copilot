# BioLit Copilot

A multi-agent biomedical literature research assistant, built as a **measurement-first**
project: every layer is priced against a free baseline before it is allowed to ship, and
three of the mechanisms that looked most promising were rejected on their own numbers.

**Status — Phases 1–4 complete; Phase 5 ran its free half and stopped there on purpose.**
There is no runnable end-to-end pipeline yet and `frontend/` is empty; what exists is the
NER → canonicalization → clustering → extraction stack, each with its own eval harness, plus a
contradiction-detection harness whose 900-pair corpus and free baselines are built and whose
paid arms were **retired by a pre-registered stop rule before they were ever called**.
16 architecture decisions record what was measured and what was rejected.

## What is actually here

| Layer | Module | Headline number |
|---|---|---|
| **Entity recognition** (Phase 2) | `biolit.ner` | F1 **0.8099** on the BC5CDR test split |
| **Canonicalization** (Phase 3) | `biolit.canon` | linking F1 **0.7842**; concept-level F1 **0.7697** |
| **Clustering / pairing** (Phase 3) | `biolit.cluster` | same-sentence pairing F1 **0.6327**, ~halving downstream LLM calls |
| **Sentence extraction** (Phase 4) | `biolit.extract` | deterministic control F1 **0.6238** — the LLM arm scored **0.3054** and was **not shipped** |
| **Contradiction detection** (Phase 5) | `biolit.critic` | 900-pair corpus, 3 free baselines at chance — **gold proxy measured invalid (π̂ 0.067) and the paid run cancelled; no paid call has ever been made** |
| **Eval harness** | `biolit_evals` | 483 tests; every run appended to a committed JSONL log |

## The part worth reading

The eval harness is the point of this project, not the pipeline. Five findings shaped it:

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

## Development

```bash
cd backend
uv sync
uv run pytest -q
uv run ruff check . && uv run pyright
```

Free evals (no credential, no API call — they download the public BC5CDR corpus on first run):

```bash
uv run python -m biolit_evals.ner_eval
uv run python -m biolit_evals.end_to_end
uv run python -m biolit_evals.extract_eval --arm control
uv run python -m biolit_evals.baselines
```

## Reading order

- `docs/EVAL_REPORT.md` — every number, its methodology, and its limitations
- `docs/DECISIONS.md` — 16 ADRs, newest first; ADR-0013 and ADR-0015 carry the standing
  findings, and ADR-0016 records why a passing test is not evidence the suite would notice a
  regression
- `docs/ARCHITECTURE.md` — how the layers fit together
- `docs/superpowers/specs/` and `docs/superpowers/plans/` — per-phase specs and plans
