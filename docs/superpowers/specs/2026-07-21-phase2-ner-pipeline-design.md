# BioLit Copilot — Phase 2 (NER Pipeline) Design Spec

- **Date:** 2026-07-21
- **Status:** Approved (design), pending spec review
- **Scope:** Phase 2 only — a local biomedical NER module (chemicals + diseases) and the NER eval harness that produces a first, defensible F1. Builds on Phase 1 (`Paper`, `Entity`, `ExtractedRecord`). Later phases (retrieval/clustering, extraction agent, contradiction, synthesis, LLMOps, backend, frontend) each get their own spec.

## 1. Goals & Non-Goals

### Goals
- A `biolit.ner` module exposing `extract_entities(text) -> list[Entity]` (CHEMICAL + DISEASE), CPU-first with GPU auto-detect — the exact function the Phase 4 Extractor agent will call to anchor clustering/contradiction.
- A standalone eval harness reporting **strict entity-level precision/recall/F1** (exact span + type) on (a) the BC5CDR test split and (b) a small hand-annotated domain sample.
- An append-only eval-run log so metric trends are demoable over time.
- A first F1 number on the board, comparable to published BC5CDR results.

### Non-Goals (deferred)
- No genes/proteins (JNLPBA) — a second model in a later increment.
- No Extractor *agent* wiring (Phase 4 consumes `extract_entities`).
- No NER HTTP service (backend/LLMOps phase), no clustering/contradiction, no table/figure extraction.

## 2. Approved Decisions
1. **Entity scope:** single BC5CDR model, chemicals + diseases only.
2. **Stack:** Hugging Face `transformers` token-classification.
3. **Checkpoint (primary):** `Francesco-A/BiomedNLP-PubMedBERT-base-uncased-abstract-bc5cdr-ner-v1` — PubMedBERT fine-tuned on full BC5CDR, emitting both entity types (BIO scheme `B/I-Chemical`, `B/I-Disease`). Community upload → implementation must confirm license and smoke-test; keep a fallback ready (e.g. `raynardj/ner-disease-ncbi-bionlp-bc5cdr-pubmed` for disease + a chemical model, or another combined checkpoint) and record whichever is used.
4. **Eval data:** BC5CDR test split (`tner/bc5cdr`) as the comparable headline **plus** a blind hand-annotated domain sample.
5. **Inference:** in-process library, CPU-first, GPU auto-detected; no service, no cloud GPU this phase.
6. **Annotation methodology:** blind, from-scratch (ADR-0006 — see §6).

## 3. Module Structure

```
backend/src/biolit/ner/
├── __init__.py
├── model.py       # NerModel: loads HF pipeline, device auto-detect, cache, batching
├── labels.py      # BC5CDR B/I-Chemical|Disease → canonical Entity.label (CHEMICAL|DISEASE)
└── extract.py     # extract_entities(text) -> list[Entity] with char spans

backend/src/biolit_evals/            # eval package (separate from library)
├── __init__.py
├── datasets.py    # BC5CDR test-split loader (tner/bc5cdr) + domain-sample loader (with provenance)
├── ner_eval.py    # run model over a dataset, compute strict entity-level P/R/F1, append to log
└── scoring.py     # entity-level P/R/F1 (seqeval) — pure, unit-tested

backend/evals/
├── gold/domain_sample.jsonl        # blind-annotated gold with provenance (see §5.2)
└── runs.jsonl                      # append-only eval-run log
```

## 4. NER Model + Inference

### 4.1 `model.py`
- `NerModel` wraps a HF `pipeline("token-classification", model=..., aggregation_strategy="simple")` so subwords merge into entity spans with char offsets.
- Device: `ner_device` config — `auto` (CUDA if available else CPU), overridable. CPU is the tested default.
- Lazy singleton load; model files cached under `ner_cache_dir`. Batched inference via `ner_batch_size`.

### 4.2 `labels.py`
- Canonical `Entity.label` values for this phase: `CHEMICAL`, `DISEASE`.
- Map every model output group label (`Chemical`/`Disease`, or `B-/I-` forms if aggregation is not applied) → canonical. A table with total coverage (unit-tested: no model tag is unmapped).

### 4.3 `extract.py`
- `extract_entities(text: str) -> list[Entity]`: runs the pipeline, applies the label map, filters by `ner_score_threshold`, returns `Entity{text,label,start,end}` with correct char offsets. Empty/whitespace text → `[]`.
- Deterministic ordering (by start offset). This is the Phase-4 anchor point; its signature is stable.

### 4.4 Config additions (`config.py`)
`ner_model_id: str`, `ner_device: str = "auto"`, `ner_cache_dir: str | None`, `ner_batch_size: int = 16`, `ner_score_threshold: float = 0.5`.

## 5. Eval Harness

### 5.1 Scoring (`scoring.py`)
- **Strict entity-level P/R/F1**: an entity counts as correct only on exact span boundaries AND correct type. Implemented via `seqeval` on aligned BIO sequences (or an equivalent span-set precision/recall). Pure function over `(gold_entities, pred_entities)` → metrics; fully unit-tested with hand-built cases (exact match, boundary-off, wrong-type, missing, spurious).

### 5.2 Datasets (`datasets.py`)
- **BC5CDR test split:** load `tner/bc5cdr` test; expose tokens + gold spans in the scorer's format.
- **Domain sample:** load `evals/gold/domain_sample.jsonl`. **Each record carries provenance**, not just text:
  ```json
  {"paper_id": "10.1000/...", "pmid": "12345678", "source": "pubmed",
   "sentence_index": 3, "text": "…", 
   "entities": [{"start": 4, "end": 13, "label": "DISEASE", "text": "…"}]}
  ```
  So every annotated sentence is traceable to the exact paper it came from if those papers are re-pulled later.

### 5.3 Eval runner (`ner_eval.py`)
- `uv run python -m biolit_evals.ner_eval --dataset {bc5cdr|domain}` — runs the model, scores, prints P/R/F1, and **appends one JSON line** to `evals/runs.jsonl`:
  ```json
  {"timestamp": "...", "model_id": "...", "dataset": "bc5cdr", "split": "test",
   "precision": 0.0, "recall": 0.0, "f1": 0.0, "n_examples": 0, "git_sha": "..."}
  ```

## 6. Domain-Sample Annotation Methodology (ADR-0006)

- **Blind, from-scratch, primary.** The gold `domain_sample.jsonl` is labeled by reading the raw abstract sentences with **no model predictions visible**. Rationale: the domain sample's entire purpose is to test the model independently of the benchmark; model-assisted pre-labeling introduces anchoring bias — annotators are measurably worse at catching false negatives (entities the model missed) when reviewing a pre-populated list than when reading cold. Pre-labeling would blind the sample to the exact failure mode it exists to surface.
- **Order of operations:** finalize the gold file first; only then run the model and score against it.
- **Optional secondary (clearly separate):** a comparison of where model-proposed spans would have influenced labeling MAY be logged as a distinct, labeled artifact — never as a substitute for the blind pass.
- **Provenance:** source PMIDs/paper ids recorded per §5.2.
- **Caveats documented in the eval report:** both the single-annotator limitation and this blind-annotation methodology are stated explicitly.
- Sentences drawn from ~15–20 real abstracts pulled via the Phase 1 PubMed client on the project's driving topics; target ~30–50 sentences.

## 7. Testing Strategy

- **Unit (fast, offline, always in CI):**
  - `labels.py`: every model tag maps to a canonical label; no unmapped tag.
  - `extract.py`: char-offset correctness on a fixed sentence, threshold filtering, empty/no-entity input, ordering. Model call mocked (no download).
  - `scoring.py`: strict entity-level P/R/F1 on hand-built cases (exact, boundary-off, wrong-type, missing, spurious) — the load-bearing correctness of the harness.
  - `datasets.py`: domain-sample loader parses provenance fields; a tiny synthetic BC5CDR-shaped fixture exercises the runner's scoring + log-append without downloads.
- **Opt-in (heavy, not in fast CI):** the real-model run over the real BC5CDR test split (downloads model + dataset), behind a pytest marker / skip so CI stays fast. This produces the reported F1.

## 8. CI
- Fast unit tests (label map, extract, scoring, dataset parsing, log append) run on every push via the existing gate.
- The heavyweight real-model eval is opt-in (marker), runnable locally / in a manual CI job.

## 9. Definition of Done
- `biolit.ner.extract_entities` returns correct CHEMICAL/DISEASE `Entity` spans; unit tests green offline.
- `biolit_evals` scores strict entity-level P/R/F1 and appends to `runs.jsonl`; scoring logic unit-tested.
- BC5CDR test-split F1 produced and recorded in `runs.jsonl` (headline number); the chosen checkpoint's license confirmed and recorded.
- `domain_sample.jsonl` created via the blind methodology with provenance; domain F1 recorded; methodology + single-annotator caveats written into the eval report.
- `DECISIONS.md` updated with ADR-0006; `ARCHITECTURE.md` notes the NER layer.
- Fast unit tests wired into CI; heavy eval opt-in.

## 10. Open implementation risks
- Checkpoint license/quality unverified until implementation — fallback required (§2.3).
- `tner/bc5cdr` schema/label alignment to the scorer must be validated on real data.
- Model download size/latency — hence the opt-in split.
