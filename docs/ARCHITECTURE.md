# BioLit Copilot — Architecture

## Phase 1 (Foundations)
Backend-forward monorepo. `Paper` is the single normalization boundary for PubMed and
bioRxiv/medRxiv. Full-text existence (`text_type`) and extraction rights (`license_tier`,
`extraction_allowed`) are independent axes. Hybrid LangGraph state: one `PipelineState`
plus pure per-node `Input`/`Output` contracts joined by `project_*`/`merge_*` adapters.
Storage schema (papers + pgvector embeddings) is defined but dormant until Phase 3.

See `docs/DECISIONS.md` for ADRs and `docs/superpowers/specs/` for the Phase 1 spec.

## NER layer (Phase 2)
`biolit.ner.extract_entities` wraps a local, fine-tuned PubMedBERT/BC5CDR token-classification
checkpoint (chemicals + diseases) behind a lazily-loaded `NerModel`. It runs CPU-first with
GPU auto-detect (`Settings.ner_device = "auto"`) and is the entity-anchoring layer the
Extractor agent will call in Phase 4 to ground extracted claims in named chemical/disease
mentions. The eval harness lives in `biolit_evals` (`biolit_evals.ner_eval`): a pure,
multiset-based strict entity-level P/R/F1 scorer (`biolit_evals/scoring.py`, exact
`(start, end, label)` match — not `seqeval`) run against both the BC5CDR test split and a
blind-annotated in-domain sample (`evals/gold/domain_sample.jsonl`, ADR-0006), with every run
appended as one JSON line to `evals/runs.jsonl`. See `docs/EVAL_REPORT.md` for the first
real numbers, methodology caveats, and limitations.
