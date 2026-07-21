# BioLit Copilot — Architecture

## Phase 1 (Foundations)
Backend-forward monorepo. `Paper` is the single normalization boundary for PubMed and
bioRxiv/medRxiv. Full-text existence (`text_type`) and extraction rights (`license_tier`,
`extraction_allowed`) are independent axes. Hybrid LangGraph state: one `PipelineState`
plus pure per-node `Input`/`Output` contracts joined by `project_*`/`merge_*` adapters.
Storage schema (papers + pgvector embeddings) is defined but dormant until Phase 3.

See `docs/DECISIONS.md` for ADRs and `docs/superpowers/specs/` for the Phase 1 spec.
