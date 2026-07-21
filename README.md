# BioLit Copilot

Multi-agent biomedical literature research assistant. Phase 1 (Foundations) is in place:
PubMed/bioRxiv data clients, honest full-text/license classification, the typed
inter-agent state layer, and a dormant pgvector storage schema.

## Development
```bash
cd backend
uv sync
uv run pytest -q
uv run ruff check . && uv run pyright
```

See `docs/superpowers/specs/` for specs, `docs/superpowers/plans/` for plans, and
`docs/DECISIONS.md` for architecture decisions.
