# BioLit Copilot — Phase 1 (Foundations) Design Spec

- **Date:** 2026-07-20
- **Status:** Approved (design), pending spec review
- **Scope:** Phase 1 only — repo scaffold, PubMed/bioRxiv data clients, dormant pgvector storage schema, and the Pydantic inter-agent state layer. Later phases (NER, retrieval, extraction, contradiction, synthesis, LLMOps, backend, frontend) each get their own spec.

## 1. Goals & Non-Goals

### Goals
- A backend-forward monorepo scaffold that later phases extend without restructuring.
- Working async PubMed (E-utilities) and bioRxiv/medRxiv clients that normalize into one shared `Paper` domain model.
- Honest, tested classification of each paper's full-text situation and its licensing/extraction rights.
- The complete typed inter-agent **state skeleton**: a shared `PipelineState` plus per-node `Input`/`Output` contracts and adapters, so every later agent node is a pure, independently testable function.
- Storage schema (papers + pgvector embeddings) defined as code and Alembic migration, but **not** running in Phase 1.
- Network-free CI: `ruff` + `pyright` + `pytest`, HTTP mocked via recorded `respx` cassettes.

### Non-Goals (deferred to later phases)
- No LangGraph node *implementations* (only their contracts exist).
- No NER model, no clustering, no contradiction detection, no synthesis.
- No running Postgres/pgvector, no repository wiring, no embeddings populated.
- No FastAPI server, no frontend (placeholder dir only).

## 2. Approved Decisions (see DECISIONS.md for ADRs)
1. **Repo structure:** single polyglot, backend-forward monorepo (ADR-0001).
2. **State layer:** hybrid — LangGraph-native shared `PipelineState` + pure per-node `Input`/`Output` contracts and adapters (ADR-0002).
3. **Infra scope:** storage schema-and-contracts now, container deferred to Phase 3 (ADR-0003).
4. **Full-text signalling:** three-state `text_type` enum, licensing modeled as a separate axis (ADR-0004).

## 3. Repository Layout

```
biolit-copilot/
├── backend/
│   ├── src/biolit/
│   │   ├── domain/          # Pydantic domain models (Paper, Entity, ...) — no I/O
│   │   ├── state/           # PipelineState + per-node Input/Output contracts + adapters
│   │   ├── clients/         # PubMedClient, BiorxivClient (async httpx) + retry/backoff
│   │   ├── storage/         # SQLAlchemy 2.0 tables + Alembic migration (defined, not run)
│   │   └── config.py        # pydantic-settings
│   ├── tests/
│   │   ├── cassettes/       # recorded respx fixtures (incl. 429-then-200, restrictive-license)
│   │   ├── clients/
│   │   ├── state/
│   │   └── storage/
│   ├── alembic/
│   ├── pyproject.toml       # uv-managed
│   └── Dockerfile           # written, unused until later
├── frontend/                # empty placeholder (Phase 8)
├── docs/
│   ├── ARCHITECTURE.md
│   ├── DECISIONS.md         # ADR log — seeded with ADR-0001..0004
│   └── superpowers/specs/
├── docker-compose.yml       # postgres+pgvector defined, not required for tests
└── README.md
```

### Tooling defaults
- **uv** for packaging/venv; **httpx** async clients; **pydantic v2** + **pydantic-settings**.
- **pytest** + **pytest-asyncio** + **respx** for record/replay HTTP (CI needs no network).
- **ruff** (lint+format) and **pyright** (type check).
- **SQLAlchemy 2.0** typed models + **Alembic** for the dormant storage layer.

## 4. Domain Models (`domain/`)

Core nouns, no I/O. The full-text and licensing story is modeled explicitly because it must propagate to Synthesis unchanged.

### 4.1 `TextType` (three-state enum)
```
TextType = full_text_available | full_text_unverified | abstract_only
```
- `full_text_available` — full text exists AND has been verified reachable (a later verification phase sets this).
- `full_text_unverified` — a full-text pointer exists (e.g. bioRxiv `jatsxml` URL, or PMC OA membership) but has not been fetched/confirmed; may 404 for freshly-posted content.
- `abstract_only` — no full-text pointer; work from abstract + metadata.

Ingest-time classification never produces `full_text_available`; it produces `full_text_unverified` or `abstract_only`. Promotion to `full_text_available` happens only after an actual fetch (Phase 4 extraction / a verification step).

### 4.2 License / extraction rights (separate axis)
Full-text **existence** and **permission to extract/display** are independent. Modeled separately:
- `Paper.license: str | None` — raw license token as reported by the source (e.g. `cc0`, `cc_by`, `cc_by_nc_nd`, `cc_no`, or a PMC license string).
- `Paper.license_tier: LicenseTier` — normalized enum: `open` (reuse OK, e.g. cc0/cc_by), `non_commercial` (NC/ND — display/extract with care), `restricted` (no-reuse, e.g. `cc_no` or PMC non-OA), `unknown`.
- `Paper.extraction_allowed: bool` — derived convenience flag (`True` only for `open`/`non_commercial` per a documented policy table). Downstream extractors and Synthesis must respect it; a paper can be `full_text_unverified` yet `extraction_allowed = False`.

### 4.3 `Paper`
Normalized record both clients emit:
`id` (canonical, DOI-or-PMID), `source` (`pubmed|biorxiv|medrxiv`), `pmid`, `doi`, `title`, `abstract`, `authors: list[Author]`, `journal`, `year`, `mesh_terms`, `categories`, `published_doi` (if a preprint was later published), `text_type: TextType`, `full_text_pointer: str | None` (jatsxml URL or PMC id), `license`, `license_tier`, `extraction_allowed`, `raw: dict` (source payload for debugging/trace).

Also stubbed for later phases (defined, not populated in Phase 1): `Entity`, `ExtractedRecord`, `Cluster`, `ContradictionFinding`, `Citation`.

## 5. State Layer (`state/`) — Hybrid

- `state/pipeline.py` — `PipelineState`, the single accumulating LangGraph model holding: `question`, `sub_queries`, `candidate_papers`, `extracted_records`, `clusters`, `contradictions`, `answer`. Fields default to empty and are filled as the graph progresses.
- `state/contracts.py` — every node's `XxxInput`/`XxxOutput` pair for the whole pipeline (`PlannerInput/Output`, `RetrieverInput/Output`, `ExtractorInput/Output`, `ClusteringInput/Output`, `CriticInput/Output`, `SynthesisInput/Output`). Phase 1 defines all of them as the typed skeleton even though only schemas exist.
- `state/adapters.py` — for each node, `project_<node>(state) -> XxxInput` and `merge_<node>(state, XxxOutput) -> PipelineState`. Node signature is `node(input: XxxInput) -> XxxOutput`; adapters keep nodes pure.

### 5.1 Merge semantics (load-bearing — see test §8.2)
`merge_*` performs **targeted, non-destructive updates**. When a node returns output touching a subset of papers/records, the merge updates exactly those entries by id and leaves all others byte-for-byte unchanged. Merges never wholesale-replace a collection with a partial one. This is the specific failure mode multi-agent state layers hit, and it is a tested contract, not an assumption.

## 6. Data Clients (`clients/`)

### 6.1 `PubMedClient` (async, E-utilities)
- `esearch(query) -> list[PMID]`, `efetch(pmids) -> list[Paper]` parsing title/abstract/authors/journal/year/MeSH/DOI.
- **`text_type` + license via PMC OA cross-reference (not a boolean from one call):**
  - Presence of a PMID in PMC is **not** sufficient. The client queries the **PMC OA Web Service** and inspects the **license** explicitly. A record can be in PMC yet under a non-commercial or otherwise restrictive license that limits extraction/display.
  - Mapping: PMC OA member with a resolvable full-text pointer → `full_text_unverified` (never `full_text_available` at ingest); not in PMC OA / no pointer → `abstract_only`. License token drives `license_tier`/`extraction_allowed` independently of `text_type`.
- **Retry with backoff (tested contract):** NCBI returns HTTP 429 under load. The client implements bounded exponential backoff with jitter and a max-attempts cap on 429/5xx. This is required because Phase 3+ issues multiple sub-queries per question. Backoff is configurable via `config.py`.
- Honors NCBI rate limits; optional `NCBI_API_KEY` from config raises the ceiling.

### 6.2 `BiorxivClient` (async, bioRxiv/medRxiv details API)
- Queries `https://api.biorxiv.org/details/{server}/{doi}` (and date-range/cursor forms). Normalizes into `Paper`.
- **`text_type` derived from the actual `jatsxml` field (verified against live API 2026-07-20), never hardcoded:**
  - Live records expose a `jatsxml` URL (JATS full-text XML path) and a `license` field (observed values include `cc0`, `cc_by_nc_nd`, `cc_no`).
  - `jatsxml` present & non-empty → `text_type = full_text_unverified` (the URL is a path that may 404 for freshly-posted preprints; promotion to `full_text_available` is a later verification step). Missing/empty `jatsxml` → `abstract_only`.
  - `license` maps to `license_tier`/`extraction_allowed` independently — a record can be `full_text_unverified` with `license: cc_no` → `restricted`, `extraction_allowed = False`.
- `dedupe(papers)` keys on DOI then PMID; preprint↔published links use `published_doi`.

## 7. Storage Layer (`storage/`) — defined, dormant

- SQLAlchemy 2.0 typed models: `papers` and `paper_embeddings` (a `vector` column via pgvector; abstract-chunk and full-text-chunk embeddings kept distinguishable).
- One Alembic migration creating both tables and the `vector` extension.
- `docker-compose.yml` defines a `postgres`+`pgvector` service.
- **No live DB in Phase 1 tests** — storage is import-and-introspect tested only (models importable, migration parses, columns/types as expected). The container is stood up and the repository wired in Phase 3 when the Retriever needs it.

## 8. Testing Strategy

All HTTP mocked via recorded `respx` cassettes → deterministic, network-free CI.

### 8.1 Client tests
- Parsing: PubMed `efetch` XML → `Paper`; bioRxiv JSON → `Paper`.
- **PMC license edge case (required):** a cassette for a PMID that IS in PMC but under a **restrictive/non-commercial license**, asserting we classify `license_tier = restricted`/`non_commercial` and set `extraction_allowed` correctly — i.e. we do NOT infer green-light from PMC presence alone.
- **PubMed 429 retry (required):** a cassette simulating **429 → then 200**, asserting the client backs off and ultimately succeeds, and that backoff (attempts/delay) is a tested contract.
- **bioRxiv `text_type` from field (required):** cassettes for (a) record with `jatsxml` present → `full_text_unverified`, (b) record with `jatsxml` missing/empty → `abstract_only`, (c) record with `jatsxml` present but `license: cc_no` → `full_text_unverified` + `restricted` + `extraction_allowed = False`.
- `dedupe` across duplicate DOIs/PMIDs.

### 8.2 State tests
- Baseline: `project_*` then `merge_*` round-trips are identity-preserving.
- **Partial-update survival (required):** seed `PipelineState` with 8 papers; run an Extractor-style output that touches only 3; assert the other **5 survive the round-trip byte-for-byte unchanged** and the 3 are updated. This is the real multi-agent failure mode and full round-trip alone won't catch it.
- Every contract model validates its example fixtures.

### 8.3 Storage tests
- Models import; Alembic migration parses/generates expected DDL; `vector` column present. No DB connection.

### 8.4 CI
- GitHub Actions on push/PR: `ruff check`, `ruff format --check`, `pyright`, `pytest`. Eval-gating CI arrives in later phases.

## 9. Definition of Done (Phase 1)
- Monorepo scaffold + tooling config present; `uv sync` + `pytest` green with no network.
- `PubMedClient` and `BiorxivClient` normalize to `Paper`, with the three required client tests passing (PMC restrictive license, 429 retry, bioRxiv jatsxml three-state).
- Full state skeleton (`PipelineState`, all contracts, all adapters) present; partial-update survival test passing.
- Storage models + migration defined and introspection-tested; container not required.
- `ARCHITECTURE.md` written; `DECISIONS.md` seeded with ADR-0001..0004.
- CI workflow runs ruff/pyright/pytest.
