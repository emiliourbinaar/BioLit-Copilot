# Architecture Decision Log

Lightweight ADRs. Newest at top. Each entry: context, decision, alternatives, consequences.

---

## ADR-0007 — Structured abstracts are concatenated with section labels
- **Date:** 2026-07-22
- **Status:** Accepted
- **Context:** `PubMedClient` parsed abstracts with `article.findtext(".//Abstract/AbstractText")`, which returns only the *first* matching element. PubMed structured abstracts carry one `<AbstractText>` per section, so we silently kept only the opening section. Verified live on PMID 27295427 (LEADER trial): 156 of 2259 characters — 7% — with METHODS, RESULTS, and CONCLUSIONS discarded. Phase 1's tests all passed because every committed cassette happened to contain an unstructured abstract; the bug surfaced only when a human-facing task read real pulled text. This is severe for this project specifically: the system exists to extract findings and detect contradictions, and those claims live in RESULTS/CONCLUSIONS.
- **Decision:** Concatenate all `<AbstractText>` elements in document order. Use `"".join(el.itertext())` per element so nested inline markup (`<i>`, `<sup>`) is preserved rather than truncating the section at its first child tag. When an element carries a `Label`, prefix that section `"<LABEL>: "` — this is how PubMed itself renders structured abstracts, so it stays faithful to the source and preserves section structure for downstream extraction. Join sections with `"\n"`. Unstructured abstracts and absent abstracts keep their existing behavior exactly (plain text; `None`).
- **Alternatives:** join section texts without labels (rejected — discards structure that the Extractor and contradiction phases can use, and PubMed renders the labels anyway); keep a list of sections on `Paper` (rejected for now — changes the Phase 1 model contract and no consumer needs per-section access yet; revisit if one does).
- **Consequences:** Abstract text is complete. **Caveat:** because the `"LABEL: "` prefixes are synthesized, character offsets into `Paper.abstract` no longer index any contiguous string in the source document. For a citation-backed system that quotes excerpts, any future offset-based quoting into an abstract must account for this — prefer quoting the section text, not raw offsets. The Phase 2 domain gold sample was rebuilt from post-fix abstracts; the pre-fix sample was discarded rather than patched.

## ADR-0006 — Domain-sample NER annotation is blind and from-scratch
- **Date:** 2026-07-21
- **Status:** Accepted
- **Context:** Phase 2's hand-annotated domain sample exists to test the NER model *independently* of the BC5CDR benchmark — specifically to surface where the model fails on our own corpus. A model-assisted "pre-label then human-verify" workflow is faster but introduces anchoring bias: annotators are measurably worse at catching false negatives (entities the model missed) when reviewing a pre-populated candidate list than when reading cold. That false-negative blindness is exactly the failure mode the sample is meant to catch.
- **Decision:** Annotate `evals/gold/domain_sample.jsonl` blind and from scratch — reading raw abstract sentences with no model predictions visible. Finalize the gold file first, then run the model and score. Any comparison of model-influenced labeling is a separate, clearly-labeled secondary artifact, never a substitute for the blind pass. Record source PMIDs/paper ids as provenance on every annotated sentence. Document both the single-annotator limitation and this methodology in the eval report.
- **Alternatives:** model-assisted annotation then human review (rejected — undermines the sample's sole purpose by biasing it toward the model's own recall gaps); benchmark-only eval (rejected earlier — loses the "works on my actual corpus" signal).
- **Consequences:** More annotation effort, but the domain F1 is an honest independent check. Provenance keeps the sample traceable if papers are re-pulled. The unbiased headline number remains the BC5CDR test split; the domain sample is a supporting cross-check.

## ADR-0005 — String-valued enums use `enum.StrEnum`
- **Date:** 2026-07-20
- **Status:** Accepted
- **Context:** The Phase 1 plan wrote enums as `class X(str, Enum)`, but the project's ruff ruleset includes `UP`, whose `UP042` rule flags that pattern and fails the required lint gate. The plan mandated both the pattern and a green gate — a conflict surfaced during Task 2.
- **Decision:** All string-valued enums inherit from `enum.StrEnum` (Python 3.12). Behavior is identical for pydantic JSON serialization and `is`/`==` comparisons; the full `UP` ruleset stays active rather than suppressing a valid rule.
- **Alternatives:** keep `(str, Enum)` and add `UP042` to ruff `ignore` (rejected — silences a legitimate modernization rule to preserve example code).
- **Consequences:** `Source`, `TextType`, `LicenseTier` (Task 2) and `ContradictionLabel` (Task 3) use `StrEnum`. Applies to all future enums.

## ADR-0004 — Full-text signalling: three-state `text_type` + separate license axis
- **Date:** 2026-07-20
- **Status:** Accepted
- **Context:** A paper's full text may *exist* yet not be *reachable yet* (bioRxiv `jatsxml` URL can 404 for freshly-posted preprints; PMC membership is not a fetch), and *existence* is independent of *permission to extract/display* (verified live: a medRxiv record with a full `jatsxml` path but `license: cc_no`). A boolean `full_text: bool` conflates all three concerns and would silently over-promise to the Synthesis agent.
- **Decision:** Model two independent axes. (1) `text_type` enum with three states — `full_text_available` (exists AND verified reachable), `full_text_unverified` (pointer exists, unconfirmed), `abstract_only`. Ingest never yields `full_text_available`; a later verification/fetch step promotes it. (2) Licensing as `license` (raw token) + `license_tier` (`open|non_commercial|restricted|unknown`) + derived `extraction_allowed`. A paper can be `full_text_unverified` yet `extraction_allowed = False`.
- **Alternatives:** boolean `full_text` (rejected — conflates existence/reachability/rights); two-state `full_text|abstract_only` with a verified flag (rejected — the user chose the explicit three-state enum to propagate uncertainty downstream).
- **Consequences:** More enum handling everywhere, but faithfulness-preserving: downstream weighting can distinguish verified full text, unverified full text, and abstract-only, and respect extraction rights separately.

## ADR-0003 — Storage: schema-and-contracts now, container deferred
- **Date:** 2026-07-20
- **Status:** Accepted
- **Context:** Phase 1 lists "pgvector setup," but embeddings/semantic search aren't consumed until the Retriever agent in Phase 3. Standing up a live DB now would front-load infra and make every Phase 1 test run depend on Docker.
- **Decision:** Define SQLAlchemy models + Alembic migration for `papers`/`paper_embeddings` and the `docker-compose.yml` in Phase 1, but keep tests DB-free (import/introspection only). Stand up the container and wire the repository in Phase 3 when there's data to justify it.
- **Alternatives:** full stack now (rejected — front-loads infra, Docker dependency in CI); defer pgvector entirely (rejected — leaves the storage schema undesigned).
- **Consequences:** Phase 1 stays fast and network/Docker-free; the storage schema decision is still locked. Repository wiring is Phase 3 work.

## ADR-0002 — State layer: hybrid (shared PipelineState + per-node contracts)
- **Date:** 2026-07-20
- **Status:** Accepted
- **Context:** LangGraph is idiomatic with a single accumulating typed state, but the project requires each node's input/output schema to be "explicit and testable in isolation," and the eval harness leans on pure node functions.
- **Decision:** Define a shared `PipelineState` for LangGraph-native orchestration AND per-node `Input`/`Output` Pydantic pairs, with adapters (`project_*`, `merge_*`) projecting/merging. Nodes are pure `node(input) -> output`. Merges are targeted and non-destructive (partial updates preserve untouched entries — a tested contract).
- **Alternatives:** single accumulating state only (rejected — implicit field coupling, harder isolated testing); per-node contracts + thin orchestration state only (rejected — more boilerplate, less LangGraph-native).
- **Consequences:** Slightly more wiring; in exchange, every node is independently unit-testable and the graph state still serializes cleanly for jobs/tracing.

## ADR-0001 — Repo structure: backend-forward monorepo
- **Date:** 2026-07-20
- **Status:** Accepted
- **Context:** The system has a substantial Python backend (agents, NER, eval) and a Next.js frontend. Solo portfolio project targeting reviewability and demoability.
- **Decision:** One git repo, `backend/` and `frontend/` siblings, with top-level `docs/`, `evals/` (later), `docker-compose.yml`. Backend is the bulk of the work.
- **Alternatives:** two separate repos (rejected — coordination overhead, worse review/demo for a solo project); Python workspace monorepo with split installable packages (rejected — upfront ceremony not yet justified; can split later if a package boundary earns it).
- **Consequences:** Everything versioned together and visible in one place. If the backend later needs hard internal boundaries, revisit toward a workspace layout.
