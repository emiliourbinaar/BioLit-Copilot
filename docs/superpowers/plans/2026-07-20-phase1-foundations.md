# BioLit Copilot — Phase 1 (Foundations) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Stand up the BioLit Copilot backend foundation — repo scaffold, async PubMed/bioRxiv clients that normalize to one `Paper` model with honest full-text/license classification, the full typed inter-agent state skeleton, and a dormant pgvector storage schema — all with network-free tests and CI.

**Architecture:** Backend-forward monorepo. A shared `Paper` domain model is the single normalization boundary for both data sources. Full-text existence (`text_type`, three-state) and extraction rights (`license_tier`/`extraction_allowed`) are modeled as independent axes. A hybrid LangGraph state layer pairs one accumulating `PipelineState` with pure per-node `Input`/`Output` contracts joined by `project_*`/`merge_*` adapters (merges are targeted and non-destructive). Storage models + Alembic migration exist as code but no DB runs in Phase 1.

**Tech Stack:** Python 3.12, uv, pydantic v2 + pydantic-settings, httpx (async), respx (HTTP mocking), pytest + pytest-asyncio, ruff, pyright, SQLAlchemy 2.0 + Alembic, pgvector, Docker.

## Global Constraints

- Python **3.12+**; all packaging/venv via **uv**.
- All clients **async** (httpx `AsyncClient`); no synchronous network code.
- **No network in tests** — every HTTP call mocked with respx against fixtures under `backend/tests/cassettes/`.
- Data-source classification rules are **tested contracts**, never assumptions:
  - `text_type ∈ {full_text_available, full_text_unverified, abstract_only}`; ingest never yields `full_text_available`.
  - PMC full-text/license comes from the **PMC OA Web Service** with explicit license inspection — never inferred from PMC presence alone.
  - bioRxiv/medRxiv `text_type` derived from the **`jatsxml`** field; never hardcoded.
  - PubMed client retries **429/5xx** with bounded exponential backoff + jitter.
  - State merges are **targeted and non-destructive** — untouched entries survive unchanged.
- Lint/type/test gate: `ruff check`, `ruff format --check`, `pyright`, `pytest` all green.
- Package import root is `biolit` (`backend/src/biolit/`), installed editable via uv.

---

### Task 1: Repo scaffold + tooling

**Files:**
- Create: `backend/pyproject.toml`
- Create: `backend/src/biolit/__init__.py`
- Create: `backend/src/biolit/config.py`
- Create: `backend/tests/__init__.py`
- Create: `backend/tests/test_smoke.py`
- Create: `frontend/.gitkeep`
- Create: `docs/ARCHITECTURE.md`

**Interfaces:**
- Produces: `biolit.__version__: str`; `biolit.config.Settings` (pydantic-settings) with fields `ncbi_api_key: str | None`, `ncbi_tool: str`, `ncbi_email: str | None`, `http_max_retries: int = 4`, `http_backoff_base_seconds: float = 0.5`, `http_backoff_max_seconds: float = 8.0`; `biolit.config.get_settings() -> Settings` (cached).

- [ ] **Step 1: Create `backend/pyproject.toml`**

```toml
[project]
name = "biolit"
version = "0.1.0"
description = "BioLit Copilot backend"
requires-python = ">=3.12"
dependencies = [
    "pydantic>=2.7",
    "pydantic-settings>=2.3",
    "httpx>=0.27",
    "sqlalchemy>=2.0",
    "alembic>=1.13",
    "pgvector>=0.3",
]

[dependency-groups]
dev = [
    "pytest>=8.2",
    "pytest-asyncio>=0.23",
    "respx>=0.21",
    "ruff>=0.5",
    "pyright>=1.1",
]

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
packages = ["src/biolit"]

[tool.pytest.ini_options]
asyncio_mode = "auto"
testpaths = ["tests"]

[tool.ruff]
line-length = 100
src = ["src", "tests"]

[tool.ruff.lint]
select = ["E", "F", "I", "UP", "B"]

[tool.pyright]
include = ["src", "tests"]
pythonVersion = "3.12"
typeCheckingMode = "standard"
```

- [ ] **Step 2: Create package init and config**

`backend/src/biolit/__init__.py`:
```python
__version__ = "0.1.0"
```

`backend/src/biolit/config.py`:
```python
from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="BIOLIT_", env_file=".env", extra="ignore")

    ncbi_api_key: str | None = None
    ncbi_tool: str = "biolit-copilot"
    ncbi_email: str | None = None

    http_max_retries: int = 4
    http_backoff_base_seconds: float = 0.5
    http_backoff_max_seconds: float = 8.0


@lru_cache
def get_settings() -> Settings:
    return Settings()
```

- [ ] **Step 3: Write the smoke test**

`backend/tests/test_smoke.py`:
```python
from biolit import __version__
from biolit.config import get_settings


def test_version_present():
    assert __version__ == "0.1.0"


def test_settings_defaults():
    settings = get_settings()
    assert settings.http_max_retries == 4
    assert settings.ncbi_tool == "biolit-copilot"
```

- [ ] **Step 4: Sync and run**

Run: `cd backend && uv sync && uv run pytest -q`
Expected: 2 passed. If `uv` resolves and installs cleanly, both tests pass.

- [ ] **Step 5: Placeholder files + ARCHITECTURE stub**

Create empty `frontend/.gitkeep`. Create `docs/ARCHITECTURE.md`:
```markdown
# BioLit Copilot — Architecture

## Phase 1 (Foundations)
Backend-forward monorepo. `Paper` is the single normalization boundary for PubMed and
bioRxiv/medRxiv. Full-text existence (`text_type`) and extraction rights (`license_tier`,
`extraction_allowed`) are independent axes. Hybrid LangGraph state: one `PipelineState`
plus pure per-node `Input`/`Output` contracts joined by `project_*`/`merge_*` adapters.
Storage schema (papers + pgvector embeddings) is defined but dormant until Phase 3.

See `docs/DECISIONS.md` for ADRs and `docs/superpowers/specs/` for the Phase 1 spec.
```

- [ ] **Step 6: Commit**

```bash
git add backend docs/ARCHITECTURE.md frontend/.gitkeep
git commit -m "chore: scaffold backend package, tooling, and config"
```

---

### Task 2: Domain enums + licensing policy

**Files:**
- Create: `backend/src/biolit/domain/__init__.py`
- Create: `backend/src/biolit/domain/enums.py`
- Create: `backend/src/biolit/domain/licensing.py`
- Test: `backend/tests/domain/__init__.py`, `backend/tests/domain/test_licensing.py`

**Interfaces:**
- Produces:
  - `Source(str, Enum)` = `pubmed | biorxiv | medrxiv`
  - `TextType(str, Enum)` = `full_text_available | full_text_unverified | abstract_only`
  - `LicenseTier(str, Enum)` = `open | non_commercial | restricted | unknown`
  - `normalize_license(raw: str | None) -> tuple[str | None, LicenseTier]` — returns `(canonical_token, tier)`
  - `extraction_allowed_for(tier: LicenseTier) -> bool`

- [ ] **Step 1: Write the failing test**

`backend/tests/domain/__init__.py`: (empty)

`backend/tests/domain/test_licensing.py`:
```python
import pytest

from biolit.domain.enums import LicenseTier
from biolit.domain.licensing import extraction_allowed_for, normalize_license


@pytest.mark.parametrize(
    "raw, token, tier",
    [
        ("CC0", "cc0", LicenseTier.open),
        ("cc0", "cc0", LicenseTier.open),
        ("CC BY", "cc_by", LicenseTier.open),
        ("cc_by", "cc_by", LicenseTier.open),
        ("CC BY-NC", "cc_by_nc", LicenseTier.non_commercial),
        ("cc_by_nc_nd", "cc_by_nc_nd", LicenseTier.non_commercial),
        ("cc_no", "cc_no", LicenseTier.restricted),
        ("NO-CC CODE", "no_cc_code", LicenseTier.restricted),
        (None, None, LicenseTier.unknown),
        ("weird-unknown", "weird_unknown", LicenseTier.unknown),
    ],
)
def test_normalize_license(raw, token, tier):
    assert normalize_license(raw) == (token, tier)


@pytest.mark.parametrize(
    "tier, allowed",
    [
        (LicenseTier.open, True),
        (LicenseTier.non_commercial, True),
        (LicenseTier.restricted, False),
        (LicenseTier.unknown, False),
    ],
)
def test_extraction_allowed(tier, allowed):
    assert extraction_allowed_for(tier) is allowed
```

- [ ] **Step 2: Run to verify it fails**

Run: `cd backend && uv run pytest tests/domain/test_licensing.py -q`
Expected: FAIL — `ModuleNotFoundError: biolit.domain.enums`.

- [ ] **Step 3: Implement enums**

`backend/src/biolit/domain/__init__.py`: (empty)

`backend/src/biolit/domain/enums.py`:
```python
from enum import Enum


class Source(str, Enum):
    pubmed = "pubmed"
    biorxiv = "biorxiv"
    medrxiv = "medrxiv"


class TextType(str, Enum):
    full_text_available = "full_text_available"
    full_text_unverified = "full_text_unverified"
    abstract_only = "abstract_only"


class LicenseTier(str, Enum):
    open = "open"
    non_commercial = "non_commercial"
    restricted = "restricted"
    unknown = "unknown"
```

- [ ] **Step 4: Implement licensing policy**

`backend/src/biolit/domain/licensing.py`:
```python
import re

from biolit.domain.enums import LicenseTier

# Canonical token -> tier. Tokens are lowercase with separators collapsed to "_".
_TIER_BY_TOKEN: dict[str, LicenseTier] = {
    "cc0": LicenseTier.open,
    "cc_by": LicenseTier.open,
    "cc_by_sa": LicenseTier.open,
    "cc_by_nc": LicenseTier.non_commercial,
    "cc_by_nc_sa": LicenseTier.non_commercial,
    "cc_by_nd": LicenseTier.non_commercial,
    "cc_by_nc_nd": LicenseTier.non_commercial,
    "cc_no": LicenseTier.restricted,
    "no_cc_code": LicenseTier.restricted,
}

_EXTRACTION_ALLOWED: dict[LicenseTier, bool] = {
    LicenseTier.open: True,
    LicenseTier.non_commercial: True,
    LicenseTier.restricted: False,
    LicenseTier.unknown: False,
}


def _canonicalize(raw: str) -> str:
    token = raw.strip().lower()
    token = re.sub(r"[\s\-]+", "_", token)
    token = re.sub(r"_+", "_", token)
    return token.strip("_")


def normalize_license(raw: str | None) -> tuple[str | None, LicenseTier]:
    """Map a source license string to (canonical_token, tier).

    Presence of a token never implies extraction rights on its own; the tier does.
    """
    if raw is None or not raw.strip():
        return None, LicenseTier.unknown
    token = _canonicalize(raw)
    return token, _TIER_BY_TOKEN.get(token, LicenseTier.unknown)


def extraction_allowed_for(tier: LicenseTier) -> bool:
    return _EXTRACTION_ALLOWED[tier]
```

- [ ] **Step 5: Run to verify pass**

Run: `cd backend && uv run pytest tests/domain/test_licensing.py -q`
Expected: PASS (all parametrized cases).

- [ ] **Step 6: Commit**

```bash
git add backend/src/biolit/domain backend/tests/domain
git commit -m "feat(domain): text_type/license enums and license policy"
```

---

### Task 3: Domain models (`Paper` + downstream stubs)

**Files:**
- Create: `backend/src/biolit/domain/paper.py`
- Create: `backend/src/biolit/domain/records.py`
- Test: `backend/tests/domain/test_paper.py`

**Interfaces:**
- Consumes: `Source`, `TextType`, `LicenseTier` from `domain.enums`.
- Produces:
  - `Author(BaseModel)`: `name: str`, `affiliation: str | None = None`
  - `Paper(BaseModel)`: fields `id: str`, `source: Source`, `pmid: str | None`, `doi: str | None`, `title: str`, `abstract: str | None`, `authors: list[Author]`, `journal: str | None`, `year: int | None`, `mesh_terms: list[str]`, `categories: list[str]`, `published_doi: str | None`, `text_type: TextType`, `full_text_pointer: str | None`, `license: str | None`, `license_tier: LicenseTier`, `extraction_allowed: bool`, `raw: dict`
  - Stub models (defined, unused in Phase 1): `Entity`, `ExtractedRecord`, `Cluster`, `ContradictionFinding`, `Citation`

- [ ] **Step 1: Write the failing test**

`backend/tests/domain/test_paper.py`:
```python
from biolit.domain.enums import LicenseTier, Source, TextType
from biolit.domain.paper import Author, Paper
from biolit.domain.records import ExtractedRecord


def _paper(**overrides) -> Paper:
    base = dict(
        id="10.1101/2020.12.30.424878",
        source=Source.biorxiv,
        pmid=None,
        doi="10.1101/2020.12.30.424878",
        title="A preprint",
        abstract="Some abstract.",
        authors=[Author(name="Doe, J.")],
        journal=None,
        year=2020,
        mesh_terms=[],
        categories=["neuroscience"],
        published_doi=None,
        text_type=TextType.full_text_unverified,
        full_text_pointer="https://example/source.xml",
        license="cc0",
        license_tier=LicenseTier.open,
        extraction_allowed=True,
        raw={"doi": "10.1101/2020.12.30.424878"},
    )
    base.update(overrides)
    return Paper(**base)


def test_paper_roundtrips_and_defaults():
    paper = _paper()
    assert paper.text_type is TextType.full_text_unverified
    assert paper.extraction_allowed is True
    dumped = paper.model_dump()
    restored = Paper(**dumped)
    assert restored == paper


def test_abstract_only_paper_can_be_restricted():
    paper = _paper(
        text_type=TextType.abstract_only,
        full_text_pointer=None,
        license="cc_no",
        license_tier=LicenseTier.restricted,
        extraction_allowed=False,
    )
    assert paper.text_type is TextType.abstract_only
    assert paper.extraction_allowed is False


def test_extracted_record_stub_defined():
    record = ExtractedRecord(paper_id="p1", entities=[], key_findings=[])
    assert record.paper_id == "p1"
```

- [ ] **Step 2: Run to verify it fails**

Run: `cd backend && uv run pytest tests/domain/test_paper.py -q`
Expected: FAIL — `ModuleNotFoundError: biolit.domain.paper`.

- [ ] **Step 3: Implement `Paper`**

`backend/src/biolit/domain/paper.py`:
```python
from pydantic import BaseModel, Field

from biolit.domain.enums import LicenseTier, Source, TextType


class Author(BaseModel):
    name: str
    affiliation: str | None = None


class Paper(BaseModel):
    id: str
    source: Source
    pmid: str | None = None
    doi: str | None = None
    title: str
    abstract: str | None = None
    authors: list[Author] = Field(default_factory=list)
    journal: str | None = None
    year: int | None = None
    mesh_terms: list[str] = Field(default_factory=list)
    categories: list[str] = Field(default_factory=list)
    published_doi: str | None = None
    text_type: TextType
    full_text_pointer: str | None = None
    license: str | None = None
    license_tier: LicenseTier = LicenseTier.unknown
    extraction_allowed: bool = False
    raw: dict = Field(default_factory=dict)
```

- [ ] **Step 4: Implement downstream stubs**

`backend/src/biolit/domain/records.py`:
```python
from enum import Enum

from pydantic import BaseModel, Field


class Entity(BaseModel):
    text: str
    label: str  # e.g. CHEMICAL, DISEASE, GENE
    start: int | None = None
    end: int | None = None


class ExtractedRecord(BaseModel):
    paper_id: str
    entities: list[Entity] = Field(default_factory=list)
    study_type: str | None = None
    sample_size: int | None = None
    key_findings: list[str] = Field(default_factory=list)


class Cluster(BaseModel):
    key: str  # e.g. "metformin|PCOS"
    paper_ids: list[str] = Field(default_factory=list)


class ContradictionLabel(str, Enum):
    agreement = "agreement"
    contradiction = "contradiction"
    insufficient_overlap = "insufficient_overlap"


class ContradictionFinding(BaseModel):
    paper_id_a: str
    paper_id_b: str
    label: ContradictionLabel
    rationale: str


class Citation(BaseModel):
    paper_id: str
    claim: str
    excerpt: str | None = None
```

- [ ] **Step 5: Run to verify pass**

Run: `cd backend && uv run pytest tests/domain/test_paper.py -q`
Expected: PASS (3 tests).

- [ ] **Step 6: Commit**

```bash
git add backend/src/biolit/domain/paper.py backend/src/biolit/domain/records.py backend/tests/domain/test_paper.py
git commit -m "feat(domain): Paper model and downstream record stubs"
```

---

### Task 4: State layer (PipelineState + contracts + adapters)

**Files:**
- Create: `backend/src/biolit/state/__init__.py`
- Create: `backend/src/biolit/state/pipeline.py`
- Create: `backend/src/biolit/state/contracts.py`
- Create: `backend/src/biolit/state/adapters.py`
- Test: `backend/tests/state/__init__.py`, `backend/tests/state/test_adapters.py`

**Interfaces:**
- Consumes: `Paper` (domain.paper); `ExtractedRecord`, `Cluster`, `ContradictionFinding`, `Citation` (domain.records).
- Produces:
  - `PipelineState(BaseModel)`: `question: str`, `sub_queries: list[str]`, `candidate_papers: list[Paper]`, `extracted_records: dict[str, ExtractedRecord]`, `clusters: list[Cluster]`, `contradictions: list[ContradictionFinding]`, `answer: str | None`, `citations: list[Citation]`
  - Contracts (subset implemented now, rest as typed stubs): `ExtractorInput(papers: list[Paper])`, `ExtractorOutput(records: list[ExtractedRecord])`; plus `PlannerInput/Output`, `RetrieverInput/Output`, `ClusteringInput/Output`, `CriticInput/Output`, `SynthesisInput/Output`.
  - Adapters: `project_extractor(state) -> ExtractorInput`, `merge_extractor(state, out) -> PipelineState`. Merge upserts `out.records` into `state.extracted_records` by `paper_id`, leaving all other state fields and untouched records unchanged.

- [ ] **Step 1: Write the failing test (partial-update survival is the key case)**

`backend/tests/state/__init__.py`: (empty)

`backend/tests/state/test_adapters.py`:
```python
from biolit.domain.enums import LicenseTier, Source, TextType
from biolit.domain.paper import Paper
from biolit.domain.records import ExtractedRecord
from biolit.state.adapters import merge_extractor, project_extractor
from biolit.state.contracts import ExtractorOutput
from biolit.state.pipeline import PipelineState


def _paper(i: int) -> Paper:
    return Paper(
        id=f"p{i}",
        source=Source.pubmed,
        title=f"Paper {i}",
        text_type=TextType.abstract_only,
        license_tier=LicenseTier.unknown,
    )


def _state_with_papers(n: int) -> PipelineState:
    return PipelineState(question="q", candidate_papers=[_paper(i) for i in range(n)])


def test_project_extractor_passes_all_candidate_papers():
    state = _state_with_papers(8)
    proj = project_extractor(state)
    assert [p.id for p in proj.papers] == [f"p{i}" for i in range(8)]


def test_merge_is_targeted_and_nondestructive():
    state = _state_with_papers(8)

    # First extractor pass touches only p0, p1, p2.
    first = ExtractorOutput(
        records=[ExtractedRecord(paper_id=f"p{i}", key_findings=[f"finding {i}"]) for i in range(3)]
    )
    state = merge_extractor(state, first)
    assert set(state.extracted_records) == {"p0", "p1", "p2"}
    original_p0 = state.extracted_records["p0"]
    original_p1 = state.extracted_records["p1"]

    # Second pass touches p2 (overlap, must update) and p3 (new).
    second = ExtractorOutput(
        records=[
            ExtractedRecord(paper_id="p2", key_findings=["updated finding 2"]),
            ExtractedRecord(paper_id="p3", key_findings=["finding 3"]),
        ]
    )
    state = merge_extractor(state, second)

    # Untouched records survive byte-for-byte (same object, unchanged value).
    assert state.extracted_records["p0"] is original_p0
    assert state.extracted_records["p1"] is original_p1
    # Overlapping record is updated; new record is added.
    assert state.extracted_records["p2"].key_findings == ["updated finding 2"]
    assert state.extracted_records["p3"].key_findings == ["finding 3"]
    assert set(state.extracted_records) == {"p0", "p1", "p2", "p3"}
    # The 8 candidate papers are all still present and unchanged.
    assert [p.id for p in state.candidate_papers] == [f"p{i}" for i in range(8)]
```

- [ ] **Step 2: Run to verify it fails**

Run: `cd backend && uv run pytest tests/state/test_adapters.py -q`
Expected: FAIL — `ModuleNotFoundError: biolit.state.pipeline`.

- [ ] **Step 3: Implement PipelineState**

`backend/src/biolit/state/__init__.py`: (empty)

`backend/src/biolit/state/pipeline.py`:
```python
from pydantic import BaseModel, Field

from biolit.domain.paper import Paper
from biolit.domain.records import Citation, Cluster, ContradictionFinding, ExtractedRecord


class PipelineState(BaseModel):
    question: str
    sub_queries: list[str] = Field(default_factory=list)
    candidate_papers: list[Paper] = Field(default_factory=list)
    extracted_records: dict[str, ExtractedRecord] = Field(default_factory=dict)
    clusters: list[Cluster] = Field(default_factory=list)
    contradictions: list[ContradictionFinding] = Field(default_factory=list)
    citations: list[Citation] = Field(default_factory=list)
    answer: str | None = None
```

- [ ] **Step 4: Implement contracts**

`backend/src/biolit/state/contracts.py`:
```python
from pydantic import BaseModel, Field

from biolit.domain.paper import Paper
from biolit.domain.records import (
    Citation,
    Cluster,
    ContradictionFinding,
    ExtractedRecord,
)


class PlannerInput(BaseModel):
    question: str


class PlannerOutput(BaseModel):
    sub_queries: list[str] = Field(default_factory=list)


class RetrieverInput(BaseModel):
    sub_queries: list[str]


class RetrieverOutput(BaseModel):
    papers: list[Paper] = Field(default_factory=list)


class ExtractorInput(BaseModel):
    papers: list[Paper]


class ExtractorOutput(BaseModel):
    records: list[ExtractedRecord] = Field(default_factory=list)


class ClusteringInput(BaseModel):
    records: list[ExtractedRecord]


class ClusteringOutput(BaseModel):
    clusters: list[Cluster] = Field(default_factory=list)


class CriticInput(BaseModel):
    clusters: list[Cluster]
    records: dict[str, ExtractedRecord]


class CriticOutput(BaseModel):
    contradictions: list[ContradictionFinding] = Field(default_factory=list)


class SynthesisInput(BaseModel):
    question: str
    records: dict[str, ExtractedRecord]
    contradictions: list[ContradictionFinding]


class SynthesisOutput(BaseModel):
    answer: str
    citations: list[Citation] = Field(default_factory=list)
```

- [ ] **Step 5: Implement adapters**

`backend/src/biolit/state/adapters.py`:
```python
from biolit.state.contracts import ExtractorInput, ExtractorOutput
from biolit.state.pipeline import PipelineState


def project_extractor(state: PipelineState) -> ExtractorInput:
    return ExtractorInput(papers=list(state.candidate_papers))


def merge_extractor(state: PipelineState, out: ExtractorOutput) -> PipelineState:
    """Upsert extracted records by paper_id without disturbing untouched entries.

    Targeted, non-destructive merge: records not present in `out` are preserved as-is
    (same object identity), and no other state field is replaced.
    """
    merged = dict(state.extracted_records)
    for record in out.records:
        merged[record.paper_id] = record
    return state.model_copy(update={"extracted_records": merged})
```

Note: `model_copy(update=...)` performs a shallow copy, so untouched `ExtractedRecord`
objects keep their identity (the `is` assertions in the test rely on this).

- [ ] **Step 6: Run to verify pass**

Run: `cd backend && uv run pytest tests/state/test_adapters.py -q`
Expected: PASS (3 tests).

- [ ] **Step 7: Commit**

```bash
git add backend/src/biolit/state backend/tests/state
git commit -m "feat(state): PipelineState, node contracts, and non-destructive adapters"
```

---

### Task 5: Shared async HTTP client with retry/backoff

**Files:**
- Create: `backend/src/biolit/clients/__init__.py`
- Create: `backend/src/biolit/clients/http.py`
- Test: `backend/tests/clients/__init__.py`, `backend/tests/clients/test_http.py`

**Interfaces:**
- Consumes: `Settings` (config).
- Produces:
  - `class RetryConfig(max_retries: int, base_seconds: float, max_seconds: float)`
  - `async def request_with_retry(client: httpx.AsyncClient, method: str, url: str, *, retry: RetryConfig, sleep=asyncio.sleep, **kwargs) -> httpx.Response` — retries on 429 and 5xx with exponential backoff + full jitter; raises `httpx.HTTPStatusError` if still failing after `max_retries`; returns the response on first 2xx.

- [ ] **Step 1: Write the failing test (429 → 200)**

`backend/tests/clients/__init__.py`: (empty)

`backend/tests/clients/test_http.py`:
```python
import httpx
import pytest
import respx

from biolit.clients.http import RetryConfig, request_with_retry


@pytest.fixture
def no_sleep():
    async def _sleep(_seconds: float) -> None:
        return None

    return _sleep


@respx.mock
async def test_retries_429_then_succeeds(no_sleep):
    route = respx.get("https://api.example/thing").mock(
        side_effect=[
            httpx.Response(429, headers={"Retry-After": "0"}),
            httpx.Response(200, json={"ok": True}),
        ]
    )
    retry = RetryConfig(max_retries=4, base_seconds=0.01, max_seconds=0.02)
    async with httpx.AsyncClient() as client:
        resp = await request_with_retry(
            client, "GET", "https://api.example/thing", retry=retry, sleep=no_sleep
        )
    assert resp.status_code == 200
    assert resp.json() == {"ok": True}
    assert route.call_count == 2


@respx.mock
async def test_gives_up_after_max_retries(no_sleep):
    respx.get("https://api.example/thing").mock(return_value=httpx.Response(429))
    retry = RetryConfig(max_retries=2, base_seconds=0.01, max_seconds=0.02)
    async with httpx.AsyncClient() as client:
        with pytest.raises(httpx.HTTPStatusError):
            await request_with_retry(
                client, "GET", "https://api.example/thing", retry=retry, sleep=no_sleep
            )


@respx.mock
async def test_success_first_try_no_retry(no_sleep):
    route = respx.get("https://api.example/ok").mock(return_value=httpx.Response(200, json={}))
    retry = RetryConfig(max_retries=4, base_seconds=0.01, max_seconds=0.02)
    async with httpx.AsyncClient() as client:
        resp = await request_with_retry(
            client, "GET", "https://api.example/ok", retry=retry, sleep=no_sleep
        )
    assert resp.status_code == 200
    assert route.call_count == 1
```

- [ ] **Step 2: Run to verify it fails**

Run: `cd backend && uv run pytest tests/clients/test_http.py -q`
Expected: FAIL — `ModuleNotFoundError: biolit.clients.http`.

- [ ] **Step 3: Implement retry helper**

`backend/src/biolit/clients/__init__.py`: (empty)

`backend/src/biolit/clients/http.py`:
```python
import asyncio
import random
from collections.abc import Awaitable, Callable
from dataclasses import dataclass

import httpx

_RETRYABLE_STATUS = {429, 500, 502, 503, 504}


@dataclass(frozen=True)
class RetryConfig:
    max_retries: int
    base_seconds: float
    max_seconds: float


def _backoff_seconds(attempt: int, retry: RetryConfig) -> float:
    # Exponential backoff with full jitter.
    ceiling = min(retry.max_seconds, retry.base_seconds * (2**attempt))
    return random.uniform(0.0, ceiling)


async def request_with_retry(
    client: httpx.AsyncClient,
    method: str,
    url: str,
    *,
    retry: RetryConfig,
    sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
    **kwargs,
) -> httpx.Response:
    """Issue a request, retrying 429/5xx with exponential backoff + full jitter.

    Returns the first non-retryable response (raising for 4xx other than 429 via
    ``raise_for_status``). Raises ``httpx.HTTPStatusError`` if retries are exhausted.
    """
    attempt = 0
    while True:
        response = await client.request(method, url, **kwargs)
        if response.status_code not in _RETRYABLE_STATUS:
            response.raise_for_status()
            return response
        if attempt >= retry.max_retries:
            response.raise_for_status()
            return response  # pragma: no cover - raise_for_status always raises here
        await sleep(_backoff_seconds(attempt, retry))
        attempt += 1
```

- [ ] **Step 4: Run to verify pass**

Run: `cd backend && uv run pytest tests/clients/test_http.py -q`
Expected: PASS (3 tests).

- [ ] **Step 5: Commit**

```bash
git add backend/src/biolit/clients/__init__.py backend/src/biolit/clients/http.py backend/tests/clients
git commit -m "feat(clients): async request helper with 429/5xx backoff+jitter"
```

---

### Task 6: PubMedClient (E-utilities + PMC OA license cross-reference)

**Files:**
- Create: `backend/src/biolit/clients/pubmed.py`
- Create: `backend/tests/cassettes/pubmed_efetch_pmc_oa.xml`
- Create: `backend/tests/cassettes/pubmed_efetch_pmc_restricted.xml`
- Create: `backend/tests/cassettes/pubmed_efetch_no_pmc.xml`
- Create: `backend/tests/cassettes/pmc_oa_open.xml`
- Create: `backend/tests/cassettes/pmc_oa_restricted.xml`
- Create: `backend/tests/cassettes/pmc_oa_not_open.xml`
- Test: `backend/tests/clients/test_pubmed.py`

**Interfaces:**
- Consumes: `request_with_retry`, `RetryConfig` (clients.http); `Settings` (config); `Paper`, `Author`; `normalize_license`, `extraction_allowed_for`; enums.
- Produces:
  - `class PubMedClient(client: httpx.AsyncClient, settings: Settings)`
  - `async def esearch(self, query: str, retmax: int = 20) -> list[str]` — returns PMIDs
  - `async def efetch(self, pmids: list[str]) -> list[Paper]` — parses records and classifies text_type/license via `_classify_pmc`
  - `async def _classify_pmc(self, pmc_id: str | None) -> tuple[TextType, str | None, str | None]` — returns `(text_type, full_text_pointer, license_raw)`; queries the PMC OA Web Service

**Classification rules (tested contract):**
- No `<ArticleId IdType="pmc">` in efetch → `abstract_only`, pointer `None`, license `None`.
- PMC id present → query PMC OA service (`.../pmc/utils/oa/oa.fcgi?id=PMCID`):
  - OA record with `license=` and a full-text link → `full_text_unverified`, pointer = link href, license = attribute (may be restrictive, e.g. `NO-CC CODE`).
  - OA service returns `<error code="idIsNotOpenAccess"/>` → `abstract_only`, pointer `None`, license `None` (in PMC but not extractable — never assume green from presence).

- [ ] **Step 1: Create fixtures**

`backend/tests/cassettes/pubmed_efetch_pmc_oa.xml`:
```xml
<?xml version="1.0"?>
<PubmedArticleSet>
  <PubmedArticle>
    <MedlineCitation>
      <PMID>11111111</PMID>
      <Article>
        <Journal><Title>Journal of Tests</Title>
          <JournalIssue><PubDate><Year>2021</Year></PubDate></JournalIssue>
        </Journal>
        <ArticleTitle>Open access paper</ArticleTitle>
        <Abstract><AbstractText>Open abstract.</AbstractText></Abstract>
        <AuthorList>
          <Author><LastName>Doe</LastName><ForeName>Jane</ForeName></Author>
        </AuthorList>
      </Article>
      <MeshHeadingList>
        <MeshHeading><DescriptorName>Metformin</DescriptorName></MeshHeading>
      </MeshHeadingList>
    </MedlineCitation>
    <PubmedData>
      <ArticleIdList>
        <ArticleId IdType="doi">10.1000/open</ArticleId>
        <ArticleId IdType="pmc">PMC1111111</ArticleId>
      </ArticleIdList>
    </PubmedData>
  </PubmedArticle>
</PubmedArticleSet>
```

`backend/tests/cassettes/pubmed_efetch_pmc_restricted.xml`: same as above but `<PMID>22222222</PMID>`, `<ArticleTitle>Restricted PMC paper</ArticleTitle>`, doi `10.1000/restricted`, and `<ArticleId IdType="pmc">PMC2222222</ArticleId>`.

`backend/tests/cassettes/pubmed_efetch_no_pmc.xml`: same shape with `<PMID>33333333</PMID>`, `<ArticleTitle>Abstract only paper</ArticleTitle>`, doi `10.1000/noPMC`, and **no** `pmc` ArticleId.

`backend/tests/cassettes/pmc_oa_open.xml`:
```xml
<?xml version="1.0"?>
<OA>
  <records>
    <record id="PMC1111111" license="CC BY">
      <link format="tgz" href="ftp://ftp.ncbi.nlm.nih.gov/pub/pmc/oa/PMC1111111.tar.gz"/>
    </record>
  </records>
</OA>
```

`backend/tests/cassettes/pmc_oa_restricted.xml`:
```xml
<?xml version="1.0"?>
<OA>
  <records>
    <record id="PMC2222222" license="NO-CC CODE">
      <link format="tgz" href="ftp://ftp.ncbi.nlm.nih.gov/pub/pmc/oa/PMC2222222.tar.gz"/>
    </record>
  </records>
</OA>
```

`backend/tests/cassettes/pmc_oa_not_open.xml`:
```xml
<?xml version="1.0"?>
<OA>
  <error code="idIsNotOpenAccess">identifier is not Open Access</error>
</OA>
```

- [ ] **Step 2: Write the failing test**

`backend/tests/clients/test_pubmed.py`:
```python
from pathlib import Path

import httpx
import pytest
import respx

from biolit.clients.pubmed import PubMedClient
from biolit.config import Settings
from biolit.domain.enums import LicenseTier, Source, TextType

CASSETTES = Path(__file__).parent.parent / "cassettes"


def _cassette(name: str) -> str:
    return (CASSETTES / name).read_text(encoding="utf-8")


def _oa_route(pmc_id: str, body: str):
    return respx.get("https://www.ncbi.nlm.nih.gov/pmc/utils/oa/oa.fcgi").mock(
        return_value=httpx.Response(200, text=body)
    ) if pmc_id else None


@pytest.fixture
def settings() -> Settings:
    return Settings(http_backoff_base_seconds=0.001, http_backoff_max_seconds=0.002)


@respx.mock
async def test_efetch_open_access(settings):
    respx.get("https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi").mock(
        return_value=httpx.Response(200, text=_cassette("pubmed_efetch_pmc_oa.xml"))
    )
    respx.get("https://www.ncbi.nlm.nih.gov/pmc/utils/oa/oa.fcgi").mock(
        return_value=httpx.Response(200, text=_cassette("pmc_oa_open.xml"))
    )
    async with httpx.AsyncClient() as http:
        papers = await PubMedClient(http, settings).efetch(["11111111"])
    paper = papers[0]
    assert paper.source is Source.pubmed
    assert paper.pmid == "11111111"
    assert paper.title == "Open access paper"
    assert paper.year == 2021
    assert paper.mesh_terms == ["Metformin"]
    assert paper.text_type is TextType.full_text_unverified
    assert paper.license_tier is LicenseTier.open
    assert paper.extraction_allowed is True


@respx.mock
async def test_efetch_pmc_but_restricted_license(settings):
    # In PMC, but under a restrictive license: must NOT be treated as extractable.
    respx.get("https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi").mock(
        return_value=httpx.Response(200, text=_cassette("pubmed_efetch_pmc_restricted.xml"))
    )
    respx.get("https://www.ncbi.nlm.nih.gov/pmc/utils/oa/oa.fcgi").mock(
        return_value=httpx.Response(200, text=_cassette("pmc_oa_restricted.xml"))
    )
    async with httpx.AsyncClient() as http:
        papers = await PubMedClient(http, settings).efetch(["22222222"])
    paper = papers[0]
    assert paper.text_type is TextType.full_text_unverified
    assert paper.license_tier is LicenseTier.restricted
    assert paper.extraction_allowed is False


@respx.mock
async def test_efetch_pmc_present_but_not_oa(settings):
    respx.get("https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi").mock(
        return_value=httpx.Response(200, text=_cassette("pubmed_efetch_pmc_restricted.xml"))
    )
    respx.get("https://www.ncbi.nlm.nih.gov/pmc/utils/oa/oa.fcgi").mock(
        return_value=httpx.Response(200, text=_cassette("pmc_oa_not_open.xml"))
    )
    async with httpx.AsyncClient() as http:
        papers = await PubMedClient(http, settings).efetch(["22222222"])
    paper = papers[0]
    assert paper.text_type is TextType.abstract_only
    assert paper.extraction_allowed is False


@respx.mock
async def test_efetch_no_pmc_is_abstract_only(settings):
    respx.get("https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi").mock(
        return_value=httpx.Response(200, text=_cassette("pubmed_efetch_no_pmc.xml"))
    )
    async with httpx.AsyncClient() as http:
        papers = await PubMedClient(http, settings).efetch(["33333333"])
    paper = papers[0]
    assert paper.text_type is TextType.abstract_only
    assert paper.full_text_pointer is None
    assert paper.license_tier is LicenseTier.unknown


@respx.mock
async def test_esearch_returns_pmids(settings):
    respx.get("https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi").mock(
        return_value=httpx.Response(
            200,
            text='<?xml version="1.0"?><eSearchResult><IdList>'
            "<Id>11111111</Id><Id>22222222</Id></IdList></eSearchResult>",
        )
    )
    async with httpx.AsyncClient() as http:
        pmids = await PubMedClient(http, settings).esearch("metformin PCOS")
    assert pmids == ["11111111", "22222222"]
```

- [ ] **Step 3: Run to verify it fails**

Run: `cd backend && uv run pytest tests/clients/test_pubmed.py -q`
Expected: FAIL — `ModuleNotFoundError: biolit.clients.pubmed`.

- [ ] **Step 4: Implement PubMedClient**

`backend/src/biolit/clients/pubmed.py`:
```python
from xml.etree import ElementTree as ET

import httpx

from biolit.clients.http import RetryConfig, request_with_retry
from biolit.config import Settings
from biolit.domain.enums import LicenseTier, Source, TextType
from biolit.domain.licensing import extraction_allowed_for, normalize_license
from biolit.domain.paper import Author, Paper

_EUTILS = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
_PMC_OA = "https://www.ncbi.nlm.nih.gov/pmc/utils/oa/oa.fcgi"


class PubMedClient:
    def __init__(self, client: httpx.AsyncClient, settings: Settings) -> None:
        self._client = client
        self._settings = settings
        self._retry = RetryConfig(
            max_retries=settings.http_max_retries,
            base_seconds=settings.http_backoff_base_seconds,
            max_seconds=settings.http_backoff_max_seconds,
        )

    def _params(self, **extra: str) -> dict[str, str]:
        params = {"tool": self._settings.ncbi_tool}
        if self._settings.ncbi_api_key:
            params["api_key"] = self._settings.ncbi_api_key
        if self._settings.ncbi_email:
            params["email"] = self._settings.ncbi_email
        params.update(extra)
        return params

    async def esearch(self, query: str, retmax: int = 20) -> list[str]:
        resp = await request_with_retry(
            self._client,
            "GET",
            f"{_EUTILS}/esearch.fcgi",
            retry=self._retry,
            params=self._params(db="pubmed", term=query, retmax=str(retmax)),
        )
        root = ET.fromstring(resp.text)
        return [el.text or "" for el in root.findall(".//IdList/Id") if el.text]

    async def efetch(self, pmids: list[str]) -> list[Paper]:
        if not pmids:
            return []
        resp = await request_with_retry(
            self._client,
            "GET",
            f"{_EUTILS}/efetch.fcgi",
            retry=self._retry,
            params=self._params(db="pubmed", id=",".join(pmids), retmode="xml"),
        )
        root = ET.fromstring(resp.text)
        papers: list[Paper] = []
        for article in root.findall(".//PubmedArticle"):
            papers.append(await self._parse_article(article))
        return papers

    async def _parse_article(self, article: ET.Element) -> Paper:
        pmid = article.findtext(".//MedlineCitation/PMID") or ""
        title = article.findtext(".//Article/ArticleTitle") or ""
        abstract = article.findtext(".//Abstract/AbstractText")
        journal = article.findtext(".//Journal/Title")
        year_text = article.findtext(".//JournalIssue/PubDate/Year")
        year = int(year_text) if year_text and year_text.isdigit() else None

        authors: list[Author] = []
        for author in article.findall(".//AuthorList/Author"):
            last = author.findtext("LastName")
            fore = author.findtext("ForeName")
            if last or fore:
                authors.append(Author(name=" ".join(p for p in (fore, last) if p)))

        mesh = [
            el.text
            for el in article.findall(".//MeshHeadingList/MeshHeading/DescriptorName")
            if el.text
        ]

        doi = None
        pmc_id = None
        for aid in article.findall(".//ArticleIdList/ArticleId"):
            id_type = aid.get("IdType")
            if id_type == "doi":
                doi = aid.text
            elif id_type == "pmc":
                pmc_id = aid.text

        text_type, pointer, license_raw = await self._classify_pmc(pmc_id)
        token, tier = normalize_license(license_raw)

        return Paper(
            id=doi or pmid,
            source=Source.pubmed,
            pmid=pmid or None,
            doi=doi,
            title=title,
            abstract=abstract,
            authors=authors,
            journal=journal,
            year=year,
            mesh_terms=mesh,
            text_type=text_type,
            full_text_pointer=pointer,
            license=token,
            license_tier=tier,
            extraction_allowed=extraction_allowed_for(tier),
            raw={"pmid": pmid, "pmc_id": pmc_id},
        )

    async def _classify_pmc(
        self, pmc_id: str | None
    ) -> tuple[TextType, str | None, str | None]:
        """Cross-reference the PMC OA Web Service; never infer rights from PMC presence."""
        if not pmc_id:
            return TextType.abstract_only, None, None
        resp = await request_with_retry(
            self._client,
            "GET",
            _PMC_OA,
            retry=self._retry,
            params={"id": pmc_id},
        )
        root = ET.fromstring(resp.text)
        if root.find(".//error") is not None:
            return TextType.abstract_only, None, None
        record = root.find(".//records/record")
        if record is None:
            return TextType.abstract_only, None, None
        license_raw = record.get("license")
        link = record.find("link")
        pointer = link.get("href") if link is not None else None
        return TextType.full_text_unverified, pointer, license_raw
```

- [ ] **Step 5: Run to verify pass**

Run: `cd backend && uv run pytest tests/clients/test_pubmed.py -q`
Expected: PASS (5 tests).

- [ ] **Step 6: Commit**

```bash
git add backend/src/biolit/clients/pubmed.py backend/tests/clients/test_pubmed.py backend/tests/cassettes
git commit -m "feat(clients): PubMed client with PMC OA license cross-reference"
```

---

### Task 7: BiorxivClient (jatsxml-driven text_type + dedupe)

**Files:**
- Create: `backend/src/biolit/clients/biorxiv.py`
- Create: `backend/tests/cassettes/biorxiv_fulltext.json`
- Create: `backend/tests/cassettes/biorxiv_no_jatsxml.json`
- Create: `backend/tests/cassettes/biorxiv_restricted.json`
- Test: `backend/tests/clients/test_biorxiv.py`

**Interfaces:**
- Consumes: `request_with_retry`, `RetryConfig`; `Settings`; `Paper`, `Author`; licensing; enums.
- Produces:
  - `class BiorxivClient(client, settings)`
  - `async def details(self, server: str, doi: str) -> list[Paper]` — `server ∈ {"biorxiv","medrxiv"}`
  - `def dedupe(papers: list[Paper]) -> list[Paper]` — module-level; keys on DOI then PMID, first occurrence wins
- **text_type rule (tested contract):** `jatsxml` present & non-empty → `full_text_unverified` with `full_text_pointer = jatsxml`; missing/empty → `abstract_only`. `license` drives tier independently.

- [ ] **Step 1: Create fixtures**

`backend/tests/cassettes/biorxiv_fulltext.json`:
```json
{
  "messages": [{"status": "ok"}],
  "collection": [
    {
      "doi": "10.1101/2020.12.30.424878",
      "title": "Full text preprint",
      "authors": "Doe, J.; Roe, R.",
      "author_corresponding": "Jane Doe",
      "author_corresponding_institution": "Test University",
      "date": "2021-01-01",
      "version": "1",
      "type": "new results",
      "license": "cc0",
      "category": "neuroscience",
      "jatsxml": "https://www.biorxiv.org/content/early/2021/01/01/2020.12.30.424878.source.xml",
      "abstract": "A full text abstract.",
      "published": "10.1038/s41586-021-00000-0",
      "server": "biorxiv"
    }
  ]
}
```

`backend/tests/cassettes/biorxiv_no_jatsxml.json`: same single record with `"doi": "10.1101/2021.02.02.111111"`, `"title": "Not yet indexed"`, `"license": "cc_by"`, `"jatsxml": ""`, `"published": "NA"`.

`backend/tests/cassettes/biorxiv_restricted.json`: same shape, `"doi": "10.1101/2020.09.09.20191205"`, `"title": "Restricted license preprint"`, `"license": "cc_no"`, `"jatsxml": "https://www.medrxiv.org/content/early/2020/09/10/2020.09.09.20191205.source.xml"`, `"server": "medrxiv"`.

- [ ] **Step 2: Write the failing test**

`backend/tests/clients/test_biorxiv.py`:
```python
from pathlib import Path

import httpx
import pytest
import respx

from biolit.clients.biorxiv import BiorxivClient, dedupe
from biolit.config import Settings
from biolit.domain.enums import LicenseTier, Source, TextType
from biolit.domain.paper import Paper

CASSETTES = Path(__file__).parent.parent / "cassettes"


def _cassette(name: str) -> str:
    return (CASSETTES / name).read_text(encoding="utf-8")


@pytest.fixture
def settings() -> Settings:
    return Settings(http_backoff_base_seconds=0.001, http_backoff_max_seconds=0.002)


@respx.mock
async def test_details_fulltext(settings):
    respx.get(url__regex=r"https://api\.biorxiv\.org/details/biorxiv/.*").mock(
        return_value=httpx.Response(200, text=_cassette("biorxiv_fulltext.json"))
    )
    async with httpx.AsyncClient() as http:
        papers = await BiorxivClient(http, settings).details("biorxiv", "10.1101/2020.12.30.424878")
    paper = papers[0]
    assert paper.source is Source.biorxiv
    assert paper.text_type is TextType.full_text_unverified
    assert paper.full_text_pointer.endswith(".source.xml")
    assert paper.license_tier is LicenseTier.open
    assert paper.extraction_allowed is True
    assert paper.published_doi == "10.1038/s41586-021-00000-0"
    assert [a.name for a in paper.authors] == ["Doe, J.", "Roe, R."]


@respx.mock
async def test_details_missing_jatsxml_is_abstract_only(settings):
    respx.get(url__regex=r"https://api\.biorxiv\.org/details/biorxiv/.*").mock(
        return_value=httpx.Response(200, text=_cassette("biorxiv_no_jatsxml.json"))
    )
    async with httpx.AsyncClient() as http:
        papers = await BiorxivClient(http, settings).details("biorxiv", "10.1101/2021.02.02.111111")
    paper = papers[0]
    assert paper.text_type is TextType.abstract_only
    assert paper.full_text_pointer is None
    assert paper.published_doi is None  # "NA" normalized to None


@respx.mock
async def test_details_fulltext_but_restricted_license(settings):
    respx.get(url__regex=r"https://api\.biorxiv\.org/details/medrxiv/.*").mock(
        return_value=httpx.Response(200, text=_cassette("biorxiv_restricted.json"))
    )
    async with httpx.AsyncClient() as http:
        papers = await BiorxivClient(http, settings).details(
            "medrxiv", "10.1101/2020.09.09.20191205"
        )
    paper = papers[0]
    assert paper.source is Source.medrxiv
    assert paper.text_type is TextType.full_text_unverified
    assert paper.license_tier is LicenseTier.restricted
    assert paper.extraction_allowed is False


def test_dedupe_keeps_first_by_doi():
    a = Paper(id="10.1/x", source=Source.biorxiv, doi="10.1/x", title="A",
              text_type=TextType.abstract_only)
    b = Paper(id="10.1/x", source=Source.biorxiv, doi="10.1/x", title="A dup",
              text_type=TextType.abstract_only)
    c = Paper(id="p9", source=Source.pubmed, pmid="9", title="C",
              text_type=TextType.abstract_only)
    result = dedupe([a, b, c])
    assert [p.title for p in result] == ["A", "C"]
```

- [ ] **Step 3: Run to verify it fails**

Run: `cd backend && uv run pytest tests/clients/test_biorxiv.py -q`
Expected: FAIL — `ModuleNotFoundError: biolit.clients.biorxiv`.

- [ ] **Step 4: Implement BiorxivClient**

`backend/src/biolit/clients/biorxiv.py`:
```python
import httpx

from biolit.clients.http import RetryConfig, request_with_retry
from biolit.config import Settings
from biolit.domain.enums import LicenseTier, Source, TextType
from biolit.domain.licensing import extraction_allowed_for, normalize_license
from biolit.domain.paper import Author, Paper

_API = "https://api.biorxiv.org/details"
_SERVER_SOURCE = {"biorxiv": Source.biorxiv, "medrxiv": Source.medrxiv}


def _clean(value: str | None) -> str | None:
    if value is None:
        return None
    trimmed = value.strip()
    if not trimmed or trimmed.upper() == "NA":
        return None
    return trimmed


def _authors(raw: str | None) -> list[Author]:
    if not raw:
        return []
    return [Author(name=name.strip()) for name in raw.split(";") if name.strip()]


class BiorxivClient:
    def __init__(self, client: httpx.AsyncClient, settings: Settings) -> None:
        self._client = client
        self._retry = RetryConfig(
            max_retries=settings.http_max_retries,
            base_seconds=settings.http_backoff_base_seconds,
            max_seconds=settings.http_backoff_max_seconds,
        )

    async def details(self, server: str, doi: str) -> list[Paper]:
        resp = await request_with_retry(
            self._client,
            "GET",
            f"{_API}/{server}/{doi}",
            retry=self._retry,
        )
        payload = resp.json()
        return [self._parse(record) for record in payload.get("collection", [])]

    def _parse(self, record: dict) -> Paper:
        server = (record.get("server") or "").lower()
        source = _SERVER_SOURCE.get(server, Source.biorxiv)

        jatsxml = _clean(record.get("jatsxml"))
        if jatsxml:
            text_type = TextType.full_text_unverified
            pointer = jatsxml
        else:
            text_type = TextType.abstract_only
            pointer = None

        token, tier = normalize_license(record.get("license"))
        year_text = (record.get("date") or "")[:4]
        year = int(year_text) if year_text.isdigit() else None
        doi = _clean(record.get("doi"))

        return Paper(
            id=doi or record.get("title", ""),
            source=source,
            doi=doi,
            title=record.get("title", ""),
            abstract=_clean(record.get("abstract")),
            authors=_authors(record.get("authors")),
            year=year,
            categories=[c for c in [_clean(record.get("category"))] if c],
            published_doi=_clean(record.get("published")),
            text_type=text_type,
            full_text_pointer=pointer,
            license=token,
            license_tier=tier,
            extraction_allowed=extraction_allowed_for(tier),
            raw=record,
        )


def dedupe(papers: list[Paper]) -> list[Paper]:
    """Drop duplicates, keying on DOI then PMID; first occurrence wins."""
    seen: set[str] = set()
    result: list[Paper] = []
    for paper in papers:
        key = paper.doi or paper.pmid or paper.id
        if key in seen:
            continue
        seen.add(key)
        result.append(paper)
    return result
```

Note the unused `LicenseTier` import is removed by ruff if flagged — keep only imports the
file uses (`Source`, `TextType` are used; drop `LicenseTier` if ruff reports F401).

- [ ] **Step 5: Run to verify pass**

Run: `cd backend && uv run pytest tests/clients/test_biorxiv.py -q`
Expected: PASS (4 tests).

- [ ] **Step 6: Commit**

```bash
git add backend/src/biolit/clients/biorxiv.py backend/tests/clients/test_biorxiv.py backend/tests/cassettes
git commit -m "feat(clients): bioRxiv/medRxiv client with jatsxml-driven text_type and dedupe"
```

---

### Task 8: Storage schema (dormant) + Alembic migration + docker-compose

**Files:**
- Create: `backend/src/biolit/storage/__init__.py`
- Create: `backend/src/biolit/storage/models.py`
- Create: `backend/alembic.ini`
- Create: `backend/alembic/env.py`
- Create: `backend/alembic/script.py.mako`
- Create: `backend/alembic/versions/0001_initial.py`
- Create: `docker-compose.yml`
- Test: `backend/tests/storage/__init__.py`, `backend/tests/storage/test_models.py`

**Interfaces:**
- Produces: `Base` (DeclarativeBase); `PaperRow` (`papers`) and `PaperEmbeddingRow` (`paper_embeddings`) with a pgvector `Vector(768)` column; `EMBEDDING_DIM = 768`.
- No DB connection in tests — import + introspection only.

- [ ] **Step 1: Write the failing test**

`backend/tests/storage/__init__.py`: (empty)

`backend/tests/storage/test_models.py`:
```python
from biolit.storage.models import EMBEDDING_DIM, Base, PaperEmbeddingRow, PaperRow


def test_tables_registered():
    tables = set(Base.metadata.tables)
    assert {"papers", "paper_embeddings"} <= tables


def test_paper_columns():
    cols = {c.name for c in PaperRow.__table__.columns}
    assert {"id", "source", "doi", "pmid", "title", "text_type", "license_tier"} <= cols


def test_embedding_has_vector_column():
    col = PaperEmbeddingRow.__table__.columns["embedding"]
    # pgvector's Vector type exposes the configured dimension.
    assert getattr(col.type, "dim", EMBEDDING_DIM) == EMBEDDING_DIM
    assert "chunk_kind" in {c.name for c in PaperEmbeddingRow.__table__.columns}
```

- [ ] **Step 2: Run to verify it fails**

Run: `cd backend && uv run pytest tests/storage/test_models.py -q`
Expected: FAIL — `ModuleNotFoundError: biolit.storage.models`.

- [ ] **Step 3: Implement models**

`backend/src/biolit/storage/__init__.py`: (empty)

`backend/src/biolit/storage/models.py`:
```python
from datetime import datetime

from pgvector.sqlalchemy import Vector
from sqlalchemy import DateTime, Integer, String, Text, func
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

EMBEDDING_DIM = 768


class Base(DeclarativeBase):
    pass


class PaperRow(Base):
    __tablename__ = "papers"

    id: Mapped[str] = mapped_column(String, primary_key=True)
    source: Mapped[str] = mapped_column(String, nullable=False)
    doi: Mapped[str | None] = mapped_column(String, nullable=True)
    pmid: Mapped[str | None] = mapped_column(String, nullable=True)
    title: Mapped[str] = mapped_column(Text, nullable=False)
    abstract: Mapped[str | None] = mapped_column(Text, nullable=True)
    year: Mapped[int | None] = mapped_column(Integer, nullable=True)
    text_type: Mapped[str] = mapped_column(String, nullable=False)
    license: Mapped[str | None] = mapped_column(String, nullable=True)
    license_tier: Mapped[str] = mapped_column(String, nullable=False)
    full_text_pointer: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )


class PaperEmbeddingRow(Base):
    __tablename__ = "paper_embeddings"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    paper_id: Mapped[str] = mapped_column(String, nullable=False, index=True)
    chunk_kind: Mapped[str] = mapped_column(String, nullable=False)  # abstract | full_text
    chunk_index: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    embedding: Mapped[list[float]] = mapped_column(Vector(EMBEDDING_DIM))
```

- [ ] **Step 4: Add Alembic scaffolding + migration**

`backend/alembic.ini` (minimal):
```ini
[alembic]
script_location = alembic
sqlalchemy.url = postgresql+psycopg://biolit:biolit@localhost:5432/biolit

[loggers]
keys = root
[handlers]
keys = console
[formatters]
keys = generic
[logger_root]
level = WARN
handlers = console
qualname =
[handler_console]
class = StreamHandler
args = (sys.stderr,)
level = NOTSET
formatter = generic
[formatter_generic]
format = %(levelname)-5.5s [%(name)s] %(message)s
```

`backend/alembic/script.py.mako`:
```mako
"""${message}

Revision ID: ${up_revision}
Revises: ${down_revision | comma,n}
"""
from alembic import op
import sqlalchemy as sa
${imports if imports else ""}

revision = ${repr(up_revision)}
down_revision = ${repr(down_revision)}
branch_labels = ${repr(branch_labels)}
depends_on = ${repr(depends_on)}


def upgrade():
    ${upgrades if upgrades else "pass"}


def downgrade():
    ${downgrades if downgrades else "pass"}
```

`backend/alembic/env.py`:
```python
from logging.config import fileConfig

from alembic import context
from sqlalchemy import engine_from_config, pool

from biolit.storage.models import Base

config = context.config
if config.config_file_name is not None:
    fileConfig(config.config_file_name)

target_metadata = Base.metadata


def run_migrations_offline() -> None:
    context.configure(
        url=config.get_main_option("sqlalchemy.url"),
        target_metadata=target_metadata,
        literal_binds=True,
    )
    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    connectable = engine_from_config(
        config.get_section(config.config_ini_section, {}),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )
    with connectable.connect() as connection:
        context.configure(connection=connection, target_metadata=target_metadata)
        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
```

`backend/alembic/versions/0001_initial.py`:
```python
"""initial papers + paper_embeddings

Revision ID: 0001
Revises:
"""
import sqlalchemy as sa
from alembic import op
from pgvector.sqlalchemy import Vector

revision = "0001"
down_revision = None
branch_labels = None
depends_on = None


def upgrade():
    op.execute("CREATE EXTENSION IF NOT EXISTS vector")
    op.create_table(
        "papers",
        sa.Column("id", sa.String(), primary_key=True),
        sa.Column("source", sa.String(), nullable=False),
        sa.Column("doi", sa.String(), nullable=True),
        sa.Column("pmid", sa.String(), nullable=True),
        sa.Column("title", sa.Text(), nullable=False),
        sa.Column("abstract", sa.Text(), nullable=True),
        sa.Column("year", sa.Integer(), nullable=True),
        sa.Column("text_type", sa.String(), nullable=False),
        sa.Column("license", sa.String(), nullable=True),
        sa.Column("license_tier", sa.String(), nullable=False),
        sa.Column("full_text_pointer", sa.Text(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
    )
    op.create_table(
        "paper_embeddings",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column("paper_id", sa.String(), nullable=False, index=True),
        sa.Column("chunk_kind", sa.String(), nullable=False),
        sa.Column("chunk_index", sa.Integer(), nullable=False),
        sa.Column("content", sa.Text(), nullable=False),
        sa.Column("embedding", Vector(768)),
    )


def downgrade():
    op.drop_table("paper_embeddings")
    op.drop_table("papers")
```

- [ ] **Step 5: Add docker-compose (defined, not required for tests)**

`docker-compose.yml`:
```yaml
services:
  db:
    image: pgvector/pgvector:pg16
    environment:
      POSTGRES_USER: biolit
      POSTGRES_PASSWORD: biolit
      POSTGRES_DB: biolit
    ports:
      - "5432:5432"
    volumes:
      - biolit_pgdata:/var/lib/postgresql/data

volumes:
  biolit_pgdata:
```

- [ ] **Step 6: Run tests + confirm migration imports**

Run: `cd backend && uv run pytest tests/storage/test_models.py -q`
Expected: PASS (3 tests).
Run: `cd backend && uv run python -c "import alembic.versions"` is not meaningful; instead verify the migration module imports:
Run: `cd backend && uv run python -c "import importlib.util, pathlib; p=pathlib.Path('alembic/versions/0001_initial.py'); spec=importlib.util.spec_from_file_location('m', p); m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m); print(m.revision)"`
Expected: prints `0001` with no import error.

- [ ] **Step 7: Commit**

```bash
git add backend/src/biolit/storage backend/alembic.ini backend/alembic docker-compose.yml backend/tests/storage
git commit -m "feat(storage): dormant papers/pgvector schema, Alembic migration, compose"
```

---

### Task 9: CI, Dockerfile, README

**Files:**
- Create: `.github/workflows/ci.yml`
- Create: `backend/Dockerfile`
- Create: `README.md`

**Interfaces:** none (infra/docs).

- [ ] **Step 1: CI workflow**

`.github/workflows/ci.yml`:
```yaml
name: ci
on:
  push:
  pull_request:

jobs:
  backend:
    runs-on: ubuntu-latest
    defaults:
      run:
        working-directory: backend
    steps:
      - uses: actions/checkout@v4
      - name: Install uv
        uses: astral-sh/setup-uv@v3
      - name: Sync
        run: uv sync
      - name: Ruff lint
        run: uv run ruff check .
      - name: Ruff format
        run: uv run ruff format --check .
      - name: Pyright
        run: uv run pyright
      - name: Pytest
        run: uv run pytest -q
```

- [ ] **Step 2: Dockerfile (written, unused this phase)**

`backend/Dockerfile`:
```dockerfile
FROM python:3.12-slim
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv
WORKDIR /app
COPY pyproject.toml ./
COPY src ./src
RUN uv sync --no-dev
CMD ["uv", "run", "python", "-c", "import biolit; print(biolit.__version__)"]
```

- [ ] **Step 3: README**

`README.md`:
```markdown
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
```

- [ ] **Step 4: Full gate locally**

Run: `cd backend && uv run ruff check . && uv run ruff format --check . && uv run pyright && uv run pytest -q`
Expected: all green (ruff clean, pyright 0 errors, all tests pass). Fix any ruff F401
unused-import findings (e.g. drop an unused `LicenseTier` import) before committing.

- [ ] **Step 5: Commit**

```bash
git add .github/workflows/ci.yml backend/Dockerfile README.md
git commit -m "chore: CI gate, backend Dockerfile, README"
```

---

## Self-Review

**Spec coverage:**
- Repo layout (spec §3) → Task 1, 8, 9.
- Domain models incl. three-state `text_type` + license axis (spec §4, ADR-0004) → Tasks 2, 3.
- Hybrid state layer with non-destructive merge (spec §5, ADR-0002) → Task 4.
- PubMed client + PMC OA license cross-reference + 429 retry (spec §6.1, user changes #1, #3) → Tasks 5, 6.
- bioRxiv client jatsxml-driven text_type (spec §6.2, user change #4) → Task 7.
- Dormant storage schema + migration + compose (spec §7, ADR-0003) → Task 8.
- Testing: PMC restrictive-license case, 429→200, bioRxiv three-state, partial-update survival (spec §8, all four user changes) → Tasks 4, 5, 6, 7.
- CI ruff/pyright/pytest (spec §8.4) → Task 9.

**Placeholder scan:** No TBD/TODO; every code step contains full code; every test has real assertions.

**Type consistency:** `PipelineState.extracted_records` is `dict[str, ExtractedRecord]` in both `pipeline.py` and the merge adapter and `SynthesisInput`/`CriticInput`. `ExtractorOutput.records` is `list[ExtractedRecord]` consistently. `request_with_retry` signature matches all call sites (positional `client, method, url`, keyword `retry`, `sleep`, `params`). `Paper` field names used by clients (`full_text_pointer`, `license`, `license_tier`, `extraction_allowed`, `published_doi`, `mesh_terms`) match `paper.py`. `normalize_license` returns `(token, tier)` used identically in both clients.
