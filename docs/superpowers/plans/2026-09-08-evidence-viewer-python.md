# Evidence Viewer — Python Half Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Produce committed, licence-sanitised JSON fixtures of four frozen pipeline runs, with a staleness pin that makes shipping an out-of-date claim fail CI.

**Architecture:** Three new modules in `biolit_evals`, mirroring the existing `acronym_export` / `relevance_export` pattern. `fixture_models.py` defines the schema and *is* the licence guard (`extra="forbid"`, no `abstract` field). `fixture_pin.py` hashes the seven modules whose behaviour determines a displayed value, normalised through the AST. `fixture_export.py` holds a pure `project_run()` plus a network-touching `main()`. One small public accessor is added to `biolit/query/ranking.py` so the exporter never imports a private name.

**Tech Stack:** Python 3.12, pydantic v2, pytest, ruff, pyright, uv.

**Spec:** `docs/superpowers/specs/2026-09-08-evidence-viewer-design.md`

## Global Constraints

- All commands run from `backend/` via `uv run`.
- CI is four steps and must be green before every commit: `uv run ruff check .`, `uv run ruff format --check .`, `uv run pyright`, `uv run pytest -q`.
- Ruff ruleset `E,F,I,UP,B`, line length 100. Imports at top of file only (E402), except deliberate function-local heavy imports inside `main()`.
- `datetime.now(UTC)`, never `timezone.utc`. `StrEnum` per ADR-0005.
- `main()` gets no direct unit test, per project convention.
- Unit tests must never download anything or touch the network.
- tdd-guard is active: **write ONE test at a time**. Adding several tests in one file write is blocked. Use write-then-Edit.
- Commit with `git commit -F -` and a bash heredoc. **No `Claude-Session:` trailer, no session URL, no agent attribution** — this is a public portfolio repo and ~200 prior commits carry none.
- Never manufacture a fake RED by writing deliberately wrong behaviour.
- Determinism fixtures use **7 reverse-inserted elements**, per repo convention.
- CPU-pin guard must stay at 0: `grep -ciE '^name = "(nvidia|triton)' uv.lock`.

---

## File Structure

| File | Responsibility |
|---|---|
| `backend/src/biolit/query/ranking.py` *(modify)* | add public `relevance_score()`; `_relevance_key` calls it |
| `backend/src/biolit_evals/fixture_models.py` *(create)* | the schema, and the guard — `extra="forbid"`, no `abstract` field anywhere |
| `backend/src/biolit_evals/fixture_pin.py` *(create)* | `PINNED_MODULES`, `source_pin()` — AST-normalised hash |
| `backend/src/biolit_evals/fixture_export.py` *(create)* | pure `project_run()` + `main()` (network) |
| `backend/tests/query/test_ranking.py` *(modify)* | one test for `relevance_score` |
| `backend/tests/evals/test_fixture_models.py` *(create)* | schema rejects unsafe shapes |
| `backend/tests/evals/test_fixture_pin.py` *(create)* | pin normalisation and sensitivity |
| `backend/tests/evals/test_fixture_export.py` *(create)* | the licence invariant, attribution, ledger integrity |
| `backend/tests/evals/test_fixture_freshness.py` *(create)* | committed fixtures match the current pin |
| `frontend/src/fixtures/*.json` *(create, Task 5)* | the four committed fixtures |

---

### Task 1: A public relevance score

The fixture displays the ranker's score so a reader can see *why* a cluster sits where it does. `_relevance_key` is private and returns a 3-tuple with a negated first element for sorting. Reaching into it from an eval module would couple the exporter to a sort implementation detail.

**Files:**
- Modify: `backend/src/biolit/query/ranking.py`
- Test: `backend/tests/query/test_ranking.py`

**Interfaces:**
- Consumes: `QueryConcepts`, `MeshTree`, `PharmacologicalActions`, `Cluster` (all existing).
- Produces: `relevance_score(cluster, concepts, *, tree, actions) -> tuple[int, float]` returning `(matched, proximity)` — **matched is positive here**, unlike the negated value inside the sort key.

- [ ] **Step 1: Write the failing test**

Append to `backend/tests/query/test_ranking.py`:

```python
def test_relevance_score_exposes_the_sort_key_without_its_negation():
    """The fixture export displays this so a reader can see why a cluster ranks where it does.
    It must be the SAME computation the sort uses -- a second implementation would drift -- but
    without the sign flip, which exists only to make `sorted` ascending.

    `MESH:MEMBER` matches via pharmacological class and `MESH:NEAR` is one edge from `MESH:QD`,
    so a correct score is (1 matched, proximity 1).
    """
    cluster = _cluster("MESH:MEMBER|MESH:NEAR")

    score = relevance_score(
        cluster, _concepts("MESH:QC", "MESH:QD"), tree=TREE, actions=ACTIONS
    )

    assert score == (1, 1)
```

Add `relevance_score` to the existing import line at the top of the file:

```python
from biolit.query.ranking import rank_clusters, relevance_score
```

- [ ] **Step 2: Run it and confirm it fails for the right reason**

Run: `uv run pytest tests/query/test_ranking.py::test_relevance_score_exposes_the_sort_key_without_its_negation -v`
Expected: FAIL — `ImportError: cannot import name 'relevance_score'`.

- [ ] **Step 3: Implement**

In `backend/src/biolit/query/ranking.py`, replace the body of `_relevance_key` so it delegates, and add the public function above it:

```python
def relevance_score(
    cluster: Cluster,
    concepts: QueryConcepts,
    *,
    tree: MeshTree,
    actions: PharmacologicalActions,
) -> tuple[int, float]:
    """The two ordinal signals behind the sort, without the sort's negation.

    Exposed so a consumer can DISPLAY why a cluster ranks where it does. It is deliberately
    the same computation `_relevance_key` consumes rather than a parallel one: a second
    implementation of a score is a second thing to keep in step, and this project has already
    recorded what happens when a displayed number and a computed number drift apart.
    """
    sides = cluster.key.split("|")
    matched = sum(1 for side in sides if side_matches(side, concepts, actions))
    residual = [
        min(
            (
                distance
                for concept_id in concepts.ids
                if (distance := tree.distance(side, concept_id)) is not None
            ),
            default=_NO_SHARED_TREE,
        )
        for side in sides
        if not side_matches(side, concepts, actions)
    ]
    return matched, max(residual, default=0.0)
```

Then reduce `_relevance_key` to:

```python
def _relevance_key(
    cluster: Cluster, concepts: QueryConcepts, tree: MeshTree, actions: PharmacologicalActions
) -> tuple:
    """Lexicographic: matched sides (more first), then hierarchy proximity, then key order.

    No threshold anywhere. Each signal is ordinal and the sort consumes it as such, so there
    is no constant to tune and none can be tuned against the relevance labels later. ADR-0022
    keeps that property: `side_matches` is a set relation, not a distance with a cutoff.

    `cluster.key` last preserves `cluster_papers`'s reproducible-and-diffable guarantee for
    clusters the score cannot separate.
    """
    matched, proximity = relevance_score(cluster, concepts, tree=tree, actions=actions)
    return (-matched, proximity, cluster.key)
```

⚠️ **Move the DEF-0003 and ADR-0022 comment blocks currently inside `_relevance_key` into `relevance_score`, next to the `residual` computation they explain.** They document the `max`-over-unmatched-sides choice, which now lives there. Losing them would delete the record of a fixed defect.

- [ ] **Step 4: Run the whole ranking suite**

Run: `uv run pytest tests/query/ -q`
Expected: PASS, 18 tests. The 11 pre-existing ranking tests must still pass unchanged — that is the check that the refactor preserved behaviour.

- [ ] **Step 5: Full CI, then commit**

```bash
uv run ruff check . && uv run ruff format --check . && uv run pyright && uv run pytest -q
```

```bash
git add src/biolit/query/ranking.py tests/query/test_ranking.py
git commit -F - <<'EOF'
refactor(query): expose relevance_score so a consumer can show why a cluster ranks

The evidence viewer displays the ranker's score. Reaching into _relevance_key
would couple a presentation module to a sort implementation detail -- including
the negation that exists only to make `sorted` ascending.

relevance_score returns (matched, proximity) with matched positive.
_relevance_key now calls it and applies the sign, so there is exactly one
implementation of the score rather than a displayed one and a computed one that
can drift.

Behaviour is unchanged: the 11 existing ranking tests pass untouched.
EOF
```

---

### Task 2: The fixture schema, which is the guard

**Files:**
- Create: `backend/src/biolit_evals/fixture_models.py`
- Test: `backend/tests/evals/test_fixture_models.py`

**Interfaces:**
- Produces: `SCHEMA_VERSION: int`, `PaperStub`, `FixtureCluster`, `FixtureFinding`, `FixtureRun` — all pydantic `BaseModel` with `model_config = ConfigDict(extra="forbid")`.
- `FixtureRun` fields: `schema_version: int`, `slug: str`, `query: str`, `generated_at: str`, `source_pin: str`, `stages: list[StageReport]`, `clusters: list[FixtureCluster]`, `answer: str`, `papers: dict[str, PaperStub]`, `findings: list[FixtureFinding]`.

- [ ] **Step 1: Write the first failing test**

Create `backend/tests/evals/test_fixture_models.py` with **exactly one** test (tdd-guard blocks multi-test file creation):

```python
import pytest
from pydantic import ValidationError

from biolit_evals.fixture_models import PaperStub


def test_a_paper_stub_refuses_an_abstract_rather_than_dropping_it():
    """⭐ THE GUARD IS THE SCHEMA, so the schema is what gets tested.

    DEF-0006: `--json-out` serialises abstracts of papers the licence gate refused -- 7 of 8
    refused papers carried one, up to 2096 characters at license_tier='unknown'. The response
    is to make the unsafe shape unrepresentable rather than to remember not to populate it.

    ⚠️ `extra="forbid"` is load-bearing and not decoration. Pydantic's DEFAULT is to ignore an
    unknown key silently, which is indistinguishable from safety until someone reads the model
    -- and a field that must always be empty is a field someone eventually fills.
    """
    with pytest.raises(ValidationError, match="abstract"):
        PaperStub(
            title="Clozapine and agranulocytosis",
            journal="J Clin Psych",
            year=2024,
            doi="10.1000/x",
            pmid="123",
            license="cc_by",
            license_tier="open",
            extraction_allowed=True,
            abstract="BACKGROUND: Clozapine is the gold standard...",
        )
```

- [ ] **Step 2: Run it and confirm it fails**

Run: `uv run pytest tests/evals/test_fixture_models.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'biolit_evals.fixture_models'`.

- [ ] **Step 3: Implement the module**

Create `backend/src/biolit_evals/fixture_models.py`:

```python
"""The evidence viewer's fixture schema. The schema IS the licence guard.

⚠️ THERE IS NO `abstract` FIELD ANYWHERE IN THIS MODULE, and that is the point. DEF-0006
records what happened when a serialisation path outgrew an argument about which contracts see
a `Paper`: `--json-out` emitted full verbatim abstracts for 7 of 8 papers the licence gate had
refused. A field that must always be empty is a field someone eventually fills, so the unsafe
shape is made unrepresentable instead.

⭐ EVERY MODEL SETS `extra="forbid"`. Pydantic's default is to IGNORE an unknown key, which
looks identical to safety from the outside -- a caller passing `abstract=...` would be told
nothing and would reasonably assume it landed. Forbidding turns that into a loud failure at
the boundary, which is where a rights error has to surface.

Paper stubs are emitted for EVERY licence tier and text for none: "20 retrieved, 8 refused,
tiers unknown/non_commercial/open" is the interesting part of the gate and needs no abstract
to tell.
"""

from pydantic import BaseModel, ConfigDict

from biolit.state.pipeline import StageReport

#: Bumped when a field is added, removed or re-meant. The frontend reads this and refuses a
#: fixture it does not understand rather than rendering a partial one.
SCHEMA_VERSION = 1


class PaperStub(BaseModel):
    """Citation metadata and licence facts. NEVER text.

    `license` and `doi` are not optional decoration: every tier the gate allows is a Creative
    Commons licence and every one of them REQUIRES ATTRIBUTION, while the synthesis stage's own
    ledger note says citation assembly is not yet built. The viewer has to supply what the
    pipeline does not, and it can only do that if the fixture carries it.
    """

    model_config = ConfigDict(extra="forbid")

    title: str
    journal: str | None = None
    year: int | None = None
    doi: str | None = None
    pmid: str | None = None
    license: str | None = None
    license_tier: str
    extraction_allowed: bool


class FixtureCluster(BaseModel):
    """One cluster as displayed: identity, membership, and why it ranks where it does."""

    model_config = ConfigDict(extra="forbid")

    key: str
    concept_names: list[str]
    paper_ids: list[str]
    rank: int
    matched: int
    proximity: float
    #: Present only where a frozen relevance label exists. The viewer MUST mark these as
    #: annotation labels from a pass whose control instrument was later found compromised
    #: (ADR-0021) -- never as ground truth the pipeline achieved.
    label: str | None = None


class FixtureFinding(BaseModel):
    """A defect pinned to this run, quoting the adjudication rather than paraphrasing it."""

    model_config = ConfigDict(extra="forbid")

    defect_id: str
    anchor: str
    headline: str
    #: Verbatim from the annotator. A paraphrase of an adjudication is a new claim.
    reason: str


class FixtureRun(BaseModel):
    """One frozen run, sanitised for publication."""

    model_config = ConfigDict(extra="forbid")

    schema_version: int
    slug: str
    query: str
    generated_at: str
    source_pin: str
    stages: list[StageReport]
    clusters: list[FixtureCluster]
    answer: str
    papers: dict[str, PaperStub]
    findings: list[FixtureFinding]
```

- [ ] **Step 4: Run it and confirm PASS**

Run: `uv run pytest tests/evals/test_fixture_models.py -v`
Expected: PASS.

- [ ] **Step 5: Add the second test via Edit (not a rewrite)**

Append to `backend/tests/evals/test_fixture_models.py`:

```python
def test_the_run_model_forbids_extras_too_so_the_guard_is_not_only_on_the_stub():
    """The stub is where an abstract would most plausibly be added, but a caller can just as
    easily hang one off the run. Every model in this schema forbids extras; this pins that the
    rule is uniform rather than remembered in one place.
    """
    with pytest.raises(ValidationError, match="abstracts"):
        FixtureRun(
            schema_version=1,
            slug="s",
            query="q",
            generated_at="2026-09-08T00:00:00+00:00",
            source_pin="deadbeef",
            stages=[],
            clusters=[],
            answer="",
            papers={},
            findings=[],
            abstracts={"pmid": "text"},
        )
```

Update the import line to `from biolit_evals.fixture_models import FixtureRun, PaperStub`.

- [ ] **Step 6: Run and confirm PASS**

Run: `uv run pytest tests/evals/test_fixture_models.py -q`
Expected: 2 passed.

- [ ] **Step 7: Full CI, then commit**

```bash
uv run ruff check . && uv run ruff format --check . && uv run pyright && uv run pytest -q
```

```bash
git add src/biolit_evals/fixture_models.py tests/evals/test_fixture_models.py
git commit -F - <<'EOF'
feat(evals): the evidence-viewer fixture schema, which is the licence guard

There is no `abstract` field anywhere in this module and that is the point.
DEF-0006 records what happened when a serialisation path outgrew an argument
about which contracts see a Paper: --json-out emitted full verbatim abstracts for
7 of 8 papers the licence gate had refused. A field that must always be empty is
a field someone eventually fills, so the unsafe shape is unrepresentable instead.

Every model sets extra="forbid". Pydantic's default is to ignore an unknown key,
which looks identical to safety from the outside -- a caller passing abstract=...
would be told nothing and would reasonably assume it landed. Both the stub and
the run model are pinned, so the rule is uniform rather than remembered in one
place.

PaperStub carries license and doi because every tier the gate allows is Creative
Commons and requires attribution, while synthesis notes that citation assembly is
not yet built. The viewer supplies what the pipeline does not.
EOF
```

---

### Task 3: The staleness pin

**Files:**
- Create: `backend/src/biolit_evals/fixture_pin.py`
- Test: `backend/tests/evals/test_fixture_pin.py`

**Interfaces:**
- Produces: `PINNED_MODULES: tuple[str, ...]` (dotted module names), `source_pin() -> str` (hex sha256), `_normalise(source: str) -> str` (AST dump with docstrings stripped).

- [ ] **Step 1: Write the first failing test**

Create `backend/tests/evals/test_fixture_pin.py` with one test:

```python
from biolit_evals.fixture_pin import _normalise


def test_the_pin_ignores_comments_and_docstrings_but_not_statements():
    """⭐ WHY THIS NORMALISES INSTEAD OF HASHING BYTES, and it is not a micro-optimisation.

    This repo's modules carry very heavy comments and docstrings, edited constantly and
    deliberately -- several of them are the only record of a fixed defect. A byte hash would
    fire on every prose edit, and a staleness check that cries wolf is a check that gets
    suppressed, which is worse than not having one.

    So the pin must be blind to comments and docstrings and sensitive to behaviour. A
    comment-only change genuinely does NOT invalidate a fixture; a changed statement does.
    """
    commented = '"""Doc one."""\n# a comment\nX = 1\n'
    reworded = '"""Doc two, entirely rewritten."""\n# a different comment\nX = 1\n'
    behavioural = '"""Doc one."""\n# a comment\nX = 2\n'

    assert _normalise(commented) == _normalise(reworded)
    assert _normalise(commented) != _normalise(behavioural)
```

- [ ] **Step 2: Run it and confirm it fails**

Run: `uv run pytest tests/evals/test_fixture_pin.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'biolit_evals.fixture_pin'`.

- [ ] **Step 3: Implement**

Create `backend/src/biolit_evals/fixture_pin.py`:

```python
"""Pin the modules whose behaviour determines a value a fixture displays.

⭐ THE RULE, so the set is derivable rather than remembered: pin every module whose behaviour
determines a value the fixture SHOWS. A fixture that claims "17 kept" after the ranker changed
is a false claim on a public page, and the requirement is that shipping one is structurally
hard rather than something anybody remembers to check.

⚠️ HASHED OVER THE PARSED AST WITH DOCSTRINGS STRIPPED, NOT OVER FILE BYTES. This repo's
modules carry very heavy comments and docstrings that are edited constantly; a byte hash would
fire on every prose edit, and a check that cries wolf gets suppressed. Normalising makes the
pin fire on behaviour and stay quiet on prose -- which is the correct trade, because a
comment-only edit genuinely does not invalidate a fixture.

⚠️ WHAT THIS DOES NOT COVER, listed rather than left unstated. `clients/pubmed.py::_parse_article`
assigns license, license_tier, doi and title, and is deliberately absent: that module is
dominated by transport that changes for reasons unrelated to what a fixture claims. Also
uncovered: the MeSH artifacts, the NER checkpoint, and NCBI's query translation. All four are
recorded only by `generated_at`, and what catches them is regeneration. If the first ever
bites, the upgrade is to hash that function's AST subtree alone -- a small extension here, not
a redesign.
"""

import ast
import hashlib
import importlib.util

#: Every module whose behaviour determines a value the fixture displays.
PINNED_MODULES: tuple[str, ...] = (
    "biolit.query.concepts",  # which clusters match
    "biolit.query.ranking",  # the order, and the score shown
    "biolit.pipeline.stages",  # every StageReport the site renders
    "biolit.cluster.pairing",  # which clusters exist at all
    "biolit.synth.template",  # renders the answer string, verbatim
    "biolit.domain.licensing",  # the tier table -> gate counts AND the tier shown
    "biolit.clients.pmc",  # parses the permissions block into the licence token
)


def _strip_docstrings(tree: ast.AST) -> ast.AST:
    for node in ast.walk(tree):
        if not isinstance(node, ast.Module | ast.ClassDef | ast.FunctionDef | ast.AsyncFunctionDef):
            continue
        body = node.body
        if body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant):
            if isinstance(body[0].value.value, str):
                node.body = body[1:]
    return tree


def _normalise(source: str) -> str:
    """Source -> a string that changes with behaviour and not with prose."""
    return ast.dump(_strip_docstrings(ast.parse(source)))


def source_pin() -> str:
    """A hex digest over every pinned module's normalised source, in declaration order."""
    digest = hashlib.sha256()
    for name in PINNED_MODULES:
        spec = importlib.util.find_spec(name)
        if spec is None or spec.origin is None:
            raise RuntimeError(
                f"fixture_pin: cannot locate {name!r}. A pinned module was renamed or removed "
                "without updating PINNED_MODULES, which would silently reduce what the pin "
                "covers -- refusing rather than hashing a smaller set."
            )
        with open(spec.origin, encoding="utf-8") as handle:
            digest.update(name.encode())
            digest.update(_normalise(handle.read()).encode())
    return digest.hexdigest()
```

- [ ] **Step 4: Run and confirm PASS**

Run: `uv run pytest tests/evals/test_fixture_pin.py -v`
Expected: PASS.

- [ ] **Step 5: Add the second test via Edit**

Append:

```python
def test_source_pin_is_stable_across_calls_and_refuses_a_missing_module(monkeypatch):
    """Stability first: a pin recomputed in the same tree must be identical, or every fixture
    would read stale on every run and the check would be worthless.

    And a renamed module must RAISE rather than hash a smaller set. Silently pinning six
    modules instead of seven is the failure mode that matters here -- the check would still
    pass, while covering less than its docstring claims.
    """
    assert source_pin() == source_pin()

    monkeypatch.setattr(
        "biolit_evals.fixture_pin.PINNED_MODULES", ("biolit.query.ranking", "biolit.no.such")
    )
    with pytest.raises(RuntimeError, match="biolit.no.such"):
        source_pin()
```

Update imports to:

```python
import pytest

from biolit_evals.fixture_pin import _normalise, source_pin
```

- [ ] **Step 6: Run and confirm PASS**

Run: `uv run pytest tests/evals/test_fixture_pin.py -q`
Expected: 2 passed.

- [ ] **Step 7: Full CI, then commit**

```bash
uv run ruff check . && uv run ruff format --check . && uv run pyright && uv run pytest -q
```

```bash
git add src/biolit_evals/fixture_pin.py tests/evals/test_fixture_pin.py
git commit -F - <<'EOF'
feat(evals): a staleness pin over the modules that determine what a fixture claims

A fixture claiming "17 kept" after the ranker changed is a false claim on a public
page. The requirement is that shipping one is structurally hard, not something
anybody remembers to check.

The set is derivable rather than remembered: pin every module whose behaviour
determines a value the fixture displays. Seven of them -- query/concepts,
query/ranking, pipeline/stages, cluster/pairing, synth/template, domain/licensing,
clients/pmc. The last three were missed on a first pass, and the gap was real: a
fixture could have shown a stale licence tier or stale answer formatting while the
pin stayed green.

Hashed over the parsed AST with docstrings stripped rather than over file bytes.
This repo edits comments and docstrings constantly -- several are the only record
of a fixed defect -- so a byte hash would fire on every prose edit, and a check
that cries wolf gets suppressed. The pin fires on behaviour and stays quiet on
prose.

A renamed or removed pinned module raises rather than hashing a smaller set:
silently covering six of seven would leave the check passing while claiming more
than it does.

Documents what it does NOT cover -- pubmed.py::_parse_article, the MeSH artifacts,
the NER checkpoint, NCBI translation drift -- with the reason the first is omitted
and the upgrade available if it bites.
EOF
```

---

### Task 4: Projection, and the licence invariant

**Files:**
- Create: `backend/src/biolit_evals/fixture_export.py`
- Test: `backend/tests/evals/test_fixture_export.py`

**Interfaces:**
- Consumes: `FixtureRun`/`PaperStub`/`FixtureCluster`/`SCHEMA_VERSION` (Task 2), `source_pin()` (Task 3), `relevance_score()` (Task 1).
- Produces:
  ```python
  def project_run(
      state: PipelineState,
      *,
      slug: str,
      concepts: QueryConcepts,
      tree: MeshTree,
      actions: PharmacologicalActions,
      names: Mapping[str, str],
      labels: Mapping[str, str],
      findings: Sequence[FixtureFinding],
      generated_at: str,
  ) -> FixtureRun
  ```
  `names` maps concept id → display name; `labels` maps cluster key → frozen relevance label.

- [ ] **Step 1: Write the first failing test — the one that would have caught DEF-0006**

Create `backend/tests/evals/test_fixture_export.py` with one test:

```python
import json

from biolit.domain.enums import LicenseTier, Source, TextType
from biolit.domain.paper import Paper
from biolit.domain.records import Cluster
from biolit.state.pipeline import PipelineState

from biolit_evals.fixture_export import project_run

SENTINEL = "ZZQX-refused-abstract-sentinel-ZZQX"


def _refused_paper() -> Paper:
    return Paper(
        id="10.1/refused",
        source=Source.pubmed,
        title="A refused paper",
        abstract=f"BACKGROUND: {SENTINEL} and more text.",
        text_type=TextType.abstract_only,
        license=None,
        license_tier=LicenseTier.unknown,
        extraction_allowed=False,
    )


def test_no_text_from_a_refused_paper_survives_projection():
    """⭐ THE TEST THAT WOULD HAVE CAUGHT DEF-0006.

    `--json-out` serialises `PipelineState` wholesale, and `candidate_papers` retains every
    retrieved `Paper` with its abstract whether or not the licence gate refused it -- measured
    at 7 of 8 refused papers carrying one, up to 2096 characters at license_tier='unknown'.

    The assertion is over the SERIALISED fixture rather than its fields, because the question
    is not "did we remember to omit the abstract attribute" but "can this string reach a
    published page by any route at all".
    """
    state = PipelineState(question="clozapine and agranulocytosis")
    state.candidate_papers = [_refused_paper()]
    state.answer = "No findings."

    run = project_run(
        state,
        slug="clozapine",
        concepts=_no_concepts(),
        tree=MeshTree({}),
        actions=PharmacologicalActions({}),
        names={},
        labels={},
        findings=[],
        generated_at="2026-09-08T00:00:00+00:00",
    )

    assert SENTINEL not in run.model_dump_json()
    assert "10.1/refused" in run.papers, "the stub is still emitted -- refusal is the story"
    assert run.papers["10.1/refused"].extraction_allowed is False
```

Add these helpers above the test:

```python
from biolit.canon.mesh_actions import PharmacologicalActions
from biolit.canon.mesh_tree import MeshTree
from biolit.query.concepts import QueryConcepts


def _no_concepts() -> QueryConcepts:
    return QueryConcepts(frozenset(), {}, ())
```

- [ ] **Step 2: Run it and confirm it fails**

Run: `uv run pytest tests/evals/test_fixture_export.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'biolit_evals.fixture_export'`.

- [ ] **Step 3: Implement `project_run` only** (leave `main()` for Task 5)

Create `backend/src/biolit_evals/fixture_export.py`:

```python
"""Project a finished `PipelineState` into a publishable, licence-sanitised fixture.

⚠️ THIS IS NOT A WRAPPER AROUND `--json-out`, AND MUST NEVER BECOME ONE. `--json-out` is the
path DEF-0006 is filed against: it writes refused papers' verbatim abstracts to disk. Driving
generation through it would create the unsafe file first and sanitise second, and the unsafe
file must never exist -- not merely never be committed. `main()` therefore calls the stage
functions in-process and hands the state straight to `project_run`.

`project_run` is pure: no network, no filesystem, no clock. Everything variable is a
parameter, so the test can construct a state containing a sentinel string and assert the
sentinel cannot reach the output by any route.
"""

from collections.abc import Mapping, Sequence
from datetime import UTC, datetime

from biolit.canon.mesh_actions import PharmacologicalActions
from biolit.canon.mesh_tree import MeshTree
from biolit.query.concepts import QueryConcepts
from biolit.query.ranking import relevance_score
from biolit.state.pipeline import PipelineState

from biolit_evals.fixture_models import (
    SCHEMA_VERSION,
    FixtureCluster,
    FixtureFinding,
    FixtureRun,
    PaperStub,
)
from biolit_evals.fixture_pin import source_pin


def project_run(
    state: PipelineState,
    *,
    slug: str,
    concepts: QueryConcepts,
    tree: MeshTree,
    actions: PharmacologicalActions,
    names: Mapping[str, str],
    labels: Mapping[str, str],
    findings: Sequence[FixtureFinding],
    generated_at: str | None = None,
) -> FixtureRun:
    """Sanitised projection. Abstracts cannot survive it, because the schema has no field."""
    papers = {
        paper.id: PaperStub(
            title=paper.title,
            journal=paper.journal,
            year=paper.year,
            doi=paper.doi,
            pmid=paper.pmid,
            license=paper.license,
            license_tier=str(paper.license_tier),
            extraction_allowed=paper.extraction_allowed,
        )
        for paper in state.candidate_papers
    }
    clusters = []
    for rank, cluster in enumerate(state.clusters, start=1):
        matched, proximity = relevance_score(cluster, concepts, tree=tree, actions=actions)
        sides = cluster.key.split("|")
        clusters.append(
            FixtureCluster(
                key=cluster.key,
                concept_names=[names.get(side, side) for side in sides],
                paper_ids=list(cluster.paper_ids),
                rank=rank,
                matched=matched,
                proximity=proximity,
                label=labels.get(cluster.key),
            )
        )
    return FixtureRun(
        schema_version=SCHEMA_VERSION,
        slug=slug,
        query=state.question,
        generated_at=generated_at or datetime.now(UTC).isoformat(),
        source_pin=source_pin(),
        stages=list(state.stages),
        clusters=clusters,
        answer=state.answer or "",
        papers=papers,
        findings=list(findings),
    )
```

- [ ] **Step 4: Run and confirm PASS**

Run: `uv run pytest tests/evals/test_fixture_export.py -v`
Expected: PASS.

- [ ] **Step 5: Add the attribution test via Edit**

Append:

```python
def test_every_cluster_paper_resolves_to_a_stub_carrying_licence_and_doi():
    """Attribution is structural, not editorial. Every tier the gate ALLOWS is a Creative
    Commons licence and every one of them requires attribution, while the synthesis stage's
    own ledger note says citation assembly is not yet built. The viewer has to supply it, and
    can only do so if every paper a cluster names carries the fields to supply it with.
    """
    allowed = Paper(
        id="10.1/ok",
        source=Source.pubmed,
        title="An allowed paper",
        abstract="Text that is licensed for reuse.",
        doi="10.1/ok",
        text_type=TextType.abstract_only,
        license="cc_by",
        license_tier=LicenseTier.open,
        extraction_allowed=True,
    )
    state = PipelineState(question="q")
    state.candidate_papers = [allowed]
    state.clusters = [Cluster(key="MESH:A|MESH:B", paper_ids=["10.1/ok"])]

    run = project_run(
        state,
        slug="s",
        concepts=_no_concepts(),
        tree=MeshTree({}),
        actions=PharmacologicalActions({}),
        names={"MESH:A": "Alpha", "MESH:B": "Beta"},
        labels={},
        findings=[],
        generated_at="2026-09-08T00:00:00+00:00",
    )

    for cluster in run.clusters:
        for paper_id in cluster.paper_ids:
            stub = run.papers[paper_id]
            assert stub.license and stub.doi, "a cited paper must carry what attribution needs"
    assert run.clusters[0].concept_names == ["Alpha", "Beta"]
```

- [ ] **Step 6: Run and confirm PASS**

Run: `uv run pytest tests/evals/test_fixture_export.py -q`
Expected: 2 passed.

- [ ] **Step 7: Add the ledger-integrity test via Edit**

Append:

```python
def test_the_ledger_survives_projection_with_its_arithmetic_intact():
    """`StageReport` documents that where the units match, `n_in - sum(dropped) == n_out` is
    checkable. The site renders those numbers, so the projection must not quietly reshape them.

    Copied by reference rather than rebuilt: a projection that reconstructs a StageReport is a
    second place for the ledger to be wrong, and the first rendered ledger already misreported
    three of seven stages by conflating `dropped` with `noted`.
    """
    state = PipelineState(question="q")
    state.stages = [
        StageReport(
            name="licence_gate",
            status=StageStatus.completed,
            n_in=20,
            n_out=12,
            dropped={"licence_refused:none": 8},
        )
    ]

    run = project_run(
        state,
        slug="s",
        concepts=_no_concepts(),
        tree=MeshTree({}),
        actions=PharmacologicalActions({}),
        names={},
        labels={},
        findings=[],
        generated_at="2026-09-08T00:00:00+00:00",
    )

    stage = run.stages[0]
    assert stage.n_in - sum(stage.dropped.values()) == stage.n_out
    assert json.loads(run.model_dump_json())["stages"][0]["dropped"] == {
        "licence_refused:none": 8
    }
```

Add `StageReport, StageStatus` to the `biolit.state.pipeline` import line.

- [ ] **Step 8: Run, full CI, commit**

Run: `uv run pytest tests/evals/test_fixture_export.py -q` → 3 passed.

```bash
uv run ruff check . && uv run ruff format --check . && uv run pyright && uv run pytest -q
```

```bash
git add src/biolit_evals/fixture_export.py tests/evals/test_fixture_export.py
git commit -F - <<'EOF'
feat(evals): licence-sanitised projection of a run into a publishable fixture

project_run is pure -- no network, no filesystem, no clock -- so the invariant
test can build a state holding a sentinel string and assert the sentinel cannot
reach the output by any route. That is the test that would have caught DEF-0006,
and it asserts over the SERIALISED fixture rather than its fields: the question is
not whether we remembered to omit an attribute but whether the string can reach a
published page at all.

Refused papers still get a stub. Their refusal is the interesting part of the
licence gate -- "20 retrieved, 8 refused, tiers unknown/non_commercial/open" -- and
telling it needs no abstract.

Two further invariants are structural rather than editorial: every paper a cluster
names carries the license and doi attribution requires, and the stage ledger is
copied by reference so its arithmetic survives projection. A projection that
rebuilt StageReport would be a second place for the ledger to be wrong, and the
first rendered ledger already misreported three of seven stages.

Explicitly not a wrapper around --json-out, which is the path DEF-0006 is filed
against; generation calls the stages in-process so the unsafe file never exists.
EOF
```

---

### Task 5: The generator, and the four fixtures

⚠️ **This task touches the network and is not TDD.** `main()` gets no unit test per project convention. Its output is verified by Task 6 plus the invariant already pinned in Task 4.

**Prerequisites — verify before starting:**

```bash
uv run python -m biolit.canon.build_mesh_tree
uv run python -m biolit.canon.build_mesh_actions
ls ~/.cache/huggingface/hub | grep BiomedNLP   # the NER checkpoint must be present
```

**Files:**
- Modify: `backend/src/biolit_evals/fixture_export.py` (add `main()` and `FEATURED`)
- Create: `frontend/src/fixtures/{statins-rhabdomyolysis,isotretinoin-depression,cisplatin-nephrotoxicity,metformin-lactic-acidosis}.json`

- [ ] **Step 1: Add `FEATURED` and `main()` to `fixture_export.py`**

Append to the module:

```python
#: The four runs, and the finding each exists to show. Slugs are stable: the frontend routes
#: on them.
FEATURED: dict[str, str] = {
    "statins-rhabdomyolysis": "statins and rhabdomyolysis",
    "isotretinoin-depression": "isotretinoin and depression",
    "cisplatin-nephrotoxicity": "cisplatin nephrotoxicity",
    "metformin-lactic-acidosis": "metformin and lactic acidosis",
}

DEFAULT_OUT = "../frontend/src/fixtures"
DEFAULT_MAX_PAPERS = 40


def main(argv: list[str] | None = None) -> None:
    """Regenerate fixtures by running the real pipeline in-process.

    ⚠️ IN-PROCESS, NEVER THROUGH `--json-out`. That path writes refused papers' abstracts to
    disk (DEF-0006); the unsafe artifact must never exist, not merely never be committed.

    `main()` gets no direct unit test per project convention; `project_run` and `source_pin`
    are tested in their own modules.
    """
    import argparse
    import asyncio
    import gzip
    import json
    from pathlib import Path

    import httpx

    from biolit.canon.canonicalize import canonicalize
    from biolit.canon.linker import DictionaryLinker
    from biolit.canon.mesh import MeshDictionary
    from biolit.clients.pubmed import PubMedClient
    from biolit.cluster.pairing import SameSentencePairing
    from biolit.config import get_settings
    from biolit.ner.extract import extract_entities
    from biolit.ner.model import NerModel
    from biolit.pipeline.stages import (
        cluster_stage,
        critic_stub,
        entities_stage,
        records_stage,
        retrieve_stage,
        select_stage,
        synthesis_stage,
    )
    from biolit.query.concepts import resolve_query_concepts
    from biolit.state.pipeline import PipelineState

    parser = argparse.ArgumentParser(description="Regenerate evidence-viewer fixtures.")
    parser.add_argument("--slug", default=None, help="one slug, or all of FEATURED")
    parser.add_argument("--out", default=DEFAULT_OUT)
    parser.add_argument("--max-papers", type=int, default=DEFAULT_MAX_PAPERS)
    args = parser.parse_args(argv)

    slugs = [args.slug] if args.slug else list(FEATURED)
    unknown = [s for s in slugs if s not in FEATURED]
    if unknown:
        raise SystemExit(f"unknown slug(s): {unknown}; known: {sorted(FEATURED)}")

    settings = get_settings()
    tree = MeshTree.from_artifact(settings.mesh_tree_artifact_path)
    actions = PharmacologicalActions.from_artifact(settings.mesh_actions_artifact_path)
    dictionary = MeshDictionary.from_artifact(settings.mesh_artifact_path)
    linker = DictionaryLinker(dictionary)
    ner = NerModel.load(settings)

    names: dict[str, str] = {}
    with gzip.open(settings.mesh_artifact_path, "rt", encoding="utf-8") as handle:
        for _surface, entries in json.load(handle).items():
            for concept_id, name, _exact in entries:
                names.setdefault(concept_id, name)

    labels: dict[str, dict[str, str]] = {}
    labels_path = Path("evals/gold/relevance_labels.jsonl")
    if labels_path.exists():
        for line in labels_path.read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            if not row["is_distractor"]:
                labels.setdefault(row["query"], {})[row["cluster_key"]] = row["label"]

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    async def build(slug: str) -> None:
        query = FEATURED[slug]
        async with httpx.AsyncClient(timeout=60) as http:
            client = PubMedClient(http, settings)
            found = await client.esearch_detailed(query, retmax=args.max_papers)
            papers = await client.efetch(found.pmids)

        state = PipelineState(question=query)
        state.candidate_papers = papers
        state.stages.append(retrieve_stage(found.pmids, papers))

        def extract(text: str):
            found_entities = extract_entities(
                text, ner, score_threshold=settings.ner_score_threshold
            )
            return canonicalize(found_entities, text, linker=linker)

        entities_by_paper, entity_report = entities_stage(papers, extract=extract)
        state.stages.append(entity_report)

        outcome = records_stage(papers, entities_by_paper)
        state.extracted_records = outcome.records
        # TWO reports, not one: the licence gate and the extractor are separate ledger rows,
        # and the site renders `licence_refused` as its own line.
        state.stages.extend([outcome.licence, outcome.extract])

        texts = {paper.id: paper.abstract or "" for paper in papers}
        clusters, cluster_report = cluster_stage(
            list(outcome.records.values()), texts=texts, pairing=SameSentencePairing()
        )
        state.stages.append(cluster_report)

        concepts = resolve_query_concepts(found.concept_terms, lookup=linker.link)
        clusters, select_report = select_stage(clusters, concepts, tree=tree, actions=actions)
        state.clusters = clusters
        state.stages.append(select_report)

        state.stages.append(critic_stub(len(clusters)))
        answer, synthesis_report = synthesis_stage(
            clusters, outcome.records, {paper.id: paper for paper in papers}
        )
        state.answer = answer
        state.stages.append(synthesis_report)

        run = project_run(
            state,
            slug=slug,
            concepts=concepts,
            tree=tree,
            actions=actions,
            names=names,
            labels=labels.get(query, {}),
            findings=[],
        )
        path = out_dir / f"{slug}.json"
        path.write_text(run.model_dump_json(indent=2), encoding="utf-8")
        print(f"{slug}: {len(run.clusters)} clusters, {len(run.papers)} papers -> {path}")

    for slug in slugs:
        asyncio.run(build(slug))


if __name__ == "__main__":
    main()
```

✅ **Signatures verified against `biolit/pipeline/__main__.py` while writing this plan**, and a first draft had six of them wrong. Worth knowing, because each is a trap a reasonable guess falls into: `retrieve_stage(pmids, papers)` is **synchronous** and returns only a report — the papers come from `client.efetch()`; `entities_stage` takes **one** `extract` callable that does NER *and* canonicalization; `records_stage` returns an outcome carrying **two** stage reports (`.licence` and `.extract`), so appending one loses the licence row the site is built to show; `cluster_stage` needs `texts=` as well as `pairing=`; the model loads via `NerModel.load(settings)`, not its constructor; and `extract_entities` takes the model **positionally**.

- [ ] **Step 2: Generate the four fixtures**

```bash
cd backend
PYTHONIOENCODING=utf-8 uv run python -m biolit_evals.fixture_export
```

Expected: four lines, one per slug. **`statins-rhabdomyolysis` must report 17 clusters** — if it reports 12, the MeSH actions artifact was not built and ADR-0022 is not in effect.

- [ ] **Step 3: Verify the invariant on real output**

```bash
PYTHONIOENCODING=utf-8 uv run python - <<'PY'
import json, pathlib
for p in sorted(pathlib.Path("../frontend/src/fixtures").glob("*.json")):
    d = json.loads(p.read_text(encoding="utf-8"))
    blob = p.read_text(encoding="utf-8")
    assert "abstract" not in blob, f"{p.name}: the string 'abstract' appears"
    refused = [k for k, v in d["papers"].items() if not v["extraction_allowed"]]
    print(f"{p.name:<34} clusters={len(d['clusters']):>3} papers={len(d['papers']):>3} "
          f"refused={len(refused):>3} pin={d['source_pin'][:12]}")
PY
```

- [ ] **Step 4: Commit the generator and the fixtures**

```bash
git add src/biolit_evals/fixture_export.py ../frontend/src/fixtures
git commit -F - <<'EOF'
feat(evals): regenerate the four evidence-viewer fixtures, in-process

Runs the real stage functions and hands the state straight to project_run. It does
NOT shell out to `python -m biolit.pipeline --json-out`: that is the path DEF-0006
is filed against, and driving generation through it would write refused papers'
abstracts to disk before any sanitising step could run. The unsafe file must never
exist, not merely never be committed.

Fixtures are committed under frontend/src/fixtures/, unlike data/, which is
gitignored wholesale. Regeneration needs live NCBI and local NER and is a
build-time step run deliberately, never at view time.

main() gets no direct unit test per project convention; project_run and
source_pin are tested in their own modules, and the next commit adds a freshness
check over the committed output.
EOF
```

---

### Task 6: The freshness check

**Files:**
- Create: `backend/tests/evals/test_fixture_freshness.py`

- [ ] **Step 1: Write the test**

```python
import json
import pathlib

import pytest

from biolit_evals.fixture_export import FEATURED
from biolit_evals.fixture_models import SCHEMA_VERSION, FixtureRun
from biolit_evals.fixture_pin import source_pin

FIXTURES = pathlib.Path(__file__).resolve().parents[3] / "frontend" / "src" / "fixtures"


@pytest.mark.parametrize("slug", sorted(FEATURED))
def test_every_committed_fixture_matches_the_current_pin(slug: str):
    """⭐ THIS IS THE CHECK THAT MAKES A STALE CLAIM HARD TO SHIP.

    A fixture saying "17 kept" after the ranker changed is a false claim on a public page. The
    pin covers the seven modules whose behaviour determines a displayed value, so a change to
    any of them turns this red until the fixtures are regenerated.

    ⚠️ That is the intended cost, stated so nobody is surprised by it: a behaviour change and a
    fixture refresh are one unit of work, and the refresh needs live NCBI. The TEST stays
    hermetic -- it only compares hashes -- so the project's "unit tests never touch the
    network" rule is intact; only the remedy needs a connection.
    """
    path = FIXTURES / f"{slug}.json"
    assert path.exists(), f"missing fixture {path}; run `uv run python -m biolit_evals.fixture_export`"

    run = FixtureRun.model_validate_json(path.read_text(encoding="utf-8"))

    assert run.schema_version == SCHEMA_VERSION
    assert run.source_pin == source_pin(), (
        f"{slug}.json is STALE: it was generated against different behaviour in a pinned "
        f"module. Regenerate with:\n"
        f"    uv run python -m biolit_evals.fixture_export --slug {slug}"
    )
```

- [ ] **Step 2: Run and confirm PASS**

Run: `uv run pytest tests/evals/test_fixture_freshness.py -v`
Expected: 4 passed.

- [ ] **Step 3: Prove the check bites**

Make a *behavioural* edit to a pinned module — e.g. in `biolit/synth/template.py`, change a literal string constant — then run the test.

Run: `uv run pytest tests/evals/test_fixture_freshness.py -q`
Expected: **4 FAILED**, each naming the regeneration command.

Then revert the edit and confirm 4 passed. **A freshness check that cannot fail is theatre; this step is how you know it works.**

- [ ] **Step 4: Prove it does NOT bite on prose**

Add a comment line to `biolit/query/ranking.py`, run the test, confirm **4 passed**, then revert.

This is the half that makes the check survivable in a repo that edits comments constantly.

- [ ] **Step 5: Full CI, then commit**

```bash
uv run ruff check . && uv run ruff format --check . && uv run pyright && uv run pytest -q
echo "cpu-pin: $(grep -ciE '^name = \"(nvidia|triton)' uv.lock)"
```

```bash
git add tests/evals/test_fixture_freshness.py
git commit -F - <<'EOF'
test(evals): committed fixtures must match the current pin

A fixture claiming "17 kept" after the ranker changed is a false claim on a public
page, so the check is structural rather than remembered: a behaviour change in any
of the seven pinned modules turns this red until fixtures are regenerated, and the
failure message carries the exact regeneration command.

Verified in both directions rather than merely observed passing. A behavioural
edit to a pinned module fails all four; a comment-only edit fails none. The second
half is what makes the check survivable in a repo that edits comments constantly
-- a check that cries wolf is a check that gets suppressed.

The test stays hermetic: it compares hashes and reads committed JSON. Only the
remedy needs a network connection.
EOF
```

---

## Self-Review

**Spec coverage.** §2 four runs → Task 5's `FEATURED`. §3 regeneration → Task 5. §4 schema and `abstract`-absent → Task 2; the invariant → Task 4 Step 1; attribution → Task 4 Step 5. §5 generator → Task 5; in-process requirement → Task 5's module docstring and commit message; pin and its seven modules → Task 3; the pin's consequence → Task 6's docstring. §6 tests 1–6 → Tasks 2, 3, 4, 6. §7 site structure and §8 stack → **deliberately out of scope; they belong to the Astro plan**, which is sequenced after this one because it consumes the fixture shape this plan commits.

**Placeholder scan.** No TBD/TODO. Every code step carries real code. The one flagged uncertainty — the stage-function signatures in Task 5 — is called out explicitly with instructions to verify against `__main__.py` rather than left as a silent guess.

**Type consistency.** `project_run` is called with identical keyword arguments in Tasks 4 and 5. `FixtureCluster` uses flat `matched`/`proximity` fields rather than the spec's nested `score: {matched, proximity}` sketch — **a deliberate simplification**, since a two-field object buys nothing over two fields and pydantic models nest awkwardly in JSON consumed by TypeScript. `relevance_score` returns `tuple[int, float]` in Task 1 and is unpacked as such in Task 4. `PINNED_MODULES` is a `tuple[str, ...]` in Task 3 and monkeypatched as a tuple in its own test.

**One deviation from the spec to note at review time:** the spec's schema sketch shows `score: {matched, proximity}`; this plan flattens it. Say so if you want the nested shape instead — it changes Task 2's model and Task 4's construction, nothing else.
