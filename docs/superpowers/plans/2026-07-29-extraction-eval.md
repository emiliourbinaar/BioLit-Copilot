# Phase 4 Extraction Eval Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Measure whether an LLM extractor of `key_findings` earns its place over a free deterministic control, on BC5CDR Test-500.

**Architecture:** An `Extractor` Protocol (mirroring `Linker` and `PairingStrategy`) with two implementations — a deterministic co-occurrence control and `claude-opus-5`. The model returns **sentence indices**, never character offsets; the harness converts indices to spans via `sentence_spans`, which makes a hallucinated span structurally impossible rather than merely detected. The eval scores three arms through one code path and decomposes the control's false negatives into three actionable buckets.

**Tech Stack:** Python 3.12, uv, pydantic v2, `anthropic` SDK (new dependency), pytest, ruff, pyright.

**Spec:** `docs/superpowers/specs/2026-07-29-extraction-eval-design.md` — read it before Task 1.

## Global Constraints

- All commands run from `backend/`. Gate = `uv run ruff check .` + `uv run ruff format --check .` + `uv run pyright` + `uv run pytest`.
- Ruff ruleset `E,F,I,UP,B`, line length 100. Imports at top of file only (E402) except deliberate function-local heavy imports inside `main()`.
- String enums use `enum.StrEnum` (ADR-0005). Timestamps use `datetime.now(UTC)`, never `timezone.utc`.
- **CPU-pin guard:** `grep -ciE '^name = "(nvidia|triton)' uv.lock` must return **0**. Re-check after any `uv add`.
- Never fabricate gold MeSH IDs, PMIDs, or abstract text. Use only ids/pmids that appear in this plan or in the real corpus.
- Never modify `pyproject.toml` for tdd-guard config; never set `tdd_guard_project_root` there; never hand-edit anything under `.claude/tdd-guard/`.
- Do not commit `backend/data/` or downloads (gitignored). Run logs **are** committed.
- `main()` gets no direct unit test — precedent: `end_to_end.main()`, `ner_eval.main()`, `cluster_eval.main()`.
- Real-model tests go behind the existing `heavy` marker (`pyproject.toml:50`; `addopts = "-m 'not heavy'"`).
- **Commit code before running any eval.** `git_sha()` records HEAD and ignores a dirty tree, so a run log line can otherwise name a sha that does not contain the code that produced it.
- Determinism fixtures use **7 reverse-inserted elements, not 2**. On the clustering branch a 2-element fixture let a dropped `sorted()` escape on 4 of 12 `PYTHONHASHSEED` values.
- **Gate discipline:** if an anchor fires, the harness is wrong. Report the mismatch; do not adjust an anchor to match an observation; do not commit the run as a result.
- Use `git commit -F -` with a heredoc. Never PowerShell here-strings in the Bash tool. Do not push.

## Verified interfaces this plan builds on

Copied from source so no task has to guess:

```python
# biolit/domain/records.py
class Entity(BaseModel):   # text, label, start, end, canonical_id, canonical_name
class ExtractedRecord(BaseModel):
    paper_id: str
    entities: list[Entity] = Field(default_factory=list)
    study_type: str | None = None
    sample_size: int | None = None
    key_findings: list[str] = Field(default_factory=list)   # ← Task 1 changes this

# biolit/domain/paper.py — Paper has: id, source, pmid, title, abstract: str | None,
#   text_type, license_tier, extraction_allowed: bool, ...

# biolit/ner/windowing.py
def sentence_spans(text: str) -> list[tuple[int, int]]: ...
# NOTE: spans do NOT cover text exactly — inter-sentence whitespace is in no span.

# biolit/cluster/pairing.py
def linked_ids(entities: Sequence[Entity], label: EntityLabel) -> set[str]: ...
def sentence_index(spans: Sequence[tuple[int, int]], position: int) -> int | None: ...

# biolit_evals/end_to_end.py
@dataclass(frozen=True)
class ConceptMetrics:  # tp, fp, fn, precision, recall, f1
def metrics_from_counts(tp: int, fp: int, fn: int) -> ConceptMetrics: ...

# biolit_evals/mesh_gold.py
@dataclass(frozen=True)
class GoldMention:  # pmid, start, end, text, label: EntityLabel, mesh_ids: tuple[str, ...]
@dataclass(frozen=True)
class GoldDocument:  # pmid, text, mentions: list[GoldMention]

# biolit_evals/mesh_gold_download.py
TEST_MEMBER: str
def load_bc5cdr_documents(zip_url: str, member: str = TEST_MEMBER) -> list[GoldDocument]: ...
def load_bc5cdr_cid_relations(zip_url: str, member: str = TEST_MEMBER) -> dict[str, set[tuple[str, str]]]: ...

# biolit_evals/cluster_eval.py — reuse unchanged
def assert_dataset_size(dataset: str, n_documents: int) -> None: ...
```

## Two refinements to the spec, surfaced while planning

Both are implementation consequences, not changes of intent. **Raise them with the controller before Task 1 if anything here reads as a scope change.**

1. **The licence gate cannot live in `LlmExtractor` alone.** `Extractor.findings()` returns `list[Finding]`; it does not create the `ExtractedRecord` it would need to suppress. So the enforcement point is a record-assembly function, `build_record()` (Task 2), which returns `ExtractedRecord | None`. `LlmExtractor.findings()` *also* returns `[]` for a non-extractable paper, so neither layer alone is load-bearing — fail-closed twice.
2. **The gold-sentence count anchor is a regression pin, not an independent validation.** Unlike gold cluster counts (500→80, 1500→325), which come from independently published corpus statistics, no published statistic exists for gold *sentences* — the value is established by this project. Calling it a loader-correctness check would overstate it: it catches a *change* in gold construction, not an *error* in it. Task 4 labels it accordingly and adds a genuinely independent invariant alongside it (every gold sentence traces to ≥1 gold CID relation, so the count of relations with a gold sentence must be ≤ 1066 on Test-500 — a figure already verified on this corpus).

## File structure

| file | responsibility |
|---|---|
| `src/biolit/domain/records.py` (modify) | `Finding`; `ExtractedRecord.key_findings: list[Finding]` |
| `src/biolit/extract/__init__.py` (create) | package marker |
| `src/biolit/extract/base.py` (create) | `Extractor` Protocol, `findings_from_sentence_indices`, `build_record` |
| `src/biolit/extract/deterministic.py` (create) | `SameSentenceAsEntitiesExtractor` |
| `src/biolit/extract/llm.py` (create) | `LlmExtractor` |
| `src/biolit_evals/extract_eval.py` (create) | gold sentences, metrics, buckets, anchors, `run_extract_eval`, `main` |
| `tests/extract/test_base.py` (create) | index→Finding conversion, `build_record` licence gate |
| `tests/extract/test_deterministic.py` (create) | the control |
| `tests/extract/test_llm.py` (create) | stub-client LLM extractor |
| `tests/evals/test_extract_eval.py` (create) | gold, metrics, buckets, anchors, runner |

---

### Task 1: `Finding` and the `ExtractedRecord` field change

**Files:**
- Modify: `src/biolit/domain/records.py`
- Test: `tests/domain/test_records.py`

**Interfaces:**
- Consumes: nothing.
- Produces: `Finding(text: str, start: int, end: int, sentence_index: int)`; `ExtractedRecord.key_findings: list[Finding]`.

- [ ] **Step 1: Write the failing test**

Append to `tests/domain/test_records.py` (create the file if absent, importing `pytest`, `ValidationError` from `pydantic`, and `ExtractedRecord`, `Finding` from `biolit.domain.records`):

```python
def test_key_findings_hold_findings_with_offsets_not_bare_strings():
    # THE DISCRIMINATING TEST for the type change. A bare string used to be valid; it must
    # not be now, or a caller could silently keep the old shape and lose the offsets a
    # Citation needs to point at a location.
    finding = Finding(text="Two patients developed acidosis.", start=57, end=89, sentence_index=2)
    record = ExtractedRecord(paper_id="1", key_findings=[finding])
    assert record.key_findings[0].sentence_index == 2
    assert record.key_findings[0].start == 57
    with pytest.raises(ValidationError):
        ExtractedRecord(paper_id="1", key_findings=["Two patients developed acidosis."])
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/domain/test_records.py::test_key_findings_hold_findings_with_offsets_not_bare_strings -v`
Expected: FAIL with `ImportError: cannot import name 'Finding'`.

- [ ] **Step 3: Write minimal implementation**

In `src/biolit/domain/records.py`, add above `ExtractedRecord`:

```python
class Finding(BaseModel):
    """One finding-bearing sentence, located in the source abstract.

    Offsets are carried so a Citation can point at a location and a repeated sentence is
    unambiguous. `sentence_index` is retained because it is the Extractor's actual output --
    keeping it makes a run log auditable against the prompt without re-deriving the split.
    """

    text: str
    start: int
    end: int
    sentence_index: int
```

Then change `ExtractedRecord.key_findings`:

```python
    key_findings: list[Finding] = Field(default_factory=list)
```

- [ ] **Step 4: Run the full suite**

Run: `uv run pytest -q`
Expected: PASS. If anything else set `key_findings` to strings it fails here — fix those call sites in this task.

- [ ] **Step 5: Commit**

```bash
git add src/biolit/domain/records.py tests/domain/test_records.py
git commit -F - <<'EOF'
feat(domain): key_findings carry offsets, not bare strings

A Finding records text plus start/end plus the sentence index that produced it.
Offsets are what let a Citation point at a location; without them a repeated
sentence is ambiguous. sentence_index is kept because it is the Extractor's
actual output, which makes a run log auditable against the prompt.
EOF
```

---

### Task 2: `Extractor` Protocol, index→`Finding` conversion, and the licence gate

**Files:**
- Create: `src/biolit/extract/__init__.py`, `src/biolit/extract/base.py`
- Create: `tests/extract/__init__.py`, `tests/extract/test_base.py`

**Interfaces:**
- Consumes: `Finding` (Task 1); `sentence_spans`.
- Produces:
  - `class Extractor(Protocol): def findings(self, paper: Paper) -> list[Finding]: ...`
  - `findings_from_sentence_indices(text: str, indices: Iterable[int]) -> list[Finding]`
  - `build_record(paper: Paper, *, entities: Sequence[Entity], extractor: Extractor) -> ExtractedRecord | None`

- [ ] **Step 1: Write the failing test — out-of-range indices are dropped**

`tests/extract/test_base.py`:

```python
from biolit.domain.enums import EntityLabel
from biolit.domain.paper import Paper
from biolit.domain.records import Entity, Finding
from biolit.extract.base import build_record, findings_from_sentence_indices


def test_out_of_range_indices_are_dropped_not_clamped_or_guessed():
    # Structured outputs cannot express numerical bounds, so the schema guarantees integers
    # but not that they are in range. A hallucinated index must yield NO finding -- clamping
    # to the last sentence would invent a citation the model never chose.
    text = "Metformin was given. Acidosis followed."
    out = findings_from_sentence_indices(text, [1, 7, -1])
    assert [f.sentence_index for f in out] == [1]
    assert out[0].text == "Acidosis followed."
    assert text[out[0].start : out[0].end] == out[0].text
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/extract/test_base.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'biolit.extract'`.

- [ ] **Step 3: Write minimal implementation**

`src/biolit/extract/__init__.py` — empty file.

`src/biolit/extract/base.py`:

```python
from collections.abc import Iterable, Sequence
from typing import Protocol

from biolit.domain.paper import Paper
from biolit.domain.records import Entity, ExtractedRecord, Finding
from biolit.ner.windowing import sentence_spans


class Extractor(Protocol):
    """Turns one paper into its finding-bearing sentences.

    Injected keyword-only into `build_record`, mirroring the `Linker` and `PairingStrategy`
    seams: a different extractor later becomes a constructor argument, not a rewrite.
    """

    def findings(self, paper: Paper) -> list[Finding]: ...


def findings_from_sentence_indices(text: str, indices: Iterable[int]) -> list[Finding]:
    """Convert sentence indices into located Findings, DROPPING out-of-range indices.

    This is the whole reason extractors return indices rather than character offsets: an
    index either addresses a sentence or it does not, so a fabricated span is structurally
    impossible rather than merely detected. Out-of-range indices are dropped -- never
    clamped, never guessed -- consistent with every other fail-closed decision here.
    """
    spans = sentence_spans(text)
    out: list[Finding] = []
    for index in sorted(set(indices)):
        if not 0 <= index < len(spans):
            continue
        start, end = spans[index]
        out.append(Finding(text=text[start:end], start=start, end=end, sentence_index=index))
    return out


def build_record(
    paper: Paper, *, entities: Sequence[Entity], extractor: Extractor
) -> ExtractedRecord | None:
    """Assemble one paper's record, or None when its licence forbids extraction.

    THE SINGLE LICENCE ENFORCEMENT POINT, and it suppresses the WHOLE record rather than
    just the findings. ExtractedRecord is not text-free: Entity.text and Finding.text both
    carry verbatim abstract substrings, so emitting entities for a non-extractable paper
    would leak its text downstream even with zero findings.

    Sufficient as the only gate because `Paper` appears in exactly two contracts --
    RetrieverOutput and ExtractorInput -- so the Extractor is the last node that ever sees
    one. Critic and Synthesis take only ExtractedRecord / Cluster / ContradictionFinding.

    Distinct from a safety refusal, which empties key_findings ONLY: a refusal is one LLM
    call declining and has no bearing on entities from the separate NER/linking stage.
    """
    if not paper.extraction_allowed:
        return None
    return ExtractedRecord(
        paper_id=paper.id, entities=list(entities), key_findings=extractor.findings(paper)
    )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/extract/test_base.py -v`
Expected: PASS.

- [ ] **Step 5: Write the licence-gate test**

Append to `tests/extract/test_base.py`:

```python
class _AlwaysFinds:
    def findings(self, paper):
        return [Finding(text="x", start=0, end=1, sentence_index=0)]


def _paper(*, allowed: bool) -> Paper:
    return Paper(
        id="p1",
        source=Source.pubmed,
        title="t",
        abstract="Metformin was given. Acidosis followed.",
        text_type=TextType.abstract_only,
        extraction_allowed=allowed,
    )


def test_a_paper_whose_licence_forbids_extraction_yields_no_record_at_all():
    # NOT "no findings" -- no RECORD. Entity.text carries verbatim abstract substrings, so a
    # record with entities and zero findings still leaks the text downstream. The entity here
    # is deliberately non-empty: a gate that only emptied key_findings would pass a weaker
    # test and still leak.
    entity = Entity(
        text="Metformin", label=EntityLabel.CHEMICAL, start=0, end=9, canonical_id="MESH:D008687"
    )
    assert build_record(_paper(allowed=False), entities=[entity], extractor=_AlwaysFinds()) is None

    allowed = build_record(_paper(allowed=True), entities=[entity], extractor=_AlwaysFinds())
    assert allowed is not None
    assert allowed.entities[0].text == "Metformin"
    assert len(allowed.key_findings) == 1
```

Add to the imports at the top of the test file: `from biolit.domain.enums import Source, TextType`. Both members used here are **verified to exist**: `Source` is `pubmed | biorxiv | medrxiv`, `TextType` is `full_text_available | full_text_unverified | abstract_only`.

- [ ] **Step 6: Run tests to verify they pass**

Run: `uv run pytest tests/extract/test_base.py -v`
Expected: PASS (both tests).

- [ ] **Step 7: Mutation-verify the gate**

Temporarily change `build_record`'s guard body from `return None` to `pass`. Run the tests; the licence test must fail. Revert and confirm `git diff src/biolit/extract/base.py` is empty. Record the exact failure text in the report.

- [ ] **Step 8: Commit**

```bash
git add src/biolit/extract tests/extract
git commit -F - <<'EOF'
feat(extract): Extractor seam, index->Finding conversion, licence gate

findings_from_sentence_indices is the reason extractors return sentence indices
rather than character offsets: an index either addresses a sentence or it does
not, so a fabricated span becomes structurally impossible instead of merely
detected. Out-of-range indices are dropped -- never clamped, never guessed.

build_record is the single licence enforcement point and it suppresses the WHOLE
record, not just the findings. ExtractedRecord is not text-free -- Entity.text
and Finding.text both carry verbatim abstract substrings -- so emitting entities
for a non-extractable paper would leak its text downstream even with zero
findings. It is sufficient as the only gate because Paper appears in exactly two
contracts, making the Extractor the last node that ever sees one.
EOF
```

---

### Task 3: `SameSentenceAsEntitiesExtractor` — the deterministic control

**Files:**
- Create: `src/biolit/extract/deterministic.py`
- Create: `tests/extract/test_deterministic.py`

**Interfaces:**
- Consumes: `findings_from_sentence_indices`, `sentence_index`, `sentence_spans`, `Finding`.
- Produces: `SameSentenceAsEntitiesExtractor(entities_by_paper: Mapping[str, Sequence[Entity]])` satisfying `Extractor`.

- [ ] **Step 1: Write the failing test**

`tests/extract/test_deterministic.py`:

```python
from biolit.domain.enums import EntityLabel, Source, TextType
from biolit.domain.paper import Paper
from biolit.domain.records import Entity
from biolit.extract.deterministic import SameSentenceAsEntitiesExtractor

# "Metformin was given. Acidosis followed. Insulin fell."
#  0                 20 22               40 42
TEXT = "Metformin was given. Acidosis followed. Insulin fell."


def _paper() -> Paper:
    return Paper(
        id="p1",
        source=Source.pubmed,
        title="t",
        abstract=TEXT,
        text_type=TextType.abstract_only,
        extraction_allowed=True,
    )


def _ent(label: EntityLabel, start: int, canonical_id: str | None) -> Entity:
    return Entity(text="x", label=label, start=start, end=start + 1, canonical_id=canonical_id)


def test_only_sentences_holding_both_a_linked_chemical_and_disease_are_selected():
    # Sentence 0 has a chemical only, sentence 1 a disease only, sentence 2 both -> only 2.
    # A selector that returned any sentence with any entity would return all three.
    entities = [
        _ent(EntityLabel.CHEMICAL, 0, "MESH:D008687"),
        _ent(EntityLabel.DISEASE, 21, "MESH:D000138"),
        _ent(EntityLabel.CHEMICAL, 40, "MESH:D007328"),
        _ent(EntityLabel.DISEASE, 48, "MESH:D000138"),
    ]
    extractor = SameSentenceAsEntitiesExtractor({"p1": entities})
    assert [f.sentence_index for f in extractor.findings(_paper())] == [2]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/extract/test_deterministic.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'biolit.extract.deterministic'`.

Before writing the implementation, confirm the fixture's sentence boundaries are what the test assumes:
`uv run python -c "from biolit.ner.windowing import sentence_spans; print(sentence_spans('Metformin was given. Acidosis followed. Insulin fell.'))"`
Adjust the offsets in the test to the real spans if they differ — **do not** adjust the implementation to fit wrong offsets.

- [ ] **Step 3: Write minimal implementation**

`src/biolit/extract/deterministic.py`:

```python
from collections.abc import Mapping, Sequence

from biolit.cluster.pairing import sentence_index
from biolit.domain.enums import EntityLabel
from biolit.domain.paper import Paper
from biolit.domain.records import Entity, Finding
from biolit.extract.base import findings_from_sentence_indices
from biolit.ner.windowing import sentence_spans


class SameSentenceAsEntitiesExtractor:
    """Selects sentences holding both a linked chemical and a linked disease. No model.

    THE CONTROL, and the reason an LLM's score is attributable at all -- the same role the
    character n-gram TF-IDF control played in Phase 3C. Without it a gain cannot be credited
    to the LLM rather than to the task being easy.

    Bounded by NER and linking by construction, which is exactly what makes it the
    entity-conditioned arm: it cannot reach a relation whose endpoint was never extracted.
    """

    def __init__(self, entities_by_paper: Mapping[str, Sequence[Entity]]) -> None:
        self._entities_by_paper = entities_by_paper

    def findings(self, paper: Paper) -> list[Finding]:
        text = paper.abstract or ""
        spans = sentence_spans(text)
        chemicals: set[int] = set()
        diseases: set[int] = set()
        for entity in self._entities_by_paper.get(paper.id, ()):
            # Fails closed on both, matching SameSentencePairing: an unlinked id is
            # unscoreable against gold, and an entity with no offset cannot be placed.
            if entity.canonical_id is None or entity.start is None:
                continue
            index = sentence_index(spans, entity.start)
            if index is None:
                continue
            if entity.label is EntityLabel.CHEMICAL:
                chemicals.add(index)
            elif entity.label is EntityLabel.DISEASE:
                diseases.add(index)
        return findings_from_sentence_indices(text, chemicals & diseases)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/extract/test_deterministic.py -v`
Expected: PASS.

- [ ] **Step 5: Write the fail-closed test**

```python
def test_unlinked_and_unplaceable_entities_are_both_excluded():
    # TWO INDEPENDENT GUARDS -- test both, or reducing the condition to one disjunct passes.
    # This is the disjunction gap that recurred five times on the clustering branch.
    # (a) canonical_id None: sentence 2 has both labels but the chemical never linked.
    # (b) start None: sentence 2's disease has no offset.
    unlinked = [
        _ent(EntityLabel.CHEMICAL, 40, None),
        _ent(EntityLabel.DISEASE, 48, "MESH:D000138"),
    ]
    assert SameSentenceAsEntitiesExtractor({"p1": unlinked}).findings(_paper()) == []

    no_offset = Entity(
        text="x", label=EntityLabel.DISEASE, start=None, end=None, canonical_id="MESH:D000138"
    )
    unplaceable = [_ent(EntityLabel.CHEMICAL, 40, "MESH:D007328"), no_offset]
    assert SameSentenceAsEntitiesExtractor({"p1": unplaceable}).findings(_paper()) == []
```

- [ ] **Step 6: Run tests, then mutation-verify each guard separately**

Run: `uv run pytest tests/extract/test_deterministic.py -v` → PASS.

Then, one at a time: reduce the guard to `if entity.canonical_id is None:` (drops the offset check) and confirm a test fails; revert. Reduce it to `if entity.start is None:` and confirm a test fails; revert. **Both mutations must be caught separately** — if only one is, the test does not pin both disjuncts. Confirm `git diff` clean and record both failure texts.

- [ ] **Step 7: Commit**

```bash
git add src/biolit/extract/deterministic.py tests/extract/test_deterministic.py
git commit -F - <<'EOF'
feat(extract): deterministic same-sentence control

Selects sentences holding both a linked chemical and a linked disease, with no
model. This is what makes an LLM score attributable -- the role Phase 3C's
char-n-gram TF-IDF control played. It is bounded by NER and linking by
construction, which is what makes it the entity-conditioned arm.

Fails closed on canonical_id None and start None independently, with each guard
mutation-verified separately: covering one disjunct and calling the condition
pinned was the single most repeated defect on the clustering branch.
EOF
```

---

### Task 4: Gold finding sentences

**Files:**
- Create: `src/biolit_evals/extract_eval.py`
- Create: `tests/evals/test_extract_eval.py`

**Interfaces:**
- Consumes: `GoldDocument`, `GoldMention`, `sentence_spans`, `sentence_index`.
- Produces: `gold_finding_sentences(documents: Sequence[GoldDocument], relations: Mapping[str, set[tuple[str, str]]]) -> dict[str, set[int]]`

- [ ] **Step 1: Write the failing test**

`tests/evals/test_extract_eval.py`:

```python
from biolit.domain.enums import EntityLabel
from biolit_evals.extract_eval import gold_finding_sentences
from biolit_evals.mesh_gold import GoldDocument, GoldMention

TEXT = "Metformin was given. Acidosis followed metformin use."


def _m(start: int, end: int, label: EntityLabel, mesh_ids: tuple[str, ...]) -> GoldMention:
    return GoldMention(
        pmid="1", start=start, end=end, text=TEXT[start:end], label=label, mesh_ids=mesh_ids
    )


def test_a_gold_sentence_needs_both_endpoints_of_one_relation_in_it():
    # Sentence 0 holds the chemical alone. Sentence 1 holds BOTH endpoints -> gold = {1}.
    # A construction that only required one endpoint would return {0, 1}.
    doc = GoldDocument(
        pmid="1",
        text=TEXT,
        mentions=[
            _m(0, 9, EntityLabel.CHEMICAL, ("MESH:D008687",)),
            _m(21, 29, EntityLabel.DISEASE, ("MESH:D000138",)),
            _m(39, 48, EntityLabel.CHEMICAL, ("MESH:D008687",)),
        ],
    )
    relations = {"1": {("MESH:D008687", "MESH:D000138")}}
    assert gold_finding_sentences([doc], relations) == {"1": {1}}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/evals/test_extract_eval.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'biolit_evals.extract_eval'`.

Verify the fixture's spans first: `uv run python -c "from biolit.ner.windowing import sentence_spans; print(sentence_spans('Metformin was given. Acidosis followed metformin use.'))"` and correct the offsets in the test if needed.

- [ ] **Step 3: Write minimal implementation**

`src/biolit_evals/extract_eval.py`:

```python
from collections.abc import Mapping, Sequence

from biolit.cluster.pairing import sentence_index
from biolit.domain.enums import EntityLabel
from biolit.ner.windowing import sentence_spans
from biolit_evals.mesh_gold import GoldDocument


def gold_finding_sentences(
    documents: Sequence[GoldDocument],
    relations: Mapping[str, set[tuple[str, str]]],
) -> dict[str, set[int]]:
    """Sentences where BOTH endpoints of at least one gold CID relation are gold-annotated.

    PROXY, NOT GROUND TRUTH. BC5CDR annotates CID relations at DOCUMENT level; sentence-level
    co-occurrence is this project's inference about where the relation is asserted. Some
    qualifying sentences state background rather than a finding, and a paper's actual key
    finding may concern efficacy, which CID does not annotate at all. A high score against
    this gold means "selects sentences containing the annotated relation" -- NOT "selects the
    paper's key finding". Cite it that way.
    """
    gold: dict[str, set[int]] = {}
    for document in documents:
        pairs = relations.get(document.pmid)
        if not pairs:
            continue
        spans = sentence_spans(document.text)
        by_label: dict[EntityLabel, dict[int, set[str]]] = {
            EntityLabel.CHEMICAL: {},
            EntityLabel.DISEASE: {},
        }
        for mention in document.mentions:
            index = sentence_index(spans, mention.start)
            if index is None or mention.label not in by_label:
                continue
            by_label[mention.label].setdefault(index, set()).update(mention.mesh_ids)
        chemicals = by_label[EntityLabel.CHEMICAL]
        diseases = by_label[EntityLabel.DISEASE]
        for index in set(chemicals) & set(diseases):
            if any(c in chemicals[index] and d in diseases[index] for c, d in pairs):
                gold.setdefault(document.pmid, set()).add(index)
    return gold
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/evals/test_extract_eval.py -v`
Expected: PASS.

- [ ] **Step 5: Write the cross-pair test**

```python
def test_two_endpoints_present_but_not_of_the_same_relation_is_not_a_gold_sentence():
    # Sentence 1 holds chemical A and disease B, but the only gold relation is (A, C).
    # A construction that checked "any chemical and any disease co-occur" would wrongly
    # return {1} -- that is the cross_product error, one level down.
    doc = GoldDocument(
        pmid="1",
        text=TEXT,
        mentions=[
            _m(0, 9, EntityLabel.CHEMICAL, ("MESH:D008687",)),
            _m(21, 29, EntityLabel.DISEASE, ("MESH:D000138",)),
            _m(39, 48, EntityLabel.CHEMICAL, ("MESH:D008687",)),
        ],
    )
    relations = {"1": {("MESH:D008687", "MESH:D011085")}}
    assert gold_finding_sentences([doc], relations) == {}
```

- [ ] **Step 6: Run tests to verify they pass**

Run: `uv run pytest tests/evals/test_extract_eval.py -v`
Expected: PASS (both).

- [ ] **Step 7: Mutation-verify the relation check**

Change the `any(...)` condition to `True`. The cross-pair test must fail. Revert; confirm `git diff` clean; record the failure text.

- [ ] **Step 8: Commit**

```bash
git add src/biolit_evals/extract_eval.py tests/evals/test_extract_eval.py
git commit -F - <<'EOF'
feat(evals): gold finding sentences from gold mentions and CID relations

A gold sentence holds BOTH endpoints of at least one gold CID relation. Checking
"any chemical and any disease co-occur" instead would be the cross_product error
one level down, so a test pins the cross-pair case explicitly.

The docstring states loudly that this is a PROXY: BC5CDR annotates relations at
document level, so sentence-level co-occurrence is our inference about where the
relation is asserted. A high score means "selects sentences containing the
annotated relation", not "selects the paper's key finding".
EOF
```

---

### Task 5: Sentence-selection metrics and the two anchors

**Files:**
- Modify: `src/biolit_evals/extract_eval.py`
- Test: `tests/evals/test_extract_eval.py`

**Interfaces:**
- Consumes: `metrics_from_counts`, `ConceptMetrics`.
- Produces:
  - `sentence_metrics(pred: Mapping[str, set[int]], gold: Mapping[str, set[int]]) -> ConceptMetrics`
  - `assert_gold_sentence_recall_anchor(metrics: ConceptMetrics, *, arm: str) -> None`
  - `assert_gold_sentence_regression_pin(n_documents: int, n_gold_sentences: int) -> None`
  - `_GOLD_SENTENCE_PINS: dict[int, int]` (populated in Task 8, `{}` until then)

- [ ] **Step 1: Write the failing test — asymmetric pmid sets**

```python
def test_sentence_metrics_cover_papers_present_on_only_one_side():
    # pmid "1" shared -> tp. pmid "2" gold-only (the arm produced nothing for it, e.g. NER
    # found no entities) -> fn. pmid "3" pred-only (selected a sentence in a paper with no
    # gold relation) -> fp. An `&` mutant on `set(pred) | set(gold)` iterates only "1" and
    # silently drops BOTH the whole-document miss and the whole-document hallucination.
    gold = {"1": {0, 1}, "2": {4}}
    pred = {"1": {1, 2}, "3": {0}}
    m = sentence_metrics(pred, gold)
    assert (m.tp, m.fp, m.fn) == (1, 2, 2)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/evals/test_extract_eval.py::test_sentence_metrics_cover_papers_present_on_only_one_side -v`
Expected: FAIL with `ImportError: cannot import name 'sentence_metrics'`.

- [ ] **Step 3: Write minimal implementation**

Add to `extract_eval.py` (and import `ConceptMetrics`, `metrics_from_counts` from `biolit_evals.end_to_end`):

```python
def sentence_metrics(
    pred: Mapping[str, set[int]], gold: Mapping[str, set[int]]
) -> ConceptMetrics:
    """Micro-averaged P/R/F1 over (paper_id, sentence_index) pairs.

    PRECISION IS LOAD-BEARING AND RECALL IS NOT QUOTABLE ALONE: selecting every sentence
    scores recall 1.0. Report `mean_sentences_per_paper` beside every arm.

    Iterates the UNION of pmids so a paper present on only one side still counts -- an
    intersection would drop whole-document misses and whole-document hallucinations from
    both denominators.
    """
    tp = fp = fn = 0
    for pmid in set(pred) | set(gold):
        predicted, expected = pred.get(pmid, set()), gold.get(pmid, set())
        tp += len(predicted & expected)
        fp += len(predicted - expected)
        fn += len(expected - predicted)
    return metrics_from_counts(tp, fp, fn)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/evals/test_extract_eval.py -v`
Expected: PASS.

- [ ] **Step 5: Mutation-verify the union**

Change `set(pred) | set(gold)` to `set(pred) & set(gold)`. The test must fail with tp/fp/fn `(1, 1, 0)` instead of `(1, 2, 2)`. Revert; confirm clean; record the text.

- [ ] **Step 6: Write the anchor tests**

```python
import pytest

from biolit_evals.end_to_end import metrics_from_counts
from biolit_evals.extract_eval import (
    assert_gold_sentence_recall_anchor,
    assert_gold_sentence_regression_pin,
)


def test_the_recall_anchor_catches_even_a_one_in_a_thousand_miss():
    # control-gold recall is 1.0000 BY CONSTRUCTION: every gold sentence holds both gold
    # endpoints, so a co-occurrence selector on gold mentions cannot miss one.
    # The near-miss case (999 tp, 1 fn -> 0.9990) is what discriminates against a loosened
    # tolerance: round(.,4) raises, round(.,2) would not.
    assert_gold_sentence_recall_anchor(metrics_from_counts(10, 5, 0), arm="control-gold")
    with pytest.raises(SystemExit, match="gold-sentence recall"):
        assert_gold_sentence_recall_anchor(metrics_from_counts(999, 0, 1), arm="control-gold")


def test_the_regression_pin_passes_untabulated_sizes_and_catches_a_changed_count():
    # A pin, not a validation: it catches a CHANGE in gold construction, not an ERROR in it.
    # Untabulated sizes must pass so unit fixtures need no entry.
    assert_gold_sentence_regression_pin(3, 99)
    assert_gold_sentence_regression_pin(0, 0)
```

- [ ] **Step 7: Run to verify they fail, then implement**

Run: `uv run pytest tests/evals/test_extract_eval.py -v` → FAIL on the imports.

```python
_GOLD_SENTENCE_PINS: dict[int, int] = {}


def assert_gold_sentence_recall_anchor(metrics: ConceptMetrics, *, arm: str) -> None:
    """HARNESS correctness. A co-occurrence selector running on GOLD mentions must recall
    every gold sentence, because a gold sentence is DEFINED as one holding both gold
    endpoints. A miss means the harness is wrong -- sentence splitting, offset handling, or
    MeSH id prefixing -- not that the selector underperformed. Do NOT relax to match an
    observation.
    """
    if round(metrics.recall, 4) != 1.0:
        raise SystemExit(
            f"{arm}: gold-sentence recall is {metrics.recall:.4f}, expected exactly 1.0000 "
            f"(fn={metrics.fn}). A gold sentence holds both gold endpoints by definition, so "
            "a co-occurrence selector on gold mentions cannot miss one. The harness is wrong."
        )


def assert_gold_sentence_regression_pin(n_documents: int, n_gold_sentences: int) -> None:
    """REGRESSION PIN -- deliberately weaker than the cluster-count anchor it resembles.

    Gold cluster counts (500->80, 1500->325) come from independently published corpus
    statistics. No published statistic exists for gold SENTENCES: the value is established
    by this project, so this catches a CHANGE in gold construction, not an ERROR in it.
    Labelled honestly rather than dressed up as loader validation.
    """
    expected = _GOLD_SENTENCE_PINS.get(n_documents)
    if expected is not None and n_gold_sentences != expected:
        raise SystemExit(
            f"gold-sentence pin: {n_documents} documents yielded {n_gold_sentences} gold "
            f"sentences, pinned at {expected}. Gold construction changed. If the change was "
            "deliberate, update the pin IN THE SAME COMMIT as the change and say why."
        )
```

- [ ] **Step 8: Run tests to verify they pass**

Run: `uv run pytest tests/evals/test_extract_eval.py -v`
Expected: PASS.

- [ ] **Step 9: Mutation-verify the tolerance**

Change `round(metrics.recall, 4)` to `round(metrics.recall, 2)`. The near-miss test must fail with `DID NOT RAISE SystemExit`. Revert; confirm clean; record the text.

- [ ] **Step 10: Commit**

```bash
git add src/biolit_evals/extract_eval.py tests/evals/test_extract_eval.py
git commit -F - <<'EOF'
feat(evals): sentence-selection metrics and two anchors

sentence_metrics iterates the UNION of pmids: an intersection would drop
whole-document misses and hallucinations from both denominators, and the test
pins that asymmetry explicitly because covering one side of a disjunction and
calling it pinned recurred five times on the clustering branch.

The recall anchor is tight at round(.,4) and its test uses the 0.9990 near-miss
case, which is what discriminates against a loosened tolerance -- round(.,2)
would pass a harness dropping 1 gold sentence in 1000.

The count check is named a REGRESSION PIN, not an anchor. Gold cluster counts
came from published corpus statistics; no such statistic exists for gold
sentences, so this value is established by us and catches a CHANGE in gold
construction rather than an ERROR in it. Naming it accurately matters more than
having it sound stronger.
EOF
```

---

### Task 6: The three-bucket false-negative decomposition and its closure anchor

**Files:**
- Modify: `src/biolit_evals/extract_eval.py`
- Test: `tests/evals/test_extract_eval.py`

**Interfaces:**
- Consumes: `linked_ids`, `sentence_index`, `sentence_spans`.
- Produces:
  - `@dataclass(frozen=True) class MissBuckets: endpoint_lost: int; never_co_sentential: int; co_sentential_elsewhere: int; total: int`
  - `classify_misses(documents, relations, gold, pred, *, entities_by_paper) -> MissBuckets`
  - `assert_bucket_closure(buckets: MissBuckets, *, n_false_negatives: int) -> None`

- [ ] **Step 1: Write the failing test — all three buckets, distinct and non-zero**

```python
from biolit_evals.extract_eval import MissBuckets, classify_misses

def test_each_miss_bucket_is_populated_distinctly():
    # THREE papers, one per bucket, with DISTINCT non-zero counts so a swapped or combined
    # implementation fails. Conflating them would make "the ceiling" unactionable, the same
    # defect as Phase 3C's diluted mentions_wrong/fp ratio.
    #   pA -> endpoint_lost: the disease never linked anywhere.
    #   pB -> never_co_sentential: both linked, but in different sentences.
    #   pC -> co_sentential_elsewhere: both co-occur in sentence 0, gold sentence is 1.
    ...  # full fixture written in Step 3 alongside the implementation
```

Write this test concretely using the same `TEXT`/`_m` helpers already in the file, three `GoldDocument`s (`pA`, `pB`, `pC`), a `relations` dict giving each one gold relation, `gold = gold_finding_sentences(...)`, `pred` set to what the control would select, and `entities_by_paper` built from `Entity` objects that realise each bucket's condition. Assert:

```python
    buckets = classify_misses(docs, relations, gold, pred, entities_by_paper=entities)
    assert (buckets.endpoint_lost, buckets.never_co_sentential, buckets.co_sentential_elsewhere) == (1, 2, 3)
```

Choose fixture sizes so the three counts are `1, 2, 3` — distinct and non-zero, so a swap fails. Use additional gold sentences per paper to reach 2 and 3.

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/evals/test_extract_eval.py -v`
Expected: FAIL with `ImportError: cannot import name 'MissBuckets'`.

- [ ] **Step 3: Write minimal implementation**

```python
@dataclass(frozen=True)
class MissBuckets:
    """Why control-real missed each gold sentence. Three buckets, three different fixes.

    endpoint_lost           -- >=1 endpoint has no linked mention anywhere in the paper.
                               UNRECOVERABLE by any window or pairing mechanism. This is the
                               40.3% population from ADR-0013, and the subset on which the
                               LLM arm's recall is the direct proof of bottleneck escape.
    never_co_sentential     -- both endpoints linked somewhere, no sentence holds both.
                               Recoverable by a wider window; prices same-paragraph variants.
    co_sentential_elsewhere -- both linked AND co-sentential, but in a sentence other than
                               the gold one. Gold-vs-real span disagreement; costs a false
                               positive as well as this false negative.

    Evaluated in that order and mutually exclusive, so the three sum to total.
    """

    endpoint_lost: int
    never_co_sentential: int
    co_sentential_elsewhere: int
    total: int


def classify_misses(
    documents: Sequence[GoldDocument],
    relations: Mapping[str, set[tuple[str, str]]],
    gold: Mapping[str, set[int]],
    pred: Mapping[str, set[int]],
    *,
    entities_by_paper: Mapping[str, Sequence[Entity]],
) -> MissBuckets:
    lost = never = elsewhere = 0
    for document in documents:
        missed = gold.get(document.pmid, set()) - pred.get(document.pmid, set())
        if not missed:
            continue
        entities = entities_by_paper.get(document.pmid, ())
        chemicals = linked_ids(entities, EntityLabel.CHEMICAL)
        diseases = linked_ids(entities, EntityLabel.DISEASE)
        spans = sentence_spans(document.text)
        per_sentence: dict[int, tuple[set[str], set[str]]] = {}
        for entity in entities:
            if entity.canonical_id is None or entity.start is None:
                continue
            index = sentence_index(spans, entity.start)
            if index is None:
                continue
            chem, dis = per_sentence.setdefault(index, (set(), set()))
            if entity.label is EntityLabel.CHEMICAL:
                chem.add(entity.canonical_id)
            elif entity.label is EntityLabel.DISEASE:
                dis.add(entity.canonical_id)
        pairs = relations.get(document.pmid, set())
        for index in sorted(missed):
            reachable = [(c, d) for c, d in pairs if c in chemicals and d in diseases]
            if not reachable:
                lost += 1
                continue
            co_sentential = any(
                c in chem and d in dis for chem, dis in per_sentence.values() for c, d in reachable
            )
            if not co_sentential:
                never += 1
            else:
                elsewhere += 1
    return MissBuckets(
        endpoint_lost=lost,
        never_co_sentential=never,
        co_sentential_elsewhere=elsewhere,
        total=lost + never + elsewhere,
    )


def assert_bucket_closure(buckets: MissBuckets, *, n_false_negatives: int) -> None:
    """HARNESS correctness: the three buckets must account for every false negative.

    A cheap identity that catches a misclassified bucket, in the spirit of the three-way
    reachable-share / oracle-recall / cross-product-recall agreement at 0.5966. Nothing in
    the code forces this to hold, so its holding is evidence.
    """
    if buckets.total != n_false_negatives:
        raise SystemExit(
            f"bucket closure: {buckets.total} classified misses "
            f"({buckets.endpoint_lost} lost + {buckets.never_co_sentential} never "
            f"co-sentential + {buckets.co_sentential_elsewhere} elsewhere) != "
            f"{n_false_negatives} false negatives. A miss was misclassified or double-counted."
        )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/evals/test_extract_eval.py -v`
Expected: PASS.

- [ ] **Step 5: Write the closure test**

```python
def test_bucket_closure_raises_when_a_miss_is_unaccounted():
    assert_bucket_closure(MissBuckets(1, 2, 3, 6), n_false_negatives=6)
    with pytest.raises(SystemExit, match="bucket closure"):
        assert_bucket_closure(MissBuckets(1, 2, 3, 6), n_false_negatives=7)
```

- [ ] **Step 6: Run tests, then mutation-verify the ordering**

Run: `uv run pytest tests/evals/test_extract_eval.py -v` → PASS.

Swap the `if not reachable` branch to run *after* the co-sentential check. The three-bucket test must fail (counts shift between buckets). Revert; confirm clean; record the text.

- [ ] **Step 7: Commit**

```bash
git add src/biolit_evals/extract_eval.py tests/evals/test_extract_eval.py
git commit -F - <<'EOF'
feat(evals): decompose control-real's misses into three actionable buckets

endpoint_lost (unrecoverable -- the 40.3% population), never_co_sentential
(prices a wider window), and co_sentential_elsewhere (gold-vs-real span
disagreement, which costs a false positive too). Reporting one conflated ceiling
would make it unactionable, the same defect as Phase 3C's diluted
mentions_wrong/fp ratio.

The fixture gives the three buckets distinct non-zero counts (1/2/3) so a
swapped or combined implementation fails rather than passing on coincidence.
assert_bucket_closure requires them to sum to the false-negative total -- nothing
in the code forces that, so its holding is evidence.
EOF
```

---

### Task 7: `LlmExtractor`

**Files:**
- Modify: `backend/pyproject.toml` (add `anthropic` dependency)
- Create: `src/biolit/extract/llm.py`
- Create: `tests/extract/test_llm.py`

**Interfaces:**
- Consumes: `findings_from_sentence_indices`, `sentence_spans`, `Finding`, `Paper`.
- Produces: `LlmExtractor(client, *, model: str = "claude-opus-5", effort: str = "medium")` satisfying `Extractor`, with public counters `refusals: int`, `out_of_range: int`, `licence_skipped: int`.

- [ ] **Step 1: Add the dependency and re-check the CPU pin**

```bash
uv add anthropic
grep -ciE '^name = "(nvidia|triton)' uv.lock   # MUST print 0
uv run pytest -q                               # MUST still pass
```

If the pin guard prints anything other than 0, stop and report — do not proceed.

- [ ] **Step 2: Write the failing test — happy path with a stub client**

`tests/extract/test_llm.py`:

```python
import json
from types import SimpleNamespace

from biolit.domain.enums import Source, TextType
from biolit.domain.paper import Paper
from biolit.extract.llm import LlmExtractor

TEXT = "Metformin was given. Acidosis followed. Insulin fell."


def _paper(*, allowed: bool = True, abstract: str | None = TEXT) -> Paper:
    return Paper(
        id="p1",
        source=Source.pubmed,
        title="t",
        abstract=abstract,
        text_type=TextType.abstract_only,
        extraction_allowed=allowed,
    )


class _StubClient:
    """Records the request and returns a canned response. Tests never touch the API."""

    def __init__(self, *, text: str = '{"finding_sentences": [1]}', stop_reason: str = "end_turn"):
        self._text, self._stop_reason = text, stop_reason
        self.calls: list[dict] = []
        self.messages = SimpleNamespace(create=self._create)

    def _create(self, **kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(
            stop_reason=self._stop_reason,
            content=[SimpleNamespace(type="text", text=self._text)],
        )


def test_the_prompt_numbers_sentences_and_indices_become_located_findings():
    client = _StubClient()
    findings = LlmExtractor(client).findings(_paper())
    assert [f.sentence_index for f in findings] == [1]
    assert findings[0].text == "Acidosis followed."
    assert TEXT[findings[0].start : findings[0].end] == findings[0].text
    # The model must see numbered sentences -- that is the whole interface.
    sent = json.dumps(client.calls[0]["messages"])
    assert "[0] Metformin was given." in sent
    assert "[1] Acidosis followed." in sent
```

- [ ] **Step 3: Run test to verify it fails**

Run: `uv run pytest tests/extract/test_llm.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'biolit.extract.llm'`.

- [ ] **Step 4: Write minimal implementation**

`src/biolit/extract/llm.py`:

```python
import json
from typing import Any

from biolit.domain.paper import Paper
from biolit.domain.records import Finding
from biolit.extract.base import findings_from_sentence_indices
from biolit.ner.windowing import sentence_spans

_SYSTEM = """You identify which sentences of a biomedical abstract state the study's findings.

A finding is a result the study reports: an observed effect, an adverse event, a measured
outcome. Background, methods, and motivation are not findings.

You will receive the abstract as a numbered list of sentences. Return the indices of the
sentences that state findings. Return an empty list if none do. Do not return an index that
is not in the list you were given."""

_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {"finding_sentences": {"type": "array", "items": {"type": "integer"}}},
    "required": ["finding_sentences"],
    "additionalProperties": False,
}


class LlmExtractor:
    """Selects finding-bearing sentences with an LLM, by INDEX rather than by offset.

    The model never computes a character offset: it sees numbered sentences and returns
    indices, and the harness converts them via sentence_spans. That is what makes a
    fabricated span structurally impossible instead of merely detected.

    Note the schema gap: structured outputs cannot express numerical bounds, so it guarantees
    a list of integers but NOT that they are in range. findings_from_sentence_indices drops
    out-of-range values and `out_of_range` counts them as a reliability diagnostic.
    """

    def __init__(self, client: Any, *, model: str = "claude-opus-5", effort: str = "medium") -> None:
        self._client, self._model, self._effort = client, model, effort
        self.refusals = 0
        self.out_of_range = 0
        self.licence_skipped = 0

    def findings(self, paper: Paper) -> list[Finding]:
        # Redundant with build_record's gate, deliberately: neither layer alone is
        # load-bearing for a licence decision. build_record is the enforcement point --
        # it suppresses the whole record, which this cannot do since it returns findings.
        if not paper.extraction_allowed:
            self.licence_skipped += 1
            return []
        text = paper.abstract or ""
        spans = sentence_spans(text)
        if not spans or not text.strip():
            return []
        numbered = "\n".join(f"[{i}] {text[start:end]}" for i, (start, end) in enumerate(spans))
        response = self._client.messages.create(
            model=self._model,
            max_tokens=1024,
            system=[{"type": "text", "text": _SYSTEM, "cache_control": {"type": "ephemeral"}}],
            output_config={
                "effort": self._effort,
                "format": {"type": "json_schema", "schema": _SCHEMA},
            },
            messages=[{"role": "user", "content": numbered}],
        )
        # Check stop_reason BEFORE reading content: on a refusal, content is empty or partial.
        # A refusal empties key_findings only -- it says nothing about entities, which come
        # from the separate deterministic NER/linking stage.
        if response.stop_reason == "refusal":
            self.refusals += 1
            return []
        payload = next((b.text for b in response.content if b.type == "text"), "")
        indices = json.loads(payload)["finding_sentences"]
        self.out_of_range += sum(1 for i in indices if not 0 <= i < len(spans))
        return findings_from_sentence_indices(text, indices)
```

- [ ] **Step 5: Run test to verify it passes**

Run: `uv run pytest tests/extract/test_llm.py -v`
Expected: PASS.

- [ ] **Step 6: Write the three failure-mode tests**

```python
def test_a_refusal_yields_no_findings_and_is_counted_without_reading_content():
    # stop_reason must be checked BEFORE content: a refusal's content is empty or partial,
    # so an implementation that parsed first would raise or return garbage.
    client = _StubClient(text="", stop_reason="refusal")
    extractor = LlmExtractor(client)
    assert extractor.findings(_paper()) == []
    assert extractor.refusals == 1


def test_an_out_of_range_index_is_dropped_and_counted():
    client = _StubClient(text='{"finding_sentences": [1, 99]}')
    extractor = LlmExtractor(client)
    assert [f.sentence_index for f in extractor.findings(_paper())] == [1]
    assert extractor.out_of_range == 1


def test_a_licence_forbidden_paper_is_never_sent_to_the_api():
    # The strongest form of this assertion: not merely "no findings", but NO CALL AT ALL.
    client = _StubClient()
    extractor = LlmExtractor(client)
    assert extractor.findings(_paper(allowed=False)) == []
    assert client.calls == []
    assert extractor.licence_skipped == 1
```

- [ ] **Step 7: Run all tests, then mutation-verify the refusal ordering**

Run: `uv run pytest tests/extract/test_llm.py -v` → PASS (4 tests).

Move the `stop_reason` check to *after* the `json.loads` line. The refusal test must fail (a `JSONDecodeError` on the empty payload). Revert; confirm clean; record the text.

- [ ] **Step 8: Run every gate**

```bash
uv run pytest -q && uv run ruff check . && uv run ruff format --check . && uv run pyright
grep -ciE '^name = "(nvidia|triton)' uv.lock
```

- [ ] **Step 9: Commit**

```bash
git add backend/pyproject.toml backend/uv.lock src/biolit/extract/llm.py tests/extract/test_llm.py
git commit -F - <<'EOF'
feat(extract): LLM extractor returning sentence indices, not offsets

claude-opus-5 with structured outputs, a prompt-cached system block (Opus 5's
512-token cache minimum means it caches), and adaptive thinking left on by
default. The model sees numbered sentences and returns indices; the harness
converts them. It never computes a character offset, which is what makes a
fabricated span structurally impossible rather than merely detected.

Three failure modes, each with its own test and its own counter:
- Refusal: stop_reason checked BEFORE reading content, since a refusal's content
  is empty or partial. Empties key_findings ONLY -- entities come from the
  separate deterministic NER stage and are unaffected.
- Out-of-range index: dropped and counted. Structured outputs cannot express
  numerical bounds, so the schema guarantees integers but not their range.
- Licence forbidden: asserted to make NO API CALL AT ALL, not merely to return
  nothing. Redundant with build_record's gate on purpose -- neither layer alone
  is load-bearing for a licence decision.

Tests use a stub client and never touch the API. CPU-pin guard still 0 after
adding the anthropic dependency.
EOF
```

---

### Task 8: `run_extract_eval` and `main()`

**Files:**
- Modify: `src/biolit_evals/extract_eval.py`
- Test: `tests/evals/test_extract_eval.py`

**Interfaces:**
- Consumes: everything from Tasks 2–7, plus `assert_dataset_size` from `cluster_eval`, `git_sha` from `_meta`.
- Produces: `run_extract_eval(...) -> dict`, `main(argv: list[str] | None = None) -> None`, `DEFAULT_LOG = "evals/extract_runs.jsonl"`.

- [ ] **Step 1: Write the failing test**

```python
def test_the_run_logs_one_line_per_arm_with_buckets_and_bucket_a_recall(tmp_path):
    # The runner's contract: three scored arms, the bucket decomposition, the bucket-(a)
    # restricted LLM recall, and mean sentences per paper for every arm. Assert on the
    # WRITTEN line, not just the return value -- a field computed but not persisted is not
    # reproducible.
    ...
```

Build the fixture from two `GoldDocument`s, a `relations` dict, an `entities_by_paper` map, and a `_StubClient`-backed `LlmExtractor` (copy the stub into this test module — do not import across test files). Assert:

```python
    line = run_extract_eval(
        documents=docs,
        relations=relations,
        entities_by_paper=entities,
        papers=papers,
        llm_extractor=LlmExtractor(_StubClient()),
        dataset="unit",
        log_path=str(log),
        git_sha="deadbee",
        now="2026-07-29T00:00:00+00:00",
    )
    assert set(line["arms"]) == {"control-gold", "control-real", "llm"}
    assert line["arms"]["control-gold"]["sentence"]["recall"] == 1.0
    assert line["arms"]["control-real"]["miss_buckets"]["total"] == (
        line["arms"]["control-real"]["sentence"]["fn"]
    )
    assert "recall_on_endpoint_lost" in line["arms"]["llm"]
    for arm in line["arms"].values():
        assert "mean_sentences_per_paper" in arm
    written = json.loads(log.read_text(encoding="utf-8").strip())
    assert written["arms"]["llm"]["diagnostics"]["refusals"] == 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/evals/test_extract_eval.py -v`
Expected: FAIL with `ImportError: cannot import name 'run_extract_eval'`.

- [ ] **Step 3: Write minimal implementation**

Add `run_extract_eval` to `extract_eval.py`. It must, in order:

1. `assert_dataset_size(dataset, len(documents))` — before any scoring, so a mis-declared corpus halts.
2. Build `gold = gold_finding_sentences(documents, relations)`.
3. `assert_gold_sentence_regression_pin(len(documents), sum(len(s) for s in gold.values()))`.
4. Build gold-mention-derived entities for `control-gold` (one `Entity` per (mention, mesh_id), `canonical_id=mesh_id`, spans from the mention — the same one-entity-per-(mention,id) decision the clustering plan pinned; a zero-id mention yields one entity with `canonical_id=None`).
5. Score all three arms with `sentence_metrics`, recording per-arm `mean_sentences_per_paper`.
6. `assert_gold_sentence_recall_anchor(control_gold_metrics, arm="control-gold")`.
7. `classify_misses(...)` for `control-real`, then `assert_bucket_closure(buckets, n_false_negatives=control_real_metrics.fn)`.
8. Compute the LLM arm's `recall_on_endpoint_lost`: restrict gold to sentences classified into bucket (a) and score LLM predictions against that restriction only. **This is the bottleneck-escape proof, so it must be its own field, not inferred.**
9. Emit diagnostics per arm (`refusals`, `out_of_range`, `licence_skipped` for the LLM arm; `0` for controls).
10. Write one JSON line and return it.

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/evals/test_extract_eval.py -v`
Expected: PASS.

- [ ] **Step 5: Add `main()`**

`main(argv)` takes `--arm {control,llm,all}` (default `all`) and, with function-local heavy imports, loads Test-500 documents and relations, runs the real NER + linking pipeline to build `entities_by_paper`, constructs `Paper` objects from the gold documents with `extraction_allowed=True`, builds an `anthropic.Anthropic()` client, and calls `run_extract_eval` with `dataset="bc5cdr_test500"`, `log_path=DEFAULT_LOG`, `git_sha=git_sha()`, `now=datetime.now(UTC).isoformat()`. Print each arm's P/R/F1, the bucket split, the bucket-(a) recall, and the diagnostics. **No direct unit test for `main()`** — per the standing precedent.

- [ ] **Step 6: Verify the CLI parses**

Run: `uv run python -m biolit_evals.extract_eval --help`
Expected: usage text, exit 0.

- [ ] **Step 7: Run every gate**

```bash
uv run pytest -q && uv run ruff check . && uv run ruff format --check . && uv run pyright
```

- [ ] **Step 8: Commit**

```bash
git add src/biolit_evals/extract_eval.py tests/evals/test_extract_eval.py
git commit -F - <<'EOF'
feat(evals): extraction eval runner for all three arms

Scores control-gold, control-real, and the LLM arm through one code path, with
assert_dataset_size ahead of any scoring so a mis-declared corpus halts before
producing numbers.

Four gates on the real path: the dataset-size guard, the gold-sentence
regression pin, the control-gold recall anchor at round(.,4), and bucket
closure against control-real's false-negative total.

recall_on_endpoint_lost is its own logged field rather than something a reader
infers. An aggregate comparison cannot rule out the LLM merely being better at
the shared part of the task without ever reaching what control-real
structurally cannot -- so the bottleneck-escape claim needs the restricted
number, the same evidentiary standard as citing same_sentence recall 0.7346 >
oracle 0.6967 instead of arguing structurally.

Every arm logs mean_sentences_per_paper, because selecting every sentence scores
recall 1.0 and no recall figure is quotable without the selection rate beside it.
EOF
```

---

### Task 9: The real run — controller-executed

**Not dispatched to an implementer.** There is no code to write; the deliverable is the measurement, and the anchors need controller judgment. Same call as Phase 3C's final measurement and the clustering eval's Task 9.

- [ ] **Step 1: Confirm the tree is clean and committed**

```bash
git status --porcelain    # MUST be empty
git log --oneline -1
```

`git_sha()` records HEAD and ignores a dirty tree. A run started with uncommitted changes produces a log line naming a sha that does not contain the code that produced it — this cost a repeated multi-minute run on the clustering branch.

- [ ] **Step 2: Run the control arms first (free, no API)**

```bash
uv run python -m biolit_evals.extract_eval --arm control
```

Record: `control-gold` P/R/F1 (**recall must be exactly 1.0000** or the anchor fires), `control-real` P/R/F1, the three bucket counts and their closure, and mean sentences per paper for both.

- [ ] **Step 3: Populate the regression pin**

The first run reports the gold-sentence count for 500 documents. Add it to `_GOLD_SENTENCE_PINS` as `{500: <observed>}`, in a commit whose message states that the value was established by this run and is a pin rather than an independent validation. Re-run the control arms to confirm the pin passes.

- [ ] **Step 4: Cross-check the independent invariant**

Count gold CID relations that have at least one gold sentence. It must be **≤ 1066** (the verified Test-500 relation count). Report the share. Relations with no gold sentence are the "asserted across sentences" population — report that count too; it is a real finding about the proxy, not a defect.

- [ ] **Step 5: Run the LLM arm twice**

```bash
uv run python -m biolit_evals.extract_eval --arm llm
uv run python -m biolit_evals.extract_eval --arm llm
```

Two runs because `temperature` is not accepted on Opus 5 and adaptive thinking is on, so output is not deterministic. Report the delta between runs as a stability diagnostic. Reporting a single LLM number as if reproducible would be dishonest. Expect **under $2 per run**.

- [ ] **Step 6: Report the decision numbers**

Bring back: all three arms' P/R/F1; `control-real`'s bucket split; **the LLM arm's `recall_on_endpoint_lost`** (the bottleneck-escape proof); the LLM's refusal count, out-of-range rate, and run-to-run delta; and mean sentences per paper for every arm. State plainly whether the LLM arm earns its place over the free control, and note that a non-zero refusal rate is an operational finding for a biomedical product, not merely a diagnostic.

- [ ] **Step 7: Commit the run log**

```bash
git add backend/evals/extract_runs.jsonl
git commit -F -   # message: the numbers, the anchors that passed, the two-run delta
```

---

## Self-review

**Spec coverage.** §2 scope → Task 1 (only `key_findings` changes). §3.1 `Finding` → Task 1. §3.2 sentence indices + out-of-range dropping → Tasks 2, 7. §3.3 `Extractor` seam → Tasks 2, 3, 7. §4 both failure modes → Tasks 2 (licence, whole record), 7 (refusal, findings only) with the distinction asserted in both. §5.1 gold → Task 4. §5.2 three arms + `control-gold` fully scored → Task 8. §5.3 precision + selection rate → Tasks 5, 8. §5.4 buckets → Task 6. §5.5 bucket-(a) recall → Task 8 step 3.8. §5.6 diagnostics → Tasks 7, 8. §6 anchors 1–4 → Tasks 5, 6, 8 (anchor 3 reused unchanged). §7 testing → every task, stub client in 7. §8 risks: two-run variance → Task 9 step 5; refusals → Task 7 + Task 9 step 6; contamination and proxy validity are write-up obligations, not code. §9 deliverables → Tasks 8, 9; the `EVAL_REPORT.md` section and ADR follow the run and are out of this plan's scope. §10 non-scope respected — no `study_type`/`sample_size`, no window variants, no LangGraph node.

**Gap found and closed:** the spec's §6 anchor 2 assumed a tabulated gold-sentence count that does not exist yet. Task 5 defines the pin with an empty table so unit fixtures pass, and Task 9 step 3 populates it from the first real run in a commit that says so. Named a regression pin, not an anchor.

**Placeholder scan.** Tasks 6 step 1 and 8 step 1 give assertion targets plus explicit fixture-construction instructions rather than complete literal fixtures, because both need offsets that must be read off the real `sentence_spans` output rather than guessed — each carries a step to verify the real spans first. Every other code step is complete. No "TBD", no "add error handling", no "similar to Task N".

**Type consistency.** `Finding(text, start, end, sentence_index)` identical in Tasks 1, 2, 3, 7. `findings_from_sentence_indices(text, indices)` identical in Tasks 2, 3, 7. `build_record(paper, *, entities, extractor)` identical in Tasks 2, 8. `sentence_metrics(pred, gold)` — pred first, matching `key_metrics`' existing argument order — identical in Tasks 5, 8. `MissBuckets(endpoint_lost, never_co_sentential, co_sentential_elsewhere, total)` identical in Tasks 6, 8. `SameSentenceAsEntitiesExtractor(entities_by_paper)` identical in Tasks 3, 8. `LlmExtractor(client, *, model, effort)` with counters `refusals` / `out_of_range` / `licence_skipped` identical in Tasks 7, 8.
