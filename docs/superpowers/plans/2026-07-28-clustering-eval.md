# Clustering Eval Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Measure what naive chemical–disease pairing costs in precision, and whether a free same-sentence heuristic closes enough of the gap that CID relation extraction is unnecessary.

**Architecture:** A `PairingStrategy` protocol (mirroring the existing `Linker` seam) with two implementations — `CrossProductPairing` and the `SameSentencePairing` control. `cluster_papers` groups papers by `"chemical|disease"` key, keeping only keys held by ≥2 papers. The eval scores two arms (gold entities over all 1500 docs; real pipeline over the held-out test 500) at three metric levels, with paper-pair primary.

**Tech Stack:** Python 3.12, uv, pydantic v2, pytest. No new dependencies — sentence splitting reuses the existing regex in `biolit.ner.windowing`.

**Spec:** `docs/superpowers/specs/2026-07-28-clustering-eval-design.md`

## Global Constraints

- All commands run from `backend/` via `uv run`.
- ruff ruleset `E,F,I,UP,B`, line length 100. Imports at top of file (`E402`) — never mid-file.
- `enum.StrEnum` for string enums (ADR-0005). `datetime.now(UTC)`, never `timezone.utc`.
- Gate = `uv run ruff check .` + `uv run ruff format --check .` + `uv run pyright` + `uv run pytest`, all green before any task is done.
- CPU pin guard: `grep -ciE '^name = "(nvidia|triton)' uv.lock` MUST return 0.
- **No new dependencies.**
- Never fabricate gold MeSH ids, PMIDs, or abstract text. Fixtures use real corpus-shaped data; the real CID lines cited in Task 1 were observed in `CDR_Data.zip` and may be used verbatim.
- Do not commit `backend/data/` or downloads (gitignored). Run logs (`evals/*.jsonl`) ARE committed.
- tdd-guard hook is ACTIVE: one test at a time, RED before GREEN. **Never manufacture a fake RED by deliberately writing wrong behavior.** If a test cannot fail honestly, stop and say so.
- Never modify `pyproject.toml` for tdd-guard config. Never hand-edit `.claude/tdd-guard/`.
- Commit with conventional-commit subjects. Use `git commit -F -` with a heredoc for multi-line messages — do NOT use PowerShell here-strings in the Bash tool.
- Do not push.

## File Structure

**Create:**
- `src/biolit/cluster/__init__.py` — package marker.
- `src/biolit/cluster/pairing.py` — `PairingStrategy` protocol, `CrossProductPairing`, `SameSentencePairing`. One responsibility: turn one paper's entities into `(chemical_id, disease_id)` pairs.
- `src/biolit/cluster/group.py` — `cluster_papers`, `PairingDiagnostics`, `pairing_diagnostics`. One responsibility: turn many papers' pairs into `Cluster`s.
- `src/biolit_evals/cluster_eval.py` — metrics, anchors, workload, runner, `main()`.
- `tests/cluster/__init__.py`, `tests/cluster/test_pairing.py`, `tests/cluster/test_group.py`
- `tests/evals/test_cluster_eval.py`

**Modify:**
- `src/biolit/ner/windowing.py` — promote `_sentence_spans` → `sentence_spans` (second consumer).
- `src/biolit_evals/mesh_gold.py` — add `parse_pubtator_cid`.
- `src/biolit_evals/mesh_gold_download.py` — add `load_bc5cdr_cid_relations`, export the three member paths.

---

### Task 1: CID relation loader

**Files:**
- Modify: `src/biolit_evals/mesh_gold.py`
- Modify: `src/biolit_evals/mesh_gold_download.py`
- Test: `tests/evals/test_datasets.py`

**Interfaces:**
- Consumes: `reconcile_mesh_id(raw: str) -> tuple[str, ...]` (already exists in `mesh_gold.py`) and `parse_pubtator`.
- Produces: `parse_pubtator_cid(text: str) -> dict[str, set[tuple[str, str]]]`; `load_bc5cdr_cid_relations(zip_url: str, member: str = TEST_MEMBER) -> dict[str, set[tuple[str, str]]]`; module constants `TRAINING_MEMBER`, `DEVELOPMENT_MEMBER`, `TEST_MEMBER`.

- [ ] **Step 1: Write the failing test**

Add to `tests/evals/test_datasets.py`:

```python
def test_cid_lines_parse_into_per_document_chemical_disease_pairs():
    # Real observed CDR_TestSet lines. Bare ids on the wire; reconcile_mesh_id prefixes
    # them so a CID pair and a gold mention id are directly comparable.
    text = "\n".join(
        [
            "8701013\tCID\tD015738\tD003693",
            "22836123\tCID\tD016572\tD057049",
            "22836123\tCID\tD000305\tD012595",
        ]
    )
    assert parse_pubtator_cid(text) == {
        "8701013": {("MESH:D015738", "MESH:D003693")},
        "22836123": {
            ("MESH:D016572", "MESH:D057049"),
            ("MESH:D000305", "MESH:D012595"),
        },
    }
```

Add `parse_pubtator_cid` to the existing `from biolit_evals.mesh_gold import ...` block at the top of the file.

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/evals/test_datasets.py::test_cid_lines_parse_into_per_document_chemical_disease_pairs -v`
Expected: FAIL with `ImportError: cannot import name 'parse_pubtator_cid'`.

If the import error prevents collection, add an empty stub `def parse_pubtator_cid(text): ...` returning `{}` so the failure is an assertion, not a collection error — per the tdd-guard convention already used in this repo.

- [ ] **Step 3: Implement `parse_pubtator_cid`**

In `src/biolit_evals/mesh_gold.py`, directly after `parse_pubtator`:

```python
def parse_pubtator_cid(text: str) -> dict[str, set[tuple[str, str]]]:
    """Parse PubTator CID relation lines into per-document (chemical, disease) id pairs.

    A CID line is `PMID<tab>CID<tab>chemicalID<tab>diseaseID` -- exactly four fields, so
    `parse_pubtator` (which requires six) already skips it and the two parsers cannot
    disagree about what a line is. Ids arrive as bare accessions and go through the same
    `reconcile_mesh_id` the mention parser uses, so a relation pair and a gold mention id
    are directly comparable -- which is what makes anchor 1 (key recall == 1.0) meaningful.
    """
    relations: dict[str, set[tuple[str, str]]] = {}
    for line in text.splitlines():
        parts = line.split("\t")
        if len(parts) != 4 or parts[1] != "CID":
            continue
        pmid, _, raw_chemical, raw_disease = parts
        for chemical in reconcile_mesh_id(raw_chemical):
            for disease in reconcile_mesh_id(raw_disease):
                relations.setdefault(pmid, set()).add((chemical, disease))
    return relations
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/evals/test_datasets.py::test_cid_lines_parse_into_per_document_chemical_disease_pairs -v`
Expected: PASS.

- [ ] **Step 5: Write the discriminating test (mutant killer)**

A parser keyed on "has at least 4 fields" would swallow mention lines. Add:

```python
def test_a_mention_line_is_not_read_as_a_relation_and_a_relation_is_not_read_as_a_mention():
    # The two parsers share a file and must not disagree about what a line is. A CID
    # parser keyed on ">= 4 fields" would eat mention lines and silently invent relations.
    mention_line = "8701013\t0\t9\tmetformin\tChemical\tD008687"
    cid_line = "8701013\tCID\tD015738\tD003693"
    assert parse_pubtator_cid(mention_line) == {}
    assert parse_pubtator(cid_line) == []
```

- [ ] **Step 6: Run it and confirm it passes**

Run: `uv run pytest tests/evals/test_datasets.py -v -k "relation"`
Expected: both PASS.

To prove it discriminates, temporarily change `len(parts) != 4` to `len(parts) < 4` and confirm the new test FAILS; then revert. Record the observed failure in the commit message.

- [ ] **Step 7: Add the download-side loader**

In `src/biolit_evals/mesh_gold_download.py`, replace the `_TEST_MEMBER` constant with three public ones and add the loader:

```python
# PubTator files inside CDR_Data.zip (BioCreative V CDR corpus).
TRAINING_MEMBER = "CDR_Data/CDR.Corpus.v010516/CDR_TrainingSet.PubTator.txt"
DEVELOPMENT_MEMBER = "CDR_Data/CDR.Corpus.v010516/CDR_DevelopmentSet.PubTator.txt"
TEST_MEMBER = "CDR_Data/CDR.Corpus.v010516/CDR_TestSet.PubTator.txt"
_TEST_MEMBER = TEST_MEMBER  # back-compat for existing default arguments


def load_bc5cdr_cid_relations(
    zip_url: str, member: str = TEST_MEMBER
) -> dict[str, set[tuple[str, str]]]:
    """Download CDR_Data.zip and parse one split's gold chemical-induced-disease relations.

    Same source and member scheme as `load_bc5cdr_documents`. Heavy/manual (network + ~20 MB).
    """
    resp = httpx.get(zip_url, follow_redirects=True, timeout=300.0)
    resp.raise_for_status()
    with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
        pubtator = zf.read(member).decode("utf-8")
    return parse_pubtator_cid(pubtator)
```

Add `parse_pubtator_cid` to the existing `from biolit_evals.mesh_gold import (...)` block.

- [ ] **Step 8: Gate and commit**

```bash
uv run ruff check . && uv run ruff format --check . && uv run pyright && uv run pytest -q
```

```bash
git add src/biolit_evals/mesh_gold.py src/biolit_evals/mesh_gold_download.py tests/evals/test_datasets.py
git commit -F - <<'EOF'
feat(evals): parse BC5CDR gold CID relations

CDR_Data.zip ships document-level chemical-induced-disease relations that the mention
parser skips (four fields, not six). These are the gold "these papers belong together"
labels the clustering eval scores against: two papers cluster together iff they share a
CID pair.

Ids reuse reconcile_mesh_id, so a relation pair and a gold mention id are directly
comparable -- that comparability is what makes the key-recall anchor meaningful.

The discriminating test pins that the two parsers cannot disagree about what a line is:
under a "> = 4 fields" mutant it fails (a mention line is read as a relation).
EOF
```

---

### Task 2: `PairingStrategy` protocol and `CrossProductPairing`

**Files:**
- Create: `src/biolit/cluster/__init__.py`, `src/biolit/cluster/pairing.py`
- Create: `tests/cluster/__init__.py`, `tests/cluster/test_pairing.py`

**Interfaces:**
- Consumes: `Entity` (`biolit.domain.records`) with fields `text, label, start, end, canonical_id, canonical_name`; `EntityLabel` (`biolit.domain.enums`).
- Produces: `PairingStrategy` protocol with `pairs(entities: Sequence[Entity], text: str) -> set[tuple[str, str]]`; `CrossProductPairing`. Pairs are `(chemical_id, disease_id)` — **chemical always first**, so a pair is unambiguous without carrying labels. Task 4 renders each as `f"{chemical_id}|{disease_id}"`.
- **Entities carry exactly one `canonical_id` each.** Multi-id gold mentions are expanded into multiple `Entity` objects upstream — see the interface decision in Task 8. No pairing code ever sees an id set.

- [ ] **Step 1: Write the failing test**

`tests/cluster/test_pairing.py`:

```python
from biolit.cluster.pairing import CrossProductPairing
from biolit.domain.enums import EntityLabel
from biolit.domain.records import Entity

CHEM = EntityLabel.CHEMICAL
DIS = EntityLabel.DISEASE


def _e(label, cid, start=0):
    return Entity(text="x", label=label, start=start, end=start + 1, canonical_id=cid)


def test_cross_product_pairs_every_chemical_with_every_disease():
    entities = [
        _e(CHEM, "MESH:D008687"),
        _e(CHEM, "MESH:D007328"),
        _e(DIS, "MESH:D011085"),
    ]
    assert CrossProductPairing().pairs(entities, "irrelevant text") == {
        ("MESH:D008687", "MESH:D011085"),
        ("MESH:D007328", "MESH:D011085"),
    }
```

Create `tests/cluster/__init__.py` (empty) and `src/biolit/cluster/__init__.py` (empty).

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/cluster/test_pairing.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'biolit.cluster.pairing'`. If collection is blocked, create `pairing.py` with an empty `class CrossProductPairing: ...` stub so the failure is an assertion.

- [ ] **Step 3: Implement**

`src/biolit/cluster/pairing.py`:

```python
from collections.abc import Sequence
from typing import Protocol

from biolit.domain.enums import EntityLabel
from biolit.domain.records import Entity


class PairingStrategy(Protocol):
    """Turns one paper's entities into (chemical_id, disease_id) pairs.

    Injected keyword-only into `cluster_papers`, mirroring the `Linker` seam: a real CID
    relation extractor later becomes a constructor argument, not a rewrite.
    """

    def pairs(self, entities: Sequence[Entity], text: str) -> set[tuple[str, str]]: ...


def _linked_ids(entities: Sequence[Entity], label: EntityLabel) -> set[str]:
    """Canonical ids of one label. NIL entities are excluded: a `nil:<surface>` endpoint
    is unscoreable against gold CID, so admitting one would inject an unmeasurable
    population into a measurement whose whole purpose is precision. The cost of that
    exclusion is reported by `pairing_diagnostics`, not assumed away."""
    return {e.canonical_id for e in entities if e.label is label and e.canonical_id is not None}


class CrossProductPairing:
    """Every linked chemical paired with every linked disease in the same paper.

    The naive baseline. On gold entities it cannot miss a gold pair -- both endpoints are
    annotated, so the cross-product necessarily contains every gold pair, which is why
    key recall is 1.0 by construction (anchor 1). Its entire error is precision.
    """

    def pairs(self, entities: Sequence[Entity], text: str) -> set[tuple[str, str]]:
        chemicals = _linked_ids(entities, EntityLabel.CHEMICAL)
        diseases = _linked_ids(entities, EntityLabel.DISEASE)
        return {(c, d) for c in chemicals for d in diseases}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/cluster/test_pairing.py -v` → PASS.

- [ ] **Step 5: Add the NIL-exclusion and empty-side tests**

```python
def test_a_nil_entity_forms_no_pairs():
    # NIL endpoints are unscoreable against gold CID; their cost is reported separately.
    entities = [_e(CHEM, None), _e(DIS, "MESH:D011085")]
    assert CrossProductPairing().pairs(entities, "t") == set()


def test_a_paper_with_no_disease_forms_no_pairs():
    entities = [_e(CHEM, "MESH:D008687"), _e(CHEM, "MESH:D007328")]
    assert CrossProductPairing().pairs(entities, "t") == set()
```

- [ ] **Step 6: Run and gate**

```bash
uv run pytest tests/cluster -v
uv run ruff check . && uv run ruff format --check . && uv run pyright
```

- [ ] **Step 7: Commit**

```bash
git add src/biolit/cluster tests/cluster
git commit -F - <<'EOF'
feat(cluster): PairingStrategy protocol and CrossProductPairing

The seam mirrors the Linker protocol: a real CID relation extractor later becomes a
constructor argument rather than a rewrite, which is what let the embedding fallback be
measured without touching canonicalize.

CrossProductPairing is the naive baseline. On gold entities it cannot miss a gold pair --
both endpoints are annotated -- so key recall is 1.0 by construction and the entire error
is precision. NIL entities form no pairs; that exclusion's cost gets measured, not assumed.
EOF
```

---

### Task 3: `SameSentencePairing` and the public `sentence_spans`

**Files:**
- Modify: `src/biolit/ner/windowing.py`
- Modify: `src/biolit/cluster/pairing.py`
- Test: `tests/cluster/test_pairing.py`

**Interfaces:**
- Consumes: `CrossProductPairing` (Task 2), `_linked_ids`.
- Produces: `sentence_spans(text: str) -> list[tuple[int, int]]` (public, in `biolit.ner.windowing`); `SameSentencePairing`.

- [ ] **Step 1: Promote `_sentence_spans` to public**

In `src/biolit/ner/windowing.py`, rename `_sentence_spans` → `sentence_spans` and update every internal call site in that file. Update `tests/ner/` references. Add to its docstring:

```python
def sentence_spans(text: str) -> list[tuple[int, int]]:
    """Split `text` into (start, end) spans at sentence boundaries, covering it exactly.

    Public because it has a second consumer: `biolit.cluster.pairing.SameSentencePairing`.
    The splitter is deliberately simple and mis-splits abbreviations ("e.g. metformin").
    In windowing that is harmless (one-sentence overlap absorbs it); in pairing a mis-split
    can drop a real pair or invent one, which is a measured cost of the heuristic, not a bug
    to hide -- both strategies are reported side by side precisely so it is visible.
    """
```

- [ ] **Step 2: Run the existing suite to confirm the rename broke nothing**

Run: `uv run pytest -q`
Expected: same pass count as before the rename.

- [ ] **Step 3: Write the failing discriminating test (mutant killer)**

This is the test that kills a `SameSentencePairing` which silently returns the cross-product. **Both assertions matter** — the second is what makes the first non-vacuous.

```python
from biolit.cluster.pairing import CrossProductPairing, SameSentencePairing

def test_same_sentence_pairing_differs_from_cross_product_across_a_sentence_boundary():
    # Sentence 1 holds the chemical and one disease; sentence 2 holds another disease.
    # A strategy that ignores sentence boundaries returns BOTH pairs and fails here.
    text = "Metformin caused nausea. Separately, hepatic injury was observed."
    chem_at = text.index("Metformin")
    nausea_at = text.index("nausea")
    injury_at = text.index("hepatic injury")
    entities = [
        Entity(text="Metformin", label=CHEM, start=chem_at, end=chem_at + 9,
               canonical_id="MESH:D008687"),
        Entity(text="nausea", label=DIS, start=nausea_at, end=nausea_at + 6,
               canonical_id="MESH:D009325"),
        Entity(text="hepatic injury", label=DIS, start=injury_at, end=injury_at + 14,
               canonical_id="MESH:D056486"),
    ]
    assert SameSentencePairing().pairs(entities, text) == {
        ("MESH:D008687", "MESH:D009325")
    }
    # The discriminator: cross-product genuinely returns MORE here, so a same-sentence
    # implementation that just delegates to it cannot pass both assertions.
    assert CrossProductPairing().pairs(entities, text) == {
        ("MESH:D008687", "MESH:D009325"),
        ("MESH:D008687", "MESH:D056486"),
    }
```

- [ ] **Step 4: Run it to verify it fails**

Run: `uv run pytest tests/cluster/test_pairing.py -k same_sentence -v`
Expected: FAIL — `SameSentencePairing` does not exist. Add an empty stub only if collection is blocked.

- [ ] **Step 5: Implement**

Append to `src/biolit/cluster/pairing.py` (and add `from biolit.ner.windowing import sentence_spans` to the top import block — **not** mid-file, ruff `E402`):

```python
def _sentence_index(spans: Sequence[tuple[int, int]], position: int) -> int | None:
    for index, (start, end) in enumerate(spans):
        if start <= position < end:
            return index
    return None


class SameSentencePairing:
    """Pair a chemical and a disease only when their spans fall in the same sentence.

    The free deterministic control against CrossProductPairing. Without it, a "CID relation
    extraction is required" conclusion cannot be told apart from "one line of sentence logic
    was missing" -- the same attribution discipline as the TF-IDF control in Phase 3C.

    FAILS CLOSED on an entity with no offsets: it cannot be placed in a sentence, so it
    forms no pairs and `pairing_diagnostics` counts it. Silently treating it as
    "sentence 0" would put it in the same sentence as the document's opening entities and
    invent pairs; silently returning nothing would be indistinguishable from "this paper
    had no pairs", which is the failure mode `check_document_context` exists to prevent.
    """

    def pairs(self, entities: Sequence[Entity], text: str) -> set[tuple[str, str]]:
        spans = sentence_spans(text)
        by_sentence: dict[int, tuple[set[str], set[str]]] = {}
        for entity in entities:
            if entity.canonical_id is None or entity.start is None:
                continue
            index = _sentence_index(spans, entity.start)
            if index is None:
                continue
            chemicals, diseases = by_sentence.setdefault(index, (set(), set()))
            if entity.label is EntityLabel.CHEMICAL:
                chemicals.add(entity.canonical_id)
            elif entity.label is EntityLabel.DISEASE:
                diseases.add(entity.canonical_id)
        out: set[tuple[str, str]] = set()
        for chemicals, diseases in by_sentence.values():
            out |= {(c, d) for c in chemicals for d in diseases}
        return out
```

- [ ] **Step 6: Run to verify it passes**

Run: `uv run pytest tests/cluster/test_pairing.py -v` → PASS.

Then prove the test discriminates: temporarily make `SameSentencePairing.pairs` return `CrossProductPairing().pairs(entities, text)` and confirm the test FAILS. Revert. Record the observed failure in the commit message.

- [ ] **Step 7: Add the dedicated `sentence_spans` and fail-closed tests**

`sentence_spans` now has a second consumer, so this path is owned here rather than inherited from the NER windowing tests.

```python
def test_same_sentence_pairing_uses_sentence_spans_boundaries():
    # Two sentences, one chemical and one disease in each. Only the within-sentence pairs
    # may appear -- this pins the dependency on sentence_spans, not just on "same text".
    text = "Aspirin caused ulcers. Metformin caused acidosis."
    entities = [
        Entity(text="Aspirin", label=CHEM, start=0, end=7, canonical_id="MESH:D001241"),
        Entity(text="ulcers", label=DIS, start=text.index("ulcers"),
               end=text.index("ulcers") + 6, canonical_id="MESH:D014456"),
        Entity(text="Metformin", label=CHEM, start=text.index("Metformin"),
               end=text.index("Metformin") + 9, canonical_id="MESH:D008687"),
        Entity(text="acidosis", label=DIS, start=text.index("acidosis"),
               end=text.index("acidosis") + 8, canonical_id="MESH:D000138"),
    ]
    assert SameSentencePairing().pairs(entities, text) == {
        ("MESH:D001241", "MESH:D014456"),
        ("MESH:D008687", "MESH:D000138"),
    }


def test_an_entity_without_offsets_fails_closed_and_forms_no_pairs():
    # start is None -> unplaceable in any sentence. It must not be silently bucketed into
    # sentence 0, which would invent pairs with the document's opening entities.
    text = "Metformin caused nausea."
    entities = [
        Entity(text="Metformin", label=CHEM, start=0, end=9, canonical_id="MESH:D008687"),
        Entity(text="nausea", label=DIS, start=None, end=None, canonical_id="MESH:D009325"),
    ]
    assert SameSentencePairing().pairs(entities, text) == set()
```

- [ ] **Step 8: Gate and commit**

```bash
uv run ruff check . && uv run ruff format --check . && uv run pyright && uv run pytest -q
```

```bash
git add src/biolit/ner/windowing.py src/biolit/cluster/pairing.py tests/
git commit -F - <<'EOF'
feat(cluster): SameSentencePairing control; promote sentence_spans to public

The free deterministic control against blind cross-product. Without it a "CID relation
extraction is required" conclusion cannot be told apart from "one line of sentence logic
was missing" -- the same attribution discipline as the Phase 3C TF-IDF control.

The discriminating test is the point: sentence 1 holds a chemical and one disease,
sentence 2 another disease, and it asserts BOTH strategies' outputs. An implementation
that delegates to CrossProductPairing fails it (verified by mutating it to do exactly
that). Asserting only the same-sentence side would pass under that mutant.

Fails closed on an entity with no offsets rather than bucketing it into sentence 0, which
would invent pairs with the document's opening entities.

sentence_spans is public because pairing is a second consumer. Its mis-splits are harmless
in windowing (overlap absorbs them) but real in pairing -- a measured cost of the
heuristic, visible because both strategies are always reported side by side.
EOF
```

---

### Task 4: `cluster_papers` and pairing diagnostics

**Files:**
- Create: `src/biolit/cluster/group.py`
- Create: `tests/cluster/test_group.py`

**Interfaces:**
- Consumes: `PairingStrategy` (Task 2); `ExtractedRecord` (`paper_id: str`, `entities: list[Entity]`); `Cluster` (`key: str`, `paper_ids: list[str]`) from `biolit.domain.records`.
- Produces: `cluster_papers(records, *, texts, pairing, min_size=2) -> list[Cluster]`; `PairingDiagnostics`; `pairing_diagnostics(records, *, texts) -> PairingDiagnostics`.

- [ ] **Step 1: Write the failing test**

`tests/cluster/test_group.py`:

```python
from biolit.cluster.group import cluster_papers
from biolit.cluster.pairing import CrossProductPairing
from biolit.domain.enums import EntityLabel
from biolit.domain.records import Entity, ExtractedRecord

CHEM = EntityLabel.CHEMICAL
DIS = EntityLabel.DISEASE


def _rec(pid, pairs):
    entities = []
    for i, (c, d) in enumerate(pairs):
        entities.append(Entity(text="c", label=CHEM, start=i, end=i + 1, canonical_id=c))
        entities.append(Entity(text="d", label=DIS, start=i, end=i + 1, canonical_id=d))
    return ExtractedRecord(paper_id=pid, entities=entities)


def test_a_key_held_by_only_one_paper_is_not_a_cluster():
    # min_size=2: the Critic compares papers WITHIN a cluster, so a singleton key is inert.
    # This is what makes cross-product over-generation survivable downstream.
    records = [
        _rec("A", [("MESH:D008687", "MESH:D011085")]),
        _rec("B", [("MESH:D008687", "MESH:D011085")]),
        _rec("C", [("MESH:D001241", "MESH:D014456")]),
    ]
    clusters = cluster_papers(
        records, texts={"A": "t", "B": "t", "C": "t"}, pairing=CrossProductPairing()
    )
    assert [c.key for c in clusters] == ["MESH:D008687|MESH:D011085"]
    assert clusters[0].paper_ids == ["A", "B"]
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run pytest tests/cluster/test_group.py -v`
Expected: FAIL — module missing. Stub only if collection is blocked.

- [ ] **Step 3: Implement `cluster_papers`**

`src/biolit/cluster/group.py`:

```python
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from biolit.cluster.pairing import PairingStrategy, _sentence_index
from biolit.domain.enums import EntityLabel
from biolit.domain.records import Cluster, ExtractedRecord
from biolit.ner.windowing import sentence_spans


def cluster_papers(
    records: Sequence[ExtractedRecord],
    *,
    texts: Mapping[str, str],
    pairing: PairingStrategy,
    min_size: int = 2,
) -> list[Cluster]:
    """Group papers by "chemical|disease" key, keeping only keys held by >= min_size papers.

    `texts` is passed separately because `ExtractedRecord` carries no text and
    `SameSentencePairing` needs it -- matching `canonicalize(entities, text, *, linker)`,
    which already takes text explicitly, rather than amending the Phase-1 contract.

    Singletons are dropped because the Critic compares papers WITHIN a cluster, so a key
    held by one paper is inert. Output is ordered (clusters by key, paper_ids sorted) so
    runs are reproducible and diffable.
    """
    by_key: dict[str, set[str]] = {}
    for record in records:
        text = texts.get(record.paper_id, "")
        for chemical_id, disease_id in pairing.pairs(record.entities, text):
            by_key.setdefault(f"{chemical_id}|{disease_id}", set()).add(record.paper_id)
    return [
        Cluster(key=key, paper_ids=sorted(paper_ids))
        for key, paper_ids in sorted(by_key.items())
        if len(paper_ids) >= min_size
    ]
```

- [ ] **Step 4: Run to verify it passes**

Run: `uv run pytest tests/cluster/test_group.py -v` → PASS.

- [ ] **Step 5: Write the failing diagnostics test**

```python
from biolit.cluster.group import pairing_diagnostics

def test_nil_diagnostics_report_which_side_was_unlinked():
    # DISEASE has NIL'd at roughly double CHEMICAL's rate throughout canonicalization, so
    # the population this exclusion makes unmeasurable is probably not uniform. A single
    # combined counter would hide that.
    records = [
        ExtractedRecord(
            paper_id="A",
            entities=[
                Entity(text="c", label=CHEM, start=0, end=1, canonical_id=None),
                Entity(text="d", label=DIS, start=2, end=3, canonical_id=None),
                Entity(text="d2", label=DIS, start=4, end=5, canonical_id=None),
                Entity(text="ok", label=CHEM, start=6, end=7, canonical_id="MESH:D008687"),
                Entity(text="np", label=DIS, start=None, end=None, canonical_id="MESH:D011085"),
            ],
        )
    ]
    diagnostics = pairing_diagnostics(records, texts={"A": "Some sentence here."})
    assert diagnostics.nil_chemical_mentions == 1
    assert diagnostics.nil_disease_mentions == 2
    assert diagnostics.unplaceable_entities == 1
    assert diagnostics.n_papers == 1
```

- [ ] **Step 6: Run to verify it fails, then implement**

Run: `uv run pytest tests/cluster/test_group.py -k diagnostics -v` → FAIL.

Append to `group.py`:

```python
@dataclass(frozen=True)
class PairingDiagnostics:
    """The cost of decisions 3 and 4 in the spec, measured rather than assumed.

    NIL counts are split by side because DISEASE has NIL'd at roughly double CHEMICAL's
    rate throughout canonicalization (29.2% vs 14.9% on perfect spans). If that holds here,
    the population excluded from this measurement is not uniform, and a single combined
    counter would hide it.

    Computed independently of the pairing strategy: these are properties of the entity
    data, so the `PairingStrategy` protocol stays single-method.
    """

    n_papers: int
    nil_chemical_mentions: int
    nil_disease_mentions: int
    papers_with_no_linked_chemical: int
    papers_with_no_linked_disease: int
    unplaceable_entities: int


def pairing_diagnostics(
    records: Sequence[ExtractedRecord], *, texts: Mapping[str, str]
) -> PairingDiagnostics:
    nil_chemical = nil_disease = unplaceable = 0
    no_chemical = no_disease = 0
    for record in records:
        spans = sentence_spans(texts.get(record.paper_id, ""))
        linked_chemical = linked_disease = 0
        for entity in record.entities:
            if entity.canonical_id is None:
                if entity.label is EntityLabel.CHEMICAL:
                    nil_chemical += 1
                elif entity.label is EntityLabel.DISEASE:
                    nil_disease += 1
                continue
            if entity.label is EntityLabel.CHEMICAL:
                linked_chemical += 1
            elif entity.label is EntityLabel.DISEASE:
                linked_disease += 1
            if entity.start is None or _sentence_index(spans, entity.start) is None:
                unplaceable += 1
        no_chemical += 1 if linked_chemical == 0 else 0
        no_disease += 1 if linked_disease == 0 else 0
    return PairingDiagnostics(
        n_papers=len(records),
        nil_chemical_mentions=nil_chemical,
        nil_disease_mentions=nil_disease,
        papers_with_no_linked_chemical=no_chemical,
        papers_with_no_linked_disease=no_disease,
        unplaceable_entities=unplaceable,
    )
```

- [ ] **Step 7: Run, gate, commit**

```bash
uv run pytest tests/cluster -v
uv run ruff check . && uv run ruff format --check . && uv run pyright && uv run pytest -q
```

```bash
git add src/biolit/cluster/group.py tests/cluster/test_group.py
git commit -F - <<'EOF'
feat(cluster): cluster_papers with min_size=2 and pairing diagnostics

texts is passed separately because ExtractedRecord carries no text and same-sentence
pairing needs it -- matching canonicalize(entities, text, *, linker) rather than amending
the Phase-1 contract.

Singleton keys are not clusters: the Critic compares papers within a cluster, so a key one
paper holds is inert. That is what makes cross-product over-generation survivable, and it
is why key-level and cluster-level numbers diverge.

NIL counts are split by chemical vs disease side. DISEASE has NIL'd at roughly double
CHEMICAL's rate throughout canonicalization, so the population this exclusion makes
unmeasurable is probably not uniform -- a combined counter would hide that.
EOF
```

---

### Task 5: The three metric levels

**Files:**
- Create: `src/biolit_evals/cluster_eval.py`
- Create: `tests/evals/test_cluster_eval.py`

**Interfaces:**
- Consumes: `Cluster`; `concept_counts(gold_ids: set[str], pred_ids: set[str]) -> tuple[int, int, int]` and `metrics_from_counts(tp, fp, fn) -> ConceptMetrics` from `biolit_evals.end_to_end`. `ConceptMetrics` has `tp, fp, fn, precision, recall, f1`.
- Produces: `key_metrics(pred, gold) -> ConceptMetrics`; `cluster_key_metrics(pred_clusters, gold_clusters) -> ConceptMetrics`; `paper_pair_metrics(pred_clusters, gold_clusters) -> ConceptMetrics`; `paper_pairs(clusters) -> set[tuple[str, str]]`; `gold_clusters_from_relations(relations, min_size=2) -> list[Cluster]`.

- [ ] **Step 1: Write the failing test for `key_metrics`**

`tests/evals/test_cluster_eval.py`:

```python
from biolit.domain.records import Cluster
from biolit_evals.cluster_eval import (
    cluster_key_metrics,
    gold_clusters_from_relations,
    key_metrics,
    paper_pair_metrics,
)


def test_key_metrics_score_predicted_pairs_per_document_against_gold_cid():
    gold = {"1": {("MESH:D008687", "MESH:D011085")}}
    pred = {"1": {("MESH:D008687", "MESH:D011085"), ("MESH:D001241", "MESH:D011085")}}
    m = key_metrics(pred, gold)
    assert (m.tp, m.fp, m.fn) == (1, 1, 0)
    assert m.recall == 1.0
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run pytest tests/evals/test_cluster_eval.py -v` → FAIL (module missing).

- [ ] **Step 3: Implement the key and cluster-key metrics**

`src/biolit_evals/cluster_eval.py`:

```python
from collections.abc import Mapping, Sequence

from biolit.domain.records import Cluster
from biolit_evals.end_to_end import ConceptMetrics, concept_counts, metrics_from_counts


def _key(pair: tuple[str, str]) -> str:
    return f"{pair[0]}|{pair[1]}"


def key_metrics(
    pred: Mapping[str, set[tuple[str, str]]], gold: Mapping[str, set[tuple[str, str]]]
) -> ConceptMetrics:
    """Micro-averaged P/R/F1 over (pmid, key), set-per-document -- DIAGNOSTIC, not primary.

    Reuses concept_counts so the set semantics match every other concept-level number in
    this project.
    """
    tp = fp = fn = 0
    for pmid in set(pred) | set(gold):
        d_tp, d_fp, d_fn = concept_counts(
            {_key(p) for p in gold.get(pmid, set())},
            {_key(p) for p in pred.get(pmid, set())},
        )
        tp, fp, fn = tp + d_tp, fp + d_fp, fn + d_fn
    return metrics_from_counts(tp, fp, fn)


def cluster_key_metrics(
    pred_clusters: Sequence[Cluster], gold_clusters: Sequence[Cluster]
) -> ConceptMetrics:
    """Is a predicted multi-paper cluster's KEY a gold multi-paper cluster key? DIAGNOSTIC.

    Blind to which papers landed in the cluster -- that is exactly what paper_pair_metrics
    measures and why this one cannot stand in for it.
    """
    pred_keys = {c.key for c in pred_clusters}
    gold_keys = {c.key for c in gold_clusters}
    return metrics_from_counts(*concept_counts(gold_keys, pred_keys))


def gold_clusters_from_relations(
    relations: Mapping[str, set[tuple[str, str]]], min_size: int = 2
) -> list[Cluster]:
    """Gold clusters: papers sharing a gold CID pair. Same min_size rule as cluster_papers."""
    by_key: dict[str, set[str]] = {}
    for pmid, pairs in relations.items():
        for pair in pairs:
            by_key.setdefault(_key(pair), set()).add(pmid)
    return [
        Cluster(key=key, paper_ids=sorted(pmids))
        for key, pmids in sorted(by_key.items())
        if len(pmids) >= min_size
    ]
```

- [ ] **Step 4: Run to verify it passes**

Run: `uv run pytest tests/evals/test_cluster_eval.py -v` → PASS.

- [ ] **Step 5: Write the DISCRIMINATING failing test for `paper_pair_metrics`**

**This is the most important test in the plan.** It exists to prevent the primary metric from silently aliasing a diagnostic one. Without it, `paper_pair_metrics` could be `cluster_key_metrics` under another name and every other test would still pass — and the entire recommendation rests on the number it produces.

```python
def test_paper_pair_metrics_are_not_an_alias_of_cluster_key_metrics():
    # Same key, but the prediction recovers only 2 of the 3 papers in the gold cluster.
    # Cluster-key level cannot see the miss (the key matches exactly, F1 == 1.0);
    # paper-pair level does (2 of 3 gold pairs missing, recall 1/3).
    # A paper_pair_metrics that secretly computes cluster-key agreement returns F1 1.0
    # here and fails.
    gold = [Cluster(key="MESH:D008687|MESH:D011085", paper_ids=["A", "B", "C"])]
    pred = [Cluster(key="MESH:D008687|MESH:D011085", paper_ids=["A", "B"])]

    ck = cluster_key_metrics(pred, gold)
    assert ck.f1 == 1.0  # blind to the missing paper

    pp = paper_pair_metrics(pred, gold)
    assert (pp.tp, pp.fp, pp.fn) == (1, 0, 2)
    assert pp.precision == 1.0
    assert pp.recall == 1 / 3
    assert pp.f1 == 0.5
```

- [ ] **Step 6: Run to verify it fails**

Run: `uv run pytest tests/evals/test_cluster_eval.py -k alias -v`
Expected: FAIL — `paper_pair_metrics` does not exist.

- [ ] **Step 7: Implement `paper_pairs` and `paper_pair_metrics`**

```python
def paper_pairs(clusters: Sequence[Cluster]) -> set[tuple[str, str]]:
    """Every unordered paper pair a cluster set implies, as sorted 2-tuples.

    This is the Critic's actual unit of work: ContradictionFinding is
    (paper_id_a, paper_id_b, label, rationale).
    """
    out: set[tuple[str, str]] = set()
    for cluster in clusters:
        ids = sorted(cluster.paper_ids)
        for i in range(len(ids)):
            for j in range(i + 1, len(ids)):
                out.add((ids[i], ids[j]))
    return out


def paper_pair_metrics(
    pred_clusters: Sequence[Cluster], gold_clusters: Sequence[Cluster]
) -> ConceptMetrics:
    """PRIMARY metric: did the right papers end up comparable?

    Primary because ContradictionFinding's unit of work is a pair of papers, so every
    false pair is a wasted or wrong Critic comparison -- and the Critic will be LLM-backed,
    making that direct cost. Deliberately NOT cluster-key agreement: a prediction can match
    a gold key exactly while recovering only part of its paper set, and this level sees
    that where cluster_key_metrics cannot.
    """
    return metrics_from_counts(
        *concept_counts(
            {f"{a}|{b}" for a, b in paper_pairs(gold_clusters)},
            {f"{a}|{b}" for a, b in paper_pairs(pred_clusters)},
        )
    )
```

- [ ] **Step 8: Run to verify it passes, then prove it discriminates**

Run: `uv run pytest tests/evals/test_cluster_eval.py -v` → PASS.

Temporarily replace `paper_pair_metrics`'s body with `return cluster_key_metrics(pred_clusters, gold_clusters)` and confirm the test FAILS (expect `assert 1.0 == 0.5`). Revert. **Record the observed failure text in the commit message** — it is the evidence the guarantee is real.

- [ ] **Step 9: Gate and commit**

```bash
uv run ruff check . && uv run ruff format --check . && uv run pyright && uv run pytest -q
```

```bash
git add src/biolit_evals/cluster_eval.py tests/evals/test_cluster_eval.py
git commit -F - <<'EOF'
feat(evals): three clustering metric levels, paper-pair primary

Paper-pair is primary because ContradictionFinding's unit of work is a pair of papers:
every false pair is a wasted or wrong Critic comparison, and the Critic will be LLM-backed,
so that is direct cost. Key-level and cluster-key-level are diagnostic, not competing
candidates for primary.

The discriminating test exists to prevent the primary metric from silently aliasing a
diagnostic one. This is a stronger guarantee than a regression test: without it,
paper_pair_metrics could BE cluster_key_metrics under another name, every other test would
still pass, and the number the entire build-or-not recommendation rests on would be
measuring something else.

Its fixture is a gold cluster of three papers where the prediction recovers two. The key
matches exactly, so cluster-key F1 is 1.0 and cannot see the miss; paper-pair recall is
1/3 and F1 0.5. Verified by mutating paper_pair_metrics to delegate to cluster_key_metrics:
the test fails with "assert 1.0 == 0.5".
EOF
```

---

### Task 6: Critic workload and top-5 concentration

**Files:**
- Modify: `src/biolit_evals/cluster_eval.py`
- Test: `tests/evals/test_cluster_eval.py`

**Interfaces:**
- Consumes: `Cluster`.
- Produces: `Workload` dataclass (`n_clusters, n_paper_pairs, largest_cluster, top5_pair_share`); `workload(clusters) -> Workload`.

- [ ] **Step 1: Write the failing test**

```python
from biolit_evals.cluster_eval import workload


def test_top5_share_exposes_one_oversized_cluster_dominating_the_critic_budget():
    # One 25-paper cluster is 300 comparisons on its own; five 2-paper clusters are 5.
    # An aggregate pair count cannot show that concentration; this can.
    clusters = [Cluster(key=f"k{i}", paper_ids=[f"p{i}_{j}" for j in range(2)]) for i in range(5)]
    clusters.append(Cluster(key="big", paper_ids=[f"b{j}" for j in range(25)]))
    w = workload(clusters)
    assert w.n_clusters == 6
    assert w.n_paper_pairs == 305
    assert w.largest_cluster == 25
    assert round(w.top5_pair_share, 4) == round(304 / 305, 4)
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run pytest tests/evals/test_cluster_eval.py -k top5 -v` → FAIL.

- [ ] **Step 3: Implement**

Append to `cluster_eval.py` (add `from dataclasses import dataclass` to the top import block):

```python
@dataclass(frozen=True)
class Workload:
    """What clustering hands the Critic. Raw counts, not only aggregates.

    top5_pair_share prices the quadratic risk directly: a cluster of n papers is
    n*(n-1)/2 comparisons, so a single 25-paper cluster is 300 on its own. An aggregate
    pair count cannot show how much of the Critic's budget one oversized cluster burns.
    """

    n_clusters: int
    n_paper_pairs: int
    largest_cluster: int
    top5_pair_share: float


def workload(clusters: Sequence[Cluster]) -> Workload:
    sizes = sorted((len(c.paper_ids) for c in clusters), reverse=True)
    pairs = [n * (n - 1) // 2 for n in sizes]
    total = sum(pairs)
    return Workload(
        n_clusters=len(clusters),
        n_paper_pairs=total,
        largest_cluster=sizes[0] if sizes else 0,
        top5_pair_share=sum(pairs[:5]) / total if total else 0.0,
    )
```

- [ ] **Step 4: Run to verify it passes**

Run: `uv run pytest tests/evals/test_cluster_eval.py -v` → PASS.

- [ ] **Step 5: Gate and commit**

```bash
uv run ruff check . && uv run ruff format --check . && uv run pyright && uv run pytest -q
git add src/biolit_evals/cluster_eval.py tests/evals/test_cluster_eval.py
git commit -F - <<'EOF'
feat(evals): Critic workload counts with top-5 cluster concentration

Follows the Phase 3C mention-level lesson: log the underlying count, not only the
aggregate. A cluster of n papers is n*(n-1)/2 Critic comparisons, so one 25-paper cluster
is 300 by itself -- top5_pair_share prices that concentration directly, which a total
pair count cannot show.
EOF
```

---

### Task 7: The two anchors

**Files:**
- Modify: `src/biolit_evals/cluster_eval.py`
- Test: `tests/evals/test_cluster_eval.py`

**Interfaces:**
- Consumes: `ConceptMetrics`.
- Produces: `assert_key_recall_anchor(metrics: ConceptMetrics, *, arm: str) -> None`; `assert_gold_cluster_anchor(n_documents: int, n_gold_clusters: int) -> None`. Both raise `SystemExit` on mismatch.

- [ ] **Step 1: Write the failing test**

```python
import pytest

from biolit_evals.cluster_eval import assert_gold_cluster_anchor, assert_key_recall_anchor
from biolit_evals.end_to_end import metrics_from_counts


def test_the_key_recall_anchor_raises_when_cross_product_misses_a_gold_pair():
    # Cross-product on gold entities cannot miss a gold pair -- both endpoints are
    # annotated. Recall below 1.0 means the harness is wrong (gold parsing, MeSH id
    # prefixing, label assignment), not that the number is interesting.
    assert_key_recall_anchor(metrics_from_counts(10, 5, 0), arm="A")  # recall 1.0, fine
    with pytest.raises(SystemExit, match="key recall"):
        assert_key_recall_anchor(metrics_from_counts(9, 5, 1), arm="A")


def test_the_gold_cluster_anchor_checks_the_LOADER_not_clustering_quality():
    assert_gold_cluster_anchor(500, 80)
    assert_gold_cluster_anchor(1500, 325)
    with pytest.raises(SystemExit, match="gold cluster"):
        assert_gold_cluster_anchor(500, 79)
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run pytest tests/evals/test_cluster_eval.py -k anchor -v` → FAIL.

- [ ] **Step 3: Implement**

```python
_GOLD_CLUSTER_ANCHORS = {500: 80, 1500: 325}


def assert_key_recall_anchor(metrics: ConceptMetrics, *, arm: str) -> None:
    """HARNESS correctness. Cross-product pairing on GOLD entities must recall every gold
    CID pair, because a gold pair always connects two annotated entities and the
    cross-product of those entities' ids necessarily contains it.

    A deviation means the harness is wrong -- gold parsing, MeSH id prefixing, or label
    assignment -- not that the result is interesting. Do NOT relax this to match an
    observation.
    """
    if round(metrics.recall, 4) != 1.0:
        raise SystemExit(
            f"{arm}: cross-product key recall is {metrics.recall:.4f}, expected exactly "
            f"1.0000 (fn={metrics.fn}). A gold CID pair connects two annotated entities, "
            "so the cross-product cannot miss one. The harness is wrong."
        )


def assert_gold_cluster_anchor(n_documents: int, n_gold_clusters: int) -> None:
    """LOADER correctness -- NOT a clustering-quality check.

    This validates `load_bc5cdr_cid_relations` against independently established gold
    statistics. "The loader reproduces known gold counts" says NOTHING about whether the
    pairing strategy is accurate; do not read a passing anchor as evidence about clustering.
    """
    expected = _GOLD_CLUSTER_ANCHORS.get(n_documents)
    if expected is None:
        return
    if n_gold_clusters != expected:
        raise SystemExit(
            f"gold cluster anchor: {n_documents} documents yielded {n_gold_clusters} "
            f"multi-paper gold clusters, expected {expected}. The CID loader is wrong; "
            "this says nothing about clustering quality either way."
        )
```

- [ ] **Step 4: Run to verify it passes**

Run: `uv run pytest tests/evals/test_cluster_eval.py -v` → PASS.

- [ ] **Step 5: Gate and commit**

```bash
uv run ruff check . && uv run ruff format --check . && uv run pyright && uv run pytest -q
git add src/biolit_evals/cluster_eval.py tests/evals/test_cluster_eval.py
git commit -F - <<'EOF'
feat(evals): key-recall and gold-cluster anchors for the clustering eval

Two anchors that check DIFFERENT things and must not be conflated.

Anchor 1 (harness): cross-product key recall on gold entities must be exactly 1.0000,
true by construction since a gold CID pair connects two annotated entities. It already did
work during design -- CID lines carry bare ids while mentions carry MESH: prefixed ones,
and recall landing on 1.0000 is what proved the prefixing correct. Enforced rather than
remembered.

Anchor 2 (loader): 80 gold clusters over 500 docs, 325 over 1500. This validates the CID
loader against independently established statistics and says NOTHING about whether the
pairing strategy is accurate. The docstring states that explicitly so a future reader
cannot read a passing anchor as evidence about clustering quality.
EOF
```

---

### Task 8: The runner and `main()` — including the Arm-A interface decision

**Files:**
- Modify: `src/biolit_evals/cluster_eval.py`
- Test: `tests/evals/test_cluster_eval.py`

**Interfaces:**
- Consumes: everything from Tasks 1–7; `GoldDocument`/`GoldMention` (`mesh_ids: tuple[str, ...]`); `git_sha()` from `biolit_evals._meta`.
- Produces: `synthesize_records(documents) -> tuple[list[ExtractedRecord], dict[str, str]]`; `ArmResult`; `run_cluster_eval(...) -> dict`; `main(argv=None) -> None`.

> **INTERFACE DECISION — multi-id gold mentions on synthesized Entities.**
> A `GoldMention` carries `mesh_ids: tuple[str, ...]` and may hold more than one id
> (BC5CDR uses `|` for composite mentions). `Entity.canonical_id` is `str | None`.
>
> **Decision: `Entity.canonical_id` stays `str | None` — unchanged — and each
> `(mention, id)` becomes its own `Entity` at the same span.** A 2-id mention becomes two
> `Entity` objects sharing `start`/`end`.
>
> **Why, and what was rejected:**
> - *Rejected — widen `canonical_id` to a set for this path.* It is a Phase-1 domain-model
>   change affecting NER, canonicalization, storage and the Extractor, for one eval's
>   convenience. The production path genuinely produces one id per entity.
> - *Rejected — pack ids into one string and split at pairing time.* `|` is already the
>   cluster-key delimiter, so `"D001|D002"` would be indistinguishable from a key, and it
>   pushes id-set logic into `PairingStrategy`, which both arms share. Arm B would then
>   carry parsing code that can never fire.
> - *Chosen — one Entity per (mention, id).* `PairingStrategy` stays "one entity, one id"
>   in both arms, the id cross-product falls out of the existing pairing logic for free,
>   and no production type changes for a measurement.
>
> **Known consequence, stated not hidden:** duplicate spans at identical offsets inflate
> raw entity counts in `pairing_diagnostics` for Arm A. Same-sentence pairing is unaffected
> (identical offsets are always in the same sentence). Arm B is unaffected entirely.

- [ ] **Step 1: Write the failing test for `synthesize_records`**

```python
from biolit_evals.cluster_eval import synthesize_records
from biolit_evals.mesh_gold import GoldDocument, GoldMention


def test_a_multi_id_gold_mention_becomes_one_entity_per_id():
    # INTERFACE DECISION: Entity.canonical_id stays str|None; each (mention, id) is its own
    # Entity at the same span. Packing ids into one string would collide with "|", the
    # cluster-key delimiter, and push id-set logic into the pairing code both arms share.
    doc = GoldDocument(
        pmid="1",
        text="Metformin and nausea.",
        mentions=[
            GoldMention(pmid="1", start=0, end=9, text="Metformin",
                        label=EntityLabel.CHEMICAL, mesh_ids=("MESH:D008687",)),
            GoldMention(pmid="1", start=14, end=20, text="nausea",
                        label=EntityLabel.DISEASE,
                        mesh_ids=("MESH:D009325", "MESH:D012640")),
        ],
    )
    records, texts = synthesize_records([doc])
    assert len(records) == 1
    ids = [(e.label, e.canonical_id, e.start) for e in records[0].entities]
    assert ids == [
        (EntityLabel.CHEMICAL, "MESH:D008687", 0),
        (EntityLabel.DISEASE, "MESH:D009325", 14),
        (EntityLabel.DISEASE, "MESH:D012640", 14),
    ]
    assert texts == {"1": "Metformin and nausea."}
```

- [ ] **Step 2: Run to verify it fails, then implement**

Run: `uv run pytest tests/evals/test_cluster_eval.py -k multi_id -v` → FAIL.

```python
def synthesize_records(
    documents: Sequence[GoldDocument],
) -> tuple[list[ExtractedRecord], dict[str, str]]:
    """Build ExtractedRecords from gold mentions so BOTH arms run the same production code
    path (`cluster_papers`) instead of a parallel gold-only implementation.

    INTERFACE DECISION -- multi-id gold mentions: `Entity.canonical_id` stays `str | None`
    and each (mention, id) becomes its own Entity at the same span. Widening canonical_id
    to a set would be a Phase-1 domain change for one eval's convenience; packing ids into
    one delimited string would collide with "|", the cluster-key delimiter, and push id-set
    logic into the PairingStrategy both arms share. Consequence, stated not hidden:
    duplicate spans inflate Arm A's raw entity counts in pairing_diagnostics.
    """
    records: list[ExtractedRecord] = []
    texts: dict[str, str] = {}
    for document in documents:
        entities = [
            Entity(
                text=mention.text,
                label=mention.label,
                start=mention.start,
                end=mention.end,
                canonical_id=mesh_id,
            )
            for mention in document.mentions
            for mesh_id in mention.mesh_ids
        ]
        records.append(ExtractedRecord(paper_id=document.pmid, entities=entities))
        texts[document.pmid] = document.text
    return records, texts
```

Add `Entity`, `ExtractedRecord`, `GoldDocument` to the top import block.

- [ ] **Step 3: Run to verify it passes**

Run: `uv run pytest tests/evals/test_cluster_eval.py -v` → PASS.

- [ ] **Step 4: Write the failing test for `run_cluster_eval`**

```python
def test_run_cluster_eval_scores_both_strategies_and_writes_one_log_line(tmp_path):
    # Two documents sharing one gold CID pair -> exactly one gold cluster of size 2.
    docs = [
        GoldDocument(pmid=p, text="Metformin caused nausea.",
                     mentions=[
                         GoldMention(pmid=p, start=0, end=9, text="Metformin",
                                     label=EntityLabel.CHEMICAL, mesh_ids=("MESH:D008687",)),
                         GoldMention(pmid=p, start=17, end=23, text="nausea",
                                     label=EntityLabel.DISEASE, mesh_ids=("MESH:D009325",)),
                     ])
        for p in ("1", "2")
    ]
    relations = {p: {("MESH:D008687", "MESH:D009325")} for p in ("1", "2")}
    log = tmp_path / "cluster_runs.jsonl"
    result = run_cluster_eval(
        documents=docs, relations=relations, arm="A", dataset="unit",
        log_path=str(log), git_sha="deadbee", now="2026-07-28T00:00:00+00:00",
    )
    assert result["n_gold_clusters"] == 1
    for strategy in ("cross_product", "same_sentence"):
        assert result["strategies"][strategy]["paper_pair"]["f1"] == 1.0
    assert len(log.read_text(encoding="utf-8").strip().splitlines()) == 1
```

- [ ] **Step 5: Run to verify it fails, then implement the runner**

```python
def run_cluster_eval(
    *,
    documents: Sequence[GoldDocument] | None = None,
    records: Sequence[ExtractedRecord] | None = None,
    texts: Mapping[str, str] | None = None,
    relations: Mapping[str, set[tuple[str, str]]],
    arm: str,
    dataset: str,
    log_path: str,
    git_sha: str,
    now: str,
) -> dict:
    """Score both pairing strategies for one arm and append one JSON line to `log_path`.

    Pass `documents` for Arm A (records are synthesized from gold mentions) or
    `records`+`texts` for Arm B (produced by the real pipeline). Every impure input is
    injected -- log_path, git_sha, now -- so this stays offline-testable, matching
    run_e2e_eval and run_canon_eval.
    """
    if documents is not None:
        records, texts = synthesize_records(documents)
    assert records is not None and texts is not None

    gold = gold_clusters_from_relations(relations)
    assert_gold_cluster_anchor(len(records), len(gold))

    strategies: dict[str, object] = {}
    for name, strategy in (
        ("cross_product", CrossProductPairing()),
        ("same_sentence", SameSentencePairing()),
    ):
        predicted_pairs = {
            r.paper_id: strategy.pairs(r.entities, texts.get(r.paper_id, "")) for r in records
        }
        clusters = cluster_papers(records, texts=texts, pairing=strategy)
        km = key_metrics(predicted_pairs, relations)
        if name == "cross_product" and documents is not None:
            assert_key_recall_anchor(km, arm=arm)
        strategies[name] = {
            "key": asdict(km),
            "cluster_key": asdict(cluster_key_metrics(clusters, gold)),
            "paper_pair": asdict(paper_pair_metrics(clusters, gold)),
            "workload": asdict(workload(clusters)),
        }

    line = {
        "timestamp": now,
        "git_sha": git_sha,
        "dataset": dataset,
        "arm": arm,
        "n_documents": len(records),
        "n_gold_clusters": len(gold),
        "n_gold_paper_pairs": len(paper_pairs(gold)),
        "diagnostics": asdict(pairing_diagnostics(records, texts=texts)),
        "strategies": strategies,
    }
    Path(log_path).parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(line) + "\n")
    return line
```

Add to the top import block: `import json`, `from dataclasses import asdict, dataclass`, `from pathlib import Path`, and the `biolit.cluster` imports.

- [ ] **Step 6: Run to verify it passes**

Run: `uv run pytest tests/evals/test_cluster_eval.py -v` → PASS.

- [ ] **Step 7: Add `main()` — no direct unit test**

Per the standing ruling and the `end_to_end.main()` / `ner_eval.main()` precedent, `main()` gets **no direct unit test**; keep it to pure wiring and printing. If tdd-guard blocks an incremental `Edit`, land it with a single `Write` of the file (the project's write-then-Edit convention) — do **not** hand-edit `.claude/tdd-guard/` and do **not** invent a test for it.

```python
DEFAULT_LOG = "evals/cluster_runs.jsonl"


def main(argv: list[str] | None = None) -> None:
    # Heavy imports are local so importing this module for scoring stays cheap and offline,
    # same pattern as end_to_end.main.
    import argparse
    from datetime import UTC, datetime

    from biolit.canon.canonicalize import canonicalize
    from biolit.canon.linker import DictionaryLinker
    from biolit.canon.mesh import MeshDictionary
    from biolit.config import get_settings
    from biolit.ner.extract import extract_entities
    from biolit.ner.model import NerModel
    from biolit_evals._meta import git_sha
    from biolit_evals.mesh_gold_download import (
        DEVELOPMENT_MEMBER,
        TEST_MEMBER,
        TRAINING_MEMBER,
        load_bc5cdr_cid_relations,
        load_bc5cdr_documents,
    )

    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", choices=["A", "B"], required=True)
    args = parser.parse_args(argv)
    settings = get_settings()
    url = settings.bc5cdr_cdr_zip_url

    if args.arm == "A":
        # All three splits: no model runs, so the NER checkpoint's training split is not a
        # leakage risk here, and multi-document clusters are sparse enough to want 1500.
        members = (TRAINING_MEMBER, DEVELOPMENT_MEMBER, TEST_MEMBER)
        documents = [d for m in members for d in load_bc5cdr_documents(url, m)]
        relations: dict[str, set[tuple[str, str]]] = {}
        for member in members:
            relations.update(load_bc5cdr_cid_relations(url, member))
        line = run_cluster_eval(
            documents=documents, relations=relations, arm="A", dataset="bc5cdr_all1500",
            log_path=DEFAULT_LOG, git_sha=git_sha(), now=datetime.now(UTC).isoformat(),
        )
    else:
        # Test split ONLY: the NER checkpoint was fine-tuned on BC5CDR's training split, so
        # any arm running real NER must stay held out or the number is contaminated.
        documents = load_bc5cdr_documents(url, TEST_MEMBER)
        relations = load_bc5cdr_cid_relations(url, TEST_MEMBER)
        dictionary = MeshDictionary.from_artifact(settings.mesh_artifact_path)
        linker = DictionaryLinker(dictionary)
        model = NerModel.load(settings)
        records = []
        texts = {}
        for document in documents:
            preds = extract_entities(
                document.text, model, score_threshold=settings.ner_score_threshold
            )
            canon = canonicalize(preds, document.text, linker=linker)
            records.append(ExtractedRecord(paper_id=document.pmid, entities=list(canon)))
            texts[document.pmid] = document.text
        line = run_cluster_eval(
            records=records, texts=texts, relations=relations, arm="B",
            dataset="bc5cdr_test500", log_path=DEFAULT_LOG, git_sha=git_sha(),
            now=datetime.now(UTC).isoformat(),
        )

    print(
        f"arm={line['arm']} docs={line['n_documents']} "
        f"gold_clusters={line['n_gold_clusters']} gold_pairs={line['n_gold_paper_pairs']}"
    )
    print(f"  diagnostics: {line['diagnostics']}")
    for name, s in line["strategies"].items():
        pp, ck, k, w = s["paper_pair"], s["cluster_key"], s["key"], s["workload"]
        print(f"\n=== {name} ===")
        print(f"  PAPER-PAIR (primary): P={pp['precision']:.4f} R={pp['recall']:.4f} "
              f"F1={pp['f1']:.4f} (tp={pp['tp']} fp={pp['fp']} fn={pp['fn']})")
        print(f"  cluster-key (diag):   P={ck['precision']:.4f} R={ck['recall']:.4f} "
              f"F1={ck['f1']:.4f}")
        print(f"  key (diag):           P={k['precision']:.4f} R={k['recall']:.4f} "
              f"F1={k['f1']:.4f}")
        print(f"  Critic workload: clusters={w['n_clusters']} pairs={w['n_paper_pairs']} "
              f"largest={w['largest_cluster']} top5_share={w['top5_pair_share']:.3f}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 8: Verify `--help` resolves without heavy imports**

Run: `uv run python -m biolit_evals.cluster_eval --help`
Expected: argparse usage prints; no model download, no CDR fetch.

- [ ] **Step 9: Gate and commit**

```bash
uv run ruff check . && uv run ruff format --check . && uv run pyright && uv run pytest -q
grep -ciE '^name = "(nvidia|triton)' uv.lock   # must print 0
```

```bash
git add src/biolit_evals/cluster_eval.py tests/evals/test_cluster_eval.py
git commit -F - <<'EOF'
feat(evals): clustering eval runner and CLI for both arms

Arm A synthesizes ExtractedRecords from gold mentions so both arms exercise the same
production code path (cluster_papers) rather than a parallel gold-only implementation, and
runs over all 1500 documents -- no model runs, so the NER checkpoint's training split is
not a leakage risk there. Arm B stays held out to the test 500 because it runs real NER.

INTERFACE DECISION -- multi-id gold mentions: Entity.canonical_id stays str|None and each
(mention, id) becomes its own Entity at the same span. Widening canonical_id to a set would
be a Phase-1 domain change for one eval's convenience; packing ids into one delimited
string would collide with "|", the cluster-key delimiter, and would push id-set logic into
the PairingStrategy that both arms share, leaving Arm B carrying parsing that can never
fire. Consequence stated rather than hidden: duplicate spans inflate Arm A's raw entity
counts in pairing_diagnostics; same-sentence pairing is unaffected since identical offsets
are always in one sentence.

main() has no direct unit test, matching the end_to_end.main()/ner_eval.main() precedent;
it is pure wiring over tested functions.
EOF
```

---

### Task 9: Run both arms and report

**Files:**
- Modify: `backend/evals/cluster_runs.jsonl` (created by the run; committed)

- [ ] **Step 1: Run Arm A**

```bash
uv run python -m biolit_evals.cluster_eval --arm A
```

Expected: the gold-cluster anchor passes (325 clusters over 1500 documents) and the key-recall anchor passes (cross-product recall exactly 1.0000). **If either raises `SystemExit`, STOP** — the harness is wrong. Do not adjust an anchor to match an observation; report the mismatch.

- [ ] **Step 2: Run Arm B**

```bash
uv run python -m biolit_evals.cluster_eval --arm B
```

Expected: the gold-cluster anchor passes (80 clusters over 500 documents). The key-recall anchor does **not** apply — Arm B's entities come from real NER and linking, which genuinely can miss a gold pair's endpoint.

- [ ] **Step 3: Sanity-check against the design-time figures**

Arm A cross-product should land near the provisional gold-entity figures in the spec: ~10.8 keys/doc and ~18.8% cluster-key precision on the test-500 subset. **These were measured on test-500 while Arm A now runs all 1500, so they will not match exactly** — a large divergence in the same direction as the spec's table is expected; a divergence in the opposite direction means something is wrong and should be raised rather than explained away.

- [ ] **Step 4: Commit the run log**

```bash
git add backend/evals/cluster_runs.jsonl
git commit -F - <<'EOF'
feat(evals): clustering eval run log, both arms

Anchors passed: gold-cluster counts reproduce 325 (all 1500) and 80 (test 500), and
cross-product key recall on gold entities is exactly 1.0000.
EOF
```

- [ ] **Step 5: Report back**

Bring the controller: the 4-configuration table at all three metric levels, the Critic workload with top-5 concentration, the NIL-side split, both anchors' status, and a recommendation on whether CID relation extraction earns its own phase. State the key-vs-cluster-level gap explicitly wherever the same-sentence numbers appear, and label Arm A as a ceiling everywhere it is cited.

---

## Self-Review

**Spec coverage.** Every spec section maps to a task: CID loader → 1; `PairingStrategy`/`CrossProductPairing` → 2; `SameSentencePairing` + public `sentence_spans` + fail-closed → 3; `cluster_papers`, `min_size=2`, NIL-by-side diagnostics → 4; three metric levels + the aliasing mutant-killer → 5; workload + top-5 → 6; both anchors with the loader-vs-harness distinction → 7; two arms, corpus split, multi-id interface decision → 8; the run and deliverable → 9.

**Known gaps, deliberate:** the spec's provisional 18.8%/36.9% figures are re-measured in Task 9 rather than asserted in a test — they are results, not invariants. The ARCHITECTURE.md deviation (NIL keys) is documented in the spec and enforced by `_linked_ids`, with its cost measured; no task revises `ARCHITECTURE.md`, which should be updated only once the eval's recommendation is known.

**Type consistency.** `pairs()` returns `set[tuple[str, str]]` in Tasks 2, 3, 5, 8. `Cluster.key` is the `f"{chemical}|{disease}"` string in Tasks 4, 5, 7. `ConceptMetrics` is the return of all three metric functions. `sentence_spans` (public) is used identically in Tasks 3 and 4. `_sentence_index` is defined in Task 3 and imported by Task 4.
