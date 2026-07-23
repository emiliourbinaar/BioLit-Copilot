# Entity Canonicalization (NEN) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a `canonicalize(entities, text)` layer that links each detected CHEMICAL/DISEASE span to a stable MeSH concept ID, so downstream clustering keys on a concept rather than a fragile surface string.

**Architecture:** A new `biolit.canon` package sits between `extract_entities` (Phase 2, untouched) and clustering. It runs a test-first structural fragment-merge (the ADR-0008 fix), then an offline dictionary linker over a CTD→MeSH alias table, behind a fallback-agnostic `Linker` protocol. A `biolit_evals` harness measures linking precision/recall/F1 plus NIL and tiebreak rates against BC5CDR gold MeSH IDs and a blind domain sample.

**Tech Stack:** Python 3.12, uv, pydantic v2, pytest (+ `heavy` marker), httpx (CTD download), stdlib `csv`/`gzip`/`zipfile`/`json`. No new third-party runtime deps.

## Global Constraints

- Run all commands from `backend/` (e.g. `cd backend`). Use `uv run <cmd>`.
- Gate for every task: `uv run ruff check .` + `uv run ruff format --check .` + `uv run pyright` + `uv run pytest` must all pass. ruff ruleset is `E,F,I,UP,B`.
- Datetimes: `datetime.now(UTC)` (ADR-0005 / UP017) — never `datetime.now(timezone.utc)`.
- String-valued enums inherit `enum.StrEnum` (ADR-0005). Reuse the existing `EntityLabel` from `biolit.domain.enums` — do NOT define a new label type.
- tdd-guard hook is active. Write a test file with ONE test function, then add further test functions via Edit (a single Write adding many test funcs can be blocked). Never hand-edit anything under `.claude/tdd-guard/`; never put tdd-guard config in `pyproject.toml`.
- Heavy/real-data code (CTD download, BC5CDR gold download, real linker run) is opt-in behind the `heavy` pytest marker; the default `pytest` run must stay network-free and use fixtures.
- No real download happens until the fixture-based units (Tasks 1–8) are committed (ADR-0006 verification-before-download).
- MeSH/OMIM ID prefixes are preserved on `canonical_id` (`MESH:D…`, `OMIM:…`). BC5CDR gold IDs are bare (`D…`, `-1` = unlinkable) and get reconciled to prefixed form on comparison.
- Never fabricate gold MeSH IDs, PMIDs, or abstract text.

---

### Task 1: Add canonical fields to `Entity`

**Files:**
- Modify: `backend/src/biolit/domain/records.py`
- Test: `backend/tests/domain/test_entity.py` (append)

**Interfaces:**
- Consumes: `biolit.domain.records.Entity`, `biolit.domain.enums.EntityLabel`.
- Produces: `Entity` with optional `canonical_id: str | None = None` and `canonical_name: str | None = None` (both default `None`, meaning NIL/unlinked).

- [ ] **Step 1: Write the failing test** — append to `backend/tests/domain/test_entity.py`:

```python
def test_entity_canonical_fields_default_to_none():
    ent = Entity.model_validate({"text": "metformin", "label": "CHEMICAL", "start": 0, "end": 9})
    assert ent.canonical_id is None
    assert ent.canonical_name is None


def test_entity_carries_canonical_id_and_name():
    ent = Entity.model_validate(
        {
            "text": "metformin",
            "label": "CHEMICAL",
            "start": 0,
            "end": 9,
            "canonical_id": "MESH:D008687",
            "canonical_name": "Metformin",
        }
    )
    assert ent.canonical_id == "MESH:D008687"
    assert ent.canonical_name == "Metformin"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd backend && uv run pytest tests/domain/test_entity.py -v`
Expected: FAIL (`test_entity_carries_canonical_id_and_name` — pydantic ignores unknown keys, so `canonical_id` stays unset / `AttributeError` or the assert fails).

- [ ] **Step 3: Add the fields** — in `backend/src/biolit/domain/records.py`, change the `Entity` model:

```python
class Entity(BaseModel):
    text: str
    label: EntityLabel  # type-checked at the Phase 4 extraction seam
    start: int | None = None
    end: int | None = None
    canonical_id: str | None = None  # e.g. "MESH:D008687" / "OMIM:125853"; None = NIL
    canonical_name: str | None = None  # CTD preferred name; None = NIL
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd backend && uv run pytest tests/domain/test_entity.py -v`
Expected: PASS (all 4 tests).

- [ ] **Step 5: Commit**

```bash
git add backend/src/biolit/domain/records.py backend/tests/domain/test_entity.py
git commit -m "feat(domain): add canonical_id/canonical_name to Entity"
```

---

### Task 2: `mesh.py` core — normalization, concepts, and dictionary lookup with tiebreak

**Files:**
- Create: `backend/src/biolit/canon/__init__.py`
- Create: `backend/src/biolit/canon/mesh.py`
- Create: `backend/tests/canon/__init__.py`
- Create: `backend/tests/canon/test_mesh.py`

**Interfaces:**
- Produces:
  - `normalize_surface(s: str) -> str` — casefold, strip, collapse internal whitespace.
  - `MeshConcept(id: str, name: str)` frozen dataclass.
  - `AliasEntry(concept: MeshConcept, is_preferred_name: bool)` frozen dataclass.
  - `LinkResult(concept: MeshConcept | None, tiebroken: bool)` frozen dataclass.
  - `MeshDictionary(aliases: dict[str, list[AliasEntry]])` with `lookup(surface: str) -> LinkResult`.
- `lookup` tiebreak rule: if the normalized surface has entries spanning >1 distinct `concept.id`, prefer entries with `is_preferred_name=True`; among the winning pool pick the lexicographically smallest `concept.id`; set `tiebroken=True`. A single distinct id → `tiebroken=False`. No entries → `LinkResult(None, False)`.

- [ ] **Step 1: Create the package** — create empty `backend/src/biolit/canon/__init__.py` and empty `backend/tests/canon/__init__.py`.

- [ ] **Step 2: Write the first failing test** — create `backend/tests/canon/test_mesh.py`:

```python
from biolit.canon.mesh import (
    AliasEntry,
    MeshConcept,
    MeshDictionary,
    normalize_surface,
)


def test_normalize_surface_casefolds_and_collapses_whitespace():
    assert normalize_surface("  GLP-1  Receptor   Agonists ") == "glp-1 receptor agonists"
```

- [ ] **Step 3: Run to verify it fails**

Run: `cd backend && uv run pytest tests/canon/test_mesh.py -v`
Expected: FAIL (`ModuleNotFoundError: biolit.canon.mesh`).

- [ ] **Step 4: Implement `mesh.py` core** — create `backend/src/biolit/canon/mesh.py`:

```python
import re
from dataclasses import dataclass

_WHITESPACE = re.compile(r"\s+")


def normalize_surface(s: str) -> str:
    """Casefold, strip, and collapse internal whitespace so the alias table and a
    lookup query are normalized identically (they MUST use this same function)."""
    return _WHITESPACE.sub(" ", s.strip()).casefold()


@dataclass(frozen=True)
class MeshConcept:
    id: str  # prefixed: "MESH:D008687", "OMIM:125853"
    name: str


@dataclass(frozen=True)
class AliasEntry:
    concept: MeshConcept
    is_preferred_name: bool


@dataclass(frozen=True)
class LinkResult:
    concept: MeshConcept | None
    tiebroken: bool


class MeshDictionary:
    def __init__(self, aliases: dict[str, list[AliasEntry]]) -> None:
        self._aliases = aliases

    def lookup(self, surface: str) -> LinkResult:
        entries = self._aliases.get(normalize_surface(surface))
        if not entries:
            return LinkResult(None, False)
        distinct_ids = {e.concept.id for e in entries}
        if len(distinct_ids) == 1:
            return LinkResult(entries[0].concept, False)
        preferred = [e for e in entries if e.is_preferred_name]
        pool = preferred if preferred else entries
        concept = min((e.concept for e in pool), key=lambda c: c.id)
        return LinkResult(concept, True)
```

- [ ] **Step 5: Run to verify pass**

Run: `cd backend && uv run pytest tests/canon/test_mesh.py -v`
Expected: PASS.

- [ ] **Step 6: Add lookup tests via Edit** — append to `backend/tests/canon/test_mesh.py`:

```python
def _dict() -> MeshDictionary:
    metformin = MeshConcept(id="MESH:D008687", name="Metformin")
    other = MeshConcept(id="MESH:D000001", name="Aspirin Variant")
    return MeshDictionary(
        {
            "metformin": [AliasEntry(metformin, True)],
            "glucophage": [AliasEntry(metformin, False)],
            # ambiguous: two distinct concepts, one via preferred name
            "ambig": [AliasEntry(metformin, False), AliasEntry(other, True)],
            # ambiguous, no preferred name -> smallest id wins
            "ambignopref": [AliasEntry(metformin, False), AliasEntry(other, False)],
        }
    )


def test_lookup_exact_and_synonym_hit_are_not_tiebroken():
    d = _dict()
    r_name = d.lookup("Metformin")
    r_syn = d.lookup("glucophage")
    assert r_name.concept is not None and r_name.concept.id == "MESH:D008687"
    assert r_syn.concept is not None and r_syn.concept.id == "MESH:D008687"
    assert r_name.tiebroken is False and r_syn.tiebroken is False


def test_lookup_miss_returns_nil():
    assert _dict().lookup("nonexistent term").concept is None


def test_lookup_ambiguous_prefers_preferred_name_and_flags_tiebreak():
    r = _dict().lookup("ambig")
    assert r.concept is not None and r.concept.id == "MESH:D000001"  # preferred-name entry
    assert r.tiebroken is True


def test_lookup_ambiguous_no_preferred_picks_smallest_id():
    r = _dict().lookup("ambignopref")
    assert r.concept is not None and r.concept.id == "MESH:D000001"  # lexicographically smallest
    assert r.tiebroken is True
```

- [ ] **Step 7: Run to verify pass**

Run: `cd backend && uv run pytest tests/canon/test_mesh.py -v`
Expected: PASS (5 tests).

- [ ] **Step 8: Gate + commit**

```bash
cd backend && uv run ruff check . && uv run ruff format --check . && uv run pyright && uv run pytest -q
cd .. && git add backend/src/biolit/canon backend/tests/canon
git commit -m "feat(canon): MeshDictionary lookup with deterministic ambiguity tiebreak"
```

---

### Task 3: `mesh.py` — CTD parsing, alias-table build, artifact round-trip

**Files:**
- Modify: `backend/src/biolit/canon/mesh.py`
- Test: `backend/tests/canon/test_mesh.py` (append)
- Create: `backend/tests/canon/fixtures/ctd_chemicals_sample.tsv`
- Create: `backend/tests/canon/fixtures/ctd_diseases_sample.tsv`

**Interfaces:**
- Consumes: `MeshConcept`, `AliasEntry`, `normalize_surface`, `MeshDictionary` (Task 2).
- Produces:
  - `build_alias_table(chem_rows: list[list[str]], disease_rows: list[list[str]]) -> dict[str, list[AliasEntry]]`.
    - Chemical row indices: name=0, id=1 (bare MeSH like `D008687`; prefixed to `MESH:…`), synonyms=7 (`|`-separated).
    - Disease row indices: name=0, id=1 (already prefixed `MESH:…`/`OMIM:…`), synonyms=7.
    - Preferred-name alias from the name column (`is_preferred_name=True`); each synonym is a non-preferred alias. Empty aliases skipped. Duplicate `(concept.id, is_preferred_name)` pairs under one alias are de-duplicated.
  - `MeshDictionary.save_artifact(path: str) -> None` and `MeshDictionary.from_artifact(path: str) -> MeshDictionary` (gzip-JSON round-trip).

- [ ] **Step 1: Create fixture TSVs** — create `backend/tests/canon/fixtures/ctd_chemicals_sample.tsv` (tab-separated; the `#` line is a header comment that must be skipped by the caller, not by the parser):

```
# ChemicalName	ChemicalID	CasRN	Definition	ParentIDs	TreeNumbers	ParentTreeNumbers	Synonyms
Metformin	D008687	657-24-9			D02.078			Glucophage|Dimethylbiguanide
Aspirin	D001241	50-78-2			D02.241			Acetylsalicylic Acid
```

Create `backend/tests/canon/fixtures/ctd_diseases_sample.tsv`:

```
# DiseaseName	DiseaseID	AltDiseaseIDs	Definition	ParentIDs	TreeNumbers	ParentTreeNumbers	Synonyms	SlimMappings
Polycystic Ovary Syndrome	MESH:D011085			C13.351			Stein-Leventhal Syndrome|PCOS
Diabetes Mellitus	MESH:D003920			C18.452			
```

- [ ] **Step 2: Write the failing test** — append to `backend/tests/canon/test_mesh.py`:

```python
from pathlib import Path

from biolit.canon.mesh import build_alias_table

_FIX = Path(__file__).parent / "fixtures"


def _rows(name: str) -> list[list[str]]:
    lines = (_FIX / name).read_text(encoding="utf-8").splitlines()
    return [ln.split("\t") for ln in lines if ln and not ln.startswith("#")]


def test_build_alias_table_prefixes_chemicals_and_keeps_disease_prefix():
    table = build_alias_table(_rows("ctd_chemicals_sample.tsv"), _rows("ctd_diseases_sample.tsv"))
    # chemical preferred name -> MESH-prefixed id
    met = table["metformin"]
    assert len(met) == 1 and met[0].concept.id == "MESH:D008687" and met[0].is_preferred_name
    # chemical synonym -> non-preferred, same concept
    assert table["glucophage"][0].concept.id == "MESH:D008687"
    assert table["glucophage"][0].is_preferred_name is False
    # disease id prefix preserved as-is
    assert table["pcos"][0].concept.id == "MESH:D011085"
```

- [ ] **Step 3: Run to verify it fails**

Run: `cd backend && uv run pytest tests/canon/test_mesh.py::test_build_alias_table_prefixes_chemicals_and_keeps_disease_prefix -v`
Expected: FAIL (`ImportError: cannot import name 'build_alias_table'`).

- [ ] **Step 4: Implement parsing + build** — append to `backend/src/biolit/canon/mesh.py`:

```python
def _add_alias(
    table: dict[str, list[AliasEntry]], alias: str, concept: MeshConcept, is_pref: bool
) -> None:
    key = normalize_surface(alias)
    if not key:
        return
    bucket = table.setdefault(key, [])
    for existing in bucket:
        if existing.concept.id == concept.id and existing.is_preferred_name == is_pref:
            return
    bucket.append(AliasEntry(concept, is_pref))


def _ingest_rows(
    table: dict[str, list[AliasEntry]], rows: list[list[str]], id_prefix: str
) -> None:
    for row in rows:
        if len(row) < 2:
            continue
        name = row[0].strip()
        raw_id = row[1].strip()
        if not name or not raw_id:
            continue
        concept = MeshConcept(id=f"{id_prefix}{raw_id}", name=name)
        _add_alias(table, name, concept, True)
        synonyms = row[7] if len(row) > 7 else ""
        for syn in synonyms.split("|"):
            if syn.strip():
                _add_alias(table, syn, concept, False)


def build_alias_table(
    chem_rows: list[list[str]], disease_rows: list[list[str]]
) -> dict[str, list[AliasEntry]]:
    """Build a normalized alias -> [AliasEntry] table from CTD chemical + disease rows.

    CTD chemical IDs are bare MeSH accessions (`D008687`) and get a `MESH:` prefix; CTD
    disease IDs already carry their `MESH:`/`OMIM:` prefix and are used verbatim.
    """
    table: dict[str, list[AliasEntry]] = {}
    _ingest_rows(table, chem_rows, "MESH:")
    _ingest_rows(table, disease_rows, "")
    return table
```

- [ ] **Step 5: Run to verify pass**

Run: `cd backend && uv run pytest tests/canon/test_mesh.py -v`
Expected: PASS.

- [ ] **Step 6: Write the artifact round-trip test** — append to `backend/tests/canon/test_mesh.py`:

```python
from biolit.canon.mesh import MeshDictionary


def test_artifact_round_trip(tmp_path):
    table = build_alias_table(_rows("ctd_chemicals_sample.tsv"), _rows("ctd_diseases_sample.tsv"))
    d = MeshDictionary(table)
    path = str(tmp_path / "mesh.json.gz")
    d.save_artifact(path)
    reloaded = MeshDictionary.from_artifact(path)
    r = reloaded.lookup("Glucophage")
    assert r.concept is not None and r.concept.id == "MESH:D008687"
    assert r.concept.name == "Metformin"
```

- [ ] **Step 7: Run to verify it fails**

Run: `cd backend && uv run pytest tests/canon/test_mesh.py::test_artifact_round_trip -v`
Expected: FAIL (`AttributeError: 'MeshDictionary' object has no attribute 'save_artifact'`).

- [ ] **Step 8: Implement artifact I/O** — add `import gzip` and `import json` at the top of `mesh.py`, then add these methods to `MeshDictionary`:

```python
    def save_artifact(self, path: str) -> None:
        payload = {
            alias: [[e.concept.id, e.concept.name, e.is_preferred_name] for e in entries]
            for alias, entries in self._aliases.items()
        }
        with gzip.open(path, "wt", encoding="utf-8") as fh:
            json.dump(payload, fh)

    @classmethod
    def from_artifact(cls, path: str) -> "MeshDictionary":
        with gzip.open(path, "rt", encoding="utf-8") as fh:
            payload = json.load(fh)
        aliases: dict[str, list[AliasEntry]] = {
            alias: [AliasEntry(MeshConcept(id=i, name=n), bool(p)) for i, n, p in rows]
            for alias, rows in payload.items()
        }
        return cls(aliases)
```

- [ ] **Step 9: Run to verify pass + gate**

Run: `cd backend && uv run pytest tests/canon/test_mesh.py -v && uv run ruff check . && uv run pyright`
Expected: PASS, no ruff/pyright errors.

- [ ] **Step 10: Commit**

```bash
git add backend/src/biolit/canon/mesh.py backend/tests/canon/test_mesh.py backend/tests/canon/fixtures
git commit -m "feat(canon): build CTD->MeSH alias table + gzip artifact round-trip"
```

---

### Task 4: `fragments.py` — structural fragment merge (the ADR-0008 fix)

**Files:**
- Create: `backend/src/biolit/canon/fragments.py`
- Create: `backend/tests/canon/test_fragments.py`

**Interfaces:**
- Consumes: `biolit.domain.records.Entity`, `biolit.domain.enums.EntityLabel`.
- Produces:
  - `FragmentCandidate(text: str, label: EntityLabel, start: int, end: int, source_indices: tuple[int, ...])` frozen dataclass.
  - `merge_fragments(entities: list[Entity], text: str) -> list[FragmentCandidate]`.
- Merge rule: sort entities by `start`. A maximal run of ≥2 consecutive same-label entities where each gap `text[prev.end:next.start]` is **connector-only** (empty, or all chars in `{-, ‑, ‒, –, /}` — no whitespace) is a candidate **iff** the run contains at least one **fragment-shaped** token. `text` field of the candidate is the source slice `text[run_start:run_end]` (preserves the hyphen). `source_indices` are indices into the input `entities` list.
- **Fragment-shaped** token := NOT a plain word, where plain word := matches `^[A-Za-z]{3,}$` and is not all-uppercase. (So `aspirin` is a plain word; `GLP`, `1RA`, `is`, `CF` are fragment-shaped.)

- [ ] **Step 1: Write the first (positive) failing test** — create `backend/tests/canon/test_fragments.py`:

```python
from biolit.canon.fragments import merge_fragments
from biolit.domain.enums import EntityLabel
from biolit.domain.records import Entity

CHEMICAL = EntityLabel.CHEMICAL
DISEASE = EntityLabel.DISEASE


def test_merges_hyphen_split_abbreviation_fragments():
    # "GLP-1RA" fragmented by the tokenizer into "GLP" + "1RA"
    text = "GLP-1RA therapy"
    ents = [
        Entity(text="GLP", label=CHEMICAL, start=0, end=3),
        Entity(text="1RA", label=CHEMICAL, start=4, end=7),
    ]
    cands = merge_fragments(ents, text)
    assert len(cands) == 1
    assert cands[0].text == "GLP-1RA"
    assert cands[0].start == 0 and cands[0].end == 7
    assert cands[0].source_indices == (0, 1)
```

- [ ] **Step 2: Run to verify it fails**

Run: `cd backend && uv run pytest tests/canon/test_fragments.py -v`
Expected: FAIL (`ModuleNotFoundError: biolit.canon.fragments`).

- [ ] **Step 3: Implement `fragments.py`** — create `backend/src/biolit/canon/fragments.py`:

```python
import re
from dataclasses import dataclass

from biolit.domain.enums import EntityLabel
from biolit.domain.records import Entity

# Hyphen/dash/slash connectors that indicate a split token rather than a phrase boundary.
# Whitespace is deliberately excluded: "GLP 1RA" is two words, "GLP-1RA" is one split token.
_CONNECTORS = frozenset("-‐‑‒–/")
_PLAIN_WORD = re.compile(r"[A-Za-z]{3,}$")


@dataclass(frozen=True)
class FragmentCandidate:
    text: str
    label: EntityLabel
    start: int
    end: int
    source_indices: tuple[int, ...]


def _connector_only(sep: str) -> bool:
    return all(ch in _CONNECTORS for ch in sep)  # all("") is True -> zero-gap counts


def _is_fragment_shaped(token: str) -> bool:
    """A plain lowercase/Titlecase word (>=3 letters) is a standalone term; everything
    else (has a digit, is all-caps, or is <3 chars) looks like a tokenizer fragment."""
    if _PLAIN_WORD.match(token) and not token.isupper():
        return False
    return True


def merge_fragments(entities: list[Entity], text: str) -> list[FragmentCandidate]:
    order = sorted(range(len(entities)), key=lambda i: entities[i].start or 0)
    candidates: list[FragmentCandidate] = []
    i = 0
    n = len(order)
    while i < n:
        j = i
        while j + 1 < n:
            a = entities[order[j]]
            b = entities[order[j + 1]]
            if a.label != b.label or a.end is None or b.start is None:
                break
            if not _connector_only(text[a.end : b.start]):
                break
            j += 1
        run = order[i : j + 1]
        if len(run) >= 2 and any(_is_fragment_shaped(entities[k].text) for k in run):
            start = entities[run[0]].start
            end = entities[run[-1]].end
            if start is not None and end is not None:
                candidates.append(
                    FragmentCandidate(
                        text=text[start:end],
                        label=entities[run[0]].label,
                        start=start,
                        end=end,
                        source_indices=tuple(run),
                    )
                )
        i = j + 1
    return candidates
```

- [ ] **Step 4: Run to verify pass**

Run: `cd backend && uv run pytest tests/canon/test_fragments.py -v`
Expected: PASS.

- [ ] **Step 5: Add the adversarial (must-NOT-merge) test via Edit** — append to `backend/tests/canon/test_fragments.py`:

```python
def test_does_not_merge_two_distinct_hyphen_adjacent_plain_words():
    # Guard for the merged-candidate-first lookup order: two genuinely distinct,
    # correctly-split same-label entities that happen to be hyphen-adjacent must NOT be
    # merged, or a bad merge could coincidentally resolve to a real (wrong) shared MeSH id.
    text = "aspirin-warfarin interaction"
    ents = [
        Entity(text="aspirin", label=CHEMICAL, start=0, end=7),
        Entity(text="warfarin", label=CHEMICAL, start=8, end=16),
    ]
    assert merge_fragments(ents, text) == []


def test_does_not_merge_across_whitespace():
    text = "GLP 1RA"
    ents = [
        Entity(text="GLP", label=CHEMICAL, start=0, end=3),
        Entity(text="1RA", label=CHEMICAL, start=4, end=7),
    ]
    assert merge_fragments(ents, text) == []


def test_does_not_merge_different_labels():
    text = "insulin-resistance"  # even if connector-joined, a CHEMICAL + DISEASE never merge
    ents = [
        Entity(text="insulin", label=CHEMICAL, start=0, end=7),
        Entity(text="resistance", label=DISEASE, start=8, end=18),
    ]
    assert merge_fragments(ents, text) == []
```

- [ ] **Step 6: Run to verify pass + gate**

Run: `cd backend && uv run pytest tests/canon/test_fragments.py -v && uv run ruff check . && uv run pyright`
Expected: PASS (4 tests), clean gate.

- [ ] **Step 7: Commit**

```bash
git add backend/src/biolit/canon/fragments.py backend/tests/canon/test_fragments.py
git commit -m "feat(canon): structural fragment merge with adversarial must-not-merge guards"
```

---

### Task 5: `linker.py` — `Linker` protocol + `DictionaryLinker`

**Files:**
- Create: `backend/src/biolit/canon/linker.py`
- Create: `backend/tests/canon/test_linker.py`

**Interfaces:**
- Consumes: `biolit.canon.mesh.MeshDictionary`, `biolit.canon.mesh.LinkResult`.
- Produces:
  - `Linker` typing `Protocol` with `link(self, surface: str) -> LinkResult`.
  - `DictionaryLinker(dictionary: MeshDictionary)` implementing `Linker` by delegating to `dictionary.lookup`.
- This is the fallback-agnostic seam: a future `SapBertLinker` implements the same `Protocol` with no change to `canonicalize`.

- [ ] **Step 1: Write the failing test** — create `backend/tests/canon/test_linker.py`:

```python
from biolit.canon.linker import DictionaryLinker, Linker
from biolit.canon.mesh import AliasEntry, MeshConcept, MeshDictionary


def test_dictionary_linker_delegates_to_dictionary():
    concept = MeshConcept(id="MESH:D008687", name="Metformin")
    d = MeshDictionary({"metformin": [AliasEntry(concept, True)]})
    linker: Linker = DictionaryLinker(d)
    result = linker.link("Metformin")
    assert result.concept is not None and result.concept.id == "MESH:D008687"
    assert linker.link("nope").concept is None
```

- [ ] **Step 2: Run to verify it fails**

Run: `cd backend && uv run pytest tests/canon/test_linker.py -v`
Expected: FAIL (`ModuleNotFoundError: biolit.canon.linker`).

- [ ] **Step 3: Implement `linker.py`** — create `backend/src/biolit/canon/linker.py`:

```python
from typing import Protocol

from biolit.canon.mesh import LinkResult, MeshDictionary


class Linker(Protocol):
    def link(self, surface: str) -> LinkResult: ...


class DictionaryLinker:
    """Offline linker over a CTD->MeSH alias table. Normalization/tiebreak live in
    MeshDictionary.lookup; this is a thin adapter so a future embedding-based linker can
    implement the same Linker protocol without touching canonicalize()."""

    def __init__(self, dictionary: MeshDictionary) -> None:
        self._dictionary = dictionary

    def link(self, surface: str) -> LinkResult:
        return self._dictionary.lookup(surface)
```

- [ ] **Step 4: Run to verify pass + gate**

Run: `cd backend && uv run pytest tests/canon/test_linker.py -v && uv run pyright`
Expected: PASS, clean.

- [ ] **Step 5: Commit**

```bash
git add backend/src/biolit/canon/linker.py backend/tests/canon/test_linker.py
git commit -m "feat(canon): Linker protocol + DictionaryLinker adapter"
```

---

### Task 6: `canonicalize.py` — orchestration

**Files:**
- Create: `backend/src/biolit/canon/canonicalize.py`
- Modify: `backend/src/biolit/canon/__init__.py`
- Create: `backend/tests/canon/test_canonicalize.py`

**Interfaces:**
- Consumes: `merge_fragments` (Task 4), `Linker` (Task 5), `Entity` (Task 1).
- Produces: `canonicalize(entities: list[Entity], text: str, *, linker: Linker) -> list[Entity]`.
  - Try every merged candidate first: if it links, all its `source_indices` entities inherit that concept. Remaining entities are looked up on their own surface. NIL stays `None`. Returns new `Entity` objects (via `model_copy(update=…)`) — inputs are not mutated.
  - `linker` is an injected keyword arg (DI, mirroring `run_eval`), so tests pass a fixture linker and production passes a `DictionaryLinker`.

- [ ] **Step 1: Write the first failing test** — create `backend/tests/canon/test_canonicalize.py`:

```python
from biolit.canon.canonicalize import canonicalize
from biolit.canon.linker import DictionaryLinker
from biolit.canon.mesh import AliasEntry, MeshConcept, MeshDictionary
from biolit.domain.enums import EntityLabel
from biolit.domain.records import Entity

CHEMICAL = EntityLabel.CHEMICAL


def _linker() -> DictionaryLinker:
    glp = MeshConcept(id="MESH:D000067299", name="Glucagon-Like Peptide-1 Receptor Agonists")
    met = MeshConcept(id="MESH:D008687", name="Metformin")
    return DictionaryLinker(
        MeshDictionary(
            {
                "glp-1ra": [AliasEntry(glp, True)],
                "metformin": [AliasEntry(met, True)],
            }
        )
    )


def test_merged_fragment_links_all_constituents_to_one_concept():
    text = "GLP-1RA and metformin"
    ents = [
        Entity(text="GLP", label=CHEMICAL, start=0, end=3),
        Entity(text="1RA", label=CHEMICAL, start=4, end=7),
        Entity(text="metformin", label=CHEMICAL, start=12, end=21),
    ]
    out = canonicalize(ents, text, linker=_linker())
    assert out[0].canonical_id == "MESH:D000067299"
    assert out[1].canonical_id == "MESH:D000067299"  # both fragments share the merged concept
    assert out[2].canonical_id == "MESH:D008687"  # individual lookup
```

- [ ] **Step 2: Run to verify it fails**

Run: `cd backend && uv run pytest tests/canon/test_canonicalize.py -v`
Expected: FAIL (`ModuleNotFoundError: biolit.canon.canonicalize`).

- [ ] **Step 3: Implement `canonicalize.py`** — create `backend/src/biolit/canon/canonicalize.py`:

```python
from biolit.canon.fragments import merge_fragments
from biolit.canon.linker import Linker
from biolit.domain.records import Entity


def canonicalize(entities: list[Entity], text: str, *, linker: Linker) -> list[Entity]:
    """Populate canonical_id/canonical_name on each entity via MeSH linking.

    Merged fragment candidates are tried first: when one links, every entity it spans
    inherits that concept (the ADR-0008 fix). Entities not resolved via a merge fall back
    to an individual lookup on their own surface form. Unlinked entities stay NIL (None).
    """
    resolved: dict[int, tuple[str, str]] = {}
    for cand in merge_fragments(entities, text):
        result = linker.link(cand.text)
        if result.concept is not None:
            for idx in cand.source_indices:
                resolved[idx] = (result.concept.id, result.concept.name)

    out: list[Entity] = []
    for i, entity in enumerate(entities):
        if i in resolved:
            cid, cname = resolved[i]
        else:
            r = linker.link(entity.text)
            cid, cname = (r.concept.id, r.concept.name) if r.concept is not None else (None, None)
        out.append(entity.model_copy(update={"canonical_id": cid, "canonical_name": cname}))
    return out
```

- [ ] **Step 4: Run to verify pass**

Run: `cd backend && uv run pytest tests/canon/test_canonicalize.py -v`
Expected: PASS.

- [ ] **Step 5: Add NIL + no-mutation tests via Edit** — append to `backend/tests/canon/test_canonicalize.py`:

```python
def test_unlinked_entity_stays_nil():
    text = "unobtainium"
    ents = [Entity(text="unobtainium", label=CHEMICAL, start=0, end=11)]
    out = canonicalize(ents, text, linker=_linker())
    assert out[0].canonical_id is None and out[0].canonical_name is None


def test_does_not_mutate_input_entities():
    text = "metformin"
    ents = [Entity(text="metformin", label=CHEMICAL, start=0, end=9)]
    canonicalize(ents, text, linker=_linker())
    assert ents[0].canonical_id is None  # original untouched
```

- [ ] **Step 6: Export from package** — set `backend/src/biolit/canon/__init__.py` to:

```python
from biolit.canon.canonicalize import canonicalize

__all__ = ["canonicalize"]
```

- [ ] **Step 7: Run to verify pass + gate**

Run: `cd backend && uv run pytest tests/canon -v && uv run ruff check . && uv run pyright`
Expected: PASS (all canon tests), clean gate.

- [ ] **Step 8: Commit**

```bash
git add backend/src/biolit/canon/canonicalize.py backend/src/biolit/canon/__init__.py backend/tests/canon/test_canonicalize.py
git commit -m "feat(canon): canonicalize() orchestration (merge-first, NIL-safe, pure)"
```

---

### Task 7: `mesh_gold.py` — MeSH-ID reconciliation, PubTator + domain gold parsing

**Files:**
- Create: `backend/src/biolit_evals/mesh_gold.py`
- Create: `backend/tests/evals/test_mesh_gold.py`
- Create: `backend/tests/evals/fixtures/pubtator_sample.txt`
- Create: `backend/tests/evals/fixtures/domain_norm_fixture.jsonl`

**Interfaces:**
- Consumes: `biolit.domain.enums.EntityLabel`, `biolit.ner.labels.canonical_label`.
- Produces:
  - `GoldMention(pmid: str, start: int, end: int, text: str, label: EntityLabel, mesh_ids: tuple[str, ...])` frozen dataclass. `mesh_ids` are prefixed (`MESH:…`); empty tuple = unlinkable.
  - `reconcile_mesh_id(raw: str) -> tuple[str, ...]`: splits on `|`, drops `-1`/empty, prefixes bare accessions with `MESH:`, keeps already-prefixed (`OMIM:`, `MESH:`) verbatim.
  - `parse_pubtator(text: str) -> list[GoldMention]`: parses PubTator mention lines `PMID\tstart\tend\tmention\ttype\tid`; `type` mapped via `canonical_label` (rows whose type is not CHEMICAL/DISEASE are skipped).
  - `load_domain_norm_sample(path: str) -> list[GoldMention]`: JSONL with `{pmid, text, entities:[{start,end,label,text,mesh_id}]}`.

- [ ] **Step 1: Create fixtures** — create `backend/tests/evals/fixtures/pubtator_sample.txt` (mention lines are TAB-separated):

```
1234|t|Metformin in polycystic ovary syndrome
1234|a|Metformin improved CFD outcomes.
1234	0	9	Metformin	Chemical	D008687
1234	13	38	polycystic ovary syndrome	Disease	D011085
1234	19	22	CFD	Chemical	-1
1234	0	0	IgnoredGene	Gene	D000001
```

Create `backend/tests/evals/fixtures/domain_norm_fixture.jsonl`:

```
{"pmid": "1234", "text": "Metformin treats PCOS", "entities": [{"start": 0, "end": 9, "label": "CHEMICAL", "text": "Metformin", "mesh_id": "D008687"}, {"start": 17, "end": 21, "label": "DISEASE", "text": "PCOS", "mesh_id": "MESH:D011085"}]}
```

- [ ] **Step 2: Write the first failing test** — create `backend/tests/evals/test_mesh_gold.py`:

```python
from biolit_evals.mesh_gold import reconcile_mesh_id


def test_reconcile_prefixes_bare_and_preserves_prefixed_and_drops_unlinkable():
    assert reconcile_mesh_id("D008687") == ("MESH:D008687",)
    assert reconcile_mesh_id("MESH:D011085") == ("MESH:D011085",)
    assert reconcile_mesh_id("OMIM:125853") == ("OMIM:125853",)
    assert reconcile_mesh_id("-1") == ()
    assert reconcile_mesh_id("D1|D2") == ("MESH:D1", "MESH:D2")
```

- [ ] **Step 3: Run to verify it fails**

Run: `cd backend && uv run pytest tests/evals/test_mesh_gold.py -v`
Expected: FAIL (`ModuleNotFoundError: biolit_evals.mesh_gold`).

- [ ] **Step 4: Implement `mesh_gold.py`** — create `backend/src/biolit_evals/mesh_gold.py`:

```python
import json
from dataclasses import dataclass

from biolit.domain.enums import EntityLabel
from biolit.ner.labels import canonical_label


@dataclass(frozen=True)
class GoldMention:
    pmid: str
    start: int
    end: int
    text: str
    label: EntityLabel
    mesh_ids: tuple[str, ...]  # prefixed ("MESH:D011085"); () == unlinkable


def reconcile_mesh_id(raw: str) -> tuple[str, ...]:
    """Reconcile a BC5CDR/domain gold id string to prefixed MeSH/OMIM ids.

    BC5CDR ids are bare accessions (`D011085`) with `-1` for unlinkable and `|` joining
    composite mentions; CTD/domain ids may already carry a `MESH:`/`OMIM:` prefix.
    """
    ids: list[str] = []
    for part in raw.split("|"):
        part = part.strip()
        if not part or part == "-1":
            continue
        ids.append(part if ":" in part else f"MESH:{part}")
    return tuple(ids)


def parse_pubtator(text: str) -> list[GoldMention]:
    mentions: list[GoldMention] = []
    for line in text.splitlines():
        parts = line.split("\t")
        if len(parts) < 6:
            continue  # title/abstract (`PMID|t|…`) and blank lines have no tabs
        pmid, start, end, mention, raw_type, raw_id = parts[:6]
        label = canonical_label(raw_type)
        if label is None:
            continue  # skip out-of-scope types (e.g. Gene)
        mentions.append(
            GoldMention(
                pmid=pmid,
                start=int(start),
                end=int(end),
                text=mention,
                label=label,
                mesh_ids=reconcile_mesh_id(raw_id),
            )
        )
    return mentions


def load_domain_norm_sample(path: str) -> list[GoldMention]:
    mentions: list[GoldMention] = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            pmid = str(rec.get("pmid") or rec.get("paper_id") or "")
            for ent in rec.get("entities", []):
                label = canonical_label(ent["label"])
                if label is None:
                    raise ValueError(f"non-canonical label in {pmid}: {ent['label']}")
                mentions.append(
                    GoldMention(
                        pmid=pmid,
                        start=int(ent["start"]),
                        end=int(ent["end"]),
                        text=ent["text"],
                        label=label,
                        mesh_ids=reconcile_mesh_id(str(ent["mesh_id"])),
                    )
                )
    return mentions
```

- [ ] **Step 5: Run to verify pass**

Run: `cd backend && uv run pytest tests/evals/test_mesh_gold.py -v`
Expected: PASS.

- [ ] **Step 6: Add parser tests via Edit** — append to `backend/tests/evals/test_mesh_gold.py`:

```python
from pathlib import Path

from biolit.domain.enums import EntityLabel
from biolit_evals.mesh_gold import load_domain_norm_sample, parse_pubtator

_FIX = Path(__file__).parent / "fixtures"


def test_parse_pubtator_extracts_mentions_and_skips_non_target_types():
    mentions = parse_pubtator((_FIX / "pubtator_sample.txt").read_text(encoding="utf-8"))
    assert len(mentions) == 3  # Gene row skipped
    met = mentions[0]
    assert met.text == "Metformin" and met.label == EntityLabel.CHEMICAL
    assert met.mesh_ids == ("MESH:D008687",)
    assert mentions[2].mesh_ids == ()  # CFD, unlinkable (-1)


def test_load_domain_norm_sample_parses_mesh_ids():
    mentions = load_domain_norm_sample(str(_FIX / "domain_norm_fixture.jsonl"))
    assert [m.mesh_ids for m in mentions] == [("MESH:D008687",), ("MESH:D011085",)]
```

- [ ] **Step 7: Run to verify pass + gate**

Run: `cd backend && uv run pytest tests/evals/test_mesh_gold.py -v && uv run ruff check . && uv run pyright`
Expected: PASS (3 tests), clean.

- [ ] **Step 8: Commit**

```bash
git add backend/src/biolit_evals/mesh_gold.py backend/tests/evals/test_mesh_gold.py backend/tests/evals/fixtures/pubtator_sample.txt backend/tests/evals/fixtures/domain_norm_fixture.jsonl
git commit -m "feat(evals): MeSH-id reconciliation + PubTator/domain gold parsing"
```

---

### Task 8: `linking_scoring.py` — linking P/R/F1 + NIL + tiebreak metrics

**Files:**
- Create: `backend/src/biolit_evals/linking_scoring.py`
- Create: `backend/tests/evals/test_linking_scoring.py`

**Interfaces:**
- Consumes: `biolit.canon.linker.Linker`, `biolit_evals.mesh_gold.GoldMention`.
- Produces:
  - `LinkingMetrics(n, correct, linked, tiebroken, precision, recall, f1, nil_rate, tiebreak_rate)` frozen dataclass (`n/correct/linked/tiebroken` int; the rest float).
  - `score_linking(gold: list[GoldMention], linker: Linker) -> LinkingMetrics`.
- Evaluated on **linkable gold only** (mentions with non-empty `mesh_ids`), feeding gold surface forms straight to the linker (isolating linking from NER). A prediction is `correct` when its `concept.id` is in the mention's `mesh_ids`. `precision = correct/linked`, `recall = correct/n`, `f1` harmonic, `nil_rate = (n-linked)/n`, `tiebreak_rate = tiebroken/n`. Empty eval set → all-zero metrics.

- [ ] **Step 1: Write the failing test** — create `backend/tests/evals/test_linking_scoring.py`:

```python
import pytest

from biolit.canon.linker import DictionaryLinker
from biolit.canon.mesh import AliasEntry, MeshConcept, MeshDictionary
from biolit.domain.enums import EntityLabel
from biolit_evals.linking_scoring import score_linking
from biolit_evals.mesh_gold import GoldMention

CHEM = EntityLabel.CHEMICAL


def _gold(text, ids):
    return GoldMention(pmid="1", start=0, end=len(text), text=text, label=CHEM, mesh_ids=ids)


def _linker():
    met = MeshConcept(id="MESH:D008687", name="Metformin")
    asp = MeshConcept(id="MESH:D001241", name="Aspirin")
    ambig_a = MeshConcept(id="MESH:D000001", name="A")
    ambig_b = MeshConcept(id="MESH:D000002", name="B")
    return DictionaryLinker(
        MeshDictionary(
            {
                "metformin": [AliasEntry(met, True)],
                "aspirin": [AliasEntry(asp, True)],
                "ambig": [AliasEntry(ambig_a, False), AliasEntry(ambig_b, False)],
            }
        )
    )


def test_score_linking_computes_pr_f1_nil_and_tiebreak():
    gold = [
        _gold("Metformin", ("MESH:D008687",)),  # correct
        _gold("Aspirin", ("MESH:D999999",)),  # linked but wrong id
        _gold("Unknownium", ("MESH:D111111",)),  # NIL
        _gold("ambig", ("MESH:D000001",)),  # correct via tiebreak (smallest id)
        _gold("Unlinkable", ()),  # excluded from the eval set entirely
    ]
    m = score_linking(gold, _linker())
    assert m.n == 4  # the unlinkable gold mention is dropped
    assert m.correct == 2  # Metformin + ambig
    assert m.linked == 3  # Metformin, Aspirin, ambig (Unknownium is NIL)
    assert m.precision == pytest.approx(2 / 3)
    assert m.recall == pytest.approx(2 / 4)
    assert m.f1 == pytest.approx(2 * (2 / 3) * 0.5 / ((2 / 3) + 0.5))
    assert m.nil_rate == pytest.approx(1 / 4)
    assert m.tiebreak_rate == pytest.approx(1 / 4)
```

- [ ] **Step 2: Run to verify it fails**

Run: `cd backend && uv run pytest tests/evals/test_linking_scoring.py -v`
Expected: FAIL (`ModuleNotFoundError: biolit_evals.linking_scoring`).

- [ ] **Step 3: Implement `linking_scoring.py`** — create `backend/src/biolit_evals/linking_scoring.py`:

```python
from dataclasses import dataclass

from biolit.canon.linker import Linker
from biolit_evals.mesh_gold import GoldMention


@dataclass(frozen=True)
class LinkingMetrics:
    n: int
    correct: int
    linked: int
    tiebroken: int
    precision: float
    recall: float
    f1: float
    nil_rate: float
    tiebreak_rate: float


def score_linking(gold: list[GoldMention], linker: Linker) -> LinkingMetrics:
    """Score linking on gold *mentions* (surface fed straight to the linker), isolating
    linking quality from NER. Evaluated on linkable gold only (mesh_ids non-empty)."""
    items = [g for g in gold if g.mesh_ids]
    n = len(items)
    correct = linked = tiebroken = 0
    for g in items:
        r = linker.link(g.text)
        if r.tiebroken:
            tiebroken += 1
        if r.concept is not None:
            linked += 1
            if r.concept.id in g.mesh_ids:
                correct += 1
    precision = correct / linked if linked else 0.0
    recall = correct / n if n else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    nil_rate = (n - linked) / n if n else 0.0
    tiebreak_rate = tiebroken / n if n else 0.0
    return LinkingMetrics(
        n=n,
        correct=correct,
        linked=linked,
        tiebroken=tiebroken,
        precision=precision,
        recall=recall,
        f1=f1,
        nil_rate=nil_rate,
        tiebreak_rate=tiebreak_rate,
    )
```

- [ ] **Step 4: Run to verify pass**

Run: `cd backend && uv run pytest tests/evals/test_linking_scoring.py -v`
Expected: PASS.

- [ ] **Step 5: Add the empty-set guard test via Edit** — append to `backend/tests/evals/test_linking_scoring.py`:

```python
def test_score_linking_empty_set_is_all_zero():
    m = score_linking([], _linker())
    assert (m.n, m.correct, m.linked) == (0, 0, 0)
    assert (m.precision, m.recall, m.f1, m.nil_rate, m.tiebreak_rate) == (0.0, 0.0, 0.0, 0.0, 0.0)
```

- [ ] **Step 6: Run + gate**

Run: `cd backend && uv run pytest tests/evals/test_linking_scoring.py -v && uv run ruff check . && uv run pyright`
Expected: PASS (2 tests), clean.

- [ ] **Step 7: Commit**

```bash
git add backend/src/biolit_evals/linking_scoring.py backend/tests/evals/test_linking_scoring.py
git commit -m "feat(evals): linking P/R/F1 with NIL rate + ambiguous-tiebreak rate"
```

---

### Task 9: Eval runner, run log, CTD builder + BC5CDR gold downloader (heavy)

**Files:**
- Create: `backend/src/biolit_evals/canon_eval.py`
- Create: `backend/src/biolit/canon/build_mesh.py`
- Modify: `backend/src/biolit/config.py`
- Modify: `backend/.gitignore` (create if absent)
- Create: `backend/tests/evals/test_canon_eval.py`
- Create: `backend/tests/evals/test_canon_smoke.py` (heavy-marked)

**Interfaces:**
- Consumes: `score_linking` / `LinkingMetrics` (Task 8), `GoldMention` (Task 7), `Linker` (Task 5), `MeshDictionary` (Task 3).
- Produces:
  - `run_canon_eval(*, gold: list[GoldMention], linker: Linker, dataset: str, artifact_source: str, log_path: str, git_sha: str, now: str) -> LinkingMetrics` — scores, then appends one JSON line (fixed 12-key schema below) to `log_path`, creating its parent dir. Injection mirrors `run_eval`.
  - Log schema keys (exact): `timestamp, git_sha, dataset, artifact_source, n, correct, linked, precision, recall, f1, nil_rate, tiebreak_rate`.
  - `build_mesh.py` CLI: `python -m biolit.canon.build_mesh` downloads CTD vocabularies, builds the artifact, saves to `settings.mesh_artifact_path`. Heavy/manual.
  - `canon_eval.main()` CLI: `python -m biolit_evals.canon_eval --dataset {bc5cdr,domain}`. Heavy/manual.
- Config additions: `mesh_artifact_path: str = "data/canon/mesh_aliases.json.gz"`, `ctd_chemicals_url: str = "https://ctdbase.org/reports/CTD_chemicals.tsv.gz"`, `ctd_diseases_url: str = "https://ctdbase.org/reports/CTD_diseases.tsv.gz"`, `bc5cdr_cdr_zip_url: str = "https://huggingface.co/datasets/bigbio/bc5cdr/resolve/main/CDR_Data.zip"`.

- [ ] **Step 1: Write the failing runner test** — create `backend/tests/evals/test_canon_eval.py`:

```python
import json

from biolit.canon.linker import DictionaryLinker
from biolit.canon.mesh import AliasEntry, MeshConcept, MeshDictionary
from biolit.domain.enums import EntityLabel
from biolit_evals.canon_eval import run_canon_eval
from biolit_evals.mesh_gold import GoldMention

CHEM = EntityLabel.CHEMICAL

_EXPECTED_KEYS = {
    "timestamp",
    "git_sha",
    "dataset",
    "artifact_source",
    "n",
    "correct",
    "linked",
    "precision",
    "recall",
    "f1",
    "nil_rate",
    "tiebreak_rate",
}


def test_run_canon_eval_scores_and_appends_one_log_line(tmp_path):
    met = MeshConcept(id="MESH:D008687", name="Metformin")
    linker = DictionaryLinker(MeshDictionary({"metformin": [AliasEntry(met, True)]}))
    gold = [
        GoldMention(pmid="1", start=0, end=9, text="Metformin", label=CHEM, mesh_ids=("MESH:D008687",))
    ]
    log = tmp_path / "nested" / "canon_runs.jsonl"  # parent must be created
    metrics = run_canon_eval(
        gold=gold,
        linker=linker,
        dataset="fixture",
        artifact_source="fixture",
        log_path=str(log),
        git_sha="abc1234",
        now="2026-07-22T00:00:00+00:00",
    )
    assert metrics.correct == 1
    lines = log.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    assert set(json.loads(lines[0]).keys()) == _EXPECTED_KEYS
```

- [ ] **Step 2: Run to verify it fails**

Run: `cd backend && uv run pytest tests/evals/test_canon_eval.py -v`
Expected: FAIL (`ModuleNotFoundError: biolit_evals.canon_eval`).

- [ ] **Step 3: Implement the runner** — create `backend/src/biolit_evals/canon_eval.py`:

```python
import argparse
import json
import subprocess
from datetime import UTC, datetime
from pathlib import Path

from biolit.canon.linker import DictionaryLinker, Linker
from biolit_evals.linking_scoring import LinkingMetrics, score_linking
from biolit_evals.mesh_gold import GoldMention

DEFAULT_LOG = "evals/canon_runs.jsonl"
DOMAIN_NORM_GOLD = "evals/gold/domain_normalization_sample.jsonl"


def run_canon_eval(
    *,
    gold: list[GoldMention],
    linker: Linker,
    dataset: str,
    artifact_source: str,
    log_path: str,
    git_sha: str,
    now: str,
) -> LinkingMetrics:
    """Score linking on `gold` and append one JSON line to `log_path`. All impure inputs
    (gold, linker, log_path, git_sha, now) are injected so this stays offline-testable."""
    metrics = score_linking(gold, linker)
    line = {
        "timestamp": now,
        "git_sha": git_sha,
        "dataset": dataset,
        "artifact_source": artifact_source,
        "n": metrics.n,
        "correct": metrics.correct,
        "linked": metrics.linked,
        "precision": metrics.precision,
        "recall": metrics.recall,
        "f1": metrics.f1,
        "nil_rate": metrics.nil_rate,
        "tiebreak_rate": metrics.tiebreak_rate,
    }
    Path(log_path).parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(line) + "\n")
    return metrics


def _git_sha() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"]).decode().strip()
    except Exception:  # best-effort metadata; never fail an eval over it
        return "unknown"


def main(argv: list[str] | None = None) -> None:
    # Heavy imports (artifact load, gold download) are local so importing this module for
    # `run_canon_eval` stays cheap and offline.
    from biolit.canon.mesh import MeshDictionary
    from biolit.config import get_settings
    from biolit_evals.mesh_gold import load_domain_norm_sample
    from biolit_evals.mesh_gold_download import load_bc5cdr_norm_gold

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["bc5cdr", "domain"], required=True)
    args = parser.parse_args(argv)

    settings = get_settings()
    linker = DictionaryLinker(MeshDictionary.from_artifact(settings.mesh_artifact_path))
    if args.dataset == "bc5cdr":
        gold = load_bc5cdr_norm_gold(settings.bc5cdr_cdr_zip_url)
    else:
        gold = load_domain_norm_sample(DOMAIN_NORM_GOLD)

    metrics = run_canon_eval(
        gold=gold,
        linker=linker,
        dataset=args.dataset,
        artifact_source=settings.mesh_artifact_path,
        log_path=DEFAULT_LOG,
        git_sha=_git_sha(),
        now=datetime.now(UTC).isoformat(),
    )
    print(
        f"{args.dataset}: P={metrics.precision:.4f} R={metrics.recall:.4f} F1={metrics.f1:.4f} "
        f"(correct={metrics.correct}/{metrics.n}, linked={metrics.linked}, "
        f"NIL={metrics.nil_rate:.3f}, tiebreak={metrics.tiebreak_rate:.3f})"
    )


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run to verify pass**

Run: `cd backend && uv run pytest tests/evals/test_canon_eval.py -v`
Expected: PASS. (The `main()` heavy imports are not exercised by this test.)

- [ ] **Step 5: Add config settings** — in `backend/src/biolit/config.py`, add to `Settings` (after `ner_score_threshold`):

```python
    mesh_artifact_path: str = "data/canon/mesh_aliases.json.gz"
    ctd_chemicals_url: str = "https://ctdbase.org/reports/CTD_chemicals.tsv.gz"
    ctd_diseases_url: str = "https://ctdbase.org/reports/CTD_diseases.tsv.gz"
    bc5cdr_cdr_zip_url: str = "https://huggingface.co/datasets/bigbio/bc5cdr/resolve/main/CDR_Data.zip"
```

- [ ] **Step 6: Gitignore the data dir** — ensure `backend/.gitignore` contains these lines (append if the file exists, create it with these lines otherwise):

```
data/
*.tsv.gz
CDR_Data.zip
```

- [ ] **Step 7: Implement the CTD builder (heavy, not unit-tested)** — create `backend/src/biolit/canon/build_mesh.py`:

```python
import csv
import gzip
import io
from pathlib import Path

import httpx

from biolit.canon.mesh import MeshDictionary, build_alias_table
from biolit.config import get_settings


def _download_tsv_gz(url: str) -> list[list[str]]:
    """Download a gzipped CTD TSV and return its data rows (comment lines dropped)."""
    resp = httpx.get(url, follow_redirects=True, timeout=120.0)
    resp.raise_for_status()
    text = gzip.decompress(resp.content).decode("utf-8")
    reader = csv.reader(io.StringIO(text), delimiter="\t")
    return [row for row in reader if row and not row[0].startswith("#")]


def main() -> None:
    settings = get_settings()
    print("Downloading CTD chemicals + diseases ...")
    chem_rows = _download_tsv_gz(settings.ctd_chemicals_url)
    disease_rows = _download_tsv_gz(settings.ctd_diseases_url)
    table = build_alias_table(chem_rows, disease_rows)
    out = Path(settings.mesh_artifact_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    MeshDictionary(table).save_artifact(str(out))
    print(f"Wrote {len(table)} aliases to {out}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 8: Implement the BC5CDR gold downloader (heavy)** — create `backend/src/biolit_evals/mesh_gold_download.py`:

```python
import io
import zipfile

import httpx

from biolit_evals.mesh_gold import GoldMention, parse_pubtator

# PubTator file inside CDR_Data.zip (BioCreative V CDR corpus, test split).
_TEST_MEMBER = "CDR_Data/CDR.Corpus.v010516/CDR_TestSet.PubTator.txt"


def load_bc5cdr_norm_gold(zip_url: str, member: str = _TEST_MEMBER) -> list[GoldMention]:
    """Download CDR_Data.zip (ungated bigbio/bc5cdr mirror) and parse its test-split
    PubTator file into gold mentions carrying MeSH IDs. Heavy/manual (network + ~20 MB)."""
    resp = httpx.get(zip_url, follow_redirects=True, timeout=300.0)
    resp.raise_for_status()
    with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
        pubtator = zf.read(member).decode("utf-8")
    return parse_pubtator(pubtator)
```

- [ ] **Step 9: Add heavy-marked real-data smoke tests** — create `backend/tests/evals/test_canon_smoke.py` (mirrors the Phase 2 `heavy` real-model smoke; deselected by the default `-m 'not heavy'` addopts):

```python
import pytest

from biolit.config import get_settings


@pytest.mark.heavy
def test_real_ctd_dictionary_links_known_chemical():
    from biolit.canon.build_mesh import _download_tsv_gz
    from biolit.canon.linker import DictionaryLinker
    from biolit.canon.mesh import MeshDictionary, build_alias_table

    s = get_settings()
    chem = _download_tsv_gz(s.ctd_chemicals_url)
    dis = _download_tsv_gz(s.ctd_diseases_url)
    linker = DictionaryLinker(MeshDictionary(build_alias_table(chem, dis)))
    result = linker.link("Metformin")
    assert result.concept is not None and result.concept.id == "MESH:D008687"


@pytest.mark.heavy
def test_real_bc5cdr_gold_parses_nonempty_with_mesh_ids():
    from biolit_evals.mesh_gold_download import load_bc5cdr_norm_gold

    gold = load_bc5cdr_norm_gold(get_settings().bc5cdr_cdr_zip_url)
    assert len(gold) > 1000  # BC5CDR test split has thousands of mentions
    assert any(g.mesh_ids for g in gold)  # at least some are linkable
```

Run (to confirm they are collected but deselected by default): `cd backend && uv run pytest tests/evals/test_canon_smoke.py -v`
Expected: `2 deselected` (they only run with `uv run pytest -m heavy`).

- [ ] **Step 10: Run gate on everything** — the two download modules are imported only inside `main()` / heavy paths, so the default test run must stay green:

Run: `cd backend && uv run ruff check . && uv run ruff format --check . && uv run pyright && uv run pytest -q`
Expected: PASS (all fast tests), clean ruff/pyright. `httpx` is already a project dependency (verify with `uv run python -c "import httpx"`).

- [ ] **Step 11: Commit**

```bash
git add backend/src/biolit_evals/canon_eval.py backend/src/biolit_evals/mesh_gold_download.py backend/src/biolit/canon/build_mesh.py backend/src/biolit/config.py backend/.gitignore backend/tests/evals/test_canon_eval.py backend/tests/evals/test_canon_smoke.py
git commit -m "feat(evals): canon eval runner + run log, CTD builder + BC5CDR gold downloader (heavy)"
```

---

### Task 10: Real run, blind domain sample, and documentation

This task produces the first real numbers and documents methodology. The real download/build steps are run manually by the implementer (they need network) and their outputs are NOT committed (gitignored). Only the blind-annotated gold sample and the docs are committed.

**Files:**
- Create: `backend/evals/gold/domain_normalization_sample.jsonl` (blind-annotated)
- Modify: `docs/EVAL_REPORT.md`
- Modify: `docs/ARCHITECTURE.md`

**Interfaces:**
- Consumes: everything above via the two CLIs.

- [ ] **Step 1: Build the MeSH artifact (manual, heavy)**

Run: `cd backend && uv run python -m biolit.canon.build_mesh`
Expected: prints `Wrote <N> aliases to data/canon/mesh_aliases.json.gz` with N in the hundreds of thousands. If the CTD host rate-limits, retry; do not fabricate an artifact.

- [ ] **Step 2: Run the benchmark eval (manual, heavy)**

Run: `cd backend && uv run python -m biolit_evals.canon_eval --dataset bc5cdr`
Expected: prints P/R/F1, correct/n, NIL rate, tiebreak rate, and appends one line to `evals/canon_runs.jsonl`. Record these numbers for the report.

- [ ] **Step 3: Build the blind domain normalization sample**

Annotate `backend/evals/gold/domain_normalization_sample.jsonl` from the SAME abstracts used by `evals/gold/domain_sample.jsonl` (reuse their pmids/text). For each CHEMICAL/DISEASE mention, record the correct MeSH ID. **Methodology (ADR-0006, and state it in the eval report):** annotate **blind** — meaning without viewing `DictionaryLinker`'s output — but annotators **may and should consult the MeSH thesaurus/browser** (https://meshb.nlm.nih.gov/) to find the correct concept IDs. "Blind" here means *not anchored to the system's predictions*, NOT annotating with zero reference tools. One JSON object per line:

```
{"pmid": "<real pmid>", "text": "<sentence>", "entities": [{"start": <int>, "end": <int>, "label": "CHEMICAL", "text": "<surface>", "mesh_id": "MESH:D008687"}]}
```

Use `mesh_id` `"-1"` for a mention with no correct MeSH concept. Do not fabricate IDs — verify each against the MeSH browser.

- [ ] **Step 4: Validate the sample loads and run the domain eval (manual, heavy)**

Run: `cd backend && uv run python -c "from biolit_evals.mesh_gold import load_domain_norm_sample; print(len(load_domain_norm_sample('evals/gold/domain_normalization_sample.jsonl')), 'mentions')"`
Then: `cd backend && uv run python -m biolit_evals.canon_eval --dataset domain`
Expected: both succeed; record the domain P/R/F1/NIL/tiebreak numbers.

- [ ] **Step 5: Write the eval-report section** — append a `## Entity canonicalization (NEN)` section to `docs/EVAL_REPORT.md` containing: the benchmark and domain metrics tables (P/R/F1, NIL rate, tiebreak rate, n); a one-line statement that linking is scored on **gold mentions** to isolate it from NER; the **blind-methodology paragraph verbatim** from Step 3 (blind = not anchored to linker output; MeSH browser consultation is expected); a note that the CTD/MeSH artifact and BC5CDR gold are downloaded, not committed (reproduce via the two CLIs); and, if `nil_rate` is high, an explicit "the SapBERT fallback (deferred, ADR-0008) is warranted iff this NIL rate is unacceptable" line so the deferred decision is data-anchored.

- [ ] **Step 6: Update the architecture doc** — add a `## Canonicalization layer (Phase 3)` section to `docs/ARCHITECTURE.md`: `biolit.canon.canonicalize(entities, text)` links spans to MeSH via a CTD dictionary behind the fallback-agnostic `Linker` protocol, sits between `extract_entities` and clustering, and is evaluated by `biolit_evals.canon_eval`. Note the offline artifact build (`python -m biolit.canon.build_mesh`).

- [ ] **Step 7: Gate + commit** (only the sample and docs are committed; gitignored artifacts/downloads are not)

```bash
cd backend && uv run pytest -q && cd ..
git add backend/evals/gold/domain_normalization_sample.jsonl docs/EVAL_REPORT.md docs/ARCHITECTURE.md
git commit -m "docs(eval): canonicalization results + blind normalization methodology"
```

- [ ] **Step 8: Verify nothing gitignored was staged**

Run: `git status --porcelain` and confirm `data/`, `*.tsv.gz`, `CDR_Data.zip`, and `evals/canon_runs.jsonl` are NOT listed as tracked/added.

---

## Notes for the executor

- **Deferred (do NOT build):** the SapBERT/embedding fallback linker — the `Linker` protocol seam is enough; add a second implementation only if Task 10's NIL rate justifies it.
- **Deferred (separate sub-project):** the clustering step that consumes `canonical_id` (with `nil:<normalized-surface>` fallback keys) — this plan only guarantees clustering *can* key on `canonical_id`.
- If any design detail is decided during implementation (e.g. an additional normalization rule), add an ADR addendum in `docs/DECISIONS.md` rather than silently diverging from this plan or the spec.
