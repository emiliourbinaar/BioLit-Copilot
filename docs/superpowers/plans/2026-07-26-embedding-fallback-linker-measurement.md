# Embedding Fallback Linker Measurement — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Produce the threshold curve showing what a real embedding-based fallback linker costs in precision across the full 3209-mention NIL population, so the decision to build or reject it rests on measured false-positive cost rather than a recall-only ceiling.

**Architecture:** The dictionary pipeline runs once to collect every NIL mention. Each NIL surface is scored once against a brute-force cosine index over all 551,669 aliases (SapBERT, and a char n-gram TF-IDF control), yielding one top-1 candidate + score per mention per arm. The threshold sweep is then **pure set arithmetic** over those cached candidates — no model or pipeline re-run per threshold.

**Tech Stack:** Python 3.12, uv, PyTorch (CPU-pinned), transformers, scikit-learn + scipy (new), numpy, pytest.

## Global Constraints

- All commands run from `backend/` via `uv run`.
- Ruff ruleset `E,F,I,UP,B`; line length 100. Pyright must report 0 errors.
- CPU-pin guard: `grep -ciE '^name = "(nvidia|triton)' uv.lock` MUST return 0 after any dependency change.
- Never modify `pyproject.toml` for tdd-guard config; never set `tdd_guard_project_root` there.
- Never hand-edit anything under `.claude/tdd-guard/`.
- StrEnum for string enums (ADR-0005). `datetime.now(UTC)`, never `timezone.utc`.
- `backend/data/` is gitignored — never commit artifacts or downloads. Run logs (`evals/*.jsonl`) ARE committed.
- Never fabricate gold MeSH ids, PMIDs, or abstract text. Synthetic ids are permitted **only** in unit fixtures that never enter scoring, and must be commented as synthetic.
- **tdd-guard is active.** Write exactly ONE test at a time, watch it fail, then implement. On an import error the guard requires an empty stub first. Do not add untested display/printing code.

## Deviations from the spec (deliberate, flagged)

1. **Sweep lives in a new `biolit_evals/fallback_sweep.py`, not in `end_to_end.py`.** `end_to_end.py` owns the permanent 23-key run-log schema used as a longitudinal record; a one-shot 4-arm measurement must not bloat it. The sweep reuses `concept_counts` and `metrics_from_counts` from `end_to_end` and writes its own `evals/fallback_sweep.jsonl`.
2. **The measurement does not instantiate a `Linker`.** Building one would force a full pipeline re-run per threshold. Scoring each NIL surface once and sweeping arithmetically is exactly equivalent in semantics, and preserves the NIL-only blast radius. A real `Linker` implementation is build-phase work, only if the measurement justifies it.

## File Structure

| File | Responsibility |
|---|---|
| `src/biolit/canon/mesh.py` (modify) | `build_concept_labels`, `_ctd_concept_id` helper, `LabelMapDiagnostics`, `diagnose_label_map` |
| `src/biolit/canon/build_mesh.py` (modify) | Write second artifact; halt on orphans |
| `src/biolit/config.py` (modify) | Paths + model settings |
| `src/biolit_evals/candidate_index.py` (new) | `Candidate`, `LabelFilter`, `best_concept` — max-over-aliases + label filter. Pure, model-free |
| `src/biolit_evals/tfidf_index.py` (new) | Char n-gram TF-IDF scorer (control arm) |
| `src/biolit_evals/embedding_index.py` (new) | SapBERT encoder + brute-force cosine |
| `src/biolit_evals/fallback_sweep.py` (new) | NIL collection, breakpoint sweep, mention counters, run log |

---

### Task 1: Concept→label map with orphan and multi-label diagnostics

**Files:**
- Modify: `backend/src/biolit/canon/mesh.py`
- Test: `backend/tests/canon/test_concept_labels.py` (create)

**Interfaces:**
- Consumes: existing `_read_ctd_dump`, `AliasEntry`, `MeshConcept` in `mesh.py`.
- Produces:
  - `build_concept_labels(chem_text: str, disease_text: str) -> dict[str, set[EntityLabel]]`
  - `LabelMapDiagnostics(n_concepts: int, n_multi_label: int, orphans: tuple[str, ...])`
  - `diagnose_label_map(aliases: dict[str, list[AliasEntry]], labels: dict[str, set[EntityLabel]]) -> LabelMapDiagnostics`
  - `_ctd_concept_id(raw_id: str) -> str`

Fixtures are **inline TSV strings**, not the shared files in `tests/canon/fixtures/`. Existing tests assert on those files' contents; adding an overlap row would perturb them. Inline text also documents the overlap case at its point of use.

- [ ] **Step 1: Write the failing test (ONE test only — tdd-guard)**

Create `backend/tests/canon/test_concept_labels.py`:

```python
from biolit.canon.mesh import build_concept_labels
from biolit.domain.enums import EntityLabel

# Minimal CTD-shaped dumps. MESH:D000000 is a SYNTHETIC id used only to exercise the
# both-dumps overlap path; it never enters scoring. The real overlap count comes from
# the build diagnostic run against real CTD, not from this fixture.
_CHEM = (
    "# ChemicalName\tChemicalID\tMESHSynonyms\n"
    "Metformin\tMESH:D008687\tGlucophage\n"
    "Overlap Agent\tMESH:D000000\t\n"
)
_DIS = (
    "# DiseaseName\tDiseaseID\tSynonyms\n"
    "Diabetes Mellitus\tMESH:D003920\t\n"
    "Overlap Agent\tMESH:D000000\t\n"
)


def test_chemical_and_disease_concepts_get_their_source_label():
    labels = build_concept_labels(_CHEM, _DIS)
    assert labels["MESH:D008687"] == {EntityLabel.CHEMICAL}
    assert labels["MESH:D003920"] == {EntityLabel.DISEASE}
```

- [ ] **Step 2: Run it and confirm it fails**

Run: `uv run pytest tests/canon/test_concept_labels.py -v`
Expected: `ImportError: cannot import name 'build_concept_labels'`.

Per tdd-guard, an import error requires an **empty stub first**. Add to `mesh.py`:

```python
def build_concept_labels(chem_text: str, disease_text: str) -> dict[str, "set[EntityLabel]"]:
    return {}
```

with `from biolit.domain.enums import EntityLabel` at the top. Re-run: expect `KeyError: 'MESH:D008687'`.

- [ ] **Step 3: Implement**

First extract the id-prefixing rule so it cannot drift between the two builders. In `mesh.py`, add:

```python
def _ctd_concept_id(raw_id: str) -> str:
    """CTD ids already carry their MESH:/OMIM: prefix; only prefix a bare accession."""
    return raw_id if ":" in raw_id else f"MESH:{raw_id}"
```

and replace the equivalent inline expression inside `_ingest_ctd` with a call to it.

Then implement:

```python
def build_concept_labels(chem_text: str, disease_text: str) -> dict[str, set[EntityLabel]]:
    """Map concept id -> the set of CTD dumps it appears in.

    A SET, not a scalar: a concept present in both dumps carries both labels, and
    last-write-wins would hide that. Columns resolve by name, as in build_alias_table.
    """
    labels: dict[str, set[EntityLabel]] = {}
    for text, label, name_col, id_col in (
        (chem_text, EntityLabel.CHEMICAL, "ChemicalName", "ChemicalID"),
        (disease_text, EntityLabel.DISEASE, "DiseaseName", "DiseaseID"),
    ):
        header, rows = _read_ctd_dump(text)
        if not header:
            raise ValueError(f"CTD header row not found (expected a '# {name_col}...' line)")
        idx = {col: i for i, col in enumerate(header)}
        name_i, id_i = idx[name_col], idx[id_col]
        for row in rows:
            if len(row) <= max(name_i, id_i):
                continue
            if not row[name_i].strip() or not row[id_i].strip():
                continue
            labels.setdefault(_ctd_concept_id(row[id_i].strip()), set()).add(label)
    return labels
```

- [ ] **Step 4: Run and confirm PASS**

Run: `uv run pytest tests/canon/test_concept_labels.py -v` → PASS.
Then `uv run pytest tests/canon -q` to confirm the `_ctd_concept_id` extraction broke nothing.

- [ ] **Step 5: Write the overlap test (one test)**

```python
def test_a_concept_in_both_dumps_carries_both_labels():
    # Last-write-wins would silently return a single label here.
    labels = build_concept_labels(_CHEM, _DIS)
    assert labels["MESH:D000000"] == {EntityLabel.CHEMICAL, EntityLabel.DISEASE}
```

- [ ] **Step 6: Run — expect PASS immediately**

The set semantics already satisfy it. This is a **characterization test** pinning behavior that a future refactor to a scalar would break. Note that in the commit message rather than pretending it was RED.

- [ ] **Step 7: Write the orphan-detection test (one test)**

```python
from biolit.canon.mesh import AliasEntry, MeshConcept, diagnose_label_map


def test_a_concept_missing_from_the_label_map_is_reported_as_an_orphan():
    # Built deliberately orphaned so the diagnostic is PROVEN to fire. Expected zero on
    # real data, but "expected zero" is exactly the assumption this project has twice
    # found to be false.
    aliases = {"metformin": [AliasEntry(MeshConcept("MESH:D008687", "Metformin"), True)]}
    diag = diagnose_label_map(aliases, {})
    assert diag.orphans == ("MESH:D008687",)
    assert diag.n_concepts == 1
```

- [ ] **Step 8: Run, confirm failure, implement**

Expected: `ImportError` → add empty stub returning `LabelMapDiagnostics(0, 0, ())` → re-run → assertion failure → implement:

```python
@dataclass(frozen=True)
class LabelMapDiagnostics:
    n_concepts: int
    n_multi_label: int
    orphans: tuple[str, ...]


def diagnose_label_map(
    aliases: dict[str, list[AliasEntry]], labels: dict[str, set[EntityLabel]]
) -> LabelMapDiagnostics:
    """Verify the label map covers the alias table.

    An orphan is not cosmetic: LabelFilter refuses concepts absent from the map, so an
    orphan silently drops a legitimate candidate from the label-constrained arms and
    makes constraint look worse than it is. Both artifacts derive from the same CTD text
    in the same run, so any orphan means the derivation is misaligned.
    """
    concept_ids = {e.concept.id for entries in aliases.values() for e in entries}
    return LabelMapDiagnostics(
        n_concepts=len(concept_ids),
        n_multi_label=sum(1 for v in labels.values() if len(v) > 1),
        orphans=tuple(sorted(concept_ids - labels.keys())),
    )
```

- [ ] **Step 9: Full gate and commit**

```bash
uv run pytest -q && uv run ruff check . && uv run ruff format --check . && uv run pyright
git add src/biolit/canon/mesh.py tests/canon/test_concept_labels.py
git commit -F - <<'EOF'
feat(canon): concept->label map with orphan and multi-label diagnostics

Set-valued, not scalar: a concept in both CTD dumps carries both labels, and
last-write-wins would hide the overlap. Extracts _ctd_concept_id so the id
prefixing rule cannot drift between the alias and label builders.

diagnose_label_map reports orphans -- alias-table concepts with no label entry.
Expected zero, but verified: LabelFilter refuses unmapped concepts, so an orphan
would silently drop candidates from the label-constrained arms and make label
constraint look worse than it is.

The overlap test passed on first run (set semantics already correct); it is a
characterization test pinning behavior a scalar refactor would break, not a RED
cycle.
EOF
```

---

### Task 2: Persist the label map from the same CTD download

**Files:**
- Modify: `backend/src/biolit/canon/build_mesh.py`
- Modify: `backend/src/biolit/config.py`
- Test: `backend/tests/test_canon_config.py` (extend)

**Interfaces:**
- Consumes: `build_concept_labels`, `diagnose_label_map` (Task 1).
- Produces: `data/canon/concept_labels.json.gz`; `Settings.concept_labels_path: str`.

Both artifacts come from **one** download so they cannot drift.

- [ ] **Step 1: Write the failing config test**

Append to `backend/tests/test_canon_config.py`:

```python
def test_concept_labels_path_defaults_alongside_the_alias_artifact():
    assert get_settings().concept_labels_path == "data/canon/concept_labels.json.gz"
```

- [ ] **Step 2: Run, confirm AttributeError**

Run: `uv run pytest tests/test_canon_config.py -v`

- [ ] **Step 3: Add the setting**

In `config.py`, directly below `mesh_artifact_path`:

```python
    concept_labels_path: str = "data/canon/concept_labels.json.gz"
```

- [ ] **Step 4: Run, confirm PASS**

- [ ] **Step 5: Wire the builder (no new test — exercised by the heavy smoke run)**

Rewrite `main()` in `build_mesh.py`:

```python
def main() -> None:
    settings = get_settings()
    print("Downloading CTD chemicals + diseases ...")
    chem_text = _download_ctd(settings.ctd_chemicals_url)
    disease_text = _download_ctd(settings.ctd_diseases_url)

    table = build_alias_table(chem_text, disease_text)
    labels = build_concept_labels(chem_text, disease_text)

    diag = diagnose_label_map(table, labels)
    print(
        f"concepts={diag.n_concepts} multi_label={diag.n_multi_label} "
        f"orphans={len(diag.orphans)}"
    )
    if diag.orphans:
        raise SystemExit(
            f"{len(diag.orphans)} alias-table concept(s) have no label entry, "
            f"e.g. {diag.orphans[:5]}. Both artifacts derive from the same CTD text in "
            "this run, so this means the derivation is misaligned. Halting rather than "
            "silently dropping candidates from the label-constrained arms."
        )

    out = Path(settings.mesh_artifact_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    MeshDictionary(table).save_artifact(str(out))

    labels_out = Path(settings.concept_labels_path)
    with gzip.open(labels_out, "wt", encoding="utf-8") as fh:
        json.dump({cid: sorted(v.value for v in labs) for cid, labs in labels.items()}, fh)
    print(f"Wrote {len(table)} aliases to {out} and {len(labels)} labels to {labels_out}")
```

Add `import json` and the new imports at the top.

- [ ] **Step 6: Rebuild the artifacts and record the diagnostics**

```bash
uv run python -m biolit.canon.build_mesh
```

Record the printed `concepts= multi_label= orphans=` line — the multi-label count is a spec deliverable. **If `orphans` is nonzero the run halts; stop and report rather than working around it.**

- [ ] **Step 7: Gate and commit**

```bash
uv run pytest -q && uv run ruff check . && uv run pyright
git add src/biolit/canon/build_mesh.py src/biolit/config.py tests/test_canon_config.py
git commit -m "feat(canon): persist concept label map from the same CTD download"
```

---

### Task 3: Candidate selection — max over aliases, with label filter

**Files:**
- Create: `backend/src/biolit_evals/candidate_index.py`
- Test: `backend/tests/evals/test_candidate_index.py`

**Interfaces:**
- Produces:
  - `Candidate(concept_id: str, score: float)`
  - `LabelFilter(labels: dict[str, set[EntityLabel]], required: EntityLabel)` with `.admits(concept_id) -> bool`
  - `best_concept(scores: Sequence[float], alias_concept_ids: Sequence[str], *, label_filter: LabelFilter | None = None) -> Candidate | None`

Model-free and testable with a hand-built score vector. Taking the global argmax over admissible alias rows is identical to max-over-aliases-then-argmax-over-concepts, and is what both arms use.

- [ ] **Step 1: Write the failing test (ONE)**

```python
from biolit_evals.candidate_index import best_concept


def test_best_concept_takes_the_max_over_a_concepts_aliases_not_its_preferred_name():
    # D1's PREFERRED name scores poorly while its synonym scores best. Returning the
    # preferred-name row would pick D2 and pass a naive implementation.
    alias_concept_ids = ["MESH:D1", "MESH:D1", "MESH:D2"]
    scores = [0.10, 0.91, 0.55]
    assert best_concept(scores, alias_concept_ids) == Candidate("MESH:D1", 0.91)
```

Import `Candidate` alongside `best_concept`.

- [ ] **Step 2: Run, confirm ImportError; add empty stub**

Create `candidate_index.py` with stubs (guard requires a stub on import error):

```python
@dataclass(frozen=True)
class Candidate:
    concept_id: str
    score: float


def best_concept(scores, alias_concept_ids, *, label_filter=None):
    return None
```

Re-run: expect `assert None == Candidate(...)`.

- [ ] **Step 3: Implement**

```python
from collections.abc import Sequence
from dataclasses import dataclass

from biolit.domain.enums import EntityLabel


@dataclass(frozen=True)
class Candidate:
    concept_id: str
    score: float


@dataclass(frozen=True)
class LabelFilter:
    """Admits a concept only if it carries the mention's label.

    A concept ABSENT from the map is refused. That is why build_mesh halts on orphans:
    silent absence would look like a label mismatch and understate the constrained arms.
    """

    labels: dict[str, set[EntityLabel]]
    required: EntityLabel

    def admits(self, concept_id: str) -> bool:
        return self.required in self.labels.get(concept_id, set())


def best_concept(
    scores: Sequence[float],
    alias_concept_ids: Sequence[str],
    *,
    label_filter: LabelFilter | None = None,
) -> Candidate | None:
    """Top-1 concept for one query.

    Global argmax over admissible alias rows == max-over-aliases then argmax over
    concepts, since a concept's score IS its best alias score.
    """
    best: Candidate | None = None
    for score, concept_id in zip(scores, alias_concept_ids, strict=True):
        if label_filter is not None and not label_filter.admits(concept_id):
            continue
        if best is None or score > best.score:
            best = Candidate(concept_id, float(score))
    return best
```

- [ ] **Step 4: Run, confirm PASS**

- [ ] **Step 5: Write the label-filter test (one)**

```python
def test_label_filter_excludes_a_higher_scoring_candidate_of_the_wrong_label():
    alias_concept_ids = ["MESH:D1", "MESH:D2"]
    scores = [0.99, 0.40]
    f = LabelFilter({"MESH:D1": {EntityLabel.CHEMICAL}, "MESH:D2": {EntityLabel.DISEASE}},
                    EntityLabel.DISEASE)
    assert best_concept(scores, alias_concept_ids, label_filter=f) == Candidate("MESH:D2", 0.40)
```

- [ ] **Step 6: Run, confirm PASS; then write the unmapped-concept test (one)**

```python
def test_an_unmapped_concept_is_refused_by_the_label_filter():
    # The failure mode build_mesh's orphan halt exists to prevent.
    f = LabelFilter({}, EntityLabel.CHEMICAL)
    assert best_concept([0.99], ["MESH:D1"], label_filter=f) is None
```

- [ ] **Step 7: Gate and commit**

```bash
uv run pytest tests/evals/test_candidate_index.py -q && uv run ruff check . && uv run pyright
git add src/biolit_evals/candidate_index.py tests/evals/test_candidate_index.py
git commit -m "feat(evals): max-over-aliases candidate selection with label filter"
```

---

### Task 4: TF-IDF control arm

**Files:**
- Create: `backend/src/biolit_evals/tfidf_index.py`
- Modify: `backend/pyproject.toml` (dependencies only — NOT tdd-guard config)
- Test: `backend/tests/evals/test_tfidf_index.py`

**Interfaces:**
- Produces: `TfidfScorer(aliases: Sequence[str])` with `.scores(queries: Sequence[str]) -> np.ndarray` of shape `(len(queries), len(aliases))`.

This is the **control**: without it a SapBERT gain is unattributable between semantic generalization and fuzzy string matching.

- [ ] **Step 1: Add dependencies and verify the CPU pin**

In `pyproject.toml`, add to `[project].dependencies`:

```toml
    "scikit-learn>=1.5",
    "scipy>=1.13",
```

Then:

```bash
uv sync
grep -ciE '^name = "(nvidia|triton)' uv.lock
```

The grep MUST print `0`. If not, stop and report — do not proceed.

- [ ] **Step 2: Write the failing test (ONE)**

```python
from biolit_evals.tfidf_index import TfidfScorer


def test_char_ngrams_score_a_misspelling_closest_to_its_true_alias():
    scorer = TfidfScorer(["metformin", "aspirin", "polycystic ovary syndrome"])
    row = scorer.scores(["metformine"])[0]
    assert row.argmax() == 0
    assert row[0] > row[1]
```

- [ ] **Step 3: Run, confirm ImportError; add empty stub, then implement**

```python
from collections.abc import Sequence

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from biolit.canon.mesh import normalize_surface


class TfidfScorer:
    """Character n-gram TF-IDF over the alias table -- the non-neural control arm.

    Uses the SAME normalize_surface as the dictionary linker so the control differs from
    SapBERT only in the scoring function, not in preprocessing.
    """

    def __init__(self, aliases: Sequence[str]) -> None:
        self._vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 3))
        self._matrix = self._vec.fit_transform(normalize_surface(a) for a in aliases)

    def scores(self, queries: Sequence[str]) -> np.ndarray:
        q = self._vec.transform(normalize_surface(x) for x in queries)
        # TfidfVectorizer L2-normalizes rows, so the dot product IS cosine similarity.
        return np.asarray((q @ self._matrix.T).todense())
```

- [ ] **Step 4: Run, confirm PASS**

- [ ] **Step 5: Gate and commit**

```bash
uv run pytest tests/evals/test_tfidf_index.py -q && uv run ruff check . && uv run pyright
git add pyproject.toml uv.lock src/biolit_evals/tfidf_index.py tests/evals/test_tfidf_index.py
git commit -m "feat(evals): char n-gram TF-IDF control arm"
```

---

### Task 5: SapBERT encoder and throughput pilot

**Files:**
- Create: `backend/src/biolit_evals/embedding_index.py`
- Modify: `backend/src/biolit/config.py`
- Test: `backend/tests/evals/test_embedding_index.py`

**Interfaces:**
- Produces:
  - `encode(texts: Sequence[str], *, model_id: str, batch_size: int, device: str, max_length: int = 32) -> np.ndarray` — L2-normalized CLS embeddings, shape `(n, 768)`.
  - `cosine_scores(queries: np.ndarray, index: np.ndarray, *, chunk: int = 4096) -> np.ndarray`

- [ ] **Step 1: Add settings**

In `config.py`:

```python
    sapbert_model_id: str = "cambridgeltl/SapBERT-from-PubMedBERT-fulltext"
    sapbert_batch_size: int = 128
    sapbert_max_length: int = 32
    embedding_index_path: str = "data/canon/alias_embeddings.npy"
```

- [ ] **Step 2: Write the failing test for cosine_scores (ONE — pure numpy, no model)**

```python
import numpy as np

from biolit_evals.embedding_index import cosine_scores


def test_cosine_scores_chunking_matches_a_single_matmul():
    rng = np.random.default_rng(0)
    q = rng.normal(size=(5, 8)).astype(np.float32)
    idx = rng.normal(size=(37, 8)).astype(np.float32)
    q /= np.linalg.norm(q, axis=1, keepdims=True)
    idx /= np.linalg.norm(idx, axis=1, keepdims=True)
    # chunk deliberately does not divide 37, so an off-by-one in the loop shows up.
    np.testing.assert_allclose(cosine_scores(q, idx, chunk=10), q @ idx.T, rtol=1e-5)
```

- [ ] **Step 3: Run, confirm ImportError; stub; implement**

```python
def cosine_scores(queries: np.ndarray, index: np.ndarray, *, chunk: int = 4096) -> np.ndarray:
    """Exact brute-force cosine over an L2-normalized index, chunked over index rows.

    Exact, not ANN: the query side is a few thousand mentions so approximate search buys
    nothing, and it would inject approximation error into a measurement whose whole
    purpose is attributing error to the model.
    """
    out = np.empty((queries.shape[0], index.shape[0]), dtype=np.float32)
    for start in range(0, index.shape[0], chunk):
        block = index[start : start + chunk]
        out[:, start : start + block.shape[0]] = queries @ block.T
    return out
```

- [ ] **Step 4: Run, confirm PASS**

- [ ] **Step 5: Implement `encode` (no unit test — covered by the heavy pilot in Step 6)**

```python
def encode(
    texts: Sequence[str],
    *,
    model_id: str,
    batch_size: int,
    device: str,
    max_length: int = 32,
) -> np.ndarray:
    """L2-normalized [CLS] embeddings. CLS pooling is SapBERT's own inference procedure."""
    tok = AutoTokenizer.from_pretrained(model_id)
    model = AutoModel.from_pretrained(model_id).to(device).eval()
    out: list[np.ndarray] = []
    with torch.inference_mode():
        for start in range(0, len(texts), batch_size):
            batch = [normalize_surface(t) for t in texts[start : start + batch_size]]
            enc = tok(batch, padding=True, truncation=True,
                      max_length=max_length, return_tensors="pt").to(device)
            cls = model(**enc).last_hidden_state[:, 0, :]
            cls = torch.nn.functional.normalize(cls, p=2, dim=1)
            out.append(cls.cpu().numpy().astype(np.float32))
    return np.concatenate(out, axis=0)
```

- [ ] **Step 6: Run the throughput pilot — SPEC-REQUIRED GATE**

Add `backend/tests/evals/test_embedding_pilot.py`:

```python
import time

import pytest

from biolit.config import get_settings
from biolit_evals.embedding_index import encode


@pytest.mark.heavy
def test_pilot_throughput_on_5000_aliases():
    """Establishes the real encode rate BEFORE committing to the 551,669-alias build."""
    import gzip, json
    s = get_settings()
    with gzip.open(s.mesh_artifact_path, "rt", encoding="utf-8") as fh:
        aliases = list(json.load(fh).keys())[:5000]
    t0 = time.perf_counter()
    vecs = encode(aliases, model_id=s.sapbert_model_id,
                  batch_size=s.sapbert_batch_size, device="cpu",
                  max_length=s.sapbert_max_length)
    elapsed = time.perf_counter() - t0
    rate = len(aliases) / elapsed
    print(f"\n{rate:.0f} aliases/sec -> {551669 / rate / 60:.1f} min for the full table")
    assert vecs.shape == (5000, 768)
```

Run: `uv run pytest tests/evals/test_embedding_pilot.py -m heavy -s -v`

**Report the projected full-build time before continuing.** If it projects beyond ~60 minutes, stop and raise it as a scope question rather than absorbing it.

- [ ] **Step 7: Gate and commit**

```bash
uv run pytest -q && uv run ruff check . && uv run pyright
git add src/biolit_evals/embedding_index.py src/biolit/config.py tests/evals/test_embedding_index.py tests/evals/test_embedding_pilot.py
git commit -m "feat(evals): SapBERT encoder and exact chunked cosine, with throughput pilot"
```

---

### Task 6: The threshold sweep with mention-level counters

**Files:**
- Create: `backend/src/biolit_evals/fallback_sweep.py`
- Test: `backend/tests/evals/test_fallback_sweep.py`

**Interfaces:**
- Consumes: `Candidate` (Task 3); `concept_counts`, `metrics_from_counts` from `biolit_evals.end_to_end`.
- Produces:
  - `NilMention(pmid: str, surface: str, label: EntityLabel, gold_ids: frozenset[str], gold_aligned: bool)`
  - `SweepPoint(threshold, precision, recall, f1, tp, fp, fn, mentions_fired, mentions_correct, mentions_wrong, mentions_wrong_unaligned)`
  - `sweep(nils, candidates, doc_gold, doc_pred) -> list[SweepPoint]`

`doc_gold` / `doc_pred` are `dict[str, set[str]]` — pmid → concept ids — from the **baseline dictionary-only** run. `nils[i]` pairs with `candidates[i]`.

- [ ] **Step 1: Write the failing anchor test (ONE)**

```python
from biolit_evals.candidate_index import Candidate
from biolit_evals.fallback_sweep import NilMention, sweep
from biolit.domain.enums import EntityLabel

CHEM = EntityLabel.CHEMICAL


def _nil(pmid, surface, gold, aligned=True):
    return NilMention(pmid=pmid, surface=surface, label=CHEM,
                      gold_ids=frozenset(gold), gold_aligned=aligned)


def test_the_highest_threshold_fires_nothing_and_reproduces_the_baseline():
    # Anchor: a sweep whose top point does not reproduce baseline is a broken harness.
    nils = [_nil("1", "biguanide", {"MESH:D008687"})]
    cands = [Candidate("MESH:D008687", 0.80)]
    doc_gold = {"1": {"MESH:D008687"}}
    doc_pred = {"1": set()}
    top = sweep(nils, cands, doc_gold, doc_pred)[0]
    assert top.mentions_fired == 0
    assert (top.tp, top.fp, top.fn) == (0, 0, 1)
```

- [ ] **Step 2: Run, confirm ImportError; stub; implement**

```python
@dataclass(frozen=True)
class NilMention:
    pmid: str
    surface: str
    label: EntityLabel
    gold_ids: frozenset[str]
    gold_aligned: bool


@dataclass(frozen=True)
class SweepPoint:
    threshold: float
    precision: float
    recall: float
    f1: float
    tp: int
    fp: int
    fn: int
    mentions_fired: int
    mentions_correct: int
    mentions_wrong: int
    mentions_wrong_unaligned: int


def _point(threshold, nils, candidates, doc_gold, doc_pred) -> SweepPoint:
    added: dict[str, set[str]] = {}
    fired = correct = wrong = wrong_unaligned = 0
    for nil, cand in zip(nils, candidates, strict=True):
        if cand is None or cand.score < threshold:
            continue
        fired += 1
        if cand.concept_id in nil.gold_ids:
            correct += 1
        else:
            wrong += 1
            if not nil.gold_aligned:
                wrong_unaligned += 1
        added.setdefault(nil.pmid, set()).add(cand.concept_id)

    tp = fp = fn = 0
    for pmid, gold in doc_gold.items():
        pred = doc_pred.get(pmid, set()) | added.get(pmid, set())
        d_tp, d_fp, d_fn = concept_counts(gold, pred)
        tp, fp, fn = tp + d_tp, fp + d_fp, fn + d_fn
    m = metrics_from_counts(tp, fp, fn)
    return SweepPoint(threshold, m.precision, m.recall, m.f1, tp, fp, fn,
                      fired, correct, wrong, wrong_unaligned)


def sweep(nils, candidates, doc_gold, doc_pred) -> list[SweepPoint]:
    """Exact curve: every distinct top-1 score is a breakpoint.

    Not a fixed grid -- a grid can straddle the region where F1 peaks and misreport the
    shape. Re-scoring is pure set arithmetic over the documents, so this is cheap.
    """
    scores = sorted({c.score for c in candidates if c is not None}, reverse=True)
    # Descending. The first point sits ABOVE every score, so nothing fires: the baseline
    # anchor. Each subsequent point admits one more distinct score.
    thresholds = [scores[0] + 1.0] + scores if scores else [1.0]
    return [_point(t, nils, candidates, doc_gold, doc_pred) for t in thresholds]
```

- [ ] **Step 3: Run, confirm PASS**

- [ ] **Step 4: Write the mention-understatement test (ONE) — the point of the whole task**

```python
def test_two_wrong_links_to_the_same_concept_cost_one_fp_but_count_as_two_mentions():
    # THE understatement this logging exists to expose: concept scoring is
    # set-per-document, so both wrong links collapse into a single fp.
    nils = [_nil("1", "agent a", {"MESH:D1"}), _nil("1", "agent b", {"MESH:D1"})]
    cands = [Candidate("MESH:D9", 0.90), Candidate("MESH:D9", 0.90)]
    point = sweep(nils, cands, {"1": {"MESH:D1"}}, {"1": set()})[-1]
    assert point.mentions_fired == 2
    assert point.mentions_wrong == 2
    assert point.fp == 1  # <- the aggregate absorbs one of the two errors
```

- [ ] **Step 5: Run, confirm PASS**

Passes on the existing implementation. It is a **characterization test** pinning the exact discrepancy the spec requires reporting — say so in the commit rather than implying a RED cycle.

- [ ] **Step 6: Write the unaligned-attribution test (ONE)**

```python
def test_wrong_links_on_non_gold_aligned_mentions_are_counted_separately():
    # A mention with no gold alignment (spurious/truncated NER span, or GOLD_UNLINKABLE
    # where NIL was correct). Invisible to a 1136-scoped measurement.
    nils = [_nil("1", "noise", set(), aligned=False)]
    cands = [Candidate("MESH:D9", 0.90)]
    point = sweep(nils, cands, {"1": {"MESH:D1"}}, {"1": set()})[-1]
    assert (point.mentions_wrong, point.mentions_wrong_unaligned) == (1, 1)
    assert point.fp == 1
```

- [ ] **Step 7: Gate and commit**

```bash
uv run pytest tests/evals/test_fallback_sweep.py -q && uv run ruff check . && uv run pyright
git add src/biolit_evals/fallback_sweep.py tests/evals/test_fallback_sweep.py
git commit -F - <<'EOF'
feat(evals): breakpoint threshold sweep with mention-level counters

Exact curve over every distinct top-1 score rather than a fixed grid, which can
straddle the F1 peak and misreport the shape. The top point sits above every
score so nothing fires -- the baseline anchor that proves the harness.

Logs mentions_fired/correct/wrong/wrong_unaligned alongside concept metrics.
Concept scoring is set-per-document, so several wrong mention links collapse to
one fp and the aggregate UNDERSTATES cost -- the mirror of the mention-count
overstatement caught twice here. Two tests characterize that gap directly rather
than only asserting the aggregate.
EOF
```

---

### Task 7: Wire the runner and produce the curve

**Files:**
- Modify: `backend/src/biolit_evals/fallback_sweep.py` (add `collect_nils`, `main`)
- Create: `backend/evals/fallback_sweep.jsonl` (committed run log)

**Interfaces:**
- Consumes: everything above, plus `canonicalize`, `merge_fragments`, `classify_exact_link`, `GoldDocument`.
- Produces: `collect_nils(docs, linker) -> tuple[list[NilMention], dict[str, set[str]], dict[str, set[str]]]`

- [ ] **Step 1: Write the failing test for `collect_nils` (ONE)**

```python
def test_collect_nils_returns_only_mentions_the_dictionary_left_unlinked():
    # Blast radius: a dictionary hit must never reach the fallback.
    ...  # build a two-entity GoldDocument where one surface is in the stub dictionary
    nils, doc_gold, doc_pred = collect_nils([doc], linker=stub)
    assert [n.surface for n in nils] == ["unlinkable surface"]
```

Build the stub linker from `MeshDictionary` with a hand-written alias table, matching the pattern in `tests/canon/test_linker.py`.

- [ ] **Step 2: Run, confirm failure, implement `collect_nils`**

Mirror `score_end_to_end`'s loop: `merge_fragments` → `canonicalize` → for each predicted entity with `canonical_id is None`, emit a `NilMention`. Set `gold_ids`/`gold_aligned` from `classify_exact_link` — aligned exactly when it returns a record with status `NIL`. Build `doc_pred` from the linked entities and `doc_gold` from `doc.mentions`.

- [ ] **Step 3: Run, confirm PASS**

- [ ] **Step 4: Add `main()` — run all four arms**

Load aliases + label map, build both scorers, score every NIL surface once per arm, then `sweep` per arm. Four arms: `sapbert_unconstrained`, `sapbert_labeled`, `tfidf_unconstrained`, `tfidf_labeled`. Append one row per run to `evals/fallback_sweep.jsonl` with `timestamp`, `git_sha`, `n_nil`, `n_aliases`, the full curve per arm, and the 1136-slice by-product.

- [ ] **Step 5: Verify the anchors before reading any result**

```bash
uv run python -m biolit_evals.fallback_sweep --dataset bc5cdr
```

For **every** arm, confirm the first sweep point reports `mentions_fired == 0`, `f1 == 0.7697`, `fp == 312`. **Any deviation invalidates the sweep — fix the harness before reading the curve.** Also confirm `n_nil == 3209`.

- [ ] **Step 6: Commit the run log and report the curve**

```bash
git add evals/fallback_sweep.jsonl src/biolit_evals/fallback_sweep.py tests/evals/test_fallback_sweep.py
git commit -m "eval: fallback linker threshold sweep, four arms"
```

Report back: the curve per arm, the argmax-F1 breakpoint, the `mentions_wrong / fp` ratio at that point, and the labeled 1136-slice by-product. **Do not merge and do not recommend building** — the decision is the user's.

---

## Self-Review

**Spec coverage:** concept→label map w/ set semantics → T1; orphan diagnostic → T1/T2; multi-label diagnostic → T1/T2; NIL-only blast radius → T7 S1; full-alias index → T5/T7; max-over-aliases → T3; brute-force exact → T5; SapBERT → T5; TF-IDF control → T4; label-constrained + unconstrained arms → T7 S4; breakpoint sweep → T6; both threshold anchors → T6 S1 + T7 S5; mention-level counters → T6; 1136 by-product labeled → T7 S6; throughput pilot → T5 S6; negative-result handling → T7 S6.

**Placeholders:** Task 7 Step 1 uses `...` for fixture construction and Step 4 describes `main()` prose-only. Both are wiring over interfaces fully specified in Tasks 1–6, and `main()` is I/O the tdd-guard forbids testing as display code. Accepted deliberately — flagged rather than hidden.

**Type consistency:** `Candidate(concept_id, score)` consistent T3/T6/T7. `best_concept(scores, alias_concept_ids, *, label_filter)` consistent. `LabelFilter(labels, required)` consistent. `build_concept_labels -> dict[str, set[EntityLabel]]` consumed as `dict[str, set[EntityLabel]]` by `LabelFilter`; serialized to sorted `list[str]` in T2 and must be **re-hydrated to `set[EntityLabel]`** on load in T7 — noted explicitly since it is the one place the on-disk and in-memory types differ.
