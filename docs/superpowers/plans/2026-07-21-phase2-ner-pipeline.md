# BioLit Copilot — Phase 2 (NER Pipeline) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A local biomedical NER module (`biolit.ner`) that turns text into CHEMICAL/DISEASE `Entity` spans, plus a `biolit_evals` harness that reports strict entity-level precision/recall/F1 on the BC5CDR test split and a blind-annotated domain sample, producing a first F1 on the board.

**Architecture:** `biolit.ner` wraps a Hugging Face token-classification pipeline (PubMedBERT fine-tuned on BC5CDR) behind `extract_entities(text, model) -> list[Entity]` — the exact call the Phase 4 Extractor will use. Heavy imports (torch/transformers/datasets) are lazy so the pure logic (label mapping, span scoring, dataset parsing) is unit-tested offline with injected fakes; the real model/dataset download is an opt-in path. Eval scoring is a pure span-set P/R/F1 (no seqeval). Results append to `evals/runs.jsonl`.

**Tech Stack:** Python 3.12, uv, Hugging Face `transformers` + `torch` (CPU), `datasets`, pydantic v2, pytest + pytest-asyncio, ruff, pyright.

## Global Constraints

- Python **3.12+**; packaging via **uv**. Torch resolves to the **CPU** build on PyPI for Windows/Linux (no CUDA index).
- `extract_entities(text: str, model, *, score_threshold: float) -> list[Entity]` is the stable Phase-4 anchor; `Entity` is the existing `biolit.domain.records.Entity` (`text,label,start,end`).
- Canonical entity labels this phase: exactly `"CHEMICAL"` and `"DISEASE"`.
- **Strict entity-level scoring:** an entity is correct only on exact `(start, end, label)` match. Corpus metrics are **micro-averaged** (sum per-example TP/FP/FN; never pool span tuples across examples).
- **Heavy imports are lazy** (inside functions): `torch`/`transformers` only inside `NerModel.load`; `datasets` only inside `load_bc5cdr_test`. Fast tests never trigger a model or dataset download — they inject fakes / use fixtures.
- **Blind annotation (ADR-0006):** the domain gold file is labeled from raw text with NO NER-model output visible; the model is run only AFTER the gold file is finalized. The annotator (a subagent) must not load or run the BC5CDR model while annotating.
- Eval-run log line schema (append-only `evals/runs.jsonl`): `{timestamp, model_id, dataset, split, precision, recall, f1, tp, fp, fn, n_examples, git_sha}`.
- Lint/type/test gate: `ruff check`, `ruff format --check`, `pyright`, `pytest` all green (fast subset). The real-model eval is opt-in (pytest marker `heavy`, deselected by default).
- Package import roots: `biolit` and `biolit_evals`, both under `backend/src/`.

---

### Task 1: Dependencies, config, package wiring, checkpoint verification

**Files:**
- Modify: `backend/pyproject.toml`
- Modify: `backend/src/biolit/config.py`
- Create: `backend/src/biolit_evals/__init__.py`
- Create: `backend/src/biolit/ner/__init__.py`
- Test: `backend/tests/test_ner_config.py`

**Interfaces:**
- Produces: `Settings` gains `ner_model_id: str`, `ner_device: str = "auto"`, `ner_cache_dir: str | None = None`, `ner_batch_size: int = 16`, `ner_score_threshold: float = 0.5`. New importable packages `biolit.ner` and `biolit_evals`. A registered pytest marker `heavy`.

- [ ] **Step 1: Add deps + eval package + marker to `backend/pyproject.toml`**

Add to `[project].dependencies` (after `pgvector>=0.3`):
```toml
    "transformers>=4.44",
    "torch>=2.2",
    "datasets>=2.20",
```
Add the eval package to the wheel targets:
```toml
[tool.hatch.build.targets.wheel]
packages = ["src/biolit", "src/biolit_evals"]
```
Register the marker under `[tool.pytest.ini_options]` (keep existing keys):
```toml
markers = [
    "heavy: downloads real model/dataset; deselected by default (run with -m heavy)",
]
addopts = "-m 'not heavy'"
```

- [ ] **Step 2: Add NER settings to `backend/src/biolit/config.py`**

Insert these fields into `Settings` (after the existing `http_*` fields):
```python
    ner_model_id: str = "Francesco-A/BiomedNLP-PubMedBERT-base-uncased-abstract-bc5cdr-ner-v1"
    ner_device: str = "auto"  # auto | cpu | cuda
    ner_cache_dir: str | None = None
    ner_batch_size: int = 16
    ner_score_threshold: float = 0.5
```

- [ ] **Step 3: Create empty package inits**

Create `backend/src/biolit/ner/__init__.py` (empty) and `backend/src/biolit_evals/__init__.py` (empty).

- [ ] **Step 4: Write the config test**

`backend/tests/test_ner_config.py`:
```python
import biolit_evals  # noqa: F401  - package importable
import biolit.ner  # noqa: F401
from biolit.config import Settings


def test_ner_settings_defaults():
    settings = Settings()
    assert settings.ner_device == "auto"
    assert settings.ner_batch_size == 16
    assert settings.ner_score_threshold == 0.5
    assert "bc5cdr" in settings.ner_model_id.lower()
```

- [ ] **Step 5: Sync and run the gate**

Run: `cd backend && uv sync && uv run pytest -q && uv run ruff check . && uv run pyright`
Expected: `uv sync` installs transformers/torch/datasets (torch is a large CPU wheel — this is expected and one-time). New test passes; full suite still green; ruff/pyright clean.

- [ ] **Step 6: Verify the checkpoint (license + availability, NO weight download)**

Fetch the model card for `Francesco-A/BiomedNLP-PubMedBERT-base-uncased-abstract-bc5cdr-ner-v1` (WebFetch `https://huggingface.co/<id>` and `https://huggingface.co/<id>/raw/main/config.json`). Confirm: (a) the model exists and is a `*ForTokenClassification` with `id2label` containing Chemical + Disease labels; (b) the license. Record findings in the report. If the license is missing/non-permissive or the model is unusable, STOP and report — the fallback is to use a permissively-licensed combined BC5CDR checkpoint (or `raynardj/ner-disease-ncbi-bionlp-bc5cdr-pubmed` for disease alongside a chemical model) and update `ner_model_id`. Do not download weights in this task.

- [ ] **Step 7: Commit**

```bash
git add backend/pyproject.toml backend/uv.lock backend/src/biolit/config.py backend/src/biolit/ner/__init__.py backend/src/biolit_evals/__init__.py backend/tests/test_ner_config.py
git commit -m "feat(ner): deps, config, eval package scaffold; verify BC5CDR checkpoint"
```

---

### Task 2: Label mapping (`ner/labels.py`)

**Files:**
- Create: `backend/src/biolit/ner/labels.py`
- Test: `backend/tests/ner/__init__.py`, `backend/tests/ner/test_labels.py`

**Interfaces:**
- Produces: `CHEMICAL = "CHEMICAL"`, `DISEASE = "DISEASE"`; `canonical_label(raw: str) -> str | None` — maps any model tag form (`Chemical`, `disease`, `B-Chemical`, `I-Disease`) to a canonical label, else `None`.

- [ ] **Step 1: Write the failing test**

`backend/tests/ner/__init__.py`: (empty)

`backend/tests/ner/test_labels.py`:
```python
import pytest

from biolit.ner.labels import CHEMICAL, DISEASE, canonical_label


@pytest.mark.parametrize(
    "raw, expected",
    [
        ("Chemical", CHEMICAL),
        ("chemical", CHEMICAL),
        ("B-Chemical", CHEMICAL),
        ("I-Chemical", CHEMICAL),
        ("Disease", DISEASE),
        ("B-Disease", DISEASE),
        ("I-Disease", DISEASE),
        ("O", None),
        ("Gene", None),
        ("", None),
    ],
)
def test_canonical_label(raw, expected):
    assert canonical_label(raw) == expected
```

- [ ] **Step 2: Run to verify it fails**

Run: `cd backend && uv run pytest tests/ner/test_labels.py -q`
Expected: FAIL — `ModuleNotFoundError: biolit.ner.labels`.

- [ ] **Step 3: Implement**

`backend/src/biolit/ner/labels.py`:
```python
CHEMICAL = "CHEMICAL"
DISEASE = "DISEASE"

_CANONICAL: dict[str, str] = {"chemical": CHEMICAL, "disease": DISEASE}


def canonical_label(raw: str) -> str | None:
    """Map a model tag (Chemical, disease, B-Chemical, I-Disease, ...) to a canonical label.

    Strips any BIO prefix and matches case-insensitively; unknown tags return None.
    """
    if not raw:
        return None
    key = raw.split("-")[-1].strip().lower()
    return _CANONICAL.get(key)
```

- [ ] **Step 4: Run to verify pass**

Run: `cd backend && uv run pytest tests/ner/test_labels.py -q`
Expected: PASS (10 cases).

- [ ] **Step 5: Commit**

```bash
git add backend/src/biolit/ner/labels.py backend/tests/ner
git commit -m "feat(ner): BC5CDR tag -> canonical CHEMICAL/DISEASE label mapping"
```

---

### Task 3: Model wrapper + entity extraction (`ner/model.py`, `ner/extract.py`)

**Files:**
- Create: `backend/src/biolit/ner/model.py`
- Create: `backend/src/biolit/ner/extract.py`
- Test: `backend/tests/ner/test_extract.py`

**Interfaces:**
- Consumes: `canonical_label` (labels), `Entity` (domain.records), `Settings` (config).
- Produces:
  - `class Predictor(Protocol)`: `__call__(self, text: str) -> list[dict]` (each dict has `entity_group`, `score`, `word`, `start`, `end`).
  - `class NerModel`: attribute `predictor: Predictor`; classmethod `load(settings: Settings) -> NerModel` (lazy torch/transformers import; builds a token-classification pipeline with `aggregation_strategy="simple"`, device from `ner_device`).
  - `extract_entities(text: str, model: NerModel, *, score_threshold: float = 0.5) -> list[Entity]` — maps predictor output → `Entity`, drops unmapped labels and sub-threshold spans, sorts by start offset. Empty/whitespace text → `[]`.

- [ ] **Step 1: Write the failing test (inject a fake predictor — no download)**

`backend/tests/ner/test_extract.py`:
```python
from biolit.domain.records import Entity
from biolit.ner.extract import extract_entities
from biolit.ner.model import NerModel


def _fake(spans):
    return NerModel(predictor=lambda text: spans)


def test_extract_maps_labels_and_spans():
    spans = [
        {"entity_group": "Chemical", "score": 0.99, "word": "metformin", "start": 0, "end": 9},
        {"entity_group": "Disease", "score": 0.97, "word": "PCOS", "start": 14, "end": 18},
    ]
    result = extract_entities("metformin in PCOS", _fake(spans))
    assert result == [
        Entity(text="metformin", label="CHEMICAL", start=0, end=9),
        Entity(text="PCOS", label="DISEASE", start=14, end=18),
    ]


def test_extract_drops_unmapped_and_subthreshold():
    spans = [
        {"entity_group": "Gene", "score": 0.99, "word": "TP53", "start": 0, "end": 4},
        {"entity_group": "Disease", "score": 0.10, "word": "cancer", "start": 8, "end": 14},
    ]
    result = extract_entities("TP53 in cancer", _fake(spans), score_threshold=0.5)
    assert result == []


def test_extract_sorts_by_start():
    spans = [
        {"entity_group": "Disease", "score": 0.9, "word": "PCOS", "start": 14, "end": 18},
        {"entity_group": "Chemical", "score": 0.9, "word": "metformin", "start": 0, "end": 9},
    ]
    result = extract_entities("metformin in PCOS", _fake(spans))
    assert [e.start for e in result] == [0, 14]


def test_extract_empty_text():
    assert extract_entities("   ", _fake([{"entity_group": "Disease", "score": 1.0,
                                           "word": "x", "start": 0, "end": 1}])) == []
```

- [ ] **Step 2: Run to verify it fails**

Run: `cd backend && uv run pytest tests/ner/test_extract.py -q`
Expected: FAIL — `ModuleNotFoundError: biolit.ner.model`.

- [ ] **Step 3: Implement `model.py`**

`backend/src/biolit/ner/model.py`:
```python
from dataclasses import dataclass
from typing import Protocol

from biolit.config import Settings


class Predictor(Protocol):
    def __call__(self, text: str) -> list[dict]: ...


@dataclass
class NerModel:
    predictor: Predictor

    @classmethod
    def load(cls, settings: Settings) -> "NerModel":
        # Heavy imports are lazy so the pure logic stays importable/testable offline.
        import torch
        from transformers import pipeline

        use_cuda = settings.ner_device in ("auto", "cuda") and torch.cuda.is_available()
        device = 0 if use_cuda else -1
        pipe = pipeline(
            "token-classification",
            model=settings.ner_model_id,
            aggregation_strategy="simple",
            device=device,
            batch_size=settings.ner_batch_size,
        )

        def predict(text: str) -> list[dict]:
            if not text or not text.strip():
                return []
            return list(pipe(text))

        return cls(predictor=predict)
```

- [ ] **Step 4: Implement `extract.py`**

`backend/src/biolit/ner/extract.py`:
```python
from biolit.domain.records import Entity
from biolit.ner.labels import canonical_label
from biolit.ner.model import NerModel


def extract_entities(
    text: str, model: NerModel, *, score_threshold: float = 0.5
) -> list[Entity]:
    """Run the NER model over `text` and return canonical CHEMICAL/DISEASE entities.

    Drops spans whose label is not canonical or whose score is below threshold.
    Returns entities sorted by start offset. Empty/whitespace text -> [].
    """
    if not text or not text.strip():
        return []
    entities: list[Entity] = []
    for span in model.predictor(text):
        raw = span.get("entity_group") or span.get("entity") or ""
        label = canonical_label(raw)
        if label is None:
            continue
        if float(span.get("score", 1.0)) < score_threshold:
            continue
        start = span.get("start")
        end = span.get("end")
        word = span.get("word")
        if word is None and start is not None and end is not None:
            word = text[start:end]
        entities.append(Entity(text=word or "", label=label, start=start, end=end))
    entities.sort(key=lambda e: (e.start if e.start is not None else 0))
    return entities
```

- [ ] **Step 5: Run to verify pass**

Run: `cd backend && uv run pytest tests/ner/test_extract.py -q`
Expected: PASS (4 tests). No model download occurs (predictor injected).

- [ ] **Step 6: Commit**

```bash
git add backend/src/biolit/ner/model.py backend/src/biolit/ner/extract.py backend/tests/ner/test_extract.py
git commit -m "feat(ner): NerModel wrapper (lazy HF pipeline) and extract_entities anchor"
```

---

### Task 4: Strict entity-level scoring (`biolit_evals/scoring.py`)

**Files:**
- Create: `backend/src/biolit_evals/scoring.py`
- Test: `backend/tests/evals/__init__.py`, `backend/tests/evals/test_scoring.py`

**Interfaces:**
- Consumes: `Entity` (domain.records).
- Produces:
  - `@dataclass(frozen=True) class PRF`: `precision: float`, `recall: float`, `f1: float`, `tp: int`, `fp: int`, `fn: int`.
  - `score_corpus(pairs: list[tuple[list[Entity], list[Entity]]]) -> PRF` — micro-averaged strict entity-level scoring over `(gold, pred)` per-example pairs (correct = exact `(start, end, label)`; counts summed per example).

- [ ] **Step 1: Write the failing test**

`backend/tests/evals/__init__.py`: (empty)

`backend/tests/evals/test_scoring.py`:
```python
from biolit.domain.records import Entity
from biolit_evals.scoring import PRF, score_corpus


def _e(start, end, label):
    return Entity(text="x", label=label, start=start, end=end)


def test_perfect_match():
    gold = [_e(0, 9, "CHEMICAL")]
    prf = score_corpus([(gold, list(gold))])
    assert prf == PRF(1.0, 1.0, 1.0, tp=1, fp=0, fn=0)


def test_boundary_off_is_wrong():
    gold = [_e(0, 9, "CHEMICAL")]
    pred = [_e(0, 8, "CHEMICAL")]  # off-by-one end
    prf = score_corpus([(gold, pred)])
    assert (prf.tp, prf.fp, prf.fn) == (0, 1, 1)


def test_wrong_type_is_wrong():
    gold = [_e(0, 4, "DISEASE")]
    pred = [_e(0, 4, "CHEMICAL")]
    prf = score_corpus([(gold, pred)])
    assert (prf.tp, prf.fp, prf.fn) == (0, 1, 1)


def test_missing_and_spurious():
    prf = score_corpus([([_e(0, 4, "DISEASE")], []), ([], [_e(0, 4, "CHEMICAL")])])
    assert (prf.tp, prf.fp, prf.fn) == (0, 1, 1)


def test_micro_average_across_examples():
    # Two examples each with the same (0,4,DISEASE) span must NOT collide/dedupe.
    gold = [_e(0, 4, "DISEASE")]
    prf = score_corpus([(gold, list(gold)), (gold, list(gold))])
    assert (prf.tp, prf.fp, prf.fn) == (2, 0, 0)
    assert prf.f1 == 1.0


def test_empty_corpus_is_zero():
    prf = score_corpus([])
    assert prf == PRF(0.0, 0.0, 0.0, 0, 0, 0)
```

- [ ] **Step 2: Run to verify it fails**

Run: `cd backend && uv run pytest tests/evals/test_scoring.py -q`
Expected: FAIL — `ModuleNotFoundError: biolit_evals.scoring`.

- [ ] **Step 3: Implement**

`backend/src/biolit_evals/scoring.py`:
```python
from dataclasses import dataclass

from biolit.domain.records import Entity


@dataclass(frozen=True)
class PRF:
    precision: float
    recall: float
    f1: float
    tp: int
    fp: int
    fn: int


def _keys(entities: list[Entity]) -> set[tuple[int | None, int | None, str]]:
    return {(e.start, e.end, e.label) for e in entities}


def score_corpus(pairs: list[tuple[list[Entity], list[Entity]]]) -> PRF:
    """Micro-averaged strict entity-level P/R/F1.

    Correct = exact (start, end, label). TP/FP/FN are counted per example and summed,
    so identical spans in different examples never collide.
    """
    tp = fp = fn = 0
    for gold, pred in pairs:
        gold_keys = _keys(gold)
        pred_keys = _keys(pred)
        tp += len(gold_keys & pred_keys)
        fp += len(pred_keys - gold_keys)
        fn += len(gold_keys - pred_keys)
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return PRF(precision, recall, f1, tp, fp, fn)
```

- [ ] **Step 4: Run to verify pass**

Run: `cd backend && uv run pytest tests/evals/test_scoring.py -q`
Expected: PASS (6 tests).

- [ ] **Step 5: Commit**

```bash
git add backend/src/biolit_evals/scoring.py backend/tests/evals
git commit -m "feat(evals): pure micro-averaged strict entity-level P/R/F1 scorer"
```

---

### Task 5: Datasets — BC5CDR loader + domain-sample loader (`biolit_evals/datasets.py`)

**Files:**
- Create: `backend/src/biolit_evals/datasets.py`
- Create: `backend/tests/evals/fixtures/domain_sample_fixture.jsonl`
- Test: `backend/tests/evals/test_datasets.py`

**Interfaces:**
- Consumes: `Entity` (domain.records).
- Produces:
  - `bio_tags_to_spans(tokens: list[str], tags: list[str]) -> tuple[str, list[Entity]]` — joins tokens with single spaces, returns `(text, entities)` with char offsets, using `canonical_label` on each tag (BIO grouping).
  - `load_domain_sample(path: str) -> list[tuple[str, list[Entity]]]` — parses the provenance JSONL into `(text, gold_entities)` pairs; validates spans are within text bounds and labels are canonical.
  - `load_bc5cdr_test() -> list[tuple[str, list[Entity]]]` — lazy `datasets` import; loads `tner/bc5cdr` test split via `bio_tags_to_spans`. (Heavy; used only by the runner / opt-in tests.)

- [ ] **Step 1: Create the domain-sample fixture**

`backend/tests/evals/fixtures/domain_sample_fixture.jsonl` (two lines):
```json
{"paper_id": "10.1000/x", "pmid": "111", "source": "pubmed", "sentence_index": 0, "text": "metformin treats PCOS", "entities": [{"start": 0, "end": 9, "label": "CHEMICAL", "text": "metformin"}, {"start": 17, "end": 21, "label": "DISEASE", "text": "PCOS"}]}
{"paper_id": "10.1000/y", "pmid": "222", "source": "pubmed", "sentence_index": 1, "text": "no entities here", "entities": []}
```

- [ ] **Step 2: Write the failing test**

`backend/tests/evals/test_datasets.py`:
```python
from pathlib import Path

from biolit.domain.records import Entity
from biolit_evals.datasets import bio_tags_to_spans, load_domain_sample

FIXTURE = Path(__file__).parent / "fixtures" / "domain_sample_fixture.jsonl"


def test_bio_tags_to_spans_char_offsets():
    tokens = ["metformin", "treats", "PCOS"]
    tags = ["B-Chemical", "O", "B-Disease"]
    text, entities = bio_tags_to_spans(tokens, tags)
    assert text == "metformin treats PCOS"
    assert entities == [
        Entity(text="metformin", label="CHEMICAL", start=0, end=9),
        Entity(text="PCOS", label="DISEASE", start=17, end=21),
    ]


def test_bio_tags_multitoken_entity():
    tokens = ["chronic", "kidney", "disease", "improves"]
    tags = ["B-Disease", "I-Disease", "I-Disease", "O"]
    text, entities = bio_tags_to_spans(tokens, tags)
    assert entities == [Entity(text="chronic kidney disease", label="DISEASE", start=0, end=22)]


def test_load_domain_sample_parses_provenance_and_spans():
    pairs = load_domain_sample(str(FIXTURE))
    assert len(pairs) == 2
    text0, ents0 = pairs[0]
    assert text0 == "metformin treats PCOS"
    assert {e.label for e in ents0} == {"CHEMICAL", "DISEASE"}
    assert ents0[0].text == text0[ents0[0].start : ents0[0].end]  # spans align to text
    assert pairs[1][1] == []  # no-entity line
```

- [ ] **Step 3: Run to verify it fails**

Run: `cd backend && uv run pytest tests/evals/test_datasets.py -q`
Expected: FAIL — `ModuleNotFoundError: biolit_evals.datasets`.

- [ ] **Step 4: Implement**

`backend/src/biolit_evals/datasets.py`:
```python
import json

from biolit.domain.records import Entity
from biolit.ner.labels import canonical_label


def bio_tags_to_spans(tokens: list[str], tags: list[str]) -> tuple[str, list[Entity]]:
    """Join tokens with single spaces and convert BIO tags to char-offset entities."""
    text_parts: list[str] = []
    offsets: list[tuple[int, int]] = []
    cursor = 0
    for i, tok in enumerate(tokens):
        if i > 0:
            text_parts.append(" ")
            cursor += 1
        start = cursor
        text_parts.append(tok)
        cursor += len(tok)
        offsets.append((start, cursor))
    text = "".join(text_parts)

    entities: list[Entity] = []
    cur_label: str | None = None
    cur_start = 0
    cur_end = 0
    for (start, end), tag in zip(offsets, tags, strict=True):
        label = canonical_label(tag)
        is_begin = tag.upper().startswith("B-") or (label is not None and cur_label is None)
        if label is None:
            if cur_label is not None:
                entities.append(
                    Entity(text=text[cur_start:cur_end], label=cur_label,
                           start=cur_start, end=cur_end)
                )
                cur_label = None
            continue
        if cur_label is not None and (label != cur_label or is_begin and tag.upper().startswith("B-")):
            entities.append(
                Entity(text=text[cur_start:cur_end], label=cur_label, start=cur_start, end=cur_end)
            )
            cur_label = None
        if cur_label is None:
            cur_label, cur_start, cur_end = label, start, end
        else:
            cur_end = end
    if cur_label is not None:
        entities.append(
            Entity(text=text[cur_start:cur_end], label=cur_label, start=cur_start, end=cur_end)
        )
    return text, entities


def load_domain_sample(path: str) -> list[tuple[str, list[Entity]]]:
    """Parse the blind-annotated domain gold JSONL (with provenance) into (text, entities)."""
    pairs: list[tuple[str, list[Entity]]] = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            text = rec["text"]
            entities: list[Entity] = []
            for ent in rec.get("entities", []):
                start, end, label = ent["start"], ent["end"], ent["label"]
                if not (0 <= start < end <= len(text)):
                    raise ValueError(f"span out of bounds in {rec.get('pmid')}: {ent}")
                if label not in ("CHEMICAL", "DISEASE"):
                    raise ValueError(f"non-canonical label in {rec.get('pmid')}: {label}")
                entities.append(Entity(text=text[start:end], label=label, start=start, end=end))
            pairs.append((text, entities))
    return pairs


def load_bc5cdr_test() -> list[tuple[str, list[Entity]]]:
    """Load the BC5CDR test split from `tner/bc5cdr` (lazy heavy import)."""
    from datasets import load_dataset

    ds = load_dataset("tner/bc5cdr", split="test")
    id2label = {
        0: "O", 1: "B-Chemical", 2: "B-Disease", 3: "I-Disease", 4: "I-Chemical",
    }
    pairs: list[tuple[str, list[Entity]]] = []
    for row in ds:
        tokens = row["tokens"]
        tags = [id2label[t] if isinstance(t, int) else t for t in row["tags"]]
        pairs.append(bio_tags_to_spans(tokens, tags))
    return pairs
```

Note: the `tner/bc5cdr` tag id→label map is asserted in Task 8 against the real dataset's `features`; if it differs, Task 8 corrects `id2label` there.

- [ ] **Step 5: Run to verify pass**

Run: `cd backend && uv run pytest tests/evals/test_datasets.py -q`
Expected: PASS (3 tests). No dataset download (only `bio_tags_to_spans` + `load_domain_sample` exercised).

- [ ] **Step 6: Commit**

```bash
git add backend/src/biolit_evals/datasets.py backend/tests/evals/test_datasets.py backend/tests/evals/fixtures
git commit -m "feat(evals): BC5CDR BIO->span conversion and provenance domain-sample loader"
```

---

### Task 6: Eval runner + run log (`biolit_evals/ner_eval.py`)

**Files:**
- Create: `backend/src/biolit_evals/ner_eval.py`
- Test: `backend/tests/evals/test_ner_eval.py`

**Interfaces:**
- Consumes: `score_corpus`/`PRF` (scoring), `extract_entities`/`NerModel` (ner), loaders (datasets), `Entity`.
- Produces:
  - `run_eval(examples, model, *, dataset, split, model_id, log_path, git_sha, now, score_threshold=0.5) -> PRF` — runs `extract_entities` per example, scores with `score_corpus`, appends one JSON line (the Global-Constraints schema) to `log_path`, returns the `PRF`. Pure w.r.t. injected `examples`/`model`/`log_path`.
  - `main(argv=None) -> None` — CLI `--dataset {bc5cdr|domain}` wiring real `Settings`, `NerModel.load`, the loaders, `evals/runs.jsonl`, and the current git sha.

- [ ] **Step 1: Write the failing test (inject fake model + synthetic examples + temp log)**

`backend/tests/evals/test_ner_eval.py`:
```python
import json

from biolit.domain.records import Entity
from biolit.ner.model import NerModel
from biolit_evals.ner_eval import run_eval


def test_run_eval_scores_and_appends_log(tmp_path):
    # Gold examples: (text, gold_entities). The fake model predicts from a lookup by text.
    examples = [
        ("metformin in PCOS", [Entity(text="metformin", label="CHEMICAL", start=0, end=9)]),
    ]
    preds = {
        "metformin in PCOS": [
            {"entity_group": "Chemical", "score": 0.99, "word": "metformin", "start": 0, "end": 9}
        ]
    }
    model = NerModel(predictor=lambda text: preds.get(text, []))
    log = tmp_path / "runs.jsonl"

    prf = run_eval(
        examples, model, dataset="synthetic", split="test", model_id="fake",
        log_path=str(log), git_sha="abc1234", now="2026-07-21T00:00:00Z",
    )
    assert prf.f1 == 1.0
    line = json.loads(log.read_text(encoding="utf-8").strip())
    assert line["dataset"] == "synthetic"
    assert line["f1"] == 1.0
    assert line["n_examples"] == 1
    assert line["git_sha"] == "abc1234"
    assert line["model_id"] == "fake"
    assert {"timestamp", "precision", "recall", "tp", "fp", "fn"} <= line.keys()


def test_run_eval_appends_not_overwrites(tmp_path):
    log = tmp_path / "runs.jsonl"
    model = NerModel(predictor=lambda text: [])
    run_eval([("x", [])], model, dataset="a", split="test", model_id="m",
             log_path=str(log), git_sha="s", now="t")
    run_eval([("x", [])], model, dataset="b", split="test", model_id="m",
             log_path=str(log), git_sha="s", now="t")
    assert len([ln for ln in log.read_text(encoding="utf-8").splitlines() if ln.strip()]) == 2
```

- [ ] **Step 2: Run to verify it fails**

Run: `cd backend && uv run pytest tests/evals/test_ner_eval.py -q`
Expected: FAIL — `ModuleNotFoundError: biolit_evals.ner_eval`.

- [ ] **Step 3: Implement**

`backend/src/biolit_evals/ner_eval.py`:
```python
import argparse
import json
import subprocess
from datetime import datetime, timezone

from biolit.config import get_settings
from biolit.domain.records import Entity
from biolit.ner.extract import extract_entities
from biolit.ner.model import NerModel
from biolit_evals.datasets import load_bc5cdr_test, load_domain_sample
from biolit_evals.scoring import PRF, score_corpus

DEFAULT_LOG = "evals/runs.jsonl"
DOMAIN_GOLD = "evals/gold/domain_sample.jsonl"


def run_eval(
    examples: list[tuple[str, list[Entity]]],
    model: NerModel,
    *,
    dataset: str,
    split: str,
    model_id: str,
    log_path: str,
    git_sha: str,
    now: str,
    score_threshold: float = 0.5,
) -> PRF:
    pairs: list[tuple[list[Entity], list[Entity]]] = []
    for text, gold in examples:
        pred = extract_entities(text, model, score_threshold=score_threshold)
        pairs.append((gold, pred))
    prf = score_corpus(pairs)
    line = {
        "timestamp": now,
        "model_id": model_id,
        "dataset": dataset,
        "split": split,
        "precision": prf.precision,
        "recall": prf.recall,
        "f1": prf.f1,
        "tp": prf.tp,
        "fp": prf.fp,
        "fn": prf.fn,
        "n_examples": len(examples),
        "git_sha": git_sha,
    }
    with open(log_path, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(line) + "\n")
    return prf


def _git_sha() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"]).decode().strip()
    except Exception:  # noqa: BLE001 - sha is best-effort metadata
        return "unknown"


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["bc5cdr", "domain"], required=True)
    args = parser.parse_args(argv)

    settings = get_settings()
    model = NerModel.load(settings)
    if args.dataset == "bc5cdr":
        examples = load_bc5cdr_test()
        split = "test"
    else:
        examples = load_domain_sample(DOMAIN_GOLD)
        split = "domain"

    prf = run_eval(
        examples, model, dataset=args.dataset, split=split, model_id=settings.ner_model_id,
        log_path=DEFAULT_LOG, git_sha=_git_sha(),
        now=datetime.now(timezone.utc).isoformat(), score_threshold=settings.ner_score_threshold,
    )
    print(f"{args.dataset}: P={prf.precision:.4f} R={prf.recall:.4f} F1={prf.f1:.4f} "
          f"(tp={prf.tp} fp={prf.fp} fn={prf.fn}, n={len(examples)})")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run to verify pass**

Run: `cd backend && uv run pytest tests/evals/test_ner_eval.py -q`
Expected: PASS (2 tests).

- [ ] **Step 5: Full fast gate**

Run: `cd backend && uv run pytest -q && uv run ruff check . && uv run ruff format --check . && uv run pyright`
Expected: all green (heavy tests deselected by default via `addopts`).

- [ ] **Step 6: Commit**

```bash
git add backend/src/biolit_evals/ner_eval.py backend/tests/evals/test_ner_eval.py
git commit -m "feat(evals): NER eval runner with micro-F1 and append-only run log"
```

---

### Task 7: Blind domain-sample annotation (data generation — ADR-0006)

**Files:**
- Create: `backend/evals/gold/domain_sample.jsonl`
- Test: `backend/tests/evals/test_domain_gold.py`

**Interfaces:** none (produces committed gold data + a validation test).

**CRITICAL (ADR-0006):** Annotate BLIND. Do NOT load, import, or run `biolit.ner` / the BC5CDR model at any point in this task. Label from the raw abstract text only.

- [ ] **Step 1: Pull real abstracts (network allowed — this is data generation, not a test)**

Use the Phase 1 `PubMedClient` to fetch abstracts for the project's driving topics. From `backend/`, run a one-off script (write to a scratch file, not committed):
```python
import asyncio, httpx
from biolit.clients.pubmed import PubMedClient
from biolit.config import get_settings

async def go():
    async with httpx.AsyncClient(timeout=30) as http:
        c = PubMedClient(http, get_settings())
        pmids = []
        for q in ["metformin PCOS insulin resistance", "statins cardiovascular disease",
                  "metformin type 2 diabetes"]:
            pmids += await c.esearch(q, retmax=8)
        papers = await c.efetch(list(dict.fromkeys(pmids)))
        for p in papers:
            if p.abstract:
                print(p.pmid, p.doi, repr(p.abstract[:600]))
asyncio.run(go())
```
Run: `cd backend && uv run python /path/to/scratch/pull_abstracts.py`. Collect ~15–20 abstracts with usable text.

- [ ] **Step 2: Annotate ~30–50 sentences BLIND**

Split abstracts into sentences. Reading ONLY the raw sentence text (no model output), label every CHEMICAL (drugs/chemicals) and DISEASE (diseases/conditions) mention with exact character offsets into that sentence's `text`. Write `backend/evals/gold/domain_sample.jsonl`, one sentence per line, in the loader's schema:
```json
{"paper_id": "<doi-or-pmid>", "pmid": "<pmid>", "source": "pubmed", "sentence_index": <int>, "text": "<sentence>", "entities": [{"start": <int>, "end": <int>, "label": "CHEMICAL|DISEASE", "text": "<surface>"}]}
```
Include sentences with zero entities too (they matter for precision). Ensure every `text` slice `text[start:end]` equals the entity `text`. Aim for ~30–50 sentences across the pulled papers.

- [ ] **Step 3: Write the validation test**

`backend/tests/evals/test_domain_gold.py`:
```python
from pathlib import Path

from biolit_evals.datasets import load_domain_sample

GOLD = Path(__file__).parents[2] / "evals" / "gold" / "domain_sample.jsonl"


def test_domain_gold_loads_and_is_wellformed():
    pairs = load_domain_sample(str(GOLD))
    assert len(pairs) >= 30  # target sample size
    total_entities = 0
    for text, ents in pairs:
        for e in ents:
            assert e.label in ("CHEMICAL", "DISEASE")
            assert e.start is not None and e.end is not None
            assert text[e.start : e.end] == e.text  # offsets align
            total_entities += 1
    assert total_entities > 0  # sample is not vacuous
```

- [ ] **Step 4: Run validation**

Run: `cd backend && uv run pytest tests/evals/test_domain_gold.py -q`
Expected: PASS — gold file loads via the loader, ≥30 sentences, all spans well-formed. (`load_domain_sample` already raises on out-of-bounds spans or non-canonical labels.)

- [ ] **Step 5: Commit**

```bash
git add backend/evals/gold/domain_sample.jsonl backend/tests/evals/test_domain_gold.py
git commit -m "data(evals): blind-annotated BC5CDR-style domain sample with provenance (ADR-0006)"
```

---

### Task 8: Produce the headline F1 + eval report (heavy, opt-in)

**Files:**
- Create: `backend/tests/evals/test_bc5cdr_smoke.py` (marked `heavy`)
- Create: `docs/EVAL_REPORT.md`
- Modify: `docs/ARCHITECTURE.md`
- Modify: `backend/evals/runs.jsonl` (produced by running the eval)

**Interfaces:** none (produces the numbers + report).

- [ ] **Step 1: Confirm the `tner/bc5cdr` tag mapping against the real dataset**

Run (heavy, one-off): `cd backend && uv run python -c "from datasets import load_dataset; ds=load_dataset('tner/bc5cdr', split='test'); print(ds.features); print(ds[0]['tokens'][:8], ds[0]['tags'][:8])"`
Confirm the integer→label mapping used in `datasets.load_bc5cdr_test` matches `ds.features` (the tner label2id). If it differs, correct `id2label` in `biolit_evals/datasets.py` and re-run Task 5's tests. Record the confirmed mapping in the report.

- [ ] **Step 2: Add a heavy smoke test for the real model**

`backend/tests/evals/test_bc5cdr_smoke.py`:
```python
import pytest

from biolit.config import get_settings
from biolit.ner.extract import extract_entities
from biolit.ner.model import NerModel


@pytest.mark.heavy
def test_real_model_extracts_known_entities():
    model = NerModel.load(get_settings())  # downloads weights
    ents = extract_entities("Metformin is used to treat type 2 diabetes.", model)
    labels = {e.label for e in ents}
    assert "CHEMICAL" in labels and "DISEASE" in labels
```

Run: `cd backend && uv run pytest tests/evals/test_bc5cdr_smoke.py -m heavy -q`
Expected: PASS — real model loads and returns both entity types. (First run downloads weights; may take minutes. May be run in the background.)

- [ ] **Step 3: Produce the two F1 numbers**

Run (heavy; the BC5CDR run may take many minutes on CPU — run in the background if needed):
```bash
cd backend && uv run python -m biolit_evals.ner_eval --dataset bc5cdr
cd backend && uv run python -m biolit_evals.ner_eval --dataset domain
```
Each prints P/R/F1 and appends a line to `backend/evals/runs.jsonl`. Capture both outputs.

- [ ] **Step 4: Write `docs/EVAL_REPORT.md`**

Record: the chosen checkpoint + confirmed license; the BC5CDR test-split strict entity-level P/R/F1 (headline) with `n_examples`; the domain-sample P/R/F1; and — explicitly — the methodology caveats: **blind from-scratch annotation (ADR-0006)** and the **single-annotator** limitation, plus that the BC5CDR number is the unbiased benchmark and the domain number is a supporting independent cross-check. Include the `runs.jsonl` lines.

- [ ] **Step 5: Update `docs/ARCHITECTURE.md`**

Add a short "NER layer (Phase 2)" note: `biolit.ner.extract_entities` (PubMedBERT/BC5CDR, chemicals+diseases, CPU-first) is the entity-anchoring layer the Extractor agent will call; eval harness in `biolit_evals` with strict entity-level F1 logged to `evals/runs.jsonl`.

- [ ] **Step 6: Commit**

```bash
git add backend/tests/evals/test_bc5cdr_smoke.py docs/EVAL_REPORT.md docs/ARCHITECTURE.md backend/evals/runs.jsonl
git commit -m "eval(ner): BC5CDR + domain F1 on the board, eval report with methodology caveats"
```

---

## Self-Review

**Spec coverage:**
- `biolit.ner` module + `extract_entities` anchor (spec §3, §4) → Tasks 2, 3.
- CPU-first, GPU auto-detect, lazy heavy imports (spec §2.5, §4.1) → Task 3.
- Config additions (spec §4.4) → Task 1.
- Strict entity-level P/R/F1 (spec §5.1) → Task 4 (pure span-set scorer replacing seqeval — noted deviation, equivalent metric, fewer deps).
- BC5CDR test-split loader + domain loader with provenance (spec §5.2) → Task 5.
- Eval runner + `runs.jsonl` schema (spec §5.3) → Task 6.
- Blind from-scratch annotation + provenance + methodology caveats (spec §6, ADR-0006) → Tasks 7, 8.
- BC5CDR test-split F1 headline + checkpoint license confirmed + eval report (spec §9) → Tasks 1(step 6), 8.
- Fast-unit vs opt-in heavy CI split (spec §7, §8) → Task 1 (marker/addopts), heavy tests marked in Task 8.
- ARCHITECTURE.md NER note, DECISIONS ADR-0006 (spec §9) → Task 8 (ADR-0006 already committed with the spec).

**Placeholder scan:** No TBD/TODO. F1 numbers and annotations are runtime outputs of specified code/procedures, not placeholders. Every code step has complete code; every test asserts concrete behavior.

**Type consistency:** `Entity` (`text,label,start,end`) used identically across `extract.py`, `scoring.py`, `datasets.py`, `ner_eval.py`. `NerModel.predictor` returns `list[dict]` consumed by `extract_entities`. `PRF` fields (`precision,recall,f1,tp,fp,fn`) match the log-line schema and the runner. `run_eval` signature matches its test call. Canonical labels `"CHEMICAL"/"DISEASE"` consistent in `labels.py`, `datasets.py` validation, and tests. `load_domain_sample`/`load_bc5cdr_test`/`bio_tags_to_spans` return `list[tuple[str, list[Entity]]]` consumed uniformly by the runner.
