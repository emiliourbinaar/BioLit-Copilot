# End-to-End Canonicalization Eval Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Measure the real production path — `extract_entities` → `canonicalize` (merge + link) → `canonical_id` — against gold MeSH IDs, with a permanent categorical census explaining every point of loss.

**Architecture:** Document-level gold loaders feed natural text to the NER model; predictions run through `canonicalize`; scoring is concept-level (micro-averaged, set-per-document, pooled and per label) alongside an outcome census (exact / mergeable / truncated / missed, with truncation sub-classified) and a descriptive merge audit. Results append to `evals/e2e_runs.jsonl`.

**Tech Stack:** Python 3.12, uv, pytest (+ `heavy` marker), stdlib `dataclasses`/`enum`/`collections`/`json`. No new dependencies.

## Global Constraints

- Run all commands from `backend/`. Use `uv run <cmd>` — plain `pytest`/`python` will not work.
- Gate for every task: `uv run ruff check .` + `uv run ruff format --check .` + `uv run pyright` + `uv run pytest`, all passing. ruff ruleset `E,F,I,UP,B`.
- String-valued enums inherit `enum.StrEnum` (ADR-0005). Datetimes use `datetime.now(UTC)` (never `timezone.utc`).
- Reuse the existing `EntityLabel` (`biolit.domain.enums`), `Entity` (`biolit.domain.records`), `GoldMention` / `parse_pubtator` / `reconcile_mesh_id` (`biolit_evals.mesh_gold`), `merge_fragments` (`biolit.canon.fragments`), `canonicalize` (`biolit.canon.canonicalize`), `Linker` (`biolit.canon.linker`). Do NOT redefine any of them.
- Default `pytest` must stay offline: no model load, no downloads. Real-data code is `@pytest.mark.heavy` (deselected by `addopts = "-m 'not heavy'"`).
- tdd-guard is active. NEVER hand-edit anything under `.claude/tdd-guard/`. Write the failing test first; if a single Write/Edit adding several test functions is blocked, add one, run pytest, then add the next via a separate Edit.
- Use the Write/Edit tools for all files — never PowerShell redirects (they add a UTF-8 BOM, which has already caused one defect in this project).
- **Out of scope — do not do these:** any change to `merge_fragments` or its heuristic; truncation/boundary recovery; scoping a SapBERT fallback; clustering.
- PubTator document text is `title + " " + abstract` — verified: 9809/9809 mentions across all 500 BC5CDR test documents satisfy `text[start:end] == mention_text`.

---

### Task 1: `GoldDocument` + `parse_pubtator_documents`

**Files:**
- Modify: `backend/src/biolit_evals/mesh_gold.py`
- Create: `backend/tests/evals/fixtures/pubtator_two_docs.txt`
- Test: `backend/tests/evals/test_mesh_gold.py` (append)

**Interfaces:**
- Consumes: `GoldMention`, `parse_pubtator` (already in `mesh_gold.py`).
- Produces:
  - `GoldDocument(pmid: str, text: str, mentions: list[GoldMention])` frozen dataclass.
  - `parse_pubtator_documents(text: str) -> list[GoldDocument]` — splits a PubTator dump on blank lines; per document takes the `PMID|t|` title and `PMID|a|` abstract, sets `text = title + " " + abstract`, and delegates mention parsing to the existing `parse_pubtator` (which already ignores non-mention lines).

- [ ] **Step 1: Create the two-document fixture** — create `backend/tests/evals/fixtures/pubtator_two_docs.txt` using the **Write tool**. Mention lines are TAB-separated; documents are separated by ONE blank line. Offsets below are correct for `title + " " + abstract`:

```
1|t|Metformin therapy.
1|a|Metformin treats PCOS.
1	0	9	Metformin	Chemical	D008687
1	19	28	Metformin	Chemical	D008687
1	36	40	PCOS	Disease	D011085

2|t|Aspirin study.
2|a|Aspirin and cancer.
2	0	7	Aspirin	Chemical	D001241
2	15	22	Aspirin	Chemical	D001241
2	27	33	cancer	Disease	D009369
```

- [ ] **Step 2: Write the failing cross-validation test** — append to `backend/tests/evals/test_mesh_gold.py`:

```python
def test_parse_pubtator_documents_matches_flat_parser():
    # The document parser must not lose, duplicate, or reorder mentions relative to the
    # already-validated flat parser. Both read offsets from the same mention lines, so
    # this specifically pins the document *segmentation*, not the offsets.
    raw = (_FIX / "pubtator_two_docs.txt").read_text(encoding="utf-8")
    docs = parse_pubtator_documents(raw)
    flat = parse_pubtator(raw)
    assert [m for d in docs for m in d.mentions] == flat
    assert len(docs) == 2
    assert [d.pmid for d in docs] == ["1", "2"]
```

- [ ] **Step 3: Run to verify it fails**

Run: `cd backend && uv run pytest tests/evals/test_mesh_gold.py::test_parse_pubtator_documents_matches_flat_parser -v`
Expected: FAIL — `ImportError: cannot import name 'parse_pubtator_documents'` (add it to the file's existing import from `biolit_evals.mesh_gold`).

- [ ] **Step 4: Implement** — append to `backend/src/biolit_evals/mesh_gold.py`:

```python
@dataclass(frozen=True)
class GoldDocument:
    pmid: str
    text: str
    mentions: list[GoldMention]


def parse_pubtator_documents(text: str) -> list[GoldDocument]:
    """Parse a PubTator dump into documents carrying their own text and mentions.

    Document text is `title + " " + abstract`: PubTator offsets are expressed against that
    concatenation. Verified against the real corpus -- 9809 of 9809 mentions across all 500
    BC5CDR test documents satisfy `text[start:end] == mention text` under it, and a
    zero-length separator breaks alignment. Mention parsing delegates to `parse_pubtator`,
    which already ignores the title/abstract lines, so the two parsers cannot disagree
    about a mention -- only about document segmentation.
    """
    documents: list[GoldDocument] = []
    for block in text.split("\n\n"):
        if not block.strip():
            continue
        pmid = ""
        title = ""
        abstract = ""
        for line in block.splitlines():
            if not title and "|t|" in line:
                pmid, title = line.split("|t|", 1)
            elif not abstract and "|a|" in line:
                _, abstract = line.split("|a|", 1)
        if not pmid:
            continue
        documents.append(
            GoldDocument(pmid=pmid, text=title + " " + abstract, mentions=parse_pubtator(block))
        )
    return documents
```

- [ ] **Step 5: Run to verify it passes**

Run: `cd backend && uv run pytest tests/evals/test_mesh_gold.py -v`
Expected: PASS.

- [ ] **Step 6: Add the offset-basis assertion test via Edit** — append to `backend/tests/evals/test_mesh_gold.py`:

```python
def test_parse_pubtator_documents_offsets_index_the_document_text():
    # The cross-validation test above cannot catch a wrong text reconstruction, because
    # both parsers read offsets from the same mention lines. Only this pins the
    # title/abstract separator: every mention must slice out of the document text.
    for path in ("pubtator_two_docs.txt", "pubtator_sample.txt"):
        raw = (_FIX / path).read_text(encoding="utf-8")
        docs = parse_pubtator_documents(raw)
        assert docs, path
        for doc in docs:
            for m in doc.mentions:
                assert doc.text[m.start : m.end] == m.text, (path, doc.pmid, m)
```

- [ ] **Step 7: Run to verify it passes**

Run: `cd backend && uv run pytest tests/evals/test_mesh_gold.py -v`
Expected: PASS. If a mention fails to slice, the fixture offsets are wrong — fix the fixture, not the parser.

- [ ] **Step 8: Gate + commit**

```bash
cd backend && uv run ruff check . && uv run ruff format --check . && uv run pyright && uv run pytest -q
cd .. && git add backend/src/biolit_evals/mesh_gold.py backend/tests/evals/test_mesh_gold.py backend/tests/evals/fixtures/pubtator_two_docs.txt
git commit -m "feat(evals): GoldDocument + document-level PubTator parsing"
```

---

### Task 2: `load_domain_norm_documents`

**Files:**
- Modify: `backend/src/biolit_evals/mesh_gold.py`
- Test: `backend/tests/evals/test_mesh_gold.py` (append)

**Interfaces:**
- Consumes: `GoldDocument` (Task 1), `GoldMention`, `reconcile_mesh_id`, `canonical_label`.
- Produces: `load_domain_norm_documents(path: str) -> list[GoldDocument]` — one `GoldDocument` per JSONL record (each record is one sentence). Applies exactly the same bounds and span/text validation as `load_domain_norm_sample`.

- [ ] **Step 1: Write the failing test** — append to `backend/tests/evals/test_mesh_gold.py`:

```python
def test_load_domain_norm_documents_carries_text_and_matches_flat_loader():
    docs = load_domain_norm_documents(str(_FIX / "domain_norm_fixture.jsonl"))
    flat = load_domain_norm_sample(str(_FIX / "domain_norm_fixture.jsonl"))
    assert [m for d in docs for m in d.mentions] == flat
    assert docs[0].text == "Metformin treats PCOS"
    for doc in docs:
        for m in doc.mentions:
            assert doc.text[m.start : m.end] == m.text


def test_load_domain_norm_documents_rejects_span_text_mismatch(tmp_path):
    rec = {
        "pmid": "1",
        "text": "Metformin treats PCOS",
        "entities": [
            {"start": 0, "end": 9, "label": "CHEMICAL", "text": "WRONG", "mesh_id": "D008687"}
        ],
    }
    bad = tmp_path / "bad.jsonl"
    bad.write_text(json.dumps(rec) + "\n", encoding="utf-8")
    with pytest.raises(ValueError):
        load_domain_norm_documents(str(bad))
```

- [ ] **Step 2: Run to verify it fails**

Run: `cd backend && uv run pytest tests/evals/test_mesh_gold.py -k domain_norm_documents -v`
Expected: FAIL — `ImportError: cannot import name 'load_domain_norm_documents'`.

- [ ] **Step 3: Refactor the shared record parsing, then add the loader** — in `backend/src/biolit_evals/mesh_gold.py`, replace the body of `load_domain_norm_sample` so both loaders share one validated implementation, and add the new loader:

```python
def _mentions_from_record(rec: dict[str, Any], pmid: str) -> list[GoldMention]:
    """Parse and validate one domain-gold JSONL record's entities.

    Shared by both domain loaders so the bounds and span/text checks -- which are what stop
    an annotator off-by-one from silently corrupting the gold -- cannot drift apart.
    """
    text = rec["text"]
    mentions: list[GoldMention] = []
    for ent in rec.get("entities", []):
        start, end = int(ent["start"]), int(ent["end"])
        if not (0 <= start < end <= len(text)):
            raise ValueError(f"span out of bounds in {pmid}: {ent}")
        label = canonical_label(ent["label"])
        if label is None:
            raise ValueError(f"non-canonical label in {pmid}: {ent['label']}")
        surface = text[start:end]
        recorded_text = ent["text"]
        if recorded_text != surface:
            raise ValueError(
                f"gold span text mismatch in {pmid}: recorded text "
                f"{recorded_text!r} does not match text[{start}:{end}] = {surface!r}"
            )
        mentions.append(
            GoldMention(
                pmid=pmid,
                start=start,
                end=end,
                text=surface,
                label=label,
                mesh_ids=reconcile_mesh_id(str(ent["mesh_id"])),
            )
        )
    return mentions


def _iter_domain_records(path: str) -> Iterator[tuple[str, dict[str, Any]]]:
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            yield str(rec.get("pmid") or rec.get("paper_id") or ""), rec


def load_domain_norm_sample(path: str) -> list[GoldMention]:
    mentions: list[GoldMention] = []
    for pmid, rec in _iter_domain_records(path):
        mentions.extend(_mentions_from_record(rec, pmid))
    return mentions


def load_domain_norm_documents(path: str) -> list[GoldDocument]:
    """Load the blind domain normalization gold as documents (one per annotated sentence)."""
    return [
        GoldDocument(pmid=pmid, text=rec["text"], mentions=_mentions_from_record(rec, pmid))
        for pmid, rec in _iter_domain_records(path)
    ]
```

Add `from collections.abc import Iterator` and `from typing import Any` to the imports at the top of the file. Delete the old inline body of `load_domain_norm_sample` — it is fully replaced above.

- [ ] **Step 4: Run to verify all mesh_gold tests pass** (including the pre-existing ones, which must be unaffected by the refactor)

Run: `cd backend && uv run pytest tests/evals/test_mesh_gold.py -v`
Expected: PASS, all tests.

- [ ] **Step 5: Gate + commit**

```bash
cd backend && uv run ruff check . && uv run ruff format --check . && uv run pyright && uv run pytest -q
cd .. && git add backend/src/biolit_evals/mesh_gold.py backend/tests/evals/test_mesh_gold.py
git commit -m "feat(evals): document-level domain gold loader on shared validated parsing"
```

---

### Task 3: `outcome_census.py` — outcome classification

**Files:**
- Create: `backend/src/biolit_evals/outcome_census.py`
- Create: `backend/tests/evals/test_outcome_census.py`

**Interfaces:**
- Consumes: `EntityLabel`, `Entity`, `GoldMention`.
- Produces:
  - `Outcome(StrEnum)`: `EXACT`, `MERGEABLE`, `TRUNCATED`, `MISSED`.
  - `TruncationKind(StrEnum)`: `PREFIX_OF_GOLD`, `SUFFIX_OF_GOLD`, `INTERIOR_OR_OTHER`, `NOT_TRUNCATED`.
  - `OutcomeRecord(label: EntityLabel, outcome: Outcome, truncation: TruncationKind, char_delta: int)` frozen dataclass. `char_delta` is `gold_length - prediction_length` (positive when the prediction is shorter); it is `0` unless `outcome is TRUNCATED`.
  - `classify_outcome(gold: GoldMention, predictions: list[Entity]) -> OutcomeRecord`.
- Classification rules, in order: only same-label predictions with non-`None` offsets are considered. Exact `(start, end)` match → `EXACT`. No overlapping prediction → `MISSED`. ≥2 overlapping → `MERGEABLE`. Exactly 1 overlapping → `TRUNCATED`, sub-classified: `pred.start == gold.start and pred.end < gold.end` → `PREFIX_OF_GOLD` (suffix dropped); `pred.end == gold.end and pred.start > gold.start` → `SUFFIX_OF_GOLD` (prefix dropped); otherwise `INTERIOR_OR_OTHER` (includes predictions extending past gold).

- [ ] **Step 1: Write the first failing test** — create `backend/tests/evals/test_outcome_census.py`:

```python
from biolit.domain.enums import EntityLabel
from biolit.domain.records import Entity
from biolit_evals.mesh_gold import GoldMention
from biolit_evals.outcome_census import Outcome, TruncationKind, classify_outcome

CHEMICAL = EntityLabel.CHEMICAL
DISEASE = EntityLabel.DISEASE


def _gold(start, end, text, label=CHEMICAL):
    return GoldMention(
        pmid="1", start=start, end=end, text=text, label=label, mesh_ids=("MESH:D1",)
    )


def _pred(start, end, text, label=CHEMICAL):
    return Entity(text=text, label=label, start=start, end=end)


def test_exact_match():
    rec = classify_outcome(_gold(0, 9, "metformin"), [_pred(0, 9, "metformin")])
    assert rec.outcome is Outcome.EXACT
    assert rec.truncation is TruncationKind.NOT_TRUNCATED
    assert rec.char_delta == 0
```

- [ ] **Step 2: Run to verify it fails**

Run: `cd backend && uv run pytest tests/evals/test_outcome_census.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'biolit_evals.outcome_census'`.

- [ ] **Step 3: Implement** — create `backend/src/biolit_evals/outcome_census.py`:

```python
from dataclasses import dataclass
from enum import StrEnum

from biolit.domain.enums import EntityLabel
from biolit.domain.records import Entity
from biolit_evals.mesh_gold import GoldMention


class Outcome(StrEnum):
    EXACT = "EXACT"
    MERGEABLE = "MERGEABLE"
    TRUNCATED = "TRUNCATED"
    MISSED = "MISSED"


class TruncationKind(StrEnum):
    PREFIX_OF_GOLD = "PREFIX_OF_GOLD"  # prediction starts at gold, ends early: suffix dropped
    SUFFIX_OF_GOLD = "SUFFIX_OF_GOLD"  # prediction ends at gold, starts late: prefix dropped
    INTERIOR_OR_OTHER = "INTERIOR_OR_OTHER"
    NOT_TRUNCATED = "NOT_TRUNCATED"


@dataclass(frozen=True)
class OutcomeRecord:
    label: EntityLabel
    outcome: Outcome
    truncation: TruncationKind
    char_delta: int  # gold length - prediction length; 0 unless TRUNCATED


def _overlaps(a_start: int, a_end: int, b_start: int, b_end: int) -> bool:
    return max(a_start, b_start) < min(a_end, b_end)


def classify_outcome(gold: GoldMention, predictions: list[Entity]) -> OutcomeRecord:
    """Classify how the NER predictions covered one gold mention.

    This is the categorical breakdown behind the end-to-end number: it separates spans the
    model got right from the three distinct ways it can get them wrong, so a change in the
    headline score can be attributed to a mechanism rather than guessed at.
    """
    same_label = [
        p
        for p in predictions
        if p.label is gold.label and p.start is not None and p.end is not None
    ]
    if any(p.start == gold.start and p.end == gold.end for p in same_label):
        return OutcomeRecord(gold.label, Outcome.EXACT, TruncationKind.NOT_TRUNCATED, 0)

    overlapping = [
        p
        for p in same_label
        if p.start is not None
        and p.end is not None
        and _overlaps(p.start, p.end, gold.start, gold.end)
    ]
    if not overlapping:
        return OutcomeRecord(gold.label, Outcome.MISSED, TruncationKind.NOT_TRUNCATED, 0)
    if len(overlapping) >= 2:
        return OutcomeRecord(gold.label, Outcome.MERGEABLE, TruncationKind.NOT_TRUNCATED, 0)

    pred = overlapping[0]
    assert pred.start is not None and pred.end is not None
    if pred.start == gold.start and pred.end < gold.end:
        kind = TruncationKind.PREFIX_OF_GOLD
    elif pred.end == gold.end and pred.start > gold.start:
        kind = TruncationKind.SUFFIX_OF_GOLD
    else:
        kind = TruncationKind.INTERIOR_OR_OTHER
    delta = (gold.end - gold.start) - (pred.end - pred.start)
    return OutcomeRecord(gold.label, Outcome.TRUNCATED, kind, delta)
```

- [ ] **Step 4: Run to verify it passes**

Run: `cd backend && uv run pytest tests/evals/test_outcome_census.py -v`
Expected: PASS.

- [ ] **Step 5: Add the remaining category tests via Edit** — append to `backend/tests/evals/test_outcome_census.py`. These use the exact shapes observed in the real corpus probe:

```python
def test_mergeable_when_two_adjacent_predictions_cover_gold():
    # Real case: gold "GLP-1RAs" predicted as "GLP" + "1RA".
    rec = classify_outcome(
        _gold(0, 8, "GLP-1RAs"), [_pred(0, 3, "GLP"), _pred(4, 7, "1RA")]
    )
    assert rec.outcome is Outcome.MERGEABLE


def test_truncated_prefix_of_gold_is_a_dropped_suffix():
    # Real case: gold "CFD" predicted as "CF".
    rec = classify_outcome(_gold(0, 3, "CFD"), [_pred(0, 2, "CF")])
    assert rec.outcome is Outcome.TRUNCATED
    assert rec.truncation is TruncationKind.PREFIX_OF_GOLD
    assert rec.char_delta == 1


def test_truncated_suffix_of_gold_is_a_dropped_prefix():
    # Real case: gold "insulin resistance" predicted as "resistance".
    rec = classify_outcome(_gold(0, 18, "insulin resistance"), [_pred(8, 18, "resistance")])
    assert rec.outcome is Outcome.TRUNCATED
    assert rec.truncation is TruncationKind.SUFFIX_OF_GOLD
    assert rec.char_delta == 8


def test_truncated_interior_when_neither_boundary_matches():
    rec = classify_outcome(_gold(0, 20, "a" * 20), [_pred(5, 15, "b" * 10)])
    assert rec.outcome is Outcome.TRUNCATED
    assert rec.truncation is TruncationKind.INTERIOR_OR_OTHER


def test_missed_when_no_overlapping_prediction():
    rec = classify_outcome(_gold(0, 9, "metformin"), [_pred(20, 26, "cancer")])
    assert rec.outcome is Outcome.MISSED


def test_other_label_predictions_are_ignored():
    # A DISEASE prediction sitting exactly on a CHEMICAL gold span is not a match.
    rec = classify_outcome(
        _gold(0, 9, "metformin", CHEMICAL), [_pred(0, 9, "metformin", DISEASE)]
    )
    assert rec.outcome is Outcome.MISSED
```

- [ ] **Step 6: Run + gate**

Run: `cd backend && uv run pytest tests/evals/test_outcome_census.py -v && uv run ruff check . && uv run pyright`
Expected: PASS (7 tests), clean.

- [ ] **Step 7: Commit**

```bash
git add backend/src/biolit_evals/outcome_census.py backend/tests/evals/test_outcome_census.py
git commit -m "feat(evals): NER outcome classification (exact/mergeable/truncated/missed)"
```

---

### Task 4: `census()` aggregation, pooled and per label

**Files:**
- Modify: `backend/src/biolit_evals/outcome_census.py`
- Test: `backend/tests/evals/test_outcome_census.py` (append)

**Interfaces:**
- Consumes: `OutcomeRecord`, `Outcome`, `TruncationKind` (Task 3).
- Produces:
  - `Census(total: int, outcomes: dict[str, int], truncation: dict[str, int], by_label: dict[str, dict[str, int]], truncation_by_label: dict[str, dict[str, int]])` frozen dataclass. All keys are plain strings (the `StrEnum` values) so the whole structure is JSON-serializable for the run log.
  - `census(records: list[OutcomeRecord]) -> Census`. `truncation` and `truncation_by_label` count only records whose outcome is `TRUNCATED`.

- [ ] **Step 1: Write the failing test** — append to `backend/tests/evals/test_outcome_census.py`:

```python
from biolit_evals.outcome_census import OutcomeRecord, census


def test_census_counts_pooled_and_per_label():
    records = [
        OutcomeRecord(CHEMICAL, Outcome.EXACT, TruncationKind.NOT_TRUNCATED, 0),
        OutcomeRecord(CHEMICAL, Outcome.TRUNCATED, TruncationKind.PREFIX_OF_GOLD, 1),
        OutcomeRecord(DISEASE, Outcome.TRUNCATED, TruncationKind.SUFFIX_OF_GOLD, 8),
        OutcomeRecord(DISEASE, Outcome.MISSED, TruncationKind.NOT_TRUNCATED, 0),
    ]
    c = census(records)
    assert c.total == 4
    assert c.outcomes == {"EXACT": 1, "TRUNCATED": 2, "MISSED": 1}
    # only TRUNCATED records contribute to the truncation breakdown
    assert c.truncation == {"PREFIX_OF_GOLD": 1, "SUFFIX_OF_GOLD": 1}
    assert c.by_label["CHEMICAL"] == {"EXACT": 1, "TRUNCATED": 1}
    assert c.by_label["DISEASE"] == {"TRUNCATED": 1, "MISSED": 1}
    assert c.truncation_by_label["CHEMICAL"] == {"PREFIX_OF_GOLD": 1}
    assert c.truncation_by_label["DISEASE"] == {"SUFFIX_OF_GOLD": 1}


def test_census_of_no_records_is_empty():
    c = census([])
    assert c.total == 0
    assert c.outcomes == {} and c.truncation == {}
    assert c.by_label == {} and c.truncation_by_label == {}
```

- [ ] **Step 2: Run to verify it fails**

Run: `cd backend && uv run pytest tests/evals/test_outcome_census.py -k census -v`
Expected: FAIL — `ImportError: cannot import name 'census'`.

- [ ] **Step 3: Implement** — append to `backend/src/biolit_evals/outcome_census.py` (add `from collections import Counter` to the imports):

```python
@dataclass(frozen=True)
class Census:
    """Categorical breakdown of how NER covered the gold mentions.

    Keys are plain strings (StrEnum values) so the whole structure serializes straight into
    the JSONL run log.
    """

    total: int
    outcomes: dict[str, int]
    truncation: dict[str, int]
    by_label: dict[str, dict[str, int]]
    truncation_by_label: dict[str, dict[str, int]]


def census(records: list[OutcomeRecord]) -> Census:
    outcomes: Counter[str] = Counter()
    truncation: Counter[str] = Counter()
    by_label: dict[str, Counter[str]] = {}
    truncation_by_label: dict[str, Counter[str]] = {}
    for record in records:
        label = record.label.value
        outcomes[record.outcome.value] += 1
        by_label.setdefault(label, Counter())[record.outcome.value] += 1
        if record.outcome is Outcome.TRUNCATED:
            truncation[record.truncation.value] += 1
            truncation_by_label.setdefault(label, Counter())[record.truncation.value] += 1
    return Census(
        total=len(records),
        outcomes=dict(outcomes),
        truncation=dict(truncation),
        by_label={k: dict(v) for k, v in by_label.items()},
        truncation_by_label={k: dict(v) for k, v in truncation_by_label.items()},
    )
```

- [ ] **Step 4: Run + gate**

Run: `cd backend && uv run pytest tests/evals/test_outcome_census.py -v && uv run ruff check . && uv run pyright`
Expected: PASS (9 tests), clean.

- [ ] **Step 5: Commit**

```bash
git add backend/src/biolit_evals/outcome_census.py backend/tests/evals/test_outcome_census.py
git commit -m "feat(evals): census aggregation pooled and per label"
```

---

### Task 5: Concept-level metrics (micro-averaged, set-per-document)

**Files:**
- Create: `backend/src/biolit_evals/end_to_end.py`
- Create: `backend/tests/evals/test_end_to_end.py`

**Interfaces:**
- Produces:
  - `ConceptMetrics(tp: int, fp: int, fn: int, precision: float, recall: float, f1: float)` frozen dataclass.
  - `concept_counts(gold_ids: set[str], pred_ids: set[str]) -> tuple[int, int, int]` returning `(tp, fp, fn)` for ONE document.
  - `metrics_from_counts(tp: int, fp: int, fn: int) -> ConceptMetrics`.
- Averaging is **micro**: callers sum `(tp, fp, fn)` across documents and call `metrics_from_counts` ONCE on the totals. Within a document the ids are **sets**, so a concept mentioned repeatedly counts once.

- [ ] **Step 1: Write the failing test** — create `backend/tests/evals/test_end_to_end.py`:

```python
import pytest

from biolit_evals.end_to_end import concept_counts, metrics_from_counts


def test_concept_counts_for_one_document():
    tp, fp, fn = concept_counts({"MESH:A", "MESH:B"}, {"MESH:A", "MESH:C"})
    assert (tp, fp, fn) == (1, 1, 1)


def test_metrics_are_micro_averaged_not_macro():
    # Doc 1: tp=1 fp=0 fn=0 (perfect). Doc 2: tp=1 fp=3 fn=0 (precision 0.25).
    # Micro precision over pooled totals = 2/6 = 0.3333, while the MACRO average of the
    # two per-document precisions would be (1.0 + 0.25)/2 = 0.625. Pinning the micro value
    # is what makes a macro-average regression fail this test.
    d1 = concept_counts({"MESH:A"}, {"MESH:A"})
    d2 = concept_counts({"MESH:B"}, {"MESH:B", "MESH:X", "MESH:Y", "MESH:Z"})
    tp = d1[0] + d2[0]
    fp = d1[1] + d2[1]
    fn = d1[2] + d2[2]
    m = metrics_from_counts(tp, fp, fn)
    assert (m.tp, m.fp, m.fn) == (2, 3, 0)
    assert m.precision == pytest.approx(2 / 5)
    assert m.recall == pytest.approx(1.0)
    assert m.f1 == pytest.approx(2 * (2 / 5) * 1.0 / ((2 / 5) + 1.0))


def test_metrics_from_counts_all_zero_is_safe():
    m = metrics_from_counts(0, 0, 0)
    assert (m.precision, m.recall, m.f1) == (0.0, 0.0, 0.0)
```

- [ ] **Step 2: Run to verify it fails**

Run: `cd backend && uv run pytest tests/evals/test_end_to_end.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'biolit_evals.end_to_end'`.

- [ ] **Step 3: Implement** — create `backend/src/biolit_evals/end_to_end.py`:

```python
from dataclasses import dataclass


@dataclass(frozen=True)
class ConceptMetrics:
    tp: int
    fp: int
    fn: int
    precision: float
    recall: float
    f1: float


def concept_counts(gold_ids: set[str], pred_ids: set[str]) -> tuple[int, int, int]:
    """(tp, fp, fn) over the concept SETS of a single document.

    Set semantics are deliberate: at concept level the question is whether a paper mentions
    a concept at all, so an entity repeated five times in one abstract counts once. Multiset
    counting would let one frequently-repeated entity dominate the corpus score.
    """
    return (
        len(gold_ids & pred_ids),
        len(pred_ids - gold_ids),
        len(gold_ids - pred_ids),
    )


def metrics_from_counts(tp: int, fp: int, fn: int) -> ConceptMetrics:
    """Build metrics from MICRO-averaged totals: tp/fp/fn summed over all documents, with
    precision/recall/F1 computed once from those sums -- not a macro average of
    per-document scores."""
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return ConceptMetrics(tp=tp, fp=fp, fn=fn, precision=precision, recall=recall, f1=f1)
```

- [ ] **Step 4: Run to verify it passes**

Run: `cd backend && uv run pytest tests/evals/test_end_to_end.py -v`
Expected: PASS (3 tests).

- [ ] **Step 5: Add the set-semantics test via Edit** — append to `backend/tests/evals/test_end_to_end.py`:

```python
def test_repeated_concept_in_one_document_counts_once():
    # Both gold and predictions mention the same concept repeatedly; sets collapse it, so
    # a single much-repeated entity cannot dominate the corpus totals.
    tp, fp, fn = concept_counts({"MESH:A"}, {"MESH:A"})
    assert (tp, fp, fn) == (1, 0, 0)
```

- [ ] **Step 6: Run + gate**

Run: `cd backend && uv run pytest tests/evals/test_end_to_end.py -v && uv run ruff check . && uv run pyright`
Expected: PASS (4 tests), clean.

- [ ] **Step 7: Commit**

```bash
git add backend/src/biolit_evals/end_to_end.py backend/tests/evals/test_end_to_end.py
git commit -m "feat(evals): micro-averaged set-per-document concept metrics"
```

---

### Task 6: `score_end_to_end` — wire predictions through canonicalize

**Files:**
- Modify: `backend/src/biolit_evals/end_to_end.py`
- Test: `backend/tests/evals/test_end_to_end.py` (append)

**Interfaces:**
- Consumes: `concept_counts` / `metrics_from_counts` / `ConceptMetrics` (Task 5), `classify_outcome` / `census` / `Census` (Tasks 3–4), `GoldDocument` (Task 1), `merge_fragments`, `canonicalize`, `Linker`, `EntityLabel`, `Entity`.
- Produces:
  - `E2EMetrics` frozen dataclass with fields: `n_documents: int`, `concepts: ConceptMetrics`, `concepts_by_label: dict[str, ConceptMetrics]`, `n_predicted: int`, `n_predicted_linked: int`, `e2e_nil_rate: float`, `census: Census`, `merge_candidates: int`, `merge_candidates_linked: int`, `merge_candidates_matching_gold: int`, `merged_constituents: int`.
  - `score_end_to_end(documents: list[GoldDocument], *, predict: Callable[[str], list[Entity]], linker: Linker) -> E2EMetrics`.
- `predict` is injected so tests run without a model. `e2e_nil_rate` = `(n_predicted - n_predicted_linked) / n_predicted`, or `0.0` when nothing was predicted.

- [ ] **Step 1: Write the failing test** — append to `backend/tests/evals/test_end_to_end.py`:

```python
from biolit.canon.linker import DictionaryLinker
from biolit.canon.mesh import AliasEntry, MeshConcept, MeshDictionary
from biolit.domain.enums import EntityLabel
from biolit.domain.records import Entity
from biolit_evals.end_to_end import score_end_to_end
from biolit_evals.mesh_gold import GoldDocument, GoldMention

CHEMICAL = EntityLabel.CHEMICAL


def _linker():
    met = MeshConcept(id="MESH:D008687", name="Metformin")
    return DictionaryLinker(MeshDictionary({"metformin": [AliasEntry(met, True)]}))


def test_score_end_to_end_scores_concepts_and_censuses_outcomes():
    doc = GoldDocument(
        pmid="1",
        text="metformin treats PCOS",
        mentions=[
            GoldMention(
                pmid="1", start=0, end=9, text="metformin",
                label=CHEMICAL, mesh_ids=("MESH:D008687",),
            ),
            GoldMention(
                pmid="1", start=17, end=21, text="PCOS",
                label=EntityLabel.DISEASE, mesh_ids=("MESH:D011085",),
            ),
        ],
    )
    # Predict the chemical exactly; miss the disease entirely.
    preds = [Entity(text="metformin", label=CHEMICAL, start=0, end=9)]
    m = score_end_to_end([doc], predict=lambda _t: preds, linker=_linker())

    assert m.n_documents == 1
    assert (m.concepts.tp, m.concepts.fp, m.concepts.fn) == (1, 0, 1)
    assert m.census.outcomes == {"EXACT": 1, "MISSED": 1}
    assert m.concepts_by_label["CHEMICAL"].tp == 1
    assert m.concepts_by_label["DISEASE"].fn == 1
    assert m.n_predicted == 1 and m.n_predicted_linked == 1
    assert m.e2e_nil_rate == pytest.approx(0.0)
    assert m.merge_candidates == 0
```

- [ ] **Step 2: Run to verify it fails**

Run: `cd backend && uv run pytest tests/evals/test_end_to_end.py -k score_end_to_end -v`
Expected: FAIL — `ImportError: cannot import name 'score_end_to_end'`.

- [ ] **Step 3: Implement** — append to `backend/src/biolit_evals/end_to_end.py` (add these imports at the top of the file):

```python
from collections.abc import Callable

from biolit.canon.canonicalize import canonicalize
from biolit.canon.fragments import merge_fragments
from biolit.canon.linker import Linker
from biolit.domain.enums import EntityLabel
from biolit.domain.records import Entity
from biolit_evals.mesh_gold import GoldDocument
from biolit_evals.outcome_census import Census, OutcomeRecord, census, classify_outcome
```

```python
@dataclass(frozen=True)
class E2EMetrics:
    n_documents: int
    concepts: ConceptMetrics
    concepts_by_label: dict[str, ConceptMetrics]
    n_predicted: int
    n_predicted_linked: int
    e2e_nil_rate: float
    census: Census
    merge_candidates: int
    merge_candidates_linked: int
    merge_candidates_matching_gold: int
    merged_constituents: int


def score_end_to_end(
    documents: list[GoldDocument],
    *,
    predict: Callable[[str], list[Entity]],
    linker: Linker,
) -> E2EMetrics:
    """Score the real path: predictions -> canonicalize (merge + link) -> canonical ids.

    `predict` is injected so this runs offline in tests. Concept scoring is micro-averaged
    over documents with set semantics inside each document; the census explains the result
    categorically, and the merge audit is raw counts (n is expected to be small).
    """
    totals = [0, 0, 0]
    label_totals: dict[str, list[int]] = {}
    records: list[OutcomeRecord] = []
    n_predicted = n_linked = 0
    cand_total = cand_linked = cand_gold = merged_constituents = 0

    for doc in documents:
        preds = predict(doc.text)
        individual = [linker.link(p.text) for p in preds]
        candidates = merge_fragments(preds, doc.text)
        gold_spans = {(m.start, m.end, m.label) for m in doc.mentions}
        for cand in candidates:
            cand_total += 1
            if linker.link(cand.text).concept is not None:
                cand_linked += 1
                merged_constituents += sum(
                    1 for i in cand.source_indices if individual[i].concept is None
                )
            if (cand.start, cand.end, cand.label) in gold_spans:
                cand_gold += 1

        canon = canonicalize(preds, doc.text, linker=linker)
        n_predicted += len(canon)
        n_linked += sum(1 for e in canon if e.canonical_id is not None)

        records.extend(classify_outcome(m, preds) for m in doc.mentions)

        gold_ids = {i for m in doc.mentions for i in m.mesh_ids}
        pred_ids = {e.canonical_id for e in canon if e.canonical_id is not None}
        counts = concept_counts(gold_ids, pred_ids)
        totals = [totals[j] + counts[j] for j in range(3)]

        for label in (EntityLabel.CHEMICAL, EntityLabel.DISEASE):
            g = {i for m in doc.mentions if m.label is label for i in m.mesh_ids}
            p = {
                e.canonical_id
                for e in canon
                if e.label is label and e.canonical_id is not None
            }
            slot = label_totals.setdefault(label.value, [0, 0, 0])
            per = concept_counts(g, p)
            for j in range(3):
                slot[j] += per[j]

    return E2EMetrics(
        n_documents=len(documents),
        concepts=metrics_from_counts(totals[0], totals[1], totals[2]),
        concepts_by_label={
            k: metrics_from_counts(v[0], v[1], v[2]) for k, v in sorted(label_totals.items())
        },
        n_predicted=n_predicted,
        n_predicted_linked=n_linked,
        e2e_nil_rate=(n_predicted - n_linked) / n_predicted if n_predicted else 0.0,
        census=census(records),
        merge_candidates=cand_total,
        merge_candidates_linked=cand_linked,
        merge_candidates_matching_gold=cand_gold,
        merged_constituents=merged_constituents,
    )
```

- [ ] **Step 4: Run to verify it passes**

Run: `cd backend && uv run pytest tests/evals/test_end_to_end.py -v`
Expected: PASS.

- [ ] **Step 5: Add the merge-audit test via Edit** — append to `backend/tests/evals/test_end_to_end.py`:

```python
def test_merge_audit_counts_candidates_and_nil_gap_fills():
    # "GLP" + "1RA" is the ADR-0008 shape: neither fragment links alone, the merged
    # surface does, so both constituents inherit it and the audit records one candidate.
    glp = MeshConcept(id="MESH:D000067299", name="GLP-1 Receptor Agonists")
    linker = DictionaryLinker(MeshDictionary({"glp-1ra": [AliasEntry(glp, True)]}))
    doc = GoldDocument(
        pmid="1",
        text="GLP-1RA therapy",
        mentions=[
            GoldMention(
                pmid="1", start=0, end=7, text="GLP-1RA",
                label=CHEMICAL, mesh_ids=("MESH:D000067299",),
            )
        ],
    )
    preds = [
        Entity(text="GLP", label=CHEMICAL, start=0, end=3),
        Entity(text="1RA", label=CHEMICAL, start=4, end=7),
    ]
    m = score_end_to_end([doc], predict=lambda _t: preds, linker=linker)
    assert m.merge_candidates == 1
    assert m.merge_candidates_linked == 1
    assert m.merge_candidates_matching_gold == 1
    assert m.merged_constituents == 2
    assert m.census.outcomes == {"MERGEABLE": 1}
    assert m.e2e_nil_rate == pytest.approx(0.0)
```

- [ ] **Step 6: Run + gate**

Run: `cd backend && uv run pytest tests/evals/test_end_to_end.py -v && uv run ruff check . && uv run ruff format --check . && uv run pyright`
Expected: PASS, clean.

- [ ] **Step 7: Commit**

```bash
git add backend/src/biolit_evals/end_to_end.py backend/tests/evals/test_end_to_end.py
git commit -m "feat(evals): end-to-end scoring through canonicalize with census + merge audit"
```

---

### Task 7: Runner, run log, and CLI

**Files:**
- Modify: `backend/src/biolit_evals/end_to_end.py`
- Test: `backend/tests/evals/test_end_to_end.py` (append)
- Create: `backend/tests/evals/test_e2e_smoke.py` (heavy)

**Interfaces:**
- Consumes: `score_end_to_end` / `E2EMetrics` (Task 6), `parse_pubtator_documents` / `load_domain_norm_documents` (Tasks 1–2), `MeshDictionary`, `DictionaryLinker`, `NerModel`, `extract_entities`.
- Produces: `run_e2e_eval(*, documents, predict, linker, dataset, artifact_source, n_aliases, log_path, git_sha, now) -> E2EMetrics`, appending ONE JSON line with EXACTLY these 18 keys: `timestamp, git_sha, dataset, artifact_source, n_aliases, n_documents, tp, fp, fn, precision, recall, f1, e2e_nil_rate, n_predicted, n_predicted_linked, census, concepts_by_label, merge_audit`. Creates the log's parent directory; appends, never truncates. Plus `main(argv)` for `python -m biolit_evals.end_to_end --dataset {bc5cdr,domain}`.

- [ ] **Step 1: Write the failing runner test** — append to `backend/tests/evals/test_end_to_end.py`:

```python
import json

from biolit_evals.end_to_end import run_e2e_eval

_EXPECTED_KEYS = {
    "timestamp",
    "git_sha",
    "dataset",
    "artifact_source",
    "n_aliases",
    "n_documents",
    "tp",
    "fp",
    "fn",
    "precision",
    "recall",
    "f1",
    "e2e_nil_rate",
    "n_predicted",
    "n_predicted_linked",
    "census",
    "concepts_by_label",
    "merge_audit",
}


def test_run_e2e_eval_appends_one_log_line_with_exact_schema(tmp_path):
    doc = GoldDocument(
        pmid="1",
        text="metformin",
        mentions=[
            GoldMention(
                pmid="1", start=0, end=9, text="metformin",
                label=CHEMICAL, mesh_ids=("MESH:D008687",),
            )
        ],
    )
    preds = [Entity(text="metformin", label=CHEMICAL, start=0, end=9)]
    log = tmp_path / "nested" / "e2e_runs.jsonl"
    m = run_e2e_eval(
        documents=[doc],
        predict=lambda _t: preds,
        linker=_linker(),
        dataset="fixture",
        artifact_source="fixture",
        n_aliases=1,
        log_path=str(log),
        git_sha="abc1234",
        now="2026-07-23T00:00:00+00:00",
    )
    assert m.concepts.tp == 1
    lines = log.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    record = json.loads(lines[0])
    assert set(record.keys()) == _EXPECTED_KEYS
    assert record["census"]["outcomes"] == {"EXACT": 1}
```

- [ ] **Step 2: Run to verify it fails**

Run: `cd backend && uv run pytest tests/evals/test_end_to_end.py -k run_e2e_eval -v`
Expected: FAIL — `ImportError: cannot import name 'run_e2e_eval'`.

- [ ] **Step 3: Implement** — append to `backend/src/biolit_evals/end_to_end.py` (add `import argparse`, `import json`, `import subprocess`, `from dataclasses import asdict`, `from datetime import UTC, datetime`, `from pathlib import Path` to the imports):

```python
DEFAULT_LOG = "evals/e2e_runs.jsonl"
DOMAIN_NORM_GOLD = "evals/gold/domain_normalization_sample.jsonl"


def run_e2e_eval(
    *,
    documents: list[GoldDocument],
    predict: Callable[[str], list[Entity]],
    linker: Linker,
    dataset: str,
    artifact_source: str,
    n_aliases: int,
    log_path: str,
    git_sha: str,
    now: str,
) -> E2EMetrics:
    """Score end to end and append one JSON line to `log_path`.

    Every impure input is injected (documents, predict, linker, log_path, git_sha, now) so
    this stays offline-testable, matching `run_eval` and `run_canon_eval`.
    """
    m = score_end_to_end(documents, predict=predict, linker=linker)
    line = {
        "timestamp": now,
        "git_sha": git_sha,
        "dataset": dataset,
        "artifact_source": artifact_source,
        "n_aliases": n_aliases,
        "n_documents": m.n_documents,
        "tp": m.concepts.tp,
        "fp": m.concepts.fp,
        "fn": m.concepts.fn,
        "precision": m.concepts.precision,
        "recall": m.concepts.recall,
        "f1": m.concepts.f1,
        "e2e_nil_rate": m.e2e_nil_rate,
        "n_predicted": m.n_predicted,
        "n_predicted_linked": m.n_predicted_linked,
        "census": asdict(m.census),
        "concepts_by_label": {k: asdict(v) for k, v in m.concepts_by_label.items()},
        "merge_audit": {
            "candidates": m.merge_candidates,
            "candidates_linked": m.merge_candidates_linked,
            "candidates_matching_gold": m.merge_candidates_matching_gold,
            "merged_constituents": m.merged_constituents,
        },
    }
    Path(log_path).parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(line) + "\n")
    return m


def _git_sha() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"]).decode().strip()
    except Exception:  # best-effort metadata; never fail an eval over it
        return "unknown"


def main(argv: list[str] | None = None) -> None:
    # Heavy imports are local so importing this module for scoring stays cheap and offline.
    from biolit.canon.linker import DictionaryLinker
    from biolit.canon.mesh import MeshDictionary
    from biolit.config import get_settings
    from biolit.ner.extract import extract_entities
    from biolit.ner.model import NerModel
    from biolit_evals.mesh_gold import load_domain_norm_documents
    from biolit_evals.mesh_gold_download import load_bc5cdr_documents

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["bc5cdr", "domain"], required=True)
    args = parser.parse_args(argv)

    settings = get_settings()
    dictionary = MeshDictionary.from_artifact(settings.mesh_artifact_path)
    linker = DictionaryLinker(dictionary)
    model = NerModel.load(settings)

    if args.dataset == "bc5cdr":
        documents = load_bc5cdr_documents(settings.bc5cdr_cdr_zip_url)
    else:
        documents = load_domain_norm_documents(DOMAIN_NORM_GOLD)

    m = run_e2e_eval(
        documents=documents,
        predict=lambda text: extract_entities(
            text, model, score_threshold=settings.ner_score_threshold
        ),
        linker=linker,
        dataset=args.dataset,
        artifact_source=settings.mesh_artifact_path,
        n_aliases=len(dictionary),
        log_path=DEFAULT_LOG,
        git_sha=_git_sha(),
        now=datetime.now(UTC).isoformat(),
    )
    c = m.concepts
    print(
        f"{args.dataset}: concept P={c.precision:.4f} R={c.recall:.4f} F1={c.f1:.4f} "
        f"(tp={c.tp} fp={c.fp} fn={c.fn}, docs={m.n_documents}) "
        f"e2e_NIL={m.e2e_nil_rate:.3f}"
    )
    print(f"  census: {m.census.outcomes}")
    print(f"  truncation: {m.census.truncation}")
    print(f"  by label: {m.census.by_label}")
    print(
        f"  merge audit: candidates={m.merge_candidates} linked={m.merge_candidates_linked} "
        f"matching_gold={m.merge_candidates_matching_gold} "
        f"constituents={m.merged_constituents}"
    )


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Add the document-level BC5CDR downloader** — append to `backend/src/biolit_evals/mesh_gold_download.py`:

```python
def load_bc5cdr_documents(zip_url: str, member: str = _TEST_MEMBER) -> list[GoldDocument]:
    """Download CDR_Data.zip and parse its test split into documents with their own text.

    Heavy/manual (network + ~20 MB), same source and member as `load_bc5cdr_norm_gold`.
    """
    resp = httpx.get(zip_url, follow_redirects=True, timeout=300.0)
    resp.raise_for_status()
    with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
        pubtator = zf.read(member).decode("utf-8")
    return parse_pubtator_documents(pubtator)
```

Update that file's import line to `from biolit_evals.mesh_gold import GoldDocument, GoldMention, parse_pubtator, parse_pubtator_documents`.

- [ ] **Step 5: Run to verify the runner test passes**

Run: `cd backend && uv run pytest tests/evals/test_end_to_end.py -v`
Expected: PASS.

- [ ] **Step 6: Add the heavy real-data smoke test** — create `backend/tests/evals/test_e2e_smoke.py`:

```python
import pytest

from biolit.config import get_settings


@pytest.mark.heavy
def test_real_bc5cdr_documents_parse_and_offsets_align():
    from biolit_evals.mesh_gold_download import load_bc5cdr_documents

    docs = load_bc5cdr_documents(get_settings().bc5cdr_cdr_zip_url)
    assert len(docs) == 500  # BC5CDR test split
    total = 0
    for doc in docs:
        for m in doc.mentions:
            total += 1
            assert doc.text[m.start : m.end] == m.text, (doc.pmid, m)
    assert total == 9809  # matches the Phase 2 NER gold's tp+fn
```

Run: `cd backend && uv run pytest tests/evals/test_e2e_smoke.py -v`
Expected: `1 deselected` (heavy, deselected by default). Do NOT run the real download here.

- [ ] **Step 7: Gate + commit**

```bash
cd backend && uv run ruff check . && uv run ruff format --check . && uv run pyright && uv run pytest -q
cd .. && git add backend/src/biolit_evals/end_to_end.py backend/src/biolit_evals/mesh_gold_download.py backend/tests/evals/test_end_to_end.py backend/tests/evals/test_e2e_smoke.py
git commit -m "feat(evals): end-to-end runner, run log, and CLI"
```

---

### Task 8: Real runs and reporting

This task produces the findings. The heavy runs need network and the local MeSH artifact; their outputs (`backend/data/`, downloads) are gitignored and must NOT be committed. `backend/evals/e2e_runs.jsonl` IS committed, matching the `runs.jsonl` / `canon_runs.jsonl` precedent.

**Files:**
- Modify: `docs/EVAL_REPORT.md`
- Modify: `docs/ARCHITECTURE.md`
- Commit: `backend/evals/e2e_runs.jsonl`

- [ ] **Step 1: Ensure the MeSH artifact exists** (skip if `backend/data/canon/mesh_aliases.json.gz` is already present)

Run: `cd backend && uv run python -m biolit.canon.build_mesh`
Expected: `Wrote <N> aliases to data/canon/mesh_aliases.json.gz` with N in the hundreds of thousands.

- [ ] **Step 2: Run the heavy smoke test against real data**

Run: `cd backend && uv run pytest tests/evals/test_e2e_smoke.py -m heavy -v`
Expected: PASS — 500 documents, 9809 mentions, every offset slicing correctly.

- [ ] **Step 3: Run the domain eval**

Run: `cd backend && uv run python -m biolit_evals.end_to_end --dataset domain`
Expected: prints concept P/R/F1, e2e NIL, census, truncation breakdown, per-label census, merge audit; appends one line to `evals/e2e_runs.jsonl`. Record every number.

- [ ] **Step 4: Run the BC5CDR eval** (the full-scale run — ~500 abstracts through the NER model; expect several minutes)

Run: `cd backend && uv run python -m biolit_evals.end_to_end --dataset bc5cdr`
Expected: same output shape at full scale; one more log line. Record every number.

- [ ] **Step 5: Write the report section** — append a `# End-to-end canonicalization (Phase 3B)` section to `docs/EVAL_REPORT.md` containing:
  - Concept-level P/R/F1 **pooled and per label** for both corpora, stating explicitly that averaging is **micro** (tp/fp/fn summed across documents) with **set semantics inside each document**, and that concept-level is the primary metric because clustering consumes concept sets, not spans.
  - The full census per corpus: `EXACT` / `MERGEABLE` / `TRUNCATED` / `MISSED`, with the truncation sub-breakdown (`PREFIX_OF_GOLD` = suffix dropped, `SUFFIX_OF_GOLD` = prefix dropped, `INTERIOR_OR_OTHER`) and the per-label split.
  - `e2e_nil_rate` **beside** the gold-surface NIL (0.259 BC5CDR / 0.406 domain) with an explicit statement that the two have **different populations and denominators** — one over model predictions, one over gold mentions — so they are comparable in direction only and must never be subtracted.
  - The merge audit as raw counts, labelled descriptive-not-a-metric.
  - A plain statement of **whether the 49-sentence probe's finding holds at full scale**: that `merge_fragments` recovers approximately nothing while truncation dominates.
  - Whether truncation looks **systematic** — consistently suffix-dropping? concentrated in one label? — **without proposing a fix**; the remedy is a separate future decision.
  - A note that this eval runs on natural text for both corpora (BC5CDR from `CDR_Data.zip`, not the space-joined `tner/bc5cdr` tokens), so the Phase 2 tokenization confound does not apply to these numbers.
  - The raw `e2e_runs.jsonl` lines.

- [ ] **Step 6: Update the architecture doc** — add a short paragraph to the canonicalization section of `docs/ARCHITECTURE.md` noting that `biolit_evals.end_to_end` measures the production path (`extract_entities` → `canonicalize` → `canonical_id`) with concept-level micro-averaged metrics plus the outcome census, and that it is the only eval exercising `merge_fragments`.

- [ ] **Step 7: Gate + commit**

```bash
cd backend && uv run pytest -q && cd ..
git add docs/EVAL_REPORT.md docs/ARCHITECTURE.md backend/evals/e2e_runs.jsonl
git commit -m "docs(eval): end-to-end canonicalization results + outcome census"
```

- [ ] **Step 8: Confirm nothing gitignored was staged**

Run: `git status --porcelain` and `git ls-files backend/data | wc -l` (must print `0`).

---

## Notes for the executor

- **Do not** change `merge_fragments` or its heuristic, attempt truncation/boundary recovery, scope a SapBERT fallback, or start clustering. This plan only measures.
- If the census shows truncation dominating, that is an expected finding to **report**, not a defect to fix here.
- If any design detail is decided during implementation, add an ADR addendum in `docs/DECISIONS.md` rather than silently diverging.
