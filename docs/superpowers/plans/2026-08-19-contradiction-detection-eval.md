# Phase 5 Contradiction Detection Eval — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the free half of the Phase 5 Critic eval — a CTD-derived contradiction gold
standard, its harness, three free baselines, and the two LLM arms wired but unrun — so the
decision to spend money is made on measured numbers rather than on an estimate.

**Architecture:** A committed manifest (`evals/gold/contradiction_pairs.jsonl`) is the frozen
gold artifact, built from CTD's `DirectEvidence` field with abstracts fetched from PubMed and
never committed. Every arm — LLM pair judgment, LLM direction decomposition, and three free
baselines — implements one `Critic` protocol returning the Phase 1 `ContradictionFinding`
type, so all six are scored through one code path and compared pairwise with McNemar's test.

**Tech Stack:** Python 3.12, uv, pydantic v2, pytest, ruff, pyright, httpx. No new
dependencies. `anthropic` is never imported by `biolit` — the client is taken as `Any`.

**Spec:** `docs/superpowers/specs/2026-08-19-contradiction-detection-eval-design.md`

## Global Constraints

- **All commands run from `backend/` via `uv run`.**
- **tdd-guard is active.** Write ONE failing test, run it, implement, run again, commit. Adding
  more than one new test in a single edit is blocked. Never manufacture a fake RED by writing
  deliberately wrong behaviour.
- **Never fabricate gold MeSH IDs, PMIDs, or abstract text.** Every fixture id in this plan is
  either a real id taken from a real record or an obviously-synthetic one (`C000001`,
  `D000001`) used only in unit tests that never touch gold.
- **Unit tests must never download anything or touch the network.** Network-touching tests
  carry `@pytest.mark.heavy` and are deselected by default (`addopts = "-m 'not heavy'"`).
- **No paid API call is made by any task in this plan.** Tasks 11–13 build and test the LLM
  arms against fakes only. Running them costs money and requires explicit user authorization.
- Ruff ruleset `E,F,I,UP,B`, line length 100. Imports at top of file only (E402), except
  deliberate function-local heavy imports in `main()`.
- String enums use `enum.StrEnum` (ADR-0005). Use `datetime.now(UTC)`, never `timezone.utc`.
- **ADR-0014:** every operand of a compound boolean condition must be independently exercised
  by a fixture. Triage first: unreachable → delete; removing it turns a gate red → pinned,
  record which gate; neither → flag, do not test and do not delete.
- **Determinism fixtures use 7 reverse-inserted elements, not 2.**
- Gate before every commit: `uv run ruff check . && uv run ruff format --check . && uv run pyright && uv run pytest -q`
- `git_sha()` records HEAD and ignores a dirty tree — commit code before running any eval.
- Commit with `git commit -F -` and a heredoc. Never PowerShell here-strings in the Bash tool.
- Do not commit `backend/data/` or downloads (gitignored). Run logs ARE committed.
- CPU-pin guard: `grep -ciE '^name = "(nvidia|triton)' uv.lock` must return 0.

---

## File Structure

**Create:**
- `src/biolit_evals/ctd_directions.py` — CTD DirectEvidence parsing + ID normalization
- `src/biolit_evals/contradiction_gold.py` — label function, candidate generation, sampling, manifest I/O
- `src/biolit_evals/contradiction_corpus.py` — abstract fetch, cache, per-class drop-rate report
- `src/biolit/critic/__init__.py`, `base.py`, `llm.py`, `direction.py` — the arms
- `src/biolit_evals/critic_baselines.py` — majority, direction lexicon, concept overlap
- `src/biolit_evals/critic_scoring.py` — macro-F1, confusion, CIs, prevalence projection, McNemar
- `src/biolit_evals/critic_cost.py` — token accounting, authorization bound, kill-switch
- `src/biolit_evals/critic_eval.py` — runner, run log, CLI
- `src/biolit_evals/annotation_export.py` — blind annotation sheet + Gate 1/Gate 2
- Tests mirroring each under `tests/evals/` and `tests/critic/`

**Modify:**
- `src/biolit/config.py` — add `ctd_chemicals_diseases_url`

---

### Task 1: CTD DirectEvidence loader

**Files:**
- Create: `src/biolit_evals/ctd_directions.py`
- Modify: `src/biolit/config.py`
- Test: `tests/evals/test_ctd_directions.py`

**Interfaces:**
- Consumes: nothing.
- Produces: `normalize_mesh_id(value: str) -> str`; `Direction` (StrEnum: `marker_mechanism`,
  `therapeutic`); `parse_ctd_directions(lines: Iterable[str]) -> dict[str, dict[tuple[str, str], frozenset[Direction]]]`
  keyed `{pmid: {(chemical_id, disease_id): directions}}`; `CTD_HEADER_MARKERS`.

**Why this task exists as written:** both parsing rules below produced a wrong answer during
design probing, and one of them — the unnormalized join — reports **zero contradictions**,
which is indistinguishable from a true negative result. The tests are the positive control.

- [ ] **Step 1: Write the failing test for header-by-content selection**

```python
# tests/evals/test_ctd_directions.py
from biolit_evals.ctd_directions import parse_ctd_directions

_HEADER = (
    "# ChemicalName\tChemicalID\tCasRN\tDiseaseName\tDiseaseID\tDirectEvidence"
    "\tInferenceGeneSymbol\tInferenceScore\tOmimIDs\tPubMedIDs"
)


def test_header_is_selected_by_content_not_by_last_comment_line():
    """CTD puts further '#' lines AFTER the column header, so 'last comment wins' yields an
    empty header and silently parses zero rows."""
    lines = [
        "# Report created: Thu Jul 30 13:59:07 EDT 2026",
        "# Fields:",
        _HEADER,
        "#",
        "aspirin\tC000001\t\tfever\tMESH:D000001\ttherapeutic\t\t\t\t111",
    ]
    parsed = parse_ctd_directions(lines)
    assert parsed == {"111": {("D000001", "C000001"): frozenset()}} or parsed != {}
```

Replace the final assertion with the exact expected mapping once `Direction` exists; the
point of step 2 is that it fails for "module not found", not for the assertion's shape.

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/evals/test_ctd_directions.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'biolit_evals.ctd_directions'`

- [ ] **Step 3: Write the implementation**

```python
# src/biolit_evals/ctd_directions.py
from collections import defaultdict
from collections.abc import Iterable
from enum import StrEnum

# The header is the comment line containing BOTH of these. Selection is by content because
# CTD emits further '#' lines after it -- a "last comment line wins" rule picks up an empty
# one, yields no columns, and parses zero rows while looking like a clean empty result.
CTD_HEADER_MARKERS = ("ChemicalName", "DirectEvidence")


class Direction(StrEnum):
    marker_mechanism = "marker/mechanism"
    therapeutic = "therapeutic"


def normalize_mesh_id(value: str) -> str:
    """Strip a namespace prefix so ids from different sources join.

    CTD writes ChemicalID bare (`C046983`) and DiseaseID prefixed (`MESH:D054198`); BC5CDR
    prefixes both. AN UNNORMALIZED JOIN MATCHES NOTHING AND REPORTS ZERO CONTRADICTIONS,
    which reads exactly like a true negative -- it did, twice, during design probing.
    """
    return value.split(":", 1)[1] if ":" in value else value


def parse_ctd_directions(
    lines: Iterable[str],
) -> dict[str, dict[tuple[str, str], frozenset[Direction]]]:
    """{pmid: {(chemical_id, disease_id): directions}} over DIRECT-evidence rows only.

    Rows without DirectEvidence are CTD's gene-inferred associations -- no paper asserts the
    relation, so they carry no direction and are dropped.
    """
    header: list[str] | None = None
    accumulated: dict[str, dict[tuple[str, str], set[Direction]]] = defaultdict(dict)
    for line in lines:
        if line.startswith("#"):
            if all(marker in line for marker in CTD_HEADER_MARKERS):
                header = line.lstrip("#").strip().split("\t")
            continue
        if header is None:
            raise ValueError(
                "parse_ctd_directions: reached a data row before the column header. "
                f"Expected a '#' line containing all of {CTD_HEADER_MARKERS}."
            )
        row = dict(zip(header, line.rstrip("\n").split("\t"), strict=False))
        evidence = (row.get("DirectEvidence") or "").strip()
        pmids = (row.get("PubMedIDs") or "").strip()
        if not evidence or not pmids:
            continue
        key = (
            normalize_mesh_id(row["ChemicalID"].strip()),
            normalize_mesh_id(row["DiseaseID"].strip()),
        )
        directions = {Direction(d) for d in evidence.split("|") if d}
        for pmid in pmids.split("|"):
            accumulated[pmid].setdefault(key, set()).update(directions)
    return {
        pmid: {key: frozenset(values) for key, values in keys.items()}
        for pmid, keys in accumulated.items()
    }
```

- [ ] **Step 4: Run test to verify it passes, then fix the assertion to be exact**

Run: `uv run pytest tests/evals/test_ctd_directions.py -v`
Then tighten the assertion to:
```python
    assert parsed == {"111": {("C000001", "D000001"): frozenset({Direction.therapeutic})}}
```
Expected: PASS. This also pins normalization — `MESH:D000001` must appear as `D000001`.

- [ ] **Step 5: Add the missing-header test (one test, per tdd-guard)**

```python
def test_data_row_before_header_raises_rather_than_returning_empty():
    """LOUDER, not quieter (ADR-0014): a silent {} here is a wrong answer that names nothing."""
    with pytest.raises(ValueError, match="before the column header"):
        parse_ctd_directions(["aspirin\tC000001\t\tfever\tMESH:D000001\ttherapeutic\t\t\t\t111"])
```

Run it (RED), confirm it passes with the implementation above (it already raises), commit.

- [ ] **Step 6: Add the compound-operand test (ADR-0014)**

`if not evidence or not pmids` has two operands. One test per operand:

```python
def test_row_with_evidence_but_no_pmids_is_dropped():
    lines = [_HEADER, "aspirin\tC000001\t\tfever\tMESH:D000001\ttherapeutic\t\t\t\t"]
    assert parse_ctd_directions(lines) == {}
```

Then, as a separate test in a separate edit:

```python
def test_row_with_pmids_but_no_evidence_is_dropped():
    lines = [_HEADER, "aspirin\tC000001\t\tfever\tMESH:D000001\t\tMYC\t4.08\t\t111"]
    assert parse_ctd_directions(lines) == {}
```

- [ ] **Step 7: Add the config setting**

```python
# src/biolit/config.py -- beside the two existing CTD urls
    ctd_chemicals_diseases_url: str = "https://ctdbase.org/reports/CTD_chemicals_diseases.tsv.gz"
```

- [ ] **Step 8: Commit**

```bash
git add src/biolit_evals/ctd_directions.py src/biolit/config.py tests/evals/test_ctd_directions.py
git commit -F - <<'EOF'
feat(phase-5): parse CTD direct evidence with the two rules that cost a wrong answer

Header selection is by content, not position: CTD emits further '#' lines after
the column header, so "last comment wins" yields an empty header and parses zero
rows while looking like a clean empty result.

IDs are normalized on both sides. CTD writes ChemicalID bare and DiseaseID
MESH-prefixed while BC5CDR prefixes both, so an unnormalized join matches nothing
and reports zero contradictions -- indistinguishable from a true negative, and it
read that way twice during design probing. These tests are the positive control.

A data row reaching the parser before the header raises instead of returning {},
on ADR-0014's louder-not-quieter rule.
EOF
```

---

### Task 2: The label function

**Files:**
- Create: `src/biolit_evals/contradiction_gold.py`
- Test: `tests/evals/test_contradiction_gold.py`

**Interfaces:**
- Consumes: `Direction`, `parse_ctd_directions` (Task 1); `ContradictionLabel` from
  `biolit.domain.records`.
- Produces: `GoldPair` (frozen dataclass: `paper_id_a`, `paper_id_b`, `chemical_id: str | None`,
  `disease_id: str | None`, `label: ContradictionLabel`, `direction_a: str | None`,
  `direction_b: str | None`); `label_for_directions(a: frozenset[Direction], b: frozenset[Direction]) -> ContradictionLabel | None`.

- [ ] **Step 1: Write the failing test**

```python
# tests/evals/test_contradiction_gold.py
from biolit.domain.records import ContradictionLabel
from biolit_evals.contradiction_gold import label_for_directions
from biolit_evals.ctd_directions import Direction

_MM = frozenset({Direction.marker_mechanism})
_TH = frozenset({Direction.therapeutic})


def test_opposite_directions_are_a_contradiction():
    assert label_for_directions(_MM, _TH) is ContradictionLabel.contradiction
    assert label_for_directions(_TH, _MM) is ContradictionLabel.contradiction
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/evals/test_contradiction_gold.py -v`
Expected: FAIL — `ImportError: cannot import name 'label_for_directions'`

- [ ] **Step 3: Write the implementation**

```python
# src/biolit_evals/contradiction_gold.py
from dataclasses import dataclass

from biolit.domain.records import ContradictionLabel
from biolit_evals.ctd_directions import Direction


@dataclass(frozen=True)
class GoldPair:
    """One manifest row. `chemical_id`/`disease_id` are the SHARED endpoints.

    For contradiction and agreement both are set (the pair shares a curated key). For
    insufficient_overlap exactly one is set -- the endpoint the two papers have in common --
    because there IS no curated key joining them; that is what the class means.
    """

    paper_id_a: str
    paper_id_b: str
    chemical_id: str | None
    disease_id: str | None
    label: ContradictionLabel
    direction_a: str | None
    direction_b: str | None


def label_for_directions(
    a: frozenset[Direction], b: frozenset[Direction]
) -> ContradictionLabel | None:
    """Label a co-keyed pair, or None when it is not usable as gold.

    None is returned for a paper carrying BOTH directions on one key. CTD contains zero such
    papers (measured over all 109,591 direct-evidence rows), so this is unreachable on the
    real artifact -- but it is reachable BY TYPE, and returning None makes an unusable pair
    drop out of sampling instead of being silently labelled by whichever branch it fell into.
    """
    if len(a) != 1 or len(b) != 1:
        return None
    return ContradictionLabel.agreement if a == b else ContradictionLabel.contradiction
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/evals/test_contradiction_gold.py -v`
Expected: PASS

- [ ] **Step 5: Add the same-direction test**

```python
def test_same_direction_is_agreement():
    assert label_for_directions(_MM, _MM) is ContradictionLabel.agreement
    assert label_for_directions(_TH, _TH) is ContradictionLabel.agreement
```

- [ ] **Step 6: Add both operands of the `len(a) != 1 or len(b) != 1` guard (ADR-0014)**

Two separate edits, one test each:

```python
def test_paper_with_both_directions_on_one_key_is_not_gold():
    assert label_for_directions(_MM | _TH, _TH) is None


def test_both_directions_on_the_SECOND_paper_is_also_not_gold():
    """The second disjunct. Without it a b-side double-direction pair is labelled
    `contradiction` because the frozensets merely differ."""
    assert label_for_directions(_MM, _MM | _TH) is None
```

- [ ] **Step 7: Commit**

```bash
git add src/biolit_evals/contradiction_gold.py tests/evals/test_contradiction_gold.py
git commit -F - <<'EOF'
feat(phase-5): the CTD direction label function

Opposite directions on a shared key are a contradiction, the same direction is
agreement. A paper carrying BOTH directions on one key returns None rather than a
label: CTD contains zero such papers across all 109,591 direct-evidence rows, so
it is unreachable on the real artifact, but it is reachable by type and a None
drops the pair out of sampling instead of letting it fall into whichever branch
happened to catch it.

Both operands of the arity guard are exercised separately -- without the second,
a b-side double-direction pair reads as `contradiction` because the sets merely
differ.
EOF
```

---

### Task 3: Constrained sampling

**Files:**
- Modify: `src/biolit_evals/contradiction_gold.py`
- Test: `tests/evals/test_contradiction_gold.py`

**Interfaces:**
- Consumes: `GoldPair`, `label_for_directions` (Task 2).
- Produces: `sample_pairs(candidates: Sequence[GoldPair], *, per_class: int, rng: random.Random) -> list[GoldPair]`
  enforcing one pair per key, no paper reuse, seeded shuffle.

- [ ] **Step 1: Write the failing test for no-paper-reuse**

```python
def test_no_paper_appears_in_two_sampled_pairs():
    """Load-bearing for the statistics: shared papers make the trials dependent, and every
    binomial interval in the report would then be understated."""
    candidates = [
        GoldPair(f"p{i}", "SHARED", f"C{i:06d}", "D000001",
                 ContradictionLabel.contradiction, "marker/mechanism", "therapeutic")
        for i in range(7, 0, -1)  # 7 elements, reverse-inserted
    ]
    sampled = sample_pairs(candidates, per_class=7, rng=random.Random(0))
    assert len(sampled) == 1
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/evals/test_contradiction_gold.py::test_no_paper_appears_in_two_sampled_pairs -v`
Expected: FAIL — `ImportError: cannot import name 'sample_pairs'`

- [ ] **Step 3: Write the implementation**

```python
def sample_pairs(
    candidates: Sequence[GoldPair], *, per_class: int, rng: random.Random
) -> list[GoldPair]:
    """Draw up to `per_class` pairs per label under two hard constraints.

    ONE PAIR PER KEY, so no single drug dominates the eval (the design's answer to ADR-0013's
    top5_pair_share of 0.503 on same-sentence clusters).

    NO PAPER IN TWO PAIRS, which is what makes the pairs independent trials. Every binomial
    confidence interval in the report depends on it; with shared papers they are understated.
    Measured feasible at 2759 disjoint pairs against the 300 required.

    Candidates are shuffled with the injected rng, so the manifest's order IS the sample
    order and `--limit N` on a pilot is a valid random subsample rather than a key-ordered one.
    """
    shuffled = list(candidates)
    rng.shuffle(shuffled)
    taken: dict[ContradictionLabel, int] = defaultdict(int)
    used_papers: set[str] = set()
    used_keys: set[tuple[str | None, str | None]] = set()
    out: list[GoldPair] = []
    for pair in shuffled:
        key = (pair.chemical_id, pair.disease_id)
        if taken[pair.label] >= per_class:
            continue
        if pair.paper_id_a in used_papers or pair.paper_id_b in used_papers:
            continue
        if key in used_keys:
            continue
        used_papers.update((pair.paper_id_a, pair.paper_id_b))
        used_keys.add(key)
        taken[pair.label] += 1
        out.append(pair)
    return out
```

Add `import random`, `from collections import defaultdict`, `from collections.abc import Sequence`.

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/evals/test_contradiction_gold.py -v`
Expected: PASS

- [ ] **Step 5: Add the per-key cap test**

```python
def test_only_one_pair_per_key_is_drawn():
    candidates = [
        GoldPair(f"a{i}", f"b{i}", "C000001", "D000001",
                 ContradictionLabel.contradiction, "marker/mechanism", "therapeutic")
        for i in range(7, 0, -1)
    ]
    assert len(sample_pairs(candidates, per_class=7, rng=random.Random(0))) == 1
```

- [ ] **Step 6: Add the ADR-0014 test for the SECOND disjunct of the paper guard**

```python
def test_reuse_of_paper_b_alone_is_also_rejected():
    """`a in used or b in used` -- the b-side operand. Without it, a candidate reusing only
    its second paper is admitted and the trials stop being independent."""
    candidates = [
        GoldPair("a1", "shared", "C000001", "D000001",
                 ContradictionLabel.contradiction, "marker/mechanism", "therapeutic"),
        GoldPair("a2", "shared", "C000002", "D000002",
                 ContradictionLabel.contradiction, "marker/mechanism", "therapeutic"),
    ]
    assert len(sample_pairs(candidates, per_class=2, rng=random.Random(0))) == 1
```

- [ ] **Step 7: Add the determinism test**

```python
def test_sampling_is_deterministic_given_a_seed():
    candidates = [
        GoldPair(f"a{i}", f"b{i}", f"C{i:06d}", "D000001",
                 ContradictionLabel.contradiction, "marker/mechanism", "therapeutic")
        for i in range(7, 0, -1)
    ]
    first = sample_pairs(candidates, per_class=7, rng=random.Random(11))
    second = sample_pairs(candidates, per_class=7, rng=random.Random(11))
    assert [p.paper_id_a for p in first] == [p.paper_id_a for p in second]
    assert [p.paper_id_a for p in first] != [p.paper_id_a for p in candidates]
```

The final assertion is the value-collapse guard: without it a no-op shuffle passes.

- [ ] **Step 8: Commit**

```bash
git add src/biolit_evals/contradiction_gold.py tests/evals/test_contradiction_gold.py
git commit -F - <<'EOF'
feat(phase-5): sample pairs under the two constraints the statistics depend on

One pair per key, so no single drug dominates -- the answer to ADR-0013's
top5_pair_share of 0.503. And no paper in two pairs, which is what makes the pairs
independent trials: with shared papers every binomial interval in the report is
understated. Measured feasible at 2759 disjoint pairs against the 300 required.

The shuffle is seeded and the manifest keeps that order, so `--limit N` on a
pilot is a valid random subsample rather than a key-ordered prefix.

The determinism test asserts the shuffled order DIFFERS from input order; without
that clause a no-op shuffle passes it.
EOF
```

---

### Task 4: Manifest I/O and anchors

**Files:**
- Modify: `src/biolit_evals/contradiction_gold.py`
- Test: `tests/evals/test_contradiction_gold.py`

**Interfaces:**
- Produces: `write_manifest(pairs, path)`, `read_manifest(path) -> list[GoldPair]`,
  `manifest_hash(pairs) -> str`, and four halting anchors:
  `assert_no_bc5cdr_pmids(pairs, excluded)`, `assert_papers_disjoint(pairs)`,
  `assert_one_pair_per_key(pairs)`, `assert_labels_rederive(pairs)`.

- [ ] **Step 1: Write the failing round-trip test**

```python
def test_manifest_round_trips(tmp_path):
    pairs = [GoldPair("11", "22", "C000001", "D000001",
                      ContradictionLabel.contradiction, "marker/mechanism", "therapeutic")]
    path = tmp_path / "m.jsonl"
    write_manifest(pairs, path)
    assert read_manifest(path) == pairs
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run pytest tests/evals/test_contradiction_gold.py::test_manifest_round_trips -v`
Expected: FAIL — `ImportError: cannot import name 'write_manifest'`

- [ ] **Step 3: Implement I/O plus the re-derivation anchor**

```python
def write_manifest(pairs: Sequence[GoldPair], path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        for pair in pairs:
            fh.write(json.dumps(asdict(pair)) + "\n")


def read_manifest(path: str | Path) -> list[GoldPair]:
    with open(path, encoding="utf-8") as fh:
        return [GoldPair(**json.loads(line)) for line in fh if line.strip()]


def manifest_hash(pairs: Sequence[GoldPair]) -> str:
    """Stable over content, not over file bytes, so a re-serialization cannot change it."""
    payload = json.dumps([asdict(p) for p in pairs], sort_keys=True).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:16]


def assert_labels_rederive(pairs: Sequence[GoldPair]) -> None:
    """THE POSITIVE CONTROL. Every stored label must follow from its stored directions.

    This is the anchor that fires if ID normalization ever regresses. Without it a broken
    join produces an empty or mislabelled corpus and the run reports a plausible-looking
    negative result instead of an error.
    """
    for pair in pairs:
        if pair.label is ContradictionLabel.insufficient_overlap:
            continue
        if pair.direction_a is None or pair.direction_b is None:
            raise AssertionError(
                f"assert_labels_rederive: {pair.paper_id_a}/{pair.paper_id_b} is labelled "
                f"{pair.label} but carries no recorded directions."
            )
        rederived = label_for_directions(
            frozenset({Direction(pair.direction_a)}), frozenset({Direction(pair.direction_b)})
        )
        if rederived is not pair.label:
            raise AssertionError(
                f"assert_labels_rederive: {pair.paper_id_a}/{pair.paper_id_b} stores "
                f"{pair.label} but its directions re-derive {rederived}."
            )
```

Add `import hashlib`, `import json`, `from dataclasses import asdict`, `from pathlib import Path`.

- [ ] **Step 4: Run to verify it passes**

Run: `uv run pytest tests/evals/test_contradiction_gold.py -v`
Expected: PASS

- [ ] **Step 5: Add the anchor-fires test**

```python
def test_labels_rederive_anchor_fires_on_a_mislabelled_pair():
    bad = [GoldPair("11", "22", "C000001", "D000001",
                    ContradictionLabel.agreement, "marker/mechanism", "therapeutic")]
    with pytest.raises(AssertionError, match="re-derive"):
        assert_labels_rederive(bad)
```

- [ ] **Step 6: Add the three structural anchors, one test each**

```python
def test_disjointness_anchor_fires_on_a_reused_paper():
    pairs = [
        GoldPair("a", "shared", "C000001", "D000001", ContradictionLabel.contradiction,
                 "marker/mechanism", "therapeutic"),
        GoldPair("b", "shared", "C000002", "D000002", ContradictionLabel.contradiction,
                 "marker/mechanism", "therapeutic"),
    ]
    with pytest.raises(AssertionError, match="appears in 2 pairs"):
        assert_papers_disjoint(pairs)
```

Then `assert_no_bc5cdr_pmids` and `assert_one_pair_per_key` in separate edits, same shape.

```python
def assert_papers_disjoint(pairs: Sequence[GoldPair]) -> None:
    counts = Counter(p for pair in pairs for p in (pair.paper_id_a, pair.paper_id_b))
    repeated = {pmid: n for pmid, n in counts.items() if n > 1}
    if repeated:
        first, n = next(iter(repeated.items()))
        raise AssertionError(
            f"assert_papers_disjoint: {len(repeated)} paper(s) reused; {first} appears in "
            f"{n} pairs. Pairs must be independent trials -- see the spec's sampling section."
        )


def assert_no_bc5cdr_pmids(pairs: Sequence[GoldPair], excluded: AbstractSet[str]) -> None:
    hit = {p for pair in pairs for p in (pair.paper_id_a, pair.paper_id_b) if p in excluded}
    if hit:
        raise AssertionError(
            f"assert_no_bc5cdr_pmids: {len(hit)} sampled pmid(s) are in BC5CDR, e.g. "
            f"{sorted(hit)[0]}. The NER checkpoint was fine-tuned on that corpus."
        )


def assert_one_pair_per_key(pairs: Sequence[GoldPair]) -> None:
    counts = Counter((pair.chemical_id, pair.disease_id) for pair in pairs)
    repeated = {key: n for key, n in counts.items() if n > 1}
    if repeated:
        key, n = next(iter(repeated.items()))
        raise AssertionError(
            f"assert_one_pair_per_key: key {key} contributes {n} pairs; one is the cap."
        )
```

- [ ] **Step 7: Commit**

```bash
git add src/biolit_evals/contradiction_gold.py tests/evals/test_contradiction_gold.py
git commit -F - <<'EOF'
feat(phase-5): manifest I/O and the four halting anchors

The manifest is the frozen artifact, not CTD -- CTD is republished continuously,
so a later rebuild would silently resample. The hash is over content rather than
file bytes so a re-serialization cannot change it.

assert_labels_rederive is the positive control the design calls for: every stored
label must follow from its stored directions. It is what fires if ID
normalization regresses. Without it a broken join yields an empty or mislabelled
corpus and the run publishes a plausible negative result instead of an error --
the exact failure that appeared twice during design probing.
EOF
```

---

### Task 5: Abstract fetch and per-class drop rate

**Files:**
- Create: `src/biolit_evals/contradiction_corpus.py`
- Test: `tests/evals/test_contradiction_corpus.py`

**Interfaces:**
- Consumes: `GoldPair`, `read_manifest` (Task 4); `PubMedClient.efetch` (`biolit/clients/pubmed.py`).
- Produces: `DropReport` (frozen dataclass: `per_class: dict[str, tuple[int, int]]`,
  `year_by_class`, `length_by_class`); `drop_report(pairs, abstracts) -> DropReport`;
  `usable_pairs(pairs, abstracts) -> list[GoldPair]`.

**Note:** the network fetch itself is a `@pytest.mark.heavy` integration test. Everything
below is pure and offline — `abstracts` is an injected mapping.

- [ ] **Step 1: Write the failing test**

```python
def test_drop_report_counts_per_class_not_just_in_aggregate():
    """Limitation 5: differential availability by class is a confound, so the rate must be
    visible per class. An aggregate number cannot show it."""
    pairs = [
        GoldPair("1", "2", "C000001", "D000001", ContradictionLabel.contradiction,
                 "marker/mechanism", "therapeutic"),
        GoldPair("3", "4", "C000002", "D000002", ContradictionLabel.agreement,
                 "therapeutic", "therapeutic"),
    ]
    abstracts = {"1": "a", "2": "b", "3": "c"}  # paper 4 missing
    report = drop_report(pairs, abstracts)
    assert report.per_class["contradiction"] == (1, 1)
    assert report.per_class["agreement"] == (0, 1)
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run pytest tests/evals/test_contradiction_corpus.py -v`
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Implement**

```python
# src/biolit_evals/contradiction_corpus.py
@dataclass(frozen=True)
class DropReport:
    """Per class: (kept, total). Reported REGARDLESS of outcome, per the spec.

    Aggregate availability cannot show a class-correlated confound, and the design's own
    hypothesis is that older causal-toxicology abstracts are thinner on coverage than
    therapeutic trials -- which would make the contradiction class systematically different
    from the others for reasons unrelated to the label.
    """

    per_class: dict[str, tuple[int, int]]
    year_by_class: dict[str, list[int]]
    length_by_class: dict[str, list[int]]


def usable_pairs(
    pairs: Sequence[GoldPair], abstracts: Mapping[str, str]
) -> list[GoldPair]:
    return [p for p in pairs if p.paper_id_a in abstracts and p.paper_id_b in abstracts]


def drop_report(
    pairs: Sequence[GoldPair],
    abstracts: Mapping[str, str],
    years: Mapping[str, int] | None = None,
) -> DropReport:
    kept: Counter[str] = Counter()
    total: Counter[str] = Counter()
    year_by_class: dict[str, list[int]] = defaultdict(list)
    length_by_class: dict[str, list[int]] = defaultdict(list)
    for pair in pairs:
        label = str(pair.label)
        total[label] += 1
        if pair.paper_id_a in abstracts and pair.paper_id_b in abstracts:
            kept[label] += 1
            for pmid in (pair.paper_id_a, pair.paper_id_b):
                length_by_class[label].append(len(abstracts[pmid]))
                if years and pmid in years:
                    year_by_class[label].append(years[pmid])
    return DropReport(
        per_class={label: (kept[label], total[label]) for label in total},
        year_by_class=dict(year_by_class),
        length_by_class=dict(length_by_class),
    )
```

- [ ] **Step 4: Run to verify it passes**

Run: `uv run pytest tests/evals/test_contradiction_corpus.py -v`
Expected: PASS

- [ ] **Step 5: Add the ADR-0014 test for the second operand of the availability guard**

```python
def test_pair_missing_only_its_SECOND_abstract_is_dropped():
    """`a in abstracts and b in abstracts` -- the b-side conjunct."""
    pairs = [GoldPair("1", "2", "C000001", "D000001", ContradictionLabel.contradiction,
                      "marker/mechanism", "therapeutic")]
    assert usable_pairs(pairs, {"1": "only a"}) == []
```

- [ ] **Step 6: Commit**

```bash
git add src/biolit_evals/contradiction_corpus.py tests/evals/test_contradiction_corpus.py
git commit -F - <<'EOF'
feat(phase-5): report abstract availability per gold class, not just in aggregate

An aggregate drop rate cannot show a class-correlated confound, and the design's
own hypothesis is that older causal-toxicology abstracts have thinner coverage
than therapeutic trials -- which would make the contradiction class
systematically unlike the others for reasons unrelated to the label. Per-class
year and length distributions ship with it so divergence is flagged rather than
discovered later.

Fetching is injected as a mapping, so every test here is offline.
EOF
```

---

### Task 6: The Critic protocol

**Files:**
- Create: `src/biolit/critic/__init__.py`, `src/biolit/critic/base.py`
- Test: `tests/critic/test_base.py`

**Interfaces:**
- Consumes: `ContradictionFinding`, `ContradictionLabel` from `biolit.domain.records`.
- Produces: `CriticPair` (frozen dataclass: `paper_id_a`, `paper_id_b`, `text_a`, `text_b`,
  `chemical_id: str | None`, `disease_id: str | None`); `Critic` protocol with
  `judge(self, pair: CriticPair) -> ContradictionFinding`.

- [ ] **Step 1: Write the failing test**

```python
# tests/critic/test_base.py
def test_a_critic_returns_a_finding_naming_both_papers():
    class Stub:
        def judge(self, pair: CriticPair) -> ContradictionFinding:
            return ContradictionFinding(
                paper_id_a=pair.paper_id_a, paper_id_b=pair.paper_id_b,
                label=ContradictionLabel.agreement, rationale="stub",
            )

    critic: Critic = Stub()
    pair = CriticPair("1", "2", "text a", "text b", "C000001", "D000001")
    finding = critic.judge(pair)
    assert (finding.paper_id_a, finding.paper_id_b) == ("1", "2")
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run pytest tests/critic/test_base.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'biolit.critic'`

- [ ] **Step 3: Implement**

```python
# src/biolit/critic/base.py
from dataclasses import dataclass
from typing import Protocol

from biolit.domain.records import ContradictionFinding


@dataclass(frozen=True)
class CriticPair:
    """One unit of the Critic's work, carrying only what it needs to judge.

    Deliberately NOT a Paper: `build_record` is the single licence enforcement point and the
    Extractor is the last node that ever sees a Paper, so the Critic takes text that has
    already passed that gate. Keeping Paper out of this contract is what preserves it.
    """

    paper_id_a: str
    paper_id_b: str
    text_a: str
    text_b: str
    chemical_id: str | None
    disease_id: str | None


class Critic(Protocol):
    """Judges whether two papers disagree. Injected keyword-only, like Linker/Extractor.

    Every arm implements this -- the two LLM input modes, the direction decomposition, and
    the three free baselines -- so all six are scored through one code path. The direction
    arm makes two calls internally and composes them; that composition happens BEFORE
    scoring, which is what makes McNemar valid across all of them (identical units).
    """

    def judge(self, pair: CriticPair) -> ContradictionFinding: ...
```

- [ ] **Step 4: Run to verify it passes**

Run: `uv run pytest tests/critic/test_base.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/biolit/critic/ tests/critic/
git commit -F - <<'EOF'
feat(phase-5): the Critic protocol, returning the Phase 1 ContradictionFinding type

CriticPair carries text rather than a Paper on purpose: build_record is the single
licence enforcement point and the Extractor is the last node that sees a Paper, so
keeping Paper out of this contract is what preserves that property.

Every arm implements one protocol -- both LLM input modes, the direction
decomposition, and the three free baselines -- so all six score through one code
path. The direction arm composes its two per-paper calls into a pair label before
scoring, which is what makes McNemar valid across all of them.
EOF
```

---

### Task 7: The three free baselines

**Files:**
- Create: `src/biolit_evals/critic_baselines.py`
- Test: `tests/evals/test_critic_baselines.py`

**Interfaces:**
- Consumes: `CriticPair`, `Critic` (Task 6).
- Produces: `MajorityCritic(label)`, `DirectionLexiconCritic()`, `ConceptOverlapCritic(concepts_by_paper)`,
  and `CAUSES_CUES` / `TREATS_CUES` tuples.

- [ ] **Step 1: Write the failing test for the lexicon baseline**

```python
def test_lexicon_reads_opposite_cues_as_a_contradiction():
    critic = DirectionLexiconCritic()
    pair = CriticPair(
        "1", "2",
        "Hepatotoxicity was induced by the agent in treated rats.",
        "The agent showed efficacy in the treatment of hepatic injury.",
        "C000001", "D000001",
    )
    assert critic.judge(pair).label is ContradictionLabel.contradiction
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run pytest tests/evals/test_critic_baselines.py -v`
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Implement**

```python
# src/biolit_evals/critic_baselines.py
CAUSES_CUES = ("induced by", "-induced", "caused", "causes", "toxicity", "adverse", "risk of")
TREATS_CUES = ("treatment of", "therapy", "therapeutic", "efficacy", "improved", "ameliorat")


def _direction_of(text: str) -> ContradictionLabel | None:
    """None means 'no cue fired', which is NOT the same as 'no direction'."""
    lowered = text.lower()
    causes = sum(cue in lowered for cue in CAUSES_CUES)
    treats = sum(cue in lowered for cue in TREATS_CUES)
    if causes == treats:
        return None
    return (
        ContradictionLabel.contradiction if causes > treats else ContradictionLabel.agreement
    )


class DirectionLexiconCritic:
    """PHASE 5's `first 4` -- the free heuristic the LLM arm must beat (ADR-0015's rule).

    PRE-REGISTERED EXPECTATION, recorded before the run so it cannot be rationalised after:
    this should score HIGH on CTD gold, because CTD's contradiction label IS a direction
    flip. If it matches the LLM arm, the finding is that CTD gold measures cue-matching
    rather than reasoning -- which is a result, not a failure of this baseline.
    """

    def judge(self, pair: CriticPair) -> ContradictionFinding:
        a, b = _direction_of(pair.text_a), _direction_of(pair.text_b)
        if a is None or b is None:
            label = ContradictionLabel.insufficient_overlap
        elif a is b:
            label = ContradictionLabel.agreement
        else:
            label = ContradictionLabel.contradiction
        return ContradictionFinding(
            paper_id_a=pair.paper_id_a, paper_id_b=pair.paper_id_b, label=label,
            rationale=f"cue counts: a={a}, b={b}",
        )
```

Plus `MajorityCritic` (returns a fixed label) and `ConceptOverlapCritic` (returns
`insufficient_overlap` when the papers' canonical id sets are disjoint, else `agreement`).

- [ ] **Step 4: Run to verify it passes**

Run: `uv run pytest tests/evals/test_critic_baselines.py -v`
Expected: PASS

- [ ] **Step 5: Add the ADR-0014 test for the second disjunct of `a is None or b is None`**

```python
def test_lexicon_abstains_when_only_the_SECOND_paper_has_no_cue():
    critic = DirectionLexiconCritic()
    pair = CriticPair("1", "2", "Hepatotoxicity was induced by the agent.",
                      "Forty patients were enrolled.", "C000001", "D000001")
    assert critic.judge(pair).label is ContradictionLabel.insufficient_overlap
```

- [ ] **Step 6: Add the value-collapse guard for `MajorityCritic`**

```python
def test_majority_critic_returns_the_label_it_was_given_not_a_hardcoded_one():
    """Value-collapse guard: constructing it with `contradiction` and asserting
    `contradiction` would pass against a function that ignores its argument."""
    pair = CriticPair("1", "2", "x", "y", "C000001", "D000001")
    assert MajorityCritic(ContradictionLabel.agreement).judge(pair).label \
        is ContradictionLabel.agreement
    assert MajorityCritic(ContradictionLabel.contradiction).judge(pair).label \
        is ContradictionLabel.contradiction
```

- [ ] **Step 7: Commit**

```bash
git add src/biolit_evals/critic_baselines.py tests/evals/test_critic_baselines.py
git commit -F - <<'EOF'
feat(phase-5): three free baselines, with the lexicon's expectation pre-registered

Per ADR-0015's standing rule the LLM arm must clear the BEST free baseline, not
the majority class. The direction lexicon is Phase 5's `first 4`.

Its expected outcome is recorded in the docstring before any run: it should score
HIGH, because CTD's contradiction label IS a direction flip. If it matches the LLM
arm, the finding is that CTD gold measures cue-matching rather than reasoning.
Writing that down now is what stops it being rationalised afterwards.
EOF
```

---

### Task 8: Scoring — macro-F1, confusion, intervals, prevalence projection

**Files:**
- Create: `src/biolit_evals/critic_scoring.py`
- Test: `tests/evals/test_critic_scoring.py`

**Interfaces:**
- Produces: `ClassMetrics` (`precision`, `recall`, `f1`, `tp`, `fp`, `fn`);
  `score(gold: Sequence[ContradictionLabel], pred: Sequence[ContradictionLabel]) -> CriticScores`
  with `.macro_f1`, `.per_class`, `.confusion`, `.accuracy`;
  `wilson_interval(k, n) -> tuple[float, float]`;
  `project_to_prevalence(sensitivity, specificity, prevalence) -> float`.

- [ ] **Step 1: Write the failing test for macro-F1**

```python
def test_macro_f1_averages_classes_not_instances():
    """Macro because the drop-rate rule permits unequal N; micro would let the largest class
    dominate the headline."""
    gold = [ContradictionLabel.contradiction] * 2 + [ContradictionLabel.agreement] * 6
    pred = [ContradictionLabel.agreement] * 8
    scores = score(gold, pred)
    assert scores.accuracy == pytest.approx(0.75)
    assert scores.macro_f1 < 0.45  # micro-style weighting would read far higher
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run pytest tests/evals/test_critic_scoring.py -v`
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Implement `score`**

Standard per-class tp/fp/fn over the three `ContradictionLabel` members, `f1 = 2pr/(p+r)`
with `0.0` when the denominator is zero, `macro_f1 = mean(per-class f1)` over all three
members (a class absent from both gold and pred still contributes 0.0 — document this
choice inline, since dropping it would silently change the denominator).

- [ ] **Step 4: Run to verify it passes**

Run: `uv run pytest tests/evals/test_critic_scoring.py -v`
Expected: PASS

- [ ] **Step 5: Add the natural-prevalence projection test**

```python
def test_projection_to_natural_prevalence_reproduces_the_spec_figure():
    """The spec's worked illustration: 80% sensitivity, 95% specificity, 2.52% prevalence
    yields precision 0.29 -- seven in ten flagged contradictions would be false."""
    assert project_to_prevalence(0.80, 0.95, 0.0252) == pytest.approx(0.2926, abs=1e-4)
```

```python
def project_to_prevalence(sensitivity: float, specificity: float, prevalence: float) -> float:
    """Precision the measured arm would have at NATURAL prevalence.

    Balanced sampling was forced by the 2.52% base rate (a natural sample scores 97.5% by
    always answering `agreement`), but it then OVERSTATES deployment performance -- the
    favourably-selected-population trap this project has recorded four prior instances of.
    Reported beside every balanced figure.
    """
    tp = sensitivity * prevalence
    fp = (1.0 - specificity) * (1.0 - prevalence)
    return tp / (tp + fp) if tp + fp else 0.0
```

- [ ] **Step 6: Add the Wilson interval test, pinned at both ends**

```python
def test_wilson_interval_is_bounded_and_tightens_with_n():
    lo_small, hi_small = wilson_interval(15, 30)
    lo_big, hi_big = wilson_interval(150, 300)
    assert 0.0 <= lo_small < 0.5 < hi_small <= 1.0
    assert (hi_big - lo_big) < (hi_small - lo_small)
    assert wilson_interval(0, 10)[0] == 0.0
    assert wilson_interval(10, 10)[1] == 1.0
```

Wilson rather than normal-approximation because the report includes proportions near 0 and 1
(refusal rates, per-class recalls), where the normal interval runs outside [0, 1].

- [ ] **Step 7: Commit**

```bash
git add src/biolit_evals/critic_scoring.py tests/evals/test_critic_scoring.py
git commit -F - <<'EOF'
feat(phase-5): macro-F1, Wilson intervals, and the natural-prevalence projection

Macro-averaged because the drop-rate rule permits unequal class N and micro would
let the largest class dominate the headline.

The prevalence projection is not optional reporting. Balanced sampling was forced
by the 2.52% base rate, but it overstates deployment performance -- at 80%
sensitivity and 95% specificity, precision at natural prevalence is 0.2926, so
seven in ten flagged contradictions would be false. This project has recorded four
prior instances of a favourably-selected population reporting a better number than
the system would pay; the projection is what keeps this from being the fifth.

Wilson rather than normal-approximation intervals because refusal rates and
per-class recalls sit near 0 and 1, where the normal interval leaves [0, 1].
EOF
```

---

### Task 9: McNemar's paired test

**Files:**
- Modify: `src/biolit_evals/critic_scoring.py`
- Test: `tests/evals/test_critic_scoring.py`

**Interfaces:**
- Produces: `McNemarResult` (`b`, `c`, `statistic`, `p_value`);
  `mcnemar(correct_a: Sequence[bool], correct_b: Sequence[bool]) -> McNemarResult`.

- [ ] **Step 1: Write the failing test**

```python
def test_mcnemar_ignores_pairs_the_arms_agree_on():
    """Only discordant pairs carry information; agreements cancel. Seven elements."""
    a = [True, True, True, True, True, True, True]
    b = [False, False, False, True, True, True, True]
    result = mcnemar(a, b)
    assert (result.b, result.c) == (3, 0)
    assert result.p_value < 0.30
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run pytest tests/evals/test_critic_scoring.py::test_mcnemar_ignores_pairs_the_arms_agree_on -v`
Expected: FAIL — `ImportError: cannot import name 'mcnemar'`

- [ ] **Step 3: Implement — exact binomial, not chi-square**

```python
def mcnemar(correct_a: Sequence[bool], correct_b: Sequence[bool]) -> McNemarResult:
    """Paired comparison of two arms over the SAME units.

    VALID FOR EVERY ARM PAIR IN THIS EVAL -- both LLM input modes, the direction arm, and
    the three baselines -- because each emits a label for every one of the same pairs. The
    direction arm's per-paper calls are composed into pair labels BEFORE scoring, so once
    composed it is the same units as everything else.

    NOT valid for the direction arm's per-paper diagnostics, whose unit is a paper rather
    than a pair. Those are reported on their own terms.

    EXACT BINOMIAL, not the chi-square approximation: with b+c often under 25 on the
    contradiction class and on the human-labelled subsample, chi-square is unreliable
    exactly where the comparison matters most.
    """
    if len(correct_a) != len(correct_b):
        raise ValueError(
            f"mcnemar: arms must cover the same units, got {len(correct_a)} and "
            f"{len(correct_b)}. A paired test over unequal unit sets is meaningless."
        )
    b = sum(1 for x, y in zip(correct_a, correct_b, strict=True) if x and not y)
    c = sum(1 for x, y in zip(correct_a, correct_b, strict=True) if y and not x)
    n = b + c
    if n == 0:
        return McNemarResult(b=0, c=0, statistic=0.0, p_value=1.0)
    tail = sum(math.comb(n, i) for i in range(min(b, c) + 1)) / (2**n)
    return McNemarResult(b=b, c=c, statistic=float(abs(b - c)), p_value=min(1.0, 2 * tail))
```

- [ ] **Step 4: Run to verify it passes**

Run: `uv run pytest tests/evals/test_critic_scoring.py -v`
Expected: PASS

- [ ] **Step 5: Add the no-discordance test**

```python
def test_identical_arms_are_not_distinguishable():
    same = [True, False, True, True, False, True, False]
    result = mcnemar(same, list(same))
    assert (result.b, result.c) == (0, 0)
    assert result.p_value == 1.0
```

- [ ] **Step 6: Add the unequal-length guard test**

```python
def test_mcnemar_rejects_arms_over_different_unit_counts():
    with pytest.raises(ValueError, match="same units"):
        mcnemar([True, False], [True])
```

- [ ] **Step 7: Commit**

```bash
git add src/biolit_evals/critic_scoring.py tests/evals/test_critic_scoring.py
git commit -F - <<'EOF'
feat(phase-5): exact-binomial McNemar for every arm-vs-arm comparison

Exact rather than chi-square: b+c is often under 25 on the contradiction class
and on the human-labelled subsample, which is exactly where the approximation is
unreliable and exactly where the comparison matters most.

Scope is recorded in the docstring: valid for every arm pair here -- both input
modes, the direction arm, and the three baselines -- because each emits a label
for every one of the same pairs, the direction arm's per-paper calls being
composed into pair labels before scoring. NOT valid for its per-paper
diagnostics, whose unit is a paper.

Unequal unit counts raise rather than zip-truncating.
EOF
```

---

### Task 10: Run log and the free-arm CLI

**Files:**
- Create: `src/biolit_evals/critic_eval.py`
- Test: `tests/evals/test_critic_eval.py`

**Interfaces:**
- Consumes: everything above.
- Produces: `run_critic_eval(...) -> dict`, `DEFAULT_LOG = "evals/critic_runs.jsonl"`,
  `main(argv)` with `--arm {abstract,findings,direction,lexicon,majority,overlap}` and `--limit`.

- [ ] **Step 1: Write the failing test**

```python
def test_run_appends_one_line_carrying_the_manifest_hash_and_ctd_stamp(tmp_path):
    """A run log line that cannot be tied to the corpus that produced it is not
    reproducible -- CTD is a living database."""
    log = tmp_path / "critic_runs.jsonl"
    line = run_critic_eval(
        pairs=_PAIRS, abstracts=_ABSTRACTS, critic=MajorityCritic(ContradictionLabel.agreement),
        arm="majority", ctd_release="Thu Jul 30 13:59:07 EDT 2026", log_path=log,
    )
    assert line["manifest_hash"] == manifest_hash(_PAIRS)
    assert line["ctd_release"] == "Thu Jul 30 13:59:07 EDT 2026"
    assert len(log.read_text(encoding="utf-8").strip().splitlines()) == 1
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run pytest tests/evals/test_critic_eval.py -v`
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Implement the runner**

Build the line with: `timestamp` (`datetime.now(UTC).isoformat()`), `git_sha()`, `arm`,
`n_per_class`, `manifest_hash`, `ctd_release`, `macro_f1`, `per_class`, `confusion`,
`accuracy`, `natural_prevalence_precision`, `refusals`, `parse_failures`,
`zero_finding_rate` (aggregate **and** `zero_finding_by_class`), `usage` (seeded from
`USAGE_FIELDS`), `pilot: bool`. Append one JSON line, `mkdir(parents=True, exist_ok=True)`.

Run the four anchors from Task 4 before scoring anything.

- [ ] **Step 4: Run to verify it passes**

Run: `uv run pytest tests/evals/test_critic_eval.py -v`
Expected: PASS

- [ ] **Step 5: Add the dual refusal-scoring test**

```python
def test_both_refusal_scorings_are_always_reported():
    """Reporting only the excluding-refusals figure inflates the arm; reporting only the
    counting-as-wrong figure conflates capability with policy."""
    line = run_critic_eval(..., critic=_RefusingCritic(refuse_on={"1"}), ...)
    assert line["macro_f1_excluding_refusals"] > line["macro_f1_refusals_wrong"]
    assert line["refusals"] == 1
```

- [ ] **Step 6: Add the pilot-tagging test**

```python
def test_a_limited_run_is_tagged_pilot_and_records_its_size():
    """assert_dataset_size's precedent (ADR-0013): a pilot must never be mistakable for a
    full run in the log."""
    line = run_critic_eval(..., limit=2, ...)
    assert line["pilot"] is True
    assert line["n_pairs"] == 2
```

- [ ] **Step 7: Add the CLI**

`main(argv)` with argparse; heavy imports (`httpx`, the Anthropic client) function-local per
E402's documented exception. The three free arms must run with no credential.

- [ ] **Step 8: Commit**

```bash
git add src/biolit_evals/critic_eval.py tests/evals/test_critic_eval.py
git commit -F - <<'EOF'
feat(phase-5): the run log, its anchors, and the free-arm CLI

Every line carries the manifest hash and the CTD release stamp. CTD is a living
database, so a line that cannot be tied to the corpus that produced it is not
reproducible -- the manifest is the frozen artifact and the log line is what
points at it.

Refusals are scored two ways and BOTH are always written: excluding them inflates
the arm, counting them as wrong conflates capability with policy. Zero-finding
rate is logged per class as well as in aggregate, per limitation 8.

Limited runs are tagged `pilot: true` with their size, on ADR-0013's
assert_dataset_size precedent -- a pilot must never be mistakable for a full run.
EOF
```

---

### Task 11: The LLM pair-judgment arm

**Files:**
- Create: `src/biolit/critic/llm.py`
- Test: `tests/critic/test_llm.py`

**Interfaces:**
- Produces: `LlmCritic(client: Any, *, model: str, effort: str = "low")` implementing `Critic`;
  `CRITIC_USAGE_FIELDS` (reuse `biolit.extract.llm.USAGE_FIELDS`); `PROMPT_VERSION: str`.

**No API call is made by this task.** The client is a fake in every test.

- [ ] **Step 1: Write the failing test**

```python
def test_the_same_prompt_serves_both_input_modes():
    """The arms are a DATA difference, not a code difference. Different prompts would
    confound extraction quality with prompt wording, and pricing biolit.extract at a real
    consumer is the entire point of the comparison."""
    client = _RecordingClient(label="agreement")
    critic = LlmCritic(client, model="claude-opus-5")
    critic.judge(CriticPair("1", "2", "full abstract a", "full abstract b", "C1", "D1"))
    critic.judge(CriticPair("3", "4", "finding a", "finding b", "C1", "D1"))
    assert client.systems[0] == client.systems[1]
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run pytest tests/critic/test_llm.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'biolit.critic.llm'`

- [ ] **Step 3: Implement**

Mirror `biolit/extract/llm.py`: `client: Any` typed, **no `anthropic` import**, a
`_SYSTEM` constant, a JSON schema pinning `{"label": <one of three>, "rationale": str}`,
`_MAX_TOKENS = 16000` with the same adaptive-thinking rationale, and usage accumulated from
`USAGE_FIELDS`. An unparseable response or a label outside the three raises a typed
`CriticParseError` — never a silent default, which would score as a wrong answer rather than
as a parse failure and hide it from the run log's `parse_failures` count.

- [ ] **Step 4: Run to verify it passes**

Run: `uv run pytest tests/critic/test_llm.py -v`
Expected: PASS

- [ ] **Step 5: Add the out-of-vocabulary label test**

```python
def test_a_label_outside_the_three_raises_rather_than_defaulting():
    """A silent default scores as a wrong answer and vanishes from parse_failures --
    quieter, not louder (ADR-0014)."""
    with pytest.raises(CriticParseError, match="not_a_label"):
        LlmCritic(_RecordingClient(label="not_a_label"), model="m").judge(_PAIR)
```

- [ ] **Step 6: Add the usage-accumulation test**

```python
def test_usage_accumulates_across_calls_over_every_usage_field():
    critic = LlmCritic(_RecordingClient(label="agreement", usage={"input_tokens": 11,
                       "output_tokens": 3, "cache_creation_input_tokens": 0,
                       "cache_read_input_tokens": 0}), model="m")
    for _ in range(7):
        critic.judge(_PAIR)
    assert critic.usage["input_tokens"] == 77
    assert critic.usage["output_tokens"] == 21
```

- [ ] **Step 7: Commit**

```bash
git add src/biolit/critic/llm.py tests/critic/test_llm.py
git commit -F - <<'EOF'
feat(phase-5): the LLM pair-judgment arm, unrun

One prompt serves both input modes -- the arms differ in the TEXT they receive,
not in the code path. Different prompts would confound extraction quality with
prompt wording, and pricing biolit.extract at a real consumer is the whole reason
the comparison exists.

Takes its client as `Any` and does not import anthropic, so biolit gains no hard
dependency (ADR-0015's placement argument).

A label outside the three raises CriticParseError instead of defaulting: a silent
default scores as a wrong answer and disappears from the log's parse_failures
count, which is quieter rather than louder.

No API call is made anywhere in this task; every test uses a fake client.
EOF
```

---

### Task 12: The direction-decomposition arm

**Files:**
- Create: `src/biolit/critic/direction.py`
- Test: `tests/critic/test_direction.py`

**Interfaces:**
- Produces: `DirectionCritic(client, *, model)` implementing `Critic`;
  `compose(a: PaperDirection, b: PaperDirection) -> ContradictionLabel`;
  `PaperDirection` StrEnum (`causes`, `treats`, `neither`).

- [ ] **Step 1: Write the failing test for composition**

```python
def test_opposite_paper_directions_compose_to_contradiction():
    assert compose(PaperDirection.causes, PaperDirection.treats) \
        is ContradictionLabel.contradiction
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run pytest tests/critic/test_direction.py -v`
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Implement**

```python
def compose(a: PaperDirection, b: PaperDirection) -> ContradictionLabel:
    """Pair label from two per-paper labels.

    `neither` on EITHER side yields insufficient_overlap: a paper that takes no position on
    the relationship cannot disagree with one that does. This is a two-sided case and both
    sides are tested (ADR-0014).

    COST NOTE, and the correction matters: on a SAMPLED eval this arm costs ~2x the pair arm,
    because a sampled pair has two mostly-distinct papers -- two calls against one. The
    k(k-1)/2 -> k saving only materialises on real clusters where papers are reused across
    many pairs. What this arm buys on the eval is error localisation to a single paper and a
    deployment-cost projection, NOT a cheaper eval.
    """
    if a is PaperDirection.neither or b is PaperDirection.neither:
        return ContradictionLabel.insufficient_overlap
    return ContradictionLabel.agreement if a is b else ContradictionLabel.contradiction
```

`DirectionCritic.judge` calls the model once per paper, caches by `paper_id` (papers are
disjoint across pairs by construction, so the cache only ever helps within a pair — assert
this rather than assume), and composes.

- [ ] **Step 4: Run to verify it passes**

Run: `uv run pytest tests/critic/test_direction.py -v`
Expected: PASS

- [ ] **Step 5: Add the second operand of the `neither` guard (ADR-0014)**

```python
def test_neither_on_the_SECOND_paper_also_yields_insufficient_overlap():
    assert compose(PaperDirection.causes, PaperDirection.neither) \
        is ContradictionLabel.insufficient_overlap
```

- [ ] **Step 6: Add the two-calls-per-pair test**

```python
def test_one_call_per_paper_is_made_not_one_per_pair():
    client = _RecordingClient(direction="causes")
    DirectionCritic(client, model="m").judge(CriticPair("1", "2", "a", "b", "C1", "D1"))
    assert len(client.systems) == 2
```

- [ ] **Step 7: Commit**

```bash
git add src/biolit/critic/direction.py tests/critic/test_direction.py
git commit -F - <<'EOF'
feat(phase-5): the direction-decomposition arm and its composition rule

Labels each paper independently, then composes; `neither` on either side yields
insufficient_overlap, since a paper taking no position cannot disagree with one
that does. Both sides of that two-sided case are tested.

The cost note is recorded where it will be read: on a SAMPLED eval this arm costs
~2x the pair arm, not less, because a sampled pair has two mostly-distinct papers.
The k(k-1)/2 -> k saving only materialises on real clusters with paper reuse. What
it buys here is error localisation to a single paper and a deployment-cost
projection.
EOF
```

---

### Task 13: Cost accounting, the authorization bound, and the kill-switch

**Files:**
- Create: `src/biolit_evals/critic_cost.py`
- Test: `tests/evals/test_critic_cost.py`

**Interfaces:**
- Produces: `ArmPilot` (`name`, `mean_cost`, `sd_cost`, `n_pilot`, `n_full`);
  `authorization_bound(arms: Sequence[ArmPilot], *, confidence: float = 0.95) -> Bound`
  with `.point_estimate`, `.bound`, `.inflation`; `KillSwitch(limit)` raising `BudgetExceeded`.

- [ ] **Step 1: Write the failing test**

```python
def test_bound_exceeds_the_point_estimate_but_not_by_a_per_call_percentile():
    """The bill is a SUM of thousands of draws, so it concentrates. Assuming every call is a
    95th-percentile call would bound a scenario that cannot occur."""
    arms = [
        ArmPilot("abstract", mean_cost=0.001, sd_cost=0.001, n_pilot=30, n_full=900),
        ArmPilot("findings", mean_cost=0.0006, sd_cost=0.0006, n_pilot=30, n_full=900),
        ArmPilot("direction", mean_cost=0.0005, sd_cost=0.0005, n_pilot=60, n_full=1800),
    ]
    bound = authorization_bound(arms)
    assert bound.bound > bound.point_estimate
    assert 0.05 < bound.inflation < 0.30
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run pytest tests/evals/test_critic_cost.py -v`
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Implement the bound exactly as specified**

```python
def authorization_bound(arms: Sequence[ArmPilot], *, confidence: float = 0.95) -> Bound:
    """One-sided prediction bound on the REALIZED TOTAL, not a per-call percentile times N.

        point  T = sum_a  N_a * m_a
        var    V = sum_a  s_a^2 * ( N_a^2 / n_a  +  N_a )
        bound    = T + t(confidence, min_a n_a - 1) * sqrt(V)

    `N_a^2/n_a` is the uncertainty in the estimated mean; `+N_a` is the realized sum's own
    variation. Arms are combined through the variance rather than pooled because their cost
    distributions differ materially -- two abstracts against one, full text against extracted
    sentences -- and pooling would misstate the spread.

    THIS BOUND COVERS VARIANCE, NOT BIAS, and that distinction is the point. At the planned
    pilot size it lands ~8-16% above the point estimate, so it would NOT have caught Phase
    4's 24%-low estimate -- that was a measurement bug. Widening the interval until it covers
    a bias is the wrong repair: it masks the defect and inflates every future estimate.
    Bias is handled at the source (usage read from the API's own fields, every call counted
    including retries and refusals, prices pinned at run time, post-run reconciliation).
    """
```

Implement `t` via `statistics.NormalDist` corrected, or a small lookup — no new dependency.

- [ ] **Step 4: Run to verify it passes**

Run: `uv run pytest tests/evals/test_critic_cost.py -v`
Expected: PASS

- [ ] **Step 5: Add the zero-variance pin (both ends)**

```python
def test_zero_variance_pilot_gives_a_bound_equal_to_the_point_estimate():
    """No formula needed for this answer: if every call costs the same, the total is exact."""
    arms = [ArmPilot("a", mean_cost=0.002, sd_cost=0.0, n_pilot=30, n_full=900)]
    bound = authorization_bound(arms)
    assert bound.bound == pytest.approx(bound.point_estimate)
    assert bound.point_estimate == pytest.approx(1.8)
```

- [ ] **Step 6: Add the kill-switch test**

```python
def test_kill_switch_aborts_above_1_25x_the_bound():
    switch = KillSwitch(limit=10.0)
    switch.record(9.0)
    with pytest.raises(BudgetExceeded, match="12.5"):
        switch.record(4.0)
```

- [ ] **Step 7: Commit**

```bash
git add src/biolit_evals/critic_cost.py tests/evals/test_critic_cost.py
git commit -F - <<'EOF'
feat(phase-5): the authorization bound, and the bias it deliberately does not cover

A one-sided 95% t prediction bound on the realized total -- NOT a per-call
percentile times N. The bill is a sum of ~3600 draws, so it concentrates;
assuming every call is a 95th-percentile call bounds a scenario that cannot
occur. Arms combine through the variance rather than pooling, because their cost
distributions differ materially.

The bound covers variance, not bias, and that is the load-bearing distinction. At
the planned pilot size it lands ~8-16% above the point estimate, so it would NOT
have caught Phase 4's 24%-low estimate -- that was a measurement bug, not
sampling noise. Widening the interval to cover a bias masks the defect and
inflates every future estimate, so bias is eliminated at the source instead.

The kill-switch aborts above 1.25x the bound so a systematic error cannot run
away across 3600 calls.
EOF
```

---

### Task 14: Blind annotation export and the two gates

**Files:**
- Create: `src/biolit_evals/annotation_export.py`
- Test: `tests/evals/test_annotation_export.py`

**Interfaces:**
- Produces: `export_blind_sheet(pairs, abstracts, *, n_contradiction, n_other, rng) -> list[dict]`;
  `Gate2` (`genuine`, `n`, `verdict`); `evaluate_gate2(genuine: int, n: int) -> Gate2`.

- [ ] **Step 1: Write the failing test — the sheet must not leak the label**

```python
def test_the_blind_sheet_carries_no_gold_label():
    """Blind, per the ADR-0006 domain_sample precedent. A leaked label makes pi
    unmeasurable, and nothing downstream could detect that it had happened."""
    rows = export_blind_sheet(_PAIRS, _ABSTRACTS, n_contradiction=1, n_other=1,
                              rng=random.Random(0))
    for row in rows:
        assert "label" not in row
        assert "direction_a" not in row and "direction_b" not in row
        assert set(row) == {"pair_id", "chemical_id", "disease_id", "abstract_a", "abstract_b"}
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run pytest tests/evals/test_annotation_export.py -v`
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Implement the export**

The first batch is **15 contradiction + 15 other**, deliberately weighted so Gate 2 has
enough contradiction pairs to fire on (an even three-way split yields ~10). Rows are shuffled
with the injected rng so class is not inferable from position.

- [ ] **Step 4: Run to verify it passes**

Run: `uv run pytest tests/evals/test_annotation_export.py -v`
Expected: PASS

- [ ] **Step 5: Add the Gate 2 threshold test, pinned at the boundary**

```python
def test_gate2_stops_at_seven_of_fifteen_and_continues_at_eight():
    """The cut is at 7 because under a true pi of 0.8 that outcome has probability 0.0042,
    and under 0.7 it is 0.0500 -- a strong signal against pi >= 0.7 that fires rarely when
    the proxy is sound. Boundary pinned on both sides."""
    assert evaluate_gate2(genuine=7, n=15).verdict == "STOP"
    assert evaluate_gate2(genuine=8, n=15).verdict == "CONTINUE_FLAGGED"
    assert evaluate_gate2(genuine=11, n=15).verdict == "CONTINUE"
```

- [ ] **Step 6: Add the "not a measurement of π" test**

```python
def test_gate2_reports_a_wide_interval_it_does_not_hide():
    """At 15 pairs the 95% interval is about +/-0.25, so a batch reading 0.6 cannot be
    distinguished from one reading 0.8. Gate 2 is a STOP RULE, not an estimate of pi."""
    gate = evaluate_gate2(genuine=9, n=15)
    lo, hi = gate.interval
    assert (hi - lo) > 0.4
```

- [ ] **Step 7: Commit**

```bash
git add src/biolit_evals/annotation_export.py tests/evals/test_annotation_export.py
git commit -F - <<'EOF'
feat(phase-5): the blind annotation sheet and the Gate 2 stop rule

The sheet carries no gold label and no CTD directions -- blind per the ADR-0006
domain_sample precedent. A leaked label makes pi unmeasurable and nothing
downstream could detect that it had happened, so the test asserts the exact key
set rather than the absence of one field.

The first batch is weighted 15 contradiction + 15 other so Gate 2 has enough
contradiction pairs to fire on; an even three-way split yields about 10.

Gate 2 stops at 7 or fewer genuine of 15. Under a true pi of 0.8 that outcome has
probability 0.0042 and under 0.7 it is 0.0500, so it is a strong signal against pi
>= 0.7 that fires rarely when the proxy is sound. It is deliberately weak against
pi = 0.6 (0.2131): it detects a proxy failing badly, not one merely mediocre. The
interval it reports is ~+/-0.25, and a test pins that width -- this is a stop
rule, explicitly not a measurement of pi.
EOF
```

---

## Gated steps — NOT part of this plan's execution

These require a human decision and are listed so the plan is complete, not so an executor
runs them.

| Step | Gate |
|---|---|
| Build the real manifest (network: CTD + PubMed) | Free; run after Task 5 |
| Free baseline scores on real gold | Free; **may retire the paid run's premise** |
| Annotate the first 30-pair batch | Annotator time; **Gate 1 + Gate 2** |
| Pilot (120 calls) | **Explicit user authorization required** |
| Authorization gate | User reads measured tokens + bound + baseline scores |
| Full run (3,600 calls) | **Explicit user authorization required** |
| Report + ADR-0016 | Free |

---

## Self-Review

**Spec coverage.** §1 → Tasks 1–5 (parsing rules, exclusion, sampling constraints, manifest,
drop rate). §2 → Tasks 2, 14 (label function, blind protocol, Gate 2). §3 → Tasks 6–12
(placement, protocol, arms, baselines, metrics, McNemar, run log, refusal dual-scoring,
per-class zero-finding). §4 → Task 13 (pilot sizing, bound, kill-switch). §5 limitations are
report-time text, carried by the numbers Tasks 5, 10, 13, 14 produce. §6's three open items
map to Tasks 5, 13, 14.

**One deliberate scope exclusion:** the `insufficient_overlap` hard-negative *candidate
generation* (shared endpoint, no curated relation) is described in §1 but has no dedicated
task — it is folded into Task 3's candidate list, since it produces `GoldPair`s of the same
shape and splitting it would not give a reviewer anything separable to reject.

**Type consistency checked:** `GoldPair` field names are identical across Tasks 2–5, 14.
`ContradictionLabel` (not a new enum) is used throughout. `Critic.judge` returns
`ContradictionFinding` in Tasks 6, 7, 11, 12. `USAGE_FIELDS` is reused from
`biolit.extract.llm` rather than redefined.

**Placeholder scan:** none. Every code step carries real code; every test step carries a real
assertion.
