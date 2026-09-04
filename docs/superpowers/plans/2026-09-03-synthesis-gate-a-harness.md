# Synthesis Gate A Harness Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the deterministic template arm and the complete metric suite for Synthesis Gate A, and confirm the metrics run end-to-end against the template's own output on real pipeline clusters — with no LLM call and no spend.

**Architecture:** A production-candidate template (`biolit.synth.template`) renders a cluster as text. A metric module (`biolit_evals.synth_metrics`) scores any output string against the cluster it came from, using only deterministic checks. A corpus module freezes real pipeline clusters and draws the size-stratified sample. A runner CLI wires them together and reports the template arm's baseline. The LLM arm is deliberately absent from this plan.

**Tech Stack:** Python 3.12, uv, pydantic v2, pytest, ruff, pyright, tdd-guard.

**Spec:** `docs/superpowers/specs/2026-09-03-synthesis-gate-a-design.md`

## Global Constraints

- All commands run from `backend/` via `uv run`.
- Ruff ruleset `E,F,I,UP,B`; line length 100.
- **CI runs four steps and all four must pass before any commit:** `uv run ruff check .`, `uv run ruff format --check .`, `uv run pyright` (bare — it covers `tests/` too), `uv run pytest -q`.
- String-valued enums use `enum.StrEnum` (ADR-0005). Use `datetime.now(UTC)`, never `timezone.utc`.
- Imports at the top of the file (E402). The one exception is deliberate function-local heavy imports inside `main()`.
- **`main()` gets no direct unit test** — project convention.
- **Unit tests must never download anything or touch the network.** The corpus freeze is a CLI step, not a test.
- **No per-token prices in code** — a committed rate rots into a wrong cost estimate (see `biolit/extract/llm.py`).
- Any sampling entry point takes a **required `--seed`**, never defaulted.
- Commit messages carry **no `Claude-Session:` trailer, session URL, or agent attribution**.
- Never fabricate PMIDs, MeSH ids, or abstract text.
- **No LLM call, no credential read, no paid API call anywhere in this plan.**

## File Structure

| File | Responsibility |
|---|---|
| `src/biolit/synth/__init__.py` | package marker |
| `src/biolit/synth/template.py` | `render_cluster` — the deterministic control, a **production candidate** (mirrors `biolit/extract/deterministic.py`) |
| `src/biolit_evals/synth_metrics.py` | all five metrics + the aggregate; pure functions over strings and domain objects |
| `src/biolit_evals/synth_corpus.py` | freeze real pipeline clusters, stratify by size, draw the sample |
| `src/biolit_evals/synth_gate_a.py` | `main()` only — runs the template arm over the sample and writes the report |
| `tests/synth/test_template.py` | template tests |
| `tests/evals/test_synth_metrics.py` | metric tests |
| `tests/evals/test_synth_corpus.py` | sampling tests |

---

### Task 1: The deterministic template

**Files:**
- Create: `backend/src/biolit/synth/__init__.py`
- Create: `backend/src/biolit/synth/template.py`
- Create: `backend/tests/synth/__init__.py`
- Test: `backend/tests/synth/test_template.py`

**Interfaces:**
- Consumes: `Cluster`, `ExtractedRecord`, `Paper`, `Finding` from `biolit.domain`.
- Produces: `render_cluster(cluster: Cluster, records: Mapping[str, ExtractedRecord], papers: Mapping[str, Paper]) -> str`

- [ ] **Step 1: Write the failing test**

Create `backend/tests/synth/__init__.py` as an empty file, then `backend/tests/synth/test_template.py`:

```python
from biolit.domain.enums import Source, TextType
from biolit.domain.paper import Paper
from biolit.domain.records import Cluster, ExtractedRecord, Finding
from biolit.synth.template import render_cluster


def _paper(pid: str, year: int | None, journal: str | None) -> Paper:
    return Paper(
        id=pid,
        source=Source.pubmed,
        pmid=pid,
        title=f"Title {pid}",
        text_type=TextType.abstract_only,
        year=year,
        journal=journal,
    )


def _record(pid: str, *sentences: str) -> ExtractedRecord:
    return ExtractedRecord(
        paper_id=pid,
        key_findings=[
            Finding(text=text, start=0, end=len(text), sentence_index=i)
            for i, text in enumerate(sentences)
        ],
    )


def test_render_cluster_lists_papers_oldest_first():
    cluster = Cluster(key="metformin|PCOS", paper_ids=["2", "1"])
    papers = {"1": _paper("1", 2007, "N Engl J Med"), "2": _paper("2", 2019, "Hum Reprod")}
    records = {"1": _record("1", "Clomiphene beat metformin."), "2": _record("2", "OHSS fell.")}

    out = render_cluster(cluster, records, papers)

    assert out.index("PMID 1") < out.index("PMID 2")
    assert "metformin — PCOS" in out
    assert "2 papers, 2007–2019." in out
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/synth/test_template.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'biolit.synth'`

- [ ] **Step 3: Write minimal implementation**

Create `backend/src/biolit/synth/__init__.py` as an empty file, then `backend/src/biolit/synth/template.py`:

```python
"""The deterministic Synthesis control: render a cluster as text, inventing nothing.

This is the arm Gate A asks an LLM to beat (spec of 2026-09-03). It is a PRODUCTION
CANDIDATE, not eval scaffolding -- if Gate A returns "no demonstrated advantage", this
module IS the Synthesis stage, which is why it lives under `biolit.synth` alongside
`biolit.extract.deterministic` rather than under `biolit_evals`.

It makes no agreement or disagreement claim about the papers, because ADR-0018 established
that the pipeline cannot support one.
"""

from collections.abc import Mapping

from biolit.domain.paper import Paper
from biolit.domain.records import Cluster, ExtractedRecord

NO_FINDING_MARKER = "(no finding sentence extracted)"


def _year_range(years: list[int]) -> str:
    if not years:
        return "year unknown"
    lo, hi = min(years), max(years)
    return str(lo) if lo == hi else f"{lo}–{hi}"


def render_cluster(
    cluster: Cluster,
    records: Mapping[str, ExtractedRecord],
    papers: Mapping[str, Paper],
) -> str:
    """Render one cluster: a heading, a count, and every paper with its finding sentences.

    Sorted by year then paper id, so ordering is deterministic and carries no implicit
    ranking -- a reader must not be able to infer importance from position.

    A paper with no extracted findings is printed with `NO_FINDING_MARKER` rather than
    dropped. `zero_findings` is already a tracked ledger key, and silently omitting those
    papers would inflate this arm's coverage against the very metric coverage measures.
    """
    ordered = sorted(
        cluster.paper_ids,
        key=lambda pid: (papers[pid].year if papers[pid].year is not None else 9999, pid),
    )
    years = [papers[pid].year for pid in ordered if papers[pid].year is not None]

    lines = [
        f"## {cluster.key.replace('|', ' — ')}",
        f"{len(ordered)} papers, {_year_range(years)}.",
        "",
    ]
    for pid in ordered:
        paper = papers[pid]
        stamp = " · ".join(
            part
            for part in (
                str(paper.year) if paper.year is not None else None,
                paper.journal,
                f"PMID {paper.pmid or pid}",
            )
            if part
        )
        lines.append(f"- {stamp}")
        findings = records[pid].key_findings if pid in records else []
        if findings:
            lines.extend(f'  "{finding.text}"' for finding in findings)
        else:
            lines.append(f"  {NO_FINDING_MARKER}")
    return "\n".join(lines)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/synth/test_template.py -q`
Expected: PASS

- [ ] **Step 5: Add the zero-findings test**

Append to `backend/tests/synth/test_template.py`:

```python
def test_a_paper_with_no_findings_is_marked_not_dropped():
    """`zero_findings` is a tracked ledger key. Omitting those papers would inflate this
    arm's coverage against the very metric coverage is meant to measure."""
    cluster = Cluster(key="a|b", paper_ids=["1", "2"])
    papers = {"1": _paper("1", 2001, "J One"), "2": _paper("2", 2002, "J Two")}
    records = {"1": _record("1", "A finding.")}

    out = render_cluster(cluster, records, papers)

    assert "PMID 2" in out
    assert "(no finding sentence extracted)" in out
```

- [ ] **Step 6: Run the full file**

Run: `uv run pytest tests/synth/test_template.py -q`
Expected: 2 passed

- [ ] **Step 7: Run all four CI checks**

```bash
uv run ruff check . && uv run ruff format --check . && uv run pyright && uv run pytest -q
```
Expected: all clean. If `ruff format --check` reports a file, run `uv run ruff format .` and re-run all four.

- [ ] **Step 8: Commit**

```bash
git add backend/src/biolit/synth backend/tests/synth
git commit -m "feat(synth): deterministic cluster template, the Gate A control"
```

---

### Task 2: Source view, numerals, and support rate

**Files:**
- Create: `backend/src/biolit_evals/synth_metrics.py`
- Test: `backend/tests/evals/test_synth_metrics.py`

**Interfaces:**
- Consumes: Task 1's `render_cluster` (tests only); `Cluster`, `ExtractedRecord`, `Paper`.
- Produces:
  - `SourceView` frozen dataclass with fields `text: str`, `numerals: frozenset[str]`, `paper_ids: tuple[str, ...]`, `n_papers: int`
  - `build_source_view(cluster, records, papers) -> SourceView`
  - `numerals(text: str) -> list[str]`
  - `Support` frozen dataclass: `supported: int`, `total: int`, `unsupported: tuple[str, ...]`, `rate: float`
  - `support_rate(output: str, source: SourceView) -> Support`

- [ ] **Step 1: Write the failing test**

Create `backend/tests/evals/test_synth_metrics.py`:

```python
from biolit.domain.enums import Source, TextType
from biolit.domain.paper import Paper
from biolit.domain.records import Cluster, ExtractedRecord, Finding
from biolit_evals.synth_metrics import build_source_view, numerals, support_rate


def _paper(pid: str, year: int = 2010) -> Paper:
    return Paper(
        id=pid,
        source=Source.pubmed,
        pmid=pid,
        title=f"Title {pid}",
        text_type=TextType.abstract_only,
        year=year,
        journal="J Test",
    )


def _record(pid: str, *sentences: str) -> ExtractedRecord:
    return ExtractedRecord(
        paper_id=pid,
        key_findings=[
            Finding(text=text, start=0, end=len(text), sentence_index=i)
            for i, text in enumerate(sentences)
        ],
    )


def _fixture(**findings: str):
    cluster = Cluster(key="a|b", paper_ids=sorted(findings))
    papers = {pid: _paper(pid) for pid in findings}
    records = {pid: _record(pid, text) for pid, text in findings.items()}
    return cluster, records, papers


def test_numerals_finds_integers_decimals_and_percentages():
    assert numerals("HbA1c fell 1.5% in 42 of 100 patients") == ["1.5", "42", "100"]


def test_support_rate_counts_a_numeral_absent_from_source_as_unsupported():
    cluster, records, papers = _fixture(p1="HbA1c fell 1.5% over 12 weeks.")
    source = build_source_view(cluster, records, papers)

    got = support_rate("HbA1c fell 1.5% over 24 weeks.", source)

    assert got.unsupported == ("24",)
    assert got.supported == 1
    assert got.total == 2
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/evals/test_synth_metrics.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'biolit_evals.synth_metrics'`

- [ ] **Step 3: Write minimal implementation**

Create `backend/src/biolit_evals/synth_metrics.py`:

```python
"""Deterministic metrics for Synthesis Gate A.

Every function here scores an output STRING against the cluster it was generated from. No
gold standard, no annotation, no network: a claim is checkable because the source text is
right there. That is the property the Critic's task lacked and the reason Gate A can be run
in a day rather than over two annotation rounds (spec of 2026-09-03, §1).

THE SUITE IS DELIBERATELY ASYMMETRIC. Support and coverage are DISQUALIFIERS for the LLM
arm, not scores: the template scores 1.0 on both by construction, so comparing the arms
there would be a score one arm cannot lose -- ADR-0015's tautology from the other side. Only
compression against `dcr` is comparative.
"""

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from biolit.domain.paper import Paper
from biolit.domain.records import Cluster, ExtractedRecord

#: Integers and decimals. Percent signs and units are excluded from the capture so "1.5%"
#: and "1.5" compare equal -- an output that drops the unit has not invented a number.
#:
#: The guard is a LEADING lookbehind, not a pair of word boundaries. What must be excluded
#: is a digit buried inside an identifier -- the "1" in "HbA1c", the "1" in the paper id
#: "p1" -- and those are recognised by what precedes them. A trailing boundary would also
#: reject the unit written flush against the number, which is how doses and durations are
#: normally written: "500mg" would yield nothing and "1.5mg/kg" would yield a phantom "1".
#: The "." in the lookbehind stops a version-like "v1.2.3" from contributing "2.3".
_NUMERAL = re.compile(r"(?<![\w.])\d+(?:\.\d+)?")


def numerals(text: str) -> list[str]:
    """Every numeral in `text`, in order, with duplicates kept."""
    return _NUMERAL.findall(text)


@dataclass(frozen=True)
class SourceView:
    """Everything an output about this cluster may legitimately draw on."""

    text: str
    numerals: frozenset[str]
    paper_ids: tuple[str, ...]
    n_papers: int


def build_source_view(
    cluster: Cluster,
    records: Mapping[str, ExtractedRecord],
    papers: Mapping[str, Paper],
) -> SourceView:
    """Collect the cluster's finding sentences and paper metadata into one checkable view."""
    parts: list[str] = [cluster.key]
    for pid in cluster.paper_ids:
        paper = papers[pid]
        # TITLE IS DELIBERATELY EXCLUDED. Spec §5 gives each arm the year, journal, PMID and
        # finding sentences -- never the title. Counting a title's numerals as "source" would
        # let an arm cite a figure it was never shown and have it scored as supported: a false
        # pass on a disqualifier, which is the score-that-cannot-be-lost shape ADR-0015 exists
        # to catch. If an arm is ever given titles, this list must change in the same commit.
        parts.extend(str(x) for x in (paper.journal, paper.year, paper.pmid) if x)
        if pid in records:
            parts.extend(f.text for f in records[pid].key_findings)
    text = "\n".join(parts)
    return SourceView(
        text=text,
        numerals=frozenset(numerals(text)),
        paper_ids=tuple(cluster.paper_ids),
        n_papers=len(cluster.paper_ids),
    )


@dataclass(frozen=True)
class Support:
    supported: int
    total: int
    unsupported: tuple[str, ...]

    @property
    def rate(self) -> float:
        return 1.0 if self.total == 0 else self.supported / self.total


def support_rate(output: str, source: SourceView) -> Support:
    """Fraction of the output's numerals that the source can account for.

    A numeral is supported if it appears in the source, OR equals the cluster's paper count
    -- the one aggregate an arm may legitimately introduce ("six papers report...").

    ⚠️ THIS METRIC IS STRICT IN A WAY THAT PENALISES EXACTLY WHAT AN LLM IS FOR. A true
    aggregate it cannot verify ("five of six measured HbA1c") counts against the arm.
    Building a test biased against one arm is the same error family as scoring an arm on a
    population it cannot lose on, pointed the other way. The mitigation is that `unsupported`
    is returned ITEMISED, never as a bare rate, so a reader can tell a hallucinated number
    from an unverifiable-but-true one. Do not report `rate` without it.
    """
    found = numerals(output)
    allowed = source.numerals | {str(source.n_papers)}
    unsupported = tuple(n for n in found if n not in allowed)
    return Support(
        supported=len(found) - len(unsupported), total=len(found), unsupported=unsupported
    )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/evals/test_synth_metrics.py -q`
Expected: 2 passed

- [ ] **Step 5: Add the paper-count exemption test**

Append to `backend/tests/evals/test_synth_metrics.py`:

```python
def test_the_cluster_paper_count_is_a_supported_numeral():
    """The one aggregate an arm may legitimately introduce. Without this exemption the
    metric would penalise 'three papers report...' on a three-paper cluster."""
    cluster, records, papers = _fixture(p1="Alpha.", p2="Beta.", p3="Gamma.")
    source = build_source_view(cluster, records, papers)

    got = support_rate("3 papers discuss this pair.", source)

    assert got.unsupported == ()
    assert got.rate == 1.0
```

- [ ] **Step 6: Run test, then verify it is not vacuous**

Run: `uv run pytest tests/evals/test_synth_metrics.py -q` → 3 passed.

Then temporarily change `allowed = source.numerals | {str(source.n_papers)}` to `allowed = source.numerals` and re-run. Expected: `test_the_cluster_paper_count_is_a_supported_numeral` FAILS. Restore the line and re-run to confirm 3 passed. A test that passes with the behaviour removed pins nothing.

- [ ] **Step 7: Run all four CI checks**

```bash
uv run ruff check . && uv run ruff format --check . && uv run pyright && uv run pytest -q
```

- [ ] **Step 8: Commit**

```bash
git add backend/src/biolit_evals/synth_metrics.py backend/tests/evals/test_synth_metrics.py
git commit -m "feat(evals): source view, numeral extraction, and support rate"
```

---

### Task 3: Entity hallucination via MeSH alias scan

**Files:**
- Modify: `backend/src/biolit_evals/synth_metrics.py`
- Test: `backend/tests/evals/test_synth_metrics.py`

**Interfaces:**
- Consumes: `SourceView` from Task 2.
- Produces:
  - `MAX_ALIAS_WORDS = 6`
  - `mesh_concepts(text: str, aliases: Mapping[str, str]) -> set[str]`
  - `hallucinated_concepts(output: str, source: SourceView, aliases: Mapping[str, str]) -> tuple[str, ...]`
  - `load_aliases(path: str) -> dict[str, str]`

**Background the implementer needs:** `data/canon/mesh_aliases.json.gz` is **alias-major** — `{alias: [[MESH:id, canonical_name, is_primary], ...]}` — and holds **551,669 aliases** (measured 2026-09-03). Ids carry a `MESH:` prefix that the rest of the project strips. Alias word counts run 1–36, but **n-grams of up to 6 words cover 98.7%** of them, which is why the scan caps there. The file is gitignored; it is built by the Phase 3A canon build step.

- [ ] **Step 1: Write the failing test**

Append to `backend/tests/evals/test_synth_metrics.py`:

```python
def test_an_entity_absent_from_the_source_is_reported_as_hallucinated():
    """The dangerous failure: inventing a drug or disease that no source paper mentions."""
    cluster, records, papers = _fixture(p1="Metformin lowered glucose.")
    source = build_source_view(cluster, records, papers)
    aliases = {"metformin": "D008687", "rosiglitazone": "D000077154"}

    got = hallucinated_concepts("Metformin and rosiglitazone lowered glucose.", source, aliases)

    assert got == ("D000077154",)


def test_a_multi_word_alias_is_matched():
    cluster, records, papers = _fixture(p1="Nothing relevant here.")
    source = build_source_view(cluster, records, papers)
    aliases = {"polycystic ovary syndrome": "D011085"}

    got = hallucinated_concepts("Patients with polycystic ovary syndrome improved.", source, aliases)

    assert got == ("D011085",)
```

Add `hallucinated_concepts` to the import line at the top of the file.

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/evals/test_synth_metrics.py -q`
Expected: FAIL — `ImportError: cannot import name 'hallucinated_concepts'`

- [ ] **Step 3: Write minimal implementation**

Append to `backend/src/biolit_evals/synth_metrics.py`:

```python
#: Alias word-count cap for the n-gram scan. MeSH aliases run 1-36 words, but 1-6 covers
#: 98.7% of the 551,669 in the artifact (measured 2026-09-03). Scanning to 36 would multiply
#: the lookup cost sixfold to reach chemical names no generated prose will contain. The
#: truncation is a real limitation and is reported rather than hidden.
MAX_ALIAS_WORDS = 6

_WORD_SPLIT = re.compile(r"[^a-z0-9\-]+")


def _alias_words(text: str) -> list[str]:
    """Lowercase `text` and split it into the exact word tokens both sides key on.

    `load_aliases` and `mesh_concepts` MUST both go through this helper, in lockstep, to
    build a lookup key -- that is the whole invariant the alias table depends on. A key
    built any other way (e.g. plain `.lower()`, keeping punctuation) can never be found:
    the scanner's n-gram keys are always plain word tokens rejoined with single spaces, so
    a comma or other punctuation left in the loader's key -- as in the MeSH inverted form
    "Diabetes Mellitus, Type 2", which is the descriptor's own primary alias -- makes that
    concept permanently unreachable from either side. That drift is the defect this helper
    exists to prevent from recurring.
    """
    return [w for w in _WORD_SPLIT.split(text.lower()) if w]


def mesh_concepts(text: str, aliases: Mapping[str, str]) -> set[str]:
    """Every MeSH concept id whose alias appears in `text`.

    Word n-grams are looked up in the alias table rather than scanning 551,669 aliases as
    substrings: the n-gram form is O(len(text)) with a hash lookup per gram, and the
    substring form is O(n_aliases) per call.
    """
    words = _alias_words(text)
    found: set[str] = set()
    for start in range(len(words)):
        for size in range(1, MAX_ALIAS_WORDS + 1):
            if start + size > len(words):
                break
            concept = aliases.get(" ".join(words[start : start + size]))
            if concept is not None:
                found.add(concept)
    return found


def hallucinated_concepts(
    output: str, source: SourceView, aliases: Mapping[str, str]
) -> tuple[str, ...]:
    """MeSH concepts named in the output that no source text mentions, sorted for stability.

    ⚠️ SCOPE, STATED SO IT IS NOT OVERREAD. This catches an invented *MeSH-linkable entity*
    -- a drug or disease that is not there. It does NOT catch an invented study design, an
    invented relationship between two real entities, or a real entity attributed to the
    wrong paper. A zero here is not a clean bill of health; it is the absence of one
    specific, dangerous, mechanically-detectable failure.
    """
    return tuple(sorted(mesh_concepts(output, aliases) - mesh_concepts(source.text, aliases)))


def load_aliases(path: str) -> dict[str, str]:
    """{alias: MeSH id} from the Phase 3A artifact, which is alias-major and prefixes ids.

    Primary aliases win; a non-primary one is kept only if nothing else claims that alias.
    """
    import gzip
    import json

    out: dict[str, str] = {}
    with gzip.open(path, "rt", encoding="utf-8") as fh:
        for alias, entries in json.load(fh).items():
            key = " ".join(_alias_words(alias))
            if not key:
                continue
            for raw_id, _canonical, is_primary in entries:
                if is_primary or key not in out:
                    out[key] = raw_id.split(":", 1)[1] if ":" in raw_id else raw_id
    return out
```

Note: `load_aliases` uses function-local imports for `gzip`/`json` following the precedent in `biolit_evals/annotation_export.py::_concept_names`.

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/evals/test_synth_metrics.py -q`
Expected: 5 passed

- [ ] **Step 5: Run all four CI checks**

```bash
uv run ruff check . && uv run ruff format --check . && uv run pyright && uv run pytest -q
```

- [ ] **Step 6: Commit**

```bash
git add backend/src/biolit_evals/synth_metrics.py backend/tests/evals/test_synth_metrics.py
git commit -m "feat(evals): MeSH-alias entity hallucination detection"
```

---

### Task 4: Coverage

**Files:**
- Modify: `backend/src/biolit_evals/synth_metrics.py`
- Test: `backend/tests/evals/test_synth_metrics.py`

**Interfaces:**
- Produces:
  - `Coverage` frozen dataclass: `covered: int`, `n_papers: int`, `missing: tuple[str, ...]`, with a `rate` property
  - `coverage(output: str, cluster: Cluster, papers: Mapping[str, Paper]) -> Coverage`

- [ ] **Step 1: Write the failing test**

Append to `backend/tests/evals/test_synth_metrics.py`:

```python
def test_coverage_reports_the_papers_the_output_never_references():
    cluster, records, papers = _fixture(p1="Alpha.", p2="Beta.", p3="Gamma.")

    got = coverage("Discussion of PMID p1 and PMID p3 only.", cluster, papers)

    assert got.missing == ("p2",)
    assert got.covered == 2
    assert got.n_papers == 3
```

Add `coverage` to the import line.

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/evals/test_synth_metrics.py -q`
Expected: FAIL — `ImportError: cannot import name 'coverage'`

- [ ] **Step 3: Write minimal implementation**

Append to `backend/src/biolit_evals/synth_metrics.py`:

```python
@dataclass(frozen=True)
class Coverage:
    covered: int
    n_papers: int
    missing: tuple[str, ...]

    @property
    def rate(self) -> float:
        return 1.0 if self.n_papers == 0 else self.covered / self.n_papers


# SUPERSEDED IN FLIGHT. The version below is what shipped, not what this plan first
# specified. The original used raw substring matching (`pid not in output`), which
# credits a paper whose id merely occurs inside another's -- Ruling 8. Review then found
# two further defects in the opposite direction: a punctuated id (DOI, or bioRxiv's title
# fallback) could never be credited at all (Ruling 11), and spec §2's year+journal
# reference was simply not implemented (Ruling 12).
def _contains_phrase(haystack: Sequence[str], phrase: Sequence[str]) -> bool:
    """Whether `phrase` occurs as a CONTIGUOUS run of tokens inside `haystack`.

    An empty phrase is never present -- there is nothing there to have been cited.
    """
    phrase = tuple(phrase)
    if not phrase:
        return False
    haystack = tuple(haystack)
    n = len(phrase)
    return any(haystack[i : i + n] == phrase for i in range(len(haystack) - n + 1))


def coverage(output: str, cluster: Cluster, papers: Mapping[str, Paper]) -> Coverage:
    """Fraction of the cluster's papers the output carries an identifying reference for.

    Spec §2 defines a content unit's paper reference as PMID / year+journal; this checks a
    paper's own id, its PMID, and its year+journal together. A DISQUALIFIER for the LLM arm,
    not a score: the template covers every paper by construction, so its 1.0 here says
    nothing about its quality.

    An id or PMID is checked as a CONTIGUOUS TOKEN PHRASE, via `_alias_words` and
    `_contains_phrase`, not by single-token equality: `Paper.id` is `doi or pmid`, and
    bioRxiv's client sets `id=doi or title` with no `pmid` at all. A DOI or title routinely
    contains characters `_alias_words` treats as separators, so it tokenises into several
    words -- a single-token check could never credit it, even cited verbatim. Phrase matching
    still isn't a substring test: a paper's id or PMID must appear as its OWN run of tokens,
    not merely inside a longer one -- `'1234567' in 'PMID 12345678'` is true as a substring,
    but `['1234567']` is not a contiguous run of `['pmid', '12345678']`, so a cluster
    containing both PMIDs still scores the shorter one missing against an output that only
    ever named the longer one. Coverage is a disqualifier, so a false "covered" is a
    disqualifier that cannot fire -- the exact shape this harness exists to catch.

    A paper is ALSO covered when its year appears as a token and its journal appears as a
    contiguous token phrase -- spec §5 hands each arm both, so "the 2019 N Engl J Med study"
    is an identification the spec allows. This path fires only when the (year, journal) pair
    is UNIQUE within the cluster, computed from the cluster's own papers, never from the
    output: two or more papers sharing a (year, journal) cannot be told apart by it, so a
    single mention such as "the 2019 NEJM papers" would otherwise cover several papers at
    once -- a disqualifier satisfiable in bulk is a disqualifier that cannot fire, the same
    shape the id/PMID phrase match exists to close. Papers that share a (year, journal) stay
    creditable by id/PMID alone; a paper missing either a year or a journal is simply not
    creditable this way.

    ⚠️ RESIDUAL LIMITATION, IN TWO PARTS. An id, PMID or journal name written flush against
    other word characters, with no separating space or punctuation (e.g. "PMID12345678"),
    tokenises as one longer word and is counted missing even when a human reader would call
    it present. And journal matching is literal: an arm that writes a journal's abbreviation
    ("NEJM") where the stored field says "N Engl J Med" is not credited, because expanding
    abbreviations would need a mapping this project does not have and inventing one would put
    an unmeasured heuristic inside a disqualifier. Both err toward strictness against the LLM
    arm, which is why `missing` is returned ITEMISED and must never be reported as a bare rate
    -- the same caveat `support_rate` already carries. If the pilot shows an arm identifying
    papers by abbreviation, that is a reason to revisit this, and the itemised list is what
    will show it.
    """
    output_words = _alias_words(output)

    # Uniqueness is keyed on the NORMALISED journal, because that is what the match below
    # compares. Keying the raw field instead would let "N Engl J Med" and "N. Engl. J. Med."
    # count as two distinct journals that a single mention nonetheless matches -- reopening
    # the bulk-credit hole this guard exists to close, through the same drift Ruling 6 records.
    year_journal_counts: dict[tuple[int, tuple[str, ...]], int] = {}
    for pid in cluster.paper_ids:
        paper = papers[pid]
        if paper.year is not None and paper.journal is not None:
            key = (paper.year, tuple(_alias_words(paper.journal)))
            year_journal_counts[key] = year_journal_counts.get(key, 0) + 1

    def _identified(pid: str) -> bool:
        if _contains_phrase(output_words, _alias_words(pid)):
            return True
        pmid = papers[pid].pmid
        if pmid is not None and _contains_phrase(output_words, _alias_words(pmid)):
            return True
        paper = papers[pid]
        if paper.year is not None and paper.journal is not None:
            journal_words = _alias_words(paper.journal)
            key = (paper.year, tuple(journal_words))
            if year_journal_counts[key] == 1 and _contains_phrase(output_words, [str(paper.year)]):
                return _contains_phrase(output_words, journal_words)
        return False

    missing = tuple(pid for pid in cluster.paper_ids if not _identified(pid))
    return Coverage(
        covered=len(cluster.paper_ids) - len(missing),
        n_papers=len(cluster.paper_ids),
        missing=missing,
    )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/evals/test_synth_metrics.py -q`
Expected: 6 passed

- [ ] **Step 5: Commit**

```bash
uv run ruff check . && uv run ruff format --check . && uv run pyright && uv run pytest -q
git add backend/src/biolit_evals/synth_metrics.py backend/tests/evals/test_synth_metrics.py
git commit -m "feat(evals): cluster coverage metric"
```

---

### Task 5: Distinguishing-content retention

**Files:**
- Modify: `backend/src/biolit_evals/synth_metrics.py`
- Test: `backend/tests/evals/test_synth_metrics.py`

**Interfaces:**
- Produces:
  - `distinguishing_tokens(cluster, records) -> dict[str, frozenset[str]]`
  - `DCR` frozen dataclass: `retained: int`, `scorable: int`, `indistinguishable: tuple[str, ...]`, `lost: tuple[str, ...]`, with a `rate` property
  - `dcr(output, cluster, records) -> DCR`

- [ ] **Step 1: Write the failing test**

Append to `backend/tests/evals/test_synth_metrics.py`:

```python
def test_dcr_counts_a_paper_retained_when_one_distinguishing_token_survives():
    cluster, records, papers = _fixture(
        p1="Metformin reduced hirsutism scores.",
        p2="Metformin reduced ovulation latency.",
    )

    got = dcr("Metformin reduced hirsutism in one report.", cluster, records)

    assert got.retained == 1
    assert got.lost == ("p2",)
    assert got.scorable == 2


def test_a_paper_with_no_distinguishing_tokens_is_excluded_from_the_denominator():
    """Two papers with identical findings cannot be told apart by any output, so scoring
    an arm on them would penalise it for the corpus rather than for its own behaviour."""
    cluster, records, papers = _fixture(p1="Identical finding.", p2="Identical finding.")

    got = dcr("Nothing in particular.", cluster, records)

    assert got.scorable == 0
    assert got.indistinguishable == ("p1", "p2")
    assert got.rate == 1.0
```

Add `dcr` to the import line.

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/evals/test_synth_metrics.py -q`
Expected: FAIL — `ImportError: cannot import name 'dcr'`

- [ ] **Step 3: Write minimal implementation**

Append to `backend/src/biolit_evals/synth_metrics.py`:

```python
#: Tokens shorter than this carry no distinguishing power in biomedical prose -- they are
#: articles, prepositions and units, present in every paper's findings.
_MIN_TOKEN_LEN = 4


def distinguishing_tokens(
    cluster: Cluster, records: Mapping[str, ExtractedRecord]
) -> dict[str, frozenset[str]]:
    """{paper_id: tokens appearing in that paper's findings and NO other paper's in the
    cluster}. These are what an output must preserve to have characterised the group rather
    than merely summarised it into one indistinguishable blur."""
    per_paper: dict[str, frozenset[str]] = {}
    for pid in cluster.paper_ids:
        findings = records[pid].key_findings if pid in records else []
        text = " ".join(f.text for f in findings).lower()
        per_paper[pid] = frozenset(
            w for w in _WORD_SPLIT.split(text) if len(w) >= _MIN_TOKEN_LEN
        )
    return {
        pid: tokens - frozenset().union(*(per_paper[o] for o in per_paper if o != pid))
        if len(per_paper) > 1
        else tokens
        for pid, tokens in per_paper.items()
    }


@dataclass(frozen=True)
class DCR:
    retained: int
    scorable: int
    indistinguishable: tuple[str, ...]
    lost: tuple[str, ...]

    @property
    def rate(self) -> float:
        return 1.0 if self.scorable == 0 else self.retained / self.scorable


def dcr(output: str, cluster: Cluster, records: Mapping[str, ExtractedRecord]) -> DCR:
    """Fraction of scorable papers whose distinguishing content survives into the output.

    THE ONLY COMPARATIVE AXIS, alongside compression. A paper with no distinguishing tokens
    -- identical findings to another in its cluster -- is excluded from the denominator and
    reported separately: no output could tell those apart, so counting them would score an
    arm on a property of the corpus rather than on its own behaviour.
    """
    lowered = output.lower()
    tokens = distinguishing_tokens(cluster, records)
    indistinguishable = tuple(pid for pid, t in tokens.items() if not t)
    scorable = [pid for pid, t in tokens.items() if t]
    lost = tuple(pid for pid in scorable if not any(t in lowered for t in tokens[pid]))
    return DCR(
        retained=len(scorable) - len(lost),
        scorable=len(scorable),
        indistinguishable=indistinguishable,
        lost=lost,
    )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/evals/test_synth_metrics.py -q`
Expected: 8 passed

- [ ] **Step 5: Commit**

```bash
uv run ruff check . && uv run ruff format --check . && uv run pyright && uv run pytest -q
git add backend/src/biolit_evals/synth_metrics.py backend/tests/evals/test_synth_metrics.py
git commit -m "feat(evals): distinguishing-content retention"
```

---

### Task 6: Compression and the aggregate

**Files:**
- Modify: `backend/src/biolit_evals/synth_metrics.py`
- Test: `backend/tests/evals/test_synth_metrics.py`

**Interfaces:**
- Produces:
  - `compression(output: str, source: SourceView) -> float`
  - `OutputScore` frozen dataclass: `support: Support`, `coverage: Coverage`, `dcr: DCR`, `compression: float`, `hallucinated: tuple[str, ...]`, `judgment_terms: tuple[str, ...]`
  - `JUDGMENT_TERMS: frozenset[str]`
  - `judgment_language(output: str) -> tuple[str, ...]`
  - `score_output(output, cluster, records, papers, aliases) -> OutputScore`

- [ ] **Step 1: Write the failing test**

Append to `backend/tests/evals/test_synth_metrics.py`:

```python
def test_judgment_language_is_detected_so_it_can_be_logged():
    """ADR-0018 established the pipeline cannot support an agreement claim. An LLM asked to
    characterise a cluster will volunteer one unprompted, so the rate is logged as a
    compliance diagnostic -- and deliberately never scored (spec §5.1)."""
    assert judgment_language("The findings are conflicting.") == ("conflicting",)
    assert judgment_language("Six papers measured HbA1c.") == ()


def test_score_output_runs_every_metric_over_one_output():
    cluster, records, papers = _fixture(p1="Metformin reduced hirsutism.", p2="Ovulation rose.")
    out = render_cluster(cluster, records, papers)

    score = score_output(out, cluster, records, papers, aliases={})

    assert score.coverage.rate == 1.0
    assert score.support.rate == 1.0
    assert score.compression <= 1.5
    assert score.hallucinated == ()
```

Add `judgment_language`, `score_output` to the metrics import line, and add
`from biolit.synth.template import render_cluster` to the test file's imports.

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/evals/test_synth_metrics.py -q`
Expected: FAIL — `ImportError: cannot import name 'judgment_language'`

- [ ] **Step 3: Write minimal implementation**

Append to `backend/src/biolit_evals/synth_metrics.py`:

```python
def compression(output: str, source: SourceView) -> float:
    """Output length over source length. Below 1.0 means the arm said it shorter."""
    return len(output) / len(source.text) if source.text else 0.0


#: Vocabulary that asserts an epistemic relation between papers. ADR-0018 established the
#: pipeline cannot support such a claim, so its presence is a COMPLIANCE signal.
JUDGMENT_TERMS = frozenset(
    {
        "agree",
        "agreement",
        "conflict",
        "conflicting",
        "consistent",
        "contradict",
        "contradictory",
        "contradiction",
        "disagree",
        "disagreement",
        "discrepant",
        "inconsistent",
        "mixed",
        "opposing",
    }
)


def judgment_language(output: str) -> tuple[str, ...]:
    """Judgment vocabulary present in the output, sorted.

    LOGGED, NEVER SCORED. It measures instruction compliance, not characterisation quality,
    and folding it into a threshold would hand the gate a second comparative axis by the
    back door -- which is how ADR-0017's forbidden metric would return under a new name.
    """
    words = {w for w in _WORD_SPLIT.split(output.lower()) if w}
    return tuple(sorted(words & JUDGMENT_TERMS))


@dataclass(frozen=True)
class OutputScore:
    support: Support
    coverage: Coverage
    dcr: DCR
    compression: float
    hallucinated: tuple[str, ...]
    judgment_terms: tuple[str, ...]


def score_output(
    output: str,
    cluster: Cluster,
    records: Mapping[str, ExtractedRecord],
    papers: Mapping[str, Paper],
    aliases: Mapping[str, str],
) -> OutputScore:
    """Run every Gate A metric over one arm's output for one cluster."""
    source = build_source_view(cluster, records, papers)
    return OutputScore(
        support=support_rate(output, source),
        coverage=coverage(output, cluster, papers),
        dcr=dcr(output, cluster, records),
        compression=compression(output, source),
        hallucinated=hallucinated_concepts(output, source, aliases),
        judgment_terms=judgment_language(output),
    )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/evals/test_synth_metrics.py -q`
Expected: 10 passed

- [ ] **Step 5: Commit**

```bash
uv run ruff check . && uv run ruff format --check . && uv run pyright && uv run pytest -q
git add backend/src/biolit_evals/synth_metrics.py backend/tests/evals/test_synth_metrics.py
git commit -m "feat(evals): compression, judgment-language diagnostic, and the aggregate"
```

---

### Task 7: Corpus freeze and size-stratified sampling

**Files:**
- Create: `backend/src/biolit_evals/synth_corpus.py`
- Test: `backend/tests/evals/test_synth_corpus.py`

**Interfaces:**
- Consumes: `Cluster` and `PipelineState` JSON dumps produced by `python -m biolit.pipeline --json-out`.
- Produces:
  - `SIZE_BANDS: tuple[tuple[str, int, int], ...]` = `(("small", 2, 3), ("medium", 4, 7), ("large", 8, 10_000))`
  - `band_for(size: int) -> str | None`
  - `sample_clusters(clusters: Sequence[Cluster], *, rng: random.Random, per_band: int = 10) -> list[Cluster]`
  - `sample_hash(clusters: Sequence[Cluster]) -> str`

- [ ] **Step 1: Write the failing test**

Create `backend/tests/evals/test_synth_corpus.py`:

```python
import random

import pytest

from biolit.domain.records import Cluster
from biolit_evals.synth_corpus import band_for, sample_clusters


def _clusters(sizes: list[int]) -> list[Cluster]:
    return [
        Cluster(key=f"k{i}", paper_ids=[f"p{i}_{j}" for j in range(size)])
        for i, size in enumerate(sizes)
    ]


def test_band_for_maps_sizes_to_the_three_bands():
    assert band_for(2) == "small"
    assert band_for(5) == "medium"
    assert band_for(28) == "large"
    assert band_for(1) is None


def test_sample_clusters_draws_the_quota_from_every_band():
    """Stratification is required by the measured size skew, not a preference: top-5
    clusters carry 50.3% of comparisons, so an unstratified draw is nearly all small ones."""
    pool = _clusters([2] * 20 + [5] * 20 + [9] * 20)

    got = sample_clusters(pool, rng=random.Random(7), per_band=10)

    assert len(got) == 30
    counts = {b: sum(1 for c in got if band_for(len(c.paper_ids)) == b) for b in
              ("small", "medium", "large")}
    assert counts == {"small": 10, "medium": 10, "large": 10}


def test_sample_clusters_refuses_a_band_it_cannot_fill():
    """A short band would silently change what the gate measures -- the same reason
    `sample_batch` refuses a short stratum in the Alamri harness."""
    pool = _clusters([2] * 20 + [5] * 20 + [9] * 3)

    with pytest.raises(ValueError, match="large"):
        sample_clusters(pool, rng=random.Random(7), per_band=10)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/evals/test_synth_corpus.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'biolit_evals.synth_corpus'`

- [ ] **Step 3: Write minimal implementation**

Create `backend/src/biolit_evals/synth_corpus.py`:

```python
"""Freeze real pipeline clusters and draw the size-stratified Gate A sample.

Sampling is stratified because the size skew is extreme and measured: `top5_pair_share` on
Arm B `same_sentence` is 0.503 -- half of all cluster comparisons come from five clusters --
the largest Arm B cluster holds 11 papers and the largest Arm A cluster 28. An unstratified
draw would be almost entirely 2-paper clusters, and a 2-paper cluster is a different task
from a 12-paper one.
"""

import random
from collections import defaultdict
from collections.abc import Sequence

from biolit.domain.records import Cluster

#: (name, min_papers, max_papers), inclusive. A cluster of 1 is not a cluster.
SIZE_BANDS: tuple[tuple[str, int, int], ...] = (
    ("small", 2, 3),
    ("medium", 4, 7),
    ("large", 8, 10_000),
)


def band_for(size: int) -> str | None:
    """The band a cluster of `size` papers belongs to, or None if it is unusable."""
    for name, low, high in SIZE_BANDS:
        if low <= size <= high:
            return name
    return None


def sample_clusters(
    clusters: Sequence[Cluster], *, rng: random.Random, per_band: int = 10
) -> list[Cluster]:
    """Draw `per_band` clusters from each size band, in band order.

    RAISES on a band it cannot fill. A short band silently changes what the gate measures --
    the same failure the Alamri sampler refuses for the same reason -- and a per-band metric
    computed over four clusters instead of ten would be reported as though it were the
    designed comparison.
    """
    by_band: dict[str, list[Cluster]] = defaultdict(list)
    for cluster in clusters:
        band = band_for(len(cluster.paper_ids))
        if band is not None:
            by_band[band].append(cluster)

    drawn: list[Cluster] = []
    for name, _low, _high in SIZE_BANDS:
        pool = sorted(by_band.get(name, []), key=lambda c: c.key)
        if len(pool) < per_band:
            raise ValueError(
                f"sample_clusters: band {name!r} holds {len(pool)} clusters, need {per_band}. "
                "Widen the query set rather than shrinking the quota -- a short band would "
                "change what the gate measures without saying so."
            )
        rng.shuffle(pool)
        drawn.extend(pool[:per_band])
    return drawn
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/evals/test_synth_corpus.py -q`
Expected: 3 passed

- [ ] **Step 5: Write the failing test for the sample hash**

Spec §6 requires the sample be frozen with a manifest hash, so a later reader can confirm
which clusters the numbers came from. Append to `backend/tests/evals/test_synth_corpus.py`:

```python
def test_sample_hash_tracks_content_not_order():
    a = _clusters([2, 5, 9])
    b = list(reversed(a))

    assert sample_hash(a) == sample_hash(b)
    assert sample_hash(a) != sample_hash(_clusters([2, 5, 8]))
```

Add `sample_hash` to the import line.

- [ ] **Step 6: Run test to verify it fails**

Run: `uv run pytest tests/evals/test_synth_corpus.py -q`
Expected: FAIL — `ImportError: cannot import name 'sample_hash'`

- [ ] **Step 7: Implement the sample hash**

Append to `backend/src/biolit_evals/synth_corpus.py`:

```python
def sample_hash(clusters: Sequence[Cluster]) -> str:
    """Stable over content, not over order or file bytes.

    Sorted before hashing so a re-ordered draw of the same clusters hashes identically: the
    hash answers "which clusters were scored", and draw order is already pinned by the seed.
    Same posture as `contradiction_gold.manifest_hash` -- content, never serialization.
    """
    import hashlib
    import json

    payload = json.dumps(
        sorted([c.key, sorted(c.paper_ids)] for c in clusters), sort_keys=True
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:16]
```

- [ ] **Step 8: Run test to verify it passes**

Run: `uv run pytest tests/evals/test_synth_corpus.py -q`
Expected: 4 passed

- [ ] **Step 9: Commit**

```bash
uv run ruff check . && uv run ruff format --check . && uv run pyright && uv run pytest -q
git add backend/src/biolit_evals/synth_corpus.py backend/tests/evals/test_synth_corpus.py
git commit -m "feat(evals): size-stratified cluster sampling and sample hash for Gate A"
```

---

### Task 8: The Gate A runner, and the end-to-end confirmation

This is the task the whole plan exists to reach: **proving the metrics run against real pipeline output before any money is on the table.**

**Files:**
- Create: `backend/src/biolit_evals/synth_gate_a.py`

**Interfaces:**
- Consumes: everything above. `main()` only — no direct unit test, per project convention.
- Produces: `data/synth/gate_a_template.json` (gitignored — carries abstract-derived text) and a committed run-log line in `evals/synth_runs.jsonl`.

- [ ] **Step 1: Write the runner**

Create `backend/src/biolit_evals/synth_gate_a.py`:

```python
"""Run the Gate A template arm over the frozen cluster sample.

TEMPLATE ARM ONLY. The LLM arm is deliberately absent: the spec's §7 makes the pilot and the
full run separate authorisation steps, and this module must be able to run to completion
with no credential present and no call made.
"""

DEFAULT_STATES = "data/synth/states"
DEFAULT_ALIASES = "data/canon/mesh_aliases.json.gz"
DEFAULT_OUT = "data/synth/gate_a_template.json"
DEFAULT_LOG = "evals/synth_runs.jsonl"


def main(argv: list[str] | None = None) -> None:
    """Score the deterministic template over the stratified sample. No LLM, no credential."""
    import argparse
    import json
    import random
    import statistics
    from collections import Counter
    from datetime import UTC, datetime
    from pathlib import Path

    from biolit.domain.paper import Paper
    from biolit.domain.records import Cluster, ExtractedRecord
    from biolit.synth.template import render_cluster
    from biolit_evals._meta import git_sha
    from biolit_evals.synth_corpus import band_for, sample_clusters, sample_hash
    from biolit_evals.synth_metrics import load_aliases, score_output

    parser = argparse.ArgumentParser(description="Gate A: score the template arm.")
    parser.add_argument("--states", default=DEFAULT_STATES)
    parser.add_argument("--aliases", default=DEFAULT_ALIASES)
    parser.add_argument("--out", default=DEFAULT_OUT)
    parser.add_argument("--log", default=DEFAULT_LOG)
    parser.add_argument("--per-band", type=int, default=10)
    parser.add_argument(
        "--seed",
        type=int,
        required=True,
        help="Required: it fixes which clusters are sampled. Never defaulted.",
    )
    args = parser.parse_args(argv)

    clusters: list[Cluster] = []
    records: dict[str, ExtractedRecord] = {}
    papers: dict[str, Paper] = {}
    for path in sorted(Path(args.states).glob("*.json")):
        state = json.loads(path.read_text(encoding="utf-8"))
        clusters.extend(Cluster(**c) for c in state["clusters"])
        records.update(
            {pid: ExtractedRecord(**r) for pid, r in state["extracted_records"].items()}
        )
        papers.update({p["id"]: Paper(**p) for p in state["candidate_papers"]})
    print(f"{len(clusters)} clusters, {len(papers)} papers from {args.states}")

    sample = sample_clusters(clusters, rng=random.Random(args.seed), per_band=args.per_band)
    aliases = load_aliases(args.aliases)

    rows = []
    for cluster in sample:
        output = render_cluster(cluster, records, papers)
        score = score_output(output, cluster, records, papers, aliases)
        rows.append(
            {
                "key": cluster.key,
                "band": band_for(len(cluster.paper_ids)),
                "n_papers": len(cluster.paper_ids),
                "support_rate": score.support.rate,
                "unsupported": list(score.support.unsupported),
                "coverage_rate": score.coverage.rate,
                "dcr_rate": score.dcr.rate,
                "dcr_scorable": score.dcr.scorable,
                "dcr_indistinguishable": list(score.dcr.indistinguishable),
                "compression": score.compression,
                "hallucinated": list(score.hallucinated),
                "judgment_terms": list(score.judgment_terms),
                "output": output,
            }
        )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")

    def mean(field: str, subset: list[dict]) -> float | None:
        values = [r[field] for r in subset]
        return statistics.fmean(values) if values else None

    per_band = {
        band: {
            "n": len([r for r in rows if r["band"] == band]),
            "support_rate": mean("support_rate", [r for r in rows if r["band"] == band]),
            "coverage_rate": mean("coverage_rate", [r for r in rows if r["band"] == band]),
            "dcr_rate": mean("dcr_rate", [r for r in rows if r["band"] == band]),
            "compression": mean("compression", [r for r in rows if r["band"] == band]),
        }
        for band in ("small", "medium", "large")
    }
    entry = {
        "timestamp": datetime.now(UTC).isoformat(),
        "git_sha": git_sha(),
        "step": "synth_gate_a",
        "arm": "template",
        "seed": args.seed,
        "sample_hash": sample_hash(sample),
        "n_clusters": len(rows),
        "pooled": {
            "support_rate": mean("support_rate", rows),
            "coverage_rate": mean("coverage_rate", rows),
            "dcr_rate": mean("dcr_rate", rows),
            "compression": mean("compression", rows),
        },
        "per_band": per_band,
        "hallucinated_total": sum(len(r["hallucinated"]) for r in rows),
        "judgment_term_clusters": sum(1 for r in rows if r["judgment_terms"]),
        "dcr_indistinguishable_papers": sum(len(r["dcr_indistinguishable"]) for r in rows),
    }
    with Path(args.log).open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(entry) + "\n")

    print(f"wrote {out_path} and logged to {args.log}")
    print(f"pooled: {entry['pooled']}")
    print(f"per band: {json.dumps(per_band, indent=2)}")
    print(f"clusters using judgment language: {entry['judgment_term_clusters']}/{len(rows)}")
    print(f"papers with no distinguishing tokens: {entry['dcr_indistinguishable_papers']}")
    print(f"strata: {dict(sorted(Counter(r['band'] for r in rows).items()))}")
```

- [ ] **Step 2: Verify it imports and the CLI parses**

Run: `uv run python -m biolit_evals.synth_gate_a --help`
Expected: usage text listing `--seed` as required.

- [ ] **Step 3: Freeze real clusters from the live pipeline**

This is the only network step in the plan. It is **free** — PubMed only, no LLM, no credential.

```bash
mkdir -p data/synth/states
for q in "metformin and lactic acidosis" \
         "statins and rhabdomyolysis" \
         "warfarin and bleeding risk" \
         "cisplatin nephrotoxicity" \
         "amiodarone pulmonary toxicity" \
         "lithium and thyroid dysfunction" \
         "NSAIDs and gastrointestinal bleeding" \
         "isotretinoin and depression"; do
  slug=$(echo "$q" | tr ' ' '_')
  uv run python -m biolit.pipeline --query "$q" --max-papers 60 \
    --json-out "data/synth/states/${slug}.json"
done
```

The eight queries are drug–adverse-event pairs chosen to produce co-mention clusters in the shape the extractor emits. **This is a judgment call, not a derived set**, and it is recorded as one: the queries were picked for topical spread, not sampled from any frame. `--max-papers 60` is chosen so the large band (8+ papers) can fill; if it cannot, widen the query set rather than shrinking `--per-band`.

- [ ] **Step 4: Confirm the bands can actually be filled**

```bash
uv run python -c "
import json, glob
from collections import Counter
from biolit_evals.synth_corpus import band_for
sizes = [len(c['paper_ids']) for f in glob.glob('data/synth/states/*.json')
         for c in json.load(open(f, encoding='utf-8'))['clusters']]
print('clusters:', len(sizes))
print('bands:', Counter(band_for(s) for s in sizes))
"
```

Expected: `small`, `medium` and `large` each ≥ 10. **If `large` is short, add queries and re-run Step 3 — do not lower `--per-band`.** A short band changes what the gate measures.

- [ ] **Step 5: Run the template arm end-to-end**

```bash
uv run python -m biolit_evals.synth_gate_a --seed 20260903
```

Expected, and each is a real check rather than a formality:
- `support_rate` **1.0** pooled — the template quotes source sentences verbatim, so anything below 1.0 means the metric is miscounting, not that the template invented a number.
- `coverage_rate` **1.0** pooled — every paper is listed by construction.
- `hallucinated_total` **0** — the template introduces no entity.
- `compression` slightly **above 1.0** — the template adds a heading and per-paper stamps to the source text.
- `judgment_term_clusters` **0** — the template contains no judgment vocabulary.

⚠️ **Any deviation is a metric bug, not an arm result.** The template's values on the three disqualifiers are known in advance precisely because they hold by construction; that is what makes this a usable self-test of the harness. Investigate and fix before proceeding.

Record `dcr_rate` and `dcr_indistinguishable_papers` — the template's DCR is the reference point the LLM arm is later compared against, and the indistinguishable count says how much of the sample no arm could ever score on.

- [ ] **Step 6: Confirm the artifact is gitignored and the log is not**

```bash
git check-ignore -v backend/data/synth/gate_a_template.json
git status --short
```
Expected: the JSON is ignored (it carries abstract-derived sentences); `evals/synth_runs.jsonl` shows as a new tracked file.

- [ ] **Step 7: Run all four CI checks**

```bash
uv run ruff check . && uv run ruff format --check . && uv run pyright && uv run pytest -q
```

- [ ] **Step 8: Commit**

```bash
git add backend/src/biolit_evals/synth_gate_a.py backend/evals/synth_runs.jsonl
git commit -m "feat(evals): Gate A template arm, scored end-to-end on real clusters"
```

---

## Done criterion

The harness is ready when Step 5's five expected values all hold on real pipeline clusters. At that point the metric suite has been exercised against real output end-to-end, the template arm's baseline numbers exist, and **the LLM arm becomes a separate authorisation request** carrying real pilot token counts — per spec §7, and deliberately not part of this plan.
