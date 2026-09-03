"""The deterministic Synthesis control: render a cluster as text, inventing nothing.

This is the arm Gate A asks an LLM to beat (spec of 2026-09-03). It is a PRODUCTION
CANDIDATE, not eval scaffolding -- if Gate A returns "no demonstrated advantage", this
module IS the Synthesis stage, which is why it lives under `biolit.synth` alongside
`biolit.extract.deterministic` rather than under `biolit_evals`.

It makes no agreement or disagreement claim about the papers, because ADR-0018 established
that the pipeline cannot support one.
"""

from collections.abc import Mapping
from typing import cast

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
    years = cast(list[int], [papers[pid].year for pid in ordered if papers[pid].year is not None])

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
