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
from collections.abc import Mapping
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
