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


#: Alias word-count cap for the n-gram scan. MeSH aliases run 1-36 words, but 1-6 covers
#: 98.7% of the 551,669 in the artifact (measured 2026-09-03). Scanning to 36 would multiply
#: the lookup cost sixfold to reach chemical names no generated prose will contain. The
#: truncation is a real limitation and is reported rather than hidden.
MAX_ALIAS_WORDS = 6

_WORD_SPLIT = re.compile(r"[^a-z0-9\-]+")


def mesh_concepts(text: str, aliases: Mapping[str, str]) -> set[str]:
    """Every MeSH concept id whose alias appears in `text`.

    Word n-grams are looked up in the alias table rather than scanning 551,669 aliases as
    substrings: the n-gram form is O(len(text)) with a hash lookup per gram, and the
    substring form is O(n_aliases) per call.
    """
    words = [w for w in _WORD_SPLIT.split(text.lower()) if w]
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
            key = alias.lower()
            for raw_id, _canonical, is_primary in entries:
                if is_primary or key not in out:
                    out[key] = raw_id.split(":", 1)[1] if ":" in raw_id else raw_id
    return out
