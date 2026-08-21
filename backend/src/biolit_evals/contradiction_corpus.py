from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from biolit_evals.contradiction_gold import GoldPair


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


def usable_pairs(pairs: Sequence[GoldPair], abstracts: Mapping[str, str]) -> list[GoldPair]:
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
