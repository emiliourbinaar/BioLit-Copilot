from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum

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


class ClassOutcome(StrEnum):
    """The three rows of the spec's pre-committed drop rule, one per class.

    The rule is stated in terms of RESULTING N rather than drop rate on purpose: N is what
    threatens the eval, and defining materiality on the rate would leave the judgement call
    exactly where the pre-commitment exists to remove it.
    """

    at_target = "at_target"
    accepted_smaller = "accepted_smaller"
    below_floor = "below_floor"


@dataclass(frozen=True)
class ComposedCorpus:
    """The result of the spec's topping-up rule: the pairs that survived the abstract filter,
    capped per class, in the pool's recorded order -- plus each class's resulting N and the
    rule row that N lands on."""

    pairs: list[GoldPair]
    n_by_class: dict[str, int]
    outcome_by_class: dict[str, ClassOutcome]


def _classify(n: int, *, per_class: int, floor: int) -> ClassOutcome:
    """`n >= floor` is INCLUSIVE: a class landing on exactly the floor is accepted at its
    smaller N. Only below it does topping up become the required response."""
    if n >= per_class:
        return ClassOutcome.at_target
    if n >= floor:
        return ClassOutcome.accepted_smaller
    return ClassOutcome.below_floor


def compose_corpus(
    pool: Sequence[GoldPair],
    abstracts: Mapping[str, str],
    *,
    per_class: int,
    floor: int = 200,
) -> ComposedCorpus:
    """Drop pairs whose abstracts are missing, then take the first `per_class` per class IN
    THE POOL'S RECORDED ORDER.

    Order is the whole point. The pool is drawn seeded and pre-ordered BEFORE any fetching,
    at 3x the target, so that unmeasured abstract availability can be absorbed by consuming
    more of a fixed list. Re-sorting or re-drawing here would turn the sample into a function
    of which abstracts happened to be fetchable -- exactly the selection effect the pre-order
    exists to prevent.

    Filtering delegates to `usable_pairs` rather than repeating its two-sided membership
    guard: `drop_report` duplicates that conjunct inline and consequently needed its own
    separate a-side witness test to stay honest. One copy, one witness.
    """
    # Seeded from the POOL's classes, not the survivors', so a class wiped out entirely by
    # the abstract filter reports n=0/below_floor instead of vanishing from the report. A
    # missing key and a zero read very differently to whoever applies the drop rule.
    taken: Counter[str] = Counter({str(p.label): 0 for p in pool})
    out: list[GoldPair] = []
    for pair in usable_pairs(pool, abstracts):
        label = str(pair.label)
        if taken[label] >= per_class:
            continue
        taken[label] += 1
        out.append(pair)
    n_by_class = dict(taken)
    return ComposedCorpus(
        pairs=out,
        n_by_class=n_by_class,
        outcome_by_class={
            label: _classify(n, per_class=per_class, floor=floor) for label, n in n_by_class.items()
        },
    )


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
