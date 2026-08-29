import zipfile
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

from biolit_evals.contradiction_gold import GoldPair
from biolit_evals.mesh_gold import parse_pubtator_documents
from biolit_evals.mesh_gold_download import (
    DEVELOPMENT_MEMBER,
    TEST_MEMBER,
    TRAINING_MEMBER,
)

# ALL THREE splits. The NER checkpoint was fine-tuned on BC5CDR, so a paper from any split is
# contaminated -- not just the test split, which is what every other consumer of CDR_Data.zip
# in this project defaults to.
BC5CDR_MEMBERS = (TRAINING_MEMBER, DEVELOPMENT_MEMBER, TEST_MEMBER)


def bc5cdr_pmids_from_zip(path: str | Path) -> frozenset[str]:
    """Every pmid in BC5CDR (1,500 across the three splits), for use as the exclusion set.

    A too-small exclusion set does not fail loudly: `assert_no_bc5cdr_pmids` checks the pool
    against whatever set it is handed, so an exclusion covering only the test split leaves
    1,000 contaminated pmids eligible AND leaves the anchor passing. The anchor would be
    agreeing with itself.
    """
    with zipfile.ZipFile(path) as zf:
        texts = [zf.read(member).decode("utf-8") for member in BC5CDR_MEMBERS]
    return frozenset(doc.pmid for text in texts for doc in parse_pubtator_documents(text))


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


def available_abstracts(
    fetched: Mapping[str, tuple[str | None, int | None]],
) -> dict[str, str]:
    """Keep only pmids that actually carry abstract text.

    PubMed answers for a pmid it knows even when that record has no abstract -- older papers,
    editorials, letters -- so `efetch_abstracts` reports (None, year) and the pmid IS a key in
    its result. Passing that result straight to `usable_pairs` would not error: membership is
    True, the pair survives, and the corpus gains a pair the Critic must judge with no text on
    one side. That scores as a model error rather than a missing input, and understates the
    drop rate this step exists to measure by exactly the count of such papers.
    """
    return {pmid: text for pmid, (text, _year) in fetched.items() if text}


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


DEFAULT_MANIFEST = "evals/gold/contradiction_pairs.jsonl"
DEFAULT_LOG = "evals/contradiction_corpus_runs.jsonl"


def main(argv: list[str] | None = None) -> None:
    """Build the seeded candidate pool and write it as the committed manifest.

    `--seed` is REQUIRED, not defaulted. The seed plus the CTD release fixes which corpus
    exists; a default would let a rebuild against a newer CTD produce a different corpus while
    looking like the same reproducible command. Both are written into the run log alongside
    the manifest hash, so a later reader can say exactly which draw produced which numbers.
    """
    # Heavy imports local to main, same pattern as cluster_eval.main.
    import argparse
    import gzip
    import json
    from datetime import UTC, datetime

    from biolit_evals._meta import git_sha
    from biolit_evals.contradiction_gold import build_pool, manifest_hash, write_manifest
    from biolit_evals.ctd_directions import ctd_release_stamp, parse_ctd_directions

    parser = argparse.ArgumentParser(description="Build the Phase 5 contradiction gold pool.")
    parser.add_argument("--ctd", required=True, help="Path to CTD_chemicals_diseases.tsv.gz.")
    parser.add_argument("--bc5cdr-zip", required=True, help="Path to CDR_Data.zip.")
    parser.add_argument(
        "--seed",
        type=int,
        required=True,
        help="Shuffle seed. Required: it fixes which corpus exists and is recorded in the log.",
    )
    parser.add_argument(
        "--per-class",
        type=int,
        default=900,
        help="Pool size per class (3x the 300 target, per the spec's topping-up rule).",
    )
    parser.add_argument("--out", default=DEFAULT_MANIFEST)
    parser.add_argument("--log", default=DEFAULT_LOG)
    args = parser.parse_args(argv)

    # Two passes over the gzip: the stamp scan stops at the first data row, so it is cheap.
    with gzip.open(args.ctd, "rt", encoding="utf-8") as fh:
        release = ctd_release_stamp(fh)
    with gzip.open(args.ctd, "rt", encoding="utf-8") as fh:
        directions = parse_ctd_directions(fh)

    excluded = bc5cdr_pmids_from_zip(args.bc5cdr_zip)
    pool = build_pool(directions, excluded=excluded, per_class=args.per_class, seed=args.seed)
    write_manifest(pool, args.out)

    line = {
        "timestamp": datetime.now(UTC).isoformat(),
        "git_sha": git_sha(),
        "seed": args.seed,
        "per_class": args.per_class,
        "ctd_release": release,
        "manifest": str(args.out),
        "manifest_hash": manifest_hash(pool),
        "pool_by_class": dict(sorted(Counter(str(p.label) for p in pool).items())),
        "n_pmids_in_ctd_direct_evidence": len(directions),
        "excluded_pmids_count": len(excluded),
    }
    Path(args.log).parent.mkdir(parents=True, exist_ok=True)
    with open(args.log, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(line) + "\n")
    print(json.dumps(line, indent=2))


if __name__ == "__main__":
    main()
