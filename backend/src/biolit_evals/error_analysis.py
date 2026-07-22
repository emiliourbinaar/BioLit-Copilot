import argparse
from collections import Counter
from dataclasses import dataclass, field

from biolit.domain.enums import EntityLabel
from biolit.domain.records import Entity
from biolit_evals.scoring import score_corpus

# Paths below are relative to the `backend/` directory, which is where this
# project's commands are run from. Mirrors `ner_eval.py`'s DOMAIN_GOLD constant so
# both scripts read the same gold file.
DOMAIN_GOLD = "evals/gold/domain_sample.jsonl"

_TOP_N = 10


@dataclass(frozen=True)
class ErrorReport:
    """Per-example-set error breakdown backing the "Interpreting the gap" section."""

    n_fn: int = 0
    n_fp: int = 0
    fn_overlapping: int = 0
    fp_overlapping: int = 0
    fn_common: list[tuple[tuple[str, EntityLabel], int]] = field(default_factory=list)
    fp_common: list[tuple[tuple[str, EntityLabel], int]] = field(default_factory=list)


def _key(e: Entity) -> tuple[int | None, int | None, str]:
    return (e.start, e.end, e.label)


def _split_errors(gold: list[Entity], pred: list[Entity]) -> tuple[list[Entity], list[Entity]]:
    """Split one example's (gold, pred) into unmatched-gold (FN) and unmatched-pred (FP).

    Uses the same exact `(start, end, label)` matching semantics as
    `biolit_evals.scoring.score_corpus` (Counter/multiset intersection), so the totals
    here agree with the aggregate TP/FP/FN counts the eval runner reports.
    """
    matched = Counter(_key(e) for e in gold) & Counter(_key(e) for e in pred)
    remaining = dict(matched)
    fn: list[Entity] = []
    for e in gold:
        k = _key(e)
        if remaining.get(k, 0) > 0:
            remaining[k] -= 1
        else:
            fn.append(e)
    remaining = dict(matched)
    fp: list[Entity] = []
    for e in pred:
        k = _key(e)
        if remaining.get(k, 0) > 0:
            remaining[k] -= 1
        else:
            fp.append(e)
    return fn, fp


def _overlaps(a: Entity, b: Entity) -> bool:
    """True if `a` and `b` share any character range. None offsets never overlap."""
    if a.start is None or a.end is None or b.start is None or b.end is None:
        return False
    return max(a.start, b.start) < min(a.end, b.end)


def _any_overlap(entity: Entity, others: list[Entity]) -> bool:
    return any(_overlaps(entity, other) for other in others)


def analyze(pairs: list[tuple[list[Entity], list[Entity]]]) -> ErrorReport:
    """Compute the FN/FP boundary-disagreement breakdown across a corpus of examples.

    For each example, FN/FP are the gold/pred spans left over after exact
    `(start, end, label)` matching. "Overlapping" counts how many of those leftover
    spans share a character range with a leftover span on the *other* side of the
    *same example* -- i.e. the model found something in roughly the right place but
    the boundary/label didn't line up exactly, as opposed to missing or
    hallucinating an entity outright.
    """
    all_fn: list[Entity] = []
    all_fp: list[Entity] = []
    fn_overlapping = 0
    fp_overlapping = 0
    for gold, pred in pairs:
        fn, fp = _split_errors(gold, pred)
        for e in fn:
            if _any_overlap(e, fp):
                fn_overlapping += 1
        for e in fp:
            if _any_overlap(e, fn):
                fp_overlapping += 1
        all_fn.extend(fn)
        all_fp.extend(fp)

    fn_common = Counter((e.text, e.label) for e in all_fn).most_common(_TOP_N)
    fp_common = Counter((e.text, e.label) for e in all_fp).most_common(_TOP_N)

    return ErrorReport(
        n_fn=len(all_fn),
        n_fp=len(all_fp),
        fn_overlapping=fn_overlapping,
        fp_overlapping=fp_overlapping,
        fn_common=fn_common,
        fp_common=fp_common,
    )


def _print_report(dataset: str, report: ErrorReport) -> None:
    total = report.n_fn + report.n_fp
    total_overlap = report.fn_overlapping + report.fp_overlapping
    print(f"=== error analysis: {dataset} ===")
    print()
    print(f"{'':17}{'count':>8}  of which overlap a span on the other side")
    fn_pct = f"{100 * report.fn_overlapping / report.n_fn:.0f}%" if report.n_fn else "n/a"
    fp_pct = f"{100 * report.fp_overlapping / report.n_fp:.0f}%" if report.n_fp else "n/a"
    total_pct = f"{100 * total_overlap / total:.0f}%" if total else "n/a"
    print(f"{'False negatives':17}{report.n_fn:>8}  {report.fn_overlapping} ({fn_pct})")
    print(f"{'False positives':17}{report.n_fp:>8}  {report.fp_overlapping} ({fp_pct})")
    print(f"{'Total':17}{total:>8}  {total_overlap} ({total_pct})")
    print()
    print("Most common false negatives (missed gold spans), (surface, label):")
    for (surface, label), count in report.fn_common:
        print(f"  {count:>3}  {surface!r} ({label})")
    print()
    print("Most common false positives (spurious predictions), (surface, label):")
    for (surface, label), count in report.fp_common:
        print(f"  {count:>3}  {surface!r} ({label})")


def main(argv: list[str] | None = None) -> None:
    # Heavy imports are lazy (model load, dataset download) so importing this module
    # for its pure `analyze` logic stays cheap and offline-testable.
    from biolit.config import get_settings
    from biolit.ner.extract import extract_entities
    from biolit.ner.model import NerModel
    from biolit_evals.datasets import load_bc5cdr_test, load_domain_sample

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["bc5cdr", "domain"], required=True)
    args = parser.parse_args(argv)

    settings = get_settings()
    model = NerModel.load(settings)
    if args.dataset == "bc5cdr":
        examples = load_bc5cdr_test()
    else:
        examples = load_domain_sample(DOMAIN_GOLD)

    pairs: list[tuple[list[Entity], list[Entity]]] = []
    for text, gold in examples:
        pred = extract_entities(text, model, score_threshold=settings.ner_score_threshold)
        pairs.append((gold, pred))

    # Cross-check: the aggregate FN/FP this script reports must agree with the
    # scoring harness's own TP/FP/FN, since both are derived from the same exact
    # (start, end, label) matching rule.
    prf = score_corpus(pairs)
    report = analyze(pairs)
    if report.n_fn != prf.fn or report.n_fp != prf.fp:
        raise AssertionError(
            f"error_analysis FN/FP ({report.n_fn}/{report.n_fp}) disagree with "
            f"score_corpus FN/FP ({prf.fn}/{prf.fp}) -- matching logic has diverged."
        )

    _print_report(args.dataset, report)


if __name__ == "__main__":
    main()
