"""Free baselines for the sentence-selection eval, so no arm is quoted against nothing.

WHY THIS MODULE EXISTS. `recall_on_endpoint_lost` is scored against bucket (a) of
`control-real`'s misses -- a gold subset DEFINED from that control's own failures, so
`control-real` scores exactly 0 on it by construction. A claim was once shipped on that
guaranteed-zero comparison and had to be retracted (ADR-0015). The two baselines that
overturned it were computed ad hoc and never committed, which left the replacement numbers
LESS reproducible than the retracted one. Everything here is deterministic given a seed,
runs offline against the committed run log, and makes NO API call.

Selection shape throughout is `pmid -> set[sentence index]`, the same shape
`extract_eval.sentence_metrics` scores and the same shape the arms produce, so a baseline
and an arm go through ONE scorer rather than two that can drift.
"""

import json
import math
import random
import statistics
from collections.abc import Callable, Iterable, Mapping, Sequence
from collections.abc import Set as AbstractSet
from dataclasses import dataclass

from biolit_evals.extract_eval import sentence_metrics

Selection = dict[str, set[int]]


def n_selected(selection: Mapping[str, AbstractSet[int]]) -> int:
    """Total (pmid, sentence index) pairs in a selection -- the only rate this module quotes.

    Matches `extract_eval._arm_report`'s `n_selected` exactly, which is what makes a baseline's
    count comparable to an arm's. Recall is bought with volume, so no recall figure here is
    quotable without this beside it.
    """
    return sum(len(indices) for indices in selection.values())


def _reject_non_positive_k(k: int, *, caller: str) -> None:
    """Shared so the two positional selectors cannot disagree about what `k` may be.

    ADR-0014 THIRD BRANCH -- kept because removing it makes an eventual failure QUIETER, not
    because the branch is unreachable. `k <= 0` is perfectly reachable (a caller sweeping
    `range(4)` reaches 0 on its first step), and neither selector crashes on it: `first_k`
    hits `min(k, count) <= 0` and `last_k` a start at or past `count`, so BOTH return every
    document mapped to the empty set. That selection scores P=R=F1=0.0000 at n=0 and prints
    as a table row -- a plausible-looking "this heuristic is worthless" result attributed to
    the heuristic instead of to the argument. Raising here is the louder choice.
    """
    if k < 1:
        raise ValueError(
            f"{caller}: k must be >= 1, got {k}. A non-positive k selects nothing from every "
            "document and scores P=R=F1=0 at n=0, which reads as a worthless heuristic rather "
            "than as the bad argument it is."
        )


def first_k(sentence_counts: Mapping[str, int], k: int) -> Selection:
    """The first `k` sentences of every document, clamped to the document's own length.

    CLAMPING IS NOT COSMETIC. Test-500 documents run from 1 to 22 sentences, so `first 4`
    over-runs many of them. An unclamped selector would emit indices the document does not
    have; those are absent from gold, so they land in `fp` (deflating precision) AND inflate
    the selection count -- the quantity every rate-matched comparison in this module is
    built on. A baseline padded to an arm's budget would then be padded to the wrong budget.
    """
    _reject_non_positive_k(k, caller="first_k")
    return {pmid: set(range(min(k, count))) for pmid, count in sentence_counts.items()}


def last_k(sentence_counts: Mapping[str, int], k: int) -> Selection:
    """The last `k` sentences of every document, clamped at index 0.

    Not the mirror image of `first_k` at corpus level: the window starts at a different
    offset in every document. That is why `first 4` and `last 4` select the SAME 1985
    sentences on Test-500 and still score differently -- the comparison is informative
    precisely because the count is held equal.
    """
    _reject_non_positive_k(k, caller="last_k")
    return {pmid: set(range(max(0, count - k), count)) for pmid, count in sentence_counts.items()}


def select_at_rate(sentence_counts: Mapping[str, int], p: float, rng: random.Random) -> Selection:
    """Each sentence selected INDEPENDENTLY with probability `p` -- the rate-matched null.

    `rng` is an explicit `random.Random`, never the module-level `random`: a baseline whose
    value depends on global interpreter state cannot be reproduced from a seed, which is the
    defect this whole module exists to fix.

    `sorted(...)` IS LOAD-BEARING, NOT TIDINESS. The rng is consumed in iteration order, so
    without it the same seed assigns a different draw to every sentence depending on how the
    caller happened to build its dict. Two mappings with identical contents would then yield
    two different "rate-matched nulls" and nothing printed would say why.

    THE RANGE CHECK IS ADR-0014's THIRD BRANCH, KEPT: `rng.random()` returns [0.0, 1.0), so
    `p < 0` is always-False and `p > 1` is always-True. An out-of-range p produces a
    perfectly well-formed selection -- empty, or the whole corpus -- while
    `bernoulli_recall_moments` would report its mean as `p` itself. A recall of 1.1 beside an
    empirical 1.0, with no exception at all, is the quietest failure in this module.
    """
    if not 0.0 <= p <= 1.0:
        raise ValueError(
            f"select_at_rate: p must be in [0, 1], got {p}. `rng.random()` returns [0.0, 1.0), "
            "so an out-of-range p silently saturates to the empty selection or to the whole "
            "corpus instead of failing."
        )
    return {
        pmid: {index for index in range(count) if rng.random() < p}
        for pmid, count in sorted(sentence_counts.items())
    }


def pad_to_budget(
    base: Mapping[str, AbstractSet[int]],
    sentence_counts: Mapping[str, int],
    budget: int,
    rng: random.Random,
) -> Selection:
    """`base` plus uniformly-sampled unselected sentences until the total is EXACTLY `budget`.

    THIS IS WHAT MAKES A POSITIONAL BASELINE COMPARABLE TO AN ARM. `first 4` selects 1985
    sentences on Test-500 and the LLM `low` arm selected 2274, so their recalls are not
    comparable as they stand -- recall is bought with volume. Padding the cheaper selector up
    to the arm's own count removes the volume advantage and leaves only the question worth
    asking: at the same budget, does the arm choose better sentences than a free heuristic?

    `sorted(...)` on the candidate pool is load-bearing for the same reason as in
    `select_at_rate`: `rng.sample` draws by POSITION, so an unsorted pool makes the padding a
    function of the caller's dict insertion order as well as of the seed.
    """
    padded: Selection = {pmid: set(indices) for pmid, indices in base.items()}
    n_base = n_selected(padded)
    if budget < n_base:
        raise ValueError(
            f"pad_to_budget: budget {budget} is below the {n_base} sentences already in "
            "`base`. Padding adds sentences and cannot shrink a selection -- match the "
            "budget to an arm's `n_selected`, or start from a smaller base."
        )
    pool = sorted(
        (pmid, index)
        for pmid, count in sentence_counts.items()
        for index in range(count)
        if index not in padded.get(pmid, ())
    )
    n_corpus = sum(sentence_counts.values())
    if budget > n_corpus:
        raise ValueError(
            f"pad_to_budget: budget {budget} exceeds the {n_corpus} sentences in the corpus, "
            "so no selection can reach it. A budget is an arm's `n_selected`, which is bounded "
            "by the corpus -- suspect a count taken from a different corpus or a `--limit` run."
        )
    for pmid, index in rng.sample(pool, budget - n_base):
        padded.setdefault(pmid, set()).add(index)
    return padded


@dataclass(frozen=True)
class Moments:
    """A closed-form mean and SD. Prints as `mean +/- sd`, never as a bare number.

    Deliberately NOT a float and deliberately without `__float__`, so `float(moments)` is a
    `TypeError` rather than a silently-dropped SD. Every figure in this module that depends
    on a random draw has a spread, and a spread quoted without its width is the shape of
    claim this whole correction pass exists to remove.
    """

    mean: float
    sd: float

    def __str__(self) -> str:
        return f"{self.mean:.4f} +/- {self.sd:.4f}"


def bernoulli_recall_moments(p: float, n_gold: int) -> Moments:
    """Exact recall moments of `select_at_rate` on a gold subset of size `n_gold`.

    NO SIMULATION IS NEEDED FOR THE MEAN, AND SIMULATING IT IS WORSE THAN USELESS. Under
    independent Bernoulli(p) selection every gold sentence is selected with probability p
    regardless of whether it is gold, so the number of recalled sentences is Binomial(n, p)
    and E[recall] = p EXACTLY -- on bucket (a)'s 270 sentences, on the full 1145, on any
    subset at all. The earlier ad-hoc work simulated this and landed about two Monte-Carlo
    standard errors low; the closed form removes that noise entirely.

    The SD is `sqrt(p(1-p)/n_gold)`, which is where `n_gold` DOES enter: on bucket (a) at
    p = 0.4655 it is 0.0304, so an arm's 0.4704 sits 0.16 SD from the null. That is the
    number the retracted headline needed and did not have.

    NO GUARD ON `p` OR `n_gold` HERE, on ADR-0014's three-way triage, and the reasoning is
    recorded rather than assumed. Both bad-argument branches ARE reachable (a caller could
    pass an empty bucket, or a p it computed wrongly), so neither is a delete-as-unreachable
    case. But every one of them already raises before returning: `n_gold == 0` is a
    `ZeroDivisionError` on the division, `n_gold < 0` and `p` outside [0, 1] make
    `p * (1 - p) / n_gold` negative and `math.sqrt` raises `ValueError: math domain error`.
    A guard would improve the MESSAGE without changing the LOUDNESS, which is the third
    branch: flagged, not added. The guard that IS worth its keep sits in `select_at_rate`,
    where an out-of-range p produces a well-formed selection and no exception at all.
    """
    return Moments(mean=p, sd=math.sqrt(p * (1.0 - p) / n_gold))


MIN_SEEDS = 200


@dataclass(frozen=True)
class Distribution:
    """An empirical spread over many seeds. Prints as `mean +/- sd [min, max] (n=...)`.

    MIN AND MAX ARE CARRIED, NOT JUST THE SD, because the question this module actually has
    to answer about the padded baseline is not "how wide is it" but "does the WORST draw
    still beat the arm". A mean and an SD leave that to the reader's normal approximation of
    a statistic that is not normal; the observed extremes answer it directly.

    Like `Moments`, it has no `__float__`, so `float(spread)` raises `TypeError` instead of
    silently discarding the width.
    """

    mean: float
    sd: float
    minimum: float
    maximum: float
    n_draws: int

    def __str__(self) -> str:
        return (
            f"{self.mean:.4f} +/- {self.sd:.4f} "
            f"[{self.minimum:.4f}, {self.maximum:.4f}] (n={self.n_draws})"
        )


def distribution(values: Sequence[float]) -> Distribution:
    """Summarise many draws of one figure.

    REFUSES FEWER THAN `MIN_SEEDS` DRAWS, and that refusal is the module's main defence
    against the defect it was written to fix. The ad-hoc computation this replaces reported
    `first-4 + random pad to 2274` scoring 0.5444 on bucket (a) FROM ONE SEED, with no
    spread -- the same unmeasured-narrative shape as the headline it was correcting. Because
    every stochastic figure here reaches the report through this function, a single draw
    cannot become a quoted number by accident: it is not merely discouraged, it raises.
    """
    if len(values) < MIN_SEEDS:
        raise ValueError(
            f"distribution: {len(values)} draw(s) is fewer than the {MIN_SEEDS} required. Any "
            "figure that depends on a random draw is reported as a distribution, never as a "
            "single draw -- a one-seed number reads as a measurement and is not one."
        )
    return Distribution(
        mean=statistics.fmean(values),
        sd=statistics.stdev(values),
        minimum=min(values),
        maximum=max(values),
        n_draws=len(values),
    )


@dataclass(frozen=True)
class Scores:
    """One selection's row: what it cost, what it scored, and what it scored on bucket (a)."""

    n_selected: int
    rate: float
    precision: float
    recall: float
    f1: float
    recall_on_endpoint_lost: float
    tp_on_endpoint_lost: int
    n_endpoint_lost: int


def score_selection(
    selection: Mapping[str, AbstractSet[int]],
    *,
    gold: Mapping[str, AbstractSet[int]],
    endpoint_lost: Mapping[str, AbstractSet[int]],
    n_sentences: int,
) -> Scores:
    """Score a selection through `extract_eval.sentence_metrics` -- the arms' own scorer.

    NOT a reimplementation. A baseline scored by a second copy of P/R/F1 is not a comparator
    for an arm scored by the first: any divergence between the two would read as a difference
    between the SELECTORS. One scorer, both sides.

    `endpoint_lost` is bucket (a)'s membership, read off the committed run log rather than
    recomputed, for the same reason: it is the gold the arm's `recall_on_endpoint_lost` was
    actually scored against, and a re-derivation is a second thing to keep in agreement.

    PRECISION ON THE RESTRICTED GOLD IS NOT REPORTED, matching `run_extract_eval`. Against a
    gold restricted to one bucket every correct selection outside that bucket counts as a
    false positive, so a precision computed there would be a number about nothing.
    """
    aggregate = sentence_metrics(selection, gold)
    restricted = sentence_metrics(selection, endpoint_lost)
    return Scores(
        n_selected=n_selected(selection),
        rate=n_selected(selection) / n_sentences,
        precision=aggregate.precision,
        recall=aggregate.recall,
        f1=aggregate.f1,
        recall_on_endpoint_lost=restricted.recall,
        tp_on_endpoint_lost=restricted.tp,
        n_endpoint_lost=restricted.tp + restricted.fn,
    )


@dataclass(frozen=True)
class ScoreDistribution:
    """A stochastic selector's row. EVERY field is a `Distribution`, including `n_selected`.

    `n_selected` is a spread and not an int on purpose: `pad_to_budget` fixes it by
    construction, so its distribution is a point mass -- but `select_at_rate` draws it
    Binomial(4885, p), and a rate-matched null quoted at "n = 2274" would be claiming a
    budget it only hits on average. One type for both keeps the report honest about which
    is which without the reader having to know.

    There is no field of type `float` anywhere in this class, which is the structural half
    of the same rule `distribution`'s `MIN_SEEDS` check enforces: a stochastic figure cannot
    be read out of this object as a bare number.
    """

    n_selected: Distribution
    rate: Distribution
    precision: Distribution
    recall: Distribution
    f1: Distribution
    recall_on_endpoint_lost: Distribution


def score_over_seeds(
    make_selection: Callable[[random.Random], Mapping[str, AbstractSet[int]]],
    *,
    gold: Mapping[str, AbstractSet[int]],
    endpoint_lost: Mapping[str, AbstractSet[int]],
    n_sentences: int,
    seeds: Iterable[int],
) -> ScoreDistribution:
    """Score `make_selection` once per seed and report every column as a distribution.

    THE ONLY WAY A STOCHASTIC SELECTOR REACHES THE REPORT. `make_selection` receives a fresh
    `random.Random(seed)` per draw, so the whole row is reproducible from the seed range
    alone, and each column goes through `distribution`, which refuses fewer than `MIN_SEEDS`
    draws. A caller cannot get a one-seed number out of this path at all.
    """
    rows = [
        score_selection(
            make_selection(random.Random(seed)),
            gold=gold,
            endpoint_lost=endpoint_lost,
            n_sentences=n_sentences,
        )
        for seed in seeds
    ]
    return ScoreDistribution(
        n_selected=distribution([float(row.n_selected) for row in rows]),
        rate=distribution([row.rate for row in rows]),
        precision=distribution([row.precision for row in rows]),
        recall=distribution([row.recall for row in rows]),
        f1=distribution([row.f1 for row in rows]),
        recall_on_endpoint_lost=distribution([row.recall_on_endpoint_lost for row in rows]),
    )


def beats_on_every_seed(spread: Distribution, threshold: float) -> bool:
    """Did the WORST observed draw clear `threshold`?

    "The positional baseline beats the LLM arm on bucket (a)" is a claim about every draw,
    not about the average one, and the difference is exactly what a single ad-hoc seed could
    not report: a mean that clears the arm while the lower tail does not is a claim that
    fails on some seeds and no one would know which.

    STRICT `>`, so a tie does not count as beating -- an equal draw is not evidence for the
    baseline. And this is an OBSERVED-EXTREME test, not a distributional one: it says the
    claim held on all `n_draws` seeds, never that it holds with some confidence on unseen
    ones. Report it in those words.
    """
    return spread.minimum > threshold


# Attribute name -> column heading. `Scores` and `ScoreDistribution` share every attribute
# name here, which is what lets one renderer serve both row kinds without a second column
# order to keep in step.
_COLUMNS = (
    ("n_selected", "n"),
    ("rate", "rate"),
    ("precision", "P"),
    ("recall", "R"),
    ("f1", "F1"),
    ("recall_on_endpoint_lost", "R@endpoint_lost"),
)
_WIDTH = 21
HEADER = f"{'selector':<34}" + "".join(f"{head:>{_WIDTH}}" for _, head in _COLUMNS)


def _cell(value: float | int | Distribution) -> str:
    """Render one table cell. A `Distribution` ALWAYS carries its SD into the string.

    This is the last place a stochastic figure could leak as a point estimate. `distribution`
    stops a one-seed number being COMPUTED; this stops a many-seed number being PRINTED as
    though it were exact, so no column of the table can be lifted into prose as a bare value.
    """
    if isinstance(value, Distribution):
        return f"{value.mean:.4f}+/-{value.sd:.4f}"
    return f"{value:.4f}" if isinstance(value, float) else str(value)


def format_row(label: str, scores: Scores | ScoreDistribution) -> str:
    """One table row, deterministic or stochastic, in the `HEADER` column order.

    Takes the union type rather than two functions so the two row kinds cannot fall out of
    column alignment -- a table whose stochastic rows print their columns in a different
    order than its deterministic ones invites exactly the cross-row comparison it looks like
    it is supporting.
    """
    return f"{label:<34}" + "".join(
        f"{_cell(getattr(scores, name)):>{_WIDTH}}" for name, _ in _COLUMNS
    )


@dataclass(frozen=True)
class LlmArm:
    """One logged LLM arm, and the bucket-(a) gold it was scored against.

    `scores` is READ off the log, never recomputed: the arm's selections exist only in the
    process that made the calls, so the log is the sole record of what it scored. That is
    also why nothing in this module can re-run the arm -- and why nothing here needs a
    credential.

    `endpoint_lost` is read for the OPPOSITE reason. It could be recomputed from the corpus,
    but re-deriving it would be a second copy of `classify_misses`' bucketing to keep in
    agreement with the first, and the baselines must be scored against exactly the gold the
    arm was scored against or the comparison is between two different populations.
    """

    timestamp: str
    git_sha: str
    model: str
    effort: str
    scores: Scores
    endpoint_lost: dict[str, set[int]]


def load_llm_arms(log_path: str, *, dataset: str, n_sentences: int) -> list[LlmArm]:
    """Every logged LLM arm for `dataset`, in file order, with bucket (a)'s membership.

    FILTERS ON THE DATASET TAG. `extract_eval.main`'s `--limit N` tags a pilot run
    `bc5cdr_test500_limitN`, so a substring match would score 500-document baselines against
    a 20-document pilot's arm -- the exact corpus-mismatch `assert_dataset_size` exists to
    prevent one level up. Equality only.

    `n_sentences` COMES FROM THE SAME CORPUS LOAD THAT PRODUCES THE BASELINES, and is the
    only figure here not read from the log: the log records `n_documents`, never the sentence
    count. Supplying it from the corpus is what makes the arm's selection RATE and the
    baselines' rates the same quantity.

    CHECKED: every selected line carries the SAME bucket-(a) membership. `control-real` is
    deterministic, so on one corpus at one commit they must agree -- and if they do not,
    "recall on endpoint_lost" names two different gold subsets in one table, with each arm
    silently scored against its own. Raising is the only honest option: there is no correct
    way to pick which membership the baselines should use.
    """
    arms: list[LlmArm] = []
    with open(log_path, encoding="utf-8") as fh:
        for raw in fh:
            raw = raw.strip()
            if not raw:
                continue
            line = json.loads(raw)
            if line["dataset"] != dataset or "llm" not in line["arms"]:
                continue
            logged = line["arms"]["llm"]
            restricted = logged["recall_on_endpoint_lost"]
            aggregate = logged["sentence"]
            arms.append(
                LlmArm(
                    timestamp=line["timestamp"],
                    git_sha=line["git_sha"],
                    model=logged["model"],
                    effort=logged["effort"],
                    scores=Scores(
                        n_selected=logged["n_selected"],
                        rate=logged["n_selected"] / n_sentences,
                        precision=aggregate["precision"],
                        recall=aggregate["recall"],
                        f1=aggregate["f1"],
                        recall_on_endpoint_lost=restricted["recall"],
                        tp_on_endpoint_lost=restricted["tp"],
                        n_endpoint_lost=restricted["n_gold_sentences"],
                    ),
                    endpoint_lost={
                        pmid: set(indices)
                        for pmid, indices in line["arms"]["control-real"]["miss_buckets"][
                            "endpoint_lost_sentences"
                        ].items()
                    },
                )
            )
    if not arms:
        raise ValueError(
            f"load_llm_arms: no line in {log_path!r} has dataset == {dataset!r} AND an `llm` "
            "arm. Control-only runs carry no arm to compare against, and a `--limit N` pilot "
            f"is tagged {dataset}_limitN, which is deliberately NOT this tag."
        )
    for other in arms[1:]:
        if other.endpoint_lost != arms[0].endpoint_lost:
            raise ValueError(
                f"load_llm_arms: run {other.timestamp} carries a different bucket-(a) "
                f"membership from run {arms[0].timestamp}. `control-real` is deterministic, so "
                "on one corpus at one commit these must agree -- and if they do not, "
                "`recall_on_endpoint_lost` names a DIFFERENT gold subset per row, with no way "
                "to choose which one the baselines should be scored against."
            )
    return arms


def main(argv: list[str] | None = None) -> None:
    # Heavy imports are local so importing this module for scoring stays cheap and offline,
    # matching `extract_eval.main` and `cluster_eval.main`. NOTHING BELOW NEEDS A CREDENTIAL:
    # the arms are read from the committed log, never re-run, so this costs nothing but the
    # corpus download.
    import argparse

    from biolit.config import get_settings
    from biolit.ner.windowing import sentence_spans
    from biolit_evals.extract_eval import (
        DEFAULT_LOG,
        assert_gold_sentence_regression_pin,
        gold_finding_sentences,
    )
    from biolit_evals.mesh_gold_download import (
        TEST_MEMBER,
        load_bc5cdr_cid_relations,
        load_bc5cdr_documents,
    )

    parser = argparse.ArgumentParser(
        description=(
            "Free baselines for the sentence-selection eval: positional, rate-matched random, "
            "and budget-padded selectors scored beside the logged LLM arms. Makes NO API call."
        )
    )
    parser.add_argument("--log", default=DEFAULT_LOG, help="Run log to read the LLM arms from.")
    parser.add_argument(
        "--seeds",
        type=int,
        default=MIN_SEEDS,
        help=(
            f"Seeds per stochastic row (default and MINIMUM {MIN_SEEDS}). Fewer is refused by "
            "`distribution`: a figure that depends on a random draw is reported as a "
            "distribution or not at all."
        ),
    )
    args = parser.parse_args(argv)

    settings = get_settings()
    url = settings.bc5cdr_cdr_zip_url
    documents = load_bc5cdr_documents(url, TEST_MEMBER)
    relations = load_bc5cdr_cid_relations(url, TEST_MEMBER)
    # The SAME string the gold and the miss buckets split (`title + " " + abstract`), for the
    # same reason `extract_eval.main` passes it whole: any other segmentation shifts every
    # sentence index between these baselines and the arm they are compared against.
    counts = {document.pmid: len(sentence_spans(document.text)) for document in documents}
    n_sentences = sum(counts.values())
    gold = gold_finding_sentences(documents, relations)
    n_gold = n_selected(gold)
    # The arm's own gold-construction pin, reused rather than re-checked: if gold construction
    # has drifted since the logged run, these baselines are scored against a different gold
    # than the arm was, and every comparison below is between two populations.
    assert_gold_sentence_regression_pin(len(documents), n_gold)

    arms = load_llm_arms(args.log, dataset="bc5cdr_test500", n_sentences=n_sentences)
    endpoint_lost = arms[0].endpoint_lost
    n_lost = n_selected(endpoint_lost)
    # THE BUDGET IS THE FIRST LOGGED FULL-CORPUS ARM'S SELECTION COUNT. A fixed rule with no
    # branch, so the padded row cannot quietly re-target a different run between invocations;
    # the arm it matches is named in the output beside it.
    budget = arms[0].scores.n_selected
    rate = budget / n_sentences
    seeds = range(args.seeds)

    print(
        f"BC5CDR Test-500: {len(documents)} documents, {n_sentences} sentences, {n_gold} gold, "
        f"{n_lost} of them in bucket (a) `endpoint_lost`."
    )
    print(f"Budget for the padded row: {budget} sentences (rate {rate:.4f}), from {arms[0].effort}")
    print(f"  run {arms[0].timestamp} at sha {arms[0].git_sha}. No API call was made.\n")
    print(HEADER)

    for k in (2, 4):
        print(
            format_row(
                f"first {k}",
                score_selection(
                    first_k(counts, k),
                    gold=gold,
                    endpoint_lost=endpoint_lost,
                    n_sentences=n_sentences,
                ),
            )
        )
    print(
        format_row(
            "last 4",
            score_selection(
                last_k(counts, 4), gold=gold, endpoint_lost=endpoint_lost, n_sentences=n_sentences
            ),
        )
    )

    base = first_k(counts, 4)
    padded = score_over_seeds(
        lambda rng: pad_to_budget(base, counts, budget, rng),
        gold=gold,
        endpoint_lost=endpoint_lost,
        n_sentences=n_sentences,
        seeds=seeds,
    )
    print(format_row(f"first 4 + random pad to {budget}", padded))
    random_null = score_over_seeds(
        lambda rng: select_at_rate(counts, rate, rng),
        gold=gold,
        endpoint_lost=endpoint_lost,
        n_sentences=n_sentences,
        seeds=seeds,
    )
    print(format_row(f"rate-matched random p={rate:.4f}", random_null))

    for arm in arms:
        print(format_row(f"LLM {arm.effort} ({arm.timestamp[:16]})", arm.scores))

    print(f"\nStochastic rows are mean +/- SD over {args.seeds} seeds, NOT a single draw.")
    for label, spread in (("first 4 + pad", padded), ("rate-matched random", random_null)):
        observed = spread.recall_on_endpoint_lost
        print(f"  {label:<22} recall on endpoint_lost: {observed}")

    closed = bernoulli_recall_moments(rate, n_lost)
    print(
        f"\nCLOSED FORM for the rate-matched null on bucket (a): {closed}. Under independent "
        f"Bernoulli(p)\n  selection E[recall] = p EXACTLY on any gold subset, so the mean needs "
        "no simulation at all;\n  the empirical row above is a CHECK on the plumbing, not the "
        "source of the number."
    )

    print("\nIS THE POSITIONAL BASELINE'S WIN ON BUCKET (a) ROBUST ACROSS SEEDS?")
    print("  Judged on the WORST observed draw, not the mean -- and over these seeds only.")
    for arm in arms:
        arm_recall = arm.scores.recall_on_endpoint_lost
        spread = padded.recall_on_endpoint_lost
        verdict = "ROBUST" if beats_on_every_seed(spread, arm_recall) else "NOT ROBUST"
        print(
            f"  vs LLM {arm.effort} ({arm.timestamp[:16]}) at {arm_recall:.4f}: {verdict} -- "
            f"worst of {spread.n_draws} draws is {spread.minimum:.4f}"
        )
    print(
        "\nEvery row above is free: the corpus is public and the arms are READ from the run "
        "log.\nNo baseline here is a claim that the arm is worthless -- they are the "
        "comparators that\nmake `recall_on_endpoint_lost` interpretable at all, since "
        "control-real scores 0 on it\nBY CONSTRUCTION (ADR-0015)."
    )


if __name__ == "__main__":
    main()
