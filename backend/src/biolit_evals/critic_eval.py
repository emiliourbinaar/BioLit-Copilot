"""The eval runner, its run log, and the free-arm CLI -- the piece that ties every Critic arm
together.

`run_critic_eval` is ARM-AGNOSTIC: it takes a `Critic` instance (the injected-object pattern
`Extractor`/`Linker` already use elsewhere in this project) and knows nothing about which of
the six arms produced it. The three free baselines are wired here; the two LLM input modes and
the direction-decomposition arm (Tasks 11/12) plug into the SAME runner without touching it.

REFUSAL SIGNALLING, established here because no Critic implementation exists yet to establish
it. `Critic.judge()` always returns a `ContradictionFinding` -- the protocol has no room for
"declined to answer" -- so a Critic that can refuse reports it the same way `LlmExtractor`
reports its own refusals: a stateful counter, read with `getattr(critic, "refusals", 0)`. This
runner reads that counter IMMEDIATELY BEFORE AND AFTER each `judge()` call and treats an
increase as "this pair was refused", which gives per-pair granularity (needed for the dual
macro-F1 split below) without requiring a shared exception type that would make `biolit.critic`
depend on this eval-only module, or vice versa. A critic with no `refusals` attribute at all --
every arm wired in this task -- never refuses, by the same structural-zero convention
`extract_eval.py`'s `_diagnostics` already uses.

`parse_failures` is counted TWO WAYS, aggregate only (no dual scoring is specified for it, so
no `_excluding_parse_failures` macro-F1 exists): via the same counter-diffing convention as
`refusals`, AND by catching any exception `judge()` raises. The second path exists because
Task 11's `LlmCritic` raises a typed `CriticParseError` -- not a counter -- on unparseable
output, precisely so a mid-corpus parse failure costs one pair instead of aborting the whole
run and producing zero log lines. A pair whose `judge()` call raised has no `ContradictionFinding`
to score, so it is treated exactly like a refusal for scoring purposes (excluded from
`macro_f1_excluding_refusals`, forced wrong in the primary/`_refusals_wrong` figures) even
though it is counted under `parse_failures`, not `refusals`.
"""

import json
from collections import Counter
from collections.abc import Mapping, Sequence
from collections.abc import Set as AbstractSet
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path

from biolit.critic.base import Critic, CriticPair
from biolit.domain.records import ContradictionFinding, ContradictionLabel
from biolit.extract.llm import USAGE_FIELDS
from biolit_evals._meta import git_sha
from biolit_evals.contradiction_gold import (
    GoldPair,
    assert_labels_rederive,
    assert_no_bc5cdr_pmids,
    assert_one_pair_per_key,
    assert_papers_disjoint,
    manifest_hash,
)
from biolit_evals.critic_cost import KillSwitch, cost_of
from biolit_evals.critic_scoring import project_to_prevalence, score

DEFAULT_LOG = "evals/critic_runs.jsonl"

# CTD's own contradiction prevalence, pinned in the spec's worked illustration. Every
# `natural_prevalence_precision` in this project's run log is projected at this one figure, so
# runs stay comparable to each other rather than each choosing its own denominator.
_NATURAL_PREVALENCE = 0.0252

# For the "refusals counted as wrong" scoring: a refused pair's actual returned label is
# DISCARDED (see module docstring) rather than trusted, so it cannot accidentally score as
# correct. Each gold label maps to some OTHER label, deterministically -- which one is
# arbitrary, only that it is never the gold label itself.
_ALWAYS_WRONG: dict[ContradictionLabel, ContradictionLabel] = {
    ContradictionLabel.agreement: ContradictionLabel.contradiction,
    ContradictionLabel.contradiction: ContradictionLabel.agreement,
    ContradictionLabel.insufficient_overlap: ContradictionLabel.agreement,
}


def run_critic_eval(
    *,
    pairs: Sequence[GoldPair],
    abstracts: Mapping[str, str],
    critic: Critic,
    arm: str,
    ctd_release: str,
    log_path: str | Path,
    excluded: AbstractSet[str] = frozenset(),
    limit: int | None = None,
    kill_switch: KillSwitch | None = None,
    prices: Mapping[str, float] | None = None,
) -> dict:
    """Score one Critic arm over `pairs` and append one JSON line to `log_path`.

    `--limit` COUNTS PAIRS: `limit` truncates `pairs` to its first N elements BEFORE anything
    else happens, matching `sample_pairs`' own documented guarantee that manifest order IS the
    (random, pre-shuffled) sample order -- so `limit` draws a valid subsample, not a
    key-ordered prefix of a differently-ordered file.

    THE FOUR TASK-4 ANCHORS RUN BEFORE ANY SCORING, over exactly the (possibly limited) pairs
    about to be scored: a corpus that fails one must never produce a log line. All four raise
    `AssertionError` on failure (see `contradiction_gold.py`), so the caller sees exactly the
    same halting behaviour the anchors document on their own.

    `manifest_hash` IS COMPUTED OVER THE FULL `pairs` ARGUMENT AS RECEIVED, not the limited
    subset: it identifies the corpus artifact this run drew from, the same way `git_sha`
    identifies the code and `ctd_release` identifies the upstream data. `pilot` and `n_pairs`
    say how much of that artifact this particular line actually scored.

    `abstracts` builds each `CriticPair`'s text via `.get(pid, "")` -- a paper missing from
    `abstracts` is NOT filtered out (this runner does not call `usable_pairs`/`drop_report`;
    those are the corpus-level report, not the scoring path). It is passed through with empty
    text instead, which is exactly what makes `zero_finding_rate` a measurement of what the
    runner actually received rather than a filter silently applied ahead of it.

    `excluded_pmids_count` IS LOGGED, NOT JUST CHECKED. `excluded` defaults to empty, and every
    caller without a real BC5CDR pmid list (this task's CLI included -- no download is wired
    into it) leaves `assert_no_bc5cdr_pmids` a structural no-op. Without a count in the line, a
    reader of `evals/critic_runs.jsonl` cannot tell an armed check from a silent pass-through,
    and the NER checkpoint being fine-tuned on BC5CDR makes that distinction a correctness
    question, not a cosmetic one.

    `kill_switch`/`prices` ARE THE SPEND SAFEGUARD (spec §4), BOTH OPTIONAL AND BOTH DEFAULT
    `None` so every existing caller and free-arm run is unaffected. When either is `None`,
    nothing is priced or recorded -- the free arms have no cost and must not be burdened by
    it. When both are supplied, the critic's `usage` counter is diffed around each `judge()`
    call (the same convention `refusals` already uses), the INCREMENTAL usage is priced with
    `cost_of`, and `kill_switch.record(...)` is called. That call sits AFTER the try/except
    that guards `judge()`, never inside it: `BudgetExceeded` must propagate out of this
    function and abort the run, not be caught by the broad `except Exception` that counts
    parse failures -- swallowing it would turn the one documented spend safeguard into a
    silently-counted parse failure, which is worse than not having it at all.
    """
    scored = list(pairs) if limit is None else list(pairs)[:limit]

    assert_no_bc5cdr_pmids(scored, excluded)
    assert_papers_disjoint(scored)
    assert_one_pair_per_key(scored)
    assert_labels_rederive(scored)

    gold: list[ContradictionLabel] = []
    pred_refusals_wrong: list[ContradictionLabel] = []
    gold_excluding_refusals: list[ContradictionLabel] = []
    pred_excluding_refusals: list[ContradictionLabel] = []
    refusals = 0
    parse_failures = 0
    zero_total = 0
    zero_empty = 0
    zero_total_by_class: Counter[str] = Counter()
    zero_empty_by_class: Counter[str] = Counter()
    measured_cost = 0.0

    for pair in scored:
        label = str(pair.label)
        text_a = abstracts.get(pair.paper_id_a, "")
        text_b = abstracts.get(pair.paper_id_b, "")
        for text in (text_a, text_b):
            zero_total += 1
            zero_total_by_class[label] += 1
            if not text.strip():
                zero_empty += 1
                zero_empty_by_class[label] += 1

        critic_pair = CriticPair(
            pair.paper_id_a, pair.paper_id_b, text_a, text_b, pair.chemical_id, pair.disease_id
        )
        refusals_before = getattr(critic, "refusals", 0)
        parse_failures_before = getattr(critic, "parse_failures", 0)
        usage_before = dict(getattr(critic, "usage", {}))
        # UNGUARDED CALLER TRUST WOULD BREAK TASK 11. None of the three free arms wired in
        # this task ever raises, but `LlmCritic.judge()` raises a typed `CriticParseError` on
        # unparseable output SPECIFICALLY so this runner can count it -- left unguarded, the
        # first bad LLM response aborts the whole corpus mid-run and produces ZERO log lines,
        # the opposite of what a `parse_failures` counter is for. Catching `Exception` broadly
        # costs nothing today (nothing here raises) and needs no import of a type that does not
        # exist yet on this branch.
        try:
            finding: ContradictionFinding | None = critic.judge(critic_pair)
        except Exception:
            finding = None
        # KILL-SWITCH ACCOUNTING SITS HERE, DELIBERATELY OUTSIDE THE `try/except` ABOVE.
        # `usage` is accumulated by the critic BEFORE it raises or refuses (see `LlmCritic`/
        # `DirectionCritic`), so the incremental usage is measured the same way regardless of
        # how this call ended. `kill_switch.record(...)` is called unguarded so a
        # `BudgetExceeded` it raises propagates straight out of this function -- placing it
        # inside the `except Exception` above (or wrapping it in one of its own) would count a
        # budget overrun as an ordinary parse failure and let the run continue, which is
        # exactly the failure mode this safeguard exists to prevent.
        if kill_switch is not None and prices is not None:
            usage_after = getattr(critic, "usage", {})
            incremental_usage = {
                field: usage_after.get(field, 0) - usage_before.get(field, 0)
                for field in usage_after
            }
            incremental_cost = cost_of(incremental_usage, prices)
            kill_switch.record(incremental_cost)
            measured_cost += incremental_cost
        refused = getattr(critic, "refusals", 0) > refusals_before
        # A raised exception IS a parse failure by definition (see above), on top of whatever
        # `parse_failures` counter-diffing already catches for a critic that flags one without
        # raising.
        failed = finding is None or getattr(critic, "parse_failures", 0) > parse_failures_before
        refusals += int(refused)
        parse_failures += int(failed)

        # A refusal or an unparseable response both mean NO USABLE ANSWER for this pair: neither
        # can be trusted as a real prediction, and `_ALWAYS_WRONG` is what keeps `finding.label`
        # for a refusal from being read at all (see the module docstring). Branched on
        # `finding is None` FIRST -- rather than a combined `refused or finding is None` flag --
        # so every `finding.label` read below is inside a branch where `finding` is narrowed to
        # non-`None`, not merely believed to be by a boolean this branch does not check itself.
        gold.append(pair.label)
        if finding is None:
            pred_refusals_wrong.append(_ALWAYS_WRONG[pair.label])
        else:
            pred_refusals_wrong.append(_ALWAYS_WRONG[pair.label] if refused else finding.label)
            if not refused:
                gold_excluding_refusals.append(pair.label)
                pred_excluding_refusals.append(finding.label)

    scores = score(gold, pred_refusals_wrong)
    scores_excluding_refusals = score(gold_excluding_refusals, pred_excluding_refusals)

    contradiction_metrics = scores.per_class[ContradictionLabel.contradiction]
    natural_prevalence_precision = project_to_prevalence(
        contradiction_metrics.recall, contradiction_metrics.specificity, _NATURAL_PREVALENCE
    )

    zero_finding_rate = zero_empty / zero_total if zero_total else 0.0
    zero_finding_by_class = {
        cls: (zero_empty_by_class[cls] / n if (n := zero_total_by_class[cls]) else 0.0)
        for cls in zero_total_by_class
    }

    line = {
        "timestamp": datetime.now(UTC).isoformat(),
        "git_sha": git_sha(),
        "arm": arm,
        "n_pairs": len(scored),
        "n_per_class": dict(Counter(str(pair.label) for pair in scored)),
        "manifest_hash": manifest_hash(pairs),
        "ctd_release": ctd_release,
        "excluded_pmids_count": len(excluded),
        "macro_f1": scores.macro_f1,
        "macro_f1_excluding_refusals": scores_excluding_refusals.macro_f1,
        "macro_f1_refusals_wrong": scores.macro_f1,
        "per_class": {str(k): asdict(v) for k, v in scores.per_class.items()},
        "confusion": {
            str(gold_label): {str(pred_label): n for pred_label, n in row.items()}
            for gold_label, row in scores.confusion.items()
        },
        "accuracy": scores.accuracy,
        "natural_prevalence_precision": natural_prevalence_precision,
        "refusals": refusals,
        "parse_failures": parse_failures,
        "zero_finding_rate": zero_finding_rate,
        "zero_finding_by_class": zero_finding_by_class,
        "usage": {field: getattr(critic, "usage", {}).get(field, 0) for field in USAGE_FIELDS},
        # 0.0 whenever `kill_switch`/`prices` are not both supplied -- the same structural-zero
        # convention `usage` already uses for a critic with no cost. This is INCREMENTAL cost
        # summed over the run, priced from `usage` diffs the same way `kill_switch.record` was.
        "measured_cost": measured_cost,
        "pilot": limit is not None,
    }

    path = Path(log_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(line) + "\n")
    return line


def main(argv: list[str] | None = None) -> None:
    """CLI for all six arms -- the three free baselines and the three paid LLM arms.

    `abstract` and `findings` both construct `LlmCritic` (`biolit.critic.llm`) -- they differ
    ONLY IN THE TEXT handed to it, never in code, which is the whole point of the two-mode
    comparison (spec §3). `direction` constructs `DirectionCritic` (`biolit.critic.direction`)
    and, like `abstract`, is scored over `--abstracts`: it is a decomposition strategy, not a
    third input mode, and the spec's pricing table (§4) lists it as a single arm rather than an
    abstract/findings pair.

    `findings` has NO FALLBACK to `--abstracts` when `--findings` is not supplied: it fails
    loudly via `parser.error` instead. A silent fallback would convert the findings arm into
    the abstract arm on exactly the papers where extraction failed -- the failure mode the
    spec's "zero-finding policy" section names explicitly and rules out.

    The client injection mirrors `extract_eval.py`'s `main()` exactly: `anthropic` is imported
    here, function-local (the documented E402 exception, matching every other `main()` in this
    package), and `anthropic.Anthropic(max_retries=5)` is handed to the critic. The three free
    arms build no client and need no credential -- this function never reads
    `ANTHROPIC_API_KEY` itself; only the SDK client (once constructed) does that internally.

    `--budget-usd` ARMS THE SPEND GUARD (spec §4) FOR THE THREE PAID ARMS, AND ONLY THOSE.
    Without it, a paid arm launched from this CLI would run with no live cost accounting at
    all -- `run_critic_eval`'s `kill_switch`/`prices` parameters would simply stay `None`, and
    `BudgetExceeded` could never fire. That is the exact unguarded state this flag exists to
    close, so it is REQUIRED (via a loud `parser.error`, not a silent default) for
    `--arm abstract/findings/direction`, and the three free arms must neither require nor
    accept it being meaningful (they are always called with `kill_switch=None, prices=None`).
    The per-token price map comes from `biolit.config.critic_prices_per_token(args.model)`,
    which raises `biolit.config.PriceGuardError` (a model with no hand-verified entry in
    `biolit.config.CRITIC_PRICES_PER_MTOK`, or a price table older than its verified-staleness
    limit) rather than silently substituting a default price -- caught here and converted to
    the same `parser.error(...)` refusal a human sees, exactly as a missing `--model` already
    is.
    """
    import argparse

    import anthropic

    from biolit.config import PriceGuardError, critic_prices_per_token
    from biolit.critic.direction import DirectionCritic
    from biolit.critic.llm import LlmCritic
    from biolit_evals.contradiction_gold import read_manifest
    from biolit_evals.critic_baselines import (
        ConceptOverlapCritic,
        DirectionLexiconCritic,
        MajorityCritic,
    )

    parser = argparse.ArgumentParser(
        description="Score a Critic arm against the contradiction-detection manifest."
    )
    parser.add_argument(
        "--arm",
        choices=["abstract", "findings", "direction", "lexicon", "majority", "overlap"],
        required=True,
        help="Which Critic arm to run.",
    )
    parser.add_argument("--manifest", required=True, help="Path to the gold manifest.")
    parser.add_argument(
        "--abstracts", required=True, help="Path to a JSON {paper_id: abstract text} file."
    )
    parser.add_argument(
        "--findings",
        default=None,
        help=(
            "Path to a JSON {paper_id: extracted finding-sentence text} file, REQUIRED for "
            "--arm findings. There is no fallback to --abstracts: silently substituting the "
            "abstract on a paper where extraction failed would convert the findings arm into "
            "the abstract arm on exactly the papers most likely to flatter it."
        ),
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Model name for --arm abstract/findings/direction. REQUIRED for those three arms.",
    )
    parser.add_argument(
        "--effort",
        choices=["low", "medium", "high", "xhigh", "max"],
        default="low",
        help="Reasoning effort for --arm abstract/findings/direction (default: low).",
    )
    parser.add_argument(
        "--budget-usd",
        type=float,
        default=None,
        help=(
            "Dollar budget limit for --arm abstract/findings/direction, REQUIRED for those "
            "three arms -- arms KillSwitch (biolit_evals.critic_cost), which aborts the run "
            "once cumulative recorded spend exceeds 1.25x this limit. The three free arms do "
            "not require it and ignore it if given."
        ),
    )
    parser.add_argument(
        "--ctd-release", required=True, help="CTD release stamp for the manifest being scored."
    )
    parser.add_argument(
        "--limit", type=int, default=None, help="Score only the first N pairs (tags the run pilot)."
    )
    parser.add_argument(
        "--majority-label",
        choices=[label.value for label in ContradictionLabel],
        default=ContradictionLabel.agreement.value,
        help="Fixed label for --arm majority (default: agreement, CTD's natural majority).",
    )
    parser.add_argument(
        "--excluded-pmids",
        default=None,
        help="Optional path to a text file of BC5CDR pmids (one per line) for the halting anchor.",
    )
    parser.add_argument(
        "--concepts",
        default=None,
        help=(
            "Path to a JSON {paper_id: [concept_id, ...]} file, REQUIRED for --arm overlap. "
            "There is no manifest-derived fallback: a pair's two papers share exactly one "
            "curated key by construction and no paper repeats across pairs, so a set built "
            "from the manifest's own chemical_id/disease_id fields is always IDENTICAL between "
            "the two papers, and ConceptOverlapCritic would answer `agreement` on every pair "
            "-- a degenerate baseline, not a real one."
        ),
    )
    args = parser.parse_args(argv)

    pairs = read_manifest(args.manifest)
    with open(args.abstracts, encoding="utf-8") as fh:
        abstracts = json.load(fh)

    excluded: frozenset[str] = frozenset()
    if args.excluded_pmids is not None:
        with open(args.excluded_pmids, encoding="utf-8") as fh:
            excluded = frozenset(line.strip() for line in fh if line.strip())

    text_source = abstracts
    kill_switch: KillSwitch | None = None
    prices: dict[str, float] | None = None
    if args.arm in ("abstract", "findings", "direction"):
        # THE THREE PAID ARMS. `model` has no sensible default -- an eval log line must name
        # the model that produced it -- so it is required here rather than defaulted, loudly,
        # the same way `--concepts` is required for `--arm overlap` below.
        if args.model is None:
            parser.error(f"--arm {args.arm} requires --model.")
        # THE SPEND GUARD IS ARMED HERE, BEFORE ANY CLIENT IS CONSTRUCTED. A paid arm launched
        # with no `--budget-usd` would run `run_critic_eval` with `kill_switch=None,
        # prices=None` -- the exact unguarded state this flag exists to close -- so it fails
        # loudly via `parser.error` rather than silently defaulting to unguarded.
        if args.budget_usd is None:
            parser.error(
                f"--arm {args.arm} requires --budget-usd. Running a paid arm with no spend "
                "guard armed is the exact unguarded state this flag exists to prevent -- there "
                "is no default budget."
            )
        # THE PRICE TABLE LOOKUP CAN ALSO REFUSE, via a named `PriceGuardError` (a model with
        # no hand-verified entry, or a table older than its staleness limit) rather than a
        # `None` sentinel a future caller could forget to check -- caught here and converted
        # to the same `parser.error(...)` refusal text a human already sees. You cannot run a
        # model whose cost you cannot price, and you cannot trust a price nobody re-verified.
        try:
            prices = critic_prices_per_token(args.model)
        except PriceGuardError as exc:
            parser.error(str(exc))
        kill_switch = KillSwitch(limit=args.budget_usd)
        client = anthropic.Anthropic(max_retries=5)
        if args.arm == "direction":
            # NOT a third input mode: `direction` is a decomposition strategy scored over the
            # same full-abstract text as `--arm abstract` (spec Section 4's pricing table lists
            # it as one arm, not an abstract/findings pair).
            critic: Critic = DirectionCritic(client, model=args.model, effort=args.effort)
        else:
            critic = LlmCritic(client, model=args.model, effort=args.effort)
            if args.arm == "findings":
                # NO FALLBACK TO --abstracts. A silent fallback would convert this arm into
                # the abstract arm on exactly the papers where extraction failed -- the
                # spec's "zero-finding policy" section rules this out explicitly.
                if args.findings is None:
                    parser.error(
                        "--arm findings requires --findings, a JSON {paper_id: extracted "
                        "finding-sentence text} file. There is no fallback to --abstracts: "
                        "silently substituting the abstract on a paper where extraction "
                        "failed would convert the findings arm into the abstract arm on "
                        "exactly the papers most likely to flatter it."
                    )
                with open(args.findings, encoding="utf-8") as fh:
                    text_source = json.load(fh)
    elif args.arm == "majority":
        critic = MajorityCritic(ContradictionLabel(args.majority_label))
    elif args.arm == "lexicon":
        critic = DirectionLexiconCritic()
    else:
        # `overlap` needs a REAL paper -> concept-id map, and there is no manifest-derived
        # fallback for it: every pair's two papers share exactly one curated key by
        # construction (`assert_one_pair_per_key`) and no paper repeats across pairs
        # (`assert_papers_disjoint`), so a set built from `chemical_id`/`disease_id` alone is
        # ALWAYS IDENTICAL between a pair's two papers. `ConceptOverlapCritic` would then answer
        # `agreement` on every single pair -- a baseline that silently sets ADR-0015's "beat the
        # best free baseline" bar to nothing, disguised as a real one in the run log. Failing
        # loudly here is louder, not quieter (ADR-0014's standing preference), and cheaper than
        # someone trusting a `overlap` log line that was never a real comparison.
        if args.concepts is None:
            parser.error(
                "--arm overlap requires --concepts. A manifest-derived concept set is always "
                "identical between a pair's two papers (they share one curated key, and no "
                "paper repeats across pairs), so ConceptOverlapCritic would answer `agreement` "
                "on every pair -- a degenerate baseline, not a real one. Supply a real "
                "paper_id -> concept_ids mapping, e.g. from entity linking over the abstracts."
            )
        with open(args.concepts, encoding="utf-8") as fh:
            concepts_by_paper = {pid: set(ids) for pid, ids in json.load(fh).items()}
        critic = ConceptOverlapCritic(concepts_by_paper)

    line = run_critic_eval(
        pairs=pairs,
        abstracts=text_source,
        critic=critic,
        arm=args.arm,
        ctd_release=args.ctd_release,
        log_path=DEFAULT_LOG,
        excluded=excluded,
        limit=args.limit,
        kill_switch=kill_switch,
        prices=prices,
    )

    print(f"arm={line['arm']} n_pairs={line['n_pairs']} sha={line['git_sha']}")
    if line["pilot"]:
        print("  PILOT RUN -- a limited corpus. Do not quote these numbers as the full run.")
    print(
        f"  macro_f1={line['macro_f1']:.4f} "
        f"(excluding_refusals={line['macro_f1_excluding_refusals']:.4f}, "
        f"refusals_wrong={line['macro_f1_refusals_wrong']:.4f}) accuracy={line['accuracy']:.4f}"
    )
    print(f"  natural_prevalence_precision={line['natural_prevalence_precision']:.4f}")
    print(
        f"  refusals={line['refusals']} parse_failures={line['parse_failures']} "
        f"zero_finding_rate={line['zero_finding_rate']:.4f}"
    )


if __name__ == "__main__":
    main()
