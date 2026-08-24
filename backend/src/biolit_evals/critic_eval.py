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
`extract_eval.py`'s `_diagnostics` already uses. `parse_failures` is read the same way but only
in aggregate (`getattr` once, no per-pair diff): no dual scoring is specified for it, so no
per-pair identity is needed.
"""

import json
from collections import Counter
from collections.abc import Mapping, Sequence
from collections.abc import Set as AbstractSet
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path

from biolit.critic.base import Critic, CriticPair
from biolit.domain.records import ContradictionLabel
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
        finding = critic.judge(critic_pair)
        refused = getattr(critic, "refusals", 0) > refusals_before
        failed = getattr(critic, "parse_failures", 0) > parse_failures_before
        refusals += int(refused)
        parse_failures += int(failed)

        gold.append(pair.label)
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
        "pilot": limit is not None,
    }

    path = Path(log_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(line) + "\n")
    return line


def main(argv: list[str] | None = None) -> None:
    """CLI for the three free arms -- `lexicon`, `majority`, `overlap`. No credential needed.

    `abstract`, `findings` and `direction` are Tasks 11/12's paid LLM arms and are not wired
    here; adding them is a branch each on the if/elif below plus a new `--arm` choice, without
    touching `run_critic_eval` at all.

    Every input is a local file: a manifest (`--manifest`, `read_manifest`'s own format) and an
    abstracts cache (`--abstracts`, a JSON object `{pmid: text}`). Nothing here makes a network
    call, so no import here needs to be heavy -- there is no `httpx`/`anthropic` client to build
    for a free arm. `argparse` is still function-local, matching every other `main()` in this
    package, so importing this module for scoring alone stays cheap.
    """
    import argparse

    from biolit_evals.contradiction_gold import read_manifest
    from biolit_evals.critic_baselines import (
        ConceptOverlapCritic,
        DirectionLexiconCritic,
        MajorityCritic,
    )

    parser = argparse.ArgumentParser(
        description="Score a free Critic baseline against the contradiction-detection manifest."
    )
    parser.add_argument(
        "--arm",
        choices=["lexicon", "majority", "overlap"],
        required=True,
        help="Which free baseline to run.",
    )
    parser.add_argument("--manifest", required=True, help="Path to the gold manifest.")
    parser.add_argument(
        "--abstracts", required=True, help="Path to a JSON {paper_id: abstract text} file."
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
    args = parser.parse_args(argv)

    pairs = read_manifest(args.manifest)
    with open(args.abstracts, encoding="utf-8") as fh:
        abstracts = json.load(fh)

    excluded: frozenset[str] = frozenset()
    if args.excluded_pmids is not None:
        with open(args.excluded_pmids, encoding="utf-8") as fh:
            excluded = frozenset(line.strip() for line in fh if line.strip())

    if args.arm == "majority":
        critic: Critic = MajorityCritic(ContradictionLabel(args.majority_label))
    elif args.arm == "lexicon":
        critic = DirectionLexiconCritic()
    else:
        # `overlap` needs a paper -> concept-id map. No entity-linking pipeline is wired into
        # this free-arm CLI, so this uses the manifest's OWN keyed endpoints as each paper's
        # concept set -- honest about what it is (the pair's own chemical_id/disease_id, not a
        # real linker's output), and enough to exercise the baseline offline.
        concepts_by_paper: dict[str, set[str]] = {}
        for pair in pairs:
            ids = {cid for cid in (pair.chemical_id, pair.disease_id) if cid is not None}
            concepts_by_paper.setdefault(pair.paper_id_a, set()).update(ids)
            concepts_by_paper.setdefault(pair.paper_id_b, set()).update(ids)
        critic = ConceptOverlapCritic(concepts_by_paper)

    line = run_critic_eval(
        pairs=pairs,
        abstracts=abstracts,
        critic=critic,
        arm=args.arm,
        ctd_release=args.ctd_release,
        log_path=DEFAULT_LOG,
        excluded=excluded,
        limit=args.limit,
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
