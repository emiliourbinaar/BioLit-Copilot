from collections import Counter
from dataclasses import dataclass, field

import pytest

from biolit.critic.base import CriticPair
from biolit.domain.records import ContradictionFinding, ContradictionLabel
from biolit.extract.llm import USAGE_FIELDS
from biolit_evals.contradiction_gold import GoldPair, manifest_hash
from biolit_evals.critic_baselines import MajorityCritic
from biolit_evals.critic_cost import BudgetExceeded, KillSwitch
from biolit_evals.critic_eval import main, run_critic_eval

# Three disjoint pairs, one per gold class, satisfying every Task 4 anchor:
#   - papers disjoint (six distinct paper ids)
#   - one pair per (chemical_id, disease_id) key (three distinct keys)
#   - labels rederive from directions (agreement: same direction; contradiction: opposite;
#     insufficient_overlap: no directions, exactly one endpoint set)
#   - none of these ids are BC5CDR pmids (the anchor is checked against an empty default set)
_PAIRS = [
    GoldPair("1", "2", "C1", "D1", ContradictionLabel.agreement, "therapeutic", "therapeutic"),
    GoldPair(
        "3", "4", "C2", "D2", ContradictionLabel.contradiction, "therapeutic", "marker/mechanism"
    ),
    GoldPair("5", "6", "C3", None, ContradictionLabel.insufficient_overlap, None, None),
]

_ABSTRACTS = {
    "1": "The drug showed therapeutic efficacy in the treatment of the disease.",
    "2": "Therapy improved outcomes for patients with the disease.",
    "3": "The compound showed therapeutic efficacy in treatment of the disease.",
    "4": "Hepatotoxicity was induced by the compound; adverse events were reported.",
    "5": "An unrelated abstract about a different chemical entirely.",
    "6": "",  # deliberately empty -- exercises zero_finding_rate
}


@dataclass
class _RefusingCritic:
    """Refuses on a configurable set of paper ids, answers correctly everywhere else -- so the
    dual-refusal test can tell `macro_f1_excluding_refusals` and `macro_f1_refusals_wrong`
    apart. `refusals` is a plain stateful counter, mirroring `LlmExtractor`'s own convention:
    see `critic_eval`'s module docstring for why `run_critic_eval` reads it by diffing around
    each `judge()` call rather than via a dedicated exception type.
    """

    refuse_on: set[str]
    answers: dict[tuple[str, str], ContradictionLabel] = field(default_factory=dict)
    refusals: int = 0

    def judge(self, pair: CriticPair) -> ContradictionFinding:
        if pair.paper_id_a in self.refuse_on or pair.paper_id_b in self.refuse_on:
            self.refusals += 1
            return ContradictionFinding(
                paper_id_a=pair.paper_id_a,
                paper_id_b=pair.paper_id_b,
                label=ContradictionLabel.agreement,
                rationale="refused",
            )
        label = self.answers[(pair.paper_id_a, pair.paper_id_b)]
        return ContradictionFinding(
            paper_id_a=pair.paper_id_a, paper_id_b=pair.paper_id_b, label=label, rationale="ok"
        )


@dataclass
class _ParseFailingCritic:
    """Increments `parse_failures` on a configured set of paper ids, mirroring the same
    counter-diffing convention `refusals` uses -- so the two aggregate diagnostics can be
    proven independent rather than one silently reporting the other's count."""

    fail_on: set[str]
    parse_failures: int = 0

    def judge(self, pair: CriticPair) -> ContradictionFinding:
        if pair.paper_id_a in self.fail_on or pair.paper_id_b in self.fail_on:
            self.parse_failures += 1
        return ContradictionFinding(
            paper_id_a=pair.paper_id_a,
            paper_id_b=pair.paper_id_b,
            label=ContradictionLabel.agreement,
            rationale="x",
        )


@dataclass
class _RaisingCritic:
    """Raises on a configured set of paper ids, mirroring the shape Task 11's `LlmCritic` uses
    for `CriticParseError` -- proving the runner survives an unguarded `judge()` call instead of
    aborting the whole corpus and producing zero log lines."""

    raise_on: set[str]

    def judge(self, pair: CriticPair) -> ContradictionFinding:
        if pair.paper_id_a in self.raise_on or pair.paper_id_b in self.raise_on:
            raise ValueError("simulated parse failure")
        return ContradictionFinding(
            paper_id_a=pair.paper_id_a,
            paper_id_b=pair.paper_id_b,
            label=ContradictionLabel.agreement,
            rationale="ok",
        )


def test_run_appends_one_line_carrying_the_manifest_hash_and_ctd_stamp(tmp_path):
    """A run log line that cannot be tied to the corpus that produced it is not
    reproducible -- CTD is a living database."""
    log = tmp_path / "critic_runs.jsonl"
    line = run_critic_eval(
        pairs=_PAIRS,
        abstracts=_ABSTRACTS,
        critic=MajorityCritic(ContradictionLabel.agreement),
        arm="majority",
        ctd_release="Thu Jul 30 13:59:07 EDT 2026",
        log_path=log,
    )
    assert line["manifest_hash"] == manifest_hash(_PAIRS)
    assert line["ctd_release"] == "Thu Jul 30 13:59:07 EDT 2026"
    assert len(log.read_text(encoding="utf-8").strip().splitlines()) == 1


def test_both_refusal_scorings_are_always_reported(tmp_path):
    """Reporting only the excluding-refusals figure inflates the arm; reporting only the
    counting-as-wrong figure conflates capability with policy."""
    answers = {(p.paper_id_a, p.paper_id_b): p.label for p in _PAIRS}
    critic = _RefusingCritic(refuse_on={"1"}, answers=answers)
    line = run_critic_eval(
        pairs=_PAIRS,
        abstracts=_ABSTRACTS,
        critic=critic,
        arm="lexicon",
        ctd_release="Thu Jul 30 13:59:07 EDT 2026",
        log_path=tmp_path / "critic_runs.jsonl",
    )
    assert line["macro_f1_excluding_refusals"] > line["macro_f1_refusals_wrong"]
    assert line["refusals"] == 1


def test_a_limited_run_is_tagged_pilot_and_records_its_size(tmp_path):
    """assert_dataset_size's precedent (ADR-0013): a pilot must never be mistakable for a
    full run in the log."""
    line = run_critic_eval(
        pairs=_PAIRS,
        abstracts=_ABSTRACTS,
        critic=MajorityCritic(ContradictionLabel.agreement),
        arm="majority",
        ctd_release="Thu Jul 30 13:59:07 EDT 2026",
        log_path=tmp_path / "critic_runs.jsonl",
        limit=2,
    )
    assert line["pilot"] is True
    assert line["n_pairs"] == 2


def test_a_failing_anchor_halts_before_any_log_line_is_written(tmp_path):
    """The four Task 4 anchors run before scoring: a corpus that fails one must never produce
    a log line. Two pairs sharing paper '1' fails `assert_papers_disjoint`."""
    broken = [
        GoldPair("1", "2", "C1", "D1", ContradictionLabel.agreement, "therapeutic", "therapeutic"),
        GoldPair("1", "3", "C9", "D9", ContradictionLabel.agreement, "therapeutic", "therapeutic"),
    ]
    log = tmp_path / "critic_runs.jsonl"
    with pytest.raises(AssertionError, match="assert_papers_disjoint"):
        run_critic_eval(
            pairs=broken,
            abstracts=_ABSTRACTS,
            critic=MajorityCritic(ContradictionLabel.agreement),
            arm="majority",
            ctd_release="x",
            log_path=log,
        )
    assert not log.exists()


def test_the_bc5cdr_anchor_halts_on_a_supplied_excluded_pmid(tmp_path):
    """`excluded` defaults to empty (the other tests never trip this anchor); supplying a real
    BC5CDR pmid must still halt the run before a log line is written."""
    log = tmp_path / "critic_runs.jsonl"
    with pytest.raises(AssertionError, match="BC5CDR"):
        run_critic_eval(
            pairs=_PAIRS,
            abstracts=_ABSTRACTS,
            critic=MajorityCritic(ContradictionLabel.agreement),
            arm="majority",
            ctd_release="x",
            log_path=log,
            excluded={"1"},
        )
    assert not log.exists()


def test_zero_finding_rate_is_computed_aggregate_and_per_class_from_pair_text(tmp_path):
    """Computed from the CriticPairs the runner receives -- the fraction of papers whose text
    is empty -- not special-cased per arm, so a class-correlated rate would surface here rather
    than being hidden behind a misleading arm-specific 0.0 (limitation 8). Paper '6' is the
    only empty abstract in `_ABSTRACTS`, and it belongs to the insufficient_overlap pair."""
    line = run_critic_eval(
        pairs=_PAIRS,
        abstracts=_ABSTRACTS,
        critic=MajorityCritic(ContradictionLabel.agreement),
        arm="majority",
        ctd_release="x",
        log_path=tmp_path / "critic_runs.jsonl",
    )
    assert line["zero_finding_rate"] == pytest.approx(1 / 6)
    assert line["zero_finding_by_class"]["insufficient_overlap"] == pytest.approx(0.5)
    assert line["zero_finding_by_class"]["agreement"] == 0.0
    assert line["zero_finding_by_class"]["contradiction"] == 0.0


def test_usage_is_seeded_from_usage_fields_with_structural_zeros_for_a_free_arm(tmp_path):
    """`usage`'s keys come from `USAGE_FIELDS`, the same tuple `LlmExtractor` seeds and
    accumulates with -- a free arm has no `usage` attribute at all, so every field reads as
    the structural zero `.get(field, 0)` gives it, matching `extract_eval.py`'s own controls."""
    from biolit.extract.llm import USAGE_FIELDS

    line = run_critic_eval(
        pairs=_PAIRS,
        abstracts=_ABSTRACTS,
        critic=MajorityCritic(ContradictionLabel.agreement),
        arm="majority",
        ctd_release="x",
        log_path=tmp_path / "critic_runs.jsonl",
    )
    assert set(line["usage"]) == set(USAGE_FIELDS)
    assert all(value == 0 for value in line["usage"].values())


def test_natural_prevalence_precision_matches_the_contradiction_class_projection(tmp_path):
    """`natural_prevalence_precision` uses the contradiction class's sensitivity and
    specificity from `score()` at prevalence 0.0252 -- checked directly against
    `project_to_prevalence` rather than trusted by construction."""
    from biolit_evals.critic_scoring import project_to_prevalence

    line = run_critic_eval(
        pairs=_PAIRS,
        abstracts=_ABSTRACTS,
        critic=MajorityCritic(ContradictionLabel.agreement),
        arm="majority",
        ctd_release="x",
        log_path=tmp_path / "critic_runs.jsonl",
    )
    contradiction = line["per_class"]["contradiction"]
    expected = project_to_prevalence(contradiction["recall"], contradiction["specificity"], 0.0252)
    assert line["natural_prevalence_precision"] == pytest.approx(expected)


def test_parse_failures_are_counted_independently_of_refusals(tmp_path):
    """`parse_failures` and `refusals` are two different aggregate diagnostics, read off two
    different counters -- a critic that only fails to parse must not be reported as refusing."""
    critic = _ParseFailingCritic(fail_on={"3"})
    line = run_critic_eval(
        pairs=_PAIRS,
        abstracts=_ABSTRACTS,
        critic=critic,
        arm="lexicon",
        ctd_release="x",
        log_path=tmp_path / "critic_runs.jsonl",
    )
    assert line["parse_failures"] == 1
    assert line["refusals"] == 0


def test_excluded_pmids_count_reflects_the_set_actually_passed(tmp_path):
    """A log line cannot say whether its BC5CDR contamination check was armed or a silent
    empty-set pass-through unless the size of `excluded` is itself in the line."""
    line = run_critic_eval(
        pairs=_PAIRS,
        abstracts=_ABSTRACTS,
        critic=MajorityCritic(ContradictionLabel.agreement),
        arm="majority",
        ctd_release="x",
        log_path=tmp_path / "critic_runs.jsonl",
        excluded={"99", "100", "101"},
    )
    assert line["excluded_pmids_count"] == 3


def test_manifest_hash_under_limit_covers_the_full_pairs_not_the_scored_subset(tmp_path):
    """`manifest_hash` identifies the corpus artifact this run drew from -- the same way
    `git_sha` names the code -- not the subset a `--limit`ed pilot actually scored. `pilot`
    and `n_pairs` already say how much of that artifact this line scored."""
    line = run_critic_eval(
        pairs=_PAIRS,
        abstracts=_ABSTRACTS,
        critic=MajorityCritic(ContradictionLabel.agreement),
        arm="majority",
        ctd_release="x",
        log_path=tmp_path / "critic_runs.jsonl",
        limit=2,
    )
    assert line["manifest_hash"] == manifest_hash(_PAIRS)
    assert line["manifest_hash"] != manifest_hash(_PAIRS[:2])


@dataclass
class _CostlyCritic:
    """A minimal Critic whose `usage` grows by a fixed amount every `judge()` call, for
    pricing the kill switch. Mirrors `LlmCritic`'s convention: `usage` is a `Counter` seeded
    from `USAGE_FIELDS`, incremented before the pair is answered."""

    per_call_tokens: int
    usage: Counter[str] = field(default_factory=lambda: Counter(dict.fromkeys(USAGE_FIELDS, 0)))

    def judge(self, pair: CriticPair) -> ContradictionFinding:
        self.usage["input_tokens"] += self.per_call_tokens
        return ContradictionFinding(
            paper_id_a=pair.paper_id_a,
            paper_id_b=pair.paper_id_b,
            label=ContradictionLabel.agreement,
            rationale="ok",
        )


def test_budget_exceeded_propagates_and_is_not_counted_as_a_parse_failure(tmp_path):
    """`BudgetExceeded` is the one documented spend safeguard (spec §4): it must ABORT the
    run, not be swallowed by the runner's broad `except Exception` around `judge()`. If it
    were swallowed, the run would complete normally with this pair counted as a parse
    failure -- turning the safety mechanism into a silently-counted failure, worse than not
    having it. `pytest.raises` here proves both halves at once: a swallowed
    `BudgetExceeded` would let `run_critic_eval` return a completed line instead of raising."""
    critic = _CostlyCritic(per_call_tokens=1000)
    kill_switch = KillSwitch(limit=0.001)
    prices: dict[str, float] = dict.fromkeys(USAGE_FIELDS, 1.0)  # $1/token -- trips on call 1
    log = tmp_path / "critic_runs.jsonl"
    with pytest.raises(BudgetExceeded):
        run_critic_eval(
            pairs=_PAIRS,
            abstracts=_ABSTRACTS,
            critic=critic,
            arm="abstract",
            ctd_release="x",
            log_path=log,
            kill_switch=kill_switch,
            prices=prices,
        )
    assert not log.exists()


def test_an_unguarded_judge_exception_is_counted_as_a_parse_failure_not_a_crash(tmp_path):
    """Task 11's LlmCritic raises CriticParseError on unparseable output SPECIFICALLY so the
    runner can count it in parse_failures. An unguarded call would abort the corpus mid-run and
    produce zero log lines -- the opposite of the intent -- so the run must still complete."""
    critic = _RaisingCritic(raise_on={"3"})
    log = tmp_path / "critic_runs.jsonl"
    line = run_critic_eval(
        pairs=_PAIRS,
        abstracts=_ABSTRACTS,
        critic=critic,
        arm="lexicon",
        ctd_release="x",
        log_path=log,
    )
    assert line["parse_failures"] == 1
    assert line["n_pairs"] == 3
    assert len(log.read_text(encoding="utf-8").strip().splitlines()) == 1


def test_main_refuses_a_paid_arm_with_no_budget_usd(tmp_path, capsys):
    """`--budget-usd` is what ARMS the spend guard (spec §4) for a paid arm -- omitting it must
    fail loudly via `parser.error`, not silently run the arm with no cost accounting at all,
    which is the exact unguarded state this flag exists to close. This must fire BEFORE any
    Anthropic client is constructed, so the test needs no credential and makes no network call.
    `argparse.ArgumentParser.error` prints its message to stderr and exits with status 2 --
    the `SystemExit` itself carries only that status code, not the text -- so `capsys` is what
    pins the refusal to the budget check specifically rather than the pre-existing `--model`
    check (both raise the same `SystemExit(2)`)."""
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text("", encoding="utf-8")
    abstracts = tmp_path / "abstracts.json"
    abstracts.write_text("{}", encoding="utf-8")
    with pytest.raises(SystemExit):
        main(
            [
                "--arm",
                "abstract",
                "--manifest",
                str(manifest),
                "--abstracts",
                str(abstracts),
                "--model",
                "claude-sonnet-5",
                "--ctd-release",
                "x",
            ]
        )
    assert "--budget-usd" in capsys.readouterr().err


def test_main_refuses_a_paid_arm_against_an_unpriced_model(tmp_path, capsys):
    """A model absent from `biolit.config.CRITIC_PRICES_PER_MTOK` has no hand-verified rate --
    `critic_prices_per_token` returns `None` for it, and `main` must refuse rather than fall
    back to a default price. Fires before any Anthropic client is constructed (same reasoning
    as the budget-usd refusal above), so this needs no credential and makes no network call."""
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text("", encoding="utf-8")
    abstracts = tmp_path / "abstracts.json"
    abstracts.write_text("{}", encoding="utf-8")
    with pytest.raises(SystemExit):
        main(
            [
                "--arm",
                "abstract",
                "--manifest",
                str(manifest),
                "--abstracts",
                str(abstracts),
                "--model",
                "gpt-4o",
                "--budget-usd",
                "5.0",
                "--ctd-release",
                "x",
            ]
        )
    assert "gpt-4o" in capsys.readouterr().err
