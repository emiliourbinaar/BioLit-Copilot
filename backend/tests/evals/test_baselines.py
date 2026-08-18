import json
import math
import random
import statistics

import pytest

from biolit.domain.enums import EntityLabel
from biolit.ner.windowing import sentence_spans
from biolit_evals.baselines import (
    MIN_SEEDS,
    BaselineReport,
    beats_on_every_seed,
    bernoulli_recall_moments,
    distribution,
    first_k,
    format_row,
    last_k,
    load_llm_arms,
    n_selected,
    pad_to_budget,
    render_report,
    run_baselines,
    score_over_seeds,
    score_selection,
    select_at_rate,
)
from biolit_evals.mesh_gold import GoldDocument, GoldMention

# Three documents of DIFFERENT lengths, so a selector that ignored the per-document count
# and emitted a fixed index set would disagree on at least one of them.
COUNTS = {"a": 5, "b": 3, "c": 1}


def test_first_k_takes_the_leading_k_sentence_indices_of_every_document():
    assert first_k(COUNTS, 2) == {"a": {0, 1}, "b": {0, 1}, "c": {0}}


def test_first_k_clamps_to_the_document_length_rather_than_inventing_indices():
    # "c" has ONE sentence, so `first 4` must yield {0} and not {0, 1, 2, 3}. Three phantom
    # indices are absent from gold, so they land in `fp` and deflate precision for sentences
    # the document does not have -- and they inflate the selection COUNT, which is the
    # quantity every rate-matched comparison here is built on.
    assert first_k(COUNTS, 4) == {"a": {0, 1, 2, 3}, "b": {0, 1, 2}, "c": {0}}


def test_last_k_takes_the_trailing_k_sentence_indices_of_every_document():
    # NOT the mirror of `first_k` on the corpus: the trailing window starts at a DIFFERENT
    # offset per document, which is the whole reason `last 4` and `first 4` select the same
    # 1985 sentences yet score differently.
    assert last_k(COUNTS, 2) == {"a": {3, 4}, "b": {1, 2}, "c": {0}}


def test_last_k_clamps_at_zero_rather_than_running_off_the_front():
    # `count - k` is NEGATIVE for "b" (3-4) and "c" (1-4). Unclamped, `range(-1, 3)` yields
    # -1, and a negative sentence index is never in gold, so it scores as a false positive
    # for a sentence that cannot exist.
    assert last_k(COUNTS, 4) == {"a": {1, 2, 3, 4}, "b": {0, 1, 2}, "c": {0}}


@pytest.mark.parametrize("selector", [first_k, last_k])
@pytest.mark.parametrize("k", [0, -1])
def test_a_non_positive_k_is_rejected_rather_than_silently_selecting_nothing(selector, k):
    # ADR-0014, THIRD BRANCH -- the failure without this guard is QUIETER, not absent. Both
    # degenerate values reach `range()` as an empty range: `first_k` via `min(k, count)`,
    # `last_k` via a start at or past `count`. Unguarded, every document maps to the empty
    # set and the row prints P=R=F1=0.0000 at n=0, which reads as "this heuristic selects
    # nothing useful" and blames the baseline for an argument error. The guard names `k`.
    with pytest.raises(ValueError, match="k must be >= 1"):
        selector(COUNTS, k)


def test_n_selected_sums_across_documents_rather_than_counting_documents():
    # 2 + 2 + 1: "c" contributes ONE, not two, because `first_k` clamped it. A count that
    # multiplied documents by k would say 6 and every rate in the table would be wrong.
    assert n_selected(first_k(COUNTS, 2)) == 5


def test_select_at_rate_at_p_one_takes_every_sentence_and_at_p_zero_takes_none():
    # The two endpoints pin that the draw is per SENTENCE, not per document: a per-document
    # coin at p=1 would also select everything, but `first_k(COUNTS, max(COUNTS.values()))`
    # is the only shape that can equal it here, and the p=0 arm rules out a selector that
    # simply returns the whole corpus regardless of p.
    everything = {"a": {0, 1, 2, 3, 4}, "b": {0, 1, 2}, "c": {0}}
    assert select_at_rate(COUNTS, 1.0, random.Random(0)) == everything
    assert select_at_rate(COUNTS, 0.0, random.Random(0)) == {"a": set(), "b": set(), "c": set()}


# SEVEN documents, and the second mapping inserts them in REVERSE order. Seven, not two:
# a two-element fixture leaves a dropped `sorted()` surviving on 4 of 12 PYTHONHASHSEED
# values, because two keys land in the same relative order often enough to pass by luck.
# With 7 the chance of the two insertion orders producing the same iteration order is 1/7!.
_FORWARD = {"d1": 3, "d2": 3, "d3": 3, "d4": 3, "d5": 3, "d6": 3, "d7": 3}
_REVERSED = {pmid: count for pmid, count in reversed(list(_FORWARD.items()))}


def test_select_at_rate_ignores_the_insertion_order_of_the_counts_mapping():
    # The rng is consumed IN ITERATION ORDER, so without a `sorted()` the same seed assigns
    # a different draw to every sentence depending on how the caller happened to build its
    # dict. Two "identical corpora" would then produce two different rate-matched nulls and
    # nothing in the printed table would say why.
    assert select_at_rate(_FORWARD, 0.5, random.Random(11)) == select_at_rate(
        _REVERSED, 0.5, random.Random(11)
    )


@pytest.mark.parametrize("p", [-0.1, 1.1])
def test_a_rate_outside_zero_to_one_is_rejected_rather_than_saturating(p):
    # ADR-0014, THIRD BRANCH again, and the SILENT case this time -- `rng.random()` returns
    # [0.0, 1.0), so `< -0.1` is always False and `< 1.1` is always True. An out-of-range p
    # therefore produces a perfectly well-formed selection (empty, or the whole corpus) whose
    # closed-form mean the module would then report as `p` -- a recall of 1.1 printed beside
    # an empirical 1.0, with no exception anywhere.
    with pytest.raises(ValueError, match=r"p must be in \[0, 1\]"):
        select_at_rate(COUNTS, p, random.Random(0))


def test_pad_to_budget_reaches_the_budget_exactly_and_keeps_every_base_sentence():
    # EXACTLY, not approximately: the entire point of padding is to compare a positional
    # baseline with an arm at an IDENTICAL selection count, since recall is trivially bought
    # with volume. A padded selection one sentence over or under is not a rate-matched
    # comparison at all.
    base = first_k(COUNTS, 1)  # 3 sentences: {"a": {0}, "b": {0}, "c": {0}}
    padded = pad_to_budget(base, COUNTS, 7, random.Random(3))
    assert n_selected(padded) == 7
    for pmid, indices in base.items():
        assert indices <= padded[pmid]


def test_pad_to_budget_only_ever_adds_sentences_the_document_actually_has():
    # Padding the WHOLE corpus (budget == every sentence) is the strictest form of this:
    # any index outside `range(count)` would show up here, and phantom indices score as
    # false positives against a gold that cannot contain them.
    padded = pad_to_budget(first_k(COUNTS, 1), COUNTS, 9, random.Random(3))
    assert padded == {"a": {0, 1, 2, 3, 4}, "b": {0, 1, 2}, "c": {0}}


def test_pad_to_budget_ignores_the_insertion_order_of_the_counts_mapping():
    # `rng.sample` draws by POSITION in the candidate pool, so an unsorted pool makes the
    # padding a function of the caller's dict insertion order as well as of the seed. The
    # same seven documents, inserted in reverse, must pad identically -- otherwise "seed 0"
    # names two different selections and the reported mean is not reproducible from the seed
    # the report quotes.
    forward = pad_to_budget(first_k(_FORWARD, 1), _FORWARD, 14, random.Random(11))
    backward = pad_to_budget(first_k(_REVERSED, 1), _REVERSED, 14, random.Random(11))
    assert forward == backward


def test_a_budget_below_the_base_selection_is_rejected_because_padding_cannot_shrink():
    # `first 4` selects 1985 sentences; asking to "pad" it to 1000 is a caller error, not a
    # request to drop 985. Unguarded, `budget - n_selected(padded)` is -985 and `rng.sample`
    # raises "Sample larger than population or is negative" -- which names neither the
    # budget nor the base, and reads like an internal sampling bug.
    with pytest.raises(ValueError, match="budget 2 is below the 3 sentences"):
        pad_to_budget(first_k(COUNTS, 1), COUNTS, 2, random.Random(0))


def test_a_budget_above_the_whole_corpus_is_rejected_naming_the_corpus_size():
    # The corpus holds 9 sentences. `rng.sample` would raise the same opaque "Sample larger
    # than population" here as for the under-budget case, so the two very different caller
    # errors would be indistinguishable from the traceback.
    with pytest.raises(ValueError, match="budget 10 exceeds the 9 sentences"):
        pad_to_budget(first_k(COUNTS, 1), COUNTS, 10, random.Random(0))


def test_expected_recall_under_independent_selection_is_p_on_every_gold_subset():
    # THE REASON SIMULATION IS UNNECESSARY FOR THE MEAN. Each gold sentence is selected with
    # probability p independently of whether it is gold, so E[recall] = p for ANY subset --
    # bucket (a)'s 270 sentences and the whole 1145 alike. Three different subset sizes, one
    # mean: an implementation that let n leak into the mean would disagree on at least two.
    assert bernoulli_recall_moments(0.4655, 270).mean == 0.4655
    assert bernoulli_recall_moments(0.4655, 1145).mean == 0.4655
    assert bernoulli_recall_moments(0.4655, 1).mean == 0.4655


def test_the_closed_form_sd_shrinks_with_the_square_root_of_the_gold_subset_size():
    # Written out rather than re-deriving the formula: an implementation that forgot the
    # `/ n_gold` would report 0.4989 on bucket (a) instead of 0.0304, and an arm 0.16 SD from
    # the null would look 16x closer to it than it is.
    assert bernoulli_recall_moments(0.5, 100).sd == pytest.approx(0.05)
    assert bernoulli_recall_moments(0.5, 400).sd == pytest.approx(0.025)
    assert bernoulli_recall_moments(0.4655, 270).sd == pytest.approx(0.0303565, abs=1e-7)


def test_closed_form_moments_render_with_their_spread_and_refuse_to_become_one_number():
    # The type is the enforcement. A `Moments` interpolated into a report string carries its
    # SD with it, and `float(moments)` is a TypeError rather than a silently dropped spread --
    # so the shape of claim this module exists to remove cannot be made by accident.
    moments = bernoulli_recall_moments(0.4655, 270)
    assert f"{moments}" == "0.4655 +/- 0.0304"
    with pytest.raises(TypeError):
        float(moments)  # pyright: ignore[reportArgumentType]


def test_a_distribution_carries_the_full_spread_not_only_the_mean():
    # min and max are carried as well as the SD because the question the report has to answer
    # about the padded baseline is not "how wide is it" but "does the WORST draw still beat
    # the arm". A mean and an SD alone leave that to the reader's normal-approximation.
    values = [float(i) for i in range(MIN_SEEDS)]  # 0 .. MIN_SEEDS-1
    spread = distribution(values)
    assert spread.n_draws == MIN_SEEDS
    assert spread.mean == pytest.approx((MIN_SEEDS - 1) / 2)
    assert spread.minimum == 0.0
    assert spread.maximum == float(MIN_SEEDS - 1)
    assert spread.sd == pytest.approx(statistics.stdev(values))


def test_a_single_draw_cannot_become_a_distribution():
    # THE POINT OF THE MODULE, enforced rather than advised. The ad-hoc work this replaces
    # reported the padded baseline's 0.5444 on bucket (a) from ONE seed and no spread. Every
    # stochastic figure here reaches the report through `distribution`, so the one-seed
    # number is not merely discouraged -- it raises before it can be printed.
    with pytest.raises(ValueError, match=f"1 draw\\(s\\) is fewer than the {MIN_SEEDS}"):
        distribution([0.5444])
    with pytest.raises(ValueError, match=f"fewer than the {MIN_SEEDS}"):
        distribution([0.0] * (MIN_SEEDS - 1))


def test_a_distribution_renders_its_extremes_and_seed_count_beside_the_mean():
    spread = distribution([0.5] * (MIN_SEEDS - 1) + [0.6])
    assert f"{spread}" == f"0.5005 +/- 0.0071 [0.5000, 0.6000] (n={MIN_SEEDS})"
    with pytest.raises(TypeError):
        float(spread)  # pyright: ignore[reportArgumentType]


# 9 sentences over 3 documents; 3 of them gold; ONE of those gold sentences is in the
# restricted bucket. Hand-computable end to end, and the bucket is a strict SUBSET of gold,
# which is what bucket (a) is on the real corpus (270 of 1145).
GOLD = {"a": {0, 1}, "b": {2}}
ENDPOINT_LOST = {"a": {1}}


def test_score_selection_reports_the_aggregate_and_the_restricted_score_together():
    # first_k(COUNTS, 2) selects a:{0,1} b:{0,1} c:{0} -- 5 of 9 sentences.
    #   tp = 2 (a0, a1), fp = 3 (b0, b1, c0), fn = 1 (b2)
    #   P = 2/5, R = 2/3, F1 = 0.5, rate = 5/9
    # The restricted score is against ENDPOINT_LOST ALONE, so b0/b1/c0 are NOT its false
    # positives -- a precision computed against a one-bucket gold would be a number about
    # nothing, which is why `extract_eval` reports recall there and this does the same.
    scores = score_selection(
        first_k(COUNTS, 2), gold=GOLD, endpoint_lost=ENDPOINT_LOST, n_sentences=9
    )
    assert scores.n_selected == 5
    assert scores.rate == pytest.approx(5 / 9)
    assert scores.precision == pytest.approx(0.4)
    assert scores.recall == pytest.approx(2 / 3)
    assert scores.f1 == pytest.approx(0.5)
    assert scores.recall_on_endpoint_lost == pytest.approx(1.0)
    assert (scores.tp_on_endpoint_lost, scores.n_endpoint_lost) == (1, 1)


def test_score_over_seeds_collapses_to_the_deterministic_score_when_the_draw_is_ignored():
    # A selector that ignores its rng must produce a zero-width distribution centred on the
    # single-selection score. This pins the plumbing -- every field routed to the field of
    # the same name -- without any randomness to reason about, so a transposed pair of
    # columns (precision reported as recall) cannot hide behind a spread.
    fixed = first_k(COUNTS, 2)
    one = score_selection(fixed, gold=GOLD, endpoint_lost=ENDPOINT_LOST, n_sentences=9)
    many = score_over_seeds(
        lambda _rng: fixed,
        gold=GOLD,
        endpoint_lost=ENDPOINT_LOST,
        n_sentences=9,
        seeds=range(MIN_SEEDS),
    )
    for field in ("n_selected", "rate", "precision", "recall", "f1", "recall_on_endpoint_lost"):
        spread = getattr(many, field)
        assert spread.n_draws == MIN_SEEDS
        assert spread.sd == 0.0
        assert spread.mean == pytest.approx(getattr(one, field))
        assert spread.minimum == spread.maximum == pytest.approx(getattr(one, field))


def test_the_empirical_mean_recall_over_seeds_agrees_with_the_closed_form():
    # TWO INDEPENDENT ROUTES TO ONE NUMBER: `select_at_rate` really does draw each sentence
    # independently at rate p, and `score_over_seeds` routes the resulting recall to the
    # column of that name. Agreement with `bernoulli_recall_moments` -- computed from p
    # alone, with no simulation at all -- is the evidence that simulating the mean is
    # unnecessary rather than merely asserted to be.
    #
    # WHAT THIS DOES NOT PIN, established by mutation rather than assumed: replacing the
    # per-draw `random.Random(seed)` with ONE generator shared across all draws leaves this
    # test green. A shared generator keeps advancing, so the draws stay independent samples
    # of the same quantity and both moments still match. What it destroys is REPRODUCIBILITY
    # from the seeds, which is a different property and is pinned by the test below. An
    # earlier version of this comment claimed the agreement caught the shared generator; it
    # does not, and the mutant proved it.
    counts = {f"d{i:03d}": 20 for i in range(50)}  # 1000 sentences
    gold = {f"d{i:03d}": {0, 1, 2, 3} for i in range(50)}  # 200 gold sentences
    closed = bernoulli_recall_moments(0.5, 200)
    spread = score_over_seeds(
        lambda rng: select_at_rate(counts, 0.5, rng),
        gold=gold,
        endpoint_lost=gold,
        n_sentences=1000,
        seeds=range(MIN_SEEDS),
    )
    assert spread.recall.sd > 0.0  # the seeds really are varying
    # 200 draws of a statistic whose own SD is `closed.sd` have a Monte-Carlo SE of
    # closed.sd / sqrt(200) ~= 0.0025, so 0.01 is a four-SE band.
    assert spread.recall.mean == pytest.approx(closed.mean, abs=0.01)
    assert spread.recall.sd == pytest.approx(closed.sd, rel=0.15)


def test_score_over_seeds_gives_the_same_answer_twice_for_the_same_seeds():
    # REPRODUCIBILITY IS THE WHOLE POINT OF THE MODULE, and it is a strictly stronger demand
    # than "the draws are independent". `score_over_seeds` must build a FRESH
    # `random.Random(seed)` per draw; share one generator across draws and every moment above
    # still matches the closed form -- the numbers are still valid samples -- but the row is
    # no longer a function of the seeds, so a second run prints different figures and the
    # committed report cannot be reproduced. That is the exact failure this module exists to
    # end, and only this test sees it.
    counts = {f"d{i:03d}": 20 for i in range(50)}
    gold = {f"d{i:03d}": {0, 1, 2, 3} for i in range(50)}
    runs = [
        score_over_seeds(
            lambda rng: select_at_rate(counts, 0.5, rng),
            gold=gold,
            endpoint_lost=gold,
            n_sentences=1000,
            seeds=range(MIN_SEEDS),
        )
        for _ in range(2)
    ]
    assert runs[0] == runs[1]


def _log_line(
    *,
    dataset: str = "bc5cdr_test500",
    effort: str = "low",
    n_selected_: int = 5,
    lost: dict[str, list[int]] | None = None,
    with_llm: bool = True,
    timestamp: str = "2026-08-17T00:59:11+00:00",
    lost_tp: int = 1,
) -> str:
    """One `extract_runs.jsonl` line, shaped exactly as `run_extract_eval` writes it."""
    arms: dict[str, object] = {
        "control-real": {
            "miss_buckets": {"endpoint_lost_sentences": {"a": [1]} if lost is None else lost}
        }
    }
    if with_llm:
        arms["llm"] = {
            "model": "claude-opus-5",
            "effort": effort,
            "n_selected": n_selected_,
            "sentence": {"precision": 0.4, "recall": 2 / 3, "f1": 0.5, "tp": 2, "fp": 3, "fn": 1},
            # tp, fn and n_gold_sentences are DELIBERATELY three different numbers. With
            # tp == n_gold_sentences a reader that took the bucket size from `tp` would be
            # indistinguishable from one that took it from `n_gold_sentences` -- and on the
            # real log those two are 127 and 270, so the confusion would report the arm as
            # recalling 127 of 127. `lost_tp` moves all three together rather than letting a
            # caller set a recall that its own tp and fn contradict.
            "recall_on_endpoint_lost": {
                "recall": lost_tp / 3,
                "tp": lost_tp,
                "fn": 3 - lost_tp,
                "n_gold_sentences": 3,
            },
        }
    return json.dumps(
        {
            "timestamp": timestamp,
            "git_sha": "3c029c0",
            "dataset": dataset,
            "n_documents": 500,
            "n_gold_sentences": 1145,
            "arms": arms,
        }
    )


def test_load_llm_arms_reads_the_arms_score_and_bucket_a_membership_off_the_log(tmp_path):
    # The arm's row is READ, never recomputed: the log records what it scored, and its
    # selections are gone once the run ends. Bucket (a)'s membership is read for the opposite
    # reason -- it CAN be recomputed, and re-deriving it would be a second copy of the
    # bucketing logic to keep in agreement with `classify_misses`.
    path = tmp_path / "runs.jsonl"
    path.write_text(_log_line() + "\n", encoding="utf-8")
    (arm,) = load_llm_arms(str(path), dataset="bc5cdr_test500", n_sentences=9)
    assert (arm.model, arm.effort, arm.git_sha) == ("claude-opus-5", "low", "3c029c0")
    assert arm.endpoint_lost == {"a": {1}}
    assert arm.scores.n_selected == 5
    assert arm.scores.rate == pytest.approx(5 / 9)
    assert arm.scores.precision == pytest.approx(0.4)
    assert arm.scores.recall == pytest.approx(2 / 3)
    assert arm.scores.f1 == pytest.approx(0.5)
    assert arm.scores.recall_on_endpoint_lost == pytest.approx(1 / 3)
    assert (arm.scores.tp_on_endpoint_lost, arm.scores.n_endpoint_lost) == (1, 3)


def test_a_limit_pilot_and_a_control_only_run_are_both_skipped(tmp_path):
    # `--limit 20` tags its line `bc5cdr_test500_limit20`. A SUBSTRING match on the dataset
    # tag would pull that pilot in and score 500-document baselines against a 20-document
    # arm -- the corpus mismatch `assert_dataset_size` exists to prevent one level up. A
    # control-only line has no `llm` key at all and is simply not an arm.
    path = tmp_path / "runs.jsonl"
    path.write_text(
        "\n".join(
            [
                _log_line(dataset="bc5cdr_test500_limit20", n_selected_=90),
                _log_line(with_llm=False),
                _log_line(effort="high", n_selected_=7),
                "",
            ]
        ),
        encoding="utf-8",
    )
    arms = load_llm_arms(str(path), dataset="bc5cdr_test500", n_sentences=9)
    assert [(arm.effort, arm.scores.n_selected) for arm in arms] == [("high", 7)]


def test_disagreeing_bucket_a_membership_across_runs_is_rejected(tmp_path):
    # `control-real` is deterministic, so two full-corpus lines at one commit MUST agree.
    # If they do not, "recall on endpoint_lost" names a different gold subset per row and
    # the table silently compares each selector against its own population -- a sharper form
    # of the very defect (a score quoted against an undeclared comparator) this module
    # exists to fix. There is no correct membership to pick, so it raises.
    path = tmp_path / "runs.jsonl"
    path.write_text(
        _log_line() + "\n" + _log_line(lost={"a": [1], "b": [0]}) + "\n", encoding="utf-8"
    )
    with pytest.raises(ValueError, match="different bucket-\\(a\\) membership"):
        load_llm_arms(str(path), dataset="bc5cdr_test500", n_sentences=9)


def test_a_log_with_no_matching_arm_is_rejected_rather_than_returning_an_empty_table(tmp_path):
    # An empty list would print a table of baselines with NOTHING to compare them against --
    # the shape of the retracted claim (a number quoted with no live comparator), just with
    # the missing side being the arm instead of the baseline.
    path = tmp_path / "runs.jsonl"
    path.write_text(_log_line(with_llm=False) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="no line in .* has dataset"):
        load_llm_arms(str(path), dataset="bc5cdr_test500", n_sentences=9)


def test_a_stochastic_row_cannot_render_any_cell_as_a_bare_number(tmp_path):
    # THE LAST PLACE A SINGLE DRAW COULD LEAK. `distribution` stops a one-seed figure being
    # COMPUTED; this stops a many-seed figure being PRINTED as though it were exact. Every
    # cell of a stochastic row carries its SD, so no column of the table can be copied into
    # prose as a point estimate by reading it off the wrong row.
    stochastic = score_over_seeds(
        lambda _rng: first_k(COUNTS, 2),
        gold=GOLD,
        endpoint_lost=ENDPOINT_LOST,
        n_sentences=9,
        seeds=range(MIN_SEEDS),
    )
    row = format_row("padded", stochastic)
    assert row.count("+/-") == 6  # n, rate, P, R, F1, R@endpoint_lost -- every one of them
    # The deterministic row is the contrast: exact figures, no spread, because there is none.
    exact = format_row(
        "first 2",
        score_selection(first_k(COUNTS, 2), gold=GOLD, endpoint_lost=ENDPOINT_LOST, n_sentences=9),
    )
    assert "+/-" not in exact
    assert "0.4000" in exact and "0.6667" in exact


def test_beating_an_arm_is_judged_on_the_worst_seed_not_on_the_mean():
    # "The positional baseline beats the LLM arm on bucket (a)" is a claim about EVERY draw,
    # not about the average one. A spread whose mean clears the arm but whose worst draw does
    # not is exactly the case the one-seed ad-hoc computation could not distinguish -- it had
    # a single draw and no way to know which side of this line it fell on.
    spread = distribution([0.52] * (MIN_SEEDS - 1) + [0.46])
    assert spread.mean > 0.4704  # the mean clears the arm ...
    assert not beats_on_every_seed(spread, 0.4704)  # ... and the claim is still not robust
    assert beats_on_every_seed(distribution([0.52] * MIN_SEEDS), 0.4704)
    # Ties do not count as beating: an equal draw is not evidence for the baseline.
    assert not beats_on_every_seed(distribution([0.4704] * MIN_SEEDS), 0.4704)


# --------------------------------------------------------------------------------------
# `run_baselines` -- the orchestration that used to sit inside `main` and was untestable
# there. FOUND BY MUTATION: five values `main` chose survived every mutant with the whole
# suite green -- the `assert_gold_sentence_regression_pin` CALL, the budget arm (`arms[0]`
# vs `arms[-1]`, which silently re-targets the padded row at a DIFFERENT logged run and
# changes a number quoted as this branch's headline), the rate denominator, the padded
# row's base `k`, and the positional sweep. `main` keeps no direct test, matching
# `end_to_end.main` / `ner_eval.main` / `cluster_eval.main`; everything it orchestrated
# that can be tested moved into `run_baselines` and `render_report`, which are tested here.
#
# SIX sentences per document, deliberately. `first 4` must differ from `first 3` and from
# `first 5` on this fixture, and the padded budget must differ between the two logged arms
# -- a fixture where the mutated and original values coincide pins nothing, which is the
# trap that produced this same finding twice already on this branch.
_SENTENCES = (
    "Metformin was given.",
    "Acidosis followed metformin use.",
    "Aspirin caused fever.",
    "Filler four here.",
    "Filler five here.",
    "Filler six here.",
)
_TEXT = " ".join(_SENTENCES)
_MET, _ACIDOSIS = "MESH:D008687", "MESH:D000138"
_ASPIRIN, _FEVER = "MESH:D001241", "MESH:D005334"


def _corpus_mention(pmid: str, needle: str, label: EntityLabel, mesh_id: str) -> GoldMention:
    # Offsets are LOCATED in the text rather than written down, so a reworded fixture cannot
    # silently detach a mention from the sentence it is supposed to sit in.
    start = _TEXT.index(needle)
    return GoldMention(
        pmid=pmid,
        start=start,
        end=start + len(needle),
        text=needle,
        label=label,
        mesh_ids=(mesh_id,),
    )


def _corpus_document(pmid: str) -> GoldDocument:
    return GoldDocument(
        pmid=pmid,
        text=_TEXT,
        mentions=[
            _corpus_mention(pmid, "Acidosis", EntityLabel.DISEASE, _ACIDOSIS),
            _corpus_mention(pmid, "metformin use", EntityLabel.CHEMICAL, _MET),
            _corpus_mention(pmid, "Aspirin", EntityLabel.CHEMICAL, _ASPIRIN),
            _corpus_mention(pmid, "fever", EntityLabel.DISEASE, _FEVER),
        ],
    )


_CORPUS = [_corpus_document(pmid) for pmid in ("d1", "d2", "d3")]
_RELATIONS = {document.pmid: {(_MET, _ACIDOSIS), (_ASPIRIN, _FEVER)} for document in _CORPUS}
# 3 documents x 6 sentences = 18; sentences 1 and 2 of each are gold, so 6 gold sentences.
_COUNTS = {document.pmid: len(sentence_spans(document.text)) for document in _CORPUS}

# Bucket (a) holds ONE sentence inside `first 4` (d1's index 1) and ONE outside it (d2's
# index 5). Both inside would make the padded row's restricted recall a constant 1.0 and
# the ROBUST/NOT ROBUST verdict unfalsifiable; both outside would make it depend only on
# the padding. One of each keeps the worst observed draw at 0.5000 and still varying.
_LOST = {"d1": [1], "d2": [5]}


def _two_arm_log(tmp_path) -> str:
    """A log with TWO full-corpus LLM lines whose selection counts DIFFER.

    Two, and different, is the whole point: with one line, or with two carrying the same
    `n_selected`, `arms[0]` and `arms[-1]` collapse to the same budget and the mutant that
    re-targets the padded row at a different run survives untouched.
    """
    path = tmp_path / "runs.jsonl"
    path.write_text(
        "\n".join(
            [
                _log_line(
                    effort="low",
                    n_selected_=14,
                    lost=_LOST,
                    timestamp="2026-08-17T00:59:11+00:00",
                    lost_tp=1,
                ),
                _log_line(
                    effort="high",
                    n_selected_=16,
                    lost=_LOST,
                    timestamp="2026-08-17T01:51:00+00:00",
                    lost_tp=2,
                ),
                "",
            ]
        ),
        encoding="utf-8",
    )
    return str(path)


def _report(tmp_path, log_path: str | None = None) -> BaselineReport:
    return run_baselines(
        documents=_CORPUS,
        relations=_RELATIONS,
        log_path=_two_arm_log(tmp_path) if log_path is None else log_path,
        n_seeds=MIN_SEEDS,
    )


def test_run_baselines_derives_the_corpus_shape_from_the_documents_it_was_handed(tmp_path):
    # The counts come from `sentence_spans(document.text)` -- the SAME string gold and the
    # miss buckets split -- so a baseline's sentence indices mean what the arm's meant.
    report = _report(tmp_path)
    assert (report.n_documents, report.n_sentences, report.n_gold) == (3, 18, 6)
    assert report.n_endpoint_lost == 2


# FIVE HUNDRED documents, because `_GOLD_SENTENCE_PINS` carries an entry for 500 documents
# and for NO other corpus size. On the 3-document fixture above the pin is a no-op, so a
# `run_baselines` with the pin CALL deleted passes every other test in this file unchanged --
# the exact value-collapse that let the mutant survive in the first place. At 500 documents
# this corpus yields 1000 gold sentences against the pinned 1145, so the call is the only
# thing here that can raise.
_PINNED_CORPUS = [_corpus_document(f"p{index:03d}") for index in range(500)]
_PINNED_RELATIONS = {
    document.pmid: {(_MET, _ACIDOSIS), (_ASPIRIN, _FEVER)} for document in _PINNED_CORPUS
}


def test_run_baselines_puts_the_corpus_through_the_arms_own_gold_construction_pin(tmp_path):
    # WHAT THE CALL DEFENDS is the property this module's docstring claims outright: these
    # baselines are scored against the SAME gold the logged arm was scored against. Drop the
    # call and a corpus whose gold construction has drifted is scored in silence, every row of
    # the table compares two populations, and nothing printed says so -- the module would be
    # reproducing the defect it exists to fix. `SystemExit` rather than `ValueError` because
    # the pin is a runner-level stop, matching how `run_extract_eval` reaches it.
    with pytest.raises(SystemExit, match="gold-sentence pin"):
        run_baselines(
            documents=_PINNED_CORPUS,
            relations=_PINNED_RELATIONS,
            log_path=_two_arm_log(tmp_path),
            n_seeds=MIN_SEEDS,
        )


def test_the_budget_comes_from_the_first_logged_arm_and_not_from_the_last(tmp_path):
    # THE MUTANT THIS EXISTS FOR is `arms[0]` -> `arms[-1]`, which re-targets the padded row at
    # a DIFFERENT logged run and moves a number this branch quotes as its headline: on the real
    # log it swaps the 2274-sentence `low` run for the 2269-sentence `high` one. Nothing printed
    # contradicts it, because the row's label and its `n` both follow whatever budget was used.
    #
    # The fixture log carries TWO full-corpus arms whose `n_selected` DIFFER (14 and 16). Two
    # equal counts, or a single logged line, would collapse `arms[0]` and `arms[-1]` onto one
    # value and pin nothing at all.
    report = _report(tmp_path)
    assert report.budget == 14
    assert (report.budget_arm.effort, report.budget_arm.timestamp) == (
        "low",
        "2026-08-17T00:59:11+00:00",
    )
    # The budget is not merely STORED as 14 -- the padded row was actually built to it.
    # `pad_to_budget` fixes `n_selected` by construction, so this column is a point mass at
    # whichever budget the row really used.
    assert report.padded.n_selected.mean == 14.0
    assert report.padded.n_selected.minimum == report.padded.n_selected.maximum == 14.0
    # And the column that carries the claim moves with it: padding to 16 instead of 14 draws
    # 4 of the 6 unselected sentences instead of 2, lifting bucket-(a) recall from 0.68 to 0.85.
    assert report.padded.recall_on_endpoint_lost.mean == pytest.approx(0.68, abs=0.03)


def test_the_selection_rate_is_the_budget_over_the_corpus_sentence_count(tmp_path):
    # `budget / n_sentences` is the rate the WHOLE comparison is matched on: it is the p the
    # rate-matched null draws at, the p its closed-form recall equals, and the figure printed
    # beside the budget. An off-by-one denominator leaves every row well-formed and every one
    # of them describing a corpus one sentence larger than the one that was scored.
    report = _report(tmp_path)
    assert report.rate == pytest.approx(14 / 18)
    # `bernoulli_recall_moments(rate, n_lost).mean` IS the rate exactly, so this pins that the
    # rate reached the closed form rather than only the `rate` field. The SD is the half that
    # depends on bucket (a)'s size, and it is written out rather than re-derived.
    assert report.closed_form.mean == pytest.approx(14 / 18)
    assert report.closed_form.sd == pytest.approx(math.sqrt((14 / 18) * (4 / 18) / 2))


def test_the_padded_row_is_built_on_the_first_four_sentences_its_label_claims(tmp_path):
    # THE BASE IS THE ONLY THING THAT CAN BE CHECKED AGAINST THE CLAIM. The padded row is
    # LABELLED "first 4 + random pad to N" and its `n_selected` is the budget whatever base it
    # started from, so a base built with the wrong `k` prints a row that contradicts nothing.
    # Six sentences per document is deliberate: `first 3`, `first 4` and `first 5` are three
    # different selections here, where four-sentence documents would collapse two of them.
    report = _report(tmp_path)
    assert report.padded_base == {"d1": {0, 1, 2, 3}, "d2": {0, 1, 2, 3}, "d3": {0, 1, 2, 3}}
    assert n_selected(report.padded_base) == 12
    # The base actually REACHED the padded row, not just the report field: a 9-sentence
    # `first 3` base padded to the same budget of 14 draws 5 of the 9 remaining sentences
    # instead of 2 of 6, which lifts bucket-(a) recall from 0.68 to 0.78.
    assert report.padded.recall_on_endpoint_lost.mean == pytest.approx(0.68, abs=0.03)


def test_the_positional_sweep_is_first_2_first_4_and_last_4_in_that_order(tmp_path):
    # BOTH HALVES OF EVERY ROW ARE PINNED -- the label AND the score -- because the two mutants
    # here fail differently. `for k in (2, 4)` -> `(2, 5)` renames the row as well as rescoring
    # it; a `first_k(counts, k + 1)` would keep the label and change only the score, and a row
    # whose label disagrees with its numbers is the worse of the two.
    #
    # Gold is sentences 1 and 2 of every document, so the three rows carry three different
    # (n, recall) pairs and no neighbouring k reproduces any of them: `first 3` recalls 1.0 at
    # n=9, `first 5` 1.0 at n=15, `last 3` 0.0 at n=9, `last 5` 1.0 at n=15.
    report = _report(tmp_path)
    assert [(label, scores.n_selected, scores.recall) for label, scores in report.positional] == [
        ("first 2", 6, 0.5),
        ("first 4", 12, 1.0),
        ("last 4", 12, 0.5),
    ]


def test_the_rendered_report_names_the_run_the_budget_was_taken_from(tmp_path):
    # `budget_arm` is CARRIED rather than re-derived at print time. A renderer that named the
    # arm by re-indexing `report.arms` could name a different run than the budget came from,
    # and the reader would be told the padded row matches an arm it does not match -- worse
    # than the wrong budget alone, because the mismatch is asserted rather than merely present.
    lines = render_report(_report(tmp_path))
    assert "Budget for the padded row: 14 sentences (rate 0.7778), from low" in lines
    assert "  run 2026-08-17T00:59:11+00:00 at sha 3c029c0. No API call was made." in lines
    assert any(line.startswith("first 4 + random pad to 14") for line in lines)
    assert any(line.startswith("rate-matched random p=0.7778") for line in lines)


def test_the_rendered_verdict_is_judged_against_every_arm_on_the_worst_draw(tmp_path):
    # The verdict loop is the ONLY caller of `beats_on_every_seed`, so a verdict computed
    # inside an unreachable `main` was a judgement nothing checked. The two fixture arms sit on
    # OPPOSITE sides of the padded row's worst observed draw (0.5000) -- `low` scored 1/3 on
    # bucket (a), `high` 2/3 -- so one row must read ROBUST and the other NOT ROBUST. A fixture
    # where both arms fell the same side would leave a hard-wired verdict indistinguishable
    # from a computed one, and the row is also the only place the arm's own score is printed
    # beside the baseline's WORST draw rather than its mean.
    lines = render_report(_report(tmp_path))
    assert [line for line in lines if " -- worst of " in line] == [
        "  vs LLM low (2026-08-17T00:59) at 0.3333: ROBUST -- worst of 200 draws is 0.5000",
        "  vs LLM high (2026-08-17T01:51) at 0.6667: NOT ROBUST -- worst of 200 draws is 0.5000",
    ]


def test_the_seed_count_the_caller_asked_for_is_the_seed_count_that_is_drawn(tmp_path):
    # MIN_SEEDS + 1, not MIN_SEEDS: every other test here passes exactly MIN_SEEDS, so a
    # `range(n_seeds)` hard-wired back to `range(MIN_SEEDS)` would agree with all of them and
    # the `--seeds` flag would silently stop doing anything. The printed report reads the
    # OBSERVED `n_draws` rather than the requested count, so a request that was not honoured
    # would not even show up in the prose beneath the table.
    report = run_baselines(
        documents=_CORPUS,
        relations=_RELATIONS,
        log_path=_two_arm_log(tmp_path),
        n_seeds=MIN_SEEDS + 1,
    )
    assert report.padded.recall_on_endpoint_lost.n_draws == MIN_SEEDS + 1
    assert report.random_null.recall_on_endpoint_lost.n_draws == MIN_SEEDS + 1
    assert f"over {MIN_SEEDS + 1} seeds" in "\n".join(render_report(report))
