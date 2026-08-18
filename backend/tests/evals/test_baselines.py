import json
import random
import statistics

import pytest

from biolit_evals.baselines import (
    MIN_SEEDS,
    beats_on_every_seed,
    bernoulli_recall_moments,
    distribution,
    first_k,
    format_row,
    last_k,
    load_llm_arms,
    n_selected,
    pad_to_budget,
    score_over_seeds,
    score_selection,
    select_at_rate,
)

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
            # recalling 127 of 127.
            "recall_on_endpoint_lost": {
                "recall": 1 / 3,
                "tp": 1,
                "fn": 2,
                "n_gold_sentences": 3,
            },
        }
    return json.dumps(
        {
            "timestamp": "2026-08-17T00:59:11+00:00",
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
