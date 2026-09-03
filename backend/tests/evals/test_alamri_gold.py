import random
import re
from collections import Counter
from pathlib import Path

import pytest

from biolit.domain.records import ContradictionLabel
from biolit_evals.alamri_gold import (
    build_distractors,
    build_pairs,
    key_terms,
    manifest_hash,
    pair_signals,
    parse_corpus,
    question_id,
    read_manifest,
    sample_batch,
    write_manifest,
)

FIXTURE = Path(__file__).parent / "fixtures" / "alamri_fixture.xml"

Q1 = (
    "In women with pre-eclampsia, does treatment with L-arginine, compared to placebo, "
    "reduce blood pressure"
)
Q2 = "Does aspirin therapy lower stroke incidence in elderly cohorts"
Q3 = "In pregnant women, does arginine supplementation reduce blood pressure at term"


@pytest.fixture
def claims():
    return parse_corpus(FIXTURE.read_text(encoding="utf-8"))


def test_parse_corpus_reads_every_claim(claims):
    assert len(claims) == 13


def test_ys_no_pairs_under_one_question_are_labelled_contradiction(claims):
    pairs = build_pairs(claims)
    pair = next(p for p in pairs if {p.paper_id_a, p.paper_id_b} == {"201", "202"})
    assert pair.label is ContradictionLabel.contradiction


def test_a_paper_is_never_paired_with_itself(claims):
    """PMID 20228403 in the real corpus carries both a YS and a NO claim under one question
    (olmesartan vs valsartan), which would derive a paper that contradicts itself. The
    fixture reproduces that shape with PMID 105."""
    pairs = build_pairs(claims)
    assert [p for p in pairs if p.paper_id_a == p.paper_id_b] == []


def test_the_same_paper_pair_under_two_questions_gets_distinct_pair_ids(claims):
    """101/102 appear together under both Q1 and Q3. Three such pairs exist in the real
    corpus, so a pair_id built from the two PMIDs alone would collide and merge two
    distinct judgments into one manifest row."""
    both = [p for p in build_pairs(claims) if {p.paper_id_a, p.paper_id_b} == {"101", "102"}]
    assert len(both) == 2
    assert len({p.pair_id for p in both}) == 2


def test_same_assertion_pairs_under_one_question_are_labelled_agreement(claims):
    pairs = build_pairs(claims)
    pair = next(p for p in pairs if {p.paper_id_a, p.paper_id_b} == {"101", "104"})
    assert pair.label is ContradictionLabel.agreement


def test_key_terms_drops_stopwords_and_short_tokens():
    terms = key_terms(Q2)
    assert {"aspirin", "stroke", "incidence"} <= terms
    assert "does" not in terms
    assert "in" not in terms


def test_pair_signals_flags_a_differing_population_qualifier():
    """The dual-pharmacology shape that retired the CTD proxy: two compatible findings
    separated by the population they were measured in, not a disagreement."""
    signals = pair_signals(
        Q1,
        "Supplementation with L-arginine reduced blood pressure in pregnant women.",
        "L-arginine supplementation did not lower blood pressure in this cohort.",
    )
    assert "population_mismatch" in signals


def test_pair_signals_flags_a_claim_that_does_not_address_its_question():
    signals = pair_signals(
        Q1,
        "Arginine supplementation reduced blood pressure significantly.",
        "Mortality was unchanged over the follow-up period.",
    )
    assert "off_question" in signals


def test_pair_signals_is_empty_when_neither_hazard_fires():
    signals = pair_signals(
        Q1,
        "Arginine supplementation reduced blood pressure significantly.",
        "L-arginine supplementation did not lower blood pressure in this cohort.",
    )
    assert signals == ()


def test_contradiction_pairs_are_stratified_by_whether_a_signal_fired(claims):
    """The stratification the whole pass turns on: pi-hat is estimated separately over
    these two subsets to test whether the lexical proxy is an actionable filter."""
    pairs = {p.pair_id: p for p in build_pairs(claims)}
    flagged = next(p for p in pairs.values() if (p.paper_id_a, p.paper_id_b) == ("101", "102"))
    clean = next(p for p in pairs.values() if (p.paper_id_a, p.paper_id_b) == ("104", "102"))
    assert flagged.stratum == "flagged"
    assert flagged.signals == ("population_mismatch",)
    assert clean.stratum == "clean"
    assert clean.signals == ()


def test_distractors_only_pair_key_term_disjoint_questions(claims):
    """A distractor is the known-unrelated anchor: two papers that genuinely address
    different questions. Q1 and Q3 share arginine/blood/pressure, so joining them would
    produce a pair the annotator could reasonably call related, destroying the anchor."""
    joined = {frozenset((p.question_id_a, p.question_id_b)) for p in build_distractors(claims)}
    assert frozenset((question_id(Q1), question_id(Q2))) in joined
    assert frozenset((question_id(Q2), question_id(Q3))) in joined
    assert frozenset((question_id(Q1), question_id(Q3))) not in joined


@pytest.fixture
def pool(claims):
    return build_pairs(claims) + build_distractors(claims)


QUOTAS = {"flagged": 2, "clean": 2, "agreement": 2, "distractor": 1}


def test_sample_batch_fills_every_stratum_quota(pool):
    batch = sample_batch(pool, rng=random.Random(7), quotas=QUOTAS, cap_paper=3, cap_question=9)
    assert Counter(p.stratum for p in batch) == Counter(QUOTAS)


def test_sample_batch_never_exceeds_the_per_paper_cap(pool):
    """The response to the corpus's non-independence: 254 papers carry 728 candidates at
    median 4 appearances and max 22, so an uncapped draw lets a few papers dominate."""
    batch = sample_batch(pool, rng=random.Random(7), quotas=QUOTAS, cap_paper=3, cap_question=9)
    appearances = Counter(pid for p in batch for pid in (p.paper_id_a, p.paper_id_b))
    assert max(appearances.values()) <= 3


def test_sample_batch_counts_a_distractor_against_both_its_questions(pool):
    """A distractor spans two questions, so it must consume a slot in each. Getting this
    wrong is what made the design's original cap of 2 look feasible when it is not."""
    with pytest.raises(ValueError, match="distractor"):
        sample_batch(
            pool,
            rng=random.Random(7),
            quotas={"distractor": 2},
            cap_paper=3,
            cap_question=1,
        )


def test_sample_batch_is_reproducible_from_its_seed(pool):
    """The sample must be recoverable from the seed and the manifest hash alone, since the
    sheet itself carries abstract text and is never committed."""

    def draw(seed: int):
        return sample_batch(
            pool, rng=random.Random(seed), quotas=QUOTAS, cap_paper=3, cap_question=9
        )

    first, again, other = draw(20260902), draw(20260902), draw(1)
    assert [p.pair_id for p in first] == [p.pair_id for p in again]
    assert [p.pair_id for p in first] != [p.pair_id for p in other]


def test_manifest_round_trips_through_disk(pool, tmp_path):
    """`signals` is a tuple and `label` a StrEnum; JSON round-trips both to something that
    compares `==` but not `is`, and the stratum split is read back with `is`."""
    path = tmp_path / "pairs.jsonl"
    write_manifest(pool, path)
    restored = read_manifest(path)
    assert restored == list(pool)
    assert manifest_hash(restored) == manifest_hash(pool)


def test_pair_ids_do_not_reveal_the_stratum(pool):
    """The annotator sees `pair_id` in the sheet heading, and this project's sole annotator
    knows the design -- so any id shape that distinguishes a distractor from a within-question
    pair would let the strictness anchor be answered from the id rather than the abstracts.
    An opaque digest is the only shape that carries nothing."""
    shapes = {p.stratum: {re.fullmatch(r"p[0-9a-f]{12}", p.pair_id) is not None} for p in pool}
    assert shapes == {s: {True} for s in shapes}
