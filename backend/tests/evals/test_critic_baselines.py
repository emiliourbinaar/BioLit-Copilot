from biolit.critic.base import CriticPair
from biolit.domain.records import ContradictionLabel
from biolit_evals.critic_baselines import (
    ConceptOverlapCritic,
    DirectionLexiconCritic,
    MajorityCritic,
)


def test_lexicon_reads_opposite_cues_as_a_contradiction():
    critic = DirectionLexiconCritic()
    pair = CriticPair(
        "1",
        "2",
        "Hepatotoxicity was induced by the agent in treated rats.",
        "The agent showed efficacy in the treatment of hepatic injury.",
        "C000001",
        "D000001",
    )
    assert critic.judge(pair).label is ContradictionLabel.contradiction


def test_lexicon_abstains_when_only_the_SECOND_paper_has_no_cue():
    critic = DirectionLexiconCritic()
    pair = CriticPair(
        "1",
        "2",
        "Hepatotoxicity was induced by the agent.",
        "Forty patients were enrolled.",
        "C000001",
        "D000001",
    )
    assert critic.judge(pair).label is ContradictionLabel.insufficient_overlap


def test_concept_overlap_critic_finds_agreement_on_a_PARTIALLY_shared_concept_set():
    """Multi-member, non-identical sets that share exactly one id -- real intersection, not `==`.

    Branch coverage for the `agreement` arm via a genuine partial overlap, distinct from the
    disjoint-sets test below (which is the one that mutation-testing confirmed actually kills a
    `concepts_a == concepts_b` mutant -- checked by running it, not by reasoning about it: an
    equality check on these two NON-identical, non-empty sets still returns "not equal", which
    coincides with the correct `agreement` answer here).
    """
    critic = ConceptOverlapCritic({"1": {"C000001", "C000002"}, "2": {"C000002", "D000002"}})
    pair = CriticPair("1", "2", "x", "y", "C000001", "D000001")
    assert critic.judge(pair).label is ContradictionLabel.agreement


def test_lexicon_agrees_when_both_papers_share_the_same_direction():
    critic = DirectionLexiconCritic()
    pair = CriticPair(
        "1",
        "2",
        "Hepatotoxicity was induced by the agent.",
        "The agent caused an adverse reaction with risk of hepatic injury.",
        "C000001",
        "D000001",
    )
    assert critic.judge(pair).label is ContradictionLabel.agreement


def test_majority_critic_returns_the_label_it_was_given_not_a_hardcoded_one():
    """Value-collapse guard: constructing it with `contradiction` and asserting
    `contradiction` would pass against a function that ignores its argument."""
    pair = CriticPair("1", "2", "x", "y", "C000001", "D000001")
    assert (
        MajorityCritic(ContradictionLabel.agreement).judge(pair).label
        is ContradictionLabel.agreement
    )
    assert (
        MajorityCritic(ContradictionLabel.contradiction).judge(pair).label
        is ContradictionLabel.contradiction
    )


def test_concept_overlap_critic_finds_insufficient_overlap_on_disjoint_concept_sets():
    """Two DIFFERENT, non-empty, disjoint sets -- not two empty sets, which would also collapse
    `isdisjoint` and `==` onto the same answer. This is the fixture that mutation-testing
    confirmed kills a `concepts_a == concepts_b` stand-in for `isdisjoint`: the sets here are
    unequal AND disjoint, so the equality mutant answers `agreement` where the correct answer
    is `insufficient_overlap`.
    """
    critic = ConceptOverlapCritic({"1": {"C000001", "D000001"}, "2": {"C000002", "D000002"}})
    pair = CriticPair("1", "2", "x", "y", "C000001", "D000001")
    assert critic.judge(pair).label is ContradictionLabel.insufficient_overlap
