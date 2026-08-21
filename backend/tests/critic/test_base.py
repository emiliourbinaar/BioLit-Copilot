from biolit.critic.base import Critic, CriticPair, PaperDirection, compose
from biolit.domain.records import ContradictionFinding, ContradictionLabel


def test_opposite_paper_directions_compose_to_contradiction():
    assert compose(PaperDirection.causes, PaperDirection.treats) is ContradictionLabel.contradiction


def test_same_paper_directions_compose_to_agreement():
    assert compose(PaperDirection.causes, PaperDirection.causes) is ContradictionLabel.agreement


def test_neither_on_the_FIRST_paper_yields_insufficient_overlap():
    assert (
        compose(PaperDirection.neither, PaperDirection.treats)
        is ContradictionLabel.insufficient_overlap
    )


def test_neither_on_the_SECOND_paper_also_yields_insufficient_overlap():
    assert (
        compose(PaperDirection.causes, PaperDirection.neither)
        is ContradictionLabel.insufficient_overlap
    )


def test_a_critic_returns_a_finding_naming_both_papers():
    class Stub:
        def judge(self, pair: CriticPair) -> ContradictionFinding:
            return ContradictionFinding(
                paper_id_a=pair.paper_id_a,
                paper_id_b=pair.paper_id_b,
                label=ContradictionLabel.agreement,
                rationale="stub",
            )

    critic: Critic = Stub()
    pair = CriticPair("1", "2", "text a", "text b", "C000001", "D000001")
    finding = critic.judge(pair)
    assert (finding.paper_id_a, finding.paper_id_b) == ("1", "2")
