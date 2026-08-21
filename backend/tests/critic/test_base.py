from biolit.critic.base import Critic, CriticPair
from biolit.domain.records import ContradictionFinding, ContradictionLabel


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
