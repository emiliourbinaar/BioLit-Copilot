from dataclasses import dataclass
from enum import StrEnum
from typing import Protocol

from biolit.domain.records import ContradictionFinding, ContradictionLabel


@dataclass(frozen=True)
class CriticPair:
    """One unit of the Critic's work, carrying only what it needs to judge.

    Deliberately NOT a Paper: `build_record` is the single licence enforcement point and the
    Extractor is the last node that ever sees a Paper, so the Critic takes text that has
    already passed that gate. Keeping Paper out of this contract is what preserves it.
    """

    paper_id_a: str
    paper_id_b: str
    text_a: str
    text_b: str
    chemical_id: str | None
    disease_id: str | None


class Critic(Protocol):
    """Judges whether two papers disagree. Injected keyword-only, like Linker/Extractor.

    Every arm implements this -- the two LLM input modes, the direction decomposition, and
    the three free baselines -- so all six are scored through one code path. The direction
    arm makes two calls internally and composes them; that composition happens BEFORE
    scoring, which is what makes McNemar valid across all of them (identical units).
    """

    def judge(self, pair: CriticPair) -> ContradictionFinding: ...


class PaperDirection(StrEnum):
    causes = "causes"
    treats = "treats"
    neither = "neither"


def compose(a: PaperDirection, b: PaperDirection) -> ContradictionLabel:
    """Pair label from two per-paper directions.

    `neither` on EITHER side yields insufficient_overlap: a paper that takes no position on
    the relationship cannot disagree with one that does. Two-sided case -- test both sides.

    SHARED ON PURPOSE. Both the free direction-lexicon baseline and the LLM direction arm
    compose two per-paper directions into a pair label. If they composed by different rules
    the two arms would not be comparable, and comparing them is the entire point of the eval.
    One shared function makes the rule identical by construction rather than by coincidence.
    """
    if a is PaperDirection.neither or b is PaperDirection.neither:
        return ContradictionLabel.insufficient_overlap
    return ContradictionLabel.agreement if a is b else ContradictionLabel.contradiction
