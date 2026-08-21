from dataclasses import dataclass
from typing import Protocol

from biolit.domain.records import ContradictionFinding


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
