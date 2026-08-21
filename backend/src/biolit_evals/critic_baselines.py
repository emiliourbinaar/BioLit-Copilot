"""The three free baselines an LLM Critic arm must beat before it earns its cost.

ADR-0015's standing rule is that an arm is compared against the BEST FREE BASELINE, not merely
a majority-class comparator -- a rate-matched or majority-only bar can be cleared by a mechanism
that is doing nothing interesting at all. `DirectionLexiconCritic` is this phase's version of
that bar: see its own docstring for the pre-registered expectation of how it should score,
written BEFORE any run so the number cannot be rationalised after the fact.

Every baseline here is deterministic, offline, and makes NO API call -- that is what "free"
means in this module's name.
"""

from collections.abc import Mapping
from collections.abc import Set as AbstractSet
from dataclasses import dataclass

from biolit.critic.base import CriticPair, PaperDirection, compose
from biolit.domain.records import ContradictionFinding, ContradictionLabel

# Cue substrings, matched case-insensitively. Deliberately crude -- this is a LEXICON baseline,
# not an NLP system, and its whole point is to be the cheapest mechanism that could plausibly
# work so an LLM arm has something real to clear.
CAUSES_CUES = ("induced by", "-induced", "caused", "causes", "toxicity", "adverse", "risk of")
TREATS_CUES = ("treatment of", "therapy", "therapeutic", "efficacy", "improved", "ameliorat")


def _direction_of(text: str) -> PaperDirection:
    """The per-paper direction a single abstract asserts, by counting lexicon cues.

    `PaperDirection.neither` means 'no cue fired, or the cues tied' -- NOT 'no direction'. A
    single paper's direction is a fact about that paper alone; `PaperDirection.neither` is the
    honest label for 'this lexicon found no evidence either way', and it is `compose` -- not
    this function -- that decides what a `neither` paper does to a PAIR's label.
    """
    lowered = text.lower()
    causes = sum(cue in lowered for cue in CAUSES_CUES)
    treats = sum(cue in lowered for cue in TREATS_CUES)
    if causes == treats:
        return PaperDirection.neither
    return PaperDirection.causes if causes > treats else PaperDirection.treats


class DirectionLexiconCritic:
    """PHASE 5's `first 4` -- the free heuristic the LLM arm must beat (ADR-0015's rule).

    PRE-REGISTERED EXPECTATION, recorded before the run so it cannot be rationalised after:
    this baseline is expected to score HIGH on CTD gold, because CTD's contradiction label IS a
    direction flip -- the gold standard was built by the same kind of cue that this lexicon
    matches. If an LLM arm merely MATCHES this baseline, the correct conclusion is that CTD gold
    measures cue-matching rather than reasoning, not that the LLM arm has earned its cost. That
    is a finding about the eval, not a failure of this baseline, and writing it down now is what
    stops it being explained away after the numbers are in.

    Composes two per-paper directions with `compose` -- the SAME function the direction-
    decomposition LLM arm uses (`biolit.critic.direction.DirectionCritic`, Task 12) -- so this
    baseline and that arm are comparable by construction rather than by two rules that happen
    to agree today.
    """

    def judge(self, pair: CriticPair) -> ContradictionFinding:
        a, b = _direction_of(pair.text_a), _direction_of(pair.text_b)
        label = compose(a, b)
        return ContradictionFinding(
            paper_id_a=pair.paper_id_a,
            paper_id_b=pair.paper_id_b,
            label=label,
            rationale=f"lexicon direction: a={a.value}, b={b.value}",
        )


@dataclass(frozen=True)
class MajorityCritic:
    """Always returns the same fixed label, regardless of the pair -- the majority-class floor.

    Not a serious comparator on its own (ADR-0015 explicitly rejects a majority-class bar as
    sufficient), but every arm must clear it too, and it costs nothing to compute.
    """

    label: ContradictionLabel

    def judge(self, pair: CriticPair) -> ContradictionFinding:
        return ContradictionFinding(
            paper_id_a=pair.paper_id_a,
            paper_id_b=pair.paper_id_b,
            label=self.label,
            rationale=f"majority baseline: fixed label {self.label.value}",
        )


@dataclass(frozen=True)
class ConceptOverlapCritic:
    """`insufficient_overlap` when the two papers share no canonical concept id, else `agreement`.

    The cheapest possible baseline: it never emits `contradiction` at all, because a bag of
    shared or unshared concept ids carries no directional information. It exists to test whether
    an arm is doing better than simply noticing the two papers are ABOUT the same thing.

    `concepts_by_paper` is injected (paper_id -> the set of canonical ids attested for it)
    rather than derived here, matching how `CriticPair` itself takes text that has already
    passed extraction -- this baseline does no linking of its own.
    """

    concepts_by_paper: Mapping[str, AbstractSet[str]]

    def judge(self, pair: CriticPair) -> ContradictionFinding:
        concepts_a = self.concepts_by_paper.get(pair.paper_id_a, frozenset())
        concepts_b = self.concepts_by_paper.get(pair.paper_id_b, frozenset())
        label = (
            ContradictionLabel.insufficient_overlap
            if concepts_a.isdisjoint(concepts_b)
            else ContradictionLabel.agreement
        )
        return ContradictionFinding(
            paper_id_a=pair.paper_id_a,
            paper_id_b=pair.paper_id_b,
            label=label,
            rationale=f"concept overlap: a={sorted(concepts_a)}, b={sorted(concepts_b)}",
        )
