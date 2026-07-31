from collections.abc import Mapping, Sequence

from biolit.cluster.pairing import sentence_index
from biolit.domain.enums import EntityLabel
from biolit.domain.paper import Paper
from biolit.domain.records import Entity, Finding
from biolit.extract.base import findings_from_sentence_indices
from biolit.ner.windowing import sentence_spans


class SameSentenceAsEntitiesExtractor:
    """Selects sentences holding both a linked chemical and a linked disease. No model.

    THE CONTROL, and the reason an LLM's score is attributable at all -- the same role the
    character n-gram TF-IDF control played in Phase 3C. Without it a gain cannot be credited
    to the LLM rather than to the task being easy.

    Bounded by NER and linking by construction, which is exactly what makes it the
    entity-conditioned arm: it cannot reach a relation whose endpoint was never extracted.
    """

    def __init__(self, entities_by_paper: Mapping[str, Sequence[Entity]]) -> None:
        self._entities_by_paper = entities_by_paper

    def findings(self, paper: Paper) -> list[Finding]:
        text = paper.abstract or ""
        spans = sentence_spans(text)
        chemicals: set[int] = set()
        diseases: set[int] = set()
        for entity in self._entities_by_paper.get(paper.id, ()):
            # Fails closed on both, matching SameSentencePairing: an unlinked id is
            # unscoreable against gold, and an entity with no offset cannot be placed.
            if entity.canonical_id is None or entity.start is None:
                continue
            index = sentence_index(spans, entity.start)
            if index is None:
                continue
            if entity.label is EntityLabel.CHEMICAL:
                chemicals.add(index)
            elif entity.label is EntityLabel.DISEASE:
                diseases.add(index)
        return findings_from_sentence_indices(text, chemicals & diseases)
