from collections.abc import Sequence
from typing import Protocol

from biolit.domain.enums import EntityLabel
from biolit.domain.records import Entity
from biolit.ner.windowing import sentence_spans


class PairingStrategy(Protocol):
    """Turns one paper's entities into (chemical_id, disease_id) pairs.

    Injected keyword-only into `cluster_papers`, mirroring the `Linker` seam: a real CID
    relation extractor later becomes a constructor argument, not a rewrite.
    """

    def pairs(self, entities: Sequence[Entity], text: str) -> set[tuple[str, str]]: ...


def _linked_ids(entities: Sequence[Entity], label: EntityLabel) -> set[str]:
    """Canonical ids of one label. NIL entities are excluded: a `nil:<surface>` endpoint
    is unscoreable against gold CID, so admitting one would inject an unmeasurable
    population into a measurement whose whole purpose is precision. The cost of that
    exclusion is reported by `pairing_diagnostics`, not assumed away."""
    return {e.canonical_id for e in entities if e.label is label and e.canonical_id is not None}


class CrossProductPairing:
    """Every linked chemical paired with every linked disease in the same paper.

    The naive baseline. On gold entities it cannot miss a gold pair -- both endpoints are
    annotated, so the cross-product necessarily contains every gold pair, which is why
    key recall is 1.0 by construction (anchor 1). Its entire error is precision.
    """

    def pairs(self, entities: Sequence[Entity], text: str) -> set[tuple[str, str]]:
        chemicals = _linked_ids(entities, EntityLabel.CHEMICAL)
        diseases = _linked_ids(entities, EntityLabel.DISEASE)
        return {(c, d) for c in chemicals for d in diseases}


def sentence_index(spans: Sequence[tuple[int, int]], position: int) -> int | None:
    """Index of the sentence span containing `position`, or None if it falls in no span.

    Public because `biolit.cluster.group.pairing_diagnostics` is a second consumer -- same
    reason `sentence_spans` was promoted, rather than importing a private name across
    modules and coupling group.py to this module's internals.
    """
    for index, (start, end) in enumerate(spans):
        if start <= position < end:
            return index
    return None


class SameSentencePairing:
    """Pair a chemical and a disease only when their spans fall in the same sentence.

    The free deterministic control against CrossProductPairing. Without it, a "CID relation
    extraction is required" conclusion cannot be told apart from "one line of sentence logic
    was missing" -- the same attribution discipline as the TF-IDF control in Phase 3C.

    FAILS CLOSED on an entity with no offsets: it cannot be placed in a sentence, so it
    forms no pairs and `pairing_diagnostics` counts it. Silently treating it as
    "sentence 0" would put it in the same sentence as the document's opening entities and
    invent pairs; silently returning nothing would be indistinguishable from "this paper
    had no pairs", which is the failure mode `check_document_context` exists to prevent.
    """

    def pairs(self, entities: Sequence[Entity], text: str) -> set[tuple[str, str]]:
        spans = sentence_spans(text)
        by_sentence: dict[int, tuple[set[str], set[str]]] = {}
        for entity in entities:
            if entity.canonical_id is None or entity.start is None:
                continue
            index = sentence_index(spans, entity.start)
            if index is None:
                continue
            chemicals, diseases = by_sentence.setdefault(index, (set(), set()))
            if entity.label is EntityLabel.CHEMICAL:
                chemicals.add(entity.canonical_id)
            elif entity.label is EntityLabel.DISEASE:
                diseases.add(entity.canonical_id)
        out: set[tuple[str, str]] = set()
        for chemicals, diseases in by_sentence.values():
            out |= {(c, d) for c in chemicals for d in diseases}
        return out
